"""Batch runner for the FL/FLB -> platform chemical Pyomo model.

Each enabled dataset is solved once for every configured objective.  All
solutions are written to one workbook, using sheet names such as
``CN_FLB_Env`` and ``EU_FL_Eco``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    SolverFactory,
    Var,
    maximize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition


OBJECTIVE_SHEETS = {"Eco": "profit", "Env": "GWP"}


def read_vector(path: Path, sheet_name: str) -> pd.Series:
    frame = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    # FL and CH sheets may contain a second, descriptive ``label`` column.
    value_columns = [column for column in frame.columns if column != "label"]
    if len(value_columns) != 1:
        raise ValueError(
            f"{path.name}/{sheet_name}: expected one value column, "
            f"found {len(value_columns)}"
        )
    result = pd.to_numeric(frame[value_columns[0]], errors="raise").dropna()
    result.index = pd.to_numeric(result.index, errors="raise")
    if result.index.has_duplicates:
        raise ValueError(f"{path.name}/{sheet_name}: duplicate indices")
    return result


def read_labels(path: Path, sheet_name: str, indices: pd.Index) -> pd.Series:
    """Read optional human-readable labels, falling back to model indices."""
    frame = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    frame.index = pd.to_numeric(frame.index, errors="raise")
    if "label" not in frame.columns:
        return pd.Series({index: str(index) for index in indices})
    labels = frame["label"].reindex(indices)
    return labels.where(labels.notna(), labels.index.to_series().astype(str)).astype(str)


def load_dataset(item: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    path = Path(item["file"])
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    sheets = item.get("sheets", {})
    eta = pd.read_excel(
        path, sheet_name=sheets.get("eta", "eta"), index_col=0
    ).apply(pd.to_numeric, errors="raise")
    eta.index = pd.to_numeric(eta.index, errors="raise")
    eta.columns = pd.to_numeric(eta.columns, errors="raise")

    fl = read_vector(path, sheets.get("FL", "FL"))
    ch = read_vector(path, sheets.get("CH", "CH"))
    fl_labels = read_labels(path, sheets.get("FL", "FL"), fl.index)
    ch_labels = read_labels(path, sheets.get("CH", "CH"), ch.index)
    coefficients = {
        label: read_vector(path, sheets.get(sheet, sheet))
        for label, sheet in OBJECTIVE_SHEETS.items()
    }

    # The vector sheets define the model sets.  This avoids hard-coded
    # RangeSet lengths and safely ignores stale/formatted rows in eta.
    missing_rows = fl.index.difference(eta.index)
    missing_cols = ch.index.difference(eta.columns)
    if len(missing_rows) or len(missing_cols):
        raise ValueError(
            f"{path.name}: eta misses FL rows {missing_rows.tolist()} "
            f"or CH columns {missing_cols.tolist()}"
        )
    eta = eta.loc[fl.index, ch.index]
    for label, series in coefficients.items():
        missing = ch.index.difference(series.index)
        if len(missing):
            raise ValueError(
                f"{path.name}: {OBJECTIVE_SHEETS[label]} misses indices "
                f"{missing.tolist()}"
            )
        coefficients[label] = series.loc[ch.index]

    return {
        "path": path,
        "region": item["region"],
        "input_tag": item["input_tag"],
        "fl": fl,
        "ch": ch,
        "eta": eta,
        "coefficients": coefficients,
        "fl_labels": fl_labels,
        "ch_labels": ch_labels,
    }


def build_model(data: dict[str, Any], objective_label: str) -> ConcreteModel:
    fl, ch, eta = data["fl"], data["ch"], data["eta"]
    coefficient = data["coefficients"][objective_label]

    model = ConcreteModel()
    model.I = Set(initialize=fl.index.tolist(), ordered=True)
    model.J = Set(initialize=ch.index.tolist(), ordered=True)
    model.FL = Param(model.I, initialize=fl.to_dict())
    model.CH = Param(model.J, initialize=ch.to_dict())
    model.eta = Param(
        model.I,
        model.J,
        initialize={
            (i, j): float(eta.loc[i, j]) for i in fl.index for j in ch.index
        },
    )
    model.objective_coefficient = Param(
        model.J, initialize=coefficient.to_dict()
    )
    model.x = Var(model.I, within=NonNegativeReals)
    model.y = Var(model.J, within=NonNegativeReals)
    model.w = Var(model.I, model.J, within=NonNegativeReals)
    model.obj = Objective(
        expr=sum(
            model.y[j] * model.objective_coefficient[j] for j in model.J
        ),
        sense=maximize,
    )
    model.x_balance = Constraint(
        model.I,
        rule=lambda m, i: m.x[i] == sum(m.w[i, j] for j in m.J),
    )
    model.y_balance = Constraint(
        model.J,
        rule=lambda m, j: m.y[j]
        == sum(m.w[i, j] * m.eta[i, j] for i in m.I),
    )
    model.demand_limit = Constraint(
        model.J, rule=lambda m, j: m.y[j] <= m.CH[j]
    )
    model.feedstock_limit = Constraint(
        model.I, rule=lambda m, i: m.x[i] <= m.FL[i]
    )
    return model


def solve_scenario(
    data: dict[str, Any],
    objective_label: str,
    solver_name: str,
    solver_executable: str | None,
) -> tuple[ConcreteModel, dict[str, Any]]:
    model = build_model(data, objective_label)
    kwargs = {"executable": solver_executable} if solver_executable else {}
    solver = SolverFactory(solver_name, **kwargs)
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{solver_name}' is unavailable. Check solver.executable "
            "in batch_config.json."
        )
    results = solver.solve(model)
    status = results.solver.status
    termination = results.solver.termination_condition
    acceptable = (
        status == SolverStatus.ok
        and termination
        in {TerminationCondition.optimal, TerminationCondition.feasible}
    )
    if not acceptable:
        raise RuntimeError(
            f"{data['region']}_{data['input_tag']}_{objective_label}: "
            f"solver status={status}, termination={termination}"
        )
    summary = {
        "Region": data["region"],
        "Input tag": data["input_tag"],
        "Objective": objective_label,
        "Objective value": value(model.obj),
        "Solver status": str(status),
        "Termination": str(termination),
        "Input file": str(data["path"]),
        "Number of I": len(model.I),
        "Number of J": len(model.J),
    }
    return model, summary


def write_scenario(
    writer: pd.ExcelWriter,
    sheet_name: str,
    model: ConcreteModel,
    summary: dict[str, Any],
) -> None:
    summary_frame = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
    x_frame = pd.DataFrame(
        {
            "i": list(model.I),
            "Available_FL": [value(model.FL[i]) for i in model.I],
            "Used_x": [value(model.x[i]) for i in model.I],
        }
    )
    y_frame = pd.DataFrame(
        {
            "j": list(model.J),
            "Demand_CH": [value(model.CH[j]) for j in model.J],
            "Produced_y": [value(model.y[j]) for j in model.J],
            "Objective_coefficient": [
                value(model.objective_coefficient[j]) for j in model.J
            ],
        }
    )
    w_frame = pd.DataFrame(
        [
            {"i": i, "j": j, "Allocated_w": value(model.w[i, j])}
            for i in model.I
            for j in model.J
        ]
    )

    row = 0
    for title, frame in (
        ("SUMMARY", summary_frame),
        ("X - feedstock use", x_frame),
        ("Y - chemical production", y_frame),
        ("W - allocation", w_frame),
    ):
        pd.DataFrame([[title]]).to_excel(
            writer, sheet_name=sheet_name, startrow=row, header=False, index=False
        )
        frame.to_excel(
            writer, sheet_name=sheet_name, startrow=row + 1, index=False
        )
        row += len(frame) + 4

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.column_dimensions["A"].width = 24
    worksheet.column_dimensions["B"].width = 24
    worksheet.column_dimensions["C"].width = 24
    worksheet.column_dimensions["D"].width = 24


def analysis_frames(
    data: dict[str, Any], model: ConcreteModel, optimized_objective: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create path contributions and regional demand-fulfilment results."""
    path_rows = []
    for i in model.I:
        for j in model.J:
            allocated = value(model.w[i, j])
            path_production = allocated * value(model.eta[i, j])
            path_rows.append(
                {
                    "Region": data["region"],
                    "Input tag": data["input_tag"],
                    "Optimized objective": optimized_objective,
                    "Feedstock ID": i,
                    "Feedstock name": data["fl_labels"].loc[i],
                    "Chemical ID": j,
                    "Chemical name": data["ch_labels"].loc[j],
                    "Allocated feedstock (w)": allocated,
                    "Conversion efficiency (eta)": value(model.eta[i, j]),
                    "Path chemical production": path_production,
                    "Economic contribution": path_production
                    * float(data["coefficients"]["Eco"].loc[j]),
                    "Environmental contribution": path_production
                    * float(data["coefficients"]["Env"].loc[j]),
                }
            )

    demand_rows = []
    for j in model.J:
        produced = value(model.y[j])
        demand = value(model.CH[j])
        demand_rows.append(
            {
                "Region": data["region"],
                "Input tag": data["input_tag"],
                "Optimized objective": optimized_objective,
                "Chemical ID": j,
                "Chemical name": data["ch_labels"].loc[j],
                "Regional production": produced,
                "Regional demand (upper limit)": demand,
                "Demand fulfilment (%)": produced / demand * 100
                if demand != 0
                else pd.NA,
            }
        )

    calculated_totals = {
        "Region": data["region"],
        "Input tag": data["input_tag"],
        "Optimized objective": optimized_objective,
        "Economic value": sum(
            value(model.y[j]) * float(data["coefficients"]["Eco"].loc[j])
            for j in model.J
        ),
        "Environmental value": sum(
            value(model.y[j]) * float(data["coefficients"]["Env"].loc[j])
            for j in model.J
        ),
    }
    return pd.DataFrame(path_rows), pd.DataFrame(demand_rows), calculated_totals


def run(config_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config_dir = config_path.parent.resolve()
    output_path = Path(config["output_file"])
    if not output_path.is_absolute():
        output_path = (config_dir / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path = Path(
        config.get("analysis_output_file", "optimization_analysis.xlsx")
    )
    if not analysis_path.is_absolute():
        analysis_path = (config_dir / analysis_path).resolve()
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    solver_config = config.get("solver", {})
    solver_name = solver_config.get("name", "glpk")
    solver_executable = solver_config.get("executable") or None
    objectives = config.get("objectives", ["Env", "Eco"])
    unknown = set(objectives).difference(OBJECTIVE_SHEETS)
    if unknown:
        raise ValueError(f"Unknown objectives: {sorted(unknown)}")

    enabled = [item for item in config["datasets"] if item.get("enabled", True)]
    expected_names = [
        f"{item['region']}_{item['input_tag']}_{objective}"
        for item in enabled
        for objective in objectives
    ]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("Duplicate region/input_tag/objective sheet name")
    too_long = [name for name in expected_names if len(name) > 31]
    if too_long:
        raise ValueError(f"Excel sheet names exceed 31 characters: {too_long}")

    all_paths = []
    all_demand = []
    all_totals = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for item in enabled:
            data = load_dataset(item, config_dir)
            for objective in objectives:
                sheet_name = f"{data['region']}_{data['input_tag']}_{objective}"
                print(f"Running {sheet_name} ...", flush=True)
                model, summary = solve_scenario(
                    data, objective, solver_name, solver_executable
                )
                write_scenario(writer, sheet_name, model, summary)
                paths, demand, totals = analysis_frames(data, model, objective)
                all_paths.append(paths)
                all_demand.append(demand)
                all_totals.append(totals)

    with pd.ExcelWriter(analysis_path, engine="openpyxl") as writer:
        pd.DataFrame(all_totals).to_excel(
            writer, sheet_name="Objective totals", index=False
        )
        pd.concat(all_paths, ignore_index=True).to_excel(
            writer, sheet_name="Path contributions", index=False
        )
        pd.concat(all_demand, ignore_index=True).to_excel(
            writer, sheet_name="Demand fulfilment", index=False
        )
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(
                    max(len(str(cell.value or "")) for cell in column_cells) + 2,
                    32,
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
        demand_sheet = writer.sheets["Demand fulfilment"]
        for cell in demand_sheet["H"][1:]:
            cell.number_format = '0.00"%"'
    print(f"Additional analysis: {analysis_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all configured region/input/objective scenarios."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("batch_config.json"),
    )
    args = parser.parse_args()
    output = run(args.config.resolve())
    print(f"Finished. Results: {output}")


if __name__ == "__main__":
    main()
