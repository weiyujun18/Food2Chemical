"""Independent Monte Carlo runner for the FL/FLB -> chemical Pyomo model.

The mathematical model is imported from ``batch_optimize.py`` and is not
modified here. Monte Carlo inputs are read by column name from the MC_* sheets.
Each region/scenario is validated deterministically before stochastic runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.environ import SolverFactory, value
from pyomo.opt import SolverStatus, TerminationCondition
from scipy import stats
from scipy.stats import qmc

from ..optimization.batch_optimize import build_model, load_dataset


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
N_ITERATIONS = 500
RANDOM_SEED = 42
SAMPLING_METHOD = "latin_hypercube"
PROGRESS_INTERVAL = 25
OBJECTIVES = {"environmental": "Env", "economic": "Eco"}
QUANTILES = [0.05, 0.50, 0.95]
CONVERGENCE_CHECKPOINTS = [
    10, 20, 50, 100, 200, 300, 400, 500,
    1000, 2000, 3000, 4000, 5000,
]
SUPPORTED_DISTRIBUTIONS = {"constant", "triangular", "normal", "lognormal"}
VALIDATION_RTOL = 1e-7
VALIDATION_ATOL = 1e-5


@dataclass(frozen=True)
class ParameterSpec:
    parameter_type: str
    parameter_name: str
    distribution: str
    base: float
    lower: float
    mode: float
    upper: float
    feedstock_id: int | None = None
    feedstock_name: str = ""
    chemical_id: int | None = None
    chemical_name: str = ""

    @property
    def uncertain(self) -> bool:
        return self.distribution != "constant" and self.lower < self.upper


def normalized(text: Any) -> str:
    return " ".join(str(text).strip().casefold().replace("_", " ").split())


def find_column(frame: pd.DataFrame, aliases: list[str], sheet: str) -> str:
    lookup = {normalized(column): column for column in frame.columns}
    for alias in aliases:
        if normalized(alias) in lookup:
            return lookup[normalized(alias)]
    raise ValueError(
        f"{sheet}: missing required column; expected one of {aliases}; "
        f"actual columns={list(frame.columns)}"
    )


def optional_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {normalized(column): column for column in frame.columns}
    return next((lookup[normalized(a)] for a in aliases if normalized(a) in lookup), None)


def numeric(row: pd.Series, column: str, sheet: str, row_number: int) -> float:
    try:
        result = float(pd.to_numeric(row[column], errors="raise"))
    except Exception as exc:
        raise ValueError(
            f"{sheet} row {row_number}: {column!r} must be numeric; "
            f"found {row[column]!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{sheet} row {row_number}: {column!r} is not finite")
    return result


def yes(value_: Any) -> bool:
    return normalized(value_) in {"yes", "y", "true", "1", "include"}


def parse_parameter_sheet(
    workbook: Path, sheet: str, parameter_type: str
) -> list[ParameterSpec]:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    if frame.empty:
        raise ValueError(f"{workbook.name}/{sheet}: sheet is empty")

    distribution_col = find_column(frame, ["Distribution"], sheet)
    lower_col = find_column(frame, ["Lower", "lower bound"], sheet)
    mode_col = find_column(frame, ["Mode", "central", "median"], sheet)
    upper_col = find_column(frame, ["Upper", "upper bound"], sheet)
    include_col = optional_column(frame, ["Include", "enabled"])
    feedstock_id_col = optional_column(frame, ["Feedstock ID", "FL ID", "FLB ID"])
    feedstock_name_col = optional_column(frame, ["Feedstock name", "FL name", "FLB name"])
    chemical_id_col = optional_column(frame, ["Chemical ID", "PC ID"])
    chemical_name_col = optional_column(frame, ["Chemical name", "PC name"])
    base_aliases = {
        "eta": ["Base eta", "eta", "base value"],
        "availability": ["Base availability", "availability", "base value"],
        "gwp": ["Base GWP saving", "base GWP", "base value"],
        "economic_margin": ["Base economic margin", "base profit", "base value"],
    }
    base_col = find_column(frame, base_aliases[parameter_type], sheet)

    if parameter_type in {"eta", "availability"} and feedstock_id_col is None:
        raise ValueError(f"{sheet}: missing Feedstock ID column")
    if parameter_type in {"eta", "gwp", "economic_margin"} and chemical_id_col is None:
        raise ValueError(f"{sheet}: missing Chemical ID column")

    specs = []
    for row_index, row in frame.iterrows():
        row_number = row_index + 2
        base = numeric(row, base_col, sheet, row_number)
        lower = numeric(row, lower_col, sheet, row_number)
        mode = numeric(row, mode_col, sheet, row_number)
        upper = numeric(row, upper_col, sheet, row_number)
        distribution = normalized(row[distribution_col])
        included = True if include_col is None else yes(row[include_col])
        if not included or lower == upper or distribution in {"", "nan", "none"}:
            distribution = "constant"
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"{workbook.name}/{sheet} row {row_number}: unsupported "
                f"distribution {row[distribution_col]!r}"
            )
        if lower > upper:
            raise ValueError(
                f"{workbook.name}/{sheet} row {row_number}: lower {lower} > upper {upper}"
            )
        if not lower <= mode <= upper:
            raise ValueError(
                f"{workbook.name}/{sheet} row {row_number}: mode {mode} is outside "
                f"[{lower}, {upper}]"
            )
        if parameter_type == "eta":
            if lower < 0 or upper > 1:
                raise ValueError(
                    f"{workbook.name}/{sheet} row {row_number}: eta range "
                    f"[{lower}, {upper}] is outside physical range [0, 1]"
                )
        if parameter_type == "availability" and lower < 0:
            raise ValueError(
                f"{workbook.name}/{sheet} row {row_number}: availability lower bound is negative"
            )
        if distribution == "lognormal" and (lower <= 0 or mode <= 0 or upper <= 0):
            raise ValueError(
                f"{workbook.name}/{sheet} row {row_number}: lognormal bounds/mode must be > 0"
            )

        feedstock_id = int(row[feedstock_id_col]) if feedstock_id_col else None
        chemical_id = int(row[chemical_id_col]) if chemical_id_col else None
        feedstock_name = (
            str(row[feedstock_name_col]).strip() if feedstock_name_col else ""
        )
        chemical_name = (
            str(row[chemical_name_col]).strip() if chemical_name_col else ""
        )
        key_parts = [parameter_type]
        if feedstock_id is not None:
            key_parts.append(f"FL_{feedstock_id}")
        if chemical_id is not None:
            key_parts.append(f"CH_{chemical_id}")
        specs.append(
            ParameterSpec(
                parameter_type=parameter_type,
                parameter_name="__".join(key_parts),
                distribution=distribution,
                base=base,
                lower=lower,
                mode=mode,
                upper=upper,
                feedstock_id=feedstock_id,
                feedstock_name=feedstock_name,
                chemical_id=chemical_id,
                chemical_name=chemical_name,
            )
        )
    return specs


def load_mc_input(workbook: Path, region: str, scenario: str) -> dict[str, Any]:
    if not workbook.exists():
        raise FileNotFoundError(f"Monte Carlo input file not found: {workbook}")
    required_sheets = {"eta", "FL", "CH", "profit", "GWP", "MC_eta", "MC_FL", "MC_GWP", "MC_profit"}
    actual_sheets = set(pd.ExcelFile(workbook).sheet_names)
    missing = required_sheets.difference(actual_sheets)
    if missing:
        raise ValueError(
            f"{workbook.name}: missing sheets {sorted(missing)}; actual={sorted(actual_sheets)}"
        )
    data = load_dataset(
        {"file": str(workbook), "region": region, "input_tag": scenario},
        workbook.parent,
    )
    specs = (
        parse_parameter_sheet(workbook, "MC_eta", "eta")
        + parse_parameter_sheet(workbook, "MC_FL", "availability")
        + parse_parameter_sheet(workbook, "MC_GWP", "gwp")
        + parse_parameter_sheet(workbook, "MC_profit", "economic_margin")
    )
    validate_mapping(workbook, data, specs)
    data["parameter_specs"] = specs
    return data


def validate_mapping(workbook: Path, data: dict[str, Any], specs: list[ParameterSpec]) -> None:
    i_set, j_set = set(data["fl"].index), set(data["ch"].index)
    seen = set()
    for spec in specs:
        key = (spec.parameter_type, spec.feedstock_id, spec.chemical_id)
        if key in seen:
            raise ValueError(f"{workbook.name}: duplicate MC parameter mapping {key}")
        seen.add(key)
        if spec.feedstock_id is not None and spec.feedstock_id not in i_set:
            raise ValueError(
                f"{workbook.name}: {spec.parameter_name} maps to missing feedstock ID {spec.feedstock_id}"
            )
        if spec.chemical_id is not None and spec.chemical_id not in j_set:
            raise ValueError(
                f"{workbook.name}: {spec.parameter_name} maps to missing chemical ID {spec.chemical_id}"
            )
    expected = (
        {("eta", int(i), int(j)) for i in i_set for j in j_set}
        | {("availability", int(i), None) for i in i_set}
        | {("gwp", None, int(j)) for j in j_set}
        | {("economic_margin", None, int(j)) for j in j_set}
    )
    missing = expected.difference(seen)
    if missing:
        raise ValueError(f"{workbook.name}: missing {len(missing)} parameter mappings; examples={list(missing)[:8]}")


def sample_from_u(spec: ParameterSpec, u: float) -> float:
    if not spec.uncertain:
        return spec.base
    u = float(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
    if spec.distribution == "triangular":
        c = (spec.mode - spec.lower) / (spec.upper - spec.lower)
        result = stats.triang.ppf(u, c=c, loc=spec.lower, scale=spec.upper - spec.lower)
    elif spec.distribution == "normal":
        sigma = max((spec.upper - spec.lower) / 6.0, np.finfo(float).eps)
        a, b = (spec.lower - spec.mode) / sigma, (spec.upper - spec.mode) / sigma
        result = stats.truncnorm.ppf(u, a=a, b=b, loc=spec.mode, scale=sigma)
    elif spec.distribution == "lognormal":
        sigma = max((math.log(spec.upper) - math.log(spec.lower)) / 6.0, np.finfo(float).eps)
        shape = sigma
        distribution = stats.lognorm(s=shape, scale=spec.mode)
        lo_cdf, hi_cdf = distribution.cdf(spec.lower), distribution.cdf(spec.upper)
        result = distribution.ppf(lo_cdf + u * (hi_cdf - lo_cdf))
    else:
        result = spec.base
    return float(np.clip(result, spec.lower, spec.upper))


def make_sample_matrix(specs: list[ParameterSpec], n: int, seed: int, method: str) -> tuple[np.ndarray, list[int], str]:
    uncertain_indices = [index for index, spec in enumerate(specs) if spec.uncertain]
    dimensions = len(uncertain_indices)
    if dimensions == 0:
        return np.empty((n, 0)), uncertain_indices, "constant_only"
    if normalized(method) in {"latin hypercube", "lhs"}:
        try:
            return qmc.LatinHypercube(d=dimensions, seed=seed).random(n), uncertain_indices, "latin_hypercube"
        except Exception as exc:
            logging.warning("Latin Hypercube failed (%s); using ordinary random sampling", exc)
    rng = np.random.default_rng(seed)
    return rng.random((n, dimensions)), uncertain_indices, "random"


def sampled_values(specs: list[ParameterSpec], matrix: np.ndarray, uncertain_indices: list[int], iteration_index: int) -> list[float]:
    column_for_spec = {spec_index: column for column, spec_index in enumerate(uncertain_indices)}
    return [
        sample_from_u(spec, matrix[iteration_index, column_for_spec[index]])
        if index in column_for_spec else spec.base
        for index, spec in enumerate(specs)
    ]


def apply_sample(base_data: dict[str, Any], specs: list[ParameterSpec], values_: list[float]) -> dict[str, Any]:
    data = dict(base_data)
    data["fl"] = base_data["fl"].copy(deep=True)
    data["eta"] = base_data["eta"].copy(deep=True)
    data["coefficients"] = {
        name: series.copy(deep=True) for name, series in base_data["coefficients"].items()
    }
    for spec, sampled in zip(specs, values_):
        if spec.parameter_type == "eta":
            data["eta"].loc[spec.feedstock_id, spec.chemical_id] = sampled
        elif spec.parameter_type == "availability":
            data["fl"].loc[spec.feedstock_id] = sampled
        elif spec.parameter_type == "gwp":
            data["coefficients"]["Env"].loc[spec.chemical_id] = sampled
        elif spec.parameter_type == "economic_margin":
            data["coefficients"]["Eco"].loc[spec.chemical_id] = sampled
    return data


def solve_once(data: dict[str, Any], objective_code: str, solver: Any) -> tuple[Any | None, str, str]:
    model = build_model(data, objective_code)
    try:
        results = solver.solve(model, load_solutions=False)
        status = results.solver.status
        termination = results.solver.termination_condition
        label = f"{status}/{termination}"
        if status == SolverStatus.ok and termination in {
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }:
            model.solutions.load_from(results)
            return model, label, ""
        return None, label, f"Solver did not return a usable solution: {label}"
    except Exception as exc:
        return None, "solver_error", f"{type(exc).__name__}: {exc}"


def solution_metrics(model: Any, data: dict[str, Any]) -> dict[str, float]:
    total_available = sum(value(model.FL[i]) for i in model.I)
    total_used = sum(value(model.x[i]) for i in model.I)
    return {
        "objective_value": value(model.obj),
        "total_gwp_saving": sum(
            value(model.y[j]) * float(data["coefficients"]["Env"].loc[j]) for j in model.J
        ),
        "total_economic_benefit": sum(
            value(model.y[j]) * float(data["coefficients"]["Eco"].loc[j]) for j in model.J
        ),
        "total_fl_or_flb_available": total_available,
        "total_fl_or_flb_used": total_used,
        "total_fl_or_flb_unused": total_available - total_used,
    }


def deterministic_validation(
    data: dict[str, Any], solver: Any, output_dir: Path, file_suffix: str
) -> bool:
    rows = []
    specs = data["parameter_specs"]
    central_data = apply_sample(data, specs, [spec.base for spec in specs])
    # The reference side uses batch_optimize.build_model directly; the MC side
    # passes an independently copied central sample through the MC data path.
    for objective_name, objective_code in OBJECTIVES.items():
        original_model, original_status, original_error = solve_once(data, objective_code, solver)
        mc_model, mc_status, mc_error = solve_once(central_data, objective_code, solver)
        if original_model is None or mc_model is None:
            rows.append({
                "result_metric": f"{objective_name}:solver",
                "original_model_value": original_status,
                "new_mc_script_deterministic_value": mc_status,
                "absolute_difference": np.nan,
                "relative_difference": np.nan,
                "validation_pass": False,
                "notes": original_error or mc_error,
            })
            continue
        original = solution_metrics(original_model, data)
        new = solution_metrics(mc_model, central_data)
        for metric in original:
            absolute = abs(original[metric] - new[metric])
            denominator = max(abs(original[metric]), VALIDATION_ATOL)
            relative = absolute / denominator
            passed = bool(np.isclose(original[metric], new[metric], rtol=VALIDATION_RTOL, atol=VALIDATION_ATOL))
            rows.append({
                "result_metric": f"{objective_name}:{metric}",
                "original_model_value": original[metric],
                "new_mc_script_deterministic_value": new[metric],
                "absolute_difference": absolute,
                "relative_difference": relative,
                "validation_pass": passed,
                "notes": "Original batch model and MC pathway use identical MC base inputs",
            })
    validation = pd.DataFrame(rows)
    validation.to_csv(
        output_dir / f"deterministic_validation_{file_suffix}.csv", index=False
    )
    return bool(validation["validation_pass"].all())


def sampled_parameter_rows(specs: list[ParameterSpec], values_: list[float], iteration: int, region: str, scenario: str) -> list[dict[str, Any]]:
    return [
        {
            "iteration": iteration,
            "parameter_type": spec.parameter_type,
            "region": region,
            "scenario": scenario,
            "flb_source": spec.feedstock_name,
            "chemical": spec.chemical_name,
            "parameter_name": spec.parameter_name,
            "sampled_value": sampled,
            "distribution": spec.distribution,
            "lower_bound": spec.lower,
            "upper_bound": spec.upper,
        }
        for spec, sampled in zip(specs, values_)
    ]


def result_rows(model: Any, data: dict[str, Any], region: str, scenario: str, objective: str, iteration: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chemical_rows = []
    for j in model.J:
        production, demand = value(model.y[j]), value(model.CH[j])
        chemical_rows.append({
            "region": region,
            "scenario": scenario,
            "objective": objective,
            "iteration": iteration,
            "chemical": data["ch_labels"].loc[j],
            "production": production,
            "demand": demand,
            "fulfillment_rate": production / demand if demand != 0 else np.nan,
            "sampled_gwp": float(data["coefficients"]["Env"].loc[j]),
            "sampled_economic_margin": float(data["coefficients"]["Eco"].loc[j]),
        })
    allocation_rows = []
    for i in model.I:
        for j in model.J:
            allocated, eta = value(model.w[i, j]), value(model.eta[i, j])
            allocation_rows.append({
                "region": region,
                "scenario": scenario,
                "objective": objective,
                "iteration": iteration,
                "flb_source": data["fl_labels"].loc[i],
                "chemical": data["ch_labels"].loc[j],
                "allocated_amount": allocated,
                "sampled_eta": eta,
                "resulting_product_amount": allocated * eta,
            })
    return chemical_rows, allocation_rows


def statistical_summary(iterations: pd.DataFrame, chemicals: pd.DataFrame) -> pd.DataFrame:
    records = []
    metric_columns = [
        "objective_value", "total_gwp_saving", "total_economic_benefit",
        "total_fl_or_flb_used", "total_fl_or_flb_unused",
    ]

    def add(group: pd.DataFrame, objective: str, metric_type: str, metric: str, chemical: str = "") -> None:
        series = pd.to_numeric(group[metric], errors="coerce").dropna()
        if series.empty:
            return
        records.append({
            "objective": objective,
            "metric_type": metric_type,
            "metric": metric,
            "chemical": chemical,
            "count": len(series),
            "mean": series.mean(),
            "standard_deviation": series.std(ddof=1),
            "minimum": series.min(),
            "maximum": series.max(),
            "5th_percentile": series.quantile(QUANTILES[0]),
            "median": series.quantile(QUANTILES[1]),
            "95th_percentile": series.quantile(QUANTILES[2]),
        })

    successful = iterations.loc[iterations["error_message"].fillna("").eq("")]
    for objective, group in successful.groupby("objective"):
        for metric in metric_columns:
            add(group, objective, "iteration", metric)
    for (objective, chemical), group in chemicals.groupby(["objective", "chemical"]):
        add(group, objective, "chemical", "production", chemical)
        add(group, objective, "chemical", "fulfillment_rate", chemical)
    return pd.DataFrame(records)


def convergence_table(iterations: pd.DataFrame, n_iterations: int) -> pd.DataFrame:
    rows = []
    checkpoints = sorted(set(c for c in CONVERGENCE_CHECKPOINTS + [n_iterations] if c <= n_iterations))
    metrics = ["objective_value", "total_gwp_saving", "total_economic_benefit"]
    for objective, group in iterations.groupby("objective"):
        group = group.sort_values("iteration")
        for checkpoint in checkpoints:
            subset = group.loc[group["iteration"] <= checkpoint]
            for metric in metrics:
                series = pd.to_numeric(subset[metric], errors="coerce").dropna()
                rows.append({
                    "objective": objective,
                    "iteration_checkpoint": checkpoint,
                    "metric": metric,
                    "cumulative_mean": series.mean() if not series.empty else np.nan,
                    "successful_iterations_included": len(series),
                })
    return pd.DataFrame(rows)


def create_plots(
    iterations: pd.DataFrame, output_dir: Path, file_suffix: str
) -> None:
    successful = iterations.loc[iterations["error_message"].fillna("").eq("")].copy()
    if successful.empty:
        return
    fig = plt.figure(figsize=(16, 8.5))
    axes = fig.subplot_mosaic(
        [
            ["convergence", "gwp_environmental", "economic_environmental"],
            ["convergence", "gwp_economic", "economic_economic"],
        ],
        gridspec_kw={"width_ratios": [1.15, 1, 1]},
    )
    colors = {"environmental": "#4F858C", "economic": "#A86F58"}

    convergence_ax = axes["convergence"]
    economic = successful.loc[successful["objective"] == "economic"].sort_values("iteration")
    environmental = successful.loc[successful["objective"] == "environmental"].sort_values("iteration")
    economic_line = convergence_ax.plot(
        economic["iteration"], economic["objective_value"].expanding().mean(),
        color=colors["economic"], linewidth=1.4, label="Economic objective",
    )[0]
    convergence_ax.set(
        title="Objective cumulative mean", xlabel="Iteration",
        ylabel="Economic objective cumulative mean",
    )
    convergence_ax.tick_params(axis="y", colors=colors["economic"])
    convergence_ax.yaxis.label.set_color(colors["economic"])
    environmental_ax = convergence_ax.twinx()
    environmental_line = environmental_ax.plot(
        environmental["iteration"],
        environmental["objective_value"].expanding().mean(),
        color=colors["environmental"], linewidth=1.4,
        label="Environmental objective",
    )[0]
    environmental_ax.set_ylabel(
        "Environmental objective cumulative mean",
        color=colors["environmental"],
    )
    environmental_ax.tick_params(axis="y", colors=colors["environmental"])
    convergence_ax.legend(
        [economic_line, environmental_line],
        [economic_line.get_label(), environmental_line.get_label()],
        frameon=False, loc="best",
    )

    histogram_specs = (
        ("gwp_environmental", "environmental", "total_gwp_saving", "Total GWP saving — environmental optimization", "GWP saving"),
        ("gwp_economic", "economic", "total_gwp_saving", "Total GWP saving — economic optimization", "GWP saving"),
        ("economic_environmental", "environmental", "total_economic_benefit", "Total economic benefit — environmental optimization", "Economic benefit"),
        ("economic_economic", "economic", "total_economic_benefit", "Total economic benefit — economic optimization", "Economic benefit"),
    )
    for panel, objective, metric, title, xlabel in histogram_specs:
        subset = successful.loc[successful["objective"] == objective, metric].dropna()
        axes[panel].hist(
            subset, bins=30, color=colors[objective], edgecolor="white", linewidth=0.7
        )
        axes[panel].set(title=title, xlabel=xlabel, ylabel="Frequency")

    for ax in axes.values():
        ax.grid(axis="y", color="#E3E5E7", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"mc_diagnostic_plots_{file_suffix}.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def safe_write_csv(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_csv(path, index=False)
    except PermissionError as exc:
        raise PermissionError(f"Cannot write {path}; close the file if it is open") from exc


def append_rows_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Append a bounded batch of rows, keeping large 5,000-run outputs off RAM."""
    if not rows:
        return
    try:
        pd.DataFrame(rows).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )
    except PermissionError as exc:
        raise PermissionError(f"Cannot write {path}; close the file if it is open") from exc
    rows.clear()


def run_scenario(
    workbook: Path,
    region: str,
    scenario: str,
    output_root: Path,
    n_iterations: int,
    seed: int,
    method: str,
    solver: Any,
) -> dict[str, Any]:
    output_dir = output_root / f"{region}_{scenario}"
    file_suffix = f"{region}_{scenario}"
    output_dir.mkdir(parents=True, exist_ok=True)
    allocation_path = output_dir / f"mc_allocation_results_{file_suffix}.csv"
    parameter_path = output_dir / f"mc_sampled_parameters_{file_suffix}.csv"
    # A new scenario run replaces any incomplete streamed output from an older run.
    for streamed_path in (allocation_path, parameter_path):
        if streamed_path.exists():
            streamed_path.unlink()
    logging.info("Processing %s: region=%s scenario=%s", workbook, region, scenario)
    data = load_mc_input(workbook, region, scenario)
    if not deterministic_validation(data, solver, output_dir, file_suffix):
        raise RuntimeError(
            f"Deterministic validation failed for {region}_{scenario}; see "
            f"{output_dir / f'deterministic_validation_{file_suffix}.csv'}"
        )
    logging.info("Deterministic validation passed for %s_%s", region, scenario)

    specs = data["parameter_specs"]
    matrix, uncertain_indices, actual_method = make_sample_matrix(specs, n_iterations, seed, method)
    logging.info("Sampling method=%s, dimensions=%d", actual_method, len(uncertain_indices))
    iteration_rows: list[dict[str, Any]] = []
    chemical_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    success = infeasible = solver_errors = 0

    for iteration_index in range(n_iterations):
        iteration = iteration_index + 1
        values_ = sampled_values(specs, matrix, uncertain_indices, iteration_index)
        sampled_data = apply_sample(data, specs, values_)
        parameter_rows.extend(sampled_parameter_rows(specs, values_, iteration, region, scenario))
        for objective_name, objective_code in OBJECTIVES.items():
            model, solver_status, error_message = solve_once(sampled_data, objective_code, solver)
            base_row = {
                "region": region, "scenario": scenario, "objective": objective_name,
                "iteration": iteration, "random_seed": seed,
                "solver_status": solver_status, "error_message": error_message,
            }
            if model is None:
                base_row.update({key: np.nan for key in (
                    "objective_value", "total_gwp_saving", "total_economic_benefit",
                    "total_fl_or_flb_available", "total_fl_or_flb_used", "total_fl_or_flb_unused",
                )})
                if "infeasible" in solver_status.casefold():
                    infeasible += 1
                else:
                    solver_errors += 1
            else:
                base_row.update(solution_metrics(model, sampled_data))
                chemicals_, allocations_ = result_rows(
                    model, sampled_data, region, scenario, objective_name, iteration
                )
                chemical_rows.extend(chemicals_)
                allocation_rows.extend(allocations_)
                success += 1
            iteration_rows.append(base_row)
        if iteration % PROGRESS_INTERVAL == 0 or iteration == n_iterations:
            append_rows_to_csv(allocation_rows, allocation_path)
            append_rows_to_csv(parameter_rows, parameter_path)
            logging.info(
                "%s_%s progress %d/%d; successful solves=%d infeasible=%d solver_errors=%d",
                region, scenario, iteration, n_iterations, success, infeasible, solver_errors,
            )

    iterations = pd.DataFrame(iteration_rows)
    chemicals = pd.DataFrame(chemical_rows)
    safe_write_csv(iterations, output_dir / f"mc_iteration_summary_{file_suffix}.csv")
    safe_write_csv(chemicals, output_dir / f"mc_chemical_results_{file_suffix}.csv")
    safe_write_csv(
        statistical_summary(iterations, chemicals),
        output_dir / f"mc_statistical_summary_{file_suffix}.csv",
    )
    safe_write_csv(
        convergence_table(iterations, n_iterations),
        output_dir / f"mc_convergence_{file_suffix}.csv",
    )
    create_plots(iterations, output_dir, file_suffix)
    return {
        "region": region, "scenario": scenario, "successful_solves": success,
        "infeasible_solves": infeasible, "solver_errors": solver_errors,
        "expected_solves": n_iterations * len(OBJECTIVES), "output_dir": str(output_dir),
    }


def discover_inputs(input_dir: Path) -> list[tuple[Path, str, str]]:
    results = []
    for region in ("CN", "EU", "US"):
        candidates = [
            (input_dir / f"FL2CH_{region}_MC_input_eta.xlsx", "FLB"),
            (input_dir / f"FL2CH_{region}_FL_MC_input_eta.xlsx", "FL"),
        ]
        for path, scenario in candidates:
            if path.exists():
                results.append((path, region, scenario))
    if not results:
        raise FileNotFoundError(f"No Monte Carlo input workbooks found in {input_dir}")
    return results


def compare_existing_inputs(mc_inputs: list[tuple[Path, str, str]], model_input_root: Path) -> list[str]:
    warnings = []
    for mc_path, region, scenario in mc_inputs:
        deterministic = (
            model_input_root
            / scenario
            / f"FL2CH_{region}{'_FL' if scenario == 'FL' else ''}.xlsx"
        )
        if not deterministic.exists():
            warnings.append(f"No existing deterministic input for {region}_{scenario}: {deterministic}")
            continue
        for sheet in ("eta", "FL", "CH", "GWP", "profit"):
            left = pd.read_excel(deterministic, sheet_name=sheet, index_col=0)
            right = pd.read_excel(mc_path, sheet_name=sheet, index_col=0)
            columns = [column for column in left.columns if normalized(column) != "label"]
            a = left[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
            b = right[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
                difference = np.nanmax(np.abs(a - b)) if a.shape == b.shape else np.nan
                warnings.append(
                    f"{region}_{scenario}/{sheet}: MC base differs from existing deterministic "
                    f"input (max absolute difference={difference})"
                )
    return warnings


def configure_logging(output_root: Path, run_tag: str = "") -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_tag}" if run_tag else ""
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output_root / f"mc_run{suffix}.log", mode="w", encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers, force=True,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run reproducible Monte Carlo optimization analysis")
    parser.add_argument("--input-dir", type=Path, default=repo_root / "data" / "monte_carlo_inputs")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "monte_carlo")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--sampling-method", default=SAMPLING_METHOD)
    parser.add_argument("--solver", default="glpk")
    parser.add_argument(
        "--solver-executable",
        default=None,
        help="Optional solver executable. By default, resolve the solver from PATH.",
    )
    parser.add_argument(
        "--only", default="",
        help="Optional single scenario such as CN_FLB; useful for parallel independent runs.",
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    output_root = args.output_dir.resolve()
    run_tag = args.only.strip().upper()
    configure_logging(output_root, run_tag)
    inputs = discover_inputs(args.input_dir.resolve())
    if run_tag:
        inputs = [item for item in inputs if f"{item[1]}_{item[2]}" == run_tag]
        if not inputs:
            raise ValueError(f"Unknown --only scenario {args.only!r}")
    logging.info("Found %d input scenario(s): %s", len(inputs), [(r, s) for _, r, s in inputs])
    for warning in compare_existing_inputs(inputs, repo_root / "data" / "model_inputs"):
        logging.warning("Input audit: %s", warning)

    solver = SolverFactory(args.solver, executable=args.solver_executable or None)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"Solver {args.solver!r} is unavailable; executable={args.solver_executable!r}")

    run_summaries = []
    for workbook, region, scenario in inputs:
        run_summaries.append(
            run_scenario(
                workbook, region, scenario, output_root, args.iterations,
                args.seed, args.sampling_method, solver,
            )
        )
    run_summary = pd.DataFrame(run_summaries)
    suffix = f"_{run_tag}" if run_tag else ""
    safe_write_csv(run_summary, output_root / f"mc_run_summary{suffix}.csv")
    with (output_root / f"mc_run_configuration{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "iterations": args.iterations, "random_seed": args.seed,
                "sampling_method_requested": args.sampling_method,
                "objectives": list(OBJECTIVES), "quantiles": QUANTILES,
            }, handle, indent=2,
        )
    logging.info("All Monte Carlo analyses finished. Results: %s", output_root)


if __name__ == "__main__":
    main()
