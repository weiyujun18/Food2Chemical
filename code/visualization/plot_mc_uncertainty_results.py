"""Publication figures for the 5,000-run Monte Carlo optimization analysis.

The script reads iteration-level summaries, calculates mean and 5th/50th/95th
percentiles, exports a tidy summary table, and creates separate FLB and FL
point-range figures. Deterministic optimization results are overlaid as asterisks.
Values are explicitly scaled from kg CO2-eq to Mt CO2-eq and from USD to
billion USD for readable publication axes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# -----------------------------------------------------------------------------
# User-editable configuration
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_INPUT = (
    REPO_ROOT
    / "figures"
    / "input_data"
    / "MC_uncertainty_summary_with_optimization.csv"
)
OUTPUT_DIR = REPO_ROOT / "figures" / "main"

FIGURE_SIZE = (10.2, 8.0)
SELECTED_FIGURE_SIZE = (16.2, 6.3)
PNG_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONT_SIZE = 12.5
LABEL_FONT_SIZE = 13.5
TITLE_FONT_SIZE = 15.5
PANEL_FONT_SIZE = 14.5
MARKER_SIZE = 8.0
ERROR_LINE_WIDTH = 1.8
CAP_SIZE = 5.0
OPTIMIZATION_MARKER_SIZE = 90
OPTIMIZATION_X_OFFSET = 0.08

TEXT_COLOR = "#303438"
GRID_COLOR = "#DDE2E5"
SCENARIO_COLORS = {
    "environmental": "#245B66",  # GWP-saving maximization
    "economic": "#A86F58",       # Economic-margin maximization
}
SCENARIO_LABELS = {
    "environmental": "GWP optimization",
    "economic": "Economic optimization",
}
REGION_LABELS = {
    "CN": "China",
    "US": "United States",
    "EU": "European Union",
}
SELECTED_REGION_LABELS = {
    "CN": "China",
    "US": "The U.S.",
    "EU": "the EU",
}
REGION_ORDER = ["CN", "US", "EU"]
SCENARIO_ORDER = ["environmental", "economic"]

INDICATORS = {
    "total_gwp_saving": {
        "label": "Total GWP saving",
        "unit": r"Mt CO$_2$-eq yr$^{-1}$",
        "scale_factor": 1e9,  # input: kg CO2-eq; 1 Mt = 1e9 kg
        "optimization_column": "Environmental value",
    },
    "total_economic_benefit": {
        "label": "Total economic margin",
        "unit": r"billion USD yr$^{-1}$",
        "scale_factor": 1e9,  # input: USD; 1 billion USD = 1e9 USD
        "optimization_column": "Economic value",
    },
}


def find_iteration_files(input_dir: Path) -> list[Path]:
    """Find one iteration-summary CSV for every region/scope directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Monte Carlo result directory not found: {input_dir}")
    files = sorted(input_dir.glob("*_*/mc_iteration_summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No mc_iteration_summary_*.csv files found under {input_dir}")
    return files


def read_iteration_results(input_dir: Path) -> pd.DataFrame:
    """Read, validate, and combine successful iteration-level results."""
    required = {
        "region", "scenario", "objective", "iteration", "solver_status",
        "total_gwp_saving", "total_economic_benefit",
    }
    frames: list[pd.DataFrame] = []
    for path in find_iteration_files(input_dir):
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        frames.append(frame[list(required)].copy())

    data = pd.concat(frames, ignore_index=True)
    data["region"] = data["region"].astype(str).str.upper().str.strip()
    data["scenario"] = data["scenario"].astype(str).str.upper().str.strip()
    data["objective"] = data["objective"].astype(str).str.lower().str.strip()
    successful = data["solver_status"].astype(str).str.lower().str.contains("optimal")
    dropped = int((~successful).sum())
    if dropped:
        print(f"Excluded {dropped:,} non-optimal result rows from plotting.")
    data = data.loc[successful].copy()

    for column in INDICATORS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[list(INDICATORS)].isna().any().any():
        bad = int(data[list(INDICATORS)].isna().any(axis=1).sum())
        raise ValueError(f"Found {bad:,} optimal rows with non-numeric output values")

    expected = {(region, scope, objective) for region in REGION_ORDER
                for scope in ("FLB", "FL") for objective in SCENARIO_ORDER}
    actual = set(data[["region", "scenario", "objective"]].itertuples(index=False, name=None))
    missing_groups = expected.difference(actual)
    if missing_groups:
        raise ValueError(f"Missing Monte Carlo groups: {sorted(missing_groups)}")
    return data


def calculate_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate requested statistics in explicitly labelled display units."""
    rows: list[dict[str, object]] = []
    group_columns = ["scenario", "region", "objective"]
    for keys, group in data.groupby(group_columns, observed=True, sort=False):
        scope, region, objective = keys
        for indicator, metadata in INDICATORS.items():
            raw = group[indicator].to_numpy(dtype=float)
            scale = float(metadata["scale_factor"])
            values = raw / scale
            mean = float(np.mean(values))
            p05, median, p95 = np.quantile(values, [0.05, 0.50, 0.95])
            relative_width = np.nan if np.isclose(mean, 0.0) else (p95 - p05) / abs(mean) * 100
            rows.append({
                "feedstock_scope": scope,
                "region": REGION_LABELS[region],
                "region_code": region,
                "optimization_scenario": SCENARIO_LABELS[objective],
                "objective_code": objective,
                "output_indicator": metadata["label"],
                "output_column": indicator,
                "display_unit": metadata["unit"].replace("$", ""),
                "input_to_display_scale_factor": scale,
                "n_successful_iterations": len(values),
                "mean": mean,
                "5th_percentile": float(p05),
                "median": float(median),
                "95th_percentile": float(p95),
                "relative_uncertainty_width_percent": float(relative_width),
            })
    result = pd.DataFrame(rows)
    scope_order = pd.Categorical(result["feedstock_scope"], ["FLB", "FL"], ordered=True)
    region_order = pd.Categorical(result["region_code"], REGION_ORDER, ordered=True)
    objective_order = pd.Categorical(result["objective_code"], SCENARIO_ORDER, ordered=True)
    indicator_order = pd.Categorical(result["output_column"], list(INDICATORS), ordered=True)
    result = (result.assign(_scope=scope_order, _region=region_order,
                            _objective=objective_order, _indicator=indicator_order)
              .sort_values(["_scope", "_region", "_objective", "_indicator"])
              .drop(columns=["_scope", "_region", "_objective", "_indicator"])
              .reset_index(drop=True))
    return result


def read_optimization_results(workbook: Path) -> pd.DataFrame:
    """Read deterministic objective totals and convert them to display units."""
    if not workbook.is_file():
        raise FileNotFoundError(f"Optimization analysis workbook not found: {workbook}")
    frame = pd.read_excel(workbook, sheet_name="Objective totals")
    required = {
        "Region", "Input tag", "Optimized objective",
        "Economic value", "Environmental value",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{workbook.name}/Objective totals: missing columns {sorted(missing)}"
        )

    frame = frame[list(required)].copy()
    frame["Region"] = frame["Region"].astype(str).str.upper().str.strip()
    frame["Input tag"] = frame["Input tag"].astype(str).str.upper().str.strip()
    objective_codes = {"ENV": "environmental", "ECO": "economic"}
    frame["objective_code"] = (
        frame["Optimized objective"].astype(str).str.upper().str.strip()
        .map(objective_codes)
    )
    if frame["objective_code"].isna().any():
        unknown = sorted(
            frame.loc[frame["objective_code"].isna(), "Optimized objective"]
            .astype(str).unique()
        )
        raise ValueError(f"Unknown deterministic optimization objectives: {unknown}")

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        for indicator, metadata in INDICATORS.items():
            source_column = str(metadata["optimization_column"])
            raw_value = pd.to_numeric(row[source_column], errors="raise")
            rows.append({
                "feedstock_scope": row["Input tag"],
                "region_code": row["Region"],
                "objective_code": row["objective_code"],
                "output_column": indicator,
                "optimization_result": float(raw_value)
                / float(metadata["scale_factor"]),
            })
    result = pd.DataFrame(rows)
    keys = ["feedstock_scope", "region_code", "objective_code", "output_column"]
    if result.duplicated(keys).any():
        duplicates = result.loc[result.duplicated(keys, keep=False), keys]
        raise ValueError(
            "Duplicate deterministic optimization groups: "
            f"{duplicates.drop_duplicates().to_dict('records')}"
        )
    return result


def add_optimization_results(
    summary: pd.DataFrame, optimization_results: pd.DataFrame
) -> pd.DataFrame:
    """Attach one deterministic value to each Monte Carlo summary row."""
    keys = ["feedstock_scope", "region_code", "objective_code", "output_column"]
    combined = summary.merge(
        optimization_results, on=keys, how="left", validate="one_to_one"
    )
    if combined["optimization_result"].isna().any():
        missing = combined.loc[combined["optimization_result"].isna(), keys]
        raise ValueError(
            "Missing deterministic optimization results for: "
            f"{missing.to_dict('records')}"
        )
    return combined


def global_y_limits(summary: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Use identical indicator limits in both figures to support scope comparison."""
    limits: dict[str, tuple[float, float]] = {}
    for indicator in INDICATORS:
        subset = summary[summary["output_column"] == indicator]
        low = min(
            0.0,
            float(subset["5th_percentile"].min()),
            float(subset["optimization_result"].min()),
        )
        high = max(
            float(subset["95th_percentile"].max()),
            float(subset["optimization_result"].max()),
        )
        span = high - low
        limits[indicator] = (low - 0.03 * span, high + 0.12 * span)
    return limits


def plot_scope(summary: pd.DataFrame, scope: str,
               y_limits: dict[str, tuple[float, float]]) -> None:
    """Create the requested two-panel point-range figure for one feedstock scope."""
    scope_data = summary[summary["feedstock_scope"] == scope]
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True,
                             gridspec_kw={"hspace": 0.12}, facecolor="white")
    offsets = {"environmental": -0.16, "economic": 0.16}
    x_base = np.arange(len(REGION_ORDER), dtype=float)

    for panel_index, (ax, indicator) in enumerate(zip(axes, INDICATORS)):
        metadata = INDICATORS[indicator]
        for objective in SCENARIO_ORDER:
            subset = (scope_data[(scope_data["output_column"] == indicator) &
                                 (scope_data["objective_code"] == objective)]
                      .set_index("region_code").loc[REGION_ORDER])
            means = subset["mean"].to_numpy(dtype=float)
            p05 = subset["5th_percentile"].to_numpy(dtype=float)
            medians = subset["median"].to_numpy(dtype=float)
            p95 = subset["95th_percentile"].to_numpy(dtype=float)
            x = x_base + offsets[objective]
            yerr = np.vstack((np.maximum(0, means - p05), np.maximum(0, p95 - means)))
            ax.errorbar(x, means, yerr=yerr, fmt="o", markersize=MARKER_SIZE,
                        color=SCENARIO_COLORS[objective], markeredgecolor="white",
                        markeredgewidth=0.8, ecolor=SCENARIO_COLORS[objective],
                        elinewidth=ERROR_LINE_WIDTH, capsize=CAP_SIZE, capthick=ERROR_LINE_WIDTH,
                        zorder=3)
            # Median is a small dark horizontal marker, distinct from the mean point.
            ax.scatter(x, medians, marker="_", s=95, linewidths=1.5,
                       color=TEXT_COLOR, zorder=4)
            # Deterministic (non-Monte-Carlo) optimization result.
            optimization_values = subset["optimization_result"].to_numpy(dtype=float)
            ax.scatter(x + OPTIMIZATION_X_OFFSET, optimization_values,
                       marker=r"$\ast$",
                       s=OPTIMIZATION_MARKER_SIZE, color=TEXT_COLOR,
                       linewidths=0.8, zorder=5)

        ax.set_ylabel(f"{metadata['label']}\n({metadata['unit']})", fontsize=LABEL_FONT_SIZE)
        ax.set_ylim(*y_limits[indicator])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.75, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=BASE_FONT_SIZE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#55585C")
        ax.spines[["left", "bottom"]].set_linewidth(0.8)
        ax.text(-0.105, 1.04, chr(ord("a") + panel_index), transform=ax.transAxes,
                fontsize=PANEL_FONT_SIZE, fontweight="bold", va="bottom", ha="left")

    axes[-1].set_xticks(x_base, [REGION_LABELS[region] for region in REGION_ORDER])
    axes[-1].set_xlabel("Region", fontsize=LABEL_FONT_SIZE)
    axes[0].tick_params(axis="x", which="both", length=0)
    scope_title = "FLB-based scenarios" if scope == "FLB" else "FL-based scenarios"
    fig.suptitle(f"Monte Carlo uncertainty: {scope_title}", fontsize=TITLE_FONT_SIZE,
                 fontweight="semibold", color=TEXT_COLOR, y=0.985)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=SCENARIO_COLORS[obj], markeredgecolor="white",
               label=SCENARIO_LABELS[obj])
        for obj in SCENARIO_ORDER
    ]
    legend_handles.append(Line2D([0], [0], marker="_", linestyle="none", markersize=11,
                                 color=TEXT_COLOR, label="Median"))
    legend_handles.append(Line2D([0], [0], marker=r"$\ast$", linestyle="none",
                                 markersize=12, color=TEXT_COLOR,
                                 label="Deterministic optimization"))
    axes[0].legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.22),
                   ncol=4, frameon=False, fontsize=11.0, columnspacing=1.2)

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.10, top=0.88)
    stem = OUTPUT_DIR / f"MC_uncertainty_{scope}_results_with_optimization"
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created figure set: {stem}.[png|svg|pdf]")


def plot_selected_scope(
    summary: pd.DataFrame,
    scope: str,
    y_limits: dict[str, tuple[float, float]],
) -> None:
    """Plot only the principal output of each matching optimization objective."""
    scope_data = summary[summary["feedstock_scope"] == scope]
    panels = (
        ("total_gwp_saving", "environmental", "Total GWP saving"),
        ("total_economic_benefit", "economic", "Potential economic margin"),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=SELECTED_FIGURE_SIZE, facecolor="white",
        gridspec_kw={"wspace": 0.30},
    )
    x_base = np.arange(len(REGION_ORDER), dtype=float)

    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        indicator, objective, axis_label = panel
        metadata = INDICATORS[indicator]
        subset = (
            scope_data[
                (scope_data["output_column"] == indicator)
                & (scope_data["objective_code"] == objective)
            ]
            .set_index("region_code")
            .loc[REGION_ORDER]
        )
        means = subset["mean"].to_numpy(dtype=float)
        p05 = subset["5th_percentile"].to_numpy(dtype=float)
        medians = subset["median"].to_numpy(dtype=float)
        p95 = subset["95th_percentile"].to_numpy(dtype=float)
        yerr = np.vstack(
            (np.maximum(0, means - p05), np.maximum(0, p95 - means))
        )
        ax.errorbar(
            x_base, means, yerr=yerr, fmt="o", markersize=MARKER_SIZE,
            color=SCENARIO_COLORS[objective], markeredgecolor="white",
            markeredgewidth=0.8, ecolor=SCENARIO_COLORS[objective],
            elinewidth=ERROR_LINE_WIDTH, capsize=CAP_SIZE,
            capthick=ERROR_LINE_WIDTH, zorder=3,
        )
        ax.scatter(
            x_base, medians, marker="_", s=95, linewidths=1.5,
            color=TEXT_COLOR, zorder=4,
        )
        optimization_values = subset["optimization_result"].to_numpy(dtype=float)
        ax.scatter(
            x_base + OPTIMIZATION_X_OFFSET, optimization_values,
            marker=r"$\ast$", s=OPTIMIZATION_MARKER_SIZE,
            color=TEXT_COLOR, linewidths=0.8, zorder=5,
        )

        ax.set_ylabel(
            f"{axis_label}\n({metadata['unit']})", fontsize=LABEL_FONT_SIZE
        )
        ax.set_ylim(*y_limits[indicator])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.75, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=BASE_FONT_SIZE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#55585C")
        ax.spines[["left", "bottom"]].set_linewidth(0.8)
        ax.text(
            -0.105, 1.04, chr(ord("a") + panel_index), transform=ax.transAxes,
            fontsize=PANEL_FONT_SIZE, fontweight="bold", va="bottom", ha="left",
        )

    for ax in axes:
        ax.set_xticks(
            x_base, [SELECTED_REGION_LABELS[region] for region in REGION_ORDER]
        )
        ax.set_xlabel("Region", fontsize=LABEL_FONT_SIZE)
    scope_title = "FLB-based scenarios" if scope == "FLB" else "FL-based scenarios"
    fig.suptitle(
        f"Monte Carlo uncertainty: {scope_title}",
        fontsize=TITLE_FONT_SIZE, fontweight="semibold", color=TEXT_COLOR, y=0.985,
    )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=SCENARIO_COLORS[objective], markeredgecolor="white",
            label=SCENARIO_LABELS[objective],
        )
        for objective in SCENARIO_ORDER
    ]
    legend_handles.extend([
        Line2D(
            [0], [0], marker="_", linestyle="none", markersize=11,
            color=TEXT_COLOR, label="Median",
        ),
        Line2D(
            [0], [0], marker=r"$\ast$", linestyle="none", markersize=12,
            color=TEXT_COLOR, label="Deterministic optimization",
        ),
    ])
    fig.legend(
        handles=legend_handles, loc="center left", bbox_to_anchor=(0.805, 0.50),
        ncol=1, frameon=False, fontsize=11.0, labelspacing=1.15,
    )

    fig.subplots_adjust(left=0.09, right=0.79, bottom=0.16, top=0.82)
    stem = OUTPUT_DIR / f"MC_uncertainty_{scope}_selected_horizontal_with_optimization"
    fig.savefig(
        stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created selected figure set: {stem}.[png|svg|pdf]")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Create Monte Carlo uncertainty figures from publication source data."
    )
    parser.add_argument("--summary-input", type=Path, default=SUMMARY_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "svg.fonttype": "none",  # preserve editable text in SVG
        "pdf.fonttype": 42,       # embed editable TrueType text in PDF
    })
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_input.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Monte Carlo figure source data not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {
        "feedstock_scope", "region", "region_code", "optimization_scenario",
        "objective_code", "output_indicator", "output_column", "display_unit",
        "mean", "5th_percentile", "median", "95th_percentile",
        "optimization_result",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{summary_path.name}: missing columns {sorted(missing)}")
    limits = global_y_limits(summary)
    for scope in ("FLB", "FL"):
        plot_scope(summary, scope, limits)
        plot_selected_scope(summary, scope, limits)
    print(f"Created Monte Carlo figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
