"""Create regional FL-to-chemical heatmaps with optimized production bubbles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable

from .allocation_chord import read_result_sheet


REGIONS = [
    ("CN", "China"),
    ("US", "the U.S."),
    ("EU", "the EU"),
]
COLUMN_CAPTIONS = ["Yield", "Max. GWP saving", "Max. economic margin"]
BUBBLE_COLOR = "#4F858C"
EPSILON = 1e-8


def normalize_label(value: object) -> str:
    label = " ".join(str(value).strip().split())
    aliases = {
        "potatoe": "Potato",
        "whole wheat": "Wheat",
    }
    return aliases.get(label.casefold(), label.title())


def load_region(
    input_dir: Path, region: str, scenario: str
) -> dict[str, object]:
    suffix = "_FL" if scenario == "FL" else ""
    path = input_dir / f"FL2CH_{region}{suffix}.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Input workbook not found: {path}")

    eta = pd.read_excel(path, sheet_name="eta", index_col=0).apply(
        pd.to_numeric, errors="raise"
    )
    fl = pd.read_excel(path, sheet_name="FL", index_col=0)
    ch = pd.read_excel(path, sheet_name="CH", index_col=0)
    gwp = pd.read_excel(path, sheet_name="GWP", index_col=0)
    profit = pd.read_excel(path, sheet_name="profit", index_col=0)
    for frame, name in ((fl, "FL"), (ch, "CH")):
        if "label" not in frame.columns:
            raise ValueError(f"{path.name}/{name}: missing label column")

    fl_ids = pd.to_numeric(fl.index, errors="raise").astype(int)
    ch_ids = pd.to_numeric(ch.index, errors="raise").astype(int)
    eta.index = pd.to_numeric(eta.index, errors="raise").astype(int)
    eta.columns = pd.to_numeric(eta.columns, errors="raise").astype(int)
    eta = eta.loc[fl_ids, ch_ids]

    def coefficient(frame: pd.DataFrame, sheet: str) -> pd.Series:
        value_columns = [column for column in frame.columns if column != "label"]
        if len(value_columns) != 1:
            raise ValueError(f"{path.name}/{sheet}: expected one value column")
        series = pd.to_numeric(frame[value_columns[0]], errors="raise")
        series.index = pd.to_numeric(series.index, errors="raise").astype(int)
        return series.loc[ch_ids]

    gwp_values = coefficient(gwp, "GWP")
    profit_values = coefficient(profit, "profit")
    yield_values = eta.to_numpy(dtype=float)
    gwp_matrix = yield_values * gwp_values.to_numpy(dtype=float)[None, :]
    profit_matrix = yield_values * profit_values.to_numpy(dtype=float)[None, :]

    def positive_log10(matrix: np.ndarray) -> np.ndarray:
        result = np.full(matrix.shape, np.nan, dtype=float)
        positive = matrix > 0
        result[positive] = np.log10(matrix[positive])
        return result

    return {
        "path": path,
        "eta": eta,
        "fl_labels": [normalize_label(value) for value in fl.loc[fl_ids, "label"]],
        "ch_labels": [str(value).strip() for value in ch.loc[ch_ids, "label"]],
        "matrices": [
            yield_values,
            positive_log10(gwp_matrix),
            positive_log10(profit_matrix),
        ],
    }


def pathway_production(
    results_workbook: Path,
    sheet_name: str,
    eta: pd.DataFrame,
) -> np.ndarray:
    _, _, allocation = read_result_sheet(results_workbook, sheet_name)
    result = np.zeros(eta.shape, dtype=float)
    i_position = {int(value): index for index, value in enumerate(eta.index)}
    j_position = {int(value): index for index, value in enumerate(eta.columns)}
    for row in allocation.itertuples(index=False):
        i, j = int(row.i), int(row.j)
        if i not in i_position or j not in j_position:
            raise ValueError(f"{sheet_name}: allocation mapping ({i}, {j}) not in FL input")
        product = float(row.Allocated_w) * float(eta.loc[i, j])
        if product > EPSILON:
            result[i_position[i], j_position[j]] = product
    return result


def sequential_cmap(colors: list[str], bad: str = "#F3F3F1") -> mcolors.Colormap:
    cmap = mcolors.LinearSegmentedColormap.from_list("muted", colors)
    cmap.set_bad(bad)
    return cmap


def finite_norm(matrices: list[np.ndarray]) -> mcolors.Normalize:
    values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    if values.size == 0:
        return mcolors.Normalize(0, 1)
    low, high = float(values.min()), float(values.max())
    if low == high:
        high = low + 1
    return mcolors.Normalize(low, high)


def bubble_sizes(values: np.ndarray, maximum: float) -> np.ndarray:
    return 80 + 1500 * np.log10(1 + values) / np.log10(1 + maximum)


def format_production(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.2f}"


def draw(
    datasets: list[dict[str, object]],
    production: dict[tuple[str, str], np.ndarray],
    output: Path,
    dpi: int,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 14})
    row_counts = [len(data["fl_labels"]) for data in datasets]
    fig, axes = plt.subplots(
        3, 3, figsize=(27, 25),
        gridspec_kw={"height_ratios": row_counts},
        constrained_layout=False,
    )

    cmaps = [
        sequential_cmap(["#F1F5F4", "#A7C5BE", "#4F858C", "#245B66"]),
        sequential_cmap(["#F4F3EC", "#B2BEA3", "#81997B", "#3F624D"]),
        sequential_cmap(["#F7F1EE", "#E6BFA9", "#D59C80", "#A86F58"]),
    ]
    norms = [
        finite_norm([data["matrices"][column] for data in datasets])
        for column in range(3)
    ]
    max_production = max(matrix.max() for matrix in production.values())

    for row, ((region_code, region_name), data) in enumerate(zip(REGIONS, datasets)):
        ch_labels = data["ch_labels"]
        fl_labels = data["fl_labels"]
        n_rows, n_cols = len(fl_labels), len(ch_labels)
        for column in range(3):
            ax = axes[row, column]
            matrix = np.ma.masked_invalid(data["matrices"][column])
            ax.imshow(
                matrix, cmap=cmaps[column], norm=norms[column],
                aspect="auto", interpolation="nearest", origin="upper",
            )
            ax.set_xticks(np.arange(n_cols), ch_labels)
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", rotation=52, labelsize=13.0, pad=5)
            ax.set_yticks(np.arange(n_rows), fl_labels)
            ax.tick_params(axis="y", labelsize=13.0)
            ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_color("#666A6D")
                spine.set_linewidth(0.75)

            if column in (1, 2):
                objective = "Env" if column == 1 else "Eco"
                bubbles = production[(region_code, objective)]
                bubble_y, bubble_x = np.nonzero(bubbles > EPSILON)
                amounts = bubbles[bubble_y, bubble_x]
                ax.scatter(
                    bubble_x, bubble_y,
                    s=bubble_sizes(amounts, max_production),
                    color=BUBBLE_COLOR, alpha=0.58,
                    edgecolor="white", linewidth=0.9, zorder=4,
                )
                for x, y, amount in zip(bubble_x, bubble_y, amounts):
                    ax.text(
                        x, y, format_production(amount),
                        ha="center", va="center", fontsize=10.5,
                        color="#172A35", fontweight="semibold", zorder=5,
                    )

            if column == 0:
                ax.text(
                    -0.24, 0.5, region_name, transform=ax.transAxes,
                    rotation=90, ha="center", va="center",
                    fontsize=20, fontweight="semibold", color="#303438",
                )

    # Shared, right-sided heatmap color scales.
    colorbar_positions = [0.70, 0.46, 0.22]
    colorbar_labels = [
        "Yield (eta)",
        r"log$_{10}$(yield × GWP saving)",
        r"log$_{10}$(yield × economic margin)",
    ]
    for column, (bottom, label) in enumerate(zip(colorbar_positions, colorbar_labels)):
        color_ax = fig.add_axes([0.925, bottom, 0.014, 0.16])
        colorbar = fig.colorbar(
            ScalarMappable(norm=norms[column], cmap=cmaps[column]), cax=color_ax
        )
        colorbar.set_label(label, fontsize=14)
        colorbar.ax.tick_params(labelsize=12)
        colorbar.outline.set_linewidth(0.7)

    # Column captions are deliberately placed below the final row.
    for column, caption in enumerate(COLUMN_CAPTIONS):
        position = axes[2, column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2, 0.025, caption,
            ha="center", va="center", fontsize=19, fontweight="semibold",
            color="#303438",
        )

    reference_production = [1, 100, 1000]
    bubble_handles = [
        axes[2, 2].scatter(
            [], [], s=bubble_sizes(np.array([amount]), max_production)[0],
            color=BUBBLE_COLOR, alpha=0.58, edgecolor="white", linewidth=0.9,
        )
        for amount in reference_production
    ]
    fig.legend(
        bubble_handles, [f"{amount:,}" for amount in reference_production],
        title="Pathway production (kt)", frameon=False,
        loc="lower left", bbox_to_anchor=(0.915, 0.055),
        fontsize=12.5, title_fontsize=14, labelspacing=1.2,
    )
    fig.subplots_adjust(left=0.14, right=0.89, top=0.95, bottom=0.065, wspace=0.32, hspace=0.40)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Plot FL/FLB-CH heatmaps and optimized pathway bubbles")
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Scenario-specific model-input directory; defaults to data/model_inputs/<scenario>.",
    )
    parser.add_argument(
        "--results", type=Path,
        default=repo_root / "figures" / "input_data" / "optimization_results.xlsx",
    )
    parser.add_argument(
        "--scenario", choices=["FL", "FLB"], default="FL",
        help="Use FL-only inputs or the broader FLB inputs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    scenario = args.scenario.upper()
    output = (
        args.output.resolve() if args.output
        else repo_root / "figures" / "main" / f"{scenario.casefold()}_ch_heatmap_bubbles.png"
    )
    input_dir = (
        args.input_dir.resolve()
        if args.input_dir
        else repo_root / "data" / "model_inputs" / scenario
    )
    datasets = [
        load_region(input_dir, code, scenario)
        for code, _ in REGIONS
    ]
    production = {}
    for (region_code, _), data in zip(REGIONS, datasets):
        for objective in ("Env", "Eco"):
            sheet = f"{region_code}_{scenario}_{objective}"
            production[(region_code, objective)] = pathway_production(
                args.results.resolve(), sheet, data["eta"]
            )
    draw(datasets, production, output, args.dpi)
    for suffix in (".png", ".pdf", ".svg"):
        print(f"Created: {output.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
