"""Plot stacked economic and environmental benefits from optimization analysis.

Two figures are created, one for FL and one for FLB.  Within each figure,
countries and optimized objectives are compared in separate economic and
environmental panels.  Each bar is stacked by platform chemical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

from .figure_style import CHEMICAL_COLORS, MUTED_ACADEMIC_PALETTE


REQUIRED_COLUMNS = {
    "Region",
    "Input tag",
    "Optimized objective",
    "Chemical name",
    "Economic contribution",
    "Environmental contribution",
}
REGION_ORDER = ("EU", "US", "CN")  # barh bottom-to-top -> CN, US, EU
OBJECTIVE_ORDER = ("Env", "Eco")
METRICS = (
    ("Environmental contribution", "GWP saving"),
    ("Economic contribution", "Potential economic margin"),
)


def read_contributions(workbook: Path) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name="Path contributions")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{workbook.name}/Path contributions is missing columns: "
            f"{sorted(missing)}"
        )
    for column, _ in METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("Region", "Input tag", "Optimized objective", "Chemical name"):
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def ordered_values(found: pd.Series, preferred: tuple[str, ...]) -> list[str]:
    values = list(dict.fromkeys(found.dropna().astype(str)))
    return [item for item in preferred if item in values] + sorted(
        set(values).difference(preferred)
    )


def chemical_colors(chemicals: list[str]) -> dict[str, str]:
    """Use the exact platform-chemical colors from the chord figures."""
    result = {}
    fallback_index = 0
    for chemical in sorted(chemicals, key=str.casefold):
        key = " ".join(chemical.split()).upper()
        if key in CHEMICAL_COLORS:
            result[chemical] = CHEMICAL_COLORS[key]
        else:
            result[chemical] = MUTED_ACADEMIC_PALETTE[fallback_index]
            fallback_index += 1
    return result


def draw_stacked_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    regions: list[str],
    objective: str,
    chemicals: list[str],
    colors: dict[str, str],
) -> None:
    selected = data.loc[data["Optimized objective"] == objective]
    grouped = selected.groupby(["Region", "Chemical name"], observed=True)[metric].sum()
    y = np.arange(len(regions), dtype=float)
    positive_left = np.zeros(len(regions))
    negative_left = np.zeros(len(regions))

    for chemical in chemicals:
        values = np.array(
            [grouped.get((region, chemical), 0.0) for region in regions],
            dtype=float,
        )
        left = np.where(values >= 0, positive_left, negative_left)
        ax.barh(
            y,
            values,
            left=left,
            height=0.48,
            color=colors[chemical],
            edgecolor="white",
            linewidth=0.7,
        )
        positive_left += np.where(values >= 0, values, 0.0)
        negative_left += np.where(values < 0, values, 0.0)

    ax.set_yticks(y, regions)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.grid(axis="x", which="major", color="#E3E5E7", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#55585C")
        ax.spines[side].set_linewidth(0.75)


def plot_feedstock(
    frame: pd.DataFrame,
    input_tag: str,
    output_dir: Path,
    dpi: int,
    formats: list[str],
) -> list[Path]:
    data = frame.loc[frame["Input tag"].str.casefold() == input_tag.casefold()].copy()
    if data.empty:
        raise ValueError(f"No rows found for Input tag={input_tag!r}")

    regions = ordered_values(data["Region"], REGION_ORDER)
    objectives = ordered_values(data["Optimized objective"], OBJECTIVE_ORDER)
    chemicals = sorted(data["Chemical name"].unique(), key=str.casefold)
    colors = chemical_colors(chemicals)

    fig, axes = plt.subplots(
        len(objectives), len(METRICS), figsize=(13.8, 8.2), sharey=True,
        sharex="col",
        constrained_layout=False, squeeze=False,
    )
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for row, objective in enumerate(objectives):
        for column, (metric, column_title) in enumerate(METRICS):
            ax = axes[row, column]
            draw_stacked_bars(
                ax, data, metric, regions, objective, chemicals, colors
            )
            if row < len(objectives) - 1:
                ax.tick_params(axis="x", labelbottom=False)
                ax.xaxis.get_offset_text().set_visible(False)
            if row == len(objectives) - 1:
                unit_label = (
                    "GWP saving (kg CO$_2$-eq)" if metric.startswith("Environmental")
                    else "Potential economic margin (USD)"
                )
                ax.set_xlabel(unit_label, fontsize=10.5, labelpad=7, color="#303438")
            ax.text(
                -0.045, 1.025, panel_labels[row * len(METRICS) + column],
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
                fontweight="semibold", color="#303438",
                clip_on=False,
            )

    fig.suptitle(
        f"{input_tag} feedstock",
        fontsize=15,
        fontweight="semibold",
        color="#303438",
        y=0.965,
    )
    legend = [Patch(facecolor=colors[name], label=name) for name in chemicals]
    fig.legend(
        handles=legend,
        title="Platform chemical",
        loc="center left",
        bbox_to_anchor=(0.875, 0.5),
        ncol=1,
        frameon=False,
    )
    legend_object = fig.legends[-1]
    legend_object.get_title().set_fontweight("semibold")
    fig.supylabel("Region", x=0.025, fontsize=12)
    fig.subplots_adjust(
        left=0.082, right=0.85, top=0.76, bottom=0.14, wspace=0.07, hspace=0.34
    )

    # Plain column and row headings replace the former grey facet strips.
    for column, (_, heading) in enumerate(METRICS):
        position = axes[0, column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2, 0.845, heading,
            ha="center", va="center", fontsize=12, fontweight="semibold",
            color="#303438",
        )
    row_headings = (
        "Carbon-optimized allocation",
        "Economic-optimized allocation",
    )
    for row, heading in enumerate(row_headings):
        position = axes[row, 0].get_position()
        fig.text(
            position.x0, position.y1 + 0.038, heading,
            ha="left", va="bottom", fontsize=11, fontweight="semibold",
            color="#303438",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in formats:
        output = output_dir / f"optimization_benefits_{input_tag}.{extension}"
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(output)
    plt.close(fig)
    return outputs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create FL and FLB stacked benefit charts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "figures" / "input_data" / "optimization_analysis.xlsx",
        help="Optimization analysis workbook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures" / "supplementary" / "optimization_analysis",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"]
    )
    args = parser.parse_args()

    frame = read_contributions(args.input.resolve())
    outputs = []
    for input_tag in ("FL", "FLB"):
        outputs.extend(
            plot_feedstock(
                frame, input_tag, args.output_dir.resolve(), args.dpi, args.formats
            )
        )
    print("Created:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
