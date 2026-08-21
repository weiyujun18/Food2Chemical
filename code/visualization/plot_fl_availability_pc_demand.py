"""Plot regional FL, food by-product availability, and PC demand."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Wedge
from matplotlib.ticker import MultipleLocator


REGIONS = {
    "China": "CN",
    "United States": "US",
    "Europe": "EU",
}

# Exact colors from the established muted chord-chart palette.
FL_COLORS = {
    "Apple": "#334E68",
    "Banana": "#627D98",
    "Barley": "#9FB3C8",
    "Corn": "#245B66",
    "Orange": "#98AA8C",
    "Potato": "#646B4F",
    "Rapeseed": "#9E9768",
    "Rice grain": "#8C6D32",
    "Sugar beet": "#D9BD8B",
    "Sugarcane": "#C98C70",
    "Sweet potato": "#E6BFA9",
    "Tomato": "#BB7470",
    "Wheat": "#D3948E",
}
BUBBLE_COLOR = "#4F858C"
REGION_ORDER = ["Europe", "United States", "China"]  # bottom-to-top
BYPRODUCT_REGION_COLORS = {
    "China": "#245B66",
    "United States": "#A86F58",
    "Europe": "#81997B",
}
BYPRODUCT_PALETTE = [
    "#243B53", "#486581", "#829AB1", "#245B66", "#4F858C", "#86ADAA",
    "#3F624D", "#6B876C", "#98AA8C", "#646B4F", "#8B895D", "#B0A576",
    "#8C6D32", "#B58F4A", "#D0AD70", "#A86F58", "#C98C70", "#DEAD94",
    "#A96562", "#C9827D", "#DDA6A1", "#8E6874", "#B28992", "#CEACB0",
]


def clean_fl_name(value: object) -> str:
    key = " ".join(str(value).strip().casefold().split())
    aliases = {
        "corn": "Corn",
        "rice grain": "Rice grain",
        "whole wheat": "Wheat",
        "wheat": "Wheat",
        "sugarcane": "Sugarcane",
        "potato": "Potato",
        "potatoe": "Potato",
        "tomato": "Tomato",
        "sweet potato": "Sweet potato",
        "apple": "Apple",
        "orange": "Orange",
        "banana": "Banana",
        "sugar beet": "Sugar beet",
        "barley": "Barley",
        "rapeseed": "Rapeseed",
    }
    return aliases.get(key, str(value).strip().title())


def read_value_label_sheet(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, index_col=0)
    required = {"Column1", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name}/{sheet}: missing columns {sorted(missing)}; "
            f"actual={list(frame.columns)}"
        )
    result = frame[["Column1", "label"]].copy()
    result["Column1"] = pd.to_numeric(result["Column1"], errors="raise")
    if result["Column1"].isna().any() or (result["Column1"] < 0).any():
        raise ValueError(f"{path.name}/{sheet}: values must be finite and non-negative")
    result["label"] = result["label"].astype(str).str.strip()
    return result


def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fl_rows, demand_rows = [], []
    chemical_order: list[str] | None = None
    for region_name, region_code in REGIONS.items():
        path = input_dir / f"FL2CH_{region_code}_FL.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Input workbook not found: {path}")
        fl = read_value_label_sheet(path, "FL")
        ch = read_value_label_sheet(path, "CH")
        current_order = ch["label"].tolist()
        if chemical_order is None:
            chemical_order = current_order
        elif set(current_order) != set(chemical_order):
            raise ValueError(f"{path.name}/CH: chemical labels differ between regions")

        for row in fl.itertuples(index=False):
            fl_rows.append(
                {
                    "region": region_name,
                    "feedstock": clean_fl_name(row.label),
                    "availability_mt": float(row.Column1) / 1000.0,
                }
            )
        for row in ch.itertuples(index=False):
            demand_rows.append(
                {
                    "region": region_name,
                    "chemical": row.label,
                    "demand_kt": float(row.Column1),
                }
            )
    return pd.DataFrame(fl_rows), pd.DataFrame(demand_rows), chemical_order or []


def load_byproduct_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Food by-product workbook not found: {path}")
    frame = pd.read_excel(path, sheet_name="Sheet1")
    required = {"region", "byproduct", "quantity", "percent_of_total"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    aliases = {
        "china": "China", "theu.s.": "United States", "the u.s.": "United States",
        "us": "United States", "the eu": "Europe", "eu": "Europe", "europe": "Europe",
    }
    result = frame[list(required)].copy()
    result["region"] = (
        result["region"].astype(str).str.strip().str.casefold().map(aliases)
    )
    result["byproduct"] = result["byproduct"].astype(str).str.strip()
    result["quantity"] = pd.to_numeric(result["quantity"], errors="raise")
    result["percent_of_total"] = pd.to_numeric(result["percent_of_total"], errors="raise")
    result = result.dropna(subset=["region", "byproduct", "quantity", "percent_of_total"])
    if (result["quantity"] < 0).any():
        raise ValueError(f"{path.name}: quantity must be non-negative")
    return result


def draw_byproduct_radial_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw all three regions in one sunburst, matching the chapter 3 chart."""
    inner_radius, region_radius = 0.18, 0.43
    product_radius, bar_start_radius, max_bar_height = 0.66, 0.76, 0.55
    total_percent = data["percent_of_total"].sum()
    region_totals = data.groupby("region")["percent_of_total"].sum()
    scale_max = np.ceil(data["quantity"].max() / 10000) * 10000
    region_order = ["China", "United States", "Europe"]
    region_labels = {
        "China": "China",
        "United States": "The U.S.",
        "Europe": "The EU",
    }

    # Quantitative circular grid/tick axis.
    ticks = np.arange(10000, scale_max + 10000, 10000)
    tick_angle = np.radians(72)
    for tick in ticks:
        radius = bar_start_radius + tick / scale_max * max_bar_height
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor="#D9E0E3",
                                linewidth=1.0, linestyle=(0, (3, 3)), zorder=0))
        ax.text(radius * np.cos(tick_angle) + 0.025, radius * np.sin(tick_angle),
                f"{tick / 1000:.0f}k", ha="left", va="center", fontsize=11.5,
                color="#59636A",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
                      "pad": 1.5}, zorder=7)
    ax.text(1.62, 1.47, "FLB quantity\n" + r"(kt wet mass yr$^{-1}$)",
            ha="left", va="top", linespacing=1.05, fontsize=10.5,
            fontweight="semibold", color="#303438")

    start_angle = 90.0
    stream_labels: list[tuple[float, float, str]] = []
    for region in region_order:
        subset = data[data["region"] == region].copy()
        if subset.empty:
            continue
        region_angle = 360.0 * float(region_totals.loc[region]) / total_percent
        color = BYPRODUCT_REGION_COLORS[region]
        ax.add_patch(Wedge((0, 0), region_radius, start_angle, start_angle + region_angle,
                           width=region_radius - inner_radius, facecolor=color,
                           alpha=0.95, edgecolor="white", linewidth=2.2, zorder=2))
        middle = np.radians(start_angle + region_angle / 2)
        rr = (inner_radius + region_radius) / 2
        ax.text(rr * np.cos(middle), rr * np.sin(middle), region_labels[region],
                ha="center", va="center", fontsize=11.5, fontweight="semibold",
                color="white", zorder=6)
        product_start = start_angle
        for row in subset.itertuples(index=False):
            product_angle = 360.0 * float(row.percent_of_total) / total_percent
            product_end = product_start + product_angle
            mid_deg = product_start + product_angle / 2
            ax.add_patch(Wedge((0, 0), product_radius, product_start, product_end,
                               width=product_radius - region_radius, facecolor=color,
                               alpha=0.58, edgecolor="white", linewidth=1.2, zorder=2))
            height = float(row.quantity) / scale_max * max_bar_height
            gap = min(product_angle * 0.12, 0.7)
            ax.add_patch(Wedge((0, 0), bar_start_radius + height,
                               product_start + gap, product_end - gap, width=height,
                               facecolor=color, alpha=0.90, edgecolor="white",
                               linewidth=0.75, zorder=3))
            if float(row.percent_of_total) >= 0.5:
                label_radius = bar_start_radius + height + 0.035
                stream_labels.append((mid_deg, label_radius, str(row.byproduct)))
            product_start = product_end
        start_angle += region_angle

    # Spread labels belonging to adjacent narrow streams.  The wedges stay at
    # their true angles; only the text anchors are nudged to prevent collisions.
    previous_label_angle: float | None = None
    minimum_label_gap = 5.5
    for mid_deg, label_radius, label in stream_labels:
        label_angle_deg = mid_deg
        if previous_label_angle is not None:
            label_angle_deg = max(label_angle_deg, previous_label_angle + minimum_label_gap)
        previous_label_angle = label_angle_deg
        label_angle = np.radians(label_angle_deg)
        x = label_radius * np.cos(label_angle)
        y = label_radius * np.sin(label_angle)
        left_half = 90 < label_angle_deg % 360 < 270
        ax.text(x, y, label, ha="right" if left_half else "left", va="center",
                rotation=label_angle_deg + 180 if left_half else label_angle_deg,
                rotation_mode="anchor", fontsize=11.0, fontweight="semibold",
                color="#303438",
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.78}, zorder=8)

    ax.add_patch(plt.Circle((0, 0), inner_radius, facecolor="white",
                            edgecolor="#697277", linewidth=1.0, zorder=5))
    ax.text(0, 0, "Food\nbyproduct", ha="center", va="center", fontsize=11.5,
            linespacing=1.05, fontweight="semibold", color="#303438", zorder=6)
    handles = [
        Patch(facecolor=BYPRODUCT_REGION_COLORS[r], label=region_labels[r])
        for r in region_order
    ]
    legend = ax.legend(handles=handles, title="Region", loc="center left",
                       bbox_to_anchor=(0.83, 0.5), frameon=False, fontsize=12.5,
                       title_fontsize=13, labelspacing=0.9)
    legend.get_title().set_fontweight("semibold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.62, 2.30)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")


def draw_byproduct_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw food by-product availability as a horizontal stacked bar chart."""
    byproducts = data["byproduct"].drop_duplicates().tolist()
    colors = {
        name: BYPRODUCT_PALETTE[index % len(BYPRODUCT_PALETTE)]
        for index, name in enumerate(byproducts)
    }
    pivot = data.pivot_table(
        index="region", columns="byproduct", values="quantity",
        aggfunc="sum", fill_value=0,
    )
    y = np.arange(len(REGION_ORDER))
    left = np.zeros(len(REGION_ORDER))
    for name in byproducts:
        values = np.array([
            pivot.loc[region, name] if region in pivot.index and name in pivot.columns else 0
            for region in REGION_ORDER
        ])
        ax.barh(y, values, left=left, height=0.58, color=colors[name],
                edgecolor="white", linewidth=0.8)
        left += values

    maximum = left.max()
    for yi, total in zip(y, left):
        ax.text(total + maximum * 0.012, yi, f"{total:,.0f}", ha="left", va="center",
                color="#303438", fontsize=12)
    ax.set_yticks(y, REGION_ORDER)
    ax.set_xlabel(r"Food by-product availability (kt wet mass yr$^{-1}$)", fontsize=15)
    ax.set_xlim(0, maximum * 1.12)
    major_step = 20000 if maximum >= 80000 else 10000
    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.grid(axis="x", color="#E3E5E7", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=colors[name], label=name) for name in byproducts]
    ax.legend(handles=handles, title="Food by-product type", frameon=False,
              ncol=2, loc="center left", bbox_to_anchor=(1.015, 0.5),
              columnspacing=1.0, handlelength=1.2, fontsize=10.5,
              title_fontsize=12.5, borderaxespad=0)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#55585C")
        ax.spines[side].set_linewidth(0.8)


def draw_single_region_sunburst(
    ax: plt.Axes,
    data: pd.DataFrame,
    region: str,
    scale_max: float,
    show_scale: bool = False,
) -> None:
    """Draw one regional food by-product sunburst using a shared radial scale."""
    subset = data[data["region"] == region].copy()
    total = subset["quantity"].sum()
    color = BYPRODUCT_REGION_COLORS[region]
    inner_radius, product_radius = 0.27, 0.67
    bar_start_radius, max_bar_height = 0.75, 0.47

    ticks = np.arange(20000, scale_max + 1, 20000)
    for tick in ticks:
        radius = bar_start_radius + tick / scale_max * max_bar_height
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor="#D9E0E3",
                                linewidth=0.65, linestyle=(0, (3, 3)), zorder=0))
        if show_scale:
            angle = np.radians(77)
            ax.text(radius * np.cos(angle) + 0.012, radius * np.sin(angle),
                    f"{tick / 1000:.0f}k", ha="left", va="center", fontsize=10.5,
                    color="#59636A",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.05})

    start_angle = 90.0
    for row in subset.itertuples(index=False):
        angle_size = 360.0 * float(row.quantity) / total
        end_angle = start_angle + angle_size
        mid_deg = start_angle + angle_size / 2
        mid = np.radians(mid_deg)
        ax.add_patch(Wedge((0, 0), product_radius, start_angle, end_angle,
                           width=product_radius - inner_radius, facecolor=color,
                           alpha=0.58, edgecolor="white", linewidth=1.0, zorder=2))
        height = float(row.quantity) / scale_max * max_bar_height
        gap = min(angle_size * 0.12, 0.8)
        ax.add_patch(Wedge((0, 0), bar_start_radius + height,
                           start_angle + gap, end_angle - gap, width=height,
                           facecolor=color, alpha=0.90, edgecolor="white",
                           linewidth=0.7, zorder=3))
        if float(row.percent_of_total) >= 0.5:
            label_radius = bar_start_radius + height + 0.035
            x, y = label_radius * np.cos(mid), label_radius * np.sin(mid)
            left_half = 90 < mid_deg % 360 < 270
            ax.text(x, y, str(row.byproduct),
                    ha="right" if left_half else "left", va="center",
                    rotation=mid_deg + 180 if left_half else mid_deg,
                    rotation_mode="anchor", fontsize=13, fontweight="semibold",
                    color="#303438", zorder=8)
        start_angle = end_angle

    ax.add_patch(plt.Circle((0, 0), inner_radius, facecolor=color,
                            edgecolor="white", linewidth=1.0, zorder=5))
    display_region = "United\nStates" if region == "United States" else region
    ax.text(0, 0, display_region, ha="center", va="center", fontsize=13,
            linespacing=1.0, fontweight="semibold", color="white", zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.43, 1.43)
    ax.set_ylim(-1.43, 1.43)
    ax.axis("off")


def format_demand(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.2f}"


def bubble_area(values: np.ndarray, maximum: float) -> np.ndarray:
    # Matplotlib scatter uses points squared. Log scaling keeps small demands
    # visible while preserving monotonic area differences across four orders.
    return 45.0 + 2350.0 * np.log10(1.0 + values) / np.log10(1.0 + maximum)


def save_standalone_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    output: Path,
    dpi: int,
) -> None:
    """Save one panel, including its title, labels, annotations, and legend."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.padded(0.12)
    fig.savefig(output, dpi=dpi, bbox_inches=bbox, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches=bbox, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches=bbox, facecolor="white")


def make_figure(
    fl: pd.DataFrame,
    demand: pd.DataFrame,
    byproduct: pd.DataFrame,
    chemical_order: list[str],
    output: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13.0,
            "axes.labelcolor": "#303438",
            "xtick.color": "#404448",
            "ytick.color": "#404448",
        }
    )
    fig = plt.figure(figsize=(21.5, 19.0), facecolor="white")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.30, 1.10], hspace=0.29)
    ax_fl = fig.add_subplot(grid[0])
    ax_byproduct_frame = fig.add_subplot(grid[1])
    ax_byproduct_frame.axis("off")
    ax_byproduct = ax_byproduct_frame.inset_axes([0, 0, 1, 1])
    ax_pc = fig.add_subplot(grid[2])

    # Panel (a): horizontal stacked bars.
    feedstocks = [name for name in FL_COLORS if name in set(fl["feedstock"])]
    y = np.arange(len(REGION_ORDER))
    left = np.zeros(len(REGION_ORDER))
    pivot = fl.pivot_table(
        index="region", columns="feedstock", values="availability_mt",
        aggfunc="sum", fill_value=0,
    )
    for feedstock in feedstocks:
        values = np.array(
            [pivot.loc[region, feedstock] if feedstock in pivot.columns else 0 for region in REGION_ORDER]
        )
        ax_fl.barh(
            y, values, left=left, height=0.58,
            color=FL_COLORS[feedstock], edgecolor="white", linewidth=0.8,
        )
        left += values
    for yi, total in zip(y, left):
        ax_fl.text(
            total + left.max() * 0.012, yi, f"{total:.1f}",
            ha="left", va="center", color="#303438", fontsize=12.0,
        )
    ax_fl.set_yticks(y, REGION_ORDER)
    ax_fl.set_xlabel(r"FL availability (Mt wet mass yr$^{-1}$)", fontsize=15)
    ax_fl.set_title("a. Food loss availability", loc="left", fontweight="semibold", fontsize=17, pad=12)
    ax_fl.set_xlim(0, left.max() * 1.12)
    ax_fl.xaxis.set_major_locator(MultipleLocator(10))
    ax_fl.grid(axis="x", color="#E3E5E7", linewidth=0.65, alpha=0.85)
    ax_fl.set_axisbelow(True)
    legend_handles = [Patch(facecolor=FL_COLORS[name], label=name) for name in feedstocks]
    ax_fl.legend(
        handles=legend_handles, title="Food-loss type", frameon=False,
        ncol=1, loc="center left", bbox_to_anchor=(1.015, 0.5),
        columnspacing=1.3, handlelength=1.2, fontsize=11.5,
        title_fontsize=12.5, borderaxespad=0,
    )

    # Panel (b): food by-product availability.
    draw_byproduct_radial_panel(ax_byproduct, byproduct)
    ax_byproduct_frame.set_title("b. Food by-product availability", loc="left",
                                 fontweight="semibold", fontsize=17, pad=12)

    # Panel (c): PC-demand bubble matrix.
    chemical_x = {chemical: index for index, chemical in enumerate(chemical_order)}
    region_y = {region: index for index, region in enumerate(REGION_ORDER)}
    x_values = demand["chemical"].map(chemical_x).to_numpy()
    y_values = demand["region"].map(region_y).to_numpy()
    demand_values = demand["demand_kt"].to_numpy(dtype=float)
    max_demand = demand_values.max()
    ax_pc.scatter(
        x_values, y_values,
        s=bubble_area(demand_values, max_demand),
        color=BUBBLE_COLOR, alpha=0.78,
        edgecolor="white", linewidth=0.8,
    )
    for x_value, y_value, amount in zip(x_values, y_values, demand_values):
        ax_pc.annotate(
            format_demand(amount), (x_value, y_value), xytext=(0, -28.35),
            textcoords="offset points", ha="center", va="top",
            fontsize=10.5, color="#303438",
        )
    display_labels = [chemical.replace(" ", "\n") for chemical in chemical_order]
    ax_pc.set_xticks(range(len(chemical_order)), display_labels)
    ax_pc.set_yticks(range(len(REGION_ORDER)), REGION_ORDER)
    ax_pc.set_xlim(-0.65, len(chemical_order) - 0.35)
    ax_pc.set_ylim(-0.6, len(REGION_ORDER) - 0.25)
    ax_pc.set_xlabel("Platform chemical", fontsize=15)
    ax_pc.set_ylabel("Region", fontsize=15)
    ax_pc.set_title("c. Platform-chemical demand", loc="left", fontweight="semibold", fontsize=17, pad=14)
    ax_pc.grid(color="#E3E5E7", linewidth=0.65, alpha=0.8)
    ax_pc.set_axisbelow(True)

    reference_values = [1, 10, 100, 1000, 5000]
    size_handles = [
        ax_pc.scatter([], [], s=bubble_area(np.array([v]), max_demand)[0],
                      color=BUBBLE_COLOR, alpha=0.78, edgecolor="white", linewidth=0.8)
        for v in reference_values
    ]
    ax_pc.legend(
        size_handles, [f"{v:,}" for v in reference_values],
        title=r"PC demand (kt yr$^{-1}$)", frameon=False,
        loc="center left", bbox_to_anchor=(1.015, 0.5),
        labelspacing=1.15, borderaxespad=0, fontsize=11.5,
        title_fontsize=12.5,
    )

    for ax in (ax_fl, ax_pc):
        ax.tick_params(axis="both", labelsize=12.5)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#55585C")
            ax.spines[side].set_linewidth(0.8)

    fig.subplots_adjust(left=0.10, right=0.84, top=0.975, bottom=0.055)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    pdf_output = output.with_suffix(".pdf")
    try:
        fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")
    except PermissionError:
        pdf_output = output.with_name(f"{output.stem}_updated.pdf")
        fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")
        print(f"Original PDF is open; wrote updated PDF to: {pdf_output}")

    # Also export panels (a) and (c) as self-contained figures so they can be
    # assembled manually with an independently prepared panel (b).
    panel_a_output = output.with_name(f"{output.stem}_panel_a.png")
    panel_c_output = output.with_name(f"{output.stem}_panel_c.png")
    save_standalone_panel(fig, ax_fl, panel_a_output, dpi)
    save_standalone_panel(fig, ax_pc, panel_c_output, dpi)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Plot FL availability and PC demand")
    parser.add_argument(
        "--input-dir", type=Path,
        default=repo_root / "data" / "model_inputs" / "FL",
    )
    parser.add_argument(
        "--byproduct-input", type=Path,
        default=repo_root / "figures" / "input_data" / "FLB_Qty_Sunburst.xlsx",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "figures" / "main" / "fl_byproduct_availability_pc_demand.png",
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    fl, demand, chemical_order = load_data(args.input_dir.resolve())
    byproduct = load_byproduct_data(args.byproduct_input.resolve())
    make_figure(fl, demand, byproduct, chemical_order, args.output.resolve(), args.dpi)
    print(f"Created: {args.output.resolve()}")
    print(f"Created: {args.output.resolve().with_suffix('.svg')}")
    print(f"Created standalone panel a: {args.output.resolve().with_name(args.output.stem + '_panel_a.png')}")
    print(f"Created standalone panel c: {args.output.resolve().with_name(args.output.stem + '_panel_c.png')}")


if __name__ == "__main__":
    main()
