"""Create one FL-to-CH chord chart for every optimization result sheet.

The allocation ``w[i, j]`` is drawn from each FL node to each CH node.  The
remaining amount, ``Available_FL - Used_x``, is drawn to an ``Unused FL`` node.
Input labels are read from the ``label`` columns of the source workbook named
in each result sheet's Summary section.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path as MplPath

from .figure_style import CHEMICAL_COLORS, MUTED_ACADEMIC_PALETTE


EPSILON = 1e-8
UNUSED_LABEL = "Unused FL"

# Fixed, low-saturation academic palette.  The order deliberately moves from
# navy/blue through teal/green/olive/ochre to peach/salmon/blush.  No automatic
# Matplotlib colormap or randomly generated color is used.
def _find_row(raw: pd.DataFrame, text: str) -> int:
    matches = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq(text)]
    if len(matches) != 1:
        raise ValueError(f"Expected one '{text}' section, found {len(matches)}")
    return int(matches[0])


def _table_after(
    raw: pd.DataFrame, section: str, row_count: int, columns: list[str]
) -> pd.DataFrame:
    section_row = _find_row(raw, section)
    header_row = section_row + 1
    table = raw.iloc[header_row + 1 : header_row + 1 + row_count, : len(columns)].copy()
    table.columns = columns
    return table.reset_index(drop=True)


def read_result_sheet(
    workbook: Path, sheet_name: str
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
    summary_row = _find_row(raw, "SUMMARY")
    summary_table = raw.iloc[summary_row + 2 : summary_row + 11, :2].dropna(how="all")
    summary = dict(zip(summary_table.iloc[:, 0], summary_table.iloc[:, 1]))
    number_i, number_j = int(summary["Number of I"]), int(summary["Number of J"])

    x = _table_after(
        raw,
        "X - feedstock use",
        number_i,
        ["i", "Available_FL", "Used_x"],
    )
    w = _table_after(
        raw,
        "W - allocation",
        number_i * number_j,
        ["i", "j", "Allocated_w"],
    )
    for frame, columns in ((x, x.columns), (w, w.columns)):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return summary, x, w


def read_labels(input_file: Path) -> tuple[dict[Any, str], dict[Any, str]]:
    labels: dict[str, dict[Any, str]] = {}
    for sheet in ("FL", "CH"):
        frame = pd.read_excel(input_file, sheet_name=sheet, index_col=0)
        if "label" not in frame.columns:
            raise ValueError(f"{input_file.name}/{sheet}: missing 'label' column")
        series = frame["label"].dropna()
        labels[sheet] = {index: str(label).strip() for index, label in series.items()}
    return labels["FL"], labels["CH"]


def make_links(
    x: pd.DataFrame,
    w: pd.DataFrame,
    fl_labels: dict[Any, str],
    ch_labels: dict[Any, str],
) -> tuple[list[str], list[str], list[tuple[str, str, float]]]:
    fl_nodes = [f"FL | {fl_labels.get(i, f'FL {i:g}')}" for i in x["i"]]
    j_values = list(dict.fromkeys(w["j"].tolist()))
    ch_nodes = [f"CH | {ch_labels.get(j, f'CH {j:g}')}" for j in j_values]
    fl_node = dict(zip(x["i"], fl_nodes))
    ch_node = dict(zip(j_values, ch_nodes))

    links: list[tuple[str, str, float]] = []
    for row in w.itertuples(index=False):
        amount = max(0.0, float(row.Allocated_w))
        if amount > EPSILON:
            links.append((fl_node[row.i], ch_node[row.j], amount))

    for row in x.itertuples(index=False):
        unused = max(0.0, float(row.Available_FL) - float(row.Used_x))
        if unused > EPSILON:
            links.append((fl_node[row.i], UNUSED_LABEL, unused))
    return fl_nodes, ch_nodes + [UNUSED_LABEL], links


def _label_key(node: str) -> str:
    """Normalize a visible label so its color is stable across regions."""
    label = node.split(" | ", 1)[-1]
    return " ".join(label.split()).casefold()


def build_global_colors(all_nodes: list[str]) -> dict[str, str]:
    """Assign the fixed muted palette consistently across all charts."""
    keys = sorted({_label_key(node) for node in all_nodes if node != UNUSED_LABEL})
    colors = {
        key: MUTED_ACADEMIC_PALETTE[index % len(MUTED_ACADEMIC_PALETTE)]
        for index, key in enumerate(keys)
    }
    # Reuse the explicit shared dictionary. These values equal the colors
    # produced by the original global mapping, so chord styling is unchanged.
    colors.update({name.casefold(): color for name, color in CHEMICAL_COLORS.items()})
    return colors


def _polar(angle: float, radius: float) -> np.ndarray:
    return np.array([radius * math.cos(angle), radius * math.sin(angle)])


def _ribbon_path(
    source_angles: tuple[float, float],
    target_angles: tuple[float, float],
    radius: float,
) -> MplPath:
    s0, s1 = source_angles
    t0, t1 = target_angles
    source_arc = np.linspace(s0, s1, 7)
    target_arc = np.linspace(t0, t1, 7)
    vertices = [_polar(source_arc[0], radius)]
    codes = [MplPath.MOVETO]
    vertices.extend(_polar(a, radius) for a in source_arc[1:])
    codes.extend([MplPath.LINETO] * (len(source_arc) - 1))

    # Two cubic Bezier curves pull the ribbon through the centre.
    vertices.extend(
        [np.zeros(2), np.zeros(2), _polar(target_arc[0], radius)]
    )
    codes.extend([MplPath.CURVE4] * 3)
    vertices.extend(_polar(a, radius) for a in target_arc[1:])
    codes.extend([MplPath.LINETO] * (len(target_arc) - 1))
    vertices.extend(
        [np.zeros(2), np.zeros(2), _polar(source_arc[0], radius)]
    )
    codes.extend([MplPath.CURVE4] * 3)
    vertices.append(_polar(source_arc[0], radius))
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(np.asarray(vertices), codes)


def draw_chord(
    fl_nodes: list[str],
    target_nodes: list[str],
    links: list[tuple[str, str, float]],
    title: str,
    output_stem: Path,
    dpi: int,
    formats: list[str],
    global_colors: dict[str, str],
) -> None:
    nodes = fl_nodes + target_nodes
    totals = {node: 0.0 for node in nodes}
    for source, target, amount in links:
        totals[source] += amount
        totals[target] += amount
    nodes = [node for node in nodes if totals[node] > EPSILON]
    fl_nodes = [node for node in fl_nodes if node in nodes]
    target_nodes = [node for node in target_nodes if node in nodes]

    # Two visible gaps distinguish FL sources from CH/Unused destinations.
    gap = math.radians(1.25)
    group_gap = math.radians(7.0)
    usable = 2 * math.pi - gap * (len(nodes) - 2) - 2 * group_gap
    scale = usable / sum(totals[node] for node in nodes)
    angles: dict[str, tuple[float, float]] = {}
    cursor = math.radians(95)
    for group_index, group in enumerate((fl_nodes, target_nodes)):
        for node in group:
            start = cursor
            end = start + totals[node] * scale
            angles[node] = (start, end)
            cursor = end + gap
        cursor += 2 * group_gap - gap if group_index == 0 else 0

    fl_colors = {node: global_colors[_label_key(node)] for node in fl_nodes}
    target_colors = {
        node: (
            (0.72, 0.72, 0.72, 1.0)
            if node == UNUSED_LABEL
            else global_colors[_label_key(node)]
        )
        for node in target_nodes
    }
    node_colors = {**fl_colors, **target_colors}

    # Allocate a non-overlapping sub-arc to every link at both endpoints.
    offsets = {node: angles[node][0] for node in nodes}
    ribbon_segments = []
    for source, target, amount in sorted(links, key=lambda item: (item[0], item[1])):
        if source not in angles or target not in angles:
            continue
        source_segment = (offsets[source], offsets[source] + amount * scale)
        target_segment = (offsets[target], offsets[target] + amount * scale)
        offsets[source] = source_segment[1]
        offsets[target] = target_segment[1]
        ribbon_segments.append((source, target, source_segment, target_segment))

    fig, ax = plt.subplots(figsize=(24, 24), subplot_kw={"aspect": "equal"})
    for source, target, source_segment, target_segment in ribbon_segments:
        color = fl_colors[source]
        if target == UNUSED_LABEL:
            color = (0.58, 0.58, 0.58, 1.0)
        patch = PathPatch(
            _ribbon_path(source_segment, target_segment, radius=0.52),
            facecolor=color,
            edgecolor="none",
            alpha=0.43,
            zorder=1,
        )
        ax.add_patch(patch)

    label_midpoints = {
        node: (angles[node][0] + angles[node][1]) / 2 for node in nodes
    }

    for node in nodes:
        start, end = angles[node]
        ax.add_patch(
            Wedge(
                (0, 0),
                0.58,
                math.degrees(start),
                math.degrees(end),
                width=0.055,
                facecolor=node_colors[node],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        )
        label_angle = label_midpoints[node]
        x, y = _polar(label_angle, 0.66)
        label = node.split(" | ", 1)[-1]
        angle_degrees = math.degrees(label_angle) % 360
        if 90 < angle_degrees < 270:
            rotation, horizontal = angle_degrees + 180, "right"
        else:
            rotation, horizontal = angle_degrees, "left"
        ax.text(
            x,
            y,
            label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=horizontal,
            va="center",
            fontsize=23,
            fontweight="semibold",
            color="#222222",
        )

    ax.text(
        -1.46,
        1.34,
        "Food loss/by-product stream",
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
        color="#333333",
    )
    ax.text(
        1.46,
        1.34,
        "Platform chemical",
        fontsize=22,
        fontweight="bold",
        ha="right",
        va="top",
        color="#333333",
    )
    ax.set_title(title, fontsize=28, fontweight="bold", pad=42)
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")
    fig.tight_layout()
    for extension in formats:
        fig.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def locate_model_input(stored_path: str, input_tag: str, repo_root: Path) -> Path:
    """Resolve historical absolute paths against the repository inputs."""
    original = Path(stored_path)
    if original.is_file():
        return original
    candidate = repo_root / "data" / "model_inputs" / input_tag.upper() / original.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Model input not found at historical path {original} or repository path {candidate}"
    )


def run(result_file: Path, output_dir: Path, dpi: int, formats: list[str]) -> None:
    result_file = result_file.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_names = pd.ExcelFile(result_file).sheet_names
    if not sheet_names:
        raise ValueError(f"No sheets found in {result_file}")

    # Read first, then build one color dictionary shared by all 12 charts.
    chart_data = []
    every_node: list[str] = []
    for sheet_name in sheet_names:
        summary, x, w = read_result_sheet(result_file, sheet_name)
        input_file = locate_model_input(
            str(summary["Input file"]), str(summary["Input tag"]), repo_root
        )
        fl_labels, ch_labels = read_labels(input_file)
        fl_nodes, target_nodes, links = make_links(x, w, fl_labels, ch_labels)
        chart_data.append((sheet_name, fl_nodes, target_nodes, links))
        every_node.extend(fl_nodes)
        every_node.extend(target_nodes)
    global_colors = build_global_colors(every_node)

    for sheet_name, fl_nodes, target_nodes, links in chart_data:
        print(f"Drawing {sheet_name} ...", flush=True)
        draw_chord(
            fl_nodes,
            target_nodes,
            links,
            title=sheet_name,
            output_stem=output_dir / safe_filename(sheet_name),
            dpi=dpi,
            formats=formats,
            global_colors=global_colors,
        )
    print(f"Finished. {len(sheet_names)} charts saved to {output_dir.resolve()}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create FL-to-CH chord charts from every result sheet."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=repo_root / "figures" / "input_data" / "optimization_results.xlsx",
        help="Optimization result workbook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures" / "supplementary" / "allocation_chord",
        help="Directory for chart files.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=["png", "svg"],
    )
    args = parser.parse_args()
    run(args.results, args.output_dir, args.dpi, args.formats)


if __name__ == "__main__":
    main()
