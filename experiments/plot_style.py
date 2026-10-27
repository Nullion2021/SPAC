"""Shared plotting style for paper-facing experiment figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PALETTE = {
    "teal": "#0F766E",
    "blue": "#2563EB",
    "orange": "#C2410C",
    "gold": "#CA8A04",
    "slate": "#475569",
    "gray": "#94A3B8",
    "rose": "#BE123C",
    "ink": "#0F172A",
    "grid": "#D7DEE8",
    "panel": "#F8FAFC",
}

ATTACK_COLORS = {
    "random_proxy": "#2563EB",
    "query_adaptive": "#7C3AED",
    "distilled_proxy": "#C2410C",
    "single_machine_full_model": "#0F766E",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_air_gap",
    ["#E2E8F0", "#93C5FD", "#2563EB"],
)


def setup_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 360,
        }
    )


def style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], alpha=0.45, linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.0)
    ax.tick_params(length=0)


def style_heatmap_axis(ax, rows: int, cols: int) -> None:
    ax.set_facecolor(PALETTE["panel"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)


def annotate_heatmap(ax, data) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = float(data[i, j])
            text_color = "white" if value >= 0.55 else PALETTE["ink"]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color=text_color,
            )

