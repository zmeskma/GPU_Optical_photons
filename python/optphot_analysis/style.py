"""Shared matplotlib style for the figures in docs/figures.

Colours follow a validated colour-vision-deficiency-safe categorical order
(blue, orange, aqua, ...), always assigned in that fixed order. In CPU/GPU
comparisons blue = CPU and orange = GPU. Analytic expectations are always
drawn in neutral ink, so colour only ever identifies simulated data.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Sequential (one hue, light -> dark) and diverging (blue <-> grey <-> red).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32d2c"])


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def pull_panel(ax, x, pulls, color=BLUE, logx=False, marker="o"):
    """Residual panel: pulls with +-1 and +-3 sigma reference bands."""
    ax.axhspan(-1, 1, color=GRID, alpha=0.8, lw=0)
    ax.axhline(0, color=INK_SECONDARY, lw=1)
    for s in (-3, 3):
        ax.axhline(s, color=INK_SECONDARY, lw=0.8, ls=":")
    ax.plot(x, pulls, marker, color=color, ms=5, ls="none")
    ax.set_ylim(-4.5, 4.5)
    ax.set_ylabel("pull (σ)")
    if logx:
        ax.set_xscale("log")


# Optional prefix for figure file names (e.g. "gpu_" for GPU-backend runs).
FILE_PREFIX = ""


def save(fig, path):
    import os
    head, tail = os.path.split(path)
    path = os.path.join(head, FILE_PREFIX + tail)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
