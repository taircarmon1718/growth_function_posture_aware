#!/usr/bin/env python3
"""
Publication graphical abstract (Elsevier-oriented layout).

Target canvas: 1328 x 531 px at 300 DPI (width x height in pixels).
Outputs: graphical_abstract.png and graphical_abstract.pdf next to input images.

Input directory: figures/for_abstract/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "figures" / "for_abstract"
OUT_PNG = INPUT_DIR / "graphical_abstract.png"
OUT_PDF = INPUT_DIR / "graphical_abstract.pdf"

FILES = {
    "raw": "rawimage.jpg",
    "segmentation": "larva_segmentation.png",
    "detection": "larva_detection.png",
    "valid": "validd.png",
    "invalid": "invalid.png",
    "length1": "automated_length1.png",
    "length2": "automated_length2.png",
    "growth": "population-level.png",
}

# Elsevier-style target: 1328 x 531 px at 300 DPI
DPI = 300
FIG_W_IN = 1328 / DPI
FIG_H_IN = 531 / DPI

ARROW_COLOR = "#2c3e50"
ARROW_LW = 2.2
ARROW_MUTATION = 18


def trim_uniform_border(
    img: np.ndarray,
    *,
    dark_thresh: float = 12.0,
    margin: int = 2,
) -> np.ndarray:
    """Remove near-uniform dark border rows/columns; keep aspect ratio."""
    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        gray = img.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    row_ok = np.any(gray > dark_thresh, axis=1)
    col_ok = np.any(gray > dark_thresh, axis=0)
    if not row_ok.any() or not col_ok.any():
        return img
    ys = np.where(row_ok)[0]
    xs = np.where(col_ok)[0]
    y0, y1 = max(ys[0] - margin, 0), min(ys[-1] + margin + 1, h)
    x0, x1 = max(xs[0] - margin, 0), min(xs[-1] + margin + 1, w)
    return img[y0:y1, x0:x1]


def load_rgb(path: Path, trim: bool = True) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    if trim:
        arr = trim_uniform_border(arr)
    return arr


def show_image(ax, arr: np.ndarray) -> None:
    ax.imshow(arr, interpolation="nearest", aspect="equal")
    ax.axis("off")


def draw_horizontal_arrow(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyArrowPatch(
            (0.05, 0.5),
            (0.95, 0.5),
            transform=ax.transData,
            arrowstyle="-|>",
            mutation_scale=ARROW_MUTATION,
            color=ARROW_COLOR,
            linewidth=ARROW_LW,
            clip_on=False,
        )
    )


def draw_vertical_arrow(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyArrowPatch(
            (0.5, 0.92),
            (0.5, 0.08),
            transform=ax.transData,
            arrowstyle="-|>",
            mutation_scale=ARROW_MUTATION,
            color=ARROW_COLOR,
            linewidth=ARROW_LW,
            clip_on=False,
        )
    )


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for _key, name in FILES.items():
        p = INPUT_DIR / name
        if not p.is_file():
            raise FileNotFoundError(f"Required file missing: {p}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )

    raw = load_rgb(INPUT_DIR / FILES["raw"])
    seg = load_rgb(INPUT_DIR / FILES["segmentation"])
    det = load_rgb(INPUT_DIR / FILES["detection"])
    vld = load_rgb(INPUT_DIR / FILES["valid"])
    inv = load_rgb(INPUT_DIR / FILES["invalid"])
    len1 = load_rgb(INPUT_DIR / FILES["length1"])
    len2 = load_rgb(INPUT_DIR / FILES["length2"])
    pop = load_rgb(INPUT_DIR / FILES["growth"])

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")

    # Cols 0–4: pipeline; col 5: vertical arrow (full height). Top: In → Seg → Det; bottom: Posture → Morph → Growth
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.0, 0.11, 1.0, 0.11, 1.0, 0.1],
        hspace=0.22,
        wspace=0.12,
        left=0.03,
        right=0.97,
        top=0.86,
        bottom=0.10,
    )

    # --- Row 0 ---
    ax_in = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_seg = fig.add_subplot(gs[0, 2])
    ax_a2 = fig.add_subplot(gs[0, 3])
    ax_det = fig.add_subplot(gs[0, 4])
    ax_varrow = fig.add_subplot(gs[0:2, 5])

    show_image(ax_in, raw)
    show_image(ax_seg, seg)
    show_image(ax_det, det)
    draw_horizontal_arrow(ax_a1)
    draw_horizontal_arrow(ax_a2)

    ax_in.set_title("Input Image", fontsize=8, fontweight="bold", pad=4)
    ax_seg.set_title("Segmentation", fontsize=8, fontweight="bold", pad=4)
    ax_det.set_title("Detection", fontsize=8, fontweight="bold", pad=4)

    # --- Row 1 ---
    # Bottom row: Posture filtering (two small panels), arrow, Morphometrics (len1 then len2), arrow, Growth
    gs_post = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.06)
    ax_v = fig.add_subplot(gs_post[0, 0])
    ax_i = fig.add_subplot(gs_post[0, 1])
    ax_a3 = fig.add_subplot(gs[1, 1])
    # Morphometric measurement: place len1 on the left, len2 on the right (order fixed)
    gs_morph = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 2], wspace=0.06)
    ax_l1 = fig.add_subplot(gs_morph[0, 0])  # len1 left
    ax_l2 = fig.add_subplot(gs_morph[0, 1])  # len2 right
    ax_a4 = fig.add_subplot(gs[1, 3])
    ax_g = fig.add_subplot(gs[1, 4])

    # Display images in logical left-to-right order
    show_image(ax_v, vld)
    show_image(ax_i, inv)
    show_image(ax_l1, len1)
    show_image(ax_l2, len2)
    show_image(ax_g, pop)

    # Draw connecting arrows
    draw_horizontal_arrow(ax_a3)
    draw_horizontal_arrow(ax_a4)
    draw_vertical_arrow(ax_varrow)

    fig.canvas.draw()

    # Posture section title + sublabels (figure coordinates after layout)
    pos_v = ax_v.get_position()
    pos_i = ax_i.get_position()
    cx = 0.5 * (pos_v.x0 + pos_i.x1)
    fig.text(cx, pos_v.y1 + 0.028, "Posture Filtering", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_v.text(0.5, -0.05, "Valid", transform=ax_v.transAxes, ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax_v.text(0.5, -0.16, "✓", transform=ax_v.transAxes, ha="center", va="top", fontsize=11, color="#1e7e34")
    ax_i.text(0.5, -0.05, "Invalid", transform=ax_i.transAxes, ha="center", va="top", fontsize=6.5, fontweight="bold")
    ax_i.text(0.5, -0.16, "✗", transform=ax_i.transAxes, ha="center", va="top", fontsize=11, color="#c0392b")

    # Morphometrics title above len1 and len2
    pos_l1 = ax_l1.get_position()
    pos_l2 = ax_l2.get_position()
    cx_m = 0.5 * (pos_l1.x0 + pos_l2.x1)
    fig.text(cx_m, pos_l1.y1 + 0.028, "Morphometric Measurement", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_g.set_title("Growth Modeling", fontsize=8, fontweight="bold", pad=4)

    fig.savefig(OUT_PNG, dpi=DPI, format="png")
    fig.savefig(OUT_PDF, dpi=DPI, format="pdf")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
