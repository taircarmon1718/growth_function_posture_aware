#!/usr/bin/env python3
"""Assemble 2×3 pipeline figure from fixed Picture1–6 files in this directory only."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FIG_DIR = Path(__file__).resolve().parent
OUT = FIG_DIR / "pipeline_figure.png"
CAPTION = FIG_DIR / "pipeline_figure_caption.txt"

FILES = [
    FIG_DIR / "Picture1.jpg",
    FIG_DIR / "Picture2.png",
    FIG_DIR / "Picture3.png",
    FIG_DIR / "Picture4.png",
    FIG_DIR / "Picture5.png",
    FIG_DIR / "Picture6.png",
]
LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]
DPI = 300


def main() -> None:
    for p in FILES:
        if not p.is_file():
            raise FileNotFoundError(p)

    images = [mpimg.imread(str(p)) for p in FILES]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, img, lab in zip(axes_flat, images, LABELS):
        ax.imshow(img, aspect="equal", interpolation="nearest")
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            lab,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color="white",
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=0.55, pad=2.0, edgecolor="none"),
        )

    fig.savefig(str(OUT), dpi=DPI, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)

    cap = (
        "(A) Raw RGB image. (B) Processed binary mask. (C) Candidate detections. "
        "(D) shows a larva retained after hierarchical machine learning filtering (valid + posture). "
        "(E) Binary mask of the selected larva. (F) Morphometric extraction (PCA length)."
    )
    CAPTION.write_text(cap + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
