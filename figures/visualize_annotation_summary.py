#!/usr/bin/env python3
"""
visualize_annotation_summary.py
================================
Read-only inspection script for larva annotation quality labels.

PART 1 — Prints annotation statistics from larva_quality_labels.xlsx.
PART 2 — Creates a publication-ready side-by-side figure comparing
          one GOOD vs one NOT-GOOD labelled larva example,
          using only the visual crop panels (original / mask / overlay).

Output figure saved to:
    figures/annotation_comparison.png

Does NOT modify any existing file.
"""

import sys
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()          # figures/
PROJECT_DIR  = SCRIPT_DIR.parent                        # growth_function_posture_aware/
ANALYSIS_DIR = PROJECT_DIR / "analysis_full"
LABELS_FILE  = ANALYSIS_DIR / "larva_quality_labels.xlsx"
OUTPUT_FIG   = SCRIPT_DIR  / "annotation_comparison.png"

RANDOM_SEED  = 42


# ──────────────────────────────────────────────────────────────
#  PART 1  ─  LOAD LABELS
# ──────────────────────────────────────────────────────────────
def load_labels() -> pd.DataFrame:
    """Load larva_quality_labels.xlsx and return as DataFrame."""
    if not LABELS_FILE.exists():
        print(f"❌  Labels file not found: {LABELS_FILE}")
        sys.exit(1)

    df = pd.read_excel(LABELS_FILE)
    df["date"]           = df["date"].astype(str)
    df["image_name"]     = df["image_name"].astype(str)
    df["larva_filename"] = df["larva_filename"].astype(str)
    df["is_valid_larva"] = pd.to_numeric(df["is_valid_larva"], errors="coerce").fillna(0).astype(int)
    df["shape_score"]    = pd.to_numeric(df["shape_score"],    errors="coerce").fillna(0).astype(int)
    return df


# ──────────────────────────────────────────────────────────────
#  PART 1  ─  PRINT STATISTICS
# ──────────────────────────────────────────────────────────────
def print_statistics(df: pd.DataFrame) -> None:
    """Print a clean annotation summary to the console."""
    n = len(df)
    sep = "─" * 56

    print(f"\n{'═' * 56}")
    print(f"  ANNOTATION SUMMARY")
    print(f"{'═' * 56}")
    print(f"  Labels file : {LABELS_FILE.relative_to(PROJECT_DIR)}")
    print(f"  Total labeled larvae : {n}")
    print(sep)

    # ── is_valid_larva ───────────────────────────────────────
    n_valid   = int((df["is_valid_larva"] == 1).sum())
    n_invalid = int((df["is_valid_larva"] == 0).sum())
    print(f"  is_valid_larva")
    print(f"    1 (valid)   : {n_valid:>5}  ({100 * n_valid   / n:.1f} %)")
    print(f"    0 (invalid) : {n_invalid:>5}  ({100 * n_invalid / n:.1f} %)")
    print(sep)

    # ── shape_score ──────────────────────────────────────────
    for score, label in [(0, "No-Good"), (1, "OK     "), (2, "Great  ")]:
        cnt = int((df["shape_score"] == score).sum())
        print(f"  shape_score == {score}  ({label}) : {cnt:>5}  ({100 * cnt / n:.1f} %)")
    print(sep)

    # ── posture-suitable ─────────────────────────────────────
    n_posture = int(((df["is_valid_larva"] == 1) & (df["shape_score"] >= 1)).sum())
    print(f"  Posture-suitable  (valid=1 & shape≥1) : {n_posture:>5}  ({100 * n_posture / n:.1f} %)")
    print(f"{'═' * 56}\n")


# ──────────────────────────────────────────────────────────────
#  PART 2  ─  SELECT EXAMPLES
# ──────────────────────────────────────────────────────────────
def _try_load_image(row: pd.Series) -> tuple[np.ndarray | None, Path]:
    """Return (bgr_image_or_None, path)."""
    img_path = (
        ANALYSIS_DIR
        / row["date"]
        / row["image_name"]
        / "larvae_reports"
        / row["larva_filename"]
    )
    if not img_path.exists():
        return None, img_path
    img = cv2.imread(str(img_path))
    return img, img_path


def select_examples(
    df: pd.DataFrame,
    rng: random.Random,
) -> tuple[tuple[pd.Series, np.ndarray] | None,
           tuple[pd.Series, np.ndarray] | None]:
    """
    Pick one GOOD and one NOT-GOOD example with a loadable image.

    GOOD     : is_valid_larva == 1 AND shape_score >= 1
    NOT-GOOD : is_valid_larva == 0 OR  shape_score == 0
    """
    good_pool    = df[(df["is_valid_larva"] == 1) & (df["shape_score"] >= 1)]
    notgood_pool = df[(df["is_valid_larva"] == 0) | (df["shape_score"] == 0)]

    def _pick(pool: pd.DataFrame, label: str):
        candidates = pool.sample(frac=1, random_state=rng.randint(0, 2**31)).iterrows()
        for _, row in candidates:
            img, path = _try_load_image(row)
            if img is not None:
                return row, img
        print(f"  ⚠  No loadable image found for category: {label}")
        return None

    good    = _pick(good_pool,    "GOOD")
    notgood = _pick(notgood_pool, "NOT-GOOD")
    return good, notgood


# ──────────────────────────────────────────────────────────────
#  PART 2  ─  CLEAN VISUAL PANEL
# ──────────────────────────────────────────────────────────────
def clean_visual_panel(bgr: np.ndarray) -> np.ndarray:
    """
    Extract only the three pure image panels (original | mask | overlay)
    from a larva report image, discarding the right-side black metrics text panel.

    The report layout from batch_analysis_pipeline.py is:
        [ gray_crop | mask_crop | overlay ]  +  [ 300 px black text panel ]
    Each of the three visual sub-panels is exactly the same width.

    Strategy:
      1. Find where the black text panel starts (right-to-left scan).
      2. The remaining visual strip width must be divisible by 3 —
         round down to the nearest multiple of 3 so sub-panels are equal.
      3. Return exactly those pixels; no text, no padding.
    """
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w_tot = gray.shape

    # ── Step 1: find right edge of visual content ─────────────
    # The black text panel has near-zero mean intensity per column.
    visual_w = w_tot
    for x in range(w_tot - 1, w_tot // 2, -1):
        if np.mean(gray[:, x]) > 15:
            visual_w = x + 1
            break

    # Fallback: if scan found nothing, assume text panel is rightmost 27 %
    if visual_w == w_tot:
        visual_w = int(w_tot * 0.73)

    # ── Step 2: snap to a multiple of 3 (three equal sub-panels) ─
    visual_w = (visual_w // 3) * 3

    return bgr[:, :visual_w]


# ──────────────────────────────────────────────────────────────
#  PART 2  ─  CREATE COMPARISON FIGURE
# ──────────────────────────────────────────────────────────────
def create_comparison_visualization(
    good_pair:    tuple[pd.Series, np.ndarray] | None,
    notgood_pair: tuple[pd.Series, np.ndarray] | None,
) -> None:
    """
    Save a minimal 2 × 3 figure:
        Row 0 (top)    : posture-suitable   — original | mask | overlay
        Row 1 (bottom) : posture-unsuitable — original | mask | overlay
    Absolutely no text, no axes, no titles, no borders anywhere.
    """
    # ── Extract and split into 3 sub-panels ───────────────────
    def _split_panels(pair: tuple | None) -> list[np.ndarray]:
        if pair is None:
            placeholder = np.full((120, 120, 3), 245, dtype=np.uint8)
            return [placeholder, placeholder, placeholder]
        _, bgr  = pair
        strip   = clean_visual_panel(bgr)          # shape: (h, 3*panel_w, 3)
        panel_w = strip.shape[1] // 3
        return [
            strip[:, 0          : panel_w  ],      # original grayscale
            strip[:, panel_w    : 2*panel_w],      # binary mask
            strip[:, 2*panel_w  :           ],     # overlay + skeleton
        ]

    good_panels    = _split_panels(good_pair)      # list of 3 BGR arrays
    notgood_panels = _split_panels(notgood_pair)

    # ── Normalise heights per column so rows align ────────────
    # Each column pair shares the same width; heights may differ slightly.
    def _resize_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        return cv2.resize(img, (img.shape[1], target_h), interpolation=cv2.INTER_AREA)

    for col in range(3):
        target_h = max(good_panels[col].shape[0], notgood_panels[col].shape[0])
        good_panels[col]    = _resize_h(good_panels[col],    target_h)
        notgood_panels[col] = _resize_h(notgood_panels[col], target_h)

    # ── Build pixel-level canvas ──────────────────────────────
    gap_col = 4    # white vertical gap between sub-panels (px)
    gap_row = 8    # white horizontal gap between the two rows (px)

    def _row_strip(panels: list[np.ndarray]) -> np.ndarray:
        row_h = max(p.shape[0] for p in panels)
        pieces = []
        for i, p in enumerate(panels):
            if i > 0:
                pieces.append(np.full((row_h, gap_col, 3), 255, dtype=np.uint8))
            pad_h = row_h - p.shape[0]
            if pad_h:
                p = np.vstack([p, np.full((pad_h, p.shape[1], 3), 255, dtype=np.uint8)])
            pieces.append(p)
        return np.hstack(pieces)

    good_strip    = _row_strip(good_panels)
    notgood_strip = _row_strip(notgood_panels)

    # Match widths
    target_w = max(good_strip.shape[1], notgood_strip.shape[1])
    def _pad_w(img: np.ndarray, w: int) -> np.ndarray:
        if img.shape[1] == w:
            return img
        return np.hstack([img, np.full((img.shape[0], w - img.shape[1], 3), 255, dtype=np.uint8)])

    good_strip    = _pad_w(good_strip,    target_w)
    notgood_strip = _pad_w(notgood_strip, target_w)

    separator = np.full((gap_row, target_w, 3), 255, dtype=np.uint8)
    canvas_bgr = np.vstack([good_strip, separator, notgood_strip])
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    # ── Pixel-perfect figure — zero margins ───────────────────
    dpi   = 300
    h_px, w_px = canvas_rgb.shape[:2]
    fig   = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor="white")
    ax    = fig.add_axes((0, 0, 1, 1))
    ax.imshow(canvas_rgb, interpolation="lanczos")
    ax.axis("off")

    fig.savefig(
        OUTPUT_FIG,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"  ✓  Figure saved → {OUTPUT_FIG}")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main() -> None:
    rng = random.Random(RANDOM_SEED)

    # ── PART 1 ───────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  PART 1 — LOADING ANNOTATION DATA")
    print("═" * 56)
    df = load_labels()
    print_statistics(df)

    # ── PART 2 ───────────────────────────────────────────────
    print("═" * 56)
    print("  PART 2 — SELECTING EXAMPLES")
    print("═" * 56)

    good_pair, notgood_pair = select_examples(df, rng)

    if good_pair is not None:
        row, _ = good_pair
        print(f"  GOOD example     : {row['date']} / {row['image_name']} / {row['larva_filename']}")
        print(f"                     valid={row['is_valid_larva']}  shape_score={row['shape_score']}")
    else:
        print("  ⚠  No GOOD example found.")

    if notgood_pair is not None:
        row, _ = notgood_pair
        print(f"  NOT-GOOD example : {row['date']} / {row['image_name']} / {row['larva_filename']}")
        print(f"                     valid={row['is_valid_larva']}  shape_score={row['shape_score']}")
    else:
        print("  ⚠  No NOT-GOOD example found.")

    print(f"\n  Generating figure …")
    create_comparison_visualization(good_pair, notgood_pair)

    print("\n" + "═" * 56)
    print("  DONE")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    main()

