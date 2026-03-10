#!/usr/bin/env python3
"""
export_growth_subset_images.py
==============================
Exports the exact subset of larva images used in growth analysis:
    predicted_valid == 1  AND  predicted_posture == 1

Source : dual_larva_models/predictions/predictions_all_larvae.xlsx
Images : analysis_full/<date>/<image_name>/larvae_reports/<larva_filename>
Output : dual_larva_models/growth_subset_images/<date>/<larva_filename>

No models are retrained. No existing pipeline is modified.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.resolve()
PREDICTIONS  = ROOT_DIR / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
ANALYSIS_DIR = ROOT_DIR / "analysis_full"
OUTPUT_DIR   = ROOT_DIR / "dual_larva_models_geodesic2" / "growth_subset_images"

OVERWRITE    = False          # set True to overwrite already-copied files
PIXEL_TO_MM  = 0.232255814

VALID_CONF_THRESHOLD   = 0.90   # minimum valid_confidence to include larva
POSTURE_CONF_THRESHOLD = 0.90   # minimum posture_confidence to include larva

# Chronological date order (Oct → Nov)
DATE_ORDER = [
    "19.10", "20.10", "21.10", "24.10", "25.10",
    "26.10", "27.10", "29.10", "31.10", "3.11",
]


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def date_sort_key(d: str) -> tuple[int, int]:
    """Return (month, day) so 31.10 sorts before 3.11."""
    try:
        day, mon = str(d).split(".")
        return (int(mon), int(day))
    except Exception:
        return (999, 999)


def normalise_date(raw: str) -> str:
    """Fix Excel date quirk: '19.1' → '19.10'."""
    parts = str(raw).split(".")
    if len(parts) == 2 and parts[1] == "1":
        return f"{parts[0]}.10"
    return str(raw)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – LOAD AND FILTER PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def load_subset(predictions_file: Path) -> pd.DataFrame:
    """
    Load predictions_all_larvae.xlsx and return only rows where:
        predicted_valid   == 1
        predicted_posture == 1
        valid_confidence   >= VALID_CONF_THRESHOLD
        posture_confidence >= POSTURE_CONF_THRESHOLD
    """
    print(f"  Loading : {predictions_file}")

    if not predictions_file.exists():
        print(f"❌  Predictions file not found: {predictions_file}")
        sys.exit(1)

    df = pd.read_excel(predictions_file)
    print(f"  Total rows in file : {len(df)}")

    # Normalise date column
    df["date"] = df["date"].astype(str).apply(normalise_date)

    # Coerce confidence columns to numeric (guard against NaN)
    df["valid_confidence"]   = pd.to_numeric(df.get("valid_confidence",   0), errors="coerce").fillna(0.0)
    df["posture_confidence"] = pd.to_numeric(df.get("posture_confidence", 0), errors="coerce").fillna(0.0)

    # Step A: basic label filter (valid=1 & posture=1)
    label_mask = (df["predicted_valid"] == 1) & (df["predicted_posture"] == 1)
    after_labels = df[label_mask].copy()
    print(f"  Rows after label filter (valid=1 & posture=1)        : {len(after_labels)}")

    # Step B: confidence threshold filter
    conf_mask = (
        (after_labels["valid_confidence"]   >= VALID_CONF_THRESHOLD) &
        (after_labels["posture_confidence"] >= POSTURE_CONF_THRESHOLD)
    )
    subset = after_labels[conf_mask].copy()

    removed_by_conf = len(after_labels) - len(subset)
    print(f"  Rows removed by confidence thresholds                 : {removed_by_conf}")
    print(f"    (valid_conf >= {VALID_CONF_THRESHOLD}, posture_conf >= {POSTURE_CONF_THRESHOLD})")
    print(f"  Final subset size                                      : {len(subset)}")

    if subset.empty:
        print("❌  No rows match the filter criteria. Exiting.")
        sys.exit(1)

    # Confidence statistics for the final subset
    vc_mean = subset["valid_confidence"].mean()
    vc_std  = subset["valid_confidence"].std()
    pc_mean = subset["posture_confidence"].mean()
    pc_std  = subset["posture_confidence"].std()
    print(f"\n  Confidence statistics (final subset):")
    print(f"    valid_confidence   : mean={vc_mean:.4f}  std={vc_std:.4f}")
    print(f"    posture_confidence : mean={pc_mean:.4f}  std={pc_std:.4f}")

    return subset


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _draw_overlay(img: np.ndarray,
                  valid_conf: float,
                  posture_conf: float,
                  body_length_mm: float) -> np.ndarray:
    """
    Extend the canvas downward with a clean white information panel.
    The original image is NEVER touched — no text is drawn on the larva.

    Layout:
        ┌──────────────────────────────┐  ← original image (unchanged)
        │                              │
        │         larva crop           │
        │                              │
        └──────────────────────────────┘
        │ Valid: 0.923 │ Posture: 0.871 │ Length: 3.47 mm │  ← panel
        └──────────────────────────────────────────────────┘
    """
    h, w = img.shape[:2]

    # ── Typography ────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.38, min(w, h) / 900)
    thickness  = 1
    pad        = max(6, int(font_scale * 18))

    fields = [
        ("Valid conf",    f"{valid_conf:.3f}"),
        ("Posture conf",  f"{posture_conf:.3f}"),
        ("Body length",   f"{body_length_mm:.2f} mm"),
    ]

    # Measure panel height from a sample string
    (_, char_h), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
    panel_h = char_h + pad * 2

    # ── Build output canvas ───────────────────────────────────
    # Panel is same width as the image; sits below it.
    panel = np.full((panel_h, w, 3), 250, dtype=np.uint8)   # near-white

    # Thin separator line at top of panel
    cv2.line(panel, (0, 0), (w, 0), (180, 180, 180), 1)

    # Evenly distribute the three fields across the panel width
    n_fields = len(fields)
    col_w    = w // n_fields

    for i, (label, value) in enumerate(fields):
        text   = f"{label}: {value}"
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Centre text within its column
        x_col_centre = i * col_w + col_w // 2
        x            = max(pad, x_col_centre - tw // 2)
        y            = pad + char_h

        # Label in dark grey, value slightly darker
        cv2.putText(panel, text, (x, y),
                    font, font_scale, (40, 40, 40),
                    thickness, cv2.LINE_AA)

        # Vertical dividers between columns
        if i > 0:
            cv2.line(panel, (i * col_w, pad // 2),
                     (i * col_w, panel_h - pad // 2),
                     (190, 190, 190), 1)

    # ── Stack: original image on top, panel below ─────────────
    out = np.vstack([img, panel])
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – EXPORT ANNOTATED IMAGES INTO DATE FOLDERS
# ─────────────────────────────────────────────────────────────────────────────
def copy_images(subset: pd.DataFrame) -> dict[str, list[float]]:
    """
    Load each larva image, draw a confidence + body-length overlay,
    and save it to OUTPUT_DIR/<date>/<larva_filename>.
    Returns a dict  {date: [body_length_mm, ...]}  for the exported larvae.
    """
    lengths_by_date: dict[str, list[float]] = {}
    total_copied  = 0
    total_missing = 0
    total_skipped = 0

    dates_in_data = sorted(subset["date"].unique(), key=date_sort_key)

    for date in dates_in_data:
        date_subset = subset[subset["date"] == date]
        date_out    = OUTPUT_DIR / date
        date_out.mkdir(parents=True, exist_ok=True)

        copied_this_date  = 0
        missing_this_date = 0
        lengths: list[float] = []

        for _, row in date_subset.iterrows():
            src = (
                ANALYSIS_DIR
                / str(row["date"])
                / str(row["image_name"])
                / "larvae_reports"
                / str(row["larva_filename"])
            )
            dst = date_out / str(row["larva_filename"])

            bl           = float(row.get("body_length_mm", 0.0))
            valid_conf   = float(row.get("valid_confidence",   0.0))
            posture_conf = float(row.get("posture_confidence", 0.0))

            # Skip if already exists and OVERWRITE is disabled
            if dst.exists() and not OVERWRITE:
                total_skipped += 1
                if bl > 0:
                    lengths.append(bl)
                continue

            if not src.exists():
                missing_this_date += 1
                total_missing     += 1
                continue

            # Load → annotate → save
            try:
                img = cv2.imread(str(src))
                if img is None:
                    raise ValueError("cv2.imread returned None")

                annotated = _draw_overlay(img, valid_conf, posture_conf, bl)
                cv2.imwrite(str(dst), annotated)

                copied_this_date += 1
                total_copied     += 1
                if bl > 0:
                    lengths.append(bl)

            except Exception as exc:
                print(f"    ⚠  Could not process {src.name}: {exc}")
                missing_this_date += 1

        lengths_by_date[date] = lengths

        print(
            f"  {date:<8}  total={len(date_subset):>5}  "
            f"exported={copied_this_date:>5}  "
            f"missing={missing_this_date:>4}  "
            f"skipped(exist)={total_skipped:>4}"
        )

    print(f"\n  ✓  Total exported : {total_copied}")
    print(f"  ⚠  Total missing  : {total_missing}")
    print(f"  –  Total skipped  : {total_skipped}")

    return lengths_by_date


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – COMPUTE SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_summaries(
    subset: pd.DataFrame,
    lengths_by_date: dict[str, list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-date and global summary DataFrames.
    """
    rows = []
    dates_present = sorted(lengths_by_date.keys(), key=date_sort_key)

    for date in dates_present:
        date_df = subset[subset["date"] == date]
        arr = np.array(lengths_by_date[date])

        n_larvae = len(date_df)
        mean_mm   = float(np.mean(arr))   if len(arr) > 0 else 0.0
        median_mm = float(np.median(arr)) if len(arr) > 0 else 0.0
        std_mm    = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        rows.append({
            "date":                 date,
            "number_of_larvae":     n_larvae,
            "mean_body_length_mm":  round(mean_mm,   4),
            "median_body_length_mm": round(median_mm, 4),
            "std_body_length_mm":   round(std_mm,    4),
        })

    per_date_df = pd.DataFrame(rows)

    # Global
    all_lengths = np.concatenate(
        [np.array(v) for v in lengths_by_date.values() if v]
    )
    global_df = pd.DataFrame([{
        "total_larvae":          int(len(subset)),
        "dates_included":        len(dates_present),
        "mean_body_length_mm":   round(float(np.mean(all_lengths)),   4) if len(all_lengths) > 0 else 0,
        "median_body_length_mm": round(float(np.median(all_lengths)), 4) if len(all_lengths) > 0 else 0,
        "std_body_length_mm":    round(float(np.std(all_lengths, ddof=1)), 4) if len(all_lengths) > 1 else 0,
        "generated_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])

    return per_date_df, global_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – SAVE SUMMARIES
# ─────────────────────────────────────────────────────────────────────────────
def save_summaries(per_date_df: pd.DataFrame, global_df: pd.DataFrame):
    per_date_path = OUTPUT_DIR / "summary_per_date.csv"
    global_path   = OUTPUT_DIR / "global_summary.csv"

    per_date_df.to_csv(per_date_path, index=False)
    global_df.to_csv(global_path, index=False)

    print(f"\n  ✓  summary_per_date.csv  → {per_date_path}")
    print(f"  ✓  global_summary.csv    → {global_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 – CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_console_table(per_date_df: pd.DataFrame):
    print()
    print("  " + "─" * 52)
    print(f"  {'Date':<10} {'Count':>7} {'Mean Length (mm)':>18} {'Median (mm)':>12}")
    print("  " + "─" * 52)
    for _, row in per_date_df.iterrows():
        print(
            f"  {row['date']:<10} "
            f"{int(row['number_of_larvae']):>7}  "
            f"{row['mean_body_length_mm']:>17.3f}  "
            f"{row['median_body_length_mm']:>11.3f}"
        )
    print("  " + "─" * 52)
    total = per_date_df["number_of_larvae"].sum()
    print(f"  {'TOTAL':<10} {int(total):>7}")
    print("  " + "─" * 52)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 – CONFIDENCE-LEVEL EXAMPLE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def visualise_confidence_examples(
    subset: pd.DataFrame,
    n_per_band: int = 3,
    save_path: Path | None = None,
) -> None:
    """
    Sample n_per_band larvae from three posture-confidence bands:
        low    : posture_confidence < 0.85
        medium : 0.85 <= posture_confidence <= 0.93
        high   : posture_confidence > 0.93

    For each example, load the exported (annotated) image from OUTPUT_DIR.
    Arrange all examples in a grid:   rows = bands,  cols = examples.
    Save to OUTPUT_DIR/confidence_examples.png  (or save_path if given).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = [
        ("Low  (<0.85)",       subset[subset["posture_confidence"] <  0.85]),
        ("Med  (0.85–0.93)",   subset[(subset["posture_confidence"] >= 0.85) &
                                      (subset["posture_confidence"] <= 0.93)]),
        ("High (>0.93)",       subset[subset["posture_confidence"] >  0.93]),
    ]

    # ── Sample rows ───────────────────────────────────────────
    sampled: list[tuple[str, list[pd.Series]]] = []
    for label, band_df in bands:
        if band_df.empty:
            sampled.append((label, []))
            print(f"  ⚠  Confidence band '{label}' has 0 samples.")
            continue
        n   = min(n_per_band, len(band_df))
        rows = [band_df.iloc[i] for i in
                np.random.default_rng(42).choice(len(band_df), n, replace=False)]
        sampled.append((label, rows))

    n_cols = n_per_band
    n_rows = len(bands)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.8),
        facecolor="#f7f7f7",
    )
    # Guarantee 2-D axes array
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    band_colors = ["#d9534f", "#f0ad4e", "#5cb85c"]   # red / amber / green

    for row_idx, (band_label, rows) in enumerate(sampled):
        color = band_colors[row_idx]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#eeeeee")
            ax.set_xticks([])
            ax.set_yticks([])

            # Coloured border to indicate band
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

            if col_idx >= len(rows):
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="#aaaaaa", fontsize=14)
                continue

            row = rows[col_idx]
            # Try exported (annotated) image first; fall back to source
            img_path = (
                OUTPUT_DIR
                / str(row["date"])
                / str(row["larva_filename"])
            )
            if not img_path.exists():
                img_path = (
                    ANALYSIS_DIR
                    / str(row["date"])
                    / str(row["image_name"])
                    / "larvae_reports"
                    / str(row["larva_filename"])
                )

            if img_path.exists():
                bgr = cv2.imread(str(img_path))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(rgb, aspect="auto")
                else:
                    ax.text(0.5, 0.5, "load error", ha="center", va="center",
                            transform=ax.transAxes, color="red", fontsize=8)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="#888888", fontsize=8)

            # Subtitle: date + conf values
            subtitle = (
                f"{row['date']}  "
                f"vc={float(row['valid_confidence']):.3f}  "
                f"pc={float(row['posture_confidence']):.3f}"
            )
            ax.set_xlabel(subtitle, fontsize=6.5, color="#333333", labelpad=3)

        # Row label on the left
        axes[row_idx, 0].set_ylabel(
            band_label, fontsize=8, color=color, fontweight="bold", labelpad=6
        )

    fig.suptitle(
        "Larva Confidence Examples — Low / Medium / High",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.6)

    out_path = save_path or (OUTPUT_DIR / "confidence_examples.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  confidence_examples.png → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("GROWTH SUBSET IMAGE EXPORT")
    print("Filter: predicted_valid == 1  AND  predicted_posture == 1")
    print("=" * 60)
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source    : {PREDICTIONS.relative_to(ROOT_DIR)}")
    print(f"  Images    : {ANALYSIS_DIR.relative_to(ROOT_DIR)}")
    print(f"  Output    : {OUTPUT_DIR.relative_to(ROOT_DIR)}")
    print(f"  Overwrite : {OVERWRITE}")

    # Create top-level output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1 ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 1 — Loading and filtering predictions")
    print("-" * 60)
    subset = load_subset(PREDICTIONS)

    # ── Step 2 ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 2 — Copying images to date folders")
    print("-" * 60)
    lengths_by_date = copy_images(subset)

    # ── Step 3 ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 3 — Computing summary statistics")
    print("-" * 60)
    per_date_df, global_df = compute_summaries(subset, lengths_by_date)

    # ── Step 4 ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 4 — Saving CSV summaries")
    print("-" * 60)
    save_summaries(per_date_df, global_df)

    # ── Step 5 ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 5 — Per-Date Console Summary")
    print("-" * 60)
    print_console_table(per_date_df)

    # ── Step 6 ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 6 — Confidence-level example visualisation")
    print("-" * 60)
    visualise_confidence_examples(subset, n_per_band=3)

    # ── Done ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GROWTH SUBSET IMAGE EXPORT COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

