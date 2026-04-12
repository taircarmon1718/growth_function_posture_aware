#!/usr/bin/env python3
"""Raw larval morphometrics dataset analysis.

This script scans the segmentation pipeline output directory
`analysis_full_binary_masks_only/` and computes descriptive statistics
for the **raw** larval dataset (before any ML filtering).

Outputs are written to:
    raw_dataset_analysis/
        figures/
        tables/
        reports/

It does NOT modify any existing data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.decomposition import PCA as skPCA


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
INPUT_ROOT = PROJECT_ROOT / "analysis_full_binary_masks_only"
OUTPUT_ROOT = PROJECT_ROOT / "raw_dataset_analysis"
FIG_DIR = OUTPUT_ROOT / "figures"
TAB_DIR = OUTPUT_ROOT / "tables"
REP_DIR = OUTPUT_ROOT / "reports"

PIXEL_TO_MM = 0.232255814


def _ensure_dirs() -> None:
    """Create output directory structure if it does not exist."""
    for d in (OUTPUT_ROOT, FIG_DIR, TAB_DIR, REP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _scan_morphometrics() -> pd.DataFrame:
    """Scan all dates/images and load morphometrics.csv into one DataFrame.

    Expected structure:
        analysis_full_binary_masks_only/
            DATE/
                IMAGE_NAME/
                    morphometrics.csv
    """
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    records = []

    for date_dir in sorted(INPUT_ROOT.iterdir()):
        if not date_dir.is_dir():
            continue
        date = date_dir.name

        for img_dir in sorted(date_dir.iterdir()):
            if not img_dir.is_dir():
                continue
            image_name = img_dir.name
            csv_path = img_dir / "morphometrics.csv"
            if not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:  # pragma: no cover - defensive
                print(f"⚠️  Failed to read {csv_path}: {e}")
                continue

            if df.empty:
                continue

            # Add date and image_name columns
            df["date"] = date
            df["image_name"] = image_name

            records.append(df)

    if not records:
        raise RuntimeError(f"No morphometrics.csv files found under {INPUT_ROOT}")

    full_df = pd.concat(records, ignore_index=True)

    # Drop unwanted date 18.10 from the raw analysis
    full_df = full_df[full_df["date"] != "18.10"].reset_index(drop=True)

    # Reorder columns: date, image_name, larva_id, then others
    cols = list(full_df.columns)
    preferred_order = ["date", "image_name", "larva_id"]
    front = [c for c in preferred_order if c in cols]
    rest = [c for c in cols if c not in front]
    full_df = full_df[front + rest]

    return full_df


def _compute_pca_body_length(mask: np.ndarray) -> float:
    """Compute body length (in pixels) from a binary mask using the same
    PCA-based algorithm as compute_geodesic_body_length() in the pipeline.

    Algorithm:
      1. Extract all foreground pixels from the binary mask
      2. Compute PCA on the pixel coordinates
      3. Take the first principal component (major axis)
      4. Project all mask pixels onto that axis
      5. Body length = max(projection) - min(projection)

    Very small masks (<5 pixels) return 0.0.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) < 5:
        return 0.0
    # (row, col) -> (x, y)
    points = coords[:, [1, 0]].astype(float)
    centroid = np.mean(points, axis=0)
    pca = skPCA(n_components=1)
    pca.fit(points)
    axis = pca.components_[0]
    centered = points - centroid
    projections = centered @ axis
    length_px = float(projections.max() - projections.min())
    return length_px


def _attach_body_length_from_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute body length from binary larva masks using PCA-based length.

    For each row (date, image_name, larva_id) this function loads the
    corresponding mask from:
        analysis_full_binary_masks_only/date/image_name/larvae_reports/larva_XXX.png

    It then computes body_length_px and body_length_mm and returns an
    updated DataFrame that includes these columns.
    """
    # We create shallow copy so original is untouched
    df = df.copy()

    bl_px_list = []

    for idx, row in df.iterrows():
        date = str(row["date"])
        image_name = str(row["image_name"])
        larva_id = row.get("larva_id", None)

        # Construct filename; assume larva_id encodes index or name
        # Common pattern in pipeline: larva_001.png etc.
        if isinstance(larva_id, str) and larva_id.lower().endswith(".png"):
            fname = larva_id
        else:
            try:
                n = int(str(larva_id))
                fname = f"larva_{n:03d}.png"
            except Exception:
                # Fallback: treat larva_id as already a filename-like string
                fname = str(larva_id)

        mask_path = INPUT_ROOT / date / image_name / "larvae_reports" / fname
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # If mask is missing, fall back to 0.0
            bl_px = 0.0
        else:
            bl_px = _compute_pca_body_length(mask)
        bl_px_list.append(bl_px)

    df["body_length_px"] = bl_px_list
    df["body_length_mm"] = df["body_length_px"] * PIXEL_TO_MM

    return df


def _compute_per_image_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute larvae-per-image statistics table."""
    # larvae count per (date, image)
    grp = df.groupby(["date", "image_name"], as_index=False).agg(
        n_larvae=("larva_id", "size"),
        mean_body_length=("body_length_mm", "mean"),
    )
    return grp


def _compute_per_date_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-date body length and count statistics."""
    def _q(x, q):
        return np.quantile(x, q) if len(x) else np.nan

    grp = df.groupby("date", as_index=False).agg(
        n_larvae=("larva_id", "size"),
        mean_body_length=("body_length_mm", "mean"),
        std_body_length=("body_length_mm", "std"),
        median_body_length=("body_length_mm", "median"),
        p05_body_length=("body_length_mm", lambda x: _q(x, 0.05)),
        p95_body_length=("body_length_mm", lambda x: _q(x, 0.95)),
    )
    return grp


def _save_tables(df: pd.DataFrame, per_img: pd.DataFrame, per_date: pd.DataFrame) -> None:
    """Save CSV tables in TAB_DIR."""
    df.to_csv(TAB_DIR / "raw_dataset_full.csv", index=False)
    per_date.to_csv(TAB_DIR / "per_date_statistics.csv", index=False)
    per_img.to_csv(TAB_DIR / "per_image_statistics.csv", index=False)


def _plot_distributions(df: pd.DataFrame) -> None:
    """Create summary distribution figures.

    Includes a publication-quality body-length histogram focused on the
    biologically relevant range (0–150 px ≈ 0–34.8 mm) with mean/median
    markers and outlier count annotation. All statistics are in mm.
    """
    sns.set(style="whitegrid", context="talk")

    # --- Publication-quality body length distribution in mm ---
    bl_mm = df["body_length_mm"].astype(float).dropna()
    if not bl_mm.empty:
        # Threshold equivalent of 150 px in mm
        thr_mm = 150 * PIXEL_TO_MM
        inliers = bl_mm[(bl_mm >= 0) & (bl_mm <= thr_mm)]
        outliers = bl_mm[bl_mm > thr_mm]

        n_total = len(bl_mm)
        n_out = len(outliers)
        mean_bl = inliers.mean() if len(inliers) else np.nan
        med_bl = inliers.median() if len(inliers) else np.nan

        plt.figure(figsize=(8, 6))
        sns.histplot(inliers, bins=45, kde=False, color="tab:blue", edgecolor="black")
        plt.xlim(0, thr_mm)
        plt.xlabel("Body length (mm)", fontsize=14)
        plt.ylabel("Number of detected larvae", fontsize=14)
        plt.title("Distribution of raw body-length measurements", fontsize=16)

        if np.isfinite(mean_bl):
            plt.axvline(mean_bl, color="tab:orange", linestyle="--", linewidth=2,
                        label=f"Mean = {mean_bl:.2f} mm")
        if np.isfinite(med_bl):
            plt.axvline(med_bl, color="tab:green", linestyle="-.", linewidth=2,
                        label=f"Median = {med_bl:.2f} mm")

        text_lines = [
            f"Total larvae: {n_total}",
            f"Median (≤{thr_mm:.2f} mm): {med_bl:.2f} mm" if np.isfinite(med_bl) else "Median: n/a",
            f"Mean  (≤{thr_mm:.2f} mm): {mean_bl:.2f} mm" if np.isfinite(mean_bl) else "Mean: n/a",
            f"Outliers >{thr_mm:.2f} mm: {n_out}",
        ]
        txt = "\n".join(text_lines)
        ax = plt.gca()
        ax.text(
            0.98,
            0.97,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.9),
        )

        plt.legend(loc="upper left", fontsize=11, frameon=True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "raw_length_distribution_clean.png", dpi=300)
        plt.close()

    # --- Area distribution (full range) ---
    if "area" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df["area"].dropna(), bins=50, kde=True, color="tab:green")
        plt.xlabel("Area (pixels)")
        plt.ylabel("Count")
        plt.title("Raw area distribution")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "raw_area_distribution.png", dpi=300)
        plt.close()

    # --- Mean width distribution ---
    if "mean_width" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df["mean_width"].dropna(), bins=50, kde=True, color="tab:orange")
        plt.xlabel("Mean width (pixels)")
        plt.ylabel("Count")
        plt.title("Raw mean width distribution")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "raw_mean_width_distribution.png", dpi=300)
        plt.close()

    # --- Curvature ratio distribution ---
    if "curvature_ratio" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df["curvature_ratio"].dropna(), bins=50, kde=True, color="tab:red")
        plt.xlabel("Curvature ratio")
        plt.ylabel("Count")
        plt.title("Raw curvature ratio distribution")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "raw_curvature_ratio_distribution.png", dpi=300)
        plt.close()


def _sorted_dates_with_311_last(dates: pd.Series | list[str]) -> list[str]:
    """Return sorted list of dates with '3.11' forced to be the last element.

    Dates are treated lexicographically, then '3.11' is moved to the end if present.
    """
    unique = sorted(set(dates))
    if "3.11" in unique:
        unique = [d for d in unique if d != "3.11"] + ["3.11"]
    return unique


def _plot_temporal(per_img: pd.DataFrame, per_date: pd.DataFrame) -> None:
    """Figures depending on per-image and per-date statistics."""
    sns.set(style="whitegrid", context="talk")

    # Histogram of larvae per image
    plt.figure(figsize=(8, 6))
    sns.histplot(per_img["n_larvae"], bins=30, kde=False, color="tab:purple")
    plt.xlabel("Larvae per image")
    plt.ylabel("Number of images")
    plt.title("Distribution of larvae per image")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "larvae_per_image_histogram.png", dpi=300)
    plt.close()

    # Larvae per day bar plot (with 3.11 last)
    plt.figure(figsize=(10, 6))
    order = _sorted_dates_with_311_last(per_date["date"])
    sns.barplot(data=per_date, x="date", y="n_larvae", order=order, color="tab:blue")
    plt.xlabel("Date")
    plt.ylabel("Number of larvae")
    plt.title("Larvae per day (raw dataset)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "larvae_per_day.png", dpi=300)
    plt.close()

    # Body length per day boxplot (per-image means, same date order)
    plt.figure(figsize=(10, 6))
    merged = per_img.merge(per_date[["date"]], on="date")
    sns.boxplot(data=merged, x="date", y="mean_body_length", order=order, color="lightgray")
    plt.xlabel("Date")
    plt.ylabel("Mean body length per image (pixels)")
    plt.title("Raw body length per day (per-image means)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "raw_length_per_day_boxplot.png", dpi=300)
    plt.close()


def _write_text_report(df: pd.DataFrame, per_img: pd.DataFrame, per_date: pd.DataFrame) -> None:
    """Write a plain-text summary report to REP_DIR/raw_dataset_summary.txt."""
    n_dates = df["date"].nunique()
    n_images = df[["date", "image_name"]].drop_duplicates().shape[0]
    n_larvae = len(df)

    # Larvae per image stats
    lpi = per_img["n_larvae"].astype(float)
    lpi_mean = lpi.mean()
    lpi_std = lpi.std()
    lpi_min = lpi.min()
    lpi_max = lpi.max()

    # Body length overall stats (mm)
    bl = df["body_length_mm"].astype(float)
    bl_mean = bl.mean()
    bl_std = bl.std()
    bl_median = bl.median()
    bl_min = bl.min()
    bl_max = bl.max()

    lines = []
    lines.append("RAW DATASET SUMMARY")
    lines.append("-------------------")
    lines.append(f"Dates: {n_dates}")
    lines.append(f"Images: {n_images}")
    lines.append(f"Total larvae: {n_larvae}")
    lines.append("")
    lines.append("Larvae per image:")
    lines.append(f"  mean ± std: {lpi_mean:.2f} ± {lpi_std:.2f}")
    lines.append(f"  min: {lpi_min:.0f}")
    lines.append(f"  max: {lpi_max:.0f}")
    lines.append("")
    lines.append("Body length (mm):")
    lines.append(f"  mean:   {bl_mean:.2f}")
    lines.append(f"  std:    {bl_std:.2f}")
    lines.append(f"  median: {bl_median:.2f}")
    lines.append(f"  range:  [{bl_min:.2f}, {bl_max:.2f}]")

    report_path = REP_DIR / "raw_dataset_summary.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Also print to console
    print("\n" + "RAW DATASET SUMMARY")
    print("-------------------")
    print(f"Dates: {n_dates}")
    print(f"Images: {n_images}")
    print(f"Total larvae: {n_larvae}")
    print("")
    print("Larvae per image:")
    print(f"  mean ± std: {lpi_mean:.2f} ± {lpi_std:.2f}")
    print(f"  min: {lpi_min:.0f}")
    print(f"  max: {lpi_max:.0f}")
    print("")
    print("Body length (pixels):")
    print(f"  mean:   {bl_mean:.2f}")
    print(f"  std:    {bl_std:.2f}")
    print(f"  median: {bl_median:.2f}")
    print(f"  range:  [{bl_min:.2f}, {bl_max:.2f}]")


def _write_per_date_summary(per_date: pd.DataFrame) -> None:
    """Write additional summary statistics by date.

    Saves a human-readable table to reports/per_date_summary.txt and
    prints a compact overview to the console.
    """
    # Sort dates with 3.11 last for readability
    order = _sorted_dates_with_311_last(per_date["date"])
    per_date_sorted = per_date.set_index("date").loc[order].reset_index()

    # Save as a nicely formatted text table
    lines = ["PER-DATE BODY LENGTH STATISTICS (mm)", "-----------------------------------"]
    header = (
        f"{'Date':>8}  {'n':>6}  {'Mean':>8}  {'Std':>8}  "
        f"{'Median':>8}  {'P05':>8}  {'P95':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in per_date_sorted.iterrows():
        lines.append(
            f"{str(r['date']):>8}  "
            f"{int(r['n_larvae']):6d}  "
            f"{r['mean_body_length']:8.2f}  "
            f"{r['std_body_length']:8.2f}  "
            f"{r['median_body_length']:8.2f}  "
            f"{r['p05_body_length']:8.2f}  "
            f"{r['p95_body_length']:8.2f}"
        )

    out_path = REP_DIR / "per_date_summary.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Print a compact console view
    print("\nPER-DATE SUMMARY (mm)")
    print("---------------------")
    for _, r in per_date_sorted.iterrows():
        print(
            f"{r['date']}: n={int(r['n_larvae'])}, "
            f"mean={r['mean_body_length']:.2f} mm, "
            f"median={r['median_body_length']:.2f} mm, "
            f"P05={r['p05_body_length']:.2f}, P95={r['p95_body_length']:.2f}"
        )


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    _ensure_dirs()

    print("======================================================================")
    print("RAW LARVAL DATASET ANALYSIS")
    print("======================================================================")
    print(f"Input directory:  {INPUT_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}\n")

    print("Loading morphometrics...")
    df = _scan_morphometrics()

    # Recompute PCA-based body length from masks
    df = _attach_body_length_from_masks(df)
    print(f"  Loaded {len(df)} larvae from {df['date'].nunique()} dates")

    # 4. Compute statistics
    print("Computing per-image and per-date statistics...")
    per_img = _compute_per_image_stats(df)
    per_date = _compute_per_date_stats(df)

    # 5. Save tables
    print("Saving tables...")
    _save_tables(df, per_img, per_date)

    # 6. Figures
    print("Generating figures...")
    _plot_distributions(df)
    _plot_temporal(per_img, per_date)

    # 7. Text report and console summary
    _write_text_report(df, per_img, per_date)
    _write_per_date_summary(per_date)

    print("\nAnalysis complete. Outputs written to:")
    print(f"  Tables:  {TAB_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Report:  {REP_DIR / 'raw_dataset_summary.txt'}")
    print(f"  Per-date summary: {REP_DIR / 'per_date_summary.txt'}")
    print("======================================================================")


if __name__ == "__main__":  # pragma: no cover
    main()
