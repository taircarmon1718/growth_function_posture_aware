#!/usr/bin/env python3
"""
Visualize raw vs filtered detections using pipeline-consistent filtering:

    (valid_confidence >= VALID_THRESHOLD) AND (predicted_posture == 1)

Key guarantees:
- EXACT image match only (no substring matching)
- Same predictions source as pipeline
- De-duplicated image IDs
- Max 2 images per date
- Deterministic larva_filename -> larva_id -> morphometrics matching
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import csv
import re

import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook


# --- CONFIG ---
PROJECT_ROOT = Path("/Users/taircarmon/Desktop/growth_function_posture_aware")
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_full_binary_masks_only"
PREDICTIONS_XLSX = PROJECT_ROOT / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
VALID_THRESHOLD = 0.85
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRAW_COLOR = (0, 0, 255)  # BGR red
DRAW_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def _read_predictions_rows(xlsx_path: Path) -> list[dict]:
    wb = load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    if header is None:
        raise RuntimeError("Predictions sheet is empty.")

    cols = {str(c).strip().lower(): i for i, c in enumerate(header)}
    required = ["date", "image_name", "larva_filename", "valid_confidence", "predicted_posture"]
    missing = [k for k in required if k not in cols]
    if missing:
        raise RuntimeError(f"Missing required columns in predictions file: {missing}")

    out = []
    for r in rows:
        if r is None:
            continue
        out.append(
            {
                "date": str(r[cols["date"]]).strip(),
                "image_name": str(r[cols["image_name"]]).strip(),
                "larva_filename": str(r[cols["larva_filename"]]).strip(),
                "valid_confidence": _to_float(r[cols["valid_confidence"]]),
                "predicted_posture": _to_int(r[cols["predicted_posture"]]),
            }
        )
    return out


def larva_id_from_filename(name: str):
    if not isinstance(name, str):
        return None
    m = re.search(r"(\d+)", Path(name).stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_centroids(morph_csv: Path) -> dict[int, tuple[float, float]]:
    centroids = {}
    if not morph_csv.exists():
        return centroids
    with open(morph_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                larva_id = int(float(row["larva_id"]))
                cx = float(row["centroid_x"])
                cy = float(row["centroid_y"])
            except Exception:
                continue
            centroids[larva_id] = (cx, cy)
    return centroids


def get_filtered_rows_from_df(pred_rows: list[dict], image_id: str, date_str: str) -> tuple[list[dict], list[dict]]:
    # EXACT MATCH (CRITICAL): no substring matching
    df_img = [r for r in pred_rows if r["image_name"] == image_id and r["date"] == date_str]

    # EXACT pipeline filter:
    filtered_df = [
        r
        for r in df_img
        if r["valid_confidence"] is not None
        and r["valid_confidence"] >= VALID_THRESHOLD
        and r["predicted_posture"] == 1
    ]
    return df_img, filtered_df


def draw_filtered_detections(clean_img_bgr, filtered_rows: list[dict], analysis_dir: Path):
    out = clean_img_bgr.copy()
    centroids = load_centroids(analysis_dir / "morphometrics.csv")
    drawn_count = 0
    missing_coord_count = 0

    for row in filtered_rows:
        larva_filename = row["larva_filename"]
        larva_id = larva_id_from_filename(larva_filename)
        if larva_id is None:
            missing_coord_count += 1
            continue

        mask_path = analysis_dir / "larvae_reports" / larva_filename
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None:
            missing_coord_count += 1
            continue

        h, w = mask.shape[:2]
        if larva_id not in centroids:
            missing_coord_count += 1
            continue

        cx, cy = centroids[larva_id]
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))
        cv2.rectangle(out, (x1, y1), (x2, y2), DRAW_COLOR, DRAW_THICKNESS)
        cv2.putText(out, f"{larva_id}", (x1, max(0, y1 - 6)), FONT, 0.5, DRAW_COLOR, 1, cv2.LINE_AA)
        drawn_count += 1

    return out, drawn_count, missing_coord_count


def find_clean_image(date_folder: str, image_id: str) -> Path | None:
    c1 = PROJECT_ROOT / date_folder / "images" / f"{image_id}.JPG"
    c2 = PROJECT_ROOT / date_folder / "images" / f"{image_id}.jpg"
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    return None


def main():
    if not PREDICTIONS_XLSX.exists():
        raise FileNotFoundError(f"Predictions file missing: {PREDICTIONS_XLSX}")
    print(f"[INFO] Predictions source: {PREDICTIONS_XLSX}")

    pred_rows = _read_predictions_rows(PREDICTIONS_XLSX)
    if not pred_rows:
        print("No prediction rows found.")
        return

    # Remove duplicates
    image_pairs = sorted(set((r["date"], r["image_name"]) for r in pred_rows))

    # Limit to max 2 images per date
    images_per_date = defaultdict(list)
    for d, image_id in image_pairs:
        images_per_date[d].append(image_id)

    filtered_image_pairs = []
    for d, imgs in images_per_date.items():
        filtered_image_pairs.extend((d, img) for img in sorted(set(imgs))[:2])

    print(f"[INFO] Unique date-image pairs: {len(image_pairs)}")
    print(f"[INFO] Processing limited pairs (max 2/date): {len(filtered_image_pairs)}")

    for date_folder, image_id in filtered_image_pairs:
        analysis_dir = ANALYSIS_ROOT / date_folder / image_id
        overlay_path = analysis_dir / "overlay.png"
        clean_img_path = find_clean_image(date_folder, image_id)
        if not overlay_path.exists() or clean_img_path is None:
            continue

        df_img, filtered_df = get_filtered_rows_from_df(pred_rows, image_id, date_folder)
        print(f"[DEBUG] {image_id}: total={len(df_img)}, filtered={len(filtered_df)}")
        if filtered_df:
            print("[DEBUG] first rows used for drawing:")
            for r in filtered_df[:5]:
                print(
                    f"  {r['larva_filename']}, "
                    f"valid_confidence={r['valid_confidence']:.4f}, "
                    f"predicted_posture={r['predicted_posture']}"
                )

        img_overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        img_clean = cv2.imread(str(clean_img_path), cv2.IMREAD_COLOR)
        if img_overlay is None or img_clean is None:
            continue

        img_filtered, drawn_count, missing_coords = draw_filtered_detections(img_clean, filtered_df, analysis_dir)

        if drawn_count == 0:
            if len(filtered_df) == 0:
                print(
                    f"[WARNING] {image_id}: No detections drawn because filtering removed all samples "
                    f"(valid_confidence < {VALID_THRESHOLD} or predicted_posture != 1)."
                )
            else:
                print(
                    f"[WARNING] {image_id}: No detections drawn due to missing coordinates/masks "
                    f"(missing={missing_coords})."
                )

        left_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(left_rgb)
        axes[0].set_title("Raw detections (noisy)", fontsize=14)
        axes[0].axis("off")
        axes[1].imshow(right_rgb)
        axes[1].set_title("Filtered detections (posture-aware)", fontsize=14)
        axes[1].axis("off")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"filtered_detections_{date_folder}_{image_id}.png"
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved: {out_path}")

    print(f"[INFO] Done. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)

#!/usr/bin/env python3
"""
Visualize raw vs filtered detections using pipeline-consistent filtering:

    (valid_confidence >= VALID_THRESHOLD) AND (predicted_posture == 1)

Key guarantees:
- EXACT image match only (no substring matching)
- Same predictions source as pipeline
- De-duplicated image IDs
- Max 2 images per date
- Deterministic larva_filename -> larva_id -> morphometrics matching
"""
# duplicate block retained below; keep inert

from pathlib import Path
from collections import defaultdict
import csv
import re

import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook


# --- CONFIG ---
PROJECT_ROOT = Path("/Users/taircarmon/Desktop/growth_function_posture_aware")
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_full_binary_masks_only"
PREDICTIONS_XLSX = PROJECT_ROOT / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
VALID_THRESHOLD = 0.85
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRAW_COLOR = (0, 0, 255)  # BGR red
DRAW_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def _read_predictions_rows(xlsx_path: Path) -> list[dict]:
    wb = load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    if header is None:
        raise RuntimeError("Predictions sheet is empty.")

    cols = {str(c).strip().lower(): i for i, c in enumerate(header)}
    required = ["date", "image_name", "larva_filename", "valid_confidence", "predicted_posture"]
    missing = [k for k in required if k not in cols]
    if missing:
        raise RuntimeError(f"Missing required columns in predictions file: {missing}")

    out = []
    for r in rows:
        if r is None:
            continue
        out.append(
            {
                "date": str(r[cols["date"]]).strip(),
                "image_name": str(r[cols["image_name"]]).strip(),
                "larva_filename": str(r[cols["larva_filename"]]).strip(),
                "valid_confidence": _to_float(r[cols["valid_confidence"]]),
                "predicted_posture": _to_int(r[cols["predicted_posture"]]),
            }
        )
    return out


def larva_id_from_filename(name: str):
    if not isinstance(name, str):
        return None
    m = re.search(r"(\d+)", Path(name).stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_centroids(morph_csv: Path) -> dict[int, tuple[float, float]]:
    centroids = {}
    if not morph_csv.exists():
        return centroids
    with open(morph_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                larva_id = int(float(row["larva_id"]))
                cx = float(row["centroid_x"])
                cy = float(row["centroid_y"])
            except Exception:
                continue
            centroids[larva_id] = (cx, cy)
    return centroids


def get_filtered_rows_from_df(pred_rows: list[dict], image_id: str, date_str: str) -> tuple[list[dict], list[dict]]:
    # EXACT MATCH (CRITICAL): no substring matching
    df_img = [r for r in pred_rows if r["image_name"] == image_id and r["date"] == date_str]

    # EXACT pipeline filter:
    filtered_df = [
        r
        for r in df_img
        if r["valid_confidence"] is not None
        and r["valid_confidence"] >= VALID_THRESHOLD
        and r["predicted_posture"] == 1
    ]
    return df_img, filtered_df


def draw_filtered_detections(clean_img_bgr, filtered_rows: list[dict], analysis_dir: Path):
    out = clean_img_bgr.copy()
    centroids = load_centroids(analysis_dir / "morphometrics.csv")
    drawn_count = 0
    missing_coord_count = 0

    for row in filtered_rows:
        larva_filename = row["larva_filename"]
        larva_id = larva_id_from_filename(larva_filename)
        if larva_id is None:
            missing_coord_count += 1
            continue

        mask_path = analysis_dir / "larvae_reports" / larva_filename
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None:
            missing_coord_count += 1
            continue

        h, w = mask.shape[:2]
        if larva_id not in centroids:
            missing_coord_count += 1
            continue

        cx, cy = centroids[larva_id]
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))
        cv2.rectangle(out, (x1, y1), (x2, y2), DRAW_COLOR, DRAW_THICKNESS)
        cv2.putText(out, f"{larva_id}", (x1, max(0, y1 - 6)), FONT, 0.5, DRAW_COLOR, 1, cv2.LINE_AA)
        drawn_count += 1

    return out, drawn_count, missing_coord_count


def find_clean_image(date_folder: str, image_id: str) -> Path | None:
    c1 = PROJECT_ROOT / date_folder / "images" / f"{image_id}.JPG"
    c2 = PROJECT_ROOT / date_folder / "images" / f"{image_id}.jpg"
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    return None


def main():
    if not PREDICTIONS_XLSX.exists():
        raise FileNotFoundError(f"Predictions file missing: {PREDICTIONS_XLSX}")
    print(f"[INFO] Predictions source: {PREDICTIONS_XLSX}")

    pred_rows = _read_predictions_rows(PREDICTIONS_XLSX)
    if not pred_rows:
        print("No prediction rows found.")
        return

    # Remove duplicates
    image_pairs = sorted(set((r["date"], r["image_name"]) for r in pred_rows))

    # Limit to max 2 images per date
    images_per_date = defaultdict(list)
    for d, image_id in image_pairs:
        images_per_date[d].append(image_id)

    filtered_image_pairs = []
    for d, imgs in images_per_date.items():
        filtered_image_pairs.extend((d, img) for img in sorted(set(imgs))[:2])

    print(f"[INFO] Unique date-image pairs: {len(image_pairs)}")
    print(f"[INFO] Processing limited pairs (max 2/date): {len(filtered_image_pairs)}")

    for date_folder, image_id in filtered_image_pairs:
        analysis_dir = ANALYSIS_ROOT / date_folder / image_id
        overlay_path = analysis_dir / "overlay.png"
        clean_img_path = find_clean_image(date_folder, image_id)

        if not overlay_path.exists() or clean_img_path is None:
            continue

        df_img, filtered_df = get_filtered_rows_from_df(pred_rows, image_id, date_folder)
        print(f"[DEBUG] {image_id}: total={len(df_img)}, filtered={len(filtered_df)}")
        if len(df_img) > 0:
            print("[DEBUG] first rows used for drawing:")
            for r in filtered_df[:5]:
                print(
                    f"  {r['larva_filename']}, "
                    f"valid_confidence={r['valid_confidence']:.4f}, "
                    f"predicted_posture={r['predicted_posture']}"
                )

        img_overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        img_clean = cv2.imread(str(clean_img_path), cv2.IMREAD_COLOR)
        if img_overlay is None or img_clean is None:
            continue

        img_filtered, drawn_count, missing_coords = draw_filtered_detections(img_clean, filtered_df, analysis_dir)

        if drawn_count == 0:
            if len(filtered_df) == 0:
                print(
                    f"[WARNING] {image_id}: No detections drawn because filtering removed all samples "
                    f"(valid_confidence < {VALID_THRESHOLD} or predicted_posture != 1)."
                )
            else:
                print(
                    f"[WARNING] {image_id}: No detections drawn due to missing coordinates/masks "
                    f"(missing={missing_coords})."
                )

        left_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(left_rgb)
        axes[0].set_title("Raw detections (noisy)", fontsize=14)
        axes[0].axis("off")
        axes[1].imshow(right_rgb)
        axes[1].set_title("Filtered detections (posture-aware)", fontsize=14)
        axes[1].axis("off")
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"filtered_detections_{date_folder}_{image_id}.png"
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved: {out_path}")

    print(f"[INFO] Done. Output directory: {OUTPUT_DIR}")


if False and __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Show raw overlay vs posture-aware filtered detections for one or several images.

This script scans the analysis_full_binary_masks_only root for overlay.png files and for
each image it loads predictions, filters by the pipeline rule
(valid_confidence >= VALID_THRESHOLD AND predicted_posture == 1), draws filtered
detections on the clean original image, and saves a side-by-side figure to outputs/figures/.

Designed to be run from the project environment where dependencies (pandas, openpyxl, cv2)
are available.
"""
from pathlib import Path
import csv
import sys
import re
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
PROJECT_ROOT = Path("/Users/taircarmon/Desktop/growth_function_posture_aware")
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_full_binary_masks_only"
PREDICTIONS_XLSX = PROJECT_ROOT / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
VALID_THRESHOLD = 0.85

DRAW_COLOR = (0, 0, 255)  # BGR red
DRAW_THICKNESS = 3
POINT_RADIUS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- helpers (kept from previous script) ---

def _to_int01(v) -> int | None:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if abs(f - 1.0) < 1e-8:
        return 1
    if abs(f - 0.0) < 1e-8:
        return 0
    return None


def _to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_centroids(morph_csv: Path) -> dict[int, tuple[float, float]]:
    centroids = {}
    if not morph_csv.exists():
        return centroids
    with open(morph_csv, "r", newline="") as f:
        # try to detect header
        header = f.readline().strip().split(",")
        f.seek(0)
        try:
            reader = csv.DictReader(f)
            if "larva_id" in reader.fieldnames and "centroid_x" in reader.fieldnames and "centroid_y" in reader.fieldnames:
                for row in reader:
                    try:
                        larva_id = int(float(row["larva_id"]))
                        cx = float(row["centroid_x"])
                        cy = float(row["centroid_y"])
                        centroids[larva_id] = (cx, cy)
                    except Exception:
                        continue
                return centroids
        except Exception:
            pass
        # fallback: read as raw CSV with numeric columns; assume last two columns are centroid x,y and first is id
        f.seek(0)
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) < 3:
                continue
            try:
                larva_id = int(float(parts[0]))
                cx = float(parts[-2])
                cy = float(parts[-1])
                centroids[larva_id] = (cx, cy)
            except Exception:
                continue
    return centroids


def larva_id_from_filename(name: str) -> int | None:
    if not isinstance(name, str):
        return None
    stem = Path(name).stem  # larva_030
    m = re.search(r"(\d+)", stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def draw_filtered_detections(clean_img_bgr: np.ndarray, filtered_rows: list[dict], analysis_dir: Path) -> tuple[np.ndarray, int, int]:
    out = clean_img_bgr.copy()
    centroids = load_centroids(analysis_dir / "morphometrics.csv")
    drawn_count = 0
    missing_coord_count = 0

    for row in filtered_rows:
        larva_filename = row.get("larva_filename")
        larva_id = larva_id_from_filename(larva_filename)
        if larva_id is None:
            missing_coord_count += 1
            continue

        # try to find larva mask in larvae_reports
        mask_path = analysis_dir / "larvae_reports" / larva_filename
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # fallback to centroid only
                pass
            else:
                # if mask loaded but centroids present, we still prefer centroid positioning
                pass

        if larva_id in centroids:
            cx, cy = centroids[larva_id]
            # draw a small rectangle centered at centroid using mask size if available
            x1 = int(round(cx - 10))
            y1 = int(round(cy - 10))
            x2 = int(round(cx + 10))
            y2 = int(round(cy + 10))
            cv2.rectangle(out, (x1, y1), (x2, y2), DRAW_COLOR, DRAW_THICKNESS)
            cv2.putText(out, f"{larva_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_COLOR, 1, cv2.LINE_AA)
            drawn_count += 1
        else:
            missing_coord_count += 1

    return out, drawn_count, missing_coord_count


# --- main multi-image processing ---

def get_filtered_rows_from_df(
    pred_df: pd.DataFrame,
    image_id: str,
    date_value: str | None = None,
    valid_threshold: float = VALID_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # normalize column names lower-case mapping
    cols = {c.lower(): c for c in pred_df.columns}
    required = ["image_name", "larva_filename", "valid_confidence", "predicted_posture", "date"]
    for r in required:
        if r not in cols:
            raise RuntimeError(f"Predictions file missing required column: {r}")

    date_col = cols["date"]
    image_col = cols["image_name"]
    valid_col = cols["valid_confidence"]
    posture_col = cols["predicted_posture"]

    # EXACT match only (no substring matching).
    df_img = pred_df[pred_df[image_col].astype(str) == image_id].copy()
    if date_value is not None:
        df_img = df_img[df_img[date_col].astype(str) == str(date_value)].copy()

    # EXACT pipeline filter: (valid_confidence >= VALID_THRESHOLD) AND (predicted_posture == 1)
    v = pd.to_numeric(df_img[valid_col], errors="coerce")
    p = pd.to_numeric(df_img[posture_col], errors="coerce")
    filtered_df = df_img[(v >= valid_threshold) & (p == 1)].copy()
    return df_img, filtered_df


def main():
    if not ANALYSIS_ROOT.exists():
        print("ERROR: analysis root not found:", ANALYSIS_ROOT)
        sys.exit(1)
    if not PREDICTIONS_XLSX.exists():
        print("ERROR: predictions file not found:", PREDICTIONS_XLSX)
        sys.exit(1)

    # load predictions once
    try:
        pred_df = pd.read_excel(PREDICTIONS_XLSX)
    except Exception as e:
        print("ERROR reading predictions XLSX:", e)
        sys.exit(1)

    # detect required columns (case-insensitive)
    cols_lower = {c.lower(): c for c in pred_df.columns}
    required = ["date", "image_name", "larva_filename", "valid_confidence", "predicted_posture"]
    missing = [c for c in required if c not in cols_lower]
    if missing:
        print("ERROR: missing required columns in predictions:", missing)
        print("Available columns:", list(pred_df.columns))
        sys.exit(1)
    date_col = cols_lower["date"]
    image_col = cols_lower["image_name"]
    larva_col = cols_lower["larva_filename"]

    # Build deduplicated (date, image_id) pairs.
    date_image_pairs = []
    for _, row in pred_df[[date_col, image_col]].dropna().drop_duplicates().iterrows():
        d = str(row[date_col]).strip()
        img = str(row[image_col]).strip()
        if not d or not img:
            continue
        date_image_pairs.append((d, img))
    # Remove duplicates robustly and sort
    date_image_pairs = sorted(set(date_image_pairs), key=lambda x: (x[0], x[1]))

    if not date_image_pairs:
        print("No image IDs found in predictions file.")
        sys.exit(0)

    # Group by date and keep at most 2 images per date.
    images_per_date = defaultdict(list)
    for d, img in date_image_pairs:
        images_per_date[d].append(img)
    filtered_pairs = []
    for d, imgs in images_per_date.items():
        for img in sorted(set(imgs))[:2]:
            filtered_pairs.append((d, img))
    filtered_pairs = sorted(filtered_pairs, key=lambda x: (x[0], x[1]))

    print(
        f"Found {len(date_image_pairs)} unique date-image pairs in predictions; "
        f"processing {len(filtered_pairs)} pairs (max 2 images per date)."
    )

    for date_folder, image_id in filtered_pairs:
        analysis_dir = ANALYSIS_ROOT / date_folder / image_id
        overlay_path = analysis_dir / "overlay.png"
        clean_img_path = PROJECT_ROOT / date_folder / "images" / f"{image_id}.JPG"
        if not overlay_path.exists():
            continue
        if not clean_img_path.exists():
            alt = PROJECT_ROOT / date_folder / "images" / f"{image_id}.jpg"
            if alt.exists():
                clean_img_path = alt
            else:
                # no clean image, skip
                continue

        # EXACT match only + exact pipeline filter logic.
        df_img, filtered_df = get_filtered_rows_from_df(
            pred_df=pred_df,
            image_id=image_id,
            date_value=date_folder,
            valid_threshold=VALID_THRESHOLD,
        )
        print(f"[DEBUG] {image_id}: total={len(df_img)}, filtered={len(filtered_df)}")
        if df_img.empty:
            continue

        # collect filtered rows into simple dicts
        filtered_rows = []
        for _, r in filtered_df.iterrows():
            larva_fn = r[larva_col]
            filtered_rows.append({"larva_filename": str(larva_fn) if larva_fn is not None else None})

        print(f"\nProcessing overlay {image_id} (date {date_folder}): total preds={len(df_img)}, filtered={len(filtered_rows)})")

        img_overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        img_clean = cv2.imread(str(clean_img_path), cv2.IMREAD_COLOR)
        if img_overlay is None or img_clean is None:
            continue

        # draw using masks/morphometrics
        img_filtered, drawn_count, missing = draw_filtered_detections(img_clean, filtered_rows, analysis_dir)
        print(" Drawn detections:", drawn_count, "missing coords:", missing)

        # If no detections passed the filter, annotate the right image with a clear message
        if drawn_count == 0:
            h_img, w_img = img_filtered.shape[:2]
            msg = "No filtered detections (valid_conf >= {:.2f} & posture==1)".format(VALID_THRESHOLD)
            # place message near top-left with background rectangle for readability
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
            pad = 8
            cv2.rectangle(img_filtered, (10 - pad, 10 - pad), (10 + tw + pad, 10 + th + pad), (255, 255, 255), -1)
            cv2.putText(img_filtered, msg, (10, 10 + th), FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # compose combined image and save with cv2 for speed
        try:
            left = img_overlay.copy()
            right = img_filtered.copy()
            if left.shape[0] != right.shape[0]:
                h = min(left.shape[0], right.shape[0])
                left = left[:h, :]
                right = right[:h, :]

            # draw titles
            def draw_title(img, text, x=10, y=30):
                cv2.putText(img, text, (x, y), FONT, 0.8, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(img, text, (x, y), FONT, 0.8, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            draw_title(left, "Raw detections (noisy)")
            draw_title(right, "Filtered detections (growth-consistent)")
            combined = np.hstack([left, right])
            out_path = OUTPUT_DIR / f"filtered_detections_{date_folder}_{image_id}.png"
            if cv2.imwrite(str(out_path), combined):
                print(" Saved:", out_path)
            else:
                print("Failed to write:", out_path)
        except Exception as e:
            print("Error saving combined image for", image_id, "->", e)
        finally:
            plt.close('all')

    print("Processing complete. Figures saved to:", OUTPUT_DIR)
