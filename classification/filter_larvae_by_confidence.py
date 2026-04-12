#!/usr/bin/env python3
"""
Filter Larvae by Confidence Score
==================================

Inference-time filtering (aligned with dual_larva_pipeline_full_labels.predict_all):

- predicted_valid = 1 iff P(valid=1) >= 0.85; valid_confidence = proba[valid][predicted class]
- predicted_posture = argmax(posture proba); posture_confidence = proba[posture][predicted class]
- KEEP iff predicted_valid==1 AND valid_confidence>=0.85 AND predicted_posture==1

Feature rows use cache flag _valid only to mean “features extracted” (not a training label filter).

Cache columns are optionally audited against extract_features(); mismatches only warn unless
FILTER_STRICT_FEATURE_CACHE=1. Larva PNG must exist and be readable; rows without a valid
extract_features() result are skipped (no zero-filled fabricated vectors).

Output:
- Copies filtered larvae to filtered_larvae_by_date/<date>/
- Original crops from full-res photo when available (no boxes); else cleaned overlay, not binary masks
"""

from pathlib import Path
import os
import sys
import shutil

import json
import numpy as np
import pandas as pd
import joblib
import cv2
from sklearn.decomposition import PCA as skPCA

from larva_feature_extraction import (
    FEATURE_KEYS,
    assert_model_input_matrix,
    cache_row_matches_extract,
    extract_features,
    features_dict_to_vector,
    inference_features_from_cache_row,
)

ROOT_DIR = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full_binary_masks_only"
CACHE_FILE = ROOT_DIR / "dual_larva_models_geodesic2" / "cached_all_features.pkl"
VALID_MODEL_FILE = ROOT_DIR / "dual_larva_models_geodesic2" / "valid_model" / "model.pkl"
POSTURE_MODEL_FILE = ROOT_DIR / "dual_larva_models_geodesic2" / "posture_model" / "model.pkl"
PREDICTIONS_XLSX = ROOT_DIR / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
TRAINING_FEATURE_STATS_JSON = ROOT_DIR / "dual_larva_models_geodesic2" / "training_feature_stats.json"
OUTPUT_DIR = ROOT_DIR / "filtered_larvae_by_date"

VALID_THRESHOLD = 0.85

# If set, any cache vs extract_features mismatch raises (debugging). Default: warn only.
_STRICT_CACHE_AUDIT = os.environ.get("FILTER_STRICT_FEATURE_CACHE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Default OFF: match training predict_all (7 feats from pkl + body_length from extract).
# Set to 1 to rebuild all 8 from extract_features() — can change who passes the filter vs xlsx.
_FULL_EXTRACT = os.environ.get("FILTER_FULL_EXTRACT_FEATURES", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

EXCLUDED_DATES = {'18.10', '18.1'}

PIXEL_TO_MM = 0.232255814


def fix_date(d):
    """Normalize date string."""
    s = str(d).strip()
    if s.endswith('.0'):
        s = s[:-2]
    parts = s.split('.')
    if len(parts) == 2:
        day, mon = parts[0].strip(), parts[1].strip()
        if mon == '1':
            return f"{day}.10"
        return f"{day}.{mon}"
    return s


def date_sort_key(name):
    """Sort key for dates."""
    try:
        day, mon = str(name).split('.')
        return (int(mon), int(day))
    except Exception:
        return (999, 999)


def is_excluded(date_str):
    """Check if date is excluded."""
    return fix_date(date_str) in EXCLUDED_DATES or str(date_str) in EXCLUDED_DATES


def create_pca_length_visualization(mask_image_path: str, output_path: str) -> float:
    """Compute PCA length from a binary larva mask and draw a minimal overlay (paper-style)."""
    img = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    # Treat any non-zero pixel as foreground. No re-threshold, no inversion.
    mask = (img > 0).astype(np.uint8)
    H, W = mask.shape

    if np.count_nonzero(mask) < 5:
        return 0.0

    # Coordinates over the FULL image
    ys, xs = np.where(mask > 0)
    coords = np.stack([ys, xs], axis=1)          # (row, col)
    points = coords[:, [1, 0]].astype(float)     # (x, y)

    # PCA length (pipeline style) in pixels
    centroid = points.mean(axis=0)
    pca = skPCA(n_components=1)
    pca.fit(points)
    direction = pca.components_[0]

    centered = points - centroid
    projections = centered @ direction
    min_proj = float(projections.min())
    max_proj = float(projections.max())
    length_px = float(max_proj - min_proj)
    length_mm = length_px * PIXEL_TO_MM

    # Visualization on full-size canvas: white larva on black
    base_uint8 = (mask * 255).astype(np.uint8)
    vis = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)

    # Endpoints on PCA axis (in full-image coordinates)
    p1 = centroid + min_proj * direction
    p2 = centroid + max_proj * direction
    pt1 = (int(round(p1[0])), int(round(p1[1])))
    pt2 = (int(round(p2[0])), int(round(p2[1])))

    cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
    cv2.circle(vis, pt1, 2, (0, 0, 255), -1)
    cv2.circle(vis, pt2, 2, (255, 0, 0), -1)

    text = f"{length_mm:.2f} mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.25
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    margin = 2
    x = margin
    y = margin + text_size[1]
    if y + 2 > H:
        y = max(text_size[1] + 2, H - 2)
    if x + text_size[0] + 2 > W:
        x = max(2, W - text_size[0] - 2)

    cv2.putText(vis, text, (x, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)
    return length_px


def load_cache() -> pd.DataFrame:
    """Load feature cache."""
    sys.stdout.write(f"Loading cache: {CACHE_FILE}\n")
    sys.stdout.flush()
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache not found: {CACHE_FILE}")
    sys.stdout.write("Cache file found, loading...\n")
    sys.stdout.flush()
    df = joblib.load(CACHE_FILE)
    sys.stdout.write(f"  ✓ Loaded {len(df)} larvae from cache\n")
    sys.stdout.flush()
    return df


def load_valid_model():
    """Load valid (quality) classifier."""
    sys.stdout.write(f"Loading valid model: {VALID_MODEL_FILE}\n")
    sys.stdout.flush()
    if not VALID_MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {VALID_MODEL_FILE}")
    model = joblib.load(VALID_MODEL_FILE)
    sys.stdout.write("  ✓ Valid model loaded\n")
    sys.stdout.flush()
    return model


def load_posture_model():
    """Load posture classifier."""
    sys.stdout.write(f"Loading posture model: {POSTURE_MODEL_FILE}\n")
    sys.stdout.flush()
    if not POSTURE_MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {POSTURE_MODEL_FILE}")
    model = joblib.load(POSTURE_MODEL_FILE)
    sys.stdout.write("  ✓ Posture model loaded\n")
    sys.stdout.flush()
    return model


def _warn_if_classes_not_standard(model, name: str) -> None:
    """Ensure proba[:, 1] is P(positive class 1) for binary sklearn estimators."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return
    c = np.asarray(classes).ravel()
    if len(c) != 2:
        sys.stdout.write(
            f"  ⚠️  {name}: expected 2 classes, got classes_={c!r} — verify proba[:, 1] semantics.\n"
        )
        sys.stdout.flush()
        return
    if int(c[1]) != 1:
        sys.stdout.write(
            f"  ⚠️  {name}: classes_[1] != 1 (classes_={c!r}) — proba[:, 1] may not be P(class 1).\n"
        )
        sys.stdout.flush()


def run_inference_training_aligned(valid_model, posture_model, X: np.ndarray):
    """
    Match dual_larva_pipeline_full_labels.predict_all:

    - v_pred = 1 iff P(valid=1) >= VALID_THRESHOLD; valid_confidence = v_prob[v_pred] (column for predicted class).
    - Posture only defined in training after v_pred==1; here we still run posture proba on all rows for batching,
      but the keep rule uses v_pred and p_pred = argmax(posture_proba) like the workbook.

    Returns:
      v_pred, v_conf (workbook-style), p_pred (argmax), p_conf (prob of predicted posture class),
      v_prob1 (P(valid=1)), p_prob1 (P(posture=1)) for diagnostics.
    """
    v_proba = valid_model.predict_proba(X)
    n = v_proba.shape[0]
    v_prob1 = v_proba[:, 1].astype(float)
    v_pred = (v_prob1 >= VALID_THRESHOLD).astype(np.int32)
    v_conf = v_proba[np.arange(n), v_pred.astype(int)].astype(float)

    p_proba = posture_model.predict_proba(X)
    p_pred = np.argmax(p_proba, axis=1).astype(np.int32)
    p_conf = p_proba[np.arange(n), p_pred].astype(float)
    p_prob1 = p_proba[:, 1].astype(float)

    return v_pred, v_conf, p_pred, p_conf, v_prob1, p_prob1


def inference_keep_mask_training_rule(
    v_pred: np.ndarray,
    v_conf: np.ndarray,
    p_pred: np.ndarray,
) -> np.ndarray:
    """Same as workbook intent: predicted_valid==1 AND valid_confidence>=thr AND predicted_posture==1."""
    return (v_pred == 1) & (v_conf >= VALID_THRESHOLD) & (p_pred == 1)


def _print_prob_distribution_line(tag: str, a: np.ndarray) -> None:
    a = np.asarray(a, dtype=float)
    sys.stdout.write(
        f"  {tag}: min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}  "
        f"median={float(np.median(a)):.4f}\n"
    )
    sys.stdout.flush()


def _print_prob_distribution_extended(tag: str, a: np.ndarray) -> None:
    a = np.asarray(a, dtype=float)
    qs = (10, 25, 75, 90)
    pv = [float(np.percentile(a, q)) for q in qs]
    sys.stdout.write(
        f"  {tag} percentiles p10,p25,p75,p90: {pv[0]:.4f}, {pv[1]:.4f}, {pv[2]:.4f}, {pv[3]:.4f}\n"
    )
    sys.stdout.flush()


def _compare_counts_to_saved_predictions(n_keep_script: int) -> bool:
    """
    Warn if exported workbook row counts disagree (xlsx may use different valid_confidence semantics).
    Returns True if counts differ enough to trigger extended debug output.
    """
    if not PREDICTIONS_XLSX.exists():
        sys.stdout.write(f"  (No reference file at {PREDICTIONS_XLSX} — skip cross-check.)\n")
        sys.stdout.flush()
        return False
    try:
        ref = pd.read_excel(PREDICTIONS_XLSX)
    except Exception as e:
        sys.stdout.write(f"  ⚠️  Could not read predictions workbook for cross-check: {e}\n")
        sys.stdout.flush()
        return False
    need = {"predicted_valid", "predicted_posture"}
    if not need.issubset(set(ref.columns)):
        sys.stdout.write("  ⚠️  Predictions workbook missing predicted_valid / predicted_posture — skip cross-check.\n")
        sys.stdout.flush()
        return False
    pv = pd.to_numeric(ref["predicted_valid"], errors="coerce")
    pp = pd.to_numeric(ref["predicted_posture"], errors="coerce")
    n_xlsx = int(((pv == 1) & (pp == 1)).sum())
    sys.stdout.write(
        f"  Reference (xlsx rows with predicted_valid==1 AND predicted_posture==1): {n_xlsx}\n"
    )
    sys.stdout.flush()
    diff = abs(n_keep_script - n_xlsx)
    denom = max(n_keep_script, n_xlsx, 1)
    significant = diff > 20 or (diff / denom) > 0.03
    if significant:
        sys.stdout.write(
            "  ⚠️  WARNING: kept larvae count differs notably from xlsx flag counts.\n"
            "     If FILTER_FULL_EXTRACT_FEATURES=1, X differs from predict_all (cache-based).\n"
            "     Otherwise small drift can come from sklearn version or row ordering.\n"
        )
        sys.stdout.flush()
    return significant


def _resolve_larva_image_path(row: pd.Series) -> Path:
    p = row.get("_img_path")
    if p is not None and str(p).strip():
        return Path(str(p))
    date_str = fix_date(row["date"])
    image_name = str(row["image_name"])
    larva_filename = str(row["larva_filename"])
    return ANALYSIS_DIR / date_str / image_name / "larvae_reports" / larva_filename


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build X from real larva PNGs (``extract_features``). Rows with missing/unreadable images are dropped.

    - **Default:** same representation as ``predict_all``: cached scalars from the pkl when present,
      any missing cache field filled from the same ``extract_features`` call (never synthetic zeros).
      ``body_length`` always from the image.
    - **FILTER_FULL_EXTRACT_FEATURES=1:** all eight from ``extract_features`` only (can disagree with xlsx).

    Optional audit only in full-extract mode.
    """
    df_valid = df[df["_valid"] == True].copy()
    if _FULL_EXTRACT:
        sys.stdout.write(
            f"Building X via FULL extract_features() ({len(df_valid)} larvae) "
            f"[FILTER_FULL_EXTRACT_FEATURES=1 — may NOT match predictions_all_larvae.xlsx] …\n"
        )
    else:
        sys.stdout.write(
            f"Building X pipeline-aligned: pkl + refreshed body_length ({len(df_valid)} larvae) "
            f"[same as dual_larva_pipeline_full_labels.predict_all] …\n"
        )
    if _STRICT_CACHE_AUDIT and _FULL_EXTRACT:
        sys.stdout.write("  (FILTER_STRICT_FEATURE_CACHE=1 — strict cache audit enabled)\n")
    sys.stdout.flush()

    rows_out: list[np.ndarray] = []
    kept_rows: list[pd.Series] = []
    n_mismatch_rows = 0
    mismatch_samples: list[str] = []
    n_skipped_no_image = 0

    for idx, (_, row) in enumerate(df_valid.iterrows()):
        if idx > 0 and idx % 1000 == 0:
            sys.stdout.write(f"  Processed {idx}/{len(df_valid)}\n")
            sys.stdout.flush()

        img_path = _resolve_larva_image_path(row)

        if _FULL_EXTRACT:
            feats = extract_features(img_path)
            if feats is None:
                n_skipped_no_image += 1
                continue
            row_mismatch = False
            for k in FEATURE_KEYS:
                if k == "body_length":
                    continue
                ok = cache_row_matches_extract(row, feats, key=k, strict=_STRICT_CACHE_AUDIT)
                if not ok:
                    row_mismatch = True
                    if _STRICT_CACHE_AUDIT:
                        raise RuntimeError(
                            f"FEATURE CACHE MISMATCH (strict): column '{k}' vs extract_features() "
                            f"for {img_path}."
                        )
            if row_mismatch:
                n_mismatch_rows += 1
                if len(mismatch_samples) < 5:
                    mismatch_samples.append(str(img_path))
        else:
            feats = inference_features_from_cache_row(
                row, img_path=img_path, full_extract=False
            )
            if feats is None:
                n_skipped_no_image += 1
                continue

        rows_out.append(features_dict_to_vector(feats))
        kept_rows.append(row.copy())

    if n_skipped_no_image:
        sys.stdout.write(
            f"  ⚠️  Skipped {n_skipped_no_image} rows (larva PNG missing or extract_features failed — no fake zeros).\n"
        )
        sys.stdout.flush()

    if not rows_out:
        raise RuntimeError(
            "No rows left after feature extraction (all images missing or unreadable?)."
        )

    df_valid = pd.DataFrame(kept_rows).reset_index(drop=True)
    X = np.vstack(rows_out).astype(np.float64, copy=False)
    assert_model_input_matrix(X)
    sys.stdout.write(f"  ✓ Feature matrix {X.shape} (order={FEATURE_KEYS})\n")
    if _FULL_EXTRACT and n_mismatch_rows > 0 and not _STRICT_CACHE_AUDIT:
        sys.stdout.write(
            f"  ⚠️  Cache audit: {n_mismatch_rows}/{len(df_valid)} rows differ from pkl on ≥1 feature.\n"
        )
        for s in mismatch_samples:
            sys.stdout.write(f"      example: {s}\n")
        sys.stdout.flush()
    sys.stdout.flush()
    return X, df_valid


def _print_feature_matrix_diagnostics(X: np.ndarray) -> None:
    sys.stdout.write("Inference feature matrix diagnostics (per column):\n")
    sys.stdout.flush()
    for j, name in enumerate(FEATURE_KEYS):
        col = X[:, j]
        sys.stdout.write(
            f"  {name:20s}  mean={float(np.mean(col)):.6g}  std={float(np.std(col)):.6g}  "
            f"min={float(np.min(col)):.6g}  max={float(np.max(col)):.6g}\n"
        )
        sys.stdout.flush()

    if not TRAINING_FEATURE_STATS_JSON.exists():
        sys.stdout.write(
            f"  (No {TRAINING_FEATURE_STATS_JSON.name} — run training pipeline to enable distribution comparison.)\n"
        )
        sys.stdout.flush()
        return

    try:
        with open(TRAINING_FEATURE_STATS_JSON, encoding="utf-8") as fh:
            ref = json.load(fh)
    except Exception as e:
        sys.stdout.write(f"  ⚠️  Could not load training stats JSON: {e}\n")
        sys.stdout.flush()
        return

    warn = False
    for j, name in enumerate(FEATURE_KEYS):
        col = X[:, j]
        cm, cs = float(np.mean(col)), float(np.std(col))
        r = ref.get(name)
        if not r:
            continue
        rm, rs = float(r["mean"]), float(r["std"])
        if abs(rm) > 1e-12 and abs(cm - rm) / abs(rm) > 0.25:
            warn = True
        if rs > 1e-12 and abs(cs - rs) / rs > 2.0:
            warn = True
    if warn:
        sys.stdout.write(
            "  ⚠️  WARNING: At least one feature mean/std differs notably from "
            "training_feature_stats.json (expected if inference cohort ≠ training mix).\n"
        )
        sys.stdout.flush()


def _strip_overlay_drawings_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """Remove colored detection overlays (boxes, IDs) from an overlay crop; keep larva/petri appearance."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim != 3:
        return crop_bgr
    b = crop_bgr[:, :, 0].astype(np.int16)
    g = crop_bgr[:, :, 1].astype(np.int16)
    r = crop_bgr[:, :, 2].astype(np.int16)
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    anno_mask = ((chroma > 22).astype(np.uint8) * 255)
    if np.count_nonzero(anno_mask) == 0:
        return crop_bgr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    anno_mask = cv2.dilate(anno_mask, k, iterations=1)
    return cv2.inpaint(crop_bgr, anno_mask, 3, cv2.INPAINT_TELEA)


def _crop_overlay_for_larva(date_str: str, image_name: str, larva_filename: str) -> np.ndarray | None:
    """Crop larva region: prefer raw photo (no boxes), else overlay with drawings stripped, else segmentation."""
    overlay_path = ANALYSIS_DIR / date_str / image_name / 'overlay.png'
    seg_path = ANALYSIS_DIR / date_str / image_name / 'segmentation_mask.png'
    original_jpg = ROOT_DIR / date_str / 'images' / f'{image_name}.JPG'
    original_jpg_lower = ROOT_DIR / date_str / 'images' / f'{image_name}.jpg'
    mask_path = ANALYSIS_DIR / date_str / image_name / 'larvae_reports' / larva_filename
    morph_path = ANALYSIS_DIR / date_str / image_name / 'morphometrics.csv'

    if (not mask_path.exists()) or (not morph_path.exists()):
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Determine desired crop size from the mask image dimensions, with ~20% padding
    h, w = mask.shape[:2]
    if h < 2 or w < 2:
        return None

    PAD_FRAC = 0.20
    h_pad = int(round(h * PAD_FRAC))
    w_pad = int(round(w * PAD_FRAC))
    h2 = h + 2 * h_pad
    w2 = w + 2 * w_pad

    # Parse larva_id from filename: larva_001.png -> 1
    try:
        stem = Path(larva_filename).stem
        larva_id = int(stem.split('_')[-1])
    except Exception:
        return None

    # Read centroid from morphometrics.csv
    try:
        mdf = pd.read_csv(morph_path)
        row = mdf[mdf['larva_id'] == larva_id]
        if len(row) == 0:
            return None
        cx = float(row.iloc[0]['centroid_x'])
        cy = float(row.iloc[0]['centroid_y'])
    except Exception:
        return None

    # 1) Original RGB (no pipeline boxes), 2) overlay with drawings removed, 3) segmentation last
    src = None
    used_overlay = False
    if original_jpg.exists():
        src = cv2.imread(str(original_jpg), cv2.IMREAD_COLOR)
    if src is None and original_jpg_lower.exists():
        src = cv2.imread(str(original_jpg_lower), cv2.IMREAD_COLOR)
    if src is None and overlay_path.exists():
        src = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        used_overlay = True
    if src is None and seg_path.exists():
        seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if seg is not None:
            src = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)

    if src is None:
        return None

    # Crop centered on centroid with padded size
    x0 = int(round(cx - w2 / 2))
    y0 = int(round(cy - h2 / 2))
    x1 = x0 + w2
    y1 = y0 + h2

    # Clamp to image bounds
    x0 = max(0, min(x0, src.shape[1] - 1))
    y0 = max(0, min(y0, src.shape[0] - 1))
    x1 = max(1, min(x1, src.shape[1]))
    y1 = max(1, min(y1, src.shape[0]))

    crop = src[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    # Ensure crop is exactly (h2, w2) by padding if needed
    if crop.shape[0] != h2 or crop.shape[1] != w2:
        out = np.zeros((h2, w2, 3), dtype=np.uint8)
        out[:crop.shape[0], :crop.shape[1]] = crop
        crop = out

    if used_overlay:
        crop = _strip_overlay_drawings_bgr(crop)

    return crop


def main():
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()
    sys.stdout.write("FILTER LARVAE BY CONFIDENCE\n")
    sys.stdout.flush()
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()
    sys.stdout.write(f"VALID_THRESHOLD = {VALID_THRESHOLD}\n")
    sys.stdout.flush()
    sys.stdout.write(
        "Posture decision: argmax(posture proba) — matches training predict_all (not fixed 0.5 on p[:,1]).\n"
    )
    sys.stdout.flush()
    if _FULL_EXTRACT:
        sys.stdout.write("  FILTER_FULL_EXTRACT_FEATURES=1 (X from full extract; compare to xlsx carefully).\n")
    else:
        sys.stdout.write("  Default X = pkl + body_length refresh (aligned with predict_all).\n")
    sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Load cache and models
    cache_df = load_cache()
    valid_model = load_valid_model()
    posture_model = load_posture_model()
    _warn_if_classes_not_standard(valid_model, "valid_model")
    _warn_if_classes_not_standard(posture_model, "posture_model")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Prepare features
    X, df_valid = prepare_features(cache_df)
    assert_model_input_matrix(X)
    _print_feature_matrix_diagnostics(X)
    sys.stdout.write(f"Feature matrix: {X.shape}  dtype={X.dtype}  n_features={X.shape[1]}\n")
    sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Predict (training-aligned)
    sys.stdout.write("Running predictions (training-aligned: v_pred threshold + posture argmax)...\n")
    sys.stdout.flush()
    try:
        v_pred, v_conf, p_pred, p_conf, v_prob1, p_prob1 = run_inference_training_aligned(
            valid_model, posture_model, X
        )
    except Exception as e:
        sys.stdout.write(f"❌ Prediction failed: {e}\n")
        sys.stdout.flush()
        return

    keep_mask = inference_keep_mask_training_rule(v_pred, v_conf, p_pred)
    n_eval = len(df_valid)
    n_pass_v = int(np.sum(v_pred == 1))
    n_pass_p = int(np.sum(p_pred == 1))
    n_keep = int(np.sum(keep_mask))

    sys.stdout.write("Filtering stages (all evaluated larvae):\n")
    sys.stdout.flush()
    sys.stdout.write(f"  predicted_valid==1 (P(valid=1)>={VALID_THRESHOLD}): {n_pass_v}\n")
    sys.stdout.flush()
    sys.stdout.write(f"  predicted_posture==1 (argmax posture):               {n_pass_p}\n")
    sys.stdout.flush()
    sys.stdout.write(
        f"  final keep (valid==1 AND valid_conf>={VALID_THRESHOLD} AND posture==1): {n_keep}\n"
    )
    sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("After valid filter (predicted_valid==1):", n_pass_v)
    print("After full training rule (keep):", n_keep)
    sys.stdout.flush()

    _print_prob_distribution_line("P(valid=1) = v_prob[:,1]", v_prob1)
    _print_prob_distribution_line("P(posture=1) = p_prob[:,1]", p_prob1)
    sys.stdout.flush()

    kept_indices = np.where(keep_mask)[0]
    df_kept = df_valid.iloc[kept_indices].copy()
    df_kept["predicted_valid"] = v_pred[kept_indices]
    df_kept["valid_confidence"] = v_conf[kept_indices]
    df_kept["predicted_posture"] = p_pred[kept_indices]
    df_kept["posture_confidence"] = p_conf[kept_indices]
    df_kept["v_prob_valid1"] = v_prob1[kept_indices]
    df_kept["p_prob_posture1"] = p_prob1[kept_indices]

    sys.stdout.write(f"✓ Predictions complete\n")
    sys.stdout.flush()
    sys.stdout.write(f"  Total larvae evaluated (feature cache _valid): {n_eval}\n")
    sys.stdout.flush()
    sys.stdout.write(f"  Total larvae kept (strict rule): {n_keep}\n")
    sys.stdout.flush()
    sys.stdout.write(f"  Retention rate: {100 * n_keep / max(n_eval, 1):.1f}%\n")
    sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    significant_diff_vs_xlsx = _compare_counts_to_saved_predictions(n_keep)
    if significant_diff_vs_xlsx:
        sys.stdout.write("  Extended diagnostics (v_prob1 / p_prob1):\n")
        sys.stdout.flush()
        _print_prob_distribution_extended("v_prob1", v_prob1)
        _print_prob_distribution_extended("p_prob1", p_prob1)
        sys.stdout.write(
            "  Stage counts again: "
            f"v>={VALID_THRESHOLD} → {n_pass_v}; "
            f"p_pred==1 → {n_pass_p}; "
            f"both → {n_keep}\n"
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Group by date
    if 'date' not in df_kept.columns:
        sys.stdout.write("❌ Column 'date' not found in cache\n")
        sys.stdout.flush()
        return

    df_kept['date_fixed'] = df_kept['date'].apply(fix_date)
    date_groups = df_kept.groupby('date_fixed').size()
    date_groups = date_groups.sort_index(key=lambda x: x.map(date_sort_key))

    sys.stdout.write("Count per date:\n")
    sys.stdout.flush()
    for date_str, count in date_groups.items():
        sys.stdout.write(f"  {date_str}: {count} larvae\n")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Copy files to output directory
    sys.stdout.write("Copying filtered larvae...\n")
    sys.stdout.flush()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    viz_count = 0
    skipped_count = 0

    for idx, row in df_kept.iterrows():
        date_str = row['date_fixed']
        image_name = str(row['image_name'])
        larva_filename = str(row['larva_filename'])

        # Source paths
        src_mask = ANALYSIS_DIR / date_str / image_name / 'larvae_reports' / larva_filename

        # Destination paths
        dst_dir = OUTPUT_DIR / date_str
        dst_mask = dst_dir / larva_filename
        dst_pca_viz = dst_dir / f"{larva_filename[:-4]}_pca_length.png"
        dst_original = dst_dir / f"{larva_filename[:-4]}_original.png"

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy mask
        if src_mask.exists():
            try:
                shutil.copy2(src_mask, dst_mask)
                copied_count += 1

                # Create PCA length visualization **from the copied mask**
                try:
                    create_pca_length_visualization(str(dst_mask), str(dst_pca_viz))
                    viz_count += 1
                except Exception:
                    pass

            except Exception as e:
                sys.stdout.write(f"  ⚠️  Failed to copy {src_mask}: {e}\n")
                sys.stdout.flush()
                skipped_count += 1
        else:
            skipped_count += 1

        # Create and save original crop (raw photo preferred; overlay stripped of boxes if used)
        crop = _crop_overlay_for_larva(date_str, image_name, larva_filename)
        if crop is not None:
            try:
                cv2.imwrite(str(dst_original), crop)
            except Exception:
                pass

    sys.stdout.write(f"✓ Copy complete\n")
    sys.stdout.flush()
    sys.stdout.write(f"  Copied: {copied_count} larvae\n")
    sys.stdout.flush()
    sys.stdout.write(f"  Visualizations created: {viz_count} larvae\n")
    sys.stdout.flush()
    sys.stdout.write(f"  Skipped: {skipped_count} larvae\n")
    sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.stdout.write(f"Output directory: {OUTPUT_DIR}\n")
    sys.stdout.flush()
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
