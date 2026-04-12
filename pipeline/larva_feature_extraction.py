"""
Single source of truth for larva morphometric features used by:
  - dual_larva_pipeline_full_labels.py (training + cache + inference in predict_all)
  - filter_larvae_by_confidence.py (batch inference from images)

Do not duplicate extract_features / compute_geodesic_body_length elsewhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
from sklearn.decomposition import PCA as skPCA

# Strict feature order for model input (length 8).
FEATURE_KEYS = [
    "area",
    "aspect_ratio",
    "solidity",
    "perimeter",
    "mean_width",
    "max_width",
    "pca_variance_ratio",
    "body_length",
]

N_FEATURES = len(FEATURE_KEYS)


def compute_geodesic_body_length(mask: np.ndarray) -> float:
    """
    PCA major-axis extent on binary mask (same as legacy 'geodesic' name in training pipeline).
    `mask` must be binary {0,1} on the bbox crop, matching extract_features.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) < 5:
        return 0.0
    points = coords[:, [1, 0]]
    centroid = np.mean(points, axis=0)
    pca = skPCA(n_components=1)
    pca.fit(points)
    pca_vector = pca.components_[0]
    centered_points = points - centroid
    projections = np.dot(centered_points, pca_vector)
    length_px = float(np.max(projections) - np.min(projections))
    return length_px


# Backward-compatible alias (inference scripts must not define a second implementation).
compute_body_length = compute_geodesic_body_length


def _empty_feature_dict() -> dict[str, float]:
    return {k: 0.0 for k in FEATURE_KEYS}


def extract_features(img_path: Path) -> Optional[dict[str, float]]:
    """
    Full feature vector from one larva mask PNG path.
    Identical mask path to training: BGR read → gray → Otsu → invert if mean>0.5 → bbox crop → metrics.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (bw > 0).astype(np.uint8)
    if np.mean(bw) > 0.5:
        bw = 1 - bw

    feats = _empty_feature_dict()
    area = int(np.sum(bw))
    feats["area"] = float(area)

    if area < 1:
        return feats

    ys, xs = np.where(bw > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    feats["aspect_ratio"] = float(bbox_h) / float(bbox_w) if bbox_w > 0 else 0.0

    crop = bw[y_min : y_max + 1, x_min : x_max + 1]

    cnts, _ = cv2.findContours((crop * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        feats["perimeter"] = float(perim)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        feats["solidity"] = float(cnt_area / hull_area) if hull_area > 0 else 0.0
    else:
        feats["perimeter"] = 0.0
        feats["solidity"] = 0.0

    widths = []
    for row_i in range(crop.shape[0]):
        row_pixels = np.where(crop[row_i] > 0)[0]
        if len(row_pixels) >= 2:
            widths.append(int(row_pixels[-1] - row_pixels[0] + 1))
    if widths:
        feats["mean_width"] = float(np.mean(widths))
        feats["max_width"] = float(np.max(widths))
    else:
        feats["mean_width"] = 0.0
        feats["max_width"] = 0.0

    coords = np.column_stack([ys - ys.mean(), xs - xs.mean()]).astype(float)
    if coords.shape[0] >= 3:
        try:
            pca = skPCA(n_components=2)
            pca.fit(coords)
            ev = pca.explained_variance_
            feats["pca_variance_ratio"] = float(ev[1] / ev[0]) if ev[0] > 0 else 0.0
        except Exception:
            feats["pca_variance_ratio"] = 0.0
    else:
        feats["pca_variance_ratio"] = 0.0

    try:
        feats["body_length"] = float(compute_geodesic_body_length(crop))
    except Exception:
        feats["body_length"] = 0.0

    return feats


def _finite_float_from_cache(v: Any) -> Optional[float]:
    """Return float if v is a usable cached scalar; None if missing / NaN / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def inference_features_from_cache_row(
    row: Mapping[str, Any],
    img_path: Optional[Path] = None,
    *,
    full_extract: bool = False,
) -> Optional[dict[str, float]]:
    """
    Build the 8-D feature dict from real image data (larva PNG) and optionally the pkl row.

    - Always reads the larva crop with ``extract_features(path)``. If the file is missing or
      unreadable, returns ``None`` (caller must skip the row — no fabricated zero vectors).
    - ``full_extract=True``: all eight keys come from ``extract_features`` (pure image).
    - ``full_extract=False`` (training ``predict_all`` alignment): use cached scalars from ``row``
      when present and finite; for any missing key, use the value from ``extract_features`` on the
      same image (still real data, not zeros).
    """
    path = img_path
    if path is None:
        p = row.get("_img_path")
        path = Path(str(p)) if p and str(p).strip() else None
    if path is None or not path.exists():
        return None
    full = extract_features(path)
    if full is None:
        return None

    if full_extract:
        return {k: float(full[k]) for k in FEATURE_KEYS}

    result: dict[str, float] = {}
    for k in FEATURE_KEYS:
        if k == "body_length":
            result[k] = float(full["body_length"])
            continue
        cached = _finite_float_from_cache(row.get(k))
        result[k] = float(cached) if cached is not None else float(full[k])
    return result


def features_dict_to_vector(feats: Mapping[str, Any]) -> np.ndarray:
    """Row vector in FEATURE_KEYS order, float64."""
    return np.array([float(feats.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float64)


def assert_model_input_matrix(X: np.ndarray) -> None:
    if X.dtype != np.float64:
        raise TypeError(f"X must be float64, got {X.dtype}")
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"X must have shape (n, {N_FEATURES}), got {X.shape}")
    if not np.all(np.isfinite(X)):
        bad = int(np.sum(~np.isfinite(X)))
        raise ValueError(f"X contains {bad} non-finite values (NaN/inf)")


def cache_row_matches_extract(
    row: Mapping[str, Any],
    extracted: Mapping[str, float],
    *,
    key: str,
    rtol: float = 1e-3,
    atol: float = 1e-2,
    strict: bool = False,
) -> bool:
    """Return True if cached scalar matches freshly extracted value for one feature."""
    if key not in row:
        return True
    try:
        c = float(row[key])
        e = float(extracted[key])
    except (TypeError, ValueError, KeyError):
        return True
    if strict:
        s_rtol, s_atol = 1e-5, 1e-4
        if key == "area":
            return abs(c - e) <= 1.0 + 1e-6
        return abs(c - e) <= s_atol + s_rtol * max(abs(c), abs(e), 1.0)
    # Loose audit: legacy caches / OpenCV Otsu can differ slightly; inference always uses `extracted`.
    if key == "area":
        return abs(c - e) <= max(10.0, 0.03 * max(abs(c), abs(e), 1.0))
    return abs(c - e) <= atol + rtol * max(abs(c), abs(e), 1.0)
