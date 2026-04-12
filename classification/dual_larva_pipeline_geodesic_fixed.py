#!/usr/bin/env python3
"""
dual_larva_pipeline_geodesic_fixed.py
======================================
FIXED VERSION - Uses new cache and results in dual_larva_models_geodesic2/

This script uses a SEPARATE output directory and cache from the original.
It is functionally IDENTICAL to the main geodesic pipeline.

Output directory: dual_larva_models_geodesic2/
Feature cache:    dual_larva_models_geodesic2/cached_all_features.pkl

All code logic, algorithms, thresholds, features, and models are unchanged.
Only the output directory path differs from the original version.

Note: This version includes skeleton filtering fixes (< 20 pixel thresholds)
to prevent constant body_length artifacts.
"""


from pathlib import Path
import sys
import shutil
import traceback
import warnings
from typing import Optional, List, Tuple
from collections import deque

import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA as skPCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================
ROOT_DIR     = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full_binary_masks_only"
LABELS_FILE  = ANALYSIS_DIR / "larva_quality_labels.xlsx"
OUTPUT_DIR   = ROOT_DIR / "dual_larva_models_geodesic2"      # ← FIXED VERSION (separate cache)

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
PIXEL_TO_MM    = 0.232255814
CONF_THRESHOLD = 0.8
VALID_THRESHOLD = 0.8

### SPEED OPTIMIZATION BLOCK ###
FAST_MODE      = True          # True → 3-fold CV + lighter estimators; False → original 5-fold
N_FOLDS        = 3 if FAST_MODE else 5

EXCLUDED_DATES = {'18.10', '18.1'}

### SMART UNIFIED FEATURE CACHE ###
# Single DataFrame holding features for ALL larvae (labeled + unlabeled).
# Persisted to disk as cached_all_features.pkl.
# Key: (date, image_name, larva_filename)
_UNIFIED_CACHE_DF: Optional[pd.DataFrame] = None
_UNIFIED_CACHE_PATH: Optional[Path] = None

### RUNTIME BODY LENGTH CACHE ###
# In-memory cache to avoid recomputing body_length multiple times per run.
# Key: (date, image_name, larva_filename)
# Value: computed body_length (float)
# This cache exists ONLY in memory and is NOT saved to disk.
_BODY_LENGTH_RUNTIME_CACHE: dict = {}


# ============================================================
#  DIRECTORY SETUP
# ============================================================
def create_output_structure():
    global _UNIFIED_CACHE_PATH
    dirs = [
        OUTPUT_DIR / "valid_model"   / "figures",
        OUTPUT_DIR / "posture_model" / "figures",
        OUTPUT_DIR / "predictions",
        OUTPUT_DIR / "high_confidence" / "valid",
        OUTPUT_DIR / "high_confidence" / "posture",
        OUTPUT_DIR / "examples" / "valid",
        OUTPUT_DIR / "examples" / "posture",
        OUTPUT_DIR / "figures",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    ### SMART UNIFIED FEATURE CACHE ###
    _UNIFIED_CACHE_PATH = OUTPUT_DIR / "cached_all_features.pkl"
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"  FAST_MODE={FAST_MODE}  N_FOLDS={N_FOLDS}  cache={_UNIFIED_CACHE_PATH.name}")


# ============================================================
#  DATE UTILITIES
# ============================================================
def fix_date(d):
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
    try:
        day, mon = str(name).split('.')
        return (int(mon), int(day))
    except Exception:
        return (999, 999)


def is_excluded(date_str):
    return fix_date(date_str) in EXCLUDED_DATES or str(date_str) in EXCLUDED_DATES


# ============================================================
#  SKELETONIZATION  — matches skeleton_length_figure.py exactly
# ============================================================
def _skeletonize(binary_img):
    """Produce a clean 1-pixel-wide skeleton using scikit-image (with fallback)."""
    try:
        from skimage.morphology import skeletonize as _sk
        bw = (binary_img > 0).astype(bool)
        skel = _sk(bw).astype(np.uint8)
        return skel
    except ImportError:
        pass
    # Fallback: OpenCV morphological thinning
    img    = (binary_img > 0).astype(np.uint8)
    skel   = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(200):
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        skel   = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
        img    = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# ============================================================
#  GEODESIC BFS UTILITIES — from skeleton_length_figure.py
# ============================================================
def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """For every skeleton pixel, count its 8-connected skeleton neighbours."""
    ker = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, ker) * skel.astype(np.uint8)


def _bfs_distances(skel: np.ndarray, source: tuple) -> dict:
    """
    Single-source BFS from `source` over skeleton pixels.
    Returns dict: (r, c) → geodesic distance from source.
    Diagonal step cost = √2, axis-aligned = 1.0.
    """
    dist  = {source: 0.0}
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        d    = dist[(r, c)]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                nb_rc  = (nr, nc)
                if nb_rc in dist:
                    continue
                if (0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]
                        and skel[nr, nc] > 0):
                    step = 1.4142 if (dr != 0 and dc != 0) else 1.0
                    dist[nb_rc] = d + step
                    queue.append(nb_rc)
    return dist


def _bfs_path_length(skel: np.ndarray, start: tuple, goal: tuple) -> float:
    """
    BFS on skeleton from start to goal.
    Returns geodesic length (diagonal=√2, axis-aligned=1).
    """
    visited = {start: 0.0}
    queue   = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            return visited[goal]
        r, c = cur
        d = visited[cur]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                nb_rc = (nr, nc)
                if nb_rc in visited:
                    continue
                if (0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]
                        and skel[nr, nc] > 0):
                    step = 1.4142 if (dr != 0 and dc != 0) else 1.0
                    visited[nb_rc] = d + step
                    queue.append(nb_rc)
    return 0.0


def _find_topology(skel: np.ndarray):
    """
    Topology detection matching skeleton_length_figure.py exactly.
      Endpoint  = skeleton pixel with 8-connected degree == 1
      Junction  = skeleton pixel with 8-connected degree >= 3
    Junction selection: maximise sum of geodesic distances to endpoints.
    """
    nb        = _neighbour_count(skel)
    ep_arr    = np.argwhere((skel > 0) & (nb == 1))
    jn_arr    = np.argwhere((skel > 0) & (nb >= 3))

    endpoints = [tuple(p) for p in ep_arr]
    jn_candidates = [tuple(p) for p in jn_arr]

    if not jn_candidates:
        return endpoints, []

    if len(jn_candidates) == 1:
        return endpoints, [jn_candidates[0]]

    best_jn    = jn_candidates[0]
    best_score = -1.0
    for jn in jn_candidates:
        dists = _bfs_distances(skel, jn)
        score = sum(dists.get(ep, 0.0) for ep in endpoints)
        if score > best_score:
            best_score = score
            best_jn    = jn

    return endpoints, [best_jn]


def compute_geodesic_body_length(mask: np.ndarray) -> float:
    """
    Compute body length using PCA-based major axis projection.

    Algorithm:
      1. Extract all foreground pixels from the binary mask
      2. Compute PCA on the pixel coordinates
      3. Take the first principal component (major axis)
      4. Project all mask pixels onto this axis
      5. Body length = max(projection) - min(projection)

    This approach measures the extent along the principal axis of variation.

    Args:
        mask: Binary mask of the larva

    Returns:
        Body length in pixels (projection range along principal axis)
    """
    # Extract foreground pixel coordinates
    coords = np.argwhere(mask > 0)  # Returns (row, col) = (y, x)

    # Handle very small masks
    if len(coords) < 5:
        return 0.0

    # Convert to (x, y) format for PCA
    points = coords[:, [1, 0]]  # Swap to (col, row) = (x, y)

    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Apply PCA to find principal axis
    pca = skPCA(n_components=1)
    pca.fit(points)

    # Get principal component (direction vector)
    pca_vector = pca.components_[0]

    # Project all points onto the principal axis
    # Projection = dot product with principal component
    centered_points = points - centroid
    projections = np.dot(centered_points, pca_vector)

    # Find min and max projections
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # Body length is the range of projections
    length_px = max_proj - min_proj

    return float(length_px)


def compute_pca_endpoint_length(mask: np.ndarray) -> float:
    coords = np.argwhere(mask > 0)

    if len(coords) < 5:
        return 0.0

    points = coords[:, [1, 0]]  # (x, y)

    centroid = np.mean(points, axis=0)

    pca = skPCA(n_components=1)
    pca.fit(points)

    direction = pca.components_[0]

    centered = points - centroid
    projections = np.dot(centered, direction)

    i_min = np.argmin(projections)
    i_max = np.argmax(projections)

    p1 = points[i_min]
    p2 = points[i_max]

    # Euclidean distance (this already includes sqrt(2) when needed)
    length_px = np.linalg.norm(p2 - p1)

    return float(length_px)


# ============================================================
#  FEATURE EXTRACTION  — IMPROVED (A) + GEODESIC BODY LENGTH
# ============================================================
_FEAT_KEYS = [
    'area', 'aspect_ratio', 'solidity', 'perimeter',
    'mean_width', 'max_width', 'pca_variance_ratio', 'body_length'
    'ellipse_ratio', 'perimeter',
    'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
    # skeleton
    'skeleton_length', 'num_endpoints', 'num_junctions',
    'vertical_extent', 'horizontal_extent', 'verticality',
    # posture / width
    'mean_width', 'max_width', 'width_peak_ratio', 'curvature_ratio',
    # pca
    """Simplified feature extraction — compute only the stable features listed in _FEAT_KEYS.
    Any removed/unstable features are intentionally omitted to reduce noise.
    """
    'pca_endpoint_length',
    # ── IMPROVEMENT A: new discriminative ratio features ──────
    'skeleton_area_ratio',
    'perimeter_area_ratio',
    'compactness',
    # ── IMPROVEMENT D: rotation-invariant T-shape topology ────
    't_branch_ratio',
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    'junction_deviation',
    'branch_length_std',
    # ── IMPROVEMENT E: grayscale intensity statistics ─────────
    'intensity_mean',
    'intensity_std',
    'intensity_median',
    'intensity_min',
    feats = {k: 0.0 for k in _FEAT_KEYS}

    area = int(np.sum(bw))
    feats['area'] = float(area)


def _empty_features():
    ys, xs = np.where(bw > 0)


    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    if not img_path.exists():
        return None
    crop = bw[y_min:y_max + 1, x_min:x_max + 1]
    if img is None:
        return None
    # Contour-based perimeter and solidity
    cnts, _ = cv2.findContours((crop * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
        cnt = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(cnt, True)
        feats['perimeter'] = float(perim)
        hull = cv2.convexHull(cnt)
        bw = 1 - bw
        cnt_area = cv2.contourArea(cnt)
    feats = {}
    area  = int(np.sum(bw))
        feats['perimeter'] = 0.0
        feats['solidity'] = 0.0

    # Width-based features (stable proxies for posture)
        return _empty_features()

    ys, xs  = np.where(bw > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h  = y_max - y_min + 1
        feats['mean_width'] = float(np.mean(widths))
        feats['max_width'] = float(np.max(widths))

        feats['mean_width'] = 0.0
        feats['max_width'] = 0.0

    # PCA variance ratio (2nd / 1st) — robust estimate
    feats['pca_endpoint_length'] = compute_pca_endpoint_length(crop)

    cnts, _ = cv2.findContours(
        (crop * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts:
        cnt      = max(cnts, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perim    = cv2.arcLength(cnt, True)

        feats['circularity'] = float(4 * np.pi * cnt_area / (perim ** 2)) if perim > 0 else 0.0
    # Body length computed using the PCA-based or geodesic routine (keep existing geodesic function)
    try:
        # Use existing routines: prefer geodesic if available; otherwise PCA endpoint length
        skel = _skeletonize(crop)
        bl = compute_geodesic_body_length(skel) if 'compute_geodesic_body_length' in globals() else compute_pca_endpoint_length(crop)
        feats['body_length'] = float(bl)
    except Exception:
        feats['body_length'] = 0.0
            feats['ellipse_ratio'] = float(MA / ma) if ma > 0 else 0.0
    # Ensure all keys present
    for k in _FEAT_KEYS:
        if k not in feats:
        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        for i in range(7):
            v = hu[i]
            feats[f'hu{i + 1}'] = float(-np.sign(v) * np.log10(abs(v) + 1e-10))
    else:
        for k in ['circularity', 'perimeter', 'solidity', 'ellipse_ratio']:
            feats[k] = 0.0
        for i in range(7):
            feats[f'hu{i + 1}'] = 0.0

    # skeleton — using scikit-image skeletonize (same as skeleton_length_figure.py)
    skel    = _skeletonize(crop)
    skel_px = int(np.sum(skel))
    feats['skeleton_length'] = float(skel_px)

    if skel_px > 1:
        ker = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        nb  = cv2.filter2D(skel.astype(np.uint8), -1, ker)
        feats['num_endpoints'] = float(int(np.sum((skel == 1) & (nb == 1))))
        feats['num_junctions'] = float(int(np.sum((skel == 1) & (nb >= 3))))
        sy, sx = np.where(skel > 0)
        v_ext  = int(sy.max() - sy.min())
        h_ext  = int(sx.max() - sx.min())
        feats['vertical_extent']   = float(v_ext)
        feats['horizontal_extent'] = float(h_ext)
        feats['verticality']       = float(v_ext) / float(h_ext + 1)
    else:
        for k in ['num_endpoints', 'num_junctions',
                  'vertical_extent', 'horizontal_extent', 'verticality']:
            feats[k] = 0.0

    # ── IMPROVEMENT D: rotation-invariant T-shape topology ───
    t_branch_ratio    = 0.0
    endpoint_dev      = 0.0
    junction_dev      = 0.0
    branch_len_std    = 0.0

    if skel_px > 5:
        _ker  = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        _nb   = cv2.filter2D(skel.astype(np.uint8), -1, _ker)

        _ep_mask  = (skel == 1) & (_nb == 1)
        _jn_mask  = (skel == 1) & (_nb >= 3)

        _n_ep = int(_ep_mask.sum())
        _n_jn = int(_jn_mask.sum())

        endpoint_dev  = float(abs(_n_ep - 3))
        junction_dev  = float(abs(_n_jn - 1))

        if _n_ep == 3 and _n_jn == 1:
            _jn_coords = np.argwhere(_jn_mask)
            _ep_coords = np.argwhere(_ep_mask)
            _jn_rc     = tuple(_jn_coords[0])

            _branch_lens = []
            for ep in _ep_coords:
                _bl = _bfs_path_length(skel, tuple(ep), _jn_rc)
                _branch_lens.append(_bl)

            if _branch_lens:
                _branch_lens_arr  = np.array(_branch_lens)
                _longest          = float(_branch_lens_arr.max())
                _short_mean       = float(_branch_lens_arr[_branch_lens_arr < _longest].mean()) \
                                    if (_branch_lens_arr < _longest).any() else 0.0
                t_branch_ratio    = _longest / (_short_mean + 1.0)
                branch_len_std    = float(_branch_lens_arr.std())

    feats['t_branch_ratio']     = t_branch_ratio
    feats['endpoint_deviation'] = endpoint_dev
    feats['junction_deviation'] = junction_dev
    feats['branch_length_std']  = branch_len_std

    # width-based posture
    widths = []
    for row_i in range(crop.shape[0]):
        row_pixels = np.where(crop[row_i] > 0)[0]
        if len(row_pixels) >= 2:
            widths.append(int(row_pixels[-1] - row_pixels[0] + 1))
    if widths:
        mean_w = float(np.mean(widths))
        max_w  = float(np.max(widths))
        feats['mean_width']       = mean_w
        feats['max_width']        = max_w
        feats['width_peak_ratio'] = max_w / mean_w if mean_w > 0 else 0.0
        feats['curvature_ratio']  = float(np.std(widths)) / mean_w if mean_w > 0 else 0.0
    else:
        feats['mean_width'] = feats['max_width'] = feats['width_peak_ratio'] = feats['curvature_ratio'] = 0.0

    # PCA variance ratio
    coords = np.column_stack([ys - ys.mean(), xs - xs.mean()]).astype(float)
    if coords.shape[0] >= 3:
        try:
            pca = skPCA(n_components=2)
            pca.fit(coords)
            ev = pca.explained_variance_
            feats['pca_variance_ratio'] = float(ev[1] / ev[0]) if ev[0] > 0 else 0.0
        except Exception:
            feats['pca_variance_ratio'] = 0.0
    else:
        feats['pca_variance_ratio'] = 0.0

    # ── IMPROVEMENT A: compute new ratio features ─────────────
    feats['skeleton_area_ratio']  = feats['skeleton_length'] / (feats['area'] + 1)
    feats['perimeter_area_ratio'] = feats['perimeter']       / (feats['area'] + 1)
    feats['compactness']          = (feats['perimeter'] ** 2) / (feats['area'] + 1)

    # ── IMPROVEMENT E: intensity statistics (mask pixels only) ─
    mask_pixels = crop_gray[crop > 0].astype(np.float32)
    if mask_pixels.size > 0:
        feats['intensity_mean']   = float(np.mean(mask_pixels))
        feats['intensity_std']    = float(np.std(mask_pixels))
        feats['intensity_median'] = float(np.median(mask_pixels))
        feats['intensity_min']    = float(np.min(mask_pixels))
        feats['intensity_max']    = float(np.max(mask_pixels))
        feats['intensity_range']  = feats['intensity_max'] - feats['intensity_min']
        hist, _ = np.histogram(mask_pixels, bins=256, range=(0, 256))
        hist    = hist[hist > 0].astype(np.float64)
        prob    = hist / hist.sum()
        feats['intensity_entropy'] = float(-np.sum(prob * np.log2(prob)))
    else:
        for k in ['intensity_mean', 'intensity_std', 'intensity_median',
                  'intensity_min', 'intensity_max', 'intensity_range',
                  'intensity_entropy']:
            feats[k] = 0.0

    # ═══════════════════════════════════════════════════════════
    #  GEODESIC BODY LENGTH - NOT stored in cache
    #  Will be computed dynamically after loading cache
    #  (This allows changing the filtering logic without invalidating cache)
    # ═══════════════════════════════════════════════════════════
    # feats['body_length'] = compute_geodesic_body_length(skel)  # REMOVED from cache
    # NOTE: body_length will be computed in _get_cached_feats() dynamically

    return feats


def extract_morphometrics(date: str, image_name: str, larva_filename: str) -> dict:
    """Kept for API compatibility. Geodesic body_length is computed in extract_features()."""
    return {'body_length': 0.0}


def build_feature_row(date: str, image_name: str, fname: str,
                      img_path: Path) -> Optional[dict]:
    """Extract features for a single larva image. No caching here — use unified cache."""
    feats = extract_features(img_path)
    if feats is None:
        return None
    # body_length is already set by geodesic computation inside extract_features()
    return feats


# ============================================================
#  SMART UNIFIED FEATURE CACHE
# ============================================================

def _cache_key(date: str, image_name: str, larva_filename: str) -> str:
    return f"{date}_{image_name}_{larva_filename}"


def build_unified_feature_cache(labels_df: pd.DataFrame,
                                all_larvae: List[Tuple]) -> pd.DataFrame:
    """
    Build or update a single feature cache covering ALL larvae
    (labeled + unlabeled from collect_all_larvae).

    • Loads existing cache from disk if available.
    • Identifies larvae NOT yet in cache.
    • Extracts features ONLY for missing larvae.
    • Appends new rows, saves updated cache to disk.
    • Returns full cache DataFrame.

    Columns: date, image_name, larva_filename, _cache_key,
             all _FEAT_KEYS, _valid (bool).
    """
    global _UNIFIED_CACHE_DF
    import time as _time

    # ── 1. Collect the full universe of larvae we need ────────
    universe = {}  # cache_key → (date, image_name, fname, img_path)

    # From labels (may not exist on disk yet — build path)
    for _, r in labels_df.iterrows():
        date = fix_date(r['date'])
        image_name = str(r['image_name'])
        fname      = str(r['larva_filename'])
        img_path   = ANALYSIS_DIR / date / image_name / 'larvae_reports' / fname
        key = _cache_key(date, image_name, fname)
        universe[key] = (date, image_name, fname, img_path)

    # From all_larvae scan (includes unlabeled)
    for (date, img_name, fname, fpath) in all_larvae:
        key = _cache_key(date, img_name, fname)
        if key not in universe:
            universe[key] = (date, img_name, fname, fpath)

    total_needed = len(universe)

    # ── 2. Load existing disk cache ───────────────────────────
    cached_keys = set()
    if _UNIFIED_CACHE_PATH is not None and _UNIFIED_CACHE_PATH.exists():
        _UNIFIED_CACHE_DF = joblib.load(_UNIFIED_CACHE_PATH)
        if '_cache_key' in _UNIFIED_CACHE_DF.columns:
            cached_keys = set(_UNIFIED_CACHE_DF['_cache_key'].values)
        print(f"  ✓ Loaded feature cache ({len(cached_keys)} larvae)")
    else:
        _UNIFIED_CACHE_DF = pd.DataFrame()

    # ── 3. Find missing larvae ────────────────────────────────
    missing_keys = [k for k in universe if k not in cached_keys]
    n_cached = total_needed - len(missing_keys)

    if not missing_keys:
        _build_cache_index()
        print(f"  ✓ Feature cache ready ({total_needed} larvae, 0 new)")
        return _UNIFIED_CACHE_DF

    print(f"  Extracting features for {len(missing_keys)} new larvae "
          f"({n_cached} already cached) …")

    # ── 4. Extract features only for missing larvae ───────────
    t0 = _time.time()
    new_rows = []
    for idx, key in enumerate(missing_keys):
        if (idx + 1) % 200 == 0:
            elapsed = _time.time() - t0
            pct = 100.0 * (idx + 1) / len(missing_keys)
            print(f"    … {idx + 1}/{len(missing_keys)}  ({pct:.0f}%, {elapsed:.0f}s)")

        date, image_name, fname, img_path = universe[key]
        feats = extract_features(img_path)

        row = {
            '_cache_key':     key,
            'date':           date,
    """Disabled topology weighting — return constant 1.0 to remove topology influence."""
    return 1.0
            '_img_path':      str(img_path),  # Store path for dynamic body_length computation
        }
        if feats is not None:
            row.update(feats)
            row['_valid'] = True
        else:
            # Fill feature columns with 0 so DataFrame stays rectangular
            for fk in _FEAT_KEYS:
                row[fk] = 0.0
            row['_valid'] = False
        new_rows.append(row)

    elapsed = _time.time() - t0
    n_ok = sum(1 for r in new_rows if r['_valid'])
    print(f"  ✓ Extracted {n_ok}/{len(missing_keys)} new features in {elapsed:.1f}s")

    # ── 5. Append to cache and save ──────────────────────────
    new_df = pd.DataFrame(new_rows)
    if len(_UNIFIED_CACHE_DF) > 0:
        # Use uniform weight to remove topology-based influence
        sample_w = 1.0
        _UNIFIED_CACHE_DF = new_df

    # Deduplicate by cache key (keep last = freshest)
    _UNIFIED_CACHE_DF = _UNIFIED_CACHE_DF.drop_duplicates(
        subset='_cache_key', keep='last'
    ).reset_index(drop=True)

    if _UNIFIED_CACHE_PATH is not None:
        joblib.dump(_UNIFIED_CACHE_DF, _UNIFIED_CACHE_PATH)
        print(f"  ✓ Saved updated cache → {_UNIFIED_CACHE_PATH.name}")

    # Rebuild O(1) lookup index
    _build_cache_index()

    print(f"  ✓ Feature cache ready (total {len(_UNIFIED_CACHE_DF)} larvae)")
    return _UNIFIED_CACHE_DF
    # Report uniform weighting
    print(f"    Sample weights: uniform (all=1.0).")
### SMART CACHE INDEX — O(1) lookup instead of O(N) pandas scan ###
_CACHE_INDEX: Optional[dict] = None    # _cache_key → row index in _UNIFIED_CACHE_DF


def _build_cache_index():
    """Build a dict mapping _cache_key → row index for O(1) lookups."""
    global _CACHE_INDEX
    if _UNIFIED_CACHE_DF is None or len(_UNIFIED_CACHE_DF) == 0:
        _CACHE_INDEX = {}
        return
    _CACHE_INDEX = dict(zip(
        _UNIFIED_CACHE_DF['_cache_key'].values,
        _UNIFIED_CACHE_DF.index.values
    ))


def _get_cached_feats(date: str, image_name: str, fname: str) -> Optional[dict]:
    """
    Retrieve a single feature row from the unified cache. Returns None if not found or invalid.

    NOTE: body_length is NOT stored in cache. It is computed dynamically from the image
    to allow changing filtering logic without invalidating the entire cache.

        # Use uniform sample weight (disable shape/topology weighting)
        sample_w = 1.0
    """
    global _CACHE_INDEX, _BODY_LENGTH_RUNTIME_CACHE
    if _UNIFIED_CACHE_DF is None or len(_UNIFIED_CACHE_DF) == 0:
        return None
    if _CACHE_INDEX is None:
        _build_cache_index()
    key = _cache_key(date, image_name, fname)
    idx = _CACHE_INDEX.get(key)
    if idx is None:
        return None
    row = _UNIFIED_CACHE_DF.iloc[idx]
    if not row.get('_valid', False):
        return None

    # Get all cached features
    result = {k: float(row[k]) if k in row.index else 0.0 for k in _FEAT_KEYS}
    # Report uniform weighting
    print(f"    Sample weights: uniform (all=1.0).")
    runtime_key = (date, image_name, fname)

    # Check runtime cache first to avoid redundant computation
    if runtime_key in _BODY_LENGTH_RUNTIME_CACHE:
        result['body_length'] = _BODY_LENGTH_RUNTIME_CACHE[runtime_key]
        return result

    # Dynamically compute body_length from the image (first time only)
    body_length = 0.0
    img_path_str = row.get('_img_path', None)
    if img_path_str is not None:
        img_path = Path(img_path_str)
        if img_path.exists():
            try:
                # Load image and recompute skeleton
                img = cv2.imread(str(img_path))
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    bw = (bw > 0).astype(np.uint8)
                    if np.mean(bw) > 0.5:
                        bw = 1 - bw

                    # Get crop region
                    ys, xs = np.where(bw > 0)
                    if len(ys) > 0:
                        y_min, y_max = int(ys.min()), int(ys.max())
                        x_min, x_max = int(xs.min()), int(xs.max())
                        crop = bw[y_min:y_max + 1, x_min:x_max + 1]

                        # Compute body_length using PCA on the mask
                        # NOTE: No topology filtering here - all larvae get measured
                        # Topology quality is handled via sample weighting during training
                        body_length = compute_geodesic_body_length(crop)
            except Exception:
                body_length = 0.0

    # Store in runtime cache for future calls
    _BODY_LENGTH_RUNTIME_CACHE[runtime_key] = body_length
    result['body_length'] = body_length

    return result


# ============================================================
#  DATASET BUILDERS  (read from unified cache, no feature extraction)
# ============================================================

def compute_topology_weight(feats: dict) -> float:
    """
    Compute a continuous sample weight based on topology/skeleton features.

    Returns a float in [min_w, max_w]. Uses smooth transforms so small
    feature changes produce gradual weight changes. Designed to be
    conservative and robust to missing features.

    Expected keys in feats (all optional):
      - 'endpoint_deviation' (smaller is better)
      - 'junction_deviation' (smaller is better)
      - 't_branch_ratio' (higher tends to indicate clear T-shape)
      - 'branch_length_std' (smaller is better)
      - 'skeleton_area_ratio' (moderate values preferred)
      - 'skeleton_length' (absolute skeleton pixels)
      - 'area' (component area)

    The function maps these signals into a combined score [0,1], then
    linearly into the output weight range.
    """
    try:
        import numpy as _np
    except Exception:
        # fallback: use global numpy
        _np = np

    # output bounds (conservative)
    min_w, max_w = 0.5, 2.0

    # Safe extraction with defaults that represent weak/neutral evidence
    ep_dev = float(feats.get('endpoint_deviation', 5.0))
    jn_dev = float(feats.get('junction_deviation', 5.0))
    t_branch = float(feats.get('t_branch_ratio', 0.0))
    branch_std = float(feats.get('branch_length_std', 1.0))
    skel_area_ratio = float(feats.get('skeleton_area_ratio', 1.0))
    skel_len = float(feats.get('skeleton_length', 0.0))
    area = float(feats.get('area', 0.0))

    # Quick bailouts for tiny/invalid masks (keep minimal weight)
    if skel_len < 5 or area < 5:
        return float(min_w)

    # Endpoint and junction closeness: transform into [0..1], higher=better
    # Use slightly larger sigma to reduce sensitivity to small integer deviations
    sigma_ep = 3.0
    sigma_jn = 1.5
    ep_score = _np.exp(- (ep_dev / sigma_ep) ** 2) if not _np.isnan(ep_dev) else 0.0
    jn_score = _np.exp(- (jn_dev / sigma_jn) ** 2) if not _np.isnan(jn_dev) else 0.0

    topology_score = 0.6 * ep_score + 0.4 * jn_score

    # T-branch presence boost (saturating)
    t_score = _np.tanh(t_branch / 2.0) if not _np.isnan(t_branch) else 0.0

    # Penalize high branch length variance (so large std reduces trust, but smoothly)
    branch_penalty = 1.0 / (1.0 + (branch_std / (2.0 + abs(t_branch)))) if branch_std >= 0 else 1.0

    # Skeleton-area ratio: prefer moderate values; high ratio -> lower score
    skel_score = 1.0 - _np.tanh(skel_area_ratio / 8.0) if not _np.isnan(skel_area_ratio) else 0.5

    # Combine signals (topology is dominant but other cues included)
    raw = (0.55 * topology_score) + (0.30 * t_score) + (0.15 * skel_score)
    raw *= branch_penalty

    # Avoid zeroing-out: nudge raw into (0.05, 0.95) so weights are smoother
    raw = float(_np.clip(raw, 0.0, 1.0))
    raw = 0.05 + 0.95 * raw

    # Map into [min_w, max_w]
    weight = min_w + (max_w - min_w) * raw
    # Final safety clamp
    weight = float(_np.clip(weight, min_w, max_w))
    return float(weight)


def build_valid_dataset(labels_df: pd.DataFrame):
    print("  Building valid-larva dataset …")
    rows, labels, weights, metas = [], [], [], []
    skipped = 0
    total_considered = 0
    for _, r in labels_df.iterrows():
        date = fix_date(r['date'])
        if is_excluded(date):
            continue
        total_considered += 1
        image_name = str(r['image_name'])
        fname      = str(r['larva_filename'])
        feats = _get_cached_feats(date, image_name, fname)
        if feats is None:
            skipped += 1
            continue

        # Compute topology-based sample weight
        sample_w = compute_topology_weight(feats)

        rows.append(feats)
        labels.append(int(r['is_valid_larva']))
        weights.append(sample_w)
        metas.append({'date': date, 'image_name': image_name, 'larva_filename': fname})

    X    = pd.DataFrame(rows)[_FEAT_KEYS].fillna(0.0)
    y    = np.array(labels, dtype=int)
    w    = np.array(weights, dtype=float)
    meta = pd.DataFrame(metas)
    skip_pct = 100.0 * skipped / max(total_considered, 1)
    print(f"    Considered : {total_considered}  |  Loaded : {len(X)}  |  Skipped : {skipped} ({skip_pct:.1f}%)")
    if skip_pct > 5.0:
        print(f"    ⚠️  WARNING: more than 5% of rows were skipped ({skip_pct:.1f}%)")
    print(f"    Classes    : {dict(zip(*np.unique(y, return_counts=True)))}")

    # Report weighting statistics
    unique_weights = sorted(set(weights))
    print(f"    Topology weights: {unique_weights}")
    for wt in unique_weights:
        count = int((w == wt).sum())
        pct = 100.0 * count / len(w)
        print(f"      weight={wt:.1f}: {count} samples ({pct:.1f}%)")

    # OPTIONAL: Save diagnostics about weight distribution (conservative, small file)
    try:
        import matplotlib.pyplot as _plt
        out_dir = OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        # percentiles
        pctiles = np.percentile(w, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        pct_df = pd.DataFrame({'percentile': [0,1,5,25,50,75,95,99,100], 'weight': pctiles})
        pct_df.to_csv(out_dir / 'weight_percentiles_valid.csv', index=False)
        # histogram
        _plt.figure(figsize=(6,4))
        _plt.hist(w, bins=30, color='C0', edgecolor='k', alpha=0.7)
        _plt.title('Valid dataset sample weight distribution')
        _plt.xlabel('sample weight')
        _plt.ylabel('count')
        _plt.tight_layout()
        _plt.savefig(out_dir / 'weight_histogram_valid.png', dpi=150)
        _plt.close()
        print(f"    ✓ Saved weight diagnostics: weight_percentiles_valid.csv, weight_histogram_valid.png")
    except Exception:
        pass

    return X, y, w, meta


def build_posture_dataset(labels_df: pd.DataFrame):
    print("  Building posture dataset …")
    rows, labels, weights, metas = [], [], [], []
    skipped = 0
    total_considered = 0
    valid_df = labels_df[labels_df['is_valid_larva'] == 1]
    for _, r in valid_df.iterrows():
        date = fix_date(r['date'])
        if is_excluded(date):
            continue
        total_considered += 1
        image_name = str(r['image_name'])
        fname      = str(r['larva_filename'])
        feats = _get_cached_feats(date, image_name, fname)
        if feats is None:
            skipped += 1
            continue
        posture_label = 1 if int(r.get('shape_score', 0)) >= 1 else 0

        # Conservative combination of shape and topology weights
        # Keep shape boost but avoid multiplicative over-amplification.
        shape_weight = 2.0 if int(r.get('shape_score', 0)) == 2 else 1.0
        topology_weight = compute_topology_weight(feats)
        # Combine as an anchored additive blend around 1.0 with clipping.
        # Coefficients chosen conservatively so neither signal dominates alone.
        sample_w = 1.0 + 0.6 * (shape_weight - 1.0) + 0.8 * (topology_weight - 1.0)
        sample_w = float(np.clip(sample_w, 0.5, 2.0))

        rows.append(feats)
        labels.append(posture_label)
        weights.append(sample_w)
        metas.append({'date': date, 'image_name': image_name, 'larva_filename': fname})

    X    = pd.DataFrame(rows)[_FEAT_KEYS].fillna(0.0)
    y    = np.array(labels,  dtype=int)
    w    = np.array(weights, dtype=float)
    meta = pd.DataFrame(metas)
    skip_pct = 100.0 * skipped / max(total_considered, 1)
    print(f"    Considered : {total_considered}  |  Loaded : {len(X)}  |  Skipped : {skipped} ({skip_pct:.1f}%)")
    if skip_pct > 5.0:
        print(f"    ⚠️  WARNING: more than 5% of rows were skipped ({skip_pct:.1f}%)")
    print(f"    Classes    : {dict(zip(*np.unique(y, return_counts=True)))}")

    # Report weighting statistics
    unique_weights = sorted(set(weights))
    print(f"    Combined weights (conservative blend): {[f'{w:.2f}' for w in unique_weights]}")
    for wt in unique_weights:
        count = int((np.abs(w - wt) < 0.01).sum())
        pct = 100.0 * count / len(w)
        print(f"      weight={wt:.1f}: {count} samples ({pct:.1f}%)")

    # OPTIONAL: Save diagnostics about weight distribution for posture dataset
    try:
        import matplotlib.pyplot as _plt
        out_dir = OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        pctiles = np.percentile(w, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        pct_df = pd.DataFrame({'percentile': [0,1,5,25,50,75,95,99,100], 'weight': pctiles})
        pct_df.to_csv(out_dir / 'weight_percentiles_posture.csv', index=False)
        _plt.figure(figsize=(6,4))
        _plt.hist(w, bins=30, color='C1', edgecolor='k', alpha=0.7)
        _plt.title('Posture dataset sample weight distribution')
        _plt.xlabel('sample weight')
        _plt.ylabel('count')
        _plt.tight_layout()
        _plt.savefig(out_dir / 'weight_histogram_posture.png', dpi=150)
        _plt.close()
        print(f"    ✓ Saved weight diagnostics: weight_percentiles_posture.csv, weight_histogram_posture.png")
    except Exception:
        pass

    return X, y, w, meta


# ============================================================
#  UPSAMPLING
# ============================================================
def upsample(X: pd.DataFrame, y: np.ndarray):
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return X, y
    target = int(np.sqrt(counts.min() * counts.max()))
    target = max(target, counts.min())
    Xd = X.copy()
    Xd['__y__'] = y
    parts = []
    for cls in unique:
        chunk = Xd[Xd['__y__'] == cls]
        if len(chunk) < target:
            chunk = resample(chunk, replace=True,
                             n_samples=target, random_state=RANDOM_STATE)
        parts.append(chunk)
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    return out.drop(columns=['__y__']), out['__y__'].values.astype(int)


# ============================================================
#  BINARY MODEL TRAINING
# ============================================================
def train_binary_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    out_dir: Path,
    class_names: List[str],
    use_class1_f1: bool = False,
    sample_weight: Optional[np.ndarray] = None,
):
    criterion_label = "class-1 F1" if use_class1_f1 else "weighted F1"
    print(f"\n  ── {model_name} ──────────────────────────────────")
    print(f"    Selection criterion : {criterion_label}")
    print(f"    CV strategy         : StratifiedKFold(n_splits={N_FOLDS})")
    if sample_weight is not None:
        print(f"    Sample weighting    : enabled (unique weights: {sorted(set(sample_weight))})")

        # Normalize sample weights to mean 1.0 for numerical stability
        try:
            sw = np.asarray(sample_weight, dtype=float)
            # Use nanmean to be robust to NaNs; only normalize if mean is positive
            mean_sw = float(np.nanmean(sw)) if sw.size > 0 else 0.0
            if mean_sw > 0 and not np.isnan(mean_sw):
                sw = sw / mean_sw
                sample_weight = sw
                print(f"    Sample weights normalized to mean=1.0 (original mean={mean_sw:.3f})")
        except Exception:
            # If normalization fails, keep original weights unchanged
            pass

    if len(X) < 20 or len(np.unique(y)) < 2:
        print("    ⚠️  Insufficient data – skipping training.")
        return None, None, None, None

    def _score(y_true, y_pred):
        if use_class1_f1:
            per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            return float(per_class[1]) if len(per_class) > 1 else 0.0
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)

    ### SPEED OPTIMIZATION BLOCK — lighter estimators in FAST_MODE ###
    def _make_rf():
        return RandomForestClassifier(
            n_estimators=200 if FAST_MODE else 500, max_depth=None,
            min_samples_split=2, min_samples_leaf=1,
            class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    def _make_svm():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel='rbf', C=50, gamma='scale',
                probability=True, class_weight='balanced',
                random_state=RANDOM_STATE,
            )),
        ])
    def _make_gb():
        return GradientBoostingClassifier(
            n_estimators=150 if FAST_MODE else 300, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            random_state=RANDOM_STATE,
        )

    model_factories = {
        'RandomForest':      _make_rf,
        'SVM':               _make_svm,
        'GradientBoosting':  _make_gb,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}

    for name, factory in model_factories.items():
        print(f"\n    [{name}] cross-validation …")
        fold_accs, fold_scores = [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr_raw, y_va = y[tr_idx],      y[va_idx]

            ### SPEED OPTIMIZATION BLOCK — no upsampling in CV ###
            # class_weight='balanced' handles imbalance; upsample only at final retrain
            mdl = factory()
            mdl.fit(X_tr_raw, y_tr_raw)
            y_pred = mdl.predict(X_va)

            acc   = accuracy_score(y_va, y_pred)
            score = _score(y_va, y_pred)
            fold_accs.append(acc)
            fold_scores.append(score)
            print(f"      fold {fold_idx+1}/{N_FOLDS}  acc={acc:.3f}  {criterion_label}={score:.3f}")

        mean_acc   = float(np.mean(fold_accs))
        std_acc    = float(np.std(fold_accs))
        mean_score = float(np.mean(fold_scores))
        std_score  = float(np.std(fold_scores))
        print(f"      → mean acc={mean_acc:.3f}±{std_acc:.3f}  "
              f"mean {criterion_label}={mean_score:.3f}±{std_score:.3f}")

        cv_results[name] = {
            'mean_score': mean_score, 'std_score': std_score,
            'mean_acc':   mean_acc,   'std_acc':   std_acc,
            'factory':    factory,
        }

    best_key  = max(cv_results, key=lambda k: cv_results[k]['mean_score'])
    best_info = cv_results[best_key]
    print(f"\n    ✓ Best model : {best_key}")
    print(f"      mean {criterion_label} = {best_info['mean_score']:.3f} ± {best_info['std_score']:.3f}")
    print(f"      mean acc            = {best_info['mean_acc']:.3f} ± {best_info['std_acc']:.3f}")

    print(f"\n    Retraining {best_key} on full dataset ({len(X)} samples) …")
    X_full_up, y_full_up = upsample(X, y)
    best_model = best_info['factory']()

    if sample_weight is not None:
        print(f"    Applying sample_weight to {best_key} fit …")
        try:
            if best_key == 'SVM':
                best_model.fit(X, y, svc__sample_weight=sample_weight)
            else:
                best_model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            print(f"    ⚠  {best_key} does not support sample_weight — fitting without.")
            best_model.fit(X_full_up, y_full_up)
    else:
        best_model.fit(X_full_up, y_full_up)

    joblib.dump(best_model, out_dir / 'model.pkl')
    print(f"    ✓ model.pkl saved")

    y_full_pred = best_model.predict(X)
    full_report = classification_report(y, y_full_pred,
                                        target_names=class_names, zero_division=0)
    print(f"\n  Full-data classification report ({model_name}):")
    print(full_report)
    full_report_path = out_dir / 'full_data_classification_report.txt'
    with open(full_report_path, 'w') as fh:
        fh.write(f"Model     : {best_key}\n")
        fh.write(f"Criterion : {criterion_label}\n")
        fh.write(f"CV mean {criterion_label} : {best_info['mean_score']:.4f} ± {best_info['std_score']:.4f}\n")
        fh.write(f"CV mean accuracy         : {best_info['mean_acc']:.4f} ± {best_info['std_acc']:.4f}\n")
        fh.write(f"Training samples (after upsample): {len(X_full_up)}\n\n")
        fh.write("Classification report on FULL dataset (after retrain):\n")
        fh.write(full_report)
    print(f"    ✓ full_data_classification_report.txt saved")

    with open(out_dir / 'classification_report.txt', 'w') as fh:
        fh.write(f"Best model : {best_key}\n")
        fh.write(f"Selection criterion : {criterion_label}\n")
        fh.write(f"CV strategy : StratifiedKFold(n_splits={N_FOLDS})\n\n")
        for name, res in cv_results.items():
            fh.write(f"{name}:\n")
            fh.write(f"  mean {criterion_label} = {res['mean_score']:.4f} ± {res['std_score']:.4f}\n")
            fh.write(f"  mean accuracy         = {res['mean_acc']:.4f} ± {res['std_acc']:.4f}\n\n")
        fh.write("\nFull-data report (best model retrained on all data):\n")
        fh.write(full_report)

    return best_model, None, None, None


# ============================================================
#  CONFUSION MATRIX FIGURE
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names: List[str],
                          title: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✓ {save_path.name} saved")


# ============================================================
#  CLASS DISTRIBUTION PIE
# ============================================================
def plot_class_distribution(y: np.ndarray, class_names: List[str],
                             title: str, save_path: Path):
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_names[u] for u in unique]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%',
            colors=colors[:len(unique)], startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✓ {save_path.name} saved")


# ============================================================
#  COLLECT ALL LARVAE
# ============================================================
def collect_all_larvae() -> List[Tuple]:
    all_larvae = []
    for date_dir in sorted(ANALYSIS_DIR.iterdir(), key=lambda p: date_sort_key(p.name)):
        if not date_dir.is_dir() or not date_dir.name[0].isdigit():
            continue
        if is_excluded(date_dir.name):
            continue
        for img_dir in sorted(date_dir.iterdir()):
            if not img_dir.is_dir():
                continue
            lr_dir = img_dir / 'larvae_reports'
            if not lr_dir.exists():
                continue
            for f in sorted(lr_dir.iterdir()):
                if f.name.startswith('larva_') and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    all_larvae.append((date_dir.name, img_dir.name, f.name, f))
    return all_larvae


# ============================================================
#  PREDICTION STAGE (cascaded)
# ============================================================
def predict_all(all_larvae: List[Tuple], valid_model, posture_model) -> pd.DataFrame:
    total = len(all_larvae)
    print(f"\n  Running predictions on {total} larvae …")
    ### SMART UNIFIED FEATURE CACHE — read features from cache, no extraction ###
    import time as _time
    t0 = _time.time()
    cache_hits = 0
    rows = []
    for i, (date, img_name, fname, fpath) in enumerate(all_larvae):
        if (i + 1) % 2000 == 0:
            elapsed = _time.time() - t0
            print(f"    … {i + 1}/{total}  ({elapsed:.0f}s)")
        feats = _get_cached_feats(date, img_name, fname)
        if feats is not None:
            cache_hits += 1
        base  = {
            'date': date, 'image_name': img_name, 'larva_filename': fname,
            'predicted_valid': 0, 'valid_confidence': 0.0,
            'predicted_posture': np.nan, 'posture_confidence': np.nan,
            'body_length_px': 0.0, 'body_length_mm': 0.0,
        }
        if feats is None:
            rows.append(base)
            continue
        Xr = pd.DataFrame([feats])[_FEAT_KEYS].fillna(0.0)
        try:
            v_prob = valid_model.predict_proba(Xr)[0]
            v_pred = 1 if float(v_prob[1]) >= VALID_THRESHOLD else 0
            v_conf = float(v_prob[v_pred])
        except Exception:
            v_pred, v_conf = 0, 0.0
        base['predicted_valid']  = v_pred
        base['valid_confidence'] = v_conf
        bl_px = feats.get('body_length', 0.0)
        base['body_length_px'] = bl_px
        base['body_length_mm'] = bl_px * PIXEL_TO_MM
        if v_pred == 1 and posture_model is not None:
            try:
                p_prob = posture_model.predict_proba(Xr)[0]
                p_pred = int(np.argmax(p_prob))
                p_conf = float(p_prob[p_pred])
            except Exception:
                p_pred, p_conf = 0, 0.0
            base['predicted_posture']  = p_pred
            base['posture_confidence'] = p_conf
        rows.append(base)
    elapsed = _time.time() - t0
    print(f"  ✓ Predictions complete: {total} larvae in {elapsed:.1f}s  "
          f"(cache hits: {cache_hits}/{total})")
    return pd.DataFrame(rows)


# ============================================================
#  SAVE PREDICTIONS
# ============================================================
def save_predictions(df: pd.DataFrame):
    pred_dir = OUTPUT_DIR / 'predictions'
    out = pred_dir / 'predictions_all_larvae.xlsx'
    df.to_excel(out, index=False)
    print(f"  ✓ {out}")
    valid_df = df[df['predicted_valid'] == 1]
    out2 = pred_dir / 'valid_predictions.xlsx'
    valid_df.to_excel(out2, index=False)
    print(f"  ✓ {out2}  ({len(valid_df)} rows)")
    posture_df = df[df['predicted_posture'] == 1]
    out3 = pred_dir / 'posture_predictions.xlsx'
    posture_df.to_excel(out3, index=False)
    print(f"  ✓ {out3}  ({len(posture_df)} rows)")


# ============================================================
#  HIGH-CONFIDENCE COPY
# ============================================================
def copy_high_confidence(df: pd.DataFrame, label_col: str, conf_col: str,
                          subset_label: str, dest_subdir: str):
    subset = df[
        (df[label_col] == 1) &
        (df[conf_col].notna()) &
        (df[conf_col].astype(float) >= CONF_THRESHOLD)
    ]
    copied = 0
    for _, row in subset.iterrows():
        src = (ANALYSIS_DIR / str(row['date']) / str(row['image_name'])
               / 'larvae_reports' / str(row['larva_filename']))
        if not src.exists():
            continue
        dst_dir = OUTPUT_DIR / 'high_confidence' / dest_subdir / str(row['date'])
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst_dir / src.name)
            copied += 1
        except Exception:
            pass
    print(f"  ✓ Copied {copied} high-confidence {subset_label} larvae")


# ============================================================
#  EXAMPLES
# ============================================================
def export_examples(df: pd.DataFrame, label_col: str, conf_col: str,
                    dest_subdir: str):
    subset = df[(df[label_col] == 1) & df[conf_col].notna()]
    for date in sorted(subset['date'].unique(), key=date_sort_key):
        date_rows = subset[subset['date'] == date]
        if date_rows.empty:
            continue
        best = date_rows.sort_values(conf_col, ascending=False).iloc[0]
        src  = (ANALYSIS_DIR / str(best['date']) / str(best['image_name'])
                / 'larvae_reports' / str(best['larva_filename']))
        if not src.exists():
            continue
        dst_dir = OUTPUT_DIR / 'examples' / dest_subdir / str(date)
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst_dir / src.name)
        except Exception:
            pass
    print(f"  ✓ Examples exported → examples/{dest_subdir}/")


# ============================================================
#  VISUALISATIONS
# ============================================================
def _avg_length_plot(df: pd.DataFrame, label: str, save_path: Path,
                     csv_path: Path, color: str = 'steelblue'):
    sub = df[df['body_length_px'] > 0].copy()
    if sub.empty:
        return
    stats = (
        sub.groupby('date')['body_length_mm']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_mm', 'std': 'std_mm', 'count': 'n'})
    )
    stats = stats.sort_values('date', key=lambda s: [date_sort_key(d) for d in s])
    stats['std_mm'] = stats['std_mm'].fillna(0)

    # ── DIAGNOSTIC: Warn about suspicious constant-length dates ──
    suspicious_dates = stats[(stats['std_mm'] < 0.01) & (stats['n'] >= 5)]
    if len(suspicious_dates) > 0:
        print(f"\n  ⚠️  WARNING: Detected dates with std ≈ 0 (constant body length):")
        for _, row in suspicious_dates.iterrows():
            print(f"      {row['date']}: n={int(row['n'])}, mean={row['mean_mm']:.3f}mm, std={row['std_mm']:.6f}")
            print(f"      → This suggests degenerate skeletons or artifacts!")

    stats.to_csv(csv_path, index=False)
    x = range(len(stats))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(x, stats['mean_mm'], yerr=stats['std_mm'],
                fmt='o-', capsize=5, markersize=9, color=color, linewidth=2)
    for xi, row in zip(x, stats.itertuples()):
        yoff = row.std_mm + stats['mean_mm'].max() * 0.04
        ax.annotate(
            f"n={int(row.n)}\n{row.mean_mm:.2f} mm",
            xy=(xi, row.mean_mm + yoff),
            ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.7),
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(stats['date'].tolist(), rotation=45, ha='right')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Body Length (mm)')
    ax.set_title(f'Average Body Length per Date – {label}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✓ {save_path.name} saved")


def create_all_visualisations(df: pd.DataFrame,
                               valid_y_te, valid_y_pred,
                               posture_y_te, posture_y_pred):
    print("\n  Creating visualisations …")
    fig_root = OUTPUT_DIR / 'figures'
    for col, label, color in [
        ('predicted_valid',   'Valid Larvae',    '#4ecdc4'),
        ('predicted_posture', 'Correct Posture', '#45b7d1'),
    ]:
        sub = df[df[col] == 1]
        if sub.empty:
            continue
        counts = sub.groupby('date').size()
        dates  = sorted(counts.index, key=date_sort_key)
        counts = counts.reindex(dates)
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(counts)), counts.values, color=color, edgecolor='gray')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(int(val)), ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        ax.set_title(f'{label} Count per Date')
        plt.tight_layout()
        fname = f"count_per_date_{col.split('_')[1]}.png"
        plt.savefig(fig_root / fname, dpi=150)
        plt.close()
        print(f"    ✓ {fname} saved")

    valid_df   = df[df['predicted_valid'] == 1].copy()
    posture_df = df[df['predicted_posture'] == 1].copy()
    valid_df['body_length_mm']   = valid_df['body_length_px'] * PIXEL_TO_MM
    posture_df['body_length_mm'] = posture_df['body_length_px'] * PIXEL_TO_MM
    _avg_length_plot(valid_df,   'Valid Larvae',
                     fig_root / 'avg_length_per_date_valid.png',
                     fig_root / 'length_statistics_valid.csv',   color='steelblue')
    _avg_length_plot(posture_df, 'Correct Posture Larvae',
                     fig_root / 'avg_length_per_date_posture.png',
                     fig_root / 'length_statistics_posture.csv', color='darkgreen')

    if valid_y_te is not None and valid_y_pred is not None:
        plot_confusion_matrix(
            valid_y_te, valid_y_pred,
            class_names=['Not Valid', 'Valid'],
            title='Confusion Matrix – Valid Larva Model (Geodesic)',
            save_path=OUTPUT_DIR / 'valid_model' / 'figures' / 'confusion_matrix.png',
        )
    if posture_y_te is not None and posture_y_pred is not None:
        plot_confusion_matrix(
            posture_y_te, posture_y_pred,
            class_names=['Bad Posture', 'Correct Posture'],
            title='Confusion Matrix – Posture Model (Geodesic)',
            save_path=OUTPUT_DIR / 'posture_model' / 'figures' / 'confusion_matrix.png',
        )


# ============================================================
#  SUMMARY PRINTER
# ============================================================
def print_summary(df: pd.DataFrame):
    total   = len(df)
    n_valid = int((df['predicted_valid'] == 1).sum())
    n_post  = int((df['predicted_posture'] == 1).sum())
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  Total larvae processed : {total}")
    print(f"  Predicted valid        : {n_valid} ({100 * n_valid / max(total, 1):.1f} %)")
    print(f"  Predicted correct pose : {n_post} ({100 * n_post / max(total, 1):.1f} %)")
    print(f"\n  Per-date breakdown (valid):")
    for date in sorted(df['date'].unique(), key=date_sort_key):
        dv = int((df[df['date'] == date]['predicted_valid'] == 1).sum())
        dp = int((df[df['date'] == date]['predicted_posture'] == 1).sum())
        print(f"    {date:<8}  valid={dv:5d}   posture={dp:5d}")


# ============================================================
#  MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("DUAL LARVA CLASSIFICATION PIPELINE  [GEODESIC BODY LENGTH]")
    print("  Model 1 → Valid Larva  (binary: 0=not-valid, 1=valid)")
    print("  Model 2 → Posture      (binary: 0=bad-pose,  1=T-like)")
    print("  Body length: geodesic BFS on scikit-image skeleton")
    print(f"  FAST_MODE={FAST_MODE}  N_FOLDS={N_FOLDS}")
    print(f"  Output: {OUTPUT_DIR.name}/")
    print("=" * 70)

    if not LABELS_FILE.exists():
        print(f"❌ Labels file not found: {LABELS_FILE}")
        sys.exit(1)
    if not ANALYSIS_DIR.exists():
        print(f"❌ Analysis directory not found: {ANALYSIS_DIR}")
        sys.exit(1)

    create_output_structure()

    labels_df = pd.read_excel(LABELS_FILE)
    labels_df['date'] = labels_df['date'].astype(str)

    unique_label_dates  = sorted(labels_df['date'].apply(fix_date).unique(),
                                 key=date_sort_key)
    analysis_folders    = sorted(
        [d.name for d in ANALYSIS_DIR.iterdir()
         if d.is_dir() and d.name[0].isdigit()],
        key=date_sort_key,
    )
    missing_folders = [d for d in unique_label_dates if d not in analysis_folders]
    print(f"\n✓ Loaded {len(labels_df)} label rows")
    print(f"  is_valid_larva : {labels_df['is_valid_larva'].value_counts().to_dict()}")
    print(f"  shape_score    : {labels_df['shape_score'].value_counts().to_dict()}")
    print(f"\n  ── DATE DIAGNOSTICS ──────────────────────────────────")
    print(f"  Unique dates in labels        : {unique_label_dates}")
    print(f"  Date folders in analysis_full : {analysis_folders}")
    print(f"  Missing folders               : {missing_folders if missing_folders else 'none'}")
    print(f"  Total labeled rows            : {len(labels_df)}")
    print(f"  ──────────────────────────────────────────────────────")

    # ── SMART UNIFIED FEATURE CACHE ───────────────────────────
    # Collect all larvae FIRST so the cache covers everything
    all_larvae = collect_all_larvae()
    print(f"\n  Found {len(all_larvae)} larvae on disk (18.10 excluded)")
    print("\n  ── BUILDING UNIFIED FEATURE CACHE ────────────────────")
    build_unified_feature_cache(labels_df, all_larvae)
    print(f"  ──────────────────────────────────────────────────────")

    # MODEL 1 — Valid Larva
    print("\n" + "=" * 70)
    print("STEP 1 – VALID LARVA CLASSIFIER  (class-1 F1 selection)")
    print("=" * 70)
    XV, yV, wV, _ = build_valid_dataset(labels_df)
    valid_model, Xte_V, yte_V, ypred_V = train_binary_model(
        XV, yV,
        model_name='Valid Larva',
        out_dir=OUTPUT_DIR / 'valid_model',
        class_names=['Not Valid', 'Valid'],
        use_class1_f1=True,
        sample_weight=wV,
    )
    if valid_model is None:
        print("❌ Could not train valid-larva model – aborting.")
        sys.exit(1)
    plot_class_distribution(
        yV, ['Not Valid', 'Valid'],
        title='Valid Larva Label Distribution',
        save_path=OUTPUT_DIR / 'valid_model' / 'figures' / 'class_distribution.png',
    )

    # MODEL 2 — Posture
    print("\n" + "=" * 70)
    print("STEP 2 – POSTURE / T-SHAPE CLASSIFIER  (weighted F1 selection)")
    print("=" * 70)
    XP, yP, wP, _ = build_posture_dataset(labels_df)
    posture_model, Xte_P, yte_P, ypred_P = train_binary_model(
        XP, yP,
        model_name='Posture',
        out_dir=OUTPUT_DIR / 'posture_model',
        class_names=['Bad Posture', 'Correct Posture'],
        use_class1_f1=False,
        sample_weight=wP,
    )
    plot_class_distribution(
        yP, ['Bad Posture', 'Correct Posture'],
        title='Posture Label Distribution',
        save_path=OUTPUT_DIR / 'posture_model' / 'figures' / 'class_distribution.png',
    )

    # STEP 3 — Predict all larvae
    print("\n" + "=" * 70)
    print("STEP 3 – PREDICT ALL LARVAE (CASCADED)")
    print("=" * 70)
    # all_larvae already collected before training (for unified cache)
    print(f"  Using {len(all_larvae)} larvae (18.10 excluded)")
    preds_df = predict_all(all_larvae, valid_model, posture_model)

    print("\n  Saving prediction files …")
    save_predictions(preds_df)

    print("\n  Copying high-confidence larvae …")
    copy_high_confidence(preds_df, 'predicted_valid',   'valid_confidence',   'Valid',   'valid')
    copy_high_confidence(preds_df, 'predicted_posture', 'posture_confidence', 'Posture', 'posture')

    print("\n  Exporting examples …")
    export_examples(preds_df, 'predicted_valid',   'valid_confidence',   'valid')
    export_examples(preds_df, 'predicted_posture', 'posture_confidence', 'posture')

    # STEP 4 — Visualisations
    print("\n" + "=" * 70)
    print("STEP 4 – VISUALISATIONS")
    print("=" * 70)
    create_all_visualisations(preds_df, yte_V, ypred_V, yte_P, ypred_P)

    print_summary(preds_df)

    print("\n" + "=" * 70)
    print("DUAL LARVA CLASSIFICATION PIPELINE [GEODESIC] COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
    except Exception as exc:
        print(f"\n❌ Fatal error: {exc}")
        traceback.print_exc()
        sys.exit(1)

