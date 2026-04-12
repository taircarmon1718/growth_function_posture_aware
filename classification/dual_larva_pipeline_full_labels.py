#!/usr/bin/env python3
"""
dual_larva_pipeline_full_labels.py
===================================
FULL-LABELS VERSION - Trains on ALL labeled larvae (~1200), not just ~120.

This script is identical to dual_larva_pipeline_geodesic_fixed.py EXCEPT:

KEY DIFFERENCE:
    The dataset builders (build_valid_dataset, build_posture_dataset) now
    extract features directly from images when cache matching fails,
    ensuring maximum coverage of labeled samples.

This allows training on ~1000-1200 labeled larvae instead of the ~120
that matched via the unstable larva_filename identifier.

Output directory: dual_larva_models_geodesic2/
Feature cache:    dual_larva_models_geodesic2/cached_all_features.pkl

All algorithmic changes from the fixed version are preserved.
Model logic, features, thresholds, and training procedure are unchanged.
"""

from pathlib import Path
import json
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

from larva_feature_extraction import FEATURE_KEYS as _FEAT_KEYS, extract_features

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

warnings.filterwarnings('ignore')

ROOT_DIR     = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full_binary_masks_only"
LABELS_FILE  = ANALYSIS_DIR / "larva_quality_labels.xlsx"
OUTPUT_DIR   = ROOT_DIR / "dual_larva_models_geodesic2"

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
PIXEL_TO_MM    = 0.232255814
CONF_THRESHOLD = 0.85
VALID_THRESHOLD = 0.85

FAST_MODE      = True
N_FOLDS        = 3 if FAST_MODE else 5

EXCLUDED_DATES = {'18.10', '18.1'}

_UNIFIED_CACHE_DF: Optional[pd.DataFrame] = None
_UNIFIED_CACHE_PATH: Optional[Path] = None

_BODY_LENGTH_RUNTIME_CACHE: dict = {}

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
    _UNIFIED_CACHE_PATH = OUTPUT_DIR / "cached_all_features.pkl"
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"  FAST_MODE={FAST_MODE}  N_FOLDS={N_FOLDS}  cache={_UNIFIED_CACHE_PATH.name}")

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

def _skeletonize(binary_img):
    try:
        from skimage.morphology import skeletonize as _sk
        bw = (binary_img > 0).astype(bool)
        skel = _sk(bw).astype(np.uint8)
        return skel
    except ImportError:
        pass
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

def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    ker = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, ker) * skel.astype(np.uint8)

def _bfs_distances(skel: np.ndarray, source: tuple) -> dict:
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

def extract_morphometrics(date: str, image_name: str, larva_filename: str) -> dict:
    return {'body_length': 0.0}

def build_feature_row(date: str, image_name: str, fname: str,
                      img_path: Path) -> Optional[dict]:
    feats = extract_features(img_path)
    if feats is None:
        return None
    return feats

def _cache_key(date: str, image_name: str, larva_filename: str) -> str:
    return f"{date}_{image_name}_{larva_filename}"

def build_unified_feature_cache(labels_df: pd.DataFrame,
                                all_larvae: List[Tuple]) -> pd.DataFrame:
    global _UNIFIED_CACHE_DF
    import time as _time

    universe = {}

    for _, r in labels_df.iterrows():
        date = fix_date(r['date'])
        image_name = str(r['image_name'])
        fname      = str(r['larva_filename'])
        img_path   = ANALYSIS_DIR / date / image_name / 'larvae_reports' / fname
        key = _cache_key(date, image_name, fname)
        universe[key] = (date, image_name, fname, img_path)

    for (date, img_name, fname, fpath) in all_larvae:
        key = _cache_key(date, img_name, fname)
        if key not in universe:
            universe[key] = (date, img_name, fname, fpath)

    total_needed = len(universe)

    cached_keys = set()
    if _UNIFIED_CACHE_PATH is not None and _UNIFIED_CACHE_PATH.exists():
        _UNIFIED_CACHE_DF = joblib.load(_UNIFIED_CACHE_PATH)
        if '_cache_key' in _UNIFIED_CACHE_DF.columns:
            cached_keys = set(_UNIFIED_CACHE_DF['_cache_key'].values)
        print(f"  ✓ Loaded feature cache ({len(cached_keys)} larvae)")
    else:
        _UNIFIED_CACHE_DF = pd.DataFrame()

    missing_keys = [k for k in universe if k not in cached_keys]
    n_cached = total_needed - len(missing_keys)

    if not missing_keys:
        _build_cache_index()
        print(f"  ✓ Feature cache ready ({total_needed} larvae, 0 new)")
        return _UNIFIED_CACHE_DF

    print(f"  Extracting features for {len(missing_keys)} new larvae "
          f"({n_cached} already cached) …")

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
            'image_name':     image_name,
            'larva_filename': fname,
            '_img_path':      str(img_path),
        }
        if feats is not None:
            row.update(feats)
            row['_valid'] = True
        else:
            for fk in _FEAT_KEYS:
                row[fk] = 0.0
            row['_valid'] = False

        new_rows.append(row)

    elapsed = _time.time() - t0
    n_ok = sum(1 for r in new_rows if r['_valid'])
    print(f"  ✓ Extracted {n_ok}/{len(missing_keys)} new features in {elapsed:.1f}s")

    new_df = pd.DataFrame(new_rows)
    if len(_UNIFIED_CACHE_DF) > 0:
        _UNIFIED_CACHE_DF = pd.concat([_UNIFIED_CACHE_DF, new_df], ignore_index=True)
    else:
        _UNIFIED_CACHE_DF = new_df

    _UNIFIED_CACHE_DF = _UNIFIED_CACHE_DF.drop_duplicates(
        subset='_cache_key', keep='last'
    ).reset_index(drop=True)

    if _UNIFIED_CACHE_PATH is not None:
        joblib.dump(_UNIFIED_CACHE_DF, _UNIFIED_CACHE_PATH)
        print(f"  ✓ Saved updated cache → {_UNIFIED_CACHE_PATH.name}")

    _build_cache_index()

    print(f"  ✓ Feature cache ready (total {len(_UNIFIED_CACHE_DF)} larvae)")
    return _UNIFIED_CACHE_DF

_CACHE_INDEX: Optional[dict] = None

def _build_cache_index():
    global _CACHE_INDEX
    if _UNIFIED_CACHE_DF is None or len(_UNIFIED_CACHE_DF) == 0:
        _CACHE_INDEX = {}
        return
    _CACHE_INDEX = dict(zip(
        _UNIFIED_CACHE_DF['_cache_key'].values,
        _UNIFIED_CACHE_DF.index.values
    ))

def _get_cached_feats(date: str, image_name: str, fname: str) -> Optional[dict]:
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

    # Build base feature dict from full key list (original + CNN)
    result = {k: float(row[k]) if k in row.index else 0.0 for k in _FEAT_KEYS}

    # Runtime body_length recomputation (unchanged)
    runtime_key = (date, image_name, fname)
    if runtime_key in _BODY_LENGTH_RUNTIME_CACHE:
        result['body_length'] = _BODY_LENGTH_RUNTIME_CACHE[runtime_key]
        return result

    body_length = 0.0
    img_path_str = row.get('_img_path', None)
    if img_path_str is not None:
        img_path = Path(img_path_str)
        if img_path.exists():
            try:
                full = extract_features(img_path)
                if full is not None:
                    body_length = float(full["body_length"])
            except Exception:
                body_length = 0.0

    _BODY_LENGTH_RUNTIME_CACHE[runtime_key] = body_length
    result['body_length'] = body_length

    return result

def compute_topology_weight(feats: dict) -> float:
    return 1.0

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

        # Try cache first, then extract from image if not found
        feats = _get_cached_feats(date, image_name, fname)
        if feats is None:
            img_path = ANALYSIS_DIR / date / image_name / 'larvae_reports' / fname
            feats = extract_features(img_path)

        if feats is None:
            skipped += 1
            continue

        # Use constant sample weight to avoid topology-based instability
        sample_w = 1.0

        rows.append(feats)
        labels.append(int(r['is_valid_larva']))
        weights.append(sample_w)
        metas.append({'date': date, 'image_name': image_name, 'larva_filename': fname})

    # Build feature DataFrame and label array
    X_all = pd.DataFrame(rows)
    missing_cols = [c for c in _FEAT_KEYS if c not in X_all.columns]
    for c in missing_cols:
        X_all[c] = 0.0
    X = X_all[_FEAT_KEYS].fillna(0.0)

    y = np.array(labels, dtype=int)

    # Make all sample weights constant (no weighting)
    w = np.ones(len(rows), dtype=float) if len(rows) > 0 else np.array([], dtype=float)

    meta = pd.DataFrame(metas)
    skip_pct = 100.0 * skipped / max(total_considered, 1)
    print(f"    Considered : {total_considered}  |  Loaded : {len(X)}  |  Skipped : {skipped} ({skip_pct:.1f}%)")
    if skip_pct > 5.0:
        print(f"    ⚠️  WARNING: more than 5% of rows were skipped ({skip_pct:.1f}%)")
    print(f"    Classes    : {dict(zip(*np.unique(y, return_counts=True)))}")

    unique_weights = sorted(set(w.tolist()))
    print(f"    Discretized topology weights: {unique_weights}")
    for wt in unique_weights:
        count = int((w == wt).sum())
        pct = 100.0 * count / max(len(w), 1)
        print(f"      weight={wt:.1f}: {count} samples ({pct:.1f}%)")

    # Print feature dimension diagnostics once
    if len(X) > 0:
        print(f"    Feature dimension (valid): base={len(_FEAT_KEYS)}, +cnn={len(_FEAT_KEYS)}")

    try:
        import matplotlib.pyplot as _plt

        out_dir = OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(w) > 0:
            pctiles = np.percentile(w, [0, 25, 50, 75, 100])
            pct_df = pd.DataFrame({'percentile': [0,25,50,75,100], 'weight': pctiles})
            pct_df.to_csv(out_dir / 'weight_percentiles_valid_discrete.csv', index=False)
            _plt.figure(figsize=(6,4))
            _plt.hist(w, bins=[0.5,0.85,1.15,1.45], color='C0', edgecolor='k', alpha=0.7)
            _plt.title('Valid dataset discretized sample weight distribution')
            _plt.xlabel('discrete sample weight')
            _plt.ylabel('count')
            _plt.tight_layout()
            _plt.savefig(out_dir / 'weight_histogram_valid_discrete.png', dpi=150)
            _plt.close()
            print(f"    ✓ Saved weight diagnostics: weight_percentiles_valid_discrete.csv, weight_histogram_valid_discrete.png")
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

        # Try cache first, then extract from image if not found
        feats = _get_cached_feats(date, image_name, fname)
        if feats is None:
            img_path = ANALYSIS_DIR / date / image_name / 'larvae_reports' / fname
            feats = extract_features(img_path)

        if feats is None:
            skipped += 1
            continue

        posture_label = 1 if int(r.get('shape_score', 0)) >= 1 else 0

        # Use constant weight to avoid noisy topology/shape weighting
        sample_w_cont = 1.0

        rows.append(feats)
        labels.append(posture_label)
        weights.append(sample_w_cont)
        metas.append({'date': date, 'image_name': image_name, 'larva_filename': fname})

    X_all = pd.DataFrame(rows)
    missing_cols = [c for c in _FEAT_KEYS if c not in X_all.columns]
    for c in missing_cols:
        X_all[c] = 0.0
    X = X_all[_FEAT_KEYS].fillna(0.0)

    y = np.array(labels,  dtype=int)

    # Make all sample weights constant (no weighting)
    w = np.ones(len(rows), dtype=float) if len(rows) > 0 else np.array([], dtype=float)

    meta = pd.DataFrame(metas)
    skip_pct = 100.0 * skipped / max(total_considered, 1)
    print(f"    Considered : {total_considered}  |  Loaded : {len(X)}  |  Skipped : {skipped} ({skip_pct:.1f}%)")
    if skip_pct > 5.0:
        print(f"    ⚠️  WARNING: more than 5% of rows were skipped ({skip_pct:.1f}%)")
    print(f"    Classes    : {dict(zip(*np.unique(y, return_counts=True)))}")

    unique_weights = sorted(set(w.tolist()))
    print(f"    Discretized combined weights: {[f'{val:.2f}' for val in unique_weights]}")
    for wt in unique_weights:
        count = int((np.abs(w - wt) < 1e-6).sum())
        pct = 100.0 * count / max(len(w), 1)
        print(f"      weight={wt:.1f}: {count} samples ({pct:.1f}%)")

    # Print feature dimension diagnostics once
    if len(X) > 0:
        print(f"    Feature dimension (posture): base={len(_FEAT_KEYS)}, +cnn={len(_FEAT_KEYS)}")

    try:
        import matplotlib.pyplot as _plt
        out_dir = OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(w) > 0:
            pctiles = np.percentile(w, [0, 25, 50, 75, 100])
            pct_df = pd.DataFrame({'percentile': [0,25,50,75,100], 'weight': pctiles})
            pct_df.to_csv(out_dir / 'weight_percentiles_posture_discrete.csv', index=False)
            _plt.figure(figsize=(6,4))
            _plt.hist(w, bins=[0.5,0.85,1.15,1.45], color='C1', edgecolor='k', alpha=0.7)
            _plt.title('Posture dataset discretized sample weight distribution')
            _plt.xlabel('discrete sample weight')
            _plt.ylabel('count')
            _plt.tight_layout()
            _plt.savefig(out_dir / 'weight_histogram_posture_discrete.png', dpi=150)
            _plt.close()
            print(f"    ✓ Saved weight diagnostics: weight_percentiles_posture_discrete.csv, weight_histogram_posture_discrete.png")
    except Exception:
        pass

    return X, y, w, meta


def _save_training_feature_stats(X_parts: List[pd.DataFrame]) -> None:
    """Persist mean/std/min/max per feature for inference-time distribution checks."""
    if not X_parts:
        return
    Xu = pd.concat(X_parts, ignore_index=True)
    stats = {}
    for col in _FEAT_KEYS:
        s = pd.to_numeric(Xu[col], errors="coerce").fillna(0.0).astype(float)
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    out = OUTPUT_DIR / "training_feature_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved training feature stats → {out.name}")


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

        try:
            sw = np.asarray(sample_weight, dtype=float)
            mean_sw = float(np.nanmean(sw)) if sw.size > 0 else 0.0
            if mean_sw > 0 and not np.isnan(mean_sw):
                sw = sw / mean_sw
                sample_weight = sw
                print(f"    Sample weights normalized to mean=1.0 (original mean={mean_sw:.3f})")
        except Exception:
            pass

    if len(X) < 20 or len(np.unique(y)) < 2:
        print("    ⚠️  Insufficient data – skipping training.")
        return None, None, None, None

    def _score(y_true, y_pred):
        if use_class1_f1:
            per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            return float(per_class[1]) if len(per_class) > 1 else 0.0
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)

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
    def _make_xgb():
        return XGBClassifier(
            n_estimators=200 if FAST_MODE else 400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ) if _HAS_XGB else None

    model_factories = {
        'RandomForest':      _make_rf,
        'SVM':               _make_svm,
        'GradientBoosting':  _make_gb,
    }
    if _HAS_XGB:
        model_factories['XGBoost'] = _make_xgb

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}

    for name, factory in model_factories.items():
        print(f"\n    [{name}] cross-validation …")
        fold_accs, fold_scores = [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr_raw, y_va = y[tr_idx],      y[va_idx]

            mdl = factory()
            if mdl is None:
                print("      XGBoost requested but xgboost is not installed – skipping.")
                fold_accs.append(0.0)
                fold_scores.append(0.0)
                continue

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

def predict_all(all_larvae: List[Tuple], valid_model, posture_model) -> pd.DataFrame:
    total = len(all_larvae)
    print(f"\n  Running predictions on {total} larvae …")
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
        # Build prediction feature vector with the same keys/order as training
        feat_row = {k: float(feats.get(k, 0.0)) for k in _FEAT_KEYS}
        Xr = pd.DataFrame([feat_row])[_FEAT_KEYS].fillna(0.0)
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

def _avg_length_plot(df: pd.DataFrame, label: str, save_path: Path,
                     csv_path: Path, color: str = 'steelblue'):
    sub = df[df['body_length_px'] > 0].copy()
    if sub.empty:
        return

    # Compute per-larva weights
    def _row_weight(row):
        v = float(row.get('valid_confidence', 0.0))
        p = row.get('posture_confidence', np.nan)
        if pd.notna(p):
            return 0.7 * v + 0.3 * float(p)
        return v

    sub['__weight__'] = sub.apply(_row_weight, axis=1)

    def _weighted_stats(group: pd.DataFrame) -> pd.Series:
        lengths = group['body_length_mm'].astype(float).values
        weights = group['__weight__'].astype(float).values
        n = len(lengths)
        if n == 0:
            return pd.Series({'mean_mm': 0.0, 'std_mm': 0.0, 'n': 0})

        sum_w = float(weights.sum())
        if sum_w <= 0.0:
            # Fallback to simple mean/std if weights are degenerate
            mean_raw = float(lengths.mean())
            std_raw = float(lengths.std(ddof=0)) if n > 1 else 0.0
        else:
            mean_raw = float((weights * lengths).sum() / sum_w)
            if n > 1:
                var_w = float(((weights * (lengths - mean_raw) ** 2).sum()) / sum_w)
                std_raw = float(np.sqrt(max(var_w, 0.0)))
            else:
                std_raw = 0.0

        # No sample-size regularization; use raw mean
        final_mean = mean_raw

        return pd.Series({'mean_mm': final_mean, 'std_mm': std_raw, 'n': n})

    stats = (
        sub.groupby('date', as_index=False)
           .apply(_weighted_stats)
           .reset_index(drop=True)
    )

    stats = stats.sort_values('date', key=lambda s: [date_sort_key(d) for d in s])
    stats['std_mm'] = stats['std_mm'].fillna(0)

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

def main():
    print("\n" + "=" * 70)
    print("DUAL LARVA CLASSIFICATION PIPELINE  [GEODESIC BODY LENGTH]")
    print("  FULL-LABELS VERSION — Trains on ~1000-1200 labeled larvae")
    print("  Model 1 → Valid Larva  (binary: 0=not-valid, 1=valid)")
    print("  Model 2 → Posture      (binary: 0=bad-pose,  1=T-like)")
    print("  Body length: PCA-based major axis projection")
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

    all_larvae = collect_all_larvae()
    print(f"\n  Found {len(all_larvae)} larvae on disk (18.10 excluded)")
    print("\n  ── BUILDING UNIFIED FEATURE CACHE ────────────────────")
    build_unified_feature_cache(labels_df, all_larvae)
    print(f"  ──────────────────────────────────────────────────────")

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
    _save_training_feature_stats([XV, XP])
    plot_class_distribution(
        yP, ['Bad Posture', 'Correct Posture'],
        title='Posture Label Distribution',
        save_path=OUTPUT_DIR / 'posture_model' / 'figures' / 'class_distribution.png',
    )

    print("\n" + "=" * 70)
    print("STEP 3 – PREDICT ALL LARVAE (CASCADED)")
    print("=" * 70)
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
