#!/usr/bin/env python3
"""
Dual Larva Classification Pipeline
====================================
Two sequential binary classifiers:

  Model 1 – Valid Larva Classifier
    0 = Not a valid larva
    1 = Valid larva          (is_valid_larva column)

  Model 2 – Posture / T-Shape Classifier
    0 = Not correct posture
    1 = Correct T-like posture
        (is_valid_larva == 1 AND shape_score >= 1)

Feature groups:
  · Geometric  (area, circularity, solidity, Hu moments, …)
  · Skeleton   (skeleton_length, endpoints, junctions, extents, verticality)
  · Posture    (mean_width, max_width, width_peak_ratio, curvature_ratio)
  · PCA        (pca_variance_ratio λ2/λ1)
  · Morpho     (body_length from morphometrics.csv)

Excluded dates: 18.10 / 18.1
"""

from pathlib import Path
import sys
import shutil
import traceback
import warnings
from typing import Optional, List, Tuple

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================
ROOT_DIR   = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full"
LABELS_FILE  = ANALYSIS_DIR / "larva_quality_labels.xlsx"
OUTPUT_DIR   = ROOT_DIR / "dual_larva_models"

RANDOM_STATE = 42
TEST_SIZE    = 0.20
PIXEL_TO_MM  = 0.232255814
CONF_THRESHOLD = 0.70

EXCLUDED_DATES = {'18.10', '18.1'}


# ============================================================
#  DIRECTORY SETUP
# ============================================================
def create_output_structure():
    dirs = [
        OUTPUT_DIR / "valid_model"  / "figures",
        OUTPUT_DIR / "posture_model"/ "figures",
        OUTPUT_DIR / "predictions",
        OUTPUT_DIR / "high_confidence" / "valid",
        OUTPUT_DIR / "high_confidence" / "posture",
        OUTPUT_DIR / "examples"  / "valid",
        OUTPUT_DIR / "examples"  / "posture",
        OUTPUT_DIR / "figures",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")


# ============================================================
#  DATE UTILITIES
# ============================================================
def fix_date(d):
    """Normalise label date (19.1 → 19.10)."""
    s = str(d)
    parts = s.split('.')
    if len(parts) == 2 and parts[1] == '1':
        return f"{parts[0]}.10"
    return s


def date_sort_key(name):
    """(month, day) so 31.10 < 3.11."""
    try:
        day, mon = str(name).split('.')
        return (int(mon), int(day))
    except Exception:
        return (999, 999)


def is_excluded(date_str):
    return fix_date(date_str) in EXCLUDED_DATES or str(date_str) in EXCLUDED_DATES


# ============================================================
#  SKELETONIZATION (morphological, fast)
# ============================================================
def _skeletonize(binary_img):
    img = binary_img.copy().astype(np.uint8)
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(100):
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# ============================================================
#  FEATURE EXTRACTION
# ============================================================
_FEAT_KEYS = [
    # geometric
    'area', 'area_ratio', 'aspect_ratio', 'circularity', 'solidity',
    'ellipse_ratio', 'perimeter',
    'hu1','hu2','hu3','hu4','hu5','hu6','hu7',
    # skeleton
    'skeleton_length', 'num_endpoints', 'num_junctions',
    'vertical_extent', 'horizontal_extent', 'verticality',
    # posture / width
    'mean_width', 'max_width', 'width_peak_ratio', 'curvature_ratio',
    # pca
    'pca_variance_ratio',
    # morpho
    'body_length',
    # NEW discriminative ratios
    'skeleton_area_ratio', 'perimeter_area_ratio', 'compactness',
]


def _empty_features():
    return {k: 0.0 for k in _FEAT_KEYS}


def extract_features(img_path: Path) -> Optional[dict]:
    """
    Extract all feature groups from a larva report image.
    Returns None on hard failure, _empty_features() on soft failure.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # ── Binarise ──────────────────────────────────────────────
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (bw > 0).astype(np.uint8)
    if np.mean(bw) > 0.5:          # invert if background is bright
        bw = 1 - bw

    feats = {}
    area = int(np.sum(bw))
    feats['area']       = float(area)
    feats['area_ratio'] = float(area) / float(h_img * w_img) if h_img * w_img > 0 else 0.0

    if area < 50:
        return _empty_features()

    # ── Bounding box ──────────────────────────────────────────
    ys, xs = np.where(bw > 0)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    feats['aspect_ratio'] = float(bbox_h) / float(bbox_w) if bbox_w > 0 else 0.0

    crop      = bw[y_min:y_max+1, x_min:x_max+1]
    crop_gray = gray[y_min:y_max+1, x_min:x_max+1]

    # ── Contour / geometric ───────────────────────────────────
    cnts, _ = cv2.findContours(
        (crop * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts:
        cnt      = max(cnts, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perim    = cv2.arcLength(cnt, True)

        feats['circularity'] = float(4 * np.pi * cnt_area / (perim ** 2)) if perim > 0 else 0.0
        feats['perimeter']   = float(perim)

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        feats['solidity'] = float(cnt_area / hull_area) if hull_area > 0 else 0.0

        if len(cnt) >= 5:
            (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
            feats['ellipse_ratio'] = float(MA / ma) if ma > 0 else 0.0
        else:
            feats['ellipse_ratio'] = feats['aspect_ratio']

        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        for i in range(7):
            v = hu[i]
            feats[f'hu{i+1}'] = float(-np.sign(v) * np.log10(abs(v) + 1e-10))
    else:
        for k in ['circularity','perimeter','solidity','ellipse_ratio']:
            feats[k] = 0.0
        for i in range(7):
            feats[f'hu{i+1}'] = 0.0

    # ── Skeleton features ─────────────────────────────────────
    skel = _skeletonize(crop)
    skel_px = int(np.sum(skel))
    feats['skeleton_length'] = float(skel_px)

    if skel_px > 5:
        ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        nb  = cv2.filter2D(skel.astype(np.uint8), -1, ker)
        feats['num_endpoints'] = float(int(np.sum((skel == 1) & (nb == 1))))
        feats['num_junctions'] = float(int(np.sum((skel == 1) & (nb >= 3))))
        sy, sx = np.where(skel > 0)
        v_ext = int(sy.max() - sy.min())
        h_ext = int(sx.max() - sx.min())
        feats['vertical_extent']   = float(v_ext)
        feats['horizontal_extent'] = float(h_ext)
        feats['verticality']       = float(v_ext) / float(h_ext + 1)
    else:
        for k in ['num_endpoints','num_junctions',
                  'vertical_extent','horizontal_extent','verticality']:
            feats[k] = 0.0

    # ── Width-based posture features ──────────────────────────
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
        # curvature proxy: std(width) / mean_width
        feats['curvature_ratio']  = float(np.std(widths)) / mean_w if mean_w > 0 else 0.0
    else:
        feats['mean_width'] = feats['max_width'] = feats['width_peak_ratio'] = feats['curvature_ratio'] = 0.0

    # ── PCA variance ratio ────────────────────────────────────
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

    # ── NEW discriminative ratio features ────────────────────
    feats['skeleton_area_ratio']  = feats['skeleton_length'] / (feats['area'] + 1)
    feats['perimeter_area_ratio'] = feats['perimeter']       / (feats['area'] + 1)
    feats['compactness']          = (feats['perimeter'] ** 2) / (feats['area'] + 1)

    # body_length placeholder (filled by extract_morphometrics)
    feats['body_length'] = 0.0
    return feats


def extract_morphometrics(date: str, image_name: str, larva_filename: str) -> dict:
    """Load body_length from morphometrics.csv for this larva."""
    result = {'body_length': 0.0}
    csv_path = ANALYSIS_DIR / str(date) / str(image_name) / 'morphometrics.csv'
    if not csv_path.exists():
        return result
    try:
        df  = pd.read_csv(csv_path)
        lid = int(larva_filename.replace('larva_','').replace('.png',''))
        row = df[df['larva_id'] == lid]
        if not row.empty:
            result['body_length'] = float(row['body_length'].values[0])
    except Exception:
        pass
    return result


def build_feature_row(date: str, image_name: str, fname: str, img_path: Path) -> Optional[dict]:
    """Return full combined feature dict or None on failure."""
    feats = extract_features(img_path)
    if feats is None:
        return None
    morph = extract_morphometrics(date, image_name, fname)
    feats['body_length'] = morph['body_length']
    return feats


# ============================================================
#  DATASET BUILDERS
# ============================================================
def build_valid_dataset(labels_df: pd.DataFrame):
    """
    Build binary dataset for Model 1: is_valid_larva.
    Includes ALL labeled rows (both valid and invalid).
    Excludes date 18.10.
    """
    print("  Building valid-larva dataset …")
    rows, labels, metas = [], [], []

    for _, r in labels_df.iterrows():
        date = fix_date(r['date'])
        if is_excluded(date):
            continue

        image_name = str(r['image_name'])
        fname      = str(r['larva_filename'])
        img_path   = ANALYSIS_DIR / date / image_name / 'larvae_reports' / fname

        feats = build_feature_row(date, image_name, fname, img_path)
        if feats is None:
            continue

        rows.append(feats)
        labels.append(int(r['is_valid_larva']))
        metas.append({'date': date, 'image_name': image_name, 'larva_filename': fname})

    X   = pd.DataFrame(rows)[_FEAT_KEYS].fillna(0.0)
    y   = np.array(labels, dtype=int)
    meta = pd.DataFrame(metas)
    print(f"    {len(X)} samples  |  classes {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, meta


def build_posture_dataset(labels_df: pd.DataFrame):
    """
    Build binary dataset for Model 2: T-like posture.
    Only uses labeled rows where is_valid_larva == 1.
    posture_label = 1 if shape_score >= 1, else 0.
    Excludes date 18.10.
    """
    print("  Building posture dataset …")
    rows, labels, metas = [], [], []

    valid_df = labels_df[labels_df['is_valid_larva'] == 1]

    for _, r in valid_df.iterrows():
        date = fix_date(r['date'])
        if is_excluded(date):
            continue

        image_name = str(r['image_name'])
        fname      = str(r['larva_filename'])
        img_path   = ANALYSIS_DIR / date / image_name / 'larvae_reports' / fname

        feats = build_feature_row(date, image_name, fname, img_path)
        if feats is None:
            continue

        posture_label = 1 if int(r.get('shape_score', 0)) >= 1 else 0

        rows.append(feats)
        labels.append(posture_label)
        metas.append({'date': date, 'image_name': image_name, 'larva_filename': fname})

    X    = pd.DataFrame(rows)[_FEAT_KEYS].fillna(0.0)
    y    = np.array(labels, dtype=int)
    meta = pd.DataFrame(metas)
    print(f"    {len(X)} samples  |  classes {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, meta


# ============================================================
#  UPSAMPLING
# ============================================================
def upsample(X: pd.DataFrame, y: np.ndarray):
    """Moderate upsampling: minority → geometric mean of min/max class size."""
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

    out = pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )
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
):
    """
    Train SVM / RF / GB and keep best model.
    Selection criterion:
      use_class1_f1=True  → F1 of class 1 only   (for Valid Larva model)
      use_class1_f1=False → weighted F1           (for Posture model)
    Saves model.pkl and classification_report.txt inside out_dir.
    Returns (best_model, X_test, y_test, y_pred).
    """
    print(f"\n  ── {model_name} ──────────────────────────────────")
    criterion_label = "class-1 F1" if use_class1_f1 else "weighted F1"
    print(f"    Selection criterion: {criterion_label}")

    if len(X) < 20 or len(np.unique(y)) < 2:
        print("    ⚠️  Insufficient data – skipping training.")
        return None, None, None, None

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    X_tr_up, y_tr_up = upsample(X_tr, y_tr)

    def _score(y_true, y_pred):
        if use_class1_f1:
            per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            return float(per_class[1]) if len(per_class) > 1 else 0.0
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)

    candidates = {}

    # ── RandomForest ─────────────────────────────────────────
    print("    Training RandomForest …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_tr_up, y_tr_up)
    rf_pred = rf.predict(X_te)
    rf_f1   = _score(y_te, rf_pred)
    rf_acc  = accuracy_score(y_te, rf_pred)
    print(f"      RF   Acc={rf_acc:.3f}  {criterion_label}={rf_f1:.3f}")
    candidates['RandomForest'] = (rf, rf_f1, rf_pred)

    # ── SVM ─────────────────────────────────────────────────
    print("    Training SVM (RBF) …")
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel='rbf', C=10, gamma='scale',
            probability=True, class_weight='balanced',
            random_state=RANDOM_STATE,
        )),
    ])
    svm_pipe.fit(X_tr_up, y_tr_up)
    svm_pred = svm_pipe.predict(X_te)
    svm_f1   = _score(y_te, svm_pred)
    svm_acc  = accuracy_score(y_te, svm_pred)
    print(f"      SVM  Acc={svm_acc:.3f}  {criterion_label}={svm_f1:.3f}")
    candidates['SVM'] = (svm_pipe, svm_f1, svm_pred)

    # ── GradientBoosting ─────────────────────────────────────
    print("    Training GradientBoosting …")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.08, subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_tr_up, y_tr_up)
    gb_pred = gb.predict(X_te)
    gb_f1   = _score(y_te, gb_pred)
    gb_acc  = accuracy_score(y_te, gb_pred)
    print(f"      GB   Acc={gb_acc:.3f}  {criterion_label}={gb_f1:.3f}")
    candidates['GradientBoosting'] = (gb, gb_f1, gb_pred)

    # ── Select best ──────────────────────────────────────────
    best_key  = max(candidates, key=lambda k: candidates[k][1])
    best_model, best_f1, best_pred = candidates[best_key]
    best_acc  = accuracy_score(y_te, best_pred)
    print(f"\n    ✓ Best: {best_key}  Acc={best_acc:.3f}  {criterion_label}={best_f1:.3f}")

    # Save model
    joblib.dump(best_model, out_dir / 'model.pkl')

    # Classification report
    report = classification_report(
        y_te, best_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(f"\n{report}")
    with open(out_dir / 'classification_report.txt', 'w') as fh:
        fh.write(f"Best model: {best_key}\n")
        fh.write(f"Selection criterion: {criterion_label}\n\n")
        fh.write(report)

    return best_model, X_te, y_te, best_pred


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
    """Walk analysis_full and return list of (date, img_name, fname, path)."""
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
                if f.name.startswith('larva_') and f.suffix.lower() in ('.png','.jpg','.jpeg'):
                    all_larvae.append((date_dir.name, img_dir.name, f.name, f))
    return all_larvae


# ============================================================
#  PREDICTION STAGE (cascaded)
# ============================================================
def predict_all(all_larvae: List[Tuple], valid_model, posture_model) -> pd.DataFrame:
    """
    Apply valid_model to all larvae.
    Apply posture_model only to those predicted as valid.
    """
    total = len(all_larvae)
    print(f"\n  Running predictions on {total} larvae …")

    rows = []
    for i, (date, img_name, fname, fpath) in enumerate(all_larvae):
        if (i + 1) % 1000 == 0:
            print(f"    … {i+1}/{total}")

        feats = build_feature_row(date, img_name, fname, fpath)
        base = {
            'date': date,
            'image_name': img_name,
            'larva_filename': fname,
            'predicted_valid': 0,
            'valid_confidence': 0.0,
            'predicted_posture': np.nan,
            'posture_confidence': np.nan,
            'body_length_px': 0.0,
            'body_length_mm': 0.0,
        }

        if feats is None:
            rows.append(base)
            continue

        Xr = pd.DataFrame([feats])[_FEAT_KEYS].fillna(0.0)

        # ── Model 1: valid larva ──────────────────────────────
        try:
            v_prob = valid_model.predict_proba(Xr)[0]
            v_pred = int(np.argmax(v_prob))
            v_conf = float(v_prob[v_pred])
        except Exception:
            v_pred, v_conf = 0, 0.0

        base['predicted_valid']  = v_pred
        base['valid_confidence'] = v_conf

        bl_px = feats.get('body_length', 0.0)
        base['body_length_px'] = bl_px
        base['body_length_mm'] = bl_px * PIXEL_TO_MM

        # ── Model 2: posture (only if predicted valid) ────────
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

    df = pd.DataFrame(rows)
    return df


# ============================================================
#  SAVE PREDICTIONS
# ============================================================
def save_predictions(df: pd.DataFrame):
    pred_dir = OUTPUT_DIR / 'predictions'

    # Full
    out = pred_dir / 'predictions_all_larvae.xlsx'
    df.to_excel(out, index=False)
    print(f"  ✓ {out}")

    # Valid only
    valid_df = df[df['predicted_valid'] == 1]
    out2 = pred_dir / 'valid_predictions.xlsx'
    valid_df.to_excel(out2, index=False)
    print(f"  ✓ {out2}  ({len(valid_df)} rows)")

    # Correct posture only
    posture_df = df[df['predicted_posture'] == 1]
    out3 = pred_dir / 'posture_predictions.xlsx'
    posture_df.to_excel(out3, index=False)
    print(f"  ✓ {out3}  ({len(posture_df)} rows)")


# ============================================================
#  HIGH-CONFIDENCE COPY
# ============================================================
def copy_high_confidence(df: pd.DataFrame, label_col: str, conf_col: str,
                          subset_label: str, dest_subdir: str):
    """Copy high-confidence larva images into dated sub-folders."""
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
#  EXAMPLES (1 best per date)
# ============================================================
def export_examples(df: pd.DataFrame, label_col: str, conf_col: str,
                    dest_subdir: str):
    subset = df[(df[label_col] == 1) & df[conf_col].notna()]
    for date in sorted(subset['date'].unique(), key=date_sort_key):
        date_rows = subset[subset['date'] == date]
        if date_rows.empty:
            continue
        best = date_rows.sort_values(conf_col, ascending=False).iloc[0]
        src = (ANALYSIS_DIR / str(best['date']) / str(best['image_name'])
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
    """Mean body length per date with n= and value annotations."""
    sub = df[(df['body_length_px'] > 0)].copy()
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
    """Create all plots for both models."""
    print("\n  Creating visualisations …")
    fig_root = OUTPUT_DIR / 'figures'

    # ── Count per date ────────────────────────────────────────
    for col, label, color in [
        ('predicted_valid',   'Valid Larvae',   '#4ecdc4'),
        ('predicted_posture', 'Correct Posture','#45b7d1'),
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
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
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

    # ── Avg length per date ───────────────────────────────────
    valid_df   = df[df['predicted_valid'] == 1].copy()
    posture_df = df[df['predicted_posture'] == 1].copy()

    valid_df['body_length_mm']   = valid_df['body_length_px'] * PIXEL_TO_MM
    posture_df['body_length_mm'] = posture_df['body_length_px'] * PIXEL_TO_MM

    _avg_length_plot(
        valid_df, 'Valid Larvae',
        fig_root / 'avg_length_per_date_valid.png',
        fig_root / 'length_statistics_valid.csv',
        color='steelblue',
    )
    _avg_length_plot(
        posture_df, 'Correct Posture Larvae',
        fig_root / 'avg_length_per_date_posture.png',
        fig_root / 'length_statistics_posture.csv',
        color='darkgreen',
    )

    # ── Confusion matrices ────────────────────────────────────
    if valid_y_te is not None and valid_y_pred is not None:
        plot_confusion_matrix(
            valid_y_te, valid_y_pred,
            class_names=['Not Valid', 'Valid'],
            title='Confusion Matrix – Valid Larva Model',
            save_path=OUTPUT_DIR / 'valid_model' / 'figures' / 'confusion_matrix.png',
        )

    if posture_y_te is not None and posture_y_pred is not None:
        plot_confusion_matrix(
            posture_y_te, posture_y_pred,
            class_names=['Bad Posture', 'Correct Posture'],
            title='Confusion Matrix – Posture Model',
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
    print(f"  Predicted valid        : {n_valid} ({100*n_valid/max(total,1):.1f} %)")
    print(f"  Predicted correct pose : {n_post} ({100*n_post/max(total,1):.1f} %)")

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
    print("DUAL LARVA CLASSIFICATION PIPELINE")
    print("  Model 1 → Valid Larva  (binary: 0=not-valid, 1=valid)")
    print("  Model 2 → Posture      (binary: 0=bad-pose,  1=T-like)")
    print("=" * 70)

    # ── Sanity checks ─────────────────────────────────────────
    if not LABELS_FILE.exists():
        print(f"❌ Labels file not found: {LABELS_FILE}")
        sys.exit(1)
    if not ANALYSIS_DIR.exists():
        print(f"❌ Analysis directory not found: {ANALYSIS_DIR}")
        sys.exit(1)

    create_output_structure()

    # ── Load labels ───────────────────────────────────────────
    labels_df = pd.read_excel(LABELS_FILE)
    print(f"\n✓ Loaded {len(labels_df)} label rows")
    print(f"  is_valid_larva distribution : {labels_df['is_valid_larva'].value_counts().to_dict()}")
    print(f"  shape_score distribution    : {labels_df['shape_score'].value_counts().to_dict()}")

    # ── MODEL 1: Valid Larva ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1 – VALID LARVA CLASSIFIER")
    print("=" * 70)

    XV, yV, _ = build_valid_dataset(labels_df)
    valid_model, Xte_V, yte_V, ypred_V = train_binary_model(
        XV, yV,
        model_name='Valid Larva',
        out_dir=OUTPUT_DIR / 'valid_model',
        class_names=['Not Valid', 'Valid'],
        use_class1_f1=True,
    )

    if valid_model is None:
        print("❌ Could not train valid-larva model – aborting.")
        sys.exit(1)

    plot_class_distribution(
        yV, ['Not Valid','Valid'],
        title='Valid Larva Label Distribution',
        save_path=OUTPUT_DIR / 'valid_model' / 'figures' / 'class_distribution.png',
    )

    # ── MODEL 2: Posture ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 – POSTURE / T-SHAPE CLASSIFIER")
    print("=" * 70)

    XP, yP, _ = build_posture_dataset(labels_df)
    posture_model, Xte_P, yte_P, ypred_P = train_binary_model(
        XP, yP,
        model_name='Posture',
        out_dir=OUTPUT_DIR / 'posture_model',
        class_names=['Bad Posture', 'Correct Posture'],
    )

    plot_class_distribution(
        yP, ['Bad Posture','Correct Posture'],
        title='Posture Label Distribution',
        save_path=OUTPUT_DIR / 'posture_model' / 'figures' / 'class_distribution.png',
    )

    # ── PREDICTION STAGE ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 – PREDICT ALL LARVAE (CASCADED)")
    print("=" * 70)

    all_larvae = collect_all_larvae()
    print(f"  Found {len(all_larvae)} larvae (18.10 excluded)")

    preds_df = predict_all(all_larvae, valid_model, posture_model)

    # ── Save predictions ──────────────────────────────────────
    print("\n  Saving prediction files …")
    save_predictions(preds_df)

    # ── High-confidence copies ────────────────────────────────
    print("\n  Copying high-confidence larvae …")
    copy_high_confidence(preds_df, 'predicted_valid',   'valid_confidence',
                         'Valid',   'valid')
    copy_high_confidence(preds_df, 'predicted_posture', 'posture_confidence',
                         'Posture', 'posture')

    # ── Examples ──────────────────────────────────────────────
    print("\n  Exporting examples …")
    export_examples(preds_df, 'predicted_valid',   'valid_confidence',   'valid')
    export_examples(preds_df, 'predicted_posture', 'posture_confidence', 'posture')

    # ── Visualisations ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 – VISUALISATIONS")
    print("=" * 70)

    create_all_visualisations(
        preds_df,
        yte_V, ypred_V,
        yte_P, ypred_P,
    )

    # ── Summary ───────────────────────────────────────────────
    print_summary(preds_df)

    print("\n" + "=" * 70)
    print("DUAL LARVA CLASSIFICATION PIPELINE COMPLETE")
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

