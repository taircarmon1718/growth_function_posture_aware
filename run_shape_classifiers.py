#!/usr/bin/env python3
"""
Larva Shape Classifier Pipeline - 3-Class Model (Improved)

Classifies valid larvae into 3 shape categories:
- 0 = Bad shape (not suitable for measurement)
- 1 = OK shape (acceptable for analysis)
- 2 = Great shape (ideal specimens)

Key improvements:
- Trains only on valid larvae (is_valid_larva == 1)
- Uses macro F1 for model selection
- Moderate upsampling with class_weight='balanced'
- Stratified K-Fold cross-validation
- Feature importance visualization
- Clean publication-ready plots with n and mean values
- Excludes date 18.10 from all outputs
- NEW: Body length is computed as the longest geodesic path along the medial skeleton
  (not simple skeleton pixel count)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils import resample
import heapq
import warnings
import traceback
import shutil
import seaborn as sns

warnings.filterwarnings('ignore')

# =================== Configuration ===================
ROOT_DIR = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full"
LABELS_FILE = ANALYSIS_DIR / "larva_quality_labels.xlsx"

RANDOM_STATE = 42
TEST_SIZE = 0.2
PIXEL_TO_MM = 0.232255814
HIGH_CONFIDENCE_THRESHOLD = 0.7
N_CV_FOLDS = 5

# Dates to exclude from all processing
EXCLUDED_DATES = {'18.10', '18.1'}

# Output directory
OUTPUT_DIR = ROOT_DIR / 'svm_shape_multiclass'
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)
(OUTPUT_DIR / 'examples').mkdir(exist_ok=True)
(OUTPUT_DIR / 'high_confidence').mkdir(exist_ok=True)

# Legacy folders for compatibility
for folder in ['svm_shape_1_classifier', 'svm_shape_2_classifier', 'svm_shape_1_2_combined_classifier']:
    (ROOT_DIR / folder).mkdir(exist_ok=True)
    (ROOT_DIR / folder / 'figures').mkdir(exist_ok=True)
    (ROOT_DIR / folder / 'examples').mkdir(exist_ok=True)
    (ROOT_DIR / folder / 'high_confidence').mkdir(exist_ok=True)

CLASS_NAMES = {0: 'Bad', 1: 'OK', 2: 'Great'}

# =================== Utilities ===================

def parse_date_key(name):
    """Return (month, day) so that 31.10 (Oct 31) sorts before 3.11 (Nov 3)."""
    try:
        parts = str(name).split('.')
        if len(parts) >= 2:
            day_s, mon_s = parts[0], parts[1]
            return (int(mon_s), int(day_s))
    except:
        pass
    return (999, 999)


def fix_date_format(date_val):
    """Fix date format from labels (19.1) to folder format (19.10)."""
    s = str(date_val)
    parts = s.split('.')
    if len(parts) == 2:
        day, month = parts[0], parts[1]
        if month == '1':
            return f"{day}.10"
    return s


def is_excluded_date(date_str):
    """Check if date should be excluded."""
    date_fixed = fix_date_format(date_str)
    return date_fixed in EXCLUDED_DATES or str(date_str) in EXCLUDED_DATES


def get_sorted_dates(dates):
    """Sort dates chronologically, excluding unwanted dates."""
    filtered = [d for d in dates if not is_excluded_date(d)]
    return sorted(filtered, key=parse_date_key)


# =================== FEATURE EXTRACTION ===================

def extract_features(img_path: Path):
    """Extract features from a larva image."""
    img_path = Path(img_path)
    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Threshold
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (bw > 0).astype(np.uint8)
    if np.mean(bw) > 0.5:
        bw = 1 - bw

    features = {}
    area = int(np.sum(bw))
    features['area'] = area
    features['area_ratio'] = float(area) / float(h_img * w_img) if (h_img * w_img) > 0 else 0

    if area < 50:
        return _empty_features()

    # Bounding box
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        return _empty_features()

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1

    features['bbox_h'] = bbox_h
    features['bbox_w'] = bbox_w
    features['aspect_ratio'] = float(bbox_h) / float(bbox_w) if bbox_w > 0 else 0

    crop = bw[y_min:y_max+1, x_min:x_max+1]
    crop_gray = gray[y_min:y_max+1, x_min:x_max+1]

    # Contour analysis
    contours, _ = cv2.findContours((crop * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        features['circularity'] = float(4 * np.pi * cnt_area / (perimeter ** 2)) if perimeter > 0 else 0
        features['perimeter'] = perimeter

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = float(cnt_area / hull_area) if hull_area > 0 else 0

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, MA), angle = ellipse
            features['ellipse_ratio'] = float(MA / ma) if ma > 0 else 0
            features['ellipse_angle'] = float(angle)
        else:
            features['ellipse_ratio'] = features['aspect_ratio']
            features['ellipse_angle'] = 0

        # Hu moments
        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        for i in range(7):
            val = hu[i]
            features[f'hu{i+1}'] = float(-np.sign(val) * np.log10(abs(val) + 1e-10))
    else:
        features['circularity'] = 0
        features['perimeter'] = 0
        features['solidity'] = 0
        features['ellipse_ratio'] = 0
        features['ellipse_angle'] = 0
        for i in range(7):
            features[f'hu{i+1}'] = 0

    # Intensity
    roi_vals = crop_gray[crop > 0]
    if len(roi_vals) > 0:
        features['mean_intensity'] = float(np.mean(roi_vals))
        features['std_intensity'] = float(np.std(roi_vals))
    else:
        features['mean_intensity'] = 0
        features['std_intensity'] = 0

    # Skeleton analysis
    skeleton = _skeletonize(crop)
    skel_pixels = int(np.sum(skeleton))
    features['skeleton_pixel_count'] = skel_pixels  # Keep for reference

    # NEW: Compute geodesic body length (longest path along skeleton)
    geodesic_length = _compute_body_length_from_skeleton(skeleton)
    features['body_length'] = geodesic_length  # This is now the true body length
    features['skeleton_length'] = geodesic_length  # Alias for compatibility

    if skel_pixels > 5:
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        endpoints = int(np.sum((skeleton == 1) & (neighbors == 1)))
        junctions = int(np.sum((skeleton == 1) & (neighbors >= 3)))
        features['num_endpoints'] = endpoints
        features['num_junctions'] = junctions

        skel_ys, skel_xs = np.where(skeleton > 0)
        if len(skel_ys) > 0:
            v_extent = int(skel_ys.max() - skel_ys.min())
            h_extent = int(skel_xs.max() - skel_xs.min())
            features['vertical_extent'] = v_extent
            features['horizontal_extent'] = h_extent
            features['verticality'] = float(v_extent) / float(h_extent + 1)
    else:
        features['num_endpoints'] = 0
        features['num_junctions'] = 0
        features['vertical_extent'] = 0
        features['horizontal_extent'] = 0
        features['verticality'] = 0

    return features


def _empty_features():
    """Return empty feature dict."""
    return {
        'area': 0, 'area_ratio': 0, 'bbox_h': 0, 'bbox_w': 0, 'aspect_ratio': 0,
        'circularity': 0, 'perimeter': 0, 'solidity': 0, 'ellipse_ratio': 0, 'ellipse_angle': 0,
        'hu1': 0, 'hu2': 0, 'hu3': 0, 'hu4': 0, 'hu5': 0, 'hu6': 0, 'hu7': 0,
        'mean_intensity': 0, 'std_intensity': 0,
        'body_length': 0, 'skeleton_length': 0, 'skeleton_pixel_count': 0,
        'num_endpoints': 0, 'num_junctions': 0, 'vertical_extent': 0, 'horizontal_extent': 0, 'verticality': 0
    }


def _skeletonize(binary_img):
    """Fast morphological skeletonization."""
    img = binary_img.copy().astype(np.uint8)
    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(100):
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        subset = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, subset)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def _compute_geodesic_body_length(skeleton):
    """
    Compute the longest geodesic path along the skeleton.

    This measures the true body length as the longest path between skeleton endpoints,
    accounting for diagonal connectivity (√2 for diagonal, 1 for orthogonal steps).

    This replaces simple skeleton pixel counting with a proper geodesic measurement
    that represents the main body axis only.

    Returns:
        float: The geodesic length of the longest path in pixels
    """
    if skeleton is None or np.sum(skeleton) < 2:
        return 0.0

    # Find skeleton pixel coordinates
    skel_points = np.argwhere(skeleton > 0)
    if len(skel_points) < 2:
        return 0.0

    # Build adjacency for skeleton pixels
    # Use 8-connectivity
    h, w = skeleton.shape

    # Find endpoints (pixels with only 1 neighbor)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = np.argwhere((skeleton > 0) & (neighbor_count == 1))

    # If no clear endpoints, use pixels with minimum neighbors or extremes
    if len(endpoints) < 2:
        # Fall back to using the two most distant skeleton points
        endpoints = skel_points

    if len(endpoints) < 2:
        return 0.0

    # Convert skeleton to a set for fast lookup
    skel_set = set(map(tuple, skel_points))

    # 8-connectivity neighbors with distances
    # (dy, dx, distance)
    neighbors_8 = [
        (-1, -1, np.sqrt(2)),  # top-left (diagonal)
        (-1,  0, 1.0),         # top
        (-1,  1, np.sqrt(2)),  # top-right (diagonal)
        ( 0, -1, 1.0),         # left
        ( 0,  1, 1.0),         # right
        ( 1, -1, np.sqrt(2)),  # bottom-left (diagonal)
        ( 1,  0, 1.0),         # bottom
        ( 1,  1, np.sqrt(2)),  # bottom-right (diagonal)
    ]

    def dijkstra_longest_path(start):
        """Find longest shortest path from start to any endpoint using Dijkstra."""
        dist = {tuple(start): 0.0}
        pq = [(0.0, tuple(start))]  # (distance, point)

        while pq:
            d, current = heapq.heappop(pq)

            if d > dist.get(current, float('inf')):
                continue

            y, x = current
            for dy, dx, step_dist in neighbors_8:
                ny, nx = y + dy, x + dx
                neighbor = (ny, nx)

                if neighbor in skel_set:
                    new_dist = d + step_dist
                    if new_dist < dist.get(neighbor, float('inf')):
                        dist[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))

        return dist

    # Find the longest path by running Dijkstra from each endpoint
    max_length = 0.0

    # Limit to first 10 endpoints for efficiency
    test_endpoints = endpoints[:min(10, len(endpoints))]

    for ep in test_endpoints:
        distances = dijkstra_longest_path(ep)
        if distances:
            # Find the maximum distance from this starting point
            max_dist = max(distances.values())
            if max_dist > max_length:
                max_length = max_dist

    return float(max_length)


def _compute_body_length_from_skeleton(skeleton):
    """
    Wrapper function to compute body length from skeleton.
    Returns the geodesic length (longest path along skeleton).
    """
    return _compute_geodesic_body_length(skeleton)


def extract_morphometrics(date, image_name, larva_filename):
    """
    Extract morphometric data from CSV (legacy support).

    Note: body_length is now primarily computed from the geodesic skeleton path
    in extract_features(). This function provides fallback/additional data.
    """
    csv_path = ANALYSIS_DIR / str(date) / str(image_name) / 'morphometrics.csv'
    feats = {'csv_body_length': 0.0}  # Renamed to avoid overwriting computed body_length
    if not csv_path.exists():
        return feats
    try:
        df = pd.read_csv(csv_path)
        lid = int(larva_filename.replace('larva_', '').replace('.png', ''))
        row = df[df['larva_id'] == lid]
        if not row.empty:
            feats['csv_body_length'] = float(row['body_length'].values[0])
    except:
        pass
    return feats


# =================== Dataset Building ===================

def build_dataset(labels_df):
    """
    Build feature dataset with 3-class labels.
    ONLY uses valid larvae (is_valid_larva == 1).
    Excludes date 18.10.
    """
    rows = []
    print("Building dataset (valid larvae only, excluding 18.10)...")

    # Filter to valid larvae only
    valid_larvae = labels_df[labels_df['is_valid_larva'] == 1].copy()
    print(f"  Total labels: {len(labels_df)}, Valid larvae: {len(valid_larvae)}")

    for idx, r in valid_larvae.iterrows():
        date = fix_date_format(r['date'])

        # Skip excluded dates
        if is_excluded_date(date):
            continue

        image_name = str(r['image_name'])
        fname = str(r['larva_filename'])

        img_path = ANALYSIS_DIR / date / image_name / 'larvae_reports' / fname
        feats = extract_features(img_path)
        morph = extract_morphometrics(date, image_name, fname)

        if feats is None:
            continue

        combined = {**feats, **morph}
        shape_score = int(r.get('shape_score', 0))

        rows.append((combined, shape_score, date, image_name, fname))

    if not rows:
        return pd.DataFrame(), np.array([]), pd.DataFrame()

    X = pd.DataFrame([r[0] for r in rows]).fillna(0)
    y = np.array([r[1] for r in rows])
    meta = pd.DataFrame([{'date': r[2], 'image_name': r[3], 'larva_filename': r[4]} for r in rows])

    print(f"  Built dataset: {len(rows)} samples, {len(X.columns)} features")
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Class distribution: {dict(zip([CLASS_NAMES.get(u, u) for u in unique], counts))}")

    return X, y, meta


# =================== Model Training ===================

def train_multiclass_model(X, y, out_dir: Path):
    """
    Train a 3-class model using macro F1 for selection.
    Uses class_weight='balanced' and moderate upsampling.
    Includes stratified K-fold cross-validation.
    """
    print("\n" + "=" * 50)
    print("TRAINING 3-CLASS SHAPE MODEL")
    print("=" * 50)

    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip([CLASS_NAMES.get(u, u) for u in unique], counts))
    print(f"Class distribution: {class_dist}")

    if len(X) < 20 or len(np.unique(y)) < 2:
        print("⚠️ Insufficient data")
        return None, None, None

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    # Moderate upsampling (to median class size, not max)
    X_train, y_train = _moderate_upsample(X_train, y_train)

    models = {}
    feature_names = list(X.columns)

    # Random Forest with class_weight='balanced'
    print("\n  Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"    RF CV macro F1: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_macro_f1 = f1_score(y_test, rf_pred, average='macro', zero_division=0)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"    RF Test: Accuracy={rf_acc:.3f}, Macro F1={rf_macro_f1:.3f}")
    models['rf'] = {'model': rf, 'f1': rf_macro_f1, 'acc': rf_acc, 'cv_f1': rf_cv_scores.mean()}

    # SVM with class_weight='balanced'
    print("  Training SVM...")
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            decision_function_shape='ovr'
        ))
    ])

    svm_cv_scores = cross_val_score(svm_pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"    SVM CV macro F1: {svm_cv_scores.mean():.3f} ± {svm_cv_scores.std():.3f}")

    svm_pipe.fit(X_train, y_train)
    svm_pred = svm_pipe.predict(X_test)
    svm_macro_f1 = f1_score(y_test, svm_pred, average='macro', zero_division=0)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"    SVM Test: Accuracy={svm_acc:.3f}, Macro F1={svm_macro_f1:.3f}")
    models['svm'] = {'model': svm_pipe, 'f1': svm_macro_f1, 'acc': svm_acc, 'cv_f1': svm_cv_scores.mean()}

    # Gradient Boosting
    print("  Training GradientBoosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )

    gb_cv_scores = cross_val_score(gb, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"    GB CV macro F1: {gb_cv_scores.mean():.3f} ± {gb_cv_scores.std():.3f}")

    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_macro_f1 = f1_score(y_test, gb_pred, average='macro', zero_division=0)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"    GB Test: Accuracy={gb_acc:.3f}, Macro F1={gb_macro_f1:.3f}")
    models['gb'] = {'model': gb, 'f1': gb_macro_f1, 'acc': gb_acc, 'cv_f1': gb_cv_scores.mean()}

    # Select best by macro F1
    best_name = max(models, key=lambda k: models[k]['f1'])
    best_model = models[best_name]['model']
    best_f1 = models[best_name]['f1']
    best_acc = models[best_name]['acc']

    print(f"\n  ✓ Selected: {best_name.upper()} (Macro F1={best_f1:.3f}, Accuracy={best_acc:.3f})")

    # Save model
    joblib.dump(best_model, out_dir / 'shape_model_3class.pkl')

    # Classification report
    best_pred = best_model.predict(X_test)
    report = classification_report(y_test, best_pred, target_names=['Bad', 'OK', 'Great'], zero_division=0)
    print(f"\n  Classification Report:\n{report}")
    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Macro F1: {best_f1:.3f}\n")
        f.write(f"Accuracy: {best_acc:.3f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'OK', 'Great'],
                yticklabels=['Bad', 'OK', 'Great'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_name.upper()} (Macro F1={best_f1:.3f})')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'confusion_matrix.png', dpi=150)
    plt.close()

    # Feature importance (for RF or GB)
    _plot_feature_importance(best_model, best_name, feature_names, out_dir)

    return best_model, best_name, feature_names


def _moderate_upsample(X, y):
    """
    Moderate upsampling: upsample minority classes to median class size.
    Less aggressive than full equalization.
    """
    X_df = pd.DataFrame(X).copy()
    X_df['_y'] = y

    unique, counts = np.unique(y, return_counts=True)
    median_count = int(np.median(counts))
    target_count = max(median_count, min(counts) * 2)  # At least double the smallest

    dfs = []
    for cls in unique:
        cls_df = X_df[X_df['_y'] == cls]
        if len(cls_df) < target_count:
            upsampled = resample(cls_df, replace=True, n_samples=target_count, random_state=RANDOM_STATE)
            dfs.append(upsampled)
        else:
            dfs.append(cls_df)

    result = pd.concat(dfs, ignore_index=True)
    return result.drop(columns=['_y']), result['_y'].values


def _plot_feature_importance(model, model_name, feature_names, out_dir):
    """Plot and save feature importance for tree-based models."""
    importance = None

    if model_name == 'rf' and hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif model_name == 'gb' and hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif model_name == 'svm':
        # SVM doesn't have direct feature importance
        print("  Note: SVM doesn't provide feature importance")
        return

    if importance is None:
        return

    # Sort by importance
    indices = np.argsort(importance)[::-1]
    top_n = min(15, len(feature_names))

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices[:top_n]][::-1], color='steelblue', edgecolor='black')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances - {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / 'feature_importance.png', dpi=150)
    plt.close()

    print(f"  ✓ Feature importance plot saved")


# =================== Prediction ===================

def collect_all_larvae():
    """Collect all larva image paths, excluding date 18.10."""
    all_larvae = []
    for date_dir in sorted(ANALYSIS_DIR.iterdir(), key=lambda p: parse_date_key(p.name)):
        if not date_dir.is_dir() or not date_dir.name[0].isdigit():
            continue

        # Skip excluded dates
        if is_excluded_date(date_dir.name):
            continue

        for img_dir in sorted(date_dir.iterdir()):
            if not img_dir.is_dir():
                continue
            larvae_dir = img_dir / 'larvae_reports'
            if not larvae_dir.exists():
                continue
            for f in sorted(larvae_dir.iterdir()):
                if f.name.startswith('larva_') and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    all_larvae.append((date_dir.name, img_dir.name, f.name, f))
    return all_larvae


def predict_all(all_larvae, model, out_dir: Path):
    """Predict shape class for all larvae."""
    print(f"\n  Predicting {len(all_larvae)} larvae...")

    predictions = []
    for i, (date, img_name, fname, fpath) in enumerate(all_larvae):
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1}/{len(all_larvae)}...")

        feats = extract_features(fpath)
        morph = extract_morphometrics(date, img_name, fname)

        if feats is None:
            predictions.append({
                'date': date, 'image_name': img_name, 'larva_filename': fname,
                'predicted_class': 0, 'class_name': 'Bad',
                'confidence': 0, 'prob_bad': 1.0, 'prob_ok': 0.0, 'prob_great': 0.0,
                'body_length_px': 0, 'body_length_mm': 0.0
            })
            continue

        X_row = pd.DataFrame([{**feats, **morph}]).fillna(0)

        try:
            probs = model.predict_proba(X_row)[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            if len(probs) == 3:
                prob_bad, prob_ok, prob_great = probs
            elif len(probs) == 2:
                prob_bad, prob_ok = probs
                prob_great = 0.0
            else:
                prob_bad = probs[0] if len(probs) > 0 else 0
                prob_ok = 0.0
                prob_great = 0.0
        except:
            pred_class = 0
            confidence = 0.0
            prob_bad, prob_ok, prob_great = 1.0, 0.0, 0.0

        # Use geodesic body_length from features (computed from skeleton)
        body_length_px = feats.get('body_length', 0)
        body_length_mm = body_length_px * PIXEL_TO_MM

        predictions.append({
            'date': date,
            'image_name': img_name,
            'larva_filename': fname,
            'predicted_class': pred_class,
            'class_name': CLASS_NAMES.get(pred_class, 'Unknown'),
            'confidence': confidence,
            'prob_bad': prob_bad,
            'prob_ok': prob_ok,
            'prob_great': prob_great,
            'body_length_px': body_length_px,
            'body_length_mm': body_length_mm,
            'aspect_ratio': feats.get('aspect_ratio', 0),
            'circularity': feats.get('circularity', 0),
            'solidity': feats.get('solidity', 0)
        })

    df = pd.DataFrame(predictions)

    # Save main predictions
    df.to_excel(out_dir / 'predictions_all_larvae.xlsx', index=False)
    print(f"  ✓ Saved predictions: {out_dir / 'predictions_all_larvae.xlsx'}")

    # Save to legacy folders
    for shape_val, folder in [(1, 'svm_shape_1_classifier'), (2, 'svm_shape_2_classifier')]:
        legacy_df = df.copy()
        legacy_df['predicted_shape'] = (legacy_df['predicted_class'] == shape_val).astype(int)
        legacy_df['shape_confidence'] = legacy_df[f'prob_{"ok" if shape_val == 1 else "great"}']
        legacy_df.to_excel(ROOT_DIR / folder / 'predictions_all_larvae.xlsx', index=False)

    combined_df = df.copy()
    combined_df['predicted_shape'] = (combined_df['predicted_class'] >= 1).astype(int)
    combined_df['shape_confidence'] = combined_df['prob_ok'] + combined_df['prob_great']
    combined_df.to_excel(ROOT_DIR / 'svm_shape_1_2_combined_classifier' / 'predictions_all_larvae.xlsx', index=False)

    return df


def export_results(preds_df, out_dir: Path):
    """Export high-confidence larvae and examples."""
    print("\n  Exporting results...")

    for class_val in [1, 2]:
        class_name = CLASS_NAMES[class_val]
        high = preds_df[
            (preds_df['predicted_class'] == class_val) &
            (preds_df['confidence'] >= HIGH_CONFIDENCE_THRESHOLD)
        ]

        copied = 0
        for date in high['date'].unique():
            if is_excluded_date(date):
                continue
            date_dir = out_dir / 'high_confidence' / class_name / str(date)
            date_dir.mkdir(parents=True, exist_ok=True)

            for _, row in high[high['date'] == date].iterrows():
                src = ANALYSIS_DIR / row['date'] / row['image_name'] / 'larvae_reports' / row['larva_filename']
                if src.exists():
                    try:
                        shutil.copy2(src, date_dir / row['larva_filename'])
                        copied += 1
                    except:
                        pass

        print(f"  ✓ Copied {copied} high-confidence {class_name} larvae")

    # Export examples per date
    for class_val in [1, 2]:
        class_name = CLASS_NAMES[class_val]
        examples_dir = out_dir / 'examples' / class_name

        for date in get_sorted_dates(preds_df['date'].unique()):
            date_rows = preds_df[(preds_df['date'] == date) & (preds_df['predicted_class'] == class_val)]
            if date_rows.empty:
                continue

            best = date_rows.sort_values('confidence', ascending=False).iloc[0]
            src = ANALYSIS_DIR / best['date'] / best['image_name'] / 'larvae_reports' / best['larva_filename']
            if src.exists():
                dst_dir = examples_dir / str(date)
                dst_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, dst_dir / best['larva_filename'])
                except:
                    pass

    print(f"  ✓ Exported examples per date")


# =================== Visualizations ===================

def create_visualizations(preds_df, out_dir: Path):
    """Create publication-ready visualizations."""
    print("\n  Creating visualizations...")
    figs_dir = out_dir / 'figures'

    # Filter out excluded dates
    preds_df = preds_df[~preds_df['date'].apply(is_excluded_date)].copy()

    # Class distribution pie chart
    class_counts = preds_df['predicted_class'].value_counts().sort_index()
    plt.figure(figsize=(8, 8))
    labels = [CLASS_NAMES.get(i, str(i)) for i in class_counts.index]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    wedges, texts, autotexts = plt.pie(
        class_counts.values, labels=labels, autopct='%1.1f%%',
        colors=colors[:len(labels)], textprops={'fontsize': 12}
    )
    plt.title('Predicted Shape Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figs_dir / 'class_distribution.png', dpi=150)
    plt.close()

    # Count per date for each class
    _plot_count_per_date(preds_df, figs_dir)

    # Average body length plots
    _plot_body_length(preds_df, 1, 'OK', figs_dir)
    _plot_body_length(preds_df, 2, 'Great', figs_dir)
    _plot_body_length_combined(preds_df, figs_dir)

    # Copy to legacy folders
    for fig_file in figs_dir.glob('*.png'):
        for folder in ['svm_shape_1_classifier', 'svm_shape_2_classifier', 'svm_shape_1_2_combined_classifier']:
            try:
                shutil.copy2(fig_file, ROOT_DIR / folder / 'figures' / fig_file.name)
            except:
                pass

    print(f"  ✓ Saved visualizations to {figs_dir}")


def _plot_count_per_date(preds_df, figs_dir):
    """Plot count per date for each class."""
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, class_val in enumerate([0, 1, 2]):
        class_name = CLASS_NAMES[class_val]
        class_df = preds_df[preds_df['predicted_class'] == class_val]

        if class_df.empty:
            axes[idx].text(0.5, 0.5, f'No {class_name} predictions', ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'{class_name} Count per Date', fontsize=12, fontweight='bold')
            continue

        counts = class_df.groupby('date').size()
        dates_sorted = get_sorted_dates(counts.index)
        counts = counts.reindex(dates_sorted)

        x = range(len(counts))
        bars = axes[idx].bar(x, counts.values, color=colors[idx], edgecolor='black')

        # Add count labels on bars
        for bar, count in zip(bars, counts.values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          str(count), ha='center', va='bottom', fontsize=9)

        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(counts.index, rotation=45, ha='right', fontsize=10)
        axes[idx].set_xlabel('Date', fontsize=11)
        axes[idx].set_ylabel('Count', fontsize=11)
        axes[idx].set_title(f'{class_name} Count per Date', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figs_dir / 'count_per_date_by_class.png', dpi=150)
    plt.close()


def _plot_body_length(preds_df, class_val, class_name, figs_dir):
    """Plot average body length per date for a specific class with n and mean values."""
    class_df = preds_df[(preds_df['predicted_class'] == class_val) & (preds_df['body_length_px'] > 0)]

    if class_df.empty or len(class_df) < 5:
        print(f"    ⚠️ Not enough data for {class_name} body length plot")
        return

    # Calculate statistics
    stats = class_df.groupby('date').agg({
        'body_length_px': ['mean', 'std', 'count']
    }).reset_index()
    stats.columns = ['date', 'mean_px', 'std_px', 'n']
    stats = stats.sort_values('date', key=lambda x: [parse_date_key(d) for d in x])

    # Convert to mm
    stats['mean_mm'] = stats['mean_px'] * PIXEL_TO_MM
    stats['std_mm'] = stats['std_px'].fillna(0) * PIXEL_TO_MM

    # Plot
    plt.figure(figsize=(12, 6))
    x = range(len(stats))

    plt.errorbar(x, stats['mean_mm'], yerr=stats['std_mm'],
                 fmt='o-', capsize=5, color='darkgreen', markersize=10,
                 linewidth=2, elinewidth=1.5)

    # Add mean values and n on the plot
    for i, (xi, row) in enumerate(zip(x, stats.itertuples())):
        # Mean value above point
        plt.annotate(f'{row.mean_mm:.2f}',
                    xy=(xi, row.mean_mm + row.std_mm + 0.05),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        # n value below point
        plt.annotate(f'n={row.n}',
                    xy=(xi, row.mean_mm - row.std_mm - 0.05),
                    ha='center', va='top', fontsize=8, color='gray')

    plt.xticks(x, stats['date'], rotation=45, ha='right', fontsize=11)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Mean Body Length (mm)', fontsize=12)
    plt.title(f'Average Body Length per Date - {class_name} Shape', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / f'avg_length_per_date_{class_name.lower()}.png', dpi=150)
    plt.close()

    # Save statistics
    stats.to_csv(figs_dir / f'length_statistics_{class_name.lower()}.csv', index=False)
    print(f"    ✓ {class_name} body length plot saved")


def _plot_body_length_combined(preds_df, figs_dir):
    """Plot average body length per date for OK + Great combined."""
    good_df = preds_df[(preds_df['predicted_class'] >= 1) & (preds_df['body_length_px'] > 0)]

    if good_df.empty or len(good_df) < 5:
        print(f"    ⚠️ Not enough data for combined body length plot")
        return

    # Calculate statistics
    stats = good_df.groupby('date').agg({
        'body_length_px': ['mean', 'std', 'count']
    }).reset_index()
    stats.columns = ['date', 'mean_px', 'std_px', 'n']
    stats = stats.sort_values('date', key=lambda x: [parse_date_key(d) for d in x])

    stats['mean_mm'] = stats['mean_px'] * PIXEL_TO_MM
    stats['std_mm'] = stats['std_px'].fillna(0) * PIXEL_TO_MM

    # Plot
    plt.figure(figsize=(12, 6))
    x = range(len(stats))

    plt.errorbar(x, stats['mean_mm'], yerr=stats['std_mm'],
                 fmt='s-', capsize=5, color='purple', markersize=10,
                 linewidth=2, elinewidth=1.5)

    # Add mean values and n
    for i, (xi, row) in enumerate(zip(x, stats.itertuples())):
        plt.annotate(f'{row.mean_mm:.2f}',
                    xy=(xi, row.mean_mm + row.std_mm + 0.05),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.annotate(f'n={row.n}',
                    xy=(xi, row.mean_mm - row.std_mm - 0.05),
                    ha='center', va='top', fontsize=8, color='gray')

    plt.xticks(x, stats['date'], rotation=45, ha='right', fontsize=11)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Mean Body Length (mm)', fontsize=12)
    plt.title('Average Body Length per Date - OK + Great Combined', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs_dir / 'avg_length_per_date_combined.png', dpi=150)
    plt.close()

    stats.to_csv(figs_dir / 'length_statistics_combined.csv', index=False)
    print(f"    ✓ Combined body length plot saved")


# =================== Main ===================

def main():
    print("\n" + "=" * 70)
    print("LARVA SHAPE CLASSIFIER - 3-CLASS MODEL (IMPROVED)")
    print("Classes: 0=Bad, 1=OK, 2=Great")
    print("Training on valid larvae only, excluding date 18.10")
    print("Using macro F1 for model selection")
    print("=" * 70)

    # Load labels
    if not LABELS_FILE.exists():
        print(f"❌ Labels file not found: {LABELS_FILE}")
        return

    labels_df = pd.read_excel(LABELS_FILE)
    print(f"✓ Loaded {len(labels_df)} labels")

    # Show distribution before filtering
    print(f"  Shape score distribution: {labels_df['shape_score'].value_counts().to_dict()}")
    print(f"  Valid larva distribution: {labels_df['is_valid_larva'].value_counts().to_dict()}")

    # Count valid larvae with shape scores
    valid_labels = labels_df[labels_df['is_valid_larva'] == 1]
    print(f"  Valid larvae shape scores: {valid_labels['shape_score'].value_counts().to_dict()}")

    # Build dataset (only valid larvae, excluding 18.10)
    X, y, meta = build_dataset(labels_df)

    if len(X) == 0:
        print("❌ No data to train on")
        return

    # Train model
    model, model_type, feature_names = train_multiclass_model(X, y, OUTPUT_DIR)

    if model is None:
        print("❌ Failed to train model")
        return

    # Collect all larvae (excluding 18.10)
    all_larvae = collect_all_larvae()
    print(f"\n✓ Found {len(all_larvae)} larvae to process (excluding 18.10)")

    # Predict
    preds_df = predict_all(all_larvae, model, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    for class_val in [0, 1, 2]:
        count = len(preds_df[preds_df['predicted_class'] == class_val])
        high_conf = len(preds_df[(preds_df['predicted_class'] == class_val) &
                                  (preds_df['confidence'] >= HIGH_CONFIDENCE_THRESHOLD)])
        print(f"  {CLASS_NAMES[class_val]}: {count} total, {high_conf} high-confidence")

    # Export results
    export_results(preds_df, OUTPUT_DIR)

    # Visualizations
    create_visualizations(preds_df, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
