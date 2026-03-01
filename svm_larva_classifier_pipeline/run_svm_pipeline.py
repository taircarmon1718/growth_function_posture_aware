#!/usr/bin/env python3
"""
SVM Larva Classifier Pipeline
===============================
Complete pipeline for training SVM classifier on labeled larvae
and predicting validity of all larvae in the dataset.

Steps:
1. Feature extraction from labeled larvae
2. Train SVM with GridSearchCV
3. Predict all larvae (including unlabeled)
4. Copy high-confidence predictions
5. Aggregate length statistics per date
6. Visualizations

Author: Research Pipeline
Date: 2026-02-28
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================
ROOT_DIR = Path(__file__).parent.parent  # Go up one level from pipeline folder
ANALYSIS_DIR = ROOT_DIR / "analysis_full"
LABELS_FILE = ANALYSIS_DIR / "larva_quality_labels.xlsx"

# Pipeline output directory
PIPELINE_DIR = Path(__file__).parent  # The svm_larva_classifier_pipeline folder itself

# Output paths
MODEL_FILE = PIPELINE_DIR / "svm_model.pkl"
SCALER_FILE = PIPELINE_DIR / "scaler.pkl"
REPORT_FILE = PIPELINE_DIR / "classification_report.txt"
CONFUSION_MATRIX_FILE = PIPELINE_DIR / "figures" / "confusion_matrix.png"
PREDICTIONS_FILE = PIPELINE_DIR / "predictions_all_larvae.xlsx"
AGGREGATE_FILE = PIPELINE_DIR / "aggregate_length_per_date.xlsx"
HIGH_CONF_DIR = PIPELINE_DIR / "high_confidence"
FIGURES_DIR = PIPELINE_DIR / "figures"

# Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_CONFIDENCE_THRESHOLD = 0.90

# Ensure directories exist
PIPELINE_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
HIGH_CONF_DIR.mkdir(exist_ok=True)

# ======================== FEATURE EXTRACTION ========================

def extract_image_features(img_path):
    """
    Extract numerical features from a larva image.

    Args:
        img_path: Path to larva image

    Returns:
        dict: Dictionary of features or None if failed
    """
    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # The larva report image has visual panels on the left
    # Extract only the visual portion (remove text panel if present)
    img_h, img_w = img.shape[:2]
    gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect visual content width
    visual_width = img_w
    for x in range(img_w - 1, img_w // 2, -1):
        if np.mean(gray_check[:, x]) > 20:
            visual_width = x + 1
            break

    if visual_width == img_w:
        visual_width = int(img_w * 0.75)

    img_visual = img[:, :visual_width]

    # Convert to grayscale for feature extraction
    if len(img_visual.shape) == 3:
        gray = cv2.cvtColor(img_visual, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_visual.copy()

    # The visual panels are stacked horizontally: [original | mask | overlay]
    # Extract the middle panel (mask) for shape features
    panel_width = visual_width // 3
    mask_panel = gray[:, panel_width:2*panel_width]

    # Threshold to get binary mask
    _, binary_mask = cv2.threshold(mask_panel, 50, 255, cv2.THRESH_BINARY)

    # Basic image features
    features = {}

    # Area (non-zero pixels in mask)
    features['area'] = np.count_nonzero(binary_mask)

    # Bounding box
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        features['bbox_width'] = w
        features['bbox_height'] = h
        features['aspect_ratio'] = w / h if h > 0 else 0

        # Perimeter and compactness
        perimeter = cv2.arcLength(cnt, True)
        features['perimeter'] = perimeter
        features['compactness'] = (4 * np.pi * features['area']) / (perimeter ** 2) if perimeter > 0 else 0

        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = features['area'] / hull_area if hull_area > 0 else 0
    else:
        features['bbox_width'] = 0
        features['bbox_height'] = 0
        features['aspect_ratio'] = 0
        features['perimeter'] = 0
        features['compactness'] = 0
        features['solidity'] = 0

    # Intensity features from original panel
    original_panel = gray[:, :panel_width]
    roi = original_panel[binary_mask > 0] if features['area'] > 0 else original_panel.flatten()
    if len(roi) > 0:
        features['mean_intensity'] = np.mean(roi)
        features['std_intensity'] = np.std(roi)
    else:
        features['mean_intensity'] = 0
        features['std_intensity'] = 0

    # Skeleton-based features
    if features['area'] > 10:
        # Simple skeleton approximation
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_mask, kernel, iterations=1)
        skeleton_approx = binary_mask - eroded
        features['skeleton_length'] = np.count_nonzero(skeleton_approx)

        # Curvature proxy: skeleton length / euclidean distance
        y_coords, x_coords = np.where(skeleton_approx > 0)
        if len(y_coords) > 0:
            p1 = (x_coords[0], y_coords[0])
            p2 = (x_coords[-1], y_coords[-1])
            euclidean_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            features['curvature_proxy'] = features['skeleton_length'] / euclidean_dist if euclidean_dist > 0 else 1.0
        else:
            features['curvature_proxy'] = 1.0
    else:
        features['skeleton_length'] = 0
        features['curvature_proxy'] = 1.0

    return features


def extract_morphometric_features(date, image_name, larva_filename):
    """
    Extract morphometric features from the morphometrics CSV file.

    Args:
        date: Date folder (already converted to string format)
        image_name: Image folder name
        larva_filename: Larva filename (e.g., larva_001.png)

    Returns:
        dict: Morphometric features or default values
    """
    # Try to load morphometrics CSV
    csv_path = ANALYSIS_DIR / str(date) / str(image_name) / "morphometrics.csv"

    features = {
        'body_length': 0,
        'major_axis': 0,
        'minor_axis': 0,
        'eccentricity': 0,
        'curvature_ratio': 1.0,
        'mean_width': 0
    }

    if not csv_path.exists():
        return features

    try:
        df = pd.read_csv(csv_path)

        # Extract larva ID from filename (e.g., larva_001.png -> 1)
        larva_id = int(larva_filename.replace('larva_', '').replace('.png', ''))

        # Find the row for this larva
        larva_row = df[df['larva_id'] == larva_id]

        if not larva_row.empty:
            features['body_length'] = larva_row['body_length'].values[0]
            features['major_axis'] = larva_row['major_axis'].values[0]
            features['minor_axis'] = larva_row['minor_axis'].values[0]
            features['eccentricity'] = larva_row['eccentricity'].values[0]
            features['curvature_ratio'] = larva_row['curvature_ratio'].values[0]
            features['mean_width'] = larva_row['mean_width'].values[0]
    except Exception as e:
        print(f"  Warning: Could not extract morphometrics for {larva_filename}: {e}")

    return features


def build_feature_dataset(labels_df):
    """
    Build feature dataset from labeled larvae.

    Args:
        labels_df: DataFrame with labels

    Returns:
        tuple: (X, y, feature_names, metadata_df)
    """
    print("\n" + "="*70)
    print("STEP 1: FEATURE EXTRACTION FROM LABELED LARVAE")
    print("="*70)

    feature_list = []
    labels_list = []
    metadata_list = []

    print(f"\nExtracting features from {len(labels_df)} labeled larvae...")

    for idx, row in labels_df.iterrows():
        # Convert date to string format (handles float dates like 19.1 -> "19.10")
        date_raw = row['date']
        if isinstance(date_raw, (float, int)):
            # Convert float like 19.1 to "19.10"
            date = f"{int(date_raw)}.{int(round((date_raw % 1) * 100)):02d}"
        else:
            date = str(date_raw)

        image_name = str(row['image_name'])
        larva_filename = str(row['larva_filename'])
        is_valid = row['is_valid_larva']

        # Path to larva image
        img_path = ANALYSIS_DIR / date / image_name / "larvae_reports" / larva_filename

        # Extract features
        img_features = extract_image_features(img_path)
        morph_features = extract_morphometric_features(date, image_name, larva_filename)

        if img_features is None:
            print(f"  Warning: Could not extract features from {img_path}")
            continue

        # Combine features
        combined_features = {**img_features, **morph_features}

        feature_list.append(combined_features)
        labels_list.append(is_valid)
        metadata_list.append({
            'date': date,
            'image_name': image_name,
            'larva_filename': larva_filename
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(labels_df)} larvae...")

    print(f"\n✓ Successfully extracted features from {len(feature_list)} larvae")

    # Convert to DataFrame
    X = pd.DataFrame(feature_list)
    y = np.array(labels_list)
    metadata_df = pd.DataFrame(metadata_list)

    feature_names = X.columns.tolist()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Label distribution: Valid={np.sum(y)}, Invalid={len(y)-np.sum(y)}")

    return X, y, feature_names, metadata_df


# ======================== MODEL TRAINING ========================

def train_svm_model(X, y):
    """
    Train SVM classifier with GridSearchCV.

    Args:
        X: Feature matrix
        y: Labels

    Returns:
        tuple: (best_model, scaler, X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*70)
    print("STEP 2: TRAIN SVM MODEL")
    print("="*70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Grid search
    print("\nPerforming GridSearchCV...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf']
    }

    svm = SVC(random_state=RANDOM_STATE, probability=True)
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = best_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Invalid', 'Valid'])
    print("\nClassification Report:")
    print(report)

    # Save report
    with open(REPORT_FILE, 'w') as f:
        f.write("SVM Larva Classifier - Evaluation Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Best parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV F1 score: {grid_search.best_score_:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\n✓ Classification report saved to: {REPORT_FILE}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    # Save model and scaler
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n✓ Model saved to: {MODEL_FILE}")
    print(f"✓ Scaler saved to: {SCALER_FILE}")

    return best_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test


def plot_confusion_matrix(cm):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Invalid', 'Valid'],
                yticklabels=['Invalid', 'Valid'])
    plt.title('Confusion Matrix - SVM Larva Classifier', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
    plt.close()

    print(f"✓ Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")


# ======================== PREDICTION ========================

def collect_all_larvae():
    """
    Collect all larvae from analysis_full directory.

    Returns:
        list: List of (date, image_name, larva_filename, img_path) tuples
    """
    print("\n" + "="*70)
    print("STEP 3: PREDICT ALL LARVAE")
    print("="*70)

    all_larvae = []

    # Find all date folders
    date_folders = sorted([
        d for d in ANALYSIS_DIR.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ])

    print(f"\nScanning {len(date_folders)} date folders...")

    for date_folder in date_folders:
        date_name = date_folder.name

        # Find all image folders
        image_folders = sorted([d for d in date_folder.iterdir() if d.is_dir()])

        for image_folder in image_folders:
            image_name = image_folder.name
            larvae_reports_dir = image_folder / "larvae_reports"

            if not larvae_reports_dir.exists():
                continue

            # Find all larva images
            larva_files = sorted([
                f for f in larvae_reports_dir.iterdir()
                if f.name.startswith("larva_") and f.name.endswith(".png")
            ])

            for larva_file in larva_files:
                all_larvae.append((date_name, image_name, larva_file.name, larva_file))

    print(f"✓ Found {len(all_larvae)} total larvae across all dates")

    return all_larvae


def predict_all_larvae(model, scaler, feature_names):
    """
    Predict validity for all larvae.

    Args:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        feature_names: List of feature names

    Returns:
        pd.DataFrame: Predictions dataframe
    """
    all_larvae = collect_all_larvae()

    predictions_list = []

    print(f"\nPredicting {len(all_larvae)} larvae...")

    for idx, (date, image_name, larva_filename, img_path) in enumerate(all_larvae):
        # Extract features
        img_features = extract_image_features(img_path)
        morph_features = extract_morphometric_features(date, image_name, larva_filename)

        if img_features is None:
            # Failed to extract, mark as invalid with low confidence
            predictions_list.append({
                'date': date,
                'image_name': image_name,
                'larva_filename': larva_filename,
                'predicted_is_larva': 0,
                'confidence_score': 0.0
            })
            continue

        # Combine features
        combined_features = {**img_features, **morph_features}

        # Create feature vector in correct order
        feature_vector = pd.DataFrame([combined_features])[feature_names]

        # Scale and predict
        feature_scaled = scaler.transform(feature_vector)
        prediction = model.predict(feature_scaled)[0]
        confidence = model.predict_proba(feature_scaled)[0][prediction]

        predictions_list.append({
            'date': date,
            'image_name': image_name,
            'larva_filename': larva_filename,
            'predicted_is_larva': int(prediction),
            'confidence_score': float(confidence)
        })

        if (idx + 1) % 100 == 0:
            print(f"  Predicted {idx + 1}/{len(all_larvae)} larvae...")

    predictions_df = pd.DataFrame(predictions_list)

    # Save predictions
    predictions_df.to_excel(PREDICTIONS_FILE, index=False)

    print(f"\n✓ Predictions saved to: {PREDICTIONS_FILE}")
    print(f"\nPrediction summary:")
    print(f"  Total larvae: {len(predictions_df)}")
    print(f"  Predicted valid: {predictions_df['predicted_is_larva'].sum()}")
    print(f"  Predicted invalid: {len(predictions_df) - predictions_df['predicted_is_larva'].sum()}")
    print(f"  Mean confidence: {predictions_df['confidence_score'].mean():.4f}")

    return predictions_df


# ======================== HIGH CONFIDENCE FILTERING ========================

def copy_high_confidence_larvae(predictions_df):
    """
    Copy high-confidence valid larvae to separate folder structure.

    Args:
        predictions_df: Predictions dataframe
    """
    print("\n" + "="*70)
    print("STEP 4: COPY HIGH CONFIDENCE LARVAE")
    print("="*70)

    # Filter high confidence valid larvae
    high_conf = predictions_df[
        (predictions_df['predicted_is_larva'] == 1) &
        (predictions_df['confidence_score'] >= HIGH_CONFIDENCE_THRESHOLD)
    ]

    print(f"\nHigh confidence threshold: {HIGH_CONFIDENCE_THRESHOLD}")
    print(f"High confidence larvae: {len(high_conf)}")

    copied_count = 0

    for date in high_conf['date'].unique():
        # Ensure date is a string
        date_str = str(date)
        date_dir = HIGH_CONF_DIR / date_str
        date_dir.mkdir(exist_ok=True)

        date_larvae = high_conf[high_conf['date'] == date]

        for _, row in date_larvae.iterrows():
            # Ensure all path components are strings
            src_path = ANALYSIS_DIR / str(row['date']) / str(row['image_name']) / "larvae_reports" / str(row['larva_filename'])

            if src_path.exists():
                dst_path = date_dir / str(row['larva_filename'])
                shutil.copy2(src_path, dst_path)
                copied_count += 1

    print(f"\n✓ Copied {copied_count} high-confidence larvae")
    print(f"✓ Organized by date in: {HIGH_CONF_DIR}")

    # Print per-date summary
    print("\nHigh-confidence larvae per date:")
    date_counts = high_conf['date'].value_counts().sort_index()
    for date, count in date_counts.items():
        print(f"  {date}: {count}")


# ======================== AGGREGATION ========================

def aggregate_length_per_date(predictions_df):
    """
    Aggregate length statistics per date for high-confidence larvae.

    Args:
        predictions_df: Predictions dataframe
    """
    print("\n" + "="*70)
    print("STEP 5: AGGREGATE LENGTH STATISTICS PER DATE")
    print("="*70)

    # Filter high confidence valid larvae
    high_conf = predictions_df[
        (predictions_df['predicted_is_larva'] == 1) &
        (predictions_df['confidence_score'] >= HIGH_CONFIDENCE_THRESHOLD)
    ]

    print(f"\nAggregating statistics for {len(high_conf)} high-confidence larvae...")

    # Extract morphometric data for these larvae
    morph_data = []

    for _, row in high_conf.iterrows():
        morph_features = extract_morphometric_features(
            row['date'], row['image_name'], row['larva_filename']
        )
        morph_data.append({
            'date': row['date'],
            'body_length': morph_features['body_length'],
            'major_axis': morph_features['major_axis'],
            'minor_axis': morph_features['minor_axis'],
            'mean_width': morph_features['mean_width']
        })

    morph_df = pd.DataFrame(morph_data)

    # Aggregate by date
    aggregated = morph_df.groupby('date').agg({
        'body_length': ['count', 'mean', 'std'],
        'major_axis': ['mean', 'std'],
        'minor_axis': ['mean', 'std'],
        'mean_width': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]

    # Rename for clarity
    aggregated = aggregated.rename(columns={
        'date': 'date',
        'body_length_count': 'num_larvae',
        'body_length_mean': 'avg_body_length',
        'body_length_std': 'std_body_length',
        'major_axis_mean': 'avg_major_axis',
        'major_axis_std': 'std_major_axis',
        'minor_axis_mean': 'avg_minor_axis',
        'minor_axis_std': 'std_minor_axis',
        'mean_width_mean': 'avg_width',
        'mean_width_std': 'std_width'
    })

    # Calculate CL/TL ratio (using major_axis as TL proxy)
    aggregated['avg_cl_tl_ratio'] = aggregated['avg_body_length'] / aggregated['avg_major_axis']

    # Save
    aggregated.to_excel(AGGREGATE_FILE, index=False)

    print(f"\n✓ Aggregate statistics saved to: {AGGREGATE_FILE}")
    print("\nSummary:")
    print(aggregated.to_string(index=False))


# ======================== VISUALIZATION ========================

def create_visualizations(predictions_df):
    """
    Create all visualization plots.

    Args:
        predictions_df: Predictions dataframe
    """
    print("\n" + "="*70)
    print("STEP 6: CREATE VISUALIZATIONS")
    print("="*70)

    # 1. Histogram of prediction confidence
    plt.figure(figsize=(10, 6))
    plt.hist(predictions_df['confidence_score'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Confidence Scores', fontsize=14, fontweight='bold')
    plt.axvline(HIGH_CONFIDENCE_THRESHOLD, color='r', linestyle='--',
                label=f'High Confidence Threshold ({HIGH_CONFIDENCE_THRESHOLD})')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    confidence_hist_path = FIGURES_DIR / "confidence_histogram.png"
    plt.savefig(confidence_hist_path, dpi=300)
    plt.close()
    print(f"\n✓ Confidence histogram saved to: {confidence_hist_path}")

    # 2. Bar chart: number of valid larvae per date
    valid_per_date = predictions_df[predictions_df['predicted_is_larva'] == 1].groupby('date').size()

    plt.figure(figsize=(12, 6))
    valid_per_date.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Valid Larvae', fontsize=12)
    plt.title('Number of Valid Larvae per Date', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    valid_bar_path = FIGURES_DIR / "valid_larvae_per_date.png"
    plt.savefig(valid_bar_path, dpi=300)
    plt.close()
    print(f"✓ Valid larvae bar chart saved to: {valid_bar_path}")

    # 3. Line plot: average body length per date (high confidence only)
    high_conf = predictions_df[
        (predictions_df['predicted_is_larva'] == 1) &
        (predictions_df['confidence_score'] >= HIGH_CONFIDENCE_THRESHOLD)
    ]

    # Extract lengths
    lengths_by_date = []
    for date in sorted(high_conf['date'].unique()):
        date_larvae = high_conf[high_conf['date'] == date]
        lengths = []
        for _, row in date_larvae.iterrows():
            morph = extract_morphometric_features(row['date'], row['image_name'], row['larva_filename'])
            if morph['body_length'] > 0:
                lengths.append(morph['body_length'])

        if lengths:
            lengths_by_date.append({
                'date': date,
                'avg_length': np.mean(lengths),
                'std_length': np.std(lengths)
            })

    if lengths_by_date:
        length_df = pd.DataFrame(lengths_by_date)

        plt.figure(figsize=(12, 6))
        plt.errorbar(range(len(length_df)), length_df['avg_length'],
                    yerr=length_df['std_length'], fmt='o-', capsize=5,
                    linewidth=2, markersize=8, color='darkgreen')
        plt.xticks(range(len(length_df)), length_df['date'], rotation=45, ha='right')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Body Length (pixels)', fontsize=12)
        plt.title('Average Body Length per Date (High Confidence Larvae)',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        length_line_path = FIGURES_DIR / "avg_length_per_date.png"
        plt.savefig(length_line_path, dpi=300)
        plt.close()
        print(f"✓ Average length line plot saved to: {length_line_path}")


# ======================== MAIN PIPELINE ========================

def main():
    """
    Main pipeline execution.
    """
    print("="*70)
    print("SVM LARVA CLASSIFIER PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {PIPELINE_DIR}")
    print("="*70)

    # Check if labels file exists
    if not LABELS_FILE.exists():
        print(f"\n❌ Error: Labels file not found: {LABELS_FILE}")
        print("Please run the interactive labeling tool first.")
        return

    # Load labels
    print(f"\nLoading labels from: {LABELS_FILE}")
    labels_df = pd.read_excel(LABELS_FILE)
    print(f"✓ Loaded {len(labels_df)} labeled larvae")

    # Step 1: Feature extraction
    X, y, feature_names, metadata_df = build_feature_dataset(labels_df)

    if len(X) == 0:
        print("\n❌ Error: No features extracted. Check your data.")
        return

    # Step 2: Train model
    model, scaler, X_train, X_test, y_train, y_test = train_svm_model(X, y)

    # Step 3: Predict all larvae
    predictions_df = predict_all_larvae(model, scaler, feature_names)

    # Step 4: Copy high confidence larvae
    copy_high_confidence_larvae(predictions_df)

    # Step 5: Aggregate statistics
    aggregate_length_per_date(predictions_df)

    # Step 6: Create visualizations
    create_visualizations(predictions_df)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll outputs saved in: {PIPELINE_DIR}")
    print("\nGenerated files:")
    print(f"  - Model: {MODEL_FILE.name}")
    print(f"  - Scaler: {SCALER_FILE.name}")
    print(f"  - Report: {REPORT_FILE.name}")
    print(f"  - Predictions: {PREDICTIONS_FILE.name}")
    print(f"  - Aggregated stats: {AGGREGATE_FILE.name}")
    print(f"  - High confidence larvae: {HIGH_CONF_DIR.name}/<date>/")
    print(f"  - Figures: {FIGURES_DIR.name}/")
    print("="*70)


if __name__ == "__main__":
    main()

