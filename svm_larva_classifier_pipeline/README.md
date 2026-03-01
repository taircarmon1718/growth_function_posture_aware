# SVM Larva Classifier Pipeline

Complete machine learning pipeline for automated larva quality classification using Support Vector Machines (SVM).

## Overview

This pipeline trains an SVM classifier on manually labeled larvae and applies it to all larvae in the dataset to predict validity with confidence scores.

## Pipeline Steps

### Step 1: Feature Extraction
Extracts numerical features from each labeled larva:

**Image-based features:**
- Area (pixel count)
- Bounding box dimensions (width, height)
- Aspect ratio
- Mean intensity
- Standard deviation intensity
- Perimeter
- Compactness
- Solidity

**Shape features:**
- Skeleton length approximation
- Curvature proxy
- Morphological properties

**Morphometric features (when available):**
- Body length (CL)
- Major axis length (TL proxy)
- Minor axis length
- Eccentricity
- Curvature ratio
- Mean width

### Step 2: Model Training
- Splits data: 80% train, 20% test (stratified)
- Standardizes features using StandardScaler
- Performs GridSearchCV with cross-validation:
  - `C`: [0.1, 1, 10, 100]
  - `gamma`: ['scale', 0.1, 0.01]
  - `kernel`: RBF
- Evaluates using accuracy, precision, recall, F1-score
- Saves confusion matrix visualization

### Step 3: Prediction
- Applies trained model to ALL larvae in `analysis_full/`
- Generates predictions with confidence scores
- Saves comprehensive predictions Excel file

### Step 4: High Confidence Filtering
- Filters larvae with:
  - `predicted_is_larva == 1`
  - `confidence_score >= 0.90`
- Copies high-confidence larvae to organized folder structure by date

### Step 5: Aggregation
- Computes per-date statistics for high-confidence larvae:
  - Number of larvae
  - Average body length (CL)
  - Average major axis (TL)
  - Average CL/TL ratio
  - Standard deviations

### Step 6: Visualization
Creates publication-quality plots:
- Histogram of prediction confidence scores
- Bar chart of valid larvae counts per date
- Line plot of average body length per date (with error bars)

## Requirements

```bash
pip install numpy pandas opencv-python scikit-learn matplotlib seaborn joblib openpyxl
```

## Usage

### Run the complete pipeline:

```bash
cd /Users/taircarmon/Desktop/growth_function_posture_aware
python3 svm_larva_classifier_pipeline/run_svm_pipeline.py
```

### Prerequisites:

1. Labeled data file must exist:
   ```
   analysis_full/larva_quality_labels.xlsx
   ```

2. Labeled larvae images must be in:
   ```
   analysis_full/<date>/<image_name>/larvae_reports/<larva_filename>
   ```

3. Morphometric data (optional but recommended):
   ```
   analysis_full/<date>/<image_name>/morphometrics.csv
   ```

## Output Structure

```
svm_larva_classifier_pipeline/
├── run_svm_pipeline.py                    # Main pipeline script
├── svm_model.pkl                          # Trained SVM model
├── scaler.pkl                             # Feature scaler
├── classification_report.txt              # Evaluation metrics
├── predictions_all_larvae.xlsx            # All predictions
├── aggregate_length_per_date.xlsx         # Per-date statistics
│
├── figures/
│   ├── confusion_matrix.png              # Model evaluation
│   ├── confidence_histogram.png          # Confidence distribution
│   ├── valid_larvae_per_date.png        # Count per date
│   └── avg_length_per_date.png          # Length trends
│
└── high_confidence/
    ├── 19.10/
    │   ├── larva_001.png
    │   ├── larva_003.png
    │   └── ...
    ├── 20.10/
    └── ...
```

## Output Files

### 1. Model Files

**svm_model.pkl**
- Trained SVM classifier (scikit-learn SVC)
- Can be loaded for future predictions

**scaler.pkl**
- Fitted StandardScaler for feature normalization
- Must be used before making predictions

### 2. Evaluation Report

**classification_report.txt**
```
SVM Larva Classifier - Evaluation Report
======================================================================

Best parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
Best CV F1 score: 0.9234
Test Accuracy: 0.9150

Classification Report:
              precision    recall  f1-score   support

     Invalid       0.88      0.85      0.86        20
       Valid       0.93      0.94      0.94        50

    accuracy                           0.92        70
   macro avg       0.91      0.90      0.90        70
weighted avg       0.91      0.92      0.91        70
```

### 3. Predictions File

**predictions_all_larvae.xlsx**

| date | image_name | larva_filename | predicted_is_larva | confidence_score |
|------|-----------|----------------|-------------------|------------------|
| 19.10 | IMG_3788 | larva_001.png | 1 | 0.95 |
| 19.10 | IMG_3788 | larva_002.png | 1 | 0.87 |
| 20.10 | IMG_4023 | larva_003.png | 0 | 0.92 |

**Columns:**
- `date`: Date folder
- `image_name`: Image folder name
- `larva_filename`: Larva image filename
- `predicted_is_larva`: 0 (invalid) or 1 (valid)
- `confidence_score`: Probability [0.0, 1.0]

### 4. Aggregate Statistics

**aggregate_length_per_date.xlsx**

| date | num_larvae | avg_body_length | std_body_length | avg_major_axis | avg_cl_tl_ratio |
|------|-----------|----------------|----------------|---------------|----------------|
| 19.10 | 145 | 234.5 | 45.2 | 180.3 | 1.30 |
| 20.10 | 167 | 241.8 | 38.9 | 185.7 | 1.30 |

**Columns:**
- `date`: Date folder
- `num_larvae`: Count of high-confidence valid larvae
- `avg_body_length`: Mean body length (pixels)
- `std_body_length`: Standard deviation
- `avg_major_axis`: Mean major axis length
- `std_major_axis`: Standard deviation
- `avg_minor_axis`: Mean minor axis length
- `std_minor_axis`: Standard deviation
- `avg_width`: Mean larva width
- `std_width`: Standard deviation
- `avg_cl_tl_ratio`: Average CL/TL ratio

### 5. High Confidence Larvae

**high_confidence/<date>/**
- Organized by date folder
- Contains only larvae with:
  - Predicted as valid (predicted_is_larva = 1)
  - High confidence (confidence_score >= 0.90)

### 6. Visualizations

**figures/confusion_matrix.png**
- 2x2 heatmap showing model performance
- True positives, false positives, true negatives, false negatives

**figures/confidence_histogram.png**
- Distribution of confidence scores across all predictions
- Red line indicates high-confidence threshold

**figures/valid_larvae_per_date.png**
- Bar chart showing count of valid larvae per date
- Useful for temporal analysis

**figures/avg_length_per_date.png**
- Line plot with error bars
- Shows average body length trends over time
- Only includes high-confidence larvae

## Configuration

Edit the script header to adjust parameters:

```python
# Parameters
RANDOM_STATE = 42                        # Reproducibility seed
TEST_SIZE = 0.2                          # Test split ratio
HIGH_CONFIDENCE_THRESHOLD = 0.90         # Confidence cutoff
```

## Feature Engineering

The pipeline extracts features in three categories:

### Image Features (Always Available)
Extracted directly from the larva report images using OpenCV:
- Geometric properties from binary mask
- Intensity statistics from grayscale image
- Shape descriptors from contours

### Morphometric Features (Optional)
Loaded from pre-computed CSV files if available:
- Body length, width measurements
- Axis lengths and eccentricity
- Curvature metrics

### Fallback Strategy
If morphometric data is unavailable:
- Uses default values (0 or 1.0)
- Model still functions with image features alone
- Slightly reduced accuracy expected

## Model Details

### Algorithm: Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function)
- **Hyperparameters:** Optimized via GridSearchCV
- **Probability estimates:** Enabled for confidence scores

### Training Strategy
- **Cross-validation:** 5-fold stratified CV
- **Optimization metric:** F1-score
- **Stratification:** Maintains class balance in splits

### Feature Scaling
- **Method:** StandardScaler (zero mean, unit variance)
- **Applied to:** All numerical features
- **Reason:** SVM performance depends on scaled features

## Performance Metrics

### Accuracy
Overall correctness: `(TP + TN) / Total`

### Precision
How many predicted valids are actually valid: `TP / (TP + FP)`

### Recall
How many actual valids are detected: `TP / (TP + FN)`

### F1-Score
Harmonic mean of precision and recall: `2 * (P * R) / (P + R)`

### Confusion Matrix
- **True Positive (TP):** Valid larvae correctly identified
- **False Positive (FP):** Invalid larvae misclassified as valid
- **True Negative (TN):** Invalid larvae correctly identified
- **False Negative (FN):** Valid larvae misclassified as invalid

## Expected Performance

Based on typical bioimage classification tasks:

- **Accuracy:** 85-95%
- **Precision:** 88-96%
- **Recall:** 82-94%
- **F1-Score:** 85-95%

Performance depends on:
- Quality of manual labels
- Feature informativeness
- Class balance
- Dataset size

## Troubleshooting

### Issue: "Labels file not found"
**Solution:** Run the interactive labeling tool first to create labels:
```bash
python3 interactive_labeling_tool.py
```

### Issue: Low model performance
**Possible causes:**
- Insufficient labeled samples (need 50+ per class)
- Inconsistent labeling criteria
- Poor feature quality

**Solutions:**
- Label more diverse samples
- Review labeling guidelines
- Check if morphometric features are being extracted

### Issue: Feature extraction fails
**Check:**
- Larva images exist in correct locations
- Images are readable (not corrupted)
- Image format is correct (PNG)

### Issue: Memory errors
**Solutions:**
- Process in smaller batches
- Reduce image resolution
- Close other applications

## Extending the Pipeline

### Add new features:

```python
# In extract_image_features() function
features['new_feature'] = calculate_new_feature(mask)
```

### Change model:

```python
# Replace SVM with Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=RANDOM_STATE)
```

### Adjust confidence threshold:

```python
HIGH_CONFIDENCE_THRESHOLD = 0.85  # Lower threshold = more larvae
```

## Reproducibility

The pipeline ensures reproducibility via:
- Fixed `random_state=42` for all random operations
- Saved model and scaler objects
- Deterministic train/test split
- Recorded hyperparameters

To reproduce results:
1. Use same labeled dataset
2. Use same random seed
3. Use same scikit-learn version

## Citation

If you use this pipeline in research, please cite:

```
SVM Larva Classifier Pipeline
Automated quality classification of larva microscopy images
2026
```

## License

Research use only. Modify as needed for your project.

## Support

For issues or questions:
1. Check this README
2. Review code comments
3. Verify input data format
4. Check console output for warnings

## Version History

- **v1.0** (2026-02-28): Initial release
  - Complete pipeline implementation
  - GridSearchCV hyperparameter tuning
  - High-confidence filtering
  - Comprehensive visualizations

---

**Ready to run!** Execute `python3 svm_larva_classifier_pipeline/run_svm_pipeline.py`

