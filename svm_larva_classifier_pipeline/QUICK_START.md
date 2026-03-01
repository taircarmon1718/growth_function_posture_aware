# Quick Start Guide - SVM Larva Classifier Pipeline

## Prerequisites

1. **Labeled data must exist:**
   ```
   analysis_full/larva_quality_labels.xlsx
   ```
   Create this by running: `python3 interactive_labeling_tool.py`

2. **Python packages installed:**
   ```bash
   pip3 install -r svm_larva_classifier_pipeline/requirements.txt
   ```

## Run the Pipeline (One Command)

```bash
cd /Users/taircarmon/Desktop/growth_function_posture_aware
python3 svm_larva_classifier_pipeline/run_svm_pipeline.py
```

## What Happens

The pipeline will automatically:

1. ✅ Load your labeled larvae data
2. ✅ Extract image + morphometric features
3. ✅ Train SVM classifier with GridSearchCV
4. ✅ Evaluate model performance
5. ✅ Predict ALL larvae (including unlabeled)
6. ✅ Copy high-confidence valid larvae
7. ✅ Aggregate statistics per date
8. ✅ Create visualizations

## Expected Output

```
SVM LARVA CLASSIFIER PIPELINE
======================================================================

STEP 1: FEATURE EXTRACTION FROM LABELED LARVAE
======================================================================
Extracting features from 250 labeled larvae...
✓ Successfully extracted features from 248 larvae

Feature matrix shape: (248, 16)
Features: ['area', 'bbox_width', 'bbox_height', ...]
Label distribution: Valid=198, Invalid=50

STEP 2: TRAIN SVM MODEL
======================================================================
Train set: 198 samples
Test set: 50 samples

Standardizing features...
Performing GridSearchCV...
✓ Best parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
✓ Best cross-validation F1 score: 0.9234

Evaluating on test set...
✓ Test Accuracy: 0.9150

Classification Report:
              precision    recall  f1-score   support
     Invalid       0.88      0.85      0.86        20
       Valid       0.93      0.94      0.94        50

✓ Model saved to: svm_model.pkl
✓ Scaler saved to: scaler.pkl

STEP 3: PREDICT ALL LARVAE
======================================================================
Scanning 10 date folders...
✓ Found 3,842 total larvae across all dates

Predicting 3,842 larvae...
✓ Predictions saved to: predictions_all_larvae.xlsx

Prediction summary:
  Total larvae: 3,842
  Predicted valid: 3,120
  Predicted invalid: 722
  Mean confidence: 0.8745

STEP 4: COPY HIGH CONFIDENCE LARVAE
======================================================================
High confidence threshold: 0.90
High confidence larvae: 2,456

✓ Copied 2,456 high-confidence larvae
✓ Organized by date in: high_confidence/

STEP 5: AGGREGATE LENGTH STATISTICS PER DATE
======================================================================
Aggregating statistics for 2,456 high-confidence larvae...

✓ Aggregate statistics saved to: aggregate_length_per_date.xlsx

STEP 6: CREATE VISUALIZATIONS
======================================================================
✓ Confidence histogram saved
✓ Valid larvae bar chart saved
✓ Average length line plot saved

PIPELINE COMPLETE
======================================================================
All outputs saved in: svm_larva_classifier_pipeline/
```

## View Results

### 1. Model Performance
```bash
cat svm_larva_classifier_pipeline/classification_report.txt
```

### 2. All Predictions
Open in Excel:
```
svm_larva_classifier_pipeline/predictions_all_larvae.xlsx
```

### 3. Per-Date Statistics
Open in Excel:
```
svm_larva_classifier_pipeline/aggregate_length_per_date.xlsx
```

### 4. Visualizations
```
svm_larva_classifier_pipeline/figures/
├── confusion_matrix.png
├── confidence_histogram.png
├── valid_larvae_per_date.png
└── avg_length_per_date.png
```

### 5. High-Confidence Larvae
```
svm_larva_classifier_pipeline/high_confidence/
├── 19.10/
├── 20.10/
├── 21.10/
└── ...
```

## Common Use Cases

### Filter by Confidence
```python
import pandas as pd

df = pd.read_excel('svm_larva_classifier_pipeline/predictions_all_larvae.xlsx')

# Only very high confidence (>95%)
very_high = df[df['confidence_score'] > 0.95]

# Valid larvae with medium-high confidence (80-90%)
medium = df[(df['predicted_is_larva'] == 1) & 
            (df['confidence_score'] >= 0.80) & 
            (df['confidence_score'] < 0.90)]
```

### Analyze Per Date
```python
stats = pd.read_excel('svm_larva_classifier_pipeline/aggregate_length_per_date.xlsx')

print(stats[['date', 'num_larvae', 'avg_body_length']])
```

### Load Model for New Predictions
```python
import joblib

model = joblib.load('svm_larva_classifier_pipeline/svm_model.pkl')
scaler = joblib.load('svm_larva_classifier_pipeline/scaler.pkl')

# Use on new larvae features
# predictions = model.predict(scaler.transform(new_features))
```

## Troubleshooting

### "Labels file not found"
Run the labeling tool first:
```bash
python3 interactive_labeling_tool.py
```

### Low performance (<80% accuracy)
- Need more labeled samples (aim for 100+)
- Check labeling consistency
- Review difficult cases

### Missing dependencies
```bash
pip3 install numpy pandas opencv-python scikit-learn matplotlib seaborn joblib openpyxl
```

## Configuration

To change settings, edit `run_svm_pipeline.py`:

```python
# Line 33-35
RANDOM_STATE = 42                    # Change for different split
TEST_SIZE = 0.2                      # Change train/test ratio
HIGH_CONFIDENCE_THRESHOLD = 0.90     # Lower = more larvae included
```

## Time Estimates

- Feature extraction: ~1-2 minutes for 250 labeled larvae
- Model training: ~2-5 minutes (depends on GridSearchCV)
- Prediction: ~5-10 minutes for 3,000+ larvae
- **Total pipeline: ~10-20 minutes**

## Next Steps

After running the pipeline:

1. **Review model performance** in classification_report.txt
2. **Examine confusion matrix** to understand errors
3. **Check confidence distribution** histogram
4. **Analyze high-confidence larvae** in organized folders
5. **Use aggregate statistics** for temporal analysis
6. **Filter predictions** by confidence for different analyses

## Advanced Usage

### Re-train with Different Parameters
Edit the GridSearchCV parameters in `run_svm_pipeline.py`:

```python
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],      # Add more values
    'gamma': ['scale', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly']           # Try different kernels
}
```

### Adjust Confidence Threshold
```python
HIGH_CONFIDENCE_THRESHOLD = 0.85  # More inclusive
HIGH_CONFIDENCE_THRESHOLD = 0.95  # More conservative
```

### Add More Features
In `extract_image_features()` function, add custom features:

```python
# Your custom feature
features['my_custom_feature'] = calculate_custom_metric(mask)
```

---

**Ready to run!** Execute the pipeline and get automated larva classification in minutes.

