# SVM Shape Score Multi-Class Classifier Pipeline

## Overview

This pipeline creates **TWO separate SVM classifiers**, one for each shape quality level:

1. **Shape Score 2 Classifier** - Predicts "Great Shape" specimens (score=2)
2. **Shape Score 1 Classifier** - Predicts "OK Shape" specimens (score=1)

Each classifier has its own:
- Trained SVM model
- Predictions file
- High-confidence folder structure
- Aggregate statistics
- Visualizations

## Quick Start

```bash
cd /Users/taircarmon/Desktop/growth_function_posture_aware
python3 run_shape_classifiers.py
```

## What It Does

The script automatically:

### For Shape Score 2 (Great Shape):
1. ✅ Trains SVM to predict perfect specimens (straight, extended)
2. ✅ Creates output folder: `svm_shape_2_classifier/`
3. ✅ Predicts all larvae for "great shape"
4. ✅ Copies high-confidence great shapes to organized folders
5. ✅ Generates statistics and visualizations

### For Shape Score 1 (OK Shape):
1. ✅ Trains SVM to predict acceptable specimens (slight curvature)
2. ✅ Creates output folder: `svm_shape_1_classifier/`
3. ✅ Predicts all larvae for "ok shape"
4. ✅ Copies high-confidence ok shapes to organized folders
5. ✅ Generates statistics and visualizations

## Output Structure

```
growth_function_posture_aware/
│
├── svm_shape_2_classifier/              # Great Shape (Score 2)
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── classification_report.txt
│   ├── predictions_all_larvae.xlsx
│   ├── aggregate_length_per_date.xlsx
│   │
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── confidence_histogram.png
│   │   ├── count_per_date.png
│   │   └── avg_length_per_date.png
│   │
│   └── high_confidence/
│       ├── 19.10/
│       ├── 20.10/
│       └── ...
│
└── svm_shape_1_classifier/              # OK Shape (Score 1)
    ├── svm_model.pkl
    ├── scaler.pkl
    ├── classification_report.txt
    ├── predictions_all_larvae.xlsx
    ├── aggregate_length_per_date.xlsx
    │
    ├── figures/
    │   ├── confusion_matrix.png
    │   ├── confidence_histogram.png
    │   ├── count_per_date.png
    │   └── avg_length_per_date.png
    │
    └── high_confidence/
        ├── 19.10/
        ├── 20.10/
        └── ...
```

## Training Strategy

### Shape Score 2 Classifier:
- **Positive class**: Larvae with shape_score = 2 (Great)
- **Negative class**: All other valid larvae (shape_score = 0 or 1)
- **Goal**: Identify perfect specimens for primary analysis

### Shape Score 1 Classifier:
- **Positive class**: Larvae with shape_score = 1 (OK)
- **Negative class**: All other valid larvae (shape_score = 0 or 2)
- **Goal**: Identify acceptable specimens for supplementary data

## Output Files

### Each Classifier Folder Contains:

#### 1. Model Files
- `svm_model.pkl` - Trained SVM classifier
- `scaler.pkl` - Feature standardizer

#### 2. Predictions Excel
- `predictions_all_larvae.xlsx`

**Columns:**
- `date` - Date folder
- `image_name` - Image folder
- `larva_filename` - Larva file
- `predicted_shape_2` or `predicted_shape_1` - Binary prediction (0/1)
- `confidence_score` - Probability [0.0, 1.0]

#### 3. Aggregate Statistics
- `aggregate_length_per_date.xlsx`

Per-date statistics for high-confidence predictions:
- Number of larvae
- Average body length
- Average major axis
- CL/TL ratios
- Standard deviations

#### 4. Classification Report
- `classification_report.txt`

Contains:
- Best hyperparameters
- Cross-validation scores
- Test set metrics (accuracy, precision, recall, F1)

#### 5. Visualizations (in `figures/`)

**confusion_matrix.png** - Model performance heatmap

**confidence_histogram.png** - Distribution of prediction confidence

**count_per_date.png** - Bar chart of predicted count per date

**avg_length_per_date.png** - Line plot of average body length trends

#### 6. High-Confidence Larvae (in `high_confidence/<date>/`)

Only larvae with:
- Predicted as target shape (predicted = 1)
- High confidence (confidence ≥ 0.90)

Organized by date folders for easy access.

## Use Cases

### 1. Primary Analysis Dataset
Use Shape 2 high-confidence larvae for primary measurements:

```python
import pandas as pd
df = pd.read_excel('svm_shape_2_classifier/predictions_all_larvae.xlsx')

# Filter high-confidence perfect specimens
perfect = df[(df['predicted_shape_2'] == 1) & (df['confidence_score'] >= 0.90)]
print(f"Perfect specimens: {len(perfect)}")

# These are in: svm_shape_2_classifier/high_confidence/<date>/
```

### 2. Supplementary Dataset
Use Shape 1 for additional data:

```python
df = pd.read_excel('svm_shape_1_classifier/predictions_all_larvae.xlsx')

# Acceptable specimens
acceptable = df[(df['predicted_shape_1'] == 1) & (df['confidence_score'] >= 0.80)]
print(f"Acceptable specimens: {len(acceptable)}")
```

### 3. Quality Distribution Analysis
Compare distributions:

```python
# Load both
df2 = pd.read_excel('svm_shape_2_classifier/predictions_all_larvae.xlsx')
df1 = pd.read_excel('svm_shape_1_classifier/predictions_all_larvae.xlsx')

# Count by date
shape2_per_date = df2[df2['predicted_shape_2'] == 1].groupby('date').size()
shape1_per_date = df1[df1['predicted_shape_1'] == 1].groupby('date').size()

# Compare quality over time
import matplotlib.pyplot as plt
plt.plot(shape2_per_date.index, shape2_per_date.values, label='Great Shape')
plt.plot(shape1_per_date.index, shape1_per_date.values, label='OK Shape')
plt.legend()
plt.show()
```

### 4. Temporal Analysis
Analyze how shape quality changes over dates:

```python
# Load aggregate stats
stats2 = pd.read_excel('svm_shape_2_classifier/aggregate_length_per_date.xlsx')
stats1 = pd.read_excel('svm_shape_1_classifier/aggregate_length_per_date.xlsx')

# Compare average lengths
print("Great Shape average length:", stats2['avg_body_length'].mean())
print("OK Shape average length:", stats1['avg_body_length'].mean())
```

## Configuration

Edit lines 33-48 in `run_shape_classifiers.py`:

```python
# Parameters
RANDOM_STATE = 42                    # Random seed
TEST_SIZE = 0.2                      # Train/test split
HIGH_CONFIDENCE_THRESHOLD = 0.90     # Confidence cutoff

# Shape targets - can add more if needed
SHAPE_TARGETS = {
    'shape_2': {
        'name': 'Great Shape (Score 2)',
        'target_value': 2,
        'folder': 'svm_shape_2_classifier',
        'description': 'Perfect specimens'
    },
    'shape_1': {
        'name': 'OK Shape (Score 1)',
        'target_value': 1,
        'folder': 'svm_shape_1_classifier',
        'description': 'Acceptable specimens'
    }
}
```

## Expected Runtime

For 600 labeled larvae and ~4000 total larvae:

- **Shape 2 Classifier**: ~10-15 minutes
  - Feature extraction: 2-3 min
  - Training: 3-5 min
  - Prediction: 5-7 min

- **Shape 1 Classifier**: ~10-15 minutes
  - Same breakdown

**Total runtime: ~20-30 minutes** for both classifiers

## Expected Performance

### Shape Score 2 (Great Shape):
- Usually more rare, might have fewer positive samples
- Expected accuracy: 80-92%
- High precision important to avoid false positives

### Shape Score 1 (OK Shape):
- Usually more common, balanced dataset
- Expected accuracy: 82-94%
- Good balance of precision and recall

## Advantages of Separate Classifiers

### 1. Specialized Models
Each model learns specific characteristics:
- Shape 2 model: learns what makes a "perfect" specimen
- Shape 1 model: learns what makes an "acceptable" specimen

### 2. Independent Predictions
Can use different confidence thresholds:
```python
# More conservative for Shape 2
shape2_high = df2[df2['confidence_score'] >= 0.95]

# More permissive for Shape 1
shape1_medium = df1[df1['confidence_score'] >= 0.85]
```

### 3. Separate Analysis Pipelines
- Primary analysis: Use Shape 2 high-confidence only
- Validation analysis: Use Shape 1 to verify trends
- Combined analysis: Pool both for larger sample size

### 4. Quality Control
Cross-check predictions:
```python
# Find larvae predicted as both (potential issues)
both = set(df2[df2['predicted_shape_2']==1]['larva_filename']) & \
       set(df1[df1['predicted_shape_1']==1]['larva_filename'])
```

## Comparison with Original Pipeline

| Feature | Original | Shape Multi-Class |
|---------|----------|-------------------|
| Predicts | Valid vs Invalid | Shape 2 vs Other, Shape 1 vs Other |
| Output folders | 1 | 2 (separate) |
| Models | 1 | 2 (specialized) |
| Use case | Quality screening | Quality-based selection |

## Troubleshooting

### "Not enough samples for shape_score X"

Check label distribution:
```python
df = pd.read_excel('analysis_full/larva_quality_labels.xlsx')
print(df['shape_score'].value_counts())
```

Need at least 20-30 samples per shape score for good training.

### Low performance on one classifier

This is normal if:
- One shape score is rare (class imbalance)
- One shape score has ambiguous features

Solutions:
- Label more samples
- Adjust confidence threshold
- Use different evaluation metric (F1 vs accuracy)

## Next Steps

After running both pipelines:

1. **Review both classification reports**
   ```bash
   cat svm_shape_2_classifier/classification_report.txt
   cat svm_shape_1_classifier/classification_report.txt
   ```

2. **Compare high-confidence counts**
   ```bash
   ls svm_shape_2_classifier/high_confidence/*/*.png | wc -l
   ls svm_shape_1_classifier/high_confidence/*/*.png | wc -l
   ```

3. **Analyze predictions**
   - Open Excel files to review predictions
   - Compare confidence distributions
   - Check temporal patterns

4. **Use for downstream analysis**
   - Select appropriate quality tier for your study
   - Filter by confidence as needed
   - Combine or separate based on research question

---

**Ready to run!** Execute:
```bash
python3 run_shape_classifiers.py
```

This will create both classifiers and all outputs automatically!

