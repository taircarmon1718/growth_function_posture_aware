# ✅ FIXED: Empty High-Confidence Dataset Error

## Error Fixed

```
KeyError: 'date'
Traceback in aggregate_statistics() when morph_df.groupby('date')
```

## Root Cause

When there are **0 high-confidence larvae** (confidence >= 0.90):
- The `high_conf` DataFrame is empty
- The `morph_data` list is empty
- The `morph_df` DataFrame has no rows and no columns
- Calling `morph_df.groupby('date')` fails because 'date' column doesn't exist

This happens when:
- Model predictions have low confidence (all < 0.90)
- Very few samples of target shape score in training data
- Model is uncertain about the predictions

## Solution Applied

### 1. Fixed `aggregate_statistics()` Method (lines ~527-548)

Added check for empty high-confidence predictions:

```python
# Check if there are any high-confidence predictions
if len(high_conf) == 0:
    print(f"\n⚠️  No high-confidence larvae found for {self.target_name}")
    print(f"   Skipping aggregation and saving empty file.")
    
    # Create empty DataFrame with correct columns
    empty_df = pd.DataFrame(columns=[
        'date', 'num_larvae', 'avg_body_length', 'std_body_length',
        'avg_major_axis', 'std_major_axis', 'avg_minor_axis', 'std_minor_axis',
        'avg_width', 'std_width', 'avg_cl_tl_ratio'
    ])
    empty_df.to_excel(self.aggregate_file, index=False)
    print(f"✓ Empty statistics file saved to: {self.aggregate_file}")
    return  # Exit early
```

**What this does:**
- Detects when high_conf is empty
- Saves an empty Excel file with proper column headers
- Returns early to avoid the groupby error

### 2. Fixed `create_visualizations()` Method (lines ~583-605)

Added multiple checks for empty datasets:

```python
# Check 1: For bar chart
predicted_target = predictions_df[predictions_df[target_col] == 1]

if len(predicted_target) == 0:
    print(f"⚠️  No predictions for {self.target_name}, skipping per-date visualizations")
else:
    # Create bar chart
    per_date = predicted_target.groupby('date').size()
    # ... plot code ...

# Check 2: For length trend
high_conf = predictions_df[...]

if len(high_conf) == 0:
    print(f"⚠️  No high-confidence predictions, skipping length trend plot")
    return

# Check 3: For valid length data
if lengths_by_date:
    # ... plot code ...
else:
    print(f"⚠️  No valid length data, skipping length trend plot")
```

**What this does:**
- Checks if there are any predictions before creating bar chart
- Checks if there are high-confidence predictions before length trend
- Skips plots gracefully with informative messages
- Always creates confidence histogram (uses all predictions)

## What Happens Now

### When High Confidence = 0:

**Step 4: Copy High Confidence**
```
High confidence threshold: 0.9
High confidence larvae: 0
✓ Copied 0 high-confidence larvae
```

**Step 5: Aggregate Statistics**
```
Aggregating for 0 high-confidence larvae...
⚠️  No high-confidence larvae found for Great Shape (Score 2)
   Skipping aggregation and saving empty file.
✓ Empty statistics file saved to: aggregate_length_per_date.xlsx
```

**Step 6: Visualizations**
```
✓ Confidence histogram: confidence_histogram.png
⚠️  No predictions for Great Shape (Score 2), skipping per-date visualizations
⚠️  No high-confidence predictions, skipping length trend plot
```

**Result:** Pipeline completes successfully instead of crashing!

## Why This Happens

### Common Scenarios:

1. **Insufficient Training Data**
   - Too few samples of target shape score
   - Model can't learn distinguishing features
   - Makes low-confidence predictions

2. **Class Imbalance**
   - Shape score 2 might be rare (e.g., only 10% of larvae)
   - Model biased toward more common classes
   - Conservative predictions

3. **Ambiguous Features**
   - Shape score 2 and 1 might have similar features
   - Hard to distinguish visually
   - Model uncertainty reflected in low confidence

4. **High Threshold**
   - Confidence threshold = 0.90 is very strict
   - Requires model to be 90%+ certain
   - Fewer predictions pass this threshold

## Solutions to Get More High-Confidence Predictions

### Option 1: Lower Confidence Threshold

Edit line 43 in `run_shape_classifiers.py`:
```python
HIGH_CONFIDENCE_THRESHOLD = 0.80  # Lower from 0.90 to 0.80
```

This will include more predictions (80-90% confidence range).

### Option 2: Label More Samples

Get more labeled examples of the target shape score:
```bash
python3 interactive_labeling_tool.py
```

Focus on labeling more Shape 2 specimens if that's the rare class.

### Option 3: Check Label Distribution

```python
import pandas as pd
df = pd.read_excel('analysis_full/larva_quality_labels.xlsx')
print(df['shape_score'].value_counts())
```

Ensure you have at least 30-50 samples per shape score.

### Option 4: Review Model Performance

Check the classification report:
```bash
cat svm_shape_2_classifier/classification_report.txt
```

If test accuracy is low (<80%), the model needs improvement:
- More training data
- Better features
- Different hyperparameters

## Files Modified

- `run_shape_classifiers.py`
  - Line ~527-548: `aggregate_statistics()` - added empty check
  - Line ~583-680: `create_visualizations()` - added multiple empty checks

## Output Files Created Even When Empty

Even with 0 high-confidence predictions, the pipeline now creates:

✅ `svm_shape_2_classifier/svm_model.pkl` - Trained model  
✅ `svm_shape_2_classifier/scaler.pkl` - Feature scaler  
✅ `svm_shape_2_classifier/classification_report.txt` - Performance report  
✅ `svm_shape_2_classifier/predictions_all_larvae.xlsx` - All predictions  
✅ `svm_shape_2_classifier/aggregate_length_per_date.xlsx` - Empty with headers  
✅ `svm_shape_2_classifier/figures/confusion_matrix.png` - Model evaluation  
✅ `svm_shape_2_classifier/figures/confidence_histogram.png` - Confidence dist  
❌ `svm_shape_2_classifier/figures/count_per_date.png` - Skipped (no data)  
❌ `svm_shape_2_classifier/figures/avg_length_per_date.png` - Skipped (no data)  

## Ready to Run

The pipeline now handles all edge cases gracefully:

```bash
python3 run_shape_classifiers.py
```

**Status:** Will complete successfully even with 0 high-confidence predictions! ✅

---

*Fixed: 2026-02-28*  
*Error: KeyError when aggregating empty DataFrame*  
*Solution: Early return with empty file + graceful visualization skipping*

