# Larvae Examples and Length Analysis

## ✅ Analysis Complete!

This analysis extracted example larvae from each date and calculated average body length statistics.

---

## Generated Files

### 1. `larvae_examples_per_date.png`
**Visual summary showing one example larva from each date**

- Shows 11 examples (one per date)
- Arranged in a grid layout (4 columns)
- Each panel shows:
  - Date label
  - Image folder name
  - Larva filename
  - Visual representation (no text metrics)

**Use this to:**
- Visually compare larvae across dates
- Quality check the data
- Include in presentations/reports

---

### 2. `average_length_per_date.png`
**Graph showing average body length trends over time**

**Features:**
- Blue line with circles: Mean body length per date
- Error bars: Standard deviation
- Orange triangles (▼): Minimum length
- Red triangles (▲): Maximum length
- Sample sizes (n=XXX) annotated above each point

**Key Statistics:**
- **Total larvae analyzed:** 11,399
- **11 dates:** 18.10 through 31.10 and 3.11
- **Grand mean length:** 69.42 pixels
- **Range of means:** 33.86 - 191.97 pixels

**Observations:**
- 18.10 has the highest average (191.97 ± 279.64 pixels)
- 24.10 has the lowest average (33.86 ± 88.80 pixels)
- Later dates show increasing sample sizes
- High variability in measurements (large error bars)

---

### 3. `length_statistics_per_date.csv`
**Detailed statistics table in CSV format**

**Columns:**
- `date` - Date folder name
- `num_larvae` - Number of larvae measured
- `avg_length` - Average body length (pixels)
- `std_length` - Standard deviation
- `min_length` - Minimum length
- `max_length` - Maximum length
- `median_length` - Median length

**Use this for:**
- Statistical analysis
- Import into R, Python, Excel
- Further data processing

---

### 4. `summary_report.txt`
**Text summary of all statistics**

Contains:
- Overall summary
- Per-date detailed statistics
- Grand averages
- Complete numeric breakdown

---

## Per-Date Summary

| Date | N Larvae | Avg Length (±SD) | Range | Median |
|------|----------|------------------|-------|--------|
| 18.10 | 388 | 191.97 ± 279.64 | 2-1932 | 70 |
| 19.10 | 749 | 60.97 ± 182.10 | 2-1408 | 14 |
| 20.10 | 610 | 80.47 ± 158.06 | 2-803 | 13 |
| 21.10 | 466 | 106.23 ± 217.92 | 2-1588 | 15 |
| 24.10 | 1340 | 33.86 ± 88.80 | 1-1398 | 15 |
| 25.10 | 765 | 44.74 ± 124.42 | 1-1387 | 19 |
| 26.10 | 689 | 38.85 ± 99.53 | 1-1279 | 19 |
| 27.10 | 961 | 59.08 ± 176.52 | 1-1491 | 21 |
| 29.10 | 1489 | 53.89 ± 152.07 | 1-2618 | 24 |
| 3.11 | 1938 | 49.01 ± 116.41 | 3-2531 | 27 |
| 31.10 | 2004 | 44.58 ± 83.16 | 1-1045 | 30 |

---

## Key Findings

### Sample Size Trends
- Early dates (18.10-21.10): 388-749 larvae
- Mid dates (24.10-27.10): 689-1340 larvae
- Late dates (29.10-3.11, 31.10): 1489-2004 larvae
- **Increasing sample size over time**

### Length Trends
- **Mean body length ranges from 33.86 to 191.97 pixels**
- **Median lengths increase from 13-70 pixels early to 21-30 pixels later**
- High variability within each date (large standard deviations)
- Possible outliers or debris affecting measurements

### Data Quality Notes
- Very wide ranges suggest possible segmentation issues
- Some measurements as low as 1-2 pixels (likely artifacts)
- Some measurements very high (>1000 pixels, possibly clumped larvae)
- Consider filtering outliers for refined analysis

---

## How to Use These Results

### For Publications
1. Include `larvae_examples_per_date.png` to show data coverage
2. Include `average_length_per_date.png` for temporal trends
3. Reference statistics from CSV or summary report

### For Further Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load statistics
df = pd.read_csv('larvae_examples_analysis/length_statistics_per_date.csv')

# Filter out potential outliers (e.g., length > 500 pixels)
# Re-analyze your data with filtered criteria

# Plot custom graphs
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['median_length'], marker='o')
plt.xlabel('Date')
plt.ylabel('Median Body Length (pixels)')
plt.title('Median Larva Length Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('median_length_trend.png')
```

### For Quality Control
1. Review `larvae_examples_per_date.png` for visual consistency
2. Check if examples represent typical larvae
3. Identify dates with unusual patterns
4. Cross-reference with original images if needed

---

## Recommendations

### Data Cleaning
1. **Filter extreme outliers** (e.g., length < 10 or > 500 pixels)
2. **Use median instead of mean** (more robust to outliers)
3. **Apply quality labels** from interactive labeling tool

### Refined Analysis
```python
# Example: Filter by quality labels
labels = pd.read_excel('analysis_full/larva_quality_labels.xlsx')

# Keep only high-quality larvae
high_quality = labels[
    (labels['is_valid_larva'] == 1) & 
    (labels['shape_score'] >= 1)
]

# Re-calculate statistics for high-quality only
```

### Consider Using
- SVM predictions to filter valid larvae
- Shape score 2 (great shape) for primary analysis
- Shape score 1 (ok shape) for supplementary data

---

## Script Information

**Script:** `create_larvae_examples_and_graph.py`

**To re-run:**
```bash
cd /Users/taircarmon/Desktop/growth_function_posture_aware
python3 create_larvae_examples_and_graph.py
```

**Runtime:** ~8 seconds for 11,399 larvae

**Requirements:**
- opencv-python (cv2)
- numpy
- pandas
- matplotlib
- seaborn

---

## Next Steps

1. ✅ **Review examples figure** - Check visual quality
2. ✅ **Analyze length graph** - Identify temporal patterns
3. ✅ **Export statistics** - Use CSV for further analysis
4. ⭕ **Filter outliers** - Refine measurements
5. ⭕ **Cross-reference with quality labels** - Use only validated larvae
6. ⭕ **Statistical testing** - Compare dates with ANOVA or t-tests

---

**Generated:** 2026-03-01 09:40:16  
**Total larvae:** 11,399  
**Dates analyzed:** 11  
**Output folder:** `larvae_examples_analysis/`

