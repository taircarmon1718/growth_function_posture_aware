"""
EXAMPLE SESSION OUTPUT
======================
This is what you'll see when running the interactive labeling tool.
"""

# ============================================================================
# TERMINAL OUTPUT EXAMPLE
# ============================================================================

"""
======================================================================
INTERACTIVE LARVA QUALITY LABELING TOOL
======================================================================
Start time: 2026-02-28 15:30:00
Output file: analysis_full/larva_quality_labels.xlsx
======================================================================

SCANNING PROCESSED RESULTS
======================================================================
Found 11 date folders: ['18.10', '19.10', '20.10', '21.10', '24.10', '25.10', '26.10', '27.10', '29.10', '3.11', '31.10']

📊 Inventory Summary:
   Total Images: 247
   Total Larvae: 3,842
   18.10: 27 images, 198 larvae
   19.10: 23 images, 287 larvae
   20.10: 18 images, 245 larvae
   21.10: 22 images, 312 larvae
   24.10: 25 images, 389 larvae
   25.10: 19 images, 276 larvae
   26.10: 24 images, 401 larvae
   27.10: 21 images, 356 larvae
   29.10: 20 images, 298 larvae
   3.11: 18 images, 312 larvae
   31.10: 30 images, 768 larvae

======================================================================
CREATING SAMPLING PLAN
======================================================================
Configuration:
  Images per date: 3
  Larvae per image: 8
  Random seed: 42

  18.10:
    Selected 3 images: ['IMG_2953', 'IMG_3007', 'IMG_3019']
      IMG_2953: 8 larvae
      IMG_3007: 8 larvae
      IMG_3019: 8 larvae
  19.10:
    Selected 3 images: ['IMG_3788', 'IMG_3801', 'IMG_3815']
      IMG_3788: 8 larvae
      IMG_3801: 8 larvae
      IMG_3815: 8 larvae
  20.10:
    Selected 3 images: ['IMG_4023', 'IMG_4036', 'IMG_4048']
      IMG_4023: 8 larvae
      IMG_4036: 8 larvae
      IMG_4048: 8 larvae
  21.10:
    Selected 3 images: ['IMG_4256', 'IMG_4268', 'IMG_4279']
      IMG_4256: 8 larvae
      IMG_4268: 8 larvae
      IMG_4279: 8 larvae
  24.10:
    Selected 3 images: ['IMG_5012', 'IMG_5024', 'IMG_5037']
      IMG_5012: 8 larvae
      IMG_5024: 8 larvae
      IMG_5037: 8 larvae
  25.10:
    Selected 3 images: ['IMG_5489', 'IMG_5501', 'IMG_5513']
      IMG_5489: 8 larvae
      IMG_5501: 8 larvae
      IMG_5513: 8 larvae
  26.10:
    Selected 3 images: ['IMG_5876', 'IMG_5888', 'IMG_5901']
      IMG_5876: 8 larvae
      IMG_5888: 8 larvae
      IMG_5901: 8 larvae
  27.10:
    Selected 3 images: ['IMG_6234', 'IMG_6246', 'IMG_6259']
      IMG_6234: 8 larvae
      IMG_6246: 8 larvae
      IMG_6259: 8 larvae
  29.10:
    Selected 3 images: ['IMG_7105', 'IMG_7118', 'IMG_7130']
      IMG_7105: 8 larvae
      IMG_7118: 8 larvae
      IMG_7130: 8 larvae
  3.11:
    Selected 3 images: ['IMG_7567', 'IMG_7579', 'IMG_7591']
      IMG_7567: 8 larvae
      IMG_7579: 8 larvae
      IMG_7591: 8 larvae
  31.10:
    Selected 3 images: ['IMG_7376', 'IMG_7389', 'IMG_7402']
      IMG_7376: 8 larvae
      IMG_7389: 8 larvae
      IMG_7402: 8 larvae

📋 Total samples in plan: 264

📝 No existing labels found. Starting fresh.

======================================================================
LABELING SESSION
======================================================================
  Total samples in plan: 264
  Already labeled: 0
  Remaining to label: 264
======================================================================

🖼️  Display window opened. Starting labeling...

======================================================================
Sample 1/264: 18.10/IMG_2953/larva_001.png
======================================================================

Is this a VALID LARVA? (1=Yes, 0=No, q=Quit): 1 ✓

Is POSTURE valid for measurement? (1=Yes, 0=No, q=Quit): 1 ✓

✅ Label saved: valid_larva=1, valid_posture=1

======================================================================
Sample 2/264: 18.10/IMG_2953/larva_003.png
======================================================================

Is this a VALID LARVA? (1=Yes, 0=No, q=Quit): 1 ✓

Is POSTURE valid for measurement? (1=Yes, 0=No, q=Quit): 0 ✗

✅ Label saved: valid_larva=1, valid_posture=0

======================================================================
Sample 3/264: 18.10/IMG_2953/larva_005.png
======================================================================

Is this a VALID LARVA? (1=Yes, 0=No, q=Quit): 0 ✗

Is POSTURE valid for measurement? (1=Yes, 0=No, q=Quit): 0 ✗

✅ Label saved: valid_larva=0, valid_posture=0

======================================================================
Sample 4/264: 18.10/IMG_2953/larva_007.png
======================================================================

Is this a VALID LARVA? (1=Yes, 0=No, q=Quit): 1 ✓

Is POSTURE valid for measurement? (1=Yes, 0=No, q=Quit): 1 ✓

✅ Label saved: valid_larva=1, valid_posture=1

[... continues for all samples ...]

======================================================================
Sample 50/264: 21.10/IMG_4268/larva_012.png
======================================================================

Is this a VALID LARVA? (1=Yes, 0=No, q=Quit): q (quit)

⚠️  User requested quit.

======================================================================
LABELING SESSION COMPLETE
======================================================================
  Total labels saved: 50
  Output file: analysis_full/larva_quality_labels.xlsx
  End time: 2026-02-28 16:15:00
======================================================================

📊 SUMMARY STATISTICS:
  Valid larvae: 42/50 (84.0%)
  Valid postures: 35/50 (70.0%)
======================================================================
"""

# ============================================================================
# RESUME SESSION (Later)
# ============================================================================

"""
======================================================================
INTERACTIVE LARVA QUALITY LABELING TOOL
======================================================================
Start time: 2026-02-28 17:00:00
Output file: analysis_full/larva_quality_labels.xlsx
======================================================================

SCANNING PROCESSED RESULTS
======================================================================
Found 11 date folders: ['18.10', '19.10', '20.10', ...]

📊 Inventory Summary:
   Total Images: 247
   Total Larvae: 3,842
   ...

======================================================================
CREATING SAMPLING PLAN
======================================================================
Configuration:
  Images per date: 3
  Larvae per image: 8
  Random seed: 42

  [Same sampling plan as before - reproducible!]

📋 Total samples in plan: 264

✅ Loaded 50 existing labels from larva_quality_labels.xlsx

======================================================================
LABELING SESSION
======================================================================
  Total samples in plan: 264
  Already labeled: 50
  Remaining to label: 214
======================================================================

🖼️  Display window opened. Starting labeling...

======================================================================
Sample 51/264: 21.10/IMG_4268/larva_015.png
======================================================================

[Automatically continues from where you left off!]
"""

# ============================================================================
# ANALYSIS EXAMPLE
# ============================================================================

"""
# After labeling, analyze your results:

import pandas as pd
import matplotlib.pyplot as plt

# Load labels
df = pd.read_excel('analysis_full/larva_quality_labels.xlsx')

print("="*60)
print("LARVA QUALITY ANALYSIS")
print("="*60)
print(f"Total samples labeled: {len(df)}")
print(f"Valid larvae: {df['is_valid_larva'].sum()} ({100*df['is_valid_larva'].mean():.1f}%)")
print(f"Valid postures: {df['is_valid_posture'].sum()} ({100*df['is_valid_posture'].mean():.1f}%)")

# Quality rate among valid larvae
valid_larvae_df = df[df['is_valid_larva'] == 1]
if len(valid_larvae_df) > 0:
    quality_rate = valid_larvae_df['is_valid_posture'].mean()
    print(f"\\nQuality rate (good posture | valid larva): {quality_rate:.1%}")

# Per-date statistics
print("\\nPer-Date Statistics:")
print("-"*60)
date_stats = df.groupby('date').agg({
    'larva_filename': 'count',
    'is_valid_larva': 'sum',
    'is_valid_posture': 'sum'
}).rename(columns={
    'larva_filename': 'Total',
    'is_valid_larva': 'Valid Larvae',
    'is_valid_posture': 'Good Posture'
})
print(date_stats)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart: Valid vs Invalid
valid_counts = df['is_valid_larva'].value_counts()
axes[0].pie(valid_counts, labels=['Invalid', 'Valid'], autopct='%1.1f%%',
            colors=['#ff6b6b', '#51cf66'])
axes[0].set_title('Larva Validity')

# Bar chart: Posture quality
posture_counts = df['is_valid_posture'].value_counts()
axes[1].pie(posture_counts, labels=['Bad Posture', 'Good Posture'], 
            autopct='%1.1f%%', colors=['#ffd43b', '#339af0'])
axes[1].set_title('Posture Quality')

plt.tight_layout()
plt.savefig('analysis_full/quality_summary.png', dpi=300)
print("\\nVisualization saved to: analysis_full/quality_summary.png")
"""

