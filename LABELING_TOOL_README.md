# Interactive Larva Quality Labeling Tool

## Overview

A fully automated, keyboard-driven tool for labeling larva quality in processed microscopy images. Designed for efficient, reproducible quality assessment with automatic resume capability.

## Features

✅ **Automatic Sample Detection** - Scans all date folders and larvae images  
✅ **Reproducible Random Sampling** - Fixed seed ensures consistent sampling  
✅ **Incremental Saving** - Labels saved after each sample  
✅ **Automatic Resume** - Skip already-labeled samples automatically  
✅ **No Duplicate Labels** - Prevents re-labeling the same larva  
✅ **Excel Output** - Clean, analysis-ready Excel file  
✅ **Keyboard-Only Workflow** - Press 0, 1, or q  
✅ **Exception Safe** - Handles errors gracefully  
✅ **Progress Tracking** - Real-time progress display  

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install opencv-python numpy pandas openpyxl scikit-image scipy
```

## Quick Start

```bash
# Run the labeling tool
python3 interactive_labeling_tool.py
```

## Configuration

Edit the configuration section at the top of `interactive_labeling_tool.py`:

```python
# Sampling Configuration
IMAGES_PER_DATE = 3          # Number of images to sample per date folder
LARVAE_PER_IMAGE = 8         # Number of larvae to sample per image
RANDOM_SEED = 42             # For reproducible sampling

# Display Configuration
DISPLAY_WIDTH = 1200         # Width of display window
DISPLAY_HEIGHT = 800         # Height of display window
```

## How to Use

### Step 1: Start the Tool

```bash
python3 interactive_labeling_tool.py
```

The tool will:
1. Scan `analysis_full/` directory
2. Detect all date folders (18.10, 19.10, etc.)
3. Create a random sampling plan
4. Load any existing labels
5. Skip to the first unlabeled sample
6. Open a display window

### Step 2: Label Each Larva

For each displayed larva, you'll be asked two questions:

**Question 1: Is this a VALID LARVA?**
- Press `1` for YES (it's a real larva)
- Press `0` for NO (it's debris, artifact, or not a larva)

**Question 2: Is POSTURE valid for measurement?**
- Press `1` for YES (larva is straight/measurable)
- Press `0` for NO (larva is curled, overlapping, or unmeasurable)

**Other Controls:**
- Press `q` to quit and save (safe exit)
- Press `Ctrl+C` to force quit (still saves)

### Step 3: View Results

Labels are saved to:
```
analysis_full/larva_quality_labels.xlsx
```

### Step 4: Resume Anytime

Simply re-run the script:
```bash
python3 interactive_labeling_tool.py
```

It will automatically:
- Load existing labels
- Skip already-labeled samples
- Continue from where you left off

## Output Format

### Excel File: `larva_quality_labels.xlsx`

| Column | Type | Description |
|--------|------|-------------|
| `date` | String | Date folder name (e.g., "31.10") |
| `image_name` | String | Image folder name (e.g., "IMG_7382") |
| `larva_filename` | String | Larva filename (e.g., "larva_001.png") |
| `is_valid_larva` | Integer | 1 = valid larva, 0 = not valid |
| `is_valid_posture` | Integer | 1 = good posture, 0 = bad posture |

### Example Data

```
date    image_name  larva_filename  is_valid_larva  is_valid_posture
18.10   IMG_2953    larva_001.png   1               1
18.10   IMG_2953    larva_002.png   1               0
19.10   IMG_3788    larva_003.png   0               0
```

## Workflow Example

```
======================================================================
INTERACTIVE LARVA QUALITY LABELING TOOL
======================================================================
Start time: 2026-02-28 14:30:15
Output file: analysis_full/larva_quality_labels.xlsx
======================================================================

SCANNING PROCESSED RESULTS
======================================================================
Found 11 date folders: ['18.10', '19.10', '20.10', ...]

📊 Inventory Summary:
   Total Images: 150
   Total Larvae: 1248
   18.10: 25 images, 187 larvae
   19.10: 18 images, 142 larvae
   ...

======================================================================
CREATING SAMPLING PLAN
======================================================================
Configuration:
  Images per date: 3
  Larvae per image: 8
  Random seed: 42

  18.10:
    Selected 3 images: ['IMG_2953', 'IMG_3001', 'IMG_3012']
      IMG_2953: 8 larvae
      IMG_3001: 8 larvae
      IMG_3012: 8 larvae
  ...

📋 Total samples in plan: 264

✅ Loaded 85 existing labels from larva_quality_labels.xlsx

======================================================================
LABELING SESSION
======================================================================
  Total samples in plan: 264
  Already labeled: 85
  Remaining to label: 179
======================================================================

🖼️  Display window opened. Starting labeling...

======================================================================
Sample 86/264: 18.10/IMG_3012/larva_005.png
======================================================================

Is this a VALID LARVA? (1=Yes, 0=No, q=Quit): 1 ✓

Is POSTURE valid for measurement? (1=Yes, 0=No, q=Quit): 0 ✗

✅ Label saved: valid_larva=1, valid_posture=0

[continues for each sample...]
```

## Sampling Strategy

The tool uses **stratified random sampling**:

1. **Date Level**: Randomly select N images per date folder
2. **Image Level**: Randomly select M larvae per image folder
3. **Reproducible**: Uses fixed random seed (configurable)

This ensures:
- Representative sampling across all dates
- No bias toward any particular date
- Reproducible results for validation
- Balanced coverage of your dataset

## Tips for Efficient Labeling

### Valid Larva Criteria (Question 1)

**Label as 1 (YES)** if:
- Clear larva shape visible
- Single organism
- Complete larva in frame

**Label as 0 (NO)** if:
- Debris or artifact
- Multiple overlapping larvae
- Partial larva (cut off)
- Very blurry or unclear

### Valid Posture Criteria (Question 2)

**Label as 1 (YES)** if:
- Larva is relatively straight
- Body is extended
- Measurements would be accurate
- No tight curling

**Label as 0 (NO)** if:
- Larva is tightly curled
- C-shaped or coiled
- Overlapping itself
- Extreme bending

## Progress Tracking

The tool displays:
- Total progress (e.g., "85/264 completed")
- Current sample info (date, image, larva)
- Real-time instructions
- Summary statistics on exit

## Error Handling

The tool handles:
- Missing images (skips automatically)
- Corrupted files (skips with warning)
- Keyboard interrupts (saves before exit)
- File write errors (falls back to CSV)

## Reproducibility

To ensure reproducible sampling:

1. **Fixed Random Seed**: Default is 42
2. **Sorted Traversal**: Dates and images processed in sorted order
3. **Deterministic Sampling**: Same seed = same samples every time

To change sampling, modify `RANDOM_SEED` in the configuration.

## Modifying Sampling Configuration

### Sample More/Fewer Images Per Date

```python
IMAGES_PER_DATE = 5  # Changed from 3 to 5
```

### Sample More/Fewer Larvae Per Image

```python
LARVAE_PER_IMAGE = 10  # Changed from 8 to 10
```

### Change Random Seed

```python
RANDOM_SEED = 123  # Different sampling pattern
```

## Data Analysis Example

After labeling, analyze results with pandas:

```python
import pandas as pd

# Load labels
df = pd.read_excel('analysis_full/larva_quality_labels.xlsx')

# Summary statistics
print(f"Total labeled: {len(df)}")
print(f"Valid larvae: {df['is_valid_larva'].sum()}")
print(f"Valid posture: {df['is_valid_posture'].sum()}")

# Calculate quality rate
quality_rate = (
    df[df['is_valid_larva'] == 1]['is_valid_posture'].sum() / 
    df['is_valid_larva'].sum()
)
print(f"Quality rate: {quality_rate:.2%}")

# Per-date statistics
date_stats = df.groupby('date').agg({
    'is_valid_larva': 'sum',
    'is_valid_posture': 'sum',
    'larva_filename': 'count'
})
print(date_stats)
```

## Troubleshooting

### Display Window Not Showing

Make sure you have X11/XQuartz installed on macOS:
```bash
brew install --cask xquartz
```

### Excel Write Error

If Excel writing fails, the tool automatically saves as CSV:
```
analysis_full/larva_quality_labels.csv
```

### Import Errors

Install missing dependencies:
```bash
pip install opencv-python pandas openpyxl
```

### No Larvae Found

Check that:
1. `analysis_full/` directory exists
2. Date folders contain image folders
3. Image folders have `larvae_reports/` subdirectories
4. `larva_*.png` files exist

## Architecture

### Modular Functions

- `collect_all_larvae()` - Scans directory structure
- `create_sampling_plan()` - Random sampling with seed
- `load_existing_labels()` - Load previous work
- `save_label()` - Incremental saving
- `is_already_labeled()` - Duplicate detection
- `display_larva()` - Visual presentation
- `get_user_input()` - Keyboard handling
- `main()` - Workflow orchestration

### Data Flow

```
analysis_full/
    └─> collect_all_larvae()
        └─> create_sampling_plan()
            └─> load_existing_labels()
                └─> display_larva()
                    └─> get_user_input()
                        └─> save_label()
                            └─> Excel file
```

## Performance

- Handles thousands of larvae efficiently
- Instant resume from any point
- No memory issues with large datasets
- Fast image loading with OpenCV

## License

Research use only. Modify as needed for your workflow.

## Support

For issues or questions, check:
1. Configuration settings
2. Directory structure
3. File permissions
4. Dependencies installed

## Version History

- **v1.0** (2026-02-28): Initial release with full automation

