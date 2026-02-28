# QUICK START GUIDE - Interactive Labeling Tool

## Files Created

✅ `interactive_labeling_tool.py` - Main labeling tool (production-ready)  
✅ `demo_sampling.py` - Preview sampling without labeling  
✅ `LABELING_TOOL_README.md` - Complete documentation  
✅ `requirements.txt` - Python dependencies  

## Installation (One-Time Setup)

```bash
# Navigate to project directory
cd /Users/taircarmon/Desktop/growth_function_posture_aware

# Install dependencies (if not already installed)
pip3 install opencv-python numpy pandas openpyxl

# Or use requirements file
pip3 install -r requirements.txt
```

## Usage

### Option 1: Preview Sampling (No Labeling)

See what will be sampled without starting labeling:

```bash
python3 demo_sampling.py
```

This shows:
- How many images/larvae per date
- Which specific samples will be selected
- Total number of labels needed
- Estimated time

### Option 2: Start Labeling

```bash
python3 interactive_labeling_tool.py
```

**What happens:**
1. Tool scans `analysis_full/` directory
2. Creates random sampling plan (3 images per date, 8 larvae per image)
3. Opens display window
4. Shows first larva image

**For each larva, answer two questions:**

```
Question 1: Is this a VALID LARVA? (1=Yes, 0=No, q=Quit)
  Press: 1 or 0

Question 2: Is POSTURE valid for measurement? (1=Yes, 0=No, q=Quit)
  Press: 1 or 0
```

**Controls:**
- `1` = Yes
- `0` = No  
- `q` = Quit and save

**Output:** Labels saved to `analysis_full/larva_quality_labels.xlsx`

### Option 3: Resume Labeling

If you quit and want to continue later:

```bash
python3 interactive_labeling_tool.py
```

The tool automatically:
- Loads existing labels
- Skips already-labeled samples
- Continues from where you left off

## Configuration

Edit `interactive_labeling_tool.py` lines 22-24:

```python
IMAGES_PER_DATE = 3          # Change to sample more/fewer images
LARVAE_PER_IMAGE = 8         # Change to sample more/fewer larvae
RANDOM_SEED = 42             # Change for different random sampling
```

## Output File

**Location:** `analysis_full/larva_quality_labels.xlsx`

**Columns:**
- `date` - Date folder (e.g., "31.10")
- `image_name` - Image folder (e.g., "IMG_7382")
- `larva_filename` - Larva file (e.g., "larva_001.png")
- `is_valid_larva` - 1=valid larva, 0=not valid
- `is_valid_posture` - 1=good posture, 0=bad posture

## Example Workflow

```bash
# Step 1: Preview what will be sampled
python3 demo_sampling.py

# Output shows:
# - 11 date folders found
# - Total: ~264 samples to label
# - Estimated time: ~132 minutes

# Step 2: Start labeling
python3 interactive_labeling_tool.py

# Step 3: Label for a while, then press 'q' to quit

# Step 4: Resume later
python3 interactive_labeling_tool.py
# Automatically continues from where you stopped

# Step 5: Analyze results
python3 -c "
import pandas as pd
df = pd.read_excel('analysis_full/larva_quality_labels.xlsx')
print(f'Total labeled: {len(df)}')
print(f'Valid larvae: {df[\"is_valid_larva\"].sum()}')
print(f'Valid postures: {df[\"is_valid_posture\"].sum()}')
"
```

## Directory Structure

Your project should look like:

```
growth_function_posture_aware/
├── analysis_full/                          # Processed results
│   ├── 18.10/
│   │   └── IMG_XXXX/
│   │       └── larvae_reports/
│   │           ├── larva_001.png          # These get labeled
│   │           ├── larva_002.png
│   │           └── ...
│   ├── 19.10/
│   ├── ...
│   └── larva_quality_labels.xlsx          # OUTPUT FILE
│
├── interactive_labeling_tool.py            # MAIN TOOL
├── demo_sampling.py                        # Preview sampling
├── LABELING_TOOL_README.md                 # Full documentation
└── requirements.txt                        # Dependencies
```

## Troubleshooting

**Display window doesn't appear:**
- Make sure you're not running via SSH without X forwarding
- On macOS, install XQuartz: `brew install --cask xquartz`

**"ModuleNotFoundError":**
```bash
pip3 install opencv-python pandas openpyxl numpy
```

**Excel file error:**
- Tool automatically falls back to CSV format
- Output: `larva_quality_labels.csv`

**Want to change sampling:**
- Edit configuration section in `interactive_labeling_tool.py`
- Change `RANDOM_SEED` for different samples
- Change `IMAGES_PER_DATE` or `LARVAE_PER_IMAGE` for more/fewer samples

## Features Summary

✅ Fully automated - no manual file navigation  
✅ Random sampling - reproducible with fixed seed  
✅ Incremental saving - never lose progress  
✅ Auto-resume - picks up where you left off  
✅ No duplicates - never labels same larva twice  
✅ Keyboard-only - press 0, 1, or q  
✅ Exception safe - handles errors gracefully  
✅ Progress tracking - always know where you are  
✅ Production-ready - clean, modular code  

## Time Estimate

With default settings:
- 11 dates × 3 images × 8 larvae = 264 total samples
- At 30 seconds per sample = ~132 minutes
- Can be done in multiple sessions (auto-resume)

## Next Steps

1. **Preview:** `python3 demo_sampling.py`
2. **Label:** `python3 interactive_labeling_tool.py`
3. **Analyze:** Load `larva_quality_labels.xlsx` in Excel or Python

## Questions?

See full documentation in `LABELING_TOOL_README.md`

---

**Ready to start!** Run: `python3 interactive_labeling_tool.py`

