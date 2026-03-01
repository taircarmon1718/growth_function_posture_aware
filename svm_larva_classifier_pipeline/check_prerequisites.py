#!/usr/bin/env python3
"""
Pre-flight check script for SVM pipeline
Verifies all prerequisites before running the full pipeline
"""

from pathlib import Path
import sys

print("="*70)
print("SVM PIPELINE PRE-FLIGHT CHECK")
print("="*70)

ROOT_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = ROOT_DIR / "analysis_full"
LABELS_FILE = ANALYSIS_DIR / "larva_quality_labels.xlsx"

checks_passed = 0
checks_failed = 0

# Check 1: Labels file exists
print("\n1. Checking for labels file...")
if LABELS_FILE.exists():
    print(f"   ✅ Found: {LABELS_FILE}")
    checks_passed += 1

    # Try to load it
    try:
        import pandas as pd
        df = pd.read_excel(LABELS_FILE)
        print(f"   ✅ Loaded {len(df)} labeled larvae")

        # Check columns
        required_cols = ['date', 'image_name', 'larva_filename', 'is_valid_larva']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"   ⚠️  Missing columns: {missing}")
            checks_failed += 1
        else:
            print(f"   ✅ All required columns present")
            checks_passed += 1

            # Check label distribution
            valid = df['is_valid_larva'].sum()
            invalid = len(df) - valid
            print(f"   ✅ Label distribution: {valid} valid, {invalid} invalid")
            if min(valid, invalid) < 10:
                print(f"   ⚠️  Warning: Need at least 10 samples per class for good training")
            checks_passed += 1

    except Exception as e:
        print(f"   ❌ Error loading labels: {e}")
        checks_failed += 1
else:
    print(f"   ❌ Not found: {LABELS_FILE}")
    print(f"   ℹ️  Run: python3 interactive_labeling_tool.py")
    checks_failed += 1

# Check 2: Analysis directory structure
print("\n2. Checking analysis_full directory...")
if ANALYSIS_DIR.exists():
    print(f"   ✅ Found: {ANALYSIS_DIR}")
    checks_passed += 1

    # Count date folders
    date_folders = [d for d in ANALYSIS_DIR.iterdir()
                   if d.is_dir() and d.name[0].isdigit()]
    print(f"   ✅ Found {len(date_folders)} date folders")
    checks_passed += 1

    # Sample a few larvae
    total_larvae = 0
    for df in date_folders[:3]:
        for img_folder in df.iterdir():
            if img_folder.is_dir():
                reports = img_folder / "larvae_reports"
                if reports.exists():
                    larvae = list(reports.glob("larva_*.png"))
                    total_larvae += len(larvae)

    if total_larvae > 0:
        print(f"   ✅ Sample check: Found larvae images")
        checks_passed += 1
    else:
        print(f"   ⚠️  No larvae images found in sample")
        checks_failed += 1
else:
    print(f"   ❌ Not found: {ANALYSIS_DIR}")
    checks_failed += 1

# Check 3: Required Python packages
print("\n3. Checking Python dependencies...")
required_packages = [
    'numpy', 'pandas', 'cv2', 'sklearn',
    'matplotlib', 'seaborn', 'joblib'
]

for pkg_name in required_packages:
    try:
        if pkg_name == 'cv2':
            import cv2
            pkg_display = 'opencv-python'
        elif pkg_name == 'sklearn':
            import sklearn
            pkg_display = 'scikit-learn'
        else:
            __import__(pkg_name)
            pkg_display = pkg_name

        print(f"   ✅ {pkg_display}")
        checks_passed += 1
    except ImportError:
        print(f"   ❌ {pkg_name} not installed")
        print(f"      Install: pip3 install {pkg_name}")
        checks_failed += 1

# Check 4: Output directory
print("\n4. Checking output directory...")
pipeline_dir = ROOT_DIR / "svm_larva_classifier_pipeline"
if pipeline_dir.exists():
    print(f"   ✅ Pipeline directory exists")
    checks_passed += 1

    # Check if already has outputs
    if (pipeline_dir / "svm_model.pkl").exists():
        print(f"   ℹ️  Previous model found - will be overwritten")

else:
    print(f"   ❌ Pipeline directory not found")
    checks_failed += 1

# Summary
print("\n" + "="*70)
print("PRE-FLIGHT CHECK SUMMARY")
print("="*70)
print(f"✅ Checks passed: {checks_passed}")
print(f"❌ Checks failed: {checks_failed}")

if checks_failed == 0:
    print("\n🎉 All checks passed! Ready to run the pipeline.")
    print("\nRun:")
    print("   python3 svm_larva_classifier_pipeline/run_svm_pipeline.py")
    sys.exit(0)
else:
    print("\n⚠️  Some checks failed. Please fix the issues above before running.")
    sys.exit(1)

