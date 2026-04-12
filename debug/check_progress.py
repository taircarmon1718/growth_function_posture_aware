#!/usr/bin/env python3
"""
Check pipeline progress and generate summary
"""
from pathlib import Path
import csv

ROOT_DIR = Path('/Users/taircarmon/Desktop/growth_function_posture_aware')
OUTPUT_DIR = ROOT_DIR / 'analysis_full'

def check_progress():
    print("="*70)
    print("PIPELINE PROGRESS CHECK")
    print("="*70)

    if not OUTPUT_DIR.exists():
        print("❌ analysis_full directory not found!")
        return

    # Find all date folders
    date_folders = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()])

    print(f"\n📊 Date folders processed: {len(date_folders)}")

    total_images = 0
    total_larvae = 0

    for date_folder in date_folders:
        image_folders = sorted([d for d in date_folder.iterdir() if d.is_dir()])
        total_images += len(image_folders)

        print(f"\n📁 {date_folder.name}:")
        print(f"   Images processed: {len(image_folders)}")

        # Check first few images
        for i, img_folder in enumerate(image_folders[:3]):
            files = list(img_folder.iterdir())
            csv_file = img_folder / 'morphometrics.csv'

            if csv_file.exists():
                with open(csv_file, 'r') as f:
                    num_larvae = len(list(csv.DictReader(f)))
                    total_larvae += num_larvae
                    print(f"      • {img_folder.name}: {num_larvae} larvae")
            else:
                print(f"      • {img_folder.name}: No CSV found")

        if len(image_folders) > 3:
            print(f"      ... and {len(image_folders)-3} more images")

    print(f"\n{'='*70}")
    print(f"TOTAL STATISTICS:")
    print(f"  Total Images Processed: {total_images}")
    print(f"  Total Larvae Detected: {total_larvae}")
    print("="*70)

    # Check for global summary
    global_csv = OUTPUT_DIR / 'global_summary.csv'
    if global_csv.exists():
        print(f"\n✅ Global summary exists: {global_csv}")
        with open(global_csv, 'r') as f:
            lines = len(f.readlines()) - 1  # Subtract header
            print(f"   Contains {lines} image records")
    else:
        print(f"\n⚠️  Global summary not yet created")

if __name__ == "__main__":
    check_progress()

