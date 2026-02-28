#!/usr/bin/env python3
"""
Generate Global Summary from Processed Results
"""
import csv
from pathlib import Path

ROOT_DIR = Path('/Users/taircarmon/Desktop/growth_function_posture_aware')
OUTPUT_DIR = ROOT_DIR / 'analysis_full'
GLOBAL_CSV = OUTPUT_DIR / 'global_summary.csv'

def generate_global_summary():
    print("="*70)
    print("GENERATING GLOBAL SUMMARY")
    print("="*70)

    if not OUTPUT_DIR.exists():
        print("❌ analysis_full directory not found!")
        return

    # Find all date folders
    date_folders = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()])

    print(f"\nScanning {len(date_folders)} date folders...")

    all_summaries = []

    for date_folder in date_folders:
        print(f"\n  Processing {date_folder.name}...")
        image_folders = sorted([d for d in date_folder.iterdir() if d.is_dir()])

        for img_folder in image_folders:
            csv_file = img_folder / 'morphometrics.csv'

            if csv_file.exists():
                # Read CSV and calculate statistics
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                    if rows:
                        body_lengths = [float(r['body_length']) for r in rows]
                        areas = [float(r['area']) for r in rows]
                        intensities = [float(r['mean_intensity']) for r in rows]

                        summary = {
                            'date': date_folder.name,
                            'image_name': img_folder.name,
                            'number_of_larvae': len(rows),
                            'mean_body_length': round(sum(body_lengths) / len(body_lengths), 2),
                            'mean_area': round(sum(areas) / len(areas), 2),
                            'mean_intensity': round(sum(intensities) / len(intensities), 2),
                        }
                        all_summaries.append(summary)
                        print(f"    ✓ {img_folder.name}: {len(rows)} larvae")
            else:
                # Image with no larvae detected
                summary = {
                    'date': date_folder.name,
                    'image_name': img_folder.name,
                    'number_of_larvae': 0,
                    'mean_body_length': 0,
                    'mean_area': 0,
                    'mean_intensity': 0,
                }
                all_summaries.append(summary)
                print(f"    ✓ {img_folder.name}: 0 larvae")

    # Write global summary
    if all_summaries:
        with open(GLOBAL_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)

        total_images = len(all_summaries)
        total_larvae = sum(s['number_of_larvae'] for s in all_summaries)

        print(f"\n{'='*70}")
        print("GLOBAL SUMMARY GENERATED")
        print(f"{'='*70}")
        print(f"  File: {GLOBAL_CSV}")
        print(f"  Total Images: {total_images}")
        print(f"  Total Larvae: {total_larvae}")
        print(f"  Average Larvae per Image: {total_larvae/total_images:.2f}")
        print("="*70)
    else:
        print("\n⚠️  No data found to summarize!")

if __name__ == "__main__":
    generate_global_summary()

