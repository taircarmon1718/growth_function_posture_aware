#!/usr/bin/env python3
"""
Demo Script - Shows what the labeling tool will sample
Runs without user interaction to preview the sampling plan
"""

import random
from pathlib import Path
from datetime import datetime

# Configuration (matches interactive_labeling_tool.py)
ROOT_DIR = Path(__file__).parent / "analysis_full"
IMAGES_PER_DATE = 3
LARVAE_PER_IMAGE = 8
RANDOM_SEED = 42

def demo_sampling():
    print("="*70)
    print("LABELING TOOL - SAMPLING PREVIEW (DEMO MODE)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Images per date: {IMAGES_PER_DATE}")
    print(f"  Larvae per image: {LARVAE_PER_IMAGE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print("="*70)

    # Collect inventory
    print("\n📊 Scanning analysis_full directory...\n")

    larvae_inventory = {}
    date_folders = sorted([
        d for d in ROOT_DIR.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ])

    print(f"Found {len(date_folders)} date folders:")
    for df in date_folders:
        print(f"  - {df.name}")

    for date_folder in date_folders:
        date_name = date_folder.name
        larvae_inventory[date_name] = {}

        image_folders = sorted([d for d in date_folder.iterdir() if d.is_dir()])

        for image_folder in image_folders:
            larvae_reports_dir = image_folder / "larvae_reports"
            if larvae_reports_dir.exists():
                larva_files = sorted([
                    f.name for f in larvae_reports_dir.iterdir()
                    if f.name.startswith("larva_") and f.name.endswith(".png")
                ])
                if larva_files:
                    larvae_inventory[date_name][image_folder.name] = larva_files

    # Statistics
    total_images = sum(len(images) for images in larvae_inventory.values())
    total_larvae = sum(
        len(larvae) for images in larvae_inventory.values()
        for larvae in images.values()
    )

    print(f"\n{'='*70}")
    print("INVENTORY SUMMARY")
    print(f"{'='*70}")
    print(f"Total Images: {total_images}")
    print(f"Total Larvae: {total_larvae}")
    print()

    for date_name, images in sorted(larvae_inventory.items()):
        num_images = len(images)
        num_larvae = sum(len(larvae) for larvae in images.values())
        print(f"  {date_name}: {num_images:3d} images, {num_larvae:4d} larvae")

    # Create sampling plan
    print(f"\n{'='*70}")
    print("SAMPLING PLAN (Random Seed = {})".format(RANDOM_SEED))
    print(f"{'='*70}")

    random.seed(RANDOM_SEED)
    sampling_plan = []

    for date_name, images_dict in sorted(larvae_inventory.items()):
        if not images_dict:
            continue

        available_images = list(images_dict.keys())
        num_to_sample = min(IMAGES_PER_DATE, len(available_images))
        selected_images = random.sample(available_images, num_to_sample)

        print(f"\n📁 {date_name}:")
        print(f"   Sampling {num_to_sample} out of {len(available_images)} images")

        for img_name in selected_images:
            larvae_files = images_dict[img_name]
            num_larvae_to_sample = min(LARVAE_PER_IMAGE, len(larvae_files))
            selected_larvae = random.sample(larvae_files, num_larvae_to_sample)

            print(f"      {img_name}: {num_larvae_to_sample}/{len(larvae_files)} larvae")

            for larva_file in selected_larvae:
                larva_path = ROOT_DIR / date_name / img_name / "larvae_reports" / larva_file
                sampling_plan.append((date_name, img_name, larva_file, larva_path))

    print(f"\n{'='*70}")
    print("SAMPLING SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples to label: {len(sampling_plan)}")
    print(f"Estimated time (30 sec/sample): {len(sampling_plan)*30/60:.1f} minutes")
    print("="*70)

    # Show first 10 samples
    print("\n📋 First 10 samples in queue:")
    for i, (date, img, larva, path) in enumerate(sampling_plan[:10], 1):
        exists = "✓" if path.exists() else "✗"
        print(f"  {i:3d}. {date:6s} / {img:12s} / {larva:15s} {exists}")

    if len(sampling_plan) > 10:
        print(f"  ... and {len(sampling_plan)-10} more samples")

    print(f"\n{'='*70}")
    print("To start actual labeling, run:")
    print("  python3 interactive_labeling_tool.py")
    print("="*70)

if __name__ == "__main__":
    demo_sampling()

