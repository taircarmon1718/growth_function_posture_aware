#!/usr/bin/env python3
"""
Interactive Larva Quality Labeling Tool
========================================
Randomly samples larvae from processed results and allows manual quality labeling.
Supports incremental saving, automatic resume, and reproducible sampling.

Author: Research Pipeline
Date: 2026-02-28
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
from datetime import datetime
import sys

# ======================== CONFIGURATION ========================
ROOT_DIR = Path(__file__).parent / "analysis_full"
OUTPUT_FILE = ROOT_DIR / "larva_quality_labels.xlsx"

# Sampling Configuration
IMAGES_PER_DATE = 3          # Number of images to sample per date folder
LARVAE_PER_IMAGE = 8         # Number of larvae to sample per image
RANDOM_SEED = 42             # For reproducible sampling

# Display Configuration
WINDOW_NAME = "Larva Quality Labeling"
DISPLAY_WIDTH = 1200         # Width of display window
DISPLAY_HEIGHT = 800         # Height of display window

# ======================== SAMPLING FUNCTIONS ========================

def collect_all_larvae():
    """
    Scans analysis_full directory and collects all available larvae images.

    Returns:
        dict: Dictionary mapping date -> image_name -> list of larva filenames
    """
    print("="*70)
    print("SCANNING PROCESSED RESULTS")
    print("="*70)

    if not ROOT_DIR.exists():
        print(f"❌ Error: {ROOT_DIR} not found!")
        return {}

    larvae_inventory = {}

    # Find all date folders
    date_folders = sorted([
        d for d in ROOT_DIR.iterdir()
        if d.is_dir() and d.name[0].isdigit()
    ])

    print(f"\nFound {len(date_folders)} date folders: {[d.name for d in date_folders]}")

    for date_folder in date_folders:
        date_name = date_folder.name
        larvae_inventory[date_name] = {}

        # Find all image folders
        image_folders = sorted([
            d for d in date_folder.iterdir()
            if d.is_dir()
        ])

        for image_folder in image_folders:
            image_name = image_folder.name
            larvae_reports_dir = image_folder / "larvae_reports"

            if not larvae_reports_dir.exists():
                continue

            # Find all larva PNG files
            larva_files = sorted([
                f.name for f in larvae_reports_dir.iterdir()
                if f.is_file() and f.name.startswith("larva_") and f.name.endswith(".png")
            ])

            if larva_files:
                larvae_inventory[date_name][image_name] = larva_files

    # Print inventory summary
    total_images = sum(len(images) for images in larvae_inventory.values())
    total_larvae = sum(
        len(larvae)
        for images in larvae_inventory.values()
        for larvae in images.values()
    )

    print(f"\n📊 Inventory Summary:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Larvae: {total_larvae}")

    for date_name, images in sorted(larvae_inventory.items()):
        num_images = len(images)
        num_larvae = sum(len(larvae) for larvae in images.values())
        print(f"   {date_name}: {num_images} images, {num_larvae} larvae")

    return larvae_inventory


def create_sampling_plan(larvae_inventory, random_seed=RANDOM_SEED):
    """
    Creates a reproducible random sampling plan.

    Args:
        larvae_inventory: Dictionary of all available larvae
        random_seed: Random seed for reproducibility

    Returns:
        list: List of tuples (date, image_name, larva_filename, larva_path)
    """
    print("\n" + "="*70)
    print("CREATING SAMPLING PLAN")
    print("="*70)
    print(f"Configuration:")
    print(f"  Images per date: {IMAGES_PER_DATE}")
    print(f"  Larvae per image: {LARVAE_PER_IMAGE}")
    print(f"  Random seed: {random_seed}")

    random.seed(random_seed)
    sampling_plan = []

    for date_name, images_dict in sorted(larvae_inventory.items()):
        if not images_dict:
            continue

        # Randomly select images for this date
        available_images = list(images_dict.keys())
        num_to_sample = min(IMAGES_PER_DATE, len(available_images))
        selected_images = random.sample(available_images, num_to_sample)

        print(f"\n  {date_name}:")
        print(f"    Selected {num_to_sample} images: {selected_images}")

        for image_name in selected_images:
            larvae_files = images_dict[image_name]

            # Randomly select larvae from this image
            num_larvae_to_sample = min(LARVAE_PER_IMAGE, len(larvae_files))
            selected_larvae = random.sample(larvae_files, num_larvae_to_sample)

            print(f"      {image_name}: {num_larvae_to_sample} larvae")

            # Build full paths and add to plan
            for larva_filename in selected_larvae:
                larva_path = ROOT_DIR / date_name / image_name / "larvae_reports" / larva_filename
                sampling_plan.append((date_name, image_name, larva_filename, larva_path))

    print(f"\n📋 Total samples in plan: {len(sampling_plan)}")

    return sampling_plan


def load_existing_labels():
    """
    Loads existing labels from Excel file if it exists.

    Returns:
        pd.DataFrame: Existing labels or empty DataFrame
    """
    if OUTPUT_FILE.exists():
        try:
            df = pd.read_excel(OUTPUT_FILE)
            print(f"\n✅ Loaded {len(df)} existing labels from {OUTPUT_FILE.name}")
            return df
        except Exception as e:
            print(f"\n⚠️  Warning: Could not load existing labels: {e}")
            return pd.DataFrame(columns=[
                'date', 'image_name', 'larva_filename',
                'is_valid_larva', 'is_valid_posture'
            ])
    else:
        print(f"\n📝 No existing labels found. Starting fresh.")
        return pd.DataFrame(columns=[
            'date', 'image_name', 'larva_filename',
            'is_valid_larva', 'is_valid_posture'
        ])


def save_label(labels_df, date, image_name, larva_filename, is_valid_larva, is_valid_posture):
    """
    Saves a single label to the DataFrame and Excel file.

    Args:
        labels_df: Current labels DataFrame
        date: Date folder name
        image_name: Image folder name
        larva_filename: Larva filename
        is_valid_larva: Boolean or 0/1
        is_valid_posture: Boolean or 0/1

    Returns:
        pd.DataFrame: Updated labels DataFrame
    """
    # Create new row
    new_row = pd.DataFrame([{
        'date': date,
        'image_name': image_name,
        'larva_filename': larva_filename,
        'is_valid_larva': int(is_valid_larva),
        'is_valid_posture': int(is_valid_posture)
    }])

    # Append to existing labels
    labels_df = pd.concat([labels_df, new_row], ignore_index=True)

    # Save to Excel
    try:
        labels_df.to_excel(OUTPUT_FILE, index=False)
    except Exception as e:
        print(f"\n❌ Error saving to Excel: {e}")
        # Fallback to CSV
        csv_file = OUTPUT_FILE.with_suffix('.csv')
        labels_df.to_csv(csv_file, index=False)
        print(f"   Saved to CSV instead: {csv_file}")

    return labels_df


def is_already_labeled(labels_df, date, image_name, larva_filename):
    """
    Checks if a larva has already been labeled.

    Args:
        labels_df: Labels DataFrame
        date: Date folder name
        image_name: Image folder name
        larva_filename: Larva filename

    Returns:
        bool: True if already labeled
    """
    if labels_df.empty:
        return False

    match = labels_df[
        (labels_df['date'] == date) &
        (labels_df['image_name'] == image_name) &
        (labels_df['larva_filename'] == larva_filename)
    ]

    return len(match) > 0


# ======================== DISPLAY FUNCTIONS ========================

def display_larva(larva_path, date, image_name, larva_filename, current_idx, total_samples, labels_completed):
    """
    Displays a larva image with metadata overlay.

    Args:
        larva_path: Path to larva image
        date: Date folder name
        image_name: Image folder name
        larva_filename: Larva filename
        current_idx: Current sample index
        total_samples: Total number of samples
        labels_completed: Number of labels completed so far

    Returns:
        numpy.ndarray: Display image or None if failed
    """
    if not larva_path.exists():
        print(f"❌ Image not found: {larva_path}")
        return None

    # Load image
    img = cv2.imread(str(larva_path))
    if img is None:
        print(f"❌ Failed to load: {larva_path}")
        return None

    # Create display canvas
    canvas = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 240

    # Resize image to fit while maintaining aspect ratio
    img_h, img_w = img.shape[:2]
    display_area_h = DISPLAY_HEIGHT - 200
    display_area_w = DISPLAY_WIDTH - 100

    scale = min(display_area_w / img_w, display_area_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized_img = cv2.resize(img, (new_w, new_h))

    # Center the image
    y_offset = 150
    x_offset = (DISPLAY_WIDTH - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header
    header_text = f"LARVA QUALITY LABELING TOOL"
    cv2.putText(canvas, header_text, (50, 40), font, 1.0, (0, 0, 0), 2)

    # Progress
    progress_text = f"Progress: {labels_completed}/{total_samples} completed ({current_idx}/{total_samples} current)"
    cv2.putText(canvas, progress_text, (50, 75), font, 0.6, (0, 100, 0), 2)

    # Sample info
    cv2.putText(canvas, f"Date: {date}", (50, 110), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Image: {image_name}", (400, 110), font, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Larva: {larva_filename}", (800, 110), font, 0.7, (0, 0, 0), 2)

    # Instructions at bottom
    y_base = DISPLAY_HEIGHT - 100
    cv2.rectangle(canvas, (0, y_base - 10), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (220, 220, 220), -1)

    cv2.putText(canvas, "INSTRUCTIONS:", (50, y_base + 20), font, 0.7, (0, 0, 139), 2)
    cv2.putText(canvas, "1. Is this a VALID LARVA? (1 = Yes, 0 = No)", (50, y_base + 50), font, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "2. Is POSTURE valid for measurement? (1 = Yes, 0 = No)", (50, y_base + 75), font, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "Press 'q' to quit and save", (800, y_base + 62), font, 0.6, (139, 0, 0), 2)

    return canvas


def get_user_input(prompt_text):
    """
    Gets a binary user input (0 or 1) via keyboard.

    Args:
        prompt_text: Text to display in console

    Returns:
        int or str: 0, 1, or 'q' to quit
    """
    while True:
        print(f"\n{prompt_text}", end=" ", flush=True)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('1'):
            print("1 ✓")
            return 1
        elif key == ord('0'):
            print("0 ✗")
            return 0
        elif key == ord('q') or key == ord('Q'):
            print("q (quit)")
            return 'q'
        else:
            print(f"Invalid key. Press 0, 1, or q.")


# ======================== MAIN LABELING WORKFLOW ========================

def main():
    """
    Main labeling workflow.
    """
    print("\n" + "="*70)
    print("INTERACTIVE LARVA QUALITY LABELING TOOL")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*70)

    # Step 1: Collect all available larvae
    larvae_inventory = collect_all_larvae()

    if not larvae_inventory:
        print("\n❌ No larvae found in analysis_full/")
        return

    # Step 2: Create sampling plan
    sampling_plan = create_sampling_plan(larvae_inventory)

    if not sampling_plan:
        print("\n❌ No samples selected!")
        return

    # Step 3: Load existing labels
    labels_df = load_existing_labels()

    # Step 4: Filter out already-labeled samples
    unlabeled_samples = [
        sample for sample in sampling_plan
        if not is_already_labeled(labels_df, sample[0], sample[1], sample[2])
    ]

    labels_completed = len(labels_df)
    total_samples = len(sampling_plan)

    print(f"\n{'='*70}")
    print(f"LABELING SESSION")
    print(f"{'='*70}")
    print(f"  Total samples in plan: {total_samples}")
    print(f"  Already labeled: {labels_completed}")
    print(f"  Remaining to label: {len(unlabeled_samples)}")
    print(f"{'='*70}")

    if not unlabeled_samples:
        print("\n✅ All samples have been labeled! Nothing to do.")
        print(f"Final results saved in: {OUTPUT_FILE}")
        return

    # Step 5: Create display window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print("\n🖼️  Display window opened. Starting labeling...\n")
    print("="*70)

    # Step 6: Label each sample
    try:
        for idx, (date, image_name, larva_filename, larva_path) in enumerate(unlabeled_samples, 1):
            current_idx = labels_completed + idx

            print(f"\n{'='*70}")
            print(f"Sample {current_idx}/{total_samples}: {date}/{image_name}/{larva_filename}")
            print(f"{'='*70}")

            # Display the larva
            display_img = display_larva(
                larva_path, date, image_name, larva_filename,
                current_idx, total_samples, labels_completed
            )

            if display_img is None:
                print("⚠️  Skipping due to image load error.")
                continue

            cv2.imshow(WINDOW_NAME, display_img)
            cv2.waitKey(1)  # Force refresh

            # Get user input for valid larva
            valid_larva = get_user_input("Is this a VALID LARVA? (1=Yes, 0=No, q=Quit):")

            if valid_larva == 'q':
                print("\n⚠️  User requested quit.")
                break

            # Get user input for valid posture
            valid_posture = get_user_input("Is POSTURE valid for measurement? (1=Yes, 0=No, q=Quit):")

            if valid_posture == 'q':
                print("\n⚠️  User requested quit.")
                break

            # Save the label
            labels_df = save_label(
                labels_df, date, image_name, larva_filename,
                valid_larva, valid_posture
            )

            labels_completed += 1

            print(f"✅ Label saved: valid_larva={valid_larva}, valid_posture={valid_posture}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")

    finally:
        cv2.destroyAllWindows()

        print("\n" + "="*70)
        print("LABELING SESSION COMPLETE")
        print("="*70)
        print(f"  Total labels saved: {len(labels_df)}")
        print(f"  Output file: {OUTPUT_FILE}")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # Summary statistics
        if len(labels_df) > 0:
            valid_larvae = labels_df['is_valid_larva'].sum()
            valid_postures = labels_df['is_valid_posture'].sum()

            print("\n📊 SUMMARY STATISTICS:")
            print(f"  Valid larvae: {valid_larvae}/{len(labels_df)} ({100*valid_larvae/len(labels_df):.1f}%)")
            print(f"  Valid postures: {valid_postures}/{len(labels_df)} ({100*valid_postures/len(labels_df):.1f}%)")
            print("="*70)


# ======================== ENTRY POINT ========================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

