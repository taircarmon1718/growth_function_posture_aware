#!/usr/bin/env python3
"""
Segmentation Experiments Analysis Script
=========================================

Runs controlled experiments on the larva segmentation pipeline to analyze
filtering effectiveness and identify sources of false detections.

This script:
- Does NOT modify any existing pipelines
- Does NOT overwrite any existing output directories
- Saves all experiments to segmentation_experiments/
- Uses the existing pipeline as baseline for comparison

Experiments:
  1. Baseline - Current pipeline performance
  2. Size Outlier Detection - Area distribution analysis
  3. Circularity Analysis - Identify round/bubble artifacts
  4. Elongation Analysis - Component shape analysis
  5. Combined Feature Visualization - Multi-feature scatter plots
  6. False Detection Inspection - Visual examples of rejected components
  7. Solidity Analysis - Component convexity patterns
  8. Feature Correlation - Relationship between morphometric features

Author: Research Pipeline
Date: 2026-03-15
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import warnings

warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================

ROOT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = ROOT_DIR / 'segmentation_experiments'
COMPONENT_EXAMPLES_DIR = EXPERIMENTS_DIR / 'component_examples'
DATA_DIR = ROOT_DIR / 'analysis_full_binary_masks_only'

# Create output directories
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
COMPONENT_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline Parameters (from run_pipeline_binary_masks.py)
BLACKHAT_KERNEL_SIZE = 25
MIN_LARVA_AREA = 80
MAX_AREA_FRACTION = 0.05
SOLIDITY_THRESHOLD = 0.85
ELONGATION_THRESHOLD = 1.5
BLACKHAT_INTENSITY_THRESHOLD = 40

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# Experiment parameters
MAX_IMAGES_PER_DATE = 50  # Limit for faster analysis
SAMPLE_SIZE = 10  # Number of components to visualize in each category
NUM_HISTOGRAMS = 50  # Bins for histogram plots


# ======================== SEGMENTATION (BASELINE) ========================

def segment_larvae(gray_img):
    """Segment larvae using the baseline pipeline"""
    h, w = gray_img.shape

    # Background estimation
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    bg_estimate = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.absdiff(bg_estimate, gray_img)

    # Blackhat transform
    bh_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE)
    )
    blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, bh_kernel)

    # Combine features
    saliency = cv2.addWeighted(diff, 0.7, blackhat, 0.3, 0)

    # Threshold
    _, thresh = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)

    # Connected components
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    return thresh, num_labels, labels_img, stats, centroids, blackhat


def filter_larva_components_baseline(num_labels, labels_img, stats, gray_img, blackhat, img_shape):
    """Baseline filtering logic from run_pipeline_binary_masks.py"""
    h, w = img_shape
    max_area = h * w * MAX_AREA_FRACTION
    valid_larvae = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_LARVA_AREA or area > max_area:
            continue

        comp_mask = (labels_img == i).astype(np.uint8)

        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        elongation = 1.0
        if len(cnt) >= 5:
            try:
                _, (minor_axis, major_axis), _ = cv2.fitEllipse(cnt)
                elongation = major_axis / minor_axis if minor_axis > 0 else 1.0
            except:
                pass

        mean_blackhat = cv2.mean(blackhat, mask=comp_mask)[0]

        is_elongated = (solidity < SOLIDITY_THRESHOLD and
                       elongation > ELONGATION_THRESHOLD)
        is_high_contrast = mean_blackhat > BLACKHAT_INTENSITY_THRESHOLD

        # Current logic: OR (too permissive)
        if is_elongated or is_high_contrast:
            valid_larvae.append(i)

    return valid_larvae


# ======================== COMPONENT ANALYSIS ========================

def analyze_component(i, labels_img, stats, gray_img, blackhat_img):
    """
    Comprehensive analysis of a single component.
    Returns dict with all morphometric features.
    """
    area = stats[i, cv2.CC_STAT_AREA]
    comp_mask = (labels_img == i).astype(np.uint8)

    # Geometry
    contours, _ = cv2.findContours(
        comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Convexity (Solidity)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Elongation
    elongation = 1.0
    if len(cnt) >= 5:
        try:
            _, (minor_axis, major_axis), _ = cv2.fitEllipse(cnt)
            elongation = major_axis / minor_axis if minor_axis > 0 else 1.0
        except:
            pass

    # Intensity features
    mean_blackhat = cv2.mean(blackhat_img, mask=comp_mask)[0]
    mean_gray = cv2.mean(gray_img, mask=comp_mask)[0]
    std_gray = np.std(gray_img[comp_mask > 0])

    # Skeleton features
    skeleton = skeletonize(comp_mask > 0)
    skeleton_length = np.sum(skeleton)

    # Aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 1.0

    # Passes baseline filter?
    passes_filter = False
    is_elongated = (solidity < SOLIDITY_THRESHOLD and
                    elongation > ELONGATION_THRESHOLD)
    is_high_contrast = mean_blackhat > BLACKHAT_INTENSITY_THRESHOLD
    if is_elongated or is_high_contrast:
        passes_filter = True

    return {
        'component_id': i,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'solidity': solidity,
        'elongation': elongation,
        'aspect_ratio': aspect_ratio,
        'skeleton_length': skeleton_length,
        'mean_blackhat': mean_blackhat,
        'mean_gray': mean_gray,
        'std_gray': std_gray,
        'passes_filter': passes_filter,
        'is_elongated': is_elongated,
        'is_high_contrast': is_high_contrast,
    }


# ======================== EXPERIMENT 1: BASELINE ========================

def experiment_baseline(images_dirs):
    """Run baseline pipeline and record filtering statistics"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BASELINE PIPELINE")
    print("=" * 70)

    results = []

    for date_dir in sorted(images_dirs):
        if not date_dir.is_dir():
            continue

        date_name = date_dir.name

        # Discover image files using helper
        image_files = get_image_files(date_dir)[:MAX_IMAGES_PER_DATE]
        analysis_path = DATA_DIR / date_name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:MAX_IMAGES_PER_DATE]

        if not image_files:
            continue

        print(f"\n  {date_name}: {len(image_files)} images")

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, num_labels, labels_img, stats, _, blackhat = segment_larvae(gray)
                total_components = num_labels - 1

                valid_larvae = filter_larva_components_baseline(
                    num_labels, labels_img, stats, gray, blackhat, gray.shape
                )

                # Area statistics (from detected components)
                all_areas = []
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    h, w = gray.shape
                    max_area = h * w * MAX_AREA_FRACTION

                    if area < MIN_LARVA_AREA or area > max_area:
                        continue
                    all_areas.append(area)

                area_min = np.min(all_areas) if all_areas else 0
                area_median = np.median(all_areas) if all_areas else 0
                area_max = np.max(all_areas) if all_areas else 0

                results.append({
                    'date': date_name,
                    'image_name': img_file.stem,
                    'total_components': total_components,
                    'components_kept': len(valid_larvae),
                    'components_rejected': total_components - len(valid_larvae),
                    'retention_rate_pct': round(100 * len(valid_larvae) / max(total_components, 1), 1),
                    'area_min': area_min,
                    'area_median': area_median,
                    'area_max': area_max,
                })

                print(f"    {img_file.stem}: {total_components} → {len(valid_larvae)} ({100*len(valid_larvae)//max(total_components,1)}%)")

            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
                continue

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = EXPERIMENTS_DIR / 'baseline_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n  ✓ Baseline results saved: {output_file.name}")

        # Summary statistics
        print(f"\n  Summary Statistics:")
        print(f"    Total images analyzed: {len(results)}")
        print(f"    Mean components/image: {results_df['total_components'].mean():.1f}")
        print(f"    Mean retention rate: {results_df['retention_rate_pct'].mean():.1f}%")
        print(f"    Mean area: {results_df['area_median'].mean():.1f} px²")

        return results_df
    else:
        print("  ⚠ No results collected")
        return pd.DataFrame()


# ======================== EXPERIMENT 2: SIZE ANALYSIS ========================

def experiment_size_distribution(images_dirs):
    """Analyze area distribution and identify outliers"""
        # Discover image files using helper
        image_files = get_image_files(date_dir)[:MAX_IMAGES_PER_DATE]
    print("=" * 70)

    all_areas = []
    all_components = []

    for date_dir in sorted(images_dirs)[:10]:  # Limit to first 10 dates for speed
        if not date_dir.is_dir():
            continue

        # Check for images in either direct folder or analysis_full_binary_masks_only
        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:MAX_IMAGES_PER_DATE]

        print(f"  {date_dir.name}: {len(image_files)} images")

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, num_labels, labels_img, stats, _, _ = segment_larvae(gray)

                h, w = gray.shape
                max_area = h * w * MAX_AREA_FRACTION

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]

                    if area < MIN_LARVA_AREA or area > max_area:
                        continue

                    all_areas.append(area)
                    all_components.append({
                        'area': area,
                        'image': img_file.stem,
                        'date': date_dir.name,
                    })

            except Exception as e:
                continue

    if not all_areas:
        print("  ⚠ No components found")
        return

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = {p: np.percentile(all_areas, p) for p in percentiles}

    print(f"\n  Area Distribution Percentiles:")
    for p, v in percentile_values.items():
        print(f"    P{p}: {v:.1f} px²")

    # Save statistics
    stats_df = pd.DataFrame([
        {'percentile': p, 'area_px2': v}
        for p, v in percentile_values.items()
    ])
        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
        # Discover image files using helper
        image_files = get_image_files(date_dir)[:MAX_IMAGES_PER_DATE]
    plt.savefig(EXPERIMENTS_DIR / 'area_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ Saved: area_distribution.csv, area_histogram.png")


# ======================== EXPERIMENT 3: CIRCULARITY ANALYSIS ========================

def experiment_circularity(images_dirs):
    """Analyze circularity of components to identify non-larva objects"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CIRCULARITY ANALYSIS")
    print("=" * 70)

    components_data = []

    for date_dir in sorted(images_dirs)[:10]:
        if not date_dir.is_dir():
            continue

        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:MAX_IMAGES_PER_DATE]

        print(f"  {date_dir.name}: {len(image_files)} images")

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, num_labels, labels_img, stats, _, blackhat = segment_larvae(gray)

                for i in range(1, num_labels):
                    analysis = analyze_component(i, labels_img, stats, gray, blackhat)
                    if analysis:
                        analysis['image'] = img_file.stem
                        analysis['date'] = date_dir.name
                        components_data.append(analysis)

            except Exception as e:
                continue

    if not components_data:
        print("  ⚠ No components analyzed")
        return

    df = pd.DataFrame(components_data)
        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:MAX_IMAGES_PER_DATE]
    df.to_csv(EXPERIMENTS_DIR / 'circularity_distribution.csv', index=False)

    # Summary statistics
    print(f"\n  Circularity Statistics:")
    print(f"    Components analyzed: {len(df)}")
    print(f"    Mean circularity: {df['circularity'].mean():.3f}")
    print(f"    Std circularity: {df['circularity'].std():.3f}")
    print(f"    Median circularity: {df['circularity'].median():.3f}")
    print(f"    Min circularity: {df['circularity'].min():.3f}")
    print(f"    Max circularity: {df['circularity'].max():.3f}")
        # Discover image files using helper
        image_files = get_image_files(date_dir)[:MAX_IMAGES_PER_DATE]
    plt.savefig(EXPERIMENTS_DIR / 'circularity_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Filter analysis
    high_circularity = df[df['circularity'] > 0.6]
    low_circularity = df[df['circularity'] <= 0.6]

    print(f"\n  High circularity (>0.6): {len(high_circularity)} ({100*len(high_circularity)//len(df)}%)")
    print(f"    Passing filter: {high_circularity['passes_filter'].sum()}")
    print(f"    Failing filter: {len(high_circularity) - high_circularity['passes_filter'].sum()}")

    print(f"\n  ✓ Saved: circularity_distribution.csv, circularity_histogram.png")

    return df


# ======================== EXPERIMENT 4: ELONGATION ANALYSIS ========================

def experiment_elongation(images_dirs):
    """Analyze elongation distribution"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: ELONGATION ANALYSIS")
    print("=" * 70)

    components_data = []

    for date_dir in sorted(images_dirs)[:10]:
        if not date_dir.is_dir():
            continue

        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue
        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:MAX_IMAGES_PER_DATE]
                      if f.suffix in IMAGE_EXTENSIONS][:MAX_IMAGES_PER_DATE]

        print(f"  {date_dir.name}: {len(image_files)} images")

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, num_labels, labels_img, stats, _, blackhat = segment_larvae(gray)

                for i in range(1, num_labels):
                    analysis = analyze_component(i, labels_img, stats, gray, blackhat)
                    if analysis:
                        components_data.append(analysis)

            except Exception as e:
                continue
        # Discover image files using helper
        image_files = get_image_files(date_dir)[:MAX_IMAGES_PER_DATE]
    print(f"    Components analyzed: {len(df)}")
    print(f"    Mean elongation: {df['elongation'].mean():.2f}")
    print(f"    Std elongation: {df['elongation'].std():.2f}")
    print(f"    Median elongation: {df['elongation'].median():.2f}")
    print(f"    Min elongation: {df['elongation'].min():.2f}")
    print(f"    Max elongation: {df['elongation'].max():.2f}")

    # Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['elongation'], bins=NUM_HISTOGRAMS, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(df['elongation'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["elongation"].mean():.2f}')
    ax.axvline(ELONGATION_THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Current threshold: {ELONGATION_THRESHOLD}')
    ax.set_xlabel('Elongation (major/minor axis ratio)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Elongation Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / 'elongation_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ Saved: elongation_distribution.csv, elongation_histogram.png")

    return df


# ======================== EXPERIMENT 5: COMBINED FEATURES ========================

def experiment_combined_features(images_dirs):
    """Create multi-dimensional scatter plots"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: COMBINED FEATURE VISUALIZATION")
    print("=" * 70)

    components_data = []

    for date_dir in sorted(images_dirs)[:10]:
        if not date_dir.is_dir():
            continue

        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:MAX_IMAGES_PER_DATE]

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, num_labels, labels_img, stats, _, blackhat = segment_larvae(gray)

                for i in range(1, num_labels):
                    analysis = analyze_component(i, labels_img, stats, gray, blackhat)
                    if analysis:
                        components_data.append(analysis)

            except Exception as e:
                continue

    if not components_data:
        print("  ⚠ No components analyzed")
        return

        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:10]
    print(f"  Components analyzed: {len(df)}")

    # Create multi-panel scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Area vs Elongation
    ax = axes[0, 0]
    passed = df[df['passes_filter']]
    rejected = df[~df['passes_filter']]
    ax.scatter(rejected['area'], rejected['elongation'], alpha=0.5, s=30, color='red', label='Rejected')
    ax.scatter(passed['area'], passed['elongation'], alpha=0.5, s=30, color='green', label='Passed')
    ax.set_xlabel('Area (px²)', fontweight='bold')
    ax.set_ylabel('Elongation', fontweight='bold')
    ax.set_title('Area vs Elongation')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Area vs Circularity
    ax = axes[0, 1]
    ax.scatter(rejected['area'], rejected['circularity'], alpha=0.5, s=30, color='red', label='Rejected')
    ax.scatter(passed['area'], passed['circularity'], alpha=0.5, s=30, color='green', label='Passed')
    ax.set_xlabel('Area (px²)', fontweight='bold')
    ax.set_ylabel('Circularity', fontweight='bold')
    ax.set_title('Area vs Circularity')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Area vs Solidity
    ax = axes[0, 2]
    ax.scatter(rejected['area'], rejected['solidity'], alpha=0.5, s=30, color='red', label='Rejected')
        # Discover image files using helper
        image_files = get_image_files(date_dir)[:10]
    ax.set_ylabel('Circularity', fontweight='bold')
    ax.set_title('Elongation vs Circularity')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 5: Solidity vs Elongation
    ax = axes[1, 1]
    ax.scatter(rejected['solidity'], rejected['elongation'], alpha=0.5, s=30, color='red', label='Rejected')
    ax.scatter(passed['solidity'], passed['elongation'], alpha=0.5, s=30, color='green', label='Passed')
    ax.set_xlabel('Solidity', fontweight='bold')
    ax.set_ylabel('Elongation', fontweight='bold')
    ax.set_title('Solidity vs Elongation')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 6: Skeleton Length vs Area
    ax = axes[1, 2]
    ax.scatter(rejected['area'], rejected['skeleton_length'], alpha=0.5, s=30, color='red', label='Rejected')
    ax.scatter(passed['area'], passed['skeleton_length'], alpha=0.5, s=30, color='green', label='Passed')
    ax.set_xlabel('Area (px²)', fontweight='bold')
    ax.set_ylabel('Skeleton Length (px)', fontweight='bold')
    ax.set_title('Area vs Skeleton Length')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Combined Feature Analysis: Passed (Green) vs Rejected (Red)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    print("=" * 70)

    components_data = []
    component_images = {}  # Store component images with metadata

    for date_dir in sorted(images_dirs)[:5]:  # Only first 5 dates for speed
        if not date_dir.is_dir():
            continue

        images_path = date_dir / 'images'
        analysis_path = IMAGES_DIR / date_dir.name / 'images'

        if images_path.exists():
            search_path = images_path
        elif analysis_path.exists():
            search_path = analysis_path
        else:
            continue

        image_files = [f for f in search_path.iterdir()
                      if f.suffix in IMAGE_EXTENSIONS][:10]

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, num_labels, labels_img, stats, _, blackhat = segment_larvae(gray)

                for i in range(1, num_labels):
                    analysis = analyze_component(i, labels_img, stats, gray, blackhat)
                    if analysis:
                        analysis['image_file'] = str(img_file)
                        analysis['component_idx'] = i
                        analysis['labels_img'] = labels_img.copy()
                        analysis['gray_img'] = gray.copy()
                        components_data.append(analysis)
                        component_images[len(components_data) - 1] = {
                            'comp_mask': (labels_img == i).astype(np.uint8),
                            'gray': gray,
                        }

            except Exception as e:
                continue

    if not components_data:
        print("  ⚠ No components analyzed")
        return

    df = pd.DataFrame(components_data)

    # Identify categories
    smallest_5pct_idx = df['area'].nsmallest(max(1, len(df) // 20)).index.tolist()
    largest_5pct_idx = df['area'].nlargest(max(1, len(df) // 20)).index.tolist()
        grid_out = COMPONENT_EXAMPLES_DIR / f'component_examples_{category}.png'
        plt.savefig(grid_out, dpi=100, bbox_inches='tight')
    rejected_idx = df[~df['passes_filter']].index[:20].tolist()

    categories = {
        'smallest_5pct': smallest_5pct_idx[:SAMPLE_SIZE],
        # Also save individual crop images for quick inspection
        for plot_idx, comp_idx in enumerate(indices):
            if comp_idx in component_images:
                comp_mask = component_images[comp_idx]['comp_mask']
                gray = component_images[comp_idx]['gray']
                x, y, w, h = cv2.boundingRect(comp_mask)
                crop = gray[y:y+h, x:x+w]
                crop_fn = COMPONENT_EXAMPLES_DIR / f'{category}_sample_{plot_idx+1:02d}.png'
                try:
                    cv2.imwrite(str(crop_fn), (crop).astype(np.uint8))
                except Exception:
                    pass

        'largest_5pct': largest_5pct_idx[:SAMPLE_SIZE],
        'most_circular': most_circular_idx[:SAMPLE_SIZE],
        'rejected_high_area': rejected_idx[:SAMPLE_SIZE],
    }

    print(f"  Components analyzed: {len(df)}")
    print(f"  Rejected components: {(~df['passes_filter']).sum()}")

    # Visualize each category
    for category, indices in categories.items():
        if not indices:
            continue

        n_samples = len(indices)
        fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(4 * ((n_samples + 1) // 2), 8))
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for plot_idx, comp_idx in enumerate(indices):
            row = df.iloc[comp_idx]
    # Try analysis_full_binary_masks_only first
    for img_folder in (DATA_DIR / date_folder.name).iterdir():
        if not img_folder.is_dir():
            continue
        img_path = img_folder / 'images'
        if img_path.exists():
            files = [f for f in img_path.iterdir() if f.suffix in IMAGE_EXTENSIONS]
            if files:
                return files
    return []

    if df is None or len(df) == 0:
        print("  ⚠ No data available")
        return

    print(f"  Solidity Statistics:")
    print(f"    Mean solidity: {df['solidity'].mean():.3f}")
    print(f"    Std solidity: {df['solidity'].std():.3f}")
    print(f"    Min solidity: {df['solidity'].min():.3f}")
    print(f"    Max solidity: {df['solidity'].max():.3f}")

    # Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['solidity'], bins=NUM_HISTOGRAMS, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(SOLIDITY_THRESHOLD, color='red', linestyle='--', linewidth=2,
               label=f'Current threshold: {SOLIDITY_THRESHOLD}')
    ax.axvline(df['solidity'].mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {df["solidity"].mean():.3f}')
    ax.set_xlabel('Solidity (Area / Hull Area)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Solidity Distribution (High = Compact, Low = Concave)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / 'solidity_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: solidity_histogram.png")


# ======================== EXPERIMENT 8: FEATURE CORRELATION ========================

def experiment_correlation(df):
    """Analyze correlation between morphometric features"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: FEATURE CORRELATION")
    print("=" * 70)

    if df is None or len(df) == 0:
        print("  ⚠ No data available")
        return

    # Select numeric columns
    features = ['area', 'perimeter', 'circularity', 'solidity', 'elongation',
                'aspect_ratio', 'skeleton_length', 'mean_blackhat', 'mean_gray', 'std_gray']

    available_features = [f for f in features if f in df.columns]
    corr_matrix = df[available_features].corr()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
    candidates = []

    # 1) date_folder/images
    p1 = date_folder / 'images'
    if p1.exists():
        candidates.extend([f for f in p1.iterdir() if f.suffix in IMAGE_EXTENSIONS])

    # 2) date_folder/*/images (analysis structure)
    p2_root = DATA_DIR / date_folder.name
    if p2_root.exists():
        for sub in p2_root.iterdir():
            if not sub.is_dir():
                continue
            p2 = sub / 'images'
            if p2.exists():
                candidates.extend([f for f in p2.iterdir() if f.suffix in IMAGE_EXTENSIONS])

    # 3) date_folder/*/larvae_reports (binary masks or reports)
    if p2_root.exists():
        for sub in p2_root.iterdir():
            if not sub.is_dir():
                continue
            p3 = sub / 'larvae_reports'
            if p3.exists():
                candidates.extend([f for f in p3.iterdir() if f.suffix in IMAGE_EXTENSIONS])

    # 4) date_folder/larvae_reports
    p4 = date_folder / 'larvae_reports'
    if p4.exists():
        candidates.extend([f for f in p4.iterdir() if f.suffix in IMAGE_EXTENSIONS])

    # Deduplicate and sort
    unique = sorted(list({str(p): p for p in candidates}.values()))
    return unique
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"    {available_features[i]} ↔ {available_features[j]}: {corr_val:.3f}")


# ======================== HELPER FUNCTIONS ========================

def get_image_files(date_folder):
    """
    Find image files in a date folder.
    Images can be in date/images/ or analysis_full_binary_masks_only/date/*/images/
    """
    # Try analysis_full_binary_masks_only first
    for img_folder in (DATA_DIR / date_folder.name).iterdir():
        if not img_folder.is_dir():
            continue
        img_path = img_folder / 'images'
    print("\nGenerated files:")
    print("  - baseline_results.csv")
    print("  - area_distribution.csv / area_histogram.png")
    print("  - circularity_distribution.csv / circularity_histogram.png")
    print("  - elongation_distribution.csv / elongation_histogram.png")
    print("  - combined_features_scatter.png")
    print("  - component_examples_*.png")
    print("  - solidity_histogram.png")
    print("  - feature_correlation_heatmap.png")
1. HIGH CIRCULARITY (>0.6) typically indicates:
   - Bubbles, dust, or noise artifacts
   - NOT true larvae (which are elongated)
2. LOW SOLIDITY (<0.70) indicates concave shapes:
   - Could be clumps of larvae or segmentation errors
   → Consider: solidity > 0.75 for better specificity
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        return

    # Get all date folders
    # Write a concise summary text file of results
3. ELONGATION < 1.5 indicates near-circular components:
   - Not typical larva morphology
   → Recommend stricter threshold: elongation > 2.0

   - Combine with OR logic (current: too permissive)
    summary_lines = []
    try:
        total_images = int(len(baseline_df)) if baseline_df is not None and not baseline_df.empty else 0
    except Exception:
        total_images = 0
    total_components = int(len(circularity_df)) if 'circularity_df' in locals() and circularity_df is not None else 0
    components_analyzed = int(len(combined_df)) if combined_df is not None else 0
5. FEATURE CORRELATIONS show:
    summary_lines.append(f"Segmentation Experiments Summary")
    summary_lines.append(f"Output directory: {EXPERIMENTS_DIR}")
    summary_lines.append("")
    summary_lines.append(f"Total images processed (baseline): {total_images}")
    summary_lines.append(f"Total components detected (circularity analysis): {total_components}")
    summary_lines.append(f"Components analyzed for combined features: {components_analyzed}")
SUGGESTED IMPROVEMENTS:
    # key statistics
    if total_components > 0 and 'circularity_df' in locals() and circularity_df is not None:
        summary_lines.append("")
        summary_lines.append(f"Circularity mean: {circularity_df['circularity'].mean():.3f}")
        summary_lines.append(f"Circularity median: {circularity_df['circularity'].median():.3f}")
    if components_analyzed > 0 and combined_df is not None:
        summary_lines.append("")
        if 'area' in combined_df.columns:
            summary_lines.append(f"Area median: {combined_df['area'].median():.1f} px²")
        if 'elongation' in combined_df.columns:
            summary_lines.append(f"Elongation mean: {combined_df['elongation'].mean():.2f}")
  4. Increase solidity threshold
    # list saved files in component examples
    saved_examples = sorted([p.name for p in COMPONENT_EXAMPLES_DIR.iterdir()]) if COMPONENT_EXAMPLES_DIR.exists() else []
    summary_lines.append("")
    summary_lines.append(f"Component example files ({len(saved_examples)}):")
    for fn in saved_examples[:200]:
        summary_lines.append(f"  - {fn}")

    summary_text = "\n".join(summary_lines)
    summary_path = EXPERIMENTS_DIR / 'summary.txt'
    with open(summary_path, 'w') as fh:
        fh.write(summary_text)
    print("""
    # also copy to component examples dir for convenience
    try:
        with open(COMPONENT_EXAMPLES_DIR / 'summary.txt', 'w') as fh2:
            fh2.write(summary_text)
    except Exception:
        pass
   - Bubbles, dust, or noise artifacts
    print(f"\n  ✓ Saved summary: {summary_path}")

3. ELONGATION < 1.5 indicates near-circular components:
   - Not typical larva morphology
   → Recommend stricter threshold: elongation > 2.0

4. BLACKHAT INTENSITY is often low for false positives:
   - Combine with OR logic (current: too permissive)
   → Recommend AND logic: is_elongated AND is_high_contrast

5. FEATURE CORRELATIONS show:
   - Area and skeleton_length are highly correlated (good)
   - Circularity and elongation are anti-correlated (expected)

SUGGESTED IMPROVEMENTS:
  1. Change filter logic from OR to AND
  2. Add circularity check: exclude circular objects
  3. Increase elongation threshold
  4. Increase solidity threshold
    """)


if __name__ == '__main__':
    main()
