#!/usr/bin/env python3
"""
Automated Larva Analysis Pipeline
==================================
Processes all date folders and images in the project directory.
Generates hierarchical output with segmentation masks, overlays,
morphometric reports, and comprehensive CSV summaries.

Author: Research Pipeline
Date: 2026-02-28
"""

import cv2
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import euclidean
import traceback

# ======================== CONFIGURATION ========================
ROOT_DIR = Path(__file__).parent
OUTPUT_ROOT = ROOT_DIR / 'analysis_full'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Processing Parameters
BLACKHAT_KERNEL_SIZE = 25
MIN_LARVA_AREA = 80
MAX_AREA_FRACTION = 0.05
SOLIDITY_THRESHOLD = 0.85
ELONGATION_THRESHOLD = 1.5
BLACKHAT_INTENSITY_THRESHOLD = 40

# Valid image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# ======================== SEGMENTATION ========================

def segment_larvae(gray_img):
    """
    Segments larvae from grayscale image using blackhat morphology
    and connected components analysis.

    Args:
        gray_img: Grayscale input image (numpy array)

    Returns:
        tuple: (binary_mask, num_labels, labels_img, stats, centroids)
    """
    h, w = gray_img.shape

    # Background estimation and contrast enhancement
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    bg_estimate = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.absdiff(bg_estimate, gray_img)

    # Blackhat transform for dark objects
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


def filter_larva_components(num_labels, labels_img, stats, gray_img, blackhat, img_shape):
    """
    Filters connected components to identify valid larvae based on
    geometric and intensity criteria.

    Args:
        num_labels: Number of connected components
        labels_img: Labeled image from connected components
        stats: Statistics for each component
        gray_img: Original grayscale image
        blackhat: Blackhat transformed image
        img_shape: Shape of the image (h, w)

    Returns:
        list: List of valid larva component indices
    """
    h, w = img_shape
    max_area = h * w * MAX_AREA_FRACTION
    valid_larvae = []

    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]

        # Size filter
        if area < MIN_LARVA_AREA or area > max_area:
            continue

        # Create component mask
        comp_mask = (labels_img == i).astype(np.uint8)

        # Geometry analysis
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Elongation via ellipse fitting
        elongation = 1.0
        if len(cnt) >= 5:
            try:
                _, (minor_axis, major_axis), _ = cv2.fitEllipse(cnt)
                elongation = major_axis / minor_axis if minor_axis > 0 else 1.0
            except:
                pass

        # Intensity check
        mean_blackhat = cv2.mean(blackhat, mask=comp_mask)[0]

        # Classification criteria
        is_elongated = (solidity < SOLIDITY_THRESHOLD and
                       elongation > ELONGATION_THRESHOLD)
        is_high_contrast = mean_blackhat > BLACKHAT_INTENSITY_THRESHOLD

        if is_elongated or is_high_contrast:
            valid_larvae.append(i)

    return valid_larvae


# ======================== MORPHOMETRICS ========================

def calculate_larva_morphometrics(larva_idx, component_mask, gray_img, blackhat_img):
    """
    Calculates comprehensive morphometric measurements for a single larva.

    Args:
        larva_idx: Unique identifier for the larva
        component_mask: Binary mask of the larva
        gray_img: Original grayscale image
        blackhat_img: Blackhat transformed image

    Returns:
        dict: Dictionary containing all morphometric measurements
    """
    # Skimage regionprops
    labeled_mask = label(component_mask)
    props = regionprops(labeled_mask, intensity_image=gray_img)[0]
    bh_props = regionprops(labeled_mask, intensity_image=blackhat_img)[0]

    # Skeleton analysis
    skeleton = skeletonize(component_mask > 0)
    skeleton_length = np.sum(skeleton)

    # Distance transform for width analysis
    dist_map = distance_transform_edt(component_mask)
    widths = dist_map[skeleton] * 2  # Convert radius to diameter

    # Endpoint detection for curvature
    skel_coords = np.argwhere(skeleton)
    endpoints = []
    for r, c in skel_coords:
        neighbor_count = np.sum(skeleton[max(0, r-1):r+2, max(0, c-1):c+2]) - 1
        if neighbor_count == 1:
            endpoints.append((r, c))

    # Curvature calculation
    euclidean_length = 0
    curvature_ratio = 1.0
    if len(endpoints) >= 2:
        p1, p2 = endpoints[0], endpoints[-1]
        euclidean_length = euclidean(p1, p2)
        curvature_ratio = skeleton_length / euclidean_length if euclidean_length > 0 else 1.0

    # Contour-based metrics
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = cv2.arcLength(contours[0], True) if contours else 0

    # Compile measurements
    metrics = {
        'larva_id': larva_idx,
        'area': props.area,
        'perimeter': round(perimeter, 2),
        'body_length': round(skeleton_length, 2),
        'euclidean_length': round(euclidean_length, 2),
        'curvature_ratio': round(curvature_ratio, 3),
        'mean_width': round(np.mean(widths), 2) if len(widths) > 0 else 0,
        'max_width': round(np.max(widths), 2) if len(widths) > 0 else 0,
        'min_width': round(np.min(widths), 2) if len(widths) > 0 else 0,
        'major_axis': round(getattr(props, 'axis_major_length', props.major_axis_length), 2),
        'minor_axis': round(getattr(props, 'axis_minor_length', props.minor_axis_length), 2),
        'eccentricity': round(props.eccentricity, 3),
        'orientation_deg': round(np.degrees(props.orientation), 1),
        'solidity': round(props.solidity, 3),
        'extent': round(props.extent, 3),
        'mean_intensity': round(getattr(props, 'intensity_mean', props.mean_intensity), 2),
        'std_intensity': round(np.std(gray_img[component_mask > 0]), 2),
        'mean_blackhat': round(getattr(bh_props, 'intensity_mean', bh_props.mean_intensity), 2),
        'centroid_x': round(props.centroid[1], 1),
        'centroid_y': round(props.centroid[0], 1),
    }

    return metrics, skeleton


# ======================== VISUALIZATION ========================

def create_larva_report(larva_id, gray_crop, mask_crop, skeleton_crop, metrics):
    """
    Creates a visual report for a single larva with overlays and metrics.

    Args:
        larva_id: Unique identifier
        gray_crop: Cropped grayscale image
        mask_crop: Cropped binary mask
        skeleton_crop: Cropped skeleton
        metrics: Dictionary of measurements

    Returns:
        numpy.ndarray: Report image
    """
    h, w = gray_crop.shape

    # Create three-panel visualization
    panel_original = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    panel_mask = cv2.cvtColor(mask_crop * 255, cv2.COLOR_GRAY2BGR)
    panel_overlay = panel_original.copy()

    # Overlay skeleton
    panel_overlay[skeleton_crop > 0] = [0, 0, 255]

    # Draw contours
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_overlay, contours, -1, (0, 255, 0), 1)

    # Combine panels
    visual = np.hstack((panel_original, panel_mask, panel_overlay))

    # Create text panel
    text_width = 300
    text_height = max(visual.shape[0], 400)
    report_img = np.zeros((text_height, visual.shape[1] + text_width, 3), dtype=np.uint8)
    report_img[:visual.shape[0], :visual.shape[1]] = visual

    # Render metrics
    text_lines = [
        f"Larva ID: {larva_id}",
        f"Area: {metrics['area']}",
        f"Body Length: {metrics['body_length']}",
        f"Mean Width: {metrics['mean_width']}",
        f"Curvature: {metrics['curvature_ratio']}",
        f"Eccentricity: {metrics['eccentricity']}",
        f"Orientation: {metrics['orientation_deg']}°",
        f"Solidity: {metrics['solidity']}",
        f"Intensity: {metrics['mean_intensity']}",
        f"Perimeter: {metrics['perimeter']}",
    ]

    y_offset = 30
    for line in text_lines:
        cv2.putText(
            report_img, line,
            (visual.shape[1] + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        y_offset += 25

    return report_img


def create_overlay_visualization(gray_img, labels_img, valid_larvae, stats):
    """
    Creates an overlay visualization showing all detected larvae.

    Args:
        gray_img: Original grayscale image
        labels_img: Labeled components image
        valid_larvae: List of valid larva indices
        stats: Component statistics

    Returns:
        numpy.ndarray: Overlay image with bounding boxes and labels
    """
    overlay = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    for idx, larva_comp_id in enumerate(valid_larvae, start=1):
        x = stats[larva_comp_id, cv2.CC_STAT_LEFT]
        y = stats[larva_comp_id, cv2.CC_STAT_TOP]
        w = stats[larva_comp_id, cv2.CC_STAT_WIDTH]
        h = stats[larva_comp_id, cv2.CC_STAT_HEIGHT]

        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label
        cv2.putText(
            overlay, str(idx), (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    return overlay


# ======================== BATCH PROCESSING ========================

def process_single_image(image_path, output_dir, date_folder_name):
    """
    Processes a single image through the complete analysis pipeline.

    Args:
        image_path: Path to input image
        output_dir: Output directory for this image
        date_folder_name: Name of the date folder

    Returns:
        dict: Summary statistics for this image
    """
    print(f"    Processing: {image_path.name}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"    ⚠ Failed to load image: {image_path.name}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Segmentation
    _, num_labels, labels_img, stats, centroids, blackhat = segment_larvae(gray)

    # Filter valid larvae
    valid_larvae = filter_larva_components(
        num_labels, labels_img, stats, gray, blackhat, gray.shape
    )

    print(f"    ✓ Detected {len(valid_larvae)} larvae")

    if len(valid_larvae) == 0:
        return {
            'date': date_folder_name,
            'image_name': image_path.stem,
            'number_of_larvae': 0,
            'mean_body_length': 0,
            'mean_area': 0,
            'mean_intensity': 0,
        }

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / 'larvae_reports'
    reports_dir.mkdir(exist_ok=True)

    # Create final mask
    final_mask = np.zeros_like(gray)
    for larva_comp_id in valid_larvae:
        final_mask[labels_img == larva_comp_id] = 255

    # Save segmentation mask
    cv2.imwrite(str(output_dir / 'segmentation_mask.png'), final_mask)

    # Create and save overlay
    overlay = create_overlay_visualization(gray, labels_img, valid_larvae, stats)
    cv2.imwrite(str(output_dir / 'overlay.png'), overlay)

    # Process each larva
    all_metrics = []

    for idx, larva_comp_id in enumerate(valid_larvae, start=1):
        # Extract component
        comp_mask = (labels_img == larva_comp_id).astype(np.uint8)

        # Calculate morphometrics
        metrics, skeleton = calculate_larva_morphometrics(
            idx, comp_mask, gray, blackhat
        )
        all_metrics.append(metrics)

        # Create cropped images for report
        x = stats[larva_comp_id, cv2.CC_STAT_LEFT]
        y = stats[larva_comp_id, cv2.CC_STAT_TOP]
        w = stats[larva_comp_id, cv2.CC_STAT_WIDTH]
        h = stats[larva_comp_id, cv2.CC_STAT_HEIGHT]

        pad = 10
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

        gray_crop = gray[y1:y2, x1:x2]
        mask_crop = comp_mask[y1:y2, x1:x2]
        skeleton_crop = skeleton[y1:y2, x1:x2]

        # Generate report
        report_img = create_larva_report(
            idx, gray_crop, mask_crop, skeleton_crop, metrics
        )
        cv2.imwrite(str(reports_dir / f'larva_{idx:03d}.png'), report_img)

    # Save morphometrics CSV
    if all_metrics:
        csv_path = output_dir / 'morphometrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)

    # Calculate summary statistics
    summary = {
        'date': date_folder_name,
        'image_name': image_path.stem,
        'number_of_larvae': len(all_metrics),
        'mean_body_length': round(np.mean([m['body_length'] for m in all_metrics]), 2),
        'mean_area': round(np.mean([m['area'] for m in all_metrics]), 2),
        'mean_intensity': round(np.mean([m['mean_intensity'] for m in all_metrics]), 2),
    }

    return summary


def process_date_folder(date_folder_path, output_root):
    """
    Processes all images in a single date folder.

    Args:
        date_folder_path: Path to date folder
        output_root: Root output directory

    Returns:
        list: List of summary dictionaries for all images
    """
    date_name = date_folder_path.name
    images_dir = date_folder_path / 'images'

    if not images_dir.exists():
        print(f"  ⚠ No 'images' subfolder in {date_name}")
        return []

    # Get all image files
    image_files = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"  ⚠ No images found in {date_name}/images")
        return []

    print(f"\n  Processing {len(image_files)} images in {date_name}")

    # Create output directory for this date
    date_output_dir = output_root / date_name
    date_output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for image_path in sorted(image_files):
        try:
            # Create output directory for this image
            image_output_dir = date_output_dir / image_path.stem

            # Process image
            summary = process_single_image(image_path, image_output_dir, date_name)

            if summary:
                summaries.append(summary)

        except Exception as e:
            print(f"    ✗ Error processing {image_path.name}: {str(e)}")
            traceback.print_exc()
            continue

    return summaries


def run_batch_pipeline():
    """
    Main function to run the complete batch processing pipeline.
    """
    print("=" * 70)
    print("AUTOMATED LARVA ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Output Directory: {OUTPUT_ROOT}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Detect all date folders
    date_folders = [
        d for d in ROOT_DIR.iterdir()
        if d.is_dir() and d.name[0].isdigit() and '.' in d.name
    ]

    if not date_folders:
        print("\n⚠ No date folders found!")
        return

    print(f"\nFound {len(date_folders)} date folders:")
    for df in sorted(date_folders):
        print(f"  - {df.name}")

    # Process all date folders
    all_summaries = []

    for date_folder in sorted(date_folders):
        print(f"\n{'='*70}")
        print(f"Processing folder: {date_folder.name}")
        print(f"{'='*70}")

        summaries = process_date_folder(date_folder, OUTPUT_ROOT)
        all_summaries.extend(summaries)

    # Save global summary CSV
    if all_summaries:
        global_csv_path = OUTPUT_ROOT / 'global_summary.csv'
        with open(global_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)

        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Total images processed: {len(all_summaries)}")
        print(f"Total larvae detected: {sum(s['number_of_larvae'] for s in all_summaries)}")
        print(f"Global summary saved to: {global_csv_path}")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    else:
        print("\n⚠ No images were successfully processed.")


# ======================== MAIN ========================

if __name__ == "__main__":
    run_batch_pipeline()

