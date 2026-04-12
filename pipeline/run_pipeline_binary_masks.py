#!/usr/bin/env python3
"""
Automated Larva Analysis Pipeline - Binary Masks Only
======================================================
Processes all date folders and images in the project directory.
Generates hierarchical output with segmentation masks, overlays,
morphometric reports, and saves clean binary larva masks.

Modified version: Saves ONLY clean binary masks in larvae_reports/
instead of visualization reports.
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
OUTPUT_ROOT = ROOT_DIR / 'analysis_full_binary_masks_only'
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
            except Exception:
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
        'area': int(getattr(props, 'area', 0)),
        'perimeter': round(perimeter, 2),
        'body_length': round(float(skeleton_length), 2),
        'euclidean_length': round(float(euclidean_length), 2),
        'curvature_ratio': round(curvature_ratio, 3),
        'mean_width': round(float(np.mean(widths)), 2) if widths.size > 0 else 0.0,
        'max_width': round(float(np.max(widths)), 2) if widths.size > 0 else 0.0,
        'min_width': round(float(np.min(widths)), 2) if widths.size > 0 else 0.0,
        'major_axis': round(getattr(props, 'axis_major_length', 0.0), 2),
        'minor_axis': round(getattr(props, 'axis_minor_length', 0.0), 2),
        'eccentricity': round(getattr(props, 'eccentricity', 0.0), 3),
        'orientation_deg': round(np.degrees(getattr(props, 'orientation', 0.0)), 1),
        'solidity': round(getattr(props, 'solidity', 0.0), 3),
        'extent': round(getattr(props, 'extent', 0.0), 3),
        'mean_intensity': round(float(getattr(props, 'intensity_mean', 0.0)), 2),
        'std_intensity': round(float(np.std(gray_img[component_mask > 0])) if np.any(component_mask > 0) else 0.0, 2),
        'mean_blackhat': round(float(getattr(bh_props, 'intensity_mean', 0.0)), 2),
        'centroid_x': round(float(props.centroid[1]) if hasattr(props, 'centroid') else 0.0, 1),
        'centroid_y': round(float(props.centroid[0]) if hasattr(props, 'centroid') else 0.0, 1),
    }

    return metrics, skeleton


# ======================== VISUALIZATION ========================

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
        x = int(stats[larva_comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[larva_comp_id, cv2.CC_STAT_TOP])
        w = int(stats[larva_comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[larva_comp_id, cv2.CC_STAT_HEIGHT])

        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label
        cv2.putText(overlay, str(idx), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return overlay


def create_larva_report_binary(larva_id, gray_crop, mask_crop, skeleton_crop, metrics):
    """Create a compact 3-panel larva report image.

    Panels (left to right):
      - Original grayscale crop
      - Binary mask (white on black)
      - Skeleton overlay with endpoints, axis line, and length text

    The function is intentionally minimal and clean so that the
    resulting PNGs can be used directly as figure panels.
    """
    # Ensure uint8 single-channel inputs
    gray_crop = gray_crop.astype(np.uint8)
    mask_vis = (mask_crop.astype(np.uint8) * 255)

    # Left panel: grayscale → BGR
    panel_left = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

    # Middle panel: binary mask → BGR (white larva on black)
    panel_mid = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # Right panel: skeleton overlay on grayscale
    panel_right = panel_left.copy()

    endpoints = []
    if skeleton_crop is not None:
        skel_bool = skeleton_crop.astype(bool)
        # draw skeleton in white
        panel_right[skel_bool] = (255, 255, 255)

        # Simple endpoint detection (8-connected neighbor count)
        skel_coords = np.argwhere(skel_bool)
        for r, c in skel_coords:
            r0, r1 = max(0, r - 1), min(skel_bool.shape[0], r + 2)
            c0, c1 = max(0, c - 1), min(skel_bool.shape[1], c + 2)
            neighbor_count = np.sum(skel_bool[r0:r1, c0:c1]) - 1
            if neighbor_count == 1:
                endpoints.append((c, r))  # (x, y)

        # If we have at least two endpoints, connect the two farthest by distance
        if len(endpoints) >= 2:
            # pick two maximizing squared distance
            best_i, best_j = 0, 1
            best_d2 = -1.0
            for i in range(len(endpoints)):
                xi, yi = endpoints[i]
                for j in range(i + 1, len(endpoints)):
                    xj, yj = endpoints[j]
                    d2 = (xi - xj) ** 2 + (yi - yj) ** 2
                    if d2 > best_d2:
                        best_d2 = d2
                        best_i, best_j = i, j
            p1 = endpoints[best_i]
            p2 = endpoints[best_j]

            # Draw green line between endpoints
            cv2.line(
                panel_right,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )

        # Draw endpoints as small red circles
        for x, y in endpoints:
            cv2.circle(
                panel_right,
                (int(x), int(y)),
                3,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

    # Overlay length text (in pixels) slightly inside the image
    length_px = metrics.get('body_length', 0)
    text = f"L = {length_px:.1f} px"
    cv2.putText(
        panel_right,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    # Stack panels horizontally on a white background, with tight layout
    h = max(panel_left.shape[0], panel_mid.shape[0], panel_right.shape[0])
    w_total = panel_left.shape[1] + panel_mid.shape[1] + panel_right.shape[1]
    report_img = np.full((h, w_total, 3), 255, dtype=np.uint8)

    x0 = 0
    report_img[0:panel_left.shape[0], x0:x0 + panel_left.shape[1]] = panel_left
    x0 += panel_left.shape[1]
    report_img[0:panel_mid.shape[0], x0:x0 + panel_mid.shape[1]] = panel_mid
    x0 += panel_mid.shape[1]
    report_img[0:panel_right.shape[0], x0:x0 + panel_right.shape[1]] = panel_right

    return report_img


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
    print("AUTOMATED LARVA ANALYSIS PIPELINE - BINARY MASKS ONLY")
    print("=" * 70)
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Output Directory: {OUTPUT_ROOT}")
    print(f"Mode: Saving clean binary masks (no visualization reports)")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Detect all date folders
    date_folders = [
        d for d in ROOT_DIR.iterdir()
        if d.is_dir() and d.name and d.name[0].isdigit() and '.' in d.name
    ]

    if not date_folders:
        print("\n⚠ No date folders found!")
        return

    print(f"\nFound {len(date_folders)} date folders:")
    for df in sorted(date_folders):
        print(f"  - {df.name}")

    # Process all date folders
    all_summaries = []
    for df in sorted(date_folders):
        summaries = process_date_folder(df, OUTPUT_ROOT)
        if summaries:
            all_summaries.extend(summaries)

    if all_summaries:
        global_csv_path = OUTPUT_ROOT / 'global_summary.csv'
        try:
            with open(global_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
                writer.writeheader()
                writer.writerows(all_summaries)
            print(f"\nGlobal summary saved to: {global_csv_path}")
        except Exception as e:
            print(f"⚠ Failed to save global summary: {e}")

    total_images = len(all_summaries)
    total_larvae = sum(s.get('number_of_larvae', 0) for s in all_summaries)
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"Total images processed: {total_images}")
    print(f"Total larvae detected: {total_larvae}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


# ======================== MAIN ========================

if __name__ == "__main__":
    run_batch_pipeline()
