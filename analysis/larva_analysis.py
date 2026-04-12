#!/usr/bin/env python3
import cv2
import numpy as np
import csv
from pathlib import Path
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import euclidean

# -------- CONFIG --------
INPUT = Path('analysis_primary/result_inner_circle_cropped.png')
OUTDIR = Path('analysis_primary_reports')
OUTDIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUTDIR / 'larvae_reports'
REPORTS_DIR.mkdir(exist_ok=True)

BLACKHAT_K = 25
MIN_AREA = 80
MAX_AREA_FRAC = 0.05
SOLIDITY_THRESH = 0.85


def calculate_advanced_metrics(idx, mask, gray, blackhat):
    """Calculates comprehensive morphometrics for a single larva component."""
    # Basic Props via Skimage
    props = regionprops(mask, intensity_image=gray)[0]
    bh_props = regionprops(mask, intensity_image=blackhat)[0]

    # 1. Skeleton & Distance Transform
    skeleton = skeletonize(mask > 0)
    dist_map = distance_transform_edt(mask)
    skel_coords = np.argwhere(skeleton)

    # 2. Width Metrics (Local thickness at every skeleton point)
    widths = dist_map[skeleton] * 2  # radius to diameter

    # 3. Skeleton Path Length (Correction for diagonals)
    # Using approx: adjacent = 1, diagonal = 1.414
    body_len = np.sum(skeleton)  # Simplification; for high precision use skan library

    # 4. Curvature & Endpoints (Crude approximation)
    endpoints = []
    for r, c in skel_coords:
        neighbor_count = np.sum(skeleton[r - 1:r + 2, c - 1:c + 2]) - 1
        if neighbor_count == 1:
            endpoints.append((r, c))

    euclidean_dist = 0
    curvature_ratio = 1
    if len(endpoints) >= 2:
        # Distance between furthest two endpoints
        p1, p2 = endpoints[0], endpoints[-1]
        euclidean_dist = euclidean(p1, p2)
        curvature_ratio = body_len / euclidean_dist if euclidean_dist > 0 else 1

    # 5. Advanced Shape
    circularity = (4 * np.pi * props.area) / (props.perimeter ** 2) if props.perimeter > 0 else 0

    # Compile Data
    data = {
        'id': idx,
        # Basic Size
        'area': props.area,
        'perimeter': round(props.perimeter, 2),
        'bbox_width': props.bbox[3] - props.bbox[1],
        'bbox_height': props.bbox[2] - props.bbox[0],
        'equiv_diameter': round(props.equivalent_diameter, 2),
        'solidity': round(props.solidity, 3),
        'extent': round(props.extent, 3),
        # Ellipse
        'major_axis': round(props.major_axis_length, 2),
        'minor_axis': round(props.minor_axis_length, 2),
        'eccentricity': round(props.eccentricity, 3),
        'orientation_deg': round(np.degrees(props.orientation), 1),
        # Skeleton/Width
        'skeleton_length': round(body_len, 2),
        'euclidean_length': round(euclidean_dist, 2),
        'curvature_ratio': round(curvature_ratio, 3),
        'mean_width': round(np.mean(widths), 2),
        'max_width': round(np.max(widths), 2),
        'min_width': round(np.min(widths), 2),
        # Intensity/Texture
        'mean_intensity': round(props.mean_intensity, 2),
        'std_intensity': round(np.std(gray[mask > 0]), 2),
        'mean_blackhat': round(bh_props.mean_intensity, 2),
        # Spatial
        'centroid_x': round(props.centroid[1], 1),
        'centroid_y': round(props.centroid[0], 1),
    }

    # Add Hu Moments (Shape Fingerprint)
    for i, hu in enumerate(props.moments_hu):
        data[f'hu_moment_{i + 1}'] = f"{hu:.2e}"

    return data


def create_individual_report(idx, gray_crop, mask_crop, data):
    """Refined report generator with more metrics."""
    h, w = gray_crop.shape
    left_panel = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    right_panel = left_panel.copy()

    # Visual overlays
    skeleton = skeletonize(mask_crop > 0)
    right_panel[skeleton] = [0, 0, 255]
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(right_panel, contours, -1, (0, 255, 0), 1)

    canvas = np.hstack((left_panel, right_panel))

    # Expand canvas for more text
    text_area_w = 250
    report_img = np.zeros((max(canvas.shape[0], 350), canvas.shape[1] + text_area_w, 3), dtype=np.uint8)
    report_img[:canvas.shape[0], :canvas.shape[1]] = canvas

    # Text rendering
    lines = [f"ID: {idx}"] + [f"{k}: {v}" for k, v in list(data.items())[1:15]]  # Show first 15 metrics
    for i, line in enumerate(lines):
        cv2.putText(report_img, line, (canvas.shape[1] + 10, 25 + (i * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return report_img


def process_larvae():
    img = cv2.imread(str(INPUT))
    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhancement
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_K, BLACKHAT_K))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    larvae_data = []

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA: continue

        mask = (labels == i).astype(np.uint8)

        # Calculate all requested metrics
        stats_dict = calculate_advanced_metrics(len(larvae_data) + 1, mask, gray, blackhat)
        larvae_data.append(stats_dict)

        # Cropping for report
        x, y, w, h = stats[i, :4]
        p = 10
        crop_gray = gray[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
        crop_mask = mask[max(0, y - p):y + h + p, max(0, x - p):x + w + p]

        report = create_individual_report(stats_dict['id'], crop_gray, crop_mask, stats_dict)
        cv2.imwrite(str(REPORTS_DIR / f"larva_{stats_dict['id']:03d}.png"), report)

    # Export CSV
    if larvae_data:
        keys = larvae_data[0].keys()
        with open(OUTDIR / 'comprehensive_morphometrics.csv', 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(larvae_data)

    print(f"Done. Analyzed {len(larvae_data)} larvae.")


if __name__ == "__main__":
    process_larvae()