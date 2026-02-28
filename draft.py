#!/usr/bin/env python3
import cv2
import numpy as np
import csv
from pathlib import Path
from skimage.morphology import skeletonize

# -------- CONFIG --------
INPUT = Path('analysis_primary/result_inner_circle_cropped.png')
OUTDIR = Path('analysis_primary_reports')
OUTDIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUTDIR / 'larvae_reports'
REPORTS_DIR.mkdir(exist_ok=True)

# Processing Params
BLACKHAT_K = 25
MIN_AREA = 80
MAX_AREA_FRAC = 0.05
SOLIDITY_THRESH = 0.85


# ------------------------

def create_individual_report(idx, gray_crop, mask_crop, data):
    """Creates a side-by-side comparison image with data overlay."""
    h, w = gray_crop.shape

    # 1. Prepare Left Panel: Raw Grayscale converted to BGR
    left_panel = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

    # 2. Prepare Right Panel: Analysis Visualization
    right_panel = left_panel.copy()

    # Draw Skeleton on Right Panel
    skeleton = skeletonize(mask_crop > 0)
    right_panel[skeleton] = [0, 0, 255]  # Red skeleton

    # Draw Contour
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(right_panel, contours, -1, (0, 255, 0), 1)  # Green outline

    # 3. Combine Panels
    canvas = np.hstack((left_panel, right_panel))

    # 4. Add Text Data (Create extra space at bottom for text)
    text_height = 120
    report_img = np.zeros((canvas.shape[0] + text_height, canvas.shape[1], 3), dtype=np.uint8)
    report_img[:canvas.shape[0], :] = canvas

    # Text positioning
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    color = (255, 255, 255)
    line_h = 18

    lines = [
        f"Larva ID: {idx}",
        f"Area: {data['area']} px",
        f"Body Length (Skel): {data['body_length']} px",
        f"Avg Width: {data['avg_width']} px",
        f"Solidity: {data['solidity']}",
        f"Elongation: {data['elongation']}",
        f"Orientation: {data['orientation']} deg"
    ]

    for i, line in enumerate(lines):
        y_pos = canvas.shape[0] + 20 + (i * line_h)
        cv2.putText(report_img, line, (10, y_pos), font, fs, color, 1, cv2.LINE_AA)

    return report_img


def process_larvae():
    if not INPUT.exists():
        print(f"Error: {INPUT} not found.")
        return

    # Load and Prepare
    img = cv2.imread(str(INPUT))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape

    # Enhancement Logic
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.absdiff(bg, gray)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_K, BLACKHAT_K))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    saliency = cv2.addWeighted(diff, 0.7, blackhat, 0.3, 0)

    # Segmentation
    thresh = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Connected Components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    larvae_data = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA or area > (h * w * MAX_AREA_FRAC):
            continue

        comp_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = contours[0]

        # Geometry
        solidity = area / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) > 0 else 0
        if len(cnt) >= 5:
            _, (ma, MA), angle = cv2.fitEllipse(cnt)
            elongation = MA / ma if ma > 0 else 0
        else:
            MA = ma = elongation = angle = 0

        # Classification Filter
        mean_bh = cv2.mean(blackhat, mask=comp_mask)[0]
        if (solidity < SOLIDITY_THRESH and elongation > 1.5) or mean_bh > 40:
            # Skeleton and Length
            skel = skeletonize(comp_mask > 0)
            body_len = np.sum(skel)

            current_id = len(larvae_data) + 1
            larva_stats = {
                'idx': current_id,
                'area': area,
                'body_length': round(body_len, 2),
                'avg_width': round(area / body_len, 2) if body_len > 0 else 0,
                'solidity': round(solidity, 3),
                'elongation': round(elongation, 2),
                'orientation': round(angle, 1)
            }
            larvae_data.append(larva_stats)

            # Generate Side-by-Side Report
            x, y, wb, hb = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            pad = 15
            y1, y2 = max(0, y - pad), min(h, y + hb + pad)
            x1, x2 = max(0, x - pad), min(w, x + wb + pad)

            crop_gray = gray[y1:y2, x1:x2]
            crop_mask = comp_mask[y1:y2, x1:x2]

            report = create_individual_report(current_id, crop_gray, crop_mask, larva_stats)
            cv2.imwrite(str(REPORTS_DIR / f"larva_report_{current_id:03d}.png"), report)

    # Final Summary CSV
    if larvae_data:
        with open(OUTDIR / 'full_morphometrics_summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=larvae_data[0].keys())
            writer.writeheader()
            writer.writerows(larvae_data)

    print(f"Processed {len(larvae_data)} larvae.")
    print(f"Individual reports saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    process_larvae()