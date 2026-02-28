#!/usr/bin/env python3
import cv2
import numpy as np
import csv
from pathlib import Path
from skimage.morphology import skeletonize

# -------- CONFIG --------
INPUT = Path('analysis_primary/result_inner_circle_cropped.png')
OUTDIR = Path('analysis_primary_detailed')
OUTDIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR = OUTDIR / 'individual_larvae'
CROPS_DIR.mkdir(exist_ok=True)

# Processing Params
BLACKHAT_K = 25
MIN_AREA = 80
MAX_AREA_FRAC = 0.05
SOLIDITY_THRESH = 0.85


# ------------------------

def get_morphometrics(mask, gray_img):
    """Calculates advanced stats for a single particle."""
    # 1. Skeleton Length
    skeleton = skeletonize(mask > 0)
    length = np.sum(skeleton)

    # 2. Contour-based stats
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)

    # 3. Intensity stats (Health/Opacity)
    mean_val = cv2.mean(gray_img, mask=mask)[0]

    return length, perimeter, mean_val, skeleton


def process_larvae():
    if not INPUT.exists():
        print(f"Error: {INPUT} not found.")
        return

    # Load and Prepare
    img = cv2.imread(str(INPUT))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape

    # Pre-processing (Your Logic)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.absdiff(bg, gray)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_K, BLACKHAT_K))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    saliency = cv2.addWeighted(diff, 0.7, blackhat, 0.3, 0)

    # Thresholding
    thresh = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    larvae_data = []
    vis_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    full_mask = np.zeros_like(gray)

    print(f"Analyzing {num_labels - 1} potential particles...")

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w_b, h_b = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        if area < MIN_AREA or area > (h * w * MAX_AREA_FRAC):
            continue

        comp_mask = (labels == i).astype(np.uint8)

        # Geometry Filter
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = contours[0]
        solidity = area / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) > 0 else 0

        # Ellipse fitting for elongation
        if len(cnt) >= 5:
            _, (ma, MA), angle = cv2.fitEllipse(cnt)
            elongation = MA / ma if ma > 0 else 0
        else:
            MA = ma = elongation = angle = 0

        # Classification
        mean_bh = cv2.mean(blackhat, mask=comp_mask)[0]
        if (solidity < SOLIDITY_THRESH and elongation > 1.5) or mean_bh > 40:
            # GET ADVANCED METRICS
            body_len, perimeter, avg_intensity, skel_img = get_morphometrics(comp_mask, gray)

            larvae_data.append({
                'idx': len(larvae_data) + 1,
                'area': area,
                'body_length': round(body_len, 2),
                'perimeter': round(perimeter, 2),
                'solidity': round(solidity, 3),
                'elongation': round(elongation, 2),
                'orientation': round(angle, 1),
                'mean_intensity': round(avg_intensity, 2)
            })

            # Save Crop for Verification
            # Adding a 10px padding to the crop
            pad = 10
            y1, y2 = max(0, y - pad), min(h, y + h_b + pad)
            x1, x2 = max(0, x - pad), min(w, x + w_b + pad)
            crop = vis_overlay[y1:y2, x1:x2].copy()
            cv2.imwrite(str(CROPS_DIR / f"larva_{len(larvae_data)}.png"), crop)

            # Update Visualization
            full_mask[labels == i] = 255
            cv2.rectangle(vis_overlay, (x, y), (x + w_b, y + h_b), (0, 255, 0), 2)
            cv2.putText(vis_overlay, str(len(larvae_data)), (x, y - 5), 0, 0.6, (0, 255, 0), 2)

    # Export CSV
    csv_path = OUTDIR / 'larvae_analysis_full.csv'
    if larvae_data:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=larvae_data[0].keys())
            writer.writeheader()
            writer.writerows(larvae_data)

    # Save Final Visuals
    cv2.imwrite(str(OUTDIR / '02_mask_final.png'), full_mask)
    cv2.imwrite(str(OUTDIR / '03_overlay_final.png'), vis_overlay)

    print(f"Finished. {len(larvae_data)} larvae documented in {OUTDIR}")


if __name__ == "__main__":
    process_larvae()