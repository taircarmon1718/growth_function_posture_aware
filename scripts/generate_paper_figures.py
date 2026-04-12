#!/usr/bin/env python3
from pathlib import Path
import re
import sys
import csv
# Ensure project root is on sys.path so we can import top-level modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Optional imports; fall back to local equivalents if environment has binary conflicts.
try:
    from run_pipeline_binary_masks import segment_larvae, filter_larva_components, calculate_larva_morphometrics
except Exception:
    segment_larvae = None
    filter_larva_components = None
    calculate_larva_morphometrics = None

try:
    from filter_larvae_by_confidence import _crop_overlay_for_larva, PIXEL_TO_MM
except Exception:
    _crop_overlay_for_larva = None
    PIXEL_TO_MM = 0.232255814

FILTERED_DIR = ROOT / 'filtered_larvae_by_date'
OUT_DIR = ROOT / 'outputs' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Typography style (aligned with figures/pipeline_flow_diagram.py)
FONT_FAMILY = 'sans-serif'
TITLE_SIZE = 14
BODY_SIZE = 13
SMALL_LABEL_SIZE = 12

# Apply pipeline_flow_diagram typography if available
try:
    import importlib
    pfd = importlib.import_module('figures.pipeline_flow_diagram')
    PF_FONT_FAMILY = getattr(pfd, 'FONT_FAMILY', FONT_FAMILY)
    PF_TITLE_SIZE = getattr(pfd, 'TITLE_SIZE', TITLE_SIZE)
    PF_BODY_SIZE = getattr(pfd, 'BODY_SIZE', BODY_SIZE)
    PF_SMALL_LABEL_SIZE = getattr(pfd, 'SMALL_LABEL_SIZE', SMALL_LABEL_SIZE)
    PF_RC = getattr(pfd, 'RC_PARAMS', None)
except Exception:
    PF_FONT_FAMILY = FONT_FAMILY
    PF_TITLE_SIZE = TITLE_SIZE
    PF_BODY_SIZE = BODY_SIZE
    PF_SMALL_LABEL_SIZE = SMALL_LABEL_SIZE
    PF_RC = None

if PF_RC:
    plt.rcParams.update(PF_RC)
else:
    plt.rcParams.update({
        'font.family': PF_FONT_FAMILY,
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        # match likely pipeline_flow_diagram aesthetics
        'axes.titleweight': 'normal',
        'axes.titlesize': PF_TITLE_SIZE,
        'axes.labelsize': PF_BODY_SIZE,
        'font.size': PF_BODY_SIZE,
    })


def _crop_overlay_for_larva_local(date_str, image_name, larva_filename):
    overlay_path = ROOT / 'analysis_full_binary_masks_only' / date_str / image_name / 'overlay.png'
    mask_path = ROOT / 'analysis_full_binary_masks_only' / date_str / image_name / 'larvae_reports' / larva_filename
    morph_path = ROOT / 'analysis_full_binary_masks_only' / date_str / image_name / 'morphometrics.csv'
    if (not overlay_path.exists()) or (not mask_path.exists()) or (not morph_path.exists()):
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    src = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if mask is None or src is None:
        return None

    h, w = mask.shape[:2]
    if h < 2 or w < 2:
        return None

    try:
        larva_id = int(Path(larva_filename).stem.split('_')[-1])
    except Exception:
        return None

    cx = cy = None
    try:
        with open(morph_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(float(row['larva_id'])) != larva_id:
                        continue
                except Exception:
                    continue
                cx = float(row['centroid_x'])
                cy = float(row['centroid_y'])
                break
    except Exception:
        return None
    if cx is None or cy is None:
        return None

    pad_frac = 0.20
    h2 = h + 2 * int(round(h * pad_frac))
    w2 = w + 2 * int(round(w * pad_frac))
    x0 = int(round(cx - w2 / 2))
    y0 = int(round(cy - h2 / 2))
    x1 = x0 + w2
    y1 = y0 + h2

    x0 = max(0, min(x0, src.shape[1] - 1))
    y0 = max(0, min(y0, src.shape[0] - 1))
    x1 = max(1, min(x1, src.shape[1]))
    y1 = max(1, min(y1, src.shape[0]))
    crop = src[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def _read_metrics_from_csv(date_str, image_name, larva_filename):
    morph_path = ROOT / 'analysis_full_binary_masks_only' / date_str / image_name / 'morphometrics.csv'
    if not morph_path.exists():
        return {}
    try:
        larva_id = int(Path(larva_filename).stem.split('_')[-1])
    except Exception:
        return {}
    with open(morph_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row['larva_id'])) != larva_id:
                    continue
            except Exception:
                continue
            return {
                'area': float(row.get('area', 0) or 0),
                'body_length': float(row.get('body_length', 0) or 0),
                'mean_width': float(row.get('mean_width', 0) or 0),
                'curvature_ratio': float(row.get('curvature_ratio', 0) or 0),
                'eccentricity': float(row.get('eccentricity', 0) or 0),
                'solidity': float(row.get('solidity', 0) or 0),
            }
    return {}

def _read_centroid_from_csv(date_str, image_name, larva_filename):
    morph_path = ROOT / 'analysis_full_binary_masks_only' / date_str / image_name / 'morphometrics.csv'
    if not morph_path.exists():
        return None, None
    try:
        larva_id = int(Path(larva_filename).stem.split('_')[-1])
    except Exception:
        return None, None
    with open(morph_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row['larva_id'])) != larva_id:
                    continue
            except Exception:
                continue
            try:
                cx = float(row.get('centroid_x', '') or 0.0)
                cy = float(row.get('centroid_y', '') or 0.0)
                return cx, cy
            except Exception:
                return None, None
    return None, None

# Helpers
def find_larva_masks(filtered_dir):
    pattern = re.compile(r'^larva_\d+\.png$')
    results = []
    if not filtered_dir.exists():
        return results
    for date_dir in sorted(filtered_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for f in sorted(date_dir.iterdir()):
            if f.is_file() and pattern.match(f.name):
                results.append(f)
    return results


def locate_source_image_name(date_str, larva_filename, target_area):
    date_root = ROOT / 'analysis_full_binary_masks_only' / date_str
    best = None
    if not date_root.exists():
        return None
    for image_dir in sorted(date_root.iterdir()):
        if not image_dir.is_dir():
            continue
        cand = image_dir / 'larvae_reports' / larva_filename
        if not cand.exists():
            continue
        area = load_mask_area(cand)
        diff = abs(area - target_area)
        if best is None or diff < best[1]:
            best = (image_dir.name, diff)
        if diff == 0:
            return image_dir.name
    return None if best is None else best[0]


def load_mask_area(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return int(np.count_nonzero(img > 0))


def ensure_same_size(img, mask):
    if img.shape[0:2] == mask.shape[0:2]:
        return img, mask
    # resize mask to img
    mask_r = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask_r


def draw_boxes_on_ax(ax, img_rgb, stats, valid_ids=None):
    ax.imshow(img_rgb)
    ax.axis('off')
    for i in range(1, stats.shape[0]):
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        valid = (valid_ids is None) or (i in valid_ids)
        color = 'green' if valid else 'red'
        rect = Rectangle((x, y), w, h, linewidth=1.4, edgecolor=color, facecolor='none')
        ax.add_patch(rect)


# Main processing
def main():
    masks = find_larva_masks(FILTERED_DIR)
    if not masks:
        print('No filtered larvae found under', FILTERED_DIR)
        sys.exit(1)
    # pick largest area larva
    areas = [(p, load_mask_area(p)) for p in masks]
    areas = [a for a in areas if a[1] > 0]
    if not areas:
        print('No valid mask images found')
        sys.exit(1)
    areas.sort(key=lambda x: x[1], reverse=True)
    mask_path, mask_area = areas[0]
    date_dir = mask_path.parent
    date_str = date_dir.name
    larva_filename = mask_path.name
    stem = mask_path.stem

    # Load mask
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print("Could not read mask image:", mask_path)
        sys.exit(1)

    # Resolve source image and use _crop_overlay_for_larva as requested
    image_name = locate_source_image_name(date_str, larva_filename, mask_area)
    detection_crop = None
    if image_name is not None:
        if _crop_overlay_for_larva is not None:
            detection_crop = _crop_overlay_for_larva(date_str, image_name, larva_filename)
        if detection_crop is None:
            detection_crop = _crop_overlay_for_larva_local(date_str, image_name, larva_filename)

    # Fallback to previously saved crop if needed
    if detection_crop is None:
        orig_crop_path = date_dir / f"{stem}_original.png"
        if orig_crop_path.exists():
            detection_crop = cv2.imread(str(orig_crop_path))

    # Final fallback
    if detection_crop is None:
        detection_crop = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

    # Prepare grayscale crop for mask alignment and measurement as before
    gray_crop = cv2.cvtColor(detection_crop, cv2.COLOR_BGR2GRAY)
    # ensure mask and crop align sizes
    _, mask_resized = ensure_same_size(gray_crop, mask_img)
    # create clean grayscale: larva pixels keep value, background set to 255 (white)
    clean_gray = gray_crop.copy()
    clean_gray[mask_resized == 0] = 255

    # ensure detection_view is available for downstream segmentation/visualization
    detection_view = gray_crop.copy()

    # Attempt to load the full overlay (petri dish) so we can draw bbox on full image
    full_overlay = None
    if image_name is not None:
        overlay_path = ROOT / 'analysis_full_binary_masks_only' / date_str / image_name / 'overlay.png'
        if overlay_path.exists():
            full_overlay = cv2.imread(str(overlay_path))
    if full_overlay is None:
        full_overlay = detection_crop.copy()

    # Convert BGR->RGB for color display in matplotlib
    full_overlay_rgb = cv2.cvtColor(full_overlay, cv2.COLOR_BGR2RGB)

    # Compute bounding box in full image coords using centroid and mask extents if possible
    cx, cy = _read_centroid_from_csv(date_str, image_name or '', larva_filename)
    box = None
    if cx is not None and cy is not None:
        h_mask, w_mask = mask_img.shape[:2]
        pad_frac = 0.20
        h2 = h_mask + 2 * int(round(h_mask * pad_frac))
        w2 = w_mask + 2 * int(round(w_mask * pad_frac))
        x0 = int(round(cx - w2 / 2))
        y0 = int(round(cy - h2 / 2))
        x1 = x0 + w2
        y1 = y0 + h2
        # clamp to image bounds
        x0 = max(0, min(x0, full_overlay_rgb.shape[1] - 1))
        y0 = max(0, min(y0, full_overlay_rgb.shape[0] - 1))
        x1 = max(0, min(x1, full_overlay_rgb.shape[1]))
        y1 = max(0, min(y1, full_overlay_rgb.shape[0]))
        box = (x0, y0, x1, y1)
        # draw rectangle on a copy
    panel_detection = full_overlay_rgb.copy()
    if box is not None:
        cv2.rectangle(panel_detection, (box[0], box[1]), (box[2], box[3]), (220, 20, 20), 2)

    # The 'Original' image used in figure1_segmentation is the grayscale 'gray' which is set later as detection_view.copy().
    # Use detection_view (grayscale crop) as the second panel to match 'Original'.
    panel_original = detection_view.copy()

    # Panel 3: binary mask (white larva on black background)
    bin_mask = (mask_resized > 0).astype(np.uint8) * 255

    # Convert all three image panels to 3-channel RGB for consistent sizing and display
    def to_rgb(img):
        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 3:
            return img.copy()
        return img.copy()

    img1 = to_rgb(panel_detection)
    img2 = to_rgb(panel_original)
    img3 = to_rgb(bin_mask)

    # Ensure identical displayed size: resize all to the same target (max height and width)
    shapes = [i.shape[:2] for i in (img1, img2, img3) if i is not None]
    if len(shapes) == 0:
        print('No images to display')
        return
    max_h = max(h for h, w in shapes)
    max_w = max(w for h, w in shapes)

    def resize_for_display(img, h=max_h, w=max_w):
        if img is None:
            return img
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    img1_r = resize_for_display(img1)
    img2_r = resize_for_display(img2)
    img3_r = resize_for_display(img3)

    # Panel 4: morphometric text (remove body length line)
    comp_mask = (mask_resized > 0).astype(np.uint8)
    if calculate_larva_morphometrics is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bh = cv2.morphologyEx(gray_crop, cv2.MORPH_BLACKHAT, k)
        metrics, _ = calculate_larva_morphometrics(1, comp_mask, gray_crop, bh)
    else:
        metrics = _read_metrics_from_csv(date_str, image_name, larva_filename)

    area_val = metrics.get('area', 0)
    mean_w = metrics.get('mean_width', 0.0)
    curv = metrics.get('curvature_ratio', 0.0)
    ecc = metrics.get('eccentricity', 0.0)
    solidity = metrics.get('solidity', 0.0)

    # --- Create Figure 3 using manual axis placement to match pipeline_flow_diagram style ---
    # Define colors for this figure
    FIG3_COLS = {
        "det": "#4A235A", "orig": "#1B5E20", "mask": "#1B5E20",
        "meas": "#7B1A1A", "bg": "#FFFFFF", "caption": "#1A1A1A"
    }

    # Prepare images for panels (these are the final data to be plotted, do not modify)
    img1 = panel_detection
    img2 = panel_original
    img3 = bin_mask

    # Create the figure with manual layout
    panel_images = [img1, img2, img3, None]  # Use None as a placeholder for the text panel
    captions = ['Detection', 'Original Larva', 'Binary Mask', 'Measurements']
    border_colors = [FIG3_COLS["det"], FIG3_COLS["orig"], FIG3_COLS["mask"], FIG3_COLS["meas"]]

    N = len(panel_images)
    IMG_H = 5.0
    GAP_X = 0.3
    PAD_L, PAD_R, PAD_TOP, PAD_BOT = 0.2, 0.2, 0.2, 1.0
    CAP_H = 1.1

    # Aspect-matched outer frame widths for panels 1–3 (common height, varying width).
    # This guarantees: full image visible, no crop, no distortion, no internal blank margins.
    img_widths = []
    for im in (img1, img2, img3):
        ih, iw = im.shape[:2]
        ratio = float(iw) / float(ih) if ih > 0 else 1.0
        img_widths.append(IMG_H * ratio)
    # Panel 4 text width: aligned height with a readable, balanced width.
    text_panel_w = IMG_H * 0.90
    panel_widths = [img_widths[0], img_widths[1], img_widths[2], text_panel_w]

    fig_w = PAD_L + sum(panel_widths) + (N - 1) * GAP_X + PAD_R
    fig_h = PAD_TOP + IMG_H + CAP_H + PAD_BOT

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=FIG3_COLS['bg'])

    def fx(inch): return inch / fig_w
    def fy(inch): return inch / fig_h

    x_cursor = PAD_L
    for i, (im, cap, bcol) in enumerate(zip(panel_images, captions, border_colors)):
        panel_w = panel_widths[i]
        img_left = x_cursor
        ax = fig.add_axes((fx(img_left), fy(PAD_BOT), fx(panel_w), fy(IMG_H)))
        x_cursor += panel_w + GAP_X

        if im is not None:
            # Panel 1 only: mild visual crop/zoom toward the petri dish.
            disp_im = im
            if i == 0:
                h0, w0 = im.shape[:2]
                c = 0.16  # trim 16% from each side for tighter petri-dish focus
                x0, x1 = int(round(w0 * c)), int(round(w0 * (1.0 - c)))
                y0, y1 = int(round(h0 * c)), int(round(h0 * (1.0 - c)))
                if x1 > x0 + 10 and y1 > y0 + 10:
                    disp_im = im[y0:y1, x0:x1]

            h, w = disp_im.shape[:2]
            if disp_im.ndim == 2:  # Grayscale images
                ax.imshow(
                    disp_im,
                    cmap='gray',
                    interpolation='lanczos',
                    origin='upper',
                    aspect='equal',
                )
            else:  # RGB images
                ax.imshow(
                    disp_im,
                    interpolation='lanczos',
                    origin='upper',
                    aspect='equal',
                )
            # Explicit limits: full image visible, no crop, no distortion.
            ax.set_xlim(0.0, float(w))
            ax.set_ylim(float(h), 0.0)
            ax.set_autoscale_on(False)
            ax.set_aspect('equal', adjustable='box')
            ax.margins(x=0, y=0)
        else:
            # This is the measurements panel (panel 4)
            ax.set_facecolor('white')
            text_lines = [
                f'Area: {area_val:.0f} px',
                f'Mean width: {mean_w:.1f} px',
                f'Curvature ratio: {curv:.3f}',
                f'Eccentricity: {ecc:.3f}',
                f'Solidity: {solidity:.3f}',
            ]
            ax.text(0.1, 0.5, '\n'.join(text_lines),
                    ha='left', va='center',
                    fontsize=24,
                    linespacing=1.6,
                    family=PF_FONT_FAMILY,
                    color='black', # Dark, readable text
                    transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(bcol)
            spine.set_linewidth(3.0)

        ax.text(0.03, 0.03, str(i + 1), ha='left', va='bottom', fontsize=14, fontweight='bold',
                color='white', transform=ax.transAxes, family=PF_FONT_FAMILY,
                bbox=dict(boxstyle='square,pad=0.18', facecolor=bcol, edgecolor='none', alpha=0.95))

        # Add caption below panel
        bb = ax.get_position()
        cap_cx = bb.x0 + bb.width / 2
        cap_by = bb.y0 - 0.01

        cap_text = cap
        if len(cap) > 15 and ' ' in cap:
            # Simple split for two-line captions
            mid_idx = cap.find(' ', len(cap)//2 - 4, len(cap)//2 + 4)
            if mid_idx != -1:
                cap_text = cap[:mid_idx] + '\n' + cap[mid_idx+1:]

        fig.text(cap_cx, cap_by, cap_text, ha='center', va='top', fontsize=30,
                 color=FIG3_COLS['caption'], linespacing=1.3, family=PF_FONT_FAMILY)

    out3 = OUT_DIR / 'figure3_larva_pipeline.png'
    plt.savefig(str(out3), dpi=300, bbox_inches='tight', facecolor=FIG3_COLS['bg'])
    plt.close(fig)
    print("Saved:", str(out3))
    # --- end of Figure 3 creation ---

    out1 = None
    out2 = None

    if segment_larvae is not None and filter_larva_components is not None:
        # FIGURE 1: segmentation pipeline [Original] -> [Blackhat] -> [Mask]
        gray = detection_view.copy()
        thresh, num_labels, labels_img, stats, centroids, blackhat = segment_larvae(gray)
        fig = plt.figure(figsize=(10, 3))
        axs = [fig.add_subplot(1, 3, i+1) for i in range(3)]
        axs[0].imshow(gray, cmap='gray', vmin=0, vmax=255); axs[0].axis('off'); axs[0].set_title('Original', fontsize=TITLE_SIZE, fontweight='normal', family=FONT_FAMILY)
        axs[1].imshow(blackhat, cmap='gray', vmin=0, vmax=255); axs[1].axis('off'); axs[1].set_title('Blackhat', fontsize=TITLE_SIZE, fontweight='normal', family=FONT_FAMILY)
        axs[2].imshow(thresh, cmap='gray', vmin=0, vmax=255); axs[2].axis('off'); axs[2].set_title('Mask', fontsize=TITLE_SIZE, fontweight='normal', family=FONT_FAMILY)
        out1 = OUT_DIR / 'figure1_segmentation.png'
        plt.tight_layout(); plt.savefig(str(out1), dpi=300, bbox_inches='tight'); plt.close()
        print("Saved:", str(out1))

        # FIGURE 2: filtering effect - before vs after
        valid_ids = filter_larva_components(num_labels, labels_img, stats, gray, blackhat, gray.shape)
        vis_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        fig = plt.figure(figsize=(10, 5))
        axl = fig.add_subplot(1, 2, 1)
        axr = fig.add_subplot(1, 2, 2)
        axl.set_title('Before Filtering', fontsize=TITLE_SIZE, fontweight='normal', family=FONT_FAMILY)
        axr.set_title('After Filtering', fontsize=TITLE_SIZE, fontweight='normal', family=FONT_FAMILY)
        draw_boxes_on_ax(axl, vis_rgb, stats, valid_ids=None)
        draw_boxes_on_ax(axr, vis_rgb, stats, valid_ids=set(valid_ids))
        out2 = OUT_DIR / 'figure2_filtering.png'
        plt.tight_layout(); plt.savefig(str(out2), dpi=300, bbox_inches='tight'); plt.close()
        print("Saved:", str(out2))

    # Print full paths at the end
    paths = [p for p in [out1, out2, out3] if p is not None]
    print('\nFinal saved figure paths:')
    for p in paths:
        print(str(p))

    print('Figures saved under:', OUT_DIR)

if __name__ == '__main__':
    main()
