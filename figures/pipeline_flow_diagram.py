#!/usr/bin/env python3
"""
pipeline_flow_diagram.py - Publication-quality pipeline figures
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_IMAGE = PROJECT_ROOT / "31.10" / "images" / "IMG_7376.JPG"
SAMPLE_BASE = PROJECT_ROOT / "analysis_full" / "31.10" / "IMG_7376"
LARVA_MASK = SAMPLE_BASE / "segmentation_mask.png"
LARVA_OVERLAY = SAMPLE_BASE / "overlay.png"

BLACKHAT_KERNEL_SIZE = 25
BG_KERNEL_SIZE = 50
MORPH_OPEN_SIZE = 3
MIN_LARVA_AREA = 80

COL = {
    "pre": "#1A5276", "seg": "#1A237E", "filt": "#4A235A",
    "morph": "#1B5E20", "out": "#7B1A1A", "bg": "#FFFFFF",
    "arrow": "#444444", "caption": "#1A1A1A",
}

STAGE_INFO_FIG1 = [("Grayscale image", COL["pre"]), ("Background estimation", COL["pre"]),
                   ("Contrast enhancement", COL["pre"]), ("Black-hat filtering", COL["seg"]),
                   ("Saliency thresholding", COL["seg"])]

STAGE_INFO_FIG2 = [("Morphological cleaning", COL["seg"]),
                   ("Connected components", COL["filt"]), ("Final larval mask", COL["filt"])]


def _compute_pca_length(mask_binary: np.ndarray):
    """Compute PCA major-axis length from binary mask."""
    ys, xs = np.where(mask_binary > 0)
    if len(xs) < 2:
        return 0.0, (0, 0), (0, 0), (0, 0), np.array([1.0, 0.0])
    pts = np.column_stack((xs.astype(float), ys.astype(float)))
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    direction = Vt[0]
    projections = pts_centered @ direction
    pmin, pmax = projections.min(), projections.max()
    ep1 = centroid + pmin * direction
    ep2 = centroid + pmax * direction
    length_px = float(pmax - pmin)
    return length_px, (float(ep1[0]), float(ep1[1])), (float(ep2[0]), float(ep2[1])), (float(centroid[0]), float(centroid[1])), direction


def _find_file_in_project(filename: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.rglob(filename))
    return matches[0] if matches else None


def _select_larva_from_later_date():
    """Select a larva from a later date (after day 10) for better quality."""
    CONF_THRESHOLD = 0.85
    preds_path = PROJECT_ROOT / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
    preds = pd.read_excel(preds_path)

    # Prefer later dates (31.10, 3.11, etc.) for better larva quality
    # Map date to numeric value for sorting
    date_order = {'19.10': 1, '20.10': 2, '21.10': 3, '24.10': 4, '25.10': 5,
                  '26.10': 6, '27.10': 7, '29.10': 8, '31.10': 9, '3.11': 10}

    sel = preds[(preds["predicted_valid"] == 1) & (preds["valid_confidence"] >= float(CONF_THRESHOLD))].copy()
    if sel.shape[0] == 0:
        raise RuntimeError(f"No larva satisfies filters")

    # Extract date from image_name (e.g., "IMG_3778" -> look up in data or use later dates)
    # Sort by date descending to prefer later dates
    sel['date_numeric'] = sel['date'].apply(lambda d: date_order.get(str(d).replace('.0', '.').replace(',', '.'), 0))
    sel = sel.sort_values(['date_numeric', 'image_name', 'larva_filename'], ascending=[False, True, True]).reset_index(drop=True)

    row = sel.iloc[0]
    larva_fname = str(row["larva_filename"]).strip()
    image_name = str(row.get("image_name", "")).strip()

    larva_report_path = _find_file_in_project(larva_fname)
    if larva_report_path is None:
        raise FileNotFoundError(f"Larva report file {larva_fname} not found")

    sample_base = larva_report_path.parent.parent if (larva_report_path.parent.name == 'larvae_reports') else larva_report_path.parent

    mask_path = sample_base / 'segmentation_mask.png'
    overlay_path = sample_base / 'overlay.png'

    if not mask_path.exists() or not overlay_path.exists():
        raise FileNotFoundError(f"Missing mask or overlay in {sample_base}")

    overlay_img = cv2.imread(str(overlay_path))
    mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if overlay_img is None or mask_full is None:
        raise RuntimeError("Cannot read overlay or mask")

    # Select best larva component by PCA length matching
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats((mask_full > 0).astype(np.uint8), connectivity=8)
    px_pred = float(row.get('body_length_px', np.nan)) if pd.notna(row.get('body_length_px', np.nan)) else np.nan

    best_label = 0
    best_score = float('inf')
    for lbl in range(1, num_labels2):
        area = int(stats2[lbl, cv2.CC_STAT_AREA])
        if area < MIN_LARVA_AREA or area > 50000:
            continue
        comp_mask = (labels2 == lbl).astype(np.uint8)
        comp_len_px, _, _, _, _ = _compute_pca_length(comp_mask)
        if np.isfinite(px_pred) and px_pred > 1.0:
            score = abs(comp_len_px - px_pred) / px_pred
        else:
            score = -area
        if score < best_score:
            best_score = score
            best_label = lbl

    if best_label == 0:
        best_area = 0
        for lbl in range(1, num_labels2):
            area = int(stats2[lbl, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_label = lbl

    if best_label == 0:
        raise RuntimeError("No suitable larva component found")

    chosen_label = int(best_label)
    larva_only = (labels2 == chosen_label).astype(np.uint8)
    lbl_stats = stats2[chosen_label]
    x, y, ww, hh = int(lbl_stats[cv2.CC_STAT_LEFT]), int(lbl_stats[cv2.CC_STAT_TOP]), int(lbl_stats[cv2.CC_STAT_WIDTH]), int(lbl_stats[cv2.CC_STAT_HEIGHT])

    pad = int(max(20, 0.3 * max(ww, hh)))
    y0 = max(0, y - pad)
    y1 = min(mask_full.shape[0] - 1, y + hh + pad - 1)
    x0 = max(0, x - pad)
    x1 = min(mask_full.shape[1] - 1, x + ww + pad - 1)

    # Try to find raw image
    raw_img = None
    if image_name:
        raw_candidates = sorted(PROJECT_ROOT.rglob(f"*{image_name}*"))
        if raw_candidates:
            raw_img = cv2.imread(str(raw_candidates[0]))
    if raw_img is None:
        raw_img = overlay_img

    larva_crop_bgr = raw_img[y0:y1+1, x0:x1+1].copy() if raw_img is not None else overlay_img[y0:y1+1, x0:x1+1].copy()
    larva_crop_mask = larva_only[y0:y1+1, x0:x1+1].astype(np.uint8)

    # Compute measurements
    length_px, ep1, ep2, centroid, direction = _compute_pca_length(larva_crop_mask)
    area_px = int(larva_crop_mask.sum())
    px_pred_val = px_pred if np.isfinite(px_pred) else float('nan')
    mm_pred = float(row.get('body_length_mm', np.nan)) if pd.notna(row.get('body_length_mm', np.nan)) else float('nan')
    PIXEL_TO_MM = (mm_pred / px_pred_val) if (np.isfinite(px_pred_val) and px_pred_val > 1e-6 and np.isfinite(mm_pred)) else float('nan')
    length_mm = length_px * PIXEL_TO_MM if np.isfinite(PIXEL_TO_MM) else float('nan')

    return {
        'larva_fname': larva_fname,
        'overlay_img': overlay_img,
        'larva_crop_bgr': larva_crop_bgr,
        'larva_crop_mask': larva_crop_mask,
        'length_px': length_px,
        'length_mm': length_mm,
        'area_px': area_px,
        'ep1': ep1, 'ep2': ep2, 'centroid': centroid, 'direction': direction
    }


def generate_stages(raw_path: Path) -> list:
    """Generate preprocessing and segmentation stages."""
    img_bgr = cv2.imread(str(raw_path))
    if img_bgr is None or img_bgr.size == 0:
        raise FileNotFoundError(f"Cannot open: {raw_path}")

    h, w = img_bgr.shape[:2]
    if w > 1400:
        scale = 1400 / w
        img_bgr = cv2.resize(img_bgr, (1400, int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s1 = gray.copy()

    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_KERNEL_SIZE, BG_KERNEL_SIZE))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, k_bg)
    s2 = bg.copy()

    diff = cv2.absdiff(bg, gray)
    s3 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)
    s4 = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    saliency = cv2.addWeighted(diff, 0.7, blackhat, 0.3, 0)
    _, thresh = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s5 = thresh.copy()

    k_mo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_mo)
    s6 = cleaned.copy()

    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    cc_img = np.zeros((*cleaned.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < MIN_LARVA_AREA:
            continue
        cc_img[labels_img == lbl] = rng.integers(60, 255, 3).tolist()
    s7 = cc_img

    mask_img = cv2.imread(str(LARVA_MASK), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        mask_img = cleaned
    s8 = mask_img

    return [s1, s2, s3, s4, s5, s6, s7, s8]


def _place_horizontal_panels(imgs, captions, border_cols, out_file, img_w=4.2, img_h=4.2, pad_top=0.6, pad_bot=0.75):
    """Place panels horizontally (no arrows)."""
    N = len(imgs)
    GAP_X = 0.28
    PAD_L = 0.16
    PAD_R = 0.16
    CAP_H = 1.1

    row_w = N * img_w + (N - 1) * GAP_X
    fig_w = PAD_L + row_w + PAD_R
    fig_h = pad_top + img_h + CAP_H + pad_bot

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COL['bg'])

    def fx(inch): return inch / fig_w
    def fy(inch): return inch / fig_h

    axes = []
    for i, (im, cap, bcol) in enumerate(zip(imgs, captions, border_cols)):
        img_left = PAD_L + i * (img_w + GAP_X)
        img_bottom = pad_bot
        ax = fig.add_axes((fx(img_left), fy(img_bottom), fx(img_w), fy(img_h)))

        if im is not None and im.size > 0:
            if len(im.shape) == 2:
                ax.imshow(im, cmap='gray', aspect='equal', interpolation='lanczos', vmin=0, vmax=255)
            else:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='lanczos')
        else:
            ax.set_facecolor('#EEEEEE')

        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(bcol)
            sp.set_linewidth(3.0)

        ax.text(0.03, 0.03, str(i + 1), ha='left', va='bottom', fontsize=10000, fontweight='bold',
                color='white', transform=ax.transAxes, family='sans-serif',
                bbox=dict(boxstyle='square,pad=0.18', facecolor=bcol, edgecolor='none', alpha=0.95))

        axes.append(ax)

    # Add captions
    for i, (ax, cap) in enumerate(zip(axes, captions)):
        bb = ax.get_position()
        cap_cx = bb.x0 + bb.width / 2
        cap_by = bb.y0 - CAP_H * 0.35

        lines = cap.split()
        if len(lines) >= 3 and len(cap) > 15:
            mid = len(lines) // 2
            cap_text = ' '.join(lines[:mid]) + '\n' + ' '.join(lines[mid:])
        else:
            cap_text = cap

        fig.text(cap_cx, cap_by, cap_text, ha='center', va='top', fontsize=10000,
                color=COL['caption'], linespacing=1.3, multialignment='center',
                family='sans-serif', fontweight='normal')

    plt.savefig(str(out_file), dpi=300, bbox_inches='tight', facecolor=COL['bg'])
    plt.close()
    print(f"  ✓  Saved: {out_file}")


def main():
    print("\n" + "=" * 70)
    print("PIPELINE FLOW DIAGRAM - Publication-Quality Figures")
    print("=" * 70)

    if not RAW_IMAGE.exists():
        print(f"❌  Raw image not found: {RAW_IMAGE}")
        return

    print("\n  Generating pipeline stage images...")
    try:
        stages = generate_stages(RAW_IMAGE)
        print(f"  ✓ {len(stages)} stages ready.")
    except Exception as e:
        print(f"❌  Error: {e}")
        return

    print("\n  Rendering Figure 1: Preprocessing...")
    try:
        _place_horizontal_panels(stages[0:5],
                                [c[0] for c in STAGE_INFO_FIG1],
                                [c[1] for c in STAGE_INFO_FIG1],
                                SCRIPT_DIR / 'pipeline_flow_diagram_figure1.png',
                                img_w=4.2, img_h=4.2)
    except Exception as e:
        print(f"❌  Error: {e}")

    print("\n  Rendering Figure 2: Segmentation...")
    try:
        _place_horizontal_panels(stages[5:8],
                                [c[0] for c in STAGE_INFO_FIG2],
                                [c[1] for c in STAGE_INFO_FIG2],
                                SCRIPT_DIR / 'pipeline_flow_diagram_figure2.png',
                                img_w=5.2, img_h=4.8)
    except Exception as e:
        print(f"❌  Error: {e}")

    print("\n  Rendering Figure 3: Detection & Morphometrics (real larva from later date)...")
    captions = [
        "Detection (overlay)",
        "Larva crop (later stage)",
        "Binary mask with morphometric report",
    ]
    border_cols = [COL["filt"], COL["morph"], COL["morph"]]
    overlay_panel, larva_crop_rgb, mask_panel = None, None, None

    try:
        data = _select_larva_from_later_date()

        # Panel 1: Global detection overlay (full plate)
        overlay_panel = cv2.cvtColor(data['overlay_img'], cv2.COLOR_BGR2RGB)

        # Panel 2: Larva crop (zoom-in on the chosen larva)
        larva_crop_rgb = cv2.cvtColor(data['larva_crop_bgr'], cv2.COLOR_BGR2RGB)

        # Panel 3: Binary mask + morphometric text (clean, centered)
        mask_h, mask_w = data['larva_crop_mask'].shape
        mask_panel = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
        mask_panel[data['larva_crop_mask'] > 0] = [255, 255, 255]

        # Morphometric text (area + lengths) placed close to top of mask
        text_lines = [
            f"Area: {data['area_px']} px",
            f"Length: {data['length_px']:.1f} px",
        ]
        if np.isfinite(data['length_mm']):
            text_lines.append(f"Length: {data['length_mm']:.2f} mm")

        y_text = 24
        for t in text_lines:
            cv2.putText(
                mask_panel,
                t,
                (8, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 100, 100),
                2,
                cv2.LINE_AA,
            )
            y_text += 26
    except Exception as e:
        print(f"❌  Error rendering Figure 3: {e}")

    _place_horizontal_panels(
        [overlay_panel, larva_crop_rgb, mask_panel],
        captions,
        border_cols,
        SCRIPT_DIR / 'pipeline_flow_diagram_figure3.png',
        img_w=5.2,
        img_h=5.0,
    )
