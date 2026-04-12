#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from filter_larvae_by_confidence import _crop_overlay_for_larva, PIXEL_TO_MM  # type: ignore
except Exception:
    _crop_overlay_for_larva = None
    PIXEL_TO_MM = 0.232255814

try:
    from run_pipeline_binary_masks import calculate_larva_morphometrics  # type: ignore
except Exception:
    calculate_larva_morphometrics = None


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis_full_binary_masks_only"
FILTERED_DIR = ROOT / "filtered_larvae_by_date"
OUT_PATH = ROOT / "outputs" / "figures" / "figure3_larva_pipeline.png"


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.uint8)


def _largest_larva() -> tuple[Path, str, int]:
    pat = re.compile(r"^larva_\d+\.png$")
    best = None
    for date_dir in sorted(FILTERED_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        for f in sorted(date_dir.iterdir()):
            if not f.is_file() or not pat.match(f.name):
                continue
            area = int(np.count_nonzero(_read_gray(f) > 0))
            if area <= 0:
                continue
            if best is None or area > best[2]:
                best = (f, date_dir.name, area)
    if best is None:
        raise RuntimeError("No larva masks found")
    return best


def _locate_image(date_str: str, larva_filename: str, target_area: int) -> str:
    best = None
    for d in sorted((ANALYSIS_DIR / date_str).iterdir()):
        if not d.is_dir():
            continue
        cand = d / "larvae_reports" / larva_filename
        if not cand.exists():
            continue
        area = int(np.count_nonzero(_read_gray(cand) > 0))
        diff = abs(area - target_area)
        if best is None or diff < best[1]:
            best = (d.name, diff)
        if diff == 0:
            return d.name
    if best is None:
        raise RuntimeError("Could not map larva to source image")
    return best[0]


def _crop_overlay_local(date_str: str, image_name: str, larva_filename: str) -> np.ndarray | None:
    overlay = ANALYSIS_DIR / date_str / image_name / "overlay.png"
    maskp = ANALYSIS_DIR / date_str / image_name / "larvae_reports" / larva_filename
    morph = ANALYSIS_DIR / date_str / image_name / "morphometrics.csv"
    if not overlay.exists() or not maskp.exists() or not morph.exists():
        return None
    mask = _read_gray(maskp)
    src = cv2.imread(str(overlay), cv2.IMREAD_COLOR)
    if src is None:
        return None
    h, w = mask.shape[:2]
    larva_id = int(Path(larva_filename).stem.split("_")[-1])
    cx = cy = None
    with open(morph, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(float(row["larva_id"])) != larva_id:
                    continue
            except Exception:
                continue
            cx = float(row["centroid_x"])
            cy = float(row["centroid_y"])
            break
    if cx is None or cy is None:
        return None
    pad = 0.20
    h2 = h + 2 * int(round(h * pad))
    w2 = w + 2 * int(round(w * pad))
    x0 = int(round(cx - w2 / 2))
    y0 = int(round(cy - h2 / 2))
    x1, y1 = x0 + w2, y0 + h2
    x0 = max(0, min(x0, src.shape[1] - 1))
    y0 = max(0, min(y0, src.shape[0] - 1))
    x1 = max(1, min(x1, src.shape[1]))
    y1 = max(1, min(y1, src.shape[0]))
    crop = src[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


def _metrics_csv(date_str: str, image_name: str, larva_filename: str) -> dict:
    morph = ANALYSIS_DIR / date_str / image_name / "morphometrics.csv"
    larva_id = int(Path(larva_filename).stem.split("_")[-1])
    with open(morph, "r", newline="") as f:
        for r in csv.DictReader(f):
            try:
                if int(float(r["larva_id"])) != larva_id:
                    continue
            except Exception:
                continue
            return {
                "area": float(r.get("area", 0) or 0),
                "body_length": float(r.get("body_length", 0) or 0),
                "mean_width": float(r.get("mean_width", 0) or 0),
                "curvature_ratio": float(r.get("curvature_ratio", 0) or 0),
                "eccentricity": float(r.get("eccentricity", 0) or 0),
                "solidity": float(r.get("solidity", 0) or 0),
            }
    return {}


def _tight_crop(gray: np.ndarray, mask01: np.ndarray, margin: int = 12) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return gray, mask01
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(gray.shape[0] - 1, y1 + margin)
    x1 = min(gray.shape[1] - 1, x1 + margin)
    return gray[y0 : y1 + 1, x0 : x1 + 1], mask01[y0 : y1 + 1, x0 : x1 + 1]


def _crop_from_mask(gray: np.ndarray, mask01: np.ndarray, margin: int = 6) -> np.ndarray:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return gray
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(gray.shape[0] - 1, y1 + margin)
    x1 = min(gray.shape[1] - 1, x1 + margin)
    return gray[y0 : y1 + 1, x0 : x1 + 1]


def _clean_overlay_annotations(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Remove colored overlay annotations (IDs/boxes) from overlay crops.
    We detect non-gray (high chroma) pixels and inpaint them on grayscale.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Invalid crop image")

    b = crop_bgr[:, :, 0].astype(np.int16)
    g = crop_bgr[:, :, 1].astype(np.int16)
    r = crop_bgr[:, :, 2].astype(np.int16)
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # Green/red overlay drawings and text usually have strong channel imbalance.
    anno_mask = (chroma > 22).astype(np.uint8) * 255
    if np.count_nonzero(anno_mask) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anno_mask = cv2.dilate(anno_mask, k, iterations=1)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if np.count_nonzero(anno_mask) == 0:
        return gray

    cleaned = cv2.inpaint(gray, anno_mask, 3, cv2.INPAINT_TELEA)
    return cleaned.astype(np.uint8)


def _crop_zoom_petri(overlay_bgr: np.ndarray) -> np.ndarray:
    """
    Crop around the Petri dish and zoom in.
    Keeps color overlays (IDs / boxes) intact.
    """
    h, w = overlay_bgr.shape[:2]
    gray = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

    # Try robust dish detection as a large circle.
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=100,
        param2=30,
        minRadius=int(0.25 * min(h, w)),
        maxRadius=int(0.52 * min(h, w)),
    )

    if circles is not None and len(circles) > 0:
        c = circles[0][0]
        cx, cy, r = int(round(c[0])), int(round(c[1])), int(round(c[2]))
        pad = int(round(0.08 * r))
        x0 = max(0, cx - r - pad)
        y0 = max(0, cy - r - pad)
        x1 = min(w, cx + r + pad)
        y1 = min(h, cy + r + pad)
        crop = overlay_bgr[y0:y1, x0:x1]
        if crop.size > 0:
            return crop

    # Fallback: crop to largest non-dark component.
    _, bw = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        pad_x = int(round(0.05 * cw))
        pad_y = int(round(0.05 * ch))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + cw + pad_x)
        y1 = min(h, y + ch + pad_y)
        crop = overlay_bgr[y0:y1, x0:x1]
        if crop.size > 0:
            return crop

    return overlay_bgr


def _render(detect_context_rgb: np.ndarray, original: np.ndarray, mask255: np.ndarray, metrics: dict) -> None:
    fig = plt.figure(figsize=(12, 3.5), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.10)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    titles = ["Detection", "Original larva", "Binary mask", ""]
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=11, pad=5, fontweight="normal")
        ax.axis("off")
        ax.set_facecolor("white")

    # Add small margins (2-5%) around image content.
    def _pad_img(img: np.ndarray, frac: float = 0.03, fill: int = 255) -> np.ndarray:
        h, w = img.shape[:2]
        py = max(1, int(round(h * frac)))
        px = max(1, int(round(w * frac)))
        if img.ndim == 2:
            out = np.full((h + 2 * py, w + 2 * px), fill, dtype=img.dtype)
            out[py : py + h, px : px + w] = img
        else:
            out = np.full((h + 2 * py, w + 2 * px, img.shape[2]), fill, dtype=img.dtype)
            out[py : py + h, px : px + w, :] = img
        return out

    det_show = _pad_img(detect_context_rgb, frac=0.03, fill=255)
    org_show = _pad_img(original, frac=0.03, fill=255)
    mask_show = _pad_img(mask255, frac=0.03, fill=0)

    # Detection panel: full Petri dish context with all detections.
    axs[0].imshow(det_show, interpolation="bilinear")

    # Original panel: preserve natural grayscale intensity (no remapping).
    axs[1].imshow(org_show, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    axs[2].imshow(mask_show, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    area = float(metrics.get("area", 0))
    mw = float(metrics.get("mean_width", 0))
    cr = float(metrics.get("curvature_ratio", 0))
    ec = float(metrics.get("eccentricity", 0))
    so = float(metrics.get("solidity", 0))
    label_w = 15
    rows_text = "\n".join(
        [
            f"{'Area':<{label_w}} {area:.0f} px",
            f"{'Mean width':<{label_w}} {mw:.1f} px",
            f"{'Curvature ratio':<{label_w}} {cr:.3f}",
            f"{'Eccentricity':<{label_w}} {ec:.3f}",
            f"{'Solidity':<{label_w}} {so:.3f}",
        ]
    )
    axs[3].text(
        0.10,
        0.5 + 0.20,
        "Morphometric measurements",
        transform=axs[3].transAxes,
        ha="left",
        va="center",
        fontsize=10,
        linespacing=1.2,
        family="sans-serif",
        color="black",
    )
    axs[3].text(
        0.10,
        0.5 - 0.02,
        rows_text,
        transform=axs[3].transAxes,
        ha="left",
        va="center",
        fontsize=9,
        linespacing=1.32,
        family="sans-serif",
        color="black",
    )

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    mask_path, date_str, area = _largest_larva()
    larva_filename = mask_path.name
    image_name = _locate_image(date_str, larva_filename, area)

    # Full detection-context panel from overlay.png (all detections).
    overlay_full_path = ANALYSIS_DIR / date_str / image_name / "overlay.png"
    overlay_full_bgr = cv2.imread(str(overlay_full_path), cv2.IMREAD_COLOR)
    if overlay_full_bgr is None:
        raise RuntimeError(f"Could not read full overlay: {overlay_full_path}")
    detect_context_bgr = _crop_zoom_petri(overlay_full_bgr)
    detect_context_rgb = cv2.cvtColor(detect_context_bgr, cv2.COLOR_BGR2RGB)

    crop = None
    if _crop_overlay_for_larva is not None:
        crop = _crop_overlay_for_larva(date_str, image_name, larva_filename)
    if crop is None:
        crop = _crop_overlay_local(date_str, image_name, larva_filename)
    if crop is None:
        raise RuntimeError("Could not crop detection panel")

    detect = _clean_overlay_annotations(crop)
    mask_small = (_read_gray(mask_path) > 0).astype(np.uint8)
    mask01 = cv2.resize(mask_small, (detect.shape[1], detect.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    # Keep more context to avoid over-cropped appearance.
    detect, mask01 = _tight_crop(detect, mask01, margin=18)
    # Original larva panel should visually match detection intensity:
    # use the same grayscale source, cropped tighter around larva, without white-out.
    original = _crop_from_mask(detect, mask01, margin=6)
    mask255 = (mask01 * 255).astype(np.uint8)

    if calculate_larva_morphometrics is not None:
        bh = cv2.morphologyEx(detect, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        metrics, _ = calculate_larva_morphometrics(1, mask01.astype(np.uint8), detect, bh)
    else:
        metrics = _metrics_csv(date_str, image_name, larva_filename)

    _render(detect_context_rgb, original, mask255, metrics)


if __name__ == "__main__":
    main()

