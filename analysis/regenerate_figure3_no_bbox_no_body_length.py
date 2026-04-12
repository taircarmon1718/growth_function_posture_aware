#!/opt/anaconda3/bin/python3.12
from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis_full_binary_masks_only"
FILTERED_DIR = ROOT / "filtered_larvae_by_date"
OUT_PATH = ROOT / "outputs" / "figures" / "figure3_larva_pipeline.png"


def _mask_area(mask_gray: np.ndarray) -> int:
    return int(np.count_nonzero(mask_gray > 0))


def _read_gray_cv(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _crop_overlay_like_impl(date_str: str, image_name: str, larva_filename: str, larva_id: int) -> np.ndarray:
    overlay_path = ANALYSIS_DIR / date_str / image_name / "overlay.png"
    larva_mask_path = ANALYSIS_DIR / date_str / image_name / "larvae_reports" / larva_filename
    morph_path = ANALYSIS_DIR / date_str / image_name / "morphometrics.csv"

    if not overlay_path.exists() or not larva_mask_path.exists() or not morph_path.exists():
        raise FileNotFoundError("Missing overlay/mask/morphometrics for cropping")

    mask = _read_gray_cv(larva_mask_path)
    h, w = mask.shape[:2]
    if h < 2 or w < 2:
        raise RuntimeError("Larva mask too small for crop")

    PAD_FRAC = 0.20
    h_pad = int(round(h * PAD_FRAC))
    w_pad = int(round(w * PAD_FRAC))
    h2 = h + 2 * h_pad
    w2 = w + 2 * w_pad

    cx = None
    cy = None
    with open(morph_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["larva_id"]) != larva_id:
                    continue
            except Exception:
                continue
            cx = float(row["centroid_x"])
            cy = float(row["centroid_y"])
            break
    if cx is None or cy is None:
        raise RuntimeError("Could not find centroid for larva")

    src = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"Could not read overlay: {overlay_path}")

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
        raise RuntimeError("Overlay crop empty")

    if crop.shape[0] != h2 or crop.shape[1] != w2:
        out = np.zeros((h2, w2, 3), dtype=np.uint8)
        out[: crop.shape[0], : crop.shape[1]] = crop
        crop = out

    return crop


def _find_largest_filtered_larva() -> tuple[Path, str, int]:
    pattern = re.compile(r"^larva_\d+\.png$")
    best = None  # (mask_path, date_str, area)

    if not FILTERED_DIR.exists():
        raise FileNotFoundError(f"Missing: {FILTERED_DIR}")

    for date_dir in sorted(FILTERED_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        for f in sorted(date_dir.iterdir()):
            if not f.is_file() or not pattern.match(f.name):
                continue
            mask_gray = _read_gray_cv(f)
            area = _mask_area(mask_gray)
            if area <= 0:
                continue
            if best is None or area > best[2]:
                best = (f, date_str, area)

    if best is None:
        raise RuntimeError(f"No valid larva masks found under {FILTERED_DIR}")
    return best


def _locate_analysis_image_name(date_str: str, larva_filename: str, target_area: int) -> str:
    date_root = ANALYSIS_DIR / date_str
    if not date_root.exists():
        raise FileNotFoundError(f"Missing analysis date directory: {date_root}")

    best = None  # (image_name, abs diff)
    for image_dir in sorted(date_root.iterdir()):
        if not image_dir.is_dir():
            continue
        candidate = image_dir / "larvae_reports" / larva_filename
        if not candidate.exists():
            continue
        cand_mask = _read_gray_cv(candidate)
        cand_area = _mask_area(cand_mask)
        diff = abs(int(cand_area) - int(target_area))
        if best is None or diff < best[1]:
            best = (image_dir.name, diff)
        if diff == 0:
            return image_dir.name

    if best is None:
        raise FileNotFoundError(f"Could not locate source for {date_str}/{larva_filename}")
    return best[0]


def _read_morphometrics(date_str: str, image_name: str, larva_id: int) -> dict:
    morph_path = ANALYSIS_DIR / date_str / image_name / "morphometrics.csv"
    if not morph_path.exists():
        raise FileNotFoundError(f"Missing morphometrics: {morph_path}")

    out = {}
    with open(morph_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["larva_id"]) != larva_id:
                    continue
            except Exception:
                continue
            for k in ["area", "mean_width", "curvature_ratio", "eccentricity", "solidity"]:
                if k in row and row[k] != "":
                    out[k] = float(row[k])
            break

    if not out:
        raise RuntimeError("Could not find morphometrics row")
    return out


def _render_measurements_panel(metrics: dict, panel_w: int, panel_h: int) -> Image.Image:
    panel = Image.new("L", (panel_w, panel_h), color=255)
    draw = ImageDraw.Draw(panel)

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/DejaVu Sans.ttf",
    ]
    measurement_font = None
    title_font = None
    for p in candidates:
        try:
            title_font = ImageFont.truetype(p, size=max(24, int(round(panel_h * 0.045))))
            measurement_font = ImageFont.truetype(p, size=max(21, int(round(panel_h * 0.036))))
            break
        except Exception:
            pass
    if title_font is None or measurement_font is None:
        title_font = ImageFont.load_default()
        measurement_font = ImageFont.load_default()

    lines = [
        f"Area: {metrics.get('area', 0.0):.0f} px",
        f"Mean width: {metrics.get('mean_width', 0.0):.2f} px",
        f"Curvature ratio: {metrics.get('curvature_ratio', 0.0):.3f}",
        f"Eccentricity: {metrics.get('eccentricity', 0.0):.3f}",
        f"Solidity: {metrics.get('solidity', 0.0):.3f}",
    ]

    title = "Morphometric measurements"
    margin_left = max(22, int(round(panel_w * 0.06)))
    margin_top = max(18, int(round(panel_h * 0.05)))
    spacing = max(4, int(round(panel_h * 0.015)))

    # Title
    draw.text((margin_left, margin_top), title, fill=0, font=title_font)
    y = margin_top + title_font.getbbox(title)[3] + spacing
    for ln in lines:
        draw.text((margin_left, y), ln, fill=0, font=measurement_font)
        y += measurement_font.getbbox(ln)[3] + int(max(5, round(spacing * 0.8)))

    return panel


def _add_panel_title(panel: Image.Image, title: str, font_size: int) -> Image.Image:
    w, h = panel.size
    title_h = max(54, int(round(h * 0.11)))
    out = Image.new("L", (w, h + title_h), color=255)
    out.paste(panel, (0, title_h))
    draw = ImageDraw.Draw(out)

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/DejaVu Sans.ttf",
    ]
    font = None
    for p in candidates:
        try:
            font = ImageFont.truetype(p, size=font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    tw = int(draw.textlength(title, font=font))
    tx = max(8, (w - tw) // 2)
    ty = max(6, int(round((title_h - font.getbbox(title)[3]) * 0.45)))
    draw.text((tx, ty), title, fill=0, font=font)
    return out


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mask_path, date_str, _ = _find_largest_filtered_larva()
    larva_filename = mask_path.name
    stem = Path(larva_filename).stem
    larva_id = int(stem.split("_")[-1])

    mask_gray_small = _read_gray_cv(mask_path)
    mask_bin_small = (mask_gray_small > 0).astype(np.uint8)
    area_px = _mask_area(mask_gray_small)

    image_name = _locate_analysis_image_name(date_str, larva_filename, area_px)

    crop_bgr = _crop_overlay_like_impl(date_str, image_name, larva_filename, larva_id)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    gray_crop_u8 = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # Align mask to crop size.
    crop_h, crop_w = gray_crop_u8.shape[:2]
    mask_resized = cv2.resize(mask_bin_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    clean_gray = gray_crop_u8.copy()
    clean_gray[mask_resized == 0] = 255
    mask_vis = (mask_resized * 255).astype(np.uint8)

    # Measurements (panel 4)
    metrics = _read_morphometrics(date_str, image_name, larva_id)
    # Upscale to publication-ready resolution.
    target_panel_h = 762  # close to the previous matplotlib output height at 300 dpi
    scale = target_panel_h / float(crop_h)

    panel_w_target = max(1, int(round(crop_w * scale)))
    gap = max(8, int(round(target_panel_h * 0.012)))

    # Make the measurements panel clearly larger.
    panel4_w = max(560, int(round(target_panel_h * 0.85)))
    panel4_h = target_panel_h

    p1_base = Image.fromarray(gray_crop_u8, mode="L").resize(
        (panel_w_target, panel4_h),
        resample=Image.Resampling.LANCZOS,
    )
    p3_base = Image.fromarray(mask_vis, mode="L").resize(
        (panel_w_target, panel4_h),
        resample=Image.Resampling.NEAREST,
    )
    p4_base = _render_measurements_panel(metrics, panel_w=panel4_w, panel_h=panel4_h)

    # Add titles to each panel for publication-style presentation.
    title_size = max(18, int(round(panel4_h * 0.035)))
    p1 = _add_panel_title(p1_base, "Detection view", title_size)
    p3 = _add_panel_title(p3_base, "Binary mask", title_size)
    p4 = _add_panel_title(p4_base, "Measurements", title_size)

    # Panel order: 1 (detection) -> 3 (mask) -> 4 (text)
    canvas_w = panel_w_target * 2 + panel4_w + gap * 2
    canvas_h = p1.size[1]
    canvas = Image.new("L", (canvas_w, canvas_h), color=255)

    x = 0
    for idx, p in enumerate([p1, p3, p4]):
        canvas.paste(p, (x, 0))
        x += p.size[0]
        if idx < 2:
            x += gap

    canvas.save(str(OUT_PATH), dpi=(300, 300))


if __name__ == "__main__":
    main()

