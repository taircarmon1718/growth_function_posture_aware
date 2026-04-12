#!/usr/bin/env python3
"""
Enhance raster images for small-format graphical abstracts (e.g. ~5 × 13 cm at 300+ DPI).

- Upscale (Lanczos) if below a target print width so thin lines stay visible when scaled.
- Slight contrast on luminance only (CLAHE in LAB) — preserves hue/scientific color relationships.
- Mild unsharp mask on luminance only — sharper edges and lines without halos if parameters stay moderate.
- Optional very light chroma smoothing (a/b in LAB) to reduce JPEG color speckle — does not touch L (detail).

Does not warp geometry, crop content, or invent structures. Avoids heavy bilateral/NLM to prevent oversmoothing.

Usage:
  python scripts/enhance_publication_images.py \\
      --input figures/for_abstract --output figures/for_abstract/enhanced

  python scripts/enhance_publication_images.py -i path/to/image.png -o path/to/out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _read_rgb(path: Path) -> np.ndarray:
    """uint8 RGB, shape (H,W,3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _save_png(rgb: np.ndarray, path: Path, dpi: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(
        path,
        format="PNG",
        dpi=dpi,
        compress_level=1,
    )


def upscale_min_print_width(
    rgb: np.ndarray,
    *,
    min_width_px: int,
) -> np.ndarray:
    """Lanczos upscale if width is below min_width_px; preserves aspect ratio."""
    h, w = rgb.shape[:2]
    if w >= min_width_px:
        return rgb
    scale = min_width_px / float(w)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    pil = Image.fromarray(rgb)
    pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def clahe_luminance_lab(
    rgb: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Mild CLAHE on L* only; a,b unchanged (color accuracy)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    out_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def mild_chroma_blur_lab(rgb: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    """3×3 blur on a,b only — reduces chroma noise/JPEG blocks; L (detail) untouched."""
    if ksize % 2 == 0:
        ksize += 1
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a2 = cv2.GaussianBlur(a, (ksize, ksize), 0)
    b2 = cv2.GaussianBlur(b, (ksize, ksize), 0)
    merged = cv2.merge((l, a2, b2))
    out_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def unsharp_luminance(
    rgb: np.ndarray,
    *,
    sigma: float = 1.0,
    amount: float = 0.55,
    threshold: int = 2,
) -> np.ndarray:
    """
    Unsharp mask applied to L* only, then merged back.
    `amount` ~0.4–0.7 is typical for publication; threshold skips flat regions to limit noise amplification.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32)
    blur = cv2.GaussianBlur(l_f, (0, 0), sigma)
    high = l_f - blur
    mask = np.abs(high) > threshold
    sharpened = l_f + amount * high
    sharpened = np.where(mask, sharpened, l_f)
    l_out = np.clip(sharpened, 0, 255).astype(np.uint8)
    merged = cv2.merge((l_out, a, b))
    out_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def add_white_padding(rgb: np.ndarray, pad_px: int) -> np.ndarray:
    if pad_px <= 0:
        return rgb
    h, w = rgb.shape[:2]
    out = np.full((h + 2 * pad_px, w + 2 * pad_px, 3), 255, dtype=np.uint8)
    out[pad_px : pad_px + h, pad_px : pad_px + w] = rgb
    return out


def enhance_pipeline(
    rgb: np.ndarray,
    *,
    min_width_px: int,
    clahe_clip: float,
    sharpen_sigma: float,
    sharpen_amount: float,
    sharpen_threshold: int,
    chroma_blur: bool,
    pad_px: int,
) -> np.ndarray:
    x = upscale_min_print_width(rgb, min_width_px=min_width_px)
    x = clahe_luminance_lab(x, clip_limit=clahe_clip, tile_size=8)
    if chroma_blur:
        x = mild_chroma_blur_lab(x, ksize=3)
    x = unsharp_luminance(
        x,
        sigma=sharpen_sigma,
        amount=sharpen_amount,
        threshold=sharpen_threshold,
    )
    x = add_white_padding(x, pad_px)
    return x


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            return [path]
        raise ValueError(f"Not a supported image file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    out = sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Publication-oriented image enhancement for graphical abstracts.")
    ap.add_argument("--input", "-i", type=Path, required=True, help="Image file or folder of images")
    ap.add_argument("--output", "-o", type=Path, required=True, help="Output file or output folder")
    ap.add_argument("--dpi", type=int, default=300, help="PNG DPI metadata (default 300)")
    ap.add_argument(
        "--print-width-cm",
        type=float,
        default=13.0,
        help="Reference print width (cm) for minimum pixel width at --dpi (default 13)",
    )
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit on L* (default 2.0)")
    ap.add_argument("--sharpen-sigma", type=float, default=1.0, help="Gaussian sigma for unsharp base blur")
    ap.add_argument("--sharpen-amount", type=float, default=0.55, help="Unsharp strength on L* (default 0.55)")
    ap.add_argument(
        "--sharpen-threshold",
        type=int,
        default=2,
        help="Skip sharpening where |high-pass| <= this (reduces noise lift; default 2)",
    )
    ap.add_argument(
        "--no-chroma-blur",
        action="store_true",
        help="Disable mild a/b blur (use if source has no JPEG/chroma noise)",
    )
    ap.add_argument("--pad", type=int, default=4, help="White padding in pixels (default 4)")
    args = ap.parse_args()

    min_width_px = int(round(args.print_width_cm / 2.54 * args.dpi))
    dpi_tuple = (args.dpi, args.dpi)

    inputs = collect_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No images found under {args.input}")

    if args.output.suffix.lower() == ".png" and len(inputs) > 1:
        raise SystemExit("Multiple inputs require --output to be a directory, not a single .png")

    for src in inputs:
        if args.output.suffix.lower() == ".png" and len(inputs) == 1:
            dst = args.output
        else:
            dst = args.output / f"{src.stem}_enhanced.png"

        rgb = _read_rgb(src)
        out = enhance_pipeline(
            rgb,
            min_width_px=min_width_px,
            clahe_clip=args.clahe_clip,
            sharpen_sigma=args.sharpen_sigma,
            sharpen_amount=args.sharpen_amount,
            sharpen_threshold=args.sharpen_threshold,
            chroma_blur=not args.no_chroma_blur,
            pad_px=args.pad,
        )
        _save_png(out, dst, dpi_tuple)
        print(f"{src.name} -> {dst} ({out.shape[1]}×{out.shape[0]} px, {args.dpi} DPI meta)")


if __name__ == "__main__":
    main()
