#!/usr/bin/env python3
"""
Publication pipeline figure (2×3 panels A–F), aligned with the paper methodology.

Loads ML predictions from:
  dual_larva_models_geodesic2/predictions/predictions_all_larvae.xlsx

ML retention rule (panels D–F ONLY):
  predicted_valid == 1
  AND valid_confidence >= 0.85
  AND predicted_posture == 1

Panels:
  (A) Raw RGB image (<date>/images/<image_name>.*)
  (B) Processed binary mask: segmentation_mask.png from analysis output
  (C) Rule-based candidates: overlay.png (green boxes after geometric filtering — NOT ML)
  (D) Larva retained after hierarchical ML filtering: original RGB crop (no binary, no heuristic)
  (E) Binary mask of the selected larva: larvae_reports/<larva_filename>
  (F) PCA major-axis length on the binary mask (small annotation text)

Outputs:
  outputs/figures/pipeline_figure_paper.png
  outputs/figures/pipeline_figure_paper_caption.txt

Reproducibility: set environment variable PIPELINE_FIGURE_SEED=<int> for deterministic larva choice.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "analysis_full_binary_masks_only"
PRED_XLSX = ROOT / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_SIZE = 11
LABEL_FONT_SIZE = 13
DPI = 300
VALID_CONF_THRESHOLD = 0.85
PIXEL_TO_MM = 0.232255814

plt.rcParams.update({"font.size": FONT_SIZE, "font.family": "sans-serif"})


def fix_date(d) -> str:
    s = str(d).strip()
    if s.endswith(".0"):
        s = s[:-2]
    parts = s.split(".")
    if len(parts) == 2 and parts[1].strip() == "1":
        return f"{parts[0].strip()}.10"
    return s


def _resolve_prediction_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    out = {}
    out["date"] = cols.get("date") or cols.get("img_date") or cols.get("image_date")
    out["image_name"] = cols.get("image_name") or cols.get("image") or cols.get("img")
    out["larva_filename"] = cols.get("larva_filename") or cols.get("larva_file") or cols.get("larva")
    out["predicted_valid"] = cols.get("predicted_valid") or cols.get("valid_pred") or cols.get("pred_valid")
    out["predicted_posture"] = (
        cols.get("predicted_posture") or cols.get("posture_pred") or cols.get("pred_posture")
    )
    out["valid_confidence"] = cols.get("valid_confidence") or cols.get("valid_conf")
    out["posture_confidence"] = cols.get("posture_confidence") or cols.get("posture_conf")
    required = ["date", "image_name", "larva_filename", "predicted_valid", "predicted_posture", "valid_confidence"]
    missing = [k for k in required if out.get(k) is None]
    if missing:
        raise RuntimeError(f"predictions xlsx missing required column(s): {missing}; have {list(df.columns)}")
    if out["posture_confidence"] is None:
        raise RuntimeError("predictions xlsx must include posture_confidence for validation logging")
    return out  # type: ignore[return-value]


def load_predictions() -> tuple[pd.DataFrame, dict[str, str]]:
    if not PRED_XLSX.exists():
        raise FileNotFoundError(f"Predictions workbook not found: {PRED_XLSX}")
    df = pd.read_excel(PRED_XLSX, engine="openpyxl")
    c = _resolve_prediction_columns(df)
    return df, c


def ml_pass_mask(df: pd.DataFrame, c: dict[str, str]) -> pd.Series:
    pv = pd.to_numeric(df[c["predicted_valid"]], errors="coerce")
    pp = pd.to_numeric(df[c["predicted_posture"]], errors="coerce")
    vc = pd.to_numeric(df[c["valid_confidence"]], errors="coerce")
    return (pv == 1) & (vc >= VALID_CONF_THRESHOLD) & (pp == 1)


def analysis_dir_for_row(date_fixed: str, image_name: str) -> Path:
    return DATA_ROOT / date_fixed / str(image_name)


def resolve_raw_bgr(date_fixed: str, image_name: str) -> tuple[np.ndarray, Path]:
    """Load original RGB photo from <ROOT>/<date>/images/<name>.(JPG|jpg|png|...). No guessing which photo."""
    images_dir = ROOT / date_fixed / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory missing: {images_dir}")
    stem = str(image_name)
    matches = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.stem == stem:
            matches.append(p)
    if not matches:
        raise FileNotFoundError(
            f"No source image with stem '{stem}' under {images_dir} (exact match required)."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple source images share stem '{stem}' in {images_dir}: {[m.name for m in matches]}"
        )
    path = matches[0]
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read raw image: {path}")
    return bgr, path


def require_analysis_file(p: Path, label: str) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {label} (required for paper figure): {p}")


def crop_original_rgb_for_larva(
    date_fixed: str,
    image_name: str,
    larva_filename: str,
    raw_bgr: np.ndarray,
) -> np.ndarray:
    """
    Panel (D): RGB crop from the original photo, centered like the pipeline crop.
    Uses morphometrics.csv centroid + larva mask dimensions (same contract as filter_larvae_by_confidence).
    """
    adir = analysis_dir_for_row(date_fixed, image_name)
    mask_path = adir / "larvae_reports" / larva_filename
    morph_path = adir / "morphometrics.csv"
    require_analysis_file(mask_path, "larva binary mask")
    require_analysis_file(morph_path, "morphometrics.csv")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    h, w = mask.shape[:2]
    if h < 2 or w < 2:
        raise RuntimeError(f"Degenerate mask size at {mask_path}")

    stem = Path(larva_filename).stem
    try:
        larva_id = int(stem.split("_")[-1])
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Cannot parse larva_id from filename {larva_filename!r}") from e

    mdf = pd.read_csv(morph_path)
    row = mdf[mdf["larva_id"] == larva_id]
    if len(row) == 0:
        raise RuntimeError(f"larva_id {larva_id} not found in {morph_path}")

    cx = float(row.iloc[0]["centroid_x"])
    cy = float(row.iloc[0]["centroid_y"])

    pad_frac = 0.20
    h_pad = int(round(h * pad_frac))
    w_pad = int(round(w * pad_frac))
    h2 = h + 2 * h_pad
    w2 = w + 2 * w_pad

    x0 = int(round(cx - w2 / 2))
    y0 = int(round(cy - h2 / 2))
    x1 = x0 + w2
    y1 = y0 + h2

    x0 = max(0, min(x0, raw_bgr.shape[1] - 1))
    y0 = max(0, min(y0, raw_bgr.shape[0] - 1))
    x1 = max(1, min(x1, raw_bgr.shape[1]))
    y1 = max(1, min(y1, raw_bgr.shape[0]))

    crop = raw_bgr[y0:y1, x0:x1].copy()
    if crop.size == 0:
        raise RuntimeError("Empty RGB crop for panel (D)")

    if crop.shape[0] != h2 or crop.shape[1] != w2:
        out = np.zeros((h2, w2, 3), dtype=np.uint8)
        out[: crop.shape[0], : crop.shape[1]] = crop
        crop = out
    return crop


def panel_binary_mask_clean(mask_path: Path) -> np.ndarray:
    """Panel (E): white larva on black from saved mask PNG."""
    g = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Cannot read mask for panel (E): {mask_path}")
    binm = (g > 0).astype(np.uint8) * 255
    return cv2.cvtColor(binm, cv2.COLOR_GRAY2RGB)


def compute_pca_axis_on_mask(mask_uint8: np.ndarray) -> tuple[float, tuple[int, int], tuple[int, int]]:
    if mask_uint8.ndim == 3:
        m = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)
    else:
        m = mask_uint8
    ys, xs = np.where(m > 0)
    if len(xs) < 5:
        return 0.0, (0, 0), (0, 0)
    pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projections = centered @ direction
    minp, maxp = projections.min(), projections.max()
    p1 = centroid + minp * direction
    p2 = centroid + maxp * direction
    length_px = float(maxp - minp)
    return length_px, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))


def panel_pca_morphometric(mask_gray: np.ndarray) -> np.ndarray:
    """Panel (F): binary-style background + PCA axis + small mm label."""
    length_px, p1, p2 = compute_pca_axis_on_mask(mask_gray)
    length_mm = length_px * PIXEL_TO_MM
    larva = (mask_gray > 0).astype(np.uint8) * 255
    canvas = np.ones((mask_gray.shape[0], mask_gray.shape[1], 3), dtype=np.uint8) * 255
    canvas[larva > 0] = (40, 40, 40)
    if length_px > 0:
        cv2.line(canvas, p1, p2, (34, 139, 34), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, (255, 0, 0), -1, lineType=cv2.LINE_AA)
    text = f"{length_mm:.2f} mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.38
    th = 1
    (tw, th_px), _ = cv2.getTextSize(text, font, fs, th)
    cv2.rectangle(canvas, (4, 4), (4 + tw + 6, 4 + th_px + 6), (255, 255, 255), -1)
    cv2.putText(canvas, text, (6, 4 + th_px + 2), font, fs, (0, 0, 0), th, cv2.LINE_AA)
    return canvas


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def select_image_and_larva(
    df: pd.DataFrame, c: dict[str, str], rng: random.Random
) -> tuple[str, str, pd.Series, Path]:
    """
    For each (date, image) with ≥1 ML-pass larva and complete files, collect eligible rows.
    Shuffle images, pick one, then rng.choice among ML-pass larvae with existing mask + morphometrics.
    """
    df = df.copy()
    df["_date_f"] = df[c["date"]].map(fix_date)
    df["_ml"] = ml_pass_mask(df, c)
    ml = df[df["_ml"]].copy()
    if ml.empty:
        raise RuntimeError(
            f"No rows satisfy ML rule: predicted_valid==1, valid_confidence>={VALID_CONF_THRESHOLD}, predicted_posture==1"
        )

    by_image: dict[tuple[str, str], list[pd.Series]] = {}
    for _, row in ml.iterrows():
        date_f = str(row["_date_f"])
        img_name = str(row[c["image_name"]])
        larva_fn = str(row[c["larva_filename"]])
        adir = analysis_dir_for_row(date_f, img_name)
        try:
            require_analysis_file(adir / "segmentation_mask.png", "segmentation_mask.png")
            require_analysis_file(adir / "overlay.png", "overlay.png (rule-based candidates)")
            require_analysis_file(adir / "larvae_reports" / larva_fn, "larva mask PNG")
            require_analysis_file(adir / "morphometrics.csv", "morphometrics.csv")
        except FileNotFoundError:
            continue
        key = (date_f, img_name)
        by_image.setdefault(key, []).append(row)

    if not by_image:
        raise RuntimeError(
            "No (date, image) has ML-pass larvae with full analysis outputs under "
            f"{DATA_ROOT} (segmentation_mask, overlay, larvae_reports/<larva_filename>, morphometrics.csv)."
        )

    keys = list(by_image.keys())
    rng.shuffle(keys)
    last_err: Exception | None = None
    for key in keys:
        date_f, img_name = key
        rows = by_image[key]
        try:
            raw_bgr, _ = resolve_raw_bgr(date_f, img_name)
        except (FileNotFoundError, RuntimeError) as e:
            last_err = e
            continue
        eligible: list[pd.Series] = []
        for row in rows:
            larva_fn = str(row[c["larva_filename"]])
            try:
                _ = crop_original_rgb_for_larva(date_f, img_name, larva_fn, raw_bgr)
            except (FileNotFoundError, RuntimeError) as e:
                last_err = e
                continue
            eligible.append(row)
        if not eligible:
            continue
        row = rng.choice(eligible)
        larva_fn = str(row[c["larva_filename"]])
        mask_path = analysis_dir_for_row(date_f, img_name) / "larvae_reports" / larva_fn
        return date_f, img_name, row, mask_path

    raise RuntimeError(
        "Could not find an image with at least one ML-pass larva and readable raw photo + crops. "
        f"Last error: {last_err!r}"
    )


def print_validation(row: pd.Series, c: dict[str, str], date_f: str, img_name: str) -> None:
    pv = int(pd.to_numeric(row[c["predicted_valid"]], errors="coerce"))
    pp = int(pd.to_numeric(row[c["predicted_posture"]], errors="coerce"))
    vc = float(pd.to_numeric(row[c["valid_confidence"]], errors="coerce"))
    pc = float(pd.to_numeric(row[c["posture_confidence"]], errors="coerce"))
    larva_fn = str(row[c["larva_filename"]])
    print("=== ML validation (mandatory) ===")
    print(f"  date (normalized): {date_f}")
    print(f"  image_name:        {img_name}")
    print(f"  larva_filename:    {larva_fn}")
    print(f"  predicted_valid:   {pv}  (required: 1)")
    print(f"  valid_confidence:  {vc:.6f}  (required: >= {VALID_CONF_THRESHOLD})")
    print(f"  predicted_posture: {pp}  (required: 1)")
    print(f"  posture_confidence:{pc:.6f}")
    ok = pv == 1 and vc >= VALID_CONF_THRESHOLD and pp == 1
    print(f"  PASS ML rule:      {ok}")
    if not ok:
        raise RuntimeError("Internal error: selected row does not satisfy ML filtering rule")
    print("=================================")


def build_panels(date_f: str, img_name: str, larva_fn: str) -> list[np.ndarray]:
    raw_bgr, raw_path = resolve_raw_bgr(date_f, img_name)
    print(f"Source image (panel A): {raw_path}")
    adir = analysis_dir_for_row(date_f, img_name)
    seg_path = adir / "segmentation_mask.png"
    overlay_path = adir / "overlay.png"
    mask_path = adir / "larvae_reports" / larva_fn

    require_analysis_file(seg_path, "segmentation_mask.png")
    require_analysis_file(overlay_path, "overlay.png")

    A = bgr_to_rgb(raw_bgr)

    seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if seg is None:
        raise RuntimeError(f"Failed to read segmentation mask: {seg_path}")
    B = bgr_to_rgb(cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR))

    over = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if over is None:
        raise RuntimeError(f"Failed to read overlay: {overlay_path}")
    C = bgr_to_rgb(over)

    d_bgr = crop_original_rgb_for_larva(date_f, img_name, larva_fn, raw_bgr)
    D = bgr_to_rgb(d_bgr)

    E = panel_binary_mask_clean(mask_path)

    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise RuntimeError(f"Failed to read larva mask for panel F: {mask_path}")
    F_bgr = panel_pca_morphometric(mask_gray)
    F = cv2.cvtColor(F_bgr, cv2.COLOR_BGR2RGB)

    print(f"Analysis dir: {adir}")
    print(f"Larva mask:   {mask_path}")
    stem = Path(larva_fn).stem
    try:
        rid = int(stem.split("_")[-1])
        report_path = adir / "larvae_reports" / f"report_{rid:03d}.png"
        if report_path.is_file():
            print(f"Report image (paired): {report_path}")
        else:
            print(f"Report image: no file at expected {report_path} (binary-masks-only runs may omit reports)")
    except (ValueError, IndexError):
        print("Report image: could not derive report_NNN.png from larva filename")
    return [A, B, C, D, E, F]


def save_figure_paper(panels: list[np.ndarray], out_path: Path) -> None:
    labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]
    titles = [
        "Raw image",
        "Binary segmentation",
        "Candidate detections\n(rule-based filtering)",
        "Larva retained after hierarchical\nmachine learning filtering",
        "Binary mask (selected larva)",
        "Morphometric extraction\n(PCA major-axis length)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    axes_flat = axes.flatten()
    for ax, im, lab, title in zip(axes_flat, panels, labels, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=FONT_SIZE)
        ax.axis("off")
        ax.text(
            0.02,
            0.03,
            lab,
            transform=ax.transAxes,
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=2),
        )
    plt.tight_layout(pad=0.6)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_caption(path: Path) -> None:
    text = (
        "Figure. Overview of the automated image-analysis pipeline for larval morphometrics. "
        "(A) Raw RGB photograph of the experimental arena. "
        "(B) Binary segmentation mask after preprocessing. "
        "(C) Candidate detections retained after rule-based (geometric) filtering, shown on the detection overlay. "
        "(D) Example of a single larva retained after hierarchical machine learning filtering "
        f"(predicted valid, valid_confidence≥{VALID_CONF_THRESHOLD:g}, predicted posture class 1), "
        "displayed as an RGB crop from the original image. "
        "(E) Binary mask of that same larva. "
        "(F) Morphometric extraction: principal-axis length estimated from the binary mask (PCA/SVD), with length in millimetres. "
        "Panels (D)–(F) use one larva drawn at random from the set that passes the ML filter for the selected field-of-view; "
        "panel (D) is not derived from heuristic area or aspect-ratio rules."
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    seed_s = os.environ.get("PIPELINE_FIGURE_SEED", "").strip()
    seed = int(seed_s) if seed_s else random.randrange(1 << 30)
    rng = random.Random(seed)
    print(f"PIPELINE_FIGURE_SEED={seed} (set env PIPELINE_FIGURE_SEED to reproduce larva choice)\n")

    df, c = load_predictions()
    date_f, img_name, row, _mask_path = select_image_and_larva(df, c, rng)
    print_validation(row, c, date_f, img_name)

    larva_fn = str(row[c["larva_filename"]])
    panels = build_panels(date_f, img_name, larva_fn)

    out_png = OUT_DIR / "pipeline_figure_paper.png"
    out_cap = OUT_DIR / "pipeline_figure_paper_caption.txt"
    save_figure_paper(panels, out_png)
    write_caption(out_cap)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_cap}")


if __name__ == "__main__":
    main()
