#!/usr/bin/env python3
"""
pipeline_flow_diagram.py
========================
Two-row publication-quality pipeline diagram for run_pipeline.py.

Row 1 (stages 1-5):  Pre-Processing  →  Segmentation
Row 2 (stages 6-10): Filtering  →  Detection Overlay  →  Per-Larva Report

Every panel shows the FULL petri-dish image at that pipeline stage.
Stages 1-9 are computed live from the raw source JPG.
Stage 10 is the saved per-larva 3-panel crop report.

Fixes applied:
  - Gray connector line routes along the RIGHT margin (avoids caption text)
  - Stage number (1-10) shown in bottom-left corner of every caption area

Output: figures/pipeline_flow_diagram.png
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_IMAGE    = PROJECT_ROOT / "31.10" / "images" / "IMG_7376.JPG"
SAMPLE_BASE  = PROJECT_ROOT / "analysis_full" / "31.10" / "IMG_7376"
LARVA_REPORT = SAMPLE_BASE / "larvae_reports" / "larva_012.png"
LARVA_MASK   = SAMPLE_BASE / "segmentation_mask.png"
LARVA_OVERLAY = SAMPLE_BASE / "overlay.png"
OUTPUT_FILE  = SCRIPT_DIR / "pipeline_flow_diagram.png"

# ── Pipeline parameters (exact match to run_pipeline.py) ──────────────────────
BLACKHAT_KERNEL_SIZE = 25
BG_KERNEL_SIZE       = 50
MORPH_OPEN_SIZE      = 3
MIN_LARVA_AREA       = 80

# ── Colours ────────────────────────────────────────────────────────────────────
COL = {
    "pre":     "#1A5276",   # teal-blue   – pre-processing
    "seg":     "#1A237E",   # indigo      – segmentation
    "filt":    "#4A235A",   # purple      – filtering
    "morph":   "#1B5E20",   # dark green  – overlay
    "out":     "#7B1A1A",   # dark red    – output
    "bg":      "#FFFFFF",
    "title":   "#1B2631",
    "arrow":   "#444444",
    "caption": "#1A1A1A",
}
FONT = "DejaVu Sans"

# ── Stage metadata: (badge_label, caption_text, border_colour) ────────────────
STAGE_META = [
    # ── Row 1 ──────────────────────────────────────────────────────────────
    ("1",  "Grayscale\nInput",                              COL["pre"]),
    ("2",  "Background\nEstimation\n(MORPH_DILATE 50×50)", COL["pre"]),
    ("3",  "Contrast Map\n(absdiff: bg − gray)",           COL["pre"]),
    ("4",  "Blackhat\nTransform\n(MORPH_BLACKHAT 25×25)",  COL["seg"]),
    ("5",  "Saliency Fusion\n+ Otsu Threshold\n(0.7×contrast + 0.3×blackhat)", COL["seg"]),
    # ── Row 2 ──────────────────────────────────────────────────────────────
    ("6",  "Morphological\nCleaning\n(MORPH_OPEN 3×3)",    COL["seg"]),
    ("7",  "Connected\nComponents\n(8-connectivity, colour-coded)", COL["filt"]),
    ("8",  "Final Larva\nMask\n(accepted components)",     COL["filt"]),
    ("9",  "Detection\nOverlay\n(bounding boxes + IDs)",   COL["morph"]),
    ("10", "Per-Larva\nReport\n(crop · mask · skeleton)",  COL["out"]),
]

ROW_LABELS  = [
    "STAGE 1–5  ·  Pre-Processing  →  Segmentation",
    "STAGE 6–10  ·  Filtering  →  Detection Overlay  →  Per-Larva Report",
]
ROW_COLOURS = [COL["seg"], COL["filt"]]


# ── Generate all intermediate images live from the raw source ─────────────────
def generate_stages(raw_path: Path) -> list:
    """
    Re-runs the exact run_pipeline.py segmentation chain on raw_path.
    Returns list of 10 (mode, numpy_array) tuples.
    Full petri-dish view — uniform downscale only, no side crop.
    """
    img_bgr = cv2.imread(str(raw_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open: {raw_path}")

    # Uniform downscale to ≤1400 px wide – preserves full dish
    h, w = img_bgr.shape[:2]
    max_w = 1400
    if w > max_w:
        scale   = max_w / w
        img_bgr = cv2.resize(img_bgr, (max_w, int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1 – grayscale
    s1 = ("gray", gray.copy())

    # 2 – background estimation (MORPH_DILATE)
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (BG_KERNEL_SIZE, BG_KERNEL_SIZE))
    bg   = cv2.morphologyEx(gray, cv2.MORPH_DILATE, k_bg)
    s2   = ("gray", bg.copy())

    # 3 – contrast map (absdiff), normalised for visibility
    diff     = cv2.absdiff(bg, gray)
    diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s3 = ("gray", diff_vis)

    # 4 – blackhat transform, normalised for visibility
    k_bh     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)
    bh_vis   = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s4 = ("gray", bh_vis)

    # 5 – saliency fusion + Otsu threshold
    saliency  = cv2.addWeighted(diff, 0.7, blackhat, 0.3, 0)
    _, thresh = cv2.threshold(saliency, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s5 = ("gray", thresh.copy())

    # 6 – morphological cleaning (MORPH_OPEN)
    k_mo    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_mo)
    s6 = ("gray", cleaned.copy())

    # 7 – connected components, colour-coded
    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8)
    cc_img = np.zeros((*cleaned.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < MIN_LARVA_AREA:
            continue
        cc_img[labels_img == lbl] = rng.integers(60, 255, 3).tolist()
    s7 = ("bgr", cc_img)

    # 8 – final larva mask (pre-computed full-image result)
    mask_img = cv2.imread(str(LARVA_MASK), cv2.IMREAD_GRAYSCALE)
    s8 = ("gray", mask_img if mask_img is not None else cleaned)

    # 9 – detection overlay (pre-computed full-image result)
    overlay_img = cv2.imread(str(LARVA_OVERLAY))
    s9 = ("bgr", overlay_img if overlay_img is not None else img_bgr)

    # 10 – per-larva 3-panel crop report
    report_img = cv2.imread(str(LARVA_REPORT))
    s10 = ("bgr", report_img if report_img is not None else img_bgr)

    return [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]


# ── Draw the two-row diagram ───────────────────────────────────────────────────
def draw(stage_images: list):
    assert len(stage_images) == 10, "Need exactly 10 stage images"

    # ── Layout constants (all in inches) ──────────────────────────────────
    N_COLS  = 5
    IMG_W   = 2.90    # panel image width
    IMG_H   = 2.10    # panel image height
    CAP_H   = 0.66    # caption area below each image
    GAP_X   = 0.10    # horizontal gap between panels
    BAR_H   = 0.28    # section header bar height
    ROW_GAP = 0.55    # gap between bottom of row-1 captions and row-2 bar
    PAD_L   = 0.12
    PAD_R   = 0.18    # slightly wider right margin for elbow connector
    PAD_TOP = 0.50
    PAD_BOT = 0.40

    row_w = N_COLS * IMG_W + (N_COLS - 1) * GAP_X
    fig_w = PAD_L + row_w + PAD_R
    fig_h = (PAD_TOP
             + BAR_H + IMG_H + CAP_H        # row 1
             + ROW_GAP
             + BAR_H + IMG_H + CAP_H        # row 2
             + PAD_BOT)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COL["bg"])

    def fx(inch): return inch / fig_w
    def fy(inch): return inch / fig_h

    # ── Main title ────────────────────────────────────────────────────────



    # ── Place one panel ───────────────────────────────────────────────────
    def place_panel(col_idx, row_idx, mode, img_data, step, caption, border_col):
        # Distance from figure TOP to top of this row's section bar
        row_top = (PAD_TOP
                   + row_idx * (BAR_H + IMG_H + CAP_H + ROW_GAP))

        # Section bar (drawn once per row, at col 0)
        if col_idx == 0:
            bar_ax = fig.add_axes([
                fx(PAD_L),
                fy(fig_h - row_top - BAR_H),
                fx(row_w),
                fy(BAR_H * 0.80),
            ])
            bar_ax.set_facecolor(ROW_COLOURS[row_idx])
            bar_ax.axis("off")
            bar_ax.text(0.5, 0.5, ROW_LABELS[row_idx],
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white",
                        fontfamily=FONT, transform=bar_ax.transAxes)

        # Image axes
        img_left   = PAD_L + col_idx * (IMG_W + GAP_X)
        img_bottom = fig_h - row_top - BAR_H - IMG_H
        ax = fig.add_axes([fx(img_left), fy(img_bottom), fx(IMG_W), fy(IMG_H)])

        if img_data is not None:
            if mode == "gray":
                ax.imshow(img_data, cmap="gray", aspect="auto",
                          interpolation="lanczos", vmin=0, vmax=255)
            else:
                ax.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB),
                          aspect="auto", interpolation="lanczos")
        else:
            ax.set_facecolor("#EEEEEE")

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(border_col); sp.set_linewidth(2.8)

        # Stage number — small, clean — bottom-left inside the image frame
        # Placed just above the bottom border, outside actual image pixels
        ax.text(0.03, 0.03, step,
                ha="left", va="bottom",
                fontsize=9, fontweight="bold",
                color="white", fontfamily=FONT,
                transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0.15",
                          facecolor=border_col, edgecolor="none", alpha=0.75),
                zorder=5)

        # Caption text centred below panel
        cap_cx = img_left + IMG_W / 2
        cap_by = img_bottom - CAP_H + 0.05
        fig.text(fx(cap_cx), fy(cap_by + CAP_H * 0.58),
                 caption,
                 ha="center", va="center",
                 fontsize=7.0, color=COL["caption"],
                 fontfamily=FONT, linespacing=1.4,
                 multialignment="center")

        return ax

    # ── Render all 10 panels ──────────────────────────────────────────────
    panel_axes = []
    for idx, ((mode, img_data), (step, caption, bcol)) in enumerate(
            zip(stage_images, STAGE_META)):
        row_idx = idx // N_COLS
        col_idx = idx % N_COLS
        ax = place_panel(col_idx, row_idx, mode, img_data,
                         step, caption, bcol)
        panel_axes.append((ax, row_idx, col_idx))

    # ── Horizontal arrows within each row ─────────────────────────────────
    for i in range(len(panel_axes) - 1):
        ax_l, r_l, _ = panel_axes[i]
        ax_r, r_r, _ = panel_axes[i + 1]
        if r_l != r_r:
            continue                        # skip cross-row gap
        bb_l = ax_l.get_position()
        bb_r = ax_r.get_position()
        ymid = (bb_l.y0 + bb_l.y1) / 2
        fig.add_artist(
            plt.annotate("",
                         xy=(bb_r.x0, ymid), xytext=(bb_l.x1, ymid),
                         xycoords="figure fraction",
                         textcoords="figure fraction",
                         arrowprops=dict(arrowstyle="-|>",
                                         color=COL["arrow"],
                                         lw=1.8, mutation_scale=13),
                         zorder=10)
        )

    # ── Inter-row connector: just an arrowhead at panel 6 entry ─────────
    bb2 = panel_axes[5][0].get_position()   # first panel row 2
    y_r2_mid = (bb2.y0 + bb2.y1) / 2
    lc = "#888888"

    # Small arrowhead pointing into panel 6 from its left edge
    fig.add_artist(
        plt.annotate("",
                     xy=(bb2.x0, y_r2_mid),
                     xytext=(bb2.x0 + fx(0.06), y_r2_mid),
                     xycoords="figure fraction",
                     textcoords="figure fraction",
                     arrowprops=dict(arrowstyle="-|>",
                                     color=lc, lw=1.8, mutation_scale=12),
                     zorder=6)
    )

    # ── Legend ────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(facecolor=COL["pre"],   edgecolor="none", label="Pre-Processing"),
        mpatches.Patch(facecolor=COL["seg"],   edgecolor="none", label="Segmentation"),
        mpatches.Patch(facecolor=COL["filt"],  edgecolor="none", label="Filtering & Mask"),
        mpatches.Patch(facecolor=COL["morph"], edgecolor="none", label="Detection Overlay"),
        mpatches.Patch(facecolor=COL["out"],   edgecolor="none", label="Per-Larva Report"),
    ]
    fig.legend(handles=patches, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=5,
               fontsize=7.5, framealpha=0.96,
               edgecolor="#CCCCCC", handlelength=1.3,
               title="Pipeline Stage  ·  Source: 31.10 / IMG_7376.JPG",
               title_fontsize=7.0)

    plt.savefig(str(OUTPUT_FILE), dpi=200, bbox_inches="tight",
                facecolor=COL["bg"], edgecolor="none")
    plt.close()
    print(f"  ✓  Saved: {OUTPUT_FILE}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("PIPELINE FLOW DIAGRAM  —  2 rows × 5 panels, full petri dish")
    print(f"Source : {RAW_IMAGE}")
    print("=" * 65)

    if not RAW_IMAGE.exists():
        print(f"❌  Raw image not found: {RAW_IMAGE}")
        return

    print("  Generating pipeline stage images...")
    stages = generate_stages(RAW_IMAGE)
    print(f"  {len(stages)} stages ready.")

    print("  Rendering diagram...")
    draw(stages)

    print("=" * 65)
    print(f"DONE  →  {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()

