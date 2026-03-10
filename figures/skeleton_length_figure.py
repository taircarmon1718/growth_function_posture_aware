#!/usr/bin/env python3
"""
skeleton_length_figure.py
==========================
Publication-ready 6-panel figure illustrating the skeleton-based
body-length estimation pipeline for a single posture-suitable larva.

Panels (A–F):
  A  Grayscale crop
  B  Binary mask
  C  Skeletonization (red skeleton on mask)
  D  Topology detection (endpoints=blue, junction=yellow)
  E  Branch length computation (each branch a different colour)
  F  Principal branch + length formula and mm conversion

Output:
    figures/skeleton_length_figure.png   (300 dpi, white background)

Read-only — does NOT modify any existing file.
"""

import sys
import random
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────────────────
FIGURES_DIR  = Path(__file__).parent.resolve()
PROJECT_DIR  = FIGURES_DIR.parent
ANALYSIS_DIR = PROJECT_DIR / "analysis_full"
LABELS_FILE  = ANALYSIS_DIR / "larva_quality_labels.xlsx"
OUTPUT_FIG   = FIGURES_DIR / "skeleton_length_figure.png"

PIXEL_TO_MM  = 0.232255814
RANDOM_SEED  = 42
EXCLUDE_DATE = {"18.10", "18.1"}

# Best known specimen (identified by scanning all labeled larvae)
PREFERRED_LARVA = ("3.11", "IMG_8053", "larva_039.png")

# ──────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (matplotlib colour strings)
# ──────────────────────────────────────────────────────────────
# Muted scientific palette — readable at 300 dpi print
BRANCH_COLOURS = ["#c0392b", "#2471a3", "#1e8449", "#7d3c98", "#b7770d"]
PRINCIPAL_COL  = "#c0392b"   # strong red — principal branch
SKEL_GREY      = "#4a4a4a"   # dark grey skeleton
ENDPOINT_COL   = "#2471a3"   # steel blue endpoints
JUNCTION_COL   = "#d4ac0d"   # muted gold junction


# ──────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ──────────────────────────────────────────────────────────────
def _hex_to_rgb01(hex_col: str):
    h = hex_col.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def _tight_bbox(mask: np.ndarray, pad_frac: float = 0.15):
    """
    Return (r0, r1, c0, c1) bounding box of the mask content,
    expanded by pad_frac of the largest dimension, then made square.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        h, w = mask.shape
        return 0, h, 0, w
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    # pad
    H, W    = mask.shape
    rh, rw  = r1 - r0, c1 - c0
    pad     = int(max(rh, rw) * pad_frac)
    r0 = max(0, r0 - pad);  r1 = min(H, r1 + pad)
    c0 = max(0, c0 - pad);  c1 = min(W, c1 + pad)
    # square: expand shorter side
    rh, rw = r1 - r0, c1 - c0
    if rh < rw:
        diff = rw - rh
        r0 = max(0, r0 - diff // 2)
        r1 = min(H, r0 + rw)
    elif rw < rh:
        diff = rh - rw
        c0 = max(0, c0 - diff // 2)
        c1 = min(W, c0 + rh)
    return r0, r1, c0, c1


def _draw_path_on_ax(ax, path, colour, linewidth=1.8, zorder=3):
    if len(path) < 2:
        return
    cols = [p[1] for p in path]
    rows = [p[0] for p in path]
    ax.plot(cols, rows, color=colour, linewidth=linewidth, zorder=zorder,
            solid_capstyle="round", solid_joinstyle="round")


def _strip_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _panel_letter(ax, letter):
    """Bold panel letter, top-left, subtle semi-transparent background."""
    ax.text(0.05, 0.95, letter,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="black",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.75, pad=2.0, boxstyle="round,pad=0.2"))


def _subtitle(ax, text):
    ax.set_xlabel(text, fontsize=9, color="#222222",
                  labelpad=6, loc="center")


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """For every skeleton pixel, count its 8-connected skeleton neighbours."""
    ker = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, ker) * skel.astype(np.uint8)


def _find_branch_endpoints(skel: np.ndarray, junction: tuple):
    """
    Find one true geometric endpoint PER BRANCH emerging from the junction.

    Algorithm:
      1. Remove ONLY the selected junction pixel from a skeleton copy.
         If that doesn't disconnect branches (junction cluster is thick),
         also remove its 8-neighbours that have degree >= 3.
      2. Find connected components of the remaining skeleton.
      3. Merge tiny components (< 3 pixels) into their nearest large component.
      4. For each final component, find the pixel at maximum geodesic distance
         from the junction (computed on the ORIGINAL skeleton).
      5. Reconstruct path endpoint → junction on the original skeleton.

    Returns:
        List of (length, endpoint, path) sorted longest-first.
    """
    jr, jc = junction
    nb_map = _neighbour_count(skel)

    # ── Step 1: try progressively more aggressive junction removal ─
    # Strategy A: remove only the selected junction pixel
    # Strategy B: remove the junction pixel + its degree≥3 8-neighbours
    # Strategy C: remove ALL degree≥3 pixels from skeleton
    # Use the first strategy that produces ≥3 large (≥3px) components.

    strategies = []

    # Strategy A
    cut_a = skel.copy()
    cut_a[jr, jc] = 0
    strategies.append(("junction only", cut_a))

    # Strategy B
    cut_b = skel.copy()
    cut_b[jr, jc] = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = jr + dr, jc + dc
            if (0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]
                    and nb_map[nr, nc] >= 3):
                cut_b[nr, nc] = 0
    strategies.append(("junction + deg≥3 neighbours", cut_b))

    # Strategy C
    cut_c = skel.copy()
    cut_c[(skel > 0) & (nb_map >= 3)] = 0
    strategies.append(("all deg≥3 pixels", cut_c))

    # Pick the first strategy yielding ≥3 components of size ≥3px
    skel_cut = None
    for name, candidate in strategies:
        n_lab, lab = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
        big_count = 0
        for lid in range(1, n_lab):
            if int(np.sum(lab == lid)) >= 2:
                big_count += 1
        if big_count >= 3:
            skel_cut = candidate
            print(f"    Junction removal strategy: {name} → {big_count} branches")
            break

    # Fallback: use the most aggressive strategy even if < 3 branches
    if skel_cut is None:
        skel_cut = cut_c
        n_lab, _ = cv2.connectedComponents(skel_cut.astype(np.uint8), connectivity=8)
        print(f"    Junction removal fallback: all deg≥3 → {n_lab - 1} components")

    n_labels, labels = cv2.connectedComponents(skel_cut.astype(np.uint8), connectivity=8)

    if n_labels <= 1:
        return []

    # ── Step 2: measure component sizes
    comp_sizes = {}
    comp_pixels = {}
    for label_id in range(1, n_labels):
        pixels = list(map(tuple, np.argwhere(labels == label_id)))
        comp_sizes[label_id] = len(pixels)
        comp_pixels[label_id] = pixels

    # Debug: show all component sizes before merging
    sizes_list = sorted(comp_sizes.values(), reverse=True)
    print(f"    Component sizes before merge: {sizes_list}")

    # ── Step 3: merge tiny components (< 2 pixels) into nearest large one
    MIN_COMP = 2
    large_ids = {lid for lid, sz in comp_sizes.items() if sz >= MIN_COMP}
    small_ids = {lid for lid, sz in comp_sizes.items() if sz < MIN_COMP}

    # Build set of large-component pixels for fast lookup
    large_pixel_to_label = {}
    for lid in large_ids:
        for px in comp_pixels[lid]:
            large_pixel_to_label[px] = lid

    for sid in small_ids:
        # Find nearest large-component pixel for any pixel in this small component
        best_lid = None
        best_dist = float('inf')
        for px in comp_pixels[sid]:
            for lid in large_ids:
                for lpx in comp_pixels[lid]:
                    d = abs(px[0] - lpx[0]) + abs(px[1] - lpx[1])
                    if d < best_dist:
                        best_dist = d
                        best_lid = lid
                    if d <= 1:
                        break
                if best_dist <= 1:
                    break
            if best_dist <= 1:
                break
        if best_lid is not None:
            comp_pixels[best_lid].extend(comp_pixels[sid])
            comp_sizes[best_lid] += comp_sizes[sid]
        del comp_pixels[sid]
        del comp_sizes[sid]

    # ── Step 4: for each component, find farthest pixel from junction
    full_distances = _bfs_distances(skel, junction)

    results = []
    for lid, pixels in comp_pixels.items():
        if not pixels:
            continue

        farthest = None
        max_dist = -1.0
        for px in pixels:
            d = full_distances.get(px, 0.0)
            if d > max_dist:
                max_dist = d
                farthest = px

        if farthest is None or max_dist <= 0:
            continue

        # ── Step 5: reconstruct path endpoint → junction on original skeleton
        length, path = _bfs_path(skel, farthest, junction)
        if not path:
            continue

        results.append((length, farthest, path))

    results.sort(key=lambda x: x[0], reverse=True)

    n_removed = int(np.sum((skel > 0).astype(int) - (skel_cut > 0).astype(int)))
    print(f"    Branch detection: removed {n_removed} junction px → "
          f"{len(comp_pixels)} branches (after merging fragments)")
    for i, (d, ep, _) in enumerate(results):
        print(f"      Branch {i+1}: {ep}  {d:.1f} px  ({d * PIXEL_TO_MM:.2f} mm)")

    return results


# ──────────────────────────────────────────────────────────────
#  MAIN FIGURE
# ──────────────────────────────────────────────────────────────
def build_figure(gray, mask, skel, branches, endpoints, junctions):
    """
    Publication-ready 6-panel figure (A–F).

    COORDINATE SYSTEM GUARANTEE
    ───────────────────────────
    Step 1: crop gray / mask / skel to a tight square bbox → skel_c
    Step 2: re-run _find_topology and compute_branches ON skel_c
            so every coordinate lives in [0, H) × [0, W) of the cropped array.
    Step 3: imshow displays skel_c with default extent [-0.5, W-0.5] × [-0.5, H-0.5]
            and axes limits locked to [0, W] × [H, 0].
    Step 4: scatter / plot use  x = col,  y = row  in the cropped space.
    No shifts, no external-space coordinates, no implicit rescaling.
    """
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":        9,
        "axes.linewidth":   0.6,
        "mathtext.default": "regular",
    })

    # ── 1. CROP — single source of truth ─────────────────────
    r0, r1, c0, c1 = _tight_bbox(mask, pad_frac=0.18)

    gray_c = gray[r0:r1, c0:c1].copy()
    mask_c = ((mask > 0).astype(np.uint8) * 255)[r0:r1, c0:c1].copy()
    skel_c = skel[r0:r1, c0:c1].copy()          # uint8, 0 or 1

    H, W = gray_c.shape

    # ── 2. TOPOLOGY — computed on skel_c (cropped space only) ─
    # Find junction (unchanged logic: degree>=3, max geodesic sum)
    _, jn_pts = _find_topology(skel_c)

    # ── 2b. PER-BRANCH ENDPOINT DETECTION ─────────────────────
    # Walk each branch from junction independently.
    # Returns (length, endpoint, path) per branch, sorted longest first.
    if jn_pts:
        all_branch_results = _find_branch_endpoints(skel_c, jn_pts[0])
        # Keep only the 3 longest branches (the true T-arms)
        branch_results = all_branch_results[:3]
        ep_pts = [ep for (_, ep, _) in branch_results]
        branches_c = []
        for i, (length, ep, path) in enumerate(branch_results):
            branches_c.append({
                "length_px": length,
                "path":      path,
                "endpoint":  ep,
                "junction":  jn_pts[0],
                "colour":    BRANCH_COLOURS[i % len(BRANCH_COLOURS)],
            })
    else:
        # Fallback: no junction, use old degree-based + pairwise BFS
        ep_deg1, _ = _find_topology(skel_c)
        ep_pts = ep_deg1
        branches_c = compute_branches(skel_c, ep_pts, jn_pts)

    _validate_topology(skel_c, ep_pts, jn_pts)
    print(f"  ✓  {len(ep_pts)} branch endpoints, {len(jn_pts)} junction")

    for i, b in enumerate(branches_c):
        print(f"    branch {i+1}: {b['length_px']:.1f} px  "
              f"({b['length_px'] * PIXEL_TO_MM:.2f} mm)")

    # ── 4. PRECOMPUTE skeleton pixel positions ────────────────
    sk_rows, sk_cols = np.where(skel_c > 0)

    # ── 5. CANVAS HELPER ─────────────────────────────────────
    def _mask_canvas():
        """Float RGB canvas from mask_c (black bg, white larva)."""
        m = mask_c.astype(np.float32) / 255.0
        return np.stack([m, m, m], axis=-1)

    # ── 6. AXES LOCK HELPER ───────────────────────────────────
    def _lock_axes(ax):
        """
        Lock axes limits to exact pixel extent of cropped image.
        imshow default: pixel centre (r,c) → display (x=c, y=r),
        with data range x ∈ [-0.5, W-0.5], y ∈ [-0.5, H-0.5].
        We lock to that range and disable any auto-scaling.
        """
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)   # y-axis flipped (row 0 at top)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _paint_skel(canvas, colour_hex):
        rgb = _hex_to_rgb01(colour_hex)
        canvas[sk_rows, sk_cols] = rgb

    # ── 7. FIGURE ─────────────────────────────────────────────
    panel_size = 2.1
    fig, axes = plt.subplots(
        1, 6,
        figsize=(panel_size * 6 + 0.4, panel_size + 0.65),
        dpi=300,
        facecolor="white",
        gridspec_kw={"wspace": 0.06},
    )

    # ════════════════════════════════════════════════════════
    #  A — Grayscale crop
    # ════════════════════════════════════════════════════════
    ax = axes[0]
    ax.imshow(gray_c, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    _lock_axes(ax)
    _panel_letter(ax, "A")
    _subtitle(ax, "Grayscale image")

    # ════════════════════════════════════════════════════════
    #  B — Binary mask
    # ════════════════════════════════════════════════════════
    ax = axes[1]
    ax.imshow(mask_c, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    _lock_axes(ax)
    _panel_letter(ax, "B")
    _subtitle(ax, "Binary mask")

    # ════════════════════════════════════════════════════════
    #  C — Skeleton on mask
    # ════════════════════════════════════════════════════════
    ax = axes[2]
    canvas = _mask_canvas()
    _paint_skel(canvas, SKEL_GREY)
    ax.imshow(canvas, interpolation="nearest")
    _lock_axes(ax)
    _panel_letter(ax, "C")
    _subtitle(ax, "Medial-axis skeleton")

    # ════════════════════════════════════════════════════════
    #  D — Topology (endpoints and junction)
    # ════════════════════════════════════════════════════════
    ax = axes[3]
    canvas = _mask_canvas()
    _paint_skel(canvas, "#888888")
    ax.imshow(canvas, interpolation="nearest")

    # x = col, y = row — consistent with imshow coordinate system
    if ep_pts:
        ax.scatter([p[1] for p in ep_pts], [p[0] for p in ep_pts],
                   s=28, c=ENDPOINT_COL, zorder=5,
                   linewidths=0.5, edgecolors="white")
    if jn_pts:
        ax.scatter([p[1] for p in jn_pts], [p[0] for p in jn_pts],
                   s=44, c=JUNCTION_COL, zorder=6,
                   linewidths=0.5, edgecolors="white")

    # Panel D Legend — compact, formal terminology
    ax.legend(
        handles=[
            mpatches.Patch(color=ENDPOINT_COL, label="Endpoints"),
            mpatches.Patch(color=JUNCTION_COL, label="Junction"),
        ],
        loc="lower left", bbox_to_anchor=(0.02, 0.02),
        fontsize=6.5, framealpha=0.6, edgecolor="none",
        handlelength=0.6, borderpad=0.3, labelspacing=0.2, handletextpad=0.35,
    )
    _lock_axes(ax)
    _panel_letter(ax, "D")
    _subtitle(ax, "Topology detection")

    # ════════════════════════════════════════════════════════
    #  E — Branch lengths, labels near endpoints
    # ════════════════════════════════════════════════════════
    ax = axes[4]
    canvas = _mask_canvas()
    _paint_skel(canvas, "#aaaaaa")
    ax.imshow(canvas, interpolation="nearest")

    cy, cx = H / 2.0, W / 2.0
    # choose radial origin: junction if available else centre
    origin = jn_pts[0] if jn_pts else (cy, cx)
    for b in branches_c:
        col = b["colour"]
        # path: list of (row, col) in skel_c space
        if len(b["path"]) >= 2:
            ax.plot([p[1] for p in b["path"]],
                    [p[0] for p in b["path"]],
                    color=col, linewidth=2.0, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round")
        ep = b["endpoint"]                    # (row, col) in skel_c
        ax.scatter(ep[1], ep[0], s=18, c=ENDPOINT_COL, zorder=6,
                   linewidths=0.4, edgecolors="white")

        # Offset label radially away from origin→endpoint direction
        dy = ep[0] - origin[0];  dx = ep[1] - origin[1]
        norm = max(np.hypot(dy, dx), 1.0)
        off  = max(10, int(min(H, W) * 0.18))  # >=10px, about 18% of size
        tx   = float(np.clip(ep[1] + dx / norm * off, 1.5, W - 1.5))
        ty   = float(np.clip(ep[0] + dy / norm * off, 1.5, H - 1.5))
        ax.text(tx, ty, f"{b['length_px']:.1f} px",
                fontsize=7, color=col,
                ha="center", va="center", zorder=8,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=0.5, boxstyle="round,pad=0.2"))

    if jn_pts:
        ax.scatter([p[1] for p in jn_pts], [p[0] for p in jn_pts],
                   s=38, c=JUNCTION_COL, zorder=7,
                   linewidths=0.4, edgecolors="white")

    ax.text(0.02, 0.02, r"diagonal step $= \sqrt{2}$  px",
            transform=ax.transAxes, fontsize=5.5, color="#666666",
            va="bottom", ha="left", style="italic")
    _lock_axes(ax)
    _panel_letter(ax, "E")
    _subtitle(ax, "Branch length along skeleton")

    # ════════════════════════════════════════════════════════
    #  F — Principal branch + formula in footer (true split)
    # ════════════════════════════════════════════════════════
    ax = axes[5]
    # Split the panel into image (top 75%) and footer (bottom 25%)
    fig = ax.figure
    pos = ax.get_position()
    ax.set_visible(False)
    img_ax = fig.add_axes([pos.x0, pos.y0 + pos.height * 0.25,
                           pos.width, pos.height * 0.75])
    footer_ax = fig.add_axes([pos.x0, pos.y0,
                              pos.width, pos.height * 0.25])

    canvas = _mask_canvas()
    _paint_skel(canvas, "#bbbbbb")
    img_ax.imshow(canvas, interpolation="nearest")

    principal = branches_c[0]
    if len(principal["path"]) >= 2:
        img_ax.plot([p[1] for p in principal["path"]],
                    [p[0] for p in principal["path"]],
                    color=PRINCIPAL_COL, linewidth=2.8, zorder=5,
                    solid_capstyle="round", solid_joinstyle="round")

    ep = principal["endpoint"]
    img_ax.scatter(ep[1], ep[0], s=26, c=ENDPOINT_COL, zorder=6,
                   linewidths=0.4, edgecolors="white")
    if jn_pts:
        img_ax.scatter([p[1] for p in jn_pts], [p[0] for p in jn_pts],
                       s=40, c=JUNCTION_COL, zorder=7,
                       linewidths=0.4, edgecolors="white")

    _lock_axes(img_ax)
    _panel_letter(img_ax, "F")
    # No subtitle for panel F

    # Footer layout (no overlap with image)
    footer_ax.set_facecolor("white")
    footer_ax.set_xlim(0, 1)
    footer_ax.set_ylim(0, 1)
    footer_ax.axis("off")
    footer_ax.axhline(1.0, color="#cccccc", linewidth=0.8)

    l_px = principal["length_px"]
    l_mm = l_px * PIXEL_TO_MM

    footer_lines = [
        (f"L_px = ∑ dᵢ = {l_px:.1f} px", 8.0, "#333333", "normal"),
        (f"× {PIXEL_TO_MM:.4f} mm/px",      7.0, "#555555", "normal"),
        (f"L = {l_mm:.2f} mm",              9.5, PRINCIPAL_COL, "bold"),
    ]
    y_positions = [0.68, 0.44, 0.18]
    for (txt, fs, col, fw), y in zip(footer_lines, y_positions):
        footer_ax.text(0.5, y, txt,
                       ha="center", va="center",
                       fontsize=fs, color=col, fontweight=fw,
                       math_fontfamily="dejavusans")

    return fig
def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphologically clean the binary mask:
    - Fill small holes
    - Remove tiny isolated blobs
    - Smooth contour slightly
    Returns a clean binary mask (0/255).
    """
    m = (mask > 0).astype(np.uint8) * 255
    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    clean = np.zeros_like(m)
    clean[labels == largest] = 255
    # Slight closing to fill skeleton gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=2)
    return clean


def _extract_gray_mask(report_path: Path):
    """
    From a larva report image, extract:
      - gray_crop  : the first panel (grayscale)
      - mask_crop  : the second panel (binary mask), binarised and cleaned
    Report layout: [gray | mask | overlay] + [300 px black text panel]
    """
    img = cv2.imread(str(report_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    visual_w = w - 300
    panel_w  = visual_w // 3
    if panel_w <= 0:
        return None, None

    gray_bgr = img[:, :panel_w]
    mask_bgr = img[:, panel_w : 2 * panel_w]

    gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
    mask_raw = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_raw, 50, 255, cv2.THRESH_BINARY)
    mask = _clean_mask(mask_bin)
    return gray, mask


def find_posture_suitable_larva():
    """
    Try the preferred specimen first.
    Then scan all posture-suitable larvae sorted by topology quality
    (fewest junctions, closest to 3 endpoints).
    """
    if not LABELS_FILE.exists():
        print(f"❌  Labels file not found: {LABELS_FILE}")
        sys.exit(1)

    # ── Try preferred specimen first ──────────────────────────
    pref_path = (ANALYSIS_DIR / PREFERRED_LARVA[0]
                 / PREFERRED_LARVA[1] / "larvae_reports" / PREFERRED_LARVA[2])
    if pref_path.exists():
        gray, mask = _extract_gray_mask(pref_path)
        if gray is not None and mask is not None and mask.sum() > 0:
            skel = _skeletonize(mask)
            ep, jn = _find_topology(skel)
            print(f"  ✓  Using preferred larva: {pref_path.relative_to(PROJECT_DIR)}")
            print(f"     ep={len(ep)}  jn={len(jn)}")
            return gray, mask, skel, pref_path

    # ── Fallback: scan all labeled ────────────────────────────
    df = pd.read_excel(LABELS_FILE)
    df["date"] = df["date"].astype(str)
    good = df[
        (df["is_valid_larva"] == 1) &
        (df["shape_score"] >= 1) &
        (~df["date"].isin(EXCLUDE_DATE))
    ].copy()

    rng = random.Random(RANDOM_SEED)
    indices = list(good.index)
    rng.shuffle(indices)

    candidates = []
    for idx in indices:
        row = good.loc[idx]
        path = (ANALYSIS_DIR / row["date"] / row["image_name"]
                / "larvae_reports" / row["larva_filename"])
        if not path.exists():
            continue
        gray, mask = _extract_gray_mask(path)
        if gray is None or mask is None or mask.sum() == 0:
            continue
        skel = _skeletonize(mask)
        ep, jn = _find_topology(skel)
        score = abs(len(jn) - 1) * 10 + abs(len(ep) - 3)
        candidates.append((score, path, gray, mask, skel, ep, jn))

    if not candidates:
        print("❌  No suitable larva found.")
        sys.exit(1)

    candidates.sort(key=lambda x: x[0])
    score, path, gray, mask, skel, ep, jn = candidates[0]
    print(f"  ✓  Selected: {path.relative_to(PROJECT_DIR)}")
    print(f"     ep={len(ep)}  jn={len(jn)}")
    return gray, mask, skel, path


# ──────────────────────────────────────────────────────────────
#  STEP 2  –  SKELETONISATION
# ──────────────────────────────────────────────────────────────
def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Produce a clean 1-pixel-wide skeleton using scikit-image."""
    try:
        from skimage.morphology import skeletonize as _sk
        bw = (binary > 0).astype(bool)
        skel = _sk(bw).astype(np.uint8)
        return skel
    except ImportError:
        pass
    # Fallback: OpenCV morphological thinning
    img    = (binary > 0).astype(np.uint8)
    skel   = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(200):
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        skel   = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
        img    = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# ──────────────────────────────────────────────────────────────
#  STEP 3  –  TOPOLOGY
# ──────────────────────────────────────────────────────────────
def _bfs_distances(skel: np.ndarray, source: tuple) -> dict:
    """
    Single-source BFS from `source` over skeleton pixels.
    Returns dict: (r, c) → geodesic distance from source.
    Diagonal step cost = √2, axis-aligned = 1.0.
    Only visits pixels where skel > 0.
    """
    dist  = {source: 0.0}
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        d    = dist[(r, c)]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                nb_rc  = (nr, nc)
                if nb_rc in dist:
                    continue
                if (0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]
                        and skel[nr, nc] > 0):
                    step = 1.4142 if (dr != 0 and dc != 0) else 1.0
                    dist[nb_rc] = d + step
                    queue.append(nb_rc)
    return dist


def _find_topology(skel: np.ndarray):
    """
    Deterministic topology detection matching the classification pipeline exactly.

    Rules (strict):
      Endpoint  = skeleton pixel with 8-connected degree == 1
      Junction  = skeleton pixel with 8-connected degree >= 3

    Junction selection (when multiple candidates exist):
      For each junction candidate J, compute:
          score(J) = sum of BFS geodesic distances from J to every endpoint
      Select the J with the HIGHEST score.
      → This is the pixel that lies deepest inside the structure,
        maximally separating all three branches.

    No clustering. No centroid averaging. No radius merging.
    Every returned coordinate is guaranteed to satisfy skel[r, c] > 0.
    """
    nb        = _neighbour_count(skel)
    ep_arr    = np.argwhere((skel > 0) & (nb == 1))
    jn_arr    = np.argwhere((skel > 0) & (nb >= 3))

    endpoints = [tuple(p) for p in ep_arr]
    jn_candidates = [tuple(p) for p in jn_arr]

    if not jn_candidates:
        return endpoints, []

    if len(jn_candidates) == 1:
        return endpoints, [jn_candidates[0]]

    # Multiple junction candidates → pick the one that maximises
    # the sum of geodesic distances to all endpoints.
    # Run BFS from each junction candidate once.
    best_jn    = jn_candidates[0]
    best_score = -1.0

    for jn in jn_candidates:
        dists = _bfs_distances(skel, jn)
        score = sum(dists.get(ep, 0.0) for ep in endpoints)
        if score > best_score:
            best_score = score
            best_jn    = jn

    return endpoints, [best_jn]


def _validate_topology(skel: np.ndarray, endpoints: list, junctions: list) -> None:
    """
    Verify topology points are on skeleton.

    Note: Endpoints are now defined as max-distance pixels, NOT degree==1.
    They must be on skeleton but degree is not checked.
    """
    ok = True

    for pt in endpoints:
        r, c = pt
        in_bounds = 0 <= r < skel.shape[0] and 0 <= c < skel.shape[1]
        on_skel   = in_bounds and skel[r, c] > 0
        if not on_skel:
            print(f"  ⚠  ENDPOINT INVALID: ({r},{c})  on_skel={on_skel}")
            ok = False

    for pt in junctions:
        r, c = pt
        in_bounds = 0 <= r < skel.shape[0] and 0 <= c < skel.shape[1]
        on_skel   = in_bounds and skel[r, c] > 0
        nb = _neighbour_count(skel)
        degree    = int(nb[r, c]) if in_bounds else -1
        if not on_skel or degree < 3:
            print(f"  ⚠  JUNCTION INVALID:  ({r},{c})  "
                  f"on_skel={on_skel}  degree={degree}  (expected >= 3)")
            ok = False

    if ok:
        print(f"  ✓  Topology valid: {len(endpoints)} geometric endpoints, "
              f"{len(junctions)} junction — all on skeleton.")


# ──────────────────────────────────────────────────────────────
#  STEP 4  –  BRANCH TRACING (BFS, geodesic distance)
# ──────────────────────────────────────────────────────────────
def _bfs_path(skel: np.ndarray, start: tuple, goal: tuple):
    """
    BFS on skeleton from start to goal.
    Returns (geodesic_length, list_of_(r,c) pixels in path).
    Diagonal steps cost √2, axis-aligned steps cost 1.
    """
    visited = {start: (None, 0.0)}   # node → (parent, cumulative_dist)
    queue   = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            # reconstruct path
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = visited[node][0]
            return visited[goal][1], path[::-1]
        r, c = cur
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                nb_rc = (nr, nc)
                if nb_rc in visited:
                    continue
                if (0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]
                        and skel[nr, nc] > 0):
                    step = 1.4142 if (dr != 0 and dc != 0) else 1.0
                    dist = visited[cur][1] + step
                    visited[nb_rc] = (cur, dist)
                    queue.append(nb_rc)
    return 0.0, []   # unreachable


def compute_branches(skel: np.ndarray, endpoints: list, junctions: list):
    """
    Compute geodesic branches on the skeleton.

    When a junction exists (selected by _find_topology):
        For each endpoint, run BFS along skeleton pixels to the junction.
        Branch length = geodesic distance (diagonal = √2).

    When no junction exists:
        Trace BFS between all endpoint pairs; keep unique non-overlapping paths.

    Returns list of dicts sorted by length descending, capped at 5.
    Every path coordinate satisfies skel[r, c] > 0.
    """
    branches = []

    if junctions:
        jn = junctions[0]   # single selected junction pixel (on skeleton)
        for ep in endpoints:
            length, path = _bfs_path(skel, ep, jn)
            if length > 0 and path:
                branches.append({
                    "length_px": length,
                    "path":      path,
                    "endpoint":  ep,
                    "junction":  jn,
                    "colour":    "",   # assigned below
                })
    else:
        # No junction: trace all endpoint pairs
        seen = set()
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                length, path = _bfs_path(skel, endpoints[i], endpoints[j])
                if length > 0 and path:
                    branches.append({
                        "length_px": length,
                        "path":      path,
                        "endpoint":  endpoints[i],
                        "junction":  endpoints[j],
                        "colour":    "",
                    })

    # Sort longest first; keep top 5; assign colours consistently
    branches.sort(key=lambda b: b["length_px"], reverse=True)
    branches = branches[:5]
    for i, b in enumerate(branches):
        b["colour"] = BRANCH_COLOURS[i % len(BRANCH_COLOURS)]

    return branches


# ──────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SKELETON LENGTH FIGURE")
    print("=" * 60)

    print("\n  Finding posture-suitable larva …")
    gray, mask, skel, src_path = find_posture_suitable_larva()

    # All topology detection and branch tracing is performed inside
    # build_figure() on the cropped skeleton (single coordinate system).
    print("  Building figure …")
    fig = build_figure(gray, mask, skel, [], [], [])

    fig.savefig(
        OUTPUT_FIG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.05,
    )
    plt.close(fig)

    print(f"\n  ✓  Saved → {OUTPUT_FIG}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
