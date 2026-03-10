#!/usr/bin/env python3
"""
geodesic_distance_schematic.py
================================
Generates a schematic diagram explaining the computation of geodesic
distance on a pixel grid for skeleton-based length estimation.

This is a didactic visualization for educational purposes, showing how
axis-aligned and diagonal steps are weighted differently.

Output:
    figures/geodesic_distance_schematic.png
"""

from pathlib import Path
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")  # Set non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import cv2

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = FIGURES_DIR.parent
ANALYSIS_DIR = PROJECT_DIR / "analysis_full"
LABELS_FILE = ANALYSIS_DIR / "larva_quality_labels.xlsx"
OUTPUT_FIG = FIGURES_DIR / "geodesic_distance_schematic.png"

# --- Real Larva Example Config ---
PIXEL_TO_MM = 0.232255814
RANDOM_SEED = 42
EXCLUDE_DATE = {"18.10", "18.1"}
PREFERRED_LARVA = ("3.11", "IMG_8053", "larva_039.png")

# Colors
GRID_COLOR = "#d0d0d0"
PATH_COLOR = "#34495e"  # Dark blue for the main path nodes
AXIS_STEP_COLOR = "#3498db"  # Bright blue for axis steps
DIAG_STEP_COLOR = "#e67e22"  # Bright orange for diagonal steps
TEXT_COLOR = "#2c3e50"

# --- Real Larva Example Colors ---
SKEL_GREY = "#666666"
PRINCIPAL_BRANCH_COLOR = "#e74c3c"  # Bright red for principal branch (same in both panels)
ENDPOINT_COLOR = "#3498db"
JUNCTION_COLOR = "#f1c40f"


# ──────────────────────────────────────────────────────────────
#  Larva Loading and Skeletonization Helpers
# (Adapted from skeleton_length_figure.py)
# ──────────────────────────────────────────────────────────────
def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Morphologically clean the binary mask."""
    m = (mask > 0).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1: return m
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    clean = np.zeros_like(m)
    clean[labels == largest] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
    return clean

def _extract_gray_mask(report_path: Path):
    """Extracts grayscale and mask panels from a report image."""
    img = cv2.imread(str(report_path))
    if img is None: return None, None
    h, w = img.shape[:2]
    panel_w = (w - 300) // 3
    if panel_w <= 0: return None, None
    gray = cv2.cvtColor(img[:, :panel_w], cv2.COLOR_BGR2GRAY)
    mask_raw = cv2.cvtColor(img[:, panel_w:2*panel_w], cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_raw, 50, 255, cv2.THRESH_BINARY)
    return gray, _clean_mask(mask_bin)

def _skeletonize(binary_img: np.ndarray) -> np.ndarray:
    """Computes a 1-pixel-wide skeleton."""
    try:
        from skimage.morphology import skeletonize as sk
        return (sk(binary_img > 0)).astype(np.uint8)
    except ImportError:
        img = (binary_img > 0).astype(np.uint8)
        skel = np.zeros_like(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(img) > 0:
            eroded = cv2.erode(img, kernel)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            skel = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
            img = eroded.copy()
        return skel

def _tight_bbox(mask: np.ndarray, pad_frac: float = 0.1):
    """Computes a tight, square bounding box around the mask content."""
    rows, cols = np.where(mask > 0)
    if not len(rows): return 0, mask.shape[0], 0, mask.shape[1]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    h, w = r1 - r0, c1 - c0
    pad = int(max(h, w) * pad_frac)
    r0, r1 = max(0, r0 - pad), min(mask.shape[0], r1 + pad)
    c0, c1 = max(0, c0 - pad), min(mask.shape[1], c1 + pad)
    return r0, r1, c0, c1

def find_real_larva_example():
    """Finds a suitable real larva example for visualization."""
    try:
        pref_path = ANALYSIS_DIR / PREFERRED_LARVA[0] / PREFERRED_LARVA[1] / "larvae_reports" / PREFERRED_LARVA[2]

        if pref_path.exists():
            gray, mask = _extract_gray_mask(pref_path)
            if gray is not None and mask is not None and mask.sum() > 0:
                return {"gray": gray, "mask": mask, "skel": _skeletonize(mask)}

        # If preferred doesn't work, return None to use fallback
        return None

    except Exception as e:
        print(f"Error loading real larva: {e}", file=sys.stderr)
        return None

def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Counts 8-connected neighbours for each skeleton pixel."""
    ker = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, ker) * skel.astype(np.uint8)

def _find_topology(skel: np.ndarray):
    """Finds endpoints (degree 1) and junctions (degree >= 3)."""
    nb = _neighbour_count(skel)
    endpoints = [tuple(p) for p in np.argwhere((skel > 0) & (nb == 1))]
    junctions = [tuple(p) for p in np.argwhere((skel > 0) & (nb >= 3))]
    return endpoints, junctions

def _geodesic_length(path: list) -> float:
    """Calculates the geodesic length of a path with weighted steps."""
    length = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        is_diag = abs(p1[0] - p2[0]) == 1 and abs(p1[1] - p2[1]) == 1
        length += np.sqrt(2) if is_diag else 1.0
    return length

def _bfs_path(skel, start, goal):
    """Finds the shortest path on the skeleton using BFS and returns path with its geodesic length."""
    q = deque([(start, [start])])
    visited = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == goal:
            length = _geodesic_length(path)
            return path, length
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc] > 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
    return [], 0.0 # Path not found

def compute_all_branches(skel: np.ndarray, endpoints: list, junctions: list) -> list:
    """Computes all branches from endpoints to junction(s) with different colors."""
    branches = []

    if junctions:
        # Multiple colors for different branches
        branch_colors = [PRINCIPAL_BRANCH_COLOR, "#2471a3", "#1e8449", "#7d3c98", "#b7770d"]

        junction = junctions[0]  # Use first junction
        for i, endpoint in enumerate(endpoints):
            path, length = _bfs_path(skel, endpoint, junction)
            if path and length > 0:
                color = branch_colors[i % len(branch_colors)]
                branches.append({
                    "path": path,
                    "length": length,
                    "endpoint": endpoint,
                    "junction": junction,
                    "color": color
                })

    # Sort branches by length (longest first)
    branches.sort(key=lambda b: b["length"], reverse=True)

    # Ensure the longest branch uses the principal color (matches Panel A)
    if branches:
        branches[0]["color"] = PRINCIPAL_BRANCH_COLOR

    return branches

def find_longest_geodesic_path(skel: np.ndarray, endpoints: list) -> list:
    """Finds the longest geodesic path between any two endpoints on the skeleton."""
    if len(endpoints) < 2:
        return []

    longest_path = []
    max_len = -1.0

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            path, length = _bfs_path(skel, endpoints[i], endpoints[j])
            if length > max_len:
                max_len = length
                longest_path = path

    return longest_path

# ──────────────────────────────────────────────────────────────
#  Main Visualization Function
# ──────────────────────────────────────────────────────────────
def create_simple_fallback_example():
    """Creates a simple artificial skeleton for fallback when real larva can't be loaded."""
    # Create a simple T-shaped skeleton
    skel = np.zeros((50, 50), dtype=np.uint8)

    # Vertical line (main body)
    for r in range(15, 40):
        skel[r, 25] = 1

    # Horizontal line (arms)
    for c in range(15, 35):
        skel[25, c] = 1

    # Create corresponding mask
    mask = np.zeros_like(skel, dtype=np.uint8)
    mask[skel > 0] = 255

    # Add some thickness to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    gray = mask.copy()

    return {"gray": gray, "mask": mask, "skel": skel}

def create_schematic_figure():
    """Creates and saves the geodesic distance schematic figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.linewidth": 1.0,
    })

    # --- Find real larva example ---
    larva_example = find_real_larva_example()
    if not larva_example:
        larva_example = create_simple_fallback_example()

    gray, mask, skel = larva_example["gray"], larva_example["mask"], larva_example["skel"]

    # --- Process Real Larva Data ---
    r0, r1, c0, c1 = _tight_bbox(mask)
    gray_c, mask_c, skel_c = gray[r0:r1, c0:c1], mask[r0:r1, c0:c1], skel[r0:r1, c0:c1]

    endpoints, junctions = _find_topology(skel_c)

    # Compute all branches for visualization
    all_branches = compute_all_branches(skel_c, endpoints, junctions)

    # Extract the principal branch path for Panel A schematic
    principal_path = all_branches[0]["path"] if all_branches else []

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150, gridspec_kw={'width_ratios': [1, 1]})
    fig.set_facecolor("white")

    ax1 = axes[0]
    ax2 = axes[1]

    # --- Panel A: Geodesic Distance Schematic (exact principal branch path) ---
    ax1.set_title("A: Geodesic Distance Schematic", fontsize=14, pad=20)
    ax1.grid(which='both', color=GRID_COLOR, linestyle='-', linewidth=1)
    ax1.set_aspect('equal', adjustable='box')

    # Hide axis labels and ticks initially (will be updated based on path)
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    # Hide spines
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # Display exact principal branch path on pixel grid
    total_length = 0
    formula_parts = []

    if principal_path and len(principal_path) > 1:
        # Use exact principal branch path coordinates
        path_arr = np.array(principal_path)

        # Calculate bounding box for the path to determine grid size
        min_r, max_r = path_arr[:, 0].min(), path_arr[:, 0].max()
        min_c, max_c = path_arr[:, 1].min(), path_arr[:, 1].max()

        # Add padding around the path
        padding = 2
        grid_r0 = max(0, min_r - padding)
        grid_r1 = max_r + padding + 1
        grid_c0 = max(0, min_c - padding)
        grid_c1 = max_c + padding + 1

        # Adjust axis limits to show the exact path region
        ax1.set_xlim(grid_c0 - 0.5, grid_c1 - 0.5)
        ax1.set_ylim(grid_r1 - 0.5, grid_r0 - 0.5)  # Inverted y-axis for image coordinates

        # Update grid to match the path region
        ax1.set_xticks(np.arange(grid_c0, grid_c1, 1))
        ax1.set_yticks(np.arange(grid_r0, grid_r1, 1))

        # Draw individual steps with appropriate colors
        for i in range(len(principal_path) - 1):
            p1 = principal_path[i]
            p2 = principal_path[i + 1]

            # Calculate step differences
            dr = abs(p2[0] - p1[0])
            dc = abs(p2[1] - p1[1])

            # Determine step type and cost
            if dr == 1 and dc == 1:
                # Diagonal step
                cost = np.sqrt(2)
                color = DIAG_STEP_COLOR
                label = r"$\sqrt{2}$"
            elif (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
                # Axis-aligned step
                cost = 1.0
                color = AXIS_STEP_COLOR
                label = "1"
            else:
                # Other steps (shouldn't happen in normal skeleton)
                cost = np.sqrt(dr*dr + dc*dc)
                color = DIAG_STEP_COLOR if cost > 1.1 else AXIS_STEP_COLOR
                label = f"{cost:.1f}"

            total_length += cost
            if abs(cost - np.sqrt(2)) < 0.01:
                formula_parts.append(r"\sqrt{2}")
            elif cost == 1.0:
                formula_parts.append("1")
            else:
                formula_parts.append(f"{cost:.1f}")

            # Draw step line
            ax1.plot([p1[1], p2[1]], [p1[0], p2[0]],
                    color=color, linewidth=3, solid_capstyle='round', alpha=0.8)

            # Add cost label at step midpoint
            mid_r = (p1[0] + p2[0]) / 2
            mid_c = (p1[1] + p2[1]) / 2
            ax1.text(mid_c, mid_r, label,
                    ha='center', va='center', fontsize=10, color=TEXT_COLOR, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

        # Draw the principal branch path
        ax1.plot(path_arr[:, 1], path_arr[:, 0],
                color=PRINCIPAL_BRANCH_COLOR, linewidth=2, alpha=0.6, zorder=1)

        # Draw nodes at exact pixel centers
        ax1.scatter(path_arr[:, 1], path_arr[:, 0],
                   color=PRINCIPAL_BRANCH_COLOR, s=60, zorder=5,
                   edgecolors='white', linewidth=1.5)

    # --- Panel B: Real Larva Example with colored branches ---
    ax2.set_title("B: Skeleton-based Length on Larva", fontsize=14, pad=20)
    ax2.imshow(mask_c, cmap='gray', vmin=0, vmax=255)

    # Draw skeleton in light grey first
    skel_coords = np.argwhere(skel_c > 0)
    ax2.scatter(skel_coords[:, 1], skel_coords[:, 0], color="#e0e0e0", s=1, alpha=0.6)

    # Draw all branches in different colors
    legend_elements = []
    if all_branches:
        for i, branch in enumerate(all_branches):
            if branch["path"]:
                path_arr = np.array(branch["path"])
                linewidth = 3.0 if i == 0 else 2.0  # Thicker line for principal branch
                ax2.plot(path_arr[:, 1], path_arr[:, 0],
                        color=branch["color"], linewidth=linewidth,
                        solid_capstyle='round', alpha=0.9)

                # Create legend entry
                label = f"Principal branch ({branch['length']:.1f} px)" if i == 0 else f"Branch {i+1} ({branch['length']:.1f} px)"
                legend_elements.append(plt.Line2D([0], [0], color=branch["color"], lw=2, label=label))

    # Draw endpoints and junctions
    if endpoints:
        ep_arr = np.array(endpoints)
        ax2.scatter(ep_arr[:, 1], ep_arr[:, 0], s=50, color=ENDPOINT_COLOR,
                   zorder=5, edgecolors='white', linewidth=1)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=ENDPOINT_COLOR, markersize=6, label="Endpoints", linestyle='None'))

    if junctions:
        jn_arr = np.array(junctions)
        ax2.scatter(jn_arr[:, 1], jn_arr[:, 0], s=80, color=JUNCTION_COLOR,
                   zorder=5, edgecolors='white', linewidth=1)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=JUNCTION_COLOR, markersize=8, label="Junction", linestyle='None'))

    # Add legend
    if legend_elements:
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.8)
    ax2.set_aspect('equal', adjustable='box')
    ax2.axis('off')

    # --- Formula and Explanation ---
    # Use schematic path for formula display (Panel A consistency)
    formula_str = ""
    if principal_path and len(principal_path) > 1:
        real_path_len = _geodesic_length(principal_path)

        if total_length > 0 and formula_parts:
            formula_str = r"$L_{\mathrm{px}} = \sum d_i = " + " + ".join(formula_parts) + f" = {total_length:.2f} \\,\\mathrm{{px}}$"
        else:
            formula_str = r"$L_{\mathrm{px}} = \sum d_i \approx " + f"{real_path_len:.2f} \\,\\mathrm{{px}}$"

    explanation_text = (
        r"$\mathbf{Geodesic\;Distance\;Computation}$" + "\n"
        "Skeleton pixels form a graph connected to their 8-neighborhood.\n"
        "The path is found using Breadth-First Search (BFS) with weighted steps:\n"
        r"$d_i = 1$ (axis-aligned), $d_i = \sqrt{2}$ (diagonal)"
    )

    fig.text(0.5, 0.12, explanation_text, ha='center', va='center', fontsize=11, color=TEXT_COLOR, linespacing=1.5)
    fig.text(0.5, 0.05, formula_str, ha='center', va='center', fontsize=14, color=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0.2, 1, 0.95])

    # 5. Save the figure
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"✓ Saved schematic figure to: {OUTPUT_FIG}")


# ──────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Script starting...", flush=True)
    create_schematic_figure()
    print("Script completed.", flush=True)
