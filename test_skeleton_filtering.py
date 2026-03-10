#!/usr/bin/env python3
"""
Real Dataset Skeleton Filtering Analysis
=========================================
Analyzes skeleton topology and body length computation on real larva masks.
Scans: analysis_full_binary_masks_only/<date>/<image>/larvae_reports/larva_*.png
Outputs debug visualizations to: skeleton_debug_examples/
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, deque

# ============================================================
#  CONFIGURATION
# ============================================================
ROOT_DIR = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full_binary_masks_only"
OUTPUT_DIR = ROOT_DIR / "skeleton_debug_examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Predictions file from the fixed pipeline
PREDICTIONS_FILE = ROOT_DIR / "dual_larva_models_geodesic2" / "predictions" / "posture_predictions.xlsx"

MAX_DEBUG_IMAGES = 50  # Save visualizations for first N filtered larvae
PIXEL_TO_MM = 0.232255814

# Dates to exclude from analysis
EXCLUDED_DATES = {'18.10', '18.1'}

# ============================================================
#  SKELETON FUNCTIONS (from dual_larva_pipeline_geodesic_fixed)
# ============================================================
def _skeletonize(binary_img):
    """Produce a clean 1-pixel-wide skeleton using scikit-image (with fallback)."""
    try:
        from skimage.morphology import skeletonize as _sk
        bw = (binary_img > 0).astype(bool)
        skel = _sk(bw).astype(np.uint8)
        return skel
    except ImportError:
        pass
    # Fallback: OpenCV morphological thinning
    img    = (binary_img > 0).astype(np.uint8)
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


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """For every skeleton pixel, count its 8-connected skeleton neighbours."""
    ker = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, ker) * skel.astype(np.uint8)


def _bfs_distances(skel: np.ndarray, source: tuple) -> dict:
    """
    Single-source BFS from `source` over skeleton pixels.
    Returns dict: (r, c) → geodesic distance from source.
    Diagonal step cost = √2, axis-aligned = 1.0.
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
    Topology detection matching skeleton_length_figure.py exactly.
      Endpoint  = skeleton pixel with 8-connected degree == 1
      Junction  = skeleton pixel with 8-connected degree >= 3
    Junction selection: maximise sum of geodesic distances to endpoints.
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

    best_jn    = jn_candidates[0]
    best_score = -1.0
    for jn in jn_candidates:
        dists = _bfs_distances(skel, jn)
        score = sum(dists.get(ep, 0.0) for ep in endpoints)
        if score > best_score:
            best_score = score
            best_jn    = jn

    return endpoints, [best_jn]


def compute_geodesic_body_length(skel: np.ndarray) -> float:
    """
    Compute body length as the longest geodesic distance on the skeleton.

    STRICT T/Y TOPOLOGY FILTER:
      Body length is computed ONLY for skeletons with:
        - exactly 3 endpoints
        - exactly 1 junction

      All other topologies are rejected (return 0.0).

    For valid T/Y larvae:
      body_length = longest branch (endpoint → junction) via BFS.

    Diagonal steps cost √2, axis-aligned steps cost 1.0.
    Returns length in pixels.
    """
    skel_px = int(np.sum(skel > 0))

    # Filter 1: Minimum skeleton size
    if skel_px < 2:
        return 0.0

    endpoints, junctions = _find_topology(skel)

    # STRICT T/Y TOPOLOGY FILTER
    # Accept ONLY skeletons with exactly 3 endpoints and exactly 1 junction
    if len(endpoints) != 3:
        return 0.0

    if len(junctions) != 1:
        return 0.0

    # Valid T/Y topology: compute branch lengths from junction to endpoints
    jn = junctions[0]
    dists = _bfs_distances(skel, jn)
    branch_lengths = [dists.get(ep, 0.0) for ep in endpoints]

    if not branch_lengths:
        return 0.0

    max_length = max(branch_lengths)

    # Filter: minimum body length
    if max_length < 2.0:
        return 0.0

    return max_length

# ============================================================
#  UTILITIES
# ============================================================
def load_valid_posture_predictions():
    """
    Load predictions and return a set of (date, image_name, larva_filename)
    for larvae with predicted_valid=1 AND predicted_posture=1
    """
    if not PREDICTIONS_FILE.exists():
        print(f"⚠️  Predictions file not found: {PREDICTIONS_FILE}")
        print("    Running without filtering...")
        return None

    df = pd.read_excel(PREDICTIONS_FILE)

    # Filter for valid=1 AND posture=1
    filtered = df[(df['predicted_valid'] == 1) & (df['predicted_posture'] == 1)]

    print(f"✓ Loaded predictions: {len(df)} total, {len(filtered)} with valid=1 AND posture=1")

    # Create set of (date, image_name, larva_filename) tuples
    valid_larvae = set()
    for _, row in filtered.iterrows():
        key = (str(row['date']), str(row['image_name']), str(row['larva_filename']))
        valid_larvae.add(key)

    return valid_larvae


def load_larva_mask(mask_path):
    """Load and convert larva mask to binary."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Convert to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def create_debug_visualization(mask, skel, endpoints, junctions, body_length, title_info):
    """
    Create 4-panel debug figure:
    1. Original binary mask
    2. Skeleton overlay on mask
    3. Topology (endpoints=red, junctions=blue)
    4. Summary text panel
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Original mask
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Binary Mask', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: Skeleton overlay
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    overlay[skel > 0] = [0, 255, 0]  # Green skeleton
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('Skeleton Overlay', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Panel 3: Topology visualization
    topo_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    topo_vis[skel > 0] = [255, 255, 255]  # White skeleton

    # Mark endpoints in red
    for ep in endpoints:
        cv2.circle(topo_vis, (ep[1], ep[0]), 3, (255, 0, 0), -1)

    # Mark junctions in blue
    for jn in junctions:
        cv2.circle(topo_vis, (jn[1], jn[0]), 4, (0, 0, 255), -1)

    axes[1, 0].imshow(topo_vis)
    axes[1, 0].set_title('Topology (EP=red, JN=blue)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Panel 4: Text summary
    axes[1, 1].axis('off')
    summary_text = (
        f"Date: {title_info['date']}\n"
        f"Image: {title_info['image']}\n"
        f"Larva: {title_info['larva']}\n\n"
        f"Skeleton pixels: {title_info['skel_px']}\n"
        f"Endpoints: {title_info['n_endpoints']}\n"
        f"Junctions: {title_info['n_junctions']}\n\n"
        f"Body length: {body_length:.2f} px\n"
        f"            ({body_length * PIXEL_TO_MM:.3f} mm)\n\n"
    )

    # Add warnings
    warnings = []
    if title_info['n_endpoints'] > 3:
        warnings.append("⚠️  >3 endpoints (noisy)")
    if title_info['n_junctions'] > 1:
        warnings.append("⚠️  >1 junction (complex)")
    if title_info['skel_px'] < 20:
        warnings.append("⚠️  <20 skeleton pixels (tiny)")
    if body_length < 20.0 and body_length > 0:
        warnings.append("⚠️  <20px body length (filtered)")
    if body_length == 0.0:
        warnings.append("❌ FILTERED (body_length=0)")

    if warnings:
        summary_text += "WARNINGS:\n" + "\n".join(warnings)
    else:
        summary_text += "✓ Normal topology"

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title_info['full_title'], fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================
#  MAIN ANALYSIS
# ============================================================
def main():
    print("="*70)
    print("REAL DATASET SKELETON ANALYSIS")
    print("Filtering: Valid larvae with correct posture (predictions = 1)")
    print("="*70)

    # Load predictions for filtering
    valid_larvae_set = load_valid_posture_predictions()

    print(f"\nInput directory:  {ANALYSIS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max debug images: {MAX_DEBUG_IMAGES}\n")

    # Statistics collection
    stats = {
        'total_larvae': 0,
        'skipped_by_prediction': 0,
        'filtered_larvae': 0,
        'tiny_skeletons': 0,
        'short_lengths': 0,
        'noisy_topology': 0,
        'complex_junctions': 0,
        'body_lengths': [],
        'skeleton_sizes': [],
        'endpoint_counts': [],
        'junction_counts': []
    }

    debug_image_count = 0

    # Scan dataset
    date_folders = sorted([d for d in ANALYSIS_DIR.iterdir() if d.is_dir()],
                         key=lambda x: x.name)

    for date_dir in date_folders:
        date_name = date_dir.name

        # Skip excluded dates
        if date_name in EXCLUDED_DATES:
            print(f"⏭️  Skipping excluded date: {date_name}")
            continue

        image_folders = sorted([img for img in date_dir.iterdir() if img.is_dir()])

        for image_dir in image_folders:
            image_name = image_dir.name
            larvae_dir = image_dir / "larvae_reports"

            if not larvae_dir.exists():
                continue

            larva_files = sorted(larvae_dir.glob("larva_*.png"))

            for larva_path in larva_files:
                larva_name = larva_path.name
                stats['total_larvae'] += 1

                # Check if this larva passes the prediction filter
                if valid_larvae_set is not None:
                    larva_key = (date_name, image_name, larva_name)
                    if larva_key not in valid_larvae_set:
                        stats['skipped_by_prediction'] += 1
                        continue

                # Load mask
                mask = load_larva_mask(larva_path)
                if mask is None:
                    print(f"⚠️  Could not load: {date_name}/{image_name}/{larva_name}")
                    continue

                # Compute skeleton
                skel = _skeletonize(mask)
                skel_px = int(np.sum(skel > 0))

                # Find topology
                endpoints, junctions = _find_topology(skel)
                n_endpoints = len(endpoints)
                n_junctions = len(junctions)

                # Compute body length
                body_length = compute_geodesic_body_length(skel)

                # Collect statistics
                stats['skeleton_sizes'].append(skel_px)
                stats['endpoint_counts'].append(n_endpoints)
                stats['junction_counts'].append(n_junctions)

                if body_length > 0:
                    stats['body_lengths'].append(body_length)
                else:
                    stats['filtered_larvae'] += 1

                # Check warnings
                warnings = []
                if skel_px < 20:
                    stats['tiny_skeletons'] += 1
                    warnings.append("TINY_SKEL")
                if body_length < 20.0 and body_length > 0:
                    stats['short_lengths'] += 1
                    warnings.append("SHORT_LENGTH")
                if n_endpoints > 3:
                    stats['noisy_topology'] += 1
                    warnings.append("NOISY_EP")
                if n_junctions > 1:
                    stats['complex_junctions'] += 1
                    warnings.append("COMPLEX_JN")

                warning_str = f" [{', '.join(warnings)}]" if warnings else ""

                # Print info
                print(f"Processing: {date_name:8s} / {image_name:15s} / {larva_name:15s}")
                print(f"  Skeleton pixels: {skel_px:4d}")
                print(f"  Endpoints:       {n_endpoints:4d}")
                print(f"  Junctions:       {n_junctions:4d}")
                print(f"  Body length:     {body_length:6.2f} px{warning_str}")

                # Save debug visualization for first N larvae
                if debug_image_count < MAX_DEBUG_IMAGES:
                    title_info = {
                        'date': date_name,
                        'image': image_name,
                        'larva': larva_name,
                        'skel_px': skel_px,
                        'n_endpoints': n_endpoints,
                        'n_junctions': n_junctions,
                        'full_title': f"{date_name} / {image_name} / {larva_name}"
                    }

                    fig = create_debug_visualization(mask, skel, endpoints, junctions,
                                                    body_length, title_info)

                    # Generate filename
                    safe_date = date_name.replace('.', '_')
                    safe_img = image_name.replace('.', '_')
                    safe_larva = larva_name.replace('.png', '')
                    output_name = f"debug_{safe_date}_{safe_img}_{safe_larva}_ep{n_endpoints}_jn{n_junctions}.png"

                    output_path = OUTPUT_DIR / output_name
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    debug_image_count += 1
                    print(f"  ✓ Saved debug image: {output_name}")

                print()

    # Print summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total larvae scanned:      {stats['total_larvae']}")
    print(f"Skipped by prediction:     {stats['skipped_by_prediction']}")
    print(f"Analyzed (valid+posture):  {stats['total_larvae'] - stats['skipped_by_prediction']}")
    print(f"Filtered by skeleton:      {stats['filtered_larvae']}")
    print(f"Valid body lengths:        {len(stats['body_lengths'])}")
    print()
    print(f"Tiny skeletons (<20px):    {stats['tiny_skeletons']}")
    print(f"Short lengths (<20px):     {stats['short_lengths']}")
    print(f"Noisy topology (>3 EP):    {stats['noisy_topology']}")
    print(f"Complex junctions (>1 JN): {stats['complex_junctions']}")
    print()

    if stats['body_lengths']:
        print(f"Body length statistics (valid only):")
        print(f"  n     = {len(stats['body_lengths'])}")
        print(f"  mean  = {np.mean(stats['body_lengths']):.2f} px")
        print(f"  std   = {np.std(stats['body_lengths']):.2f} px")
        print(f"  min   = {np.min(stats['body_lengths']):.2f} px")
        print(f"  max   = {np.max(stats['body_lengths']):.2f} px")

    print()
    if stats['skeleton_sizes']:
        print(f"Skeleton size statistics:")
        print(f"  mean  = {np.mean(stats['skeleton_sizes']):.1f} px")
        print(f"  std   = {np.std(stats['skeleton_sizes']):.1f} px")
        print(f"  range = [{np.min(stats['skeleton_sizes'])}, {np.max(stats['skeleton_sizes'])}]")

    print()
    print(f"Debug images saved: {debug_image_count} (max {MAX_DEBUG_IMAGES})")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Filter criteria: predicted_valid=1 AND predicted_posture=1")
    print("="*70)


if __name__ == "__main__":
    main()

