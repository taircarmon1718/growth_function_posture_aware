#!/usr/bin/env python3
"""
visualize_body_length_debug.py
===============================
Visual debugging tool for body_length computation in the geodesic pipeline.

This script:
1. Loads larvae images directly from analysis_full/
2. Applies the exact same processing as the pipeline
3. Computes skeleton and body_length
4. Creates detailed 4-panel visualizations showing each step
5. Saves debug figures for inspection

Usage:
    python visualize_body_length_debug.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, List, Optional
from collections import deque

# ============================================================
#  EXACT COPIES OF PIPELINE FUNCTIONS
# ============================================================

def _skeletonize(binary_img):
    """Exact copy from pipeline - produces 1-pixel-wide skeleton."""
    try:
        from skimage.morphology import skeletonize as _sk
        bw = (binary_img > 0).astype(bool)
        skel = _sk(bw).astype(np.uint8)
        return skel
    except ImportError:
        pass
    # Fallback: OpenCV morphological thinning
    img = (binary_img > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(200):
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Count 8-connected skeleton neighbours for each pixel."""
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
    dist = {source: 0.0}
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        d = dist[(r, c)]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                nb_rc = (nr, nc)
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
    Topology detection: endpoints (degree 1) and junctions (degree ≥ 3).
    Junction selection: maximise sum of geodesic distances to endpoints.
    """
    nb = _neighbour_count(skel)
    ep_arr = np.argwhere((skel > 0) & (nb == 1))
    jn_arr = np.argwhere((skel > 0) & (nb >= 3))

    endpoints = [tuple(p) for p in ep_arr]
    jn_candidates = [tuple(p) for p in jn_arr]

    if not jn_candidates:
        return endpoints, []

    if len(jn_candidates) == 1:
        return endpoints, [jn_candidates[0]]

    best_jn = jn_candidates[0]
    best_score = -1.0
    for jn in jn_candidates:
        dists = _bfs_distances(skel, jn)
        score = sum(dists.get(ep, 0.0) for ep in endpoints)
        if score > best_score:
            best_score = score
            best_jn = jn

    return endpoints, [best_jn]


def compute_geodesic_body_length(skel: np.ndarray) -> Tuple[float, Optional[Tuple], Optional[Tuple], dict]:
    """
    Exact copy from pipeline with added debug info.

    Returns: (body_length, point1, point2, debug_info)
    where point1 and point2 are the two points defining the body length,
    and debug_info contains intermediate computation details.
    """
    skel_px = int(np.sum(skel > 0))

    debug_info = {
        'skel_px': skel_px,
        'has_junction': False,
        'num_endpoints': 0,
        'num_junctions': 0,
        'algorithm': 'none',
        'distances': {}
    }

    # Filter 1: Minimum skeleton size
    if skel_px < 2:
        return 0.0, None, None, debug_info

    endpoints, junctions = _find_topology(skel)
    debug_info['num_endpoints'] = len(endpoints)
    debug_info['num_junctions'] = len(junctions)

    if not endpoints:
        return 0.0, None, None, debug_info

    # Case 1: junction exists — compute branch lengths
    if junctions:
        debug_info['has_junction'] = True
        debug_info['algorithm'] = 'junction_to_endpoint'

        jn = junctions[0]
        dists = _bfs_distances(skel, jn)
        debug_info['distances'] = dists

        branch_lengths = [dists.get(ep, 0.0) for ep in endpoints]
        if not branch_lengths:
            return 0.0, None, None, debug_info

        max_length = max(branch_lengths)
        max_idx = branch_lengths.index(max_length)

        # Filter 2: Minimum body length
        if max_length < 2.0:
            return 0.0, None, None, debug_info

        return max_length, jn, endpoints[max_idx], debug_info

    # Case 2: no junction — compute geodesic diameter using double BFS
    debug_info['algorithm'] = 'double_bfs'

    # Get all skeleton pixels
    skel_coords = np.argwhere(skel > 0)
    if len(skel_coords) == 0:
        return 0.0, None, None, debug_info

    # Step 1: Pick any skeleton pixel (use first one)
    start_pixel = tuple(skel_coords[0])

    # Step 2: Run BFS from start pixel to find farthest point
    dists_from_start = _bfs_distances(skel, start_pixel)
    if not dists_from_start:
        return 0.0, None, None, debug_info

    # Step 3: Find the farthest skeleton pixel from start
    farthest_pixel = max(dists_from_start.keys(), key=lambda p: dists_from_start[p])

    # Step 4: Run BFS again from the farthest pixel
    dists_from_farthest = _bfs_distances(skel, farthest_pixel)
    debug_info['distances'] = dists_from_farthest

    if not dists_from_farthest:
        return 0.0, None, None, debug_info

    # Step 5: Maximum distance is the geodesic diameter
    max_length = max(dists_from_farthest.values())
    other_end = max(dists_from_farthest.keys(), key=lambda p: dists_from_farthest[p])

    # Filter 2: Minimum body length
    if max_length < 2.0:
        return 0.0, None, None, debug_info

    return max_length, farthest_pixel, other_end, debug_info


# ============================================================
#  IMAGE PROCESSING (EXACT PIPELINE LOGIC)
# ============================================================

def process_larva_from_segmentation(image_dir: Path, larva_component_id: int,
                                     labels_img: np.ndarray, stats: np.ndarray,
                                     original_img: np.ndarray) -> dict:
    """
    Process a single larva component from the segmentation mask.
    Returns dictionary with all intermediate results for visualization.

    Args:
        image_dir: Path to the image directory (contains segmentation_mask.png)
        larva_component_id: Component ID in the labeled image
        labels_img: Labeled components image from segmentation
        stats: Component statistics from cv2.connectedComponentsWithStats
        original_img: Original grayscale image for visualization

    Returns:
        dict: Dictionary with processed results for visualization
    """
    # Extract component mask
    comp_mask = (labels_img == larva_component_id).astype(np.uint8)

    # Get bounding box
    x = stats[larva_component_id, cv2.CC_STAT_LEFT]
    y = stats[larva_component_id, cv2.CC_STAT_TOP]
    w = stats[larva_component_id, cv2.CC_STAT_WIDTH]
    h = stats[larva_component_id, cv2.CC_STAT_HEIGHT]

    # Add padding
    pad = 10
    y1, y2 = max(0, y - pad), min(original_img.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(original_img.shape[1], x + w + pad)

    # Crop the mask and original image
    crop = comp_mask[y1:y2, x1:x2]
    crop_gray = original_img[y1:y2, x1:x2]

    # Skeletonize the component mask
    skel = _skeletonize(crop)

    # Detect topology
    endpoints, junctions = _find_topology(skel)

    # Compute body length
    body_length, point1, point2, debug_info = compute_geodesic_body_length(skel)

    # Create visualization of the full binary mask (for panel 2)
    full_binary = np.zeros_like(original_img, dtype=np.uint8)
    full_binary[comp_mask > 0] = 255

    return {
        'original': cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR),
        'gray': original_img,
        'binary': full_binary,
        'crop': crop,
        'crop_gray': crop_gray,
        'skeleton': skel,
        'bbox': (x1, y1, x2, y2),
        'endpoints': endpoints,
        'junctions': junctions,
        'body_length': body_length,
        'point1': point1,
        'point2': point2,
        'debug_info': debug_info
    }


# ============================================================
#  VISUALIZATION
# ============================================================

def reconstruct_geodesic_path(skel: np.ndarray, start: tuple, end: tuple, distances: dict) -> List[tuple]:
    """
    Reconstruct the geodesic path from start to end using BFS distances.
    """
    if not distances or end not in distances:
        return [start, end]

    path = [end]
    current = end

    # Work backwards from end to start
    while current != start:
        r, c = current
        best_neighbor = None
        best_dist = float('inf')

        # Check all 8 neighbors
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if neighbor in distances and distances[neighbor] < best_dist:
                    if (0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]
                            and skel[nr, nc] > 0):
                        best_dist = distances[neighbor]
                        best_neighbor = neighbor

        if best_neighbor is None:
            break

        path.append(best_neighbor)
        current = best_neighbor

        # Safety check to prevent infinite loops
        if len(path) > skel.shape[0] * skel.shape[1]:
            break

    return list(reversed(path))


def visualize_larva_debug(result: dict, date: str, filename: str, save_path: Path):
    """
    Create simple visualization showing binary mask with length calculation overlay.
    """
    fig = plt.figure(figsize=(10, 10))

    # Single panel: Binary mask with geodesic path overlay
    ax = plt.subplot(1, 1, 1)

    # Show binary mask (cropped)
    mask_display = np.stack([result['crop']*255]*3, axis=-1).astype(np.uint8)

    # Overlay skeleton in white
    mask_display[result['skeleton'] > 0] = [200, 200, 200]

    ax.imshow(mask_display)

    debug_info = result['debug_info']
    algorithm = debug_info.get('algorithm', 'none')

    # Draw endpoints as RED circles
    for ep in result['endpoints']:
        circle = mpatches.Circle((ep[1], ep[0]), radius=2, color='red', fill=True, zorder=10)
        ax.add_patch(circle)

    # Draw junctions as BLUE circles
    for jn in result['junctions']:
        circle = mpatches.Circle((jn[1], jn[0]), radius=3, color='blue', fill=True, zorder=11)
        ax.add_patch(circle)

    if result['point1'] is not None and result['point2'] is not None:
        # Draw the two key points
        p1 = result['point1']
        p2 = result['point2']

        # Draw geodesic path in GREEN
        distances = debug_info.get('distances', {})
        if distances:
            path = reconstruct_geodesic_path(result['skeleton'], p1, p2, distances)
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 1], path_array[:, 0], 'g-', linewidth=3,
                        label='Geodesic Path', zorder=5)

        # Mark the two endpoints of measurement
        ax.plot(p1[1], p1[0], 'ro', markersize=12, label='Start Point', zorder=10)
        ax.plot(p2[1], p2[0], 'bo', markersize=12, label='End Point', zorder=10)

        # Annotate with length
        mid_y = (p1[0] + p2[0]) / 2
        mid_x = (p1[1] + p2[1]) / 2
        ax.text(mid_x, mid_y, f'{result["body_length"]:.1f} px',
                color='yellow', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
                ha='center', va='center', zorder=15)

    # Title with statistics
    skel_px = debug_info.get('skel_px', 0)
    num_ep = debug_info.get('num_endpoints', 0)
    num_jn = debug_info.get('num_junctions', 0)

    ax.set_title(f'{date} / {filename}\n' +
                 f'Algorithm: {algorithm}\n' +
                 f'Skeleton: {skel_px} px | Endpoints: {num_ep} | Junctions: {num_jn}\n' +
                 f'Body Length: {result["body_length"]:.2f} px',
                 fontsize=12, fontweight='bold', pad=15)

    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
#  MAIN FUNCTION
# ============================================================

def normalize_date_format(date_str):
    """
    Normalize date format to match directory structure.
    Excel reads '19.10' as '19.1', but directory is '19.10'.
    '3.11' stays '3.11'.

    Args:
        date_str: Date string from Excel (e.g., '19.1', '20.1', '3.11')

    Returns:
        Normalized date string (e.g., '19.10', '20.10', '3.11')
    """
    parts = date_str.split('.')
    if len(parts) == 2:
        day, month = parts
        # If month is single digit, pad with zero
        if len(month) == 1:
            return f"{day}.{month}0"
    return date_str


def main():
    """Main function to process and visualize larvae."""

    print("="*70)
    print("BODY LENGTH VISUALIZATION DEBUG TOOL")
    print("Filtering: Valid larvae with correct posture (predictions = 1)")
    print("="*70)

    # Setup paths
    base_dir = Path(__file__).parent
    analysis_dir = base_dir / "analysis_full_binary_masks_only"  # Use binary masks directory
    output_dir = base_dir / "debug_body_length_visualization"
    output_dir.mkdir(exist_ok=True)

    # Load predictions from dual_larva_pipeline_geodesic_fixed
    predictions_file = base_dir / "dual_larva_models_geodesic2" / "predictions" / "posture_predictions.xlsx"

    if not predictions_file.exists():
        print(f"\n❌ Predictions file not found: {predictions_file}")
        print("   Please run dual_larva_pipeline_geodesic_fixed.py first to generate predictions")
        return 1

    print(f"\nLoading predictions from: {predictions_file.name}")

    try:
        import pandas as pd
        predictions_df = pd.read_excel(predictions_file)

        # Convert date and image_name to strings to avoid path concatenation errors
        predictions_df['date'] = predictions_df['date'].astype(str)
        predictions_df['image_name'] = predictions_df['image_name'].astype(str)

        # Filter for valid larvae with correct posture
        # predicted_valid == 1 AND predicted_posture == 1
        valid_posture = predictions_df[
            (predictions_df['predicted_valid'] == 1) &
            (predictions_df['predicted_posture'] == 1)
        ].copy()

        print(f"✓ Loaded {len(predictions_df)} total predictions")
        print(f"✓ Found {len(valid_posture)} larvae with valid=1 AND posture=1")

        if len(valid_posture) == 0:
            print("\n❌ No larvae found with both predictions = 1")
            return 1

        # Show distribution by date
        date_counts = valid_posture['date'].value_counts().sort_index()
        print(f"\n  Distribution of valid+posture larvae by date:")
        for date, count in date_counts.items():
            print(f"    {date}: {count} larvae")

    except Exception as e:
        print(f"\n❌ Error loading predictions: {e}")
        return 1

    print(f"\nInput directory:  {analysis_dir}")
    print(f"Output directory: {output_dir}")

    if not analysis_dir.exists():
        print(f"\n❌ Analysis directory not found: {analysis_dir}")
        return 1

    # Collect one example per date from the filtered predictions
    # Use the actual date format from the data
    target_dates = ['19.1', '20.1', '21.1']  # Updated to match actual date format

    # First try target dates
    selected_larvae = []

    for date in target_dates:
        # Filter predictions for this date
        date_larvae = valid_posture[valid_posture['date'] == date]

        if len(date_larvae) == 0:
            print(f"⚠️  No valid posture larvae found in {date}")
            continue

        # Take the first one
        first_larva = date_larvae.iloc[0]
        selected_larvae.append({
            'date': first_larva['date'],
            'image_name': first_larva['image_name'],
            'larva_filename': first_larva['larva_filename'],
            'body_length_px': first_larva['body_length_px'],
            'valid_confidence': first_larva.get('valid_confidence', 0),
            'posture_confidence': first_larva.get('posture_confidence', 0)
        })
        print(f"  Selected from {date}: {first_larva['image_name']} / {first_larva['larva_filename']}")

    # If we didn't find enough, get from any available dates
    if len(selected_larvae) < 3:
        print(f"\n  Only found {len(selected_larvae)} in target dates, selecting from other dates...")

        # Get all available dates
        all_dates = valid_posture['date'].unique()
        used_dates = {larva['date'] for larva in selected_larvae}

        for date in sorted(all_dates):
            if date in used_dates:
                continue

            if len(selected_larvae) >= 3:
                break

            date_larvae = valid_posture[valid_posture['date'] == date]
            if len(date_larvae) > 0:
                first_larva = date_larvae.iloc[0]
                selected_larvae.append({
                    'date': first_larva['date'],
                    'image_name': first_larva['image_name'],
                    'larva_filename': first_larva['larva_filename'],
                    'body_length_px': first_larva['body_length_px'],
                    'valid_confidence': first_larva.get('valid_confidence', 0),
                    'posture_confidence': first_larva.get('posture_confidence', 0)
                })
                print(f"  Selected from {date}: {first_larva['image_name']} / {first_larva['larva_filename']}")

    if not selected_larvae:
        print("\n❌ No valid posture larvae found in target dates!")
        return 1

    print(f"\n📊 Processing {len(selected_larvae)} larvae (1 per date)...")
    print("="*70)

    # Process each selected larva
    success_count = 0

    for idx, larva_info in enumerate(selected_larvae):
        date = larva_info['date']
        image_name = larva_info['image_name']
        larva_filename = larva_info['larva_filename']
        predicted_length = larva_info['body_length_px']

        print(f"\n[{idx+1}/{len(selected_larvae)}] Processing: {date} / {image_name} / {larva_filename}")
        print(f"  Predicted body length: {predicted_length:.2f} px")
        print(f"  Valid confidence: {larva_info['valid_confidence']:.3f}")
        print(f"  Posture confidence: {larva_info['posture_confidence']:.3f}")

        try:
            # Normalize date format to match directory structure
            # Excel reads '19.10' as '19.1', but directory is '19.10'
            normalized_date = normalize_date_format(date)

            # Construct paths using normalized date
            img_dir = analysis_dir / normalized_date / image_name
            larva_mask_path = img_dir / 'larvae_reports' / larva_filename

            if not larva_mask_path.exists():
                print(f"  ⚠️  Larva mask not found: {larva_filename}")
                continue

            # Load the saved larva mask (binary mask from run_pipeline_binary_masks.py)
            larva_mask = cv2.imread(str(larva_mask_path), cv2.IMREAD_GRAYSCALE)

            if larva_mask is None:
                print(f"  ⚠️  Failed to load larva mask")
                continue

            # Load original grayscale image for visualization
            images_folder = base_dir / normalized_date / 'images'
            original_img = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = images_folder / f"{image_name}{ext}"
                if img_path.exists():
                    original = cv2.imread(str(img_path))
                    if original is not None:
                        original_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
                        break

            if original_img is None:
                original_img = np.zeros_like(larva_mask)

            # The larva mask is already a clean binary mask
            larva_binary = (larva_mask > 127).astype(np.uint8)

            # Check if mask is not empty
            if np.sum(larva_binary) == 0:
                print(f"  ⚠️  Empty larva mask")
                continue

            # Use the mask directly as crop
            crop = larva_binary

            # Match crop_gray size to crop
            if crop.shape[0] <= original_img.shape[0] and crop.shape[1] <= original_img.shape[1]:
                crop_gray = original_img[:crop.shape[0], :crop.shape[1]]
            else:
                crop_gray = np.zeros_like(crop, dtype=np.uint8)

            # Skeletonize
            skel = _skeletonize(crop)

            # Detect topology
            endpoints, junctions = _find_topology(skel)

            # Compute body length
            body_length, point1, point2, debug_info = compute_geodesic_body_length(skel)

            # Print statistics
            print(f"  Recomputed:")
            print(f"    Skeleton pixels:  {debug_info['skel_px']}")
            print(f"    Endpoints:        {debug_info['num_endpoints']}")
            print(f"    Junctions:        {debug_info['num_junctions']}")
            print(f"    Algorithm:        {debug_info['algorithm']}")
            print(f"    Body length:      {body_length:.2f} px")
            if predicted_length > 0:
                print(f"    Difference:       {abs(body_length - predicted_length):.2f} px")

            # Create result dictionary for visualization
            full_binary = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
            full_binary[crop > 0] = 255

            result = {
                'original': cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if original_img.ndim == 2 else original_img,
                'gray': original_img,
                'binary': full_binary,
                'crop': crop,
                'crop_gray': crop_gray,
                'skeleton': skel,
                'bbox': (0, 0, crop.shape[1], crop.shape[0]),
                'endpoints': endpoints,
                'junctions': junctions,
                'body_length': body_length,
                'point1': point1,
                'point2': point2,
                'debug_info': debug_info
            }

            # Create visualization
            larva_num = larva_filename.replace('larva_', '').replace('.png', '')
            output_filename = f"debug_{date}_{image_name}_{larva_num}_valid_posture.png"
            output_path = output_dir / output_filename

            title = f"{image_name} / {larva_filename}\nValid+Posture (conf: {larva_info['posture_confidence']:.2f})"
            visualize_larva_debug(result, date, title, output_path)
            print(f"  ✓ Saved: {output_filename}")

            success_count += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"COMPLETE: {success_count}/{len(selected_larvae)} larvae visualized")
    print(f"Criteria: predicted_valid=1 AND predicted_posture=1")
    print(f"Output: {output_dir}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

