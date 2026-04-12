#!/usr/bin/env python3
"""
visualize_larva_length_methods.py
==================================
Compare three different body length estimation methods:
1. Feret Diameter (Convex Hull)
2. PCA Major Axis
3. Skeleton Geodesic Length (Biological Centerline)

This script loads larva images and visualizes all three methods side-by-side.
"""

import sys
from pathlib import Path
from collections import deque
from typing import Tuple, List, Optional
import random

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA

# ============================================================
#  CONFIGURATIONxq
# ============================================================
ROOT_DIR = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full_binary_masks_only"
OUTPUT_DIR = ROOT_DIR / "dual_larva_models_geodesic2" / "length_method_visualization"
N_SAMPLES = 10  # Number of larvae to visualize
RANDOM_SEED = 42

# ============================================================
#  METHOD 1: FERET DIAMETER (CONVEX HULL)
# ============================================================
def compute_feret_diameter(mask: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute Feret diameter using convex hull.

    Returns:
        length: Feret diameter in pixels
        p1: First endpoint (x, y)
        p2: Second endpoint (x, y)
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, None, None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return 0.0, None, None

    hull = cv2.convexHull(contour)
    if len(hull) < 2:
        return 0.0, None, None

    hull_points = hull.reshape(-1, 2)

    # Find maximum distance between any two hull points
    max_distance = 0.0
    best_i, best_j = 0, 0

    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            dist = np.sqrt((hull_points[i][0] - hull_points[j][0]) ** 2 +
                          (hull_points[i][1] - hull_points[j][1]) ** 2)
            if dist > max_distance:
                max_distance = dist
                best_i, best_j = i, j

    p1 = hull_points[best_i]
    p2 = hull_points[best_j]

    return float(max_distance), p1, p2


# ============================================================
#  METHOD 2: PCA MAJOR AXIS
# ============================================================
def compute_pca_length(mask: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute body length using PCA principal axis projection.

    Returns:
        length: PCA length in pixels
        p1: First endpoint (x, y)
        p2: Second endpoint (x, y)
    """
    coords = np.argwhere(mask > 0)

    if len(coords) < 2:
        return 0.0, None, None

    # Convert to (x, y) format
    points = coords[:, [1, 0]]

    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Apply PCA
    pca = PCA(n_components=1)
    pca.fit(points)

    # Get principal component (direction vector)
    pca_vector = pca.components_[0]

    # Project all points onto the principal axis
    centered_points = points - centroid
    projections = np.dot(centered_points, pca_vector)

    # Find min and max projections
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # Compute endpoints
    p1 = centroid + min_proj * pca_vector
    p2 = centroid + max_proj * pca_vector

    length_px = max_proj - min_proj

    return float(length_px), p1.astype(int), p2.astype(int)


# ============================================================
#  METHOD 3: SKELETON GEODESIC LENGTH
# ============================================================
def compute_skeleton_geodesic_length(mask: np.ndarray) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """
    Compute body length using skeleton geodesic distance.

    Returns:
        length: Geodesic length in pixels
        path: List of (x, y) coordinates along the skeleton path
    """
    # Skeletonize
    skel = skeletonize(mask > 0).astype(np.uint8)

    if np.sum(skel) < 2:
        return 0.0, None

    # Find endpoints (pixels with exactly 1 neighbor)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)

    endpoints_mask = (skel == 1) & (neighbor_count == 1)
    endpoints = np.argwhere(endpoints_mask)

    if len(endpoints) < 2:
        return 0.0, None

    # Convert endpoints to (row, col) format
    endpoints_list = [(int(ep[0]), int(ep[1])) for ep in endpoints]

    # Compute geodesic distances from each endpoint using BFS
    max_length = 0.0
    best_path = None

    for start_ep in endpoints_list:
        distances, predecessors = _bfs_skeleton(skel, start_ep)

        # Check distance to all other endpoints
        for end_ep in endpoints_list:
            if start_ep == end_ep:
                continue

            dist = distances.get(end_ep, 0.0)
            if dist > max_length:
                max_length = dist
                # Reconstruct path
                path = _reconstruct_path(predecessors, start_ep, end_ep)
                best_path = path

    return float(max_length), best_path


def _bfs_skeleton(skel: np.ndarray, start: Tuple[int, int]) -> Tuple[dict, dict]:
    """
    BFS on skeleton with diagonal cost sqrt(2).

    Returns:
        distances: dict mapping (row, col) -> distance
        predecessors: dict mapping (row, col) -> (parent_row, parent_col)
    """
    rows, cols = skel.shape
    distances = {start: 0.0}
    predecessors = {start: None}
    queue = deque([start])
    visited = {start}

    # 8-connected neighbors
    neighbors_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    # Costs: sqrt(2) for diagonal, 1 for orthogonal
    costs = [
        np.sqrt(2), 1, np.sqrt(2),
        1,             1,
        np.sqrt(2), 1, np.sqrt(2)
    ]

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        for (dr, dc), cost in zip(neighbors_8, costs):
            r, c = current[0] + dr, current[1] + dc

            if 0 <= r < rows and 0 <= c < cols and skel[r, c] > 0:
                neighbor = (r, c)
                new_dist = current_dist + cost

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current

                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)

    return distances, predecessors


def _reconstruct_path(predecessors: dict, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstruct path from start to end using predecessors.

    Returns:
        path: List of (row, col) coordinates
    """
    if end not in predecessors:
        return []

    path = []
    current = end

    while current is not None:
        path.append(current)
        current = predecessors.get(current)

    path.reverse()
    return path


# ============================================================
#  METHOD 4: CENTERLINE SKELETON LENGTH
# ============================================================
def compute_centerline_skeleton_length(mask: np.ndarray) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """
    Compute true centerline length using longest geodesic path across entire skeleton.

    Unlike the skeleton geodesic method which only considers endpoint-to-endpoint paths,
    this method finds the longest path across the entire skeleton graph.

    Returns:
        length: Centerline length in pixels
        path: List of (row, col) coordinates along the path
    """
    # Skeletonize
    skel = skeletonize(mask > 0).astype(np.uint8)

    if np.sum(skel) < 2:
        return 0.0, None

    # Get all skeleton pixels
    skel_pixels = np.argwhere(skel > 0)
    skel_pixels_list = [(int(r), int(c)) for r, c in skel_pixels]

    if len(skel_pixels_list) < 2:
        return 0.0, None

    # Find longest path by checking from all skeleton pixels
    max_length = 0.0
    best_path = None

    # Sample subset of pixels to avoid O(n²) for large skeletons
    sample_size = min(20, len(skel_pixels_list))
    sampled_pixels = random.sample(skel_pixels_list, sample_size)

    for start_pixel in sampled_pixels:
        distances, predecessors = _bfs_skeleton(skel, start_pixel)

        # Find the farthest reachable pixel
        max_dist = 0.0
        farthest_pixel = None

        for pixel, dist in distances.items():
            if dist > max_dist:
                max_dist = dist
                farthest_pixel = pixel

        if max_dist > max_length:
            max_length = max_dist
            # Reconstruct path
            if farthest_pixel is not None:
                path = _reconstruct_path(predecessors, start_pixel, farthest_pixel)
                best_path = path

    return float(max_length), best_path


# ============================================================
#  METHOD 5: SPLINE CENTERLINE LENGTH
# ============================================================
def compute_spline_centerline_length(mask: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """
    Compute centerline length by fitting a smooth spline through the skeleton.

    Returns:
        length: Spline curve length in pixels
        spline_points: Array of (x, y) points along the spline
    """
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        # Fallback: return 0 if scipy not available
        return 0.0, None

    # Skeletonize
    skel = skeletonize(mask > 0).astype(np.uint8)

    if np.sum(skel) < 5:
        return 0.0, None

    # Get skeleton pixels
    skel_coords = np.argwhere(skel > 0)

    if len(skel_coords) < 5:
        return 0.0, None

    # Convert to (x, y)
    points = skel_coords[:, [1, 0]].astype(float)

    # Sort points to create an ordered path (simple approach: sort by distance from first point)
    ordered_points = [points[0]]
    remaining = list(range(1, len(points)))

    while remaining:
        last_point = ordered_points[-1]
        # Find nearest remaining point
        distances = [np.linalg.norm(points[i] - last_point) for i in remaining]
        nearest_idx = remaining[np.argmin(distances)]
        ordered_points.append(points[nearest_idx])
        remaining.remove(nearest_idx)

        # Limit path length to avoid very long computation
        if len(ordered_points) > 200:
            break

    ordered_points = np.array(ordered_points)

    if len(ordered_points) < 5:
        return 0.0, None

    try:
        # Fit spline (x, y coordinates)
        tck, u = splprep([ordered_points[:, 0], ordered_points[:, 1]], s=2.0, k=min(3, len(ordered_points) - 1))

        # Sample spline densely
        u_fine = np.linspace(0, 1, 250)
        x_fine, y_fine = splev(u_fine, tck)
        spline_points = np.column_stack([x_fine, y_fine])

        # Compute length by summing distances
        length = 0.0
        for i in range(1, len(spline_points)):
            dx = spline_points[i, 0] - spline_points[i-1, 0]
            dy = spline_points[i, 1] - spline_points[i-1, 1]
            length += np.sqrt(dx**2 + dy**2)

        return float(length), spline_points

    except Exception:
        # Spline fitting failed
        return 0.0, None


# ============================================================
#  METHOD 6: ELLIPSE MAJOR AXIS LENGTH
# ============================================================
def compute_ellipse_major_axis(mask: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute body length by fitting an ellipse and measuring its major axis.

    Returns:
        length: Major axis length in pixels
        p1: First endpoint (x, y)
        p2: Second endpoint (x, y)
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, None, None

    contour = max(contours, key=cv2.contourArea)

    # Need at least 5 points to fit ellipse
    if len(contour) < 5:
        return 0.0, None, None

    try:
        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Extract parameters
        center, axes, angle = ellipse
        center_x, center_y = center
        major_axis, minor_axis = axes

        # Major axis is the larger of the two axes
        major_length = max(major_axis, minor_axis)

        # Determine which axis is major
        if major_axis >= minor_axis:
            # Major axis is along the primary direction
            angle_rad = np.deg2rad(angle)
        else:
            # Major axis is perpendicular
            angle_rad = np.deg2rad(angle + 90)

        # Compute endpoints of major axis
        half_length = major_length / 2.0
        dx = half_length * np.cos(angle_rad)
        dy = half_length * np.sin(angle_rad)

        p1 = np.array([center_x - dx, center_y - dy], dtype=int)
        p2 = np.array([center_x + dx, center_y + dy], dtype=int)

        return float(major_length), p1, p2

    except Exception:
        # Ellipse fitting failed
        return 0.0, None, None


# ============================================================
#  VISUALIZATION
# ============================================================
def visualize_three_methods(
    img_gray: np.ndarray,
    mask: np.ndarray,
    larva_name: str,
    output_path: Path
):
    """
    Create a 3-column visualization comparing all three methods.
    """
    # Compute all three methods
    feret_len, feret_p1, feret_p2 = compute_feret_diameter(mask)
    pca_len, pca_p1, pca_p2 = compute_pca_length(mask)
    skel_len, skel_path = compute_skeleton_geodesic_length(mask)

    # Create figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert grayscale to RGB for visualization
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # --- Method 1: Feret Diameter (Blue) ---
    img1 = img_rgb.copy()
    if feret_p1 is not None and feret_p2 is not None:
        cv2.line(img1, tuple(feret_p1), tuple(feret_p2), (0, 0, 255), 2)  # Blue (BGR)
        cv2.circle(img1, tuple(feret_p1), 1, (0, 255, 0), -1)  # Green dot
        cv2.circle(img1, tuple(feret_p2), 1, (0, 255, 0), -1)  # Green dot

    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Feret Diameter\nLength = {feret_len:.2f} px', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # --- Method 2: PCA Major Axis (Green) ---
    img2 = img_rgb.copy()
    if pca_p1 is not None and pca_p2 is not None:
        cv2.line(img2, tuple(pca_p1), tuple(pca_p2), (0, 255, 0), 2)  # Green (BGR)
        cv2.circle(img2, tuple(pca_p1), 1, (255, 0, 0), -1)  # Blue dot
        cv2.circle(img2, tuple(pca_p2), 1, (255, 0, 0), -1)  # Blue dot

    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'PCA Major Axis\nLength = {pca_len:.2f} px', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # --- Method 3: Skeleton Geodesic (Red) ---
    img3 = img_rgb.copy()
    if skel_path is not None and len(skel_path) > 1:
        # Convert path from (row, col) to (x, y) for OpenCV
        path_xy = np.array([(c, r) for r, c in skel_path], dtype=np.int32)
        cv2.polylines(img3, [path_xy], False, (255, 0, 0), 2)  # Red (BGR)
        # Mark endpoints
        cv2.circle(img3, tuple(path_xy[0]), 1, (255, 255, 0), -1)  # Cyan dot
        cv2.circle(img3, tuple(path_xy[-1]), 1, (255, 255, 0), -1)  # Cyan dot

    axes[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Skeleton Geodesic\nLength = {skel_len:.2f} px', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Overall title
    fig.suptitle(f'Body Length Comparison: {larva_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return feret_len, pca_len, skel_len


def visualize_six_methods(
    img_gray: np.ndarray,
    mask: np.ndarray,
    larva_name: str,
    output_path: Path
):
    """
    Create a 2x3 grid visualization comparing all six methods.
    """
    # Compute all six methods
    feret_len, feret_p1, feret_p2 = compute_feret_diameter(mask)
    pca_len, pca_p1, pca_p2 = compute_pca_length(mask)
    skel_len, skel_path = compute_skeleton_geodesic_length(mask)
    centerline_len, centerline_path = compute_centerline_skeleton_length(mask)
    spline_len, spline_points = compute_spline_centerline_length(mask)
    ellipse_len, ellipse_p1, ellipse_p2 = compute_ellipse_major_axis(mask)

    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Convert grayscale to RGB for visualization
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # --- Method 1: Feret Diameter (Blue) ---
    img1 = img_rgb.copy()
    if feret_p1 is not None and feret_p2 is not None:
        cv2.line(img1, tuple(feret_p1), tuple(feret_p2), (0, 0, 255), 2)  # Blue (BGR)
        cv2.circle(img1, tuple(feret_p1), 1, (0, 255, 0), -1)  # Green dot
        cv2.circle(img1, tuple(feret_p2), 1, (0, 255, 0), -1)  # Green dot

    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Feret Diameter\nLength = {feret_len:.2f} px', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # --- Method 2: PCA Major Axis (Green) ---
    img2 = img_rgb.copy()
    if pca_p1 is not None and pca_p2 is not None:
        cv2.line(img2, tuple(pca_p1), tuple(pca_p2), (0, 255, 0), 2)  # Green (BGR)
        cv2.circle(img2, tuple(pca_p1), 1, (255, 0, 0), -1)  # Blue dot
        cv2.circle(img2, tuple(pca_p2), 1, (255, 0, 0), -1)  # Blue dot

    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'PCA Major Axis\nLength = {pca_len:.2f} px', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # --- Method 3: Skeleton Geodesic (Red) ---
    img3 = img_rgb.copy()
    if skel_path is not None and len(skel_path) > 1:
        path_xy = np.array([(c, r) for r, c in skel_path], dtype=np.int32)
        cv2.polylines(img3, [path_xy], False, (255, 0, 0), 2)  # Red (BGR)
        cv2.circle(img3, tuple(path_xy[0]), 1, (255, 255, 0), -1)  # Cyan dot
        cv2.circle(img3, tuple(path_xy[-1]), 1, (255, 255, 0), -1)  # Cyan dot

    axes[0, 2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Skeleton Geodesic\nLength = {skel_len:.2f} px', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    # --- Method 4: Centerline Skeleton (Magenta) ---
    img4 = img_rgb.copy()
    if centerline_path is not None and len(centerline_path) > 1:
        path_xy = np.array([(c, r) for r, c in centerline_path], dtype=np.int32)
        cv2.polylines(img4, [path_xy], False, (255, 0, 255), 2)  # Magenta (BGR)
        cv2.circle(img4, tuple(path_xy[0]), 1, (0, 255, 255), -1)  # Yellow dot
        cv2.circle(img4, tuple(path_xy[-1]), 1, (0, 255, 255), -1)  # Yellow dot

    axes[1, 0].imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Centerline Skeleton\nLength = {centerline_len:.2f} px', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # --- Method 5: Spline Centerline (Orange) ---
    img5 = img_rgb.copy()
    if spline_points is not None and len(spline_points) > 1:
        # Draw spline curve
        for i in range(len(spline_points) - 1):
            pt1 = (int(spline_points[i, 0]), int(spline_points[i, 1]))
            pt2 = (int(spline_points[i+1, 0]), int(spline_points[i+1, 1]))
            cv2.line(img5, pt1, pt2, (0, 165, 255), 2)  # Orange (BGR)
        # Mark endpoints
        pt_start = (int(spline_points[0, 0]), int(spline_points[0, 1]))
        pt_end = (int(spline_points[-1, 0]), int(spline_points[-1, 1]))
        cv2.circle(img5, pt_start, 1, (128, 0, 128), -1)  # Purple dot
        cv2.circle(img5, pt_end, 1, (128, 0, 128), -1)  # Purple dot

    axes[1, 1].imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Spline Centerline\nLength = {spline_len:.2f} px', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    # --- Method 6: Ellipse Major Axis (Cyan) ---
    img6 = img_rgb.copy()
    if ellipse_p1 is not None and ellipse_p2 is not None:
        cv2.line(img6, tuple(ellipse_p1), tuple(ellipse_p2), (255, 255, 0), 2)  # Cyan (BGR)
        cv2.circle(img6, tuple(ellipse_p1), 1, (0, 0, 255), -1)  # Red dot
        cv2.circle(img6, tuple(ellipse_p2), 1, (0, 0, 255), -1)  # Red dot

    axes[1, 2].imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Ellipse Major Axis\nLength = {ellipse_len:.2f} px', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    # Overall title
    fig.suptitle(f'Body Length Comparison (6 Methods): {larva_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return feret_len, pca_len, skel_len, centerline_len, spline_len, ellipse_len


# ============================================================
#  MAIN PIPELINE
# ============================================================
def collect_larvae() -> List[Tuple[str, str, str]]:
    """
    Collect all available larvae from the analysis directory.

    Returns:
        List of (date, image_name, larva_filename)
    """
    larvae = []

    if not ANALYSIS_DIR.exists():
        print(f"❌ Analysis directory not found: {ANALYSIS_DIR}")
        return larvae

    for date_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not date_dir.is_dir() or not date_dir.name[0].isdigit():
            continue

        date = date_dir.name

        for image_dir in sorted(date_dir.iterdir()):
            if not image_dir.is_dir():
                continue

            image_name = image_dir.name
            larvae_reports_dir = image_dir / "larvae_reports"

            if not larvae_reports_dir.exists():
                continue

            for larva_file in sorted(larvae_reports_dir.iterdir()):
                if larva_file.name.endswith('.png') and larva_file.name.startswith('larva_'):
                    larvae.append((date, image_name, larva_file.name))

    return larvae


def main():
    print("=" * 80)
    print("LARVA LENGTH METHODS VISUALIZATION - 6 METHODS")
    print("Comparing:")
    print("  1. Feret Diameter")
    print("  2. PCA Major Axis")
    print("  3. Skeleton Geodesic")
    print("  4. Centerline Skeleton")
    print("  5. Spline Centerline")
    print("  6. Ellipse Major Axis")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Collect all larvae
    all_larvae = collect_larvae()
    print(f"\n✓ Found {len(all_larvae)} larvae in dataset")

    if len(all_larvae) == 0:
        print("❌ No larvae found. Exiting.")
        sys.exit(1)

    # Randomly sample N larvae
    random.seed(RANDOM_SEED)
    sample_larvae = random.sample(all_larvae, min(N_SAMPLES, len(all_larvae)))
    print(f"✓ Selected {len(sample_larvae)} larvae for visualization")

    # Process each larva
    results = {
        'feret': [],
        'pca': [],
        'skeleton': [],
        'centerline': [],
        'spline': [],
        'ellipse': []
    }

    print("\n" + "=" * 80)
    print("PROCESSING LARVAE")
    print("=" * 80)

    for idx, (date, image_name, larva_filename) in enumerate(sample_larvae, 1):
        print(f"\n[{idx}/{len(sample_larvae)}] {date} / {image_name} / {larva_filename}")

        # Load larva image
        larva_path = ANALYSIS_DIR / date / image_name / "larvae_reports" / larva_filename

        if not larva_path.exists():
            print(f"  ⚠️  File not found: {larva_path}")
            continue

        # Load as grayscale
        img = cv2.imread(str(larva_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"  ⚠️  Failed to load image")
            continue

        # Threshold to binary mask using Otsu
        _, mask = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Visualize with all 6 methods
        output_filename = f"example_{idx:03d}_{date}_{larva_filename}"
        output_path = OUTPUT_DIR / output_filename

        try:
            feret_len, pca_len, skel_len, centerline_len, spline_len, ellipse_len = visualize_six_methods(
                img,
                mask,
                f"{date}/{larva_filename}",
                output_path
            )

            results['feret'].append(feret_len)
            results['pca'].append(pca_len)
            results['skeleton'].append(skel_len)
            results['centerline'].append(centerline_len)
            results['spline'].append(spline_len)
            results['ellipse'].append(ellipse_len)

            print(f"  ✓ Feret: {feret_len:.2f} px")
            print(f"  ✓ PCA: {pca_len:.2f} px")
            print(f"  ✓ Skeleton: {skel_len:.2f} px")
            print(f"  ✓ Centerline: {centerline_len:.2f} px")
            print(f"  ✓ Spline: {spline_len:.2f} px")
            print(f"  ✓ Ellipse: {ellipse_len:.2f} px")
            print(f"  ✓ Saved: {output_filename}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if results['feret']:
        print(f"\nAverage lengths:")
        print(f"  Feret Diameter      : {np.mean(results['feret']):.2f} px")
        print(f"  PCA Major Axis      : {np.mean(results['pca']):.2f} px")
        print(f"  Skeleton Geodesic   : {np.mean(results['skeleton']):.2f} px")
        print(f"  Centerline Skeleton : {np.mean(results['centerline']):.2f} px")
        print(f"  Spline Centerline   : {np.mean(results['spline']):.2f} px")
        print(f"  Ellipse Major Axis  : {np.mean(results['ellipse']):.2f} px")

        print(f"\nMethod differences (mean absolute):")
        methods = ['feret', 'pca', 'skeleton', 'centerline', 'spline', 'ellipse']
        method_names = ['Feret', 'PCA', 'Skeleton', 'Centerline', 'Spline', 'Ellipse']

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                diff = np.mean([abs(a - b) for a, b in zip(results[methods[i]], results[methods[j]])])
                print(f"  |{method_names[i]} - {method_names[j]}|: {diff:.2f} px")

        print(f"\nStandard deviations:")
        print(f"  Feret Diameter      : {np.std(results['feret']):.2f} px")
        print(f"  PCA Major Axis      : {np.std(results['pca']):.2f} px")
        print(f"  Skeleton Geodesic   : {np.std(results['skeleton']):.2f} px")
        print(f"  Centerline Skeleton : {np.std(results['centerline']):.2f} px")
        print(f"  Spline Centerline   : {np.std(results['spline']):.2f} px")
        print(f"  Ellipse Major Axis  : {np.std(results['ellipse']):.2f} px")

    print("\n" + "=" * 80)
    print(f"✓ COMPLETE: {len(results['feret'])} larvae visualized")
    print(f"✓ Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()



