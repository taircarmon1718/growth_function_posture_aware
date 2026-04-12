#!/usr/bin/env python3
"""
Skeleton Topology Debug Tool for Larval Binary Masks

This script analyzes the skeleton topology of larval binary masks to understand
the distribution of endpoints and junctions in the dataset.

For each larva, it:
- Skeletonizes the binary mask
- Detects endpoints (skeleton pixels with exactly 1 neighbor)
- Detects junctions (skeleton pixels with ≥3 neighbors)
- Prints detailed debug information
- Saves visualization examples for inspection

Output: topology_debug_examples/ folder with labeled visualizations
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import skeletonize
from scipy import ndimage
import sys


def find_skeleton_topology(skeleton):
    """
    Analyze skeleton topology to find endpoints and junctions.

    Parameters:
    -----------
    skeleton : np.ndarray
        Binary skeleton image (True/False or 1/0)

    Returns:
    --------
    endpoints : list of tuples
        List of (row, col) coordinates of endpoint pixels
    junctions : list of tuples
        List of (row, col) coordinates of junction pixels
    skeleton_pixel_count : int
        Total number of skeleton pixels
    """
    # Ensure binary skeleton
    skel_binary = (skeleton > 0).astype(np.uint8)

    # Count skeleton pixels
    skeleton_pixel_count = np.sum(skel_binary)

    if skeleton_pixel_count == 0:
        return [], [], 0

    # Create 3x3 kernel to count neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Don't count the center pixel

    # Count neighbors for each skeleton pixel
    neighbor_count = ndimage.convolve(skel_binary, kernel, mode='constant', cval=0)

    # Only consider neighbor counts where skeleton exists
    neighbor_count = neighbor_count * skel_binary

    # Endpoints: exactly 1 neighbor
    endpoint_mask = (neighbor_count == 1) & (skel_binary == 1)
    endpoints = list(zip(*np.where(endpoint_mask)))

    # Junctions: 3 or more neighbors
    junction_mask = (neighbor_count >= 3) & (skel_binary == 1)
    junctions = list(zip(*np.where(junction_mask)))

    return endpoints, junctions, skeleton_pixel_count


def visualize_skeleton_topology(mask, skeleton, endpoints, junctions,
                                  skeleton_px, output_path):
    """
    Create a visualization showing the skeleton with marked endpoints and junctions.

    Parameters:
    -----------
    mask : np.ndarray
        Original binary mask
    skeleton : np.ndarray
        Binary skeleton
    endpoints : list of tuples
        Endpoint coordinates
    junctions : list of tuples
        Junction coordinates
    skeleton_px : int
        Total skeleton pixels
    output_path : Path
        Where to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original binary mask
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Binary Mask', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Skeleton overlay on mask
    # Create RGB image for overlay
    overlay = np.stack([mask, mask, mask], axis=-1).astype(float)

    # Draw skeleton in white
    skel_coords = np.where(skeleton > 0)
    overlay[skel_coords[0], skel_coords[1], :] = [1.0, 1.0, 1.0]

    axes[1].imshow(overlay)
    axes[1].set_title('Skeleton Overlay', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Skeleton with topology markers
    # Start with mask as grayscale background
    topology_viz = np.stack([mask * 0.5, mask * 0.5, mask * 0.5], axis=-1)

    # Draw skeleton in white
    topology_viz[skel_coords[0], skel_coords[1], :] = [1.0, 1.0, 1.0]

    axes[2].imshow(topology_viz)

    # Mark endpoints with red circles
    if endpoints:
        ep_rows, ep_cols = zip(*endpoints)
        axes[2].scatter(ep_cols, ep_rows, c='red', s=100, marker='o',
                       edgecolors='yellow', linewidths=2, label='Endpoints', zorder=10)

    # Mark junctions with blue circles
    if junctions:
        jn_rows, jn_cols = zip(*junctions)
        axes[2].scatter(jn_cols, jn_rows, c='blue', s=150, marker='o',
                       edgecolors='cyan', linewidths=2, label='Junctions', zorder=10)

    # Add legend
    if endpoints or junctions:
        axes[2].legend(loc='upper right', fontsize=10)

    axes[2].set_title(
        f'Topology: {len(endpoints)} EP, {len(junctions)} JN\n'
        f'Skeleton pixels: {skeleton_px}',
        fontsize=12, fontweight='bold'
    )
    axes[2].axis('off')

    # Overall title
    fig.suptitle(f'{output_path.stem}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_larva_mask(mask_path, save_visualization=False, output_dir=None):
    """
    Process a single larva mask: skeletonize and analyze topology.

    Parameters:
    -----------
    mask_path : Path
        Path to the binary mask image
    save_visualization : bool
        Whether to save a visualization
    output_dir : Path
        Directory to save visualization (if enabled)

    Returns:
    --------
    dict with keys: endpoints, junctions, skeleton_px, warnings
    """
    # Load the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"  ⚠️  Failed to load: {mask_path.name}")
        return None

    # Convert to binary (white = larva, black = background)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Check if mask is empty
    if np.sum(binary_mask) == 0:
        print(f"  ⚠️  Empty mask: {mask_path.name}")
        return None

    # Skeletonize
    skeleton = skeletonize(binary_mask).astype(np.uint8)

    # Analyze topology
    endpoints, junctions, skeleton_px = find_skeleton_topology(skeleton)

    # Print debug information
    print(f"Processing: {mask_path.name}")
    print(f"  Skeleton pixels: {skeleton_px}")
    print(f"  Endpoints: {len(endpoints)}")
    print(f"  Junctions: {len(junctions)}")
    print(f"  Topology: ({len(endpoints)} endpoints, {len(junctions)} junction{'s' if len(junctions) != 1 else ''})")

    # Check for unusual cases
    warnings = []

    if len(junctions) > 1:
        msg = f"  ⚠️  WARNING: more than 1 junction ({len(junctions)} junctions detected)"
        print(msg)
        warnings.append("multiple_junctions")

    if len(endpoints) > 3:
        msg = f"  ⚠️  WARNING: more than 3 endpoints ({len(endpoints)} endpoints detected)"
        print(msg)
        warnings.append("many_endpoints")

    if len(endpoints) == 0:
        msg = f"  ⚠️  WARNING: no endpoints detected (possible loop or noise)"
        print(msg)
        warnings.append("no_endpoints")

    if skeleton_px < 10:
        msg = f"  ⚠️  WARNING: very small skeleton ({skeleton_px} pixels)"
        print(msg)
        warnings.append("tiny_skeleton")

    print()  # blank line

    # Save visualization if requested
    if save_visualization and output_dir:
        viz_filename = f"debug_{mask_path.stem}_ep{len(endpoints)}_jn{len(junctions)}.png"
        viz_path = output_dir / viz_filename

        visualize_skeleton_topology(
            binary_mask, skeleton, endpoints, junctions,
            skeleton_px, viz_path
        )

    return {
        'filename': mask_path.name,
        'endpoints': len(endpoints),
        'junctions': len(junctions),
        'skeleton_px': skeleton_px,
        'warnings': warnings
    }


def main():
    """
    Main function to process all larva masks in the dataset.
    """
    print("=" * 70)
    print("SKELETON TOPOLOGY DEBUG TOOL")
    print("=" * 70)
    print()

    # Configuration
    # Modify this path to point to your larva masks directory
    # Default: look for analysis_full_binary_masks_only folders
    ROOT_DIR = Path(__file__).parent

    # Try to find larva masks automatically
    analysis_dir = ROOT_DIR / "analysis_full_binary_masks_only"

    if not analysis_dir.exists():
        print(f"❌ Directory not found: {analysis_dir}")
        print("Please update the script with the correct path to larva masks.")
        sys.exit(1)

    # Collect all larva mask files
    print(f"📂 Scanning for larva masks in: {analysis_dir}")
    mask_files = []

    for date_dir in sorted(analysis_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        for img_dir in sorted(date_dir.iterdir()):
            if not img_dir.is_dir():
                continue

            larvae_dir = img_dir / "larvae_reports"
            if not larvae_dir.exists():
                continue

            # Find all larva_*.png files
            for mask_file in sorted(larvae_dir.glob("larva_*.png")):
                mask_files.append(mask_file)

    if not mask_files:
        print("❌ No larva mask files found!")
        print("Expected pattern: analysis_full_binary_masks_only/<date>/<image>/larvae_reports/larva_*.png")
        sys.exit(1)

    print(f"✓ Found {len(mask_files)} larva masks")
    print()

    # Create output directory for visualizations
    output_dir = ROOT_DIR / "topology_debug_examples"
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Saving visualizations to: {output_dir}")
    print()

    # Process all larvae
    print("=" * 70)
    print("PROCESSING LARVAE")
    print("=" * 70)
    print()

    # Number of visualizations to save
    MAX_VISUALIZATIONS = 20

    results = []
    viz_count = 0

    for i, mask_file in enumerate(mask_files, 1):
        # Save visualization for first N larvae
        save_viz = (viz_count < MAX_VISUALIZATIONS)

        result = process_larva_mask(mask_file, save_visualization=save_viz,
                                     output_dir=output_dir)

        if result:
            results.append(result)
            if save_viz:
                viz_count += 1

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()

    print(f"Total larvae processed: {len(results)}")
    print(f"Visualizations saved: {viz_count}")
    print()

    if results:
        # Count topology patterns
        topology_counts = {}
        for r in results:
            key = (r['endpoints'], r['junctions'])
            topology_counts[key] = topology_counts.get(key, 0) + 1

        print("Topology distribution:")
        for (ep, jn), count in sorted(topology_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(results)
            print(f"  {ep} endpoints, {jn} junction(s): {count:4d} larvae ({pct:5.1f}%)")
        print()

        # Count warnings
        warning_counts = {}
        for r in results:
            for w in r['warnings']:
                warning_counts[w] = warning_counts.get(w, 0) + 1

        if warning_counts:
            print("Warning counts:")
            for warning, count in sorted(warning_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / len(results)
                print(f"  {warning}: {count:4d} larvae ({pct:5.1f}%)")
            print()

        # Skeleton size statistics
        skeleton_sizes = [r['skeleton_px'] for r in results]
        print("Skeleton size statistics:")
        print(f"  Min:    {min(skeleton_sizes):6.1f} pixels")
        print(f"  Max:    {max(skeleton_sizes):6.1f} pixels")
        print(f"  Mean:   {np.mean(skeleton_sizes):6.1f} pixels")
        print(f"  Median: {np.median(skeleton_sizes):6.1f} pixels")
        print(f"  Std:    {np.std(skeleton_sizes):6.1f} pixels")
        print()

    print("=" * 70)
    print("COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

