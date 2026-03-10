#!/usr/bin/env python3
"""
PCA Body Length Measurement Demo
==================================
Demonstrates body length measurement using PCA principal axis on a single larva mask.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Configuration
PIXEL_TO_MM = 0.232255814
ROOT_DIR = Path(__file__).parent.resolve()
ANALYSIS_DIR = ROOT_DIR / "analysis_full_binary_masks_only"

def load_binary_mask(mask_path):
    """Load and convert image to binary mask."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Convert to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def compute_pca_body_length(mask):
    """
    Compute body length using PCA principal axis.

    Returns:
        length_px: body length in pixels
        centroid: (x, y) centroid coordinates
        pca_vector: principal component direction vector
        min_point: (x, y) tail point
        max_point: (x, y) head point
    """
    # Extract mask pixel coordinates
    coords = np.argwhere(mask > 0)  # Returns (row, col) = (y, x)

    if len(coords) == 0:
        return 0.0, None, None, None, None

    # Convert to (x, y) format for PCA
    points = coords[:, [1, 0]]  # Swap to (col, row) = (x, y)

    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Apply PCA
    pca = PCA(n_components=1)
    pca.fit(points)

    # Get principal component (direction vector)
    pca_vector = pca.components_[0]

    # Project all points onto the principal axis
    # Projection = dot product with principal component
    centered_points = points - centroid
    projections = np.dot(centered_points, pca_vector)

    # Find min and max projections
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # Body length is the range of projections
    length_px = max_proj - min_proj

    # Calculate the actual min/max points in image coordinates
    min_point = centroid + min_proj * pca_vector
    max_point = centroid + max_proj * pca_vector

    return length_px, centroid, pca_vector, min_point, max_point


def visualize_pca_measurement(mask, length_px, centroid, pca_vector, min_point, max_point):
    """
    Create visualization showing:
    - Binary mask
    - PCA principal axis (red line)
    - Head/tail points (blue circles)
    - Centroid (yellow circle)
    - Measured length annotation
    """
    # Create RGB image for visualization
    h, w = mask.shape
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(vis, cmap='gray')

    if centroid is not None:
        # Draw PCA axis as a long red line through the centroid
        # Extend the line far in both directions
        axis_length = max(h, w)  # Make it long enough
        axis_start = centroid - axis_length * pca_vector
        axis_end = centroid + axis_length * pca_vector

        ax.plot([axis_start[0], axis_end[0]],
                [axis_start[1], axis_end[1]],
                'r-', linewidth=2, label='PCA Principal Axis', alpha=0.7)

        # Draw centroid (yellow)
        ax.plot(centroid[0], centroid[1], 'yo', markersize=10,
                label='Centroid', markeredgecolor='black', markeredgewidth=1.5)

        # Draw min/max points (head and tail) in blue
        ax.plot(min_point[0], min_point[1], 'bo', markersize=12,
                label='Tail (min projection)', markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(max_point[0], max_point[1], 'bo', markersize=12,
                label='Head (max projection)', markeredgecolor='black', markeredgewidth=1.5)

        # Draw line connecting head and tail (to show measured length)
        ax.plot([min_point[0], max_point[0]],
                [min_point[1], max_point[1]],
                'b--', linewidth=1.5, alpha=0.5)

        # Annotate body length
        mid_point = (min_point + max_point) / 2
        length_mm = length_px * PIXEL_TO_MM
        ax.text(mid_point[0], mid_point[1] - 15,
                f'Length: {length_px:.2f} px\n({length_mm:.3f} mm)',
                fontsize=12, color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                ha='center')

    ax.set_title('PCA-Based Body Length Measurement', fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    return fig


def find_sample_larva():
    """Find one sample larva mask from the dataset."""
    # Search for a larva with predicted_valid=1 and predicted_posture=1
    predictions_file = ROOT_DIR / "dual_larva_models_geodesic2" / "predictions" / "posture_predictions.xlsx"

    if predictions_file.exists():
        import pandas as pd
        df = pd.read_excel(predictions_file)
        filtered = df[(df['predicted_valid'] == 1) & (df['predicted_posture'] == 1)]

        if len(filtered) > 0:
            # Get first valid larva
            row = filtered.iloc[0]
            date = str(row['date'])
            image_name = str(row['image_name'])
            larva_filename = str(row['larva_filename'])

            mask_path = ANALYSIS_DIR / date / image_name / "larvae_reports" / larva_filename
            if mask_path.exists():
                return mask_path, date, image_name, larva_filename

    # Fallback: search for any larva
    date_folders = sorted([d for d in ANALYSIS_DIR.iterdir() if d.is_dir()])
    for date_dir in date_folders:
        if date_dir.name in {'18.10', '18.1'}:
            continue

        image_folders = sorted([img for img in date_dir.iterdir() if img.is_dir()])
        for image_dir in image_folders:
            larvae_dir = image_dir / "larvae_reports"
            if larvae_dir.exists():
                larva_files = sorted(larvae_dir.glob("larva_*.png"))
                if larva_files:
                    larva_path = larva_files[0]
                    return larva_path, date_dir.name, image_dir.name, larva_path.name

    return None, None, None, None


def main():
    print("="*70)
    print("PCA BODY LENGTH MEASUREMENT DEMO")
    print("="*70)
    print()

    # Find a sample larva
    mask_path, date, image_name, larva_name = find_sample_larva()

    if mask_path is None:
        print("❌ No larva masks found in the dataset!")
        return

    print(f"Processing larva:")
    print(f"  Date:  {date}")
    print(f"  Image: {image_name}")
    print(f"  Larva: {larva_name}")
    print(f"  Path:  {mask_path}")
    print()

    # Load mask
    mask = load_binary_mask(mask_path)
    if mask is None:
        print("❌ Failed to load mask!")
        return

    print(f"✓ Loaded mask: {mask.shape[1]}x{mask.shape[0]} pixels")
    print(f"  Mask pixels: {np.sum(mask > 0)}")
    print()

    # Compute PCA body length
    length_px, centroid, pca_vector, min_point, max_point = compute_pca_body_length(mask)

    if centroid is None:
        print("❌ No mask pixels found!")
        return

    length_mm = length_px * PIXEL_TO_MM

    print("="*70)
    print("PCA MEASUREMENT RESULTS")
    print("="*70)
    print(f"Centroid:        ({centroid[0]:.2f}, {centroid[1]:.2f})")
    print(f"PCA vector:      ({pca_vector[0]:.4f}, {pca_vector[1]:.4f})")
    print(f"Tail point:      ({min_point[0]:.2f}, {min_point[1]:.2f})")
    print(f"Head point:      ({max_point[0]:.2f}, {max_point[1]:.2f})")
    print()
    print(f"Body length:     {length_px:.2f} pixels")
    print(f"Body length:     {length_mm:.3f} mm")
    print("="*70)
    print()

    # Create visualization
    print("Creating visualization...")
    fig = visualize_pca_measurement(mask, length_px, centroid, pca_vector, min_point, max_point)

    # Save figure
    output_path = ROOT_DIR / "pca_body_length_demo.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")

    # Show figure
    plt.show()

    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()

