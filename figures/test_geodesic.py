#!/usr/bin/env python3
"""
Simple test script to verify the geodesic distance computation works.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import cv2

print("Starting test script...")

# Create a simple T-shaped skeleton for testing
def create_test_skeleton():
    skel = np.zeros((30, 30), dtype=np.uint8)

    # Vertical line
    for r in range(10, 25):
        skel[r, 15] = 1

    # Horizontal line
    for c in range(8, 23):
        skel[15, c] = 1

    return skel

def _neighbour_count(skel):
    ker = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, ker) * skel.astype(np.uint8)

def _find_topology(skel):
    nb = _neighbour_count(skel)
    endpoints = [tuple(p) for p in np.argwhere((skel > 0) & (nb == 1))]
    junctions = [tuple(p) for p in np.argwhere((skel > 0) & (nb >= 3))]
    return endpoints, junctions

def _geodesic_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        is_diag = abs(p1[0] - p2[0]) == 1 and abs(p1[1] - p2[1]) == 1
        length += np.sqrt(2) if is_diag else 1.0
    return length

def _bfs_path(skel, start, goal):
    q = deque([(start, [start])])
    visited = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == goal:
            return path, _geodesic_length(path)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc] > 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
    return [], 0.0

def find_longest_path(skel, endpoints):
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

def main():
    print("Creating test skeleton...")
    skel = create_test_skeleton()

    print("Finding topology...")
    endpoints, junctions = _find_topology(skel)
    print(f"Found {len(endpoints)} endpoints, {len(junctions)} junctions")

    print("Finding longest path...")
    longest_path = find_longest_path(skel, endpoints)
    print(f"Longest path has {len(longest_path)} points")

    if longest_path:
        length = _geodesic_length(longest_path)
        print(f"Path length: {length:.2f} pixels")

    # Create a simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Show skeleton
    ax1.imshow(skel, cmap='gray')
    ax1.set_title("Test Skeleton")
    ax1.axis('off')

    # Show path
    ax2.imshow(skel, cmap='gray')
    if longest_path:
        path_arr = np.array(longest_path)
        ax2.plot(path_arr[:, 1], path_arr[:, 0], 'r-', linewidth=2)
    if endpoints:
        ep_arr = np.array(endpoints)
        ax2.scatter(ep_arr[:, 1], ep_arr[:, 0], c='blue', s=50)
    ax2.set_title("Longest Geodesic Path")
    ax2.axis('off')

    output_path = Path(__file__).parent / "test_geodesic.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved test figure to: {output_path}")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
