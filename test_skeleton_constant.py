#!/usr/bin/env python3
"""
Test what skeleton topology produces the constant length of ~14.486 px (3.364 mm).
"""
import numpy as np
import sys
from pathlib import Path

# Import the geodesic functions from the pipeline
sys.path.insert(0, str(Path(__file__).parent))
from dual_larva_classification_pipeline_geodesic import compute_geodesic_body_length

PIXEL_TO_MM = 0.232255814

# Test various small skeleton patterns
test_cases = []

# Case 1: Single horizontal line (10 pixels)
skel1 = np.zeros((20, 20), dtype=np.uint8)
skel1[10, 5:15] = 255
test_cases.append(("Horizontal line 10px", skel1))

# Case 2: Vertical line (10 pixels)
skel2 = np.zeros((20, 20), dtype=np.uint8)
skel2[5:15, 10] = 255
test_cases.append(("Vertical line 10px", skel2))

# Case 3: T-shape with short arms
skel3 = np.zeros((20, 20), dtype=np.uint8)
skel3[10, 7:14] = 255  # horizontal arm (7 pixels)
skel3[10:18, 10] = 255  # vertical body (8 pixels)
test_cases.append(("T-shape short", skel3))

# Case 4: Diagonal line
skel4 = np.zeros((20, 20), dtype=np.uint8)
for i in range(10):
    skel4[5+i, 5+i] = 255
test_cases.append(("Diagonal line 10px", skel4))

# Case 5: Single pixel
skel5 = np.zeros((20, 20), dtype=np.uint8)
skel5[10, 10] = 255
test_cases.append(("Single pixel", skel5))

# Case 6: 2 pixels
skel6 = np.zeros((20, 20), dtype=np.uint8)
skel6[10, 10:12] = 255
test_cases.append(("2 pixels horizontal", skel6))

# Case 7: 3 pixels line
skel7 = np.zeros((20, 20), dtype=np.uint8)
skel7[10, 10:13] = 255
test_cases.append(("3 pixels horizontal", skel7))

# Case 8: L-shape (10 pixels each arm)
skel8 = np.zeros((25, 25), dtype=np.uint8)
skel8[10, 5:15] = 255  # horizontal
skel8[10:20, 14] = 255  # vertical
test_cases.append(("L-shape 10+10", skel8))

# Case 9: T-shape specific topology to hit ~14.5 px
# If branch = 14.486, that could be:
# - straight line: 14 pixels (geodesic = 13)
# - with diagonals: 10 pixels + 3 diagonals * 1.414 = 10 + 4.242 = 14.242
skel9 = np.zeros((30, 30), dtype=np.uint8)
skel9[15, 10:20] = 255  # 10px horizontal
skel9[15:21, 15] = 255  # 6px vertical down from center
test_cases.append(("T-shape to ~14.5px", skel9))

# Case 10: Exact attempt at 14.486
# geodesic = 14.486 could be: 10 axial + ~3.17 diagonal steps
# 10*1.0 + 3*1.4142 = 10 + 4.2426 = 14.2426 (close!)
skel10 = np.zeros((30, 30), dtype=np.uint8)
for i in range(10):
    skel10[10, 10+i] = 255  # 9 steps = 9.0
for i in range(4):
    skel10[10+i, 19+i] = 255  # 3 diagonal steps = 4.242
test_cases.append(("10 axial + 3 diag", skel10))

print("="*70)
print("SKELETON GEODESIC LENGTH TEST")
print("="*70)
print(f"\nTarget: 14.486 px (3.364 mm)\n")

for name, skel in test_cases:
    length_px = compute_geodesic_body_length(skel)
    length_mm = length_px * PIXEL_TO_MM
    match = "✓ MATCH!" if abs(length_px - 14.486) < 0.01 else ""
    print(f"{name:25s} : {length_px:8.4f} px ({length_mm:6.3f} mm) {match}")

print("\n" + "="*70)

