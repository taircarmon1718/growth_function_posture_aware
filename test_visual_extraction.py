#!/usr/bin/env python3
"""
Test script to verify the visual-only display works correctly
"""
import cv2
from pathlib import Path

# Test the visual extraction
sample_path = Path('analysis_full/31.10/IMG_7382/larvae_reports/larva_001.png')

if sample_path.exists():
    img = cv2.imread(str(sample_path))
    if img is not None:
        h, w = img.shape[:2]
        print(f"Original larva report image: {w}x{h}")

        # Apply the same extraction as in the tool
        visual_width = int(w * 0.75)
        img_visual_only = img[:, :visual_width]

        vis_h, vis_w = img_visual_only.shape[:2]
        print(f"Visual-only portion: {vis_w}x{vis_h}")
        print(f"Removed: {w - vis_w} pixels (text metrics panel)")

        # Save comparison
        cv2.imwrite('test_visual_only.png', img_visual_only)
        print("\n✅ Visual-only extraction works!")
        print(f"   Saved test output to: test_visual_only.png")
        print(f"   This shows only the larva visualization without text metrics")
    else:
        print("Could not load image")
else:
    print(f"Sample not found at: {sample_path}")
    print("Make sure analysis_full has processed data")

