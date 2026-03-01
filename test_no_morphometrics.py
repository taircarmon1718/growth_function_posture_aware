#!/usr/bin/env python3
"""
Test script to verify the morphometric text removal works correctly
"""
import cv2
import numpy as np
from pathlib import Path

def test_visual_extraction():
    sample_path = Path('analysis_full/31.10/IMG_7382/larvae_reports/larva_001.png')

    if not sample_path.exists():
        print(f"❌ Sample not found: {sample_path}")
        return

    img = cv2.imread(str(sample_path))
    if img is None:
        print("❌ Could not load image")
        return

    img_h, img_w = img.shape[:2]
    print(f"Original image: {img_w}x{img_h}")

    # Apply the same detection logic as in the tool
    gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    visual_width = img_w
    for x in range(img_w - 1, img_w // 2, -1):
        col_mean = np.mean(gray_check[:, x])
        if col_mean > 20:
            visual_width = x + 1
            break

    # If detection failed, fall back
    if visual_width == img_w:
        visual_width = int(img_w * 0.75)
        print("⚠️  Detection failed, using fallback (75% of width)")
    else:
        print(f"✓ Detected visual content ends at x={visual_width}")

    img_visual_only = img[:, :visual_width]
    vis_h, vis_w = img_visual_only.shape[:2]

    print(f"\nVisual-only portion: {vis_w}x{vis_h}")
    print(f"Removed: {img_w - vis_w} pixels (text panel)")
    print(f"Kept: {100*vis_w/img_w:.1f}% of original width")

    # Save the result
    output_path = Path('test_no_morphometrics.png')
    cv2.imwrite(str(output_path), img_visual_only)
    print(f"\n✅ Success!")
    print(f"   Saved visual-only version to: {output_path}")
    print(f"   This is what you'll see in the labeling tool")
    print(f"   NO morphometric text should be visible!")

if __name__ == "__main__":
    test_visual_extraction()

