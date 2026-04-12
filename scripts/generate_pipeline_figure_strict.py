#!/usr/bin/env python3
"""
Generate a publication-quality 2x3 figure using ONLY the specified files in the figures folder.
Strict rules: do not load or modify any other files.
Saves: /Users/taircarmon/Desktop/growth_function_posture_aware/figures/pipeline_figure.png (300 DPI)
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path("/Users/taircarmon/Desktop/growth_function_posture_aware/figures")
OUT = FIG_DIR / "pipeline_figure.png"

FILES = [
    ("A", "Picture1.jpg"),
    ("B", "Picture2.png"),
    ("C", "Picture3.png"),
    ("D", "Picture4.png"),
    ("E", "Picture5.png"),
    ("F", "Picture6.png"),
]

# Load images exactly as provided, no modifications
images = []
for label, fname in FILES:
    p = FIG_DIR / fname
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    # read with unchanged to preserve channels
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load image: {p}")
    # Convert to RGB for matplotlib
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # drop alpha if present
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append((label, fname, img_rgb))

# Determine panel box size: use max width and max height among images
widths = [img.shape[1] for (_, _, img) in images]
heights = [img.shape[0] for (_, _, img) in images]
panel_w = max(widths)
panel_h = max(heights)

# Prepare canvases: each panel is a white canvas of size (panel_h, panel_w)
canvases = []
for label, fname, img in images:
    h, w = img.shape[0], img.shape[1]
    # compute scale to fit within panel while preserving aspect ratio
    scale = min(panel_w / w, panel_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 255
    # center the resized image
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    canvases.append((label, fname, canvas))

# Create matplotlib figure 2x3
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for ax, (label, fname, canvas) in zip(axes, canvases):
    ax.imshow(canvas)
    ax.axis('off')
    # add panel label at top-left
    ax.text(0.02, 0.03, f"({label})", transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='white',
            bbox=dict(facecolor='black', alpha=0.6, pad=4))

plt.tight_layout(pad=0.6)
fig.savefig(str(OUT), dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

