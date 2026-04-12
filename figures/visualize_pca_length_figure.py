#!/usr/bin/env python3
"""
visualize_pca_length_figure.py

Create a clean scientific 4-panel figure illustrating PCA-based body length
estimation for one larva mask. Panels (A)–(D):

(A) Binary segmentation mask
(B) Mask + PCA principal axis
(C) Projection of pixels onto axis (perpendicular lines)
(D) Body length measurement (double arrow between min/max projections)

Style: white background, black mask, blue PCA axis, red endpoints, gray
projection lines. Outputs saved to: figures/pca_length_figure.svg and .png

Usage:
    python visualize_pca_length_figure.py [--mask PATH]

If --mask is not provided the script searches for a larva mask under
`analysis_full_binary_masks_only/*/*/larvae_reports/larva_*.png` and uses the
first one found.

This is a standalone script and does not modify any existing files.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

PIXEL_TO_MM = 0.232255814


def find_first_mask(root: Path) -> Path:
    # Search for larva_*.png masks in analysis_full_binary_masks_only
    search_root = root / 'analysis_full_binary_masks_only'
    if not search_root.exists():
        return None
    for date_dir in sorted(search_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for img_dir in sorted(date_dir.iterdir()):
            lr = img_dir / 'larvae_reports'
            if not lr.exists():
                continue
            masks = sorted(lr.glob('larva_*.png'))
            if masks:
                return masks[0]
    return None


def load_binary_mask(p: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {p}")
    bw = (img > 127).astype(np.uint8)
    # Ensure mask is foreground=1 on white background when plotting
    return bw


def compute_pca_axis(points: np.ndarray):
    # points: Nx2 (x, y)
    if PCA is not None:
        pca = PCA(n_components=1)
        pca.fit(points)
        vec = pca.components_[0]
        # ensure unit norm
        vec = vec / np.linalg.norm(vec)
        centroid = pca.mean_
    else:
        # fallback: use SVD
        centroid = points.mean(axis=0)
        Xc = points - centroid
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        vec = Vt[0]
        vec = vec / np.linalg.norm(vec)
    return centroid, vec


def project_points(points: np.ndarray, centroid: np.ndarray, vec: np.ndarray):
    # return projections (scalar) and projection points in xy
    centered = points - centroid
    projs = centered.dot(vec)
    proj_points = centroid + np.outer(projs, vec)
    return projs, proj_points


def draw_panel_A(ax, mask):
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_title('(A) Binary mask')
    ax.axis('off')


def draw_panel_B(ax, mask, pmin, pmax):
    """Draw the mask and a PCA axis segment between the two extreme projections.
    pmin and pmax are (x,y) coordinates on the axis."""
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    # draw axis segment between pmin and pmax (thin blue line)
    ax.add_line(Line2D([pmin[0], pmax[0]], [pmin[1], pmax[1]], color='blue', linewidth=1.2))
    ax.set_title('(B) PCA axis')
    ax.axis('off')


def draw_panel_C(ax, mask, points, proj_pts, max_lines=300):
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    N = points.shape[0]
    # choose subset for projection lines to avoid clutter
    if N > max_lines:
        idx = np.random.choice(N, size=max_lines, replace=False)
    else:
        idx = np.arange(N)
    for i in idx:
        x, y = points[i]
        px, py = proj_pts[i]
        ax.plot([x, px], [y, py], color='#bbbbbb', linewidth=0.6, alpha=0.9)
    # draw axis as thin blue line between extremes if provided via proj_pts
    # compute axis endpoints from proj_pts min/max
    if proj_pts.shape[0] > 0:
        # find extremes along projection direction
        # proj_pts columns contain x,y; use min/max along projection scalar via bounding box
        xs = proj_pts[:, 0]
        ys = proj_pts[:, 1]
        ax.add_line(Line2D([xs.min(), xs.max()], [ys.min(), ys.max()], color='blue', linewidth=1.2))
    ax.set_title('(C) Projections onto PCA axis')
    ax.axis('off')


def draw_panel_D(ax, mask, pmin, pmax):
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    # draw thin axis segment between pmin and pmax
    ax.add_line(Line2D([pmin[0], pmax[0]], [pmin[1], pmax[1]], color='blue', linewidth=1.2))
    # draw endpoints as small red circles
    ax.add_patch(Circle((pmin[0], pmin[1]), radius=2.2, color='red'))
    ax.add_patch(Circle((pmax[0], pmax[1]), radius=2.2, color='red'))
    # draw thin double arrow between pmin and pmax
    arrow = FancyArrowPatch((pmin[0], pmin[1]), (pmax[0], pmax[1]), arrowstyle='<->', mutation_scale=10, linewidth=1.4, color='black')
    ax.add_patch(arrow)
    # annotate length in pixels and mm at midpoint (small unobtrusive text)
    dist_px = np.linalg.norm(pmax - pmin)
    dist_mm = dist_px * PIXEL_TO_MM
    mid = (pmin + pmax) / 2.0
    ax.text(mid[0], mid[1] - 8, f'{dist_px:.1f} px ({dist_mm:.3f} mm)', color='black', fontsize=9, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    ax.set_title('(D) PCA length')
    ax.axis('off')


def generate_synthetic_larva_mask(height=200, width=600, curvature=0.35,
                                  head_width=40, tail_width=6, seed=42):
    """
    Generate a synthetic elongated, slightly curved larva-like binary mask.

    The centerline is a quadratic Bezier curve from left to right with a
    single control point offset to create curvature. Width tapers from
    head_width to tail_width along the centerline.
    """
    import cv2
    np.random.seed(seed)
    H, W = int(height), int(width)
    # endpoints
    p0 = np.array([50.0, H * 0.5])
    p1 = np.array([W - 50.0, H * 0.5])
    # control point offset vertically to create curvature
    ctrl_y = H * 0.5 + curvature * H * (0.3 + 0.2 * (np.random.rand() - 0.5))
    ctrl_x = W * 0.5 + (np.random.rand() - 0.5) * (W * 0.02)
    c = np.array([ctrl_x, ctrl_y])

    n_pts = max(200, W // 2)
    t = np.linspace(0.0, 1.0, n_pts)
    # quadratic Bezier
    center = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, c) + np.outer(t ** 2, p1)

    # approximate tangents by finite differences
    d_cent = np.gradient(center, axis=0)
    # normals (perpendicular)
    normals = np.zeros_like(d_cent)
    normals[:, 0] = -d_cent[:, 1]
    normals[:, 1] = d_cent[:, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norms

    # width tapering from head to tail
    p_pow = 1.2
    widths = tail_width + (head_width - tail_width) * (1.0 - t ** p_pow)

    left_pts = center + (widths / 2.0)[:, None] * normals
    right_pts = center - (widths / 2.0)[:, None] * normals

    # build polygon (left side then reversed right side)
    poly = np.vstack([left_pts, right_pts[::-1]])
    poly_int = np.round(poly).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    try:
        cv2.fillPoly(mask, [poly_int], color=1)
    except Exception:
        # safety fallback: draw circles along centerline
        for (x, y), w in zip(center, widths):
            cv2.circle(mask, (int(round(x)), int(round(y))), int(round(w / 2)), 1, -1)

    # smooth edges slightly and threshold to get crisp mask
    if hasattr(cv2, 'GaussianBlur'):
        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)
        mask = (mask_blur > 0.2).astype(np.uint8)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', type=str, default=None, help='Path to a binary larva mask image (PNG)')
    parser.add_argument('--synthetic', action='store_true', help='Create a synthetic curved larva mask instead of loading one from disk')
    parser.add_argument('--out', type=str, default='figures/pca_length_figure', help='Output root path (no extension)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    root = Path(__file__).parent
    if args.synthetic:
        mask = generate_synthetic_larva_mask(height=220, width=660, curvature=0.35,
                                             head_width=46, tail_width=6, seed=args.seed)
    else:
        # Attempt to pick a high-confidence real larva from predictions
        preds_path = root / 'dual_larva_models_geodesic2' / 'predictions' / 'predictions_all_larvae.xlsx'
        mask = None
        if (not args.mask) and preds_path.exists():
            try:
                import pandas as _pd
                dfp = _pd.read_excel(preds_path)
                # Prefer highly confident valid predictions
                cond = (dfp.get('predicted_valid') == 1) & (dfp.get('valid_confidence').notna())
                cand = dfp[cond].sort_values('valid_confidence', ascending=False)
                if len(cand) == 0:
                    # fallback to posture predictions
                    cond2 = (dfp.get('predicted_posture') == 1) & (dfp.get('posture_confidence').notna())
                    cand = dfp[cond2].sort_values('posture_confidence', ascending=False)
                if len(cand) > 0:
                    best = cand.iloc[0]
                    date = str(best['date'])
                    image_name = str(best['image_name'])
                    fname = str(best['larva_filename'])
                    candidate_path = root / 'analysis_full_binary_masks_only' / date / image_name / 'larvae_reports' / fname
                    if candidate_path.exists():
                        mask = load_binary_mask(candidate_path)
                        print(f"Using high-confidence larva from predictions: {date}/{image_name}/{fname}")
                    else:
                        # try to locate file by filename in the analysis_full tree (robust to minor date/name mismatches)
                        search_root = root / 'analysis_full_binary_masks_only'
                        matches = list(search_root.rglob(fname)) if search_root.exists() else []
                        chosen = None
                        if matches:
                            # prefer matches that include the expected image_name in their path
                            for m in matches:
                                if image_name in str(m.parent.parent.name):
                                    chosen = m
                                    break
                            if chosen is None:
                                chosen = matches[0]
                        if chosen is not None and chosen.exists():
                            mask = load_binary_mask(chosen)
                            print(f"Using found larva file: {chosen} (predictions entry: {date}/{image_name}/{fname})")
            except Exception:
                mask = None
        # If mask not set by predictions, use explicit --mask or first found
        if mask is None:
            mask_path = Path(args.mask) if args.mask else find_first_mask(root)
            if mask_path is None or not mask_path.exists():
                raise SystemExit('No larva mask found. Provide --mask PATH or ensure analysis_full_binary_masks_only exists or use --synthetic')
            mask = load_binary_mask(mask_path)

    # get coordinates (x, y) of foreground pixels
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise SystemExit('Empty mask (no foreground pixels)')
    points = np.column_stack([xs.astype(float), ys.astype(float)])

    centroid, vec = compute_pca_axis(points)
    projs, proj_pts = project_points(points, centroid, vec)
    # find extreme projections
    min_idx = int(np.argmin(projs))
    max_idx = int(np.argmax(projs))
    pmin = proj_pts[min_idx]
    pmax = proj_pts[max_idx]

    # prepare figure with 4 horizontal panels (consistent axes limits)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    fig.patch.set_facecolor('white')
    # Set consistent extents so all panels share scale
    H, W = mask.shape
    extent = [0, W, H, 0]

    # Panel A
    draw_panel_A(axes[0], mask)
    # Panel B: draw axis segment between extremes
    draw_panel_B(axes[1], mask, pmin, pmax)
    # Panel C: draw projections (use proj_pts for axis endpoints)
    draw_panel_C(axes[2], mask, points, proj_pts, max_lines=500)
    # Panel D: draw measured length between extreme projection points
    draw_panel_D(axes[3], mask, pmin, pmax)

    # apply consistent limits and styling for thin journal-style lines
    for ax in axes:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # add panel labels (A)-(D) in upper left of each panel
    labels = ['(A)', '(B)', '(C)', '(D)']
    for ax, lab in zip(axes, labels):
        ax.text(0.02, 0.95, lab, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout(w_pad=1.0)
    out_root = Path(args.out)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    svg_out = str(out_root) + '.svg'
    png_out = str(out_root) + '.png'
    fig.savefig(svg_out, dpi=300, bbox_inches='tight')
    fig.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {svg_out}\nSaved: {png_out}')


if __name__ == '__main__':
    main()

