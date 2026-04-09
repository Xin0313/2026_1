"""
Visualization utilities for SCHPose.

Provides:
  - draw_pose: draw 3D bounding box / axes projected onto image
  - draw_segmentation: overlay segmentation mask
  - draw_code_heatmap: visualize discrete code predictions
  - overlay_mask: overlay binary mask with color
"""

import numpy as np
import cv2


def draw_pose(image, R, t, K, model_pts=None, color=(0, 255, 0), thickness=2):
    """
    Draw object pose (3D axes or bounding box projected to 2D).

    Args:
        image: np.ndarray [H, W, 3] uint8
        R: [3, 3] rotation matrix
        t: [3] translation vector
        K: [3, 3] camera intrinsics
        model_pts: [N, 3] optional model 3D points for bounding box
        color: RGB color tuple
        thickness: line thickness

    Returns:
        vis: np.ndarray [H, W, 3] with visualization
    """
    vis = image.copy()
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def project_pt(pt3d):
        pt_cam = R @ pt3d + t
        if pt_cam[2] <= 0:
            return None
        x = int(pt_cam[0] / pt_cam[2] * fx + cx)
        y = int(pt_cam[1] / pt_cam[2] * fy + cy)
        return (x, y)

    # Draw coordinate axes
    if t is not None:
        axis_len = 0.05  # 5 cm
        origin = project_pt(np.zeros(3))
        if origin is not None:
            for axis_vec, axis_color in [
                (np.array([axis_len, 0, 0]), (0, 0, 255)),   # X: blue
                (np.array([0, axis_len, 0]), (0, 255, 0)),   # Y: green
                (np.array([0, 0, axis_len]), (255, 0, 0)),   # Z: red
            ]:
                end = project_pt(axis_vec)
                if end is not None:
                    cv2.arrowedLine(vis_bgr, origin, end, axis_color, thickness, tipLength=0.3)

    # Draw model point cloud projection
    if model_pts is not None and len(model_pts) > 0:
        pts_cam = (R @ model_pts.T).T + t
        valid = pts_cam[:, 2] > 0
        if valid.sum() > 0:
            pts_valid = pts_cam[valid]
            xs = (pts_valid[:, 0] / pts_valid[:, 2] * fx + cx).astype(int)
            ys = (pts_valid[:, 1] / pts_valid[:, 2] * fy + cy).astype(int)
            H, W = image.shape[:2]
            in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            bgr_color = (color[2], color[1], color[0])
            for xi, yi in zip(xs[in_bounds], ys[in_bounds]):
                cv2.circle(vis_bgr, (xi, yi), 1, bgr_color, -1)

    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


def draw_segmentation(image, seg_mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay segmentation mask on image.

    Args:
        image: [H, W, 3] uint8 RGB
        seg_mask: [H, W] binary uint8 (foreground=1) or float [0,1]
        color: RGB color for overlay
        alpha: transparency of overlay

    Returns:
        vis: [H, W, 3] uint8
    """
    vis = image.copy().astype(np.float32)
    mask = (seg_mask > 0).astype(np.float32)
    overlay = np.zeros_like(vis)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    mask_3c = mask[:, :, None]
    vis = vis * (1 - alpha * mask_3c) + overlay * alpha * mask_3c
    return vis.clip(0, 255).astype(np.uint8)


def draw_code_heatmap(coarse_logits, fine_logits=None, colormap=cv2.COLORMAP_JET):
    """
    Visualize discrete code predictions as heatmaps.

    Args:
        coarse_logits: [n_coarse, H, W] or [H, W] (already argmaxed)
        fine_logits: optional [n_fine, H, W]
        colormap: OpenCV colormap

    Returns:
        heatmap: [H, W, 3] uint8 RGB
        (fine_heatmap: [H, W, 3] if fine_logits provided)
    """
    if coarse_logits.ndim == 3:
        coarse_ids = coarse_logits.argmax(axis=0)  # [H, W]
    else:
        coarse_ids = coarse_logits

    n_coarse = coarse_ids.max() + 1 if coarse_ids.max() > 0 else 1
    normalized = (coarse_ids.astype(np.float32) / n_coarse * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(normalized, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    if fine_logits is not None:
        if fine_logits.ndim == 3:
            fine_ids = fine_logits.argmax(axis=0)
        else:
            fine_ids = fine_logits
        n_fine = fine_ids.max() + 1 if fine_ids.max() > 0 else 1
        fine_norm = (fine_ids.astype(np.float32) / n_fine * 255).astype(np.uint8)
        fine_bgr = cv2.applyColorMap(fine_norm, colormap)
        fine_rgb = cv2.cvtColor(fine_bgr, cv2.COLOR_BGR2RGB)
        return heatmap_rgb, fine_rgb

    return heatmap_rgb


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.4, contour=True):
    """
    Overlay binary mask with optional contour.

    Args:
        image: [H, W, 3] uint8 RGB
        mask: [H, W] uint8 binary
        color: RGB tuple
        alpha: blend alpha
        contour: whether to draw mask contour

    Returns:
        vis: [H, W, 3] uint8
    """
    vis = image.copy()
    mask_bool = mask > 0

    colored = np.zeros_like(vis, dtype=np.float32)
    colored[:] = color
    vis = vis.astype(np.float32)
    vis[mask_bool] = vis[mask_bool] * (1 - alpha) + colored[mask_bool] * alpha

    if contour:
        vis_bgr = cv2.cvtColor(vis.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bgr_color = (color[2], color[1], color[0])
        cv2.drawContours(vis_bgr, contours, -1, bgr_color, 2)
        vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    return vis.clip(0, 255).astype(np.uint8)


def make_grid(images, nrow=4, padding=2):
    """
    Arrange list of images into a grid.

    Args:
        images: list of [H, W, 3] uint8 arrays (all same size)
        nrow: images per row
        padding: pixel padding between images

    Returns:
        grid: np.ndarray [H_total, W_total, 3]
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    H, W, C = images[0].shape
    ncols = nrow
    nrows = (len(images) + ncols - 1) // ncols

    total_h = nrows * H + (nrows + 1) * padding
    total_w = ncols * W + (ncols + 1) * padding
    grid = np.ones((total_h, total_w, C), dtype=np.uint8) * 128

    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        y = row * (H + padding) + padding
        x = col * (W + padding) + padding
        h, w = img.shape[:2]
        grid[y:y + h, x:x + w] = img

    return grid
