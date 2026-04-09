"""
Evaluation metrics for 6D pose estimation.

Implements:
  - ADD: Average Distance of Model Points
  - ADD-S: ADD with Symmetric matching
  - ADD(-S): ADD for non-symmetric, ADD-S for symmetric objects
  - ADD AUC: Area Under the ADD curve
  - VSD: Visible Surface Discrepancy
  - MSSD: Maximum Symmetry-aware Surface Distance
  - MSPD: Maximum Symmetry-aware Projection Distance
"""

import numpy as np
from collections import defaultdict


# ── Core Distance Metrics ─────────────────────────────────────────────────────

def compute_add(R_pred, t_pred, R_gt, t_gt, model_pts):
    """
    ADD: mean distance between transformed model points.

    Args:
        R_pred, R_gt: [3, 3] rotation matrices
        t_pred, t_gt: [3] translation vectors
        model_pts: [N, 3] 3D model points

    Returns:
        add_value: float (in same units as t)
    """
    pts_pred = (R_pred @ model_pts.T).T + t_pred
    pts_gt = (R_gt @ model_pts.T).T + t_gt
    return float(np.linalg.norm(pts_pred - pts_gt, axis=1).mean())


def compute_add_s(R_pred, t_pred, R_gt, t_gt, model_pts):
    """
    ADD-S: mean minimum distance (for symmetric objects).

    For each predicted point, find the nearest GT-transformed point.
    """
    pts_pred = (R_pred @ model_pts.T).T + t_pred   # [N, 3]
    pts_gt = (R_gt @ model_pts.T).T + t_gt         # [N, 3]
    diff = pts_pred[:, None, :] - pts_gt[None, :, :]  # [N, N, 3]
    dists = np.linalg.norm(diff, axis=2)              # [N, N]
    return float(dists.min(axis=1).mean())


def compute_add_sym(R_pred, t_pred, R_gt, t_gt, model_pts, sym_transforms=None):
    """
    Compute minimum ADD over symmetry-equivalent poses.

    Args:
        sym_transforms: list of [3, 3] rotation matrices

    Returns:
        min_add: float
    """
    base_add = compute_add(R_pred, t_pred, R_gt, t_gt, model_pts)
    if not sym_transforms:
        return base_add

    min_add = base_add
    for sym_R in sym_transforms:
        R_equiv = R_gt @ sym_R
        add_eq = compute_add(R_pred, t_pred, R_equiv, t_gt, model_pts)
        min_add = min(min_add, add_eq)

    return min_add


def compute_add_neg_s(R_pred, t_pred, R_gt, t_gt, model_pts, is_symmetric=False, sym_transforms=None):
    """
    ADD(-S): ADD for non-symmetric, ADD-S for symmetric objects.
    """
    if is_symmetric:
        return compute_add_s(R_pred, t_pred, R_gt, t_gt, model_pts)
    else:
        return compute_add(R_pred, t_pred, R_gt, t_gt, model_pts)


# ── AUC Computation ───────────────────────────────────────────────────────────

def compute_add_auc(add_values, max_threshold=0.1, n_bins=1000):
    """
    Compute ADD AUC (Area Under the Recall-Threshold Curve).

    Args:
        add_values: list or array of ADD values (normalized by object diameter)
        max_threshold: maximum threshold for AUC (default 0.1 = 10% of diameter)
        n_bins: number of threshold bins

    Returns:
        auc: float in [0, 1]
    """
    add_values = np.array(add_values)
    thresholds = np.linspace(0, max_threshold, n_bins)
    recalls = []
    for thresh in thresholds:
        recall = (add_values <= thresh).mean()
        recalls.append(recall)
    # AUC using trapezoidal rule, normalized
    auc = np.trapz(recalls, thresholds) / max_threshold
    return float(auc)


def add_accuracy_at_threshold(add_values, threshold):
    """
    Fraction of predictions with ADD < threshold.

    Args:
        add_values: array of raw ADD values (meters)
        threshold: float threshold in meters
    """
    return float((np.array(add_values) < threshold).mean())


# ── VSD ───────────────────────────────────────────────────────────────────────

def compute_vsd(R_pred, t_pred, R_gt, t_gt, depth_gt, K, model_pts,
                tau=0.02, delta=0.015):
    """
    VSD: Visible Surface Discrepancy.

    Measures discrepancy between predicted and GT depth maps for
    visible object surface, tolerant to small depth differences.

    Args:
        R_pred, t_pred: predicted pose
        R_gt, t_gt: ground-truth pose
        depth_gt: [H, W] ground-truth depth image
        K: [3, 3] camera intrinsics
        model_pts: [N, 3] model 3D points
        tau: misalignment tolerance (meters)
        delta: sensor noise tolerance

    Returns:
        vsd: float in [0, 1] (lower is better)
    """
    H, W = depth_gt.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def render_depth(R, t):
        """Simple z-buffer depth rendering."""
        depth = np.zeros((H, W), dtype=np.float32)
        pts_cam = (R @ model_pts.T).T + t

        # Project to image
        z = pts_cam[:, 2]
        valid = z > 0
        x = (pts_cam[valid, 0] / pts_cam[valid, 2] * fx + cx).astype(int)
        y = (pts_cam[valid, 1] / pts_cam[valid, 2] * fy + cy).astype(int)
        z_v = pts_cam[valid, 2]

        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x, y, z_v = x[in_bounds], y[in_bounds], z_v[in_bounds]

        for xi, yi, zi in sorted(zip(x, y, z_v), key=lambda t: -t[2]):
            if depth[yi, xi] == 0 or depth[yi, xi] > zi:
                depth[yi, xi] = zi

        return depth

    depth_pred = render_depth(R_pred, t_pred)
    depth_gt_render = render_depth(R_gt, t_gt)

    # Visible mask: pixels where GT render is visible
    vis_mask = (depth_gt_render > 0) & (np.abs(depth_gt - depth_gt_render) <= delta)

    if vis_mask.sum() == 0:
        return 1.0

    d_pred = depth_pred[vis_mask]
    d_gt = depth_gt_render[vis_mask]

    # VSD formula
    vsd_vals = np.abs(d_pred - d_gt) > tau
    # Handle pixels where prediction didn't render
    vsd_vals = vsd_vals | (d_pred == 0)

    return float(vsd_vals.mean())


# ── MSSD & MSPD ───────────────────────────────────────────────────────────────

def compute_mssd(R_pred, t_pred, R_gt, t_gt, model_pts, sym_transforms=None):
    """
    MSSD: Maximum Symmetry-aware Surface Distance.

    min over symmetry transforms of max over points of distance.
    """
    pts_pred = (R_pred @ model_pts.T).T + t_pred

    min_max_dist = float("inf")
    transforms = [np.eye(3)] + (sym_transforms or [])
    for sym_R in transforms:
        R_equiv = R_gt @ sym_R
        pts_gt_equiv = (R_equiv @ model_pts.T).T + t_gt
        max_dist = np.linalg.norm(pts_pred - pts_gt_equiv, axis=1).max()
        if max_dist < min_max_dist:
            min_max_dist = max_dist

    return float(min_max_dist)


def compute_mspd(R_pred, t_pred, R_gt, t_gt, model_pts, K, sym_transforms=None):
    """
    MSPD: Maximum Symmetry-aware Projection Distance.

    min over symmetry transforms of max over projected 2D point distances.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def project(R, t, pts):
        cam = (R @ pts.T).T + t
        z = cam[:, 2].clip(min=1e-6)
        x = cam[:, 0] / z * fx + cx
        y = cam[:, 1] / z * fy + cy
        return np.stack([x, y], axis=1)

    pts2d_pred = project(R_pred, t_pred, model_pts)

    min_max_dist = float("inf")
    transforms = [np.eye(3)] + (sym_transforms or [])
    for sym_R in transforms:
        R_equiv = R_gt @ sym_R
        pts2d_gt = project(R_equiv, t_gt, model_pts)
        max_dist = np.linalg.norm(pts2d_pred - pts2d_gt, axis=1).max()
        if max_dist < min_max_dist:
            min_max_dist = max_dist

    return float(min_max_dist)


# ── Aggregated Metrics ────────────────────────────────────────────────────────

class PoseMetrics:
    """
    Aggregated pose evaluation metrics tracker.

    Usage:
        metrics = PoseMetrics(obj_diameters, symmetric_obj_ids)
        for each prediction:
            metrics.update(obj_id, R_pred, t_pred, R_gt, t_gt, model_pts)
        results = metrics.summarize()
    """

    def __init__(self, obj_diameters=None, symmetric_obj_ids=None,
                 add_threshold=0.1, auc_max_threshold=0.1):
        """
        Args:
            obj_diameters: dict obj_id -> diameter (meters)
            symmetric_obj_ids: set of symmetric obj_ids
            add_threshold: threshold as fraction of diameter for ADD accuracy
            auc_max_threshold: max threshold for AUC (normalized)
        """
        self.obj_diameters = obj_diameters or {}
        self.symmetric_obj_ids = set(symmetric_obj_ids or [])
        self.add_threshold = add_threshold
        self.auc_max_threshold = auc_max_threshold

        self._records = defaultdict(list)  # obj_id -> list of ADD values (normalized)

    def update(self, obj_id, R_pred, t_pred, R_gt, t_gt, model_pts,
               sym_transforms=None):
        """Add one prediction result."""
        is_sym = obj_id in self.symmetric_obj_ids
        diam = self.obj_diameters.get(obj_id, 1.0)

        add_val = compute_add_neg_s(R_pred, t_pred, R_gt, t_gt, model_pts,
                                    is_symmetric=is_sym,
                                    sym_transforms=sym_transforms)
        add_normalized = add_val / diam
        self._records[obj_id].append(add_normalized)

    def summarize(self):
        """
        Compute per-object and mean metrics.

        Returns:
            dict with per-object and 'mean' entries.
        """
        results = {}
        all_adds = []

        for obj_id, adds in self._records.items():
            adds_arr = np.array(adds)
            acc = add_accuracy_at_threshold(adds_arr, self.add_threshold)
            auc = compute_add_auc(adds_arr, max_threshold=self.auc_max_threshold)
            results[obj_id] = {
                "add_accuracy": acc,
                "add_auc": auc,
                "n_samples": len(adds),
                "mean_add": float(adds_arr.mean()),
            }
            all_adds.extend(adds)

        if all_adds:
            all_arr = np.array(all_adds)
            results["mean"] = {
                "add_accuracy": add_accuracy_at_threshold(all_arr, self.add_threshold),
                "add_auc": compute_add_auc(all_arr, max_threshold=self.auc_max_threshold),
                "n_samples": len(all_adds),
                "mean_add": float(all_arr.mean()),
            }

        return results

    def reset(self):
        self._records.clear()
