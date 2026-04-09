"""
Symmetry-aware pose loss for SCHPose.

Supports:
  - ADD: Average Distance of Model Points
  - ADD-S: Average Distance with Symmetric matching
  - Symmetry-aware minimum error (over equivalence class)

Reference: BOP Challenge metrics.
"""

import torch
import torch.nn as nn
import numpy as np


class PoseLoss(nn.Module):
    """
    Symmetry-aware pose loss.

    For non-symmetric objects: ADD loss
    For symmetric objects: ADD-S (minimum distance matching)
    With symmetry transforms: minimum over all equivalent poses

    Args:
        loss_type: 'add' | 'add_s' | 'sym_min'
        max_sym: max number of symmetry transforms to enumerate
    """

    def __init__(self, loss_type="sym_min", max_sym=16):
        super().__init__()
        self.loss_type = loss_type
        self.max_sym = max_sym

    def forward(self, R_pred, t_pred, R_gt, t_gt, model_pts, is_symmetric=False, sym_transforms=None):
        """
        Compute pose loss.

        Args:
            R_pred: [3, 3] or [B, 3, 3] predicted rotation
            t_pred: [3] or [B, 3] predicted translation
            R_gt: [3, 3] or [B, 3, 3] ground-truth rotation
            t_gt: [3] or [B, 3] ground-truth translation
            model_pts: [N, 3] or [B, N, 3] model 3D points
            is_symmetric: bool or [B] bool
            sym_transforms: list of [3,3] rotation matrices (optional)

        Returns:
            loss: scalar
        """
        batched = R_pred.dim() == 3
        if not batched:
            R_pred = R_pred.unsqueeze(0)
            t_pred = t_pred.unsqueeze(0)
            R_gt = R_gt.unsqueeze(0)
            t_gt = t_gt.unsqueeze(0)
            if model_pts.dim() == 2:
                model_pts = model_pts.unsqueeze(0)
            if isinstance(is_symmetric, bool):
                is_symmetric = [is_symmetric]

        B = R_pred.shape[0]
        losses = []

        for b in range(B):
            sym_b = is_symmetric[b] if isinstance(is_symmetric, (list, tuple)) else is_symmetric

            if self.loss_type == "add_s" or (self.loss_type == "sym_min" and sym_b):
                loss_b = self._add_s_loss(
                    R_pred[b], t_pred[b], R_gt[b], t_gt[b], model_pts[b]
                )
            elif self.loss_type == "sym_min" and sym_transforms:
                loss_b = self._sym_min_loss(
                    R_pred[b], t_pred[b], R_gt[b], t_gt[b], model_pts[b], sym_transforms
                )
            else:
                loss_b = self._add_loss(
                    R_pred[b], t_pred[b], R_gt[b], t_gt[b], model_pts[b]
                )

            losses.append(loss_b)

        return torch.stack(losses).mean()

    def _transform_pts(self, R, t, pts):
        """Apply rotation R and translation t to points. pts: [N,3]"""
        return (R @ pts.T).T + t.unsqueeze(0)

    def _add_loss(self, R_pred, t_pred, R_gt, t_gt, pts):
        """ADD: mean distance between transformed model points."""
        pts_pred = self._transform_pts(R_pred, t_pred, pts)
        pts_gt = self._transform_pts(R_gt, t_gt, pts)
        return (pts_pred - pts_gt).norm(dim=1).mean()

    def _add_s_loss(self, R_pred, t_pred, R_gt, t_gt, pts):
        """
        ADD-S: mean of minimum distances (symmetric objects).
        For each predicted point, find the nearest GT point.
        """
        pts_pred = self._transform_pts(R_pred, t_pred, pts)  # [N, 3]
        pts_gt = self._transform_pts(R_gt, t_gt, pts)        # [N, 3]

        # Distance matrix [N, N]
        diff = pts_pred.unsqueeze(1) - pts_gt.unsqueeze(0)   # [N, N, 3]
        dists = diff.norm(dim=2)                              # [N, N]

        # Min over GT points for each pred point
        min_dists = dists.min(dim=1).values  # [N]
        return min_dists.mean()

    def _sym_min_loss(self, R_pred, t_pred, R_gt, t_gt, pts, sym_transforms):
        """
        Minimum ADD over all symmetry-equivalent poses.
        """
        base_add = self._add_loss(R_pred, t_pred, R_gt, t_gt, pts)
        min_loss = base_add

        for sym_R in sym_transforms:
            if not isinstance(sym_R, torch.Tensor):
                sym_R = torch.from_numpy(sym_R).to(device=R_pred.device, dtype=R_pred.dtype)
            R_equiv = R_gt @ sym_R
            add_equiv = self._add_loss(R_pred, t_pred, R_equiv, t_gt, pts)
            if add_equiv < min_loss:
                min_loss = add_equiv

        return min_loss


def compute_add(R_pred, t_pred, R_gt, t_gt, model_pts):
    """
    Compute ADD metric (numpy).

    Args:
        R_pred, R_gt: [3, 3]
        t_pred, t_gt: [3]
        model_pts: [N, 3]

    Returns:
        add_value: scalar float
    """
    pts_pred = (R_pred @ model_pts.T).T + t_pred
    pts_gt = (R_gt @ model_pts.T).T + t_gt
    return np.linalg.norm(pts_pred - pts_gt, axis=1).mean()


def compute_add_s(R_pred, t_pred, R_gt, t_gt, model_pts):
    """
    Compute ADD-S metric (numpy, for symmetric objects).
    """
    pts_pred = (R_pred @ model_pts.T).T + t_pred
    pts_gt = (R_gt @ model_pts.T).T + t_gt
    diff = pts_pred[:, None] - pts_gt[None, :]   # [N, N, 3]
    dists = np.linalg.norm(diff, axis=2)         # [N, N]
    return dists.min(axis=1).mean()
