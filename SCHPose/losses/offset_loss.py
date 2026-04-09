"""
Continuous offset (u,v) regression loss for SCHPose.

Uses Charbonnier loss (smooth L1 variant):
  rho(x) = sqrt(x^2 + epsilon^2)

Only computed over foreground pixels.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: rho(x) = sqrt(x^2 + eps^2)
    Differentiable everywhere, more robust than L2, smoother than L1.

    Args:
        epsilon: smoothing parameter
        reduction: 'mean' or 'sum'
    """

    def __init__(self, epsilon=1e-3, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [..., 2] or any shape
            target: same shape as pred
            mask: optional binary mask (same spatial shape, broadcasting OK)

        Returns:
            loss: scalar
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon ** 2)

        if mask is not None:
            if mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask
            n = mask.sum().clamp(min=1)
            if self.reduction == "mean":
                return loss.sum() / n
            return loss.sum()

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class OffsetLoss(nn.Module):
    """
    Loss for continuous (u, v) within-block coordinate regression.

    Args:
        epsilon: Charbonnier smoothing
    """

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(epsilon=epsilon)

    def forward(self, uv_pred, uv_target, mask):
        """
        Args:
            uv_pred: [B, 2, H, W] predicted (u, v) in [0, 1]
            uv_target: [B, 2, H, W] ground-truth (u, v)
            mask: [B, H, W] binary foreground mask

        Returns:
            loss: scalar
        """
        # Reshape for masked selection
        B, _, H, W = uv_pred.shape
        mask_flat = mask.reshape(-1).bool()

        uv_pred_flat = uv_pred.permute(0, 2, 3, 1).reshape(-1, 2)
        uv_target_flat = uv_target.permute(0, 2, 3, 1).reshape(-1, 2)

        if mask_flat.sum() == 0:
            return uv_pred.sum() * 0

        uv_pred_fg = uv_pred_flat[mask_flat]
        uv_target_fg = uv_target_flat[mask_flat]

        return self.charbonnier(uv_pred_fg, uv_target_fg)
