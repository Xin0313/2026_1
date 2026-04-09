"""
Uncertainty (log-variance) loss for SCHPose.

Uses Negative Log-Likelihood (NLL) formulation:
  L_unc = sum_i [ ||e_i||^2 / sigma_i^2 + log(sigma_i^2) ]

where e_i is the 2D reprojection error and sigma_i^2 = exp(log_var_i).

This loss incentivizes the network to predict high uncertainty (large sigma)
for pixels with high reprojection error, and low uncertainty for reliable pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyLoss(nn.Module):
    """
    NLL-based uncertainty loss.

    L_unc = mean_i [ ||e_i||^2 * exp(-log_var_i) + log_var_i ]

    Args:
        min_log_var: clamp log_var from below to prevent collapse
        max_log_var: clamp log_var from above
    """

    def __init__(self, min_log_var=-10.0, max_log_var=10.0):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, log_var, reproj_errors, mask):
        """
        Args:
            log_var: [B, 1, H, W] predicted log(sigma^2)
            reproj_errors: [B, H, W] squared reprojection errors per pixel
                           (computed from decoded 3D points + current pose estimate)
            mask: [B, H, W] binary foreground mask

        Returns:
            loss: scalar
        """
        mask_bool = mask.bool()

        if mask_bool.sum() == 0:
            return log_var.sum() * 0

        log_var_flat = log_var.squeeze(1)           # [B, H, W]
        log_var_fg = log_var_flat[mask_bool]         # [N_fg]
        err_fg = reproj_errors[mask_bool]            # [N_fg]

        # Clamp for stability
        log_var_fg = torch.clamp(log_var_fg, self.min_log_var, self.max_log_var)

        # NLL: err/sigma^2 + log(sigma^2)
        nll = err_fg * torch.exp(-log_var_fg) + log_var_fg

        return nll.mean()

    def forward_simple(self, log_var, mask):
        """
        Simplified version: just regularize log_var toward 0 on foreground.
        Used when reprojection errors are not available (Phase 1).

        Args:
            log_var: [B, 1, H, W]
            mask: [B, H, W]

        Returns:
            loss: scalar
        """
        mask_bool = mask.bool()
        if mask_bool.sum() == 0:
            return log_var.sum() * 0

        log_var_flat = log_var.squeeze(1)
        log_var_fg = log_var_flat[mask_bool]
        log_var_fg = torch.clamp(log_var_fg, self.min_log_var, self.max_log_var)

        # Penalize extreme values: push toward small positive uncertainty
        reg = log_var_fg ** 2
        return reg.mean()
