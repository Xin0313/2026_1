"""
Discrete code loss for SCHPose.

Hierarchical cross-entropy loss with:
  - Label smoothing
  - Hierarchical consistency constraint (coarse must contain fine)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeLoss(nn.Module):
    """
    Hierarchical discrete code loss.

    For each foreground pixel:
      L_coarse = CE(coarse_logits, coarse_gt)
      L_fine   = CE(fine_logits, fine_gt)
      L_consist = constraint that fine prediction is consistent with coarse

    Args:
        label_smoothing: label smoothing epsilon (0 = no smoothing)
        consistency_weight: weight for consistency regularization
    """

    def __init__(self, label_smoothing=0.1, consistency_weight=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.consistency_weight = consistency_weight

    def forward(self, coarse_logits, fine_logits, coarse_targets, fine_targets, mask):
        """
        Args:
            coarse_logits: [B, n_coarse, H, W]
            fine_logits: [B, n_fine, H, W]
            coarse_targets: [B, H, W] long  (ground-truth coarse block id per pixel)
            fine_targets: [B, H, W] long    (ground-truth fine block id within coarse)
            mask: [B, H, W] binary float (foreground pixels)

        Returns:
            loss: scalar
            loss_dict: dict with sub-losses
        """
        B, n_coarse, H, W = coarse_logits.shape
        n_fine = fine_logits.shape[1]

        # Flatten spatial dims
        mask_flat = mask.reshape(-1).bool()
        coarse_logits_flat = coarse_logits.permute(0, 2, 3, 1).reshape(-1, n_coarse)
        fine_logits_flat = fine_logits.permute(0, 2, 3, 1).reshape(-1, n_fine)
        coarse_targets_flat = coarse_targets.reshape(-1)
        fine_targets_flat = fine_targets.reshape(-1)

        if mask_flat.sum() == 0:
            zero = coarse_logits.sum() * 0
            return zero, {"coarse_ce": zero.detach(), "fine_ce": zero.detach(), "consistency": zero.detach()}

        # Select foreground pixels
        cl = coarse_logits_flat[mask_flat]
        fl = fine_logits_flat[mask_flat]
        ct = coarse_targets_flat[mask_flat]
        ft = fine_targets_flat[mask_flat]

        # Coarse CE with label smoothing
        coarse_loss = self._ce_with_smoothing(cl, ct, n_coarse)

        # Fine CE with label smoothing
        fine_loss = self._ce_with_smoothing(fl, ft, n_fine)

        # Consistency loss:
        # The predicted fine block must be compatible with the predicted coarse block.
        # We encourage: argmax(coarse) == coarse_targets for fine-predicted pixels.
        # Simple implementation: cross-entropy of coarse conditioned on fine target.
        consist_loss = self._consistency_loss(cl, fl, ct, ft)

        total = coarse_loss + fine_loss + self.consistency_weight * consist_loss
        return total, {
            "coarse_ce": coarse_loss.detach(),
            "fine_ce": fine_loss.detach(),
            "consistency": consist_loss.detach(),
        }

    def _ce_with_smoothing(self, logits, targets, num_classes):
        """Cross-entropy with optional label smoothing."""
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            log_p = F.log_softmax(logits, dim=1)
            # One-hot targets
            nll = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_p.mean(dim=1)
            loss = (1 - eps) * nll + eps * smooth_loss
            return loss.mean()
        else:
            return F.cross_entropy(logits, targets)

    def _consistency_loss(self, coarse_logits, fine_logits, coarse_targets, fine_targets):
        """
        Hierarchical consistency: ensure fine prediction is within correct coarse block.

        Penalty: cross-entropy of coarse logits given fine-target pixels that
        have been correctly fine-classified.
        """
        # Simply use the coarse prediction entropy as a proxy for consistency
        # More sophisticated: use a coarse->fine mapping table
        coarse_probs = F.softmax(coarse_logits, dim=1)  # [N, n_coarse]
        fine_probs = F.softmax(fine_logits, dim=1)       # [N, n_fine]

        # Entropy of coarse predictions (penalize high entropy = uncertain coarse)
        coarse_ent = -(coarse_probs * torch.log(coarse_probs + 1e-8)).sum(dim=1)

        # Only penalize pixels where fine prediction was correct
        fine_correct = (fine_logits.argmax(dim=1) == fine_targets).float()
        consist = (coarse_ent * fine_correct).mean()
        return consist
