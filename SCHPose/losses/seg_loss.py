"""
Segmentation loss for SCHPose.
Combines Focal Loss + Dice Loss for robust foreground segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense prediction.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (default 2.0)
        alpha: class balance weight (default 0.25)
        reduction: 'mean' or 'sum'
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] or [B, C] unnormalized
            targets: [B, H, W] or [B] long integers

        Returns:
            loss: scalar
        """
        B = logits.shape[0]
        C = logits.shape[1]

        if logits.dim() == 4:
            # Dense prediction
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
            targets_flat = targets.reshape(-1)
        else:
            logits_flat = logits
            targets_flat = targets

        log_p = F.log_softmax(logits_flat, dim=1)
        p = torch.exp(log_p)

        # Gather probabilities for correct class
        log_pt = log_p.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting (binary: background vs foreground)
        alpha_t = torch.where(targets_flat > 0,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))

        loss = -alpha_t * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary/multi-class segmentation.

    Args:
        smooth: smoothing factor to avoid division by zero
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] unnormalized
            targets: [B, H, W] long

        Returns:
            loss: scalar
        """
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # One-hot encode targets
        targets_oh = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        # Dice per class (exclude background class 0)
        dice_loss = 0.0
        n_classes = 0
        for c in range(1, C):
            p = probs[:, c].reshape(-1)
            t = targets_oh[:, c].reshape(-1)
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)
            n_classes += 1

        if n_classes > 0:
            return dice_loss / n_classes
        return dice_loss


class SegLoss(nn.Module):
    """
    Combined segmentation loss = Focal + Dice.

    Args:
        focal_weight: weight for Focal Loss
        dice_weight: weight for Dice Loss
        gamma: Focal Loss gamma
        alpha: Focal Loss alpha
    """

    def __init__(self, focal_weight=1.0, dice_weight=1.0, gamma=2.0, alpha=0.25):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, seg_logits, seg_targets):
        """
        Args:
            seg_logits: [B, num_objects+1, H, W]
            seg_targets: [B, H, W] long (0=background, 1..C=object classes)

        Returns:
            loss: scalar
            loss_dict: dict with 'focal' and 'dice' sub-losses
        """
        focal_loss = self.focal(seg_logits, seg_targets)
        dice_loss = self.dice(seg_logits, seg_targets)
        total = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total, {"focal": focal_loss.detach(), "dice": dice_loss.detach()}
