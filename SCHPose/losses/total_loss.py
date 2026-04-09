"""
Total loss aggregator for SCHPose.

L_total = w_seg * L_seg
        + w_code * L_code
        + w_uv * L_uv
        + w_unc * L_unc
        + w_pose * L_pose      (Phase 2 only)
        + w_consist * L_consist

Default weights: 2.0, 1.0, 5.0, 1.0, 10.0, 0.5
"""

import torch
import torch.nn as nn

from .seg_loss import SegLoss
from .code_loss import CodeLoss
from .offset_loss import OffsetLoss
from .uncertainty_loss import UncertaintyLoss
from .pose_loss import PoseLoss


class TotalLoss(nn.Module):
    """
    Aggregated total loss for SCHPose training.

    Args:
        cfg: config dict with 'loss_weights' key
        phase: training phase (1 or 2)
    """

    def __init__(self, cfg, phase=1):
        super().__init__()
        self.phase = phase

        weights = cfg.get("loss_weights", {})
        self.w_seg = weights.get("seg", 2.0)
        self.w_code = weights.get("code", 1.0)
        self.w_uv = weights.get("uv", 5.0)
        self.w_unc = weights.get("uncertainty", 1.0)
        self.w_pose = weights.get("pose", 10.0)
        self.w_consist = weights.get("consistency", 0.5)

        self.seg_loss = SegLoss()
        self.code_loss = CodeLoss()
        self.offset_loss = OffsetLoss()
        self.uncertainty_loss = UncertaintyLoss()
        self.pose_loss = PoseLoss(loss_type="sym_min")

    def set_phase(self, phase):
        """Switch training phase."""
        self.phase = phase

    def forward(self, predictions, targets):
        """
        Compute total loss.

        Args:
            predictions: dict from model forward pass, containing:
                'head_outputs': dict with 'seg_logits', 'coarse_logits', 'fine_logits',
                                'uv', 'log_var', 'sym_logits'
                'poses': list of (R, t) from PnP (Phase 2)
            targets: dict with:
                'seg_targets': [B, H, W] long segmentation labels
                'coarse_targets': [B, H, W] long coarse block labels
                'fine_targets': [B, H, W] long fine block labels
                'uv_targets': [B, 2, H, W] float UV coordinates
                'mask': [B, H, W] float foreground mask
                'R_gt': [B, 3, 3] ground-truth rotations
                't_gt': [B, 3] ground-truth translations
                'model_pts': [B, N, 3] model 3D points
                'is_symmetric': [B] bool
                'sym_transforms': list of lists of rotation matrices (per batch)
                'reproj_errors': [B, H, W] optional reprojection errors

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        head_out = predictions["head_outputs"]
        mask = targets["mask"]

        # Downsample masks/targets to prediction resolution (1/4 original)
        mask_small = self._downsample_mask(mask, head_out["seg_logits"])

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=mask.device, requires_grad=True)

        # 1. Segmentation loss
        seg_targets_small = self._downsample_labels(
            targets["seg_targets"], head_out["seg_logits"]
        )
        l_seg, seg_dict = self.seg_loss(head_out["seg_logits"], seg_targets_small)
        total_loss = total_loss + self.w_seg * l_seg
        loss_dict["seg"] = l_seg.detach()
        loss_dict.update({f"seg_{k}": v for k, v in seg_dict.items()})

        # 2. Discrete code loss
        coarse_targets_small = self._downsample_labels(
            targets["coarse_targets"], head_out["coarse_logits"]
        )
        fine_targets_small = self._downsample_labels(
            targets["fine_targets"], head_out["fine_logits"]
        )
        l_code, code_dict = self.code_loss(
            head_out["coarse_logits"],
            head_out["fine_logits"],
            coarse_targets_small,
            fine_targets_small,
            mask_small,
        )
        total_loss = total_loss + self.w_code * l_code
        loss_dict["code"] = l_code.detach()
        loss_dict.update({f"code_{k}": v for k, v in code_dict.items()})

        # 3. Continuous offset loss
        uv_targets_small = self._downsample_uv(targets["uv_targets"], head_out["uv"])
        l_uv = self.offset_loss(head_out["uv"], uv_targets_small, mask_small)
        total_loss = total_loss + self.w_uv * l_uv
        loss_dict["uv"] = l_uv.detach()

        # 4. Uncertainty loss
        if "log_var" in head_out:
            reproj_errors = targets.get("reproj_errors", None)
            if reproj_errors is not None:
                reproj_small = self._downsample_reproj(reproj_errors, head_out["log_var"])
                l_unc = self.uncertainty_loss(head_out["log_var"], reproj_small, mask_small)
            else:
                l_unc = self.uncertainty_loss.forward_simple(head_out["log_var"], mask_small)
            total_loss = total_loss + self.w_unc * l_unc
            loss_dict["uncertainty"] = l_unc.detach()

        # 5. Pose loss (Phase 2 only)
        if self.phase == 2 and "poses" in predictions and len(predictions["poses"]) > 0:
            l_pose = self._compute_pose_loss(predictions, targets)
            total_loss = total_loss + self.w_pose * l_pose
            loss_dict["pose"] = l_pose.detach()

        # 6. Consistency loss (from code_loss dict)
        if "code_consistency" in loss_dict:
            # Already weighted within code_loss; add explicit consistency term
            l_consist = loss_dict["code_consistency"]
            # Apply extra weight (consistency already included in code loss)
            # Here we track it separately in loss_dict
            loss_dict["consistency"] = l_consist

        loss_dict["total"] = total_loss.detach()
        return total_loss, loss_dict

    def _compute_pose_loss(self, predictions, targets):
        """Compute pose loss over batch."""
        poses = predictions["poses"]
        R_gt = targets["R_gt"]
        t_gt = targets["t_gt"]
        model_pts = targets.get("model_pts", None)
        is_symmetric = targets.get("is_symmetric", [False] * len(poses))
        sym_transforms = targets.get("sym_transforms", None)

        if model_pts is None or len(poses) == 0:
            return torch.tensor(0.0, device=R_gt.device)

        pose_losses = []
        for b, (R_pred, t_pred) in enumerate(poses):
            sym_t = sym_transforms[b] if sym_transforms else None
            sym_b = is_symmetric[b] if isinstance(is_symmetric, (list, tuple)) else is_symmetric
            l_b = self.pose_loss(
                R_pred, t_pred, R_gt[b], t_gt[b],
                model_pts[b],
                is_symmetric=sym_b,
                sym_transforms=sym_t,
            )
            pose_losses.append(l_b)

        return torch.stack(pose_losses).mean()

    def _downsample_mask(self, mask, ref_tensor):
        """Downsample mask to match reference spatial size."""
        import torch.nn.functional as F
        H, W = ref_tensor.shape[-2:]
        if mask.shape[-2:] == (H, W):
            return mask
        return F.interpolate(
            mask.unsqueeze(1).float(), size=(H, W), mode="nearest"
        ).squeeze(1)

    def _downsample_labels(self, labels, ref_tensor):
        """Downsample long label map to match reference spatial size."""
        import torch.nn.functional as F
        H, W = ref_tensor.shape[-2:]
        if labels.shape[-2:] == (H, W):
            return labels
        return F.interpolate(
            labels.unsqueeze(1).float(), size=(H, W), mode="nearest"
        ).squeeze(1).long()

    def _downsample_uv(self, uv, ref_tensor):
        """Downsample UV map to match reference spatial size."""
        import torch.nn.functional as F
        H, W = ref_tensor.shape[-2:]
        if uv.shape[-2:] == (H, W):
            return uv
        return F.interpolate(uv, size=(H, W), mode="bilinear", align_corners=False)

    def _downsample_reproj(self, reproj, ref_tensor):
        """Downsample reprojection error map."""
        import torch.nn.functional as F
        H, W = ref_tensor.shape[-2:]
        if reproj.shape[-2:] == (H, W):
            return reproj
        return F.interpolate(
            reproj.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)
