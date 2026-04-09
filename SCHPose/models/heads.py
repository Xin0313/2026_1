"""
Prediction heads for SCHPose.

5 decoupled prediction heads operating on FPN feature (P2, 1/4 resolution):
  (a) Seg/Det Head: segmentation logits + center offset
  (b) Discrete Code Head: coarse block logits + fine block logits
  (c) Continuous Offset Head: (u, v) within-block coordinates
  (d) Uncertainty Head: log(sigma^2) per pixel
  (e) Symmetry Class Head: symmetry equivalence class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel_size=3, padding=1):
    """Standard Conv-BN-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SegDetHead(nn.Module):
    """
    Segmentation and Detection head.

    Outputs:
      - seg_logits: [B, num_objects+1, H, W]  (background + num_objects)
      - center_offset: [B, 2, H, W]  (dx, dy) from pixel to object center
    """

    def __init__(self, in_channels, num_objects):
        super().__init__()
        self.num_objects = num_objects
        self.convs = nn.Sequential(
            conv_bn_relu(in_channels, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
        )
        self.seg_head = nn.Conv2d(256, num_objects + 1, kernel_size=1)
        self.center_head = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, feat):
        feat = self.convs(feat)
        seg_logits = self.seg_head(feat)      # [B, num_objects+1, H, W]
        center_offset = self.center_head(feat)  # [B, 2, H, W]
        return seg_logits, center_offset


class DiscreteCodeHead(nn.Module):
    """
    Discrete Code prediction head.

    Outputs per-pixel:
      - coarse_logits: [B, n_coarse, H, W]
      - fine_logits: [B, n_fine, H, W]
    """

    def __init__(self, in_channels, n_coarse=64, n_fine=16):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.convs = nn.Sequential(
            conv_bn_relu(in_channels, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
        )
        self.coarse_head = nn.Conv2d(256, n_coarse, kernel_size=1)
        self.fine_head = nn.Conv2d(256, n_fine, kernel_size=1)

    def forward(self, feat):
        feat = self.convs(feat)
        coarse_logits = self.coarse_head(feat)  # [B, n_coarse, H, W]
        fine_logits = self.fine_head(feat)       # [B, n_fine, H, W]
        return coarse_logits, fine_logits


class ContinuousOffsetHead(nn.Module):
    """
    Continuous within-block coordinate prediction head.

    Outputs per-pixel:
      - uv: [B, 2, H, W] in [0, 1] range (sigmoid activation)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(in_channels, 256),
            conv_bn_relu(256, 256),
        )
        self.uv_head = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, feat):
        feat = self.convs(feat)
        uv = torch.sigmoid(self.uv_head(feat))  # [B, 2, H, W] in [0, 1]
        return uv


class UncertaintyHead(nn.Module):
    """
    Per-pixel uncertainty (log variance) prediction head.

    Outputs:
      - log_var: [B, 1, H, W]  log(sigma^2) for each pixel
    """

    def __init__(self, in_channels):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(in_channels, 256),
            conv_bn_relu(256, 128),
        )
        self.log_var_head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, feat):
        feat = self.convs(feat)
        log_var = self.log_var_head(feat)   # [B, 1, H, W]
        # Clamp to avoid numerical issues
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return log_var


class SymmetryClassHead(nn.Module):
    """
    Symmetry equivalence class prediction head.

    Outputs per-pixel:
      - sym_logits: [B, n_sym, H, W]
    """

    def __init__(self, in_channels, n_sym=8):
        super().__init__()
        self.n_sym = n_sym
        self.convs = nn.Sequential(
            conv_bn_relu(in_channels, 256),
            conv_bn_relu(256, 256),
        )
        self.sym_head = nn.Conv2d(256, n_sym, kernel_size=1)

    def forward(self, feat):
        feat = self.convs(feat)
        sym_logits = self.sym_head(feat)   # [B, n_sym, H, W]
        return sym_logits


class SCHPoseHeads(nn.Module):
    """
    All 5 decoupled prediction heads for SCHPose.

    Args:
        in_channels: input feature channels (default 256 from FPN)
        num_objects: number of object categories
        n_coarse: number of coarse blocks
        n_fine: number of fine blocks per coarse block
        n_sym: number of symmetry equivalence classes
        use_uncertainty: whether to include uncertainty head
        use_symmetry: whether to include symmetry head
    """

    def __init__(
        self,
        in_channels=256,
        num_objects=21,
        n_coarse=64,
        n_fine=16,
        n_sym=8,
        use_uncertainty=True,
        use_symmetry=True,
    ):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.use_symmetry = use_symmetry

        self.seg_head = SegDetHead(in_channels, num_objects)
        self.code_head = DiscreteCodeHead(in_channels, n_coarse, n_fine)
        self.offset_head = ContinuousOffsetHead(in_channels)
        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(in_channels)
        if use_symmetry:
            self.symmetry_head = SymmetryClassHead(in_channels, n_sym)

    def forward(self, feat):
        """
        Args:
            feat: FPN P2 feature [B, in_channels, H/4, W/4]

        Returns:
            dict with keys:
              'seg_logits': [B, num_objects+1, H/4, W/4]
              'center_offset': [B, 2, H/4, W/4]
              'coarse_logits': [B, n_coarse, H/4, W/4]
              'fine_logits': [B, n_fine, H/4, W/4]
              'uv': [B, 2, H/4, W/4]
              'log_var': [B, 1, H/4, W/4]  (if use_uncertainty)
              'sym_logits': [B, n_sym, H/4, W/4]  (if use_symmetry)
        """
        seg_logits, center_offset = self.seg_head(feat)
        coarse_logits, fine_logits = self.code_head(feat)
        uv = self.offset_head(feat)

        outputs = {
            "seg_logits": seg_logits,
            "center_offset": center_offset,
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "uv": uv,
        }

        if self.use_uncertainty:
            outputs["log_var"] = self.uncertainty_head(feat)

        if self.use_symmetry:
            outputs["sym_logits"] = self.symmetry_head(feat)

        return outputs
