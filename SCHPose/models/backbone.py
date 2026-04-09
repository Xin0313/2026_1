"""
Backbone: ConvNeXt-Tiny + Feature Pyramid Network (FPN) for SCHPose.

Architecture:
  - ConvNeXt-Tiny pretrained on ImageNet
  - Extract 4 stage features {C2, C3, C4, C5}
  - FPN top-down pathway -> {P2, P3, P4, P5} with unified channels=256
  - Output P2 (1/4 resolution) as primary dense prediction feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class FPNBlock(nn.Module):
    """Single FPN level: lateral conv + top-down upsampling."""

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat, top_down=None):
        lateral = self.lateral(feat)
        if top_down is not None:
            top_down_up = F.interpolate(
                top_down, size=lateral.shape[-2:], mode="nearest"
            )
            lateral = lateral + top_down_up
        return self.output(lateral)


class ConvNeXtFPN(nn.Module):
    """
    ConvNeXt-Tiny + FPN backbone.

    Args:
        pretrained: load ImageNet pretrained weights
        out_channels: FPN output channels (default 256)
        out_level: which FPN level to use as main output ('P2' = 1/4 resolution)
    """

    def __init__(self, pretrained=True, out_channels=256, out_level="P2"):
        super().__init__()
        self.out_channels = out_channels
        self.out_level = out_level

        # Load ConvNeXt-Tiny backbone
        backbone = tv_models.convnext_tiny(
            weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # ConvNeXt-Tiny feature dimensions at each stage:
        #   Stage 0 (C2, stride 4):  96 channels
        #   Stage 1 (C3, stride 8):  192 channels
        #   Stage 2 (C4, stride 16): 384 channels
        #   Stage 3 (C5, stride 32): 768 channels
        self.stage_channels = [96, 192, 384, 768]

        # Extract stages from ConvNeXt features
        # torchvision ConvNeXt: features is nn.Sequential
        # features[0]: stem (downsample x4)
        # features[1]: stage 0
        # features[2]: downsample
        # features[3]: stage 1
        # features[4]: downsample
        # features[5]: stage 2
        # features[6]: downsample
        # features[7]: stage 3
        features = backbone.features
        self.stem = features[0]
        self.layer1 = features[1]   # C2: stride 4, 96ch
        self.down2 = features[2]
        self.layer2 = features[3]   # C3: stride 8, 192ch
        self.down3 = features[4]
        self.layer3 = features[5]   # C4: stride 16, 384ch
        self.down4 = features[6]
        self.layer4 = features[7]   # C5: stride 32, 768ch

        # FPN layers
        self.fpn_p5 = FPNBlock(self.stage_channels[3], out_channels)
        self.fpn_p4 = FPNBlock(self.stage_channels[2], out_channels)
        self.fpn_p3 = FPNBlock(self.stage_channels[1], out_channels)
        self.fpn_p2 = FPNBlock(self.stage_channels[0], out_channels)

    def forward(self, x):
        """
        Args:
            x: input image tensor [B, 3, H, W]

        Returns:
            dict with keys 'P2', 'P3', 'P4', 'P5'
            P2: [B, 256, H/4, W/4]
            P3: [B, 256, H/8, W/8]
            P4: [B, 256, H/16, W/16]
            P5: [B, 256, H/32, W/32]
        """
        # Bottom-up pathway
        s = self.stem(x)        # stride 4
        c2 = self.layer1(s)     # [B, 96, H/4, W/4]
        c3 = self.layer2(self.down2(c2))    # [B, 192, H/8, W/8]
        c4 = self.layer3(self.down3(c3))    # [B, 384, H/16, W/16]
        c5 = self.layer4(self.down4(c4))    # [B, 768, H/32, W/32]

        # Top-down FPN pathway
        p5 = self.fpn_p5(c5)
        p4 = self.fpn_p4(c4, p5)
        p3 = self.fpn_p3(c3, p4)
        p2 = self.fpn_p2(c2, p3)

        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}


class ResNet50FPN(nn.Module):
    """
    Alternative backbone: ResNet-50 + FPN.
    Same interface as ConvNeXtFPN.
    """

    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # C2: stride 4, 256ch
        self.layer2 = backbone.layer2   # C3: stride 8, 512ch
        self.layer3 = backbone.layer3   # C4: stride 16, 1024ch
        self.layer4 = backbone.layer4   # C5: stride 32, 2048ch

        self.fpn_p5 = FPNBlock(2048, out_channels)
        self.fpn_p4 = FPNBlock(1024, out_channels)
        self.fpn_p3 = FPNBlock(512, out_channels)
        self.fpn_p2 = FPNBlock(256, out_channels)

    def forward(self, x):
        c2 = self.layer1(self.layer0(x))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.fpn_p5(c5)
        p4 = self.fpn_p4(c4, p5)
        p3 = self.fpn_p3(c3, p4)
        p2 = self.fpn_p2(c2, p3)

        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}


def build_backbone(cfg):
    """Factory function to build backbone from config."""
    backbone_type = cfg.get("model", {}).get("backbone", "convnext_tiny")
    out_channels = cfg.get("model", {}).get("fpn_out_channels", 256)

    if backbone_type in ("convnext_tiny", "convnext"):
        return ConvNeXtFPN(pretrained=True, out_channels=out_channels)
    elif backbone_type in ("resnet50", "resnet"):
        return ResNet50FPN(pretrained=True, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
