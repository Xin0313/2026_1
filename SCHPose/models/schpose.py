"""
SCHPose: Main model integrating all components.

Architecture:
  Input RGB -> ConvNeXtFPN -> SCHPoseHeads -> HybridDecoder -> DiffPnP -> (R, t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvNeXtFPN, ResNet50FPN, build_backbone
from .heads import SCHPoseHeads
from .hybrid_decoder import HybridDecoder
from .diff_pnp import DiffPnP


class SCHPose(nn.Module):
    """
    SCHPose: Symmetry-aware Continuous-discrete Hybrid encoding for
    single-stage 6D Pose estimation.

    Single-stage, RGB-only 6D pose estimation.

    Args:
        cfg: configuration dict
        codebooks: dict mapping obj_id -> SurfaceCodebook (optional at init)
        sym_transforms: dict mapping obj_id -> list of [3,3] rotation matrices
    """

    def __init__(self, cfg, codebooks=None, sym_transforms=None):
        super().__init__()
        self.cfg = cfg
        self.codebooks = codebooks or {}
        self.sym_transforms = sym_transforms or {}

        model_cfg = cfg.get("model", {})
        dataset_cfg = cfg.get("dataset", {})

        self.num_objects = dataset_cfg.get("num_objects", 21)
        self.n_coarse = model_cfg.get("n_coarse", 64)
        self.n_fine = model_cfg.get("n_fine", 16)
        self.n_sym = model_cfg.get("n_sym", 8)
        self.fpn_out_channels = model_cfg.get("fpn_out_channels", 256)
        self.use_uncertainty = model_cfg.get("uncertainty_head", True)
        self.use_symmetry = model_cfg.get("symmetry_head", True)

        pnp_cfg = cfg.get("pnp", {})
        self.top_k = pnp_cfg.get("top_k", 512)
        self.pnp_n_iter = pnp_cfg.get("n_iter", 5)

        # Build sub-modules
        self.backbone = build_backbone(cfg)
        self.heads = SCHPoseHeads(
            in_channels=self.fpn_out_channels,
            num_objects=self.num_objects,
            n_coarse=self.n_coarse,
            n_fine=self.n_fine,
            n_sym=self.n_sym,
            use_uncertainty=self.use_uncertainty,
            use_symmetry=self.use_symmetry,
        )
        self.decoder = HybridDecoder()
        self.pnp = DiffPnP(n_iter=self.pnp_n_iter, top_k=self.top_k)

        # Scale factor for coordinate output (heads operate at 1/4 resolution)
        self.stride = 4

    def set_codebook(self, obj_id, codebook):
        """Set or update codebook for an object."""
        self.codebooks[obj_id] = codebook
        self.decoder.update_codebook(codebook)

    def forward(self, images, K, obj_ids=None, masks_gt=None, phase=1):
        """
        Forward pass.

        Args:
            images: [B, 3, H, W] RGB images, normalized to [0,1]
            K: [B, 3, 3] camera intrinsics
            obj_ids: list of length B, object ids per sample
                     (needed for codebook lookup in inference)
            masks_gt: [B, H, W] ground-truth masks (for training, Phase 1)
            phase: training phase (1 or 2)

        Returns:
            dict with:
              'head_outputs': raw head predictions (dict)
              'poses': list of (R, t) tuples per batch element (Phase 2)
              'points3d': [B, H/4, W/4, 3] decoded 3D points
        """
        B, C, H, W = images.shape

        # Normalize images (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_norm = (images - mean) / std

        # Extract features
        fpn_feats = self.backbone(images_norm)
        p2 = fpn_feats["P2"]  # [B, 256, H/4, W/4]

        # Predict from heads
        head_out = self.heads(p2)

        # Decode 3D points (differentiable, for training)
        points3d = None
        poses = []

        if phase == 2 or not self.training:
            # Soft decode for Phase 2 training
            coarse_probs = F.softmax(head_out["coarse_logits"], dim=1)
            fine_probs = F.softmax(head_out["fine_logits"], dim=1)

            # Update decoder with first batch's codebook if available
            # (In multi-object scenarios, need per-sample codebook)
            if obj_ids is not None and len(self.codebooks) > 0:
                first_obj = obj_ids[0] if isinstance(obj_ids, (list, tuple)) else obj_ids[0].item()
                if first_obj in self.codebooks:
                    self.decoder.update_codebook(self.codebooks[first_obj])

            if hasattr(self.decoder, "block_centers_t"):
                points3d = self.decoder(coarse_probs, fine_probs, head_out["uv"])

            # Solve PnP per sample
            log_var = head_out.get("log_var", None)

            for b in range(B):
                obj_id_b = None
                if obj_ids is not None:
                    obj_id_b = obj_ids[b] if isinstance(obj_ids, (list, tuple)) else obj_ids[b].item()

                # Set codebook for this object
                if obj_id_b is not None and obj_id_b in self.codebooks:
                    self.decoder.update_codebook(self.codebooks[obj_id_b])

                # Get mask for this sample
                if masks_gt is not None:
                    # Use GT mask (training) at 1/4 resolution
                    mask_b = F.interpolate(
                        masks_gt[b:b+1].unsqueeze(1).float(),
                        size=(H // self.stride, W // self.stride),
                        mode="nearest"
                    ).squeeze().bool()
                else:
                    # Use predicted segmentation (inference)
                    seg_logits_b = head_out["seg_logits"][b]
                    if obj_id_b is not None:
                        obj_idx = self._obj_id_to_idx(obj_id_b)
                        mask_b = seg_logits_b[obj_idx + 1] > seg_logits_b[0]
                    else:
                        mask_b = seg_logits_b[1:].max(dim=0).values > seg_logits_b[0]

                # Get correspondences
                lv_b = log_var[b] if log_var is not None else torch.zeros(
                    1, H // self.stride, W // self.stride, device=images.device
                )

                if hasattr(self.decoder, "block_centers_t"):
                    pts2d, pts3d_kp, weights = self.decoder.get_correspondences(
                        coarse_logits=head_out["coarse_logits"][b],
                        fine_logits=head_out["fine_logits"][b],
                        uv=head_out["uv"][b],
                        log_var=lv_b,
                        mask=mask_b,
                        top_k=self.top_k,
                    )

                    # Scale pts2d from 1/4 resolution to full resolution
                    pts2d_full = pts2d * self.stride

                    K_b = K[b]
                    sym_t = self.sym_transforms.get(obj_id_b, None)

                    if len(pts2d_full) >= 4:
                        R_b, t_b = self.pnp(
                            pts2d_full, pts3d_kp, weights, K_b, sym_t
                        )
                        poses.append((R_b, t_b))
                    else:
                        poses.append((
                            torch.eye(3, device=images.device, dtype=images.dtype),
                            torch.zeros(3, device=images.device, dtype=images.dtype),
                        ))
                else:
                    poses.append((
                        torch.eye(3, device=images.device, dtype=images.dtype),
                        torch.zeros(3, device=images.device, dtype=images.dtype),
                    ))

        return {
            "head_outputs": head_out,
            "points3d": points3d,
            "poses": poses,
            "fpn_features": fpn_feats,
        }

    def _obj_id_to_idx(self, obj_id):
        """Map obj_id to 0-indexed position in object list."""
        obj_ids = self.cfg.get("dataset", {}).get("object_ids", [])
        if obj_id in obj_ids:
            return obj_ids.index(obj_id)
        return 0

    def inference(self, image, K, obj_id=None):
        """
        Single image inference.

        Args:
            image: [3, H, W] or [1, 3, H, W] tensor
            K: [3, 3] or [1, 3, 3] tensor
            obj_id: object id (int)

        Returns:
            R: [3, 3] rotation matrix
            t: [3] translation vector
            seg_mask: [H/4, W/4] segmentation mask
        """
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if K.dim() == 2:
                K = K.unsqueeze(0)

            output = self.forward(
                image, K,
                obj_ids=[obj_id] if obj_id is not None else None,
                phase=2,
            )

            poses = output["poses"]
            if poses:
                R, t = poses[0]
            else:
                R = torch.eye(3, device=image.device)
                t = torch.zeros(3, device=image.device)

            # Segmentation mask
            head_out = output["head_outputs"]
            seg_logits = head_out["seg_logits"][0]
            seg_mask = seg_logits[1:].max(dim=0).values > seg_logits[0]

            return R, t, seg_mask


def build_model(cfg, codebooks=None, sym_transforms=None):
    """Factory function to build SCHPose model."""
    return SCHPose(cfg, codebooks=codebooks, sym_transforms=sym_transforms)
