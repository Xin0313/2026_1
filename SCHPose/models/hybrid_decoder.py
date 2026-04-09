"""
Hybrid Decoder for SCHPose.

Converts per-pixel predictions (coarse_id, fine_id, u, v) into
precise 3D surface points using the surface codebook.

Supports both:
  - NumPy-based decoding (for inference with precomputed codebook)
  - Differentiable PyTorch-based decoding (for training)
"""

import torch
import torch.nn as nn
import numpy as np


class HybridDecoder(nn.Module):
    """
    Hybrid decoder: (BlockID + continuous offset) -> precise 3D correspondence.

    For each foreground pixel with predictions:
      - coarse_logits: [n_coarse] -> coarse block id (argmax or soft)
      - fine_logits: [n_fine] -> fine block id within coarse block
      - uv: [2] -> within-block (u, v) coordinates in [0, 1]

    Returns a 3D point on the object model surface.

    Args:
        codebook: SurfaceCodebook object (loaded per object)
    """

    def __init__(self, codebook=None):
        super().__init__()
        self.codebook = codebook

        if codebook is not None:
            self._build_lookup_tables(codebook)

    def _build_lookup_tables(self, codebook):
        """
        Build differentiable lookup tables from codebook.

        block_centers_table: [n_coarse, n_fine, 3]  3D center of each block
        block_u_axis_table:   [n_coarse, n_fine, 3]
        block_v_axis_table:   [n_coarse, n_fine, 3]
        block_u_scale_table:  [n_coarse, n_fine]
        block_v_scale_table:  [n_coarse, n_fine]
        """
        n_coarse = codebook.n_coarse
        n_fine = codebook.n_fine

        centers = np.zeros((n_coarse, n_fine, 3), dtype=np.float32)
        u_axes = np.zeros((n_coarse, n_fine, 3), dtype=np.float32)
        v_axes = np.zeros((n_coarse, n_fine, 3), dtype=np.float32)
        u_scales = np.ones((n_coarse, n_fine), dtype=np.float32) * 1e-3
        v_scales = np.ones((n_coarse, n_fine), dtype=np.float32) * 1e-3

        # Default axes
        u_axes[:, :, 0] = 1.0
        v_axes[:, :, 1] = 1.0

        for (c_id, f_id), center in codebook.block_centers.items():
            if c_id < n_coarse and f_id < n_fine:
                centers[c_id, f_id] = center
                if (c_id, f_id) in codebook.block_axes:
                    ua, va = codebook.block_axes[(c_id, f_id)]
                    u_axes[c_id, f_id] = ua
                    v_axes[c_id, f_id] = va
                if (c_id, f_id) in codebook.block_scale:
                    us, vs = codebook.block_scale[(c_id, f_id)]
                    u_scales[c_id, f_id] = us
                    v_scales[c_id, f_id] = vs

        # Register as buffers (non-trainable)
        self.register_buffer("block_centers_t", torch.from_numpy(centers))
        self.register_buffer("block_u_axis_t", torch.from_numpy(u_axes))
        self.register_buffer("block_v_axis_t", torch.from_numpy(v_axes))
        self.register_buffer("block_u_scale_t", torch.from_numpy(u_scales))
        self.register_buffer("block_v_scale_t", torch.from_numpy(v_scales))

    def update_codebook(self, codebook):
        """Update the codebook (e.g., when switching objects)."""
        self.codebook = codebook
        self._build_lookup_tables(codebook)

    def forward(self, coarse_probs, fine_probs, uv, mask=None):
        """
        Decode hybrid codes to 3D points using differentiable soft lookup.

        Args:
            coarse_probs: [B, n_coarse, H, W] (softmax probabilities)
            fine_probs: [B, n_fine, H, W]
            uv: [B, 2, H, W] in [0, 1]
            mask: [B, H, W] binary foreground mask (optional)

        Returns:
            points3d: [B, H, W, 3] 3D points on model surface
        """
        if not hasattr(self, "block_centers_t"):
            raise RuntimeError("Codebook not set. Call update_codebook() first.")

        B, n_coarse, H, W = coarse_probs.shape
        n_fine = fine_probs.shape[1]

        # Soft selection of block using probability-weighted sum
        # centers: [n_coarse, n_fine, 3] -> [1, n_coarse, n_fine, 3]
        centers = self.block_centers_t.unsqueeze(0)          # [1, C, F, 3]
        u_axes = self.block_u_axis_t.unsqueeze(0)            # [1, C, F, 3]
        v_axes = self.block_v_axis_t.unsqueeze(0)            # [1, C, F, 3]
        u_scales = self.block_u_scale_t.unsqueeze(0)         # [1, C, F]
        v_scales = self.block_v_scale_t.unsqueeze(0)         # [1, C, F]

        # Reshape spatial dims
        # coarse_probs: [B, C, H*W] -> [B, H*W, C]
        N = H * W
        cp = coarse_probs.reshape(B, n_coarse, N).permute(0, 2, 1)  # [B, N, C]
        fp = fine_probs.reshape(B, n_fine, N).permute(0, 2, 1)       # [B, N, F]

        # Outer product: [B, N, C, F] joint block probability
        joint_prob = cp.unsqueeze(3) * fp.unsqueeze(2)  # [B, N, C, F]

        # Weighted center: sum over C, F dims
        # centers: [C, F, 3] -> [1, 1, C, F, 3]
        centers_e = self.block_centers_t.unsqueeze(0).unsqueeze(0)   # [1, 1, C, F, 3]
        u_axes_e = self.block_u_axis_t.unsqueeze(0).unsqueeze(0)
        v_axes_e = self.block_v_axis_t.unsqueeze(0).unsqueeze(0)
        u_scales_e = self.block_u_scale_t.unsqueeze(0).unsqueeze(0)  # [1, 1, C, F]
        v_scales_e = self.block_v_scale_t.unsqueeze(0).unsqueeze(0)

        # [B, N, C, F, 1] * [1, 1, C, F, 3] -> [B, N, 3]
        jp = joint_prob.unsqueeze(-1)  # [B, N, C, F, 1]
        weighted_center = (jp * centers_e).sum(dim=(2, 3))  # [B, N, 3]
        weighted_u_axis = (jp * u_axes_e).sum(dim=(2, 3))   # [B, N, 3]
        weighted_v_axis = (jp * v_axes_e).sum(dim=(2, 3))   # [B, N, 3]
        weighted_u_scale = (joint_prob * u_scales_e).sum(dim=(2, 3))  # [B, N]
        weighted_v_scale = (joint_prob * v_scales_e).sum(dim=(2, 3))

        # uv: [B, 2, H, W] -> [B, N, 2]
        uv_flat = uv.reshape(B, 2, N).permute(0, 2, 1)  # [B, N, 2]
        u_local = (uv_flat[:, :, 0] * 2.0 - 1.0) * weighted_u_scale  # [B, N]
        v_local = (uv_flat[:, :, 1] * 2.0 - 1.0) * weighted_v_scale  # [B, N]

        # Final 3D point
        points3d = (
            weighted_center
            + u_local.unsqueeze(-1) * weighted_u_axis
            + v_local.unsqueeze(-1) * weighted_v_axis
        )  # [B, N, 3]

        points3d = points3d.reshape(B, H, W, 3)

        return points3d

    def decode_argmax(self, coarse_logits, fine_logits, uv):
        """
        Hard (argmax) decoding for inference.

        Args:
            coarse_logits: [B, n_coarse, H, W]
            fine_logits: [B, n_fine, H, W]
            uv: [B, 2, H, W]

        Returns:
            points3d: [B, H, W, 3]
            c_ids: [B, H, W] int64
            f_ids: [B, H, W] int64
        """
        if not hasattr(self, "block_centers_t"):
            raise RuntimeError("Codebook not set. Call update_codebook() first.")

        c_ids = coarse_logits.argmax(dim=1)  # [B, H, W]
        f_ids = fine_logits.argmax(dim=1)    # [B, H, W]

        B, H, W = c_ids.shape

        # Lookup centers and axes
        centers = self.block_centers_t[c_ids, f_ids]   # [B, H, W, 3]
        u_axes = self.block_u_axis_t[c_ids, f_ids]
        v_axes = self.block_v_axis_t[c_ids, f_ids]
        u_scales = self.block_u_scale_t[c_ids, f_ids]  # [B, H, W]
        v_scales = self.block_v_scale_t[c_ids, f_ids]

        u_vals = uv[:, 0]  # [B, H, W]
        v_vals = uv[:, 1]
        u_local = (u_vals * 2.0 - 1.0) * u_scales
        v_local = (v_vals * 2.0 - 1.0) * v_scales

        points3d = (
            centers
            + u_local.unsqueeze(-1) * u_axes
            + v_local.unsqueeze(-1) * v_axes
        )

        return points3d, c_ids, f_ids

    def get_correspondences(self, coarse_logits, fine_logits, uv, log_var, mask, top_k=512):
        """
        Extract top-K 2D-3D correspondences for PnP.

        Args:
            coarse_logits: [n_coarse, H, W]  (single sample)
            fine_logits: [n_fine, H, W]
            uv: [2, H, W]
            log_var: [1, H, W]
            mask: [H, W] binary
            top_k: max number of correspondences

        Returns:
            pts2d: [K, 2] pixel coordinates (x, y)
            pts3d: [K, 3] 3D model points
            weights: [K] confidence weights
        """
        H, W = mask.shape
        device = coarse_logits.device

        # Get foreground pixels
        fg_y, fg_x = torch.where(mask > 0)

        if len(fg_y) == 0:
            return (
                torch.zeros(0, 2, device=device),
                torch.zeros(0, 3, device=device),
                torch.zeros(0, device=device),
            )

        # Extract predictions at foreground pixels
        c_ids = coarse_logits[:, fg_y, fg_x].argmax(dim=0)  # [N_fg]
        f_ids = fine_logits[:, fg_y, fg_x].argmax(dim=0)    # [N_fg]
        uv_fg = uv[:, fg_y, fg_x]                           # [2, N_fg]
        log_var_fg = log_var[0, fg_y, fg_x]                 # [N_fg]

        # Compute weights (higher weight = lower uncertainty)
        weights = torch.exp(-log_var_fg)  # [N_fg]

        # Select top-K by weight
        K = min(top_k, len(fg_y))
        topk_idx = torch.topk(weights, K, largest=True, sorted=False).indices
        c_ids_k = c_ids[topk_idx]
        f_ids_k = f_ids[topk_idx]
        uv_k = uv_fg[:, topk_idx]       # [2, K]
        weights_k = weights[topk_idx]   # [K]

        # 2D pixel coordinates (scale back to full image coords if needed)
        pts2d = torch.stack([fg_x[topk_idx].float(), fg_y[topk_idx].float()], dim=1)  # [K, 2]

        # 3D points from codebook
        centers = self.block_centers_t[c_ids_k, f_ids_k]    # [K, 3]
        u_axes = self.block_u_axis_t[c_ids_k, f_ids_k]
        v_axes = self.block_v_axis_t[c_ids_k, f_ids_k]
        u_scales = self.block_u_scale_t[c_ids_k, f_ids_k]
        v_scales = self.block_v_scale_t[c_ids_k, f_ids_k]

        u_local = (uv_k[0] * 2.0 - 1.0) * u_scales
        v_local = (uv_k[1] * 2.0 - 1.0) * v_scales
        pts3d = centers + u_local.unsqueeze(1) * u_axes + v_local.unsqueeze(1) * v_axes

        return pts2d, pts3d, weights_k
