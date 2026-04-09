"""
Differentiable Weighted PnP solver for SCHPose.

Features:
  - Importance sampling (top-K by confidence)
  - Weighted EPnP initialization
  - Differentiable Gauss-Newton iterative refinement (5 steps)
  - Symmetry-aware: enumerate symmetry transforms, pick best
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def skew_symmetric(v):
    """
    Compute skew-symmetric matrix from 3D vector.

    Args:
        v: [..., 3]

    Returns:
        S: [..., 3, 3]
    """
    S = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] = v[..., 1]
    S[..., 1, 0] = v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] = v[..., 0]
    return S


def rodrigues_to_rotation(rvec):
    """
    Convert Rodrigues rotation vector to rotation matrix.

    Args:
        rvec: [..., 3]

    Returns:
        R: [..., 3, 3]
    """
    angle = torch.norm(rvec, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = rvec / angle
    angle = angle.squeeze(-1)
    cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    K = skew_symmetric(axis)
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(K)
    R = cos_a * I + sin_a * K + (1 - cos_a) * torch.einsum("...i,...j->...ij", axis, axis)
    return R


def project_points(pts3d, R, t, K):
    """
    Project 3D points to 2D image plane.

    Args:
        pts3d: [N, 3] or [B, N, 3]
        R: [3, 3] or [B, 3, 3]
        t: [3] or [B, 3]
        K: [3, 3] or [B, 3, 3]

    Returns:
        pts2d: [N, 2] or [B, N, 2] image coordinates (x, y)
    """
    # Handle batched and non-batched inputs
    batched = pts3d.dim() == 3
    if not batched:
        pts3d = pts3d.unsqueeze(0)
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        K = K.unsqueeze(0)

    # pts3d_cam: [B, N, 3]
    pts3d_cam = torch.einsum("bij,bnj->bni", R, pts3d) + t.unsqueeze(1)

    # Project
    z = pts3d_cam[..., 2:3].clamp(min=1e-6)
    pts3d_norm = pts3d_cam / z  # [B, N, 3]

    # Apply K
    fx = K[:, 0, 0:1].unsqueeze(1)  # [B, 1, 1]
    fy = K[:, 1, 1:2].unsqueeze(1)
    cx = K[:, 0, 2:3].unsqueeze(1)
    cy = K[:, 1, 2:3].unsqueeze(1)

    x = pts3d_norm[..., 0:1] * fx + cx
    y = pts3d_norm[..., 1:2] * fy + cy
    pts2d = torch.cat([x, y], dim=-1)  # [B, N, 2]

    if not batched:
        pts2d = pts2d.squeeze(0)

    return pts2d


def weighted_epnp_init(pts2d, pts3d, weights, K):
    """
    Weighted EPnP initialization using OpenCV (CPU fallback).

    Args:
        pts2d: [N, 2] float numpy
        pts3d: [N, 3] float numpy
        weights: [N] float numpy
        K: [3, 3] float numpy

    Returns:
        R: [3, 3] numpy
        t: [3] numpy
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required for EPnP initialization.")

    N = len(pts2d)
    if N < 4:
        # Degenerate case: return identity
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    # Select top weighted points for RANSAC stability
    top_k = min(max(N, 6), N)
    order = np.argsort(-weights)[:top_k]
    pts2d_s = pts2d[order].astype(np.float64)
    pts3d_s = pts3d[order].astype(np.float64)

    dist_coeffs = np.zeros(4)
    success, rvec, tvec, _ = cv2.solvePnPRansac(
        pts3d_s,
        pts2d_s,
        K.astype(np.float64),
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=3.0,
        iterationsCount=100,
        confidence=0.99,
    )
    if not success:
        # Fallback to EPnP without RANSAC
        success2, rvec, tvec = cv2.solvePnP(
            pts3d_s[:max(6, len(pts3d_s))],
            pts2d_s[:max(6, len(pts2d_s))],
            K.astype(np.float64),
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success2:
            return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R, t


class DiffPnP(nn.Module):
    """
    Differentiable Weighted PnP solver with Gauss-Newton refinement.

    Args:
        n_iter: number of Gauss-Newton iterations
        top_k: number of top correspondences to use
        sym_transforms: list of [4,4] tensors for symmetry transforms
    """

    def __init__(self, n_iter=5, top_k=512):
        super().__init__()
        self.n_iter = n_iter
        self.top_k = top_k

    def forward(self, pts2d, pts3d, weights, K, sym_transforms=None):
        """
        Solve weighted PnP with differentiable Gauss-Newton refinement.

        Args:
            pts2d: [N, 2] or [B, N, 2] 2D pixel correspondences
            pts3d: [N, 3] or [B, N, 3] 3D model points
            weights: [N] or [B, N] per-point weights (confidence)
            K: [3, 3] or [B, 3, 3] camera intrinsics
            sym_transforms: list of [3, 3] rotation matrices (symmetry)

        Returns:
            R: [3, 3] or [B, 3, 3]
            t: [3] or [B, 3]
        """
        batched = pts2d.dim() == 3
        if not batched:
            pts2d = pts2d.unsqueeze(0)
            pts3d = pts3d.unsqueeze(0)
            weights = weights.unsqueeze(0)
            K = K.unsqueeze(0)

        B = pts2d.shape[0]
        results_R = []
        results_t = []

        for b in range(B):
            R_b, t_b = self._solve_single(
                pts2d[b], pts3d[b], weights[b], K[b], sym_transforms
            )
            results_R.append(R_b)
            results_t.append(t_b)

        R_out = torch.stack(results_R, dim=0)
        t_out = torch.stack(results_t, dim=0)

        if not batched:
            R_out = R_out.squeeze(0)
            t_out = t_out.squeeze(0)

        return R_out, t_out

    def _solve_single(self, pts2d, pts3d, weights, K, sym_transforms=None):
        """
        Solve for a single sample.

        Args:
            pts2d: [N, 2]
            pts3d: [N, 3]
            weights: [N]
            K: [3, 3]
            sym_transforms: optional list of rotation matrices [3,3]

        Returns:
            R: [3, 3], t: [3]
        """
        device = pts2d.device
        N = pts2d.shape[0]

        if N < 4:
            return (
                torch.eye(3, device=device, dtype=pts2d.dtype),
                torch.zeros(3, device=device, dtype=pts2d.dtype),
            )

        # EPnP initialization (CPU, numpy)
        p2d_np = pts2d.detach().cpu().numpy().astype(np.float64)
        p3d_np = pts3d.detach().cpu().numpy().astype(np.float64)
        w_np = weights.detach().cpu().numpy().astype(np.float64)
        K_np = K.detach().cpu().numpy().astype(np.float64)

        R_init, t_init = weighted_epnp_init(p2d_np, p3d_np, w_np, K_np)
        R = torch.from_numpy(R_init).to(device=device, dtype=pts2d.dtype)
        t = torch.from_numpy(t_init).to(device=device, dtype=pts2d.dtype)

        # Gauss-Newton refinement (differentiable)
        R, t = self._gauss_newton_refine(pts2d, pts3d, weights, K, R, t)

        # Symmetry-aware: pick best among symmetry equivalents
        if sym_transforms and len(sym_transforms) > 0:
            R, t = self._apply_symmetry_selection(
                pts2d, pts3d, weights, K, R, t, sym_transforms
            )

        return R, t

    def _gauss_newton_refine(self, pts2d, pts3d, weights, K, R, t):
        """
        Differentiable Gauss-Newton refinement of PnP.

        Minimizes: sum_i w_i * ||pi(R, t, x_i) - u_i||^2

        Args:
            pts2d: [N, 2]
            pts3d: [N, 3]
            weights: [N]
            K: [3, 3]
            R: [3, 3] initial rotation
            t: [3] initial translation

        Returns:
            R_refined: [3, 3]
            t_refined: [3]
        """
        device = pts2d.device
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        for _ in range(self.n_iter):
            # Camera-frame points
            pts_cam = (R @ pts3d.T).T + t.unsqueeze(0)  # [N, 3]
            z = pts_cam[:, 2].clamp(min=1e-6)
            x_proj = pts_cam[:, 0] / z * fx + cx
            y_proj = pts_cam[:, 1] / z * fy + cy

            # Residuals [N, 2]
            res_x = x_proj - pts2d[:, 0]
            res_y = y_proj - pts2d[:, 1]
            residuals = torch.stack([res_x, res_y], dim=1)  # [N, 2]

            # Jacobian of projection w.r.t. camera-frame point [N, 2, 3]
            # d(x_proj)/d(xc, yc, zc) = [fx/z, 0, -fx*xc/z^2]
            # d(y_proj)/d(xc, yc, zc) = [0, fy/z, -fy*yc/z^2]
            xc = pts_cam[:, 0]
            yc = pts_cam[:, 1]
            J_proj = torch.zeros(len(pts3d), 2, 3, device=device, dtype=pts2d.dtype)
            J_proj[:, 0, 0] = fx / z
            J_proj[:, 0, 2] = -fx * xc / (z * z)
            J_proj[:, 1, 1] = fy / z
            J_proj[:, 1, 2] = -fy * yc / (z * z)

            # Jacobian w.r.t. [rvec(3), t(3)] via chain rule
            # dpts_cam/dt = I
            # dpts_cam/drvec via Rodrigues: dRx/drvec = -R * skew(x)
            # Use numerical approach for rotation part to keep differentiable
            # J_t: [N, 2, 3]
            J_t = J_proj  # dpts_cam/dt = I -> J_t = J_proj

            # For rotation: dpts_cam/drvec = -R * [pts3d_i x]^T
            pts_skew = skew_symmetric(pts3d)  # [N, 3, 3]
            # dpts_cam/drvec = -skew(R*pts3d) * dR/drvec approx -R*skew(pts3d)^T
            J_R = -torch.einsum("nij,njk->nik", J_proj, (R.unsqueeze(0) @ pts_skew))

            # Full Jacobian: [N, 2, 6]
            J = torch.cat([J_R, J_t], dim=2)  # [N, 2, 6]

            # Weighted least squares: (J^T W J) delta = -J^T W r
            W = weights.unsqueeze(1)  # [N, 1]
            J_flat = J.reshape(-1, 6)     # [2N, 6]
            r_flat = residuals.reshape(-1)  # [2N]
            W_flat = W.expand(-1, 2).reshape(-1)  # [2N]

            JtWJ = (J_flat * W_flat.unsqueeze(1)).T @ J_flat  # [6, 6]
            JtWr = (J_flat * W_flat.unsqueeze(1)).T @ r_flat  # [6]

            # Add damping for stability
            damp = 1e-4 * torch.eye(6, device=device, dtype=pts2d.dtype)
            try:
                delta = torch.linalg.solve(JtWJ + damp, -JtWr)
            except RuntimeError:
                break

            # Update R and t
            delta_r = delta[:3]
            delta_t = delta[3:]
            angle = torch.norm(delta_r).clamp(max=0.1)
            if angle > 1e-8:
                axis = delta_r / angle
                R_delta = rodrigues_to_rotation(delta_r.unsqueeze(0)).squeeze(0)
                R = R_delta @ R
            t = t + delta_t

        return R, t

    def _apply_symmetry_selection(self, pts2d, pts3d, weights, K, R, t, sym_transforms):
        """
        Enumerate symmetry-equivalent poses and select the one with minimum
        weighted reprojection error.

        Args:
            sym_transforms: list of [3, 3] rotation matrices (model-space symmetries)

        Returns:
            R_best: [3, 3], t_best: [3]
        """
        device = pts2d.device

        def weighted_reproj_error(R_cand, t_cand):
            pts_cam = (R_cand @ pts3d.T).T + t_cand.unsqueeze(0)
            z = pts_cam[:, 2].clamp(min=1e-6)
            x_proj = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
            y_proj = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
            err_x = x_proj - pts2d[:, 0]
            err_y = y_proj - pts2d[:, 1]
            err = (err_x ** 2 + err_y ** 2) * weights
            return err.mean()

        best_R, best_t = R, t
        best_err = weighted_reproj_error(R, t)

        for sym_R in sym_transforms:
            if not isinstance(sym_R, torch.Tensor):
                sym_R = torch.from_numpy(sym_R).to(device=device, dtype=R.dtype)
            R_cand = R @ sym_R
            err = weighted_reproj_error(R_cand, t)
            if err < best_err:
                best_err = err
                best_R, best_t = R_cand, t

        return best_R, best_t

    def compute_weighted_reproj_error(self, pts2d, pts3d, weights, K, R, t):
        """
        Compute weighted reprojection error (for loss computation).

        Args:
            pts2d: [N, 2]
            pts3d: [N, 3]
            weights: [N]
            K: [3, 3]
            R: [3, 3]
            t: [3]

        Returns:
            error: scalar
        """
        pts_cam = (R @ pts3d.T).T + t.unsqueeze(0)
        z = pts_cam[:, 2].clamp(min=1e-6)
        x_proj = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
        y_proj = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
        err = ((x_proj - pts2d[:, 0]) ** 2 + (y_proj - pts2d[:, 1]) ** 2)
        return (err * weights).sum() / (weights.sum() + 1e-8)
