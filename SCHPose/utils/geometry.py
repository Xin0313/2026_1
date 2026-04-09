"""
Geometric utility functions for SCHPose.

Includes:
  - Rotation matrix <-> quaternion <-> axis-angle conversions
  - Transform matrix utilities
  - Point transformation helpers
"""

import numpy as np
import torch


# ── Numpy versions ─────────────────────────────────────────────────────────────

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion [w, x, y, z].

    Args:
        R: np.ndarray [3, 3] or [B, 3, 3]

    Returns:
        q: np.ndarray [4] or [B, 4]
    """
    batched = R.ndim == 3
    if not batched:
        R = R[None]

    B = R.shape[0]
    q = np.zeros((B, 4), dtype=np.float64)

    for i in range(B):
        r = R[i]
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[i, 0] = 0.25 / s
            q[i, 1] = (r[2, 1] - r[1, 2]) * s
            q[i, 2] = (r[0, 2] - r[2, 0]) * s
            q[i, 3] = (r[1, 0] - r[0, 1]) * s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q[i, 0] = (r[2, 1] - r[1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (r[0, 1] + r[1, 0]) / s
            q[i, 3] = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            q[i, 0] = (r[0, 2] - r[2, 0]) / s
            q[i, 1] = (r[0, 1] + r[1, 0]) / s
            q[i, 2] = 0.25 * s
            q[i, 3] = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            q[i, 0] = (r[1, 0] - r[0, 1]) / s
            q[i, 1] = (r[0, 2] + r[2, 0]) / s
            q[i, 2] = (r[1, 2] + r[2, 1]) / s
            q[i, 3] = 0.25 * s

    return q[0] if not batched else q


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix.

    Args:
        q: np.ndarray [4] or [B, 4]

    Returns:
        R: np.ndarray [3, 3] or [B, 3, 3]
    """
    batched = q.ndim == 2
    if not batched:
        q = q[None]

    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    B = len(q)
    R = np.zeros((B, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R[0] if not batched else R


def rotation_matrix_to_axis_angle(R):
    """
    Convert rotation matrix to Rodrigues axis-angle representation.

    Args:
        R: np.ndarray [3, 3]

    Returns:
        rvec: np.ndarray [3]
    """
    try:
        import cv2
        rvec, _ = cv2.Rodrigues(R)
        return rvec.reshape(3)
    except ImportError:
        pass

    # Manual implementation
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def axis_angle_to_rotation_matrix(rvec):
    """
    Convert Rodrigues axis-angle to rotation matrix.

    Args:
        rvec: np.ndarray [3]

    Returns:
        R: np.ndarray [3, 3]
    """
    try:
        import cv2
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        return R
    except ImportError:
        pass

    angle = np.linalg.norm(rvec)
    if angle < 1e-8:
        return np.eye(3)
    axis = rvec / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def compute_transform_matrix(R, t):
    """
    Build 4x4 homogeneous transform from R [3,3] and t [3].

    Returns:
        T: np.ndarray [4, 4]
    """
    T = np.eye(4, dtype=R.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(pts, R, t):
    """
    Transform 3D points.

    Args:
        pts: np.ndarray [N, 3]
        R: [3, 3]
        t: [3]

    Returns:
        transformed: [N, 3]
    """
    return (R @ pts.T).T + t


def invert_transform(R, t):
    """
    Compute inverse transform.

    Args:
        R: [3, 3]
        t: [3]

    Returns:
        R_inv: [3, 3], t_inv: [3]
    """
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def angular_error_deg(R_pred, R_gt):
    """
    Compute rotation error in degrees.

    Args:
        R_pred, R_gt: [3, 3] numpy

    Returns:
        angle_deg: float
    """
    R_err = R_pred @ R_gt.T
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def translation_error_m(t_pred, t_gt):
    """
    Compute translation error in meters.

    Args:
        t_pred, t_gt: [3] numpy

    Returns:
        err: float
    """
    return np.linalg.norm(t_pred - t_gt)
