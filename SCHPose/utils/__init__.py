"""
Utils module for SCHPose.
"""
from .geometry import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    axis_angle_to_rotation_matrix,
    compute_transform_matrix,
    transform_points,
)
from .symmetry import SymmetryHandler, get_symmetry_transforms
from .metrics import (
    compute_add,
    compute_add_s,
    compute_add_auc,
    compute_vsd,
    compute_mssd,
    compute_mspd,
    PoseMetrics,
)
from .visualization import (
    draw_pose,
    draw_segmentation,
    draw_code_heatmap,
    overlay_mask,
)

__all__ = [
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_axis_angle",
    "axis_angle_to_rotation_matrix",
    "compute_transform_matrix",
    "transform_points",
    "SymmetryHandler",
    "get_symmetry_transforms",
    "compute_add",
    "compute_add_s",
    "compute_add_auc",
    "compute_vsd",
    "compute_mssd",
    "compute_mspd",
    "PoseMetrics",
    "draw_pose",
    "draw_segmentation",
    "draw_code_heatmap",
    "overlay_mask",
]
