"""
Symmetry definitions and handling for SCHPose.

Provides symmetry transforms for YCB-V, LM-O, and T-LESS objects.
Symmetry transforms are represented as 3x3 rotation matrices that,
when applied to model points, produce an equivalent appearance.
"""

import numpy as np


# ── YCB-V symmetry definitions ─────────────────────────────────────────────────
# Objects with discrete or continuous rotational symmetry.
# Format: obj_id -> list of np.ndarray [3, 3] rotation matrices

def _rot_z(angle_deg):
    """Rotation around Z axis by angle_deg."""
    a = np.deg2rad(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1],
    ], dtype=np.float64)


def _rot_y(angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [0,          1, 0        ],
        [-np.sin(a), 0, np.cos(a)],
    ], dtype=np.float64)


def _rot_x(angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([
        [1, 0,          0         ],
        [0, np.cos(a), -np.sin(a) ],
        [0, np.sin(a),  np.cos(a) ],
    ], dtype=np.float64)


def _continuous_rot_z(n=36):
    """Discretize continuous Z-rotational symmetry into n steps."""
    return [_rot_z(360.0 / n * i) for i in range(1, n)]


def _continuous_rot_y(n=36):
    return [_rot_y(360.0 / n * i) for i in range(1, n)]


# YCB-V symmetric objects
YCBV_SYMMETRIES = {
    # 13: bowl - continuous Z symmetry
    13: _continuous_rot_z(36),
    # 16: wood_block - 4-fold Z symmetry
    16: [_rot_z(90 * i) for i in range(1, 4)],
    # 19: large_clamp - 2-fold symmetry
    19: [_rot_z(180)],
    # 20: extra_large_clamp - 2-fold symmetry
    20: [_rot_z(180)],
    # 21: foam_brick - 2-fold symmetry (rectangular block)
    21: [_rot_z(180), _rot_x(180), _rot_y(180)],
}

# LM-O symmetric objects
LMO_SYMMETRIES = {
    # 10: glue - continuous Z symmetry
    10: _continuous_rot_z(36),
    # 11: eggbox - 2-fold symmetry
    11: [_rot_z(180)],
}

# T-LESS: most objects have rotational symmetry
# We provide continuous Z symmetry for all T-LESS objects as default
def _tless_symmetries():
    syms = {}
    for obj_id in range(1, 31):
        # Default: continuous Z rotation (most T-LESS objects are rotationally symmetric)
        # Objects with known discrete symmetry orders:
        discrete_order = {
            1: 4, 2: 4, 3: 4, 4: 4,    # 4-fold
            5: 4, 6: 4, 7: 4, 8: 4,
            9: 3, 10: 3, 11: 3, 12: 3, # 3-fold
            13: 36, 14: 36, 15: 36,     # continuous
            16: 4, 17: 4, 18: 4,
            19: 4, 20: 4, 21: 4,
            22: 2, 23: 2, 24: 2,
            25: 4, 26: 4, 27: 4,
            28: 4, 29: 4, 30: 36,
        }.get(obj_id, 36)
        syms[obj_id] = [_rot_z(360.0 / discrete_order * i) for i in range(1, discrete_order)]
    return syms


TLESS_SYMMETRIES = _tless_symmetries()


class SymmetryHandler:
    """
    Helper class to manage symmetry transforms for multiple datasets/objects.

    Args:
        dataset_name: 'ycbv' | 'lmo' | 'tless'
        object_ids: list of object ids to handle
    """

    def __init__(self, dataset_name="ycbv", object_ids=None):
        self.dataset_name = dataset_name
        self.object_ids = object_ids or []
        self._sym_db = self._load_sym_db(dataset_name)

    def _load_sym_db(self, dataset_name):
        if dataset_name == "ycbv":
            return YCBV_SYMMETRIES
        elif dataset_name == "lmo":
            return LMO_SYMMETRIES
        elif dataset_name == "tless":
            return TLESS_SYMMETRIES
        return {}

    def is_symmetric(self, obj_id):
        """Return True if object has non-trivial symmetry."""
        return obj_id in self._sym_db

    def get_symmetries(self, obj_id):
        """
        Return list of symmetry rotation matrices [3,3] for object.
        Returns empty list for non-symmetric objects.
        """
        return self._sym_db.get(obj_id, [])

    def get_all_symmetries(self):
        """Return dict: obj_id -> list of [3,3] rotation matrices."""
        return {oid: self.get_symmetries(oid) for oid in self.object_ids}

    def apply_symmetry(self, R, obj_id):
        """
        Generate all symmetry-equivalent rotations.

        Args:
            R: [3,3] numpy rotation matrix
            obj_id: int

        Returns:
            list of [3,3] rotation matrices (includes input R)
        """
        syms = self.get_symmetries(obj_id)
        equivalents = [R]
        for sym_R in syms:
            equivalents.append(R @ sym_R)
        return equivalents


def get_symmetry_transforms(dataset_name, obj_id):
    """
    Convenience function to get symmetry transforms.

    Args:
        dataset_name: 'ycbv' | 'lmo' | 'tless'
        obj_id: int

    Returns:
        list of [3, 3] numpy rotation matrices
    """
    handler = SymmetryHandler(dataset_name)
    return handler.get_symmetries(obj_id)
