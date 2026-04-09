"""
Surface codebook generation for SCHPose.

3D model surface is partitioned into a 2-level hierarchy:
  - n_coarse blocks (coarse partition via K-Means on mesh vertices)
  - Each coarse block divided into n_fine fine blocks
  - Each fine block has a local PCA-based (u,v) parameterization

Provides:
  - generate_codebook(mesh_path, n_coarse, n_fine) -> SurfaceCodebook
  - SurfaceCodebook.decode(c_id, f_id, u, v) -> 3D point
  - SurfaceCodebook.save(path) / load(path)
"""

import os
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans


class SurfaceCodebook:
    """
    Codebook for 2-level hierarchical surface parameterization.

    Attributes:
        n_coarse: number of coarse blocks
        n_fine: number of fine blocks per coarse block
        vertices: np.ndarray [N, 3] mesh vertices
        coarse_labels: np.ndarray [N] coarse block assignment
        fine_labels: np.ndarray [N, 2] (coarse_id, fine_id) assignments
        block_centers: dict mapping (c_id, f_id) -> center point
        block_axes: dict mapping (c_id, f_id) -> (u_axis [3], v_axis [3]) PCA axes
        block_scale: dict mapping (c_id, f_id) -> (u_scale, v_scale)
        sym_mappings: list of dicts {(c_id, f_id): (c_id', f_id')} for each symmetry
    """

    def __init__(self):
        self.n_coarse = 64
        self.n_fine = 16
        self.vertices = None
        self.coarse_labels = None
        self.fine_labels = None  # [N] fine block global id (c_id * n_fine + f_id)
        self.block_centers = {}
        self.block_axes = {}
        self.block_scale = {}
        self.sym_mappings = []
        self._coarse_fine_labels = None  # [N, 2]

    def generate(self, mesh_path, n_coarse=64, n_fine=16, sym_transforms=None):
        """
        Generate codebook from mesh.

        Args:
            mesh_path: path to .ply or .obj mesh file
            n_coarse: number of coarse blocks
            n_fine: number of fine blocks per coarse block
            sym_transforms: list of 4x4 numpy arrays representing symmetry transforms
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required: pip install trimesh")

        self.n_coarse = n_coarse
        self.n_fine = n_fine

        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.array(mesh.vertices, dtype=np.float64)
        self.vertices = vertices
        N = len(vertices)

        # Step 1: Coarse K-Means clustering
        n_coarse_eff = min(n_coarse, N)
        kmeans_coarse = MiniBatchKMeans(
            n_clusters=n_coarse_eff, random_state=42, n_init=3
        )
        coarse_labels = kmeans_coarse.fit_predict(vertices)
        self.coarse_labels = coarse_labels

        # Step 2: Fine K-Means within each coarse block
        fine_global_labels = np.full(N, -1, dtype=np.int32)
        self._coarse_fine_labels = np.full((N, 2), -1, dtype=np.int32)

        for c_id in range(n_coarse_eff):
            c_mask = coarse_labels == c_id
            c_verts = vertices[c_mask]
            c_indices = np.where(c_mask)[0]

            if len(c_verts) == 0:
                continue

            n_fine_eff = min(n_fine, len(c_verts))
            if n_fine_eff < 2:
                f_labels = np.zeros(len(c_verts), dtype=np.int32)
            else:
                kmeans_fine = MiniBatchKMeans(
                    n_clusters=n_fine_eff, random_state=42, n_init=3
                )
                f_labels = kmeans_fine.fit_predict(c_verts)

            for f_id in range(n_fine_eff):
                f_mask_local = f_labels == f_id
                f_verts = c_verts[f_mask_local]
                f_indices = c_indices[f_mask_local]

                if len(f_verts) == 0:
                    continue

                # Assign global labels
                fine_global_labels[f_indices] = c_id * n_fine + f_id
                self._coarse_fine_labels[f_indices, 0] = c_id
                self._coarse_fine_labels[f_indices, 1] = f_id

                # Compute PCA parameterization for this fine block
                center = f_verts.mean(axis=0)
                self.block_centers[(c_id, f_id)] = center

                if len(f_verts) >= 3:
                    centered = f_verts - center
                    try:
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        u_axis = Vt[0]
                        v_axis = Vt[1]
                        # Compute scale (std along each axis)
                        u_proj = centered @ u_axis
                        v_proj = centered @ v_axis
                        u_scale = max(u_proj.std(), 1e-6)
                        v_scale = max(v_proj.std(), 1e-6)
                    except np.linalg.LinAlgError:
                        u_axis = np.array([1.0, 0.0, 0.0])
                        v_axis = np.array([0.0, 1.0, 0.0])
                        u_scale = v_scale = 1e-3
                else:
                    u_axis = np.array([1.0, 0.0, 0.0])
                    v_axis = np.array([0.0, 1.0, 0.0])
                    u_scale = v_scale = 1e-3

                self.block_axes[(c_id, f_id)] = (u_axis, v_axis)
                self.block_scale[(c_id, f_id)] = (u_scale, v_scale)

        self.fine_labels = fine_global_labels

        # Step 3: Symmetry mappings
        if sym_transforms:
            self._build_sym_mappings(sym_transforms)

    def _build_sym_mappings(self, sym_transforms):
        """
        Build symmetry mappings between blocks.
        For each symmetry transform T, find which (c_id, f_id) maps to which.
        """
        self.sym_mappings = []
        for T in sym_transforms:
            R = T[:3, :3]
            t = T[:3, 3]
            mapping = {}
            for (c_id, f_id), center in self.block_centers.items():
                transformed = R @ center + t
                # Find nearest block center
                best_key = None
                best_dist = float("inf")
                for (c2, f2), center2 in self.block_centers.items():
                    dist = np.linalg.norm(transformed - center2)
                    if dist < best_dist:
                        best_dist = dist
                        best_key = (c2, f2)
                if best_key is not None:
                    mapping[(c_id, f_id)] = best_key
            self.sym_mappings.append(mapping)

    def decode(self, c_id, f_id, u, v):
        """
        Decode (coarse_id, fine_id, u, v) -> 3D point on model surface.

        Args:
            c_id: int or np.ndarray, coarse block id
            f_id: int or np.ndarray, fine block id
            u: float or np.ndarray in [0,1], local u coordinate
            v: float or np.ndarray in [0,1], local v coordinate

        Returns:
            point: np.ndarray [3] or [N, 3]
        """
        scalar = np.isscalar(c_id)
        if scalar:
            c_id = np.array([c_id])
            f_id = np.array([f_id])
            u = np.array([u])
            v = np.array([v])

        points = np.zeros((len(c_id), 3), dtype=np.float64)
        for i in range(len(c_id)):
            key = (int(c_id[i]), int(f_id[i]))
            if key not in self.block_centers:
                continue
            center = self.block_centers[key]
            u_axis, v_axis = self.block_axes[key]
            u_scale, v_scale = self.block_scale[key]
            # Map u,v from [0,1] to [-scale, scale]
            u_local = (float(u[i]) * 2.0 - 1.0) * u_scale
            v_local = (float(v[i]) * 2.0 - 1.0) * v_scale
            points[i] = center + u_local * u_axis + v_local * v_axis

        return points[0] if scalar else points

    def decode_batch(self, c_ids, f_ids, us, vs):
        """
        Batch decode using torch tensors or numpy arrays.

        Args:
            c_ids: [N] int array/tensor
            f_ids: [N] int array/tensor
            us: [N] float array/tensor in [0,1]
            vs: [N] float array/tensor in [0,1]

        Returns:
            points: np.ndarray [N, 3]
        """
        import torch
        if isinstance(c_ids, torch.Tensor):
            c_ids = c_ids.cpu().numpy()
            f_ids = f_ids.cpu().numpy()
            us = us.cpu().numpy()
            vs = vs.cpu().numpy()
        return self.decode(c_ids, f_ids, us, vs)

    def get_block_points(self, c_id, f_id):
        """Return all mesh vertices in block (c_id, f_id)."""
        if self._coarse_fine_labels is None:
            return np.array([])
        mask = (self._coarse_fine_labels[:, 0] == c_id) & (
            self._coarse_fine_labels[:, 1] == f_id
        )
        return self.vertices[mask]

    def save(self, path):
        """Save codebook to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        """Load codebook from file."""
        cb = cls()
        with open(path, "rb") as f:
            state = pickle.load(f)
        cb.__dict__.update(state)
        return cb

    def __repr__(self):
        return (
            f"SurfaceCodebook(n_coarse={self.n_coarse}, n_fine={self.n_fine}, "
            f"n_vertices={len(self.vertices) if self.vertices is not None else 0}, "
            f"n_blocks={len(self.block_centers)})"
        )
