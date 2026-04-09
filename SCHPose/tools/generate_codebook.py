"""
Generate surface codebooks for SCHPose objects.

Usage:
    python tools/generate_codebook.py --config configs/ycbv.yaml

This script:
  1. Loads mesh for each object
  2. Runs 2-level K-Means clustering (n_coarse x n_fine)
  3. Builds PCA local parameterization
  4. Adds symmetry block mappings
  5. Saves codebook to disk
"""

import os
import sys
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.surface_codebook import SurfaceCodebook
from utils.symmetry import get_symmetry_transforms


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_mesh_path(dataset_root, obj_id):
    """Find mesh file for object id."""
    candidates = [
        os.path.join(dataset_root, "models", f"obj_{obj_id:06d}.ply"),
        os.path.join(dataset_root, "models", f"obj_{obj_id:06d}.obj"),
        os.path.join(dataset_root, "models_eval", f"obj_{obj_id:06d}.ply"),
        os.path.join(dataset_root, "models", f"obj_{obj_id:02d}.ply"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate SCHPose surface codebooks")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--n_coarse", type=int, default=None, help="Override n_coarse")
    parser.add_argument("--n_fine", type=int, default=None, help="Override n_fine")
    parser.add_argument("--obj_ids", type=int, nargs="+", default=None,
                        help="Specific object ids to process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    dataset_root = dataset_cfg["root"]
    codebook_dir = dataset_cfg["codebook_dir"]
    dataset_name = dataset_cfg["name"]
    object_ids = args.obj_ids or dataset_cfg["object_ids"]
    n_coarse = args.n_coarse or model_cfg.get("n_coarse", 64)
    n_fine = args.n_fine or model_cfg.get("n_fine", 16)

    os.makedirs(codebook_dir, exist_ok=True)

    print(f"Generating codebooks for {len(object_ids)} objects")
    print(f"  Dataset: {dataset_name}")
    print(f"  n_coarse: {n_coarse}, n_fine: {n_fine}")
    print(f"  Output dir: {codebook_dir}")

    for obj_id in object_ids:
        print(f"\n[{obj_id}] Processing object {obj_id}...")

        mesh_path = find_mesh_path(dataset_root, obj_id)
        if mesh_path is None:
            print(f"  WARNING: No mesh found for obj_id={obj_id}, skipping.")
            continue

        print(f"  Mesh: {mesh_path}")

        # Get symmetry transforms
        sym_transforms = get_symmetry_transforms(dataset_name, obj_id)
        if sym_transforms:
            print(f"  Symmetry transforms: {len(sym_transforms)}")

        # Generate codebook
        codebook = SurfaceCodebook()
        try:
            codebook.generate(
                mesh_path=mesh_path,
                n_coarse=n_coarse,
                n_fine=n_fine,
                sym_transforms=[
                    np.vstack([
                        np.hstack([R, np.zeros((3, 1))]),
                        [0, 0, 0, 1]
                    ])
                    for R in sym_transforms
                ] if sym_transforms else None,
            )
            print(f"  Generated: {codebook}")
        except Exception as e:
            print(f"  ERROR generating codebook: {e}")
            continue

        # Save
        save_path = os.path.join(codebook_dir, f"obj_{obj_id:06d}_codebook.pkl")
        codebook.save(save_path)
        print(f"  Saved: {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
