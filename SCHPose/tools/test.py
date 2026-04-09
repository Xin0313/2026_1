"""
Testing script for SCHPose.

Loads a checkpoint, runs inference on the test set,
computes ADD/ADD-S/AUC metrics, and outputs BOP-format results.

Usage:
    python tools/test.py --config configs/ycbv.yaml --checkpoint outputs/ycbv/best_checkpoint.pth
"""

import os
import sys
import argparse
import yaml
import json
import csv
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.bop_dataset import BOPDataset
from datasets.surface_codebook import SurfaceCodebook
from models.schpose import SCHPose
from utils.metrics import PoseMetrics, compute_add, compute_add_s
from utils.symmetry import get_symmetry_transforms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("schpose.test")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_codebooks(cfg):
    codebook_dir = cfg["dataset"].get("codebook_dir", None)
    if not codebook_dir:
        return {}
    object_ids = cfg["dataset"]["object_ids"]
    codebooks = {}
    for obj_id in object_ids:
        path = os.path.join(codebook_dir, f"obj_{obj_id:06d}_codebook.pkl")
        if os.path.isfile(path):
            codebooks[obj_id] = SurfaceCodebook.load(path)
    logger.info(f"Loaded {len(codebooks)}/{len(object_ids)} codebooks")
    return codebooks


def load_model_pts(cfg):
    """Load model 3D points for each object."""
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not available, model_pts not loaded.")
        return {}, {}

    dataset_root = cfg["dataset"]["root"]
    object_ids = cfg["dataset"]["object_ids"]
    model_pts = {}
    obj_diameters = {}

    for obj_id in object_ids:
        for ext in [".ply", ".obj"]:
            mesh_path = os.path.join(dataset_root, "models", f"obj_{obj_id:06d}{ext}")
            if not os.path.isfile(mesh_path):
                mesh_path = os.path.join(dataset_root, "models", f"obj_{obj_id:02d}{ext}")
            if os.path.isfile(mesh_path):
                mesh = trimesh.load(mesh_path, force="mesh")
                pts = np.array(mesh.vertices, dtype=np.float64)
                # Subsample for efficiency
                if len(pts) > 1000:
                    idx = np.random.choice(len(pts), 1000, replace=False)
                    pts = pts[idx]
                # Convert mm -> m if needed (BOP uses mm)
                if pts.max() > 10:
                    pts = pts / 1000.0
                model_pts[obj_id] = pts
                # Compute diameter
                from scipy.spatial.distance import cdist
                if len(pts) > 100:
                    sample = pts[np.random.choice(len(pts), 100, replace=False)]
                    diam = cdist(sample, sample).max()
                else:
                    diam = cdist(pts, pts).max()
                obj_diameters[obj_id] = float(diam)
                break

    logger.info(f"Loaded model points for {len(model_pts)} objects")
    return model_pts, obj_diameters


def run_inference(model, dataloader, device, cfg):
    """
    Run inference on dataloader.

    Returns:
        results: list of dicts with pred_R, pred_t, gt_R, gt_t, obj_id, etc.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch["image"].to(device)
            K = batch["K"].to(device)
            obj_ids = batch["obj_id"]
            R_gt = batch["R"].numpy()
            t_gt = batch["t"].numpy()

            output = model(
                images, K,
                obj_ids=list(obj_ids) if isinstance(obj_ids, torch.Tensor) else obj_ids,
                phase=2,
            )

            poses = output["poses"]
            B = images.shape[0]

            for b in range(B):
                if b < len(poses):
                    R_pred, t_pred = poses[b]
                    R_pred_np = R_pred.cpu().numpy()
                    t_pred_np = t_pred.cpu().numpy()
                else:
                    R_pred_np = np.eye(3)
                    t_pred_np = np.zeros(3)

                oid = obj_ids[b] if isinstance(obj_ids, (list, tuple)) else int(obj_ids[b])
                results.append({
                    "obj_id": oid,
                    "R_pred": R_pred_np,
                    "t_pred": t_pred_np,
                    "R_gt": R_gt[b],
                    "t_gt": t_gt[b],
                    "scene_dir": batch["scene_dir"][b] if isinstance(batch["scene_dir"], (list, tuple)) else batch["scene_dir"],
                    "img_id": int(batch["img_id"][b]) if isinstance(batch["img_id"], torch.Tensor) else batch["img_id"][b],
                })

    return results


def compute_metrics(results, model_pts, obj_diameters, symmetric_obj_ids,
                    sym_transforms_db, add_threshold=0.1):
    """Compute ADD/ADD-S metrics from results."""
    pose_metrics = PoseMetrics(
        obj_diameters=obj_diameters,
        symmetric_obj_ids=symmetric_obj_ids,
        add_threshold=add_threshold,
    )

    for r in results:
        obj_id = r["obj_id"]
        if obj_id not in model_pts:
            continue
        pts = model_pts[obj_id]
        sym_t = sym_transforms_db.get(obj_id, None)
        pose_metrics.update(
            obj_id,
            r["R_pred"], r["t_pred"],
            r["R_gt"], r["t_gt"],
            pts, sym_transforms=sym_t,
        )

    return pose_metrics.summarize()


def write_bop_results(results, output_path):
    """
    Write results in BOP CSV format.
    Format: scene_id,im_id,obj_id,score,R,t,time
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])

        for r in results:
            scene_id = os.path.basename(r.get("scene_dir", "000000"))
            try:
                scene_id = int(scene_id)
            except ValueError:
                scene_id = 0

            R_flat = " ".join(f"{v:.8f}" for v in r["R_pred"].flatten())
            t_mm = r["t_pred"] * 1000.0  # m -> mm for BOP format
            t_flat = " ".join(f"{v:.8f}" for v in t_mm)

            writer.writerow([
                scene_id,
                r["img_id"],
                r["obj_id"],
                1.0,
                R_flat,
                t_flat,
                -1,
            ])

    logger.info(f"BOP results written to {output_path}")


def print_metrics_table(metrics):
    """Print formatted metrics table."""
    print("\n" + "=" * 70)
    print(f"{'Object':<20} {'ADD-Acc':<12} {'ADD-AUC':<12} {'Mean-ADD':<12} {'N':<6}")
    print("-" * 70)

    for obj_id, m in metrics.items():
        if obj_id == "mean":
            continue
        print(
            f"obj_{obj_id:<16} {m['add_accuracy']:.4f}       "
            f"{m['add_auc']:.4f}       {m['mean_add']:.4f}       {m['n_samples']}"
        )

    if "mean" in metrics:
        m = metrics["mean"]
        print("-" * 70)
        print(
            f"{'MEAN':<20} {m['add_accuracy']:.4f}       "
            f"{m['add_auc']:.4f}       {m['mean_add']:.4f}       {m['n_samples']}"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test SCHPose")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    output_dir = Path(args.output_dir or cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Load codebooks and symmetry
    codebooks = load_codebooks(cfg)
    sym_transforms_db = {}
    dataset_name = cfg["dataset"]["name"]
    for obj_id in cfg["dataset"]["object_ids"]:
        st = get_symmetry_transforms(dataset_name, obj_id)
        if st:
            sym_transforms_db[obj_id] = st

    # Build model
    model = SCHPose(cfg, codebooks=codebooks, sym_transforms=sym_transforms_db)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load model points and diameters
    model_pts, obj_diameters = load_model_pts(cfg)

    # Build test dataset
    test_dataset = BOPDataset(cfg, split="test", augmentation=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Run inference
    t_start = time.time()
    results = run_inference(model, test_loader, device, cfg)
    t_elapsed = time.time() - t_start
    logger.info(f"Inference done: {len(results)} predictions in {t_elapsed:.1f}s "
                f"({len(results)/t_elapsed:.1f} pred/s)")

    # Write BOP results
    bop_path = str(output_dir / "bop_results.csv")
    write_bop_results(results, bop_path)

    # Compute metrics
    if model_pts:
        symmetric_obj_ids = set(cfg["dataset"].get("symmetric_obj_ids", []))
        add_threshold = cfg.get("evaluation", {}).get("add_threshold", 0.1)
        metrics = compute_metrics(
            results, model_pts, obj_diameters,
            symmetric_obj_ids, sym_transforms_db,
            add_threshold=add_threshold,
        )
        print_metrics_table(metrics)

        # Save metrics to JSON
        metrics_path = str(output_dir / "metrics.json")
        with open(metrics_path, "w") as f:
            # Convert keys to strings for JSON
            metrics_str = {str(k): v for k, v in metrics.items()}
            json.dump(metrics_str, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.warning("No model points available, skipping metric computation.")


if __name__ == "__main__":
    main()
