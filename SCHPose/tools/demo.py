"""
Single-image inference demo for SCHPose.

Usage:
    python tools/demo.py --config configs/ycbv.yaml \
                         --checkpoint outputs/ycbv/best_checkpoint.pth \
                         --image path/to/image.jpg \
                         --K "fx,fy,cx,cy" \
                         --obj_id 1
"""

import os
import sys
import argparse
import yaml
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.surface_codebook import SurfaceCodebook
from models.schpose import SCHPose
from utils.visualization import draw_pose, draw_segmentation, overlay_mask
from utils.symmetry import get_symmetry_transforms


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SCHPose demo")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--K", type=str, default=None,
                        help="Camera intrinsics as 'fx,fy,cx,cy'")
    parser.add_argument("--obj_id", type=int, default=1)
    parser.add_argument("--output", type=str, default="demo_output.jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # Load codebooks
    codebook_dir = cfg["dataset"].get("codebook_dir", None)
    codebooks = {}
    if codebook_dir:
        path = os.path.join(codebook_dir, f"obj_{args.obj_id:06d}_codebook.pkl")
        if os.path.isfile(path):
            codebooks[args.obj_id] = SurfaceCodebook.load(path)

    # Load symmetry transforms
    sym_transforms = {}
    st = get_symmetry_transforms(cfg["dataset"]["name"], args.obj_id)
    if st:
        sym_transforms[args.obj_id] = st

    # Build model
    model = SCHPose(cfg, codebooks=codebooks, sym_transforms=sym_transforms)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Error: cannot load image {args.image}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = img_rgb.shape[:2]

    # Camera intrinsics
    if args.K:
        fx, fy, cx, cy = [float(v) for v in args.K.split(",")]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        # Default: assume 80deg FOV
        fx = fy = max(H, W) * 1.2
        cx, cy = W / 2, H / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        print(f"No K provided, using default: fx={fx:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Resize to model input size
    in_H, in_W = cfg["dataset"]["image_size"]
    img_resized = cv2.resize(img_rgb, (in_W, in_H))
    scale_x = in_W / W
    scale_y = in_H / H
    K_scaled = K.copy()
    K_scaled[0] *= scale_x
    K_scaled[1] *= scale_y

    img_tensor = torch.from_numpy(
        img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0).to(device)
    K_tensor = torch.from_numpy(K_scaled).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        R, t, seg_mask = model.inference(img_tensor, K_tensor, obj_id=args.obj_id)

    R_np = R.cpu().numpy()
    t_np = t.cpu().numpy()
    seg_np = seg_mask.cpu().numpy().astype(np.uint8)

    print(f"\nPredicted pose for object {args.obj_id}:")
    print(f"  R =\n{R_np}")
    print(f"  t = {t_np}")

    # Visualization
    seg_up = cv2.resize(seg_np, (in_W, in_H), interpolation=cv2.INTER_NEAREST)
    vis = overlay_mask(img_resized, seg_up, color=(0, 255, 0), alpha=0.4)
    vis = draw_pose(vis, R_np, t_np, K_scaled)

    cv2.imwrite(args.output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()
