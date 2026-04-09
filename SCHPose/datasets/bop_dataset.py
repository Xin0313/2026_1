"""
BOP format dataset loader for SCHPose.
Supports YCB-Video, Linemod-Occluded, and T-LESS datasets.

BOP format structure:
  <dataset_root>/
    models/
      obj_<id>.ply
    train_pbr/ or train_real/
      <scene_id>/
        rgb/
          <img_id>.jpg / .png
        depth/
          <img_id>.png
        mask/
          <img_id>_<obj_idx>.png
        mask_visib/
          <img_id>_<obj_idx>.png
        scene_gt.json
        scene_camera.json
        scene_gt_info.json
    test/
      <scene_id>/  (same structure)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


class BOPDataset(Dataset):
    """
    BOP format dataset loader.

    Args:
        cfg: configuration dict/namespace
        split: 'train' or 'test'
        augmentation: augmentation object (SCHPoseAugmentation or None)
    """

    def __init__(self, cfg, split="train", augmentation=None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.augmentation = augmentation

        self.root = cfg["dataset"]["root"]
        self.object_ids = cfg["dataset"]["object_ids"]
        self.num_objects = cfg["dataset"]["num_objects"]
        self.image_size = cfg["dataset"]["image_size"]   # [H, W]
        self.codebook_dir = cfg["dataset"].get("codebook_dir", None)
        self.symmetric_obj_ids = set(cfg["dataset"].get("symmetric_obj_ids", []))

        # Determine data split directories
        if split == "train":
            self.data_dirs = self._find_split_dirs(["train_pbr", "train_real", "train"])
        else:
            self.data_dirs = self._find_split_dirs(["test"])

        # Build sample list: each sample = (scene_dir, img_id, obj_idx, obj_id)
        self.samples = self._build_sample_list()

    def _find_split_dirs(self, candidates):
        """Find all scene directories under candidate split folders."""
        dirs = []
        for cand in candidates:
            split_dir = os.path.join(self.root, cand)
            if os.path.isdir(split_dir):
                for scene in sorted(os.listdir(split_dir)):
                    scene_path = os.path.join(split_dir, scene)
                    if os.path.isdir(scene_path):
                        dirs.append(scene_path)
        return dirs

    def _build_sample_list(self):
        """Build flat list of (scene_dir, img_id, obj_idx, obj_id) tuples."""
        samples = []
        for scene_dir in self.data_dirs:
            gt_path = os.path.join(scene_dir, "scene_gt.json")
            gt_info_path = os.path.join(scene_dir, "scene_gt_info.json")
            if not os.path.isfile(gt_path):
                continue
            with open(gt_path, "r") as f:
                scene_gt = json.load(f)
            gt_info = {}
            if os.path.isfile(gt_info_path):
                with open(gt_info_path, "r") as f:
                    gt_info = json.load(f)

            for img_id_str, annots in scene_gt.items():
                img_id = int(img_id_str)
                for obj_idx, annot in enumerate(annots):
                    obj_id = annot["obj_id"]
                    if obj_id not in self.object_ids:
                        continue
                    # Filter by visibility if gt_info is available
                    if gt_info:
                        info_list = gt_info.get(img_id_str, [])
                        if obj_idx < len(info_list):
                            visib_frac = info_list[obj_idx].get("visib_fract", 1.0)
                            if visib_frac < 0.1:
                                continue
                    samples.append((scene_dir, img_id, obj_idx, obj_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_dir, img_id, obj_idx, obj_id = self.samples[idx]

        # Load RGB image
        img = self._load_image(scene_dir, img_id)
        H, W = self.image_size

        # Load camera intrinsics
        K = self._load_camera(scene_dir, img_id)

        # Load ground-truth pose
        R, t = self._load_pose(scene_dir, img_id, obj_idx)

        # Load masks
        mask_visib = self._load_mask(scene_dir, img_id, obj_idx, kind="visib")
        mask_full = self._load_mask(scene_dir, img_id, obj_idx, kind="full")

        # Resize to target resolution
        orig_h, orig_w = img.shape[:2]
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
            scale_x = W / orig_w
            scale_y = H / orig_h
            K = K.copy()
            K[0] *= scale_x
            K[1] *= scale_y
            if mask_visib is not None:
                mask_visib = cv2.resize(mask_visib, (W, H), interpolation=cv2.INTER_NEAREST)
            if mask_full is not None:
                mask_full = cv2.resize(mask_full, (W, H), interpolation=cv2.INTER_NEAREST)

        # Apply augmentation
        if self.augmentation is not None and self.split == "train":
            img, mask_visib = self.augmentation(img, mask_visib)

        # Object label index (0-indexed in object_ids list)
        obj_label = self.object_ids.index(obj_id)

        # Convert to tensors
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        K_tensor = torch.from_numpy(K.astype(np.float32))
        R_tensor = torch.from_numpy(R.astype(np.float32))
        t_tensor = torch.from_numpy(t.astype(np.float32))

        if mask_visib is not None:
            mask_tensor = torch.from_numpy(mask_visib.astype(np.float32))
        else:
            mask_tensor = torch.zeros(H, W, dtype=torch.float32)

        return {
            "image": img_tensor,                      # [3, H, W]
            "K": K_tensor,                            # [3, 3]
            "R": R_tensor,                            # [3, 3]
            "t": t_tensor,                            # [3]
            "mask": mask_tensor,                      # [H, W]
            "obj_id": obj_id,
            "obj_label": obj_label,
            "is_symmetric": obj_id in self.symmetric_obj_ids,
            "scene_dir": scene_dir,
            "img_id": img_id,
            "obj_idx": obj_idx,
        }

    def _load_image(self, scene_dir, img_id):
        """Load RGB image."""
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(scene_dir, "rgb", f"{img_id:06d}{ext}")
            if os.path.isfile(path):
                img = cv2.imread(path)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"Image not found in {scene_dir}/rgb/ for id {img_id}")

    def _load_camera(self, scene_dir, img_id):
        """Load camera intrinsics K as 3x3 numpy array."""
        cam_path = os.path.join(scene_dir, "scene_camera.json")
        with open(cam_path, "r") as f:
            cams = json.load(f)
        cam = cams[str(img_id)]
        K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
        return K

    def _load_pose(self, scene_dir, img_id, obj_idx):
        """Load rotation matrix R and translation t for the given object."""
        gt_path = os.path.join(scene_dir, "scene_gt.json")
        with open(gt_path, "r") as f:
            scene_gt = json.load(f)
        annot = scene_gt[str(img_id)][obj_idx]
        R = np.array(annot["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t = np.array(annot["cam_t_m2c"], dtype=np.float64) / 1000.0  # mm -> m
        return R, t

    def _load_mask(self, scene_dir, img_id, obj_idx, kind="visib"):
        """Load binary mask. kind='visib' or 'full'."""
        folder = "mask_visib" if kind == "visib" else "mask"
        path = os.path.join(scene_dir, folder, f"{img_id:06d}_{obj_idx:06d}.png")
        if not os.path.isfile(path):
            return None
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.uint8)
