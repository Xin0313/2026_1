"""
Data augmentation pipeline for SCHPose.
Includes ColorJitter, RandomErasing, Copy-Paste, Background Replace, Motion Blur, Gaussian Noise.
"""

import numpy as np
import cv2
import random
import os
from PIL import Image, ImageFilter, ImageEnhance


class SCHPoseAugmentation:
    """
    Complete augmentation pipeline for SCHPose training.

    Args:
        cfg: augmentation config dict with keys:
          color_jitter, random_erasing, copy_paste,
          background_replace, motion_blur, gaussian_noise
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cj_cfg = cfg.get("color_jitter", {})
        self.re_cfg = cfg.get("random_erasing", {})
        self.cp_cfg = cfg.get("copy_paste", False)
        self.bg_cfg = cfg.get("background_replace", {})
        self.mb_cfg = cfg.get("motion_blur", {})
        self.gn_cfg = cfg.get("gaussian_noise", {})

        # Load background images if available
        self.bg_images = []
        bg_dir = self.bg_cfg.get("bg_dir", None)
        if bg_dir and os.path.isdir(bg_dir):
            for fname in os.listdir(bg_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.bg_images.append(os.path.join(bg_dir, fname))

    def __call__(self, image, mask):
        """
        Apply augmentation to image and mask.

        Args:
            image: np.ndarray [H, W, 3] uint8 RGB
            mask: np.ndarray [H, W] uint8 binary mask (foreground=1)

        Returns:
            image, mask (same types)
        """
        image = image.copy()

        # 1. Color jitter
        image = self._color_jitter(image)

        # 2. Background replacement
        if self.bg_images and random.random() < self.bg_cfg.get("p", 0.3):
            image = self._replace_background(image, mask)

        # 3. Random erasing (simulate occlusion)
        if random.random() < self.re_cfg.get("p", 0.5):
            image, mask = self._random_erasing(image, mask)

        # 4. Motion blur
        if random.random() < self.mb_cfg.get("p", 0.2):
            image = self._motion_blur(image)

        # 5. Gaussian noise
        if random.random() < self.gn_cfg.get("p", 0.2):
            image = self._gaussian_noise(image)

        return image, mask

    def _color_jitter(self, image):
        """Apply random color jitter (brightness, contrast, saturation, hue)."""
        img = Image.fromarray(image)

        brightness = self.cj_cfg.get("brightness", 0.4)
        contrast = self.cj_cfg.get("contrast", 0.4)
        saturation = self.cj_cfg.get("saturation", 0.3)
        hue = self.cj_cfg.get("hue", 0.1)

        # Brightness
        if brightness > 0:
            factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        if contrast > 0:
            factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Saturation
        if saturation > 0:
            factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            img = ImageEnhance.Color(img).enhance(factor)

        # Hue (simulate with slight color shift in HSV)
        if hue > 0:
            arr = np.array(img).astype(np.float32)
            shift = random.uniform(-hue * 255, hue * 255)
            arr = np.clip(arr + shift, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return np.array(img)

    def _replace_background(self, image, mask):
        """Replace background pixels with a random background image."""
        if not self.bg_images:
            return image
        bg_path = random.choice(self.bg_images)
        try:
            bg = cv2.imread(bg_path)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]
            bg = cv2.resize(bg, (W, H))
            fg_mask = (mask > 0).astype(np.uint8)
            fg_mask_3c = np.stack([fg_mask] * 3, axis=-1)
            image = image * fg_mask_3c + bg * (1 - fg_mask_3c)
            return image.astype(np.uint8)
        except Exception:
            return image

    def _random_erasing(self, image, mask):
        """Randomly erase a rectangular region to simulate occlusion."""
        H, W = image.shape[:2]
        scale_range = self.re_cfg.get("scale", [0.02, 0.2])
        area = H * W
        erase_area = random.uniform(scale_range[0], scale_range[1]) * area
        aspect_ratio = random.uniform(0.3, 3.0)
        erase_h = int(np.sqrt(erase_area * aspect_ratio))
        erase_w = int(np.sqrt(erase_area / aspect_ratio))
        erase_h = min(erase_h, H)
        erase_w = min(erase_w, W)
        x = random.randint(0, W - erase_w)
        y = random.randint(0, H - erase_h)
        # Fill with random color
        fill_color = [random.randint(0, 255) for _ in range(3)]
        image[y:y + erase_h, x:x + erase_w] = fill_color
        mask[y:y + erase_h, x:x + erase_w] = 0
        return image, mask

    def _motion_blur(self, image):
        """Apply random motion blur."""
        ks_range = self.mb_cfg.get("kernel_size", [3, 7])
        ksize = random.choice(range(ks_range[0], ks_range[1] + 1, 2))
        angle = random.uniform(0, 360)
        # Create motion blur kernel
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = 1.0
        kernel /= ksize
        # Rotate kernel
        M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel /= kernel.sum() + 1e-8
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred.astype(np.uint8)

    def _gaussian_noise(self, image):
        """Add Gaussian noise to the image."""
        std_range = self.gn_cfg.get("std", [0.01, 0.05])
        std = random.uniform(std_range[0], std_range[1]) * 255
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation: paste objects from one image onto another
    to simulate occlusion.

    Args:
        paste_prob: probability of pasting
        max_objects: maximum number of objects to paste
    """

    def __init__(self, paste_prob=0.5, max_objects=3):
        self.paste_prob = paste_prob
        self.max_objects = max_objects
        self.object_bank = []  # list of (obj_rgb, obj_mask) tuples

    def add_to_bank(self, obj_rgb, obj_mask):
        """Add an object crop to the bank for future pasting."""
        self.object_bank.append((obj_rgb.copy(), obj_mask.copy()))
        if len(self.object_bank) > 500:
            self.object_bank.pop(0)

    def __call__(self, image, mask):
        """
        Args:
            image: np.ndarray [H, W, 3] uint8
            mask: np.ndarray [H, W] uint8 (foreground=1)

        Returns:
            augmented image, mask
        """
        if not self.object_bank or random.random() > self.paste_prob:
            return image, mask

        H, W = image.shape[:2]
        n_paste = random.randint(1, min(self.max_objects, len(self.object_bank)))

        for _ in range(n_paste):
            obj_rgb, obj_mask = random.choice(self.object_bank)
            oh, ow = obj_rgb.shape[:2]

            # Random scale
            scale = random.uniform(0.5, 1.5)
            new_h = max(10, int(oh * scale))
            new_w = max(10, int(ow * scale))
            obj_rgb_r = cv2.resize(obj_rgb, (new_w, new_h))
            obj_mask_r = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Random placement
            if new_h >= H or new_w >= W:
                continue
            x = random.randint(0, W - new_w)
            y = random.randint(0, H - new_h)

            paste_mask = (obj_mask_r > 0).astype(np.uint8)
            paste_mask_3c = np.stack([paste_mask] * 3, axis=-1)
            roi = image[y:y + new_h, x:x + new_w]
            roi[:] = roi * (1 - paste_mask_3c) + obj_rgb_r * paste_mask_3c

            # Update mask: pasted region overwrites foreground mask
            mask[y:y + new_h, x:x + new_w] = mask[y:y + new_h, x:x + new_w] * (1 - paste_mask)

        return image, mask
