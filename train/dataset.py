import numpy as np
import rasterio
import random
import torch
from pathlib import Path

class SpaceNetDataset:
    def __init__(self, rgb_dir, mask_dir, crop_size=256, augment=False) -> None:
        # Initialize the dataset with the directories for RGB images and masks, and the desired crop size
        self.crop_size = crop_size
        self.augment = augment

        rgb_dir = Path(rgb_dir)
        mask_dir = Path(mask_dir)

        self.pairs = []
        for rgb_path in rgb_dir.glob("*.tif"):
            for rgb_path in sorted(rgb_dir.glob("*.tif")):
                mask_path = mask_dir / rgb_path.name   # chip0.tif → chip0.tif
            if mask_path.exists():
                self.pairs.append((rgb_path, mask_path))

    def __len__(self) -> int:
        # Return the number of samples in the dataset
        return len(self.pairs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # Return a single sample from the dataset at the given index
        rgb_path, mask_path = self.pairs[idx]

        # ① Read the RGB image and mask using rasterio
        with rasterio.open(rgb_path) as rgb_src:
            image = rgb_src.read()  # shape: (3, 1300, 1300)  dtype: uint8
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read()  # shape: (8, 1300, 1300)  dtype: uint8

        # ② Crop the image and mask into smaller patches of size (crop_size, crop_size)
        # 画像とマスク両方とも同じ位置でクロップする必要があります。
        top  = random.randint(0, 1300 - 256)
        left = random.randint(0, 1300 - 256)

        image = image[:, top:top+256, left:left+256]  # (3, 256, 256)
        mask = mask[:, top:top+256, left:left+256]  # (8, 256, 256)

        # ③ 正規化
        image = image.astype(np.float32) / 255.0
        # 0〜255 → 0.0〜1.0

        mask  = (mask > 0).astype(np.float32)
        # 0 or 255 → 0.0 or 1.0  (二値化)

        # ④ Numpy 配列を PyTorch のテンソルに変換
        image = torch.from_numpy(image)  # (3, 256, 256)
        mask  = torch.from_numpy(mask)   # (8, 256, 256)

        # ⑤ 左右反転 (augment=True のときだけ)
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[2])  # 横方向に反転
            mask  = torch.flip(mask, dims=[1])   # 横方向に反転

        return image, mask