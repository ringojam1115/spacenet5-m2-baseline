import os
import torch
import numpy as np
import rasterio
from pathlib import Path

from train.model import build_model

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

RGB_DIR  = Path("data/raw/AOI_7_Moscow/PS-RGB")
OUT_DIR  = Path("results/predictions")
WEIGHTS  = Path("results/best_model.pth")

def predict():
    # 1. モデルを読み込む
    model = build_model()
    model.load_state_dict(torch.load("results/best_model.pth", map_location="mps", weights_only=True))
    model.eval()
    model.to("mps")

    # 2. タイル一覧を取得
    tif_list = sorted(RGB_DIR.glob("*.tif"))

    if len(tif_list) == 0:
        print(f"No tif files found in: {RGB_DIR}")
        return
    
    print(f"Found {len(tif_list)} tiles")

    # 3. 1 枚ずつ推論して保存
    for tif_path in tif_list:
        print(f"Processing {tif_path.name} ... ", end="", flush=True)

        # ① 画像を読む
        with rasterio.open(tif_path) as src:
            image = src.read()  # shape: (3, 1300, 1300), dtype: uint8
            profile = src.profile

        # ② Tensor に変換して mps に転送
        image = image.astype(np.float32) / 255.0  # 正規化
        image = torch.from_numpy(image).unsqueeze(0).to("mps")  # shape: (1, 3, H, W)

        # ③ 推論
        with torch.no_grad():
            pred = model(image)  # shape: (1, 8, H, W)
            pred = torch.sigmoid(pred)  # 値域を 0〜1 に変換

        pred = pred.squeeze(0).cpu().numpy()  # shape: (8, H, W)

        # ④ GeoTIFF として保存
        out_path = OUT_DIR / tif_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        profile.update(
            dtype=rasterio.float32,
            count=8,
            compress="lzw",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred.astype(np.float32))

        print("done")

if __name__ == "__main__":
    predict()