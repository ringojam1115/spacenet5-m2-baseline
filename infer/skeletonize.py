import numpy as np
import rasterio
from pathlib import Path
from skimage.morphology import skeletonize

INP_DIR  = Path("results/predictions")
OUT_DIR  = Path("results/skeletonized")

def skeletonize_predictions() -> None:
    # 1. タイル一覧を取得
    tif_list = sorted(INP_DIR.glob("*.tif"))

    if len(tif_list) == 0:
        print(f"No tif files found in: {INP_DIR}")
        return
    
    print(f"Found {len(tif_list)} tiles")

    # 2. 1 枚ずつ処理して保存
    for tif_path in tif_list:
        print(f"Processing {tif_path.name} ... ", end="", flush=True)

        # ① 予測マスクを読む
        with rasterio.open(tif_path) as src:
            pred_mask = src.read()  # shape: (8, H, W), dtype: float32
            profile = src.profile

        # ② スケルトン化 (細線化)
        skeleton_mask = np.zeros_like(pred_mask)
        for i in range(pred_mask.shape[0]):
            skeleton_mask[i] = skeletonize(pred_mask[i] > 0.3).astype(np.float32)

        # ③ GeoTIFF として保存
        out_path = OUT_DIR / tif_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        profile.update(
            dtype=rasterio.float32,
            count=8,
            compress="lzw",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(skeleton_mask)

        print("done")

if __name__ == "__main__":
    skeletonize_predictions()