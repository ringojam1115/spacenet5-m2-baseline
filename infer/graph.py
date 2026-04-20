import numpy as np
import rasterio
import pickle
from pathlib import Path
import sknw

INP_DIR  = Path("results/skeletonized")
OUT_DIR  = Path("results/graph")

def graph_from_skeletons() -> None:
    # 1. タイル一覧を取得
    tif_list = sorted(INP_DIR.glob("*.tif"))

    if len(tif_list) == 0:
        print(f"No tif files found in: {INP_DIR}")
        return
    
    print(f"Found {len(tif_list)} tiles")

    # 2. 1 枚ずつ処理して保存
    for tif_path in tif_list:
        print(f"Processing {tif_path.name} ... ", end="", flush=True)

        # ① スケルトン化されたマスクを読む
        with rasterio.open(tif_path) as src:
            skeleton_mask = src.read()  # shape: (8, H, W), dtype: float32

        # ② グラフ構造に変換
        graph = sknw.build_sknw(skeleton_mask[7].astype(np.uint16))  # 1 チャンネル目を使用

        # ③ ピクルファイルとして保存
        out_path = OUT_DIR / tif_path.with_suffix(".pkl").name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "wb") as f:
            pickle.dump(graph, f)

        print("done")

if __name__ == "__main__":
    graph_from_skeletons()