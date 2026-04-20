# infer/speed.py

import pickle
import numpy as np
import rasterio
from pathlib import Path

GRAPH_DIR = Path("results/graph")
PRED_DIR  = Path("results/predictions")
OUT_DIR   = Path("results/speed_graph")

BIN_TO_MPH = {
    0:  5.0,
    1: 15.0,
    2: 25.0,
    3: 35.0,
    4: 45.0,
    5: 55.0,
    6: 62.5,
}

def infer_speed() -> None:
    pkl_list = sorted(GRAPH_DIR.glob("*.pkl"))

    if len(pkl_list) == 0:
        print(f"No pkl files found in: {GRAPH_DIR}")
        return

    print(f"Found {len(pkl_list)} graphs")

    for pkl_path in pkl_list:
        print(f"Processing {pkl_path.name} ... ", end="", flush=True)

        # ① グラフを読み込む
        with open(pkl_path, "rb") as f:
            G = pickle.load(f)

        # ② 対応する予測マスクを読み込む
        #    chip0.pkl → chip0.tif
        pred_path = PRED_DIR / pkl_path.with_suffix(".tif").name
        with rasterio.open(pred_path) as src:
            pred = src.read()   # shape: (8, H, W)

        # ③ エッジごとに速度を推定して G に追加する
        for u, v, data in G.edges(data=True):
            pts = data["pts"]        # shape: (N, 2), 各点の (y, x)
            ys  = pts[:, 0]
            xs  = pts[:, 1]

            # pred の ch0〜ch6 から pts の座標の値を取り出す
            pts_pred = pred[:7, ys, xs]  # shape: (7, N)        

            # チャンネル方向に平均 → argmax でビン番号 → mph に変換
            avg_pred = pts_pred.mean(axis=1)
            bin_indices = np.argmax(avg_pred)
            mph = BIN_TO_MPH[bin_indices]

            # G[u][v]["speed_mph"] = mph に代入する
            G[u][v]["speed_mph"] = mph

        # ④ 速度付きグラフを保存する
        out_path = OUT_DIR / pkl_path.name

        # out_path を作って pickle で保存する
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(G, f)

        print("done")

if __name__ == "__main__":
    infer_speed()