"""
衛星画像 + 正解 + 予測グラフを並べて可視化する。

使い方:
    # chip0 を可視化
    python -m viz.overlay --chip 0

    # chip0〜chip4 をまとめて保存
    python -m viz.overlay --chip 0 1 2 3 4
"""

import argparse
import pickle
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import geopandas as gpd
from pathlib import Path

# --- パス設定 -----------------------------------------------------------------
RGB_DIR    = Path("data/raw/AOI_7_Moscow/PS-RGB")
GEO_DIR    = Path("data/raw/AOI_7_Moscow/geojson_roads_speed")
GRAPH_DIR  = Path("results/speed_graph")
OUT_DIR    = Path("results/viz")


def visualize(chip_id: int) -> None:
    """chip_id に対応する比較画像を生成して保存する。"""

    rgb_path   = RGB_DIR   / f"chip{chip_id}.tif"
    geo_path   = GEO_DIR   / f"chip{chip_id}.geojson"
    graph_path = GRAPH_DIR / f"chip{chip_id}.pkl"

    # ファイル存在チェック
    for p in [rgb_path, geo_path, graph_path]:
        if not p.exists():
            print(f"ファイルが見つかりません: {p}")
            return

    # --- 読み込み ---
    with rasterio.open(rgb_path) as src:
        rgb = src.read().transpose(1, 2, 0)   # (H, W, 3)
        transform = src.transform

    gdf = gpd.read_file(geo_path)

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # --- 描画 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = cm.get_cmap("RdYlGn")

    # 左: 元画像
    axes[0].imshow(rgb)
    axes[0].set_title(f"chip{chip_id}: PS-RGB")
    axes[0].axis("off")

    # 中: 正解 (GeoJSON)
    axes[1].imshow(rgb)
    for _, row in gdf.iterrows():
        coords = np.array(row.geometry.coords)
        px = ((coords[:, 0] - transform.c) / transform.a).astype(int)
        py = ((coords[:, 1] - transform.f) / transform.e).astype(int)
        axes[1].plot(px, py, color="lime", linewidth=2)
    axes[1].set_title(f"chip{chip_id}: 正解")
    axes[1].axis("off")

    # 右: 予測グラフ (速度カラー)
    axes[2].imshow(rgb)
    for u, v, data in G.edges(data=True):
        pts = data.get("pts", np.array([]))
        mph = data.get("speed_mph", 0)
        if len(pts) == 0:
            continue
        axes[2].plot(pts[:, 1], pts[:, 0],
                     color=cmap(mph / 65.0), linewidth=2)
    sm = plt.cm.ScalarMappable(cmap=cmap,
         norm=plt.Normalize(vmin=0, vmax=65))
    plt.colorbar(sm, ax=axes[2], label="Speed (mph)", shrink=0.7)
    axes[2].set_title(f"chip{chip_id}: 予測")
    axes[2].axis("off")

    # --- 保存 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"chip{chip_id}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", nargs="+", type=int, default=[0],
                        help="可視化する chip 番号 (複数指定可)")
    args = parser.parse_args()

    for chip_id in args.chip:
        visualize(chip_id)


if __name__ == "__main__":
    main()
