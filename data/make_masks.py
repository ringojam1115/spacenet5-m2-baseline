"""
GeoJSON (道路 + 速度ラベル) → 8ch 速度マスク GeoTIFF に変換する。

チャンネル構成:
  ch 0:  1〜10  mph
  ch 1: 11〜20  mph
  ch 2: 21〜30  mph
  ch 3: 31〜40  mph
  ch 4: 41〜50  mph
  ch 5: 51〜60  mph
  ch 6: 61〜65  mph
  ch 7: 全道路 (ch0〜ch6 の OR)

使い方:
  python data/make_masks.py
"""

import math
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm

# ----- パス設定 ---------------------------------------------------------------
BASE     = Path(__file__).parent / "raw" / "AOI_7_Moscow"
RGB_DIR  = BASE / "PS-RGB"
GEO_DIR  = BASE / "geojson_roads_speed"
MASK_DIR = Path(__file__).parent / "masks" / "AOI_7_Moscow"

# モスクワの UTM ゾーン(バッファ計算をメートル単位でするため)
UTM_CRS = "EPSG:32637"

# 道路の半幅 [m]
BUFFER_METERS = 2

# マスクの画素値
BURN_VALUE = 255

N_SPEED_BINS = 7   # 速度ビン数
N_CHANNELS   = 8   # 速度ビン 7 + 全道路 1


# ----- 速度 → チャンネル番号 --------------------------------------------------
def speed_to_channel(speed_mph: float) -> int:
    """速度(mph)を 0〜6 のチャンネル番号に変換する。

    ロジック: ceil(speed / 10) - 1
      例: 15 mph → ceil(15/10)=2 → ch 1
          30 mph → ceil(30/10)=3 → ch 2
          65 mph → ceil(65/10)=7 → ch 6 (上限クランプ)
    """
    ch = int(math.ceil(speed_mph / 10.0)) - 1
    return min(max(ch, 0), N_SPEED_BINS - 1)


# ----- 1 タイル分のマスク生成 -------------------------------------------------
def make_mask(image_path: Path, geojson_path: Path, output_path: Path) -> None:
    """PS-RGB タイル 1 枚に対応する 8ch マスクを生成して保存する。"""

    # 画像のメタ情報だけ取得
    with rasterio.open(image_path) as src:
        transform = src.transform
        h, w = src.height, src.width
        crs = src.crs

    # 8ch マスクを全ゼロで初期化 (チャンネル, 高さ, 幅)
    mask = np.zeros((N_CHANNELS, h, w), dtype=np.uint8)

    gdf = gpd.read_file(geojson_path)

    if len(gdf) == 0:
        _save_mask(mask, output_path, transform, crs)
        return

    # --- バッファ処理 ---
    # EPSG:4326 は度単位なのでメートルバッファのために一時的に UTM へ変換
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_utm = gdf.to_crs(UTM_CRS).copy()
        gdf_utm["geometry"] = gdf_utm.geometry.buffer(BUFFER_METERS, cap_style=2)
        gdf_back = gdf_utm.to_crs(crs)

    # --- チャンネル別にラスタライズ ---
    for _, row in gdf_back.iterrows():
        speed = row.get("inferred_speed_mph")
        if speed is None or (isinstance(speed, float) and math.isnan(speed)):
            continue

        ch = speed_to_channel(float(speed))
        burned = rasterize(
            [(row.geometry.__geo_interface__, BURN_VALUE)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        mask[ch] = np.maximum(mask[ch], burned)

    # ch7 = 全道路チャンネル
    mask[7] = np.where(mask[:7].max(axis=0) > 0, BURN_VALUE, 0).astype(np.uint8)

    _save_mask(mask, output_path, transform, crs)


# ----- GeoTIFF 保存 -----------------------------------------------------------
def _save_mask(mask: np.ndarray, output_path: Path, transform, crs) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=mask.shape[1],
        width=mask.shape[2],
        count=N_CHANNELS,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(mask)


# ----- メインループ -----------------------------------------------------------
def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(RGB_DIR.glob("*.tif"))
    print(f"{len(tif_files)} タイルのマスクを生成します → {MASK_DIR}\n")

    for tif_path in tqdm(tif_files):
        chip_id  = tif_path.stem
        geo_path = GEO_DIR / f"{chip_id}.geojson"
        out_path = MASK_DIR / f"{chip_id}.tif"

        if not geo_path.exists():
            tqdm.write(f"  SKIP (geojson なし): {chip_id}")
            continue

        make_mask(tif_path, geo_path, out_path)

    print(f"\n完了: {MASK_DIR}")


if __name__ == "__main__":
    main()
