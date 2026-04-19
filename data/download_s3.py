"""
S3 から SpaceNet 5 Moscow の PS-RGB タイルと GeoJSON を取得する。
chip0 〜 chip199 の 200 枚だけダウンロードする。

使い方:
    python data/download_s3.py
"""

import subprocess
from pathlib import Path

BUCKET = "spacenet-dataset"
BASE = "spacenet/SN5_roads/train/AOI_7_Moscow"
OUT_DIR = Path(__file__).parent / "raw" / "AOI_7_Moscow"

RGB_DIR = OUT_DIR / "PS-RGB"
GEO_DIR = OUT_DIR / "geojson_roads_speed"
N_CHIPS = 200  # ダウンロードするタイル数


def s3_cp(s3_path: str, local_path: Path) -> None:
    if local_path.exists():
        return  # 再実行時にスキップ
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "cp",
        f"s3://{BUCKET}/{s3_path}",
        str(local_path),
        "--request-payer", "requester",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  SKIP (no file?): {s3_path}")


def main():
    print(f"Downloading {N_CHIPS} chips to {OUT_DIR}\n")
    for i in range(N_CHIPS):
        rgb_key = f"{BASE}/PS-RGB/SN5_roads_train_AOI_7_Moscow_PS-RGB_chip{i}.tif"
        geo_key = f"{BASE}/geojson_roads_speed/SN5_roads_train_AOI_7_Moscow_geojson_roads_speed_chip{i}.geojson"

        rgb_path = RGB_DIR / f"chip{i}.tif"
        geo_path = GEO_DIR / f"chip{i}.geojson"

        print(f"[{i+1:3d}/{N_CHIPS}] chip{i}", end=" ... ", flush=True)
        s3_cp(rgb_key, rgb_path)
        s3_cp(geo_key, geo_path)
        print("done")

    print(f"\n完了: {RGB_DIR}")


if __name__ == "__main__":
    main()
