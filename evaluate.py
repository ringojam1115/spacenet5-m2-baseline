"""
予測マスクと正解マスクを比較して精度指標を出力する。

使い方:
    python evaluate.py
    python evaluate.py --threshold 0.5
"""

import argparse
import numpy as np
import rasterio
from pathlib import Path

PRED_DIR = Path("results/predictions")
MASK_DIR = Path("data/masks/AOI_7_Moscow")


def evaluate(threshold: float = 0.3) -> None:
    pred_paths = sorted(PRED_DIR.glob("*.tif"))

    if len(pred_paths) == 0:
        print(f"予測ファイルが見つかりません: {PRED_DIR}")
        return

    metrics = []

    for pred_path in pred_paths:
        mask_path = MASK_DIR / pred_path.name
        if not mask_path.exists():
            continue

        # ch7 (全道路チャンネル) だけ使う
        # rasterio は 1-indexed なので ch7 = band 8
        with rasterio.open(pred_path) as src:
            pred = src.read(8)
        with rasterio.open(mask_path) as src:
            gt = src.read(8)

        pred_bin = (pred > threshold)
        gt_bin   = (gt > 0)

        TP = (pred_bin &  gt_bin).sum()
        FP = (pred_bin & ~gt_bin).sum()
        FN = (~pred_bin &  gt_bin).sum()

        iou       = TP / (TP + FP + FN + 1e-6)
        f1        = 2*TP / (2*TP + FP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall    = TP / (TP + FN + 1e-6)
        metrics.append((iou, f1, precision, recall))

    metrics = np.array(metrics)

    print(f"threshold : {threshold}")
    print(f"評価タイル数: {len(metrics)}")
    print(f"\n{'指標':<12} {'平均':>8} {'最小':>8} {'最大':>8}")
    print("-" * 42)
    for i, name in enumerate(["IoU", "F1", "Precision", "Recall"]):
        print(f"{name:<12} {metrics[:, i].mean():>8.4f} "
              f"{metrics[:, i].min():>8.4f} {metrics[:, i].max():>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="道路判定の閾値 (default: 0.3)")
    args = parser.parse_args()
    evaluate(threshold=args.threshold)
