# SpaceNet 5 M2 Baseline

SpaceNet 5 の道路速度セグメンテーションを **M2 MacBook Air** で動かす学習用実装です。

[SpaceNet 5 Baseline ブログ](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-1-imagery-and-label-preparation-598af46d485e) の内容を、CUDA 不要・最新 PyTorch (MPS) で再現しました。

---

## パイプライン概要

```
衛星画像 (PS-RGB)
    ↓ data/make_masks.py     GeoJSON → 8ch 速度マスク
    ↓ train/train.py         ResNet34-UNet で学習 (Apple MPS)
    ↓ infer/predict.py       予測マスク生成
    ↓ infer/skeletonize.py   マスク → 1px 骨格線
    ↓ infer/graph.py         骨格線 → NetworkX グラフ
    ↓ infer/speed.py         エッジごとに速度を推定
    ↓ viz/overlay.py         衛星画像 + 速度グラフを可視化
```

---

## 環境

| 項目 | 内容 |
|---|---|
| OS | macOS (Apple M2) |
| Python | 3.12 |
| PyTorch | 2.x (MPS バックエンド) |
| 主要ライブラリ | segmentation-models-pytorch, rasterio, geopandas, networkx, sknw |

---

## セットアップ

```bash
git clone https://github.com/ringojam1115/spacenet5-m2-baseline.git
cd spacenet5-m2-baseline

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 動作確認
python verify_env.py
```

---

## データ取得

SpaceNet 5 のデータは AWS S3 (requester-pays) から取得します。
事前に AWS CLI の設定 (`aws configure`) が必要です。

```bash
python data/download_s3.py   # Moscow AOI の 200 タイルをダウンロード
python data/make_masks.py    # GeoJSON → 8ch 速度マスクに変換
```

---

## 学習

```bash
python -m train.train
# epoch  1/30  train: 0.3644  val: 0.3729
# ...
# epoch 30/30  train: 0.2331  val: 0.2314
# → results/best_model.pth に最良重みを保存
```

**ハイパーパラメータ** (train/train.py):

| パラメータ | 値 |
|---|---|
| encoder | ResNet34 (ImageNet pretrained) |
| crop_size | 256 × 256 |
| batch_size | 4 |
| epochs | 30 |
| lr | 0.0001 |
| loss | Dice × 0.25 + Focal × 0.75 |

---

## 推論

```bash
python -m infer.predict      # 予測マスク生成
python -m infer.skeletonize  # 骨格化
python -m infer.graph        # グラフ化
python -m infer.speed        # 速度推定
```

---

## 評価・可視化

```bash
# 精度指標 (IoU / F1 / Precision / Recall)
python evaluate.py --threshold 0.3

# 衛星画像 + 正解 + 予測グラフを並べて保存
python -m viz.overlay --chip 0 1 2
```

### 現在の精度 (200 枚, threshold=0.3)

| 指標 | 平均 | 最大 |
|---|---|---|
| IoU | 0.018 | 0.923 |
| F1 | 0.031 | 0.960 |
| Precision | 0.275 | 0.978 |
| Recall | 0.019 | 0.997 |

Precision が高く Recall が低い → 予測した箇所は概ね正しいが見逃しが多い。
データを 200 枚 → 1353 枚に増やすことで改善見込み。

---

## ディレクトリ構成

```
m2_baseline/
├── data/
│   ├── download_s3.py      S3 からタイルを取得
│   ├── make_masks.py       GeoJSON → 8ch 速度マスク
│   └── raw/ masks/         データ置き場 (git 管理外)
├── train/
│   ├── dataset.py          SpaceNetDataset
│   ├── model.py            ResNet34-UNet (SMP)
│   ├── loss.py             Dice Loss + Focal Loss
│   └── train.py            学習ループ (MPS, train/val split)
├── infer/
│   ├── predict.py          マスク推論
│   ├── skeletonize.py      骨格化
│   ├── graph.py            NetworkX グラフ化
│   └── speed.py            速度推定
├── viz/
│   └── overlay.py          可視化
├── evaluate.py             精度評価
├── verify_env.py           環境確認
└── requirements.txt
```

---

## 参考

- [SpaceNet 5 Baseline Part 1](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-1-imagery-and-label-preparation-598af46d485e)
- [SpaceNet 5 Baseline Part 2](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-2-training-a-road-speed-segmentation-model-2bc93de564d7)
- [SpaceNet 5 Baseline Part 3](https://medium.com/the-downlinq/the-spacenet-5-baseline-part-3-extracting-road-speed-vectors-from-satellite-imagery-5d07cd5e1d21)
- [cresi (元リポジトリ)](https://github.com/avanetten/cresi)
