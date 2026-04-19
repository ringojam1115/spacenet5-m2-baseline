import torch
from train.dataset import SpaceNetDataset
from torch.utils.data import DataLoader
from train.model import build_model
from train.loss import combined_loss

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def train():
    """
    1. Dataset / DataLoader
    2. model.to(DEVICE)
    3. optimizer
    4. 学習ループ
    5. 重み保存
    """
    # データセットのパス
    rgb_dir = "data/raw/AOI_7_Moscow/PS-RGB"
    mask_dir = "data/masks/AOI_7_Moscow"

    # ハイパーパラメータ
    crop_size = 256
    augment = False

    # 学習設定
    lr = 0.0001 # 学習率
    batch_size = 4 # バッチサイズ
    num_epochs = 30 # エポック数

    # 1. Dataset / DataLoader
    spaceNetDataset = SpaceNetDataset(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        crop_size=crop_size,
        augment=augment
    )

    # 2. model.to(DEVICE)
    model = build_model() # モデルの構築
    model.to("mps")  # Apple Silicon の GPU を使う場合

    # 3. optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 学習ループ
    train_loader = DataLoader(spaceNetDataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to("mps")
            masks  = masks.to("mps")
            pred = model(images)
            loss = combined_loss(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch+1}/{num_epochs}  loss: {avg_loss:.4f}")

    # 5. 重み保存
    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    train()

