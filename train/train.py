import torch
from train.dataset import SpaceNetDataset
from torch.utils.data import DataLoader, random_split
from train.model import build_model
from train.loss import combined_loss

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def train():
    """
    1. Dataset / DataLoader (train 80% / val 20%)
    2. model.to(DEVICE)
    3. optimizer
    4. 学習ループ
    5. val loss が最小のときだけ重み保存
    """
    # データセットのパス
    rgb_dir = "data/raw/AOI_7_Moscow/PS-RGB"
    mask_dir = "data/masks/AOI_7_Moscow"

    # ハイパーパラメータ
    crop_size = 256
    augment = False

    # 学習設定
    lr = 0.0001
    batch_size = 4
    num_epochs = 30

    # 1. Dataset を作って 80:20 に分割
    dataset = SpaceNetDataset(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        crop_size=crop_size,
        augment=augment,
    )
    n_train = int(len(dataset) * 0.8)   # 160 枚
    n_val   = len(dataset) - n_train     #  40 枚
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),  # 毎回同じ分割
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    # 2. model
    model = build_model()
    model.to("mps")

    # 3. optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 学習ループ
    os.makedirs("results", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        # --- train ---
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to("mps")
            masks  = masks.to("mps")
            pred = model(images)
            loss = combined_loss(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():                    # val では勾配計算しない
            for images, masks in val_loader:
                images = images.to("mps")
                masks  = masks.to("mps")
                pred = model(images)
                val_loss += combined_loss(pred, masks).item()
        val_loss /= len(val_loader)

        # val loss が最小なら保存
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model.pth")
            saved = "  → 保存"

        print(f"epoch {epoch+1:2d}/{num_epochs}"
              f"  train: {train_loss:.4f}  val: {val_loss:.4f}{saved}")

if __name__ == "__main__":
    train()

