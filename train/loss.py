import torch
from torch import Tensor
import torch.nn.functional as F

def dice_loss(pred: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    """
    引数:
        pred   : モデルの生出力 (sigmoid 前), shape=(B, 8, H, W)
        target : 正解マスク,                         shape=(B, 8, H, W), 値域 0 or 1
        smooth : ゼロ除算防止の微小値
    
    Dice係数 = 2 * |pred ∩ target| / (|pred| + |target|)
    Dice Loss = 1 - Dice係数
    
    返り値: スカラー (全チャンネル・全バッチの平均)
    """
    pred = torch.sigmoid(pred)

    # ピクセル方向に flatten する (B, C, H, W) → (B, C, -1)
    pred   = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)

    # 交差部分の面積
    intersection = (pred * target).sum(dim=2)  # shape=(B, 8)

    # 各チャンネルの Dice 係数を計算
    dice_coef = (2. * intersection + smooth) / (pred.sum(dim=2) + target.sum(dim=2) + smooth)

    # Dice Loss を計算
    dice_loss = 1 - dice_coef

    # バッチ全体の平均を返す
    return dice_loss.mean()


def focal_loss(pred: Tensor, target: Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
    """
    引数:
        pred   : モデルの生出力 (sigmoid 前), shape=(B, 8, H, W)
        target : 正解マスク,                  shape=(B, 8, H, W), 値域 0 or 1
        gamma  : 難しいサンプルへの重み付け強度
        alpha  : 正例(道路ピクセル)への重み
    
    Focal Loss = -alpha * (1 - p)^gamma * log(p)
    
    ヒント: torch.nn.functional.binary_cross_entropy_with_logits の
             weight / reduction 引数を活用する
    """
    # BCE (logits版)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    # 確率に変換
    p = torch.sigmoid(pred)

    # p_t を作る（正解側の確率）
    p_t = p * target + (1 - p) * (1 - target)

    # focal weight
    focal_weight = (1 - p_t) ** gamma

    # alphaバランス
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    # 最終loss
    loss = alpha_t * focal_weight * bce

    # バッチ全体の平均を返す
    return loss.mean()

def combined_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Dice Loss * 0.25 + Focal Loss * 0.75 を返す
    """
    return 0.25 * dice_loss(pred, target) + 0.75 * focal_loss(pred, target)