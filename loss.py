import torch  # 匯入 PyTorch
import torch.nn as nn  # 匯入神經網路模組
import torch.nn.functional as F  # 匯入常用函式（如 softmax、conv）

# -----------------------------
# Dice + BCE Loss（適用：刮傷 Segmentation）
# -----------------------------
class DiceBCELoss(nn.Module):  # 定義損失類別
    def __init__(self):  # 初始化
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()  # 二元交叉熵損失

    def forward(self, pred, target):  # 前向傳遞
        pred = torch.sigmoid(pred)  # 將輸出壓到 0~1
        bce_loss = self.bce(pred, target)  # 計算 BCE

        smooth = 1e-5  # 避免除以 0
        pred_flat = pred.view(-1)  # 攤平成一維
        target_flat = target.view(-1)  # 攤平成一維

        intersection = (pred_flat * target_flat).sum()  # 計算交集
        dice_loss = 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return bce_loss + dice_loss  # 回傳 BCE + Dice
        

# -----------------------------
# Boundary Loss（適用：端蓋偏移 Segmentation）
# -----------------------------
class BoundaryLoss(nn.Module):  # 定義邊界損失
    def __init__(self):  # 初始化
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target):  # 前向傳遞
        pred = torch.sigmoid(pred)  # 壓到 0~1
        pred_grad = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])  # 計算水平梯度
        target_grad = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])  # 計算 GT 梯度

        loss = F.l1_loss(pred_grad, target_grad)  # L1 損失比較邊界差異
        return loss  # 回傳邊界損失
