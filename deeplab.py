import torch  # 匯入 PyTorch
import torch.nn as nn  # 匯入神經網路模組
import torchvision.models as models  # 匯入 torchvision 預訓練模型

# -----------------------------
# ASPP 模組（DeepLabv3+ 核心）
# -----------------------------
class ASPP(nn.Module):  # 定義 ASPP 類別
    def __init__(self, in_ch, out_ch):  # 傳入輸入/輸出通道
        super().__init__()
        self.atrous1 = nn.Conv2d(in_ch, out_ch, 1)  # 1x1 卷積
        self.atrous6 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6)  # 空洞卷積
        self.atrous12 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)
        self.atrous18 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)

        self.conv1x1 = nn.Conv2d(out_ch * 4, out_ch, 1)  # 合併後的 1x1 卷積

    def forward(self, x):  # 前向傳遞
        x1 = self.atrous1(x)  # 分支 1
        x2 = self.atrous6(x)  # 分支 2
        x3 = self.atrous12(x)  # 分支 3
        x4 = self.atrous18(x)  # 分支 4

        x = torch.cat([x1, x2, x3, x4], dim=1)  # 合併四個分支
        return self.conv1x1(x)  # 1x1 卷積整合


# -----------------------------
# DeepLabv3+ 主體
# -----------------------------
class DeepLabV3Plus(nn.Module):  # 定義 DeepLabv3+ 類別
    def __init__(self, backbone="resnet50", num_classes=1):  # 初始化
        super().__init__()

        # 使用 ResNet50 當 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 移除最後 FC 層

        # ASPP 模組
        self.aspp = ASPP(2048, 256)  # ResNet50 最後輸出 2048 通道

        # Decoder（上採樣 + 邊界強化）
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 3x3 卷積
            nn.BatchNorm2d(128),  # BN
            nn.ReLU(inplace=True),  # ReLU
            nn.Conv2d(128, num_classes, 1)  # 1x1 卷積輸出 mask
        )

    def forward(self, x):  # 前向傳遞
        x = self.backbone(x)  # 提取特徵
        x = self.aspp(x)  # ASPP
        x = self.decoder(x)  # 解碼
        x = nn.functional.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)  # 上採樣回原尺寸
        return x  # 回傳 mask
