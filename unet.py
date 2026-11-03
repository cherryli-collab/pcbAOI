import torch  # 匯入 PyTorch
import torch.nn as nn  # 匯入神經網路模組

# -----------------------------
# 基本卷積區塊（Conv → BN → ReLU）
# -----------------------------
class ConvBlock(nn.Module):  # 定義卷積區塊
    def __init__(self, in_ch, out_ch):  # 傳入輸入/輸出通道數
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 3x3 卷積
            nn.BatchNorm2d(out_ch),  # 批次正規化
            nn.ReLU(inplace=True),  # ReLU 激活
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # 再來一層卷積
            nn.BatchNorm2d(out_ch),  # 批次正規化
            nn.ReLU(inplace=True)  # ReLU 激活
        )

    def forward(self, x):  # 前向傳遞
        return self.conv(x)  # 回傳卷積結果


# -----------------------------
# U-Net 主體
# -----------------------------
class UNet(nn.Module):  # 定義 U-Net 類別
    def __init__(self, in_channels=3, out_channels=1):  # 初始化
        super().__init__()

        # Encoder（下採樣）
        self.enc1 = ConvBlock(in_channels, 64)  # 第一層
        self.enc2 = ConvBlock(64, 128)  # 第二層
        self.enc3 = ConvBlock(128, 256)  # 第三層
        self.enc4 = ConvBlock(256, 512)  # 第四層

        # MaxPool（下採樣）
        self.pool = nn.MaxPool2d(2)  # 2x2 池化

        # Bottleneck（最底部）
        self.bottleneck = ConvBlock(512, 1024)  # 中間層

        # Decoder（上採樣）
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 反卷積上採樣
        self.dec4 = ConvBlock(1024, 512)  # 解碼卷積

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # 最終輸出層
        self.final = nn.Conv2d(64, out_channels, 1)  # 1x1 卷積輸出 mask

    def forward(self, x):  # 前向傳遞
        # Encoder
        e1 = self.enc1(x)  # 第一層
        e2 = self.enc2(self.pool(e1))  # 第二層
        e3 = self.enc3(self.pool(e2))  # 第三層
        e4 = self.enc4(self.pool(e3))  # 第四層

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 中間層

        # Decoder（上採樣 + skip connection）
        d4 = self.up4(b)  # 上採樣
        d4 = torch.cat([d4, e4], dim=1)  # skip connection
        d4 = self.dec4(d4)  # 解碼

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)  # 回傳最終 mask
