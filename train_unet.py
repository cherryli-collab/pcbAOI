# train_unet.py
import torch
from torch.utils.data import DataLoader
from unet import UNet
from ScratchDataset import ScratchDataset
from loss import DiceBCELoss

def train_unet():
    print("Training U-Net (Scratch)...")

    dataset = ScratchDataset("dataset/scratch/images", "dataset/scratch/masks")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet(3, 1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    for epoch in range(5):  # 你可以改成 50–150
        for img, mask in loader:
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[U-Net] Epoch {epoch+1}, Loss={loss.item():.4f}")

    torch.save(model.state_dict(), "output/unet.pth")
    print("U-Net training done → output/unet.pth")
