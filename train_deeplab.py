# train_deeplab.py
import torch
from torch.utils.data import DataLoader
from deeplab import DeepLabV3Plus
from CapDataset import CapDataset
from loss import BoundaryLoss

def train_deeplab():
    print("Training DeepLabv3+ (Cap Misalignment)...")

    dataset = CapDataset("dataset/cap/images", "dataset/cap/masks")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DeepLabV3Plus().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = BoundaryLoss()

    for epoch in range(5):  # 你可以改成 100–200
        for img, mask in loader:
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[DeepLab] Epoch {epoch+1}, Loss={loss.item():.4f}")

    torch.save(model.state_dict(), "output/deeplab.pth")
    print("DeepLab training done → output/deeplab.pth")
