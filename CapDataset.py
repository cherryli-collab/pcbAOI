import os
import cv2
import torch
from torch.utils.data import Dataset

class CapDataset(Dataset):
    """
    端盖偏移数据集（DeepLabv3+ 用）
    images/ 里放原图
    masks/  里放对应 mask（0/255）
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 读取影像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # 读取 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return img, mask
