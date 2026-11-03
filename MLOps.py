import os
import shutil
import time
import traceback

from train_unet import train_unet
from train_yolo import train_yolo
from train_deeplab import train_deeplab

# 必要資料夾
required_dirs = [
    "NG",
    "dataset/new_data",
    "dataset/scratch/images",
    "dataset/scratch/masks",
    "dataset/weld/images",
    "dataset/weld/masks",
    "dataset/cap/images",
    "dataset/cap/masks",
    "output",
    "deploy"
]

for d in required_dirs:
    os.makedirs(d, exist_ok=True)

NG_DIR = "NG/"
NEW_DATA_DIR = "dataset/new_data/"

def log(msg):
    print(msg)
    with open("mlops.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def auto_collect():
    moved = 0
    for f in os.listdir(NG_DIR):
        if f.endswith(".png"):
            src = os.path.join(NG_DIR, f)
            dst = os.path.join(NEW_DATA_DIR, f)
            shutil.move(src, dst)
            moved += 1
    log(f"[Collect] Moved {moved} NG images.")

def dataset_empty(path):
    return len(os.listdir(path)) == 0

def auto_retrain():
    # U-Net
    if dataset_empty("dataset/scratch/images"):
        log("[U-Net] Skip training (dataset empty).")
    else:
        try:
            log("[U-Net] Training...")
            train_unet()
        except Exception:
            log("[U-Net] ERROR:\n" + traceback.format_exc())

    # YOLO
    if dataset_empty("dataset/weld/images"):
        log("[YOLO] Skip training (dataset empty).")
    else:
        try:
            log("[YOLO] Training...")
            train_yolo()
        except Exception:
            log("[YOLO] ERROR:\n" + traceback.format_exc())

    # DeepLab
    if dataset_empty("dataset/cap/images"):
        log("[DeepLab] Skip training (dataset empty).")
    else:
        try:
            log("[DeepLab] Training...")
            train_deeplab()
        except Exception:
            log("[DeepLab] ERROR:\n" + traceback.format_exc())

def auto_update():
    try:
        shutil.copy("output/unet.pth", "deploy/unet.pth")
        shutil.copy("output/yolo.pt", "deploy/yolo.pt")
        shutil.copy("output/deeplab.pth", "deploy/deeplab.pth")
        log("[Deploy] Models updated.")
    except Exception:
        log("[Deploy] ERROR:\n" + traceback.format_exc())

def mlops_loop():
    log("=== MLOps Started ===")
    while True:
        auto_collect()
        auto_retrain()
        auto_update()
        log("=== Cycle Completed ===\n")
        time.sleep(86400)  # 24 小時

if __name__ == "__main__":
    mlops_loop()
