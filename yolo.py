import torch
from ultralytics import YOLO

def train_yolo():
    # 自動選擇 CPU 或 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 載入 YOLOv8 segmentation 模型
    model = YOLO("yolov8s-seg.pt")

    # 開始訓練
    model.train(
        data="weld.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device='cpu'   # ← 永遠不會是 0
    )

    # 導出模型
    model.export(format="pt")
    print("YOLO training done → output/yolo.pt")

if __name__ == "__main__":
    train_yolo()   # ← 不要傳任何 device 參數
