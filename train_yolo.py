from ultralytics import YOLO

def train_yolo():
    print("=== Training YOLO-Seg (Weld) ===")

    # 加载 YOLOv8 segmentation 模型
    model = YOLO("yolov8s-seg.pt")

    # 开始训练
    model.train(
        data="dataset/weld.yaml",   # 数据集配置文件
        epochs=100,                 # 训练轮数
        imgsz=640,                  # 输入尺寸
        batch=16,                   # batch size
        device="cpu"                    # GPU
    )

    # 导出模型
    model.export(format="pt")
    print("YOLO training done → output/yolo.pt")

if __name__ == "__main__":
    train_yolo()

