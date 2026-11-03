import cv2  # 影像讀取與處理
import os  # Fix: add missing import
import torch  # PyTorch 主套件
import numpy as np  # 數值運算
from unet import UNet  # 匯入 U-Net 模型
from deeplab import DeepLabV3Plus  # 匯入 DeepLabv3+ 模型
from ultralytics import YOLO  # 匯入 YOLOv8 Segmentation

# -----------------------------
# 0. 設備檢測 (CPU/GPU)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] 使用設備: {DEVICE}")

# -----------------------------
# 1. 模型載入 (容錯處理)
# -----------------------------

# U-Net（刮傷）- 使用預訓練或跳過
unet = None
try:
    unet = UNet(in_channels=3, out_channels=1).to(DEVICE)
    if os.path.exists("unet_scratch.pth"):
        unet.load_state_dict(torch.load("unet_scratch.pth", map_location=DEVICE))
        print("[✔] U-Net 已載入")
    else:
        print("[!] unet_scratch.pth 不存在，將使用隨機權重")
    unet.eval()
except Exception as e:
    print(f"[!] U-Net 載入失敗: {e}")

# YOLO-Seg（焊點）- 使用預訓練模型
yolo = None
try:
    if os.path.exists("yolov8s-seg.pt"):
        yolo = YOLO("yolov8s-seg.pt")
    else:
        print("[*] 使用 YOLOv8n-seg 預訓練模型")
        yolo = YOLO("yolov8n-seg.pt")
    yolo.to(DEVICE)
    print("[✔] YOLO 已載入")
except Exception as e:
    print(f"[!] YOLO 載入失敗: {e}")

# DeepLabv3+（端蓋偏移）- 使用預訓練或跳過
deeplab = None
try:
    deeplab = DeepLabV3Plus(backbone="resnet50").to(DEVICE)
    if os.path.exists("deeplab_cap.pth"):
        deeplab.load_state_dict(torch.load("deeplab_cap.pth", map_location=DEVICE))
        print("[✔] DeepLabv3+ 已載入")
    else:
        print("[!] deeplab_cap.pth 不存在，將使用隨機權重")
    deeplab.eval()
except Exception as e:
    print(f"[!] DeepLabv3+ 載入失敗: {e}")


# -----------------------------
# 2. 前處理函式
# -----------------------------
def preprocess(img, size=512):
    img_resized = cv2.resize(img, (size, size))  # 調整大小
    img_tensor = torch.from_numpy(img_resized).float()  # 轉成 float tensor
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # HWC → CHW 並正規化
    return img_tensor.unsqueeze(0).to(DEVICE)  # 增加 batch 維度並移到設備


# -----------------------------
# 3. 刮傷推論（U-Net）
# -----------------------------
def infer_scratch(img):
    x = preprocess(img)  # 前處理
    with torch.no_grad():  # 不計算梯度
        pred = unet(x)  # U-Net 推論
        mask = torch.sigmoid(pred)[0, 0].cpu().numpy()  # 取出 mask
        mask = (mask > 0.5).astype(np.uint8) * 255  # 二值化
    return mask  # 回傳刮傷 mask


# -----------------------------
# 4. 焊點推論（YOLO-Seg）
# -----------------------------
def infer_weld(img):
    results = yolo(img, imgsz=640, device=0)[0]  # YOLO 推論
    masks = []  # 用來存 segmentation mask
    boxes = []  # 用來存 bounding box

    if results.masks is not None:  # 如果有 segmentation 結果
        for m, b in zip(results.masks.data, results.boxes.xyxy):
            mask = m.cpu().numpy() * 255  # 轉成 numpy mask
            masks.append(mask.astype(np.uint8))  # 加入 mask
            boxes.append(b.cpu().numpy())  # 加入 bounding box

    return masks, boxes  # 回傳焊點 mask 與框框


# -----------------------------
# 5. 端蓋偏移推論（DeepLabv3+）
# -----------------------------
def infer_cap(img):
    x = preprocess(img)  # 前處理
    with torch.no_grad():  # 不計算梯度
        pred = deeplab(x)  # DeepLab 推論
        mask = torch.sigmoid(pred)[0, 0].cpu().numpy()  # 取出 mask
        mask = (mask > 0.5).astype(np.uint8) * 255  # 二值化
    return mask  # 回傳端蓋邊界 mask


# -----------------------------
# 6. 缺陷融合邏輯（最終 OK / NG）
# -----------------------------
def fuse_defects(scratch_mask, weld_masks, cap_mask):
    defects = []  # 用來存缺陷資訊

    # 刮傷判定
    if scratch_mask.sum() > 500:  # 若刮傷面積 > 500 pixels
        defects.append("Scratch")  # 加入缺陷類型

    # 焊點判定
    if len(weld_masks) > 0:  # 若 YOLO 偵測到焊點瑕疵
        defects.append("Weld")  # 加入缺陷類型

    # 端蓋偏移判定
    if cap_mask.sum() > 2000:  # 若偏移邊界面積 > 閾值
        defects.append("Cap Misalignment")  # 加入缺陷類型

    # 最終 OK / NG
    if len(defects) == 0:
        return "OK", defects  # 無缺陷 → OK
    else:
        return "NG", defects  # 有缺陷 → NG


# -----------------------------
# 7. 主推論流程（整合三模型）
# -----------------------------
def run_inference(img):
    scratch_mask = infer_scratch(img)  # 刮傷
    weld_masks, weld_boxes = infer_weld(img)  # 焊點
    cap_mask = infer_cap(img)  # 端蓋偏移

    status, defects = fuse_defects(scratch_mask, weld_masks, cap_mask)  # 融合判定

    return {
        "status": status,  # OK / NG
        "defects": defects,  # 缺陷類型
        "scratch_mask": scratch_mask,  # 刮傷 mask
        "weld_masks": weld_masks,  # 焊點 mask
        "weld_boxes": weld_boxes,  # 焊點框框
        "cap_mask": cap_mask  # 端蓋 mask
    }


# -----------------------------
# 8. 測試推論（可直接跑）
# -----------------------------
if __name__ == "__main__":
    img = cv2.imread("test_motor.jpg")  # 讀取測試影像
    result = run_inference(img)  # 執行推論

    print("最終判定：", result["status"])  # 印出 OK / NG
    print("缺陷類型：", result["defects"])  # 印出缺陷列表

    cv2.imwrite("scratch_mask.png", result["scratch_mask"])  # 儲存刮傷 mask
    cv2.imwrite("cap_mask.png", result["cap_mask"])  # 儲存端蓋 mask
