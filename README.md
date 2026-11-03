# PCBAOI

直流馬達外觀檢測（AOI）原型，包含刮傷、焊點／焊接區域與端蓋偏移的影像分割訓練程式、推論整合程式，以及 SQLite 缺陷資料庫範例。

> **目前狀態：開發原型。** 專案尚未提供依賴鎖定檔或完整的部署設定；部分訓練和 GUI 入口有使用限制，請先閱讀「已知限制」再執行。

## 功能與主要檔案

| 檔案 | 用途 |
|---|---|
| `train_unet.py`、`unet.py`、`ScratchDataset.py` | 訓練 U-Net 刮傷分割模型 |
| `train_deeplab.py`、`deeplab.py`、`CapDataset.py` | 訓練 DeepLabv3+ 端蓋偏移分割模型 |
| `train_yolo.py`、`yolo.py` | 使用 Ultralytics YOLOv8 Seg 訓練焊點／缺陷分割模型 |
| `inference.py` | 載入模型並整合刮傷、焊點與端蓋偏移推論及 OK/NG 判定 |
| `PyQt5 GUI.py` | 相機畫面與檢測結果的 PyQt5 介面原型 |
| `MLOps.py` | NG 影像收集、模型重新訓練與部署檔案複製的循環原型 |
| `init_db.py`、`sqlite3.sql` | 建立 SQLite `defects` 資料表 |
| `report_generator.py` | 從資料庫產生缺陷 Pareto 與班別 NG 比率圖表 |
| `loss.py` | 分割模型使用的損失函式 |

## 環境準備

建議使用 Python 3.10 或更新版本的虛擬環境。請在專案根目錄執行以下命令：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安裝專案程式使用的套件：

```powershell
pip install torch torchvision ultralytics opencv-python numpy PyQt5 pandas matplotlib
```

若要使用 NVIDIA GPU，請依照 PyTorch 官方安裝指引選擇與本機 CUDA 相容的 PyTorch 套件。現有 U-Net 與 DeepLabv3+ 訓練程式直接呼叫 CUDA，不能在沒有可用 CUDA GPU 的環境執行。

## 資料集

目前目錄中包含 `dataset/scratch/`、`dataset/cap/`、`dataset/weld/` 與 `dataset/new_data/`。分割資料集的影像與遮罩需同名配對：

```text
dataset/
├── scratch/
│   ├── images/       # 刮傷原圖
│   └── masks/        # 對應灰階遮罩，背景為 0、缺陷為 255
└── cap/
    ├── images/       # 端蓋原圖
    └── masks/        # 對應灰階遮罩，背景為 0、缺陷為 255
```

YOLO 分割資料通常需要影像、YOLO 格式標註，以及資料集 YAML。現有 `weld.yaml` 與 `dataset/weld.yaml` 設定的是 `train`、`val`、`test` 子目錄；請確認 YAML 中的路徑、類別名稱和分割標註與實際資料集一致，再開始訓練。僅有 `images/` 與 `masks/` 的目錄尚不足以直接滿足 Ultralytics YOLO 分割標註格式。

## 執行方式

### 訓練

在專案根目錄執行：

```powershell
python train_unet.py
python train_deeplab.py
python train_yolo.py
```

U-Net 與 DeepLabv3+ 權重會儲存至 `output/`；YOLO 使用 Ultralytics 訓練與匯出流程，輸出位置依其執行設定而定。U-Net 與 DeepLabv3+ 目前各訓練 5 個 epoch；YOLO 訓練 100 個 epoch。訓練資料路徑與模型裝置設定以各腳本中的設定為準。

### 推論

`inference.py` 定義 `run_inference(img)`，輸入 OpenCV 讀取的 BGR 影像，回傳包含 `status`、`defects` 及各模型遮罩／框座標的字典。腳本主程式會嘗試讀取 `test_motor.jpg`；執行前請在專案根目錄放入有效影像：

```powershell
python inference.py
```

推論會在目前工作目錄尋找 `unet_scratch.pth`、`deeplab_cap.pth` 與 `yolov8s-seg.pt`。前兩個分割權重不存在時，程式仍會以隨機初始化權重執行；此時輸出不具備有效檢測意義。YOLO 權重則需使用適合任務的訓練結果。

### 資料庫與報表

建立資料庫資料表：

```powershell
python init_db.py
```

資料預設寫入專案根目錄的 `aoi.db`。產生報表：

```powershell
python report_generator.py
```

此腳本會讀取 `defects` 資料表並輸出 `pareto.png` 與 `shift_rate.png`；請先確認資料表存在且已寫入資料。

### MLOps 循環

```powershell
python MLOps.py
```

此程式會持續執行：將 `NG/` 下的 PNG 移至 `dataset/new_data/`、依各訓練資料夾狀態嘗試重新訓練、複製模型至 `deploy/`，之後等待 24 小時再進入下一輪。它不會自動替新影像建立標註；請在使用前確認資料與部署流程符合需求。停止程式可在終端機按 `Ctrl+C`。

## 已知限制

- `PyQt5 GUI.py` 匯入 `inference.run_pipeline`，但目前 `inference.py` 提供的是 `run_inference`；兩者介面尚未對齊，因此 GUI 目前不能直接啟動。
- `inference.py` 的 YOLO 推論指定 `device=0`，這表示使用 CUDA 裝置 0；CPU 環境可能無法透過此推論路徑執行。
- `train_unet.py` 與 `train_deeplab.py` 使用 `.cuda()`，需要可用的 CUDA GPU。
- `train_yolo.py` 使用 `dataset/weld.yaml`，而兩份 YOLO YAML 的路徑均假設存在 `train/`、`val/` 等目錄；請依實際資料集調整 YAML。
- `MLOps.py` 會複製 `output/unet.pth`、`output/yolo.pt`、`output/deeplab.pth`。若任一模型檔不存在，複製流程會記錄錯誤；目前訓練腳本與 Ultralytics 的實際輸出位置也應先核對。
- `inference.py` 的缺陷面積門檻（500、2000 pixels）及焊點判定規則是原型邏輯，需以標註資料和現場規格驗證，不能直接視為量產判定標準。

## 資料與模型檔案

影像資料、模型權重、SQLite 資料庫、訓練輸出與日誌可能含有大型或非程式碼內容。請依團隊的資料管理規範決定是否納入版本控制，避免提交敏感影像或不必要的大型模型檔案。

---

# English

PCBAOI is a prototype automated visual inspection (AOI) system for DC motor components. It includes image-segmentation training scripts for scratches, weld/solder areas, and cap misalignment, an integrated inference module, and a SQLite defect-database example.

> **Status: Development prototype.** The project does not yet include a dependency lock file or complete deployment configuration. Some training and GUI entry points have limitations; review the notes below before running them.

## Features and key files

| File | Purpose |
|---|---|
| `train_unet.py`, `unet.py`, `ScratchDataset.py` | Train a U-Net model for scratch segmentation |
| `train_deeplab.py`, `deeplab.py`, `CapDataset.py` | Train a DeepLabv3+ model for cap-misalignment segmentation |
| `train_yolo.py`, `yolo.py` | Train an Ultralytics YOLOv8 Seg model for weld/defect segmentation |
| `inference.py` | Run scratch, weld, and cap-misalignment inference and produce an OK/NG result |
| `PyQt5 GUI.py` | PyQt5 camera-view and inspection-results GUI prototype |
| `MLOps.py` | Prototype loop for collecting NG images, retraining models, and copying deployment files |
| `init_db.py`, `sqlite3.sql` | Create the SQLite `defects` table |
| `report_generator.py` | Generate defect Pareto and shift NG-rate charts from the database |
| `loss.py` | Loss functions used by the segmentation models |

## Setup

Python 3.10 or later is recommended. Create and activate a virtual environment from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the packages used by the project:

```powershell
pip install torch torchvision ultralytics opencv-python numpy PyQt5 pandas matplotlib
```

For an NVIDIA GPU, install a PyTorch build compatible with the local CUDA environment by following the official PyTorch installation instructions. The current U-Net and DeepLabv3+ training scripts call CUDA directly and cannot run without an available CUDA GPU.

## Datasets

The project currently contains `dataset/scratch/`, `dataset/cap/`, `dataset/weld/`, and `dataset/new_data/`. For the segmentation datasets, each image and its mask must have the same filename:

```text
dataset/
├── scratch/
│   ├── images/       # Original scratch images
│   └── masks/        # Matching grayscale masks: 0 background, 255 defect
└── cap/
    ├── images/       # Original cap images
    └── masks/        # Matching grayscale masks: 0 background, 255 defect
```

YOLO segmentation datasets typically require images, YOLO-format annotations, and a dataset YAML file. The existing `weld.yaml` and `dataset/weld.yaml` expect `train`, `val`, and `test` subdirectories. Before training, verify that the YAML paths, class names, and segmentation annotations match the actual dataset. Directories containing only `images/` and `masks/` do not by themselves satisfy the Ultralytics YOLO segmentation annotation format.

## Usage

### Training

Run these commands from the project root:

```powershell
python train_unet.py
python train_deeplab.py
python train_yolo.py
```

U-Net and DeepLabv3+ weights are saved under `output/`. YOLO uses the Ultralytics training and export workflow, so its output location depends on the run configuration. U-Net and DeepLabv3+ currently train for 5 epochs each; YOLO trains for 100 epochs. See each script for the configured dataset paths and device.

### Inference

`inference.py` defines `run_inference(img)`. It accepts a BGR image loaded by OpenCV and returns a dictionary containing `status`, `defects`, and model masks/box coordinates. Its main block attempts to load `test_motor.jpg`; place a valid image with that name in the project root before running:

```powershell
python inference.py
```

Inference looks for `unet_scratch.pth`, `deeplab_cap.pth`, and `yolov8s-seg.pt` in the current working directory. If either of the first two segmentation weights is missing, the program still runs with randomly initialized weights; those results are not meaningful for inspection. The YOLO weights should also be trained for the intended task.

### Database and reports

Create the database table:

```powershell
python init_db.py
```

By default, the database is written to `aoi.db` in the project root. Generate reports with:

```powershell
python report_generator.py
```

The script reads the `defects` table and writes `pareto.png` and `shift_rate.png`. Ensure the table exists and contains data first.

### MLOps loop

```powershell
python MLOps.py
```

This program runs continuously: it moves PNG files from `NG/` to `dataset/new_data/`, attempts retraining based on the contents of each training-data directory, copies models to `deploy/`, and waits 24 hours before the next cycle. It does not create annotations for newly collected images. Verify the data and deployment workflow before use. Press `Ctrl+C` in the terminal to stop it.

## Known limitations

- `PyQt5 GUI.py` imports `inference.run_pipeline`, but `inference.py` currently provides `run_inference`; these interfaces are not aligned, so the GUI cannot be launched as-is.
- YOLO inference in `inference.py` specifies `device=0`, which selects CUDA device 0; this path may not work in a CPU-only environment.
- `train_unet.py` and `train_deeplab.py` use `.cuda()` and require an available CUDA GPU.
- `train_yolo.py` uses `dataset/weld.yaml`, and both YOLO YAML files assume directories such as `train/` and `val/` exist. Adjust the YAML to match the actual dataset.
- `MLOps.py` copies `output/unet.pth`, `output/yolo.pt`, and `output/deeplab.pth`. If any file is missing, the copy step logs an error. Verify the actual output locations produced by the training scripts and Ultralytics.
- The defect-area thresholds (500 and 2000 pixels) and weld-detection rule in `inference.py` are prototype logic. Validate them against labeled data and production requirements; they are not production-ready inspection criteria.

## Data and model files

Images, model weights, the SQLite database, training outputs, and logs may be large or contain non-code data. Follow your team's data-management policy when deciding what to include in version control, and avoid committing sensitive images or unnecessary large model files.
