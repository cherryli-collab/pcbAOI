import sys  # 系統參數
import cv2  # 相機影像
import numpy as np  # 數值運算
from PyQt5.QtWidgets import *  # GUI 元件
from PyQt5.QtGui import *  # GUI 圖片
from PyQt5.QtCore import *  # GUI 訊號

from inference import run_pipeline  # 匯入推論流程

class AOI_GUI(QWidget):  # 定義 GUI 主視窗
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DC Motor AI AOI System")  # 視窗標題
        self.setGeometry(100, 100, 1200, 700)  # 視窗大小

        layout = QHBoxLayout()  # 主水平布局
        left = QVBoxLayout()  # 左側：影像
        right = QVBoxLayout()  # 右側：結果

        self.cam_label = QLabel("Camera Feed")  # 顯示相機畫面
        left.addWidget(self.cam_label)

        self.status_label = QLabel("Status: ---")  # 顯示 OK/NG
        self.status_label.setStyleSheet("font-size: 30px;")
        right.addWidget(self.status_label)

        self.defect_list = QListWidget()  # 顯示缺陷列表
        right.addWidget(self.defect_list)

        layout.addLayout(left)
        layout.addLayout(right)
        self.setLayout(layout)

        self.timer = QTimer()  # 定時器
        self.timer.timeout.connect(self.update_frame)  # 每次觸發更新畫面
        self.timer.start(50)  # 20 FPS

        self.cap = cv2.VideoCapture(0)  # 開啟相機

    def update_frame(self):
        ret, frame = self.cap.read()  # 讀取相機影像
        if not ret:
            return

        status, defects = run_pipeline(frame)  # 執行推論

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(img))  # 顯示影像

        self.status_label.setText(f"Status: {status}")  # 更新狀態

        self.defect_list.clear()  # 清空列表
        for d in defects:
            self.defect_list.addItem(d)  # 加入缺陷

app = QApplication(sys.argv)
window = AOI_GUI()
window.show()
sys.exit(app.exec_())
