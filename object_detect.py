import cv2
import numpy as np
import torch
import sys
import os
import time
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

FILE = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(FILE)
YOLOV5_DIR = os.path.join(PROJECT_DIR, 'yolov5')
sys.path.append(YOLOV5_DIR)
sys.path.append(PROJECT_DIR)

from yolov5.models.yolo import DetectionModel
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

weights_path = r'C:\Users\suyeo\Downloads\fall_detection_project\yolov5\runs\train\weights\best_windows.pt'

with torch.serialization.safe_globals({'models.yolo.DetectionModel': DetectionModel}):
    model = DetectMultiBackend(weights_path, device='cpu')


class VideoBox:
    def __init__(self, address, frame, label, source, warning_label):
        self.video_frame = frame
        self.video_label = label
        self.warning_label = warning_label
        self.source = source
        self.cap = cv2.VideoCapture(self.source)

        self.check = 0
        self.start = 0
        self.end = 0

    def video_play(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # ----- YOLO 입력 준비 -----
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0
        img = img.unsqueeze(0)

        # ----- YOLO 추론 -----
        results = model(img)
        results = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)
        preds = results[0]

        # YOLO 추론 코드 (생략)

        detect_fall = 0
        for *xyxy, conf, cls in preds:
            if conf > 0.4:
                class_name = model.names[int(cls)]
                if class_name.lower() in ['fallen', 'lying']:
                    detect_fall = 1
                    label = f'{class_name} {conf:.2f}'  # ← WARNING 제거!
                else:
                    label = f'{class_name} {conf:.2f}'

                cv2.rectangle(frame,
                              (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])),
                              (255, 0, 0), 2)
                cv2.putText(frame, label,
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

        # --- 상태 체크 ---
        if detect_fall:
            if self.check == 0:
                self.start = time.time()
                self.check = 1
            else:
                self.end = time.time()

            elapsed = self.end - self.start if self.end and self.start else 0

            if elapsed >= 3:  # 3초 이상
                self.video_frame.setStyleSheet("background-color: red;")

                # WARNING 글자 흰색으로 표시
                if self.warning_label:
                    self.warning_label.setText("WARNING")
                    self.warning_label.setStyleSheet("""
                        color: white;
                        font-size: 80px;
                        font-weight: bold;
                        background-color: transparent;
                    """)
                    self.warning_label.show()
        else:
            # 초기화
            self.check = 0
            self.start = 0
            self.end = 0
            self.video_frame.setStyleSheet("background-color: white;")
            if self.warning_label:
                self.warning_label.hide()

        # 영상 PyQt 라벨에 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)
