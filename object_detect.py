import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import sys
import os
import time
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
 
FILE = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(FILE)
YOLOV5_DIR = os.path.join(PROJECT_DIR, 'yolov5')
sys.path.append(YOLOV5_DIR)
sys.path.append(PROJECT_DIR)

import torch
from yolov5.models.yolo import DetectionModel
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

weights_path = r'C:\Users\parkg\Desktop\fall_detection_project\yolov5\runs\train\weights\best_windows.pt'

with torch.serialization.safe_globals({'models.yolo.DetectionModel': DetectionModel}):
    model = DetectMultiBackend(weights_path, device='cpu')



class VideoBox:

    def __init__(self, address, frame, label, source, board):
        self.__check = 0
        self.__start = 0
        self.__end = 0
        self.__board_check = 0
        self.__address = address
        self.__video_frame = frame
        self.__video_label = label
        self.__source = source
        self.board = board

    def get_source(self):
        return self.__source
    
    def main_page(self): 
        cap = cv2.VideoCapture(self.__source)

        def video_play():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img.transpose((2, 0, 1))  # (H, W, C) → (C, H, W)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float()
            img /= 255.0
            img = img.unsqueeze(0)  # (1, 3, 640, 640)

            results = model(img)
            results = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)       
            preds = results[0]      # 바운딩 박스 좌표 등 추출

            detect_fall = 0
            for *xyxy, conf, cls in preds:
                if conf > 0.4:
                    class_name = model.names[int(cls)]
                    if class_name.lower() in ['fall', 'lying']:
                        label = f'fallen {conf:.2f}'
                        detect_fall = 1
                    else:
                        label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if class_name in ['fallen', 'lying']:
                        detect_fall = 1

            # 예측이 아무것도 없을 때
            if len(preds) == 0:
                self._check = 0
                self._start = 0
                self._end = 0
                self._board_check = 0
                self.video_frame.setStyleSheet("background-color: red;")
            else:
                if detect_fall == 1:
                    if self.__check == 0:
                        self.__start = time.time()
                        self.__check = 1
                    else:
                        self.__end = time.time()

                    if 5 <= (self.__end - self.__start):
                        self.__video_frame.config(bg='red')
                        if self.__board_check == 0:
                            self.__check = 2
                            self.__board_check = 1
                            self.board.update_board(self.__address, self.__source)
                else:
                    self.__check = 0
                    self.__start = 0
                    self.__end = 0
                    self.__board_check = 0
                    self.__video_frame.config(bg='white')

            video = frame
            cv2img = cv2.cvtColor(video, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.__video_label.imgtk = imgtk
            self.__video_label.configure(image=imgtk)
            self.__video_label.after(10, video_play)

        video_play()
