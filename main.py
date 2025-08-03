import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QFrame
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from object_detect import VideoBox
from board import BoardApp

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOv5 Object Detection')
        self.setGeometry(100, 100, 1400, 900)

        # main layout
        main_layout = QHBoxLayout()

        # left side (video + logs)
        left_frame = QVBoxLayout()

        # video label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(960, 720)
        self.video_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        left_frame.addWidget(QLabel('실시간 CCTV'))
        left_frame.addWidget(self.video_label)

        # right side (log)
        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setFixedSize(400, 800)

        main_layout.addLayout(left_frame)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

        # set up VideoBox with PyQt widgets
        self.video_frame = self.video_label
        self.board = PyQtBoard(self.log_box)
        self.vb = VideoBox(
            address='쓰러짐 감지!',
            frame=self,  # not used directly in PyQt
            label=self.video_label,
            source=0,
            board=self.board
        )

        # setup timer to simulate video_play
        self.timer = QTimer()
        self.timer.timeout.connect(self.vb.main_page)
        self.timer.start(30)  # approx 33 fps


class PyQtBoard:
    def __init__(self, log_widget):
        self.num = 1
        self.board_data = []
        self.log_widget = log_widget

    def update_board(self, address, source):
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_info = f"[{self.num}]\n{current_time}\n{address}\n{source}\n\n"
        self.num += 1
        self.board_data.insert(0, new_info)
        self.log_widget.clear()
        for info in self.board_data:
            self.log_widget.append(info)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
