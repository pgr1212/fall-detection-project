import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QSizePolicy
from object_detect import VideoBox


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOv5 쓰러짐 감지 시스템')
        self.resize(2000, 1500)  # 전체 창 크기 (적당히 크게)

        # 메인 레이아웃
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
        layout.setSpacing(5)

        # 1️⃣ WARNING 라벨 (상단 고정)
        self.warning_label = QLabel("WARNING", self)
        self.warning_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 중앙 정렬
        self.warning_label.setStyleSheet("""
            color: red;
            font-size: 60px;
            font-weight: bold;
            background-color: white;
        """)
        self.warning_label.hide()

        # 2️⃣ 카메라 출력 라벨 (화면을 꽉 채우도록 설정)
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 창 크기에 맞게 확장
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        # 레이아웃 배치
        layout.addWidget(self.warning_label, alignment=Qt.AlignHCenter | Qt.AlignTop)
        layout.addWidget(self.video_label, stretch=1)  # stretch=1 → 남은 공간을 모두 채움

        self.setLayout(layout)

        # VideoBox 연결
        self.vb = VideoBox(
            address='쓰러짐 감지!',
            frame=self,
            label=self.video_label,
            source=0,
            warning_label=self.warning_label
        )

        # 타이머 실행
        self.timer = QTimer()
        self.timer.timeout.connect(self.vb.video_play)
        self.timer.start(30)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
