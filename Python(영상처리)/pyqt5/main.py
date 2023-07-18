from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

class VideoThread(QThread):
    #numpy의 형식으로 변환
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # 웹캠에서 영상을 캡처
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                # 영상이 정상적으로 읽혔으면 change_pixmap_signal을 통해 이미지 전송
                self.change_pixmap_signal.emit(cv_img)
        # 영상 캡처 종료
        cap.release()

    def stop(self):
        """run_flag를 False로 설정하고 스레드가 종료될 때까지 대기"""
        self._run_flag = False
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640
        self.display_height = 480
        # 이미지를 보여줄 라벨 생성
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # 텍스트 라벨 생성
        self.textLabel = QLabel('Webcam')

        # 수직 박스 레이아웃 생성하고 두 라벨을 추가
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # 수직 박스 레이아웃을 위젯의 레이아웃으로 설정
        self.setLayout(vbox)

        # 비디오 캡처 스레드 생성
        self.thread = VideoThread()
        # change_pixmap_signal을 update_image 슬롯에 연결
        self.thread.change_pixmap_signal.connect(self.update_image)
        # 스레드 시작
        self.thread.start()

    def closeEvent(self, event):
        # 어플리케이션이 종료될 때 캡처 스레드를 종료
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """update_image 슬롯을 통해 새로운 OpenCV 이미지로 image_label을 업데이트"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """OpenCV 이미지를 QPixmap으로 변환"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
