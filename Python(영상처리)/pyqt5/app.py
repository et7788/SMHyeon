import sys
from tkinter import messagebox
import cv2
from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from ultralytics import YOLO
import logging
import opencv_ui

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType(r"C:\Users\User\Desktop\Github\Python(영상처리)\pyqt5\opencvUI.ui")[0]

#메인 윈도우 클래스
class App(QMainWindow, form_class):
    #초기화 메서드
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Qt Live Video")
        self.cb_detection.stateChanged.connect(self.state_dectection)

        # VideoThread 인스턴스 생성
        self.thread = opencv_ui.VideoThread()

        # pushButton (시작버튼)을 클릭하면 아래 fuctionStart 메서드와 연결 됨.
        self.btn_start.clicked.connect(self.functionStart)
        # pushButton (종료버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_stop.clicked.connect(self.closeEvent)
        # pushButton (좌우반전버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_hor.clicked.connect(self.functionHorizontal)
        # pushButton (상하반전버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_ver.clicked.connect(self.funtionVertical)
        # pushButton (블러처리버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_blur.clicked.connect(self.funtionBlur)
        # pushButton (히스토그램버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_hist.clicked.connect(self.funtionHist)
        # pushButton (Gaussian버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_gaussian.clicked.connect(self.funtionGaussian)
        # pushButton (Median버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_median.clicked.connect(self.funtionMedian)
        # pushButton (Bilateral버튼)을 클릭하면 아래 functionstop 메서드와 연결 됨.
        self.btn_bilateral.clicked.connect(self.funtionBilateral)
        # 비디오 레이블 크기를 저장할 변수
        self.display_width = 0
        self.display_height = 0

        # 비디오 레이블의 크기가 변경될 때 호출되는 이벤트 재정의
    def resizeEvent(self, event):
        logging.warn("resizeEvent")
        self.display_width = self.video_label.width()
        self.display_height = self.video_label.height()
    
    def state_dectection(self):
        if self.cb_detection.isChecked():
            self.thread.chk_detection = 1
        else:
            self.thread.chk_detection = 0
        logging.warn(self.thread.chk_detection)

    # 시작버튼을 눌렀을 때 실행되는 메서드
    def functionStart(self):
        logging.warn("functionStart")
        self.thread._run_flag = True
        # change_pixmap_signal을 update_image 슬롯에 연결
        self.thread.change_pixmap_signal.connect(self.update_image)
        # 스레드 시작
        self.thread.start()

    def functionHorizontal(self):
        logging.warn("horizontal")
        self.thread.btn_num = 1
        self.thread.flip_num = 1
        self.thread.flip_enabled = not self.thread.flip_enabled

    def funtionVertical(self):
        logging.warn("vertical")
        self.thread.btn_num = 1
        self.thread.flip_num = 0
        self.thread.flip_enabled = not self.thread.flip_enabled

    def funtionBlur(self):
        logging.warn("blur")
        self.thread.btn_num = 2
        self.thread.blur_num = 8
        self.thread.blur_enabled = not self.thread.blur_enabled

    def funtionHist(self):
        logging.warn("funtionHist")
        self.thread.btn_num = 3
        self.thread.hist_enabled = not self.thread.hist_enabled
        self.thread.change_pixmap_ghist.connect(self.update_ghist)
        self.thread.change_pixmap_chist.connect(self.update_chist)

    def funtionGaussian(self):
        logging.warn("Gaussian")
        self.thread.btn_num = 4
        self.thread.gaussian_size = 9
        self.thread.gaussian_enabled = not self.thread.gaussian_enabled   

    def funtionMedian(self):
        logging.warn("Median")
        self.thread.btn_num = 5
        self.thread.median_size = 9
        self.thread.median_enabled = not self.thread.median_enabled

    def funtionBilateral(self):
        logging.warn("Bilateral")
        self.thread.btn_num = 6
        self.thread.bilateral_size = 9
        self.thread.bilateral_enabled = not self.thread.bilateral_enabled

    def closeEvent(self, event):
        logging.warn("closeEvent")
        # 어플리케이션이 종료될 때 캡처 스레드를 종료
        self.thread._run_flag = False

    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        """update_image 슬롯을 통해 새로운 OpenCV 이미지로 video_label을 업데이트"""
        qt_img = self.convert_cv_qt(frame)
        self.video_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, frame):
        """OpenCV 이미지를 QPixmap으로 변환"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(QtGui.QImage)
    def update_ghist(self, hist):
        qt_ghist = self.convert_hist_qt(hist)
        self.hist_label.setPixmap(qt_ghist)

    @pyqtSlot(QtGui.QImage)
    def update_chist(self, hist):
        qt_chist = self.convert_hist_qt(hist)
        self.hist_label2.setPixmap(qt_chist)

    def convert_hist_qt(self, hist):
        # hist_label의 크기에 맞게 qimg 리사이즈
        label_width = self.hist_label.width()
        label_height = self.hist_label.height()
        qimg = hist.scaled(label_width, label_height, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(qimg)
#코드 실행시 GUI 창을 띄우는 부분
#__name__ == "__main__" : 모듈로 활용되는게 아니라 해당 .py파일에서 직접 실행되는 경우에만 코드 실행
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = App()
    myWindow.show()
    sys.exit(app.exec_())