from tkinter import messagebox
import cv2
from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import datetime
from ultralytics import YOLO
import logging
import app
import matplotlib.pylab as plt
import io
import matplotlib
matplotlib.use('agg')

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

CONFIDENCE_THRESHOLD = 0.6 #기준이 되는 정확도
# 사전 훈련된 YOLOv8 모델을 불러옵니다.
model = YOLO(r"C:\Users\User\Desktop\Github\Python(영상처리)\camdetection\yolo_model\carnum.pt")

class VideoThread(QThread):
    # 작업이 끝났을 때 메인 스레드로 결과를 전달하기 위해 시그널 numpy의 형식으로 변환
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pixmap_ghist = pyqtSignal(QtGui.QImage)
    change_pixmap_chist = pyqtSignal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.btn_num = 0 
        self.chk_detection = 0
        self.flip_enabled = False
        self.flip_num = -1
        self.blur_enabled = False
        self.blur_num = 0
        self.hist_enabled = False
        self.gaussian_size = 0
        self.gaussian_enabled = False
        self.median_size = 0
        self.median_enabled = False
        self.bilateral_size = 0
        self.bilateral_enabled = False

    # 스레드에서 실행될 작업 정의
    def run(self):
        logging.warn("run")
        # 웹캠에서 영상을 캡처
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                if self.btn_num == 1:
                    self.check_flip(frame)
                elif self.btn_num == 2:
                    self.check_blur(frame)
                elif self.btn_num == 3:
                    self.check_hist(frame)
                elif self.btn_num == 4:
                    self.check_gaussian(frame)
                elif self.btn_num == 5:
                    self.check_median(frame)
                elif self.btn_num == 6:
                    self.check_bilateral(frame)
                else :
                    self.run_frame(frame)

        # 영상 캡처 종료
        cap.release()

    def run_frame(self, frame):
        if self.chk_detection == 1:
            # 영상이 정상적으로 읽혔으면 change_pixmap_signal을 통해 이미지 전송
            self.detection(frame)
        else:
            self.normal(frame)
    
    def check_flip(self, frame):
        if self.flip_enabled:
            frame = self.flip_frame(frame)
            self.run_frame(frame)
        else :
            self.run_frame(frame)

    def check_blur(self, frame):
        if self.blur_enabled:
            frame = self.blur_frame(frame)
            self.run_frame(frame)
        else :
            self.run_frame(frame)

    def check_hist(self, frame):
        if self.hist_enabled:
            self.run_frame(frame)
            ghist = self.gray_hist_frame(frame)
            chist = self.color_hist_frame(frame)
            self.change_pixmap_ghist.emit(ghist)
            self.change_pixmap_chist.emit(chist)
        else :
            self.run_frame(frame)

    def check_gaussian(self, frame):
        if self.gaussian_enabled:
            frame = self.gaussian_frame(frame)
            self.run_frame(frame)
        else :
            self.run_frame(frame)

    def check_median(self, frame):
        if self.median_enabled:
            frame = self.median_frame(frame)
            self.run_frame(frame)
        else :
            self.run_frame(frame)

    def check_bilateral(self, frame):
        if self.bilateral_enabled:
            frame = self.bilateral_frame(frame)
            self.run_frame(frame)
        else :
            self.run_frame(frame)

    def normal(self, frame):
        self.change_pixmap_signal.emit(frame)

    def detection(self, frame):
        #객체 인식 시작 시간
        start = datetime.datetime.now()
        # YOLO 모델을 이용하여 프레임에서 객체 인식 수행
        detections = model(frame)[0]
        # 객체 검출 결과에 대해 반복합니다.
        for box in detections.boxes.data.tolist():
            # 예측에 연관된 신뢰도(확률)를 추출합니다.
            confidence = box[4]
            # 신뢰도가 최소 신뢰도보다 작은 경우 약한 검출은 건너뜁니다.
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # 신뢰도가 최소 신뢰도보다 큰 경우, 바운딩 박스 좌표를 가져옵니다.
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # 객체의 클래스 이름을 저장합니다.
            class_name = model.names[int(box[5])]

            # 바운딩 박스를 프레임에 그립니다.
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2)

        #객체 인식 종료 시간
        end = datetime.datetime.now()
        # 초당 프레임 수(FPS)를 계산하고 프레임에 표시합니다.
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 8)

        # 영상이 정상적으로 읽혔으면 change_pixmap_signal을 통해 이미지 전송
        self.change_pixmap_signal.emit(frame)

    def stop(self):
        """run_flag를 False로 설정하고 스레드가 종료될 때까지 대기"""
        self._run_flag = False
        self.wait()

    def flip_frame(self, frame):
        # 프레임을 좌우,상하 반전시키는 로직을 구현
        if self.flip_num == 1:
            flip_frame = cv2.flip(frame, 1)
        else :
            flip_frame = cv2.flip(frame, 0)

        return flip_frame
    
    def blur_frame(self, frame):
        kernel = np.ones((self.blur_num, self.blur_num), np.float32)/self.blur_num**2
        return cv2.filter2D(frame, -1, kernel)
    
    def gray_hist_frame(self, frame):
        # Grayscale 이미지로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 히스토그램 추출
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        # 0에서 255 사이로 스케일링
        hist = hist / hist.max() * 255
        # 히스토그램 값을 정수형으로 변환
        hist = np.uint8(hist)

        return self.gray_hist_plt(hist)
    
    def color_hist_frame(self, frame):
        # 히스토그램 추출
        hist_r = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([frame], [2], None, [256], [0, 256])

        # 0에서 255 사이로 스케일링
        hist_r = hist_r / hist_r.max() * 255
        hist_g = hist_g / hist_g.max() * 255
        hist_b = hist_b / hist_b.max() * 255

        # 히스토그램 값을 정수형으로 변환
        hist_r = np.uint8(hist_r)
        hist_g = np.uint8(hist_g)
        hist_b = np.uint8(hist_b)

        # 히스토그램 배열의 크기
        hist_size = len(hist_r)
        # 히스토그램 배열을 이미지로 변환
        hist_image = np.zeros((hist_size, 256, 3), dtype=np.uint8)
        for i in range(hist_size):
            # 각 픽셀값의 높이를 히스토그램 값으로 지정
            y_r = int(255 - hist_r[i])
            y_g = int(255 - hist_g[i])
            y_b = int(255 - hist_b[i])
            cv2.line(hist_image, (i, 255), (i, y_r), (255, 0, 0), 1)  # Red channel
            cv2.line(hist_image, (i, 255), (i, y_g), (0, 255, 0), 1)  # Green channel
            cv2.line(hist_image, (i, 255), (i, y_b), (0, 0, 255), 1)  # Blue channel

        # 히스토그램 그래프 그리기
        plt.plot(hist_r, color='red')
        plt.plot(hist_g, color='green')
        plt.plot(hist_b, color='blue')

        # 그래프를 이미지로 변환하여 QImage로 반환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        qimg = QtGui.QImage.fromData(buf.getvalue())

        return qimg

    def gray_hist_plt(self, hist):
        # 히스토그램 배열의 크기
        hist_size = len(hist)
        # 히스토그램 배열을 이미지로 변환
        hist_image = np.zeros((hist_size, 256), dtype=np.uint8)
        for i in range(hist_size):
            # 각 픽셀값의 높이를 히스토그램 값으로 지정
            y = int(255 - hist[i])
            cv2.line(hist_image, (i, 255), (i, y), 255)
        # 히스토그램 그래프 그리기
        plt.plot(hist)
        # 그래프를 이미지로 변환하여 QImage로 반환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        qimg = QtGui.QImage.fromData(buf.getvalue())

        return qimg

    def gaussian_frame(self, frame):
        frame = np.clip((frame / 255 + np.random.normal(scale=0.1, size  = frame.shape)) * 255, 0, 255).astype('uint8')
        gaussian = cv2.GaussianBlur(frame, (self.gaussian_size, self.gaussian_size), 0)
        return gaussian

    def median_frame(self, frame):
        N = 10000
        idx1 = np.random.randint(frame.shape[0], size = N)
        idx2 = np.random.randint(frame.shape[1], size = N)
        frame[idx1, idx2] = 0

        median = cv2.medianBlur(frame, self.median_size)

        return median
    
    def bilateral_frame(self, frame):
        frame = np.clip((frame / 255 + np.random.normal(scale=0.1, size  = frame.shape)) * 255, 0, 255).astype('uint8')
        bilateral = cv2.bilateralFilter(frame, self.bilateral_size, 75, 75)
        
        return bilateral