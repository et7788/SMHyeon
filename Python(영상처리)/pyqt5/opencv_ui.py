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

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

CONFIDENCE_THRESHOLD = 0.6 #기준이 되는 정확도
# 사전 훈련된 YOLOv8 모델을 불러옵니다.
model = YOLO(r"C:\Users\User\Desktop\Github\Python(영상처리)\camdetection\yolo_model\carnum.pt")

class VideoThread(QThread):
    # 작업이 끝났을 때 메인 스레드로 결과를 전달하기 위해 시그널 numpy의 형식으로 변환
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pixmap_hist = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.btn_num = 0 
        self.chk_detection = 0
        self.flip_enabled = False
        self.flip_num = -1
        self.blur_enabled = False
        self.blur_num = 0

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
                    self.hist_frame(frame)
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
    
    def hist_frame(self, frame):
        channels = cv2.split(frame)
        colors = ('b','g','r')
        for (ch, color) in zip (channels, colors):
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
            plt.plot(hist, color = color)

