import sys
from tkinter import messagebox
import cv2
from PyQt5 import QtGui, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType(r"C:\Users\User\Desktop\Github\Python(영상처리)\pyqt5\opencvUI.ui")[0]

GREEN = (0, 255, 0)
BLAKE = (0, 0, 0)

CONFIDENCE_THRESHOLD = 0.6 #기준이 되는 정확도
# 사전 훈련된 YOLOv8 모델을 불러옵니다.
model = YOLO(r"C:\Users\User\Desktop\Github\Python(영상처리)\camdetection\yolo_model\carnum.pt")

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
            ret, frame = cap.read()
            if ret:
                # 영상이 정상적으로 읽혔으면 change_pixmap_signal을 통해 이미지 전송
                self.change_pixmap_signal.emit(frame)
        # 영상 캡처 종료
        cap.release()

    def stop(self):
        """run_flag를 False로 설정하고 스레드가 종료될 때까지 대기"""
        self._run_flag = False
        self.wait()
        
class video_detection(QThread):
    #numpy의 형식으로 변환
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # 웹캠에서 영상을 캡처
        cap = cv2.VideoCapture(0)

        start = datetime.datetime.now()

        while self._run_flag:
            ret, frame = cap.read()

            if not ret:
                QMessageBox.information(self, "메시지", "전송 오류")
                break

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
                # 객체의 중심점 좌표를 계산합니다.

                # 확대한 바운딩 박스를 프레임에 그립니다.
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                #cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLAKE, 2)

            end = datetime.datetime.now()
            # 초당 프레임 수(FPS)를 계산하고 프레임에 표시합니다.
            fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

            if ret:
                # 영상이 정상적으로 읽혔으면 change_pixmap_signal을 통해 이미지 전송
                self.change_pixmap_signal.emit(frame)
            # 영상 캡처 종료
        cap.release()

    def stop(self):
        """run_flag를 False로 설정하고 스레드가 종료될 때까지 대기"""
        self._run_flag = False
        self.wait()


#메인 윈도우 클래스
class App(QMainWindow, form_class) :
    #초기화 메서드
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Qt Live Video")
        #pushButton (시작버튼)을 클릭하면 아래 fuctionStart 메서드와 연결 됨.
        self.btn_start.clicked.connect(self.functionStart)

         # 비디오 레이블 크기를 저장할 변수
        self.display_width = 0
        self.display_height = 0

        # 비디오 레이블의 크기가 변경될 때 호출되는 이벤트 재정의
    def resizeEvent(self, event):
        self.display_width = self.video_label.width()
        self.display_height = self.video_label.height()

    # 시작버튼을 눌렀을 때 실행되는 메서드
    def functionStart(self):

        if hasattr(self, 'video_thread') and isinstance(self.video_thread, VideoThread):
            self.video_thread.stop()
            self.video_thread.deleteLater()

        if hasattr(self, 'detection_thread') and isinstance(self.detection_thread, video_detection):
            self.detection_thread.stop()
            self.detection_thread.deleteLater()

        if self.cb_detection.isChecked():
            # 비디오 캡처 스레드 생성
            self.thread = video_detection()
            # change_pixmap_signal을 update_image 슬롯에 연결
            self.thread.change_pixmap_signal.connect(self.update_image)
            # 스레드 시작
            self.thread.start()

        else:
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


#코드 실행시 GUI 창을 띄우는 부분
#__name__ == "__main__" : 모듈로 활용되는게 아니라 해당 .py파일에서 직접 실행되는 경우에만 코드 실행
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = App()
    myWindow.show()
    sys.exit(app.exec_())