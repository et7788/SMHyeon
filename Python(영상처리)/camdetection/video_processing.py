import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
BLAKE = (0, 0, 0)
video_dir = r"C:\Users\User\Desktop\Github\Python(영상처리)\camdetection\data\test.mp4"

# 비디오 캡처 객체를 초기화합니다.
video_cap = cv2.VideoCapture(video_dir)
# 비디오 프레임 크기를 가져옵니다.
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 사전 훈련된 YOLOv8 모델을 불러옵니다.
model = YOLO(r"C:\Users\User\Desktop\Github\Python(영상처리)\camdetection\data\yolov8n.pt")
# DeepSort는 실시간 객체 추적을 수행하는 알고리즘 중 하나입니다.
# max_age는 트랙의 최대 수명을 나타내는 매개변수로, 개체의 최대 트랙 지속 시간을 결정합니다.
tracker = DeepSort(max_age=50)

def generate_frames():
    while True:
        start = datetime.datetime.now()

        ret, frame = video_cap.read()

        if not ret:
            break

        # 현재 프레임에 대해 YOLO 모델을 실행합니다.
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
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2

            # 바운딩 박스의 크기를 확대합니다.
            width = xmax - xmin
            height = ymax - ymin

            expand_ratio = 1.2  # 확대 비율 조정 가능
            expanded_width = int(width * expand_ratio)
            expanded_height = int(height * expand_ratio)

            # 확대한 바운딩 박스의 좌표를 계산합니다.
            expanded_xmin = max(center_x - expanded_width // 2, 0)
            expanded_ymin = max(center_y - expanded_height // 2, 0)
            expanded_xmax = expanded_xmin + expanded_width
            expanded_ymax = expanded_ymin + expanded_height

            # 확대한 바운딩 박스를 프레임에 그립니다.
            cv2.rectangle(frame, (expanded_xmin, expanded_ymin), (expanded_xmax, expanded_ymax), GREEN, 2)
            cv2.putText(frame, class_name, (expanded_xmin, expanded_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLAKE, 2)
                 
        # 초당 프레임 수(FPS)를 계산하고 프레임에 표시합니다.
        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # 프레임을 JPEG 이미지로 인코딩합니다.
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 클라이언트에게 프레임을 전송합니다.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
