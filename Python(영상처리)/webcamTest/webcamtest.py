#OpenCV로 카메라 연동
import cv2
#flask를 이용한 웹사이트 송출
from flask import Flask, Response

app = Flask(__name__)


def generate_frames():
    #VideoCapture(0)이 기본설정 캠
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # HTML 파일로 전송할 프레임을 jpg형태로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 프레임을 HTML 스트림으로 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.0.241', debug=True)
