from flask import Flask, Response, render_template
from video_processing import generate_frames

app = Flask(__name__)
#기본 접속 시
@app.route('/')
def index():
    return render_template('index.html')

#/video_feed로 이동 시
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
