from flask import Flask, render_template, Response, request, jsonify
import cv2

class VideoApp:
    '''
    负责将接收到的视频流推送到 远程浏览器
    '''
    def __init__(self, url = ""):
        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index_cavas', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/receive_coordinates', 'receive_coordinates', self.receive_coordinates, methods=['POST'])
        
        self.url = url
        self.cap = cv2.VideoCapture(self.url) 

    def generate(self):
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (640, 480))
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def index(self):
        return render_template('help_button.html')

    def video_feed(self):
        return Response(self.generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def receive_coordinates(self):
        data = request.json
        print(f"Received coordinates: {data}")
        self.danger_area = data
        return jsonify(status="success")

    def run(self, host='0.0.0.0', debug=False, port=5000):
        self.app.run(host = host, debug=debug, port = port)

if __name__ == '__main__':
    video_app = VideoApp(url='rtsp://admin:falconix12@10.168.2.172:554/h264/ch1/main/av_stream')
    video_app.run()
