import os
import cv2
import json
import copy
import numpy as np


from pathlib import Path
from utils.yolo import YOLOV5, CLASSES
from datetime import datetime
from os.path import join as opj

from multiprocessing import Process, Queue
from flask import Flask, render_template, Response, request, jsonify

from utils.baseclass import BaseClass
from utils.video_demo import VideoCaptureThread

    
class VideoApp(BaseClass):
    '''
    负责将接收到的视频流推送到 远程浏览器
    '''
    def __init__(self, conf_path=""):

        super().__init__(conf_path)

        # 1. 解析 json文件
        template_folder = self.conf["template_folder"]

        self.app = Flask(__name__, template_folder)
        self.app.add_url_rule('/', 'index_cavas', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/receive_coordinates', 'receive_coordinates', self.receive_coordinates, methods=['POST'])

        # 2. 创建目标检测器
        self.detector = YOLOV5(conf_path, time_measure=False)
        # 创建两个相机的缓存队列
        self.queue_camera_a = Queue(maxsize=1)
        self.queue_camera_b = Queue(maxsize=1)
        
        # 3. 创建两个生产者 线程
        if(os.name == 'nt'):
            self.producer_a = VideoCaptureThread(self.conf["camera_url_a"], 
                                                 self.queue_camera_a, "cam_a", (int(self.conf["img_w"]), int(self.conf["img_h"])))
            self.producer_b = VideoCaptureThread(self.conf["camera_url_b"], 
                                                 self.queue_camera_b, "cam_b",(int(self.conf["img_w"]), int(self.conf["img_h"])))
            # 启动线程
            self.producer_a.start()
            self.producer_b.start()

        else:
            self.producer_a = None
            self.producer_b = None

    def draw_box(self, images: list, batch_boxes):  
        #-------------------------------------------------------
        #	取整，方便画框
        #-------------------------------------------------------
        #print(batch_boxes)
        for idx, image, box_data in zip(range(2), images, batch_boxes):
            box_data = np.array(box_data)
            if box_data.shape[0] == 0:
                continue
            #print(box_data.shape[0])
            #print(len(box_data.shape))
            boxes=box_data[...,:4].astype(np.int32) 
            scores=box_data[...,4]
            classes=box_data[...,5].astype(np.int32) 

            for box, score, cl in zip(boxes, scores, classes):
                top, left, right, bottom = box
                # print('class: {}, score: {}'.format(CLASSES[cl], score))
                # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

                # cv2.rectangle 是 OpenCV 中用于在图像上绘制矩形的函数。这里用它来绘制边界框。
                # (top, left) 和 (right, bottom) 分别是矩形（边界框）的左上角和右下角坐标。
                # (255, 0, 0) 是颜色代码，这里表示蓝色。2 是线条的粗细。
                cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)

                # cv2.putText 是 OpenCV 中用于在图像上绘制文本的函数。
                # '{} {:.2f}'.format(CLASSES[cl], score) 生成标签文本，显示类别名称和置信度（保留两位小数）。
                # (top, left) 是文本的起始坐标，通常设置在边界框的左上角。
                # cv2.FONT_HERSHEY_SIMPLEX 指定字体类型。
                # 0.6 是字体缩放比例。(0, 0, 255) 是字体颜色，这里表示红色。2 是线条的粗细。
                cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                            (top, left ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            cv2.imshow(f'Detected Image {idx}', image)
        cv2.waitKey(5)  # 等待用户按键


    def generate(self):
        
        while True:
            '''
            在这里拿 yolo 后处理的数据
            '''
            # 队列都有数据
            if(self.queue_camera_a.full() and self.queue_camera_b.full()):
                # 获得两个摄像头的推理结果
                raw_img_a = self.queue_camera_a.get()
                raw_img_b = self.queue_camera_b.get()

                batch_boxes = self.detector.detect([copy.deepcopy(raw_img_a[1]), copy.deepcopy(raw_img_b[1])])
                # 解析两个摄像头的推理结果
                self.draw_box([raw_img_a[1], raw_img_b[1]], batch_boxes)


            else:
                '''
                ret = True
                frame = cv2.imread('/media/marc/DATA1/work/code/CAMERA/doc/test.jpg')
                self.detector.detect(frame, frame)
                '''
                continue
            # if not ret:
            #     break
            frame = cv2.vconcat([raw_img_a[1], raw_img_b[1]])
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

    video_app = VideoApp(conf_path="conf/conf.json")
    video_app.run()



    '''
    detector = YOLOV5()
    frame = cv2.imread('/media/marc/DATA1/work/code/CAMERA/doc/test.jpg')
    detector.detect(frame, frame)
    '''