import os
import cv2
import json
import copy
import numpy as np
from gevent import pywsgi

from pathlib import Path
from utils.yolo import DetModel, CLASSES
from datetime import datetime
from os.path import join as opj

from multiprocessing import Process, Queue
from flask import Flask, render_template, Response, request, jsonify

from utils.baseclass import BaseClass, workspace
from utils.video_demo import VideoCaptureThread

from enum import Enum

# def judge_selected_id(self):
class CameraID(Enum):
    ID_FIRST = 1
    ID_SECOND = 2

class DangerArea:
    '''危险区域类'''
    def __init__(self):
        # box 不生效
        self.box_valid = False
        self.center = []
        # x_min, y_min, x_max, y_max
        self.box_area = [0,0,0,0]
    
    def set(self, mouse_request:dict):
        '''设置 area'''
        # 设置
        if mouse_request.get("button") == "left":
            if mouse_request["startX"] == mouse_request["endX"] or mouse_request["startY"] == mouse_request["endY"]:
                return
            self.box_valid = True
            self.box_area = [mouse_request["startX"], mouse_request["startY"], mouse_request["endX"], mouse_request["endY"]]
        else:
            # 取消
            self.box_valid = False
        return
    
    @property
    def pt_lt(self):
        return (self.box_area[0], self.box_area[1])
    
    @property
    def pt_rb(self):
        return (self.box_area[2], self.box_area[3])

class VideoApp(BaseClass):
    '''
    负责将接收到的视频流推送到 远程浏览器
    '''
    def __init__(self, conf_path=""):
        super().__init__(conf_path)

        # 1. 解析 json文件
        self.app = Flask(__name__, template_folder=workspace / self.conf["template_folder"])
        self.app.add_url_rule('/', 'index_cavas', self.index)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/receive_coordinates', 'receive_coordinates', self.receive_coordinates, methods=['POST'])

        # 2. 创建目标检测器
        self.detector = DetModel(conf_path, time_measure=False)
        # 创建两个相机的缓存队列
        self.queue_camera_a = Queue(maxsize=1)
        self.queue_camera_b = Queue(maxsize=1)
        
        # 3. 创建两个生产者 线程
        if(os.name == 'nt'):
            self.producer_a = VideoCaptureThread(self.conf["camera_url_a"], 
                                                 self.queue_camera_a, "cam_a", (int(self.conf["img_w"]), int(self.conf["img_h"])), img_flip=eval(self.conf["img_flip"]))
            self.producer_b = VideoCaptureThread(self.conf["camera_url_b"], 
                                                 self.queue_camera_b, "cam_b",(int(self.conf["img_w"]), int(self.conf["img_h"])))
            # 启动线程
            self.producer_a.start()
            self.producer_b.start()

        else:
            self.producer_a = None
            self.producer_b = None

        self.danger_area_first = DangerArea()
        self.danger_area_second = DangerArea()
        self.count = 0

    @staticmethod
    def raw_data_normalize(data):
        '''
        将绘制的box规范化
        '''
        data_copy = copy.deepcopy(data)
        data["startX"] = min(data_copy["startX"], data_copy["endX"])
        data["endX"] = max(data_copy["startX"], data_copy["endX"])

        
        data["startY"] = min(data_copy["startY"], data_copy["endY"])
        data["endY"] = max(data_copy["startY"], data_copy["endY"])

        return data

    def judge_select_camera(self, mouse_request:dict):
        '''
        根据鼠标返回的信息，判断在操作哪一个相机
        '''
        # 根据中心点，判断在是哪一个相机
        center = ((mouse_request["startX"] + mouse_request["endX"]) / 2, 
                  (mouse_request["startY"] +  mouse_request["endY"]) / 2)
        
        if center[1] > (self.post_img_height - 1) or center[0] > (self.post_img_width - 1):
            # 第二个相机的话，对box进行归一化
            self.box_clip_and_normalize(mouse_request, CameraID.ID_SECOND)
            
            return CameraID.ID_SECOND
        else:
            self.box_clip_and_normalize(mouse_request, CameraID.ID_FIRST)
            return CameraID.ID_FIRST

    def box_clip_and_normalize(self, data:dict, camera_id):
        if camera_id == CameraID.ID_FIRST:
            data["endY"] = data["endY"] if data["endY"] < self.post_img_height else self.post_img_height - 1
        else:
            data["startY"] -= self.post_img_height
            data["startY"] = data["startY"] if data["startY"] >= 0 else 0

            data["endY"] -= self.post_img_height
            data["endY"] = data["endY"] if data["endY"] >= 0 else 0

        return data

    @staticmethod
    def cal_box_center(mouse_request:dict):
        return ()

    @staticmethod
    def cal_hor_lineseg_iou(range_a, range_b):
        """
        计算两个线段的水平iou
        """
        left, right = range_a
        left_d, right_d = range_b

        r_border = min(right, right_d)
        l_boarder = max(left, left_d)
        if r_border < l_boarder:
            return 0
        return (r_border - l_boarder) * 1.0 / ((right - left) + (right_d - left_d) - (r_border - l_boarder))

    @staticmethod
    def box_in_danger_area(box, dangre_area : DangerArea):
        #---------------------

        #---------------------
        left, top, right, bottom = box
        left_d, top_d, right_d, bot_d = dangre_area.box_area
        '''
        判断 目标是否在危险区域
        '''
        hor_iou = video_app.cal_hor_lineseg_iou([left, right],[left_d, right_d])
        
        if hor_iou > 0:
            if bottom >= top_d and bottom <= bot_d:
                return True
        return False

    def draw_box(self, images: list, batch_boxes):  
        # 设置 危险区域
        for idx, image in zip(range(2), images):
            if(idx == 0):
                if self.danger_area_first.box_valid:
                    cv2.rectangle(image, self.danger_area_first.pt_lt, self.danger_area_first.pt_rb, (255, 255, 255), 2)
            else:
                if self.danger_area_second.box_valid:
                    cv2.rectangle(image, self.danger_area_second.pt_lt, self.danger_area_second.pt_rb, (255, 255, 255), 2)

        #-------------------------------------------------------
        #	取整，方便画框
        #-------------------------------------------------------
        for idx, image, box_data in zip(range(len(batch_boxes)), images, batch_boxes):
            if idx == 1:
                a = 0
            box_data = np.array(box_data)
            if box_data.shape[0] == 0:
                continue
            #print(box_data.shape[0])
            #print(len(box_data.shape))
            boxes=box_data[...,:4].astype(np.int32) 
            scores=box_data[...,4]
            classes=box_data[...,5].astype(np.int32) 

            for box, score, cl in zip(boxes, scores, classes):
                left, top, right, bottom = box
                # print('class: {}, score: {}'.format(CLASSES[cl], score))
                # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

                # cv2.rectangle 是 OpenCV 中用于在图像上绘制矩形的函数。这里用它来绘制边界框。
                # (top, left) 和 (right, bottom) 分别是矩形（边界框）的左上角和右下角坐标。
                # (255, 0, 0) 是颜色代码，这里表示蓝色。2 是线条的粗细。

                if idx == 0:
                    if self.danger_area_first.box_valid and VideoApp.box_in_danger_area(box, self.danger_area_first):
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
                
                elif idx == 1:
                    if self.danger_area_second.box_valid and VideoApp.box_in_danger_area(box, self.danger_area_second):
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)

                # cv2.putText 是 OpenCV 中用于在图像上绘制文本的函数。
                # '{} {:.2f}'.format(CLASSES[cl], score) 生成标签文本，显示类别名称和置信度（保留两位小数）。
                # (top, left) 是文本的起始坐标，通常设置在边界框的左上角。
                # cv2.FONT_HERSHEY_SIMPLEX 指定字体类型。
                # 0.6 是字体缩放比例。(0, 0, 255) 是字体颜色，这里表示红色。2 是线条的粗细。
                '''
                cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                            (top, left ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            cv2.imshow(f'Detected Image {idx}', image)

        cv2.waitKey(5)  # 等待用户按键
                '''

    def generate(self):
        
        while True:
            '''
            在这里拿 yolo 后处理的数据
            '''

            # 队列都有数据
            if(self.queue_camera_a.full() and self.queue_camera_b.full()):
                # 获得两个摄像头的推理结果
                raw_img_first = self.queue_camera_a.get()
                raw_img_second = self.queue_camera_b.get()

                batch_boxes = self.detector.detect([copy.deepcopy(raw_img_first[1]), copy.deepcopy(raw_img_second[1])])
                # 解析两个摄像头的推理结果
                raw_img_first_darw = copy.deepcopy(raw_img_first[1])
                raw_img_second_darw = copy.deepcopy(raw_img_second[1])
                self.draw_box([raw_img_first_darw, raw_img_second_darw], batch_boxes)
            else:
                '''
                ret = True
                frame = cv2.imread('/media/marc/DATA1/work/code/CAMERA/doc/test.jpg')
                self.detector.detect(frame, frame)
                '''
                continue
            # if not ret:
            #     break
            
            self.count += 1
            if(self.count > 1000):
                self.count = 0
            if self.count % 5 == 0:
                frame = cv2.vconcat([raw_img_first_darw, raw_img_second_darw])
                # frame = cv2.resize(frame, (320, 384))
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
        data = video_app.raw_data_normalize(data)
        #print(f"Received coordinates: {data}")
        
        # 判断设置的哪一个相机
        camera_id = self.judge_select_camera(data)
        # 设置对应相机的危险区域
        if camera_id == CameraID.ID_FIRST:
            self.danger_area_first.set(data)
        else:
            
            self.danger_area_second.set(data)

        return jsonify(status="success")
    
    #def box_pos_normalize(self, data:dict):


    def run(self, host='0.0.0.0', debug=False, port=5000):
        self.app.run(host = host, debug=debug, port = port)

        '''
        self.server = pywsgi.WSGIServer((host, port), self.app)
        self.server.serve_forever()
        '''

if __name__ == '__main__':

    video_app = VideoApp(conf_path="conf/conf.json")
    video_app.run()



    '''
    detector = DetModel()
    frame = cv2.imread('/media/marc/DATA1/work/code/CAMERA/doc/test.jpg')
    detector.detect(frame, frame)
    '''