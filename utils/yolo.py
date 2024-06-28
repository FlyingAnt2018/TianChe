import onnxruntime
import numpy as np
import cv2
import os
import glob
from typing import List, Tuple, AnyStr
from .baseclass import BaseClass
import copy

from utils.common import decorator_timer, xywh2xyxy, nms
CLASSES = ['P']

class DetModel(BaseClass):
    def __init__(self, conf_path:str, time_measure=False):
        
        super().__init__(conf_path)
        onnxpath = self.conf["onnx_path"]
        self.conf_thre = float(self.conf["conf_thre"])
        self.time_measure = time_measure
        self.onnx_session=onnxruntime.InferenceSession(onnxpath)
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()
        print(f"DetModel input_name = {self.input_name}, output_name = {self.output_name}")
   
    #-------------------------------------------------------
    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    #-------------------------------------------------------
    
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    #-------------------------------------------------------
	#   输入图像
	#-------------------------------------------------------
    def get_input_feed(self,img_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=img_tensor
        return input_feed
    
    def pre_process(self, img: list):
        # 1. GR2RGB和HWC2CHW
        # 2. 两帧数据封装为一个 batch
        img = [im[:,:,::-1].transpose(2,0,1)[np.newaxis, :] for im in img]
        img = np.concatenate((img[0], img[1]), axis=0)

        # 将图像数据类型转换为 float32
        img=img.astype(dtype=np.float32)
        # 将像素值归一化
        img/=255.0
        
        return img

    def inference(self, img: list):

        # 准备模型输入和推理
        input_feed=self.get_input_feed(img)
        #pred=self.onnx_session.run(None,input_feed)[0]
        if self.time_measure:
            time_onnx = []
            for i in range(50):
                pred, onnx_time =self.onnx_run(self.onnx_session, input_feed)
                time_onnx.append(onnx_time)
            print(f'| onnx inference time = {np.array(time_onnx[2:]).mean()}')
        else:
            pred, onnx_time =self.onnx_run(self.onnx_session, input_feed)

        # 返回模型的预测结果 pred 
        return pred
    
    def detect(self, img:list):
        # 数据预处理
        img = self.pre_process(img)

        # 推理
        pred = self.inference(img)

        # 后处理
        outputs = self.post_process(pred)

        return outputs
    
    def post_process(self, batch_boxes, iou_thres = 0.5):
        ''' #过滤掉无用的框'''
        outputs = []
        #-------------------------------------------------------
        #   遍历batch维度
        #	删除置信度小于conf_thres的BOX
        #-------------------------------------------------------
        for org_box in batch_boxes:
            org_box=np.squeeze(org_box)
            conf = org_box[..., 4] > self.conf_thre
            box = org_box[conf == True]
            #-------------------------------------------------------
            #	通过argmax获取置信度最大的类别
            #-------------------------------------------------------
            cls_cinf = box[..., 5:]
            cls = []
            for i in range(len(cls_cinf)):
                cls.append(int(np.argmax(cls_cinf[i])))
            all_cls = list(set(cls))     
            #-------------------------------------------------------
            #   分别对每个类别进行过滤
            #	1.将第6列元素替换为类别下标
            #	2.xywh2xyxy 坐标转换
            #	3.经过非极大抑制后输出的BOX下标
            #	4.利用下标取出非极大抑制后的BOX
            #-------------------------------------------------------
            output = []

            for i in range(len(all_cls)):
                curr_cls = all_cls[i]
                curr_cls_box = []
                curr_out_box = []
                for j in range(len(cls)):
                    if cls[j] == curr_cls:
                        box[j][5] = curr_cls
                        curr_cls_box.append(box[j][:6])
                curr_cls_box = np.array(curr_cls_box)
                # curr_cls_box_old = np.copy(curr_cls_box)
                curr_cls_box = xywh2xyxy(curr_cls_box)
                curr_out_box = nms(curr_cls_box,iou_thres)
                for k in curr_out_box:
                    output.append(curr_cls_box[k])
            #output = np.array(output)
            outputs.append(output)
        
        #outputs = np.array(outputs)
        return outputs

    @decorator_timer
    def onnx_run(self, sesson, input):
        return  sesson.run(None, input)[0]

def draw(images: list, batch_boxes):  
    #-------------------------------------------------------
    #	取整，方便画框
	#-------------------------------------------------------
    print(batch_boxes)
    for idx, image, box_data in zip(range(2), images, batch_boxes):
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
            # cv2.imwrite('res.jpg',or_img) #保存该图片
            # cv2.imshow(f'Detected Image {idx}', image)
    # cv2.waitKey(0)  # 等待用户按键
    # cv2.destroyAllWindows()


if __name__=="__main__":
    onnx_path='/media/marc/DATA1/work/code/CAMERA/weights/model_bs2.onnx'
    model=DetModel(onnx_path)
    img_path = '/media/marc/DATA_DISK/Object_detection/BGVP-dataset/BGVP-main/BGVP-main/Dataset/valid/valid/'
    img_paths = glob.glob(os.path.join(img_path, '*.jpg'))[:20]
    for path in img_paths:
        #print(path)
        img = cv2.imread("/media/marc/DATA1/work/code/CAMERA/doc/test.jpg")
        img = cv2.resize(img, (640, 384))
        ori_img = copy.deepcopy(img)
        outbox = model.detect([img, img])
        if outbox != []:
            draw([ori_img, ori_img], outbox)