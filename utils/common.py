import numpy as np
from timeit import default_timer as timer


def decorator_timer(function):
    def wrapper(*args, **kwargs):
        start = timer()
        result = function(*args, **kwargs)
        elapsed = (timer() - start)
        
        return result, elapsed
    return wrapper

# 坐标转换
def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

#dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class 
#thresh: 阈值
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #-------------------------------------------------------
	#   计算框的面积
    #	置信度从大到小排序
	#-------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1] 

    # 只要还有索引（即待处理的检测框）存在，循环就会继续执行。
    while index.size > 0:
        i = index[0]
        keep.append(i)
		#-------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        #-------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]]) 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)                              
        h = np.maximum(0, y22 - y11 + 1) 

        overlaps = w * h
        #-------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        #-------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


