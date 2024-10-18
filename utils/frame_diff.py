import cv2
class FrameDiff:
    def __init__(self, mask_area=None) -> None:
        self.init = False
        self.first_frame = None
        self.mask_area = mask_area
        self.serial_del_count = 0
    
    def update_bg(self, img):
        self.first_frame = img

    def run(self, img):
        # cv2.namedWindow("raw", 0)
        # cv2.namedWindow("res", 0)
        if self.mask_area is not None:
            #cv2.imshow("raw", img)
            img[self.mask_area[0]:self.mask_area[1], self.mask_area[2]:self.mask_area[3]] = 0
            #cv2.imshow("res", img)
            #cv2.waitKey(0)
        if not self.init:
            self.init = True
            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 对灰度图进行模糊处理
            self.first_frame = cv2.GaussianBlur(gray1, (5, 5), 0)
            return False

        move = True
        # 转为灰度图
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        # 计算帧差
        diff = cv2.absdiff(self.first_frame, gray2)

        # 应用阈值
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # 膨胀操作，填补空洞
        dilated = cv2.dilate(thresh, None, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area  > 150:  # 设定最小面积阈值
                #move = False
                # 绘制轮廓和大小
                x, y, w, h = cv2.boundingRect(contour)
                # if w < 100 and h < 100:
                move = False
                self.serial_del_count = 0
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(img, f'Area: {area}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if(move):
            self.serial_del_count = self.serial_del_count + 1
        if(self.serial_del_count > 21):
            self.serial_del_count = 0
            move = False
        # 显示结果
        # cv2.imshow('Motion Detection', img)
        # cv2.waitKey(0)
        self.update_bg(gray2)
        return move