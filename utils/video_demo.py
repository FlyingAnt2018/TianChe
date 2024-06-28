import cv2
import threading
import queue
from datetime import datetime

class VideoCaptureThread(threading.Thread):
    def __init__(self, video_source, frame_queue, thread_name, img_size = (640, 384)):
        threading.Thread.__init__(self)
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.thread_name = thread_name
        self.cap = None
        self.running = True
        self.img_size = img_size

    def run(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"Error opening video source {self.video_source}")
            return
        
        while self.running:
            # 读取一帧图像
            ret, frame = self.cap.read()
            # 图像缩放
            if not ret:
                print(f"Failed to grab frame from {self.video_source}")
                break
            frame = cv2.resize(frame, self.img_size)

            # Try to put the frame in the queue
            if not self.frame_queue.full():
                # Put the frame in the queue
                time_stamp = int(datetime.now().timestamp())
                self.frame_queue.put((self.thread_name, frame, time_stamp))
            '''
            else:
                # If queue is full, replace the existing frame with the new frame
                try:
                    self.frame_queue.get_nowait()  # Remove the old frame
                except queue.Empty:
                    pass
                time_stamp  = int(datetime.now().timestamp())
                self.frame_queue.put((self.thread_name, frame, time_stamp))
            '''
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        self.cap.release()

    def stop(self):
        self.running = False

def time_sync():
    pass

def gpu_inference():
    pass

def cpu_post_process():
    pass

def push_image_brawser():
    pass

def do_inference(frame_queue_a, frame_queued_b):
    '''
    对获取到的两帧图像：
    1. 时间同步
    2. 封装到一个batch进行推理
    3. 推送 flask 并显示
    ''' 
    # 1.0 当队列数量比较多的时候，需要显式时间同步
    time_sync()

    # 2.0 图像推理
    gpu_inference()

    # 3.0 检测结果后处理
    cpu_post_process()

    while True:
        if not frame_queue_a.empty():

            thread_name_a, frame_a, ts = frame_queue_a.get()
            thread_name_b, frame_b, ts = frame_queued_b.get()
            cv2.imshow(f"Frame from {thread_name_a}", frame_a)
            cv2.imshow(f"Frame from {thread_name_b}", frame_b)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue_a = queue.Queue(maxsize=1)
    frame_queue_b = queue.Queue(maxsize=1)
    # Create threads for different video sources
    thread1 = VideoCaptureThread('rtsp://admin:falconix12@10.168.2.172:554/h264/ch1/main/av_stream', frame_queue_a, "Webcam")  # Usually, 0 is the webcam
    thread2 = VideoCaptureThread("rtsp://admin:falconix12@10.168.2.172:554/h264/ch1/main/av_stream", frame_queue_b, "Video File")  # Another video file

    # Start threads
    thread1.start()
    thread2.start()

    # Start the frame display function
    do_inference(frame_queue_a, frame_queue_b)

    # Wait for threads to finish
    thread1.join()
    thread2.join()
