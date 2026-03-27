# cameo.py
import cv2
from managers import WindowManager, CaptureManager
from hand_gesture import HandGestureRecognizer

class Cameo(object):

    def __init__(self,width=640, height=480):
        cap = cv2.VideoCapture(701)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        # 关闭 CaptureManager 自带的镜像
        self._captureManager = CaptureManager(
            cap, self._windowManager, False)
        self._gesture_recognizer = HandGestureRecognizer()

    def _draw_landmarks_on_mirror(self, image, landmarks):
        """在镜像图像上绘制手部关键点和连接线（坐标已镜像）"""
        h, w, _ = image.shape
        # 绘制关键点（坐标镜像）
        for landmark in landmarks:
            x = int((1 - landmark.x) * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # 绘制连接线（坐标镜像）
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for start, end in connections:
            x1 = int((1 - landmarks[start].x) * w)
            y1 = int(landmarks[start].y * h)
            x2 = int((1 - landmarks[end].x) * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def run(self):
        """运行主循环"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                # 获取手势数字和原始关键点
                gesture_num, landmarks = self._gesture_recognizer.get_gesture_and_landmarks(frame)
                # 创建镜像帧（用于显示）
                mirror_frame = cv2.flip(frame, 1)

                # 如果有关键点，在镜像帧上绘制（坐标已镜像）
                if landmarks is not None:
                    self._draw_landmarks_on_mirror(mirror_frame, landmarks)

                # 在镜像帧上显示文字（文字不需要镜像）
                cv2.putText(mirror_frame,
                            self._gesture_recognizer.gesture_labels[gesture_num],
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 0, 255), 3)

                # 将镜像帧替换到 CaptureManager 的 _frame 中，以便显示和录像
                self._captureManager._frame = mirror_frame

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        if keycode == 32:   # 空格键：截图
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # Tab键：开始/停止录像
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # Esc键：退出
            self._windowManager.destroyWindow()

if __name__ == "__main__":
    Cameo().run()