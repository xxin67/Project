# hand_gesture.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import math

class HandGestureRecognizer:
    def __init__(self, model_path=None, num_hands=2,
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.75):

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.gesture_labels = ["none", "one", "two", "three", "four", "five",
                               "six", "seven", "eight", "nine", "ten"]
        # 角度阈值（度），超过此值认为手指伸直
        self.straight_angle_threshold = 170

    def _angle_between_points(self, p1, p2, p3):
        """
        计算以 p2 为顶点的夹角（度数）
        输入为三个点，每个点有 x, y 属性（可以是归一化坐标或世界坐标）
        """
        # 计算向量 p2->p1 和 p2->p3
        v1 = (p1.x - p2.x, p1.y - p2.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)
        # 点积
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        # 模长
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])
        if norm1 == 0 or norm2 == 0:
            return 0
        cos_angle = dot / (norm1 * norm2)
        # 限制在 [-1, 1] 内，防止数值误差
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    def _is_finger_straight(self, landmarks, tip_idx, pip_idx, dip_idx=None):
        """
        判断手指是否伸直
        - 对于食指、中指、无名指、小指：使用 PIP、DIP、TIP 三点，以 DIP 为顶点计算角度
        - 对于拇指：使用 MCP、IP、TIP 三点，以 IP 为顶点
        返回 True 表示伸直，False 表示弯曲
        """
        if dip_idx is None:
            # 拇指：点2(MCP), 点3(IP), 点4(TIP)
            mcp = landmarks[tip_idx - 2]  # 根据拇指特殊处理，这里用参数传入
            ip = landmarks[tip_idx - 1]
            tip = landmarks[tip_idx]
            angle = self._angle_between_points(mcp, ip, tip)
        else:
            # 其他手指：PIP、DIP、TIP
            pip = landmarks[pip_idx]
            dip = landmarks[dip_idx]
            tip = landmarks[tip_idx]
            angle = self._angle_between_points(pip, dip, tip)
        return angle > self.straight_angle_threshold

    def _count_fingers(self, hand_landmarks):

        # 定义每个手指的关键点索引（指尖, 近节指间关节, 远节指间关节）
        # 拇指没有独立的近节指间，我们传参时特殊处理
        fingers = [
            (4, 2, 3),   # 拇指：指尖4，掌指关节2，指间关节3（dip_idx 传 None 但这里我们统一用三个点）
            (8, 6, 7),   # 食指：指尖8，PIP=6，DIP=7
            (12, 10, 11), # 中指：指尖12，PIP=10，DIP=11
            (16, 14, 15), # 无名指：指尖16，PIP=14，DIP=15
            (20, 18, 19)  # 小指：指尖20，PIP=18，DIP=19
        ]
        count = 0
        for i, (tip, pip, dip) in enumerate(fingers):
            if i == 0:  # 拇指
                # 使用 MCP(2)、IP(3)、TIP(4)
                mcp = hand_landmarks[tip - 2]   # 索引 2
                ip = hand_landmarks[tip - 1]    # 索引 3
                tip_pt = hand_landmarks[tip]    # 索引 4
                angle = self._angle_between_points(mcp, ip, tip_pt)
            else:
                # 其他手指：PIP, DIP, TIP
                pip_pt = hand_landmarks[pip]
                dip_pt = hand_landmarks[dip]
                tip_pt = hand_landmarks[tip]
                angle = self._angle_between_points(pip_pt, dip_pt, tip_pt)
            if angle > self.straight_angle_threshold:
                count += 1
        # 限制范围 0~10
        return min(count, 10)

    # 保留原来的距离法（改名），便于对比
    def _count_fingers_distance(self, hand_landmarks):
        wrist = hand_landmarks[0]
        index_mcp = hand_landmarks[5]

        base_dist_sq = self._dist_sq(wrist, index_mcp)
        if base_dist_sq == 0:
            base_dist_sq = 1.0
        base = base_dist_sq / 0.6

        tips = [4, 8, 12, 16, 20]
        count = 0
        for tip_idx in tips:
            tip = hand_landmarks[tip_idx]
            if tip_idx == 4:
                if self._dist_sq(index_mcp, tip) > base * 0.3:
                    count += 1
            else:
                if self._dist_sq(wrist, tip) > base:
                    count += 1
        return min(count, 10)

    def _dist_sq(self, p1, p2):
        return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

    def get_gesture_and_landmarks(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)
        gesture_num = 0
        landmarks = None
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            gesture_num = self._count_fingers(landmarks)  # 使用角度法
        return gesture_num, landmarks

    def _draw_landmarks(self, image, landmarks):
        h, w, _ = image.shape
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for start, end in connections:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保留原有 process_frame 方法（可选）
    def process_frame(self, frame_bgr):
        gesture_num, landmarks = self.get_gesture_and_landmarks(frame_bgr)
        if landmarks is not None:
            self._draw_landmarks(frame_bgr, landmarks)
        return gesture_num, frame_bgr