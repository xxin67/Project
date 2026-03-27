# hand_gesture.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class HandGestureRecognizer:
    def __init__(self, model_path=None, num_hands=2,
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.75):
        if model_path is None:
            if os.path.exists("hand_landmarker.task"):
                model_path = "hand_landmarker.task"
            else:
                model_path = "hand_landmarker.task"

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

    def _dist_sq(self, p1, p2):
        return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

    def _count_fingers(self, hand_landmarks):
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

    # 新增方法：返回手势数字和原始关键点（不绘制）
    def get_gesture_and_landmarks(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)
        gesture_num = 0
        landmarks = None
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]  # 取第一只手
            gesture_num = self._count_fingers(landmarks)
        return gesture_num, landmarks

    # 原有方法保留（如果其他地方用到）
    def process_frame(self, frame_bgr):
        gesture_num, landmarks = self.get_gesture_and_landmarks(frame_bgr)
        if landmarks is not None:
            # 绘制在原始帧上（坐标不变）
            self._draw_landmarks(frame_bgr, landmarks)
        return gesture_num, frame_bgr

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