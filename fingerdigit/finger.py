import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "fingerdigit/hand_landmarker.task"          

# 手指关键点索引
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5, 9, 13, 17]

#手指连接线，参考mediapipe的定义
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),      
    (0,5), (5,6), (6,7), (7,8),      
    (0,9), (9,10), (10,11), (11,12), 
    (0,13), (13,14), (14,15), (15,16), 
    (0,17), (17,18), (18,19), (19,20), 
    (5,9), (9,13), (13,17)           
]

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

#手指伸直判断函数
def count_fingers(hand_landmarks, handedness, frame_width, frame_height):

    landmarks = [(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in hand_landmarks]
    finger_count = 0

    # 拇指（使用X坐标）
    tip_x, _ = landmarks[FINGER_TIPS[0]]
    base_x, _ = landmarks[FINGER_BASES[0]]
    if handedness == 'Left':
        if tip_x > base_x:
            finger_count += 1
    else:  # Right
        if tip_x < base_x:
            finger_count += 1

    # 其余四指（使用Y坐标）
    for i in range(1, 5):
        tip_y = landmarks[FINGER_TIPS[i]][1]
        base_y = landmarks[FINGER_BASES[i]][1]
        if tip_y < base_y:
            finger_count += 1

    return finger_count


# ==================== 初始化 HandLandmarker ====================
def init_hand_landmarker(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


# ==================== 主程序 ====================
def main():
    cap = cv2.VideoCapture(701)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)

    landmarker = init_hand_landmarker(MODEL_PATH)

    print("按 'q' 键退出程序")
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 水平翻转（镜像效果）
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        # 转换为 RGB 并创建 MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 检测手部关键点
        timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        total_fingers = 0

        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # 获取左右手标签
                handedness = detection_result.handedness[i][0].category_name

                # 统计手指数
                finger_num = count_fingers(hand_landmarks, handedness, frame_width, frame_height)
                total_fingers += finger_num

                # 绘制手部关键点和连接线（手动绘制，避免依赖 solutions）
                # 绘制关键点（红色圆点）
                for lm in hand_landmarks:
                    cx = int(lm.x * frame_width)
                    cy = int(lm.y * frame_height)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                # 绘制连接线（绿色线条）
                for connection in HAND_CONNECTIONS:
                    start = hand_landmarks[connection[0]]
                    end = hand_landmarks[connection[1]]
                    start_point = (int(start.x * frame_width), int(start.y * frame_height))
                    end_point = (int(end.x * frame_width), int(end.y * frame_height))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                # 在手腕上方显示左右手标签和手指数
                wrist = hand_landmarks[0]
                cx = int(wrist.x * frame_width)
                cy = int(wrist.y * frame_height) - 30
                cv2.putText(frame, f"{handedness}: {finger_num}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 显示总手指数
        cv2.putText(frame, f"Total Fingers: {total_fingers}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition (MediaPipe Tasks API)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()