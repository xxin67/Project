import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 手势标签（对应伸出的手指个数，0～10）
gesture = ["none", "one", "two", "three", "four", "five",
           "six", "seven", "eight", "nine", "ten"]

# 配置 HandLandmarker 参数
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                     # 最多检测2只手
    min_hand_detection_confidence=0.75,   # 检测置信度
    min_tracking_confidence=0.75
)

# 初始化手部关键点检测器
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# 打开摄像头（请将 701 改为正确的索引，如 0）
cap = cv2.VideoCapture(701)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 定义辅助函数：计算两点之间的平方距离
def dist_sq(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

# 辅助函数：绘制手部关键点和连接线
def draw_landmarks_on_image(rgb_image, detection_result):
    """在图像上绘制手部关键点和连接线"""
    hand_landmarks_list = detection_result.hand_landmarks
    for hand_landmarks in hand_landmarks_list:
        # 绘制关键点
        for landmark in hand_landmarks:
            x = int(landmark.x * rgb_image.shape[1])
            y = int(landmark.y * rgb_image.shape[0])
            cv2.circle(rgb_image, (x, y), 3, (0, 255, 0), -1)
        # 绘制连接线（根据 MediaPipe 定义的连接关系）
        connections = [
            (0,1),(1,2),(2,3),(3,4),      # 大拇指
            (0,5),(5,6),(6,7),(7,8),      # 食指
            (0,9),(9,10),(10,11),(11,12), # 中指
            (0,13),(13,14),(14,15),(15,16), # 无名指
            (0,17),(17,18),(18,19),(19,20), # 小指
            (5,9),(9,13),(13,17)           # 手掌连接
        ]
        for start, end in connections:
            x1 = int(hand_landmarks[start].x * rgb_image.shape[1])
            y1 = int(hand_landmarks[start].y * rgb_image.shape[0])
            x2 = int(hand_landmarks[end].x * rgb_image.shape[1])
            y2 = int(hand_landmarks[end].y * rgb_image.shape[0])
            cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return rgb_image

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 RGB（MediaPipe Tasks 需要 RGB 图像）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 创建 MediaPipe Image 对象
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 进行手部检测
    detection_result = hand_landmarker.detect(mp_image)

    flag = 0   # 记录伸出的手指数量

    # 如果有检测到手部
    if detection_result.hand_landmarks:
        # 取第一只手（如果你需要处理两只手，可遍历）
        hand_landmarks = detection_result.hand_landmarks[0]

        # 获取手腕（索引0）和食指根部（索引5）的坐标
        p0 = hand_landmarks[0]
        p5 = hand_landmarks[5]

        # 基准距离（大拇指根部到手腕距离，用于判断指尖是否伸直）
        base_dist_sq = dist_sq(p0, p5)
        # 添加一个小偏移，避免除零
        base = base_dist_sq / 0.6 if base_dist_sq > 0 else 1.0

        # 各指尖索引
        tips = [4, 8, 12, 16, 20]   # 大拇指、食指、中指、无名指、小指
        for i, tip_idx in enumerate(tips):
            tip = hand_landmarks[tip_idx]
            dist_to_wrist = dist_sq(p0, tip)

            # 大拇指特殊处理（因为大拇指根部不是手腕，而是索引5）
            if tip_idx == 4:
                # 大拇指伸直时指尖与食指根部的距离应大于某个阈值
                if dist_sq(p5, tip) > base * 0.3:
                    flag += 1
            else:
                if dist_to_wrist > base:
                    flag += 1

        # 限制范围
        flag = min(flag, 10)

        # 绘制关键点和连接线（在 BGR 图像上绘制）
        draw_landmarks_on_image(frame, detection_result)

    # 显示手势结果
    cv2.putText(frame, gesture[flag], (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0, 0, 255), 3)
    cv2.imshow('MediaPipe Hand Landmarker', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()