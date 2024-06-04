import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 포즈 추적 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 카메라 캡처 객체 초기화
cap = cv2.VideoCapture(0)

# 초기 체력 설정
player1_health = 100
player2_health = 100

def draw_stickman(image, landmarks, offset_x, offset_y, color=(0, 255, 0), gloves=False, distance_scale=1.0):
    h, w, _ = image.shape

    # 기준점 (코 위치) 가져오기
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    # 스틱맨의 팔, 다리 및 몸통 그리기
    def draw_limb(start, end, color=color):
        start_point = (int(start.x * w) + offset_x, int(start.y * h) + offset_y)
        end_point = (int(end.x * w) + offset_x, int(end.y * h) + offset_y)
        cv2.line(image, start_point, end_point, color, max(int(distance_scale * 5), 1))

    # 팔
    draw_limb(left_shoulder, left_elbow)
    draw_limb(left_elbow, left_wrist)
    draw_limb(right_shoulder, right_elbow)
    draw_limb(right_elbow, right_wrist)

    # 다리
    draw_limb(left_hip, left_knee)
    draw_limb(left_knee, left_ankle)
    draw_limb(right_hip, right_knee)
    draw_limb(right_knee, right_ankle)

    # 몸통
    draw_limb(left_shoulder, right_shoulder)
    draw_limb(left_hip, right_hip)
    draw_limb(left_shoulder, left_hip)
    draw_limb(right_shoulder, right_hip)

    # 글러브 그리기
    if gloves:
        glove_size = max(int(distance_scale * 20), 1)
        cv2.circle(image, (int(left_wrist.x * w) + offset_x, int(left_wrist.y * h) + offset_y), glove_size, (0, 0, 255), -1)
        cv2.circle(image, (int(right_wrist.x * w) + offset_x, int(right_wrist.y * h) + offset_y), glove_size, (0, 0, 255), -1)

def check_collision(wrist, opponent_landmarks, width, height, offset_x=0, offset_y=0):
    for landmark in opponent_landmarks:
        distance = np.linalg.norm(
            np.array([wrist.x * width + offset_x, wrist.y * height + offset_y]) - 
            np.array([landmark.x * width + offset_x, landmark.y * height + offset_y])
        )
        if distance < 30:  # 임의의 충돌 거리
            return True
    return False

def draw_head(image, landmarks, offset_x, offset_y, color=(255, 255, 255)):
    h, w, _ = image.shape
    
    # 머리 중심 Landmark의 위치
    head_center = (int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * w) + offset_x, int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * h) + offset_y)
    
    # 머리 반지름
    head_radius = int(landmarks[mp_pose.PoseLandmark.NOSE.value].visibility * h / 5)  # 머리 크기를 Landmark의 가시성으로 조절
    
    # 머리 그리기
    cv2.circle(image, head_center, head_radius, color, -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    mid_x = width // 2
    left_frame = frame[:, :mid_x]
    right_frame = frame[:, mid_x:]
    
    image_rgb_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
    result_left = pose.process(image_rgb_left)

    image_rgb_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
    result_right = pose.process(image_rgb_right)

    # 메인 게임 화면 초기화
    game_screen = np.zeros((720, 1280, 3), dtype=np.uint8)

    if result_left.pose_landmarks:
        landmarks_left = result_left.pose_landmarks.landmark

        offset_x_left = 320  # 왼쪽 캐릭터의 x 오프셋
        offset_y_left = 100  # y 오프셋

        draw_stickman(game_screen, landmarks_left, offset_x_left, offset_y_left, color=(0, 255, 0), gloves=True)
        draw_head(game_screen, landmarks_left, offset_x_left, offset_y_left)  # 머리 그리기

        opponent_landmarks_left = [landmarks_left[mp_pose.PoseLandmark.NOSE.value], 
                                  landmarks_left[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                  landmarks_left[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                  landmarks_left[mp_pose.PoseLandmark.LEFT_HIP.value],
                                  landmarks_left[mp_pose.PoseLandmark.RIGHT_HIP.value]]
        if (check_collision(landmarks_left[mp_pose.PoseLandmark.LEFT_WRIST.value], opponent_landmarks_left, width, height, offset_x_left, offset_y_left) or
            check_collision(landmarks_left[mp_pose.PoseLandmark.RIGHT_WRIST.value], opponent_landmarks_left, width, height, offset_x_left, offset_y_left)):
            player2_health -= 1

    if result_right.pose_landmarks:
        landmarks_right = result_right.pose_landmarks.landmark

        offset_x_right = 800  # 오른쪽 캐릭터의 x 오프셋
        offset_y_right = 100  # y 오프셋

        draw_stickman(game_screen, landmarks_right, offset_x_right, offset_y_right, color=(0, 0, 255), gloves=True)
        draw_head(game_screen, landmarks_right, offset_x_right, offset_y_right)  # 머리 그리기

        opponent_landmarks_right = [landmarks_right[mp_pose.PoseLandmark.NOSE.value], 
                                   landmarks_right[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks_right[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                   landmarks_right[mp_pose.PoseLandmark.LEFT_HIP.value],
                                   landmarks_right[mp_pose.PoseLandmark.RIGHT_HIP.value]]
        if (check_collision(landmarks_right[mp_pose.PoseLandmark.LEFT_WRIST.value], opponent_landmarks_right, width, height, offset_x_right, offset_y_right) or
            check_collision(landmarks_right[mp_pose.PoseLandmark.RIGHT_WRIST.value], opponent_landmarks_right, width, height, offset_x_right, offset_y_right)):
            player1_health -= 1

    # 체력 게이지 표시
    cv2.rectangle(game_screen, (50, 50), (50 + player1_health * 2, 80), (0, 255, 0), -1)
    cv2.rectangle(game_screen, (1180, 50), (1180 - player2_health * 2, 80), (0, 0, 255), -1)

    # 카메라 화면을 작은 화면으로 표시
    frame_resized_left = cv2.resize(left_frame, (320, 240))
    frame_resized_right = cv2.resize(right_frame, (320, 240))
    game_screen[480:720, 0:320] = frame_resized_left
    game_screen[480:720, 960:1280] = frame_resized_right

    cv2.imshow('Boxing Game', game_screen)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()