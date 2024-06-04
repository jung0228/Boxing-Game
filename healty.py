import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 포즈 추적 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 카메라 캡처 객체 초기화
cap = cv2.VideoCapture(0)

# 초기 체력 설정
player1_health = 100
player2_health = 100

def draw_stickman(image, landmarks, color=(0, 255, 0), gloves=False):
    h, w, c = image.shape

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
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(image, start_point, end_point, color, 5)

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
        glove_size = 20
        cv2.circle(image, (int(left_wrist.x * w), int(left_wrist.y * h)), glove_size, (0, 0, 255), -1)
        cv2.circle(image, (int(right_wrist.x * w), int(right_wrist.y * h)), glove_size, (0, 0, 255), -1)

def check_collision(wrist, opponent_landmarks, width, height):
    for landmark in opponent_landmarks:
        distance = np.linalg.norm(np.array([wrist.x * width, wrist.y * height]) - np.array([landmark.x * width, landmark.y * height]))
        if distance < 30:  # 임의의 충돌 거리
            return True
    return False

# 전체 화면 크기 설정
screen_width = 1280
screen_height = 720

# 카메라 화면 크기 설정
camera_width = 640
camera_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    # 전체 게임 화면 초기화
    game_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        draw_stickman(game_screen, landmarks, color=(0, 255, 0), gloves=True)

        opponent_landmarks = [landmarks[mp_pose.PoseLandmark.NOSE.value], 
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]]

        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # 왼손 충돌 검사
        if check_collision(left_wrist, opponent_landmarks, width, height):
            player2_health -= 1

        # 오른손 충돌 검사
        if check_collision(right_wrist, opponent_landmarks, width, height):
            player1_health -= 1

    # 체력 게이지 표시
    cv2.rectangle(game_screen, (30, 30), (30 + player1_health * 2, 60), (0, 255, 0), -1)
    cv2.rectangle(game_screen, (screen_width - 30, 30), (screen_width - 30 - player2_health * 2, 60), (0, 0, 255), -1)

    # 카메라 화면을 작은 화면으로 표시
    frame_resized = cv2.resize(frame, (camera_width, camera_height))
    game_screen[screen_height - camera_height - 30:screen_height - 30, screen_width - camera_width - 30:screen_width - 30] = frame_resized

    cv2.imshow('Boxing Game', game_screen)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
