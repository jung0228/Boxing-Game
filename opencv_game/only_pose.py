import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 포즈 추적 모델 초기화!!!
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 카메라 캡처 객체 초기화
cap = cv2.VideoCapture(0)

def draw_stickman(image, landmarks):
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
    def draw_limb(start, end, color=(0, 255, 0)):
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(image, start_point, end_point, color, 5)

    # 팔
    draw_limb(left_shoulder, left_elbow, (0, 255, 0))
    draw_limb(left_elbow, left_wrist, (0, 255, 0))
    draw_limb(right_shoulder, right_elbow, (0, 255, 0))
    draw_limb(right_elbow, right_wrist, (0, 255, 0))

    # 다리
    draw_limb(left_hip, left_knee, (0, 255, 0))
    draw_limb(left_knee, left_ankle, (0, 255, 0))
    draw_limb(right_hip, right_knee, (0, 255, 0))
    draw_limb(right_knee, right_ankle, (0, 255, 0))

    # 몸통
    draw_limb(left_shoulder, right_shoulder, (0, 255, 0))
    draw_limb(left_hip, right_hip, (0, 255, 0))
    draw_limb(left_shoulder, left_hip, (0, 255, 0))
    draw_limb(right_shoulder, right_hip, (0, 255, 0))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    # 메인 게임 화면 초기화
    game_screen = np.zeros((720, 1280, 3), dtype=np.uint8)

    if result.pose_landmarks:
        draw_stickman(game_screen, result.pose_landmarks.landmark)

    # 카메라 화면을 작은 화면으로 표시
    frame_resized = cv2.resize(frame, (320, 240))
    game_screen[480:720, 0:320] = frame_resized

    cv2.imshow('Boxing Game', game_screen)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
