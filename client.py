import cv2
import mediapipe as mp
import numpy as np
import socket
import pickle

# 서버 정보 설정
SERVER = "127.0.0.1"
PORT = 5555
ADDR = (SERVER, PORT)

# MediaPipe 포즈 추적 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 소켓 초기화
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

while True:
    # 카메라 캡처 객체 초기화
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    # 카메라 입력 처리
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        landmarks = []
        for landmark in result.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        # 포즈 정보를 서버로 전송
        data = pickle.dumps(landmarks)
        client.sendall(data)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.close()
