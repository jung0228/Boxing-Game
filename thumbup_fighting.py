import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 손 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 체력 초기화
player1_health = 30
player2_health = 30

# 비디오 캡처 초기화 (웹캠)
cap = cv2.VideoCapture(0)

def draw_lightsaber(image, wrist, thumb, color=(0, 255, 0)):
    # Calculate the direction from wrist to thumb
    direction = (thumb[0] - wrist[0], thumb[1] - wrist[1])
    length = int(np.hypot(direction[0], direction[1]) * 2)  # Adjust length if necessary
    end_point = (wrist[0] + direction[0] * 2, wrist[1] + direction[1] * 2)
  
    # Draw the lightsaber handle
    cv2.line(image, wrist, end_point, color, 20)
  
    # Add glow effect
    for i in range(1, 6):
        cv2.line(image, wrist, end_point, (color[0], int(color[1]*0.6), int(color[2]*0.6)), 20 - i*3)
    
    return end_point

game_over = False
winner = ""

while cap.isOpened() and not game_over:
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Mediapipe로 손 찾기
    results = hands.process(image)

    # BGR로 다시 변환
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 손목 위치 저장 변수
    player1_wrist = None
    player2_wrist = None
    player1_thumb = None
    player2_thumb = None

    player1_lightsaber_end = None
    player2_lightsaber_end = None

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 손목과 엄지의 랜드마크 추출
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            wrist_pos = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
            thumb_pos = (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0]))

            # 중간 지점 계산
            middle_x = (wrist_pos[0] + thumb_pos[0]) // 2

            # 각 플레이어에 손목 위치 저장 및 색상 설정
            if player1_wrist is None:
                player1_wrist = wrist_pos
                player1_thumb = thumb_pos
                if middle_x < image.shape[1] // 2:
                    color = (255, 0, 0)  # Blue lightsaber for player 1 (left side)
                else:
                    color = (0, 0, 255)  # Red lightsaber for player 1 (right side)
                player1_lightsaber_end = draw_lightsaber(image, wrist_pos, thumb_pos, color)
            else:
                player2_wrist = wrist_pos
                player2_thumb = thumb_pos
                if middle_x < image.shape[1] // 2:
                    color = (255, 0, 0)  # Blue lightsaber for player 2 (left side)
                else:
                    color = (0, 0, 255)  # Red lightsaber for player 2 (right side)
                player2_lightsaber_end = draw_lightsaber(image, wrist_pos, thumb_pos, color)

            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 충돌 감지 및 체력 감소
    if player1_lightsaber_end and player2_wrist:
        collision_distance = 50  # 충돌 범위 조정 가능
        distance = ((player1_lightsaber_end[0] - player2_wrist[0]) ** 2 + (player1_lightsaber_end[1] - player2_wrist[1]) ** 2) ** 0.5
        if distance < collision_distance:
            player2_health -= 1

    if player2_lightsaber_end and player1_wrist:
        collision_distance = 50  # 충돌 범위 조정 가능
        distance = ((player2_lightsaber_end[0] - player1_wrist[0]) ** 2 + (player2_lightsaber_end[1] - player1_wrist[1]) ** 2) ** 0.5
        if distance < collision_distance:
            player1_health -= 1

    # 체력 상태 표시
    cv2.putText(image, f'Player 1 Health: {player1_health}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f'Player 2 Health: {player2_health}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 체력이 0 이하가 되면 게임 오버
    if player1_health <= 0:
        game_over = True
        winner = "Player 2 Wins!"
    elif player2_health <= 0:
        game_over = True
        winner = "Player 1 Wins!"

    # 이미지 출력
    cv2.imshow('Lightsaber Fight Game', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# 게임 종료 후 승자 출력
if game_over:
    game_over_image = np.zeros_like(image)
    text1 = "GAME OVER!!!"
    text2 = winner
    
    # 텍스트 크기와 위치 계산
    text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
    text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
    text1_x = (game_over_image.shape[1] - text1_size[0]) // 2
    text1_y = (game_over_image.shape[0] - text1_size[1]) // 2
    text2_x = (game_over_image.shape[1] - text2_size[0]) // 2
    text2_y = text1_y + 100

    for i in range(10):
        if i % 2 == 0:
            game_over_image = np.zeros_like(image)
        else:
            cv2.putText(game_over_image, text1, (text1_x, text1_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            cv2.putText(game_over_image, text2, (text2_x, text2_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        
        cv2.imshow('Lightsaber Fight Game', game_over_image)
        cv2.waitKey(500)

hands.close()
cap.release()
cv2.destroyAllWindows()
