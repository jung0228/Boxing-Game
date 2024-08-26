import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 손 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 체력 초기화
player1_health = 50
player2_health = 50
max_health = 50

# 비디오 캡처 초기화 (웹캠)
cap = cv2.VideoCapture(0)

# 이미지 로드 및 크기 조정
overlay_image = cv2.imread('overlay.png')  # 'overlay.png'를 원하는 이미지 파일로 교체
overlay_image = cv2.resize(overlay_image, (200, 200))

def draw_lightsaber(image, wrist, thumb, color=(0, 255, 0)):
    # Calculate the direction from wrist to thumb
    direction = (thumb[0] - wrist[0], thumb[1] - wrist[1])
    length = int(np.hypot(direction[0], direction[1]) * 2)  # Adjust length if necessary
    end_point = (wrist[0] + direction[0] * 2, wrist[1] + direction[1] * 2)
  
    # Draw the lightsaber handle
    cv2.line(image, wrist, end_point, color, 20)
  
    return end_point

def point_to_line_distance(point, line_start, line_end):
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    nearest = None
    if t < 0.0:
        nearest = line_start
    elif t > 1.0:
        nearest = line_end
    else:
        nearest = line_start + t * line_vec
    return np.linalg.norm(np.array(point) - nearest)

def draw_health_bar(image, health, max_health, position, size=(300, 20), color=(0, 255, 0)):
    x, y = position
    w, h = size
    health_ratio = health / max_health
    current_w = int(w * health_ratio)
    
    # Draw background rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (50, 50, 50), -1)
    
    # Draw health rectangle
    cv2.rectangle(image, (x, y), (x + current_w, y + h), color, -1)
    
    # Draw border
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

def overlay_image_on_hand(image, overlay_img, wrist_pos):
    overlay_height, overlay_width = overlay_img.shape[:2]
    top_left = (int(wrist_pos[0] - overlay_width / 2), int(wrist_pos[1] - overlay_height))  # 이미지를 위로 올립니다.

    # Ensure the overlay fits within the main image
    if top_left[0] < 0 or top_left[1] < 0 or top_left[0] + overlay_width > image.shape[1] or top_left[1] + overlay_height > image.shape[0]:
        return  # Skip overlay if it doesn't fit

    # Create a mask of the overlay image and its inverse mask
    gray_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Region of interest (ROI) in the main image
    roi = image[top_left[1]:top_left[1] + overlay_height, top_left[0]:top_left[0] + overlay_width]

    # Black-out the area of the overlay in the ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of overlay image.
    img2_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    # Put overlay in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    image[top_left[1]:top_left[1] + overlay_height, top_left[0]:top_left[0] + overlay_width] = dst


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

    # 중앙선 그리기
    middle_x = image.shape[1] // 2
    cv2.line(image, (middle_x, 0), (middle_x, image.shape[0]), (0, 0, 0), 5)

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
                if wrist_pos[0] < image.shape[1] // 2:
                    color = (255, 0, 0)  # Blue lightsaber for player 1 (left side)
                else:
                    color = (0, 0, 255)  # Red lightsaber for player 1 (right side)
                player1_lightsaber_end = draw_lightsaber(image, wrist_pos, thumb_pos, color)
            else:
                player2_wrist = wrist_pos
                player2_thumb = thumb_pos
                if wrist_pos[0] < image.shape[1] // 2:
                    color = (255, 0, 0)  # Blue lightsaber for player 2 (left side)
                else:
                    color = (0, 0, 255)  # Red lightsaber for player 2 (right side)
                player2_lightsaber_end = draw_lightsaber(image, wrist_pos, thumb_pos, color)

            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손 위에 이미지 오버레이
            overlay_image_on_hand(image, overlay_image, wrist_pos)

    # 충돌 감지 및 체력 감소
    if player1_lightsaber_end and player2_wrist and player2_thumb:
        collision_distance = 50  # 충돌 범위 조정 가능
        distance_to_wrist = point_to_line_distance(player2_wrist, player1_wrist, player1_lightsaber_end)
        distance_to_thumb = point_to_line_distance(player2_thumb, player1_wrist, player1_lightsaber_end)
        # play_sound("hit_sound.mp3")  # 적절한 사운드 파일 이름으로 변경
        if distance_to_wrist < collision_distance or distance_to_thumb < collision_distance:
            player1_health -= 1

    if player2_lightsaber_end and player1_wrist and player1_thumb:
        collision_distance = 50  # 충돌 범위 조정 가능
        distance_to_wrist = point_to_line_distance(player1_wrist, player2_wrist, player2_lightsaber_end)
        distance_to_thumb = point_to_line_distance(player1_thumb, player2_wrist, player2_lightsaber_end)
        # play_sound("hit_sound.mp3")  # 적절한 사운드 파일 이름으로 변경
        if distance_to_wrist < collision_distance or distance_to_thumb < collision_distance:
            player2_health -= 1

    # 체력 상태 표시
    draw_health_bar(image, player1_health, max_health, (image.shape[1] - 310, 10), size=(300, 20), color=(0, 0, 255))
    draw_health_bar(image, player2_health, max_health, (10, 10), size=(300, 20), color=(255, 0, 0))

    # 중간에 "vs" 텍스트 표시
    vs_text = "VS"
    vs_text_size = cv2.getTextSize(vs_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    vs_text_x = (image.shape[1] - vs_text_size[0]) // 2
    vs_text_y = 30
    cv2.putText(image, vs_text, (vs_text_x, vs_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 체력이 0 이하가 되면 게임 오버
    if player1_health <= 0:
        game_over = True
        winner = "BLUE Wins!"
    elif player2_health <= 0:
        game_over = True
        winner = "RED Wins!"

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
