import pygame
import sys
import cv2
import mediapipe as mp

# Basic settings
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Stickman Boxing with Pose Control")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Font settings
font = pygame.font.Font(None, 36)

# Stickman class definition
class Stickman:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.health = 100
        self.stance = "idle"  # idle, attack, defend
        self.direction = direction

    def draw(self, screen):
        head_radius = 60  # 3 times the original
        body_length = 180  # 3 times the original
        arm_length = 120  # 3 times the original

        # Draw head
        pygame.draw.circle(screen, BLACK, (self.x, self.y - head_radius - body_length), head_radius)
        # Draw body
        pygame.draw.line(screen, BLACK, (self.x, self.y - body_length), (self.x, self.y), 8)
        
        if self.stance == "idle":
            # Draw arms in idle position
            if self.direction == "left":
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x - arm_length // 2, self.y - body_length + 60), 8)
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x + arm_length // 2, self.y - body_length + 30), 8)
            else:
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x + arm_length // 2, self.y - body_length + 60), 8)
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x - arm_length // 2, self.y - body_length + 30), 8)
        elif self.stance == "attack":
            # Draw attacking arms
            if self.direction == "left":
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x - arm_length, self.y - body_length + 60), 8)
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x + arm_length // 2, self.y - body_length + 30), 8)
            else:
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x + arm_length, self.y - body_length + 60), 8)
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x - arm_length // 2, self.y - body_length + 30), 8)
        elif self.stance == "defend":
            pygame.draw.circle(screen, GREEN, (self.x, self.y - head_radius - body_length), head_radius)
            # Draw defending arms
            if self.direction == "left":
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x - arm_length // 2, self.y - body_length + 30), 8)
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x + arm_length // 2, self.y - body_length + 30), 8)
                pygame.draw.line(screen, BLACK, (self.x - arm_length // 2, self.y - body_length + 30), (self.x - arm_length // 2, self.y - body_length + 90), 8)
                pygame.draw.line(screen, BLACK, (self.x + arm_length // 2, self.y - body_length + 30), (self.x + arm_length // 2, self.y - body_length + 90), 8)
            else:
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x + arm_length // 2, self.y - body_length + 30), 8)
                pygame.draw.line(screen, BLACK, (self.x, self.y - body_length + 60), (self.x - arm_length // 2, self.y - body_length + 30), 8)
                pygame.draw.line(screen, BLACK, (self.x + arm_length // 2, self.y - body_length + 30), (self.x + arm_length // 2, self.y - body_length + 90), 8)
                pygame.draw.line(screen, BLACK, (self.x - arm_length // 2, self.y - body_length + 30), (self.x - arm_length // 2, self.y - body_length + 90), 8)

# Game loop
player1 = Stickman(200, 550, "right")
player2 = Stickman(600, 550, "left")
clock = pygame.time.Clock()

# Set up video capture using OpenCV
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_pose():
    ret, frame = cap.read()
    if not ret:
        return None, None

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        frame_h, frame_w, _ = frame.shape
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_x = int(left_wrist.x * frame_w)
        right_x = int(right_wrist.x * frame_w)
        nose_x = int(nose.x * frame_w)

        player1_action = "idle"
        player2_action = "idle"
        if nose_x < frame_w // 2:  # Left half for Player 1
            if left_x < nose_x - 50:
                player1_action = "attack"
            elif right_x < nose_x - 50:
                player1_action = "defend"
        else:  # Right half for Player 2
            if right_x > nose_x + 50:
                player2_action = "attack"
            elif left_x > nose_x + 50:
                player2_action = "defend"

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame, (player1_action, player2_action)
    return None, (None, None)

while True:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            sys.exit()

    frame, actions = detect_pose()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (400, 300))
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (200, 0))

    player1_action, player2_action = actions
    if player1_action:
        player1.stance = player1_action
    if player2_action:
        player2.stance = player2_action

    # Update health
    if player1.stance == "attack" and abs(player1.x - player2.x) < 300 and player2.stance != "defend":
        player2.health -= 10
    if player2.stance == "attack" and abs(player1.x - player2.x) < 300 and player1.stance != "defend":
        player1.health -= 10

    # Draw the players
    player1.draw(screen)
    player2.draw(screen)

    # Draw health
    health_text1 = font.render(f"Player 1 Health: {player1.health}", True, BLACK)
    health_text2 = font.render(f"Player 2 Health: {player2.health}", True, BLACK)
    screen.blit(health_text1, (50, 50))
    screen.blit(health_text2, (550, 50))

    pygame.display.flip()
    clock.tick(30)