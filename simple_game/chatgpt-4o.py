import pygame
import sys

# 기본 설정
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Stickman Boxing")

# 색상 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# 기본 폰트 설정
font = pygame.font.Font(None, 36)

# 스틱맨 클래스 정의
class Stickman:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.health = 100
        self.stance = "idle"  # idle, attack, defend
        self.direction = direction

    def draw(self, screen):
        head_radius = 60  # 3배로 키움
        body_length = 180  # 3배로 키움
        arm_length = 120  # 3배로 키움

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

# 게임 루프
player1 = Stickman(200, 550, "right")
player2 = Stickman(600, 550, "left")

clock = pygame.time.Clock()

while True:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    keys = pygame.key.get_pressed()

    # 플레이어1 조작
    if keys[pygame.K_a]:
        player1.stance = "attack"
        if abs(player1.x - player2.x) < 300 and player2.stance != "defend":
            player2.health -= 10
    elif keys[pygame.K_s]:
        player1.stance = "defend"
    else:
        player1.stance = "idle"

    # 플레이어2 조작
    if keys[pygame.K_LEFT]:
        player2.stance = "attack"
        if abs(player1.x - player2.x) < 300 and player1.stance != "defend":
            player1.health -= 10
    elif keys[pygame.K_DOWN]:
        player2.stance = "defend"
    else:
        player2.stance = "idle"
    
    # 화면을 반으로 나누기
    pygame.draw.line(screen, BLACK, (400, 0), (400, 600), 5)
    
    # 플레이어 그리기
    player1.draw(screen)
    player2.draw(screen)

    # 체력 표시
    health_text1 = font.render(f"Player 1 Health: {player1.health}", True, BLACK)
    health_text2 = font.render(f"Player 2 Health: {player2.health}", True, BLACK)
    screen.blit(health_text1, (50, 50))
    screen.blit(health_text2, (550, 50))
    
    pygame.display.flip()
    clock.tick(30)
