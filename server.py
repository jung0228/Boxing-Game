import cv2
import numpy as np
import pickle
import socket
import threading

# 서버 정보 설정
SERVER = "0.0.0.0"
PORT = 5555
ADDR = (SERVER, PORT)

# 소켓 초기화
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    while True:
        # 클라이언트로부터 데이터 수신
        data = conn.recv(4096)
        if not data:
            break
        
        # 데이터 처리
        landmarks = pickle.loads(data)
        # 여기에 데이터 처리 작업을 추가하세요

        # 클라이언트로부터 받은 포즈를 다른 클라이언트에게 다시 전송
        for client_socket in clients:
            if client_socket != conn:
                client_socket.sendall(pickle.dumps(landmarks))

    conn.close()

def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")

        # 새로운 클라이언트가 연결되면 해당 클라이언트를 clients 리스트에 추가
        clients.append(conn)

# 클라이언트 소켓을 담을 리스트 초기화
clients = []

print("[STARTING] Server is starting...")
start()
