from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from io import BytesIO

## git merge test

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image file found', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400

    # 이미지를 읽어서 OpenCV 형식으로 변환
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # 컬러 이미지로 변환

    # 외곽선 검출을 위해 Canny Edge Detection 적용
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 흰 배경에 검은 외곽선으로 변환
    result = np.zeros_like(image)
    result[:] = [255, 255, 255]     # 흰색 배경 만들기
    result[edges != 0] = [0, 0, 0]  # 검은색 외곽 보여주기

    # 클라이언트로부터 받은 색상 정보
    color = request.form.get('color')
    if color:
        try:
            color = tuple(map(int, color.split(',')))  # '255,0,0' 형식을 (255, 0, 0) 튜플로 변환
            # 외곽선 내부를 클라이언트가 선택한 색상으로 채우기
            result[edges != 0] = color
        except ValueError:
            pass

    # 결과 이미지를 메모리 버퍼에 저장
    _, buffer = cv2.imencode('.png', result)
    io_buf = BytesIO(buffer)

    return send_file(io_buf, mimetype='image/png')

@app.route('/upload-gray', methods=['POST'])
def upload_gray_image():
    if 'image' not in request.files:
        return 'No image file found', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400

    # 이미지를 읽어서 OpenCV 형식으로 변환
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # 이미지를 그레이스케일로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이미지 색상 단순화
    color_levels = 5  # 사용할 색상 레벨 수
    step = 256 // color_levels
    image = (image // step) * step

    # 색상 포화도 높이기
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * 1.5, 255)
    # image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # 처리된 이미지를 PNG 형식으로 인코딩하여 클라이언트에게 반환
    _, buffer = cv2.imencode('.png', image)
    io_buf = BytesIO(buffer)

    # 두 번째 이미지 반환
    return send_file(io_buf, mimetype='image/png')

@app.route('/upload-inverted', methods=['POST'])
def upload_inverted_image():
    if 'image' not in request.files:
        return 'No image file found', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400

    # 이미지를 읽어서 OpenCV 형식으로 변환
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # 이미지 색상 단순화
    color_levels = 4  # 사용할 색상 레벨 수
    step = 256 // color_levels
    image = (image // step) * step

    # 색상 포화도 높이기
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * 1.5, 255)
    # image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # 처리된 이미지를 PNG 형식으로 인코딩하여 클라이언트에게 반환
    _, buffer = cv2.imencode('.png', image)
    io_buf = BytesIO(buffer)

    # 두 번째 이미지 반환
    return send_file(io_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
