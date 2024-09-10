from flask import Flask, jsonify, render_template
import openai
import speech_recognition as sr
import subprocess
import os 

app = Flask(__name__)

# OpenAI API 키를 환경 변수에서 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

# 음성 인식 함수
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    try:
        recognized_text = r.recognize_google(audio)
        print("Recognized: " + recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return "Could not request results from Google Speech Recognition service; {0}".format(e)

# ChatGPT 응답 생성 함수
def get_chatgpt_response(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error during ChatGPT response generation: {str(e)}"

# Flask 라우트
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # 1. 음성 인식 실행
    recognized_text = recognize_speech()
    
    # 2. 음성 인식 결과 먼저 전송
    if recognized_text.startswith("Sorry") or "Error" in recognized_text:
        return jsonify({"recognized": recognized_text, "response": "No response from ChatGPT due to recognition error."})

    # 3. ChatGPT 응답 생성
    chatgpt_response = get_chatgpt_response(recognized_text)
    
    # 4. 인식된 텍스트와 ChatGPT 응답을 함께 반환
    return jsonify({"recognized": recognized_text, "response": chatgpt_response})

if __name__ == '__main__':
    app.run(debug=True)
