from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
# from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
# load_dotenv()

app = Flask(__name__)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """채팅 메시지 처리"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': '메시지가 비어있습니다.'}), 400
        
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        # 응답 추출
        bot_message = response.choices[0].message.content
        
        return jsonify({
            'response': bot_message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
