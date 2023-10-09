from flask import Flask, request, jsonify
import pickle
import numpy as np
import random

model = pickle.load(open('model1.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    #이 부분에서 뇌파 데이터 받아오기
    data_list = []  # 빈 리스트 생성

    data_list = request.get_json()['data_list']

    '''
    for _ in range(100):
        data = request.form.get('data')  # 데이터 수집
        data_list.append(data)  # 리스트에 데이터 추가
    '''


    #결과
    input_query = np.array([data_list], dtype=np.float32)

    result = model.predict(input_query)[0]

    return jsonify({'placement':str(result)})

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, port=8080)