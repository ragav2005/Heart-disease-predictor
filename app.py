from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from predictor import predictor
import os

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data.get('input_data')
    prediction = predictor(input_data)
    res = {
        'prediction': int(prediction[0]),
        'Accuracy': "85.5%" 
    }
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)
