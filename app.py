import numpy as np
import joblib
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load("rainfall_prediction_model.pkl")

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['Temperature'], data['Humidity'], data['Wind Speed'], data['Pressure']]])
    prediction = model.predict(features)[0]
    return jsonify({"Predicted Rainfall": prediction})

if __name__ == '__main__':
    app.run(debug=True)
