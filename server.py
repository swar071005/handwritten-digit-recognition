from flask import Flask, request, jsonify
import numpy as np
import cv2
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (8, 8))
    img = img / 16.0
    img = img.reshape(1, -1)

    prediction = model.predict(img)
    return jsonify({"digit": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
