# Python backend: serves the /predict API and (after `npm run build`) the React app.
#
# Development:  flask --app main run --port 5001      (then `npm run dev` in frontend/)
# Port 5001 because macOS uses 5000 for AirPlay Receiver.
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / 'frontend' / 'dist'

# load model
clf = joblib.load(BASE_DIR / 'ALX_model_local.joblib')

# MediaPipe hand landmarker (Tasks API; the old mp.solutions API is gone in new versions)
hands = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(BASE_DIR / 'hand_landmarker.task')),
        num_hands=1,
    )
)

app = Flask(__name__, static_folder=None)


# Returns prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    image_file = request.files['image']

    # Convert to numpy array (OpenCV format)
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    results = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))

    if results.hand_landmarks:
        # Get first hand's landmarks
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        # Predict
        prediction = clf.predict([landmarks])[0]
        return jsonify({'letter': prediction})
    else:
        return jsonify({'letter': 'none'})


# Everything else is the built React app (pages are routed in the browser).
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def frontend(path):
    if not FRONTEND_DIST.is_dir():
        return 'Frontend not built. Run `npm run build` in frontend/, or use `npm run dev`.', 404
    if path and (FRONTEND_DIST / path).is_file():
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, 'index.html')
