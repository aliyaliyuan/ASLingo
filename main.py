#Python backend
from flask import Flask
from flask import send_file
from flask import render_template
from flask import request, jsonify
#from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np
import mediapipe as mp
import joblib

#load model
clf = joblib.load('/Users/aliyahaas/Desktop/VS_Code/ASLingo/ALX_model_local.joblib')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

#Run with the line: flask --app main run
app = Flask(__name__)

@app.route('/')
def login():
    return render_template('login.html')

#Camera
@app.route('/app')
def app_page():
    return render_template('front_end.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

#Returns prediction
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
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Get first hand's landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Predict
        prediction = clf.predict([landmarks])[0]
        return jsonify({'letter': prediction})
    else:
        return jsonify({'letter': 'none'})

@app.route('/flashcards')
def flashcards():
    return render_template('flashcards.html')


