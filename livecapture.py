import mediapipe as mp
import numpy as np
import cv2
import joblib

# Load model
clf = joblib.load('/Users/aliyahaas/Desktop/VS_Code/ASLingo/ALX_model_local.joblib')

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

print("Camera opened. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates for prediction
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Predict the letter
                prediction = clf.predict([landmarks])
                
                # Display prediction on frame
                cv2.putText(frame, f"Letter: {prediction[0]}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (0, 255, 0), 3)
        
        cv2.imshow("ASL Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()