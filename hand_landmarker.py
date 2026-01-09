#Ensure mediapipe is installed in Terminal 
#pip install -q mediapipe

#install the model bundle from github (now located in the directory as .task file)

#Visualization utilities 

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

MARGIN = 10 #pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) #Vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    annotated_image = cv2.cvtColor(np.copy(rgb_image), cv2.COLOR_RGB2BGR)
    #Loop through the detected hands to visualize
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        #Draw the hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z =landmark.z) for landmark in hand_landmarks 
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
        
        #Get the top left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        #Draw handedness (left or right hand) on the image
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA
                    )
    return annotated_image

#Running inference and visualizing the results

#STEP 1: Import necessary modules

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Step 2: Generate a HandLandmarker object
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

#Step 3: Load input image
image = mp.Image.create_from_file('/path/to/img')

#Step 4: Detect hand landmarks from the input image
detection_results = detector.detect(image)

#Step 5: Process the classification results. (Visualize in this case)
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_results)
cv2.imshow('Hand Landmarks', annotated_image)
cv2.waitKey(0)
cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
