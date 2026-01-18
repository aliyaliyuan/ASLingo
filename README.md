## ASLingo

ASLingo is a ML model for ASL education that I am developinging using Google's MediaPipe 
ML model. I am not fluent in ASL, but wanted to make a tool that made ASL education more accessible. The ultimate goal is to have a model that can translate ASL based on live-video capture and 
can identify if the user is signing correctly or not. For the purposes of this prototype,
I want the model to be able to translate and give feedback on a user's ability to sign the alphabet,
100 common words in ASL, and 10 basic phrases. I recognize that ASL has more complex elements (facial expression, grammar, etc.), but I want to start simple. 

To start with, I am training the model on data I create (images of the ASL alphabet) and simple word banks from Kaggle. Eventually, I want to train the model on a robust data set obtained of signs from fluent ASL speakers from the Deaf community for authenticity. 

## Current Progress

Currently, I am training Google's MediaPipe on ASL data (images and videos of words, letters, phrases, etc.) 
The Hand Landmarker Model (hand_landmarker.py script) takes a .jpg image as input, detects the localization of 21 hand-knuckle coordinates, 
and annotates them. It uses bounding boxes to improve efficiency of this object detection model. 

## Sample of Hand Landmarker output with Letter "A" in ASL 
![a](https://github.com/user-attachments/assets/355d829d-a087-4bfa-bbe4-8c9101d9a3ca)

