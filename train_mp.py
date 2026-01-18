

#Getting sample data from you dataset
import matplotlib.pyplot as plt
import os
NUM_EXAMPLES = 5
IMAGES_PATH = "/Users/aliyahaas/Desktop/VS_Code/ASLingo/ALX_Train"

#Get list of labels from list of folder names
labels = []
for i in os.listdir(IMAGES_PATH):
    if os.path.isdir(os.path.join(IMAGES_PATH,i)):
        labels.append(i)

print("Labels:")
for i in labels:
    print(i)

#Making a New Model
#pip install -q mediapipe-model-maker

#Import modules
from mediapipe_model_maker import gesture_recognizer

#Load training image archive
data = gesture_recognizer.Dataset.from_folder(
    dirname=IMAGES_PATH,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)

#Split the archiveinto training, validation and test dataset
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

#Train the model
hparams = gesture_recognizer.HParams(export_dir="/Users/aliyahaas/Desktop/VS_Code/ASLingo/asl_model0")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss: {loss}, Test accuracy: {acc}")

#Export the model bundle
model.export_model()

#Rename in terminal
#mv asl_model0/gesture_recognizer.task ALX.task

#Download ALX.task

#import cv2
#img = cv2.imread("photo.jpg")

#Show image





