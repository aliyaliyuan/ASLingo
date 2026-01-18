## ASLingo

ASL is one of the third most widely-used languages in the United States. However, many people from the hearing community do not know any ASL. ASLingo is a ML model for ASL education that I am developinging using Google's MediaPipe 
ML model. I am not fluent in ASL, but wanted to make a tool that makes ASL education more accessible (and I'm learning ASL as I go).  The ultimate goal is to have a model that can translate ASL based on live-video capture and 
can identify if the user is signing correctly or not. The feedback will give users more confidence and increase their chances of communicating with people actually fluent in ASL. 

For the purposes of this prototype, I want the model to be able to translate and give feedback on a user's ability to sign the alphabet,
100 common words in ASL, and 10 basic phrases. I recognize that ASL has more complex elements (facial expression, grammar, etc.), but I want to start simple. 

To start with, I am training the model on data I create (images of the ASL alphabet from Stock photos and myself) and simple word banks from Kaggle. Eventually, I want to train the model on a robust data set obtained of signs from fluent ASL speakers from the Deaf community for authenticity. 

## Current Progress
![Prototype](https://github.com/user-attachments/assets/b0132854-ce1a-4449-afdf-4192f9b7f58e)

I started with 3 letters (A, L, and X). For simplicity, I used a Random Forest Classifier on the three letters using images of me signing the letters and stock images of the letters. There are only about 20-30 images per letter, so overfitting is definitely occuring (which I will be fixing by creating more robust training data sets). 

Training files can be found at https://drive.google.com/drive/folders/1xhWV4fKXKLFgCnSbBuy9rk7h4h2a4FyP?usp=drive_link 

## How To Run It
First, run Train_mp_ALX.py. This will create a landmarks.csv, which extract features from the images in your training data. These features are then used to generate the predictive model. You will need to save the model. 

Then, you will edit the livecapture.py script to point to the file path location of your model and run the script. Your camera will pop up and text that reads "Letter: " will appear and predict the letter you are trying to sign. Right now, the accuracy is quite poor due to overfitting of the model I discussed earlier. It seems the model did really well at learning "L", but often confuses "A" and "X". 

Stay tuned for updates!

## Sample of Hand Landmarker output with Letter "A" in ASL 
![a](https://github.com/user-attachments/assets/355d829d-a087-4bfa-bbe4-8c9101d9a3ca)

This depicts the 21 landmarkers from MediaPipe that help to recognize hand gestures, which will help the model to learn how to detect signs. 

## Future Direction
I will be creating a robust training data set with 100 images per classification. I will also be expanding the model to train on all letters of the ASL alphabet. Then, I will move on to basic words. Then, I will need help from fluent ASL speakers/the Deaf community. 
