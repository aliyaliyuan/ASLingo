
![logo](https://github.com/user-attachments/assets/85b3dc07-c039-405c-9dda-17481a657ed5)


ASL is the third most widely-used language in the United States. However, many people from the hearing community do not know any ASL. ASLingo is a ML model for ASL education that I am developinging using Google's MediaPipe 
ML model. I am not fluent in ASL, but wanted to make a tool that makes ASL education more accessible (and I'm learning ASL as I go).  The ultimate goal is to have a model that can translate ASL based on live-video capture and 
can identify if the user is signing correctly or not. The feedback will give users more confidence and increase their chances of communicating with people actually fluent in ASL. 

For the purposes of this prototype, I want the model to be able to translate and give feedback on a user's ability to sign the alphabet,
100 common words in ASL, and 10 basic phrases. I recognize that ASL has more complex elements (facial expression, grammar, etc.), but I want to start simple. 

To start with, I am training the model on data I create (images of the ASL alphabet from Stock photos and myself) and simple word banks from Kaggle. Eventually, I want to train the model on a robust data set obtained of signs from fluent ASL speakers from the Deaf community for authenticity. 

## Current Progress
![GUI_1](https://github.com/user-attachments/assets/a9a559e2-7ea2-41bd-842d-26c3554506e2)

I started with 3 letters (A, L, and X). For simplicity, I used a Random Forest Classifier on the three letters using images of me signing the letters and stock images of the letters. There are only about 20-30 images per letter, so overfitting is definitely occuring (which I will be fixing by creating more robust training data sets). 

Training files can be found at https://drive.google.com/drive/folders/1xhWV4fKXKLFgCnSbBuy9rk7h4h2a4FyP?usp=drive_link 

A basic GUI that captures the live video feed from your device and displays the letter being recognized (A, L, X, or None). hello.py is the back-end python script that imports flask and the trained model to create end points for the HTML file to connect with. front_end.html captures the live video, sends frames to hello.py, which uses the model to make the predictions, then sends the information back. This is the simplest GUI it will be. Eventually, I want to incorporate users, lessons, and games. 

## How To Train the Model
First, run Train_mp_ALX.py. This will create a landmarks.csv, which extract features from the images in your training data. These features are then used to generate the predictive model. You will need to save the model. 

Then, you will edit the livecapture.py script to point to the file path location of your model and run the script. Your camera will pop up and text that reads "Letter: " will appear and predict the letter you are trying to sign. Right now, the accuracy is quite poor due to overfitting of the model I discussed earlier. It seems the model did really well at learning "L", but often confuses "A" and "X". 

## How to Run the GUI

Go to the directory that hello.py is located in. Run: "flask --app hello run" in the Terminal. For this to work, the front_end.html is located in the /templates directory, and the logo is in the /static direcotry. Both /static and /templates must be in the same directory as hello.py. 

Stay tuned for updated!

## Sample of Hand Landmarker output with Letter "L" in ASL 
![Prototype](https://github.com/user-attachments/assets/b0132854-ce1a-4449-afdf-4192f9b7f58e)

This depicts the 21 landmarkers from MediaPipe that help to recognize hand gestures, which will help the model to learn how to detect signs. 

## Future Direction
I will be creating a robust training data set with 100 images per classification. I will also be expanding the model to train on all letters of the ASL alphabet. Then, I will move on to basic words. Then, I will need help from fluent ASL speakers/the Deaf community. 

I will be modifying the scripts so that the Python script is the back-end and a React script will be used for the front end. The front-end will consist of a window that includes the live camera and a box that shows the letter (and eventually word/translated sentence) being identified. 
