
![logo](https://github.com/user-attachments/assets/85b3dc07-c039-405c-9dda-17481a657ed5)


ASL is the third most taught language in the US, however, many people from the hearing community do not know any ASL. ASLingo is a ML model for ASL education that I am developinging using Google's MediaPipe 
ML model. I am not fluent in ASL, but wanted to make a tool that makes ASL education more accessible (and I'm learning ASL as I go). I am developing a model that can translate ASL based on live-video capture and 
can identify if the user is signing correctly or not. This tool is not meant to replace instructors or interpretors or interactions with real people, but rather to instill confidence and provide ample opportunities to practice. 
With confidence, users will be more likely to interact with other ASL speakers. 

For the purposes of this prototype, I want the model to be able to translate and give feedback on a user's ability to sign the alphabet,
100 common words in ASL, and 10 basic phrases. I recognize that ASL has more complex elements (facial expression, grammar, etc.), but I want to start simple. 

To start with, I am training the model on data I create (images of the ASL alphabet from Stock photos and myself) and simple word banks from Kaggle. Eventually, I want to train the model on a robust data set obtained of signs from fluent ASL speakers from the Deaf community for authenticity. 

## Current Progress
Log-in page: 

![loginpage](https://github.com/user-attachments/assets/d1108fad-0c09-4990-8440-daac25565dfc)

Menu:

![menu](https://github.com/user-attachments/assets/d91cb6ba-575f-4b7e-861e-5647b1d9cdce)

Flashcards: 

![ASLingo_flashcardfeature](https://github.com/user-attachments/assets/4d709e8b-ba1c-478e-ab7c-b7b33cfcbaba)

Live Capture Letter Detection Page: 

![predictpage](https://github.com/user-attachments/assets/20bfeea2-54cd-49ea-8002-d1e5f3023cfa)

Game:

https://github.com/user-attachments/assets/4411e29c-dd76-4ee8-8b5f-da5a39fe176b
Controls: up, down, left, right arrows

## The Prediction Model
I started with 3 letters (A, L, and X). For simplicity, I used a Random Forest Classifier on the three letters using images of me signing the letters and stock images of the letters. There are only about 20-30 images per letter, so overfitting is definitely occuring (which I will be fixing by creating more robust training data sets). 

Training files can be found at https://drive.google.com/drive/folders/1xhWV4fKXKLFgCnSbBuy9rk7h4h2a4FyP?usp=drive_link 

A basic GUI that captures the live video feed from your device and displays the letter being recognized (A, L, X, or None). hello.py is the back-end python script that imports flask and the trained model to create end points for the HTML file to connect with. front_end.html captures the live video, sends frames to hello.py, which uses the model to make the predictions, then sends the information back. This is the simplest GUI it will be. Eventually, I want to incorporate users, lessons, and games. 

The flashcard section is the first feature to be developed. Once users are set up, I will have a word/phrase/grammar bank that tracks what the user has learned so far as well as mistakes. This will inform what the user will be given in future lessons and games. 

## How To Train the Model
First, run Train_mp_ALX.py. This will create a landmarks.csv, which extract features from the images in your training data. These features are then used to generate the predictive model. You will need to save the model. 

Then, you will edit the livecapture.py script to point to the file path location of your model and run the script. Your camera will pop up and text that reads "Letter: " will appear and predict the letter you are trying to sign. 

The "L" is correctly identified all of the time, "A" is correctly identified around 60% of the time, but "X" only works on your left hand and if your hand is to the side. I will need to train on more data to prevent overfitting. Also to expand it to the rest of the alphabet (and eventually words and phrases). 

## How to Run the GUI

Go to the directory that main.py is located in. Run: "flask --app main run" in the Terminal. For this to work, the front_end.html is located in the /templates directory, and the logo is in the /static direcotry. Both /static and /templates must be in the same directory as main.py. 

Stay tuned for updates!

## Sample of Hand Landmarker output with Letter "L" in ASL 
![Prototype](https://github.com/user-attachments/assets/b0132854-ce1a-4449-afdf-4192f9b7f58e)

This depicts the 21 landmarkers from MediaPipe that help to recognize hand gestures, which will help the model to learn how to detect signs. 

## Future Direction
I will be creating a robust training data set with 100 images per classification. I will also be expanding the model to train on all letters of the ASL alphabet. Then, I will move on to basic words. Then, I will need help from fluent ASL speakers/the Deaf community. 

For the game, I want it to be a mini grocery shopping game. The user will be given a list (i.e. apple, carrot, and cake). The user (the circle on the screen) will navigate to the sections of the grocery store (will be just plain rectangles in the early prototype) and once the user clicks on the section, a screen will pop up with options. For instance, at the fruit station, the user will be shown four images -- an apple, an orange, a banana, and a strawberry. Another window also pops up (the live detection video that can predict the letter being signed). The user must sign the option they want. If they sign it correctly, it is added to their basket. The round is won when all three of the items are correctly signed (and thus added to the basket). 

I will also eventually add users. 
