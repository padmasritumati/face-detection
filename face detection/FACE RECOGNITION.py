

import cv2
# Import numpy for matrices calculations
import numpy as np
import os
def assure_path_exists(path):
dir = os.path.dirname(path)
if not os.path.exists(dir):
os.makedirs(dir)
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists(&quot;trainer/&quot;)
# Load the trained mode
recognizer.read(&#39;trainer/trainer.yml&#39;)
# Load prebuilt model for Frontal Face
cascadePath = &quot;haarcascade_frontalface_default.xml&quot;
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);
# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX
# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)
# Loop
while True:
# Read the video frame
ret, im =cam.read()
# Convert the captured frame into grayscale
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# Get all face from the video frame
faces = faceCascade.detectMultiScale(gray, 1.2,5)
# For each face in faces
for(x,y,w,h) in faces:
# Create rectangle around the face
cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
# Recognize the face belongs to which ID
id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
# Check the ID if exist
#if(confidence&lt;=50):
if(id == 1):
id = &quot;Saketh&quot; #give your name
elif(id == 2):
id = &quot;Unknown&quot; # give others
# Put text describe who is in the picture
cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
cv2.putText(im, str(id), (x,y-40), font, 1, (255,255,255), 3)
# Display the video frame with the bounded rectangle
cv2.imshow(&#39;im&#39;,im)
# If &#39;q&#39; is pressed, close program
if cv2.waitKey(10) &amp; 0xFF == ord(&#39;q&#39;):
break
# Stop the camera
cam.release()
# Close all windows
cv2.destroyAllWindows()