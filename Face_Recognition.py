import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import smtplib
import ssl
from PIL import Image
from email.message import EmailMessage
import time
import io
import sqlite3
import base64
from datetime import datetime
from datetime import date
import face_recognition


images = []
empimg=[]
classnames=[]
mydb=sqlite3.connect(os.path.expanduser('~') + r"/Desktop/faces.db")
mycursor=mydb.cursor()
mycursor.execute("select username,imagestring1,imagestring2 from facerecognition")
a=mycursor.fetchall()
#print(a)
for i in range(len(a)):
    classnames.append(a[i][0])
    empimg.append(a[i][1])
    empimg.append(a[i][2])
    b=base64.b64decode(a[i][1])
    im=Image.open(io.BytesIO(b))
    images.append(im)
    b=base64.b64decode(a[i][2])
    im=Image.open(io.BytesIO(b))
    images.append(im)
mydb.commit()
mydb.close()
print(images)
#Function for find the encoded data of the input image
def findEncodings(images):
        encodeList = []

   
        for img in images:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList


def markAttendance(name):   #save the name , current date and time in the text file
        with open('Attendance.txt', 'a+') as f:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    today=date.today()  # to display current date
                    day = today.strftime("%B %d, %Y")
                    f.writelines(f'\n{name},{dtString},{day}')

#find encodings of training images
encodeListKnown = findEncodings(images)
#print('Encoding Complete')

face_cascade = cv2.CascadeClassifier(os.path.expanduser('~') + r"/Desktop/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open(os.path.expanduser('~') + r'/Desktop/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights(os.path.expanduser('~') + r'/Desktop/antispoofing_model.h5')
print("Model loaded from disk")
start_time = time.time()
def send_email(frame,subject,body):
    # Create the email message
    sender_email = 'sender@gmail.com'
    sender_password = '' #give your App Password ,don't give your google Account password
    Receiver_email = 'receiver@gmail.com'
                
    print("Sending email......")
    em = EmailMessage()            
    em['From'] = sender_email
    em['To'] = Receiver_email
    em['subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    filename = f"detected_face.jpg"
    cv2.imwrite(filename,frame )
            
    # attach the saved image to the email message
    with open('detected_face.jpg', 'rb') as f:
        img_data = f.read()
    em.add_attachment(img_data, maintype='image', subtype='jpg', filename='detected_face.jpg')
    # Send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465,context=context) as smtp:
        smtp.login(sender_email,sender_password)
        smtp.sendmail(sender_email,Receiver_email, em.as_string())
        print("mail sent successfully")


# Initialize video capture device
cap = cv2.VideoCapture(0)

# Loop through video frames
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Convert the frame to RGB color space
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_frame,1.3,5)
    if len(faces)==0 and time.time() - start_time > 30:
                subject="faces not detected"
                body = """
                there is no employee infront of camera
                """
                send_email(frame,subject,body)
                start_time = time.time()
                print("No faces detected")

    else:
        for (x,y,w,h) in faces:
                face = frame[y-5:y+h+5,x-5:x+w+5]
                resized_face = cv2.resize(face,(160,160))
                resized_face = resized_face.astype("float") / 255.0
                # resized_face = img_to_array(resized_face)
                resized_face = np.expand_dims(resized_face, axis=0)
                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(resized_face)[0]
                #print(preds)
                elapsed_time = time.time() - start_time
                face_img_gray= rgb_frame[y:y+h, x:x+w]
                face_image = frame[y:y+h,x:x+w]
                laplacian_var = cv2.Laplacian(face_img_gray, cv2.CV_64F).var()

                if preds> 0.5 :
                    label = 'fake'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 0, 255), 2)
                    if elapsed_time >= 30:
                        subject="fake face detected"
                        body = """
                        Employee using the fake images like phone images ect
                        """
                        send_email(frame,subject,body
                                   )
                        start_time = time.time()
                        print("fake face detected")
                else:
    
                    # Detect faces in the frame
                    face_locations = face_recognition.face_locations(rgb_frame)
                    
                    # Encode faces in the frame
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # Loop through each face in the frame
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        # Compare the face with known faces in the call database
                        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
                        
                        # Find the index of the matched face
                        if True in matches:
                            match_index = matches.index(True) 
                        
                            # Draw a rectangle around the face in the frame
                            top, right, bottom, left = face_location
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Label the face in the frame with the matched name
                            name = classnames[match_index]
                            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if elapsed_time >= 30 and laplacian_var > 140:
                                filename = f"face_detection_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                                cv2.imwrite(filename,face_image )
                                markAttendance(name)
                                start_time = time.time()
                                print("succussfully image captured")
                        else:
                            name='unknown'
                            top, right, bottom, left = face_location
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            if elapsed_time >= 15:
                                subject="Unknown face detected"
                                body = """
                                Unknown person is sit in front of camera
                                """
                                send_email(frame,subject,body)
                                start_time = time.time()
                                print("Unknown face detected")
                            
    
    # Show the frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy all windows
cap.release()
cv2.destroyAllWindows()

