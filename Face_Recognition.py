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

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier(r"C:\Users\Username\Desktop/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open(r'C:\Users\Username\Desktop/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights(r'C:\Users\Username\Desktop/antispoofing_model.h5')
print("Model loaded from disk")
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
start_time = time.time()
def send_email(frame):
    # Create the email message
    sender_email = 'sender12@gmail.com'
    sender_password = ''  #give your App Password ,don't give your google Account password
    Receiver_email = 'Receiver@gmail.com'
    subject="No face detected"
    body = """
    there is no faces infront of camera
    """
                
    print("Sending email......")
    em = EmailMessage()            
    em['From'] = sender_email
    em['To'] = Receiver_email
    em['subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    imge = Image.fromarray(frame)
    imge.save('detected_face.jpg')
            
    # attach the saved image to the email message
    with open('detected_face.jpg', 'rb') as f:
        img_data = f.read()
    em.add_attachment(img_data, maintype='image', subtype='jpg', filename='detected_face.jpg')
    # Send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465,context=context) as smtp:
        smtp.login(sender_email,sender_password)
        smtp.sendmail(sender_email,Receiver_email, em.as_string())
        print("mail sent successfully")


video = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        if len(faces) == 0:
            if time.time() - start_time > 30:
                send_email(frame)
                start_time = time.time()
                print("No faces detected")
                
        else:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
                face_img_gray= gray[y:y+h, x:x+w]
                face_image = frame[y:y+h,x:x+w]
                laplacian_var = cv2.Laplacian(face_img_gray, cv2.CV_64F).var()

                if preds> 0.5 :
                    label = 'fake'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 0, 255), 2)
                    if elapsed_time >= 30:
                        send_email(frame)
                        start_time = time.time()
                        print("fake image detected")
                else:
                    label = 'real'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 255, 0), 2)
                    if elapsed_time >= 30 and laplacian_var > 140:
                        filename = f"face_detection_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                        cv2.imwrite(filename,face_image )
                        start_time = time.time()
                        print("succussfully image captured")
                    
        #cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass
video.release()        
cv2.destroyAllWindows()

