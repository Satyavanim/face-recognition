import tkinter as tk
import cv2
import base64
import sqlite3
import ctypes
import os

class FaceRegistrationForm:
    def __init__(self, master):
        self.master = master
        self.master.title("Live Face Registration Form")
        self.username_label = tk.Label(self.master, text="Enter Username:")
        self.username_label.pack()
        self.username_entry = tk.Entry(self.master)
        self.username_entry.pack()
        self.camera_label = tk.Label(self.master)
        self.camera_label.pack()
        self.capture_button = tk.Button(self.master, text="Capture", command=self.capture_face)
        self.capture_button.pack()
        self.master.protocol("WM_DELETE_WINDOW", self.close)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.count=0

    def capture_face(self):
        username = self.username_entry.get()
        datasets = 'datasets' 
                 
                 
        # These are sub data sets of folder,
        # for my faces I've used my name you can
        # change the label here
        sub_data = 'satya'    
                 
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
        self.count = 0
        while self.count<2:
                    ret, frame = self.cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.3, 4)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            face = gray[y:y + h, x:x + w]
                            #face_resize = cv2.resize(face, (width, height))
                            filename='% s/% s'+username+'.jpg'
                            cv2.imwrite( filename % (path,self.count), face)
                        self.count += 1
        if self.count >= 2:
                self.save_to_database(username)
                self.master.destroy()


    

    def save_to_database(self, username):
        path=os.path.expanduser('~') + r"/Desktop/datasets/satya"
        filename=str(0)+''+username+'.jpg'
        image1=os.path.join(path,filename)
        filename=str(1)+''+username+'.jpg'
        image2=os.path.join(path,filename)
        conn = sqlite3.connect('faces.db')
        mycursor= conn.cursor()
        mycursor.execute('''create table if not exists facerecognition
        (username TEXT NOT NULL UNIQUE,
        imagestring1 TEXT NOT NULL UNIQUE,
        imagestring2 TEXT NOT NULL UNIQUE);''')
        #mycursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS username_index ON facerecognition (username)")


        with open(image1, "rb") as image1string,open(image2, "rb") as image2string:
            converted_string1 = base64.b64encode(image1string.read())
            imagestring1=converted_string1.decode('utf-8')
            #print(imagestring1)
            converted_string2 = base64.b64encode(image2string.read())
            imagestring2=converted_string2.decode('utf-8')
            #print(imagestring2)
            res=mycursor.execute("select * from facerecognition where username=? and imagestring1=?  ",(username,imagestring1))
            s=("Insert or ignore into facerecognition(username,imagestring1,imagestring2) values(?,?,?)")
            mycursor.execute(s,(username,imagestring1,imagestring2))
            ctypes.windll.user32.MessageBoxW(0,"Data successfully save","Success")
            conn.commit()
            conn.close()
    def close(self):
        self.cap.release()
        self.master.destroy()

root = tk.Tk()
app = FaceRegistrationForm(root)
root.mainloop()


