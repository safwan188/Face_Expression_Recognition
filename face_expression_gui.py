import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

class Application:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Load model and Haar cascade
        self.model = load_model('max_conv_model1.h5')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize video capture
        self.vid = cv2.VideoCapture(video_source)

        # Create a canvas for video
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to quit
        self.btn_quit = tk.Button(window, text="Quit", width=10, command=self.window.destroy)
        self.btn_quit.pack()

        # Update periodically
        self.update()

        self.window.mainloop()

    def update(self):
        isTrue, frame = self.vid.read()
        if isTrue:
            # Image processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float')/255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=3)

                preds = self.model.predict(roi_gray)
                label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][np.argmax(preds)]
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Convert frame to PhotoImage
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

# Create and run the application
if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root, "Face Expression Recognition")
