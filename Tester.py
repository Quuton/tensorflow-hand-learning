from tensorflow.keras.models import load_model as model_loader
from tensorflow.image import resize
import numpy as np
import cv2 as cv
import os
from PIL import ImageTk as imagetk
from PIL import Image as p_image
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

target_x = 150
target_y = 150

image_path = None
model = model_loader('Models/RPS_Classifier_v1.h5')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Rock Paper Scissors Classifier')
        self.geometry('600x300')
        self.label1 = ttk.Label(self, text='No image selected')
        self.label1.grid(column=0, row=0, padx=10, pady=10)

        self.main_image = imagetk.PhotoImage(p_image.open("Images/Resources/Error.png").resize((200,200), p_image.Resampling.LANCZOS))
        self.label2 = ttk.Label(self,image = self.main_image)
        self.label2.grid(column=1, row=0, padx=10, pady=10)

        self.frame1 = ttk.Frame(master=self)
        self.frame1.grid(column=2,row=0)

        self.button1 = ttk.Button(self.frame1, text="Change image", command=self.choose_image)
        self.button1.grid(column= 0, row = 0, padx=10, pady=10)

        self.button2 = ttk.Button(self.frame1, text="Predict", command=self.predict_class)
        self.button2.grid(column= 0, row = 1, padx=10, pady=10)

        self.label3 = ttk.Label(self.frame1, text='No image given yet')
        self.label3.grid(column=0, row=3, padx=10, pady=10)
    
    def choose_image(self):
        global image_path
        image_path = filedialog.askopenfilename(initialdir = "Images/Data/Tests",title = "Select a File",filetypes = (("all files","*.*"),("PNG Files","*.png*")))
        temp_image = p_image.open(image_path)
        temp_image.thumbnail((200,200), p_image.Resampling.LANCZOS)
        self.main_image = imagetk.PhotoImage(temp_image)
        self.label2.configure(image = self.main_image)
        self.label1.configure(text=os.path.basename(image_path))

    def predict_class(self):
        global image_path
        print(image_path)
        if image_path == None:
            return

        temp_img = cv.imread(image_path)
        resized_image = resize(temp_img, (target_x, target_y))

        probabilities = model.predict(np.expand_dims(resized_image/255, 0))

        prob = str(round(max(probabilities[0]) *100)) + '%'

        if (np.argmax(probabilities) == 0):
            self.label3.configure(text = "Thats a paper\nI am {prob} sure".format(prob = prob))
        elif (np.argmax(probabilities) == 1):
            self.label3.configure(text = "Thats a rock\nI am {prob} sure".format(prob = prob))
        else:    
            self.label3.configure(text = "Those are scissors\nI am {prob} sure".format(prob = prob))

        

if __name__ == "__main__":
    app = App()
    app.mainloop()