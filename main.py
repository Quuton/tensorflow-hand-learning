
import kivy
import numpy as np
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import tensorflow as tf
kivy.require('1.9.0')

target_x = 150
target_y = 150

model = tf.keras.models.load_model('Models/RPS_Classifier_v1.h5')

class MainActivity(App):
    def build(self):
        return RootLayout()

class RootLayout(BoxLayout):
    def __init__(self):
        super(RootLayout, self).__init__()

    def capture(self):
        camera = self.ids['camera1']
        pictureData = cv2.cvtColor(np.frombuffer(camera.texture.pixels, np.uint8).reshape(camera.texture.height, camera.texture.width, 4), cv2.COLOR_BGR2RGB)
        resizedImage = cv2.resize(pictureData, [target_x, target_y])
        probabilities = model.predict(np.expand_dims(resizedImage/255, 0))
        prob = str(round(max(probabilities[0]) *100)) + '%'
        if (np.argmax(probabilities) == 0):
            self.label1.text = "Thats a paper\nI am {prob} sure".format(prob = prob)
        elif (np.argmax(probabilities) == 1):
            self.label1.text = "Thats a rock\nI am {prob} sure".format(prob = prob)
        else:
            self.label1.text = "Those are scissors\nI am {prob} sure".format(prob = prob)
        return


if __name__ == "__main__":
    mainActivity = MainActivity()
    mainActivity.run()