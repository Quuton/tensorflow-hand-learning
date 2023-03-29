from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
from tensorflow.keras.preprocessing import image as tf_img
from tensorflow.keras.optimizers import  RMSprop as rms
from tensorflow.keras import layers as tf_layer
import tensorflow as tf
import matplotlib.pyplot as mplot
import numpy as np
import cv2 as cv
import os

target_x = 150
target_y = 150

training_dataset_batch_size = 128
validation_dataset_batch_size = 128


# Rescale of image rgb values to fit 0->1 range
training = idg(rescale = 1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)

validation = idg(rescale = 1/255)

# Dataset generation
training_dataset = training.flow_from_directory('Images/Data/Training', target_size = (target_x, target_y), batch_size = training_dataset_batch_size, class_mode = 'categorical')
validation_dataset = validation.flow_from_directory('Images/Data/Validation', target_size = (target_x, target_y), batch_size = validation_dataset_batch_size, class_mode = 'categorical')




# # Configuration of convolutional model
model = tf.keras.models.Sequential()

model.add(tf_layer.Conv2D(64, (3,3), activation='relu', input_shape=(target_x, target_y,3)))
model.add(tf_layer.MaxPooling2D(2,2))

model.add(tf_layer.Conv2D(64, (3,3), activation='relu'))
model.add(tf_layer.MaxPooling2D(2,2))

model.add(tf_layer.Conv2D(128, (3,3), activation = 'relu'))
model.add(tf_layer.MaxPooling2D(2,2))

model.add(tf_layer.Conv2D(128, (3,3), activation = 'relu'))
model.add(tf_layer.MaxPooling2D(2,2))

model.add(tf_layer.Flatten())

model.add(tf_layer.Dropout(0.5))

model.add(tf_layer.Dense(512, activation='relu'))

model.add(tf_layer.Dense(3, activation='softmax'))




# Model compilatioon
model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=['categorical_accuracy'])

model_fit = model.fit(training_dataset, validation_data=validation_dataset, epochs=10)

model.save(os.path.join('models','RPS_Classifer_v1.h5'))

model.summary()
