import tensorflow as tf
from tensorflow.keras.models import load_model as model_loader
# Convert the model
model = model_loader('Models/RPS_Classifier_v1.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('RPS_Classifier_v1.tflite', 'wb') as f:
    f.write(tflite_model)