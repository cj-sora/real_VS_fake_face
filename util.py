import cv2
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import load_model
import numpy as np

model = keras.models.load_model('realfakemodel3.h5')
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])

# Process image (input) for the model
def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(img, (256,256))   
    resize = resize.numpy().astype(int)
    np.expand_dims(resize, 0).shape
    return resize

def get_prediction(img):
    pred = process(img)
    yhat = model.predict(np.expand_dims(pred/255, 0))
    if yhat > 0.9:
        pred = "real"
    else:
        pred = "fake"
    return pred

# my_img = cv2.imread("a (1).jpg")
# result = get_prediction(my_img)
# print(result)