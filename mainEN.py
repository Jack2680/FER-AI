# This is a sample Python script.
import os
import keras
import face_recognition
from PIL import Image
import glob

import numpy as np
from numpy import array
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import asarray

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams


def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)

rcParams['figure.figsize'] = 20, 10

from keras.utils import np_utils

# variables for web cam
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

model_custom = keras.models.load_model('stackedModel')
# model_custom = keras.models.load_model('stackedModel')

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    # sleep(0.5)  # this seems to work
    img_name = "opencv_frame.png"
    cv2.imwrite(img_name, frame)  # has to save the photo to use it

    # image = face_recognition.load_image_file("opencv_frame_{}.png".format(img_counter))
    image = face_recognition.load_image_file("opencv_frame.png")
    face_locations = face_recognition.face_locations(image)

    face_list = []
    count = 0
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_resize = cv2.resize(face_image, (48, 48))
        face_list.append(face_resize)
        count = count + 1

    face_data = np.array(face_list)
    face_data = face_data.astype('float32')
    face_data = face_data / 255

    # for face in face_data:
    if len(face_data) != 0:
        prediction = predict_stacked_model(model_custom, face_data)
        print("prediction:", prediction)
        classes = np.argmax(prediction, axis=1)
        print(names[classes[0]])
        print(classes)

    removingfiles = glob.glob('D:/PythonProgram/pythonProject/pythonProject/opencv_frame.png')
    for i in removingfiles:
        os.remove(i)

cam.release()

cv2.destroyAllWindows()