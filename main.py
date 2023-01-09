# This is a sample Python script.
import os
import keras
import face_recognition

import numpy as np
from numpy import array
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from pylab import rcParams

rcParams['figure.figsize'] = 20, 10

from keras.utils import np_utils


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(48, 48, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='RMSprop')

    return model


data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (48, 48))

        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255
print(img_data.shape)

num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:134] = 0
labels[135:188] = 1
labels[189:365] = 2
labels[366:440] = 3
labels[441:647] = 4
labels[648:731] = 5
labels[732:980] = 6

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def getLabel(id):
    return ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'][id]


Y = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(img_data, Y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
x_test = x_test

data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
data_generator_no_aug = ImageDataGenerator()

train_generator = data_generator_woth_aug.flow(x_train, y_train)
validation_generator = data_generator_woth_aug.flow(x_test, y_test)

model_custom = create_model()
model_custom.summary()

history = model_custom.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
)

print("Evaluate on test data")
results = model_custom.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# using the model on the group image

# imagePath = 'D:/GroupImage.png'

# image = face_recognition.load_image_file(imagePath)
# face_locations = face_recognition.face_locations(image)

# for face_location in face_locations:
# top, right, bottom, left = face_location
# face_image = image[top:bottom, left:right]
# predict_img_resize = cv2.resize(face_image, (48, 48))

# pre_data = np.array(predict_img_resize)
# pre_data = pre_data.astype('float32')
# pre_data = pre_data / 255

# prediction = model_custom.predict(pre_data)
# print(prediction)


prediction = model_custom.predict(x_test[0:1])
print("prediction:", prediction)

# Generate arg maxes for predictions
classes = np.argmax(prediction, axis=1)
print(names[classes[0]])
print(classes)
