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
# import matplotlib.image as mpimg
from pylab import rcParams

rcParams['figure.figsize'] = 20, 10

from keras.utils import np_utils

# variables for web cam
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
"""
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
"""
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

'''
imagePath = 'D:/DuoImage.png'
image = face_recognition.load_image_file(imagePath)
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
prediction = model_custom.predict(face_data[:count])
print("prediction:", prediction)
classes = np.argmax(prediction, axis=1)
print(names[classes[0]])
print(classes)
'''


def diction_output(output):
    for x in output:
        print(x)
        em_list = output[x]
        size = len(em_list)
        d = {}
        for word in em_list:
            if word in d:
                d[word] = d[word] + 1
            else:
                d[word] = 1

        for key in list(d.keys()):
            print(key, ":", d[key]/size * 100)


model_custom = keras.models.load_model('EN_model')
# model_custom = keras.models.load_model('stackedModel')

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

known_face_encodings = []
known_face_names = []
studentCount = 1
student_dict = {}

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        # print(student_dict)
        diction_output(student_dict)
        print("Escape hit, closing...")
        break

    # sleep(0.5)  # this seems to work
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    img_name = "opencv_frame.png"
    cv2.imwrite(img_name, rgb_small_frame)  # has to save the photo to use it

    # image = face_recognition.load_image_file("opencv_frame_{}.png".format(img_counter))
    image = face_recognition.load_image_file("opencv_frame.png")
    face_locations = face_recognition.face_locations(image)
    # face_encodings = face_recognition.face_encodings(image, face_locations)

    student = "Student{0}"
    '''
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        print(matches)

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            name = student.format(studentCount)
            studentCount += studentCount + 1
            known_face_names.append(name)
            known_face_encodings.append(face_encoding)
    '''
    count = 0
    # student = "Student{0}"

    for face_location in face_locations:
        face_list = []
        top, right, bottom, left = face_location

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_resize = cv2.resize(face_image, (48, 48))
        face_list.append(face_resize)
        count = count + 1

        # face_locations_en = face_recognition.face_locations(face_image)
        face_encodings = face_recognition.face_encodings(face_image)
        # only works when in for loop cant figure out why
        name = "Unkown"
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                name = known_face_names[matches.index(True)]
            else:
                name = student.format(studentCount)
                studentCount += studentCount + 1
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)
                student_dict[name] = ["a"]

        face_data = np.array(face_list)
        face_data = face_data.astype('float32')
        face_data = face_data / 255

        # for face in face_data:
        if len(face_data) != 0:
            prediction = model_custom.predict(face_data[:1])
            print("prediction:", prediction)
            classes = np.argmax(prediction, axis=1)
            print(name, names[classes[0]])
            print(classes)
            if name != "Unkown":
                student_dict[name].append(names[classes[0]])  # creating dictionary to track individual student emotions

        removingfiles = glob.glob('D:/PythonProgram/pythonProject/pythonProject/opencv_frame.png')
        for i in removingfiles:
            os.remove(i)

cam.release()

cv2.destroyAllWindows()
