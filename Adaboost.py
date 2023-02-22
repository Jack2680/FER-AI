# This is a sample Python script.
import os
import keras
import face_recognition

import numpy as np
from numpy import array, std, mean
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage import color

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
import cv2 as cv
from sklearn import datasets
import argparse
import pickle
from sklearn import metrics

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

'''
iris = datasets.load_iris()
A = iris.data
B = iris.target

# haar feature code
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='D:/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='D:/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--mouth_cascade', default='D:/mouth.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
mouth_cascade_name = args.mouth_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
mouth_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
if not mouth_cascade.load(cv.samples.findFile(mouth_cascade_name)):
    print('Error loading mouth cascade')
    exit(0)
'''

data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv.resize(input_img, (48, 48))
        imgGray = color.rgb2gray(input_img_resize)
        input_flat = imgGray.flatten()
        img_data_list.append(input_flat)

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


# Y = np_utils.to_categorical(labels, num_classes)

print(labels.shape)

x, y = shuffle(img_data, labels, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
x_test = x_test

data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
data_generator_no_aug = ImageDataGenerator()

# train_generator = data_generator_woth_aug.flow(x_train, y_train)
# validation_generator = data_generator_woth_aug.flow(x_test, y_test)

print(x_train.shape)  # has 4 features needs 2
print(y_train.shape)  # has 2 features need 1

# trainX = x_train[0].flatten(order='K')
# trainY = y_train.flatten()

'''

print(A.shape)
print(B.shape)

print(img_data.shape)
print(Y.shape)


X_train, X_test, y_train, y_test = train_test_split(img_data, Y, test_size=0.3)  # 70% training and 30% test

print(X_train.shape)
print(y_train.shape)


print(X_train.shape)
print(y_train.shape)
'''

abc = AdaBoostClassifier(random_state=96, base_estimator=RandomForestClassifier(random_state=101), n_estimators=20, learning_rate=0.01)

# Train Adaboost Classifer
abc.fit(x_train, y_train)

# save the model to disk
#filename = 'Ada_model.sav'
#pickle.dump(abc, open(filename, 'wb'))

y_true = y_test # label
# y_true = np.argmax(y_true, axis=0)
y_pred = abc.predict(x_test)
# y_pred = np.argmax(y_pred, axis=0)

print(metrics.confusion_matrix(y_true, y_pred))
# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, digits=3))

