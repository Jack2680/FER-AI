import sys
import random
from time import time

import os
import cv2 as cv
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn import metrics, ensemble
from keras.callbacks import EarlyStopping
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import AdaBoostClassifier
import face_recognition
import skimage.data
from keras.utils import np_utils
from skimage import color
from tensorflow import image
import matplotlib.image as mpimg
from sklearn.utils import shuffle

from dask import delayed
from skimage.data import lfw_subset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature#


def get_concat(img1, img2):
    morb = np.vstack((img1, img2))
    return morb


data_path = 'D:/CK+48'
# data_path = 'D:/images/train'
data_dir_list = os.listdir(data_path)

'''
test_img_anger = cv.imread('D:/CK+48/anger/S010_004_00000017.png')
test_img_contempt = cv.imread('D:/CK+48/contempt/S138_008_00000007.png')
test_img_disgust = cv.imread('D:/CK+48/disgust/S005_001_00000009.png')
test_img_happy = cv.imread('D:/CK+48/happy/S010_006_00000013.png')

face_image = cv.resize(test_img_disgust, (128, 128))

# cv.rectangle(face_image, (35, 85), (105, 115), (0, 255, 0)) # mouth
roi_mouth = face_image[85:85 + 40, 30:30 + 70] #mouth snip
mouth = np.array(roi_mouth)
mouth = cv.resize(mouth, (115, 40))

# cv.rectangle(face_image, (10, 12), (60, 52), (0, 255, 0)) # left eye
eye_snip = face_image[25:25 + 40, 10:10 + 105] # eye snip
eye = np.array(eye_snip)

# get_concat(eye, mouth)
print(mouth.shape)
print(eye.shape)

eye = cv.resize(eye, (115, 40))

get_concat(eye, mouth)
'''


# Applying the face detection method on the grayscale image


img_data_list = []



for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv.resize(input_img, (128, 128))

        mouth_snip = input_img_resize[85:85 + 40, 30:30 + 70]  # mouth snip
        eye_snip = input_img_resize[25:25 + 40, 10:10 + 105]  # eye snip
        eye_snip = cv.resize(eye_snip, (115, 40))
        concat_img = get_concat(mouth_snip, eye_snip)

        imgGray = color.rgb2gray(concat_img)
        input_flat = imgGray.flatten()
        img_data_list.append(input_flat)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255

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
