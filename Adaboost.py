# This is a sample Python script.
import os
import keras
import face_recognition
import numpy
from PIL import Image
from matplotlib import pyplot
from skimage.feature._cascade import rgb2gray
from sklearn.ensemble import AdaBoostClassifier

import numpy as np
from numpy import array, std, mean
from sklearn.tree import DecisionTreeClassifier
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
import cv2
from sklearn import datasets
import pickle
from sklearn import metrics

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# get a list of models to evaluate
def get_models():
    models = dict()
    # explore depths from 1 to 10
    for i in range(1, 11):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


iris = datasets.load_iris()
A = iris.data
B = iris.target

data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (48, 48))
        input_flat = input_img_resize.flatten()
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)
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

abc = AdaBoostClassifier(random_state=96, base_estimator=RandomForestClassifier(random_state=101), n_estimators=100, learning_rate=0.01)

# Train Adaboost Classifer
abc.fit(x_train, y_train)

# save the model to disk
filename = 'Ada_model.sav'
pickle.dump(abc, open(filename, 'wb'))

score_seen = abc.score(x_train, y_train)
score_unseen = abc.score(x_test, y_test)

print("Score seen data:", score_seen)
print("Score unseen data:", score_unseen)

