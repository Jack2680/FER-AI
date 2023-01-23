import os

from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
from sklearn.utils import shuffle

import cv2
import numpy as np
from keras.utils import np_utils
from numpy import dstack
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model
from sklearn.linear_model import LogisticRegression


def fit_model(trainX, trainY):
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

    model.add(Dense(25, input_dim=2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # fit the model
    history = model.fit(trainX, trainY, epochs=100, verbose=0)

    return model


# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'models/model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
            print(stackX.shape)
            stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
            print(stackX.shape)
            print(stackX)
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat


data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

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


# one hot encode output variable
Y = to_categorical(labels, num_classes)

x, y = shuffle(img_data, Y, random_state=2)

# split into train and test
n_train = 100
trainX, testX = x[:n_train, :], x[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
print(trainX.shape, testX.shape)

'''
# define model
# create directory for models
os.makedirs('models')
# fit and save models
n_members = 5
for i in range(n_members):
    # fit model
    model = fit_model(trainX, trainy)
    # save model
    filename = 'models/model_' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)
'''

# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# evaluate standalone models on test dataset
for model in members:
    # testy_enc = to_categorical(testy)
    _, acc = model.evaluate(testX, testy, verbose=0)
    print('Model Accuracy: %.3f' % acc)

# fit stacked model using the ensemble
model = fit_stacked_model(members, testX, testy)
# evaluate model on test set
yhat = stacked_prediction(members, model, testX)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
# fit model
# history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

'''
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
'''

'''
# learning curves of model accuracy
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
'''
