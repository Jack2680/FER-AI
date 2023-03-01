import os

from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics

import cv2
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
# from keras.utils import to_categorical
# from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from skimage import color
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pywt.data

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

'''
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
'''


# combines numpy arrays to together into a 2x2
def get_concat(img1, img2, img3, img4):
    top = np.hstack((img1, img2))
    bottom = np.hstack((img3, img4))
    result = np.vstack((top, bottom))
    return result


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


# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
        ensemble_visible = [model.input for model in members]
        ensemble_outputs = [model.output for model in members]
        merge = concatenate(ensemble_outputs)
        hidden = Dense(10, activation='relu')(merge)
        output = Dense(7, activation='softmax')(hidden)

    model = Model(inputs=ensemble_visible, outputs=output)

        # plot graph of ensemble
        # plot_model(model, show_shapes=True, to_file='model_graph.png')
        # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data
    print(inputy.shape)
    inputy_enc = to_categorical(inputy, 3)
    print(inputy_enc.shape)
    # fit model
    model.fit(X, inputy, epochs=100, verbose=0)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (64, 64))
        imgGray = color.rgb2gray(input_img_resize)
        img_data_list.append(imgGray)

img_data = np.array(img_data_list)
# img_data = img_data.astype('float32')
# img_data = img_data / 255
# print(img_data.shape)

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

processed_data = []

max_lev = 3
label_levels = 3
shape = img_data[0].shape

for img in img_data:
    for level in range(max_lev, max_lev + 1):
        c = pywt.wavedec2(img, 'db2', mode='periodization', level=level)
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]
        arr, slices = pywt.coeffs_to_array(c)
        # img_Gray = color.rgb2gray(arr)
        arr = arr.astype('float32')
        arr = arr * 255
        backtorgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        processed_data.append(backtorgb)

pro_data = np.array(processed_data)
pro_data = pro_data.astype('float32')
pro_data = pro_data / 255

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def getLabel(id):
    return ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'][id]


# one hot encode output variable
Y = to_categorical(labels, num_classes)

x, y = shuffle(pro_data, Y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
x_test = x_test

#data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
#data_generator_no_aug = ImageDataGenerator()

#train_generator = data_generator_woth_aug.flow(x_train, y_train)
#validation_generator = data_generator_woth_aug.flow(x_test, y_test)

'''
# split into train and test
n_train = 100
trainX, testX = x[:n_train, :], x[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
print(trainX.shape, testX.shape)
'''

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

# define ensemble model
stacked_model = define_stacked_model(members)

# fit stacked model on test dataset
fit_stacked_model(stacked_model, x_test, y_test)

stacked_model.summary()

stacked_model.save("stackedModel")

y_true = y_test # label
y_true = np.argmax(y_true, axis=1)
y_pred = predict_stacked_model(stacked_model, x_test)
y_pred = np.argmax(y_pred, axis=1)

print(metrics.confusion_matrix(y_true, y_pred))
# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, digits=3))

'''
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, x_test)
yhat = argmax(yhat, axis=1)
testy_arg = argmax(y_test, axis=1)
acc = accuracy_score(testy_arg, yhat)

print('Stacked Test Accuracy: %.3f' % acc)
'''

# evaluate the model
'''
_, train_acc = stacked_model.evaluate(x_train, y_train, verbose=0)
_, test_acc = stacked_model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
'''

# print('Stacked Test Accuracy: %.3f' % acc)
# evaluate the model
#_, train_acc = stacked_model.evaluate(trainX, trainy, verbose=0)
#_, test_acc = stacked_model.evaluate(trainX, trainy, verbose=0)
#print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

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
