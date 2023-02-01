# This is a sample Python script.
import os
import keras
import face_recognition
from PIL import Image
import glob

import numpy as np
from numpy import array
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import asarray
import pickle

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
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from skimage.morphology import area_closing, area_opening
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from time import sleep
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from pylab import rcParams

'''
rcParams['figure.figsize'] = 20, 10

from keras.utils import np_utils

# variables for web cam
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
'''





def get_properties(image):

    img_data = np.array(image)
    img_data = img_data.astype('float32')
    img_data = img_data / 255
    properties = ['area', 'convex_area', 'bbox_area',
                  'major_axis_length', 'minor_axis_length',
                  'perimeter', 'equivalent_diameter',
                  'mean_intensity', 'solidity', 'eccentricity']
    dataframe = pd.DataFrame(columns=properties)
    for img in img_data:
        grayscale = rgb2gray(img)
        threshold = threshold_otsu(grayscale)
        binarized = grayscale < threshold
        closed = area_closing(binarized, 1000)
        opened = area_opening(closed, 1000)
        labeled = label(opened)
        regions = regionprops(labeled)
        data = pd.DataFrame(regionprops_table(labeled, grayscale,
                                              properties=properties))
        data = data[(data.index != 0) & (data.area > 100)]
        dataframe = pd.concat([dataframe, data])
    return dataframe


RF = pickle.load(open('RF_model.sav', 'rb'))


names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

imagePath = 'D:/CK+48/anger/S010_004_00000017.png'


face_list = []

input_img = cv2.imread(imagePath)
input_img_resize = cv2.resize(input_img, (48, 48))

    # You can access the actual face itself like this:
face_list.append(input_img_resize)


face_data = np.array(face_list)
face_data = face_data.astype('float32')
face_data = face_data / 255

df = get_properties(face_data)
df['ratio_length'] = (df['major_axis_length'] /
                      df['minor_axis_length'])
df['perimeter_ratio_major'] = (df['perimeter'] /
                               df['major_axis_length'])
df['perimeter_ratio_minor'] = (df['perimeter'] /
                               df['minor_axis_length'])
df['area_ratio_convex'] = df['area'] / df['convex_area']
df['area_ratio_bbox'] = df['area'] / df['bbox_area']
df['peri_over_dia'] = df['perimeter'] / df['equivalent_diameter']


# for face in face_data:
prediction = RF.predict(df)
print("prediction:", prediction)
classes = np.argmax(prediction, axis=1)
print(names[classes[0]])
print(classes)