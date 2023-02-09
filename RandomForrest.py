import os

import cv2
import numpy as np
import pandas as pd
from PIL._imaging import display
from keras.utils import np_utils
from skimage.feature import draw_haar_like_feature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
import cv2 as cv
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

from sklearn.model_selection import GridSearchCV

Ada = pickle.load(open('Ada_model.sav', 'rb'))

idx_sorted = np.load('haar_features.npy')

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
feature_coord = np.load('haar_feature_coords.npy')

# restore np.load for future normal usage
np.load = np_load_old

test_img_anger = mpimg.imread('D:/CK+48/anger/S010_004_00000017.png')
test_img_contempt = mpimg.imread('D:/CK+48/contempt/S138_008_00000007.png')
test_img_disgust = mpimg.imread('D:/CK+48/disgust/S005_001_00000009.png')
test_img_happy = mpimg.imread('D:/CK+48/happy/S010_006_00000013.png')

test_faces_list = [test_img_anger, test_img_contempt, test_img_disgust, test_img_happy]
for test_face in test_faces_list:
    test_img_resize = cv.resize(test_face, (25, 25))
    plt.imshow(test_img_resize)
    plt.show()
    for idx in range(5):
        test_list = []
        test_img = draw_haar_like_feature(test_img_resize, 0, 0,
                                          test_img_resize.shape[1],
                                          test_img_resize.shape[0],
                                          [feature_coord[idx_sorted[idx]]])
        test_flat = test_img.flatten()
        test_list.append(test_flat)

        test_test = np.array(test_list)
        test_test = test_test.astype('float32')
        test_test = test_test / 255

        if len(test_test) != 0:
            prediction = Ada.predict(test_test)
            print("prediction:", prediction)


'''
img = cv.imread('D:/CK+48/sadness/S080_005_00000011.png')
input_img_resize = cv.resize(img, (100, 100))
imgGray = color.rgb2gray(input_img_resize)
print(input_img_resize.shape)
print(imgGray.shape)
imgplot = plt.imshow(imgGray)
plt.show()

# input_img_resize = cv.resize(input_img, (25, 25))
# imgGray = color.rgb2gray(input_img_resize)
'''
'''
data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list = []


for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (876, 637))

        img_data_list.append(input_img_resize)


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255
print(img_data[0].shape)



gray_face = rgb2gray(img_data[0])
thresh = threshold_otsu(gray_face)
binarized = gray_face < thresh

closed = area_closing(binarized)
opened = area_opening(closed)

label_im = label(opened)
regions = regionprops(label_im)

masks = []
bbox = []
for num, x in enumerate(regions):
    area = x.area
    convex_area = x.convex_area
    if num!=0 and x.area >= 100:
        masks.append(regions[num].convex_image)
        bbox.append(regions[num].bbox)
count = len(masks)
fig, axis = plt.subplots(4, int(count/4), figsize=(15, 6))
for ax, box, mask in zip(axis.flatten(), bbox, masks):
    image = gray_face[box[0]:box[2], box[1]:box[3]] * mask
    ax.imshow(image)
    ax.axis('off')

properties = ['area','convex_area','bbox_area',
              'major_axis_length', 'minor_axis_length',
              'perimeter', 'equivalent_diameter',
              'mean_intensity', 'solidity', 'eccentricity']
pd.DataFrame(regionprops_table(label_im, gray_face,
                               properties=properties)).head(10)
'''

'''
def get_properties():
    data_path = 'D:/CK+48'
    data_dir_list = os.listdir(data_path)

    img_data_list = []

    for dataset in data_dir_list:
        img_list = os.listdir(data_path + '/' + dataset)
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        for img in img_list:
            input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
            input_img_resize = cv2.resize(input_img, (876, 637))

            img_data_list.append(input_img_resize)

    img_data = np.array(img_data_list)
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


df = get_properties()
print("The shape of the dataframe is: ", df.shape)
df['ratio_length'] = (df['major_axis_length'] /
                      df['minor_axis_length'])
df['perimeter_ratio_major'] = (df['perimeter'] /
                               df['major_axis_length'])
df['perimeter_ratio_minor'] = (df['perimeter'] /
                               df['minor_axis_length'])
df['area_ratio_convex'] = df['area'] / df['convex_area']
df['area_ratio_bbox'] = df['area'] / df['bbox_area']
df['peri_over_dia'] = df['perimeter'] / df['equivalent_diameter']
# final_df = df[df.drop('type', axis=1).columns].astype(float)
# final_df = final_df.replace(np.inf, 0)
# final_df['type'] = df['type']

# x = df



num_classes = 7

num_of_samples = df.shape[0]
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
'''
'''

Y = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(df, Y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)
x_test = x_test

print(x_train.shape)
print(y_train.shape)


RF = RandomForestClassifier(max_depth=6, n_estimators=100)
RF.fit(x_train, y_train)

# save the model to disk
filename = 'RF_model.sav'
pickle.dump(RF, open(filename, 'wb'))


y_pred_RF= RF.predict(x_test)

print("Accuracy against seen data: ", RF.score(x_train, y_train))
print("Accuracy against unseen data", RF.score(x_test, y_test))

'''
'''
data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
data_generator_no_aug = ImageDataGenerator()

train_generator = data_generator_woth_aug.flow(x_train, y_train)
validation_generator = data_generator_woth_aug.flow(x_test, y_test)

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2), n_estimators=300, learning_rate=1
)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=300,
    learning_rate=1.5,
    algorithm="SAMME",
)

'''

# need to figure out how to process image data
'''
x_train = np.argmax(x_train, axis=1)
x_train = np.argmax(x_train, axis=1)
y_train = np.argmax(y_train, axis=1)

x_test = np.argmax(x_test, axis=1)
x_test = np.argmax(x_test, axis=1)
y_test = np.argmax(y_test, axis=1)
print(x_train.shape)
print(y_train.shape)
'''

'''
Ada_classifier = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                                    n_estimators=39, random_state=None)
AdaBoost = Ada_classifier.fit(x_train, y_train)

AdaBoost_pred = AdaBoost.predict(x_test)

print("The accuracy of the model is:  ", accuracy_score(y_test, AdaBoost_pred))
'''

# RF = RandomForestClassifier(max_depth=6, n_estimators=100)
# RF.fit(x_train, y_train)

'''
bdt_real.fit(x_train, y_train)
bdt_discrete.fit(x_train, y_train)

real_test_errors = []
discrete_test_errors = []

for real_test_predict, discrete_test_predict in zip(
    bdt_real.staged_predict(x_test), bdt_discrete.staged_predict(x_test)
):
    real_test_errors.append(1.0 - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(1.0 - accuracy_score(discrete_test_predict, y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1), discrete_test_errors, c="black", label="SAMME")
plt.plot(
    range(1, n_trees_real + 1),
    real_test_errors,
    c="black",
    linestyle="dashed",
    label="SAMME.R",
)
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel("Test Error")
plt.xlabel("Number of Trees")

plt.subplot(132)
plt.plot(
    range(1, n_trees_discrete + 1),
    discrete_estimator_errors,
    "b",
    label="SAMME",
    alpha=0.5,
)
plt.plot(
    range(1, n_trees_real + 1), real_estimator_errors, "r", label="SAMME.R", alpha=0.5
)
plt.legend()
plt.ylabel("Error")
plt.xlabel("Number of Trees")
plt.ylim((0.2, max(real_estimator_errors.max(), discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights, "b", label="SAMME")
plt.legend()
plt.ylabel("Weight")
plt.xlabel("Number of Trees")
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()
'''
