import sys
from time import time

import os
import cv2 as cv
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import AdaBoostClassifier
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
from skimage.feature import draw_haar_like_feature

data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = mpimg.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv.resize(input_img, (48, 48))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')

num_classes = 7

num_of_samples = img_data.shape[0] * 5
labels = np.ones((num_of_samples,), dtype='int64')

'''
labels[0:134] = 0
labels[135:188] = 1
labels[189:365] = 2
labels[366:440] = 3
labels[441:647] = 4
labels[648:731] = 5
labels[732:980] = 6
'''
# multiplying labels by 5 for all haar features.

labels[0:670] = 0
labels[671:940] = 1
labels[941:1825] = 2
labels[1826:2200] = 3
labels[2201:3235] = 4
labels[3236:3655] = 5
labels[3656:4900] = 6

print(img_data.shape)
'''
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    print("Before", frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray) # use this?
    print("After", faces)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        print("Eyes", eyes)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        mouth = mouth_cascade.detectMultiScale(faceROI)
        print("Mouth", mouth)
        for (x2, y2, w2, h2) in mouth:
            mouth_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, mouth_center, radius, (255, 0, 0), 4)
    cv.imshow('Capture - Face detection', frame)

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
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)


while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break



'''


def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


images = img_data[:980]  # look at what contains in this dataset
# images = img_data[:50]
print(images.shape)

# To speed up the example, extract the two types of features only
feature_types = ['type-2-x', 'type-2-y']

# Build a computation graph using Dask. This allows the use of multiple
# CPU cores later during the actual computation
X = delayed(extract_feature_image(img, feature_types) for img in images)
# Compute the result
t_start = time()
X = np.array(X.compute(scheduler='single-threaded'))
time_full_feature_comp = time() - t_start

# Label images (100 faces and 100 non-faces)
y = np.array([1] * 490 + [0] * 490)
# y = np.array([1] * 25 + [0] * 25)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=735,
                                                    random_state=0,
                                                    stratify=y)

# Extract all possible features
feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                            feature_type=feature_types)

# Train a random forest classifier and assess its performance
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Sort features in order of importance and plot the six most significant
idx_sorted = np.argsort(clf.feature_importances_)[::-1]  # if i could pickle this would be good

'''
fig, axes = plt.subplots(3, 2)
for idx, ax in enumerate(axes.ravel()):
    image = images[1]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[idx_sorted[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

_ = fig.suptitle('The most important features')
'''

haar_data_list = []
print(img_data.shape)

for img in img_data:
    for idx in range(5):
        applied_img = draw_haar_like_feature(img, 0, 0,
                                             img_data.shape[2],
                                             img_data.shape[1],
                                             [feature_coord[idx_sorted[idx]]])
        haar_flatten = applied_img.flatten()
        haar_data_list.append(haar_flatten)

# plt.imshow(haar_data_list[104])

haar_data = np.array(haar_data_list)
print(haar_data.shape)

a, b = shuffle(haar_data_list, labels, random_state=2)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.4, random_state=2)
a_test = a_test

data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
data_generator_no_aug = ImageDataGenerator()

abc = AdaBoostClassifier(random_state=96, base_estimator=RandomForestClassifier(random_state=101), n_estimators=100, learning_rate=0.01)

# Train Adaboost Classifer
abc.fit(a_train, b_train)

score_seen = abc.score(a_train, b_train)
score_unseen = abc.score(a_test, b_test)

print("Score seen data:", score_seen)
print("Score unseen data:", score_unseen)

# save the model to disk
filename = 'Ada_model.sav'
pickle.dump(abc, open(filename, 'wb'))

'''
cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
cdf_feature_importances /= cdf_feature_importances[-1]  # divide by max value
sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
sig_feature_percent = round(sig_feature_count /
                            len(cdf_feature_importances) * 100, 1)
print((f'{sig_feature_count} features, or {sig_feature_percent}%, '
       f'account for 70% of branch points in the random forest.'))

# Select the determined number of most informative features
feature_coord_sel = feature_coord[idx_sorted[:sig_feature_count]]
feature_type_sel = feature_type[idx_sorted[:sig_feature_count]]
# Note: it is also possible to select the features directly from the matrix X,
# but we would like to emphasize the usage of `feature_coord` and `feature_type`
# to recompute a subset of desired features.

# Build the computational graph using Dask
X = delayed(extract_feature_image(img, feature_type_sel, feature_coord_sel)
            for img in images)
# Compute the result
t_start = time()
X = np.array(X.compute(scheduler='single-threaded'))
time_subs_feature_comp = time() - t_start

# y = np.array([1] * 25 + [0] * 25)
y = np.array([1] * 490 + [0] * 490)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=30,
                                                    random_state=0,
                                                    stratify=y)

t_start = time()
clf.fit(X_train, y_train)
time_subs_train = time() - t_start

auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

summary = ((f'Computing the full feature set took '
            f'{time_full_feature_comp:.3f}s, '
            f'plus {time_full_train:.3f}s training, '
            f'for an AUC of {auc_full_features:.2f}. '
            f'Computing the restricted feature set took '
            f'{time_subs_feature_comp:.3f}s, plus {time_subs_train:.3f}s '
            f'training, for an AUC of {auc_subs_features:.2f}.'))

print(summary)
plt.show()
'''