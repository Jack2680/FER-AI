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
from skimage.feature import draw_haar_like_feature

# calculates precision for 1:100 dataset with 90 tp and 30 fp
from sklearn.metrics import precision_score

data_path = 'D:/CK+48'
# data_path = 'D:/images/train'
data_dir_list = os.listdir(data_path)
# test_img_anger = mpimg.imread('D:/CK+48/anger/S010_004_00000017.png')
# test_img_contempt = mpimg.imread('D:/CK+48/contempt/S138_008_00000007.png')
# test_img_disgust = mpimg.imread('D:/CK+48/disgust/S005_001_00000009.png')
# test_img_happy = mpimg.imread('D:/CK+48/happy/S010_006_00000013.png')

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv.resize(input_img, (128, 128))
        if dataset == "contempt":
            img_data_list.append(input_img_resize)
            img_data_list.append(input_img_resize)
            img_data_list.append(input_img_resize)
        img_data_list.append(input_img_resize)


img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255

'''
#making more contempt data as is most common facial expression
contempt_data = img_data[135:188]
contempt_data_2 = contempt_data.copy()
contempt_data_3 = contempt_data.copy()

contempt_data.append(contempt_data_2)
contempt_data.append(contempt_data_3)
img_data.append(contempt_data)
'''

num_classes = 7

num_of_samples_haar = img_data.shape[0]
# num_of_samples_ada = img_data.shape[0] * 5
haar_labels = np.ones((num_of_samples_haar,), dtype='int64')
# ada_labels = np.ones((num_of_samples_ada,), dtype='int64')

haar_labels[0:134] = 0
haar_labels[135:350] = 1
haar_labels[351:527] = 2
haar_labels[528:602] = 3
haar_labels[603:809] = 4
haar_labels[810:893] = 5
haar_labels[894:1142] = 6

'''
haar_labels[0:3992] = 0  # anger
haar_labels[3993:4428] = 1  # disgust
haar_labels[4429:8532] = 2  # fear
haar_labels[8533:15696] = 3  # happy
haar_labels[15697:20678] = 4  # neutral
haar_labels[20679:25616] = 5  # sadness
haar_labels[25617:28821] = 6  # suprise
'''
# multiplying labels by 5 for all haar features.

'''
ada_labels[0:19960] = 0
ada_labels[19961:22140] = 1
ada_labels[22141:42660] = 2
ada_labels[42661:78480] = 3
ada_labels[78481:103390] = 4
ada_labels[103390:128080] = 5
ada_labels[128081:144105] = 6
'''

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


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1,
                                             rotation_range=45, brightness_range=[0.4, 1.2])

data_generator_no_aug = ImageDataGenerator()

mix_generator = data_generator_woth_aug.flow(img_data, haar_labels)

'''
mix_images, mix_labels = shuffle(img_data, haar_labels, random_state=2)


gray_images = []
for img in mix_images:
    img = rgb2gray(img)
    gray_images.append(img)

gray_data = np.array(gray_images)
gray_data = gray_data.astype('float32')

images = gray_data[:980]  # look at what contains in this dataset
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
# y = np.array([1] * 490 + [0] * 490)
# y = np.array([1] * 25 + [0] * 25)

X_train, X_test, y_train, y_test = train_test_split(X, mix_labels[:980], train_size=735,
                                                    random_state=0,
                                                    stratify=mix_labels[:980])

# Extract all possible features
feature_coord, feature_type = \
    haar_like_feature_coord(width=images.shape[2], height=images.shape[1],
                            feature_type=feature_types)

np.save('haar_feature_coords.npy', feature_coord)

# Train a random forest classifier and assess its performance
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
t_start = time()
clf.fit(X_train, y_train)
time_full_train = time() - t_start
# auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Sort features in order of importance and plot the six most significant
idx_sorted = np.argsort(clf.feature_importances_)[::-1]

for idx in range(5):
    print(idx)
    print("------")
    print(feature_coord[idx_sorted[idx]])
    print("++++++++")

'''

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
idx_mouth = [[(18, 8), (20, 17)], [(20, 8), (22, 17)]]  # mouth coords

idx_nose = [[(10, 10), (14, 14)], [(14, 10), (15, 14)]]  # nose coords

factor = 5.12  # when value 1 it is set to work on 25, 25 image
idx_eyes = [[(round(4 * factor), round(21 * factor)), (round(6 * factor), round(4 * factor))],
            [(round(6 * factor), round(21 * factor)), (round(8 * factor), round(4 * factor))],
            [(round(10 * factor), round(10 * factor)), (round(14 * factor), round(14 * factor))],
            [(round(14 * factor), round(10 * factor)), (round(15 * factor), round(14 * factor))],
            [(round(18 * factor), round(8 * factor)), (round(20 * factor), round(17 * factor))],
            [(round(20 * factor), round(8 * factor)),
             (round(22 * factor), round(17 * factor))]]  # full haar face coords
# idx_sorted.append(idx)

# print(type(idx_sorted))
np.save('haar_features.npy', idx_eyes)

haar_data_list = []
# print(img_data.shape)

'''
for img in img_data:
    # img_resize = cv.resize(img, (200, 200))
    for idx in range(5):
        applied_img = draw_haar_like_feature(img, 0, 0,
                                             img_data.shape[2],
                                             img_data.shape[1],
                                             [feature_coord[idx_sorted[idx]]])
        print(applied_img.shape)
        haar_flatten = applied_img.flatten()
        haar_data_list.append(haar_flatten)

'''

print("applying haar")
for img in img_data:
    applied_img = draw_haar_like_feature(img, 0, 0,
                                         img_data.shape[2],
                                         img_data.shape[1],
                                         [idx_eyes])
    # print(applied_img.shape)
    haar_flatten = applied_img.flatten()
    haar_data_list.append(haar_flatten)

# plt.imshow(haar_data_list[104])

haar_data = np.array(haar_data_list)
print(haar_data.shape)
haar_data = haar_data.astype('float32')
haar_data = haar_data / 255

# print(ada_labels.shape)

a, b = shuffle(haar_data, haar_labels, random_state=2)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=2)
a_test = a_test

# data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
# data_generator_no_aug = ImageDataGenerator()

# train_generator = data_generator_woth_aug.flow(a_train, b_train)
# validation_generator = data_generator_woth_aug.flow(a_test, b_test)

# reduce n_estimators helps with overfitting
# abc = ensemble.RandomForestClassifier(random_state=96, n_estimators=20)

abc = AdaBoostClassifier(random_state=96, base_estimator=RandomForestClassifier(random_state=101), n_estimators=20,
                         learning_rate=0.01)

# abc = ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=0.01, random_state=96) # 0.04
# abc = ensemble.AdaBoostRegressor(estimator=None, *, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None, base_estimator='deprecated')

print(a_train.shape)
print(b_train.shape)
# Train Adaboost Classifer
print("fitting")
abc.fit(a_train, b_train)  # Run this to see if doesnt need to be removed

score_seen = abc.score(a_train, b_train)
score_unseen = abc.score(a_test, b_test)

print("Score seen data:", score_seen)
print("Score unseen data:", score_unseen)

# save the model to disk
filename = 'Ada_model.sav'
joblib.dump(abc, filename)

y_true = b_test  # label
y_pred = abc.predict(a_test)

print(metrics.confusion_matrix(y_true, y_pred))
# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, digits=3))

'''
act_pos1 = [1 for _ in range(100)]
act_pos2 = [2 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos1 + act_pos2 + act_neg

pred_pos1 = [0 for _ in range(50)] + [1 for _ in range(50)]
pred_pos2 = [0 for _ in range(1)] + [2 for _ in range(99)]
pred_neg = [1 for _ in range(20)] + [2 for _ in range(51)] + [0 for _ in range(9929)]
y_pred = pred_pos1 + pred_pos2 + pred_neg
# calculate prediction
precision = precision_score(y_true, y_pred, labels=[1,2], average='micro')
print('Precision: %.3f' % precision)
'''
'''
test_faces_list = [test_img_anger, test_img_contempt, test_img_disgust, test_img_happy]
for test_face in test_faces_list:
    test_img_resize = cv.resize(test_face, (25, 25))
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
            prediction = abc.predict(test_test)
            print("prediction:", prediction)
'''
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
