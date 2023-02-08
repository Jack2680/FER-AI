import os
import keras
import face_recognition
from PIL import Image
import glob
from skimage import color

import numpy as np

import cv2
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
import pickle
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

rcParams['figure.figsize'] = 20, 10

# variables for web cam
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

Ada = pickle.load(open('Ada_model.sav', 'rb'))
idx_sorted = np.load('haar_features.npy')

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
feature_coord = np.load('haar_feature_coords.npy')

# restore np.load for future normal usage
np.load = np_load_old

feature_types = ['type-2-x', 'type-2-y']

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    # sleep(0.5)  # this seems to work
    img_name = "opencv_frame.png"
    cv2.imwrite(img_name, frame)  # has to save the photo to use it

    # image = face_recognition.load_image_file("opencv_frame_{}.png".format(img_counter))
    image = face_recognition.load_image_file("opencv_frame.png")
    face_locations = face_recognition.face_locations(image)

    count = 0
    for face_location in face_locations:
        sum_emotion = []
        top, right, bottom, left = face_location

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_resize = cv2.resize(face_image, (25, 25))  # To change this to 48, 48 need to train Adaboost to be 48, 48
        faceGray = color.rgb2gray(face_resize)
        # Extract all possible features this takes to long so need to save as numpy

        for idx in range(5):
            face_list = []
            applied_img = draw_haar_like_feature(faceGray, 0, 0,
                                                 faceGray.shape[1],
                                                 faceGray.shape[0],
                                                 [feature_coord[idx_sorted[idx]]*10])
            # plt.imshow(applied_img)

            face_flat = applied_img.flatten()
            face_list.append(face_flat)

            face_data = np.array(face_list)
            face_data = face_data.astype('float32')
            face_data = face_data / 255

            # for face in face_data:
            if len(face_data) != 0:
                prediction = Ada.predict(face_data)
                sum_emotion.append(names[prediction[0]])
                print("prediction:", prediction)
                print(names[prediction[0]])
        # print(sum_emotion)
        # plt.show()

    removing_files = glob.glob('D:/PythonProgram/pythonProject/pythonProject/opencv_frame.png')
    for i in removing_files:
        os.remove(i)

cam.release()

cv2.destroyAllWindows()
