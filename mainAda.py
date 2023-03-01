import os
import keras
import face_recognition
from PIL import Image
import glob
from skimage import color

import numpy as np
from sklearn.externals import joblib

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

#Ada = pickle.load(open('Ada_model.sav', 'rb'))
Ada = joblib.load('Ada_model.sav')
print("test")

idx_sorted = np.load('haar_features.npy')

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
        print("In loop")
        top, right, bottom, left = face_location
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_Gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY) # Just to make face same as training data
        backtorgb = cv2.cvtColor(face_Gray, cv2.COLOR_GRAY2RGB)
        face_resize = cv2.resize(backtorgb, (128, 128))
        # Extract all possible features this takes to long so need to save as numpy
        face_list = []
        applied_img = draw_haar_like_feature(face_resize, 0, 0,
                                             face_resize.shape[1],
                                             face_resize.shape[0],
                                             [idx_sorted])
        plt.imshow(applied_img)
        plt.show()
        face_flat = applied_img.flatten()
        face_list.append(face_flat)

        face_data = np.array(face_list)
        face_data = face_data.astype('float32')
        face_data = face_data / 255
            # for face in face_data:
        if len(face_data) != 0:
            prediction = Ada.predict(face_data)
            print("prediction:", prediction)
            print(names[prediction[0]])

    removing_files = glob.glob('D:/PythonProgram/pythonProject/pythonProject/opencv_frame.png')
    for i in removing_files:
        os.remove(i)

cam.release()

cv2.destroyAllWindows()
