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

rcParams['figure.figsize'] = 20, 10

# variables for web cam
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0


Ada = pickle.load(open('Ada_model.sav', 'rb'))

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

    face_list = []
    count = 0
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_resize = cv2.resize(face_image, (48, 48))
        faceGray = color.rgb2gray(face_resize)
        face_flat = faceGray.flatten()
        face_list.append(face_flat)
        count = count + 1

    face_data = np.array(face_list)
    face_data = face_data.astype('float32')
    face_data = face_data / 255

    # for face in face_data:
    if len(face_data) != 0:
        prediction = Ada.predict(face_data)
        print("prediction:", prediction)
        print(names[prediction[0]])

    removingfiles = glob.glob('D:/PythonProgram/pythonProject/pythonProject/opencv_frame.png')
    for i in removingfiles:
        os.remove(i)

cam.release()

cv2.destroyAllWindows()
