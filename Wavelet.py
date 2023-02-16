import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications.densenet import layers
from matplotlib import cm
import pywt
import pywt.data

import cv2
import os
from keras.callbacks import EarlyStopping

from keras.utils import np_utils
from skimage import color
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


# combines numpy arrays to together into a 2x2
def get_concat(img1, img2, img3, img4):
    top = np.hstack((img1, img2))
    bottom = np.hstack((img3, img4))
    result = np.vstack((top, bottom))
    return result


def create_model(x_train, y_train, callback):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(52, 52, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(25, input_dim=2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=200, callbacks=[callback])

    return model


data_path = 'D:/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (48, 48))
        imgGray = color.rgb2gray(input_img_resize)
        img_data_list.append(imgGray)

img_data = np.array(img_data_list)
# img_data = img_data.astype('float32')
# img_data = img_data / 255

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


print(len(img_data))

processed_data = []

for img in img_data:
    # Load image
    holder = []

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    # print("break")
    # print(coeffs2)
    # processed_data.append(coeffs2)
    LL, (LH, HL, HH) = coeffs2
    for i, a in enumerate([LL, LH, HL, HH]):
        # processed_data.append(a)
        holder.append(a)

    concat_img = get_concat(holder[0], holder[1], holder[2], holder[3])
    concat_img = concat_img.astype('float32')
    concat_img = concat_img * 255
    backtorgb = cv2.cvtColor(concat_img, cv2.COLOR_GRAY2RGB)
    processed_data.append(backtorgb)

    '''
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        # processed_data.append(fig.add_subplot(1, 4, i + 1)) # creating array filled with all the wavelet transformed data
        ax.imshow(a, interpolation="nearest")
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
    '''

# print(img_data)
# plt.show()


pro_data = np.array(processed_data)
pro_data = pro_data.astype('float32')
pro_data = pro_data / 255
print(pro_data.shape)

# pro_data_LL = np.array(processed_data_LL)
# pro_data_LL = pro_data_LL.astype('float32')
# pro_data_LL = pro_data_LL / 255

'''
labels[0:536] = 0
labels[537:752] = 1
labels[753:1460] = 2
labels[1461:1760] = 3
labels[1761:2588] = 4
labels[2589:2924] = 5
labels[2925:3920] = 6
'''

# print("test")
# print(labels[1024])

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def getLabel(id):
    return ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'][id]


# pro_data_LL = pro_data_LL.reshape(list(pro_data_LL.shape) + [1])
# print(pro_data_LL[0].shape)
plt.imshow(pro_data[0])
plt.show()

Y = np_utils.to_categorical(labels, num_classes)

x, y = shuffle(pro_data, Y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)
x_test = x_test

data_generator_woth_aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
data_generator_no_aug = ImageDataGenerator()

train_generator = data_generator_woth_aug.flow(x_train, y_train)
validation_generator = data_generator_woth_aug.flow(x_test, y_test)

# x_train = x_train.reshape(list(x_train.shape) + [1])
callback = EarlyStopping(monitor='loss', patience=5)

# create directory for models
os.makedirs('models')
# fit and save models
n_members = 5
for i in range(n_members):
    # fit model
    model = create_model(x_train, y_train, callback)
    # save model
    filename = 'models/model_' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)

# model_custom = create_model()
# model_custom.summary()

# print(np.array(x_train).shape) # 784,2
# print(np.array(y_train).shape) # 784, 7
# history = model_custom.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=200, callbacks=[callback])

# abc = AdaBoostClassifier(random_state=96)

# model_custom = abc.fit(x_train, y_train)

# print("Evaluate on test data")
# results = model_custom.evaluate(x_test, y_test, verbose=0)
# print("test loss, test acc:", results)

# model_custom.save("wavelet")
