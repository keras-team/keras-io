"""
Title: Video Classification With TimeDistributed Layer
Author: [Sujoy K Goswami](https://www.linkedin.com/in/sujoy-kumar-goswami/)
Date created: 2022/01/09
Last modified: 2022/01/10
Description: Guide to examine any video-classification-model quickly without any GPU.
"""

"""
## Introduction

Video Classification DL Models are heavy and need huge size of data.
So it is time-consuming and expensive. Here it is shown, how to examine your model
quickly, before feeding the actual data, and that also without need of any GPU.

Here video dataset will be created; a white rectangle moving in different directions,
on a black canvas. The sample code for creating left-moving-rectangle videos is below.
"""

import numpy as np
import skvideo.io as sk
from IPython.display import Video

# creating sample video data
num_vids = 5
num_imgs = 50
img_size = 50
min_object_size = 1
max_object_size = 5

for i_vid in range(num_vids):
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    vid_name = "vid" + str(i_vid) + ".mp4"
    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    i_img = 0
    while x > 0:
        imgs[i_img, y : y + h, x : x + w] = 255  # set rectangle as foreground
        x = x - 1
        i_img = i_img + 1
    sk.vwrite(vid_name, imgs.astype(np.uint8))
Video(
    "vid3.mp4"
)  # play a video; the script and video generated should be in same folder

"""
## Data Generation and Preparation

Now dataset with 4 classes will be created where, a rectangle is moving in 4
different directions in those classes respectively.
"""


# preparing dataset
X_train = []
Y_train = []
labels = {0: "left", 1: "right", 2: "up", 3: "down"}  # 4 classes
num_vids = 40
num_imgs = 40
img_size = 40
min_object_size = 1
max_object_size = 5
# video frames with left moving object
for i_vid in range(num_vids):
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    # vid_name = 'vid' + str(i_vid) + '.mp4'
    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    i_img = 0
    while x > 0:
        imgs[i_img, y : y + h, x : x + w] = 255  # set rectangle as foreground
        x = x - 1
        i_img = i_img + 1
    X_train.append(imgs)
for i in range(0, num_imgs):
    Y_train.append(0)
# video frames with right moving object
for i_vid in range(num_vids):
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    # vid_name = 'vid' + str(i_vid) + '.mp4'
    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    i_img = 0
    while x < img_size:
        imgs[i_img, y : y + h, x : x + w] = 255  # set rectangle as foreground
        x = x + 1
        i_img = i_img + 1
    X_train.append(imgs)
for i in range(0, num_imgs):
    Y_train.append(1)
# video frames with up moving object
for i_vid in range(num_vids):
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    # vid_name = 'vid' + str(i_vid) + '.mp4'
    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    i_img = 0
    while y > 0:
        imgs[i_img, y : y + h, x : x + w] = 255  # set rectangle as foreground
        y = y - 1
        i_img = i_img + 1
    X_train.append(imgs)
for i in range(0, num_imgs):
    Y_train.append(2)
# video frames with down moving object
for i_vid in range(num_vids):
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    # vid_name = 'vid' + str(i_vid) + '.mp4'
    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    i_img = 0
    while y < img_size:
        imgs[i_img, y : y + h, x : x + w] = 255  # set rectangle as foreground
        y = y + 1
        i_img = i_img + 1
    X_train.append(imgs)
for i in range(0, num_imgs):
    Y_train.append(3)

# data pre-processing
from tensorflow.keras.utils import to_categorical

X_train = np.array(X_train, dtype=np.float32) / 255
X_train = X_train.reshape(X_train.shape[0], num_imgs, img_size, img_size, 1)
print(X_train.shape)
Y_train = np.array(Y_train, dtype=np.uint8)
Y_train = Y_train.reshape(X_train.shape[0], 1)
print(Y_train.shape)
Y_train = to_categorical(Y_train, 4)

"""
## Model Building and Training

TimeDistributed layer is used to pass temporal information of videos to the network.
**No GPU is needed.** Training gets completed within few minutes.
"""

# building model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed

model = Sequential()
model.add(
    TimeDistributed(
        Conv2D(8, (3, 3), strides=(1, 1), activation="relu", padding="same"),
        input_shape=(num_imgs, img_size, img_size, 1),
    )
)
model.add(
    TimeDistributed(
        Conv2D(8, (3, 3), kernel_initializer="he_normal", activation="relu")
    )
)
model.add(TimeDistributed(MaxPooling2D((1, 1), strides=(1, 1))))
model.add(TimeDistributed(Flatten()))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False, dropout=0.3))
model.add(Dense(4, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
# model training
model.fit(X_train, Y_train, epochs=40, verbose=1)

"""
## Model Inferencing

Model is tested on new generated video data.
"""

# model testing with new data (4 videos)
X_test = []
Y_test = []
for i_vid in range(2):
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    i_img = 0
    while x < img_size:
        imgs[i_img, y : y + h, x : x + w] = 255  # set rectangle as foreground
        x = x + 1
        i_img = i_img + 1
    X_test.append(imgs)  # 2nd class - 'right'
for i_vid in range(2):
    imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size - w)
    y = np.random.randint(0, img_size - h)
    i_img = 0
    while y < img_size:
        imgs[i_img, y : y + h, x : x + w] = 255  # set rectangle as foreground
        y = y + 1
        i_img = i_img + 1
    X_test.append(imgs)  # 4th class - 'down'
X_test = np.array(X_test, dtype=np.float32) / 255
X_test = X_test.reshape(X_test.shape[0], num_imgs, img_size, img_size, 1)
pred = np.argmax(model.predict(X_test), axis=-1)
for i in range(len(X_test)):
    print(labels[pred[i]])

"""
Clearly, the model is examined well on this synthetic dataset.
"""
