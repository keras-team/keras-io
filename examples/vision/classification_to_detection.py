"""
Title: Extending Image Classification model with Object detection
Author: [Karan Sharma]()
Date created: 20-12-2020
Last modified: 20-12-2020
Description: Sliding window implemention through convolution.
"""
"""
## Introduction: Classification and Localization



Localization is detecting the object in the classified image. The basic idea to build
such type of model is through training a Image classification model and then
convolutionly implement the sliding window object detection. Convoultion implemention is
much more fast than tradition sliding window object detection because it predict the
output of all window in just one inference. It was first introduced in
[OverFeat](https://arxiv.org/abs/1312.6229) research paper.




"""

"""
## Setup



This example uses TensorFlow 2.3.0


"""

import tensorflow as tf


"""
## Loading Dataset



Dataset is loaded through image_dataset_from_directory. Animal dataset is used for this
example from
[Kaggle](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda)
contain three classes which are dog, cat and panda.


"""

data_dir = "E:\\data"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.1,
    subset="training",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.1,
    subset="validation",
)

"""
## Visualizaing the dataset



The following code is to show the 9 random images.
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(train_ds.class_names[labels[i]])

        plt.axis("off")

"""
## Building and training Classification Model



We will use modified MobileNetV2 model for this example. Top layer is excluded which
removes the Dense layer from MobileNet model. Add a convolution layer which gives the
output in 4D array of size Batch_sizex1x1x3 where 3 is number of class.

We have to Reshape it into Batch_Sizex1x3 to make it compatible with loss function.
"""

model = tf.keras.Sequential(
    [
        tf.keras.applications.MobileNetV2(
            input_shape=(256, 256, 3), include_top=False, weights="imagenet"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=3, kernel_size=8, activation="softmax"),
        tf.keras.layers.Reshape((1, 3)),
        tf.keras.layers.Activation("softmax"),
    ]
)


model.summary()


# decay learning rate

initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=50, decay_rate=0.96, staircase=False
)


# loss function


def loss(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)


# compiling

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
    loss=loss,
    metrics="accuracy",
)


# training

import os

checkpoint_dir = data_dir

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True, verbose=1, save_best_only=True
)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=15, callbacks=[checkpoint_callback]
)

"""
## Implementing Sliding window through Convolution



We now make a model with same parameters as previous model but for bigger input images
and then transfer the weights of previously trained model to this model. <br />






"""

final_model = tf.keras.Sequential(
    [
        tf.keras.applications.MobileNetV2(
            input_shape=(640, 640, 3), include_top=False, weights="imagenet"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=3, kernel_size=8, activation="softmax"),
    ]
)

final_model.summary()


# tranfer weights

final_model.set_weights(model.get_weights())


"""
## Visualizing result

Now as an example we pass a image of cat and dog to final_model. We get an output of
(13,13,3). First 2 dimension tell about the position of all the 169 windows which is of
size (256,256). Last dimesion is for probability of different classes. </br>

Following code takes the output and find the classes of each window using 0.9 threshold.
-1 class if nothing is predicted. 0 is for cat and 1 is for dog.</br>

You can see that bottom left portion in output class matrix predicted class number 0
which is cat and on the right side class number 1 is predicted which is dog  same as the
input image.
"""

from matplotlib import image

from matplotlib import pyplot

from PIL import Image

import numpy as np

img = Image.open(
    "/content/drive/MyDrive/cat-and-dog.jpg"
)  # image extension *.png,*.jpg

new_width = 640

new_height = 640

img = img.resize((new_width, new_height), Image.ANTIALIAS)

img = np.asarray(img)

pyplot.imshow(img)

pyplot.show()

img = np.expand_dims(img, axis=0)

result = final_model.predict(img)

one = np.zeros((13, 13))

for i in range(13):

    for j in range(13):

        if np.max(result[0][i][j] > 0.9):

            m = np.where(result[0][i][j] == np.max(result[0][i][j]))

            one[i][j] = m[0]

        else:

            one[i][j] = -1

print(result.shape)

print(one)
