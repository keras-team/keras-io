"""
Title: Asynchronous ensemble with sharing of the tensorflow session.
Author: [PierrickPochelu](https://github.com/PierrickPochelu)
Date created: 2022/10/17
Last modified: 2022/10/17
Description: Asynchronous ensemble with sharing of the tensorflow session.
"""

"""
## Introduction
Asynchronous ensemble based on multiple tensorflow sessions and the built-in multiprocessing package.  The ensemble of N models is generally more accurate than base models. The asynchronous ensemble exploites more efficiently massively-parallel devices than iterating on the computing of base models.
"""

"""
### Reading data
"""
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Input, Model, optimizers, layers
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)
import numpy as np
import time
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

NB_SAMPLES = 6000  # Sampling for faster experiences
train_images = train_images[:NB_SAMPLES]
train_labels = train_labels[:NB_SAMPLES]
test_images = test_images[:NB_SAMPLES]
test_labels = test_labels[:NB_SAMPLES]

"""
### Settings
"""

ENSEMBLE_SIZE = 4
NB_EPOCHS = 1
GPUIDS = [-1, -1, -1, -1]  # asignement of models with devices
BATCH_SIZE = 16


def keras_model(x):
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(10, activation="softmax")(x)
    x = layers.Softmax()(x)
    return x


"""
## Asynchronous ensemble design
"""


def evaluate_model(model, x, y):
    start_time = time.time()
    y_pred = model.predict(x, batch_size=BATCH_SIZE)
    enlapsed = time.time() - start_time
    acc = np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))
    return {"accuracy": round(acc, 2), "time": round(enlapsed, 2)}


def from_gpu_id_to_device_name(gpuid):
    if gpuid == -1:
        return "/device:CPU:0"
    else:
        return "/device:GPU:" + str(gpuid)


class ensemble:
    def __init__(self, ensemble_size, gpus):
        self.loss = "categorical_crossentropy"
        self.opt = "adam"

        self.model_list = []
        output_list = []

        with tf.device(from_gpu_id_to_device_name(gpus[0])):
            input = Input(shape=(32, 32, 3))

        for i in range(ensemble_size):
            with tf.device(from_gpu_id_to_device_name(gpus[i])):
                input_i = tf.identity(input)
                output_i = keras_model(input_i)
                model_i = Model(inputs=input_i, outputs=output_i)
                output_list.append(output_i)
                self.model_list.append(model_i)

        with tf.device(from_gpu_id_to_device_name(gpus[0])):
            merge = tf.stack(output_list, axis=-1)
            combined_predictions = tf.reduce_mean(merge, axis=-1)

        self.ensemble = Model(inputs=input, outputs=combined_predictions)
        self.ensemble.compile(loss=self.loss, optimizer=self.opt)

    def fit(self, train_images, train_labels):
        for model_i in self.model_list:
            model_i.compile(loss=self.loss, optimizer=self.opt)
            model_i.fit(
                x=train_images, y=train_labels, batch_size=BATCH_SIZE, epochs=NB_EPOCHS
            )

    def predict(self, x, batch_size):
        return self.ensemble.predict(x, batch_size=batch_size)


"""
## Training of models
"""
ensemble = ensemble(ensemble_size=ENSEMBLE_SIZE, gpus=GPUIDS)
ensemble.fit(train_images, train_labels)

"""
## Inference
### Models evaluation
"""
for i, base_model in enumerate(ensemble.model_list):
    info = evaluate_model(base_model, test_images, test_labels)
    print(f"Model id: {i} accuracy: {info['accuracy']} time: {info['time']}")

"""
### Ensemble evaluation
"""
info = evaluate_model(ensemble, test_images, test_labels)
print(f"Ensemble accuracy: {info['accuracy']} inference time: {info['time']}")

"""
Conclusion: the asynchronous ensemble exploits well the underlying parallelism. The ensemble of N models is faster than N*T models, with T the average computing time of one model.
"""

