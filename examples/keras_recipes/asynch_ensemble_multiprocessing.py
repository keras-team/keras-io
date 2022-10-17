"""
Title: Asynchronous ensemble based on multiprocessing package.
Author: [PierrickPochelu](https://github.com/PierrickPochelu)
Date created: 2022/10/17
Last modified: 2022/10/17
Description: Asynchronous ensemble based on multiprocessing package.
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
GPUID = [-1, -1, -1, -1]  #  # asignement of models with devices
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
from multiprocessing import Queue, Process, Barrier
import os


def evaluate_model(model, x, y):
    start_time = time.time()
    y_pred = model.predict(x, batch_size=BATCH_SIZE)
    enlapsed = time.time() - start_time
    acc = np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))
    return {"accuracy": round(acc, 2), "time": round(enlapsed, 2)}


def keras_model_builder(config):
    input_shape = (32, 32, 3)
    loss = "categorical_crossentropy"
    opt = "adam"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpuid"])
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    input = Input(shape=input_shape)
    output = keras_model(input)
    model = Model(inputs=input, outputs=output)
    model.compile(loss=loss, optimizer=opt)
    return model


class MyProcess(Process):
    def __init__(
        self,
        rank: int,
        config: dict,
        dataset: list,
        shared_input_queue: Queue,
        shared_output_queue: Queue,
    ):
        Process.__init__(self, name="ModelProcessor")
        self.rank = rank
        self.config = config
        self.dataset = dataset  # List of np.ndarray xtrain, ytrain, xtest, ytest
        self.shared_input_queue = shared_input_queue  # 'go' or 'stop'
        self.shared_output_queue = shared_output_queue  # "initok" or predictions
        self.model = None
        self.info = None

    def _asynchronous_predict(self):
        finish = False
        while finish == False:
            msg = self.shared_input_queue.get()  # wait
            if msg == "go":
                out = self.model.predict(self.dataset[2], batch_size=BATCH_SIZE)
                self.shared_output_queue.put((self.rank, out))
            else:
                finish = True

    def train(self):
        self.model.fit(
            x=self.dataset[0],
            y=self.dataset[1],
            batch_size=BATCH_SIZE,
            epochs=NB_EPOCHS,
        )

    def run(self):
        self.model = keras_model_builder(self.config)
        self.train()
        info = evaluate_model(self.model, x=self.dataset[2], y=self.dataset[3])
        self.shared_output_queue.put((self.rank, info))  # notify the main process

        self._asynchronous_predict()  # run forever


input_queue = Queue()
output_queue = Queue()
processes = []

"""
## Training of models
### Build the processes
"""
for i in range(ENSEMBLE_SIZE):
    proc = MyProcess(
        rank=i,
        config={"gpuid": GPUID[i]},
        dataset=[train_images, train_labels, test_images, test_labels],
        shared_input_queue=input_queue,
        shared_output_queue=output_queue,
    )
    proc.start()  # start and wait
    processes.append(proc)
print("The building/training is launched ...")

"""
### Wait every processes is ready
"""
training_start_time = time.time()
for i in range(ENSEMBLE_SIZE):
    thread_id, msg = output_queue.get()
    if isinstance(msg, dict):
        print(
            f"Model rank: {thread_id} accuracy: {msg['accuracy']} time: {msg['time']}"
        )
    else:
        raise ValueError(f"thread {thread_id} received an unexpected message")


"""
## Inference
"""

"""
### Ensemble evaluation
"""
inference_start_time = time.time()
for process in processes:
    input_queue.put("go")

preds = np.zeros(test_labels.shape, np.float32)
for process in processes:
    thread_id, msg = output_queue.get()
    preds += msg
acc = np.mean(np.argmax(test_labels, axis=1) == np.argmax(preds, axis=1))
inf_time = time.time() - inference_start_time
print(f"Ensemble accuracy: {round(acc,2)} inference time: {round(inf_time,2)}")


"""
Conclusion: the asynchronous ensemble exploits well the underlying parallelism. The ensemble of N models is faster than N*T models, with T the average computing time of one model.
"""

"""
### Stop processes
"""
for i in range(ENSEMBLE_SIZE):
    input_queue.put("stop")
