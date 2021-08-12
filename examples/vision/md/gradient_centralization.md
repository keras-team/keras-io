# Gradient Centralization for Better Training Performance

**Author:** [Rishit Dagli](https://github.com/Rishit-dagli)<br>
**Date created:** 06/18/21<br>
**Last modified:** 06/18/21<br>
**Description:** Implement Gradient Centralization to improve training performance of DNNs.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/gradient_centralization.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/gradient_centralization.py)



## Introduction

This example implements [Gradient Centralization](https://arxiv.org/abs/2004.01461), a
new optimization technique for Deep Neural Networks by Yong et al., and demonstrates it
on Laurence Moroney's [Horses or Humans
Dataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans). Gradient
Centralization can both speedup training process and improve the final generalization
performance of DNNs. It operates directly on gradients by centralizing the gradient
vectors to have zero mean. Gradient Centralization morever improves the Lipschitzness of
the loss function and its gradient so that the training process becomes more efficient
and stable.

This example requires TensorFlow 2.2 or higher as well as `tensorflow_datasets` which can
be installed with this command:

```
pip install tensorflow-datasets
```

We will be implementing Gradient Centralization in this example but you could also use
this very easily with a package I built,
[gradient-centralization-tf](https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow).

## Setup


```python
from time import time

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
```

## Prepare the data

For this example, we will be using the [Horses or Humans
dataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans).


```python
num_classes = 2
input_shape = (300, 300, 3)
dataset_name = "horses_or_humans"
batch_size = 128
AUTOTUNE = tf.data.AUTOTUNE

(train_ds, test_ds), metadata = tfds.load(
    name=dataset_name,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
    with_info=True,
    as_supervised=True,
)

print(f"Image shape: {metadata.features['image'].shape}")
print(f"Training images: {metadata.splits['train'].num_examples}")
print(f"Test images: {metadata.splits['test'].num_examples}")
```

<div class="k-default-codeblock">
```
Image shape: (300, 300, 3)
Training images: 1027
Test images: 256

```
</div>
## Use Data Augmentation

We will rescale the data to `[0, 1]` and perform simple augmentations to our data.


```python
rescale = layers.Rescaling(1.0 / 255)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
    ]
)


def prepare(ds, shuffle=False, augment=False):
    # Rescale dataset
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1024)

    # Batch dataset
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Use buffered prefecting
    return ds.prefetch(buffer_size=AUTOTUNE)

```

Rescale and augment the data


```python
train_ds = prepare(train_ds, shuffle=True, augment=True)
test_ds = prepare(test_ds)
```

## Define a model

In this section we will define a Convolutional neural network.


```python
model = tf.keras.Sequential(
    [
        layers.Conv2D(16, (3, 3), activation="relu", input_shape=(300, 300, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.Dropout(0.5),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Dropout(0.5),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
```

## Implement Gradient Centralization

We will now
subclass the `RMSProp` optimizer class modifying the
`tf.keras.optimizers.Optimizer.get_gradients()` method where we now implement Gradient
Centralization. On a high level the idea is that let us say we obtain our gradients
through back propogation for a Dense or Convolution layer we then compute the mean of the
column vectors of the weight matrix, and then remove the mean from each column vector.

The experiments in [this paper](https://arxiv.org/abs/2004.01461) on various
applications, including general image classification, fine-grained image classification,
detection and segmentation and Person ReID demonstrate that GC can consistently improve
the performance of DNN learning.

Also, for simplicity at the moment we are not implementing gradient cliiping functionality,
however this quite easy to implement.

At the moment we are just creating a subclass for the `RMSProp` optimizer
however you could easily reproduce this for any other optimizer or on a custom
optimizer in the same way. We will be using this class in the later section when
we train a model with Gradient Centralization.


```python

class GCRMSprop(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


optimizer = GCRMSprop(learning_rate=1e-4)
```

## Training utilities

We will also create a callback which allows us to easily measure the total training time
and the time taken for each epoch since we are interested in comparing the effect of
Gradient Centralization on the model we built above.


```python

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)

```

## Train the model without GC

We now train the model we built earlier without Gradient Centralization which we can
compare to the training performance of the model trained with Gradient Centralization.


```python
time_callback_no_gc = TimeHistory()
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=["accuracy"],
)

model.summary()
```

<div class="k-default-codeblock">
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 298, 298, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 149, 149, 16)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 147, 147, 32)      4640      
_________________________________________________________________
dropout (Dropout)            (None, 147, 147, 32)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 73, 73, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 71, 71, 64)        18496     
_________________________________________________________________
dropout_1 (Dropout)          (None, 71, 71, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 35, 35, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 33, 33, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 64)        36928     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 3136)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               1606144   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 1,704,097
Trainable params: 1,704,097
Non-trainable params: 0
_________________________________________________________________

```
</div>
We also save the history since we later want to compare our model trained with and not
trained with Gradient Centralization


```python
history_no_gc = model.fit(
    train_ds, epochs=10, verbose=1, callbacks=[time_callback_no_gc]
)
```

<div class="k-default-codeblock">
```
Epoch 1/10
9/9 [==============================] - 5s 571ms/step - loss: 0.7427 - accuracy: 0.5073
Epoch 2/10
9/9 [==============================] - 6s 667ms/step - loss: 0.6757 - accuracy: 0.5433
Epoch 3/10
9/9 [==============================] - 6s 660ms/step - loss: 0.6616 - accuracy: 0.6144
Epoch 4/10
9/9 [==============================] - 6s 642ms/step - loss: 0.6598 - accuracy: 0.6203
Epoch 5/10
9/9 [==============================] - 6s 666ms/step - loss: 0.6782 - accuracy: 0.6329
Epoch 6/10
9/9 [==============================] - 6s 655ms/step - loss: 0.6550 - accuracy: 0.6524
Epoch 7/10
9/9 [==============================] - 6s 645ms/step - loss: 0.6157 - accuracy: 0.7186
Epoch 8/10
9/9 [==============================] - 6s 654ms/step - loss: 0.6095 - accuracy: 0.6913
Epoch 9/10
9/9 [==============================] - 6s 677ms/step - loss: 0.5880 - accuracy: 0.7147
Epoch 10/10
9/9 [==============================] - 6s 663ms/step - loss: 0.5814 - accuracy: 0.6933

```
</div>
## Train the model with GC

We will now train the same model, this time using Gradient Centralization,
notice our optimizer is the one using Gradient Centralization this time.


```python
time_callback_gc = TimeHistory()
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.summary()

history_gc = model.fit(train_ds, epochs=10, verbose=1, callbacks=[time_callback_gc])
```

<div class="k-default-codeblock">
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 298, 298, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 149, 149, 16)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 147, 147, 32)      4640      
_________________________________________________________________
dropout (Dropout)            (None, 147, 147, 32)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 73, 73, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 71, 71, 64)        18496     
_________________________________________________________________
dropout_1 (Dropout)          (None, 71, 71, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 35, 35, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 33, 33, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 64)        36928     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 3136)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               1606144   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 1,704,097
Trainable params: 1,704,097
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
9/9 [==============================] - 6s 673ms/step - loss: 0.6022 - accuracy: 0.7147
Epoch 2/10
9/9 [==============================] - 6s 662ms/step - loss: 0.5385 - accuracy: 0.7371
Epoch 3/10
9/9 [==============================] - 6s 673ms/step - loss: 0.4832 - accuracy: 0.7945
Epoch 4/10
9/9 [==============================] - 6s 645ms/step - loss: 0.4692 - accuracy: 0.7799
Epoch 5/10
9/9 [==============================] - 6s 720ms/step - loss: 0.4792 - accuracy: 0.7799
Epoch 6/10
9/9 [==============================] - 6s 658ms/step - loss: 0.4623 - accuracy: 0.7838
Epoch 7/10
9/9 [==============================] - 6s 651ms/step - loss: 0.4413 - accuracy: 0.8072
Epoch 8/10
9/9 [==============================] - 6s 682ms/step - loss: 0.4542 - accuracy: 0.8014
Epoch 9/10
9/9 [==============================] - 6s 649ms/step - loss: 0.4235 - accuracy: 0.8053
Epoch 10/10
9/9 [==============================] - 6s 686ms/step - loss: 0.4445 - accuracy: 0.7936

```
</div>
## Comparing performance


```python
print("Not using Gradient Centralization")
print(f"Loss: {history_no_gc.history['loss'][-1]}")
print(f"Accuracy: {history_no_gc.history['accuracy'][-1]}")
print(f"Training Time: {sum(time_callback_no_gc.times)}")

print("Using Gradient Centralization")
print(f"Loss: {history_gc.history['loss'][-1]}")
print(f"Accuracy: {history_gc.history['accuracy'][-1]}")
print(f"Training Time: {sum(time_callback_gc.times)}")
```

<div class="k-default-codeblock">
```
Not using Gradient Centralization
Loss: 0.5814347863197327
Accuracy: 0.6932814121246338
Training Time: 136.35903406143188
Using Gradient Centralization
Loss: 0.4444807469844818
Accuracy: 0.7935734987258911
Training Time: 131.61780261993408

```
</div>
Readers are encouraged to try out Gradient Centralization on different datasets from
different domains and experiment with it's effect. You are strongly advised to check out
the [original paper](https://arxiv.org/abs/2004.01461) as well - the authors present
several studies on Gradient Centralization showing how it can improve general
performance, generalization, training time as well as more efficient.

Many thanks to [Ali Mustufa Shaikh](https://github.com/ialimustufa) for reviewing this
implementation.
