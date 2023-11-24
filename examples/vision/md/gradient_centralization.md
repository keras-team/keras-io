# Gradient Centralization for Better Training Performance

**Author:** [Rishit Dagli](https://github.com/Rishit-dagli)<br>
**Date created:** 06/18/21<br>
**Last modified:** 07/25/23<br>
**Description:** Implement Gradient Centralization to improve training performance of DNNs.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/gradient_centralization.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/gradient_centralization.py)



---
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

This example requires `tensorflow_datasets` which can be installed with this command:

```
pip install tensorflow-datasets
```

---
## Setup


```python
from time import time

import keras
from keras import layers
from keras.optimizers import RMSprop
from keras import ops

from tensorflow import data as tf_data
import tensorflow_datasets as tfds

```

---
## Prepare the data

For this example, we will be using the [Horses or Humans
dataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans).


```python
num_classes = 2
input_shape = (300, 300, 3)
dataset_name = "horses_or_humans"
batch_size = 128
AUTOTUNE = tf_data.AUTOTUNE

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
---
## Use Data Augmentation

We will rescale the data to `[0, 1]` and perform simple augmentations to our data.


```python
rescale = layers.Rescaling(1.0 / 255)

data_augmentation = [
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
]


# Helper to apply augmentation
def apply_aug(x):
    for aug in data_augmentation:
        x = aug(x)
    return x


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
            lambda x, y: (apply_aug(x), y),
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

---
## Define a model

In this section we will define a Convolutional neural network.


```python
model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu"),
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

---
## Implement Gradient Centralization

We will now
subclass the `RMSProp` optimizer class modifying the
`keras.optimizers.Optimizer.get_gradients()` method where we now implement Gradient
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
                grad -= ops.mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


optimizer = GCRMSprop(learning_rate=1e-4)
```

---
## Training utilities

We will also create a callback which allows us to easily measure the total training time
and the time taken for each epoch since we are interested in comparing the effect of
Gradient Centralization on the model we built above.


```python

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)

```

---
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


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">298</span>, <span style="color: #00af00; text-decoration-color: #00af00">298</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)      │        <span style="color: #00af00; text-decoration-color: #00af00">448</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">149</span>, <span style="color: #00af00; text-decoration-color: #00af00">149</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)      │      <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">33</span>, <span style="color: #00af00; text-decoration-color: #00af00">33</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3136</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3136</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)               │  <span style="color: #00af00; text-decoration-color: #00af00">1,606,144</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                 │        <span style="color: #00af00; text-decoration-color: #00af00">513</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,704,097</span> (6.50 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,704,097</span> (6.50 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



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
 9/9 ━━━━━━━━━━━━━━━━━━━━ 24s 778ms/step - accuracy: 0.4772 - loss: 0.7405
Epoch 2/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 597ms/step - accuracy: 0.5434 - loss: 0.6861
Epoch 3/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 700ms/step - accuracy: 0.5402 - loss: 0.6911
Epoch 4/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 9s 586ms/step - accuracy: 0.5884 - loss: 0.6788
Epoch 5/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 9s 588ms/step - accuracy: 0.6570 - loss: 0.6564
Epoch 6/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 591ms/step - accuracy: 0.6671 - loss: 0.6395
Epoch 7/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 594ms/step - accuracy: 0.7010 - loss: 0.6161
Epoch 8/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 9s 593ms/step - accuracy: 0.6946 - loss: 0.6129
Epoch 9/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 699ms/step - accuracy: 0.6972 - loss: 0.5987
Epoch 10/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 11s 623ms/step - accuracy: 0.6839 - loss: 0.6197

```
</div>
---
## Train the model with GC

We will now train the same model, this time using Gradient Centralization,
notice our optimizer is the one using Gradient Centralization this time.


```python
time_callback_gc = TimeHistory()
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.summary()

history_gc = model.fit(train_ds, epochs=10, verbose=1, callbacks=[time_callback_gc])
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">298</span>, <span style="color: #00af00; text-decoration-color: #00af00">298</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)      │        <span style="color: #00af00; text-decoration-color: #00af00">448</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">149</span>, <span style="color: #00af00; text-decoration-color: #00af00">149</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)      │      <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">33</span>, <span style="color: #00af00; text-decoration-color: #00af00">33</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3136</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3136</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)               │  <span style="color: #00af00; text-decoration-color: #00af00">1,606,144</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                 │        <span style="color: #00af00; text-decoration-color: #00af00">513</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,704,097</span> (6.50 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,704,097</span> (6.50 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 12s 649ms/step - accuracy: 0.7118 - loss: 0.5594
Epoch 2/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 592ms/step - accuracy: 0.7249 - loss: 0.5817
Epoch 3/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 9s 587ms/step - accuracy: 0.8060 - loss: 0.4448
Epoch 4/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 693ms/step - accuracy: 0.8472 - loss: 0.4051
Epoch 5/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 594ms/step - accuracy: 0.8386 - loss: 0.3978
Epoch 6/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 593ms/step - accuracy: 0.8442 - loss: 0.3976
Epoch 7/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 9s 585ms/step - accuracy: 0.7409 - loss: 0.6626
Epoch 8/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 587ms/step - accuracy: 0.8191 - loss: 0.4357
Epoch 9/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 9s 587ms/step - accuracy: 0.8248 - loss: 0.3974
Epoch 10/10
 9/9 ━━━━━━━━━━━━━━━━━━━━ 10s 646ms/step - accuracy: 0.8022 - loss: 0.4589

```
</div>
---
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
Loss: 0.5345584154129028
Accuracy: 0.7604166865348816
Training Time: 112.48799777030945
Using Gradient Centralization
Loss: 0.4014038145542145
Accuracy: 0.8153935074806213
Training Time: 98.31573963165283

```
</div>
Readers are encouraged to try out Gradient Centralization on different datasets from
different domains and experiment with it's effect. You are strongly advised to check out
the [original paper](https://arxiv.org/abs/2004.01461) as well - the authors present
several studies on Gradient Centralization showing how it can improve general
performance, generalization, training time as well as more efficient.

Many thanks to [Ali Mustufa Shaikh](https://github.com/ialimustufa) for reviewing this
implementation.
