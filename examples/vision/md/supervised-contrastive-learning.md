# Supervised Contrastive Learning

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2020/11/01<br>
**Last modified:** 2020/11/01<br>
**Description:** Using supervised contrastive learning for image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/supervised-contrastive-learning.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/supervised-contrastive-learning.py)



---
## Introduction

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
(Prannay Khosla et al.) is a training methodology that outperforms cross-entropy
on supervised learning tasks.

Essentially, training an image classification model with Supervised Contrastive Learning
is peformed in two phases:

  1. Pre-training an encoder to generate feature vectors for input images such that feature
    vectors of images in the same class will be more similar compared feature vectors of
    images in other classes.
  2. Training a classifier on top of the freezed encoder.

---
## Setup


```python
!!pip install tensorflow-addons
```




```python
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
```
<div class="k-default-codeblock">
```
['Requirement already satisfied: tensorflow-addons in /Users/khalidsalama/Technology/python-venvs/py37-env/lib/python3.7/site-packages (0.11.2)',
 'Requirement already satisfied: typeguard>=2.7 in /Users/khalidsalama/Technology/python-venvs/py37-env/lib/python3.7/site-packages (from tensorflow-addons) (2.9.1)']

/Users/khalidsalama/Technology/python-venvs/py37-env/lib/python3.7/site-packages/tensorflow_addons/utils/ensure_tf_install.py:44: UserWarning: You are currently using a nightly version of TensorFlow (2.3.0-dev20200625). 
TensorFlow Addons offers no support for the nightly versions of TensorFlow. Some things might work, some other might not. 
If you encounter a bug, do not file an issue on GitHub.
  UserWarning,

```
</div>
---
## Prepare the data


```python
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train, y_test = tf.squeeze(y_train), tf.squeeze(y_test)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
```

<div class="k-default-codeblock">
```
x_train shape: (50000, 32, 32, 3)
y_train shape: (50000,)
50000 train samples
10000 test samples

```
</div>
---
## Build the encoder model

The encoder model takes the image as an input and produce a 128-dimension feature vector.


```python

def create_encoder():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
        ]
    )


encoder = create_encoder()
encoder.summary()

BATCH_SIZE = 256
NUM_EPOCHS = 50
DROPOUT = 0.5
TEMPERATURE = 0.05
```

<div class="k-default-codeblock">
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 2304)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               295040    
=================================================================
Total params: 314,432
Trainable params: 314,432
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Build the classification model

The classification model adds a fully-connected layer on top of the encoder, plus a
softmax layer with the target classes.


```python

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(DROPOUT)(features)
    features = tf.keras.layers.Dense(64)(features)
    features = tf.keras.layers.Dropout(DROPOUT)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
    )
    return model

```

---
## Experiment 1: Train the baseline classification model

In this experiment, a baseline classifier is trained normally, i.e., the encoder and the
classifier parts are trained together as a single model to minimize cross-entropy loss.


```python
encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()

history = classifier.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_test, y_test),
    verbose=0,
)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy*100, 2)}%")
```

<div class="k-default-codeblock">
```
Model: "cifar10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
sequential_1 (Sequential)    (None, 128)               314432    
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 323,338
Trainable params: 323,338
Non-trainable params: 0
_________________________________________________________________
313/313 [==============================] - 3s 8ms/step - loss: 0.9452 - sparse_categorical_accuracy: 0.7114
Test accuracy: 71.14%

```
</div>
We get to ~70.1% validation accuracy.

---
## Experiment 2: Use supervised contrastive learning

### 1. Supervised contrastive learning loss function


```python

def make_supervised_contrastive_loss_fn(temperature=1):
    def supervised_contrastive_loss(labels, feature_vectors):

        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)

        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            temperature,
        )

        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

    return supervised_contrastive_loss

```

### 2. Pretrain the encoder


```python
encoder = create_encoder()
encoder.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=make_supervised_contrastive_loss_fn(temperature=TEMPERATURE),
)

history = encoder.fit(
    x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=0
)
```

### 3. Train the classifier with the freezed encoder


```python
classifier = create_classifier(encoder, trainable=False)
history = classifier.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_test, y_test),
    verbose=0,
)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy*100, 2)}%")
```

<div class="k-default-codeblock">
```
313/313 [==============================] - 3s 8ms/step - loss: 0.8564 - sparse_categorical_accuracy: 0.7290
Test accuracy: 72.9%

```
</div>
We get to ~72.6% validation accuracy.
