# Supervised Contrastive Learning

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2020/11/30<br>
**Last modified:** 2026/08/04<br>
**Description:** Using supervised contrastive learning for image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/supervised-contrastive-learning.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/supervised-contrastive-learning.py)



---
## Introduction

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
(Prannay Khosla et al.) is a training methodology that outperforms
supervised training with crossentropy on classification tasks.

Essentially, training an image classification model with Supervised Contrastive
Learning is performed in two phases:

1. Training an encoder to learn to produce vector representations of input images such
that representations of images in the same class will be more similar compared to
representations of images in different classes.
2. Training a classifier on top of the frozen encoder.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras
from keras import layers
from keras import ops
```

---
## Prepare the data


```python
num_classes = 50
input_shape = (32, 32, 3)

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Display shapes of train and test datasets
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

```

<div class="k-default-codeblock">
```
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 1)
```
</div>

---
## Using image data augmentation


```python
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
    ]
)

# Setting the state of the normalization layer.
data_augmentation.layers[0].adapt(x_train)
```

---
## Build the encoder model

The encoder model takes the image as input and turns it into a 2048-dimensional
feature vector.


```python

def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model


encoder = create_encoder()
encoder.summary()

learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 10
dropout_rate = 0.5
temperature = 0.05
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "cifar10-encoder"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ sequential (<span style="color: #0087ff; text-decoration-color: #0087ff">Sequential</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">7</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ resnet50v2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │    <span style="color: #00af00; text-decoration-color: #00af00">23,564,800</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,564,807</span> (89.89 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,519,360</span> (89.72 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">45,447</span> (177.53 KB)
</pre>



---
## Build the classification model

The classification model adds a fully-connected layer on top of the encoder,
plus a softmax layer with the target classes.


```python

def create_classifier(encoder, trainable=True):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

```

---
## Experiment 1: Train the baseline classification model

In this experiment, a baseline classifier is trained as usual, i.e., the
encoder and the classifier parts are trained together as a single model
to minimize the crossentropy loss.


```python
encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()

history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "cifar10-classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ cifar10-encoder (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │    <span style="color: #00af00; text-decoration-color: #00af00">23,564,807</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │     <span style="color: #00af00; text-decoration-color: #00af00">1,049,088</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">5,130</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">24,619,025</span> (93.91 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">24,573,578</span> (93.74 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">45,447</span> (177.53 KB)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 457s 2s/step - loss: 1.8886 - sparse_categorical_accuracy: 0.3186

Epoch 2/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 427s 2s/step - loss: 1.4216 - sparse_categorical_accuracy: 0.4887

Epoch 3/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 451s 2s/step - loss: 1.2727 - sparse_categorical_accuracy: 0.5527

Epoch 4/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 474s 3s/step - loss: 1.1427 - sparse_categorical_accuracy: 0.6023

Epoch 5/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 420s 2s/step - loss: 1.0507 - sparse_categorical_accuracy: 0.6380

Epoch 6/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 420s 2s/step - loss: 0.9727 - sparse_categorical_accuracy: 0.6645

Epoch 7/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 434s 2s/step - loss: 0.9005 - sparse_categorical_accuracy: 0.6901

Epoch 8/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 420s 2s/step - loss: 0.8047 - sparse_categorical_accuracy: 0.7225

Epoch 9/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 415s 2s/step - loss: 0.7285 - sparse_categorical_accuracy: 0.7518

Epoch 10/10

313/313 ━━━━━━━━━━━━━━━━━━━━ 24s 69ms/step - loss: 1.4892 - sparse_categorical_accuracy: 0.6420

Test accuracy: 64.2%
```
</div>

---
## Experiment 2: Use supervised contrastive learning

In this experiment, the model is trained in two phases. In the first phase,
the encoder is pretrained to optimize the supervised contrastive loss,
described in [Prannay Khosla et al.](https://arxiv.org/abs/2004.11362).

In the second phase, the classifier is trained using the trained encoder with
its weights freezed; only the weights of fully-connected layers with the
softmax are optimized.

### 1. Supervised contrastive learning loss function


```python

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, labels, feature_vectors):
        feature_vectors = ops.normalize(feature_vectors, axis=1)

        logits = ops.divide(
            ops.matmul(feature_vectors, ops.transpose(feature_vectors)),
            self.temperature,
        )

        # Create a mask to find positive pairs (images of same class)
        labels = ops.cast(labels, "int32")
        labels = ops.reshape(labels, (-1, 1))
        mask = ops.cast(ops.equal(labels, ops.transpose(labels)), "float32")

        batch_size = ops.shape(logits)[0]
        logits_mask = 1.0 - ops.eye(batch_size)
        mask = mask * logits_mask

        logits_max = ops.max(logits, axis=1, keepdims=True)
        logits_exp = ops.exp(logits - logits_max) * logits_mask

        log_prob = (logits - logits_max) - ops.log(
            ops.sum(logits_exp, axis=1, keepdims=True) + 1e-8
        )

        mean_log_prob_pos = ops.sum(mask * log_prob, axis=1) / (
            ops.sum(mask, axis=1) + 1e-8
        )

        return ops.subtract(0.0, ops.mean(mean_log_prob_pos))


def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

```

### 2. Pretrain the encoder


```python
encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

history = encoder_with_projection_head.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=num_epochs,
)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "cifar-encoder_with_projection-head"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ cifar10-encoder (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │    <span style="color: #00af00; text-decoration-color: #00af00">23,564,807</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">262,272</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,827,079</span> (90.89 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,781,632</span> (90.72 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">45,447</span> (177.53 KB)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 429s 2s/step - loss: 5.3027

Epoch 2/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 412s 2s/step - loss: 5.0873

Epoch 3/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 414s 2s/step - loss: 4.9359

Epoch 4/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 423s 2s/step - loss: 4.8023

Epoch 5/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 417s 2s/step - loss: 4.6883

Epoch 6/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 404s 2s/step - loss: 4.5960

Epoch 7/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 407s 2s/step - loss: 4.5119

Epoch 8/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 416s 2s/step - loss: 4.4246

Epoch 9/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 443s 2s/step - loss: 4.3586

Epoch 10/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 458s 2s/step - loss: 4.3063
```
</div>

### 3. Train the classifier with the frozen encoder


```python
classifier = create_classifier(encoder, trainable=False)

history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
```

<div class="k-default-codeblock">
```
Epoch 1/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 85s 440ms/step - loss: 0.7454 - sparse_categorical_accuracy: 0.7694

Epoch 2/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 81s 432ms/step - loss: 0.6565 - sparse_categorical_accuracy: 0.7838

Epoch 3/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 80s 424ms/step - loss: 0.6488 - sparse_categorical_accuracy: 0.7876

Epoch 4/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 70s 370ms/step - loss: 0.6420 - sparse_categorical_accuracy: 0.7888

Epoch 5/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 67s 353ms/step - loss: 0.6415 - sparse_categorical_accuracy: 0.7866

Epoch 6/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 73s 389ms/step - loss: 0.6352 - sparse_categorical_accuracy: 0.7893

Epoch 7/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 74s 393ms/step - loss: 0.6352 - sparse_categorical_accuracy: 0.7887

Epoch 8/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 68s 359ms/step - loss: 0.6356 - sparse_categorical_accuracy: 0.7876

Epoch 9/10

189/189 ━━━━━━━━━━━━━━━━━━━━ 70s 371ms/step - loss: 0.6387 - sparse_categorical_accuracy: 0.7872

Epoch 10/10

313/313 ━━━━━━━━━━━━━━━━━━━━ 19s 60ms/step - loss: 0.7974 - sparse_categorical_accuracy: 0.7298

Test accuracy: 72.98%
```
</div>

We get to an improved test accuracy.

---
## Conclusion

As shown in the experiments, using the supervised contrastive learning technique
outperformed the conventional technique in terms of the test accuracy. Note that
the same training budget (i.e., number of epochs) was given to each technique.
Supervised contrastive learning pays off when the encoder involves a complex
architecture, like ResNet, and multi-class problems with many labels.
In addition, large batch sizes and multi-layer projection heads
improve its effectiveness. See the [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
paper for more details.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/supervised-contrastive-learning-cifar10)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/supervised-contrastive-learning).
