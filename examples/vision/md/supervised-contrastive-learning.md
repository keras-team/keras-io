# Supervised Contrastive Learning

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2020/11/30<br>
**Last modified:** 2020/11/30<br>
**Description:** Using supervised contrastive learning for image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/supervised-contrastive-learning.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/supervised-contrastive-learning.py)




```python
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
```

---
## Prepare the data


```python
num_classes = 10
input_shape = (32, 32, 3)

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Display shapes of train and test datasets
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

```

<div class="k-default-codeblock">
```
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170500096/170498071 [==============================] - 4s 0us/step
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 1)

```
</div>
---
## Using image data augmentation


```python
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.02),
        layers.experimental.preprocessing.RandomWidth(0.2),
        layers.experimental.preprocessing.RandomHeight(0.2),
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
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05
```

<div class="k-default-codeblock">
```
Model: "cifar10-encoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
sequential (Sequential)      (None, None, None, 3)     7         
_________________________________________________________________
resnet50v2 (Functional)      (None, 2048)              23564800  
=================================================================
Total params: 23,564,807
Trainable params: 23,519,360
Non-trainable params: 45,447
_________________________________________________________________

```
</div>
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

<div class="k-default-codeblock">
```
Model: "cifar10-classifier"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
cifar10-encoder (Functional) (None, 2048)              23564807  
_________________________________________________________________
dropout (Dropout)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               1049088   
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
Total params: 24,619,025
Trainable params: 24,573,578
Non-trainable params: 45,447
_________________________________________________________________
Epoch 1/50
189/189 [==============================] - 15s 77ms/step - loss: 1.9582 - sparse_categorical_accuracy: 0.2736
Epoch 2/50
189/189 [==============================] - 11s 60ms/step - loss: 1.5386 - sparse_categorical_accuracy: 0.4399
Epoch 3/50
189/189 [==============================] - 11s 58ms/step - loss: 1.3785 - sparse_categorical_accuracy: 0.5044
Epoch 4/50
189/189 [==============================] - 10s 55ms/step - loss: 1.2459 - sparse_categorical_accuracy: 0.5592
Epoch 5/50
189/189 [==============================] - 10s 55ms/step - loss: 1.1631 - sparse_categorical_accuracy: 0.5937
Epoch 6/50
189/189 [==============================] - 11s 58ms/step - loss: 1.1134 - sparse_categorical_accuracy: 0.6138
Epoch 7/50
189/189 [==============================] - 10s 55ms/step - loss: 1.0793 - sparse_categorical_accuracy: 0.6288
Epoch 8/50
189/189 [==============================] - 11s 56ms/step - loss: 0.9989 - sparse_categorical_accuracy: 0.6558
Epoch 9/50
189/189 [==============================] - 10s 55ms/step - loss: 0.9810 - sparse_categorical_accuracy: 0.6651
Epoch 10/50
189/189 [==============================] - 11s 56ms/step - loss: 0.9146 - sparse_categorical_accuracy: 0.6871
Epoch 11/50
189/189 [==============================] - 10s 55ms/step - loss: 0.8835 - sparse_categorical_accuracy: 0.6992
Epoch 12/50
189/189 [==============================] - 10s 55ms/step - loss: 0.8430 - sparse_categorical_accuracy: 0.7138
Epoch 13/50
189/189 [==============================] - 10s 55ms/step - loss: 0.8098 - sparse_categorical_accuracy: 0.7248
Epoch 14/50
189/189 [==============================] - 11s 56ms/step - loss: 0.7614 - sparse_categorical_accuracy: 0.7400
Epoch 15/50
189/189 [==============================] - 10s 55ms/step - loss: 0.7144 - sparse_categorical_accuracy: 0.7562
Epoch 16/50
189/189 [==============================] - 10s 55ms/step - loss: 0.6786 - sparse_categorical_accuracy: 0.7679
Epoch 17/50
189/189 [==============================] - 10s 55ms/step - loss: 0.6562 - sparse_categorical_accuracy: 0.7752
Epoch 18/50
189/189 [==============================] - 10s 55ms/step - loss: 0.6967 - sparse_categorical_accuracy: 0.7636
Epoch 19/50
189/189 [==============================] - 10s 55ms/step - loss: 0.6395 - sparse_categorical_accuracy: 0.7813
Epoch 20/50
189/189 [==============================] - 10s 55ms/step - loss: 0.6318 - sparse_categorical_accuracy: 0.7856
Epoch 21/50
189/189 [==============================] - 10s 55ms/step - loss: 0.5887 - sparse_categorical_accuracy: 0.7989
Epoch 22/50
189/189 [==============================] - 10s 55ms/step - loss: 0.5704 - sparse_categorical_accuracy: 0.8072
Epoch 23/50
189/189 [==============================] - 10s 55ms/step - loss: 0.5553 - sparse_categorical_accuracy: 0.8116
Epoch 24/50
189/189 [==============================] - 10s 55ms/step - loss: 0.5307 - sparse_categorical_accuracy: 0.8197
Epoch 25/50
189/189 [==============================] - 10s 55ms/step - loss: 0.5268 - sparse_categorical_accuracy: 0.8203
Epoch 26/50
189/189 [==============================] - 10s 55ms/step - loss: 0.4986 - sparse_categorical_accuracy: 0.8286
Epoch 27/50
189/189 [==============================] - 10s 55ms/step - loss: 0.4858 - sparse_categorical_accuracy: 0.8342
Epoch 28/50
189/189 [==============================] - 11s 56ms/step - loss: 0.4785 - sparse_categorical_accuracy: 0.8388
Epoch 29/50
189/189 [==============================] - 10s 55ms/step - loss: 0.4746 - sparse_categorical_accuracy: 0.8378
Epoch 30/50
189/189 [==============================] - 11s 57ms/step - loss: 0.4576 - sparse_categorical_accuracy: 0.8434
Epoch 31/50
189/189 [==============================] - 11s 56ms/step - loss: 0.4482 - sparse_categorical_accuracy: 0.8487
Epoch 32/50
189/189 [==============================] - 11s 56ms/step - loss: 0.6295 - sparse_categorical_accuracy: 0.7878
Epoch 33/50
 56/189 [=======>......................] - ETA: 7s - loss: 0.5332 - sparse_categorical_accuracy: 0.8191

```
</div>
We get to ~78.4% test accuracy.

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
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
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
    x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs
)
```

<div class="k-default-codeblock">
```
Model: "cifar-encoder_with_projection-head"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_8 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
cifar10-encoder (Functional) (None, 2048)              23564807  
_________________________________________________________________
dense_2 (Dense)              (None, 128)               262272    
=================================================================
Total params: 23,827,079
Trainable params: 23,781,632
Non-trainable params: 45,447
_________________________________________________________________
Epoch 1/50
189/189 [==============================] - 11s 57ms/step - loss: 5.3993
Epoch 2/50
189/189 [==============================] - 11s 56ms/step - loss: 5.1661
Epoch 3/50
189/189 [==============================] - 11s 57ms/step - loss: 5.0481
Epoch 4/50
189/189 [==============================] - 11s 56ms/step - loss: 4.9533
Epoch 5/50
189/189 [==============================] - 11s 56ms/step - loss: 4.8610
Epoch 6/50
189/189 [==============================] - 11s 57ms/step - loss: 4.7823
Epoch 7/50
189/189 [==============================] - 11s 57ms/step - loss: 4.7080
Epoch 8/50
189/189 [==============================] - 11s 56ms/step - loss: 4.6416
Epoch 9/50
189/189 [==============================] - 10s 55ms/step - loss: 4.5999
Epoch 10/50
189/189 [==============================] - 11s 56ms/step - loss: 4.5428
Epoch 11/50
189/189 [==============================] - 11s 57ms/step - loss: 4.4942
Epoch 12/50
189/189 [==============================] - 11s 58ms/step - loss: 4.4513
Epoch 13/50
189/189 [==============================] - 11s 57ms/step - loss: 4.4150
Epoch 14/50
189/189 [==============================] - 11s 57ms/step - loss: 4.3864
Epoch 15/50
189/189 [==============================] - 11s 56ms/step - loss: 4.3518
Epoch 16/50
189/189 [==============================] - 10s 56ms/step - loss: 4.3187
Epoch 17/50
189/189 [==============================] - 11s 56ms/step - loss: 4.2892
Epoch 18/50
189/189 [==============================] - 11s 57ms/step - loss: 4.2742
Epoch 19/50
189/189 [==============================] - 11s 56ms/step - loss: 4.2313
Epoch 20/50
189/189 [==============================] - 11s 56ms/step - loss: 4.2144
Epoch 21/50
189/189 [==============================] - 11s 56ms/step - loss: 4.1948
Epoch 22/50
189/189 [==============================] - 10s 55ms/step - loss: 4.1801
Epoch 23/50
189/189 [==============================] - 11s 56ms/step - loss: 4.1543
Epoch 24/50
189/189 [==============================] - 11s 56ms/step - loss: 4.1318
Epoch 25/50
189/189 [==============================] - 11s 56ms/step - loss: 4.1143
Epoch 26/50
189/189 [==============================] - 11s 56ms/step - loss: 4.0856
Epoch 27/50
189/189 [==============================] - 11s 57ms/step - loss: 4.0673
Epoch 28/50
189/189 [==============================] - 11s 57ms/step - loss: 4.0460
Epoch 29/50
189/189 [==============================] - 11s 56ms/step - loss: 4.0404
Epoch 30/50
189/189 [==============================] - 11s 56ms/step - loss: 4.0239
Epoch 31/50
189/189 [==============================] - 11s 56ms/step - loss: 4.0206
Epoch 32/50
189/189 [==============================] - 11s 56ms/step - loss: 3.9985
Epoch 33/50
189/189 [==============================] - 10s 55ms/step - loss: 4.0011
Epoch 34/50
189/189 [==============================] - 11s 56ms/step - loss: 3.9625
Epoch 35/50
189/189 [==============================] - 11s 56ms/step - loss: 3.9536
Epoch 36/50
189/189 [==============================] - 10s 55ms/step - loss: 3.9312
Epoch 37/50
189/189 [==============================] - 11s 56ms/step - loss: 3.9236
Epoch 38/50
189/189 [==============================] - 11s 56ms/step - loss: 3.9134
Epoch 39/50
189/189 [==============================] - 11s 56ms/step - loss: 3.8904
Epoch 40/50
189/189 [==============================] - 11s 56ms/step - loss: 3.8784
Epoch 41/50
189/189 [==============================] - 11s 56ms/step - loss: 3.8747
Epoch 42/50
189/189 [==============================] - 11s 56ms/step - loss: 3.8574
Epoch 43/50
189/189 [==============================] - 11s 56ms/step - loss: 3.8469
Epoch 44/50
189/189 [==============================] - 10s 55ms/step - loss: 3.8524
Epoch 45/50
189/189 [==============================] - 10s 55ms/step - loss: 3.8271
Epoch 46/50
189/189 [==============================] - 10s 55ms/step - loss: 3.8322
Epoch 47/50
189/189 [==============================] - 11s 56ms/step - loss: 3.8068
Epoch 48/50
189/189 [==============================] - 11s 56ms/step - loss: 3.7951
Epoch 49/50
189/189 [==============================] - 11s 56ms/step - loss: 3.7817
Epoch 50/50
189/189 [==============================] - 11s 56ms/step - loss: 3.7809

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
Epoch 1/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3911 - sparse_categorical_accuracy: 0.8938
Epoch 2/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3161 - sparse_categorical_accuracy: 0.9056
Epoch 3/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3101 - sparse_categorical_accuracy: 0.9041
Epoch 4/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3107 - sparse_categorical_accuracy: 0.9032
Epoch 5/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3048 - sparse_categorical_accuracy: 0.9052
Epoch 6/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2933 - sparse_categorical_accuracy: 0.9087
Epoch 7/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3023 - sparse_categorical_accuracy: 0.9054
Epoch 8/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3059 - sparse_categorical_accuracy: 0.9044
Epoch 9/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3071 - sparse_categorical_accuracy: 0.9039
Epoch 10/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2959 - sparse_categorical_accuracy: 0.9057
Epoch 11/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3082 - sparse_categorical_accuracy: 0.9040
Epoch 12/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3033 - sparse_categorical_accuracy: 0.9039
Epoch 13/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2948 - sparse_categorical_accuracy: 0.9082
Epoch 14/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3045 - sparse_categorical_accuracy: 0.9035
Epoch 15/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3042 - sparse_categorical_accuracy: 0.9039
Epoch 16/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2925 - sparse_categorical_accuracy: 0.9092
Epoch 17/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2988 - sparse_categorical_accuracy: 0.9062
Epoch 18/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2979 - sparse_categorical_accuracy: 0.9053
Epoch 19/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3024 - sparse_categorical_accuracy: 0.9033
Epoch 20/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2903 - sparse_categorical_accuracy: 0.9076
Epoch 21/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2986 - sparse_categorical_accuracy: 0.9069
Epoch 22/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3073 - sparse_categorical_accuracy: 0.9036
Epoch 23/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2969 - sparse_categorical_accuracy: 0.9053
Epoch 24/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2905 - sparse_categorical_accuracy: 0.9091
Epoch 25/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2974 - sparse_categorical_accuracy: 0.9063
Epoch 26/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3028 - sparse_categorical_accuracy: 0.9044
Epoch 27/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3006 - sparse_categorical_accuracy: 0.9052
Epoch 28/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3118 - sparse_categorical_accuracy: 0.9019
Epoch 29/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3048 - sparse_categorical_accuracy: 0.9042
Epoch 30/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2906 - sparse_categorical_accuracy: 0.9071
Epoch 31/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2970 - sparse_categorical_accuracy: 0.9061
Epoch 32/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3000 - sparse_categorical_accuracy: 0.9043
Epoch 33/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2961 - sparse_categorical_accuracy: 0.9067
Epoch 34/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3003 - sparse_categorical_accuracy: 0.9060
Epoch 35/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2997 - sparse_categorical_accuracy: 0.9053
Epoch 36/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3049 - sparse_categorical_accuracy: 0.9044
Epoch 37/50
189/189 [==============================] - 3s 16ms/step - loss: 0.3036 - sparse_categorical_accuracy: 0.9041
Epoch 38/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2923 - sparse_categorical_accuracy: 0.9075
Epoch 39/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2925 - sparse_categorical_accuracy: 0.9079
Epoch 40/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2981 - sparse_categorical_accuracy: 0.9048
Epoch 41/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2949 - sparse_categorical_accuracy: 0.9075
Epoch 42/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2929 - sparse_categorical_accuracy: 0.9071
Epoch 43/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2854 - sparse_categorical_accuracy: 0.9111
Epoch 44/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2957 - sparse_categorical_accuracy: 0.9051
Epoch 45/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2866 - sparse_categorical_accuracy: 0.9102
Epoch 46/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2823 - sparse_categorical_accuracy: 0.9094
Epoch 47/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2954 - sparse_categorical_accuracy: 0.9064
Epoch 48/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2909 - sparse_categorical_accuracy: 0.9082
Epoch 49/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2961 - sparse_categorical_accuracy: 0.9065
Epoch 50/50
189/189 [==============================] - 3s 16ms/step - loss: 0.2955 - sparse_categorical_accuracy: 0.9059
313/313 [==============================] - 2s 7ms/step - loss: 0.7674 - sparse_categorical_accuracy: 0.8077
Test accuracy: 80.77%

```
</div>
We get to ~82.6% test accuracy.

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
