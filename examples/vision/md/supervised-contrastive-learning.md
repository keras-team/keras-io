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
import keras
from keras import layers
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
189/189 [==============================] - 1335s 7s/step - loss: 1.9680 - sparse_categorical_accuracy: 0.2729
Epoch 2/50
189/189 [==============================] - 1333s 7s/step - loss: 1.5906 - sparse_categorical_accuracy: 0.4192
Epoch 3/50
189/189 [==============================] - 1358s 7s/step - loss: 1.4133 - sparse_categorical_accuracy: 0.4895
Epoch 4/50
189/189 [==============================] - 1343s 7s/step - loss: 1.2916 - sparse_categorical_accuracy: 0.5405
Epoch 5/50
189/189 [==============================] - 1328s 7s/step - loss: 1.1770 - sparse_categorical_accuracy: 0.5810
Epoch 6/50
189/189 [==============================] - 1349s 7s/step - loss: 1.1033 - sparse_categorical_accuracy: 0.6141
Epoch 7/50
189/189 [==============================] - 1326s 7s/step - loss: 1.0895 - sparse_categorical_accuracy: 0.6227
Epoch 8/50
189/189 [==============================] - 1325s 7s/step - loss: 1.0722 - sparse_categorical_accuracy: 0.6308
Epoch 9/50
189/189 [==============================] - 1308s 7s/step - loss: 0.9488 - sparse_categorical_accuracy: 0.6746
Epoch 10/50
189/189 [==============================] - 1315s 7s/step - loss: 0.8857 - sparse_categorical_accuracy: 0.6947
Epoch 11/50
189/189 [==============================] - 1334s 7s/step - loss: 0.8299 - sparse_categorical_accuracy: 0.7138
Epoch 12/50
189/189 [==============================] - 1329s 7s/step - loss: 0.8247 - sparse_categorical_accuracy: 0.7160
Epoch 13/50
189/189 [==============================] - 1328s 7s/step - loss: 0.7980 - sparse_categorical_accuracy: 0.7265
Epoch 14/50
189/189 [==============================] - 1345s 7s/step - loss: 0.7387 - sparse_categorical_accuracy: 0.7486
Epoch 15/50
189/189 [==============================] - 1314s 7s/step - loss: 0.7208 - sparse_categorical_accuracy: 0.7538
Epoch 16/50
189/189 [==============================] - 1339s 7s/step - loss: 0.7193 - sparse_categorical_accuracy: 0.7545
Epoch 17/50
189/189 [==============================] - 1296s 7s/step - loss: 0.6845 - sparse_categorical_accuracy: 0.7667
Epoch 18/50
189/189 [==============================] - 1333s 7s/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7846
Epoch 19/50
189/189 [==============================] - 1299s 7s/step - loss: 0.6095 - sparse_categorical_accuracy: 0.7928
Epoch 20/50
189/189 [==============================] - 1326s 7s/step - loss: 0.5951 - sparse_categorical_accuracy: 0.7955
Epoch 21/50
189/189 [==============================] - 1331s 7s/step - loss: 0.5798 - sparse_categorical_accuracy: 0.8036
Epoch 22/50
189/189 [==============================] - 1335s 7s/step - loss: 0.5576 - sparse_categorical_accuracy: 0.8116
Epoch 23/50
189/189 [==============================] - 1306s 7s/step - loss: 0.7917 - sparse_categorical_accuracy: 0.7409
Epoch 24/50
189/189 [==============================] - 1317s 7s/step - loss: 0.7753 - sparse_categorical_accuracy: 0.7386
Epoch 25/50
189/189 [==============================] - 1306s 7s/step - loss: 0.5985 - sparse_categorical_accuracy: 0.7987
Epoch 26/50
189/189 [==============================] - 1360s 7s/step - loss: 0.5433 - sparse_categorical_accuracy: 0.8137
Epoch 27/50
189/189 [==============================] - 1317s 7s/step - loss: 0.5240 - sparse_categorical_accuracy: 0.8253
Epoch 28/50
189/189 [==============================] - 1322s 7s/step - loss: 0.5090 - sparse_categorical_accuracy: 0.8268
Epoch 29/50
189/189 [==============================] - 1354s 7s/step - loss: 0.4879 - sparse_categorical_accuracy: 0.8330
Epoch 30/50
189/189 [==============================] - 1332s 7s/step - loss: 0.4695 - sparse_categorical_accuracy: 0.8406
Epoch 31/50
189/189 [==============================] - 1322s 7s/step - loss: 0.4551 - sparse_categorical_accuracy: 0.8455
Epoch 32/50
189/189 [==============================] - 1327s 7s/step - loss: 0.4522 - sparse_categorical_accuracy: 0.8455
Epoch 33/50
189/189 [==============================] - 1343s 7s/step - loss: 0.4364 - sparse_categorical_accuracy: 0.8539
Epoch 34/50
189/189 [==============================] - 1339s 7s/step - loss: 0.4181 - sparse_categorical_accuracy: 0.8573
Epoch 35/50
189/189 [==============================] - 1356s 7s/step - loss: 0.4107 - sparse_categorical_accuracy: 0.8605
Epoch 36/50
189/189 [==============================] - 1322s 7s/step - loss: 0.6583 - sparse_categorical_accuracy: 0.7803
Epoch 37/50
189/189 [==============================] - 1293s 7s/step - loss: 0.4956 - sparse_categorical_accuracy: 0.8298
Epoch 38/50
189/189 [==============================] - 1314s 7s/step - loss: 0.4282 - sparse_categorical_accuracy: 0.8535
Epoch 39/50
189/189 [==============================] - 1276s 7s/step - loss: 0.3992 - sparse_categorical_accuracy: 0.8639
Epoch 40/50
189/189 [==============================] - 1331s 7s/step - loss: 0.5332 - sparse_categorical_accuracy: 0.8227
Epoch 41/50
189/189 [==============================] - 1383s 7s/step - loss: 0.4282 - sparse_categorical_accuracy: 0.8542
Epoch 42/50
189/189 [==============================] - 1342s 7s/step - loss: 0.3810 - sparse_categorical_accuracy: 0.8690
Epoch 43/50
189/189 [==============================] - 1291s 7s/step - loss: 0.3630 - sparse_categorical_accuracy: 0.8744
Epoch 44/50
189/189 [==============================] - 1298s 7s/step - loss: 0.3466 - sparse_categorical_accuracy: 0.8815
Epoch 45/50
189/189 [==============================] - 1367s 7s/step - loss: 0.3411 - sparse_categorical_accuracy: 0.8847
Epoch 46/50
189/189 [==============================] - 1335s 7s/step - loss: 0.3319 - sparse_categorical_accuracy: 0.8866
Epoch 47/50
189/189 [==============================] - 1347s 7s/step - loss: 0.3277 - sparse_categorical_accuracy: 0.8873
Epoch 48/50
189/189 [==============================] - 1328s 7s/step - loss: 0.3263 - sparse_categorical_accuracy: 0.8884
Epoch 49/50
189/189 [==============================] - 1342s 7s/step - loss: 0.3066 - sparse_categorical_accuracy: 0.8949
Epoch 50/50
189/189 [==============================] - 1330s 7s/step - loss: 0.2950 - sparse_categorical_accuracy: 0.9008
313/313 [==============================] - 13s 42ms/step - loss: 0.9170 - sparse_categorical_accuracy: 0.7824
Test accuracy: 78.24%

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
189/189 [==============================] - 1331s 7s/step - loss: 5.3503
Epoch 2/50
189/189 [==============================] - 1321s 7s/step - loss: 5.1451
Epoch 3/50
189/189 [==============================] - 1387s 7s/step - loss: 5.0304
Epoch 4/50
189/189 [==============================] - 1324s 7s/step - loss: 4.9171
Epoch 5/50
189/189 [==============================] - 1307s 7s/step - loss: 4.8316
Epoch 6/50
189/189 [==============================] - 1315s 7s/step - loss: 4.7466
Epoch 7/50
189/189 [==============================] - 1347s 7s/step - loss: 4.6916
Epoch 8/50
189/189 [==============================] - 1323s 7s/step - loss: 4.6225
Epoch 9/50
189/189 [==============================] - 1341s 7s/step - loss: 4.5685
Epoch 10/50
189/189 [==============================] - 1297s 7s/step - loss: 4.5244
Epoch 11/50
189/189 [==============================] - 1361s 7s/step - loss: 4.4807
Epoch 12/50
189/189 [==============================] - 1313s 7s/step - loss: 4.4375
Epoch 13/50
189/189 [==============================] - 1329s 7s/step - loss: 4.3968
Epoch 14/50
189/189 [==============================] - 1352s 7s/step - loss: 4.3687
Epoch 15/50
189/189 [==============================] - 1368s 7s/step - loss: 4.3479
Epoch 16/50
189/189 [==============================] - 1371s 7s/step - loss: 4.3217
Epoch 17/50
189/189 [==============================] - 1329s 7s/step - loss: 4.2809
Epoch 18/50
189/189 [==============================] - 1311s 7s/step - loss: 4.2591
Epoch 19/50
189/189 [==============================] - 1371s 7s/step - loss: 4.2405
Epoch 20/50
189/189 [==============================] - 1291s 7s/step - loss: 4.2046
Epoch 21/50
189/189 [==============================] - 1329s 7s/step - loss: 4.1866
Epoch 22/50
189/189 [==============================] - 1320s 7s/step - loss: 4.1684
Epoch 23/50
189/189 [==============================] - 1356s 7s/step - loss: 4.1491
Epoch 24/50
189/189 [==============================] - 1332s 7s/step - loss: 4.1188
Epoch 25/50
189/189 [==============================] - 1262s 7s/step - loss: 4.0941
Epoch 26/50
189/189 [==============================] - 1290s 7s/step - loss: 4.0775
Epoch 27/50
189/189 [==============================] - 1313s 7s/step - loss: 4.0648
Epoch 28/50
189/189 [==============================] - 1329s 7s/step - loss: 4.0403
Epoch 29/50
189/189 [==============================] - 1349s 7s/step - loss: 4.0363
Epoch 30/50
189/189 [==============================] - 1325s 7s/step - loss: 4.0181
Epoch 31/50
189/189 [==============================] - 1275s 7s/step - loss: 3.9933
Epoch 32/50
189/189 [==============================] - 1332s 7s/step - loss: 3.9832
Epoch 33/50
189/189 [==============================] - 1320s 7s/step - loss: 3.9713
Epoch 34/50
189/189 [==============================] - 1304s 7s/step - loss: 3.9626
Epoch 35/50
189/189 [==============================] - 1289s 7s/step - loss: 3.9292
Epoch 36/50
189/189 [==============================] - 1358s 7s/step - loss: 3.9389
Epoch 37/50
189/189 [==============================] - 1322s 7s/step - loss: 3.9231
Epoch 38/50
189/189 [==============================] - 1292s 7s/step - loss: 3.9107
Epoch 39/50
189/189 [==============================] - 1374s 7s/step - loss: 3.8971
Epoch 40/50
189/189 [==============================] - 1352s 7s/step - loss: 3.8959
Epoch 41/50
189/189 [==============================] - 1346s 7s/step - loss: 3.8723
Epoch 42/50
189/189 [==============================] - 1345s 7s/step - loss: 3.8526
Epoch 43/50
189/189 [==============================] - 1341s 7s/step - loss: 3.8520
Epoch 44/50
189/189 [==============================] - 1373s 7s/step - loss: 3.8417
Epoch 45/50
189/189 [==============================] - 1363s 7s/step - loss: 3.8284
Epoch 46/50
189/189 [==============================] - 1353s 7s/step - loss: 3.8251
Epoch 47/50
189/189 [==============================] - 1323s 7s/step - loss: 3.8123
Epoch 48/50
189/189 [==============================] - 1308s 7s/step - loss: 3.8004
Epoch 49/50
189/189 [==============================] - 1334s 7s/step - loss: 3.7841
Epoch 50/50
189/189 [==============================] - 1349s 7s/step - loss: 3.7785

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
189/189 [==============================] - 55s 288ms/step - loss: 0.3736 - sparse_categorical_accuracy: 0.8949
Epoch 2/50
189/189 [==============================] - 55s 290ms/step - loss: 0.3366 - sparse_categorical_accuracy: 0.8995
Epoch 3/50
189/189 [==============================] - 53s 282ms/step - loss: 0.3127 - sparse_categorical_accuracy: 0.9044
Epoch 4/50
189/189 [==============================] - 54s 283ms/step - loss: 0.3138 - sparse_categorical_accuracy: 0.9046
Epoch 5/50
189/189 [==============================] - 54s 288ms/step - loss: 0.3080 - sparse_categorical_accuracy: 0.9044
Epoch 6/50
189/189 [==============================] - 55s 292ms/step - loss: 0.3056 - sparse_categorical_accuracy: 0.9046
Epoch 7/50
189/189 [==============================] - 54s 285ms/step - loss: 0.2998 - sparse_categorical_accuracy: 0.9074
Epoch 8/50
189/189 [==============================] - 53s 279ms/step - loss: 0.2964 - sparse_categorical_accuracy: 0.9077
Epoch 9/50
189/189 [==============================] - 54s 287ms/step - loss: 0.3046 - sparse_categorical_accuracy: 0.9051
Epoch 10/50
189/189 [==============================] - 54s 288ms/step - loss: 0.3118 - sparse_categorical_accuracy: 0.9036
Epoch 11/50
189/189 [==============================] - 54s 288ms/step - loss: 0.3114 - sparse_categorical_accuracy: 0.9026
Epoch 12/50
189/189 [==============================] - 53s 282ms/step - loss: 0.3011 - sparse_categorical_accuracy: 0.9060
Epoch 13/50
189/189 [==============================] - 54s 287ms/step - loss: 0.3086 - sparse_categorical_accuracy: 0.9044
Epoch 14/50
189/189 [==============================] - 55s 291ms/step - loss: 0.2997 - sparse_categorical_accuracy: 0.9064
Epoch 15/50
189/189 [==============================] - 53s 279ms/step - loss: 0.3095 - sparse_categorical_accuracy: 0.9040
Epoch 16/50
189/189 [==============================] - 54s 286ms/step - loss: 0.3102 - sparse_categorical_accuracy: 0.9036
Epoch 17/50
189/189 [==============================] - 54s 287ms/step - loss: 0.3027 - sparse_categorical_accuracy: 0.9054
Epoch 18/50
189/189 [==============================] - 54s 284ms/step - loss: 0.2932 - sparse_categorical_accuracy: 0.9074
Epoch 19/50
189/189 [==============================] - 54s 287ms/step - loss: 0.2997 - sparse_categorical_accuracy: 0.9042
Epoch 20/50
189/189 [==============================] - 55s 293ms/step - loss: 0.2981 - sparse_categorical_accuracy: 0.9069
Epoch 21/50
189/189 [==============================] - 55s 294ms/step - loss: 0.2911 - sparse_categorical_accuracy: 0.9094
Epoch 22/50
189/189 [==============================] - 55s 289ms/step - loss: 0.2977 - sparse_categorical_accuracy: 0.9064
Epoch 23/50
189/189 [==============================] - 55s 289ms/step - loss: 0.2998 - sparse_categorical_accuracy: 0.9046
Epoch 24/50
189/189 [==============================] - 52s 275ms/step - loss: 0.3016 - sparse_categorical_accuracy: 0.9038
Epoch 25/50
189/189 [==============================] - 54s 284ms/step - loss: 0.3031 - sparse_categorical_accuracy: 0.9053
Epoch 26/50
189/189 [==============================] - 54s 286ms/step - loss: 0.2918 - sparse_categorical_accuracy: 0.9076
Epoch 27/50
189/189 [==============================] - 54s 287ms/step - loss: 0.2991 - sparse_categorical_accuracy: 0.9076
Epoch 28/50
189/189 [==============================] - 55s 289ms/step - loss: 0.2981 - sparse_categorical_accuracy: 0.9042
Epoch 29/50
189/189 [==============================] - 55s 293ms/step - loss: 0.2976 - sparse_categorical_accuracy: 0.9064
Epoch 30/50
189/189 [==============================] - 54s 286ms/step - loss: 0.2975 - sparse_categorical_accuracy: 0.9070
Epoch 31/50
189/189 [==============================] - 54s 287ms/step - loss: 0.2976 - sparse_categorical_accuracy: 0.9053
Epoch 32/50
189/189 [==============================] - 54s 286ms/step - loss: 0.2919 - sparse_categorical_accuracy: 0.9088
Epoch 33/50
189/189 [==============================] - 55s 294ms/step - loss: 0.2930 - sparse_categorical_accuracy: 0.9090
Epoch 34/50
189/189 [==============================] - 56s 294ms/step - loss: 0.3025 - sparse_categorical_accuracy: 0.9049
Epoch 35/50
189/189 [==============================] - 55s 291ms/step - loss: 0.2986 - sparse_categorical_accuracy: 0.9056
Epoch 36/50
189/189 [==============================] - 55s 289ms/step - loss: 0.3094 - sparse_categorical_accuracy: 0.9042
Epoch 37/50
189/189 [==============================] - 53s 280ms/step - loss: 0.3009 - sparse_categorical_accuracy: 0.9058
Epoch 38/50
189/189 [==============================] - 54s 283ms/step - loss: 0.3011 - sparse_categorical_accuracy: 0.9049
Epoch 39/50
189/189 [==============================] - 54s 288ms/step - loss: 0.3045 - sparse_categorical_accuracy: 0.9045
Epoch 40/50
189/189 [==============================] - 54s 288ms/step - loss: 0.2986 - sparse_categorical_accuracy: 0.9065
Epoch 41/50
189/189 [==============================] - 53s 281ms/step - loss: 0.3058 - sparse_categorical_accuracy: 0.9039
Epoch 42/50
189/189 [==============================] - 54s 283ms/step - loss: 0.2960 - sparse_categorical_accuracy: 0.9055
Epoch 43/50
189/189 [==============================] - 53s 281ms/step - loss: 0.2966 - sparse_categorical_accuracy: 0.9064
Epoch 44/50
189/189 [==============================] - 55s 292ms/step - loss: 0.2990 - sparse_categorical_accuracy: 0.9064
Epoch 45/50
189/189 [==============================] - 55s 293ms/step - loss: 0.2997 - sparse_categorical_accuracy: 0.9052
Epoch 46/50
189/189 [==============================] - 54s 285ms/step - loss: 0.3026 - sparse_categorical_accuracy: 0.9064
Epoch 47/50
189/189 [==============================] - 53s 282ms/step - loss: 0.3053 - sparse_categorical_accuracy: 0.9045
Epoch 48/50
189/189 [==============================] - 53s 281ms/step - loss: 0.2985 - sparse_categorical_accuracy: 0.9043
Epoch 49/50
189/189 [==============================] - 54s 285ms/step - loss: 0.2979 - sparse_categorical_accuracy: 0.9055
Epoch 50/50
189/189 [==============================] - 54s 283ms/step - loss: 0.2969 - sparse_categorical_accuracy: 0.9065
313/313 [==============================] - 13s 43ms/step - loss: 0.6470 - sparse_categorical_accuracy: 0.8171
Test accuracy: 81.71%

```
</div>
We get to ~81.6% test accuracy.

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
