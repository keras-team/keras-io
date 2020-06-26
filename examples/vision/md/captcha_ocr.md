# OCR model for reading captcha

**Author:** [A_K_Nain](https://twitter.com/A_K_Nain)<br>
**Date created:** 2020/06/14<br>
**Last modified:** 2020/06/14<br>
**Description:** How to implement an OCR model using CNNs, RNNs and CTC loss.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/captcha_ocr.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/captcha_ocr.py)



---
## Introduction

This example demonstrates a simple OCR model using Functional API. Apart from
combining CNN and RNN, it also illustrates how you can instantiate a new layer
and use it as an `Endpoint` layer for implementing CTC loss. For a detailed
description on layer subclassing, please check out this
[example](https://keras.io/guides/making_new_layers_and_models_via_subclassing/#the-addmetric-method)
in the developer guides.

---
## Setup


```python
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

```

---
## Load the data: [Captcha Images](https://www.kaggle.com/fournierp/captcha-version-2-images)
Let's download the data.


```python
!curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
!unzip -qq captcha_images_v2.zip
```

<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   159  100   159    0     0    200      0 --:--:-- --:--:-- --:--:--   200
100 8863k  100 8863k    0     0  7620k      0  0:00:01  0:00:01 --:--:-- 7620k

```
</div>
The dataset contains 1040 captcha files as png images. The label for each sample is the
name of the file (excluding the '.png' part). The label for each sample is a string.
We will map each character in the string to a number for training the model. Similary,
we would be required to map the predictions of the model back to string. For this purpose
would maintain two dictionary mapping characters to numbers and numbers to characters
respectively.


```python

# Path to the data directory
data_dir = Path("./captcha_images_v2/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 200
img_height = 50

# Factor  by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each convolution block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

```

<div class="k-default-codeblock">
```
Number of images found:  1040
Number of labels found:  1040
Number of unique characters:  19
Characters present:  {'7', 'y', 'd', '8', 'f', 'b', '5', 'c', '6', 'p', 'x', '4', '3', 'n', 'w', 'e', '2', 'm', 'g'}

```
</div>
---
## Preprocessing


```python

# Mapping characters to numbers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping numbers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}

```

---
## Data Generators


```python

train_data_generator = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data_generator = (
    train_data_generator.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

valid_data_generator = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_data_generator = (
    valid_data_generator.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)
```

---
## Visualize the data


```python

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_data_generator.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()
```


![png](/img/examples/vision/captcha_ocr/captcha_ocr_13_0.png)


---
## Model


```python

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # On test time, just return the computed loss
        return y_pred


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing it to RNNs
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# Get the model
model = build_model()
model.summary()
```

<div class="k-default-codeblock">
```
Model: "ocr_model_v1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
image (InputLayer)              [(None, 200, 50, 1)] 0                                            
__________________________________________________________________________________________________
Conv1 (Conv2D)                  (None, 200, 50, 32)  320         image[0][0]                      
__________________________________________________________________________________________________
pool1 (MaxPooling2D)            (None, 100, 25, 32)  0           Conv1[0][0]                      
__________________________________________________________________________________________________
Conv2 (Conv2D)                  (None, 100, 25, 64)  18496       pool1[0][0]                      
__________________________________________________________________________________________________
pool2 (MaxPooling2D)            (None, 50, 12, 64)   0           Conv2[0][0]                      
__________________________________________________________________________________________________
reshape (Reshape)               (None, 50, 768)      0           pool2[0][0]                      
__________________________________________________________________________________________________
dense1 (Dense)                  (None, 50, 64)       49216       reshape[0][0]                    
__________________________________________________________________________________________________
dropout (Dropout)               (None, 50, 64)       0           dense1[0][0]                     
__________________________________________________________________________________________________
bidirectional (Bidirectional)   (None, 50, 256)      197632      dropout[0][0]                    
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 50, 128)      164352      bidirectional[0][0]              
__________________________________________________________________________________________________
label (InputLayer)              [(None, None)]       0                                            
__________________________________________________________________________________________________
dense2 (Dense)                  (None, 50, 20)       2580        bidirectional_1[0][0]            
__________________________________________________________________________________________________
ctc_loss (CTCLayer)             (None, 50, 20)       0           label[0][0]                      
                                                                 dense2[0][0]                     
==================================================================================================
Total params: 432,596
Trainable params: 432,596
Non-trainable params: 0
__________________________________________________________________________________________________

```
</div>
---
## Training


```python

epochs = 100
es_patience = 10
# Add early stopping
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=es_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_data_generator,
    validation_data=valid_data_generator,
    epochs=epochs,
    callbacks=[es],
)

```

<div class="k-default-codeblock">
```
Epoch 1/100
59/59 [==============================] - 3s 55ms/step - loss: 20.4639 - val_loss: 16.4358
Epoch 2/100
59/59 [==============================] - 1s 24ms/step - loss: 16.3280 - val_loss: 16.4349
Epoch 3/100
59/59 [==============================] - 1s 25ms/step - loss: 16.3248 - val_loss: 16.4418
Epoch 4/100
59/59 [==============================] - 1s 25ms/step - loss: 16.3218 - val_loss: 16.4366
Epoch 5/100
59/59 [==============================] - 1s 24ms/step - loss: 16.3030 - val_loss: 16.4514
Epoch 6/100
59/59 [==============================] - 1s 24ms/step - loss: 16.2960 - val_loss: 16.3997
Epoch 7/100
59/59 [==============================] - 1s 24ms/step - loss: 16.2485 - val_loss: 16.3081
Epoch 8/100
59/59 [==============================] - 1s 25ms/step - loss: 16.1381 - val_loss: 16.0589
Epoch 9/100
59/59 [==============================] - 1s 24ms/step - loss: 15.8330 - val_loss: 15.6132
Epoch 10/100
59/59 [==============================] - 1s 24ms/step - loss: 15.3101 - val_loss: 14.6895
Epoch 11/100
59/59 [==============================] - 1s 24ms/step - loss: 13.2795 - val_loss: 10.7522
Epoch 12/100
59/59 [==============================] - 1s 25ms/step - loss: 9.2077 - val_loss: 5.4861
Epoch 13/100
59/59 [==============================] - 1s 25ms/step - loss: 4.8549 - val_loss: 2.0471
Epoch 14/100
59/59 [==============================] - 1s 25ms/step - loss: 2.3248 - val_loss: 0.8337
Epoch 15/100
59/59 [==============================] - 1s 24ms/step - loss: 1.4187 - val_loss: 0.5065
Epoch 16/100
59/59 [==============================] - 1s 25ms/step - loss: 0.9633 - val_loss: 0.2598
Epoch 17/100
59/59 [==============================] - 1s 24ms/step - loss: 0.6201 - val_loss: 0.1746
Epoch 18/100
59/59 [==============================] - 1s 25ms/step - loss: 0.4828 - val_loss: 0.1050
Epoch 19/100
59/59 [==============================] - 1s 24ms/step - loss: 0.3048 - val_loss: 0.0673
Epoch 20/100
59/59 [==============================] - 1s 25ms/step - loss: 0.2504 - val_loss: 0.0470
Epoch 21/100
59/59 [==============================] - 1s 24ms/step - loss: 0.2388 - val_loss: 0.0555
Epoch 22/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1876 - val_loss: 0.0682
Epoch 23/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1102 - val_loss: 0.0401
Epoch 24/100
59/59 [==============================] - 1s 25ms/step - loss: 0.1279 - val_loss: 0.0243
Epoch 25/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1413 - val_loss: 0.0503
Epoch 26/100
59/59 [==============================] - 1s 25ms/step - loss: 0.1357 - val_loss: 0.0238
Epoch 27/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1380 - val_loss: 0.0140
Epoch 28/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1004 - val_loss: 0.0411
Epoch 29/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1259 - val_loss: 0.0149
Epoch 30/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0818 - val_loss: 0.0147
Epoch 31/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0746 - val_loss: 0.0104
Epoch 32/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1260 - val_loss: 0.0179
Epoch 33/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1045 - val_loss: 0.0396
Epoch 34/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0610 - val_loss: 0.0111
Epoch 35/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0750 - val_loss: 0.0233
Epoch 36/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0863 - val_loss: 0.0101
Epoch 37/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0737 - val_loss: 0.0139
Epoch 38/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0677 - val_loss: 0.0078
Epoch 39/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0402 - val_loss: 0.0069
Epoch 40/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0490 - val_loss: 0.0249
Epoch 41/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0673 - val_loss: 0.0072
Epoch 42/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0841 - val_loss: 0.0054
Epoch 43/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0796 - val_loss: 0.0073
Epoch 44/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0408 - val_loss: 0.0055
Epoch 45/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0458 - val_loss: 0.0047
Epoch 46/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0395 - val_loss: 0.0054
Epoch 47/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0254 - val_loss: 0.0043
Epoch 48/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0585 - val_loss: 0.0275
Epoch 49/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0770 - val_loss: 0.0306
Epoch 50/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0826 - val_loss: 0.0077
Epoch 51/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0242 - val_loss: 0.0037
Epoch 52/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0392 - val_loss: 0.0036
Epoch 53/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1234 - val_loss: 0.0045
Epoch 54/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0679 - val_loss: 0.0233
Epoch 55/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0438 - val_loss: 0.0040
Epoch 56/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0558 - val_loss: 0.0040
Epoch 57/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0328 - val_loss: 0.0027
Epoch 58/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0418 - val_loss: 0.0048
Epoch 59/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0324 - val_loss: 0.0021
Epoch 60/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0189 - val_loss: 0.0036
Epoch 61/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0448 - val_loss: 0.0042
Epoch 62/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0203 - val_loss: 0.0025
Epoch 63/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0838 - val_loss: 0.0998
Epoch 64/100
59/59 [==============================] - 1s 25ms/step - loss: 0.0507 - val_loss: 0.0028
Epoch 65/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0499 - val_loss: 0.0020
Epoch 66/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0123 - val_loss: 0.0044
Epoch 67/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0339 - val_loss: 0.0026
Epoch 68/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0175 - val_loss: 0.0020
Epoch 69/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0134 - val_loss: 0.0016
Epoch 70/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0651 - val_loss: 0.0029
Epoch 71/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0357 - val_loss: 0.0017
Epoch 72/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0074 - val_loss: 0.0015
Epoch 73/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0512 - val_loss: 0.0020
Epoch 74/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0293 - val_loss: 0.0017
Epoch 75/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0201 - val_loss: 0.0021
Epoch 76/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0155 - val_loss: 0.0022
Epoch 77/100
59/59 [==============================] - 1s 24ms/step - loss: 0.1752 - val_loss: 0.0062
Epoch 78/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0369 - val_loss: 0.0022
Epoch 79/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0332 - val_loss: 0.0015
Epoch 80/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0104 - val_loss: 0.0024
Epoch 81/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0089 - val_loss: 0.0011
Epoch 82/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0124 - val_loss: 0.0043
Epoch 83/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0293 - val_loss: 0.0030
Epoch 84/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0186 - val_loss: 9.2171e-04
Epoch 85/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0366 - val_loss: 0.0021
Epoch 86/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0218 - val_loss: 0.0012
Epoch 87/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0159 - val_loss: 0.0011
Epoch 88/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0125 - val_loss: 9.5702e-04
Epoch 89/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0372 - val_loss: 9.8982e-04
Epoch 90/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0517 - val_loss: 0.0025
Epoch 91/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0327 - val_loss: 0.0026
Epoch 92/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0401 - val_loss: 0.0013
Epoch 93/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0279 - val_loss: 0.0266
Epoch 94/100
59/59 [==============================] - 1s 24ms/step - loss: 0.0300 - val_loss: 0.0011

```
</div>
---
## Let's test-drive it


```python

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Let's check results on some validation samples
for batch in valid_data_generator.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()
```

<div class="k-default-codeblock">
```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
image (InputLayer)           [(None, 200, 50, 1)]      0         
_________________________________________________________________
Conv1 (Conv2D)               (None, 200, 50, 32)       320       
_________________________________________________________________
pool1 (MaxPooling2D)         (None, 100, 25, 32)       0         
_________________________________________________________________
Conv2 (Conv2D)               (None, 100, 25, 64)       18496     
_________________________________________________________________
pool2 (MaxPooling2D)         (None, 50, 12, 64)        0         
_________________________________________________________________
reshape (Reshape)            (None, 50, 768)           0         
_________________________________________________________________
dense1 (Dense)               (None, 50, 64)            49216     
_________________________________________________________________
dropout (Dropout)            (None, 50, 64)            0         
_________________________________________________________________
bidirectional (Bidirectional (None, 50, 256)           197632    
_________________________________________________________________
bidirectional_1 (Bidirection (None, 50, 128)           164352    
_________________________________________________________________
dense2 (Dense)               (None, 50, 20)            2580      
=================================================================
Total params: 432,596
Trainable params: 432,596
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From /home/nainaakash012/miniconda3/envs/tfnightly/lib/python3.7/site-packages/tensorflow/python/util/dispatch.py:201: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.

```
</div>
![png](/img/examples/vision/captcha_ocr/captcha_ocr_19_1.png)

