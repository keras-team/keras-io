# OCR model for reading Captchas

**Author:** [A_K_Nain](https://twitter.com/A_K_Nain)<br>
**Date created:** 2020/06/14<br>
**Last modified:** 2020/06/26<br>
**Description:** How to implement an OCR model using CNNs, RNNs and CTC loss.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/captcha_ocr.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/captcha_ocr.py)



---
## Introduction

This example demonstrates a simple OCR model built with the Functional API. Apart from
combining CNN and RNN, it also illustrates how you can instantiate a new layer
and use it as an "Endpoint layer" for implementing CTC loss. For a detailed
guide to layer subclassing, please check out
[this page](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)
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
100   159  100   159    0     0    164      0 --:--:-- --:--:-- --:--:--   164
100 8863k  100 8863k    0     0  4882k      0  0:00:01  0:00:01 --:--:-- 33.0M

```
</div>
The dataset contains 1040 captcha files as `png` images. The label for each sample is a string,
the name of the file (minus the file extension).
We will map each character in the string to an integer for training the model. Similary,
we will need to map the predictions of the model back to strings. For this purpose
we will maintain two dictionaries, mapping characters to integers, and integers to characters,
respectively.


```python

# Path to the data directory
data_dir = Path("./captcha_images_v2/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 200
img_height = 50

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
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
Characters present:  {'d', 'w', 'y', '4', 'f', '6', 'g', 'e', '3', '5', 'p', 'x', '2', 'c', '7', 'n', 'b', '8', 'm'}

```
</div>
---
## Preprocessing


```python

# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
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
## Create `Dataset` objects


```python

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
```

---
## Visualize the data


```python

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
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

        # At test time, just return the computed predictions
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

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

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
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

```

<div class="k-default-codeblock">
```
Epoch 1/100
59/59 [==============================] - 3s 53ms/step - loss: 21.5722 - val_loss: 16.3351
Epoch 2/100
59/59 [==============================] - 2s 27ms/step - loss: 16.3335 - val_loss: 16.3062
Epoch 3/100
59/59 [==============================] - 2s 27ms/step - loss: 16.3360 - val_loss: 16.3116
Epoch 4/100
59/59 [==============================] - 2s 27ms/step - loss: 16.3318 - val_loss: 16.3167
Epoch 5/100
59/59 [==============================] - 2s 27ms/step - loss: 16.3256 - val_loss: 16.3152
Epoch 6/100
59/59 [==============================] - 2s 29ms/step - loss: 16.3229 - val_loss: 16.3123
Epoch 7/100
59/59 [==============================] - 2s 30ms/step - loss: 16.3119 - val_loss: 16.3116
Epoch 8/100
59/59 [==============================] - 2s 27ms/step - loss: 16.2977 - val_loss: 16.3107
Epoch 9/100
59/59 [==============================] - 2s 28ms/step - loss: 16.2801 - val_loss: 16.2552
Epoch 10/100
59/59 [==============================] - 2s 28ms/step - loss: 16.2199 - val_loss: 16.1008
Epoch 11/100
59/59 [==============================] - 2s 28ms/step - loss: 16.1136 - val_loss: 15.9867
Epoch 12/100
59/59 [==============================] - 2s 30ms/step - loss: 16.0138 - val_loss: 15.8825
Epoch 13/100
59/59 [==============================] - 2s 29ms/step - loss: 15.9670 - val_loss: 15.8413
Epoch 14/100
59/59 [==============================] - 2s 29ms/step - loss: 15.9315 - val_loss: 15.8263
Epoch 15/100
59/59 [==============================] - 2s 31ms/step - loss: 15.9162 - val_loss: 15.7971
Epoch 16/100
59/59 [==============================] - 2s 31ms/step - loss: 15.8916 - val_loss: 15.7844
Epoch 17/100
59/59 [==============================] - 2s 31ms/step - loss: 15.8653 - val_loss: 15.7624
Epoch 18/100
59/59 [==============================] - 2s 31ms/step - loss: 15.8543 - val_loss: 15.7620
Epoch 19/100
59/59 [==============================] - 2s 28ms/step - loss: 15.8373 - val_loss: 15.7559
Epoch 20/100
59/59 [==============================] - 2s 27ms/step - loss: 15.8319 - val_loss: 15.7495
Epoch 21/100
59/59 [==============================] - 2s 27ms/step - loss: 15.8104 - val_loss: 15.7430
Epoch 22/100
59/59 [==============================] - 2s 29ms/step - loss: 15.8037 - val_loss: 15.7260
Epoch 23/100
59/59 [==============================] - 2s 29ms/step - loss: 15.8021 - val_loss: 15.7204
Epoch 24/100
59/59 [==============================] - 2s 28ms/step - loss: 15.7901 - val_loss: 15.7174
Epoch 25/100
59/59 [==============================] - 2s 29ms/step - loss: 15.7851 - val_loss: 15.7074
Epoch 26/100
59/59 [==============================] - 2s 27ms/step - loss: 15.7701 - val_loss: 15.7097
Epoch 27/100
59/59 [==============================] - 2s 28ms/step - loss: 15.7694 - val_loss: 15.7040
Epoch 28/100
59/59 [==============================] - 2s 28ms/step - loss: 15.7544 - val_loss: 15.7012
Epoch 29/100
59/59 [==============================] - 2s 31ms/step - loss: 15.7498 - val_loss: 15.7015
Epoch 30/100
59/59 [==============================] - 2s 31ms/step - loss: 15.7521 - val_loss: 15.6880
Epoch 31/100
59/59 [==============================] - 2s 29ms/step - loss: 15.7165 - val_loss: 15.6734
Epoch 32/100
59/59 [==============================] - 2s 27ms/step - loss: 15.6650 - val_loss: 15.5789
Epoch 33/100
59/59 [==============================] - 2s 27ms/step - loss: 15.5300 - val_loss: 15.4026
Epoch 34/100
59/59 [==============================] - 2s 27ms/step - loss: 15.3519 - val_loss: 15.2115
Epoch 35/100
59/59 [==============================] - 2s 27ms/step - loss: 15.1165 - val_loss: 14.7826
Epoch 36/100
59/59 [==============================] - 2s 27ms/step - loss: 14.7086 - val_loss: 14.4432
Epoch 37/100
59/59 [==============================] - 2s 29ms/step - loss: 14.3317 - val_loss: 13.9445
Epoch 38/100
59/59 [==============================] - 2s 29ms/step - loss: 13.9658 - val_loss: 13.6972
Epoch 39/100
59/59 [==============================] - 2s 29ms/step - loss: 13.6728 - val_loss: 13.3388
Epoch 40/100
59/59 [==============================] - 2s 28ms/step - loss: 13.3454 - val_loss: 13.0102
Epoch 41/100
59/59 [==============================] - 2s 27ms/step - loss: 13.0448 - val_loss: 12.8307
Epoch 42/100
59/59 [==============================] - 2s 28ms/step - loss: 12.7552 - val_loss: 12.6071
Epoch 43/100
59/59 [==============================] - 2s 29ms/step - loss: 12.4573 - val_loss: 12.2800
Epoch 44/100
59/59 [==============================] - 2s 31ms/step - loss: 12.1055 - val_loss: 11.9209
Epoch 45/100
59/59 [==============================] - 2s 28ms/step - loss: 11.8148 - val_loss: 11.9132
Epoch 46/100
59/59 [==============================] - 2s 28ms/step - loss: 11.4530 - val_loss: 11.4357
Epoch 47/100
59/59 [==============================] - 2s 29ms/step - loss: 11.0592 - val_loss: 11.1121
Epoch 48/100
59/59 [==============================] - 2s 27ms/step - loss: 10.7746 - val_loss: 10.8532
Epoch 49/100
59/59 [==============================] - 2s 28ms/step - loss: 10.2616 - val_loss: 10.3643
Epoch 50/100
59/59 [==============================] - 2s 28ms/step - loss: 9.8708 - val_loss: 10.0987
Epoch 51/100
59/59 [==============================] - 2s 30ms/step - loss: 9.4077 - val_loss: 9.6371
Epoch 52/100
59/59 [==============================] - 2s 29ms/step - loss: 9.0663 - val_loss: 9.2463
Epoch 53/100
59/59 [==============================] - 2s 28ms/step - loss: 8.4546 - val_loss: 8.7581
Epoch 54/100
59/59 [==============================] - 2s 28ms/step - loss: 7.9226 - val_loss: 8.1805
Epoch 55/100
59/59 [==============================] - 2s 27ms/step - loss: 7.4927 - val_loss: 7.8858
Epoch 56/100
59/59 [==============================] - 2s 28ms/step - loss: 7.0499 - val_loss: 7.3202
Epoch 57/100
59/59 [==============================] - 2s 27ms/step - loss: 6.6383 - val_loss: 7.0875
Epoch 58/100
59/59 [==============================] - 2s 28ms/step - loss: 6.1446 - val_loss: 6.9619
Epoch 59/100
59/59 [==============================] - 2s 28ms/step - loss: 5.8533 - val_loss: 6.3855
Epoch 60/100
59/59 [==============================] - 2s 28ms/step - loss: 5.5107 - val_loss: 5.9797
Epoch 61/100
59/59 [==============================] - 2s 31ms/step - loss: 5.1181 - val_loss: 5.7549
Epoch 62/100
59/59 [==============================] - 2s 31ms/step - loss: 4.6952 - val_loss: 5.5488
Epoch 63/100
59/59 [==============================] - 2s 29ms/step - loss: 4.4189 - val_loss: 5.3030
Epoch 64/100
59/59 [==============================] - 2s 28ms/step - loss: 4.1358 - val_loss: 5.1772
Epoch 65/100
59/59 [==============================] - 2s 28ms/step - loss: 3.8560 - val_loss: 5.1071
Epoch 66/100
59/59 [==============================] - 2s 28ms/step - loss: 3.5342 - val_loss: 4.6958
Epoch 67/100
59/59 [==============================] - 2s 28ms/step - loss: 3.3336 - val_loss: 4.5865
Epoch 68/100
59/59 [==============================] - 2s 27ms/step - loss: 3.0925 - val_loss: 4.3647
Epoch 69/100
59/59 [==============================] - 2s 28ms/step - loss: 2.8751 - val_loss: 4.3005
Epoch 70/100
59/59 [==============================] - 2s 27ms/step - loss: 2.7444 - val_loss: 4.0820
Epoch 71/100
59/59 [==============================] - 2s 27ms/step - loss: 2.5921 - val_loss: 4.1694
Epoch 72/100
59/59 [==============================] - 2s 28ms/step - loss: 2.3246 - val_loss: 3.9142
Epoch 73/100
59/59 [==============================] - 2s 28ms/step - loss: 2.0769 - val_loss: 3.9135
Epoch 74/100
59/59 [==============================] - 2s 29ms/step - loss: 2.0872 - val_loss: 3.9808
Epoch 75/100
59/59 [==============================] - 2s 29ms/step - loss: 1.9498 - val_loss: 3.9935
Epoch 76/100
59/59 [==============================] - 2s 28ms/step - loss: 1.8178 - val_loss: 3.7735
Epoch 77/100
59/59 [==============================] - 2s 29ms/step - loss: 1.7661 - val_loss: 3.6309
Epoch 78/100
59/59 [==============================] - 2s 31ms/step - loss: 1.6236 - val_loss: 3.7410
Epoch 79/100
59/59 [==============================] - 2s 29ms/step - loss: 1.4652 - val_loss: 3.6756
Epoch 80/100
59/59 [==============================] - 2s 27ms/step - loss: 1.3552 - val_loss: 3.4979
Epoch 81/100
59/59 [==============================] - 2s 29ms/step - loss: 1.2655 - val_loss: 3.5306
Epoch 82/100
59/59 [==============================] - 2s 29ms/step - loss: 1.2632 - val_loss: 3.2885
Epoch 83/100
59/59 [==============================] - 2s 28ms/step - loss: 1.2316 - val_loss: 3.2482
Epoch 84/100
59/59 [==============================] - 2s 30ms/step - loss: 1.1260 - val_loss: 3.4285
Epoch 85/100
59/59 [==============================] - 2s 28ms/step - loss: 1.0745 - val_loss: 3.2985
Epoch 86/100
59/59 [==============================] - 2s 29ms/step - loss: 1.0133 - val_loss: 3.2209
Epoch 87/100
59/59 [==============================] - 2s 31ms/step - loss: 0.9417 - val_loss: 3.2203
Epoch 88/100
59/59 [==============================] - 2s 28ms/step - loss: 0.9104 - val_loss: 3.1121
Epoch 89/100
59/59 [==============================] - 2s 30ms/step - loss: 0.8516 - val_loss: 3.2070
Epoch 90/100
59/59 [==============================] - 2s 28ms/step - loss: 0.8275 - val_loss: 3.0335
Epoch 91/100
59/59 [==============================] - 2s 28ms/step - loss: 0.8056 - val_loss: 3.2085
Epoch 92/100
59/59 [==============================] - 2s 28ms/step - loss: 0.7373 - val_loss: 3.0326
Epoch 93/100
59/59 [==============================] - 2s 28ms/step - loss: 0.7753 - val_loss: 2.9935
Epoch 94/100
59/59 [==============================] - 2s 28ms/step - loss: 0.7688 - val_loss: 2.9940
Epoch 95/100
59/59 [==============================] - 2s 27ms/step - loss: 0.6765 - val_loss: 3.0432
Epoch 96/100
59/59 [==============================] - 2s 29ms/step - loss: 0.6674 - val_loss: 3.1233
Epoch 97/100
59/59 [==============================] - 2s 29ms/step - loss: 0.6018 - val_loss: 2.8405
Epoch 98/100
59/59 [==============================] - 2s 28ms/step - loss: 0.6322 - val_loss: 2.8323
Epoch 99/100
59/59 [==============================] - 2s 29ms/step - loss: 0.5889 - val_loss: 2.8786
Epoch 100/100
59/59 [==============================] - 2s 28ms/step - loss: 0.5616 - val_loss: 2.9697

```
</div>
---
## Inference

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/ocr-for-captcha) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/ocr-for-captcha).

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
for batch in validation_dataset.take(1):
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

```
</div>
![png](/img/examples/vision/captcha_ocr/captcha_ocr_19_1.png)