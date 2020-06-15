"""
Title: OCR model for reading captcha
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2020/06/14
Last modified: 2020/06/14
Description: How to implement an OCR model using CNNs, RNNs and CTC loss.
"""

"""
## Introduction

This example demonstrates a simple OCR model using Functional API. Apart from
combining CNN and RNN, it also illustrates how you can instantiate a new layer
and use it as an `Endpoint` layer for implementing CTC loss. For a detailed
description on layer subclassing, please check out this [example](https://keras.io/guides/making_new_layers_and_models_via_subclassing/#the-addmetric-method)
in the developer guides.
"""

"""
## Setup
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

"""
## Load the data: [Captcha Images](https://www.kaggle.com/fournierp/captcha-version-2-images)
Let's download the data.
"""

"""shell
curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images.tar.gz
tar -zxf captcha_images.tar.gz
"""


"""
The dataset contains 1070 captcha files as png images. The label for each sample is the
name of the file (excluding the '.png' part). We will load the data as numpy arrays. The label
for each sample is a string. We will map each character in the string to a number for training
the model. Similary, we would be required to map the predictions of the model back to string.
For this purpose would maintain two dictionary mapping characters to numbers and numbers to
characters respectively.
"""

# Path to the data directory
data_dir = Path("./captcha_images/samples/")

# Get list of all the images
images = list(data_dir.glob("*.png"))
print("Number of images found: ", len(images))

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
max_length = 5


def get_img_array(img_path, size=(50, 200)):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(
        img_path, color_mode="grayscale", target_size=size
    )
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


# Let's take a look at few samples
sample_images = images[:4]
_, ax = plt.subplots(2, 2, figsize=(5, 3))
for i in range(4):
    img = get_img_array(str(sample_images[i]))
    ax[i // 2, i % 2].imshow(img[0, :, :, 0])
    ax[i // 2, i % 2].axis("off")
plt.show()

"""
## Preprocessing
"""


# Store all the characters in a set
characters = set()

# We will store the length of each captcha in a list
captcha_length = []

# Store image-label pair info
dataset = []

# Iterate over the dataset and extract the
# information needed
for img_path in images:
    # 1. Get the label associated with each image
    label = img_path.name.split(".png")[0]
    # 2. Store the length of this cpatcha
    captcha_length.append(len(label))
    # 3. Store the image-label pair info
    dataset.append((str(img_path), label))

    # 4. Store the characters present
    for ch in label:
        characters.add(ch)

# Sort the characters
characters = sorted(characters)

# Convert the dataset info into a pandas dataframe
dataset = pd.DataFrame(dataset, columns=["img_path", "label"], index=None)

# Shuffle the dataset
dataset = dataset.sample(frac=1.0).reset_index(drop=True)

print("Number of unqiue charcaters in the whole dataset: ", len(characters))
print("Maximum length of any captcha: ", max(Counter(captcha_length).keys()))
print("Characters present: ", characters)
print("Total number of samples in the dataset: ", len(dataset))


# Split the dataset into training and validation sets
training_data, validation_data = train_test_split(
    dataset, test_size=0.1, random_state=seed
)

training_data = training_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)
print("Number of training samples: ", len(training_data))
print("Number of validation samples: ", len(validation_data))

# Map characters to numbers
char_to_labels = {char: idx for idx, char in enumerate(characters)}

# Map numbers to text
labels_to_char = {val: key for key, val in char_to_labels.items()}

# Sanity check for corrupted images
def is_valid_captcha(captcha):
    for ch in captcha:
        if not ch in characters:
            return False
    return True


# We will store the data in memory as it's not a big dataset
def generate_arrays(df, resize=True, img_height=50, img_width=200):
    """Generates image array and labels array from a dataframe.
    
    Args:
        df: dataframe from which we want to read the data
        resize (bool): whether to resize images or not
        img_weidth (int): width of the resized images
        img_height (int): height of the resized images
        
    Returns:
        images (ndarray): grayscale images
        labels (ndarray): corresponding encoded labels
    """

    num_items = len(df)
    images = np.zeros((num_items, img_height, img_width, 1), dtype=np.float32)
    labels = [0] * num_items

    for i in range(num_items):
        img = get_img_array(df["img_path"][i])
        img = (img / 255.0).astype(np.float32)
        label = df["label"][i]

        # Add the current sample only if it is a valid captcha
        if is_valid_captcha(label):
            images[i, :, :] = img
            labels[i] = label

    return images, np.array(labels)


# Build training data
training_data, training_labels = generate_arrays(df=training_data)
print("Number of training images: ", training_data.shape)
print("Number of training labels: ", training_labels.shape)

# Build validation data
validation_data, validation_labels = generate_arrays(df=validation_data)
print("Number of validation images: ", validation_data.shape)
print("Number of validation labels: ", validation_labels.shape)


"""
## Data Generators
"""


class DataGenerator(keras.utils.Sequence):
    """Generates batches from a given dataset.
    
    Args:
        data: training or validation data
        labels: labels corresponding to the data
        char_map: dictionary mapping characters to numbers
        batch_size: number of samples in a single batch
        img_width: width of the resized image
        img_height: height of the resized image
        downsample_factor: combined downsampling factor used in CNN blocks
        shuffle: whether to shuffle data or not after each epoch
    Returns:
        batch_inputs: a dictionary containing batch inputs
        dummy ndarray filled with zeros having same shape as batch_input 
    """

    def __init__(
        self,
        data,
        labels,
        char_map,
        batch_size=16,
        img_width=200,
        img_height=50,
        downsample_factor=4,
        max_length=5,
        shuffle=True,
    ):
        self.data = data
        self.labels = labels
        self.char_map = char_map
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.downsample_factor = downsample_factor
        self.max_length = max_length
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        # 1. Get the next batch indices
        curr_batch_idx = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        # 2. Only necessay for the last batch
        batch_len = len(curr_batch_idx)
        # 3. Instantiate batch arrays
        batch_images = np.ones(
            (batch_len, self.img_width, self.img_height, 1), dtype=np.float32
        )
        batch_labels = np.ones((batch_len, self.max_length), dtype=np.float32)
        input_length = np.ones((batch_len, 1), dtype=np.int64) * (
            self.img_width // self.downsample_factor - 2
        )
        label_length = np.zeros((batch_len, 1), dtype=np.int64)

        for j, idx in enumerate(curr_batch_idx):
            # 4. Get the image and transpose it
            img = self.data[idx].T
            # 5. Add extra dimenison
            img = np.expand_dims(img, axis=-1)
            # 6. Get the correpsonding label
            text = self.labels[idx]
            # 7. Include the pair only if the captcha is valid
            if is_valid_captcha(text):
                label = [self.char_map[ch] for ch in text]
                batch_images[j] = img
                batch_labels[j] = label
                label_length[j] = len(text)

        # 8. Make a dictionary of inout data
        batch_inputs = {
            "input_data": batch_images,
            "input_label": batch_labels,
            "input_length": input_length,
            "label_length": label_length,
        }
        return batch_inputs, np.zeros(batch_len).astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# Get a generator object for the training data
train_data_generator = DataGenerator(
    data=training_data,
    labels=training_labels,
    char_map=char_to_labels,
    batch_size=batch_size,
    img_width=img_width,
    img_height=img_height,
    downsample_factor=downsample_factor,
    max_length=max_length,
    shuffle=True,
)

# Get a generator object for the validation data
valid_data_generator = DataGenerator(
    data=validation_data,
    labels=validation_labels,
    char_map=char_to_labels,
    batch_size=batch_size,
    img_width=img_width,
    img_height=img_height,
    downsample_factor=downsample_factor,
    max_length=max_length,
    shuffle=False,
)

"""
## Model
"""


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # On test time, just return the computed loss
        return loss


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="input_data", dtype="float32"
    )
    labels = layers.Input(name="input_label", shape=[max_length], dtype="float32")
    input_length = layers.Input(name="input_length", shape=[1], dtype="int64")
    label_length = layers.Input(name="label_length", shape=[1], dtype="int64")

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
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=output,
        name="ocr_model_v1",
    )

    # Optimizer
    sgd = keras.optimizers.SGD(
        learning_rate=0.002, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5
    )

    # Compile the model and return
    model.compile(optimizer=sgd)
    return model


# Get the model
model = build_model()
print(model.summary())

"""
## Training
"""


# Add early stopping
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_data_generator,
    validation_data=valid_data_generator,
    epochs=50,
    callbacks=[es],
)


"""
## Let's test-drive it
"""


# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="input_data").input, model.get_layer(name="dense2").output
)
print(prediction_model.summary())

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        outstr = ""
        for c in res:
            if c < len(characters) and c >= 0:
                outstr += labels_to_char[c]
        output_text.append(outstr)

    # return final text results
    return output_text


# Let's check results on some validation samples
for p, (inp_value, _) in enumerate(valid_data_generator):
    bs = inp_value["input_data"].shape[0]
    X_data = inp_value["input_data"]
    labels = inp_value["input_label"]

    preds = prediction_model.predict(X_data)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in labels:
        text = "".join([labels_to_char[int(x)] for x in label])
        orig_texts.append(text)

    _, ax = plt.subplots(3, 3, figsize=(15, 5))

    for i in range(9):
        img = (X_data[i, :, :, 0] * 255).astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 3, i % 3].imshow(img, cmap="gray")
        ax[i // 3, i % 3].set_title(title)
        ax[i // 3, i % 3].axis("off")
    break
