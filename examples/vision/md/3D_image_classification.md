# 3D Image Classification from CT Scans

**Author:** [Hasib Zunair](https://twitter.com/hasibzunair)<br>
**Date created:** 2020/09/23<br>
**Last modified:** 2020/09/23<br>
**Description:** Train a 3D convolutional neural network to predict presence of pneumonia.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/3D_image_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/3D_image_classification.py)



---
## Introduction

This example will show the steps needed to build a 3D convolutional neural network (CNN)
to predict the presence of viral pneumonia in computer tomography (CT) scans. 2D CNNs are
commonly used to process RGB images (3 channels). A 3D CNN is simply the 3D
equivalent: it takes as input a 3D volume or a sequence of 2D frames (e.g. slices in a CT scan),
3D CNNs are a powerful model for learning representations for volumetric data.

---
## References

- [A survey on Deep Learning Advances on Different 3D DataRepresentations](https://arxiv.org/pdf/1808.01462.pdf)
- [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)
- [FusionNet: 3D Object Classification Using MultipleData Representations](http://3ddl.cs.princeton.edu/2016/papers/Hegde_Zadeh.pdf)
- [Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction](https://arxiv.org/abs/2007.13224)


```python
import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
```

---
## Downloading the MosMedData:Chest CT Scans with COVID-19 Related Findings

In this example, we use a subset of the
[MosMedData: Chest CT Scans with COVID-19 Related
Findings](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). This dataset
consists of lung CT scans with COVID-19 related findings, as well as without such
findings.

We will be using the associated radiological findings of the CT scans as labels to build
a classifier to predict presence of viral pneumonia.
Hence, the task is a binary classification problem.


```python
# Download url of normal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename = os.path.join(os.getcwd(), "CT-0.zip")
keras.utils.get_file(filename, url)

# Download url of abnormal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-1.zip"
filename = os.path.join(os.getcwd(), "CT-1.zip")
keras.utils.get_file(filename, url)

# Make a directory to store the data.
os.makedirs("MosMedData")

# Unzip data in the newly created directory.
with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-1.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")
```

<div class="k-default-codeblock">
```
Downloading data from https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip
1065476096/1065471431 [==============================] - 234s 0us/step
Downloading data from https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-1.zip
 275013632/1050898487 [======>.......................] - ETA: 2:49

```
</div>
---
## Load data

The files are provided in Nifti format with the extension .nii. To read the
scans, we use the `nibabel` package.
You can install the package via `pip install nibabel`.

To process the data, we do the following:

* We first rotate the volumes by 90 degrees, so the orientation is fixed
* We resize width, height and depth.

Here we define several helper functions to process the data. These functions
will be used when building training and validation datasets.


```python
import numpy as np
import nibabel as nib
import cv2

from scipy.ndimage import zoom


def read_nifti_file(filepath):
    """Read and load volume"""
    # read file
    scan = nib.load(filepath)
    # get raw data
    scan = scan.get_fdata()
    # rotate
    scan = np.rot90(np.array(scan))
    return scan


def resize_slices(img):
    """Resize width and height"""
    # resize all slices
    flatten = [
        cv2.resize(img[:, :, i], (128, 128), interpolation=cv2.INTER_CUBIC)
        for i in range(img.shape[-1])
    ]
    # stack along the z-axis
    img = np.array(np.dstack(flatten))
    return img


def resize_depth(img):
    """Resize across z-axis"""
    # set the desired depth
    desired_depth = 64
    # get current depth
    current_depth = img.shape[-1]
    # compute depth factor
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    # resize across z-axis
    img_new = zoom(img, (1, 1, depth_factor), mode="nearest")
    return img_new


def process_scan(path):
    """Read and resize volume"""
    # read scan
    volume = read_nifti_file(path)
    # resize width and height
    volume = resize_slices(volume)
    # resize across z-axis
    volume = resize_depth(volume)
    return volume

```

Let's read the paths of the CT scans from the class directories.


```python
# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]
# Folder "CT-1" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-1", x)
    for x in os.listdir("MosMedData/CT-1")
]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))
```

<div class="k-default-codeblock">
```
CT scans with normal lung tissue: 100
CT scans with abnormal lung tissue: 100

```
</div>
Let's visualize a CT scan and it's shape.


```python
import matplotlib.pyplot as plt

# Read a scan.
img = read_nifti_file(normal_scan_paths[15])
print("Dimension of the CT scan is:", img.shape)
plt.imshow(img[:, :, 15], cmap="gray")
```

<div class="k-default-codeblock">
```
Dimension of the CT scan is: (512, 512, 38)

<matplotlib.image.AxesImage at 0x7f1230297a20>

```
</div>
![png](/img/examples/vision/3D_image_classification/3D_image_classification_10_2.png)


Since a CT scan has many slices, let's visualize a montage of the slices.


```python

def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Display 20 slices from the CT scan.
# Here we visualize 20 slices, 2 rows and 10 columns
# adapt it according to your need.
plot_slices(2, 10, 512, 512, img[:, :, :20])
```


![png](/img/examples/vision/3D_image_classification/3D_image_classification_12_0.png)


---
## Build train and validation datasets
Read the scans from the class directories and assign labels. Downsample the scans to have
shape of 128x128x64.
Lastly, split the dataset into train and validation subsets.


```python
# Read and process the scans.
# Each scan is resized across width, height, and depth.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)
```

<div class="k-default-codeblock">
```
Number of samples in train and validation are 140 and 60.

```
</div>
---
## Preprocessing and data augmentation

CT scans store raw voxel intensity in Hounsfield units (HU). They range from
-1024 to above 2000 in this dataset. Above 400 are bones with different
radiointensity, so this is used as a higher bound. A threshold between
-1000 and 400 is commonly used to normalize CT scans. The CT scans are
also augmented by rotating and blurring. There are different kinds of
preprocessing and augmentation techniques out there, this example shows a few
simple ones to get started.


```python
import random

from scipy import ndimage
from scipy.ndimage import gaussian_filter


@tf.function
def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume = volume - min / max - min
    volume_min = tf.reduce_min(volume)
    volume_max = tf.reduce_max(volume)
    normalized_volume = (volume - volume_min) / (volume_max - volume_min)
    normalized_volume = tf.expand_dims(normalized_volume, axis=3)
    return normalized_volume


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float64)
    return augmented_volume


@tf.function
def blur(volume):
    """Blur the volume"""

    def scipy_blur(volume):
        # gaussian blur
        volume = gaussian_filter(volume, sigma=1)
        return volume

    augmented_volume = tf.numpy_function(scipy_blur, [volume], tf.float64)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating, blur and normalizing."""
    # rotate data
    volume = rotate(volume)
    # blur data
    volume = blur(volume)
    # normalize
    volume = normalize(volume)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only normalizing."""
    volume = normalize(volume)
    return volume, label

```

While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates or blurs the volume and finally normalizes
it to have values between 0 and 1. For the validation data, the volumes are only normalized.


```python
# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
```

Visualize an augmented CT scan.


```python
import matplotlib.pyplot as plt

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")

# Visualize montage of slices.
# 10 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])
```

<div class="k-default-codeblock">
```
Dimension of the CT scan is: (128, 128, 64, 1)

```
</div>
![png](/img/examples/vision/3D_image_classification/3D_image_classification_20_1.png)



![png](/img/examples/vision/3D_image_classification/3D_image_classification_20_2.png)


---
## Define a 3D convolutional neural network

To make the model easier to understand, we structure it into blocks.
The architecture of the 3D CNN used in this example
is based on [this paper](https://arxiv.org/abs/2007.13224).


```python

def get_model(width=128, height=128, depth=64):
    """build a 3D convolutional neural network model"""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()
```

<div class="k-default-codeblock">
```
Model: "3dcnn"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 128, 128, 64, 1)] 0         
_________________________________________________________________
conv3d (Conv3D)              (None, 126, 126, 62, 64)  1792      
_________________________________________________________________
max_pooling3d (MaxPooling3D) (None, 63, 63, 31, 64)    0         
_________________________________________________________________
batch_normalization (BatchNo (None, 63, 63, 31, 64)    256       
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 61, 61, 29, 64)    110656    
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 30, 30, 14, 64)    0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 30, 30, 14, 64)    256       
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 28, 28, 12, 128)   221312    
_________________________________________________________________
max_pooling3d_2 (MaxPooling3 (None, 14, 14, 6, 128)    0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 6, 128)    512       
_________________________________________________________________
conv3d_3 (Conv3D)            (None, 12, 12, 4, 256)    884992    
_________________________________________________________________
max_pooling3d_3 (MaxPooling3 (None, 6, 6, 2, 256)      0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 6, 6, 2, 256)      1024      
_________________________________________________________________
global_average_pooling3d (Gl (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               131584    
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 1,352,897
Trainable params: 1,351,873
Non-trainable params: 1,024
_________________________________________________________________

```
</div>
---
## Train model


```python
# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=10)

# Train the model, doing validation at the end of each epoch.
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
```

<div class="k-default-codeblock">
```
Epoch 1/100
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0231s vs `on_train_batch_end` time: 0.0455s). Check your callbacks.
70/70 - 15s - loss: 0.7226 - acc: 0.5214 - val_loss: 0.7367 - val_acc: 0.5000
Epoch 2/100
70/70 - 15s - loss: 0.6928 - acc: 0.5500 - val_loss: 0.7400 - val_acc: 0.5000
Epoch 3/100
70/70 - 15s - loss: 0.6758 - acc: 0.5929 - val_loss: 0.9405 - val_acc: 0.5000
Epoch 4/100
70/70 - 15s - loss: 0.7240 - acc: 0.5000 - val_loss: 0.7072 - val_acc: 0.5000
Epoch 5/100
70/70 - 15s - loss: 0.6943 - acc: 0.5214 - val_loss: 0.7686 - val_acc: 0.5000
Epoch 6/100
70/70 - 15s - loss: 0.6796 - acc: 0.6500 - val_loss: 0.7111 - val_acc: 0.5500
Epoch 7/100
70/70 - 15s - loss: 0.6764 - acc: 0.5286 - val_loss: 1.0334 - val_acc: 0.4833
Epoch 8/100
70/70 - 15s - loss: 0.6250 - acc: 0.6429 - val_loss: 0.8498 - val_acc: 0.5000
Epoch 9/100
70/70 - 15s - loss: 0.6483 - acc: 0.6000 - val_loss: 1.9208 - val_acc: 0.4833
Epoch 10/100
70/70 - 15s - loss: 0.6567 - acc: 0.6000 - val_loss: 1.1385 - val_acc: 0.4667
Epoch 11/100
70/70 - 15s - loss: 0.6444 - acc: 0.6214 - val_loss: 0.9513 - val_acc: 0.4500
Epoch 12/100
70/70 - 15s - loss: 0.5996 - acc: 0.6643 - val_loss: 1.1083 - val_acc: 0.4667
Epoch 13/100
70/70 - 15s - loss: 0.6515 - acc: 0.6286 - val_loss: 0.8650 - val_acc: 0.4333
Epoch 14/100
70/70 - 15s - loss: 0.6304 - acc: 0.6357 - val_loss: 0.7985 - val_acc: 0.5833
Epoch 15/100
70/70 - 15s - loss: 0.6293 - acc: 0.6429 - val_loss: 0.9456 - val_acc: 0.4833
Epoch 16/100
70/70 - 15s - loss: 0.6008 - acc: 0.6429 - val_loss: 0.8814 - val_acc: 0.5667
Epoch 17/100
70/70 - 15s - loss: 0.6184 - acc: 0.6214 - val_loss: 1.2379 - val_acc: 0.5333
Epoch 18/100
70/70 - 15s - loss: 0.6186 - acc: 0.6500 - val_loss: 1.2624 - val_acc: 0.5667
Epoch 19/100
70/70 - 15s - loss: 0.6292 - acc: 0.6571 - val_loss: 0.9592 - val_acc: 0.4833
Epoch 20/100
70/70 - 15s - loss: 0.6073 - acc: 0.6500 - val_loss: 1.2174 - val_acc: 0.4833
Epoch 21/100
70/70 - 15s - loss: 0.6238 - acc: 0.6571 - val_loss: 1.8992 - val_acc: 0.5000
Epoch 22/100
70/70 - 15s - loss: 0.5788 - acc: 0.7071 - val_loss: 0.9556 - val_acc: 0.5333
Epoch 23/100
70/70 - 15s - loss: 0.5416 - acc: 0.6929 - val_loss: 1.3784 - val_acc: 0.5000
Epoch 24/100
70/70 - 15s - loss: 0.6038 - acc: 0.6571 - val_loss: 1.5250 - val_acc: 0.5000

<tensorflow.python.keras.callbacks.History at 0x7f122d8dfc50>

```
</div>
It is important to note that the number of samples is very small (only 200) and we don't
specify a random seed. As such, you can expect significant variance in the results. The full dataset
which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
performance is observed in both cases.

---
## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.


```python
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
```


![png](/img/examples/vision/3D_image_classification/3D_image_classification_27_0.png)


---
## Make predictions on a single CT scan.


```python
# Load best weights.
model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )
```

<div class="k-default-codeblock">
```
This model is 100.00 percent confident that CT scan is normal
This model is 0.00 percent confident that CT scan is abnormal

```
</div>