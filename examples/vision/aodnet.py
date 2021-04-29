"""
Title: An All-in-One Network for Dehazing and Beyond
Author: [Soumik Rakshit](https://github.com/soumik12345)
Date created: 2021/04/10
Last modified: 2021/04/10
Description: Dehaze images with a lightweight deep convolutional neural network.
"""
"""
# Introduction

_Image dehazing_ — improving the image quality by removing the haze from images
captured in real-world weather conditions — is an important problem is computer
vision. It can be used not only to enhance photographs clicked in hazy
conditions, but also can be used to augment other computer vision systems, such
as _object detection_ and _semantic segmentation_ as a
pre-processing step. In this example, we implement
the [All-in-One Network (AOD-Net)](https://arxiv.org/abs/1707.06543v1) — a simple and
lightweight deep convolutional neural network model that can generate clean
images directly from hazy images. The end-to-end design of the AOD-Net
enables embedding the network into other neural network models for, such as
[Faster-RCNN](https://arxiv.org/abs/1506.01497v3) and
[YOLO](https://pjreddie.com/media/files/papers/yolo.pdf), as a pre-processing
step, making it easier for such model to detect and recognize objects from
clean images.
"""

"""shell
!nvidia-smi -L
"""

import os
import gdown
import numpy as np
from PIL import Image
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt

"""
# Load the NYU2 Depth Database

The experiment in the AOD-Net paper used synthesized hazy images from the
ground-truth images with depth meta-data from the indoor 
[NYU2 Depth Database](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
We will download this dataset from Google Drive using `gdown`.
"""

gdown.download(
    "https://drive.google.com/uc?id=1sInD9Ydq8-x7WwqehE0EyRknMdSFPOat",
    "Dehaze-NYU.zip",
    quiet=False,
)

"""shell
!unzip -qq Dehaze-NYU.zip
!rm ./Dehaze-NYU.zip
"""

"""
## Define the data transformation and augmentation functions

In order to ensure that the model is fed the data efficiently during training, we will be
use the `tf.data` API to create our data loading pipeline. Due to memory constraints for
training on Google Colab, we generate random crops of size `256` from corresponding
hazing and clean image pair. For the training dataset, we use also apply random
horizontal flips in order to augment the data.
"""


def get_image_file_list(dataset_path):
    original_image_files = []
    hazy_image_paths = sorted(
        glob(str(os.path.join(dataset_path, "train_images/*.jpg")))
    )
    for image_path in hazy_image_paths:
        image_file_name = image_path.split("/")[-1]
        original_file_name = (
            image_file_name.split("_")[0] + "_" + image_file_name.split("_")[1] + ".jpg"
        )
        original_image_files.append(
            str(os.path.join(dataset_path, "original_images/" + original_file_name))
        )
    return original_image_files, hazy_image_paths


def read_images(image_files):
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(
        lambda x: tf.image.decode_png(x, channels=3),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def apply_scaling(hazy_image, original_image):
    hazy_image = tf.cast(hazy_image, tf.float32)
    original_image = tf.cast(original_image, tf.float32)
    hazy_image = hazy_image / 255.0
    original_image = original_image / 255.0
    return hazy_image, original_image


def random_flip(hazy_image, original_image):
    return tf.cond(
        tf.random.uniform(shape=(), maxval=1) < 0.5,
        lambda: (hazy_image, original_image),
        lambda: (
            tf.image.flip_left_right(hazy_image),
            tf.image.flip_left_right(original_image),
        ),
    )


def random_crop(hazy_image, original_image, low_crop_size, enhanced_crop_size):
    hazy_image_shape = tf.shape(hazy_image)[:2]
    low_w = tf.random.uniform(
        shape=(), dtype=tf.int32, maxval=hazy_image_shape[1] - low_crop_size + 1
    )
    low_h = tf.random.uniform(
        shape=(), dtype=tf.int32, maxval=hazy_image_shape[0] - low_crop_size + 1
    )
    enhanced_w = low_w
    enhanced_h = low_h
    hazy_image_cropped = hazy_image[
        low_h : low_h + low_crop_size, low_w : low_w + low_crop_size
    ]
    original_image_cropped = original_image[
        enhanced_h : enhanced_h + enhanced_crop_size,
        enhanced_w : enhanced_w + enhanced_crop_size,
    ]
    return hazy_image_cropped, original_image_cropped


def configure_dataset(
    dataset, image_crop_size, buffer_size, batch_size, is_dataset_train
):
    dataset = dataset.map(apply_scaling, num_parallel_calls=tf.data.AUTOTUNE)
    if image_crop_size > 0:
        dataset = dataset.map(
            lambda hazy, original: random_crop(
                hazy, original, image_crop_size, image_crop_size
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if is_dataset_train:
        dataset = dataset.map(random_flip, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_dataset(dataset_path, image_crop_size, buffer_size, batch_size, val_split):
    original_images, hazy_images = get_image_file_list(dataset_path=dataset_path)
    assert len(original_images) == len(hazy_images)
    print("Total Number of Image Pairs:", len(original_images))
    hazy_dataset = read_images(hazy_images)
    original_dataset = read_images(original_images)
    dataset = tf.data.Dataset.zip((hazy_dataset, original_dataset))
    cardinality = tf.data.experimental.cardinality(dataset).numpy()
    train_dataset = configure_dataset(
        dataset=dataset.skip(int(cardinality * val_split)),
        image_crop_size=image_crop_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        is_dataset_train=True,
    )
    val_dataset = configure_dataset(
        dataset=dataset.take(int(cardinality * val_split)),
        image_crop_size=image_crop_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        is_dataset_train=True,
    )
    return train_dataset, val_dataset


train_dataset, val_dataset = build_dataset(
    dataset_path="./Dehazing",
    image_crop_size=256,
    buffer_size=1024,
    batch_size=16,
    val_split=0.1,
)
print(train_dataset)
print(val_dataset)

"""
## Define the AOD-Net model

Now we will define the AOD-Net model as a subclass of  `tf.keras.Model`. The
following diagrams from the original paper summarize its architecture:

### The AOD-Net model
![](https://i.imgur.com/nmnF0cY.png)

### The K-estimation model
![](https://i.imgur.com/dq5i4uz.png)
"""


class AODNet(tf.keras.Model):
    def __init__(self, name: str, stddev: float = 0.02, weight_decay: float = 1e-4):
        super(AODNet, self).__init__(name=name)
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_3 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=5,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_4 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=7,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_5 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.relu = tf.keras.layers.ReLU(max_value=1.0)

    def call(self, inputs, *args, **kwargs):
        conv_1 = self.conv_layer_1(inputs)
        conv_2 = self.conv_layer_2(conv_1)
        concat_1 = tf.concat([conv_1, conv_2], axis=-1)
        conv_3 = self.conv_layer_3(concat_1)
        concat_2 = tf.concat([conv_2, conv_3], axis=-1)
        conv_4 = self.conv_layer_4(concat_2)
        concat_3 = tf.concat([conv_1, conv_2, conv_3, conv_4], axis=-1)
        k = self.conv_layer_5(concat_3)
        j = k * inputs - k + 1.0
        output = self.relu(j)
        return output


"""
In this example, we are training the model using a single GPU (an NVIDIA Tesla
P100) and the`tf.distribute.OneDeviceStrategy`. This
[TensorFlow distribution strategy](https://www.tensorflow.org/tutorials/distribute/keras)
will place any variables created in its scope on the specified device. We will
define and compile our model in the scope of our distribution strategy, using
Adam as the optimizer, the Mean Squared Error
as the loss function, and the PSNR (Peak Signal Noise Ratio) as our metric.
**Note:** For training the AOD-Net in a multi-GPU environment, we can also use
`tf.distribute.MirroredStrategy` (go to the
[Distributed training](https://www.tensorflow.org/guide/distributed_training)
guide more information).
"""


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

with strategy.scope():
    model = AODNet(name='AODNet', stddev=0.02, weight_decay=1e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[peak_signal_noise_ratio]
    )

"""
# Training

We train the AOD-Net for 10 epochs. On an NVIDIA Tesla P100, each epoch
can take around 190 seconds, so the model may take around 30 minutes to train.
"""

training_history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

"""
## Plot of Training and Validation Loss
"""

plt.plot(training_history.history["loss"])
plt.plot(training_history.history["val_loss"])
plt.title("Loss Curve")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

"""
## Plot of Training and Validation PSNR
"""

plt.plot(training_history.history["peak_signal_noise_ratio"])
plt.plot(training_history.history["val_peak_signal_noise_ratio"])
plt.title("PSNR Curve")
plt.ylabel("PSNR")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

"""
# Inference

Next, let's define 2 simple utility functions for inferring from
an image and plotting the results:
"""


def infer(model, image_path):
    original_image = Image.open(image_path)
    image = tf.keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return original_image, prediction[0]


def plot_result(image1, image2, title1, title2):
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 2, 1).set_title(title1)
    _ = plt.imshow(image1)
    fig.add_subplot(1, 2, 2).set_title(title2)
    _ = plt.imshow(image2)
    plt.show()


"""
We use the sample real-world test images collected from
[https://github.com/soumik12345/AODNet](https://github.com/soumik12345/AODNet) for
testing our model.
"""

"""shell
!git clone https://github.com/soumik12345/AODNet
"""

for hazy_image_file in glob("./AODNet/assets/sample_test_images/*"):
    hazy_image, predicted_image = infer(model, hazy_image_file)
    plot_result(hazy_image, predicted_image, "Hazy Image", "Predicted Image")
