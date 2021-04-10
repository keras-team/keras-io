"""
Title: An All-in-One Network for Dehazing and Beyond
Author: [Soumik Rakshit](https://github.com/soumik12345)
Date created: 2021/04/10
Last modified: 2021/04/10
Description: Train a lightweight Deep CNN for image denhazing.
"""
"""
# Introduction
Image Dehazing is an important problem is Computer Vision. It can not only be used to
enhance photographs clicked in hazy conditions, but also can be used to augment other
computer vision systems such as Object Detection and Seantic Segmentation as a
pre-processing step. In this example, we implement
[AODNet](https://arxiv.org/abs/1707.06543v1), a simple and light-weight Deep CNN model
that directly generates a clean image from a hazy image. The end-to-end design of AODNet
makes it easy to embed AOD-Net into other deep models such as Faster-RCNN and YOLO as a
pre-processing step, making it easier for such model to detect and recognize objects from
a clean image.
"""

"""shell
!nvidia-smi -L
"""

import os
import gdown
import numpy as np
from PIL import Image
from glob import glob
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt

"""
# Downloading Dataset
The authors of the AODNet paper have created synthesized hazy images using the
groundtruth images with depth meta-data from the indoor [NYU2
Depth Database](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). We will
download this dataset from Google Drive using `gdown`.
"""

gdown.download(
    "https://drive.google.com/uc?id=1sInD9Ydq8-x7WwqehE0EyRknMdSFPOat",
    "Dehaze-NYU.zip",
    quiet=False,
)

os.system("unzip -qq Dehaze-NYU.zip")
os.system("rm ./Dehaze-NYU.zip")

"""
## Data Loader
In order to ensure that the model is fed the data efficiently during training, we will be
use the `tf.data` API to create our data loading pipeline. Due to memory constraints for
training on Google Colab, we generate random crops of size `256` from corresponding
hazing and clean image pair. For the training dataset, we use also apply random rotation
and random horizontal flips in order to augment the data.
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


def random_flip(hazy_image, original_image):
    return tf.cond(
        tf.random.uniform(shape=(), maxval=1) < 0.5,
        lambda: (hazy_image, original_image),
        lambda: (
            tf.image.flip_left_right(hazy_image),
            tf.image.flip_left_right(original_image),
        ),
    )


def random_rotate(hazy_image, original_image):
    condition = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    hazy_image = tf.image.rot90(hazy_image, condition)
    original_image = tf.image.rot90(original_image, condition)
    return hazy_image, original_image


def apply_scaling(hazy_image, original_image):
    hazy_image = tf.cast(hazy_image, tf.float32)
    original_image = tf.cast(original_image, tf.float32)
    hazy_image = hazy_image / 255.0
    original_image = original_image / 255.0
    return hazy_image, original_image


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
        dataset = dataset.map(random_rotate, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(random_flip, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


class DeHazeDataLoader:
    def __init__(self, dataset_path):
        (self.original_images, self.hazy_images) = get_image_file_list(
            dataset_path=dataset_path
        )

    def __len__(self):
        assert len(self.hazy_images) == len(self.original_images)
        return len(self.hazy_images)

    def build_dataset(self, image_crop_size, buffer_size, batch_size, val_split):
        hazy_dataset = read_images(self.hazy_images)
        original_dataset = read_images(self.original_images)
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


dataloader = DeHazeDataLoader("./Dehazing")
print("Total Number of Image Pairs:", len(dataloader))
train_dataset, val_dataset = dataloader.build_dataset(
    image_crop_size=256, buffer_size=1024, batch_size=16, val_split=0.1
)
print(train_dataset)
print(val_dataset)

"""
## Building AODNet Model
Now we will build AODNet a subclass of  `tf.keras.Model`. The following diagrams taken
from the paper summarizes the architecture of the AODNet model.
### The AODNet Model
![](https://i.imgur.com/nmnF0cY.png)
### K-estimation Model
![](https://i.imgur.com/dq5i4uz.png)
"""


class AODNet(tf.keras.Model):
    def __init__(self, name: str, stddev: float = 0.02, weight_decay: float = 1e-4):
        super(AODNet, self).__init__(name=name)
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_3 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=5,
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_4 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=7,
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.initializers.random_normal(stddev=stddev),
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
        self.conv_layer_5 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.initializers.random_normal(stddev=stddev),
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
        j = tf.math.multiply(k, inputs) - k + 1.0
        output = self.relu(j)
        return output


"""
Since we are training the model using a single GPU, i.e, Nvidia Tesla P100, we would use
`tf.distribute.OneDeviceStrategy`, using this strategy will place any variables created
in its scope on the specified device. We would define and compile our model in the scope
of our distribution strategy. We would be using Adam as out optimizer, Mean Squared Error
as the loss function and PSNR (Peak Signal Noise Ratio) as our metric.
**Note:** For training AODNet in a multi-GPU environment,
`tf.distribute.MirroredStrategy` can be used as a distribution stratgey.
"""


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
with strategy.scope():
    model = AODNet(name="AODNet", stddev=0.02, weight_decay=1e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[peak_signal_noise_ratio],
    )
    model.build((1, 256, 256, 3))

model.summary()

"""
# Training
We train AODNet for 10 epochs, on Nvidia Tesla P100, an epoch takes around 190 seconds,
so the model takes around 30 minutes to train
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
We define 2 simple utility functions for inferring from an image and plotting the results
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

os.system("git clone https://github.com/soumik12345/AODNet")

for hazy_image_file in glob("./AODNet/assets/sample_test_images/*"):
    hazy_image, predicted_image = infer(model, hazy_image_file)
    plot_result(hazy_image, predicted_image, "Hazy Image", "Predicted Image")
