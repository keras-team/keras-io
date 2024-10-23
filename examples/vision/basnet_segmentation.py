"""
Title: Highly accurate boundaries segmentation using BASNet
Author: [Hamid Ali](https://github.com/hamidriasat)
Date created: 2023/05/30
Last modified: 2024/10/02
Description: Boundaries aware segmentation model trained on the DUTS dataset.
Accelerator: GPU
"""

"""
## Introduction

Deep semantic segmentation algorithms have improved a lot recently, but still fails to correctly
predict pixels around object boundaries. In this example we implement
**Boundary-Aware Segmentation Network (BASNet)**, using two stage predict and refine
architecture, and a hybrid loss it can predict highly accurate boundaries and fine structures
for image segmentation.

### References:

- [Boundary-Aware Segmentation Network for Mobile and Web Applications](https://arxiv.org/abs/2101.04704)
- [BASNet Keras Implementation](https://github.com/hamidriasat/BASNet/tree/basnet_keras)
- [Learning to Detect Salient Objects with Image-level Supervision](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_Learning_to_Detect_CVPR_2017_paper.html)
"""

"""
## Download the Data

We will use the [DUTS-TE](http://saliencydetection.net/duts/) dataset for training. It has 5,019
images but we will use 140 for training and validation to save notebook running time. DUTS is
relatively large salient object segmentation dataset. which contain diversified textures and
structures common to real-world images in both foreground and background.
"""

import os

# Because of the use of tf.image.ssim in the loss,
# this example requires TensorFlow. The rest of the code
# is backend-agnostic.
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import keras_cv
import tensorflow as tf
import keras
from keras import layers, ops

keras.config.disable_traceback_filtering()

"""
## Define Hyperparameters
"""

IMAGE_SIZE = 288
BATCH_SIZE = 4
OUT_CLASSES = 1
TRAIN_SPLIT_RATIO = 0.90

"""
## Create `PyDataset`s

We will use `load_paths()` to load and split 140 paths into train and validation set, and
convert paths into `PyDataset` object.
"""

data_dir = keras.utils.get_file(
    origin="http://saliencydetection.net/duts/download/DUTS-TE.zip",
    extract=True,
)
data_dir = os.path.join(data_dir, "DUTS-TE")


def load_paths(path, split_ratio):
    images = sorted(glob(os.path.join(path, "DUTS-TE-Image/*")))[:140]
    masks = sorted(glob(os.path.join(path, "DUTS-TE-Mask/*")))[:140]
    len_ = int(len(images) * split_ratio)
    return (images[:len_], masks[:len_]), (images[len_:], masks[len_:])


class Dataset(keras.utils.PyDataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        img_size,
        out_classes,
        batch,
        shuffle=True,
        **kwargs,
    ):
        if shuffle:
            perm = np.random.permutation(len(image_paths))
            image_paths = [image_paths[i] for i in perm]
            mask_paths = [mask_paths[i] for i in perm]
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.out_classes = out_classes
        self.batch_size = batch
        super().__init__(*kwargs)

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            x, y = self.preprocess(
                self.image_paths[i],
                self.mask_paths[i],
                self.img_size,
            )
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.stack(batch_x, axis=0)
        batch_y = np.stack(batch_y, axis=0)
        return batch_x, batch_y

    def read_image(self, path, size, mode):
        x = keras.utils.load_img(path, target_size=size, color_mode=mode)
        x = keras.utils.img_to_array(x)
        x = (x / 255.0).astype(np.float32)
        return x

    def preprocess(self, x_batch, y_batch, img_size):
        images = self.read_image(x_batch, (img_size, img_size), mode="rgb")  # image
        masks = self.read_image(y_batch, (img_size, img_size), mode="grayscale")  # mask
        return images, masks


train_paths, val_paths = load_paths(data_dir, TRAIN_SPLIT_RATIO)

train_dataset = Dataset(
    train_paths[0], train_paths[1], IMAGE_SIZE, OUT_CLASSES, BATCH_SIZE, shuffle=True
)
val_dataset = Dataset(
    val_paths[0], val_paths[1], IMAGE_SIZE, OUT_CLASSES, BATCH_SIZE, shuffle=False
)

"""
## Visualize Data
"""


def display(display_list):
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]), cmap="gray")
        plt.axis("off")
    plt.show()


for image, mask in val_dataset:
    display([image[0], mask[0]])
    break

"""
## Analyze Mask

Lets print unique values of above displayed mask. You can see despite belonging to one class, it's
intensity is changing between low(0) to high(255). This variation in intensity makes it hard for
network to generate good segmentation map for **salient or camouflaged object segmentation**.
Because of its Residual Refined Module (RMs), BASNet is good in generating highly accurate
boundaries and fine structures.
"""

print(f"Unique values count: {len(np.unique((mask[0] * 255)))}")
print("Unique values:")
print(np.unique((mask[0] * 255)).astype(int))

"""
## Building the BASNet Model

BASNet comprises of a predict-refine architecture and a hybrid loss. The predict-refine
architecture consists of a densely supervised encoder-decoder network and a residual refinement
module, which are respectively used to predict and refine a segmentation probability map.

![](https://i.imgur.com/8jaZ2qs.png)
"""


def basic_block(x_input, filters, stride=1, down_sample=None, activation=None):
    """Creates a residual(identity) block with two 3*3 convolutions."""
    residual = x_input

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(
        x_input
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="same", use_bias=False)(
        x
    )
    x = layers.BatchNormalization()(x)

    if down_sample is not None:
        residual = down_sample

    x = layers.Add()([x, residual])

    if activation is not None:
        x = layers.Activation(activation)(x)

    return x


def convolution_block(x_input, filters, dilation=1):
    """Apply convolution + batch normalization + relu layer."""
    x = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def segmentation_head(x_input, out_classes, final_size):
    """Map each decoder stage output to model output classes."""
    x = layers.Conv2D(out_classes, kernel_size=(3, 3), padding="same")(x_input)

    if final_size is not None:
        x = layers.Resizing(final_size[0], final_size[1])(x)

    return x


def get_resnet_block(_resnet, block_num):
    """Extract and return ResNet-34 block."""
    resnet_layers = [3, 4, 6, 3]  # ResNet-34 layer sizes at different block.
    return keras.models.Model(
        inputs=_resnet.get_layer(f"v2_stack_{block_num}_block1_1_conv").input,
        outputs=_resnet.get_layer(
            f"v2_stack_{block_num}_block{resnet_layers[block_num]}_add"
        ).output,
        name=f"resnet34_block{block_num + 1}",
    )


"""
## Prediction Module

Prediction module is a heavy encoder decoder structure like U-Net. The encoder includes an input
convolutional layer and six stages. First four are adopted from ResNet-34 and rest are basic
res-blocks. Since first convolution and pooling layer of ResNet-34 is skipped so we will use
`get_resnet_block()` to extract first four blocks. Both bridge and decoder uses three
convolutional layers with side outputs. The module produces seven segmentation probability
maps during training, with the last one considered the final output.
"""


def basnet_predict(input_shape, out_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    filters = 64
    num_stages = 6

    x_input = layers.Input(input_shape)

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    resnet = keras_cv.models.ResNet34Backbone(
        include_rescaling=False,
    )

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet-34 blocks.
            x = get_resnet_block(resnet, i)(x)
            encoder_blocks.append(x)
            x = layers.Activation("relu")(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            shape = x.shape
            x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)

        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block, out_classes, input_shape[:2])
        for decoder_block in decoder_blocks
    ]

    return keras.models.Model(inputs=[x_input], outputs=decoder_blocks)


"""
## Residual Refinement Module

Refinement Modules (RMs), designed as a residual block aim to refines the coarse(blurry and noisy
boundaries) segmentation maps generated by prediction module. Similar to prediction module it's
also an encode decoder structure but with light weight 4 stages, each containing one
`convolutional block()` init. At the end it adds both coarse and residual output to generate
refined output.
"""


def basnet_rrm(base_model, out_classes):
    """BASNet Residual Refinement Module(RRM) module, output fine label map."""
    num_stages = 4
    filters = 64

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        shape = x.shape
        x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)
        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters)

    x = segmentation_head(x, out_classes, None)  # Segmentation head.

    # ------------- refined = coarse + residual
    x = layers.Add()([x_input, x])  # Add prediction + refinement output

    return keras.models.Model(inputs=base_model.input[0], outputs=x)


"""
## Combine Predict and Refinement Module
"""


class BASNet(keras.Model):
    def __init__(self, input_shape, out_classes):
        """BASNet, it's a combination of two modules
        Prediction Module and Residual Refinement Module(RRM)."""

        # Prediction model.
        predict_model = basnet_predict(input_shape, out_classes)
        # Refinement model.
        refine_model = basnet_rrm(predict_model, out_classes)

        output = refine_model.outputs  # Combine outputs.
        output.extend(predict_model.output)

        # Activations.
        output = [layers.Activation("sigmoid")(x) for x in output]
        super().__init__(inputs=predict_model.input[0], outputs=output)

        self.smooth = 1.0e-9
        # Binary Cross Entropy loss.
        self.cross_entropy_loss = keras.losses.BinaryCrossentropy()
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        # Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = ops.sum(ops.abs(y_true * y_pred), axis=[1, 2, 3])
        union = ops.sum(y_true, [1, 2, 3]) + ops.sum(y_pred, [1, 2, 3])
        union = union - intersection
        return ops.mean((intersection + self.smooth) / (union + self.smooth), axis=0)

    def compute_loss(self, x, y_true, y_pred, sample_weight=None, training=False):
        total = 0.0
        for y_pred_i in y_pred:  # y_pred = refine_model.outputs + predict_model.output
            cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred_i)

            ssim_value = self.ssim_value(y_true, y_pred, max_val=1)
            ssim_loss = ops.mean(1 - ssim_value + self.smooth, axis=0)

            iou_value = self.iou_value(y_true, y_pred)
            iou_loss = 1 - iou_value

            # Add all three losses.
            total += cross_entropy_loss + ssim_loss + iou_loss
        return total


"""
## Hybrid Loss

Another important feature of BASNet is its hybrid loss function, which is a combination of
binary cross entropy, structural similarity and intersection-over-union losses, which guide
the network to learn three-level (i.e., pixel, patch and map level) hierarchy representations.
"""


basnet_model = BASNet(
    input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], out_classes=OUT_CLASSES
)  # Create model.
basnet_model.summary()  # Show model summary.

optimizer = keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
# Compile model.
basnet_model.compile(
    optimizer=optimizer,
    metrics=[keras.metrics.MeanAbsoluteError(name="mae") for _ in basnet_model.outputs],
)

"""
### Train the Model
"""

basnet_model.fit(train_dataset, validation_data=val_dataset, epochs=1)

"""
### Visualize Predictions

In paper BASNet was trained on DUTS-TR dataset, which has 10553 images. Model was trained for 400k
iterations with a batch size of eight and without a validation dataset. After training model was
evaluated on DUTS-TE dataset and achieved a mean absolute error of `0.042`.

Since BASNet is a deep model and cannot be trained in a short amount of time which is a
requirement for keras example notebook, so we will load pretrained weights from [here](https://github.com/hamidriasat/BASNet/tree/basnet_keras)
to show model prediction. Due to computer power limitation this model was trained for 120k
iterations but it still demonstrates its capabilities. For further details about
trainings parameters please check given link.
"""

"""shell
!gdown 1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg
"""


def normalize_output(prediction):
    max_value = np.max(prediction)
    min_value = np.min(prediction)
    return (prediction - min_value) / (max_value - min_value)


# Load weights.
basnet_model.load_weights("./basnet_weights.h5")

"""
### Make Predictions
"""

for (image, mask), _ in zip(val_dataset, range(1)):
    pred_mask = basnet_model.predict(image)
    display([image[0], mask[0], normalize_output(pred_mask[0][0])])
