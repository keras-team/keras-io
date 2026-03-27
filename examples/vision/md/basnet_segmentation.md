# Highly accurate boundaries segmentation using BASNet

**Author:** [Hamid Ali](https://github.com/hamidriasat)<br>
**Date created:** 2023/05/30<br>
**Last modified:** 2025/01/24<br>
**Description:** Boundaries aware segmentation model trained on the DUTS dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/basnet_segmentation.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/basnet_segmentation.py)



---
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

---
## Download the Data

We will use the [DUTS-TE](http://saliencydetection.net/duts/) dataset for training. It has 5,019
images but we will use 140 for training and validation to save notebook running time. DUTS is
relatively large salient object segmentation dataset. which contain diversified textures and
structures common to real-world images in both foreground and background.


```python
import os

# Because of the use of tf.image.ssim in the loss,
# this example requires TensorFlow. The rest of the code
# is backend-agnostic.
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import keras_hub
import tensorflow as tf
import keras
from keras import layers, ops

keras.config.disable_traceback_filtering()
```

---
## Define Hyperparameters


```python
IMAGE_SIZE = 288
BATCH_SIZE = 4
OUT_CLASSES = 1
TRAIN_SPLIT_RATIO = 0.90
```

---
## Create `PyDataset`s

We will use `load_paths()` to load and split 140 paths into train and validation set, and
convert paths into `PyDataset` object.


```python
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
```

---
## Visualize Data


```python

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
```


    
![png](/img/examples/vision/basnet_segmentation/basnet_segmentation_9_0.png)
    


---
## Analyze Mask

Lets print unique values of above displayed mask. You can see despite belonging to one class, it's
intensity is changing between low(0) to high(255). This variation in intensity makes it hard for
network to generate good segmentation map for **salient or camouflaged object segmentation**.
Because of its Residual Refined Module (RMs), BASNet is good in generating highly accurate
boundaries and fine structures.


```python
print(f"Unique values count: {len(np.unique((mask[0] * 255)))}")
print("Unique values:")
print(np.unique((mask[0] * 255)).astype(int))
```

<div class="k-default-codeblock">
```
Unique values count: 245
Unique values:
[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  61  62  63  65  66  67  68  69  70  71  73  74
  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
  93  94  95  96  97  98  99 100 101 102 103 104 105 108 109 110 111 112
 113 114 115 116 117 118 119 120 122 123 124 125 128 129 130 131 132 133
 134 135 136 137 138 139 140 141 142 144 145 146 147 148 149 150 151 152
 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 170 171
 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189
 190 191 192 193 194 195 196 197 198 199 201 202 203 204 205 206 207 208
 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226
 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244
 245 246 247 248 249 250 251 252 253 254 255]

```
</div>
---
## Building the BASNet Model

BASNet comprises of a predict-refine architecture and a hybrid loss. The predict-refine
architecture consists of a densely supervised encoder-decoder network and a residual refinement
module, which are respectively used to predict and refine a segmentation probability map.

![](https://i.imgur.com/8jaZ2qs.png)


```python

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


def get_resnet_block(resnet, block_num):
    """Extract and return a ResNet-34 block."""
    extractor_levels = ["P2", "P3", "P4", "P5"]
    num_blocks = resnet.stackwise_num_blocks
    if block_num == 0:
        x = resnet.get_layer("pool1_pool").output
    else:
        x = resnet.pyramid_outputs[extractor_levels[block_num - 1]]
    y = resnet.get_layer(f"stack{block_num}_block{num_blocks[block_num]-1}_add").output
    return keras.models.Model(
        inputs=x,
        outputs=y,
        name=f"resnet_block{block_num + 1}",
    )

```

---
## Prediction Module

Prediction module is a heavy encoder decoder structure like U-Net. The encoder includes an input
convolutional layer and six stages. First four are adopted from ResNet-34 and rest are basic
res-blocks. Since first convolution and pooling layer of ResNet-34 is skipped so we will use
`get_resnet_block()` to extract first four blocks. Both bridge and decoder uses three
convolutional layers with side outputs. The module produces seven segmentation probability
maps during training, with the last one considered the final output.


```python

def basnet_predict(input_shape, out_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    filters = 64
    num_stages = 6

    x_input = layers.Input(input_shape)

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    resnet = keras_hub.models.ResNetBackbone(
        input_conv_filters=[64],
        input_conv_kernel_sizes=[7],
        stackwise_num_filters=[64, 128, 256, 512],
        stackwise_num_blocks=[3, 4, 6, 3],
        stackwise_num_strides=[1, 2, 2, 2],
        block_type="basic_block",
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

    return keras.models.Model(inputs=x_input, outputs=decoder_blocks)

```

---
## Residual Refinement Module

Refinement Modules (RMs), designed as a residual block aim to refines the coarse(blurry and noisy
boundaries) segmentation maps generated by prediction module. Similar to prediction module it's
also an encode decoder structure but with light weight 4 stages, each containing one
`convolutional block()` init. At the end it adds both coarse and residual output to generate
refined output.


```python

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

    return keras.models.Model(inputs=[base_model.input], outputs=[x])

```

---
## Combine Predict and Refinement Module


```python

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
        super().__init__(inputs=predict_model.input, outputs=output)

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

```

---
## Hybrid Loss

Another important feature of BASNet is its hybrid loss function, which is a combination of
binary cross entropy, structural similarity and intersection-over-union losses, which guide
the network to learn three-level (i.e., pixel, patch and map level) hierarchy representations.


```python

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
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "bas_net"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">1,792</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resnet_block1       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │    <span style="color: #00af00; text-decoration-color: #00af00">222,720</span> │ conv2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resnet_block2       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │  <span style="color: #00af00; text-decoration-color: #00af00">1,118,720</span> │ activation[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_1        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resnet_block3       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">6,829,056</span> │ activation_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_2        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">256</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resnet_block4       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │ <span style="color: #00af00; text-decoration-color: #00af00">13,121,536</span> │ activation_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_3        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ max_pooling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalization │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_4        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │ max_pooling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_5        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_6        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │ activation_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_7        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_8        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │ activation_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_9        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_1     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_9[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ max_pooling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_10       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_10[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ max_pooling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_11       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_11[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_9[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_12       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_12[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_10[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ activation_11[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_13       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_13[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_11[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_14       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,296</span> │ activation_14[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_12[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│                     │                   │            │ activation_13[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_15       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_15[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_13[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_16       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_16[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_14[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_17       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_15 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_17[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_15[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_18       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>,      │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_15[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)             │            │ activation_18[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">4,719,104</span> │ concatenate[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_16[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_19       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_17 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_19[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_17[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_20       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_20[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_18[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_21       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_21[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_1       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_9[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)             │            │ resizing[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_19 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">4,719,104</span> │ concatenate_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_19[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_22       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_22[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_20[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_23       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_23[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_21[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_24       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_1          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_24[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_2       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)             │            │ resizing_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">4,719,104</span> │ concatenate_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_22[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_25       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_23 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_25[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_23[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_26       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_26[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_24[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_27       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_2          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_27[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_3       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">768</span>)              │            │ resizing_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">3,539,456</span> │ concatenate_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_25[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_28       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_26 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_28[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_26[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_29       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_27 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_29[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_27[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_30       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_3          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_30[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_4       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">640</span>)              │            │ resizing_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_28 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │  <span style="color: #00af00; text-decoration-color: #00af00">2,949,632</span> │ concatenate_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_28[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_31       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_29 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_31[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_29[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_32       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_30 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_32[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_30[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_33       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_4          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_33[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_5       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resnet_block1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">576</span>)              │            │ resizing_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_31 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │  <span style="color: #00af00; text-decoration-color: #00af00">2,654,720</span> │ concatenate_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_31[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_34       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_32 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_34[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_32[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_35       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_33 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │  <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ activation_35[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv2d_33[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_36       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">512</span>)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_34 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">4,609</span> │ activation_36[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_5          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_34[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_41 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">640</span> │ resizing_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_42 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ conv2d_41[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_42[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_37       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_37[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_43 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ max_pooling2d_2[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_43[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_38       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_3     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_38[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_44 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ max_pooling2d_3[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_44[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_39       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_4     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_39[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_45 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ max_pooling2d_4[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_45[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_40       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_5     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_40[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_46 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ max_pooling2d_5[<span style="color: #00af00; text-decoration-color: #00af00">…</span> │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_46[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_41       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_12         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_41[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_6       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_40[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │ resizing_12[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_47 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">73,792</span> │ concatenate_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_47[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_42       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_13         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_42[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_7       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_39[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │ resizing_13[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_48 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │     <span style="color: #00af00; text-decoration-color: #00af00">73,792</span> │ concatenate_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_48[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_43       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>,    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_14         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_43[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_8       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_38[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │ resizing_14[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_49 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │     <span style="color: #00af00; text-decoration-color: #00af00">73,792</span> │ concatenate_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_49[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_44       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_15         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_44[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_9       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ activation_37[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │ <span style="color: #00af00; text-decoration-color: #00af00">128</span>)              │            │ resizing_15[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_50 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │     <span style="color: #00af00; text-decoration-color: #00af00">73,792</span> │ concatenate_9[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2d_50[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_45       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">64</span>)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_51 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │        <span style="color: #00af00; text-decoration-color: #00af00">577</span> │ activation_45[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_35 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>, <span style="color: #00af00; text-decoration-color: #00af00">144</span>,  │      <span style="color: #00af00; text-decoration-color: #00af00">4,609</span> │ activation_33[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_36 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">4,609</span> │ activation_30[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_37 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">4,609</span> │ activation_27[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_38 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">18</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>) │      <span style="color: #00af00; text-decoration-color: #00af00">4,609</span> │ activation_24[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_39 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │      <span style="color: #00af00; text-decoration-color: #00af00">4,609</span> │ activation_21[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_40 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">9</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │      <span style="color: #00af00; text-decoration-color: #00af00">4,609</span> │ activation_18[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>], │
│                     │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │ conv2d_51[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_6          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_35[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_7          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_36[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_8          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_37[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_9          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_38[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_10         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_39[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ resizing_11         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2d_40[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)          │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_46       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_47       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_48       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_49       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_50       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_51       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_9[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_52       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_10[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_53       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>, <span style="color: #00af00; text-decoration-color: #00af00">288</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ resizing_11[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">108,886,792</span> (415.37 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">108,834,952</span> (415.17 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">51,840</span> (202.50 KB)
</pre>



### Train the Model


```python
basnet_model.fit(train_dataset, validation_data=val_dataset, epochs=1)
```

    
  
<div class="k-default-codeblock">
```

 31/31 ━━━━━━━━━━━━━━━━━━━━ 38319s 1245s/step - activation_46_mae: 0.2864 - loss: 17.4035 - val_activation_46_mae: 0.8427 - val_loss: 238.6499

<keras.src.callbacks.history.History at 0x1312ff970>

```
</div>
### Visualize Predictions

In paper BASNet was trained on DUTS-TR dataset, which has 10553 images. Model was trained for 400k
iterations with a batch size of eight and without a validation dataset. After training model was
evaluated on DUTS-TE dataset and achieved a mean absolute error of `0.042`.

Since BASNet is a deep model and cannot be trained in a short amount of time which is a
requirement for keras example notebook, so we will load pretrained weights from [here](https://github.com/hamidriasat/BASNet/tree/basnet_keras)
to show model prediction. Due to computer power limitation this model was trained for 120k
iterations but it still demonstrates its capabilities. For further details about
trainings parameters please check given link.


```python
import gdown

gdown.download(id="1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg", output="basnet_weights.h5")


def normalize_output(prediction):
    max_value = np.max(prediction)
    min_value = np.min(prediction)
    return (prediction - min_value) / (max_value - min_value)


# Load weights.
basnet_model.load_weights("./basnet_weights.h5")
```

<div class="k-default-codeblock">
```
Downloading...
From (original): https://drive.google.com/uc?id=1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg
From (redirected): https://drive.google.com/uc?id=1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg&confirm=t&uuid=57f729b1-764a-4fbb-a569-c6eb5dd414ef
To: /Users/laxmareddyp/Desktop/Keras-IO/keras-io/scripts/tmp_4610020/basnet_weights.h5

Python(47905) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.

```
</div>
    
  0%|                                                                                                                      | 0.00/436M [00:00<?, ?B/s]

    
  0%|▎                                                                                                            | 1.05M/436M [00:00<00:42, 10.2MB/s]

    
  0%|▌                                                                                                            | 2.10M/436M [00:00<00:42, 10.1MB/s]

    
  1%|▉                                                                                                            | 3.67M/436M [00:00<00:40, 10.7MB/s]

    
  1%|█▎                                                                                                           | 5.24M/436M [00:00<00:37, 11.5MB/s]

    
  2%|█▋                                                                                                           | 6.82M/436M [00:00<00:36, 11.8MB/s]

    
  2%|██                                                                                                           | 8.39M/436M [00:00<00:33, 12.8MB/s]

    
  2%|██▍                                                                                                          | 9.96M/436M [00:00<00:31, 13.5MB/s]

    
  3%|███                                                                                                          | 12.1M/436M [00:00<00:28, 14.9MB/s]

    
  3%|███▋                                                                                                         | 14.7M/436M [00:01<00:24, 17.0MB/s]

    
  4%|████▎                                                                                                        | 17.3M/436M [00:01<00:22, 18.2MB/s]

    
  5%|████▉                                                                                                        | 19.9M/436M [00:01<00:21, 19.6MB/s]

    
  5%|█████▋                                                                                                       | 22.5M/436M [00:01<00:20, 20.3MB/s]

    
  6%|██████▎                                                                                                      | 25.2M/436M [00:01<00:19, 20.7MB/s]

    
  6%|██████▊                                                                                                      | 27.3M/436M [00:01<00:20, 20.3MB/s]

    
  7%|███████▍                                                                                                     | 29.9M/436M [00:01<00:19, 20.7MB/s]

    
  7%|███████▉                                                                                                     | 32.0M/436M [00:01<00:19, 20.5MB/s]

    
  8%|████████▋                                                                                                    | 34.6M/436M [00:01<00:19, 20.7MB/s]

    
  8%|█████████▏                                                                                                   | 36.7M/436M [00:02<00:19, 20.7MB/s]

    
  9%|█████████▋                                                                                                   | 38.8M/436M [00:02<00:19, 20.1MB/s]

    
  9%|██████████▎                                                                                                  | 41.4M/436M [00:02<00:18, 21.4MB/s]

    
 10%|███████████                                                                                                  | 44.0M/436M [00:02<00:18, 21.5MB/s]

    
 11%|███████████▋                                                                                                 | 46.7M/436M [00:02<00:18, 21.0MB/s]

    
 11%|████████████▎                                                                                                | 49.3M/436M [00:02<00:19, 20.1MB/s]

    
 12%|████████████▉                                                                                                | 51.9M/436M [00:02<00:18, 21.2MB/s]

    
 13%|█████████████▋                                                                                               | 54.5M/436M [00:02<00:18, 21.0MB/s]

    
 13%|██████████████▎                                                                                              | 57.1M/436M [00:03<00:18, 20.8MB/s]

    
 14%|██████████████▊                                                                                              | 59.2M/436M [00:03<00:18, 20.6MB/s]

    
 14%|███████████████▎                                                                                             | 61.3M/436M [00:03<00:18, 19.8MB/s]

    
 15%|███████████████▊                                                                                             | 63.4M/436M [00:03<00:19, 19.5MB/s]

    
 15%|████████████████▍                                                                                            | 65.5M/436M [00:03<00:18, 19.7MB/s]

    
 16%|█████████████████                                                                                            | 68.2M/436M [00:03<00:18, 20.2MB/s]

    
 16%|█████████████████▌                                                                                           | 70.3M/436M [00:03<00:17, 20.4MB/s]

    
 17%|██████████████████                                                                                           | 72.4M/436M [00:03<00:18, 20.1MB/s]

    
 17%|██████████████████▋                                                                                          | 75.0M/436M [00:03<00:18, 20.0MB/s]

    
 18%|███████████████████▍                                                                                         | 77.6M/436M [00:04<00:17, 20.1MB/s]

    
 18%|████████████████████                                                                                         | 80.2M/436M [00:04<00:16, 21.1MB/s]

    
 19%|████████████████████▋                                                                                        | 82.8M/436M [00:04<00:16, 21.1MB/s]

    
 20%|█████████████████████▎                                                                                       | 85.5M/436M [00:04<00:16, 21.4MB/s]

    
 20%|██████████████████████                                                                                       | 88.1M/436M [00:04<00:16, 21.0MB/s]

    
 21%|██████████████████████▋                                                                                      | 90.7M/436M [00:04<00:16, 21.1MB/s]

    
 21%|███████████████████████▎                                                                                     | 93.3M/436M [00:04<00:16, 21.4MB/s]

    
 22%|███████████████████████▉                                                                                     | 95.9M/436M [00:04<00:15, 21.5MB/s]

    
 23%|████████████████████████▋                                                                                    | 98.6M/436M [00:05<00:15, 21.6MB/s]

    
 23%|█████████████████████████▌                                                                                    | 101M/436M [00:05<00:16, 20.1MB/s]

    
 24%|██████████████████████████                                                                                    | 103M/436M [00:05<00:16, 19.8MB/s]

    
 24%|██████████████████████████▋                                                                                   | 106M/436M [00:05<00:16, 20.2MB/s]

    
 25%|███████████████████████████▏                                                                                  | 108M/436M [00:05<00:22, 14.3MB/s]

    
 25%|███████████████████████████▊                                                                                  | 110M/436M [00:05<00:21, 15.3MB/s]

    
 26%|████████████████████████████▎                                                                                 | 112M/436M [00:05<00:19, 16.2MB/s]

    
 26%|████████████████████████████▉                                                                                 | 115M/436M [00:06<00:17, 18.2MB/s]

    
 27%|█████████████████████████████▌                                                                                | 117M/436M [00:06<00:16, 19.2MB/s]

    
 28%|██████████████████████████████▎                                                                               | 120M/436M [00:06<00:15, 20.0MB/s]

    
 28%|██████████████████████████████▉                                                                               | 123M/436M [00:06<00:15, 20.5MB/s]

    
 29%|███████████████████████████████▌                                                                              | 125M/436M [00:06<00:15, 20.7MB/s]

    
 29%|████████████████████████████████▎                                                                             | 128M/436M [00:06<00:14, 21.1MB/s]

    
 30%|████████████████████████████████▉                                                                             | 131M/436M [00:06<00:14, 20.7MB/s]

    
 31%|█████████████████████████████████▌                                                                            | 133M/436M [00:06<00:14, 21.1MB/s]

    
 31%|██████████████████████████████████▎                                                                           | 136M/436M [00:07<00:14, 21.2MB/s]

    
 32%|██████████████████████████████████▉                                                                           | 138M/436M [00:07<00:13, 21.4MB/s]

    
 32%|███████████████████████████████████▌                                                                          | 141M/436M [00:07<00:13, 21.5MB/s]

    
 33%|████████████████████████████████████▏                                                                         | 144M/436M [00:07<00:13, 21.6MB/s]

    
 34%|████████████████████████████████████▉                                                                         | 146M/436M [00:07<00:13, 21.6MB/s]

    
 34%|█████████████████████████████████████▌                                                                        | 149M/436M [00:07<00:13, 21.7MB/s]

    
 35%|██████████████████████████████████████▏                                                                       | 152M/436M [00:07<00:13, 21.2MB/s]

    
 35%|██████████████████████████████████████▉                                                                       | 154M/436M [00:07<00:13, 20.8MB/s]

    
 36%|███████████████████████████████████████▌                                                                      | 157M/436M [00:08<00:13, 20.9MB/s]

    
 36%|████████████████████████████████████████                                                                      | 159M/436M [00:08<00:14, 19.0MB/s]

    
 37%|████████████████████████████████████████▌                                                                     | 161M/436M [00:08<00:14, 19.5MB/s]

    
 37%|█████████████████████████████████████████▏                                                                    | 163M/436M [00:08<00:14, 19.5MB/s]

    
 38%|█████████████████████████████████████████▋                                                                    | 165M/436M [00:08<00:13, 19.6MB/s]

    
 38%|██████████████████████████████████████████▏                                                                   | 167M/436M [00:08<00:13, 19.6MB/s]

    
 39%|██████████████████████████████████████████▊                                                                   | 170M/436M [00:08<00:13, 20.2MB/s]

    
 39%|███████████████████████████████████████████▍                                                                  | 172M/436M [00:08<00:13, 19.2MB/s]

    
 40%|███████████████████████████████████████████▉                                                                  | 174M/436M [00:08<00:13, 19.3MB/s]

    
 40%|████████████████████████████████████████████▍                                                                 | 176M/436M [00:09<00:13, 19.4MB/s]

    
 41%|████████████████████████████████████████████▉                                                                 | 178M/436M [00:09<00:13, 19.6MB/s]

    
 41%|█████████████████████████████████████████████▍                                                                | 180M/436M [00:09<00:13, 19.2MB/s]

    
 42%|██████████████████████████████████████████████                                                                | 182M/436M [00:09<00:13, 19.1MB/s]

    
 42%|██████████████████████████████████████████████▋                                                               | 185M/436M [00:09<00:13, 18.7MB/s]

    
 43%|███████████████████████████████████████████████▏                                                              | 187M/436M [00:09<00:12, 19.2MB/s]

    
 43%|███████████████████████████████████████████████▋                                                              | 189M/436M [00:09<00:12, 19.0MB/s]

    
 44%|████████████████████████████████████████████████▎                                                             | 191M/436M [00:09<00:12, 18.9MB/s]

    
 44%|████████████████████████████████████████████████▊                                                             | 193M/436M [00:09<00:13, 18.6MB/s]

    
 45%|█████████████████████████████████████████████████▎                                                            | 196M/436M [00:10<00:12, 18.6MB/s]

    
 45%|█████████████████████████████████████████████████▊                                                            | 198M/436M [00:10<00:12, 18.4MB/s]

    
 46%|██████████████████████████████████████████████████▌                                                           | 200M/436M [00:10<00:12, 19.5MB/s]

    
 46%|███████████████████████████████████████████████████                                                           | 202M/436M [00:10<00:11, 19.8MB/s]

    
 47%|███████████████████████████████████████████████████▋                                                          | 205M/436M [00:10<00:11, 20.4MB/s]

    
 48%|████████████████████████████████████████████████████▎                                                         | 208M/436M [00:10<00:10, 20.8MB/s]

    
 48%|████████████████████████████████████████████████████▉                                                         | 210M/436M [00:10<00:11, 19.2MB/s]

    
 49%|█████████████████████████████████████████████████████▋                                                        | 213M/436M [00:10<00:10, 21.9MB/s]

    
 49%|██████████████████████████████████████████████████████▎                                                       | 215M/436M [00:11<00:10, 21.8MB/s]

    
 50%|███████████████████████████████████████████████████████                                                       | 218M/436M [00:11<00:10, 21.8MB/s]

    
 51%|███████████████████████████████████████████████████████▋                                                      | 221M/436M [00:11<00:10, 21.4MB/s]

    
 51%|████████████████████████████████████████████████████████▎                                                     | 223M/436M [00:11<00:10, 20.7MB/s]

    
 52%|████████████████████████████████████████████████████████▉                                                     | 226M/436M [00:11<00:10, 20.9MB/s]

    
 52%|█████████████████████████████████████████████████████████▋                                                    | 229M/436M [00:11<00:09, 21.2MB/s]

    
 53%|██████████████████████████████████████████████████████████▎                                                   | 231M/436M [00:11<00:09, 21.0MB/s]

    
 54%|██████████████████████████████████████████████████████████▉                                                   | 234M/436M [00:11<00:09, 20.6MB/s]

    
 54%|███████████████████████████████████████████████████████████▌                                                  | 236M/436M [00:12<00:09, 20.1MB/s]

    
 55%|████████████████████████████████████████████████████████████                                                  | 238M/436M [00:12<00:09, 19.8MB/s]

    
 55%|████████████████████████████████████████████████████████████▌                                                 | 240M/436M [00:12<00:10, 19.6MB/s]

    
 56%|█████████████████████████████████████████████████████████████                                                 | 242M/436M [00:12<00:09, 19.4MB/s]

    
 56%|█████████████████████████████████████████████████████████████▋                                                | 244M/436M [00:12<00:09, 19.2MB/s]

    
 57%|██████████████████████████████████████████████████████████████▏                                               | 246M/436M [00:12<00:09, 19.4MB/s]

    
 57%|██████████████████████████████████████████████████████████████▋                                               | 249M/436M [00:12<00:09, 19.5MB/s]

    
 57%|███████████████████████████████████████████████████████████████▏                                              | 251M/436M [00:12<00:09, 19.4MB/s]

    
 58%|███████████████████████████████████████████████████████████████▋                                              | 253M/436M [00:12<00:09, 18.5MB/s]

    
 58%|████████████████████████████████████████████████████████████████▎                                             | 255M/436M [00:13<00:10, 18.1MB/s]

    
 59%|████████████████████████████████████████████████████████████████▊                                             | 257M/436M [00:13<00:09, 18.4MB/s]

    
 60%|█████████████████████████████████████████████████████████████████▍                                            | 260M/436M [00:13<00:09, 19.4MB/s]

    
 60%|██████████████████████████████████████████████████████████████████                                            | 262M/436M [00:13<00:08, 20.1MB/s]

    
 61%|██████████████████████████████████████████████████████████████████▊                                           | 265M/436M [00:13<00:08, 20.6MB/s]

    
 61%|███████████████████████████████████████████████████████████████████▍                                          | 267M/436M [00:13<00:08, 20.5MB/s]

    
 62%|████████████████████████████████████████████████████████████████████                                          | 270M/436M [00:13<00:07, 20.9MB/s]

    
 63%|████████████████████████████████████████████████████████████████████▊                                         | 273M/436M [00:13<00:07, 21.1MB/s]

    
 63%|█████████████████████████████████████████████████████████████████████▍                                        | 275M/436M [00:13<00:07, 21.4MB/s]

    
 64%|██████████████████████████████████████████████████████████████████████                                        | 278M/436M [00:14<00:07, 20.6MB/s]

    
 64%|██████████████████████████████████████████████████████████████████████▌                                       | 280M/436M [00:14<00:07, 20.4MB/s]

    
 65%|███████████████████████████████████████████████████████████████████████▏                                      | 282M/436M [00:14<00:08, 18.9MB/s]

    
 65%|███████████████████████████████████████████████████████████████████████▋                                      | 284M/436M [00:14<00:08, 18.7MB/s]

    
 66%|████████████████████████████████████████████████████████████████████████▏                                     | 286M/436M [00:14<00:07, 18.7MB/s]

    
 66%|████████████████████████████████████████████████████████████████████████▋                                     | 288M/436M [00:14<00:08, 18.0MB/s]

    
 67%|█████████████████████████████████████████████████████████████████████████▍                                    | 291M/436M [00:14<00:07, 18.8MB/s]

    
 67%|█████████████████████████████████████████████████████████████████████████▉                                    | 293M/436M [00:14<00:07, 18.6MB/s]

    
 68%|██████████████████████████████████████████████████████████████████████████▍                                   | 295M/436M [00:15<00:07, 19.2MB/s]

    
 68%|██████████████████████████████████████████████████████████████████████████▉                                   | 297M/436M [00:15<00:07, 19.0MB/s]

    
 69%|███████████████████████████████████████████████████████████████████████████▌                                  | 299M/436M [00:15<00:07, 18.7MB/s]

    
 69%|████████████████████████████████████████████████████████████████████████████                                  | 301M/436M [00:15<00:09, 14.1MB/s]

    
 70%|████████████████████████████████████████████████████████████████████████████▋                                 | 304M/436M [00:15<00:08, 16.0MB/s]

    
 70%|█████████████████████████████████████████████████████████████████████████████▎                                | 307M/436M [00:15<00:07, 17.5MB/s]

    
 71%|██████████████████████████████████████████████████████████████████████████████                                | 309M/436M [00:15<00:06, 18.7MB/s]

    
 72%|██████████████████████████████████████████████████████████████████████████████▋                               | 312M/436M [00:16<00:06, 19.6MB/s]

    
 72%|███████████████████████████████████████████████████████████████████████████████▎                              | 315M/436M [00:16<00:06, 20.2MB/s]

    
 73%|████████████████████████████████████████████████████████████████████████████████                              | 317M/436M [00:16<00:05, 20.3MB/s]

    
 73%|████████████████████████████████████████████████████████████████████████████████▌                             | 319M/436M [00:16<00:05, 19.9MB/s]

    
 74%|█████████████████████████████████████████████████████████████████████████████████                             | 321M/436M [00:16<00:05, 19.7MB/s]

    
 74%|█████████████████████████████████████████████████████████████████████████████████▌                            | 323M/436M [00:16<00:05, 19.5MB/s]

    
 75%|██████████████████████████████████████████████████████████████████████████████████                            | 326M/436M [00:16<00:05, 19.1MB/s]

    
 75%|██████████████████████████████████████████████████████████████████████████████████▋                           | 328M/436M [00:16<00:05, 19.1MB/s]

    
 76%|███████████████████████████████████████████████████████████████████████████████████▏                          | 330M/436M [00:16<00:05, 19.0MB/s]

    
 76%|███████████████████████████████████████████████████████████████████████████████████▊                          | 332M/436M [00:17<00:05, 19.8MB/s]

    
 77%|████████████████████████████████████████████████████████████████████████████████████▌                         | 335M/436M [00:17<00:04, 20.5MB/s]

    
 77%|█████████████████████████████████████████████████████████████████████████████████████                         | 337M/436M [00:17<00:04, 20.1MB/s]

    
 78%|█████████████████████████████████████████████████████████████████████████████████████▋                        | 340M/436M [00:17<00:04, 20.5MB/s]

    
 79%|██████████████████████████████████████████████████████████████████████████████████████▎                       | 342M/436M [00:17<00:04, 20.8MB/s]

    
 79%|███████████████████████████████████████████████████████████████████████████████████████                       | 345M/436M [00:17<00:04, 21.2MB/s]

    
 80%|███████████████████████████████████████████████████████████████████████████████████████▋                      | 348M/436M [00:17<00:04, 21.4MB/s]

    
 80%|████████████████████████████████████████████████████████████████████████████████████████▎                     | 350M/436M [00:17<00:04, 21.3MB/s]

    
 81%|█████████████████████████████████████████████████████████████████████████████████████████                     | 353M/436M [00:17<00:03, 21.6MB/s]

    
 82%|█████████████████████████████████████████████████████████████████████████████████████████▋                    | 355M/436M [00:18<00:03, 21.2MB/s]

    
 82%|██████████████████████████████████████████████████████████████████████████████████████████▎                   | 358M/436M [00:18<00:03, 21.9MB/s]

    
 83%|██████████████████████████████████████████████████████████████████████████████████████████▉                   | 361M/436M [00:18<00:03, 21.8MB/s]

    
 83%|███████████████████████████████████████████████████████████████████████████████████████████▋                  | 363M/436M [00:18<00:03, 21.6MB/s]

    
 84%|████████████████████████████████████████████████████████████████████████████████████████████▎                 | 366M/436M [00:18<00:03, 21.0MB/s]

    
 85%|████████████████████████████████████████████████████████████████████████████████████████████▉                 | 369M/436M [00:18<00:03, 19.9MB/s]

    
 85%|█████████████████████████████████████████████████████████████████████████████████████████████▍                | 371M/436M [00:18<00:03, 19.4MB/s]

    
 85%|██████████████████████████████████████████████████████████████████████████████████████████████                | 373M/436M [00:18<00:03, 18.8MB/s]

    
 86%|██████████████████████████████████████████████████████████████████████████████████████████████▌               | 375M/436M [00:19<00:03, 18.9MB/s]

    
 86%|███████████████████████████████████████████████████████████████████████████████████████████████               | 377M/436M [00:19<00:03, 18.6MB/s]

    
 87%|███████████████████████████████████████████████████████████████████████████████████████████████▌              | 379M/436M [00:19<00:03, 18.3MB/s]

    
 87%|████████████████████████████████████████████████████████████████████████████████████████████████▏             | 381M/436M [00:19<00:02, 18.6MB/s]

    
 88%|████████████████████████████████████████████████████████████████████████████████████████████████▋             | 383M/436M [00:19<00:02, 18.8MB/s]

    
 88%|█████████████████████████████████████████████████████████████████████████████████████████████████▏            | 385M/436M [00:19<00:02, 18.9MB/s]

    
 89%|█████████████████████████████████████████████████████████████████████████████████████████████████▋            | 387M/436M [00:19<00:02, 19.1MB/s]

    
 89%|██████████████████████████████████████████████████████████████████████████████████████████████████▍           | 390M/436M [00:19<00:02, 19.8MB/s]

    
 90%|███████████████████████████████████████████████████████████████████████████████████████████████████           | 393M/436M [00:20<00:02, 20.5MB/s]

    
 91%|███████████████████████████████████████████████████████████████████████████████████████████████████▋          | 395M/436M [00:20<00:01, 20.9MB/s]

    
 91%|████████████████████████████████████████████████████████████████████████████████████████████████████▏         | 397M/436M [00:20<00:01, 20.2MB/s]

    
 92%|████████████████████████████████████████████████████████████████████████████████████████████████████▉         | 400M/436M [00:20<00:01, 20.7MB/s]

    
 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████▍        | 402M/436M [00:20<00:01, 19.9MB/s]

    
 93%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉        | 404M/436M [00:20<00:02, 15.0MB/s]

    
 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████▍       | 406M/436M [00:20<00:01, 15.7MB/s]

    
 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████▏      | 409M/436M [00:20<00:01, 17.3MB/s]

    
 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████▊      | 412M/436M [00:21<00:01, 18.3MB/s]

    
 95%|████████████████████████████████████████████████████████████████████████████████████████████████████████▍     | 414M/436M [00:21<00:01, 18.6MB/s]

    
 96%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▏    | 417M/436M [00:21<00:00, 19.7MB/s]

    
 96%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▊    | 419M/436M [00:21<00:00, 20.3MB/s]

    
 97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 422M/436M [00:21<00:00, 20.7MB/s]

    
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████   | 425M/436M [00:21<00:00, 21.0MB/s]

    
 98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▊  | 427M/436M [00:21<00:00, 21.3MB/s]

    
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 430M/436M [00:21<00:00, 20.5MB/s]

    
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████ | 433M/436M [00:22<00:00, 20.8MB/s]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 435M/436M [00:22<00:00, 21.1MB/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 436M/436M [00:22<00:00, 19.6MB/s]

    


### Make Predictions


```python
for (image, mask), _ in zip(val_dataset, range(1)):
    pred_mask = basnet_model.predict(image)
    display([image[0], mask[0], normalize_output(pred_mask[0][0])])
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 32s 32s/step



    
![png](/img/examples/vision/basnet_segmentation/basnet_segmentation_27_2.png)
    


---
## Relevant Chapters from Deep Learning with Python
- [Chapter 11: Image segmentation](https://deeplearningwithpython.io/chapters/chapter11_image-segmentation)
