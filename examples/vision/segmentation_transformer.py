"""
Title: Semantic segmentation with SEgmentation TRansformer
Author: [ZhiYong Chang](https://github.com/czy00000)
Date created: 2021/11/3
Last modified: 2021/11/3
Description: Implementing Transformer model for semantic segmentation.
"""


"""
# Introduction

Most recent semantic segmentation methods adopt
a fully-convolutional network (FCN) with an encoder-decoder architecture.
This example implements [Segmentation
Transformer](https://arxiv.org/abs/2012.15840).
The authors provide an alternative perspective by treating semantic segmentation as a
sequence-to-sequence prediction task. **SE**gmentation **TR**ansformer(SETR) use a pure
transformer(without convolution and resolution reduction) to encode an image as a
sequence of patches.

This example requires TensorFlow 2.5 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview) package,
which can be installed using the following command:
```python
pip install -U tensorflow-addons
```
"""

"""
# setup
"""

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


"""
## Configure the hyperparameters
"""

weight_decay = 0.0001
learning_rate = 0.001
num_epochs = 50

"""
# Prepare the data

We will use the People-clothing-segmentation Dataset for training our model.
You can download it in this
[link](https://github.com/czy00000/datasets/releases/download/semantic-segmentation/cloth.zip).
The dataset contains 1000 images and 1000 corresponding semantic segmentation masks
each of size 825 pixels by 550 pixels in PNG format. The segmentation masks belong to
59 classes, the first being the background of individuals, and the rest belong to
58 clothing classes such as shirt, hair, pants, skin, shoes, glasses and so on.
We will create a tensorflow dataset.
"""

# code reference https://keras.io/examples/vision/deeplabv3_plus/

image_size = 256
batch_size = 2
num_classes = 59
data_dir = "./cloth"
num_train_images = 900
num_val_images = 100

train_images = sorted(glob(os.path.join(data_dir, "png_images/IMAGES/*")))[
    :num_train_images
]
train_masks = sorted(glob(os.path.join(data_dir, "png_masks/MASKS/*")))[
    :num_train_images
]
val_images = sorted(glob(os.path.join(data_dir, "png_images/IMAGES/*")))[
    num_train_images : num_train_images + num_val_images
]
val_masks = sorted(glob(os.path.join(data_dir, "png_masks/MASKS/*")))[
    num_train_images : num_train_images + num_val_images
]


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[image_size, image_size])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[image_size, image_size])
        image = image / 255
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

"""
## Implement the patch extraction and encoding layer
"""


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size, self.patch_size, 1),
            strides=(1, self.patch_size, self.patch_size, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


"""
## Implement the MLP block
"""


def mlp(x, hidden_dim, out_dim, drop=0):
    x = layers.Dense(hidden_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(out_dim)(x)
    return x


"""
## Implement the Transformer block
"""


def transformer_layer(
    x, dim, num_heads, mlp_ratio=4, attention_dropout=0, projection_dropout=0
):
    residual_1 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim, dropout=attention_dropout
    )(x, x)
    x = layers.add([x, residual_1])
    residual_2 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = mlp(x, dim * mlp_ratio, dim)
    x = layers.add([x, residual_2])
    return x


"""
# Implement the Segmentation Transformer

SEgmentation TRansformer (SETR) first split an image into fixed-size patches,
linearly embed each of them, add position embeddings, and feed the resulting
sequence of vectors to a standard Transformer encoder. To
perform pixel-wise segmentation, the authors introduce different decoder designs.

**(1)Naive upsampling**: This naive decoder first
projects the transformer feature `Z` to the dimension of
category number. The authors adopt a simple 2-layer network with architecture.
After that, they simply bilinearly upsample the output to the full image resolution.

**(2)Progressive upsampling**: The authors adopt a progressive upsampling strategy
that alternates conv layers and upsampling operations. To maximally mitigate
the adversarial effect, they restrict upsampling to 2×. Hence,
a total of 4 operations are needed for reaching the full resolution from the transformer
feature `Z`.

**(3)Multi-Level feature Aggregation**: The authors select some specific transformer
layer and
reshape to 3D feature map. To enhance the interactions across different streams, they
introduce
a top-down aggregation design via element-wise addition after the first layer. An
additional
`3 × 3 conv` is applied after the element-wise additioned feature. After the third layer,
they
obtain the fused feature from all the streams via channel-wise concatenation which is then
bilinearly upsampled 4× to the full resolution.

For convenience, we use the second decoder structure.
"""


def segmentation_transformer(
    num_transformer_layer=8,
    image_size=image_size,
    patch_size=16,
    num_heads=6,
    dim=128,
    num_classes=num_classes,
):
    inputs = layers.Input(shape=(image_size, image_size, 3))
    num_patches = (image_size // patch_size) ** 2
    x = PatchExtract(patch_size)(inputs)
    x = PatchEmbedding(num_patches, dim)(x)
    # encoder
    for _ in range(num_transformer_layer):
        x = transformer_layer(
            x,
            dim,
            num_heads,
            mlp_ratio=4,
            attention_dropout=0.2,
            projection_dropout=0.2,
        )
    x = tf.reshape(
        x, shape=(-1, image_size // patch_size, image_size // patch_size, dim)
    )
    # decoder
    x = layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", activation="relu")(
        x
    )
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(
        x
    )  # x-->(H/8, W/8, dim)

    x = layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", activation="relu")(
        x
    )
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(
        x
    )  # x-->(H/4, W/4, dim)

    x = layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", activation="relu")(
        x
    )
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(
        x
    )  # x-->(H/2, W/2, dim)

    x = layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", activation="relu")(
        x
    )
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # x-->(H, W, dim)

    outputs = layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


"""
# Train on People-Cloth-Segmentation
"""

model = segmentation_transformer()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=num_epochs,
)

"""
### Let's visualize the training progress of the model.
"""

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

"""
# Inference on validation images
"""

# part code reference https://keras.io/examples/vision/deeplabv3_plus/

np.random.seed(29)
colormap = np.random.randint(255, size=(59, 3))


def inference(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlay


plt.figure(figsize=(20, 30))
k = 0
select_sample = np.random.randint(99, size=4)
for i in select_sample:
    image_tensor = read_image(val_images[i])
    truth_mask = read_image(val_masks[i], mask=True)
    prediction_mask = inference(image_tensor=image_tensor, model=model)
    prediction_colormap = decode_segmentation_masks(
        prediction_mask, colormap, num_classes
    )
    prediction_overlay = get_overlay(image_tensor, prediction_colormap)

    plt.subplot(4, 3, 1 + k * 3)
    plt.imshow(image_tensor)
    plt.axis("off")
    plt.title("Actual Image")

    plt.subplot(4, 3, 2 + k * 3)
    plt.imshow(truth_mask, cmap="jet")
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(4, 3, 3 + k * 3)
    plt.imshow(prediction_overlay)
    plt.axis("off")
    plt.title("Prediction")

    k += 1
plt.show()

"""
# Conclusion

Our model has a lot of room for improvement. We can use a pre-trained ViT encoder
and we can set a larger image resolution such as 512x512. Our dataset has only 1000
images,
so in order to get better performance, we can use data augmentation or add more samples
to train.
"""
