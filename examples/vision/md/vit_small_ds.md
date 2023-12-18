# Train a Vision Transformer on small datasets

**Author:** [Aritra Roy Gosthipaty](https://twitter.com/ariG23498)<br>
**Date created:** 2022/01/07<br>
**Last modified:** 2023/12/13<br>
**Description:** Training a ViT on smaller datasets with shifted patch tokenization and locality self-attention.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/vit_small_ds.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/vit_small_ds.py)



---
## Introduction

In the academic paper
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929),
the authors mention that Vision Transformers (ViT) are data-hungry. Therefore,
pretraining a ViT on a large-sized dataset like JFT300M and fine-tuning
it on medium-sized datasets (like ImageNet) is the only way to beat
state-of-the-art Convolutional Neural Network models.

The self-attention layer of ViT lacks **locality inductive bias** (the notion that
image pixels are locally correlated and that their correlation maps are translation-invariant).
This is the reason why ViTs need more data. On the other hand, CNNs look at images through
spatial sliding windows, which helps them get better results with smaller datasets.

In the academic paper
[Vision Transformer for Small-Size Datasets](https://arxiv.org/abs/2112.13492v1),
the authors set out to tackle the problem of locality inductive bias in ViTs.

The main ideas are:

- **Shifted Patch Tokenization**
- **Locality Self Attention**

This example implements the ideas of the paper. A large part of this
example is inspired from
[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/).


---
## Setup


```python
import math
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import layers
from keras import ops

# Setting seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)

```

---
## Prepare the data


```python
NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

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
## Configure the hyperparameters

The hyperparameters are different from the paper. Feel free to tune
the hyperparameters yourself.


```python
# DATA
BUFFER_SIZE = 512
BATCH_SIZE = 256

# AUGMENTATION
IMAGE_SIZE = 72
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# OPTIMIZER
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# TRAINING
EPOCHS = 50

# ARCHITECTURE
LAYER_NORM_EPS = 1e-6
TRANSFORMER_LAYERS = 8
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]
MLP_HEAD_UNITS = [2048, 1024]
```

---
## Use data augmentation

A snippet from the paper:

*"According to DeiT, various techniques are required to effectively
train ViTs. Thus, we applied data augmentations such as CutMix, Mixup,
Auto Augment, Repeated Augment to all models."*

In this example, we will focus solely on the novelty of the approach
and not on reproducing the paper results. For this reason, we
don't use the mentioned data augmentation schemes. Please feel
free to add to or remove from the augmentation pipeline.


```python
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)
```

---
## Implement Shifted Patch Tokenization

In a ViT pipeline, the input images are divided into patches that are
then linearly projected into tokens. Shifted patch tokenization (STP)
is introduced to combat the low receptive field of ViTs. The steps
for Shifted Patch Tokenization are as follows:

- Start with an image.
- Shift the image in diagonal directions.
- Concat the diagonally shifted images with the original image.
- Extract patches of the concatenated images.
- Flatten the spatial dimension of all patches.
- Layer normalize the flattened patches and then project it.

| ![Shifted Patch Toekenization](https://i.imgur.com/bUnHxd0.png) |
| :--: |
| Shifted Patch Tokenization [Source](https://arxiv.org/abs/2112.13492v1) |


```python

class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        target_height = self.image_size - self.half_patch
        target_width = self.image_size - self.half_patch
        crop = images[
            :,
            crop_height : crop_height + target_height,
            crop_width : crop_width + target_width,
            :,
        ]

        shift_pad = ops.image.pad_images(
            crop,
            top_padding=shift_height,
            left_padding=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = ops.concatenate(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = ops.image.extract_patches(
            images,
            (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            dilation_rate=(1, 1),
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return (tokens, patches)

```

### Visualize the patches


```python
# Get a random image from the training dataset
# and resize the image
image = x_train[np.random.choice(range(x_train.shape[0]))]
resized_image = ops.image.resize(
    ops.convert_to_tensor([image], dtype="float32"),
    size=(IMAGE_SIZE, IMAGE_SIZE),
)

# Vanilla patch maker: This takes an image and divides into
# patches as in the original ViT paper
(token, patch) = ShiftedPatchTokenization(vanilla=True)(resized_image / 255.0)
(token, patch) = (token[0], patch[0])
n = patch.shape[0]
count = 1
plt.figure(figsize=(4, 4))
for row in range(n):
    for col in range(n):
        plt.subplot(n, n, count)
        count = count + 1
        image = ops.reshape(patch[row][col], (PATCH_SIZE, PATCH_SIZE, 3))
        plt.imshow(image)
        plt.axis("off")
plt.show()

# Shifted Patch Tokenization: This layer takes the image, shifts it
# diagonally and then extracts patches from the concatinated images
(token, patch) = ShiftedPatchTokenization(vanilla=False)(resized_image / 255.0)
(token, patch) = (token[0], patch[0])
n = patch.shape[0]
shifted_images = ["ORIGINAL", "LEFT-UP", "LEFT-DOWN", "RIGHT-UP", "RIGHT-DOWN"]
for index, name in enumerate(shifted_images):
    print(name)
    count = 1
    plt.figure(figsize=(4, 4))
    for row in range(n):
        for col in range(n):
            plt.subplot(n, n, count)
            count = count + 1
            image = ops.reshape(patch[row][col], (PATCH_SIZE, PATCH_SIZE, 5 * 3))
            plt.imshow(image[..., 3 * index : 3 * index + 3])
            plt.axis("off")
    plt.show()
```

<div class="k-default-codeblock">
```
/home/suryanarayanay/miniconda3/envs/tf2.13/lib/python3.11/site-packages/keras/src/layers/layer.py:357: UserWarning: `build()` was called on layer 'shifted_patch_tokenization', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(

```
</div>
    
![png](/img/examples/vision/vit_small_ds/vit_small_ds_13_1.png)
    


<div class="k-default-codeblock">
```
ORIGINAL

```
</div>
    
![png](/img/examples/vision/vit_small_ds/vit_small_ds_13_3.png)
    


<div class="k-default-codeblock">
```
LEFT-UP

```
</div>
    
![png](/img/examples/vision/vit_small_ds/vit_small_ds_13_5.png)
    


<div class="k-default-codeblock">
```
LEFT-DOWN

```
</div>
    
![png](/img/examples/vision/vit_small_ds/vit_small_ds_13_7.png)
    


<div class="k-default-codeblock">
```
RIGHT-UP

```
</div>
    
![png](/img/examples/vision/vit_small_ds/vit_small_ds_13_9.png)
    


<div class="k-default-codeblock">
```
RIGHT-DOWN

```
</div>
    
![png](/img/examples/vision/vit_small_ds/vit_small_ds_13_11.png)
    


---
## Implement the patch encoding layer

This layer accepts projected patches and then adds positional
information to them.


```python

class PatchEncoder(layers.Layer):
    def __init__(
        self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = ops.arange(start=0, stop=self.num_patches, step=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches

```

---
## Implement Locality Self Attention

The regular attention equation is stated below.

| ![Equation of attention](https://miro.medium.com/max/396/1*P9sV1xXM10t943bXy_G9yg.png) |
| :--: |
| [Source](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634) |

The attention module takes a query, key, and value. First, we compute the
similarity between the query and key via a dot product. Then, the result
is scaled by the square root of the key dimension. The scaling prevents
the softmax function from having an overly small gradient. Softmax is then
applied to the scaled dot product to produce the attention weights.
The value is then modulated via the attention weights.

In self-attention, query, key and value come from the same input.
The dot product would result in large self-token relations rather than
inter-token relations. This also means that the softmax gives higher
probabilities to self-token relations than the inter-token relations.
To combat this, the authors propose masking the diagonal of the dot product.
This way, we force the attention module to pay more attention to the
inter-token relations.

The scaling factor is a constant in the regular attention module.
This acts like a temperature term that can modulate the softmax function.
The authors suggest a learnable temperature term instead of a constant.

| ![Implementation of LSA](https://i.imgur.com/GTV99pk.png) |
| :--: |
| Locality Self Attention [Source](https://arxiv.org/abs/2112.13492v1) |

The above two pointers make the Locality Self Attention. We have subclassed the
[`layers.MultiHeadAttention`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention)
and implemented the trainable temperature. The attention mask is built
at a later stage.


```python

class MultiHeadAttentionLSA(keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = keras.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = ops.multiply(query, 1.0 / self.tau)
        attention_scores = ops.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = ops.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

```

---
## Implement the MLP


```python

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# Build the diagonal attention mask
diag_attn_mask = 1 - ops.eye(NUM_PATCHES)
diag_attn_mask = ops.cast([diag_attn_mask], dtype="int8")
```

---
## Build the ViT


```python

def create_vit_classifier(vanilla=False):
    inputs = layers.Input(shape=INPUT_SHAPE)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(vanilla=vanilla)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder()(tokens)

    # Create multiple layers of the Transformer block.
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(NUM_CLASSES)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

```

---
## Compile, train, and evaluate the mode


```python

# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = ops.array(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = ops.cos(
            self.pi
            * (ops.cast(step, "float32") - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * ops.cast(step, "float32") + self.warmup_learning_rate
            learning_rate = ops.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return ops.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def run_experiment(model):
    total_steps = int((len(x_train) / BATCH_SIZE) * EPOCHS)
    warmup_epoch_percentage = 0.10
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
    )
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


# Run experiments with the vanilla ViT
vit = create_vit_classifier(vanilla=True)
history = run_experiment(vit)

# Run experiments with the Shifted Patch Tokenization and
# Locality Self Attention modified ViT
vit_sl = create_vit_classifier(vanilla=False)
history = run_experiment(vit_sl)
```

<div class="k-default-codeblock">
```
/home/suryanarayanay/miniconda3/envs/tf2.13/lib/python3.11/site-packages/keras/src/layers/layer.py:357: UserWarning: `build()` was called on layer 'shifted_patch_tokenization_2', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(

Epoch 1/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 106s 388ms/step - accuracy: 0.0144 - loss: 4.9827 - top-5-accuracy: 0.0674 - val_accuracy: 0.0338 - val_loss: 4.2915 - val_top-5-accuracy: 0.1594
Epoch 2/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.0326 - loss: 4.3661 - top-5-accuracy: 0.1383 - val_accuracy: 0.0614 - val_loss: 4.0716 - val_top-5-accuracy: 0.2298
Epoch 3/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 67s 378ms/step - accuracy: 0.0500 - loss: 4.1864 - top-5-accuracy: 0.1946 - val_accuracy: 0.0848 - val_loss: 3.9701 - val_top-5-accuracy: 0.2820
Epoch 4/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0587 - loss: 4.1070 - top-5-accuracy: 0.2219 - val_accuracy: 0.0812 - val_loss: 3.9043 - val_top-5-accuracy: 0.2830
Epoch 5/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0647 - loss: 4.0492 - top-5-accuracy: 0.2386 - val_accuracy: 0.0980 - val_loss: 3.8119 - val_top-5-accuracy: 0.3244
Epoch 6/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0743 - loss: 3.9836 - top-5-accuracy: 0.2607 - val_accuracy: 0.1042 - val_loss: 3.7930 - val_top-5-accuracy: 0.3304
Epoch 7/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.0754 - loss: 3.9804 - top-5-accuracy: 0.2626 - val_accuracy: 0.1006 - val_loss: 3.7506 - val_top-5-accuracy: 0.3306
Epoch 8/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0794 - loss: 3.9464 - top-5-accuracy: 0.2760 - val_accuracy: 0.1064 - val_loss: 3.7564 - val_top-5-accuracy: 0.3406
Epoch 9/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 67s 380ms/step - accuracy: 0.0820 - loss: 3.9316 - top-5-accuracy: 0.2788 - val_accuracy: 0.1070 - val_loss: 3.7597 - val_top-5-accuracy: 0.3274
Epoch 10/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0800 - loss: 3.9258 - top-5-accuracy: 0.2854 - val_accuracy: 0.1064 - val_loss: 3.7534 - val_top-5-accuracy: 0.3376
Epoch 11/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 67s 378ms/step - accuracy: 0.0868 - loss: 3.9147 - top-5-accuracy: 0.2892 - val_accuracy: 0.1188 - val_loss: 3.7024 - val_top-5-accuracy: 0.3508
Epoch 12/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 380ms/step - accuracy: 0.0931 - loss: 3.8687 - top-5-accuracy: 0.3009 - val_accuracy: 0.1214 - val_loss: 3.6852 - val_top-5-accuracy: 0.3588
Epoch 13/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0930 - loss: 3.8857 - top-5-accuracy: 0.2979 - val_accuracy: 0.1140 - val_loss: 3.7435 - val_top-5-accuracy: 0.3450
Epoch 14/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.0964 - loss: 3.8475 - top-5-accuracy: 0.3063 - val_accuracy: 0.1186 - val_loss: 3.7180 - val_top-5-accuracy: 0.3508
Epoch 15/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.0883 - loss: 3.9195 - top-5-accuracy: 0.2851 - val_accuracy: 0.1166 - val_loss: 3.7621 - val_top-5-accuracy: 0.3386
Epoch 16/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0951 - loss: 3.8740 - top-5-accuracy: 0.3024 - val_accuracy: 0.1190 - val_loss: 3.7663 - val_top-5-accuracy: 0.3442
Epoch 17/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.0950 - loss: 3.8552 - top-5-accuracy: 0.3057 - val_accuracy: 0.1300 - val_loss: 3.6928 - val_top-5-accuracy: 0.3622
Epoch 18/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 379ms/step - accuracy: 0.0998 - loss: 3.8125 - top-5-accuracy: 0.3239 - val_accuracy: 0.1242 - val_loss: 3.7050 - val_top-5-accuracy: 0.3554
Epoch 19/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1051 - loss: 3.7838 - top-5-accuracy: 0.3307 - val_accuracy: 0.1308 - val_loss: 3.6886 - val_top-5-accuracy: 0.3698
Epoch 20/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1096 - loss: 3.7654 - top-5-accuracy: 0.3396 - val_accuracy: 0.1318 - val_loss: 3.6502 - val_top-5-accuracy: 0.3734
Epoch 21/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 379ms/step - accuracy: 0.1125 - loss: 3.7372 - top-5-accuracy: 0.3426 - val_accuracy: 0.1346 - val_loss: 3.6462 - val_top-5-accuracy: 0.3764
Epoch 22/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 378ms/step - accuracy: 0.1106 - loss: 3.7680 - top-5-accuracy: 0.3365 - val_accuracy: 0.1244 - val_loss: 3.7112 - val_top-5-accuracy: 0.3582
Epoch 23/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.1136 - loss: 3.7426 - top-5-accuracy: 0.3430 - val_accuracy: 0.1346 - val_loss: 3.6812 - val_top-5-accuracy: 0.3612
Epoch 24/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 83s 379ms/step - accuracy: 0.1198 - loss: 3.7006 - top-5-accuracy: 0.3575 - val_accuracy: 0.1420 - val_loss: 3.6321 - val_top-5-accuracy: 0.3814
Epoch 25/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.1233 - loss: 3.7036 - top-5-accuracy: 0.3598 - val_accuracy: 0.1248 - val_loss: 3.7282 - val_top-5-accuracy: 0.3560
Epoch 26/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.1218 - loss: 3.6886 - top-5-accuracy: 0.3597 - val_accuracy: 0.1236 - val_loss: 3.7601 - val_top-5-accuracy: 0.3498
Epoch 27/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 83s 379ms/step - accuracy: 0.1176 - loss: 3.7587 - top-5-accuracy: 0.3406 - val_accuracy: 0.1270 - val_loss: 3.7429 - val_top-5-accuracy: 0.3476
Epoch 28/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 379ms/step - accuracy: 0.1229 - loss: 3.6951 - top-5-accuracy: 0.3578 - val_accuracy: 0.1454 - val_loss: 3.6243 - val_top-5-accuracy: 0.3808
Epoch 29/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.1254 - loss: 3.6891 - top-5-accuracy: 0.3589 - val_accuracy: 0.1362 - val_loss: 3.6371 - val_top-5-accuracy: 0.3726
Epoch 30/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1356 - loss: 3.6347 - top-5-accuracy: 0.3753 - val_accuracy: 0.1086 - val_loss: 3.7619 - val_top-5-accuracy: 0.3336
Epoch 31/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.1361 - loss: 3.6206 - top-5-accuracy: 0.3844 - val_accuracy: 0.1192 - val_loss: 3.7217 - val_top-5-accuracy: 0.3460
Epoch 32/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.1429 - loss: 3.6092 - top-5-accuracy: 0.3874 - val_accuracy: 0.1120 - val_loss: 3.8008 - val_top-5-accuracy: 0.3224
Epoch 33/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 83s 380ms/step - accuracy: 0.1452 - loss: 3.5909 - top-5-accuracy: 0.3944 - val_accuracy: 0.0920 - val_loss: 3.9773 - val_top-5-accuracy: 0.2808
Epoch 34/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 379ms/step - accuracy: 0.1106 - loss: 3.8286 - top-5-accuracy: 0.3258 - val_accuracy: 0.1310 - val_loss: 3.6728 - val_top-5-accuracy: 0.3628
Epoch 35/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 379ms/step - accuracy: 0.1313 - loss: 3.6763 - top-5-accuracy: 0.3657 - val_accuracy: 0.1074 - val_loss: 3.8060 - val_top-5-accuracy: 0.3170
Epoch 36/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1383 - loss: 3.6165 - top-5-accuracy: 0.3860 - val_accuracy: 0.1322 - val_loss: 3.6553 - val_top-5-accuracy: 0.3624
Epoch 37/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1469 - loss: 3.5917 - top-5-accuracy: 0.3905 - val_accuracy: 0.1242 - val_loss: 3.7094 - val_top-5-accuracy: 0.3452
Epoch 38/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1513 - loss: 3.5550 - top-5-accuracy: 0.3996 - val_accuracy: 0.1136 - val_loss: 3.8052 - val_top-5-accuracy: 0.3224
Epoch 39/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1578 - loss: 3.5185 - top-5-accuracy: 0.4110 - val_accuracy: 0.1138 - val_loss: 3.8225 - val_top-5-accuracy: 0.3140
Epoch 40/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1643 - loss: 3.4857 - top-5-accuracy: 0.4196 - val_accuracy: 0.1168 - val_loss: 3.7585 - val_top-5-accuracy: 0.3348
Epoch 41/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 379ms/step - accuracy: 0.1665 - loss: 3.4651 - top-5-accuracy: 0.4277 - val_accuracy: 0.1114 - val_loss: 3.8782 - val_top-5-accuracy: 0.2956
Epoch 42/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 379ms/step - accuracy: 0.1717 - loss: 3.4370 - top-5-accuracy: 0.4325 - val_accuracy: 0.1220 - val_loss: 3.8067 - val_top-5-accuracy: 0.3182
Epoch 43/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1779 - loss: 3.3973 - top-5-accuracy: 0.4457 - val_accuracy: 0.1068 - val_loss: 3.8727 - val_top-5-accuracy: 0.2994
Epoch 44/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1812 - loss: 3.4076 - top-5-accuracy: 0.4446 - val_accuracy: 0.1290 - val_loss: 3.7340 - val_top-5-accuracy: 0.3370
Epoch 45/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1812 - loss: 3.3673 - top-5-accuracy: 0.4529 - val_accuracy: 0.1134 - val_loss: 3.8447 - val_top-5-accuracy: 0.3062
Epoch 46/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1845 - loss: 3.3541 - top-5-accuracy: 0.4552 - val_accuracy: 0.0976 - val_loss: 3.9903 - val_top-5-accuracy: 0.2778
Epoch 47/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 375ms/step - accuracy: 0.1906 - loss: 3.3411 - top-5-accuracy: 0.4612 - val_accuracy: 0.1046 - val_loss: 3.9565 - val_top-5-accuracy: 0.2806
Epoch 48/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 66s 376ms/step - accuracy: 0.1958 - loss: 3.3100 - top-5-accuracy: 0.4696 - val_accuracy: 0.0894 - val_loss: 4.1331 - val_top-5-accuracy: 0.2536
Epoch 49/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1936 - loss: 3.3293 - top-5-accuracy: 0.4643 - val_accuracy: 0.1178 - val_loss: 3.8180 - val_top-5-accuracy: 0.3152
Epoch 50/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 377ms/step - accuracy: 0.1941 - loss: 3.3206 - top-5-accuracy: 0.4695 - val_accuracy: 0.1104 - val_loss: 3.9149 - val_top-5-accuracy: 0.3000
 40/40 ━━━━━━━━━━━━━━━━━━━━ 5s 114ms/step - accuracy: 0.1098 - loss: 3.9105 - top-5-accuracy: 0.2918
Test accuracy: 11.01%
Test top 5 accuracy: 29.66%
Epoch 1/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 114s 435ms/step - accuracy: 0.0130 - loss: 4.9957 - top-5-accuracy: 0.0603 - val_accuracy: 0.0184 - val_loss: 4.4445 - val_top-5-accuracy: 0.1068
Epoch 2/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 431ms/step - accuracy: 0.0187 - loss: 4.4785 - top-5-accuracy: 0.0923 - val_accuracy: 0.0328 - val_loss: 4.3426 - val_top-5-accuracy: 0.1186
Epoch 3/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 430ms/step - accuracy: 0.0258 - loss: 4.3935 - top-5-accuracy: 0.1156 - val_accuracy: 0.0314 - val_loss: 4.2596 - val_top-5-accuracy: 0.1456
Epoch 4/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 83s 435ms/step - accuracy: 0.0317 - loss: 4.3342 - top-5-accuracy: 0.1326 - val_accuracy: 0.0534 - val_loss: 4.1678 - val_top-5-accuracy: 0.1914
Epoch 5/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 431ms/step - accuracy: 0.0354 - loss: 4.2960 - top-5-accuracy: 0.1521 - val_accuracy: 0.0418 - val_loss: 4.2036 - val_top-5-accuracy: 0.1768
Epoch 6/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 417ms/step - accuracy: 0.0397 - loss: 4.2619 - top-5-accuracy: 0.1604 - val_accuracy: 0.0490 - val_loss: 4.1102 - val_top-5-accuracy: 0.2018
Epoch 7/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 417ms/step - accuracy: 0.0402 - loss: 4.2454 - top-5-accuracy: 0.1649 - val_accuracy: 0.0534 - val_loss: 4.0830 - val_top-5-accuracy: 0.2138
Epoch 8/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 84s 431ms/step - accuracy: 0.0410 - loss: 4.2215 - top-5-accuracy: 0.1703 - val_accuracy: 0.0608 - val_loss: 4.0638 - val_top-5-accuracy: 0.2174
Epoch 9/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 430ms/step - accuracy: 0.0423 - loss: 4.2098 - top-5-accuracy: 0.1746 - val_accuracy: 0.0626 - val_loss: 4.0521 - val_top-5-accuracy: 0.2340
Epoch 10/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 80s 418ms/step - accuracy: 0.0440 - loss: 4.2040 - top-5-accuracy: 0.1776 - val_accuracy: 0.0614 - val_loss: 4.0863 - val_top-5-accuracy: 0.2312
Epoch 11/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 417ms/step - accuracy: 0.0419 - loss: 4.2204 - top-5-accuracy: 0.1747 - val_accuracy: 0.0596 - val_loss: 4.1508 - val_top-5-accuracy: 0.2196
Epoch 12/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 418ms/step - accuracy: 0.0449 - loss: 4.2479 - top-5-accuracy: 0.1725 - val_accuracy: 0.0698 - val_loss: 4.1028 - val_top-5-accuracy: 0.2356
Epoch 13/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 84s 431ms/step - accuracy: 0.0472 - loss: 4.2098 - top-5-accuracy: 0.1841 - val_accuracy: 0.0722 - val_loss: 4.0688 - val_top-5-accuracy: 0.2454
Epoch 14/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 431ms/step - accuracy: 0.0518 - loss: 4.1769 - top-5-accuracy: 0.2000 - val_accuracy: 0.0722 - val_loss: 4.0601 - val_top-5-accuracy: 0.2466
Epoch 15/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 431ms/step - accuracy: 0.0563 - loss: 4.1603 - top-5-accuracy: 0.1986 - val_accuracy: 0.0718 - val_loss: 4.0230 - val_top-5-accuracy: 0.2534
Epoch 16/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 79s 418ms/step - accuracy: 0.0548 - loss: 4.1367 - top-5-accuracy: 0.2080 - val_accuracy: 0.0810 - val_loss: 4.0148 - val_top-5-accuracy: 0.2582
Epoch 17/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 417ms/step - accuracy: 0.0577 - loss: 4.1224 - top-5-accuracy: 0.2128 - val_accuracy: 0.0748 - val_loss: 4.0177 - val_top-5-accuracy: 0.2576
Epoch 18/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 84s 431ms/step - accuracy: 0.0602 - loss: 4.0958 - top-5-accuracy: 0.2233 - val_accuracy: 0.0700 - val_loss: 4.0434 - val_top-5-accuracy: 0.2538
Epoch 19/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 430ms/step - accuracy: 0.0613 - loss: 4.1042 - top-5-accuracy: 0.2232 - val_accuracy: 0.0748 - val_loss: 4.0170 - val_top-5-accuracy: 0.2634
Epoch 20/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 431ms/step - accuracy: 0.0597 - loss: 4.1195 - top-5-accuracy: 0.2211 - val_accuracy: 0.0896 - val_loss: 3.9539 - val_top-5-accuracy: 0.2854
Epoch 21/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 430ms/step - accuracy: 0.0649 - loss: 4.0599 - top-5-accuracy: 0.2390 - val_accuracy: 0.0802 - val_loss: 4.0006 - val_top-5-accuracy: 0.2712
Epoch 22/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 80s 417ms/step - accuracy: 0.0680 - loss: 4.0650 - top-5-accuracy: 0.2385 - val_accuracy: 0.0916 - val_loss: 3.9500 - val_top-5-accuracy: 0.2790
Epoch 23/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 417ms/step - accuracy: 0.0667 - loss: 4.0852 - top-5-accuracy: 0.2381 - val_accuracy: 0.0932 - val_loss: 3.9277 - val_top-5-accuracy: 0.2872
Epoch 24/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 418ms/step - accuracy: 0.0731 - loss: 4.0355 - top-5-accuracy: 0.2484 - val_accuracy: 0.0922 - val_loss: 3.9267 - val_top-5-accuracy: 0.2922
Epoch 25/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 84s 431ms/step - accuracy: 0.0748 - loss: 4.0059 - top-5-accuracy: 0.2635 - val_accuracy: 0.0942 - val_loss: 3.9181 - val_top-5-accuracy: 0.2854
Epoch 26/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 431ms/step - accuracy: 0.0796 - loss: 3.9839 - top-5-accuracy: 0.2660 - val_accuracy: 0.1054 - val_loss: 3.8624 - val_top-5-accuracy: 0.3120
Epoch 27/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 430ms/step - accuracy: 0.0852 - loss: 3.9548 - top-5-accuracy: 0.2778 - val_accuracy: 0.1012 - val_loss: 3.8862 - val_top-5-accuracy: 0.3122
Epoch 28/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 80s 417ms/step - accuracy: 0.0910 - loss: 3.9288 - top-5-accuracy: 0.2873 - val_accuracy: 0.1094 - val_loss: 3.8378 - val_top-5-accuracy: 0.3192
Epoch 29/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 417ms/step - accuracy: 0.0931 - loss: 3.9275 - top-5-accuracy: 0.2917 - val_accuracy: 0.0830 - val_loss: 4.0437 - val_top-5-accuracy: 0.2622
Epoch 30/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 430ms/step - accuracy: 0.0659 - loss: 4.1312 - top-5-accuracy: 0.2252 - val_accuracy: 0.0730 - val_loss: 4.0968 - val_top-5-accuracy: 0.2366
Epoch 31/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 430ms/step - accuracy: 0.0641 - loss: 4.1432 - top-5-accuracy: 0.2210 - val_accuracy: 0.0956 - val_loss: 3.9468 - val_top-5-accuracy: 0.2882
Epoch 32/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 80s 417ms/step - accuracy: 0.0809 - loss: 4.0316 - top-5-accuracy: 0.2580 - val_accuracy: 0.1012 - val_loss: 3.9199 - val_top-5-accuracy: 0.2926
Epoch 33/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 417ms/step - accuracy: 0.0836 - loss: 3.9901 - top-5-accuracy: 0.2690 - val_accuracy: 0.0088 - val_loss: 4.6065 - val_top-5-accuracy: 0.0606
Epoch 34/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 417ms/step - accuracy: 0.0130 - loss: 4.5710 - top-5-accuracy: 0.0715 - val_accuracy: 0.0234 - val_loss: 4.4221 - val_top-5-accuracy: 0.1234
Epoch 35/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 416ms/step - accuracy: 0.0275 - loss: 4.4182 - top-5-accuracy: 0.1194 - val_accuracy: 0.0438 - val_loss: 4.3006 - val_top-5-accuracy: 0.1820
Epoch 36/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 417ms/step - accuracy: 0.0349 - loss: 4.3419 - top-5-accuracy: 0.1467 - val_accuracy: 0.0482 - val_loss: 4.2545 - val_top-5-accuracy: 0.1948
Epoch 37/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 417ms/step - accuracy: 0.0412 - loss: 4.3049 - top-5-accuracy: 0.1622 - val_accuracy: 0.0572 - val_loss: 4.2424 - val_top-5-accuracy: 0.2096
Epoch 38/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 417ms/step - accuracy: 0.0456 - loss: 4.2676 - top-5-accuracy: 0.1777 - val_accuracy: 0.0604 - val_loss: 4.2151 - val_top-5-accuracy: 0.2160
Epoch 39/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 416ms/step - accuracy: 0.0529 - loss: 4.2212 - top-5-accuracy: 0.1956 - val_accuracy: 0.0638 - val_loss: 4.2102 - val_top-5-accuracy: 0.2164
Epoch 40/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 416ms/step - accuracy: 0.0553 - loss: 4.2052 - top-5-accuracy: 0.2000 - val_accuracy: 0.0580 - val_loss: 4.2483 - val_top-5-accuracy: 0.1942
Epoch 41/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 416ms/step - accuracy: 0.0593 - loss: 4.1926 - top-5-accuracy: 0.2096 - val_accuracy: 0.0576 - val_loss: 4.2302 - val_top-5-accuracy: 0.2036
Epoch 42/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 84s 430ms/step - accuracy: 0.0607 - loss: 4.1615 - top-5-accuracy: 0.2178 - val_accuracy: 0.0642 - val_loss: 4.2020 - val_top-5-accuracy: 0.2104
Epoch 43/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 430ms/step - accuracy: 0.0618 - loss: 4.1686 - top-5-accuracy: 0.2174 - val_accuracy: 0.0520 - val_loss: 4.2642 - val_top-5-accuracy: 0.1868
Epoch 44/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 429ms/step - accuracy: 0.0656 - loss: 4.1444 - top-5-accuracy: 0.2268 - val_accuracy: 0.0690 - val_loss: 4.1636 - val_top-5-accuracy: 0.2188
Epoch 45/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 73s 417ms/step - accuracy: 0.0692 - loss: 4.1015 - top-5-accuracy: 0.2354 - val_accuracy: 0.0450 - val_loss: 4.2696 - val_top-5-accuracy: 0.1740
Epoch 46/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 84s 432ms/step - accuracy: 0.0688 - loss: 4.1271 - top-5-accuracy: 0.2321 - val_accuracy: 0.0526 - val_loss: 4.2354 - val_top-5-accuracy: 0.1932
Epoch 47/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 80s 418ms/step - accuracy: 0.0743 - loss: 4.0733 - top-5-accuracy: 0.2484 - val_accuracy: 0.0560 - val_loss: 4.2570 - val_top-5-accuracy: 0.1894
Epoch 48/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 76s 431ms/step - accuracy: 0.0773 - loss: 4.0570 - top-5-accuracy: 0.2498 - val_accuracy: 0.0544 - val_loss: 4.2575 - val_top-5-accuracy: 0.1946
Epoch 49/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 82s 433ms/step - accuracy: 0.0826 - loss: 4.0352 - top-5-accuracy: 0.2621 - val_accuracy: 0.0466 - val_loss: 4.3723 - val_top-5-accuracy: 0.1560
Epoch 50/50
 176/176 ━━━━━━━━━━━━━━━━━━━━ 77s 434ms/step - accuracy: 0.0888 - loss: 4.0068 - top-5-accuracy: 0.2730 - val_accuracy: 0.0548 - val_loss: 4.3327 - val_top-5-accuracy: 0.1874
 40/40 ━━━━━━━━━━━━━━━━━━━━ 6s 144ms/step - accuracy: 0.0539 - loss: 4.3341 - top-5-accuracy: 0.1849
Test accuracy: 5.8%
Test top 5 accuracy: 18.77%

```
</div>
# Final Notes

With the help of Shifted Patch Tokenization and Locality Self Attention,
we were able to get ~**3-4%** top-1 accuracy gains on CIFAR100.

The ideas on Shifted Patch Tokenization and Locality Self Attention
are very intuitive and easy to implement. The authors also ablates of
different shifting strategies for Shifted Patch Tokenization in the
supplementary of the paper.

I would like to thank [Jarvislabs.ai](https://jarvislabs.ai/) for
generously helping with GPU credits.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/vit_small_ds_v2)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/vit-small-ds).
