
# Compact Convolutional Transformers

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/06/30<br>
**Last modified:** 2021/06/30<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/cct.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/cct.py)


**Description:** Compact Convolutional Transformers for efficient image classification.

As discussed in the [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929) paper,
a Transformer-based architecture for vision typically requires a larger dataset than
usual, as well as a longer pre-training schedule. [ImageNet-1k](http://imagenet.org/)
(which has about a million images) is considered to fall under the medium-sized data regime with
respect to ViTs. This is primarily because, unlike CNNs, ViTs (or a typical
Transformer-based architecture) do not have well-informed inductive biases (such as
convolutions for processing images). This begs the question: can't we combine the
benefits of convolution and the benefits of Transformers
in a single network architecture? These benefits include parameter-efficiency, and
self-attention to process long-range and global dependencies (interactions between
different regions in an image).

In [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704),
Hassani et al. present an approach for doing exactly this. They proposed the
**Compact Convolutional Transformer** (CCT) architecture. In this example, we will work on an
implementation of CCT and we will see how well it performs on the CIFAR-10 dataset.

If you are unfamiliar with the concept of self-attention or Transformers, you can read
[this chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11/r-3/312)
from  François Chollet's book *Deep Learning with Python*. This example uses
code snippets from another example,
[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/).

This example requires TensorFlow 2.5 or higher, as well as TensorFlow Addons, which can
be installed using the following command:


```python
!pip install -U -q tensorflow-addons
```

<div class="k-default-codeblock">
```
[K     |████████████████████████████████| 686kB 15.9MB/s 
[?25h

```
</div>
---
## Imports


```python
from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
```

---
## Hyperparameters and constants


```python
positional_emb = True
conv_layers = 2
projection_dim = 128

num_heads = 2
transformer_units = [
    projection_dim,
    projection_dim,
]
transformer_layers = 2
stochastic_depth_rate = 0.1

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 30
image_size = 32
```

---
## Load CIFAR-10 dataset


```python
num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
```

<div class="k-default-codeblock">
```
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170500096/170498071 [==============================] - 6s 0us/step
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 10)
x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 10)

```
</div>
---
## The CCT tokenizer

The first recipe introduced by the CCT authors is the tokenizer for processing the
images. In a standard ViT, images are organized into uniform *non-overlapping* patches.
This eliminates the boundary-level information present in between different patches. This
is important for a neural network to effectively exploit the locality information. The
figure below presents an illustration of how images are organized into patches.

![](https://i.imgur.com/IkBK9oY.png)

We already know that convolutions are quite good at exploiting locality information. So,
based on this, the authors introduce an all-convolution mini-network to produce image
patches.


```python

class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[64, 128],
        positional_emb=positional_emb,
        **kwargs,
    ):
        super(CCTTokenizer, self).__init__(**kwargs)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(
                layers.MaxPool2D(pooling_kernel_size, pooling_stride, "same")
            )

        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),
        )
        return reshaped

    def positional_embedding(self, image_size):
        # Positional embeddings are optional in CCT. Here, we calculate
        # the number of sequences and initialize an `Embedding` layer to
        # compute the positional embeddings later.
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size, image_size, 3))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]

            embed_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=projection_dim
            )
            return embed_layer, sequence_length
        else:
            return None

```

---
## Stochastic depth for regularization

[Stochastic depth](https://arxiv.org/abs/1603.09382) is a regularization technique that
randomly drops a set of layers. During inference, the layers are kept as they are. It is
very much similar to [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) but only
that it operates on a block os layers rather than individual nodes present inside a
layer. In CCT, stochastic depth is used just before the residual blocks of a Transformers
encoder.


```python
# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

```

---
## MLP for the Transformers encoder


```python

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

```

---
## Data augmentation

In the [original paper](https://arxiv.org/abs/2104.05704), the authors use
[AutoAugment](https://arxiv.org/abs/1805.09501) to induce stronger regularization. For
this example, we will be using the standard geometric augmentations like random cropping
and flipping.


```python
# Note the rescaling layer. These layers have pre-defined inference behavior.
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(scale=1.0 / 255),
        layers.experimental.preprocessing.RandomCrop(image_size, image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
    ],
    name="data_augmentation",
)
```

---
## The final CCT model

Another recipe introduced in CCT is attention pooling or sequence pooling. In ViT, only
the feature map corresponding to the class token is pooled and is then used for the
subsequent classification task (or any other downstream task). In CCT, outputs from the
Transformers encoder are weighted and then passed on to the final task-specific layer (in
this example, we do classification).


```python

def create_cct_model(
    image_size=image_size,
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):

    inputs = layers.Input(input_shape)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(augmented)

    # Apply positional embedding.
    if positional_emb:
        pos_embed, seq_length = cct_tokenizer.positional_embedding(image_size)
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    # Classify outputs.
    logits = layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

```

---
## Model training and evaluation


```python

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


cct_model = create_cct_model()
history = run_experiment(cct_model)
```

<div class="k-default-codeblock">
```
Epoch 1/30
352/352 [==============================] - 10s 18ms/step - loss: 1.9361 - accuracy: 0.3195 - top-5-accuracy: 0.8192 - val_loss: 1.6775 - val_accuracy: 0.4578 - val_top-5-accuracy: 0.9088
Epoch 2/30
352/352 [==============================] - 6s 16ms/step - loss: 1.6029 - accuracy: 0.4888 - top-5-accuracy: 0.9272 - val_loss: 1.4748 - val_accuracy: 0.5550 - val_top-5-accuracy: 0.9476
Epoch 3/30
352/352 [==============================] - 6s 16ms/step - loss: 1.4636 - accuracy: 0.5568 - top-5-accuracy: 0.9480 - val_loss: 1.4069 - val_accuracy: 0.5834 - val_top-5-accuracy: 0.9562
Epoch 4/30
352/352 [==============================] - 6s 16ms/step - loss: 1.3810 - accuracy: 0.5955 - top-5-accuracy: 0.9568 - val_loss: 1.3310 - val_accuracy: 0.6224 - val_top-5-accuracy: 0.9660
Epoch 5/30
352/352 [==============================] - 6s 16ms/step - loss: 1.3176 - accuracy: 0.6288 - top-5-accuracy: 0.9642 - val_loss: 1.2924 - val_accuracy: 0.6350 - val_top-5-accuracy: 0.9672
Epoch 6/30
352/352 [==============================] - 6s 16ms/step - loss: 1.2636 - accuracy: 0.6564 - top-5-accuracy: 0.9696 - val_loss: 1.2917 - val_accuracy: 0.6498 - val_top-5-accuracy: 0.9654
Epoch 7/30
352/352 [==============================] - 6s 16ms/step - loss: 1.2285 - accuracy: 0.6722 - top-5-accuracy: 0.9724 - val_loss: 1.1805 - val_accuracy: 0.6912 - val_top-5-accuracy: 0.9774
Epoch 8/30
352/352 [==============================] - 6s 16ms/step - loss: 1.1871 - accuracy: 0.6887 - top-5-accuracy: 0.9763 - val_loss: 1.1779 - val_accuracy: 0.6974 - val_top-5-accuracy: 0.9780
Epoch 9/30
352/352 [==============================] - 6s 16ms/step - loss: 1.1610 - accuracy: 0.7022 - top-5-accuracy: 0.9773 - val_loss: 1.1541 - val_accuracy: 0.7118 - val_top-5-accuracy: 0.9766
Epoch 10/30
352/352 [==============================] - 6s 16ms/step - loss: 1.1334 - accuracy: 0.7168 - top-5-accuracy: 0.9788 - val_loss: 1.1468 - val_accuracy: 0.7138 - val_top-5-accuracy: 0.9768
Epoch 11/30
352/352 [==============================] - 6s 16ms/step - loss: 1.1214 - accuracy: 0.7215 - top-5-accuracy: 0.9805 - val_loss: 1.1361 - val_accuracy: 0.7176 - val_top-5-accuracy: 0.9800
Epoch 12/30
352/352 [==============================] - 6s 16ms/step - loss: 1.0947 - accuracy: 0.7340 - top-5-accuracy: 0.9817 - val_loss: 1.1240 - val_accuracy: 0.7292 - val_top-5-accuracy: 0.9768
Epoch 13/30
352/352 [==============================] - 6s 16ms/step - loss: 1.0736 - accuracy: 0.7448 - top-5-accuracy: 0.9840 - val_loss: 1.0907 - val_accuracy: 0.7398 - val_top-5-accuracy: 0.9796
Epoch 14/30
352/352 [==============================] - 6s 16ms/step - loss: 1.0609 - accuracy: 0.7517 - top-5-accuracy: 0.9838 - val_loss: 1.0769 - val_accuracy: 0.7440 - val_top-5-accuracy: 0.9814
Epoch 15/30
352/352 [==============================] - 6s 16ms/step - loss: 1.0442 - accuracy: 0.7572 - top-5-accuracy: 0.9852 - val_loss: 1.0840 - val_accuracy: 0.7432 - val_top-5-accuracy: 0.9826
Epoch 16/30
352/352 [==============================] - 6s 16ms/step - loss: 1.0344 - accuracy: 0.7651 - top-5-accuracy: 0.9853 - val_loss: 1.0796 - val_accuracy: 0.7490 - val_top-5-accuracy: 0.9802
Epoch 17/30
352/352 [==============================] - 6s 16ms/step - loss: 1.0204 - accuracy: 0.7692 - top-5-accuracy: 0.9858 - val_loss: 1.0755 - val_accuracy: 0.7442 - val_top-5-accuracy: 0.9824
Epoch 18/30
352/352 [==============================] - 5s 16ms/step - loss: 1.0101 - accuracy: 0.7731 - top-5-accuracy: 0.9872 - val_loss: 1.0532 - val_accuracy: 0.7536 - val_top-5-accuracy: 0.9842
Epoch 19/30
352/352 [==============================] - 6s 16ms/step - loss: 1.0005 - accuracy: 0.7790 - top-5-accuracy: 0.9878 - val_loss: 1.0618 - val_accuracy: 0.7564 - val_top-5-accuracy: 0.9810
Epoch 20/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9942 - accuracy: 0.7795 - top-5-accuracy: 0.9875 - val_loss: 1.0458 - val_accuracy: 0.7652 - val_top-5-accuracy: 0.9836
Epoch 21/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9800 - accuracy: 0.7865 - top-5-accuracy: 0.9883 - val_loss: 1.0308 - val_accuracy: 0.7736 - val_top-5-accuracy: 0.9834
Epoch 22/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9736 - accuracy: 0.7897 - top-5-accuracy: 0.9888 - val_loss: 1.0293 - val_accuracy: 0.7704 - val_top-5-accuracy: 0.9846
Epoch 23/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9662 - accuracy: 0.7967 - top-5-accuracy: 0.9890 - val_loss: 1.0230 - val_accuracy: 0.7760 - val_top-5-accuracy: 0.9840
Epoch 24/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9616 - accuracy: 0.7955 - top-5-accuracy: 0.9896 - val_loss: 1.0184 - val_accuracy: 0.7722 - val_top-5-accuracy: 0.9822
Epoch 25/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9541 - accuracy: 0.8003 - top-5-accuracy: 0.9896 - val_loss: 1.0174 - val_accuracy: 0.7750 - val_top-5-accuracy: 0.9856
Epoch 26/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9433 - accuracy: 0.8028 - top-5-accuracy: 0.9907 - val_loss: 1.0366 - val_accuracy: 0.7644 - val_top-5-accuracy: 0.9856
Epoch 27/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9465 - accuracy: 0.8029 - top-5-accuracy: 0.9895 - val_loss: 0.9985 - val_accuracy: 0.7804 - val_top-5-accuracy: 0.9862
Epoch 28/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9287 - accuracy: 0.8125 - top-5-accuracy: 0.9909 - val_loss: 1.0238 - val_accuracy: 0.7740 - val_top-5-accuracy: 0.9812
Epoch 29/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9311 - accuracy: 0.8108 - top-5-accuracy: 0.9900 - val_loss: 0.9999 - val_accuracy: 0.7892 - val_top-5-accuracy: 0.9848
Epoch 30/30
352/352 [==============================] - 6s 16ms/step - loss: 0.9218 - accuracy: 0.8145 - top-5-accuracy: 0.9916 - val_loss: 1.0545 - val_accuracy: 0.7644 - val_top-5-accuracy: 0.9810
313/313 [==============================] - 2s 6ms/step - loss: 1.0127 - accuracy: 0.7765 - top-5-accuracy: 0.9833
Test accuracy: 77.65%
Test top 5 accuracy: 98.33%

```
</div>
Let's now visualize the training progress of the model.


```python
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()
```


![png](/img/examples/vision/cct/cct_22_0.png)


The CCT model we just trained has just **0.4 million** parameters, and it gets us to
~78% top-1 accuracy within 30 epochs. The plot above shows no signs of overfitting as
well. This means we can train this network for longers (perhaps with a bit more
regularization) and may obtain even better performance. This performance can further be
improved by additional recipes like cosine decay learning rate schedule, other data augmentation
techniques like [AutoAugment](https://arxiv.org/abs/1805.09501),
[MixUp](https://arxiv.org/abs/1710.09412) or
[Cutmix](https://arxiv.org/abs/1905.04899). The authors also present a number of
experiments to study how the number of convolution blocks, Transformers layers, etc.
affect the final performance.

For a comparison, a ViT model takes about **4.7 million** parameters and **100
epochs** of training to reach a top-1 accuracy of 78.22% on the CIFAR-10 dataset. You can
refer to
[this notebook](https://colab.research.google.com/gist/sayakpaul/1a80d9f582b044354a1a26c5cb3d69e5/image_classification_with_vision_transformer.ipynb)
to know about the experimental setup.

The authors also demonstrate the performance of Compact Convolutional Transformers on
NLP tasks and they report competitive results there.
