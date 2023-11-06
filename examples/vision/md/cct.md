# Compact Convolutional Transformers

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/06/30<br>
**Last modified:** 2021/06/30<br>
**Description:** Compact Convolutional Transformers for efficient image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/cct.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/cct.py)



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
170498071/170498071 [==============================] - 14s 0us/step
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
        super().__init__(**kwargs)

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
that it operates on a block of layers rather than individual nodes present inside a
layer. In CCT, stochastic depth is used just before the residual blocks of a Transformers
encoder.


```python

# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(x).shape[0] - 1)
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
        layers.Rescaling(scale=1.0 / 255),
        layers.RandomCrop(image_size, image_size),
        layers.RandomFlip("horizontal"),
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
352/352 [==============================] - 19s 39ms/step - loss: 1.8898 - accuracy: 0.3386 - top-5-accuracy: 0.8355 - val_loss: 1.6695 - val_accuracy: 0.4638 - val_top-5-accuracy: 0.9092
Epoch 2/30
352/352 [==============================] - 13s 37ms/step - loss: 1.5330 - accuracy: 0.5158 - top-5-accuracy: 0.9369 - val_loss: 1.4373 - val_accuracy: 0.5666 - val_top-5-accuracy: 0.9480
Epoch 3/30
352/352 [==============================] - 14s 39ms/step - loss: 1.4146 - accuracy: 0.5774 - top-5-accuracy: 0.9548 - val_loss: 1.3599 - val_accuracy: 0.6010 - val_top-5-accuracy: 0.9616
Epoch 4/30
352/352 [==============================] - 13s 36ms/step - loss: 1.3421 - accuracy: 0.6116 - top-5-accuracy: 0.9619 - val_loss: 1.2986 - val_accuracy: 0.6438 - val_top-5-accuracy: 0.9666
Epoch 5/30
352/352 [==============================] - 13s 37ms/step - loss: 1.2772 - accuracy: 0.6445 - top-5-accuracy: 0.9677 - val_loss: 1.2459 - val_accuracy: 0.6606 - val_top-5-accuracy: 0.9720
Epoch 6/30
352/352 [==============================] - 13s 37ms/step - loss: 1.2494 - accuracy: 0.6551 - top-5-accuracy: 0.9711 - val_loss: 1.2579 - val_accuracy: 0.6582 - val_top-5-accuracy: 0.9686
Epoch 7/30
352/352 [==============================] - 13s 37ms/step - loss: 1.1945 - accuracy: 0.6826 - top-5-accuracy: 0.9760 - val_loss: 1.1633 - val_accuracy: 0.7028 - val_top-5-accuracy: 0.9760
Epoch 8/30
352/352 [==============================] - 13s 36ms/step - loss: 1.1729 - accuracy: 0.6946 - top-5-accuracy: 0.9769 - val_loss: 1.1754 - val_accuracy: 0.7024 - val_top-5-accuracy: 0.9756
Epoch 9/30
352/352 [==============================] - 13s 37ms/step - loss: 1.1481 - accuracy: 0.7079 - top-5-accuracy: 0.9782 - val_loss: 1.2000 - val_accuracy: 0.6876 - val_top-5-accuracy: 0.9752
Epoch 10/30
352/352 [==============================] - 13s 38ms/step - loss: 1.1227 - accuracy: 0.7192 - top-5-accuracy: 0.9793 - val_loss: 1.1666 - val_accuracy: 0.7118 - val_top-5-accuracy: 0.9764
Epoch 11/30
352/352 [==============================] - 13s 38ms/step - loss: 1.1003 - accuracy: 0.7295 - top-5-accuracy: 0.9815 - val_loss: 1.1461 - val_accuracy: 0.7110 - val_top-5-accuracy: 0.9792
Epoch 12/30
352/352 [==============================] - 13s 37ms/step - loss: 1.0836 - accuracy: 0.7370 - top-5-accuracy: 0.9821 - val_loss: 1.0968 - val_accuracy: 0.7382 - val_top-5-accuracy: 0.9790
Epoch 13/30
352/352 [==============================] - 13s 38ms/step - loss: 1.0673 - accuracy: 0.7460 - top-5-accuracy: 0.9838 - val_loss: 1.1011 - val_accuracy: 0.7362 - val_top-5-accuracy: 0.9820
Epoch 14/30
352/352 [==============================] - 14s 38ms/step - loss: 1.0483 - accuracy: 0.7545 - top-5-accuracy: 0.9856 - val_loss: 1.0798 - val_accuracy: 0.7416 - val_top-5-accuracy: 0.9828
Epoch 15/30
352/352 [==============================] - 14s 39ms/step - loss: 1.0355 - accuracy: 0.7616 - top-5-accuracy: 0.9848 - val_loss: 1.0628 - val_accuracy: 0.7530 - val_top-5-accuracy: 0.9834
Epoch 16/30
352/352 [==============================] - 13s 36ms/step - loss: 1.0251 - accuracy: 0.7653 - top-5-accuracy: 0.9860 - val_loss: 1.0934 - val_accuracy: 0.7382 - val_top-5-accuracy: 0.9816
Epoch 17/30
352/352 [==============================] - 13s 38ms/step - loss: 1.0128 - accuracy: 0.7718 - top-5-accuracy: 0.9866 - val_loss: 1.0550 - val_accuracy: 0.7464 - val_top-5-accuracy: 0.9850
Epoch 18/30
352/352 [==============================] - 14s 38ms/step - loss: 1.0080 - accuracy: 0.7737 - top-5-accuracy: 0.9873 - val_loss: 1.0278 - val_accuracy: 0.7708 - val_top-5-accuracy: 0.9864
Epoch 19/30
352/352 [==============================] - 13s 36ms/step - loss: 0.9888 - accuracy: 0.7838 - top-5-accuracy: 0.9877 - val_loss: 1.0681 - val_accuracy: 0.7512 - val_top-5-accuracy: 0.9848
Epoch 20/30
352/352 [==============================] - 13s 37ms/step - loss: 0.9853 - accuracy: 0.7846 - top-5-accuracy: 0.9884 - val_loss: 1.0195 - val_accuracy: 0.7682 - val_top-5-accuracy: 0.9872
Epoch 21/30
352/352 [==============================] - 15s 43ms/step - loss: 0.9814 - accuracy: 0.7867 - top-5-accuracy: 0.9891 - val_loss: 1.0201 - val_accuracy: 0.7674 - val_top-5-accuracy: 0.9860
Epoch 22/30
352/352 [==============================] - 15s 43ms/step - loss: 0.9641 - accuracy: 0.7953 - top-5-accuracy: 0.9890 - val_loss: 1.0406 - val_accuracy: 0.7632 - val_top-5-accuracy: 0.9858
Epoch 23/30
352/352 [==============================] - 17s 47ms/step - loss: 0.9589 - accuracy: 0.7971 - top-5-accuracy: 0.9894 - val_loss: 1.0279 - val_accuracy: 0.7698 - val_top-5-accuracy: 0.9844
Epoch 24/30
352/352 [==============================] - 14s 39ms/step - loss: 0.9559 - accuracy: 0.7964 - top-5-accuracy: 0.9901 - val_loss: 1.0162 - val_accuracy: 0.7740 - val_top-5-accuracy: 0.9856
Epoch 25/30
352/352 [==============================] - 13s 38ms/step - loss: 0.9467 - accuracy: 0.8029 - top-5-accuracy: 0.9898 - val_loss: 0.9973 - val_accuracy: 0.7822 - val_top-5-accuracy: 0.9852
Epoch 26/30
352/352 [==============================] - 13s 38ms/step - loss: 0.9455 - accuracy: 0.8016 - top-5-accuracy: 0.9893 - val_loss: 1.0363 - val_accuracy: 0.7682 - val_top-5-accuracy: 0.9820
Epoch 27/30
352/352 [==============================] - 13s 38ms/step - loss: 0.9414 - accuracy: 0.8055 - top-5-accuracy: 0.9900 - val_loss: 0.9891 - val_accuracy: 0.7868 - val_top-5-accuracy: 0.9866
Epoch 28/30
352/352 [==============================] - 13s 36ms/step - loss: 0.9357 - accuracy: 0.8069 - top-5-accuracy: 0.9908 - val_loss: 0.9991 - val_accuracy: 0.7820 - val_top-5-accuracy: 0.9882
Epoch 29/30
352/352 [==============================] - 13s 38ms/step - loss: 0.9243 - accuracy: 0.8122 - top-5-accuracy: 0.9904 - val_loss: 0.9807 - val_accuracy: 0.7930 - val_top-5-accuracy: 0.9882
Epoch 30/30
352/352 [==============================] - 13s 38ms/step - loss: 0.9232 - accuracy: 0.8124 - top-5-accuracy: 0.9908 - val_loss: 0.9839 - val_accuracy: 0.7940 - val_top-5-accuracy: 0.9872
313/313 [==============================] - 3s 9ms/step - loss: 1.0100 - accuracy: 0.7772 - top-5-accuracy: 0.9869
Test accuracy: 77.72%
Test top 5 accuracy: 98.69%

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
well. This means we can train this network for longer (perhaps with a bit more
regularization) and may obtain even better performance. This performance can further be
improved by additional recipes like cosine decay learning rate schedule, other data augmentation
techniques like [AutoAugment](https://arxiv.org/abs/1805.09501),
[MixUp](https://arxiv.org/abs/1710.09412) or
[Cutmix](https://arxiv.org/abs/1905.04899). With these modifications, the authors present
95.1% top-1 accuracy on the CIFAR-10 dataset. The authors also present a number of
experiments to study how the number of convolution blocks, Transformers layers, etc.
affect the final performance of CCTs.

For a comparison, a ViT model takes about **4.7 million** parameters and **100
epochs** of training to reach a top-1 accuracy of 78.22% on the CIFAR-10 dataset. You can
refer to
[this notebook](https://colab.research.google.com/gist/sayakpaul/1a80d9f582b044354a1a26c5cb3d69e5/image_classification_with_vision_transformer.ipynb)
to know about the experimental setup.

The authors also demonstrate the performance of Compact Convolutional Transformers on
NLP tasks and they report competitive results there.

Example available on HuggingFace:

| Trained Model | Demo |
|------|------|
| [![🤗 Model - CCT](https://img.shields.io/badge/🤗_Model-Compact_Convolutional_Transformers-black)](https://huggingface.co/keras-io/cct) | [![🤗 Spaces - CCT](https://img.shields.io/badge/🤗_Spaces-Compact_Convolutional_Transformers-black)](https://huggingface.co/spaces/keras-io/cct) |
