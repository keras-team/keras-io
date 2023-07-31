# Neural style transfer

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2016/01/11<br>
**Last modified:** 2020/05/02<br>
**Description:** Transferring the style of a reference image to target image using gradient descent.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/neural_style_transfer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/neural_style_transfer.py)



---
## Introduction

Style transfer consists in generating an image
with the same "content" as a base image, but with the
"style" of a different picture (typically artistic).
This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":

- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.
- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).
- The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.

**Reference:** [A Neural Algorithm of Artistic Style](
  http://arxiv.org/abs/1508.06576)

---
## Setup


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

base_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_reference_image_path = keras.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
result_prefix = "paris_generated"

# Weights of the different loss components
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# Dimensions of the generated picture.
width, height = keras.utils.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
```

<div class="k-default-codeblock">
```
Downloading data from https://i.imgur.com/F28w3Ac.jpg
106496/102437 [===============================] - 0s 0us/step
114688/102437 [=================================] - 0s 0us/step
Downloading data from https://i.imgur.com/9ooB60I.jpg
942080/935806 [==============================] - 0s 0us/step
950272/935806 [==============================] - 0s 0us/step

```
</div>
---
## Let's take a look at our base (content) image and our style reference image


```python
from IPython.display import Image, display

display(Image(base_image_path))
display(Image(style_reference_image_path))
```


    
![jpeg](/img/examples/generative/neural_style_transfer/neural_style_transfer_5_0.jpg)
    



    
![jpeg](/img/examples/generative/neural_style_transfer/neural_style_transfer_5_1.jpg)
    


---
## Image preprocessing / deprocessing utilities


```python

def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.utils.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

```

---
## Compute the style transfer loss

First, we need to define 4 utility functions:

- `gram_matrix` (used to compute the style loss)
- The `style_loss` function, which keeps the generated image close to the local textures
of the style reference image
- The `content_loss` function, which keeps the high-level representation of the
generated image close to that of the base image
- The `total_variation_loss` function, a regularization loss which keeps the generated
image locally-coherent


```python
# The gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

```

Next, let's create a feature extraction model that retrieves the intermediate activations
of VGG19 (as a dict, by name).


```python
# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
80142336/80134624 [==============================] - 1s 0us/step
80150528/80134624 [==============================] - 1s 0us/step

```
</div>
Finally, here's the code that computes the style transfer loss.


```python
# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
content_layer_name = "block5_conv2"


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

```

---
## Add a tf.function decorator to loss & gradient computation

To compile it, and thus make it fast.


```python

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

```

---
## The training loop

Repeatedly run vanilla gradient descent steps to minimize the loss, and save the
resulting image every 100 iterations.

We decay the learning rate by 0.96 every 100 steps.


```python
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        keras.utils.save_img(fname, img)
```

<div class="k-default-codeblock">
```
Iteration 100: loss=11021.84
Iteration 200: loss=8516.83
Iteration 300: loss=7572.59
Iteration 400: loss=7062.75
Iteration 500: loss=6734.12
Iteration 600: loss=6498.55
Iteration 700: loss=6319.12
Iteration 800: loss=6176.68
Iteration 900: loss=6060.16
Iteration 1000: loss=5962.53
Iteration 1100: loss=5879.59
Iteration 1200: loss=5808.41
Iteration 1300: loss=5746.57
Iteration 1400: loss=5692.11
Iteration 1500: loss=5643.77
Iteration 1600: loss=5600.53
Iteration 1700: loss=5561.75
Iteration 1800: loss=5526.84
Iteration 1900: loss=5495.23
Iteration 2000: loss=5466.59
Iteration 2100: loss=5440.56
Iteration 2200: loss=5416.80
Iteration 2300: loss=5395.01
Iteration 2400: loss=5375.02
Iteration 2500: loss=5356.57
Iteration 2600: loss=5339.50
Iteration 2700: loss=5323.70
Iteration 2800: loss=5309.09
Iteration 2900: loss=5295.48
Iteration 3000: loss=5282.80
Iteration 3100: loss=5270.98
Iteration 3200: loss=5259.91
Iteration 3300: loss=5249.54
Iteration 3400: loss=5239.84
Iteration 3500: loss=5230.77
Iteration 3600: loss=5222.23
Iteration 3700: loss=5214.22
Iteration 3800: loss=5206.67
Iteration 3900: loss=5199.56
Iteration 4000: loss=5192.88

```
</div>
After 4000 iterations, you get the following result:


```python
display(Image(result_prefix + "_at_iteration_4000.png"))
```


    
![png](/img/examples/generative/neural_style_transfer/neural_style_transfer_19_0.png)
    


**Example available on HuggingFace**
Trained Model | Demo 
--- | --- 
[![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Neural%20style%20transfer-black.svg)](https://huggingface.co/keras-io/VGG19) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Neural%20style%20transfer-black.svg)](https://huggingface.co/spaces/keras-io/neural-style-transfer) 
