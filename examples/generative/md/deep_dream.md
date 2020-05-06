# Deep Dream

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2016/01/13<br>
**Last modified:** 2020/05/02<br>
**Description:** Generating Deep Dreams with Keras.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/https://github.com/keras-team/keras-io/blob/master/examples/generative/deep_dream/ipynb/deep_dream.py)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/https://github.com/keras-team/keras-io/blob/master/examples/generative/deep_dream/deep_dream.py)



---
## Introduction

"Deep dream" is an image-filtering technique which consists of taking an image
classification model, and running gradient ascent over an input image to
try to maximize the activations of specific layers (and sometimes, specific units in
specific layers) for this input. It produces hallucination-like visuals.

It was first introduced by Alexander Mordvintsev from Google in July 2015.

Process:

- Load the original image.
- Define a number of processing scales ("octaves"),
from smallest to largest.
- Resize the original image to the smallest scale.
- For every scale, starting with the smallest (i.e. current one):
    - Run gradient ascent
    - Upscale image to the next scale
    - Reinject the detail that was lost at upscaling time
- Stop when we are back to the original size.
To obtain the detail lost during upscaling, we simply
take the original image, shrink it down, upscale it,
and compare the result to the (resized) original image.

---
## Setup


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3

base_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/aGBdQyK.jpg")
result_prefix = "sky_dream"

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
layer_settings = {
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}

# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 15.0

```

This is our base image:


```python
from IPython.display import Image, display

display(Image(base_image_path))

```


![jpeg](/img/examples/generative/deep_dream/deep_dream_5_0.jpeg)


Let's set up some image preprocessing/deprocessing utilities:


```python

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate arrays.
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # Util function to convert a NumPy array into a valid image.
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Undo inception v3 preprocession
    x /= 2.0
    x += 0.5
    x *= 255.0
    # Convert to uint8 and clip to the valid range [0, 255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


```

---
## Compute the Deep Dream loss

First, build a feature extraction model to retrieve the activations of our target layers
given an input image.


```python
# Build an InceptionV3 model loaded with pre-trained ImageNet weights
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

# Set up a model that returns the activation values for every target layer
# (as a dict)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

```

The actual loss computation is very simple:


```python

def compute_loss(input_image):
    features = feature_extractor(input_image)
    # Initialize the loss
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss


```

---
## Set up the gradient ascent loop for one octave


```python

@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img


```

---
## Run the training loop, iterating over different octaves


```python
original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img)  # Make a copy
for i, shape in enumerate(successive_shapes):
    print("Processing octave %d with shape %s" % (i, shape))
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.preprocessing.image.save_img(result_prefix + ".png", deprocess_image(img.numpy()))

```

<div class="k-default-codeblock">
```
Processing octave 0 with shape (309, 464)
... Loss value at step 0: 0.59
... Loss value at step 1: 0.80
... Loss value at step 2: 1.06
... Loss value at step 3: 1.30
... Loss value at step 4: 1.53
... Loss value at step 5: 1.76
... Loss value at step 6: 1.98
... Loss value at step 7: 2.19
... Loss value at step 8: 2.41
... Loss value at step 9: 2.59
... Loss value at step 10: 2.82
... Loss value at step 11: 2.97
... Loss value at step 12: 3.18
... Loss value at step 13: 3.31
... Loss value at step 14: 3.57
... Loss value at step 15: 3.68
... Loss value at step 16: 3.92
... Loss value at step 17: 4.06
... Loss value at step 18: 4.40
... Loss value at step 19: 4.46
Processing octave 1 with shape (433, 650)
... Loss value at step 0: 0.85
... Loss value at step 1: 1.36
... Loss value at step 2: 1.84
... Loss value at step 3: 2.26
... Loss value at step 4: 2.67
... Loss value at step 5: 3.02
... Loss value at step 6: 3.31
... Loss value at step 7: 3.59
... Loss value at step 8: 3.93
... Loss value at step 9: 4.18
... Loss value at step 10: 4.37
... Loss value at step 11: 4.65
... Loss value at step 12: 5.04
... Loss value at step 13: 4.99
... Loss value at step 14: 5.56
... Loss value at step 15: 5.49
... Loss value at step 16: 5.79
... Loss value at step 17: 5.98
... Loss value at step 18: 6.38
... Loss value at step 19: 6.63
Processing octave 2 with shape (607, 910)
... Loss value at step 0: 0.93
... Loss value at step 1: 1.47
... Loss value at step 2: 1.96
... Loss value at step 3: 2.39
... Loss value at step 4: 2.77
... Loss value at step 5: 3.12
... Loss value at step 6: 3.46
... Loss value at step 7: 3.77
... Loss value at step 8: 4.12
... Loss value at step 9: 4.39
... Loss value at step 10: 4.66
... Loss value at step 11: 5.05
... Loss value at step 12: 5.20
... Loss value at step 13: 5.51
... Loss value at step 14: 5.74
... Loss value at step 15: 6.16
... Loss value at step 16: 6.38
... Loss value at step 17: 6.65
... Loss value at step 18: 6.82
... Loss value at step 19: 7.13

```
</div>
Display the result.


```python
display(Image(result_prefix + ".png"))

```


![png](/img/examples/generative/deep_dream/deep_dream_17_0.png)

