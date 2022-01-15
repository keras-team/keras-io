# Convolutional autoencoder for image denoising

**Author:** [Santiago L. Valdarrama](https://twitter.com/svpino)<br>
**Date created:** 2021/03/01<br>
**Last modified:** 2021/03/01<br>
**Description:** How to train a deep convolutional autoencoder for image denoising.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/autoencoder.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/autoencoder.py)



---
## Introduction

This example demonstrates how to implement a deep convolutional autoencoder
for image denoising, mapping noisy digits images from the MNIST dataset to
clean digits images. This implementation is based on an original blog post
titled [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
by [François Chollet](https://twitter.com/fchollet).

---
## Setup


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

```

---
## Prepare the data


```python
# Since we only need images from the dataset to encode and decode, we
# won't use the labels.
(train_data, _), (test_data, _) = mnist.load_data()

# Normalize and reshape the data
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Create a copy of the data with added noise
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
display(train_data, noisy_train_data)
```


![png](/img/examples/vision/autoencoder/autoencoder_5_0.png)


---
## Build the autoencoder

We are going to use the Functional API to build our convolutional autoencoder.


```python
input = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()
```

<div class="k-default-codeblock">
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 28, 28, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 14, 14, 32)        9248      
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 32)        9248      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 1)         289       
=================================================================
Total params: 28,353
Trainable params: 28,353
Non-trainable params: 0
_________________________________________________________________

```
</div>
Now we can train our autoencoder using `train_data` as both our input data
and target. Notice we are setting up the validation data using the same
format.


```python
autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(test_data, test_data),
)
```

<div class="k-default-codeblock">
```
Epoch 1/50
469/469 [==============================] - 20s 43ms/step - loss: 0.1354 - val_loss: 0.0735
Epoch 2/50
469/469 [==============================] - 21s 45ms/step - loss: 0.0719 - val_loss: 0.0698
Epoch 3/50
469/469 [==============================] - 22s 47ms/step - loss: 0.0695 - val_loss: 0.0682
Epoch 4/50
469/469 [==============================] - 23s 50ms/step - loss: 0.0684 - val_loss: 0.0674
Epoch 5/50
469/469 [==============================] - 24s 51ms/step - loss: 0.0676 - val_loss: 0.0669
Epoch 6/50
469/469 [==============================] - 26s 55ms/step - loss: 0.0671 - val_loss: 0.0663
Epoch 7/50
469/469 [==============================] - 27s 57ms/step - loss: 0.0667 - val_loss: 0.0660
Epoch 8/50
469/469 [==============================] - 26s 56ms/step - loss: 0.0663 - val_loss: 0.0657
Epoch 9/50
469/469 [==============================] - 28s 59ms/step - loss: 0.0642 - val_loss: 0.0639
Epoch 21/50
469/469 [==============================] - 28s 60ms/step - loss: 0.0642 - val_loss: 0.0638
Epoch 22/50
469/469 [==============================] - 29s 62ms/step - loss: 0.0632 - val_loss: 0.0629
Epoch 38/50
397/469 [========================>.....] - ETA: 4s - loss: 0.0632

```
</div>
Let's predict on our test dataset and display the original image together with
the prediction from our autoencoder.

Notice how the predictions are pretty close to the original images, although
not quite the same.


```python
predictions = autoencoder.predict(test_data)
display(test_data, predictions)
```


![png](/img/examples/vision/autoencoder/autoencoder_11_0.png)


Now that we know that our autoencoder works, let's retrain it using the noisy
data as our input and the clean data as our target. We want our autoencoder to
learn how to denoise the images.


```python
autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=100,
    batch_size=128,
    shuffle=True,
    validation_data=(noisy_test_data, test_data),
)
```

<div class="k-default-codeblock">
```
Epoch 1/100
469/469 [==============================] - 28s 59ms/step - loss: 0.1027 - val_loss: 0.0946
Epoch 2/100
469/469 [==============================] - 27s 57ms/step - loss: 0.0942 - val_loss: 0.0924
Epoch 3/100
469/469 [==============================] - 27s 58ms/step - loss: 0.0925 - val_loss: 0.0913
Epoch 4/100
469/469 [==============================] - 28s 60ms/step - loss: 0.0915 - val_loss: 0.0905
Epoch 5/100
469/469 [==============================] - 31s 66ms/step - loss: 0.0908 - val_loss: 0.0897
Epoch 6/100
469/469 [==============================] - 30s 64ms/step - loss: 0.0902 - val_loss: 0.0893
Epoch 7/100
469/469 [==============================] - 28s 60ms/step - loss: 0.0897 - val_loss: 0.0887
Epoch 8/100
469/469 [==============================] - 31s 66ms/step - loss: 0.0872 - val_loss: 0.0867
Epoch 19/100
469/469 [==============================] - 30s 64ms/step - loss: 0.0860 - val_loss: 0.0854
Epoch 35/100
469/469 [==============================] - 32s 68ms/step - loss: 0.0854 - val_loss: 0.0849
Epoch 52/100
469/469 [==============================] - 28s 60ms/step - loss: 0.0851 - val_loss: 0.0847
Epoch 68/100
469/469 [==============================] - 31s 66ms/step - loss: 0.0851 - val_loss: 0.0848
Epoch 69/100
469/469 [==============================] - 31s 65ms/step - loss: 0.0849 - val_loss: 0.0847
Epoch 84/100
469/469 [==============================] - 29s 63ms/step - loss: 0.0848 - val_loss: 0.0846

<tensorflow.python.keras.callbacks.History at 0x7fbb195a3a90>

```
</div>
Let's now predict on the noisy data and display the results of our autoencoder.

Notice how the autoencoder does an amazing job at removing the noise from the
input images.


```python
predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions)
```


![png](/img/examples/vision/autoencoder/autoencoder_15_0.png)

