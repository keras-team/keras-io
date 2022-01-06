# Image segmentation with a U-Net-like architecture

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2019/03/20<br>
**Last modified:** 2020/04/20<br>
**Description:** Image segmentation model trained from scratch on the Oxford Pets dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/oxford_pets_image_segmentation.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/oxford_pets_image_segmentation.py)



---
## Download the data


```python
!curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz
```

<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  755M  100  755M    0     0  6943k      0  0:01:51  0:01:51 --:--:-- 7129k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 18.2M  100 18.2M    0     0  5692k      0  0:00:03  0:00:03 --:--:-- 5692k

```
</div>
---
## Prepare paths of input images and target segmentation masks


```python
import os

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)
```

<div class="k-default-codeblock">
```
Number of samples: 7390
images/Abyssinian_1.jpg | annotations/trimaps/Abyssinian_1.png
images/Abyssinian_10.jpg | annotations/trimaps/Abyssinian_10.png
images/Abyssinian_100.jpg | annotations/trimaps/Abyssinian_100.png
images/Abyssinian_101.jpg | annotations/trimaps/Abyssinian_101.png
images/Abyssinian_102.jpg | annotations/trimaps/Abyssinian_102.png
images/Abyssinian_103.jpg | annotations/trimaps/Abyssinian_103.png
images/Abyssinian_104.jpg | annotations/trimaps/Abyssinian_104.png
images/Abyssinian_105.jpg | annotations/trimaps/Abyssinian_105.png
images/Abyssinian_106.jpg | annotations/trimaps/Abyssinian_106.png
images/Abyssinian_107.jpg | annotations/trimaps/Abyssinian_107.png

```
</div>
---
## What does one input image and corresponding segmentation mask look like?


```python
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

# Display input image #7
display(Image(filename=input_img_paths[9]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)
```


    
![jpeg](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_6_0.jpeg)
    



    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_6_1.png)
    


---
## Prepare `Sequence` class to load & vectorize batches of data


```python
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y

```

---
## Prepare U-Net Xception-style model


```python
from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()
```

<div class="k-default-codeblock">
```
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 160, 160, 3) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 80, 80, 32)   896         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 80, 80, 32)   128         conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 80, 80, 32)   0           batch_normalization[0][0]        
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 80, 80, 32)   0           activation[0][0]                 
__________________________________________________________________________________________________
separable_conv2d (SeparableConv (None, 80, 80, 64)   2400        activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 80, 80, 64)   256         separable_conv2d[0][0]           
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 80, 80, 64)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
separable_conv2d_1 (SeparableCo (None, 80, 80, 64)   4736        activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 80, 80, 64)   256         separable_conv2d_1[0][0]         
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 40, 40, 64)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 40, 40, 64)   2112        activation[0][0]                 
__________________________________________________________________________________________________
add (Add)                       (None, 40, 40, 64)   0           max_pooling2d[0][0]              
                                                                 conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 40, 40, 64)   0           add[0][0]                        
__________________________________________________________________________________________________
separable_conv2d_2 (SeparableCo (None, 40, 40, 128)  8896        activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 40, 40, 128)  512         separable_conv2d_2[0][0]         
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 40, 40, 128)  0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
separable_conv2d_3 (SeparableCo (None, 40, 40, 128)  17664       activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 40, 40, 128)  512         separable_conv2d_3[0][0]         
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 20, 20, 128)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 20, 20, 128)  8320        add[0][0]                        
__________________________________________________________________________________________________
add_1 (Add)                     (None, 20, 20, 128)  0           max_pooling2d_1[0][0]            
                                                                 conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 20, 20, 128)  0           add_1[0][0]                      
__________________________________________________________________________________________________
separable_conv2d_4 (SeparableCo (None, 20, 20, 256)  34176       activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 20, 20, 256)  1024        separable_conv2d_4[0][0]         
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 20, 20, 256)  0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
separable_conv2d_5 (SeparableCo (None, 20, 20, 256)  68096       activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 20, 20, 256)  1024        separable_conv2d_5[0][0]         
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 10, 10, 256)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 10, 10, 256)  33024       add_1[0][0]                      
__________________________________________________________________________________________________
add_2 (Add)                     (None, 10, 10, 256)  0           max_pooling2d_2[0][0]            
                                                                 conv2d_3[0][0]                   
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 10, 10, 256)  0           add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 10, 10, 256)  590080      activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 10, 10, 256)  1024        conv2d_transpose[0][0]           
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 10, 10, 256)  0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 10, 10, 256)  590080      activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 10, 10, 256)  1024        conv2d_transpose_1[0][0]         
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 20, 20, 256)  0           add_2[0][0]                      
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 20, 20, 256)  0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 20, 20, 256)  65792       up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
add_3 (Add)                     (None, 20, 20, 256)  0           up_sampling2d[0][0]              
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 20, 20, 256)  0           add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 20, 20, 128)  295040      activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 20, 20, 128)  512         conv2d_transpose_2[0][0]         
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 20, 20, 128)  0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 20, 20, 128)  147584      activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 20, 20, 128)  512         conv2d_transpose_3[0][0]         
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 40, 40, 256)  0           add_3[0][0]                      
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 40, 40, 128)  0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 40, 40, 128)  32896       up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
add_4 (Add)                     (None, 40, 40, 128)  0           up_sampling2d_2[0][0]            
                                                                 conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 40, 40, 128)  0           add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, 40, 40, 64)   73792       activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 40, 40, 64)   256         conv2d_transpose_4[0][0]         
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 40, 40, 64)   0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_5 (Conv2DTrans (None, 40, 40, 64)   36928       activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 40, 40, 64)   256         conv2d_transpose_5[0][0]         
__________________________________________________________________________________________________
up_sampling2d_5 (UpSampling2D)  (None, 80, 80, 128)  0           add_4[0][0]                      
__________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)  (None, 80, 80, 64)   0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 80, 80, 64)   8256        up_sampling2d_5[0][0]            
__________________________________________________________________________________________________
add_5 (Add)                     (None, 80, 80, 64)   0           up_sampling2d_4[0][0]            
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 80, 80, 64)   0           add_5[0][0]                      
__________________________________________________________________________________________________
conv2d_transpose_6 (Conv2DTrans (None, 80, 80, 32)   18464       activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 80, 80, 32)   128         conv2d_transpose_6[0][0]         
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 80, 80, 32)   0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_7 (Conv2DTrans (None, 80, 80, 32)   9248        activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 80, 80, 32)   128         conv2d_transpose_7[0][0]         
__________________________________________________________________________________________________
up_sampling2d_7 (UpSampling2D)  (None, 160, 160, 64) 0           add_5[0][0]                      
__________________________________________________________________________________________________
up_sampling2d_6 (UpSampling2D)  (None, 160, 160, 32) 0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 160, 160, 32) 2080        up_sampling2d_7[0][0]            
__________________________________________________________________________________________________
add_6 (Add)                     (None, 160, 160, 32) 0           up_sampling2d_6[0][0]            
                                                                 conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 160, 160, 3)  867         add_6[0][0]                      
==================================================================================================
Total params: 2,058,979
Trainable params: 2,055,203
Non-trainable params: 3,776
__________________________________________________________________________________________________

```
</div>
---
## Set aside a validation split


```python
import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
```

---
## Train the model


```python
# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
```

<div class="k-default-codeblock">
```
Epoch 1/15
  2/199 [..............................] - ETA: 13s - loss: 5.4602WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0462s vs `on_train_batch_end` time: 0.0935s). Check your callbacks.
199/199 [==============================] - 32s 161ms/step - loss: 0.9396 - val_loss: 3.7159
Epoch 2/15
199/199 [==============================] - 32s 159ms/step - loss: 0.4911 - val_loss: 2.2709
Epoch 3/15
199/199 [==============================] - 32s 160ms/step - loss: 0.4205 - val_loss: 0.5184
Epoch 4/15
199/199 [==============================] - 32s 159ms/step - loss: 0.3739 - val_loss: 0.4584
Epoch 5/15
199/199 [==============================] - 32s 160ms/step - loss: 0.3416 - val_loss: 0.3968
Epoch 6/15
199/199 [==============================] - 32s 159ms/step - loss: 0.3131 - val_loss: 0.4059
Epoch 7/15
199/199 [==============================] - 31s 157ms/step - loss: 0.2895 - val_loss: 0.3963
Epoch 8/15
199/199 [==============================] - 31s 156ms/step - loss: 0.2695 - val_loss: 0.4035
Epoch 9/15
199/199 [==============================] - 31s 157ms/step - loss: 0.2528 - val_loss: 0.4184
Epoch 10/15
199/199 [==============================] - 31s 157ms/step - loss: 0.2360 - val_loss: 0.3950
Epoch 11/15
199/199 [==============================] - 31s 157ms/step - loss: 0.2247 - val_loss: 0.4139
Epoch 12/15
199/199 [==============================] - 31s 157ms/step - loss: 0.2126 - val_loss: 0.3861
Epoch 13/15
199/199 [==============================] - 31s 157ms/step - loss: 0.2026 - val_loss: 0.4138
Epoch 14/15
199/199 [==============================] - 31s 156ms/step - loss: 0.1932 - val_loss: 0.4265
Epoch 15/15
199/199 [==============================] - 31s 157ms/step - loss: 0.1857 - val_loss: 0.3959

<tensorflow.python.keras.callbacks.History at 0x7f6e11107b70>

```
</div>
---
## Visualize predictions


```python
# Generate predictions for all images in the validation set

val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


# Display results for validation image #10
i = 10

# Display input image
display(Image(filename=val_input_img_paths[i]))

# Display ground-truth target mask
img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.
```


    
![jpeg](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_16_0.jpeg)
    



    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_16_1.png)
    



    
![png](/img/examples/vision/oxford_pets_image_segmentation/oxford_pets_image_segmentation_16_2.png)
    

