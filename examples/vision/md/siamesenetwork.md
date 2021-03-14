# Siamese Network

**Author:** [hazemessamm](https://twitter.com/hazemessamm)<br>
**Date created:** 2021/03/13<br>
**Last modified:** 2021/03/14<br>
**Description:** Siamese network with custom training loop and dataset generator.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/siamesenetwork.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/siamesenetwork.py)



### Setup


```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from tensorflow.keras import losses, optimizers
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import applications
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import Sequence

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
```

---
## Siamese Network

[Siamese Network](https://en.wikipedia.org/wiki/Siamese_neural_network) is used to solve
many problems like detecting question duplicates, face recognition by comparing the
similarity of the inputs by comparing their feature vectors.

First we need to have a dataset that contains 3 Images, 2 are similar and 1 is different,
they are called Anchor image, Positive Image and Negative image respectively, we need to
tell the network that the anchor image and the positive image are similar, we also need
to tell it that the anchor image and the negative image are NOT similar, we can do that
by the Triplet Loss Function.

Triplet Loss function:

L(Anchor, Positive, Negative) = max((distance(f(Anchor), f(Positive)) -
distance(f(Anchor), f(Negative)))**2, 0.0)

Note that the weights are shared which mean that we are only using one model for
prediction and training

You can find the dataset here:
https://drive.google.com/drive/folders/1qQJHA5m-vLMAkBfWEWgGW9n61gC_orHl

Also more info found here: https://sites.google.com/view/totally-looks-like-dataset

Image from:
https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942


![png](/img/examples/vision/siamesenetwork/1_0E9104t29iMBmtvq7G1G6Q.png)

First we get the paths of the datasets in siamese networks we usually have two folders
each folder has images and every image has a corresponding similar picture in the other
folder.

---
## Preparing data


```python
dataset = os.path.join("/mnt/media", "TotallyLooksLikeDataset")
anchor_images_path = os.path.join(dataset, "left")
positive_images_path = os.path.join(dataset, "right")
target_shape = (200, 200)
```

---
## SiameseDatasetGenerator

This class inherits from Sequence which is used to generate images for training, the
reason of using a generator is that there are datasets which contain a lot of high
resolution images and we cannot load all of them in our memory so we just generate
batches of them while training we inherit it so we can use it in training

1) We override the __len__ method by returning our number of batches so keras can know
how many batches available.
2) We override __getitem__ method so we can access any index of an array

### Negative images:

Negative images are just random images we sample from our dataset. every example should
contain 3 images (Anchor, Positive and Negative). The negative image should NOT be the
same as the Anchor or the Positive images, We use a set() that stores the names of the
anchor and positive images so when we sample the negative images we avoid getting any
image that exist in the set()

### Batch shuffle:

We need to shuffle the batch so we can have random examples

### Image preprocessing:

After creating a list of paths for Anchor images, positive images, negative images we
pass these lists to the preprocess_img()
because we need to load the image given the path we have and we need to convert it into
tensor by using img_to_array()


```python

class SiameseDatasetGenerator(Sequence):
    def __init__(
        self,
        anchor_images_path,
        positive_images_path,
        target_shape,
        batch_size=128,
        shuffle=True,
    ):
        self.anchor_images_path = (
            anchor_images_path  # store the path of the anchor images
        )
        self.positive_images_path = (
            positive_images_path  # store the path of the positive images
        )
        self.target_shape = target_shape  ##store image shape
        # list the contents (images) of the specified directory
        self.anchor_images = np.array(os.listdir(positive_images_path))
        self.positive_images = np.array(os.listdir(positive_images_path))
        self.batch_size = batch_size
        self.num_examples = len(self.anchor_images)
        self.num_batches = self.num_examples // batch_size
        self.shuffle = shuffle

    """
    we use __len__ method that is called
    to get the length of the batches
    it is called when we call len()
    """

    def __len__(self):
        return self.num_batches

    """
    this method allows us to get batches when we
    access the instance the same way we access a list
    e.g. dataset[0] will call __getitem__(index=0)
    """

    def __getitem__(self, index):
        current_batch = index * self.batch_size
        # here we get batches of data by using slicing
        anchor_imgs = self.anchor_images[
            current_batch : current_batch + self.batch_size
        ]
        positive_imgs = self.positive_images[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        # store the loaded images to avoid reloading them in negative images
        # we store them in a set for faster access
        loaded_examples = set([i for i in anchor_imgs])

        negative_imgs = np.array(
            self.get_negative_imgs(
                from_anchor_dir=random.choice([True, False]),
                loaded_examples=loaded_examples,
            )
        )
        anchor_imgs = np.array(
            [os.path.join(self.anchor_images_path + "/", img) for img in anchor_imgs]
        )
        positive_imgs = np.array(
            [
                os.path.join(self.positive_images_path + "/", img)
                for img in positive_imgs
            ]
        )

        if self.shuffle:
            # create a list of random numbers to use it when we shuffle the batches
            random_shuffle = random.choices(
                [*range(0, len(anchor_imgs))], k=len(anchor_imgs)
            )
            anchor_imgs = anchor_imgs[random_shuffle]
            positive_imgs = positive_imgs[random_shuffle]
            negative_imgs = negative_imgs[random_shuffle]

        anchor_imgs = self.preprocess_img(anchor_imgs)
        positive_imgs = self.preprocess_img(positive_imgs)
        negative_imgs = self.preprocess_img(negative_imgs)

        # here if the batch size equal one we just convert the images into numpy
        # and expand the dimension of this batch by adding 1 in the first axis
        if self.batch_size == 1:
            return np.expand_dims(
                np.array([anchor_imgs, positive_imgs, negative_imgs]), axis=0
            )
        # Add the batch_size dimension in the first axis by using permute()
        return tf.keras.backend.permute_dimensions(
            np.array([anchor_imgs, positive_imgs, negative_imgs]), (1, 0, 2, 3, 4)
        )

    def get_negative_imgs(self, from_anchor_dir=True, loaded_examples={}):
        # load the negative_imgs by randomly loading it from the anchor or the positive images
        negative_imgs = []
        if from_anchor_dir:
            negative_imgs = random.choices(
                [img for img in self.anchor_images if img not in loaded_examples],
                k=self.batch_size,
            )
            negative_imgs = [
                os.path.join(self.anchor_images_path + "/", img)
                for img in negative_imgs
            ]
        else:
            negative_imgs = random.choices(
                [img for img in self.positive_images if img not in loaded_examples],
                k=self.batch_size,
            )
            negative_imgs = [
                os.path.join(self.positive_images_path + "/", img)
                for img in negative_imgs
            ]
        return negative_imgs

    def preprocess_img(self, imgs):
        output = []
        for img_path in imgs:
            img = preprocessing.image.load_img(img_path, target_size=self.target_shape)
            img = preprocessing.image.img_to_array(img)
            output.append(img)
        if len(output) == 1:
            return output[0]
        return tuple(output)


dataset = SiameseDatasetGenerator(
    anchor_images_path, positive_images_path, target_shape, 64, False
)

# this function just visalize each random 3 images (anchor, positive, negative)
def visualize():
    example = dataset[random.randint(0, dataset.batch_size)]
    img1, img2, img3 = (
        preprocessing.image.array_to_img(example[:, 0][0]),
        preprocessing.image.array_to_img(example[:, 1][0]),
        preprocessing.image.array_to_img(example[:, 2][0]),
    )
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(img3)
    plt.savefig("test.png")


visualize()  # as you see we have two similar images and one different image
```


    
![png](/img/examples/vision/siamesenetwork/siamesenetwork_15_0.png)
    


---
## Loading a pre-trained model

Here we use ResNet50 architecture, we use "imagenet" weights, also we pass the image shape
Note that include_top means that we do NOT want the top layers


```python
base_cnn = applications.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)
```

---
## Fine Tuning

Here we fine tune the ResNet50 we freeze all layers that exist before "conv5_block1_out"
layer, starting from "conv5_block2_2_relu" layer we unfreeze all the layers so we can
just train these layers


```python
trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable
```

---
## Adding top layers

Here we customize the model by adding Dense layers and Batch Normalization layers. we
start with the image input then we pass the input to the base_cnn then we flatten it.
Finally we pass each layer as an input to the next layer the output layer is just a dense
layer which will act as an embedding for our images.


```python
flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="SiameseNetwork")

embedding.summary()
```

<div class="k-default-codeblock">
```
Model: "SiameseNetwork"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 200, 200, 3) 0                                            
__________________________________________________________________________________________________
conv1_pad (ZeroPadding2D)       (None, 206, 206, 3)  0           input_1[0][0]                    
__________________________________________________________________________________________________
conv1_conv (Conv2D)             (None, 100, 100, 64) 9472        conv1_pad[0][0]                  
__________________________________________________________________________________________________
conv1_bn (BatchNormalization)   (None, 100, 100, 64) 256         conv1_conv[0][0]                 
__________________________________________________________________________________________________
conv1_relu (Activation)         (None, 100, 100, 64) 0           conv1_bn[0][0]                   
__________________________________________________________________________________________________
pool1_pad (ZeroPadding2D)       (None, 102, 102, 64) 0           conv1_relu[0][0]                 
__________________________________________________________________________________________________
pool1_pool (MaxPooling2D)       (None, 50, 50, 64)   0           pool1_pad[0][0]                  
__________________________________________________________________________________________________
conv2_block1_1_conv (Conv2D)    (None, 50, 50, 64)   4160        pool1_pool[0][0]                 
__________________________________________________________________________________________________
conv2_block1_1_bn (BatchNormali (None, 50, 50, 64)   256         conv2_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_1_relu (Activation (None, 50, 50, 64)   0           conv2_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_2_conv (Conv2D)    (None, 50, 50, 64)   36928       conv2_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_2_bn (BatchNormali (None, 50, 50, 64)   256         conv2_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_2_relu (Activation (None, 50, 50, 64)   0           conv2_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_0_conv (Conv2D)    (None, 50, 50, 256)  16640       pool1_pool[0][0]                 
__________________________________________________________________________________________________
conv2_block1_3_conv (Conv2D)    (None, 50, 50, 256)  16640       conv2_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv2_block1_0_bn (BatchNormali (None, 50, 50, 256)  1024        conv2_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_3_bn (BatchNormali (None, 50, 50, 256)  1024        conv2_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv2_block1_add (Add)          (None, 50, 50, 256)  0           conv2_block1_0_bn[0][0]          
                                                                 conv2_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv2_block1_out (Activation)   (None, 50, 50, 256)  0           conv2_block1_add[0][0]           
__________________________________________________________________________________________________
conv2_block2_1_conv (Conv2D)    (None, 50, 50, 64)   16448       conv2_block1_out[0][0]           
__________________________________________________________________________________________________
conv2_block2_1_bn (BatchNormali (None, 50, 50, 64)   256         conv2_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_1_relu (Activation (None, 50, 50, 64)   0           conv2_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_2_conv (Conv2D)    (None, 50, 50, 64)   36928       conv2_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_2_bn (BatchNormali (None, 50, 50, 64)   256         conv2_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_2_relu (Activation (None, 50, 50, 64)   0           conv2_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_3_conv (Conv2D)    (None, 50, 50, 256)  16640       conv2_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv2_block2_3_bn (BatchNormali (None, 50, 50, 256)  1024        conv2_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv2_block2_add (Add)          (None, 50, 50, 256)  0           conv2_block1_out[0][0]           
                                                                 conv2_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv2_block2_out (Activation)   (None, 50, 50, 256)  0           conv2_block2_add[0][0]           
__________________________________________________________________________________________________
conv2_block3_1_conv (Conv2D)    (None, 50, 50, 64)   16448       conv2_block2_out[0][0]           
__________________________________________________________________________________________________
conv2_block3_1_bn (BatchNormali (None, 50, 50, 64)   256         conv2_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_1_relu (Activation (None, 50, 50, 64)   0           conv2_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_2_conv (Conv2D)    (None, 50, 50, 64)   36928       conv2_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_2_bn (BatchNormali (None, 50, 50, 64)   256         conv2_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_2_relu (Activation (None, 50, 50, 64)   0           conv2_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_3_conv (Conv2D)    (None, 50, 50, 256)  16640       conv2_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv2_block3_3_bn (BatchNormali (None, 50, 50, 256)  1024        conv2_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv2_block3_add (Add)          (None, 50, 50, 256)  0           conv2_block2_out[0][0]           
                                                                 conv2_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv2_block3_out (Activation)   (None, 50, 50, 256)  0           conv2_block3_add[0][0]           
__________________________________________________________________________________________________
conv3_block1_1_conv (Conv2D)    (None, 25, 25, 128)  32896       conv2_block3_out[0][0]           
__________________________________________________________________________________________________
conv3_block1_1_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_1_relu (Activation (None, 25, 25, 128)  0           conv3_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_2_conv (Conv2D)    (None, 25, 25, 128)  147584      conv3_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_2_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_2_relu (Activation (None, 25, 25, 128)  0           conv3_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_0_conv (Conv2D)    (None, 25, 25, 512)  131584      conv2_block3_out[0][0]           
__________________________________________________________________________________________________
conv3_block1_3_conv (Conv2D)    (None, 25, 25, 512)  66048       conv3_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block1_0_bn (BatchNormali (None, 25, 25, 512)  2048        conv3_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_3_bn (BatchNormali (None, 25, 25, 512)  2048        conv3_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block1_add (Add)          (None, 25, 25, 512)  0           conv3_block1_0_bn[0][0]          
                                                                 conv3_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block1_out (Activation)   (None, 25, 25, 512)  0           conv3_block1_add[0][0]           
__________________________________________________________________________________________________
conv3_block2_1_conv (Conv2D)    (None, 25, 25, 128)  65664       conv3_block1_out[0][0]           
__________________________________________________________________________________________________
conv3_block2_1_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_1_relu (Activation (None, 25, 25, 128)  0           conv3_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_2_conv (Conv2D)    (None, 25, 25, 128)  147584      conv3_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_2_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_2_relu (Activation (None, 25, 25, 128)  0           conv3_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_3_conv (Conv2D)    (None, 25, 25, 512)  66048       conv3_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block2_3_bn (BatchNormali (None, 25, 25, 512)  2048        conv3_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block2_add (Add)          (None, 25, 25, 512)  0           conv3_block1_out[0][0]           
                                                                 conv3_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block2_out (Activation)   (None, 25, 25, 512)  0           conv3_block2_add[0][0]           
__________________________________________________________________________________________________
conv3_block3_1_conv (Conv2D)    (None, 25, 25, 128)  65664       conv3_block2_out[0][0]           
__________________________________________________________________________________________________
conv3_block3_1_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_1_relu (Activation (None, 25, 25, 128)  0           conv3_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_2_conv (Conv2D)    (None, 25, 25, 128)  147584      conv3_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_2_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_2_relu (Activation (None, 25, 25, 128)  0           conv3_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_3_conv (Conv2D)    (None, 25, 25, 512)  66048       conv3_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block3_3_bn (BatchNormali (None, 25, 25, 512)  2048        conv3_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block3_add (Add)          (None, 25, 25, 512)  0           conv3_block2_out[0][0]           
                                                                 conv3_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block3_out (Activation)   (None, 25, 25, 512)  0           conv3_block3_add[0][0]           
__________________________________________________________________________________________________
conv3_block4_1_conv (Conv2D)    (None, 25, 25, 128)  65664       conv3_block3_out[0][0]           
__________________________________________________________________________________________________
conv3_block4_1_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_1_relu (Activation (None, 25, 25, 128)  0           conv3_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_2_conv (Conv2D)    (None, 25, 25, 128)  147584      conv3_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_2_bn (BatchNormali (None, 25, 25, 128)  512         conv3_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_2_relu (Activation (None, 25, 25, 128)  0           conv3_block4_2_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_3_conv (Conv2D)    (None, 25, 25, 512)  66048       conv3_block4_2_relu[0][0]        
__________________________________________________________________________________________________
conv3_block4_3_bn (BatchNormali (None, 25, 25, 512)  2048        conv3_block4_3_conv[0][0]        
__________________________________________________________________________________________________
conv3_block4_add (Add)          (None, 25, 25, 512)  0           conv3_block3_out[0][0]           
                                                                 conv3_block4_3_bn[0][0]          
__________________________________________________________________________________________________
conv3_block4_out (Activation)   (None, 25, 25, 512)  0           conv3_block4_add[0][0]           
__________________________________________________________________________________________________
conv4_block1_1_conv (Conv2D)    (None, 13, 13, 256)  131328      conv3_block4_out[0][0]           
__________________________________________________________________________________________________
conv4_block1_1_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_1_relu (Activation (None, 13, 13, 256)  0           conv4_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_2_conv (Conv2D)    (None, 13, 13, 256)  590080      conv4_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_2_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_2_relu (Activation (None, 13, 13, 256)  0           conv4_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_0_conv (Conv2D)    (None, 13, 13, 1024) 525312      conv3_block4_out[0][0]           
__________________________________________________________________________________________________
conv4_block1_3_conv (Conv2D)    (None, 13, 13, 1024) 263168      conv4_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block1_0_bn (BatchNormali (None, 13, 13, 1024) 4096        conv4_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_3_bn (BatchNormali (None, 13, 13, 1024) 4096        conv4_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block1_add (Add)          (None, 13, 13, 1024) 0           conv4_block1_0_bn[0][0]          
                                                                 conv4_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block1_out (Activation)   (None, 13, 13, 1024) 0           conv4_block1_add[0][0]           
__________________________________________________________________________________________________
conv4_block2_1_conv (Conv2D)    (None, 13, 13, 256)  262400      conv4_block1_out[0][0]           
__________________________________________________________________________________________________
conv4_block2_1_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_1_relu (Activation (None, 13, 13, 256)  0           conv4_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_2_conv (Conv2D)    (None, 13, 13, 256)  590080      conv4_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_2_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_2_relu (Activation (None, 13, 13, 256)  0           conv4_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_3_conv (Conv2D)    (None, 13, 13, 1024) 263168      conv4_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block2_3_bn (BatchNormali (None, 13, 13, 1024) 4096        conv4_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block2_add (Add)          (None, 13, 13, 1024) 0           conv4_block1_out[0][0]           
                                                                 conv4_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block2_out (Activation)   (None, 13, 13, 1024) 0           conv4_block2_add[0][0]           
__________________________________________________________________________________________________
conv4_block3_1_conv (Conv2D)    (None, 13, 13, 256)  262400      conv4_block2_out[0][0]           
__________________________________________________________________________________________________
conv4_block3_1_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_1_relu (Activation (None, 13, 13, 256)  0           conv4_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_2_conv (Conv2D)    (None, 13, 13, 256)  590080      conv4_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_2_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_2_relu (Activation (None, 13, 13, 256)  0           conv4_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_3_conv (Conv2D)    (None, 13, 13, 1024) 263168      conv4_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block3_3_bn (BatchNormali (None, 13, 13, 1024) 4096        conv4_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block3_add (Add)          (None, 13, 13, 1024) 0           conv4_block2_out[0][0]           
                                                                 conv4_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block3_out (Activation)   (None, 13, 13, 1024) 0           conv4_block3_add[0][0]           
__________________________________________________________________________________________________
conv4_block4_1_conv (Conv2D)    (None, 13, 13, 256)  262400      conv4_block3_out[0][0]           
__________________________________________________________________________________________________
conv4_block4_1_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block4_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_1_relu (Activation (None, 13, 13, 256)  0           conv4_block4_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_2_conv (Conv2D)    (None, 13, 13, 256)  590080      conv4_block4_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_2_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block4_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_2_relu (Activation (None, 13, 13, 256)  0           conv4_block4_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_3_conv (Conv2D)    (None, 13, 13, 1024) 263168      conv4_block4_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block4_3_bn (BatchNormali (None, 13, 13, 1024) 4096        conv4_block4_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block4_add (Add)          (None, 13, 13, 1024) 0           conv4_block3_out[0][0]           
                                                                 conv4_block4_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block4_out (Activation)   (None, 13, 13, 1024) 0           conv4_block4_add[0][0]           
__________________________________________________________________________________________________
conv4_block5_1_conv (Conv2D)    (None, 13, 13, 256)  262400      conv4_block4_out[0][0]           
__________________________________________________________________________________________________
conv4_block5_1_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block5_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_1_relu (Activation (None, 13, 13, 256)  0           conv4_block5_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_2_conv (Conv2D)    (None, 13, 13, 256)  590080      conv4_block5_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_2_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block5_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_2_relu (Activation (None, 13, 13, 256)  0           conv4_block5_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_3_conv (Conv2D)    (None, 13, 13, 1024) 263168      conv4_block5_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block5_3_bn (BatchNormali (None, 13, 13, 1024) 4096        conv4_block5_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block5_add (Add)          (None, 13, 13, 1024) 0           conv4_block4_out[0][0]           
                                                                 conv4_block5_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block5_out (Activation)   (None, 13, 13, 1024) 0           conv4_block5_add[0][0]           
__________________________________________________________________________________________________
conv4_block6_1_conv (Conv2D)    (None, 13, 13, 256)  262400      conv4_block5_out[0][0]           
__________________________________________________________________________________________________
conv4_block6_1_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block6_1_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_1_relu (Activation (None, 13, 13, 256)  0           conv4_block6_1_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_2_conv (Conv2D)    (None, 13, 13, 256)  590080      conv4_block6_1_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_2_bn (BatchNormali (None, 13, 13, 256)  1024        conv4_block6_2_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_2_relu (Activation (None, 13, 13, 256)  0           conv4_block6_2_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_3_conv (Conv2D)    (None, 13, 13, 1024) 263168      conv4_block6_2_relu[0][0]        
__________________________________________________________________________________________________
conv4_block6_3_bn (BatchNormali (None, 13, 13, 1024) 4096        conv4_block6_3_conv[0][0]        
__________________________________________________________________________________________________
conv4_block6_add (Add)          (None, 13, 13, 1024) 0           conv4_block5_out[0][0]           
                                                                 conv4_block6_3_bn[0][0]          
__________________________________________________________________________________________________
conv4_block6_out (Activation)   (None, 13, 13, 1024) 0           conv4_block6_add[0][0]           
__________________________________________________________________________________________________
conv5_block1_1_conv (Conv2D)    (None, 7, 7, 512)    524800      conv4_block6_out[0][0]           
__________________________________________________________________________________________________
conv5_block1_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block1_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_1_relu (Activation (None, 7, 7, 512)    0           conv5_block1_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block1_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block1_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_2_relu (Activation (None, 7, 7, 512)    0           conv5_block1_2_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_0_conv (Conv2D)    (None, 7, 7, 2048)   2099200     conv4_block6_out[0][0]           
__________________________________________________________________________________________________
conv5_block1_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block1_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block1_0_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block1_0_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block1_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block1_add (Add)          (None, 7, 7, 2048)   0           conv5_block1_0_bn[0][0]          
                                                                 conv5_block1_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block1_out (Activation)   (None, 7, 7, 2048)   0           conv5_block1_add[0][0]           
__________________________________________________________________________________________________
conv5_block2_1_conv (Conv2D)    (None, 7, 7, 512)    1049088     conv5_block1_out[0][0]           
__________________________________________________________________________________________________
conv5_block2_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block2_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_1_relu (Activation (None, 7, 7, 512)    0           conv5_block2_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block2_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block2_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_2_relu (Activation (None, 7, 7, 512)    0           conv5_block2_2_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block2_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block2_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block2_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block2_add (Add)          (None, 7, 7, 2048)   0           conv5_block1_out[0][0]           
                                                                 conv5_block2_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block2_out (Activation)   (None, 7, 7, 2048)   0           conv5_block2_add[0][0]           
__________________________________________________________________________________________________
conv5_block3_1_conv (Conv2D)    (None, 7, 7, 512)    1049088     conv5_block2_out[0][0]           
__________________________________________________________________________________________________
conv5_block3_1_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block3_1_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_1_relu (Activation (None, 7, 7, 512)    0           conv5_block3_1_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_2_conv (Conv2D)    (None, 7, 7, 512)    2359808     conv5_block3_1_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_2_bn (BatchNormali (None, 7, 7, 512)    2048        conv5_block3_2_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_2_relu (Activation (None, 7, 7, 512)    0           conv5_block3_2_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_add (Add)          (None, 7, 7, 2048)   0           conv5_block2_out[0][0]           
                                                                 conv5_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_out (Activation)   (None, 7, 7, 2048)   0           conv5_block3_add[0][0]           
__________________________________________________________________________________________________
flatten (Flatten)               (None, 100352)       0           conv5_block3_out[0][0]           
__________________________________________________________________________________________________
dense (Dense)                   (None, 512)          51380736    flatten[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 512)          2048        dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 256)          131328      batch_normalization[0][0]        
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 256)          1024        dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 256)          65792       batch_normalization_1[0][0]      
==================================================================================================
Total params: 75,168,640
Trainable params: 60,510,720
Non-trainable params: 14,657,920
__________________________________________________________________________________________________

```
</div>
---
## Model for training

This model is just used for training we pass to it three input batches (anchor images,
positive images, negative images) and the output will be the output of the model we
defined above, it will be 1 output for each input.


```python
anchor_input = layers.Input(shape=target_shape + (3,))
positive_input = layers.Input(shape=target_shape + (3,))
negative_input = layers.Input(shape=target_shape + (3,))

anchor_output = embedding(anchor_input)
positive_output = embedding(positive_input)
negative_output = embedding(negative_input)

training_model = Model(
    [anchor_input, positive_input, negative_input],
    {
        "anchor_embedding": anchor_output,
        "positive_embedding": positive_output,
        "negative_embedding": negative_output,
    },
)

```

---
## Cosine Similarity Layer

This layer just computes how similar to feature vectors are by computing it using the
Cosine Similarity
We override the call method and implement our own call method.

Check out https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity

We return the negative of the loss because we just need to know how similar they are we
do NOT need to know the loss


```python

class CosineDistance(layers.Layer):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def call(self, img1, img2):
        return -losses.CosineSimilarity(reduction=losses.Reduction.NONE)(img1, img2)

```

---
## Model subclassing

Here we customize our training process and our model.

We override the train_step() method and apply our own loss and our own training process

We also use Triplet loss function as we specified above.

Loss function explaination:

we calculate the distance between the anchor embedding and the positive embedding the
axis = -1 because we want the distance over the features of every example. We also add
alpha which act as extra margin.


```python

class SiameseModel(Model):
    def __init__(self, model, alpha=0.5):
        super(SiameseModel, self).__init__()
        self.embedding = model  # we pass the model to the class
        self.alpha = alpha

    def call(self, inputs):
        pass

    def train_step(self, data):
        # here we create a tape to record our operations so we can get the gradients
        with tf.GradientTape() as tape:
            embeddings = training_model((data[:, 0], data[:, 1], data[:, 2]))

            # Euclidean Distance between anchor and positive
            # axis=-1 so we can get distances over examples
            anchor_positive_dist = tf.reduce_sum(
                tf.square(
                    embeddings["anchor_embedding"] - embeddings["positive_embedding"]
                ),
                -1,
            )

            # Euclidean Distance between anchor and negative
            anchor_negative_dist = tf.reduce_sum(
                tf.square(
                    embeddings["anchor_embedding"] - embeddings["negative_embedding"]
                ),
                -1,
            )

            # getting the loss by subtracting the distances
            loss = anchor_positive_dist - anchor_negative_dist
            # getting the max because we don't want negative loss
            loss = tf.reduce_mean(tf.maximum(loss + self.alpha, 0.0))
        # getting the gradients [loss with respect to trainable weights]
        grads = tape.gradient(loss, training_model.trainable_weights)
        # applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(zip(grads, training_model.trainable_weights))
        return {"Loss": loss}

```

### Training


```python
siamese_model = SiameseModel(embedding)

siamese_model.compile(optimizer=optimizers.Adam(0.0001))

siamese_model.fit(dataset, epochs=2)
```

<div class="k-default-codeblock">
```
Epoch 1/2
94/94 [==============================] - 64s 568ms/step - Loss: 4.7495
Epoch 2/2
94/94 [==============================] - 55s 582ms/step - Loss: 0.3849

<tensorflow.python.keras.callbacks.History at 0x7f215c2b29d0>

```
</div>
### Inference


```python
# here we just load from the dataset an example
# we should NOT test the performace of the model
# using training data but here we are just see how did it learn
example_prediction = dataset[3]
anchor_example = preprocessing.image.array_to_img(example_prediction[:, 0][0])
positive_example = preprocessing.image.array_to_img(example_prediction[:, 1][0])
negative_example = preprocessing.image.array_to_img(example_prediction[:, 2][0])

# here we just plotting the example that we loaded
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(anchor_example)
ax2.imshow(positive_example)
ax3.imshow(negative_example)

# we add an extra dimension (batch_size dimension) in the first axis by using expand dims.
anchor_tensor = np.expand_dims(example_prediction[:, 0][0], axis=0)
positive_tensor = np.expand_dims(example_prediction[:, 1][0], axis=0)
negative_tensor = np.expand_dims(example_prediction[:, 2][0], axis=0)

anchor_embedding, positive_embedding = (
    embedding(anchor_tensor),
    embedding(positive_tensor),
)
positive_similarity = CosineDistance()(anchor_embedding, positive_embedding)
print("Similarity between similar images:", positive_similarity)

anchor_embedding, negative_embedding = (
    embedding(anchor_tensor),
    embedding(negative_tensor),
)
negative_similarity = CosineDistance()(anchor_embedding, negative_embedding)
print("Similarity between dissimilar images:", negative_similarity)
```

<div class="k-default-codeblock">
```
Similarity between similar images: tf.Tensor([0.9115795], shape=(1,), dtype=float32)
Similarity between dissimilar images: tf.Tensor([0.89747494], shape=(1,), dtype=float32)

```
</div>
    
![png](/img/examples/vision/siamesenetwork/siamesenetwork_37_1.png)
    

