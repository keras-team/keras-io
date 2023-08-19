# Highly accurate boundaries segmentation using BASNet

**Author:** [Hamid Ali](https://github.com/hamidriasat)<br>
**Date created:** 2023/05/30<br>
**Last modified:** 2023/07/13<br>
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
!wget http://saliencydetection.net/duts/download/DUTS-TE.zip
!unzip -q DUTS-TE.zip
```

<div class="k-default-codeblock">
```
--2023-08-06 19:07:37--  http://saliencydetection.net/duts/download/DUTS-TE.zip
Resolving saliencydetection.net (saliencydetection.net)... 36.55.239.177
Connecting to saliencydetection.net (saliencydetection.net)|36.55.239.177|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 139799089 (133M) [application/zip]
Saving to: ‘DUTS-TE.zip’
```
</div>
    
<div class="k-default-codeblock">
```
DUTS-TE.zip         100%[===================>] 133.32M  1.76MB/s    in 77s     
```
</div>
    
<div class="k-default-codeblock">
```
2023-08-06 19:08:55 (1.73 MB/s) - ‘DUTS-TE.zip’ saved [139799089/139799089]
```
</div>
    



```python
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import keras_cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
```

<div class="k-default-codeblock">
```
Using TensorFlow backend

```
</div>
---
## Define Hyperparameters


```python
IMAGE_SIZE = 288
BATCH_SIZE = 4
OUT_CLASSES = 1
TRAIN_SPLIT_RATIO = 0.90
DATA_DIR = "./DUTS-TE/"
```

---
## Create TensorFlow Dataset

We will use `load_paths()` to load and split 140 paths into train and validation set, and
`load_dataset()` to convert paths into `tf.data.Dataset` object.


```python

def load_paths(path, split_ratio):
    images = sorted(glob(os.path.join(path, "DUTS-TE-Image/*")))[:140]
    masks = sorted(glob(os.path.join(path, "DUTS-TE-Mask/*")))[:140]
    len_ = int(len(images) * split_ratio)
    return (images[:len_], masks[:len_]), (images[len_:], masks[len_:])


def read_image(path, size, mode):
    x = keras.utils.load_img(path, target_size=size, color_mode=mode)
    x = keras.utils.img_to_array(x)
    x = (x / 255.0).astype(np.float32)
    return x


def preprocess(x_batch, y_batch, img_size, out_classes):
    def f(_x, _y):
        _x, _y = _x.decode(), _y.decode()
        _x = read_image(_x, (img_size, img_size), mode="rgb")  # image
        _y = read_image(_y, (img_size, img_size), mode="grayscale")  # mask
        return _x, _y

    images, masks = tf.numpy_function(f, [x_batch, y_batch], [tf.float32, tf.float32])
    images.set_shape([img_size, img_size, 3])
    masks.set_shape([img_size, img_size, out_classes])
    return images, masks


def load_dataset(image_paths, mask_paths, img_size, out_classes, batch, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.cache().shuffle(buffer_size=1000)
    dataset = dataset.map(
        lambda x, y: preprocess(x, y, img_size, out_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_paths, val_paths = load_paths(DATA_DIR, TRAIN_SPLIT_RATIO)

train_dataset = load_dataset(
    train_paths[0], train_paths[1], IMAGE_SIZE, OUT_CLASSES, BATCH_SIZE, shuffle=True
)
val_dataset = load_dataset(
    val_paths[0], val_paths[1], IMAGE_SIZE, OUT_CLASSES, BATCH_SIZE, shuffle=False
)

print(f"Train Dataset: {train_dataset}")
print(f"Validation Dataset: {val_dataset}")
```

<div class="k-default-codeblock">
```
Train Dataset: <_PrefetchDataset element_spec=(TensorSpec(shape=(None, 288, 288, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 288, 288, 1), dtype=tf.float32, name=None))>
Validation Dataset: <_PrefetchDataset element_spec=(TensorSpec(shape=(None, 288, 288, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 288, 288, 1), dtype=tf.float32, name=None))>

```
</div>
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


for image, mask in val_dataset.take(1):
    display([image[0], mask[0]])
```


    
![png](/img/examples/vision/basnet_segmentation/basnet_segmentation_10_0.png)
    


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


def get_resnet_block(_resnet, block_num):
    """Extract and return ResNet-34 block."""
    resnet_layers = [3, 4, 6, 3]  # ResNet-34 layer sizes at different block.
    return keras.models.Model(
        inputs=_resnet.get_layer(f"v2_stack_{block_num}_block1_1_conv").input,
        outputs=_resnet.get_layer(
            f"v2_stack_{block_num}_block{resnet_layers[block_num]}_add"
        ).output,
        name=f"resnet34_block{block_num + 1}",
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

    resnet = keras_cv.models.ResNet34Backbone(
        include_rescaling=False,
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
            shape = keras.backend.int_shape(x)
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

    return keras.models.Model(inputs=[x_input], outputs=decoder_blocks)

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
        shape = keras.backend.int_shape(x)
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

def basnet(input_shape, out_classes):
    """BASNet, it's a combination of two modules
    Prediction Module and Residual Refinement Module(RRM)."""

    # Prediction model.
    predict_model = basnet_predict(input_shape, out_classes)
    # Refinement model.
    refine_model = basnet_rrm(predict_model, out_classes)

    output = [refine_model.output]  # Combine outputs.
    output.extend(predict_model.output)

    output = [layers.Activation("sigmoid")(_) for _ in output]  # Activations.

    return keras.models.Model(inputs=[predict_model.input], outputs=output)

```

---
## Hybrid Loss

Another important feature of BASNet is its hybrid loss function, which is a combination of
binary cross entropy, structural similarity and intersection-over-union losses, which guide
the network to learn three-level (i.e., pixel, patch and map level) hierarchy representations.


```python

class BasnetLoss(keras.losses.Loss):
    """BASNet hybrid loss."""

    def __init__(self, **kwargs):
        super().__init__(name="basnet_loss", **kwargs)
        self.smooth = 1.0e-9

        # Binary Cross Entropy loss.
        self.cross_entropy_loss = keras.losses.BinaryCrossentropy()
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        #  Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = backend.sum(backend.abs(y_true * y_pred), axis=[1, 2, 3])
        union = backend.sum(y_true, [1, 2, 3]) + backend.sum(y_pred, [1, 2, 3])
        union = union - intersection
        return backend.mean(
            (intersection + self.smooth) / (union + self.smooth), axis=0
        )

    def call(self, y_true, y_pred):
        cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred)

        ssim_value = self.ssim_value(y_true, y_pred, max_val=1)
        ssim_loss = backend.mean(1 - ssim_value + self.smooth, axis=0)

        iou_value = self.iou_value(y_true, y_pred)
        iou_loss = 1 - iou_value

        # Add all three losses.
        return cross_entropy_loss + ssim_loss + iou_loss


basnet_model = basnet(
    input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], out_classes=OUT_CLASSES
)  # Create model.
basnet_model.summary()  # Show model summary.

optimizer = keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
# Compile model.
basnet_model.compile(
    loss=BasnetLoss(),
    optimizer=optimizer,
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
)
```

<div class="k-default-codeblock">
```
Model: "model_2"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 288, 288, 3)]        0         []                            
                                                                                                  
 conv2d (Conv2D)             (None, 288, 288, 64)         1792      ['input_1[0][0]']             
                                                                                                  
 resnet34_block1 (Functiona  (None, None, None, 64)       222720    ['conv2d[0][0]']              
 l)                                                                                               
                                                                                                  
 activation (Activation)     (None, 288, 288, 64)         0         ['resnet34_block1[0][0]']     
                                                                                                  
 resnet34_block2 (Functiona  (None, None, None, 128)      1118720   ['activation[0][0]']          
 l)                                                                                               
                                                                                                  
 activation_1 (Activation)   (None, 144, 144, 128)        0         ['resnet34_block2[0][0]']     
                                                                                                  
 resnet34_block3 (Functiona  (None, None, None, 256)      6829056   ['activation_1[0][0]']        
 l)                                                                                               
                                                                                                  
 activation_2 (Activation)   (None, 72, 72, 256)          0         ['resnet34_block3[0][0]']     
                                                                                                  
 resnet34_block4 (Functiona  (None, None, None, 512)      1312153   ['activation_2[0][0]']        
 l)                                                       6                                       
                                                                                                  
 activation_3 (Activation)   (None, 36, 36, 512)          0         ['resnet34_block4[0][0]']     
                                                                                                  
 max_pooling2d (MaxPooling2  (None, 18, 18, 512)          0         ['activation_3[0][0]']        
 D)                                                                                               
                                                                                                  
 conv2d_1 (Conv2D)           (None, 18, 18, 512)          2359296   ['max_pooling2d[0][0]']       
                                                                                                  
 batch_normalization (Batch  (None, 18, 18, 512)          2048      ['conv2d_1[0][0]']            
 Normalization)                                                                                   
                                                                                                  
 activation_4 (Activation)   (None, 18, 18, 512)          0         ['batch_normalization[0][0]'] 
                                                                                                  
 conv2d_2 (Conv2D)           (None, 18, 18, 512)          2359296   ['activation_4[0][0]']        
                                                                                                  
 batch_normalization_1 (Bat  (None, 18, 18, 512)          2048      ['conv2d_2[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 add (Add)                   (None, 18, 18, 512)          0         ['batch_normalization_1[0][0]'
                                                                    , 'max_pooling2d[0][0]']      
                                                                                                  
 activation_5 (Activation)   (None, 18, 18, 512)          0         ['add[0][0]']                 
                                                                                                  
 conv2d_3 (Conv2D)           (None, 18, 18, 512)          2359296   ['activation_5[0][0]']        
                                                                                                  
 batch_normalization_2 (Bat  (None, 18, 18, 512)          2048      ['conv2d_3[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 activation_6 (Activation)   (None, 18, 18, 512)          0         ['batch_normalization_2[0][0]'
                                                                    ]                             
                                                                                                  
 conv2d_4 (Conv2D)           (None, 18, 18, 512)          2359296   ['activation_6[0][0]']        
                                                                                                  
 batch_normalization_3 (Bat  (None, 18, 18, 512)          2048      ['conv2d_4[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 add_1 (Add)                 (None, 18, 18, 512)          0         ['batch_normalization_3[0][0]'
                                                                    , 'activation_5[0][0]']       
                                                                                                  
 activation_7 (Activation)   (None, 18, 18, 512)          0         ['add_1[0][0]']               
                                                                                                  
 conv2d_5 (Conv2D)           (None, 18, 18, 512)          2359296   ['activation_7[0][0]']        
                                                                                                  
 batch_normalization_4 (Bat  (None, 18, 18, 512)          2048      ['conv2d_5[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 activation_8 (Activation)   (None, 18, 18, 512)          0         ['batch_normalization_4[0][0]'
                                                                    ]                             
                                                                                                  
 conv2d_6 (Conv2D)           (None, 18, 18, 512)          2359296   ['activation_8[0][0]']        
                                                                                                  
 batch_normalization_5 (Bat  (None, 18, 18, 512)          2048      ['conv2d_6[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 add_2 (Add)                 (None, 18, 18, 512)          0         ['batch_normalization_5[0][0]'
                                                                    , 'activation_7[0][0]']       
                                                                                                  
 activation_9 (Activation)   (None, 18, 18, 512)          0         ['add_2[0][0]']               
                                                                                                  
 max_pooling2d_1 (MaxPoolin  (None, 9, 9, 512)            0         ['activation_9[0][0]']        
 g2D)                                                                                             
                                                                                                  
 conv2d_7 (Conv2D)           (None, 9, 9, 512)            2359296   ['max_pooling2d_1[0][0]']     
                                                                                                  
 batch_normalization_6 (Bat  (None, 9, 9, 512)            2048      ['conv2d_7[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 activation_10 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_6[0][0]'
                                                                    ]                             
                                                                                                  
 conv2d_8 (Conv2D)           (None, 9, 9, 512)            2359296   ['activation_10[0][0]']       
                                                                                                  
 batch_normalization_7 (Bat  (None, 9, 9, 512)            2048      ['conv2d_8[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 add_3 (Add)                 (None, 9, 9, 512)            0         ['batch_normalization_7[0][0]'
                                                                    , 'max_pooling2d_1[0][0]']    
                                                                                                  
 activation_11 (Activation)  (None, 9, 9, 512)            0         ['add_3[0][0]']               
                                                                                                  
 conv2d_9 (Conv2D)           (None, 9, 9, 512)            2359296   ['activation_11[0][0]']       
                                                                                                  
 batch_normalization_8 (Bat  (None, 9, 9, 512)            2048      ['conv2d_9[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 activation_12 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_8[0][0]'
                                                                    ]                             
                                                                                                  
 conv2d_10 (Conv2D)          (None, 9, 9, 512)            2359296   ['activation_12[0][0]']       
                                                                                                  
 batch_normalization_9 (Bat  (None, 9, 9, 512)            2048      ['conv2d_10[0][0]']           
 chNormalization)                                                                                 
                                                                                                  
 add_4 (Add)                 (None, 9, 9, 512)            0         ['batch_normalization_9[0][0]'
                                                                    , 'activation_11[0][0]']      
                                                                                                  
 activation_13 (Activation)  (None, 9, 9, 512)            0         ['add_4[0][0]']               
                                                                                                  
 conv2d_11 (Conv2D)          (None, 9, 9, 512)            2359296   ['activation_13[0][0]']       
                                                                                                  
 batch_normalization_10 (Ba  (None, 9, 9, 512)            2048      ['conv2d_11[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_14 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_10[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_12 (Conv2D)          (None, 9, 9, 512)            2359296   ['activation_14[0][0]']       
                                                                                                  
 batch_normalization_11 (Ba  (None, 9, 9, 512)            2048      ['conv2d_12[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 add_5 (Add)                 (None, 9, 9, 512)            0         ['batch_normalization_11[0][0]
                                                                    ',                            
                                                                     'activation_13[0][0]']       
                                                                                                  
 activation_15 (Activation)  (None, 9, 9, 512)            0         ['add_5[0][0]']               
                                                                                                  
 conv2d_13 (Conv2D)          (None, 9, 9, 512)            2359808   ['activation_15[0][0]']       
                                                                                                  
 batch_normalization_12 (Ba  (None, 9, 9, 512)            2048      ['conv2d_13[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_16 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_12[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_14 (Conv2D)          (None, 9, 9, 512)            2359808   ['activation_16[0][0]']       
                                                                                                  
 batch_normalization_13 (Ba  (None, 9, 9, 512)            2048      ['conv2d_14[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_17 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_13[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_15 (Conv2D)          (None, 9, 9, 512)            2359808   ['activation_17[0][0]']       
                                                                                                  
 batch_normalization_14 (Ba  (None, 9, 9, 512)            2048      ['conv2d_15[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_18 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_14[0][0]
                                                                    ']                            
                                                                                                  
 concatenate (Concatenate)   (None, 9, 9, 1024)           0         ['activation_15[0][0]',       
                                                                     'activation_18[0][0]']       
                                                                                                  
 conv2d_16 (Conv2D)          (None, 9, 9, 512)            4719104   ['concatenate[0][0]']         
                                                                                                  
 batch_normalization_15 (Ba  (None, 9, 9, 512)            2048      ['conv2d_16[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_19 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_15[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_17 (Conv2D)          (None, 9, 9, 512)            2359808   ['activation_19[0][0]']       
                                                                                                  
 batch_normalization_16 (Ba  (None, 9, 9, 512)            2048      ['conv2d_17[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_20 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_16[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_18 (Conv2D)          (None, 9, 9, 512)            2359808   ['activation_20[0][0]']       
                                                                                                  
 batch_normalization_17 (Ba  (None, 9, 9, 512)            2048      ['conv2d_18[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_21 (Activation)  (None, 9, 9, 512)            0         ['batch_normalization_17[0][0]
                                                                    ']                            
                                                                                                  
 resizing (Resizing)         (None, 18, 18, 512)          0         ['activation_21[0][0]']       
                                                                                                  
 concatenate_1 (Concatenate  (None, 18, 18, 1024)         0         ['activation_9[0][0]',        
 )                                                                   'resizing[0][0]']            
                                                                                                  
 conv2d_19 (Conv2D)          (None, 18, 18, 512)          4719104   ['concatenate_1[0][0]']       
                                                                                                  
 batch_normalization_18 (Ba  (None, 18, 18, 512)          2048      ['conv2d_19[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_22 (Activation)  (None, 18, 18, 512)          0         ['batch_normalization_18[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_20 (Conv2D)          (None, 18, 18, 512)          2359808   ['activation_22[0][0]']       
                                                                                                  
 batch_normalization_19 (Ba  (None, 18, 18, 512)          2048      ['conv2d_20[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_23 (Activation)  (None, 18, 18, 512)          0         ['batch_normalization_19[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_21 (Conv2D)          (None, 18, 18, 512)          2359808   ['activation_23[0][0]']       
                                                                                                  
 batch_normalization_20 (Ba  (None, 18, 18, 512)          2048      ['conv2d_21[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_24 (Activation)  (None, 18, 18, 512)          0         ['batch_normalization_20[0][0]
                                                                    ']                            
                                                                                                  
 resizing_1 (Resizing)       (None, 36, 36, 512)          0         ['activation_24[0][0]']       
                                                                                                  
 concatenate_2 (Concatenate  (None, 36, 36, 1024)         0         ['resnet34_block4[0][0]',     
 )                                                                   'resizing_1[0][0]']          
                                                                                                  
 conv2d_22 (Conv2D)          (None, 36, 36, 512)          4719104   ['concatenate_2[0][0]']       
                                                                                                  
 batch_normalization_21 (Ba  (None, 36, 36, 512)          2048      ['conv2d_22[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_25 (Activation)  (None, 36, 36, 512)          0         ['batch_normalization_21[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_23 (Conv2D)          (None, 36, 36, 512)          2359808   ['activation_25[0][0]']       
                                                                                                  
 batch_normalization_22 (Ba  (None, 36, 36, 512)          2048      ['conv2d_23[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_26 (Activation)  (None, 36, 36, 512)          0         ['batch_normalization_22[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_24 (Conv2D)          (None, 36, 36, 512)          2359808   ['activation_26[0][0]']       
                                                                                                  
 batch_normalization_23 (Ba  (None, 36, 36, 512)          2048      ['conv2d_24[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_27 (Activation)  (None, 36, 36, 512)          0         ['batch_normalization_23[0][0]
                                                                    ']                            
                                                                                                  
 resizing_2 (Resizing)       (None, 72, 72, 512)          0         ['activation_27[0][0]']       
                                                                                                  
 concatenate_3 (Concatenate  (None, 72, 72, 768)          0         ['resnet34_block3[0][0]',     
 )                                                                   'resizing_2[0][0]']          
                                                                                                  
 conv2d_25 (Conv2D)          (None, 72, 72, 512)          3539456   ['concatenate_3[0][0]']       
                                                                                                  
 batch_normalization_24 (Ba  (None, 72, 72, 512)          2048      ['conv2d_25[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_28 (Activation)  (None, 72, 72, 512)          0         ['batch_normalization_24[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_26 (Conv2D)          (None, 72, 72, 512)          2359808   ['activation_28[0][0]']       
                                                                                                  
 batch_normalization_25 (Ba  (None, 72, 72, 512)          2048      ['conv2d_26[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_29 (Activation)  (None, 72, 72, 512)          0         ['batch_normalization_25[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_27 (Conv2D)          (None, 72, 72, 512)          2359808   ['activation_29[0][0]']       
                                                                                                  
 batch_normalization_26 (Ba  (None, 72, 72, 512)          2048      ['conv2d_27[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_30 (Activation)  (None, 72, 72, 512)          0         ['batch_normalization_26[0][0]
                                                                    ']                            
                                                                                                  
 resizing_3 (Resizing)       (None, 144, 144, 512)        0         ['activation_30[0][0]']       
                                                                                                  
 concatenate_4 (Concatenate  (None, 144, 144, 640)        0         ['resnet34_block2[0][0]',     
 )                                                                   'resizing_3[0][0]']          
                                                                                                  
 conv2d_28 (Conv2D)          (None, 144, 144, 512)        2949632   ['concatenate_4[0][0]']       
                                                                                                  
 batch_normalization_27 (Ba  (None, 144, 144, 512)        2048      ['conv2d_28[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_31 (Activation)  (None, 144, 144, 512)        0         ['batch_normalization_27[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_29 (Conv2D)          (None, 144, 144, 512)        2359808   ['activation_31[0][0]']       
                                                                                                  
 batch_normalization_28 (Ba  (None, 144, 144, 512)        2048      ['conv2d_29[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_32 (Activation)  (None, 144, 144, 512)        0         ['batch_normalization_28[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_30 (Conv2D)          (None, 144, 144, 512)        2359808   ['activation_32[0][0]']       
                                                                                                  
 batch_normalization_29 (Ba  (None, 144, 144, 512)        2048      ['conv2d_30[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_33 (Activation)  (None, 144, 144, 512)        0         ['batch_normalization_29[0][0]
                                                                    ']                            
                                                                                                  
 resizing_4 (Resizing)       (None, 288, 288, 512)        0         ['activation_33[0][0]']       
                                                                                                  
 concatenate_5 (Concatenate  (None, 288, 288, 576)        0         ['resnet34_block1[0][0]',     
 )                                                                   'resizing_4[0][0]']          
                                                                                                  
 conv2d_31 (Conv2D)          (None, 288, 288, 512)        2654720   ['concatenate_5[0][0]']       
                                                                                                  
 batch_normalization_30 (Ba  (None, 288, 288, 512)        2048      ['conv2d_31[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_34 (Activation)  (None, 288, 288, 512)        0         ['batch_normalization_30[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_32 (Conv2D)          (None, 288, 288, 512)        2359808   ['activation_34[0][0]']       
                                                                                                  
 batch_normalization_31 (Ba  (None, 288, 288, 512)        2048      ['conv2d_32[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_35 (Activation)  (None, 288, 288, 512)        0         ['batch_normalization_31[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_33 (Conv2D)          (None, 288, 288, 512)        2359808   ['activation_35[0][0]']       
                                                                                                  
 batch_normalization_32 (Ba  (None, 288, 288, 512)        2048      ['conv2d_33[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_36 (Activation)  (None, 288, 288, 512)        0         ['batch_normalization_32[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_34 (Conv2D)          (None, 288, 288, 1)          4609      ['activation_36[0][0]']       
                                                                                                  
 resizing_5 (Resizing)       (None, 288, 288, 1)          0         ['conv2d_34[0][0]']           
                                                                                                  
 conv2d_41 (Conv2D)          (None, 288, 288, 64)         640       ['resizing_5[0][0]']          
                                                                                                  
 conv2d_42 (Conv2D)          (None, 288, 288, 64)         36928     ['conv2d_41[0][0]']           
                                                                                                  
 batch_normalization_33 (Ba  (None, 288, 288, 64)         256       ['conv2d_42[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_37 (Activation)  (None, 288, 288, 64)         0         ['batch_normalization_33[0][0]
                                                                    ']                            
                                                                                                  
 max_pooling2d_2 (MaxPoolin  (None, 144, 144, 64)         0         ['activation_37[0][0]']       
 g2D)                                                                                             
                                                                                                  
 conv2d_43 (Conv2D)          (None, 144, 144, 64)         36928     ['max_pooling2d_2[0][0]']     
                                                                                                  
 batch_normalization_34 (Ba  (None, 144, 144, 64)         256       ['conv2d_43[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_38 (Activation)  (None, 144, 144, 64)         0         ['batch_normalization_34[0][0]
                                                                    ']                            
                                                                                                  
 max_pooling2d_3 (MaxPoolin  (None, 72, 72, 64)           0         ['activation_38[0][0]']       
 g2D)                                                                                             
                                                                                                  
 conv2d_44 (Conv2D)          (None, 72, 72, 64)           36928     ['max_pooling2d_3[0][0]']     
                                                                                                  
 batch_normalization_35 (Ba  (None, 72, 72, 64)           256       ['conv2d_44[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_39 (Activation)  (None, 72, 72, 64)           0         ['batch_normalization_35[0][0]
                                                                    ']                            
                                                                                                  
 max_pooling2d_4 (MaxPoolin  (None, 36, 36, 64)           0         ['activation_39[0][0]']       
 g2D)                                                                                             
                                                                                                  
 conv2d_45 (Conv2D)          (None, 36, 36, 64)           36928     ['max_pooling2d_4[0][0]']     
                                                                                                  
 batch_normalization_36 (Ba  (None, 36, 36, 64)           256       ['conv2d_45[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_40 (Activation)  (None, 36, 36, 64)           0         ['batch_normalization_36[0][0]
                                                                    ']                            
                                                                                                  
 max_pooling2d_5 (MaxPoolin  (None, 18, 18, 64)           0         ['activation_40[0][0]']       
 g2D)                                                                                             
                                                                                                  
 conv2d_46 (Conv2D)          (None, 18, 18, 64)           36928     ['max_pooling2d_5[0][0]']     
                                                                                                  
 batch_normalization_37 (Ba  (None, 18, 18, 64)           256       ['conv2d_46[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_41 (Activation)  (None, 18, 18, 64)           0         ['batch_normalization_37[0][0]
                                                                    ']                            
                                                                                                  
 resizing_12 (Resizing)      (None, 36, 36, 64)           0         ['activation_41[0][0]']       
                                                                                                  
 concatenate_6 (Concatenate  (None, 36, 36, 128)          0         ['activation_40[0][0]',       
 )                                                                   'resizing_12[0][0]']         
                                                                                                  
 conv2d_47 (Conv2D)          (None, 36, 36, 64)           73792     ['concatenate_6[0][0]']       
                                                                                                  
 batch_normalization_38 (Ba  (None, 36, 36, 64)           256       ['conv2d_47[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_42 (Activation)  (None, 36, 36, 64)           0         ['batch_normalization_38[0][0]
                                                                    ']                            
                                                                                                  
 resizing_13 (Resizing)      (None, 72, 72, 64)           0         ['activation_42[0][0]']       
                                                                                                  
 concatenate_7 (Concatenate  (None, 72, 72, 128)          0         ['activation_39[0][0]',       
 )                                                                   'resizing_13[0][0]']         
                                                                                                  
 conv2d_48 (Conv2D)          (None, 72, 72, 64)           73792     ['concatenate_7[0][0]']       
                                                                                                  
 batch_normalization_39 (Ba  (None, 72, 72, 64)           256       ['conv2d_48[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_43 (Activation)  (None, 72, 72, 64)           0         ['batch_normalization_39[0][0]
                                                                    ']                            
                                                                                                  
 resizing_14 (Resizing)      (None, 144, 144, 64)         0         ['activation_43[0][0]']       
                                                                                                  
 concatenate_8 (Concatenate  (None, 144, 144, 128)        0         ['activation_38[0][0]',       
 )                                                                   'resizing_14[0][0]']         
                                                                                                  
 conv2d_49 (Conv2D)          (None, 144, 144, 64)         73792     ['concatenate_8[0][0]']       
                                                                                                  
 batch_normalization_40 (Ba  (None, 144, 144, 64)         256       ['conv2d_49[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_44 (Activation)  (None, 144, 144, 64)         0         ['batch_normalization_40[0][0]
                                                                    ']                            
                                                                                                  
 resizing_15 (Resizing)      (None, 288, 288, 64)         0         ['activation_44[0][0]']       
                                                                                                  
 concatenate_9 (Concatenate  (None, 288, 288, 128)        0         ['activation_37[0][0]',       
 )                                                                   'resizing_15[0][0]']         
                                                                                                  
 conv2d_50 (Conv2D)          (None, 288, 288, 64)         73792     ['concatenate_9[0][0]']       
                                                                                                  
 batch_normalization_41 (Ba  (None, 288, 288, 64)         256       ['conv2d_50[0][0]']           
 tchNormalization)                                                                                
                                                                                                  
 activation_45 (Activation)  (None, 288, 288, 64)         0         ['batch_normalization_41[0][0]
                                                                    ']                            
                                                                                                  
 conv2d_51 (Conv2D)          (None, 288, 288, 1)          577       ['activation_45[0][0]']       
                                                                                                  
 conv2d_35 (Conv2D)          (None, 144, 144, 1)          4609      ['activation_33[0][0]']       
                                                                                                  
 conv2d_36 (Conv2D)          (None, 72, 72, 1)            4609      ['activation_30[0][0]']       
                                                                                                  
 conv2d_37 (Conv2D)          (None, 36, 36, 1)            4609      ['activation_27[0][0]']       
                                                                                                  
 conv2d_38 (Conv2D)          (None, 18, 18, 1)            4609      ['activation_24[0][0]']       
                                                                                                  
 conv2d_39 (Conv2D)          (None, 9, 9, 1)              4609      ['activation_21[0][0]']       
                                                                                                  
 conv2d_40 (Conv2D)          (None, 9, 9, 1)              4609      ['activation_18[0][0]']       
                                                                                                  
 add_6 (Add)                 (None, 288, 288, 1)          0         ['resizing_5[0][0]',          
                                                                     'conv2d_51[0][0]']           
                                                                                                  
 resizing_6 (Resizing)       (None, 288, 288, 1)          0         ['conv2d_35[0][0]']           
                                                                                                  
 resizing_7 (Resizing)       (None, 288, 288, 1)          0         ['conv2d_36[0][0]']           
                                                                                                  
 resizing_8 (Resizing)       (None, 288, 288, 1)          0         ['conv2d_37[0][0]']           
                                                                                                  
 resizing_9 (Resizing)       (None, 288, 288, 1)          0         ['conv2d_38[0][0]']           
                                                                                                  
 resizing_10 (Resizing)      (None, 288, 288, 1)          0         ['conv2d_39[0][0]']           
                                                                                                  
 resizing_11 (Resizing)      (None, 288, 288, 1)          0         ['conv2d_40[0][0]']           
                                                                                                  
 activation_46 (Activation)  (None, 288, 288, 1)          0         ['add_6[0][0]']               
                                                                                                  
 activation_47 (Activation)  (None, 288, 288, 1)          0         ['resizing_5[0][0]']          
                                                                                                  
 activation_48 (Activation)  (None, 288, 288, 1)          0         ['resizing_6[0][0]']          
                                                                                                  
 activation_49 (Activation)  (None, 288, 288, 1)          0         ['resizing_7[0][0]']          
                                                                                                  
 activation_50 (Activation)  (None, 288, 288, 1)          0         ['resizing_8[0][0]']          
                                                                                                  
 activation_51 (Activation)  (None, 288, 288, 1)          0         ['resizing_9[0][0]']          
                                                                                                  
 activation_52 (Activation)  (None, 288, 288, 1)          0         ['resizing_10[0][0]']         
                                                                                                  
 activation_53 (Activation)  (None, 288, 288, 1)          0         ['resizing_11[0][0]']         
                                                                                                  
==================================================================================================
Total params: 108886792 (415.37 MB)
Trainable params: 108834952 (415.17 MB)
Non-trainable params: 51840 (202.50 KB)
__________________________________________________________________________________________________

```
</div>
### Train the Model


```python
basnet_model.fit(train_dataset, validation_data=val_dataset, epochs=1)
```

<div class="k-default-codeblock">
```
32/32 [==============================] - 153s 2s/step - loss: 16.3507 - activation_46_loss: 2.1445 - activation_47_loss: 2.1512 - activation_48_loss: 2.0621 - activation_49_loss: 2.0755 - activation_50_loss: 2.1406 - activation_51_loss: 1.9035 - activation_52_loss: 1.8702 - activation_53_loss: 2.0031 - activation_46_mae: 0.2972 - activation_47_mae: 0.3126 - activation_48_mae: 0.2793 - activation_49_mae: 0.2887 - activation_50_mae: 0.3280 - activation_51_mae: 0.2548 - activation_52_mae: 0.2330 - activation_53_mae: 0.2564 - val_loss: 18.4498 - val_activation_46_loss: 2.3113 - val_activation_47_loss: 2.3143 - val_activation_48_loss: 2.3356 - val_activation_49_loss: 2.3093 - val_activation_50_loss: 2.3187 - val_activation_51_loss: 2.3943 - val_activation_52_loss: 2.2712 - val_activation_53_loss: 2.1952 - val_activation_46_mae: 0.2770 - val_activation_47_mae: 0.2681 - val_activation_48_mae: 0.2424 - val_activation_49_mae: 0.2691 - val_activation_50_mae: 0.2765 - val_activation_51_mae: 0.1907 - val_activation_52_mae: 0.1885 - val_activation_53_mae: 0.2938

<keras.src.callbacks.History at 0x79b024bd83a0>

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
!!gdown 1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg
```




```python

def normalize_output(prediction):
    max_value = np.max(prediction)
    min_value = np.min(prediction)
    return (prediction - min_value) / (max_value - min_value)


# Load weights.
basnet_model.load_weights("./basnet_weights.h5")
```
<div class="k-default-codeblock">
```
['Downloading...',
 'From: https://drive.google.com/uc?id=1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg',
 'To: /content/keras-io/scripts/tmp_3792671/basnet_weights.h5',
 '',
 '  0% 0.00/436M [00:00<?, ?B/s]',
 '  1% 4.72M/436M [00:00<00:25, 16.7MB/s]',
 '  4% 17.3M/436M [00:00<00:13, 31.5MB/s]',
 '  7% 30.9M/436M [00:00<00:07, 54.5MB/s]',
 '  9% 38.8M/436M [00:00<00:08, 48.2MB/s]',
 ' 12% 50.9M/436M [00:01<00:08, 45.2MB/s]',
 ' 15% 65.0M/436M [00:01<00:05, 62.2MB/s]',
 ' 17% 73.4M/436M [00:01<00:07, 50.6MB/s]',
 ' 19% 84.4M/436M [00:01<00:07, 48.3MB/s]',
 ' 23% 100M/436M [00:01<00:05, 66.7MB/s] ',
 ' 25% 110M/436M [00:02<00:05, 59.1MB/s]',
 ' 27% 118M/436M [00:02<00:06, 48.4MB/s]',
 ' 31% 135M/436M [00:02<00:05, 52.7MB/s]',
 ' 35% 152M/436M [00:02<00:04, 70.2MB/s]',
 ' 37% 161M/436M [00:03<00:04, 56.9MB/s]',
 ' 42% 185M/436M [00:03<00:04, 56.2MB/s]',
 ' 48% 210M/436M [00:03<00:03, 65.0MB/s]',
 ' 53% 231M/436M [00:03<00:02, 83.6MB/s]',
 ' 56% 243M/436M [00:04<00:02, 71.4MB/s]',
 ' 60% 261M/436M [00:04<00:02, 73.9MB/s]',
 ' 62% 272M/436M [00:04<00:02, 80.1MB/s]',
 ' 66% 286M/436M [00:04<00:01, 79.3MB/s]',
 ' 68% 295M/436M [00:04<00:01, 81.2MB/s]',
 ' 71% 308M/436M [00:04<00:01, 91.3MB/s]',
 ' 73% 319M/436M [00:04<00:01, 88.2MB/s]',
 ' 75% 329M/436M [00:05<00:01, 83.5MB/s]',
 ' 78% 339M/436M [00:05<00:01, 87.6MB/s]',
 ' 81% 353M/436M [00:05<00:00, 90.4MB/s]',
 ' 83% 362M/436M [00:05<00:00, 87.0MB/s]',
 ' 87% 378M/436M [00:05<00:00, 104MB/s] ',
 ' 89% 389M/436M [00:05<00:00, 101MB/s]',
 ' 93% 405M/436M [00:05<00:00, 115MB/s]',
 ' 96% 417M/436M [00:05<00:00, 110MB/s]',
 ' 98% 428M/436M [00:06<00:00, 91.4MB/s]',
 '100% 436M/436M [00:06<00:00, 71.3MB/s]']

```
</div>
### Make Predictions


```python
for image, mask in val_dataset.take(1):
    pred_mask = basnet_model.predict(image)
    display([image[0], mask[0], normalize_output(pred_mask[0][0])])
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 2s 2s/step

```
</div>
    
![png](/img/examples/vision/basnet_segmentation/basnet_segmentation_29_1.png)
    

