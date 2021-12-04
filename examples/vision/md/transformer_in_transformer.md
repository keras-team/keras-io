# Image classification with TNT(Transformer in Transformer)

**Author:** [ZhiYong Chang](https://github.com/czy00000)<br>
**Date created:** 2021/10/25<br>
**Last modified:** 2021/11/29<br>
**Description:** Implementing the Transformer in Transformer (TNT) model for image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/transformer_in_transformer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/transformer_in_transformer.py)



---
## Introduction
This example implements the [TNT](https://arxiv.org/abs/2103.00112)
model for image classification, and demonstrates it's performance on the CIFAR-100
dataset.
To keep training time reasonable, we will train and test a smaller model than is in the
paper(0.66M params vs 23.8M params).
TNT is a novel model for modeling both patch-level and pixel-level
representation. In each TNT block, an ***outer*** transformer block is utilized to process
patch embeddings, and an ***inner***
transformer block extracts local features from pixel embeddings. The pixel-level
feature is projected to the space of patch embedding by a linear transformation layer
and then added into the patch.
This example requires TensorFlow 2.5 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview) package for the
AdamW optimizer.
Tensorflow Addons can be installed using the following command:
```python
pip install -U tensorflow-addons
```

---
## Setup


```python
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from itertools import repeat

```

---
## Prepare the data


```python
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
```

<div class="k-default-codeblock">
```
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 100)
x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 100)

```
</div>
---
## Configure the hyperparameters


```python
weight_decay = 0.0002
learning_rate = 0.001
label_smoothing = 0.1
validation_split = 0.2
batch_size = 128
image_size = (96, 96)  # resize images to this size
patch_size = (8, 8)
num_epochs = 50
outer_block_embedding_dim = 64
inner_block_embedding_dim = 32
num_transformer_layer = 5
outer_block_num_heads = 4
inner_block_num_heads = 2
mlp_ratio = 4
attention_dropout = 0.5
projection_dropout = 0.5
first_stride = 4
```

---
## Use data augmentation


```python

def data_augmentation(inputs):
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = layers.Resizing(image_size[0], image_size[1])(x)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(factor=0.1)(x)
    x = layers.RandomContrast(factor=0.1)(x)
    x = layers.RandomZoom(height_factor=0.2, width_factor=0.2)(x)
    return x

```

---
## Implement the pixel embedding and patch embedding layer


```python

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def pixel_embed(x, image_size=image_size, patch_size=patch_size, in_dim=48, stride=4):
    _, channel, height, width = x.shape
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    inner_patch_size = [math.ceil(ps / stride) for ps in patch_size]
    x = layers.Conv2D(in_dim, kernel_size=7, strides=stride, padding="same")(x)
    # pixel extraction
    x = tf.image.extract_patches(
        images=x,
        sizes=(1, inner_patch_size[0], inner_patch_size[1], 1),
        strides=(1, inner_patch_size[0], inner_patch_size[1], 1),
        rates=(1, 1, 1, 1),
        padding="VALID",
    )
    x = tf.reshape(x, shape=(-1, inner_patch_size[0] * inner_patch_size[1], in_dim))
    x = PatchEncoder(inner_patch_size[0] * inner_patch_size[1], in_dim)(x)
    return x, num_patches, inner_patch_size


def patch_embed(
    pixel_embedding,
    num_patches,
    outer_block_embedding_dim,
    inner_block_embedding_dim,
    num_pixels,
):
    patch_embedding = tf.reshape(
        pixel_embedding, shape=(-1, num_patches, inner_block_embedding_dim * num_pixels)
    )
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = layers.Dense(outer_block_embedding_dim)(patch_embedding)
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = PatchEncoder(num_patches, outer_block_embedding_dim)(
        patch_embedding
    )
    patch_embedding = layers.Dropout(projection_dropout)(patch_embedding)
    return patch_embedding

```

---
## Implement the MLP block


```python

def mlp(x, hidden_dim, output_dim, drop_rate=0.2):
    x = layers.Dense(hidden_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(output_dim)(x)
    x = layers.Dropout(drop_rate)(x)
    return x

```

---
## Implement the TNT block


```python

def transformer_in_transformer_block(
    pixel_embedding,
    patch_embedding,
    out_embedding_dim,
    in_embedding_dim,
    num_pixels,
    out_num_heads,
    in_num_heads,
    mlp_ratio,
    attention_dropout,
    projection_dropout,
):
    # inner transformer block
    residual_in_1 = pixel_embedding
    pixel_embedding = layers.LayerNormalization(epsilon=1e-5)(pixel_embedding)
    pixel_embedding = layers.MultiHeadAttention(
        num_heads=in_num_heads, key_dim=in_embedding_dim, dropout=attention_dropout
    )(pixel_embedding, pixel_embedding)
    pixel_embedding = layers.add([pixel_embedding, residual_in_1])
    residual_in_2 = pixel_embedding
    pixel_embedding = layers.LayerNormalization(epsilon=1e-5)(pixel_embedding)
    pixel_embedding = mlp(
        pixel_embedding, in_embedding_dim * mlp_ratio, in_embedding_dim
    )
    pixel_embedding = layers.add([pixel_embedding, residual_in_2])

    # outer transformer block
    _, num_patches, channel = patch_embedding.shape
    # fuse local and global information
    fused_embedding = tf.reshape(
        pixel_embedding, shape=(-1, num_patches, in_embedding_dim * num_pixels)
    )
    fused_embedding = layers.LayerNormalization(epsilon=1e-5)(fused_embedding)
    fused_embedding = layers.Dense(out_embedding_dim)(fused_embedding)
    patch_embedding = layers.add([patch_embedding, fused_embedding])
    residual_out_1 = patch_embedding
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = layers.MultiHeadAttention(
        num_heads=out_num_heads, key_dim=out_embedding_dim, dropout=attention_dropout
    )(patch_embedding, patch_embedding)
    patch_embedding = layers.add([patch_embedding, residual_out_1])
    residual_out_2 = patch_embedding
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = mlp(
        patch_embedding, out_embedding_dim * mlp_ratio, out_embedding_dim
    )
    patch_embedding = layers.add([patch_embedding, residual_out_2])
    return pixel_embedding, patch_embedding

```

---
## Implement the TNT model
The TNT model consists of multiple TNT blocks.
In the TNT block, there are two transformer blocks where
the outer transformer block models the global relation among patch embeddings,
and the inner one extracts local structure information of pixel embeddings.
The local information is added on the patch
embedding by linearly projecting the pixel embeddings into the space of patch embedding.
Patch-level and pixel-level position embeddings are introduced in order to
retain spatial information. In orginal paper, the authors use the class token for
classification.
We use the `layers.GlobalAvgPool1D` to fuse patch information.


```python

def get_model(
    image_size=image_size,
    patch_size=patch_size,
    outer_block_embedding_dim=outer_block_embedding_dim,
    inner_block_embedding_dim=inner_block_embedding_dim,
    num_transformer_layer=num_transformer_layer,
    outer_block_num_heads=outer_block_num_heads,
    inner_block_num_heads=inner_block_num_heads,
    mlp_ratio=mlp_ratio,
    attention_dropout=attention_dropout,
    projection_dropout=projection_dropout,
    first_stride=first_stride,
):
    inputs = layers.Input(shape=input_shape)
    # Image augment
    x = data_augmentation(inputs)
    # extract pixel embedding
    pixel_embedding, num_patches, inner_patch_size = pixel_embed(
        x, image_size, patch_size, inner_block_embedding_dim, first_stride
    )
    num_pixels = inner_patch_size[0] * inner_patch_size[1]
    # extract patch embedding
    patch_embedding = patch_embed(
        pixel_embedding,
        num_patches,
        outer_block_embedding_dim,
        inner_block_embedding_dim,
        num_pixels,
    )
    # create multiple layers of the TNT block.
    for _ in range(num_transformer_layer):
        pixel_embedding, patch_embedding = transformer_in_transformer_block(
            pixel_embedding,
            patch_embedding,
            outer_block_embedding_dim,
            inner_block_embedding_dim,
            num_pixels,
            outer_block_num_heads,
            inner_block_num_heads,
            mlp_ratio,
            attention_dropout,
            projection_dropout,
        )
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    x = layers.GlobalAvgPool1D()(patch_embedding)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

---
## Train on CIFAR-100


```python
model = get_model()
model.summary()
model.compile(
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_split=validation_split,
)
```

<div class="k-default-codeblock">
```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 32, 32, 3)]  0                                            
__________________________________________________________________________________________________
rescaling (Rescaling)           (None, 32, 32, 3)    0           input_1[0][0]                    
__________________________________________________________________________________________________
resizing (Resizing)             (None, 96, 96, 3)    0           rescaling[0][0]                  
__________________________________________________________________________________________________
random_flip (RandomFlip)        (None, 96, 96, 3)    0           resizing[0][0]                   
__________________________________________________________________________________________________
random_rotation (RandomRotation (None, 96, 96, 3)    0           random_flip[0][0]                
__________________________________________________________________________________________________
random_contrast (RandomContrast (None, 96, 96, 3)    0           random_rotation[0][0]            
__________________________________________________________________________________________________
random_zoom (RandomZoom)        (None, 96, 96, 3)    0           random_contrast[0][0]            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 24, 24, 32)   4736        random_zoom[0][0]                
__________________________________________________________________________________________________
tf.image.extract_patches (TFOpL (None, 12, 12, 128)  0           conv2d[0][0]                     
__________________________________________________________________________________________________
tf.reshape (TFOpLambda)         (None, 4, 32)        0           tf.image.extract_patches[0][0]   
__________________________________________________________________________________________________
patch_encoder (PatchEncoder)    (None, 4, 32)        1184        tf.reshape[0][0]                 
__________________________________________________________________________________________________
layer_normalization_2 (LayerNor (None, 4, 32)        64          patch_encoder[0][0]              
__________________________________________________________________________________________________
multi_head_attention (MultiHead (None, 4, 32)        8416        layer_normalization_2[0][0]      
                                                                 layer_normalization_2[0][0]      
__________________________________________________________________________________________________
add (Add)                       (None, 4, 32)        0           multi_head_attention[0][0]       
                                                                 patch_encoder[0][0]              
__________________________________________________________________________________________________
layer_normalization_3 (LayerNor (None, 4, 32)        64          add[0][0]                        
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 4, 128)       4224        layer_normalization_3[0][0]      
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 4, 128)       0           dense_3[0][0]                    
__________________________________________________________________________________________________
tf.reshape_1 (TFOpLambda)       (None, 144, 128)     0           patch_encoder[0][0]              
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 4, 32)        4128        dropout_1[0][0]                  
__________________________________________________________________________________________________
layer_normalization (LayerNorma (None, 144, 128)     256         tf.reshape_1[0][0]               
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 4, 32)        0           dense_4[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 144, 64)      8256        layer_normalization[0][0]        
__________________________________________________________________________________________________
add_1 (Add)                     (None, 4, 32)        0           dropout_2[0][0]                  
                                                                 add[0][0]                        
__________________________________________________________________________________________________
layer_normalization_1 (LayerNor (None, 144, 64)      128         dense_1[0][0]                    
__________________________________________________________________________________________________
tf.reshape_2 (TFOpLambda)       (None, 144, 128)     0           add_1[0][0]                      
__________________________________________________________________________________________________
patch_encoder_1 (PatchEncoder)  (None, 144, 64)      13376       layer_normalization_1[0][0]      
__________________________________________________________________________________________________
layer_normalization_4 (LayerNor (None, 144, 128)     256         tf.reshape_2[0][0]               
__________________________________________________________________________________________________
layer_normalization_7 (LayerNor (None, 4, 32)        64          add_1[0][0]                      
__________________________________________________________________________________________________
dropout (Dropout)               (None, 144, 64)      0           patch_encoder_1[0][0]            
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 144, 64)      8256        layer_normalization_4[0][0]      
__________________________________________________________________________________________________
multi_head_attention_2 (MultiHe (None, 4, 32)        8416        layer_normalization_7[0][0]      
                                                                 layer_normalization_7[0][0]      
__________________________________________________________________________________________________
add_2 (Add)                     (None, 144, 64)      0           dropout[0][0]                    
                                                                 dense_5[0][0]                    
__________________________________________________________________________________________________
add_5 (Add)                     (None, 4, 32)        0           multi_head_attention_2[0][0]     
                                                                 add_1[0][0]                      
__________________________________________________________________________________________________
layer_normalization_5 (LayerNor (None, 144, 64)      128         add_2[0][0]                      
__________________________________________________________________________________________________
layer_normalization_8 (LayerNor (None, 4, 32)        64          add_5[0][0]                      
__________________________________________________________________________________________________
multi_head_attention_1 (MultiHe (None, 144, 64)      66368       layer_normalization_5[0][0]      
                                                                 layer_normalization_5[0][0]      
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 4, 128)       4224        layer_normalization_8[0][0]      
__________________________________________________________________________________________________
add_3 (Add)                     (None, 144, 64)      0           multi_head_attention_1[0][0]     
                                                                 add_2[0][0]                      
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 4, 128)       0           dense_8[0][0]                    
__________________________________________________________________________________________________
layer_normalization_6 (LayerNor (None, 144, 64)      128         add_3[0][0]                      
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 4, 32)        4128        dropout_5[0][0]                  
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 144, 256)     16640       layer_normalization_6[0][0]      
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 4, 32)        0           dense_9[0][0]                    
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 144, 256)     0           dense_6[0][0]                    
__________________________________________________________________________________________________
add_6 (Add)                     (None, 4, 32)        0           dropout_6[0][0]                  
                                                                 add_5[0][0]                      
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 144, 64)      16448       dropout_3[0][0]                  
__________________________________________________________________________________________________
tf.reshape_3 (TFOpLambda)       (None, 144, 128)     0           add_6[0][0]                      
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 144, 64)      0           dense_7[0][0]                    
__________________________________________________________________________________________________
layer_normalization_9 (LayerNor (None, 144, 128)     256         tf.reshape_3[0][0]               
__________________________________________________________________________________________________
layer_normalization_12 (LayerNo (None, 4, 32)        64          add_6[0][0]                      
__________________________________________________________________________________________________
add_4 (Add)                     (None, 144, 64)      0           dropout_4[0][0]                  
                                                                 add_3[0][0]                      
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 144, 64)      8256        layer_normalization_9[0][0]      
__________________________________________________________________________________________________
multi_head_attention_4 (MultiHe (None, 4, 32)        8416        layer_normalization_12[0][0]     
                                                                 layer_normalization_12[0][0]     
__________________________________________________________________________________________________
add_7 (Add)                     (None, 144, 64)      0           add_4[0][0]                      
                                                                 dense_10[0][0]                   
__________________________________________________________________________________________________
add_10 (Add)                    (None, 4, 32)        0           multi_head_attention_4[0][0]     
                                                                 add_6[0][0]                      
__________________________________________________________________________________________________
layer_normalization_10 (LayerNo (None, 144, 64)      128         add_7[0][0]                      
__________________________________________________________________________________________________
layer_normalization_13 (LayerNo (None, 4, 32)        64          add_10[0][0]                     
__________________________________________________________________________________________________
multi_head_attention_3 (MultiHe (None, 144, 64)      66368       layer_normalization_10[0][0]     
                                                                 layer_normalization_10[0][0]     
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 4, 128)       4224        layer_normalization_13[0][0]     
__________________________________________________________________________________________________
add_8 (Add)                     (None, 144, 64)      0           multi_head_attention_3[0][0]     
                                                                 add_7[0][0]                      
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 4, 128)       0           dense_13[0][0]                   
__________________________________________________________________________________________________
layer_normalization_11 (LayerNo (None, 144, 64)      128         add_8[0][0]                      
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 4, 32)        4128        dropout_9[0][0]                  
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 144, 256)     16640       layer_normalization_11[0][0]     
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 4, 32)        0           dense_14[0][0]                   
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 144, 256)     0           dense_11[0][0]                   
__________________________________________________________________________________________________
add_11 (Add)                    (None, 4, 32)        0           dropout_10[0][0]                 
                                                                 add_10[0][0]                     
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 144, 64)      16448       dropout_7[0][0]                  
__________________________________________________________________________________________________
tf.reshape_4 (TFOpLambda)       (None, 144, 128)     0           add_11[0][0]                     
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 144, 64)      0           dense_12[0][0]                   
__________________________________________________________________________________________________
layer_normalization_14 (LayerNo (None, 144, 128)     256         tf.reshape_4[0][0]               
__________________________________________________________________________________________________
layer_normalization_17 (LayerNo (None, 4, 32)        64          add_11[0][0]                     
__________________________________________________________________________________________________
add_9 (Add)                     (None, 144, 64)      0           dropout_8[0][0]                  
                                                                 add_8[0][0]                      
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 144, 64)      8256        layer_normalization_14[0][0]     
__________________________________________________________________________________________________
multi_head_attention_6 (MultiHe (None, 4, 32)        8416        layer_normalization_17[0][0]     
                                                                 layer_normalization_17[0][0]     
__________________________________________________________________________________________________
add_12 (Add)                    (None, 144, 64)      0           add_9[0][0]                      
                                                                 dense_15[0][0]                   
__________________________________________________________________________________________________
add_15 (Add)                    (None, 4, 32)        0           multi_head_attention_6[0][0]     
                                                                 add_11[0][0]                     
__________________________________________________________________________________________________
layer_normalization_15 (LayerNo (None, 144, 64)      128         add_12[0][0]                     
__________________________________________________________________________________________________
layer_normalization_18 (LayerNo (None, 4, 32)        64          add_15[0][0]                     
__________________________________________________________________________________________________
multi_head_attention_5 (MultiHe (None, 144, 64)      66368       layer_normalization_15[0][0]     
                                                                 layer_normalization_15[0][0]     
__________________________________________________________________________________________________
dense_18 (Dense)                (None, 4, 128)       4224        layer_normalization_18[0][0]     
__________________________________________________________________________________________________
add_13 (Add)                    (None, 144, 64)      0           multi_head_attention_5[0][0]     
                                                                 add_12[0][0]                     
__________________________________________________________________________________________________
dropout_13 (Dropout)            (None, 4, 128)       0           dense_18[0][0]                   
__________________________________________________________________________________________________
layer_normalization_16 (LayerNo (None, 144, 64)      128         add_13[0][0]                     
__________________________________________________________________________________________________
dense_19 (Dense)                (None, 4, 32)        4128        dropout_13[0][0]                 
__________________________________________________________________________________________________
dense_16 (Dense)                (None, 144, 256)     16640       layer_normalization_16[0][0]     
__________________________________________________________________________________________________
dropout_14 (Dropout)            (None, 4, 32)        0           dense_19[0][0]                   
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 144, 256)     0           dense_16[0][0]                   
__________________________________________________________________________________________________
add_16 (Add)                    (None, 4, 32)        0           dropout_14[0][0]                 
                                                                 add_15[0][0]                     
__________________________________________________________________________________________________
dense_17 (Dense)                (None, 144, 64)      16448       dropout_11[0][0]                 
__________________________________________________________________________________________________
tf.reshape_5 (TFOpLambda)       (None, 144, 128)     0           add_16[0][0]                     
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, 144, 64)      0           dense_17[0][0]                   
__________________________________________________________________________________________________
layer_normalization_19 (LayerNo (None, 144, 128)     256         tf.reshape_5[0][0]               
__________________________________________________________________________________________________
layer_normalization_22 (LayerNo (None, 4, 32)        64          add_16[0][0]                     
__________________________________________________________________________________________________
add_14 (Add)                    (None, 144, 64)      0           dropout_12[0][0]                 
                                                                 add_13[0][0]                     
__________________________________________________________________________________________________
dense_20 (Dense)                (None, 144, 64)      8256        layer_normalization_19[0][0]     
__________________________________________________________________________________________________
multi_head_attention_8 (MultiHe (None, 4, 32)        8416        layer_normalization_22[0][0]     
                                                                 layer_normalization_22[0][0]     
__________________________________________________________________________________________________
add_17 (Add)                    (None, 144, 64)      0           add_14[0][0]                     
                                                                 dense_20[0][0]                   
__________________________________________________________________________________________________
add_20 (Add)                    (None, 4, 32)        0           multi_head_attention_8[0][0]     
                                                                 add_16[0][0]                     
__________________________________________________________________________________________________
layer_normalization_20 (LayerNo (None, 144, 64)      128         add_17[0][0]                     
__________________________________________________________________________________________________
layer_normalization_23 (LayerNo (None, 4, 32)        64          add_20[0][0]                     
__________________________________________________________________________________________________
multi_head_attention_7 (MultiHe (None, 144, 64)      66368       layer_normalization_20[0][0]     
                                                                 layer_normalization_20[0][0]     
__________________________________________________________________________________________________
dense_23 (Dense)                (None, 4, 128)       4224        layer_normalization_23[0][0]     
__________________________________________________________________________________________________
add_18 (Add)                    (None, 144, 64)      0           multi_head_attention_7[0][0]     
                                                                 add_17[0][0]                     
__________________________________________________________________________________________________
dropout_17 (Dropout)            (None, 4, 128)       0           dense_23[0][0]                   
__________________________________________________________________________________________________
layer_normalization_21 (LayerNo (None, 144, 64)      128         add_18[0][0]                     
__________________________________________________________________________________________________
dense_24 (Dense)                (None, 4, 32)        4128        dropout_17[0][0]                 
__________________________________________________________________________________________________
dense_21 (Dense)                (None, 144, 256)     16640       layer_normalization_21[0][0]     
__________________________________________________________________________________________________
dropout_18 (Dropout)            (None, 4, 32)        0           dense_24[0][0]                   
__________________________________________________________________________________________________
dropout_15 (Dropout)            (None, 144, 256)     0           dense_21[0][0]                   
__________________________________________________________________________________________________
add_21 (Add)                    (None, 4, 32)        0           dropout_18[0][0]                 
                                                                 add_20[0][0]                     
__________________________________________________________________________________________________
dense_22 (Dense)                (None, 144, 64)      16448       dropout_15[0][0]                 
__________________________________________________________________________________________________
tf.reshape_6 (TFOpLambda)       (None, 144, 128)     0           add_21[0][0]                     
__________________________________________________________________________________________________
dropout_16 (Dropout)            (None, 144, 64)      0           dense_22[0][0]                   
__________________________________________________________________________________________________
layer_normalization_24 (LayerNo (None, 144, 128)     256         tf.reshape_6[0][0]               
__________________________________________________________________________________________________
add_19 (Add)                    (None, 144, 64)      0           dropout_16[0][0]                 
                                                                 add_18[0][0]                     
__________________________________________________________________________________________________
dense_25 (Dense)                (None, 144, 64)      8256        layer_normalization_24[0][0]     
__________________________________________________________________________________________________
add_22 (Add)                    (None, 144, 64)      0           add_19[0][0]                     
                                                                 dense_25[0][0]                   
__________________________________________________________________________________________________
layer_normalization_25 (LayerNo (None, 144, 64)      128         add_22[0][0]                     
__________________________________________________________________________________________________
multi_head_attention_9 (MultiHe (None, 144, 64)      66368       layer_normalization_25[0][0]     
                                                                 layer_normalization_25[0][0]     
__________________________________________________________________________________________________
add_23 (Add)                    (None, 144, 64)      0           multi_head_attention_9[0][0]     
                                                                 add_22[0][0]                     
__________________________________________________________________________________________________
layer_normalization_26 (LayerNo (None, 144, 64)      128         add_23[0][0]                     
__________________________________________________________________________________________________
dense_26 (Dense)                (None, 144, 256)     16640       layer_normalization_26[0][0]     
__________________________________________________________________________________________________
dropout_19 (Dropout)            (None, 144, 256)     0           dense_26[0][0]                   
__________________________________________________________________________________________________
dense_27 (Dense)                (None, 144, 64)      16448       dropout_19[0][0]                 
__________________________________________________________________________________________________
dropout_20 (Dropout)            (None, 144, 64)      0           dense_27[0][0]                   
__________________________________________________________________________________________________
add_24 (Add)                    (None, 144, 64)      0           dropout_20[0][0]                 
                                                                 add_23[0][0]                     
__________________________________________________________________________________________________
layer_normalization_27 (LayerNo (None, 144, 64)      128         add_24[0][0]                     
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 64)           0           layer_normalization_27[0][0]     
__________________________________________________________________________________________________
dense_28 (Dense)                (None, 100)          6500        global_average_pooling1d[0][0]   
==================================================================================================
Total params: 660,164
Trainable params: 660,164
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/50
313/313 [==============================] - 51s 118ms/step - loss: 4.2801 - accuracy: 0.0570 - top-5-accuracy: 0.2025 - val_loss: 4.0453 - val_accuracy: 0.0945 - val_top-5-accuracy: 0.2995
Epoch 2/50
313/313 [==============================] - 36s 114ms/step - loss: 3.9381 - accuracy: 0.1129 - top-5-accuracy: 0.3332 - val_loss: 3.8243 - val_accuracy: 0.1401 - val_top-5-accuracy: 0.3812
Epoch 3/50
313/313 [==============================] - 36s 115ms/step - loss: 3.7698 - accuracy: 0.1494 - top-5-accuracy: 0.3983 - val_loss: 3.6749 - val_accuracy: 0.1714 - val_top-5-accuracy: 0.4349
Epoch 4/50
313/313 [==============================] - 36s 115ms/step - loss: 3.6372 - accuracy: 0.1787 - top-5-accuracy: 0.4473 - val_loss: 3.6056 - val_accuracy: 0.1857 - val_top-5-accuracy: 0.4584
Epoch 5/50
313/313 [==============================] - 36s 116ms/step - loss: 3.5228 - accuracy: 0.2038 - top-5-accuracy: 0.4876 - val_loss: 3.4683 - val_accuracy: 0.2218 - val_top-5-accuracy: 0.5079
Epoch 6/50
313/313 [==============================] - 36s 115ms/step - loss: 3.4331 - accuracy: 0.2246 - top-5-accuracy: 0.5198 - val_loss: 3.4018 - val_accuracy: 0.2326 - val_top-5-accuracy: 0.5306
Epoch 7/50
313/313 [==============================] - 36s 116ms/step - loss: 3.3532 - accuracy: 0.2441 - top-5-accuracy: 0.5463 - val_loss: 3.3699 - val_accuracy: 0.2448 - val_top-5-accuracy: 0.5347
Epoch 8/50
313/313 [==============================] - 36s 115ms/step - loss: 3.2863 - accuracy: 0.2596 - top-5-accuracy: 0.5648 - val_loss: 3.2318 - val_accuracy: 0.2727 - val_top-5-accuracy: 0.5804
Epoch 9/50
313/313 [==============================] - 36s 115ms/step - loss: 3.2309 - accuracy: 0.2726 - top-5-accuracy: 0.5835 - val_loss: 3.2216 - val_accuracy: 0.2751 - val_top-5-accuracy: 0.5866
Epoch 10/50
313/313 [==============================] - 36s 115ms/step - loss: 3.1830 - accuracy: 0.2830 - top-5-accuracy: 0.5971 - val_loss: 3.1503 - val_accuracy: 0.2947 - val_top-5-accuracy: 0.6052
Epoch 11/50
313/313 [==============================] - 36s 115ms/step - loss: 3.1380 - accuracy: 0.2946 - top-5-accuracy: 0.6092 - val_loss: 3.1401 - val_accuracy: 0.2952 - val_top-5-accuracy: 0.6110
Epoch 12/50
313/313 [==============================] - 36s 115ms/step - loss: 3.1024 - accuracy: 0.3021 - top-5-accuracy: 0.6200 - val_loss: 3.0892 - val_accuracy: 0.3062 - val_top-5-accuracy: 0.6227
Epoch 13/50
313/313 [==============================] - 36s 115ms/step - loss: 3.0698 - accuracy: 0.3113 - top-5-accuracy: 0.6271 - val_loss: 3.1121 - val_accuracy: 0.2976 - val_top-5-accuracy: 0.6157
Epoch 14/50
313/313 [==============================] - 36s 115ms/step - loss: 3.0341 - accuracy: 0.3203 - top-5-accuracy: 0.6399 - val_loss: 3.0740 - val_accuracy: 0.3145 - val_top-5-accuracy: 0.6289
Epoch 15/50
313/313 [==============================] - 36s 115ms/step - loss: 3.0032 - accuracy: 0.3274 - top-5-accuracy: 0.6493 - val_loss: 3.0014 - val_accuracy: 0.3286 - val_top-5-accuracy: 0.6462
Epoch 16/50
313/313 [==============================] - 36s 116ms/step - loss: 2.9827 - accuracy: 0.3304 - top-5-accuracy: 0.6539 - val_loss: 3.0092 - val_accuracy: 0.3267 - val_top-5-accuracy: 0.6450
Epoch 17/50
313/313 [==============================] - 36s 116ms/step - loss: 2.9518 - accuracy: 0.3400 - top-5-accuracy: 0.6628 - val_loss: 2.9859 - val_accuracy: 0.3346 - val_top-5-accuracy: 0.6503
Epoch 18/50
313/313 [==============================] - 36s 115ms/step - loss: 2.9238 - accuracy: 0.3445 - top-5-accuracy: 0.6697 - val_loss: 2.9553 - val_accuracy: 0.3421 - val_top-5-accuracy: 0.6612
Epoch 19/50
313/313 [==============================] - 36s 115ms/step - loss: 2.9095 - accuracy: 0.3506 - top-5-accuracy: 0.6747 - val_loss: 2.9304 - val_accuracy: 0.3493 - val_top-5-accuracy: 0.6666
Epoch 20/50
313/313 [==============================] - 36s 114ms/step - loss: 2.8880 - accuracy: 0.3538 - top-5-accuracy: 0.6788 - val_loss: 2.9773 - val_accuracy: 0.3397 - val_top-5-accuracy: 0.6564
Epoch 21/50
313/313 [==============================] - 36s 115ms/step - loss: 2.8680 - accuracy: 0.3614 - top-5-accuracy: 0.6857 - val_loss: 2.9260 - val_accuracy: 0.3553 - val_top-5-accuracy: 0.6696
Epoch 22/50
313/313 [==============================] - 36s 114ms/step - loss: 2.8469 - accuracy: 0.3677 - top-5-accuracy: 0.6905 - val_loss: 2.9066 - val_accuracy: 0.3589 - val_top-5-accuracy: 0.6772
Epoch 23/50
313/313 [==============================] - 36s 115ms/step - loss: 2.8344 - accuracy: 0.3707 - top-5-accuracy: 0.6949 - val_loss: 2.8922 - val_accuracy: 0.3617 - val_top-5-accuracy: 0.6790
Epoch 24/50
313/313 [==============================] - 36s 115ms/step - loss: 2.8242 - accuracy: 0.3712 - top-5-accuracy: 0.6985 - val_loss: 2.8481 - val_accuracy: 0.3725 - val_top-5-accuracy: 0.6941
Epoch 25/50
313/313 [==============================] - 36s 115ms/step - loss: 2.7945 - accuracy: 0.3806 - top-5-accuracy: 0.7065 - val_loss: 2.9250 - val_accuracy: 0.3638 - val_top-5-accuracy: 0.6706
Epoch 26/50
313/313 [==============================] - 36s 114ms/step - loss: 2.7930 - accuracy: 0.3792 - top-5-accuracy: 0.7033 - val_loss: 2.8798 - val_accuracy: 0.3689 - val_top-5-accuracy: 0.6838
Epoch 27/50
313/313 [==============================] - 36s 116ms/step - loss: 2.7694 - accuracy: 0.3871 - top-5-accuracy: 0.7127 - val_loss: 2.8356 - val_accuracy: 0.3830 - val_top-5-accuracy: 0.6949
Epoch 28/50
313/313 [==============================] - 36s 115ms/step - loss: 2.7516 - accuracy: 0.3903 - top-5-accuracy: 0.7174 - val_loss: 2.8489 - val_accuracy: 0.3757 - val_top-5-accuracy: 0.6928
Epoch 29/50
313/313 [==============================] - 36s 116ms/step - loss: 2.7429 - accuracy: 0.3965 - top-5-accuracy: 0.7189 - val_loss: 2.8167 - val_accuracy: 0.3865 - val_top-5-accuracy: 0.6986
Epoch 30/50
313/313 [==============================] - 36s 115ms/step - loss: 2.7255 - accuracy: 0.3969 - top-5-accuracy: 0.7214 - val_loss: 2.8334 - val_accuracy: 0.3741 - val_top-5-accuracy: 0.6986
Epoch 31/50
313/313 [==============================] - 36s 116ms/step - loss: 2.7101 - accuracy: 0.4029 - top-5-accuracy: 0.7258 - val_loss: 2.8109 - val_accuracy: 0.3852 - val_top-5-accuracy: 0.6998
Epoch 32/50
313/313 [==============================] - 36s 116ms/step - loss: 2.7010 - accuracy: 0.4047 - top-5-accuracy: 0.7301 - val_loss: 2.7821 - val_accuracy: 0.3942 - val_top-5-accuracy: 0.7122
Epoch 33/50
313/313 [==============================] - 36s 115ms/step - loss: 2.6933 - accuracy: 0.4057 - top-5-accuracy: 0.7341 - val_loss: 2.7993 - val_accuracy: 0.3916 - val_top-5-accuracy: 0.7055
Epoch 34/50
313/313 [==============================] - 36s 115ms/step - loss: 2.6863 - accuracy: 0.4088 - top-5-accuracy: 0.7323 - val_loss: 2.7967 - val_accuracy: 0.3927 - val_top-5-accuracy: 0.7135
Epoch 35/50
313/313 [==============================] - 38s 123ms/step - loss: 2.6642 - accuracy: 0.4168 - top-5-accuracy: 0.7397 - val_loss: 2.7863 - val_accuracy: 0.3937 - val_top-5-accuracy: 0.7082
Epoch 36/50
313/313 [==============================] - 36s 115ms/step - loss: 2.6627 - accuracy: 0.4141 - top-5-accuracy: 0.7412 - val_loss: 2.7564 - val_accuracy: 0.4013 - val_top-5-accuracy: 0.7186
Epoch 37/50
313/313 [==============================] - 36s 115ms/step - loss: 2.6401 - accuracy: 0.4203 - top-5-accuracy: 0.7469 - val_loss: 2.7451 - val_accuracy: 0.4059 - val_top-5-accuracy: 0.7204
Epoch 38/50
313/313 [==============================] - 36s 116ms/step - loss: 2.6435 - accuracy: 0.4200 - top-5-accuracy: 0.7422 - val_loss: 2.7464 - val_accuracy: 0.4067 - val_top-5-accuracy: 0.7229
Epoch 39/50
313/313 [==============================] - 36s 115ms/step - loss: 2.6272 - accuracy: 0.4243 - top-5-accuracy: 0.7488 - val_loss: 2.7538 - val_accuracy: 0.4026 - val_top-5-accuracy: 0.7208
Epoch 40/50
313/313 [==============================] - 36s 115ms/step - loss: 2.6289 - accuracy: 0.4258 - top-5-accuracy: 0.7468 - val_loss: 2.7538 - val_accuracy: 0.4039 - val_top-5-accuracy: 0.7198
Epoch 41/50
313/313 [==============================] - 36s 116ms/step - loss: 2.6179 - accuracy: 0.4265 - top-5-accuracy: 0.7501 - val_loss: 2.7584 - val_accuracy: 0.3970 - val_top-5-accuracy: 0.7193
Epoch 42/50
313/313 [==============================] - 36s 115ms/step - loss: 2.6061 - accuracy: 0.4283 - top-5-accuracy: 0.7576 - val_loss: 2.7954 - val_accuracy: 0.3918 - val_top-5-accuracy: 0.7105
Epoch 43/50
313/313 [==============================] - 36s 116ms/step - loss: 2.6021 - accuracy: 0.4328 - top-5-accuracy: 0.7571 - val_loss: 2.7306 - val_accuracy: 0.4161 - val_top-5-accuracy: 0.7211
Epoch 44/50
313/313 [==============================] - 36s 116ms/step - loss: 2.5891 - accuracy: 0.4320 - top-5-accuracy: 0.7594 - val_loss: 2.7579 - val_accuracy: 0.4002 - val_top-5-accuracy: 0.7235
Epoch 45/50
313/313 [==============================] - 36s 115ms/step - loss: 2.5844 - accuracy: 0.4382 - top-5-accuracy: 0.7605 - val_loss: 2.7026 - val_accuracy: 0.4194 - val_top-5-accuracy: 0.7316
Epoch 46/50
313/313 [==============================] - 36s 116ms/step - loss: 2.5843 - accuracy: 0.4385 - top-5-accuracy: 0.7603 - val_loss: 2.6975 - val_accuracy: 0.4157 - val_top-5-accuracy: 0.7364
Epoch 47/50
313/313 [==============================] - 36s 116ms/step - loss: 2.5725 - accuracy: 0.4385 - top-5-accuracy: 0.7625 - val_loss: 2.6863 - val_accuracy: 0.4189 - val_top-5-accuracy: 0.7393
Epoch 48/50
313/313 [==============================] - 36s 116ms/step - loss: 2.5629 - accuracy: 0.4405 - top-5-accuracy: 0.7652 - val_loss: 2.6832 - val_accuracy: 0.4232 - val_top-5-accuracy: 0.7415
Epoch 49/50
313/313 [==============================] - 36s 115ms/step - loss: 2.5690 - accuracy: 0.4408 - top-5-accuracy: 0.7639 - val_loss: 2.7229 - val_accuracy: 0.4110 - val_top-5-accuracy: 0.7345
Epoch 50/50
313/313 [==============================] - 36s 115ms/step - loss: 2.5600 - accuracy: 0.4396 - top-5-accuracy: 0.7677 - val_loss: 2.6856 - val_accuracy: 0.4241 - val_top-5-accuracy: 0.7410

```
</div>
### Visualize the training progress of the model.


```python
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracies Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()
```


    
![png](/img/examples/vision/transformer_in_transformer/transformer_in_transformer_21_0.png)
    



    
![png](/img/examples/vision/transformer_in_transformer/transformer_in_transformer_21_1.png)
    


### Let's display the final results of the test on CIFAR-100.


```python
loss, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
```

<div class="k-default-codeblock">
```
313/313 [==============================] - 6s 18ms/step - loss: 2.6842 - accuracy: 0.4270 - top-5-accuracy: 0.7349
Test loss: 2.68
Test accuracy: 42.7%
Test top 5 accuracy: 73.49%

```
</div>
After 50 epochs, the TNT model achieves around 42% accuracy and
73% top-5 accuracy on the test data. It only has 0.6M parameters.
From the above loss curve, we can find that the model gradually converges,
but it never achieves state of the art performance. We could apply further data
augmentation to
obtain better performance, like [RandAugment](https://arxiv.org/abs/1909.13719),
[MixUp](https://arxiv.org/abs/1710.09412)
etc. We also can adjust the depth of model, learning rate or increase the size of
embedding. Compared to the conventional
vision transformers [ViT](https://arxiv.org/abs/2010.11929) which corrupts the local
structure
of the patch, the TNT can better preserve and model the local information
for visual recognition.
