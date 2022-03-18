# Timeseries classification with a Transformer model

**Author:** [Theodoros Ntakouris](https://github.com/ntakouris)<br>
**Date created:** 2021/06/25<br>
**Last modified:** 2021/08/05<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_classification_transformer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_classification_transformer.py)


**Description:** This notebook demonstrates how to do timeseries classification using a Transformer model.

---
## Introduction

This is the Transformer architecture from
[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
applied to timeseries instead of natural language.

This example requires TensorFlow 2.4 or higher.

---
## Load the dataset

We are going to use the same dataset and preprocessing as the
[TimeSeries Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)
example.


```python
import numpy as np


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
```

---
## Build the model

Our model processes a tensor of shape `(batch size, sequence length, features)`,
where `sequence length` is the number of time steps and `features` is each input
timeseries.

You can replace your classification RNN layers with this one: the
inputs are fully compatible!


```python
from tensorflow import keras
from tensorflow.keras import layers
```

We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`.


```python

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

```

The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer. For
this example, a `GlobalAveragePooling1D` layer is sufficient.


```python

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

```

---
## Train and evaluate


```python
input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)
```

<div class="k-default-codeblock">
```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 500, 1)]     0                                            
__________________________________________________________________________________________________
layer_normalization (LayerNorma (None, 500, 1)       2           input_1[0][0]                    
__________________________________________________________________________________________________
multi_head_attention (MultiHead (None, 500, 1)       7169        layer_normalization[0][0]        
                                                                 layer_normalization[0][0]        
__________________________________________________________________________________________________
dropout (Dropout)               (None, 500, 1)       0           multi_head_attention[0][0]       
__________________________________________________________________________________________________
tf.__operators__.add (TFOpLambd (None, 500, 1)       0           dropout[0][0]                    
                                                                 input_1[0][0]                    
__________________________________________________________________________________________________
layer_normalization_1 (LayerNor (None, 500, 1)       2           tf.__operators__.add[0][0]       
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 500, 4)       8           layer_normalization_1[0][0]      
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 500, 4)       0           conv1d[0][0]                     
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 500, 1)       5           dropout_1[0][0]                  
__________________________________________________________________________________________________
tf.__operators__.add_1 (TFOpLam (None, 500, 1)       0           conv1d_1[0][0]                   
                                                                 tf.__operators__.add[0][0]       
__________________________________________________________________________________________________
layer_normalization_2 (LayerNor (None, 500, 1)       2           tf.__operators__.add_1[0][0]     
__________________________________________________________________________________________________
multi_head_attention_1 (MultiHe (None, 500, 1)       7169        layer_normalization_2[0][0]      
                                                                 layer_normalization_2[0][0]      
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 500, 1)       0           multi_head_attention_1[0][0]     
__________________________________________________________________________________________________
tf.__operators__.add_2 (TFOpLam (None, 500, 1)       0           dropout_2[0][0]                  
                                                                 tf.__operators__.add_1[0][0]     
__________________________________________________________________________________________________
layer_normalization_3 (LayerNor (None, 500, 1)       2           tf.__operators__.add_2[0][0]     
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 500, 4)       8           layer_normalization_3[0][0]      
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 500, 4)       0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 500, 1)       5           dropout_3[0][0]                  
__________________________________________________________________________________________________
tf.__operators__.add_3 (TFOpLam (None, 500, 1)       0           conv1d_3[0][0]                   
                                                                 tf.__operators__.add_2[0][0]     
__________________________________________________________________________________________________
layer_normalization_4 (LayerNor (None, 500, 1)       2           tf.__operators__.add_3[0][0]     
__________________________________________________________________________________________________
multi_head_attention_2 (MultiHe (None, 500, 1)       7169        layer_normalization_4[0][0]      
                                                                 layer_normalization_4[0][0]      
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 500, 1)       0           multi_head_attention_2[0][0]     
__________________________________________________________________________________________________
tf.__operators__.add_4 (TFOpLam (None, 500, 1)       0           dropout_4[0][0]                  
                                                                 tf.__operators__.add_3[0][0]     
__________________________________________________________________________________________________
layer_normalization_5 (LayerNor (None, 500, 1)       2           tf.__operators__.add_4[0][0]     
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 500, 4)       8           layer_normalization_5[0][0]      
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 500, 4)       0           conv1d_4[0][0]                   
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 500, 1)       5           dropout_5[0][0]                  
__________________________________________________________________________________________________
tf.__operators__.add_5 (TFOpLam (None, 500, 1)       0           conv1d_5[0][0]                   
                                                                 tf.__operators__.add_4[0][0]     
__________________________________________________________________________________________________
layer_normalization_6 (LayerNor (None, 500, 1)       2           tf.__operators__.add_5[0][0]     
__________________________________________________________________________________________________
multi_head_attention_3 (MultiHe (None, 500, 1)       7169        layer_normalization_6[0][0]      
                                                                 layer_normalization_6[0][0]      
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 500, 1)       0           multi_head_attention_3[0][0]     
__________________________________________________________________________________________________
tf.__operators__.add_6 (TFOpLam (None, 500, 1)       0           dropout_6[0][0]                  
                                                                 tf.__operators__.add_5[0][0]     
__________________________________________________________________________________________________
layer_normalization_7 (LayerNor (None, 500, 1)       2           tf.__operators__.add_6[0][0]     
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 500, 4)       8           layer_normalization_7[0][0]      
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 500, 4)       0           conv1d_6[0][0]                   
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 500, 1)       5           dropout_7[0][0]                  
__________________________________________________________________________________________________
tf.__operators__.add_7 (TFOpLam (None, 500, 1)       0           conv1d_7[0][0]                   
                                                                 tf.__operators__.add_6[0][0]     
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 500)          0           tf.__operators__.add_7[0][0]     
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          64128       global_average_pooling1d[0][0]   
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 128)          0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2)            258         dropout_8[0][0]                  
==================================================================================================
Total params: 93,130
Trainable params: 93,130
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/200
45/45 [==============================] - 26s 499ms/step - loss: 1.0233 - sparse_categorical_accuracy: 0.5174 - val_loss: 0.7853 - val_sparse_categorical_accuracy: 0.5368
Epoch 2/200
45/45 [==============================] - 22s 499ms/step - loss: 0.9108 - sparse_categorical_accuracy: 0.5507 - val_loss: 0.7169 - val_sparse_categorical_accuracy: 0.5659
Epoch 3/200
45/45 [==============================] - 23s 509ms/step - loss: 0.8177 - sparse_categorical_accuracy: 0.5851 - val_loss: 0.6851 - val_sparse_categorical_accuracy: 0.5839
Epoch 4/200
45/45 [==============================] - 24s 532ms/step - loss: 0.7494 - sparse_categorical_accuracy: 0.6160 - val_loss: 0.6554 - val_sparse_categorical_accuracy: 0.6214
Epoch 5/200
45/45 [==============================] - 23s 520ms/step - loss: 0.7287 - sparse_categorical_accuracy: 0.6319 - val_loss: 0.6333 - val_sparse_categorical_accuracy: 0.6463
Epoch 6/200
45/45 [==============================] - 23s 509ms/step - loss: 0.7108 - sparse_categorical_accuracy: 0.6424 - val_loss: 0.6185 - val_sparse_categorical_accuracy: 0.6546
Epoch 7/200
45/45 [==============================] - 23s 512ms/step - loss: 0.6624 - sparse_categorical_accuracy: 0.6667 - val_loss: 0.6023 - val_sparse_categorical_accuracy: 0.6657
Epoch 8/200
45/45 [==============================] - 23s 518ms/step - loss: 0.6392 - sparse_categorical_accuracy: 0.6774 - val_loss: 0.5935 - val_sparse_categorical_accuracy: 0.6796
Epoch 9/200
45/45 [==============================] - 23s 513ms/step - loss: 0.5978 - sparse_categorical_accuracy: 0.6955 - val_loss: 0.5778 - val_sparse_categorical_accuracy: 0.6907
Epoch 10/200
45/45 [==============================] - 23s 511ms/step - loss: 0.5909 - sparse_categorical_accuracy: 0.6948 - val_loss: 0.5687 - val_sparse_categorical_accuracy: 0.6935
Epoch 11/200
45/45 [==============================] - 23s 513ms/step - loss: 0.5785 - sparse_categorical_accuracy: 0.7021 - val_loss: 0.5628 - val_sparse_categorical_accuracy: 0.6990
Epoch 12/200
45/45 [==============================] - 23s 514ms/step - loss: 0.5547 - sparse_categorical_accuracy: 0.7247 - val_loss: 0.5545 - val_sparse_categorical_accuracy: 0.7101
Epoch 13/200
45/45 [==============================] - 24s 535ms/step - loss: 0.5705 - sparse_categorical_accuracy: 0.7240 - val_loss: 0.5461 - val_sparse_categorical_accuracy: 0.7240
Epoch 14/200
45/45 [==============================] - 23s 517ms/step - loss: 0.5538 - sparse_categorical_accuracy: 0.7250 - val_loss: 0.5403 - val_sparse_categorical_accuracy: 0.7212
Epoch 15/200
45/45 [==============================] - 23s 515ms/step - loss: 0.5144 - sparse_categorical_accuracy: 0.7500 - val_loss: 0.5318 - val_sparse_categorical_accuracy: 0.7295
Epoch 16/200
45/45 [==============================] - 23s 512ms/step - loss: 0.5200 - sparse_categorical_accuracy: 0.7521 - val_loss: 0.5286 - val_sparse_categorical_accuracy: 0.7379
Epoch 17/200
45/45 [==============================] - 23s 515ms/step - loss: 0.4910 - sparse_categorical_accuracy: 0.7590 - val_loss: 0.5229 - val_sparse_categorical_accuracy: 0.7393
Epoch 18/200
45/45 [==============================] - 23s 514ms/step - loss: 0.5013 - sparse_categorical_accuracy: 0.7427 - val_loss: 0.5157 - val_sparse_categorical_accuracy: 0.7462
Epoch 19/200
45/45 [==============================] - 23s 511ms/step - loss: 0.4883 - sparse_categorical_accuracy: 0.7712 - val_loss: 0.5123 - val_sparse_categorical_accuracy: 0.7490
Epoch 20/200
45/45 [==============================] - 23s 514ms/step - loss: 0.4935 - sparse_categorical_accuracy: 0.7667 - val_loss: 0.5032 - val_sparse_categorical_accuracy: 0.7545
Epoch 21/200
45/45 [==============================] - 23s 514ms/step - loss: 0.4551 - sparse_categorical_accuracy: 0.7799 - val_loss: 0.4978 - val_sparse_categorical_accuracy: 0.7573
Epoch 22/200
45/45 [==============================] - 23s 516ms/step - loss: 0.4477 - sparse_categorical_accuracy: 0.7948 - val_loss: 0.4941 - val_sparse_categorical_accuracy: 0.7531
Epoch 23/200
45/45 [==============================] - 23s 518ms/step - loss: 0.4549 - sparse_categorical_accuracy: 0.7858 - val_loss: 0.4893 - val_sparse_categorical_accuracy: 0.7656
Epoch 24/200
45/45 [==============================] - 23s 516ms/step - loss: 0.4426 - sparse_categorical_accuracy: 0.7948 - val_loss: 0.4842 - val_sparse_categorical_accuracy: 0.7712
Epoch 25/200
45/45 [==============================] - 23s 520ms/step - loss: 0.4360 - sparse_categorical_accuracy: 0.8035 - val_loss: 0.4798 - val_sparse_categorical_accuracy: 0.7809
Epoch 26/200
45/45 [==============================] - 23s 515ms/step - loss: 0.4316 - sparse_categorical_accuracy: 0.8035 - val_loss: 0.4715 - val_sparse_categorical_accuracy: 0.7809
Epoch 27/200
45/45 [==============================] - 23s 518ms/step - loss: 0.4084 - sparse_categorical_accuracy: 0.8146 - val_loss: 0.4676 - val_sparse_categorical_accuracy: 0.7878
Epoch 28/200
45/45 [==============================] - 23s 515ms/step - loss: 0.3998 - sparse_categorical_accuracy: 0.8240 - val_loss: 0.4667 - val_sparse_categorical_accuracy: 0.7933
Epoch 29/200
45/45 [==============================] - 23s 514ms/step - loss: 0.3993 - sparse_categorical_accuracy: 0.8198 - val_loss: 0.4603 - val_sparse_categorical_accuracy: 0.7892
Epoch 30/200
45/45 [==============================] - 23s 515ms/step - loss: 0.4031 - sparse_categorical_accuracy: 0.8243 - val_loss: 0.4562 - val_sparse_categorical_accuracy: 0.7920
Epoch 31/200
45/45 [==============================] - 23s 511ms/step - loss: 0.3891 - sparse_categorical_accuracy: 0.8184 - val_loss: 0.4528 - val_sparse_categorical_accuracy: 0.7920
Epoch 32/200
45/45 [==============================] - 23s 516ms/step - loss: 0.3922 - sparse_categorical_accuracy: 0.8292 - val_loss: 0.4485 - val_sparse_categorical_accuracy: 0.7892
Epoch 33/200
45/45 [==============================] - 23s 516ms/step - loss: 0.3802 - sparse_categorical_accuracy: 0.8309 - val_loss: 0.4463 - val_sparse_categorical_accuracy: 0.8003
Epoch 34/200
45/45 [==============================] - 23s 514ms/step - loss: 0.3711 - sparse_categorical_accuracy: 0.8372 - val_loss: 0.4427 - val_sparse_categorical_accuracy: 0.7975
Epoch 35/200
45/45 [==============================] - 23s 512ms/step - loss: 0.3744 - sparse_categorical_accuracy: 0.8378 - val_loss: 0.4366 - val_sparse_categorical_accuracy: 0.8072
Epoch 36/200
45/45 [==============================] - 23s 511ms/step - loss: 0.3653 - sparse_categorical_accuracy: 0.8372 - val_loss: 0.4338 - val_sparse_categorical_accuracy: 0.8072
Epoch 37/200
45/45 [==============================] - 23s 512ms/step - loss: 0.3681 - sparse_categorical_accuracy: 0.8382 - val_loss: 0.4337 - val_sparse_categorical_accuracy: 0.8058
Epoch 38/200
45/45 [==============================] - 23s 512ms/step - loss: 0.3634 - sparse_categorical_accuracy: 0.8514 - val_loss: 0.4264 - val_sparse_categorical_accuracy: 0.8128
Epoch 39/200
45/45 [==============================] - 23s 512ms/step - loss: 0.3498 - sparse_categorical_accuracy: 0.8535 - val_loss: 0.4211 - val_sparse_categorical_accuracy: 0.8225
Epoch 40/200
45/45 [==============================] - 23s 514ms/step - loss: 0.3358 - sparse_categorical_accuracy: 0.8663 - val_loss: 0.4161 - val_sparse_categorical_accuracy: 0.8197
Epoch 41/200
45/45 [==============================] - 23s 512ms/step - loss: 0.3448 - sparse_categorical_accuracy: 0.8573 - val_loss: 0.4161 - val_sparse_categorical_accuracy: 0.8169
Epoch 42/200
45/45 [==============================] - 23s 512ms/step - loss: 0.3439 - sparse_categorical_accuracy: 0.8552 - val_loss: 0.4119 - val_sparse_categorical_accuracy: 0.8211
Epoch 43/200
45/45 [==============================] - 23s 510ms/step - loss: 0.3335 - sparse_categorical_accuracy: 0.8660 - val_loss: 0.4101 - val_sparse_categorical_accuracy: 0.8266
Epoch 44/200
45/45 [==============================] - 23s 510ms/step - loss: 0.3235 - sparse_categorical_accuracy: 0.8660 - val_loss: 0.4067 - val_sparse_categorical_accuracy: 0.8294
Epoch 45/200
45/45 [==============================] - 23s 510ms/step - loss: 0.3273 - sparse_categorical_accuracy: 0.8656 - val_loss: 0.4033 - val_sparse_categorical_accuracy: 0.8350
Epoch 46/200
45/45 [==============================] - 23s 513ms/step - loss: 0.3277 - sparse_categorical_accuracy: 0.8608 - val_loss: 0.3994 - val_sparse_categorical_accuracy: 0.8336
Epoch 47/200
45/45 [==============================] - 23s 519ms/step - loss: 0.3136 - sparse_categorical_accuracy: 0.8708 - val_loss: 0.3945 - val_sparse_categorical_accuracy: 0.8363
Epoch 48/200
45/45 [==============================] - 23s 518ms/step - loss: 0.3122 - sparse_categorical_accuracy: 0.8764 - val_loss: 0.3925 - val_sparse_categorical_accuracy: 0.8350
Epoch 49/200
45/45 [==============================] - 23s 519ms/step - loss: 0.3035 - sparse_categorical_accuracy: 0.8826 - val_loss: 0.3906 - val_sparse_categorical_accuracy: 0.8308
Epoch 50/200
45/45 [==============================] - 23s 512ms/step - loss: 0.2994 - sparse_categorical_accuracy: 0.8823 - val_loss: 0.3888 - val_sparse_categorical_accuracy: 0.8377
Epoch 51/200
45/45 [==============================] - 23s 514ms/step - loss: 0.3023 - sparse_categorical_accuracy: 0.8781 - val_loss: 0.3862 - val_sparse_categorical_accuracy: 0.8391
Epoch 52/200
45/45 [==============================] - 23s 515ms/step - loss: 0.3012 - sparse_categorical_accuracy: 0.8833 - val_loss: 0.3854 - val_sparse_categorical_accuracy: 0.8350
Epoch 53/200
45/45 [==============================] - 23s 513ms/step - loss: 0.2890 - sparse_categorical_accuracy: 0.8837 - val_loss: 0.3837 - val_sparse_categorical_accuracy: 0.8363
Epoch 54/200
45/45 [==============================] - 23s 513ms/step - loss: 0.2931 - sparse_categorical_accuracy: 0.8858 - val_loss: 0.3809 - val_sparse_categorical_accuracy: 0.8433
Epoch 55/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2867 - sparse_categorical_accuracy: 0.8885 - val_loss: 0.3784 - val_sparse_categorical_accuracy: 0.8447
Epoch 56/200
45/45 [==============================] - 23s 511ms/step - loss: 0.2731 - sparse_categorical_accuracy: 0.8986 - val_loss: 0.3756 - val_sparse_categorical_accuracy: 0.8488
Epoch 57/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2754 - sparse_categorical_accuracy: 0.8955 - val_loss: 0.3759 - val_sparse_categorical_accuracy: 0.8474
Epoch 58/200
45/45 [==============================] - 23s 511ms/step - loss: 0.2775 - sparse_categorical_accuracy: 0.8976 - val_loss: 0.3704 - val_sparse_categorical_accuracy: 0.8474
Epoch 59/200
45/45 [==============================] - 23s 513ms/step - loss: 0.2770 - sparse_categorical_accuracy: 0.9000 - val_loss: 0.3698 - val_sparse_categorical_accuracy: 0.8558
Epoch 60/200
45/45 [==============================] - 23s 516ms/step - loss: 0.2688 - sparse_categorical_accuracy: 0.8965 - val_loss: 0.3697 - val_sparse_categorical_accuracy: 0.8502
Epoch 61/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2716 - sparse_categorical_accuracy: 0.8972 - val_loss: 0.3710 - val_sparse_categorical_accuracy: 0.8405
Epoch 62/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2635 - sparse_categorical_accuracy: 0.9087 - val_loss: 0.3656 - val_sparse_categorical_accuracy: 0.8488
Epoch 63/200
45/45 [==============================] - 23s 520ms/step - loss: 0.2596 - sparse_categorical_accuracy: 0.8979 - val_loss: 0.3654 - val_sparse_categorical_accuracy: 0.8488
Epoch 64/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2586 - sparse_categorical_accuracy: 0.9062 - val_loss: 0.3634 - val_sparse_categorical_accuracy: 0.8530
Epoch 65/200
45/45 [==============================] - 23s 516ms/step - loss: 0.2491 - sparse_categorical_accuracy: 0.9139 - val_loss: 0.3591 - val_sparse_categorical_accuracy: 0.8530
Epoch 66/200
45/45 [==============================] - 23s 519ms/step - loss: 0.2600 - sparse_categorical_accuracy: 0.9017 - val_loss: 0.3621 - val_sparse_categorical_accuracy: 0.8516
Epoch 67/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2465 - sparse_categorical_accuracy: 0.9156 - val_loss: 0.3608 - val_sparse_categorical_accuracy: 0.8488
Epoch 68/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2502 - sparse_categorical_accuracy: 0.9101 - val_loss: 0.3557 - val_sparse_categorical_accuracy: 0.8627
Epoch 69/200
45/45 [==============================] - 23s 517ms/step - loss: 0.2418 - sparse_categorical_accuracy: 0.9104 - val_loss: 0.3561 - val_sparse_categorical_accuracy: 0.8502
Epoch 70/200
45/45 [==============================] - 23s 516ms/step - loss: 0.2463 - sparse_categorical_accuracy: 0.9049 - val_loss: 0.3554 - val_sparse_categorical_accuracy: 0.8613
Epoch 71/200
45/45 [==============================] - 23s 520ms/step - loss: 0.2372 - sparse_categorical_accuracy: 0.9177 - val_loss: 0.3548 - val_sparse_categorical_accuracy: 0.8627
Epoch 72/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2365 - sparse_categorical_accuracy: 0.9118 - val_loss: 0.3528 - val_sparse_categorical_accuracy: 0.8655
Epoch 73/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2420 - sparse_categorical_accuracy: 0.9083 - val_loss: 0.3510 - val_sparse_categorical_accuracy: 0.8655
Epoch 74/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2342 - sparse_categorical_accuracy: 0.9205 - val_loss: 0.3478 - val_sparse_categorical_accuracy: 0.8669
Epoch 75/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2337 - sparse_categorical_accuracy: 0.9062 - val_loss: 0.3484 - val_sparse_categorical_accuracy: 0.8655
Epoch 76/200
45/45 [==============================] - 23s 516ms/step - loss: 0.2298 - sparse_categorical_accuracy: 0.9153 - val_loss: 0.3478 - val_sparse_categorical_accuracy: 0.8585
Epoch 77/200
45/45 [==============================] - 23s 516ms/step - loss: 0.2218 - sparse_categorical_accuracy: 0.9243 - val_loss: 0.3467 - val_sparse_categorical_accuracy: 0.8613
Epoch 78/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2352 - sparse_categorical_accuracy: 0.9083 - val_loss: 0.3431 - val_sparse_categorical_accuracy: 0.8641
Epoch 79/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2218 - sparse_categorical_accuracy: 0.9194 - val_loss: 0.3448 - val_sparse_categorical_accuracy: 0.8613
Epoch 80/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2246 - sparse_categorical_accuracy: 0.9198 - val_loss: 0.3417 - val_sparse_categorical_accuracy: 0.8682
Epoch 81/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2168 - sparse_categorical_accuracy: 0.9201 - val_loss: 0.3397 - val_sparse_categorical_accuracy: 0.8641
Epoch 82/200
45/45 [==============================] - 23s 517ms/step - loss: 0.2254 - sparse_categorical_accuracy: 0.9153 - val_loss: 0.3373 - val_sparse_categorical_accuracy: 0.8682
Epoch 83/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2230 - sparse_categorical_accuracy: 0.9194 - val_loss: 0.3391 - val_sparse_categorical_accuracy: 0.8655
Epoch 84/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2124 - sparse_categorical_accuracy: 0.9240 - val_loss: 0.3370 - val_sparse_categorical_accuracy: 0.8682
Epoch 85/200
45/45 [==============================] - 23s 515ms/step - loss: 0.2123 - sparse_categorical_accuracy: 0.9278 - val_loss: 0.3394 - val_sparse_categorical_accuracy: 0.8571
Epoch 86/200
45/45 [==============================] - 23s 520ms/step - loss: 0.2119 - sparse_categorical_accuracy: 0.9260 - val_loss: 0.3355 - val_sparse_categorical_accuracy: 0.8627
Epoch 87/200
45/45 [==============================] - 23s 517ms/step - loss: 0.2052 - sparse_categorical_accuracy: 0.9247 - val_loss: 0.3353 - val_sparse_categorical_accuracy: 0.8738
Epoch 88/200
45/45 [==============================] - 23s 518ms/step - loss: 0.2089 - sparse_categorical_accuracy: 0.9299 - val_loss: 0.3342 - val_sparse_categorical_accuracy: 0.8779
Epoch 89/200
45/45 [==============================] - 23s 519ms/step - loss: 0.2027 - sparse_categorical_accuracy: 0.9250 - val_loss: 0.3353 - val_sparse_categorical_accuracy: 0.8793
Epoch 90/200
45/45 [==============================] - 23s 517ms/step - loss: 0.2110 - sparse_categorical_accuracy: 0.9264 - val_loss: 0.3320 - val_sparse_categorical_accuracy: 0.8752
Epoch 91/200
45/45 [==============================] - 23s 516ms/step - loss: 0.1965 - sparse_categorical_accuracy: 0.9292 - val_loss: 0.3339 - val_sparse_categorical_accuracy: 0.8710
Epoch 92/200
45/45 [==============================] - 23s 520ms/step - loss: 0.2030 - sparse_categorical_accuracy: 0.9253 - val_loss: 0.3296 - val_sparse_categorical_accuracy: 0.8752
Epoch 93/200
45/45 [==============================] - 23s 519ms/step - loss: 0.1969 - sparse_categorical_accuracy: 0.9347 - val_loss: 0.3298 - val_sparse_categorical_accuracy: 0.8807
Epoch 94/200
45/45 [==============================] - 23s 518ms/step - loss: 0.1939 - sparse_categorical_accuracy: 0.9295 - val_loss: 0.3300 - val_sparse_categorical_accuracy: 0.8779
Epoch 95/200
45/45 [==============================] - 23s 517ms/step - loss: 0.1930 - sparse_categorical_accuracy: 0.9330 - val_loss: 0.3305 - val_sparse_categorical_accuracy: 0.8766
Epoch 96/200
45/45 [==============================] - 23s 518ms/step - loss: 0.1946 - sparse_categorical_accuracy: 0.9288 - val_loss: 0.3288 - val_sparse_categorical_accuracy: 0.8669
Epoch 97/200
45/45 [==============================] - 23s 518ms/step - loss: 0.1951 - sparse_categorical_accuracy: 0.9264 - val_loss: 0.3281 - val_sparse_categorical_accuracy: 0.8682
Epoch 98/200
45/45 [==============================] - 23s 516ms/step - loss: 0.1899 - sparse_categorical_accuracy: 0.9354 - val_loss: 0.3307 - val_sparse_categorical_accuracy: 0.8696
Epoch 99/200
45/45 [==============================] - 23s 519ms/step - loss: 0.1901 - sparse_categorical_accuracy: 0.9250 - val_loss: 0.3307 - val_sparse_categorical_accuracy: 0.8710
Epoch 100/200
45/45 [==============================] - 23s 516ms/step - loss: 0.1902 - sparse_categorical_accuracy: 0.9319 - val_loss: 0.3259 - val_sparse_categorical_accuracy: 0.8696
Epoch 101/200
45/45 [==============================] - 23s 518ms/step - loss: 0.1868 - sparse_categorical_accuracy: 0.9358 - val_loss: 0.3262 - val_sparse_categorical_accuracy: 0.8724
Epoch 102/200
45/45 [==============================] - 23s 518ms/step - loss: 0.1779 - sparse_categorical_accuracy: 0.9431 - val_loss: 0.3250 - val_sparse_categorical_accuracy: 0.8710
Epoch 103/200
45/45 [==============================] - 23s 520ms/step - loss: 0.1870 - sparse_categorical_accuracy: 0.9351 - val_loss: 0.3260 - val_sparse_categorical_accuracy: 0.8724
Epoch 104/200
45/45 [==============================] - 23s 521ms/step - loss: 0.1826 - sparse_categorical_accuracy: 0.9344 - val_loss: 0.3232 - val_sparse_categorical_accuracy: 0.8766
Epoch 105/200
45/45 [==============================] - 23s 519ms/step - loss: 0.1731 - sparse_categorical_accuracy: 0.9399 - val_loss: 0.3245 - val_sparse_categorical_accuracy: 0.8724
Epoch 106/200
45/45 [==============================] - 23s 518ms/step - loss: 0.1766 - sparse_categorical_accuracy: 0.9361 - val_loss: 0.3254 - val_sparse_categorical_accuracy: 0.8682
Epoch 107/200

```
</div>
---
## Conclusions

In about 110-120 epochs (25s each on Colab), the model reaches a training
accuracy of ~0.95, validation accuracy of ~84 and a testing
accuracy of ~85, without hyperparameter tuning. And that is for a model
with less than 100k parameters. Of course, parameter count and accuracy could be
improved by a hyperparameter search and a more sophisticated learning rate
schedule, or a different optimizer.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/timeseries_transformer_classification) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/timeseries_transformer_classification).
