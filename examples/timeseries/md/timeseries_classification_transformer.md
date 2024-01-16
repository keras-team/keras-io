# Timeseries classification with a Transformer model

**Author:** [Theodoros Ntakouris](https://github.com/ntakouris)<br>
**Date created:** 2021/06/25<br>
**Last modified:** 2021/08/05<br>
**Description:** This notebook demonstrates how to do timeseries classification using a Transformer model.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_classification_transformer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_classification_transformer.py)



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
import keras
from keras import layers


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

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
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
    epochs=150,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold"> Param # </span>┃<span style="font-weight: bold"> Connected to         </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ multi_head_attenti… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">7,169</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MultiHeadAttentio…</span> │                   │         │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ multi_head_attentio… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalization │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ dropout_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">8</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv1d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">5</span> │ dropout_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ conv1d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ multi_head_attenti… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">7,169</span> │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MultiHeadAttentio…</span> │                   │         │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ multi_head_attentio… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ dropout_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ add_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">8</span> │ add_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv1d_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">5</span> │ dropout_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ conv1d_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ add_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ multi_head_attenti… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">7,169</span> │ add_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MultiHeadAttentio…</span> │                   │         │ add_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ multi_head_attentio… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ dropout_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ add_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">8</span> │ add_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv1d_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">5</span> │ dropout_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ conv1d_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ add_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ multi_head_attenti… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">7,169</span> │ add_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MultiHeadAttentio…</span> │                   │         │ add_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_10          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ multi_head_attentio… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ dropout_10[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ add_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">8</span> │ add_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_11          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv1d_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ conv1d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">5</span> │ dropout_11[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ layer_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2</span> │ conv1d_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalizatio…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ add_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ layer_normalization… │
│                     │                   │         │ add_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ global_average_poo… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">500</span>)       │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePool…</span> │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │  <span style="color: #00af00; text-decoration-color: #00af00">64,128</span> │ global_average_pool… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dropout_12          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dense[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)         │     <span style="color: #00af00; text-decoration-color: #00af00">258</span> │ dropout_12[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">93,130</span> (363.79 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">93,130</span> (363.79 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 17s 183ms/step - loss: 1.0039 - sparse_categorical_accuracy: 0.5180 - val_loss: 0.7024 - val_sparse_categorical_accuracy: 0.5908
Epoch 2/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.8639 - sparse_categorical_accuracy: 0.5625 - val_loss: 0.6370 - val_sparse_categorical_accuracy: 0.6241
Epoch 3/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.7701 - sparse_categorical_accuracy: 0.6118 - val_loss: 0.6042 - val_sparse_categorical_accuracy: 0.6602
Epoch 4/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.7522 - sparse_categorical_accuracy: 0.6167 - val_loss: 0.5794 - val_sparse_categorical_accuracy: 0.6782
Epoch 5/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.6845 - sparse_categorical_accuracy: 0.6606 - val_loss: 0.5609 - val_sparse_categorical_accuracy: 0.6893
Epoch 6/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.6760 - sparse_categorical_accuracy: 0.6653 - val_loss: 0.5520 - val_sparse_categorical_accuracy: 0.7046
Epoch 7/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.6589 - sparse_categorical_accuracy: 0.6558 - val_loss: 0.5390 - val_sparse_categorical_accuracy: 0.7129
Epoch 8/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.6416 - sparse_categorical_accuracy: 0.6675 - val_loss: 0.5299 - val_sparse_categorical_accuracy: 0.7171
Epoch 9/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.6270 - sparse_categorical_accuracy: 0.6861 - val_loss: 0.5202 - val_sparse_categorical_accuracy: 0.7295
Epoch 10/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.5995 - sparse_categorical_accuracy: 0.6969 - val_loss: 0.5135 - val_sparse_categorical_accuracy: 0.7323
Epoch 11/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.5846 - sparse_categorical_accuracy: 0.6927 - val_loss: 0.5084 - val_sparse_categorical_accuracy: 0.7420
Epoch 12/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.5837 - sparse_categorical_accuracy: 0.7163 - val_loss: 0.5042 - val_sparse_categorical_accuracy: 0.7420
Epoch 13/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.5407 - sparse_categorical_accuracy: 0.7323 - val_loss: 0.4984 - val_sparse_categorical_accuracy: 0.7462
Epoch 14/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.5302 - sparse_categorical_accuracy: 0.7446 - val_loss: 0.4958 - val_sparse_categorical_accuracy: 0.7462
Epoch 15/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.5041 - sparse_categorical_accuracy: 0.7459 - val_loss: 0.4905 - val_sparse_categorical_accuracy: 0.7503
Epoch 16/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.5122 - sparse_categorical_accuracy: 0.7506 - val_loss: 0.4842 - val_sparse_categorical_accuracy: 0.7642
Epoch 17/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.5042 - sparse_categorical_accuracy: 0.7565 - val_loss: 0.4824 - val_sparse_categorical_accuracy: 0.7656
Epoch 18/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4965 - sparse_categorical_accuracy: 0.7709 - val_loss: 0.4794 - val_sparse_categorical_accuracy: 0.7587
Epoch 19/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4860 - sparse_categorical_accuracy: 0.7649 - val_loss: 0.4733 - val_sparse_categorical_accuracy: 0.7614
Epoch 20/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.4797 - sparse_categorical_accuracy: 0.7716 - val_loss: 0.4700 - val_sparse_categorical_accuracy: 0.7642
Epoch 21/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.4946 - sparse_categorical_accuracy: 0.7638 - val_loss: 0.4668 - val_sparse_categorical_accuracy: 0.7670
Epoch 22/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4443 - sparse_categorical_accuracy: 0.7949 - val_loss: 0.4640 - val_sparse_categorical_accuracy: 0.7670
Epoch 23/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4495 - sparse_categorical_accuracy: 0.7897 - val_loss: 0.4597 - val_sparse_categorical_accuracy: 0.7739
Epoch 24/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4284 - sparse_categorical_accuracy: 0.8085 - val_loss: 0.4572 - val_sparse_categorical_accuracy: 0.7739
Epoch 25/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4353 - sparse_categorical_accuracy: 0.8060 - val_loss: 0.4548 - val_sparse_categorical_accuracy: 0.7795
Epoch 26/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4332 - sparse_categorical_accuracy: 0.8024 - val_loss: 0.4531 - val_sparse_categorical_accuracy: 0.7781
Epoch 27/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4399 - sparse_categorical_accuracy: 0.7992 - val_loss: 0.4462 - val_sparse_categorical_accuracy: 0.7864
Epoch 28/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4143 - sparse_categorical_accuracy: 0.8098 - val_loss: 0.4433 - val_sparse_categorical_accuracy: 0.7850
Epoch 29/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3950 - sparse_categorical_accuracy: 0.8373 - val_loss: 0.4421 - val_sparse_categorical_accuracy: 0.7850
Epoch 30/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.4050 - sparse_categorical_accuracy: 0.8186 - val_loss: 0.4392 - val_sparse_categorical_accuracy: 0.7878
Epoch 31/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.4152 - sparse_categorical_accuracy: 0.8162 - val_loss: 0.4361 - val_sparse_categorical_accuracy: 0.7947
Epoch 32/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3870 - sparse_categorical_accuracy: 0.8290 - val_loss: 0.4335 - val_sparse_categorical_accuracy: 0.7961
Epoch 33/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3966 - sparse_categorical_accuracy: 0.8239 - val_loss: 0.4295 - val_sparse_categorical_accuracy: 0.7961
Epoch 34/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3797 - sparse_categorical_accuracy: 0.8320 - val_loss: 0.4252 - val_sparse_categorical_accuracy: 0.8031
Epoch 35/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3798 - sparse_categorical_accuracy: 0.8336 - val_loss: 0.4222 - val_sparse_categorical_accuracy: 0.8003
Epoch 36/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3652 - sparse_categorical_accuracy: 0.8437 - val_loss: 0.4217 - val_sparse_categorical_accuracy: 0.8044
Epoch 37/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3590 - sparse_categorical_accuracy: 0.8394 - val_loss: 0.4203 - val_sparse_categorical_accuracy: 0.8072
Epoch 38/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3457 - sparse_categorical_accuracy: 0.8562 - val_loss: 0.4182 - val_sparse_categorical_accuracy: 0.8100
Epoch 39/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3668 - sparse_categorical_accuracy: 0.8379 - val_loss: 0.4147 - val_sparse_categorical_accuracy: 0.8072
Epoch 40/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3382 - sparse_categorical_accuracy: 0.8612 - val_loss: 0.4116 - val_sparse_categorical_accuracy: 0.8128
Epoch 41/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.3454 - sparse_categorical_accuracy: 0.8525 - val_loss: 0.4076 - val_sparse_categorical_accuracy: 0.8155
Epoch 42/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.3359 - sparse_categorical_accuracy: 0.8672 - val_loss: 0.4075 - val_sparse_categorical_accuracy: 0.8100
Epoch 43/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3420 - sparse_categorical_accuracy: 0.8538 - val_loss: 0.4033 - val_sparse_categorical_accuracy: 0.8197
Epoch 44/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3325 - sparse_categorical_accuracy: 0.8642 - val_loss: 0.4010 - val_sparse_categorical_accuracy: 0.8197
Epoch 45/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3201 - sparse_categorical_accuracy: 0.8715 - val_loss: 0.3993 - val_sparse_categorical_accuracy: 0.8211
Epoch 46/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.3342 - sparse_categorical_accuracy: 0.8597 - val_loss: 0.3966 - val_sparse_categorical_accuracy: 0.8294
Epoch 47/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.3171 - sparse_categorical_accuracy: 0.8714 - val_loss: 0.3955 - val_sparse_categorical_accuracy: 0.8280
Epoch 48/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.3213 - sparse_categorical_accuracy: 0.8698 - val_loss: 0.3919 - val_sparse_categorical_accuracy: 0.8294
Epoch 49/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.3063 - sparse_categorical_accuracy: 0.8822 - val_loss: 0.3907 - val_sparse_categorical_accuracy: 0.8322
Epoch 50/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2966 - sparse_categorical_accuracy: 0.8826 - val_loss: 0.3888 - val_sparse_categorical_accuracy: 0.8322
Epoch 51/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2946 - sparse_categorical_accuracy: 0.8844 - val_loss: 0.3885 - val_sparse_categorical_accuracy: 0.8308
Epoch 52/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.2930 - sparse_categorical_accuracy: 0.8948 - val_loss: 0.3865 - val_sparse_categorical_accuracy: 0.8322
Epoch 53/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2715 - sparse_categorical_accuracy: 0.9141 - val_loss: 0.3835 - val_sparse_categorical_accuracy: 0.8280
Epoch 54/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2960 - sparse_categorical_accuracy: 0.8848 - val_loss: 0.3806 - val_sparse_categorical_accuracy: 0.8252
Epoch 55/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2813 - sparse_categorical_accuracy: 0.8989 - val_loss: 0.3808 - val_sparse_categorical_accuracy: 0.8239
Epoch 56/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2708 - sparse_categorical_accuracy: 0.9076 - val_loss: 0.3784 - val_sparse_categorical_accuracy: 0.8363
Epoch 57/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2895 - sparse_categorical_accuracy: 0.8882 - val_loss: 0.3786 - val_sparse_categorical_accuracy: 0.8336
Epoch 58/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2905 - sparse_categorical_accuracy: 0.8810 - val_loss: 0.3780 - val_sparse_categorical_accuracy: 0.8363
Epoch 59/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2732 - sparse_categorical_accuracy: 0.9023 - val_loss: 0.3738 - val_sparse_categorical_accuracy: 0.8419
Epoch 60/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2698 - sparse_categorical_accuracy: 0.8962 - val_loss: 0.3733 - val_sparse_categorical_accuracy: 0.8308
Epoch 61/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.2741 - sparse_categorical_accuracy: 0.9025 - val_loss: 0.3724 - val_sparse_categorical_accuracy: 0.8391
Epoch 62/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 128ms/step - loss: 0.2713 - sparse_categorical_accuracy: 0.8973 - val_loss: 0.3698 - val_sparse_categorical_accuracy: 0.8308
Epoch 63/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.2682 - sparse_categorical_accuracy: 0.9004 - val_loss: 0.3681 - val_sparse_categorical_accuracy: 0.8363
Epoch 64/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2673 - sparse_categorical_accuracy: 0.9006 - val_loss: 0.3692 - val_sparse_categorical_accuracy: 0.8377
Epoch 65/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2585 - sparse_categorical_accuracy: 0.9056 - val_loss: 0.3684 - val_sparse_categorical_accuracy: 0.8322
Epoch 66/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2696 - sparse_categorical_accuracy: 0.8958 - val_loss: 0.3654 - val_sparse_categorical_accuracy: 0.8336
Epoch 67/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2489 - sparse_categorical_accuracy: 0.9182 - val_loss: 0.3630 - val_sparse_categorical_accuracy: 0.8405
Epoch 68/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2475 - sparse_categorical_accuracy: 0.9121 - val_loss: 0.3626 - val_sparse_categorical_accuracy: 0.8433
Epoch 69/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2398 - sparse_categorical_accuracy: 0.9195 - val_loss: 0.3607 - val_sparse_categorical_accuracy: 0.8433
Epoch 70/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2379 - sparse_categorical_accuracy: 0.9138 - val_loss: 0.3598 - val_sparse_categorical_accuracy: 0.8474
Epoch 71/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2343 - sparse_categorical_accuracy: 0.9162 - val_loss: 0.3568 - val_sparse_categorical_accuracy: 0.8447
Epoch 72/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2497 - sparse_categorical_accuracy: 0.9104 - val_loss: 0.3554 - val_sparse_categorical_accuracy: 0.8419
Epoch 73/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.2399 - sparse_categorical_accuracy: 0.9070 - val_loss: 0.3552 - val_sparse_categorical_accuracy: 0.8433
Epoch 74/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2300 - sparse_categorical_accuracy: 0.9190 - val_loss: 0.3572 - val_sparse_categorical_accuracy: 0.8419
Epoch 75/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2370 - sparse_categorical_accuracy: 0.9109 - val_loss: 0.3523 - val_sparse_categorical_accuracy: 0.8419
Epoch 76/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2324 - sparse_categorical_accuracy: 0.9172 - val_loss: 0.3512 - val_sparse_categorical_accuracy: 0.8391
Epoch 77/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2262 - sparse_categorical_accuracy: 0.9210 - val_loss: 0.3488 - val_sparse_categorical_accuracy: 0.8391
Epoch 78/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2262 - sparse_categorical_accuracy: 0.9175 - val_loss: 0.3495 - val_sparse_categorical_accuracy: 0.8419
Epoch 79/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.2226 - sparse_categorical_accuracy: 0.9270 - val_loss: 0.3487 - val_sparse_categorical_accuracy: 0.8433
Epoch 80/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2181 - sparse_categorical_accuracy: 0.9247 - val_loss: 0.3501 - val_sparse_categorical_accuracy: 0.8474
Epoch 81/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2220 - sparse_categorical_accuracy: 0.9181 - val_loss: 0.3479 - val_sparse_categorical_accuracy: 0.8460
Epoch 82/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2114 - sparse_categorical_accuracy: 0.9254 - val_loss: 0.3464 - val_sparse_categorical_accuracy: 0.8460
Epoch 83/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2148 - sparse_categorical_accuracy: 0.9196 - val_loss: 0.3467 - val_sparse_categorical_accuracy: 0.8460
Epoch 84/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.2262 - sparse_categorical_accuracy: 0.9181 - val_loss: 0.3446 - val_sparse_categorical_accuracy: 0.8474
Epoch 85/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2121 - sparse_categorical_accuracy: 0.9205 - val_loss: 0.3452 - val_sparse_categorical_accuracy: 0.8460
Epoch 86/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.2057 - sparse_categorical_accuracy: 0.9238 - val_loss: 0.3460 - val_sparse_categorical_accuracy: 0.8350
Epoch 87/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2081 - sparse_categorical_accuracy: 0.9342 - val_loss: 0.3455 - val_sparse_categorical_accuracy: 0.8488
Epoch 88/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2153 - sparse_categorical_accuracy: 0.9211 - val_loss: 0.3421 - val_sparse_categorical_accuracy: 0.8488
Epoch 89/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1977 - sparse_categorical_accuracy: 0.9366 - val_loss: 0.3413 - val_sparse_categorical_accuracy: 0.8474
Epoch 90/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1928 - sparse_categorical_accuracy: 0.9410 - val_loss: 0.3428 - val_sparse_categorical_accuracy: 0.8405
Epoch 91/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1968 - sparse_categorical_accuracy: 0.9327 - val_loss: 0.3411 - val_sparse_categorical_accuracy: 0.8474
Epoch 92/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1909 - sparse_categorical_accuracy: 0.9308 - val_loss: 0.3404 - val_sparse_categorical_accuracy: 0.8488
Epoch 93/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2067 - sparse_categorical_accuracy: 0.9285 - val_loss: 0.3371 - val_sparse_categorical_accuracy: 0.8488
Epoch 94/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1990 - sparse_categorical_accuracy: 0.9329 - val_loss: 0.3385 - val_sparse_categorical_accuracy: 0.8502
Epoch 95/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1986 - sparse_categorical_accuracy: 0.9267 - val_loss: 0.3368 - val_sparse_categorical_accuracy: 0.8433
Epoch 96/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.2069 - sparse_categorical_accuracy: 0.9235 - val_loss: 0.3346 - val_sparse_categorical_accuracy: 0.8502
Epoch 97/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1971 - sparse_categorical_accuracy: 0.9296 - val_loss: 0.3340 - val_sparse_categorical_accuracy: 0.8544
Epoch 98/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.2042 - sparse_categorical_accuracy: 0.9250 - val_loss: 0.3352 - val_sparse_categorical_accuracy: 0.8419
Epoch 99/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1998 - sparse_categorical_accuracy: 0.9271 - val_loss: 0.3334 - val_sparse_categorical_accuracy: 0.8474
Epoch 100/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1832 - sparse_categorical_accuracy: 0.9406 - val_loss: 0.3317 - val_sparse_categorical_accuracy: 0.8474
Epoch 101/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1917 - sparse_categorical_accuracy: 0.9340 - val_loss: 0.3343 - val_sparse_categorical_accuracy: 0.8433
Epoch 102/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1811 - sparse_categorical_accuracy: 0.9286 - val_loss: 0.3317 - val_sparse_categorical_accuracy: 0.8530
Epoch 103/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1733 - sparse_categorical_accuracy: 0.9396 - val_loss: 0.3340 - val_sparse_categorical_accuracy: 0.8460
Epoch 104/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1661 - sparse_categorical_accuracy: 0.9464 - val_loss: 0.3288 - val_sparse_categorical_accuracy: 0.8488
Epoch 105/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1806 - sparse_categorical_accuracy: 0.9390 - val_loss: 0.3296 - val_sparse_categorical_accuracy: 0.8516
Epoch 106/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1774 - sparse_categorical_accuracy: 0.9401 - val_loss: 0.3291 - val_sparse_categorical_accuracy: 0.8530
Epoch 107/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1689 - sparse_categorical_accuracy: 0.9463 - val_loss: 0.3290 - val_sparse_categorical_accuracy: 0.8488
Epoch 108/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1830 - sparse_categorical_accuracy: 0.9319 - val_loss: 0.3299 - val_sparse_categorical_accuracy: 0.8447
Epoch 109/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1757 - sparse_categorical_accuracy: 0.9304 - val_loss: 0.3315 - val_sparse_categorical_accuracy: 0.8488
Epoch 110/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1810 - sparse_categorical_accuracy: 0.9378 - val_loss: 0.3280 - val_sparse_categorical_accuracy: 0.8502
Epoch 111/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1628 - sparse_categorical_accuracy: 0.9522 - val_loss: 0.3276 - val_sparse_categorical_accuracy: 0.8474
Epoch 112/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1659 - sparse_categorical_accuracy: 0.9484 - val_loss: 0.3285 - val_sparse_categorical_accuracy: 0.8530
Epoch 113/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1814 - sparse_categorical_accuracy: 0.9364 - val_loss: 0.3281 - val_sparse_categorical_accuracy: 0.8474
Epoch 114/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1721 - sparse_categorical_accuracy: 0.9391 - val_loss: 0.3287 - val_sparse_categorical_accuracy: 0.8433
Epoch 115/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 127ms/step - loss: 0.1743 - sparse_categorical_accuracy: 0.9321 - val_loss: 0.3275 - val_sparse_categorical_accuracy: 0.8474
Epoch 116/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1677 - sparse_categorical_accuracy: 0.9415 - val_loss: 0.3297 - val_sparse_categorical_accuracy: 0.8391
Epoch 117/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1657 - sparse_categorical_accuracy: 0.9449 - val_loss: 0.3228 - val_sparse_categorical_accuracy: 0.8419
Epoch 118/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1787 - sparse_categorical_accuracy: 0.9316 - val_loss: 0.3230 - val_sparse_categorical_accuracy: 0.8447
Epoch 119/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1659 - sparse_categorical_accuracy: 0.9408 - val_loss: 0.3233 - val_sparse_categorical_accuracy: 0.8460
Epoch 120/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1615 - sparse_categorical_accuracy: 0.9385 - val_loss: 0.3235 - val_sparse_categorical_accuracy: 0.8460
Epoch 121/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1582 - sparse_categorical_accuracy: 0.9526 - val_loss: 0.3247 - val_sparse_categorical_accuracy: 0.8474
Epoch 122/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1577 - sparse_categorical_accuracy: 0.9497 - val_loss: 0.3263 - val_sparse_categorical_accuracy: 0.8474
Epoch 123/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1593 - sparse_categorical_accuracy: 0.9483 - val_loss: 0.3261 - val_sparse_categorical_accuracy: 0.8433
Epoch 124/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1570 - sparse_categorical_accuracy: 0.9442 - val_loss: 0.3277 - val_sparse_categorical_accuracy: 0.8419
Epoch 125/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1434 - sparse_categorical_accuracy: 0.9460 - val_loss: 0.3257 - val_sparse_categorical_accuracy: 0.8447
Epoch 126/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1589 - sparse_categorical_accuracy: 0.9414 - val_loss: 0.3237 - val_sparse_categorical_accuracy: 0.8447
Epoch 127/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1591 - sparse_categorical_accuracy: 0.9460 - val_loss: 0.3217 - val_sparse_categorical_accuracy: 0.8447
Epoch 128/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1530 - sparse_categorical_accuracy: 0.9450 - val_loss: 0.3203 - val_sparse_categorical_accuracy: 0.8474
Epoch 129/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1464 - sparse_categorical_accuracy: 0.9514 - val_loss: 0.3206 - val_sparse_categorical_accuracy: 0.8474
Epoch 130/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1437 - sparse_categorical_accuracy: 0.9526 - val_loss: 0.3231 - val_sparse_categorical_accuracy: 0.8447
Epoch 131/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1415 - sparse_categorical_accuracy: 0.9510 - val_loss: 0.3226 - val_sparse_categorical_accuracy: 0.8433
Epoch 132/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1539 - sparse_categorical_accuracy: 0.9505 - val_loss: 0.3261 - val_sparse_categorical_accuracy: 0.8405
Epoch 133/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1432 - sparse_categorical_accuracy: 0.9544 - val_loss: 0.3239 - val_sparse_categorical_accuracy: 0.8377
Epoch 134/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1368 - sparse_categorical_accuracy: 0.9567 - val_loss: 0.3200 - val_sparse_categorical_accuracy: 0.8474
Epoch 135/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1319 - sparse_categorical_accuracy: 0.9619 - val_loss: 0.3200 - val_sparse_categorical_accuracy: 0.8433
Epoch 136/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1479 - sparse_categorical_accuracy: 0.9494 - val_loss: 0.3201 - val_sparse_categorical_accuracy: 0.8502
Epoch 137/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1353 - sparse_categorical_accuracy: 0.9573 - val_loss: 0.3208 - val_sparse_categorical_accuracy: 0.8488
Epoch 138/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1349 - sparse_categorical_accuracy: 0.9584 - val_loss: 0.3213 - val_sparse_categorical_accuracy: 0.8474
Epoch 139/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1418 - sparse_categorical_accuracy: 0.9532 - val_loss: 0.3197 - val_sparse_categorical_accuracy: 0.8447
Epoch 140/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1402 - sparse_categorical_accuracy: 0.9534 - val_loss: 0.3204 - val_sparse_categorical_accuracy: 0.8488
Epoch 141/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1344 - sparse_categorical_accuracy: 0.9525 - val_loss: 0.3207 - val_sparse_categorical_accuracy: 0.8474
Epoch 142/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1448 - sparse_categorical_accuracy: 0.9494 - val_loss: 0.3192 - val_sparse_categorical_accuracy: 0.8488
Epoch 143/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1363 - sparse_categorical_accuracy: 0.9552 - val_loss: 0.3219 - val_sparse_categorical_accuracy: 0.8460
Epoch 144/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1380 - sparse_categorical_accuracy: 0.9540 - val_loss: 0.3219 - val_sparse_categorical_accuracy: 0.8474
Epoch 145/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1472 - sparse_categorical_accuracy: 0.9468 - val_loss: 0.3219 - val_sparse_categorical_accuracy: 0.8474
Epoch 146/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1402 - sparse_categorical_accuracy: 0.9622 - val_loss: 0.3217 - val_sparse_categorical_accuracy: 0.8502
Epoch 147/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1236 - sparse_categorical_accuracy: 0.9617 - val_loss: 0.3194 - val_sparse_categorical_accuracy: 0.8460
Epoch 148/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1183 - sparse_categorical_accuracy: 0.9683 - val_loss: 0.3193 - val_sparse_categorical_accuracy: 0.8488
Epoch 149/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 126ms/step - loss: 0.1189 - sparse_categorical_accuracy: 0.9618 - val_loss: 0.3237 - val_sparse_categorical_accuracy: 0.8488
Epoch 150/150
 45/45 ━━━━━━━━━━━━━━━━━━━━ 6s 125ms/step - loss: 0.1495 - sparse_categorical_accuracy: 0.9459 - val_loss: 0.3181 - val_sparse_categorical_accuracy: 0.8460
 42/42 ━━━━━━━━━━━━━━━━━━━━ 3s 44ms/step - loss: 0.3182 - sparse_categorical_accuracy: 0.8617

[0.3543623089790344, 0.843181848526001]

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
