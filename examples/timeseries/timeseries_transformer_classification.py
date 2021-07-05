"""
Title: Time Series Classification with a Self-Attention Transformer Model
Authors: [Theodoros Ntakouris](https://github.com/ntakouris)
Date created: 2021/06/25
Last modified: 2021/06/25
Description: This notebook demonstrates how to do timeseries forecasting using 
a transformer model.
"""


"""
##
This is the [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762), 
but for time series instead of natural language processing.

Over the past years, the Attention mechanism has become top of the leaderboards 
for NLP (sequential in nature) tasks. There has also beensignificant adoption
of the architecture for computer vision as well as time series. In general, 
experiments have shown that this architecturescales well with big datasets. 
Self attention might yield remarkable performance for a lot of tasks 
(not only time series), expecially with simplistic datasets and model
combinations topping off quickly at even 100% training or validation accuracy,
but there are several disadvantages of using attention.

Firstly, their complexity is quadratic with regards to the input length.
This means that the memory and compute requirements for long sequences renders
them unfit for training or deployment on resource contrained devices. Secondly,
for time series tasks where the inputs are of a few dimensions long, or even 
1-dimension long (univariate time series), there might be convergence problems.
Lastly, due to the large learning capacity of the transformer architecture,
smaller datasets might have better performance with a carefully tuned RNN 
architecture. 

Note on tensorflow version: make sure you are on `tensorflow>=2.4` to run this 
notebook. This is due to the `keras.layers.MultiHeadAttention` migrating from 
`tensorflow_addons` to keras/tensorflow.

## Dataset Load

We are going to use the same dataset and preprocessing as the 
[Time Series Classification from Scratch](
    https://keras.io/examples/timeseries/timeseries_classification_from_scratch
) keras example.
"""

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

"""
## Model Creation

The dataset is now ready and moving on, we are going to implement our model, 
which first encodes a tensor of shape `(batch size, sequence length, features)`,
where `sequence length` is the number of time steps and `features` is each input
time series. 

This is identical to the LSTM layer that keras provides -- and the good part is
that you can replace your classification RNN models with this one, because the
inputs are fully compatible!
"""

from tensorflow import keras
from tensorflow.keras import layers

"""
The MultiHeadAttention layer can perform
self-attention, by generating each of {query, key, value}
matrices, through the inputs.
This corresponds to implementing an attention mechanism by:
`Q = dense_q(x), K = dense_k(x), V = dense_v(x)` , where
each `dense_x()` is a different instance of `keras.layers.Dense`.

(The actual implementation is a much more optimized version.)

The Self-Attention mechanism is weak on it's own. We need to add a residual 
connection, (layer) normalization, dropout and projection layers for this to 
work. The following layer can also be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`, by 
(automatically) broadcasting the compute to the last feature dimension of the
input tensors.
"""


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


"""

The main part of our model is now complete. We can stack multiple of those 
`transformer_encoder` blocks and we can also proceed to add the final 
Multi-Layer Perceptron classification "head". Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model, down to a vector of features for each data point in the current 
batch. A common way to achieve this is to use some sort of pooling layer. For 
this example, a `GlobalAveragePooling1D` layer is sufficient.

Feel free to experiment with different pooling layers if you wish, 
encorporating domain expertise.

"""


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0
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


"""
## Training and Evaluation
"""

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

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)

"""
## Conclusions

In about 110-120 epochs (25s each on google colab), the model reaches a training
sparse categorical accuracy of ~0.95, validation accuracy of ~84 and a testing 
accuracy of ~85, without much hyperparameter tuning. And that is for a model 
with less than 100k parameters. Of course, parameter count and accuracy could be
improved by a hyperparameter search and a more sophisticated learning rate 
schedule, or a different optimizer, but the 
[Time Series Classification from Scratch](
    https://keras.io/examples/timeseries/timeseries_classification_from_scratch
) keras example which uses convolutional and batch normalization layers, 
contains a model that reaches > 90% test accuracy with just 25k parameters.

"""
