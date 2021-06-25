"""
Title: Time Series Classification with a Self-Attention Transformer Model
Authors: [Theodoros Ntakouris](https://github.com/ntakouris)
Date created: 2021/06/25
Last modified: 2021/06/25
Description: This notebook demonstrates how to do timeseries forecasting using a transformer model.
"""


"""
##
This is the [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762), but for time series instead of natural language processing.

Note on tensorflow version: make sure you are on `tensorflow>=2.4` to run this notebook. This is due to the `keras.layers.MultiHeadAttention` migrating from `tensorflow_addons` to keras/tensorflow.

## Dataset Load

We are going to use the same dataset and preprocessing as the [Time Series Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/) keras example.
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

The dataset is now ready and moving on, we are going to implement our model, which first encodes a tensor of shape `(batch size, sequence length, features)`, where `sequence length` is the number of time steps and `features` is each input time series. This is identical to the LSTM layer that keras provides.
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K

"""
Here, we make the MultiHeadAttention layer to perform
self-attention, by generating each of {query, key, value}
matrices, through a dense layer.
This corresponds to implementing an attention mechanism by:
`Q = dense_q(x), K = dense_k(x), V = dense_v(x)` , where
each `dense_x()` is a different instance of `keras.layers.Dense`.

(The actual implementation is a much more optimized version.)
"""

class SelfAttention(keras.Model):
    def __init__(self, head_size, num_heads=1, dropout=0, name='SelfAttention', **kwargs):
        super().__init__(name=name, **kwargs)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout, **kwargs)

    # input shape = output shape = (batch size, seq len, features)
    def call(self, inputs, **kwargs):
        x = inputs
        return self.attention(query=x, key=x, value=x, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

"""
The Self-Attention mechanism is weak on it's own. We need to add a residual connection, (layer) normalization,
dropout and projection layers for this to work. The following layer can also be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`, by (automatically) broadcasting the compute to the last feature dimension of the input tensors.
"""

class SelfAttentionBlock(keras.Model):
    def __init__(self,head_size, num_heads=1, ff_dim=None, dropout=0, name='SelfAttentionBlock',  **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = SelfAttention(head_size, num_heads, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(
            filters=ff_dim, kernel_size=1, activation='relu')

        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(
            filters=input_shape[-1], kernel_size=1)

    def call(self, inputs, training, **kwargs):
        x = self.attention_norm(inputs, **kwargs)
        x = self.attention(x, **kwargs)
        x = self.attention_dropout(x, training=training, **kwargs)
        res = x + inputs

        x = self.ff_norm(res, **kwargs)
        x = self.ff_conv1(x, **kwargs)
        x = self.ff_dropout(x, training=training, **kwargs)
        x = self.ff_conv2(x, **kwargs)
        x = self.ff_dropout(x, training=training, **kwargs)
        return x + res

    def compute_output_shape(self, input_shape):
        return input_shape

"""
Finally, we can wrap every layer up and add the ability to stack several of them together.
"""

class TransformerEncoder(keras.Model):
    def __init__(self, name='TransformerEncoder', num_heads=1, head_size=64, ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        if ff_dim is None: # sensible default
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [SelfAttentionBlock(
            num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

    def call(self, inputs, training, **kwargs):
        x = inputs # (batch size, sequence length, features)
        for attention_layer in self.attention_layers:
            x = attention_layer(x, training, **kwargs)

        return x # (batch size, sequence length, features)

"""
The main part of our model is now complete and we can now proceed to add the final Multi-Layer Perceptron classification "head".
Apart from a stack of `Dense` layers, we need to reduce the output tensor of the `TransformerEncoder` part of our model,
down to a vector of features for each data point in the current batch.
A common way to achieve this is to use some sort of pooling layer.
For this example, a `GlobalAveragePooling1D` layer is sufficient.
Feel free to experiment with different pooling layers if you wish, encorporating domain expertise.
"""

def build_model(input_shape, n_classes, head_size, num_heads=4, num_layers=2, dropout=0.2, dropout_mlps=0.5, mlp=[],ff_dim=None, activation='relu', optimizer='adamW'):
    model_input = keras.Input(shape=input_shape)
    x = model_input
    x = TransformerEncoder(num_heads=num_heads, head_size=head_size, ff_dim=None, num_layers=num_layers, dropout=dropout)(x)

    # if you use the default format 'channels_last', then the output dimensions
    # will be very small, of shape (seq len, 1)
    head = keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)

    for dim in mlp:
        head = keras.layers.Dense(dim, activation=activation)(head)
        head = keras.layers.Dropout(dropout_mlps)(head)

    model_output = keras.layers.Dense(n_classes, activation='softmax')(head)

    model = keras.Model(model_input, model_output)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

"""
## Training and Evaluation
"""

input_shape = x_train.shape[1:]

opt = keras.optimizers.Adam(learning_rate=1e-4)
model = build_model(input_shape, n_classes, num_heads=4, head_size=256, num_layers=4, mlp=[128], dropout=0.25, dropout_mlps=0.4, optimizer=opt)

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(x_train, y_train, validation_split=0.2, epochs=200, batch_size=64, callbacks=callbacks)

model.evaluate(x_test, y_test, verbose=1)

