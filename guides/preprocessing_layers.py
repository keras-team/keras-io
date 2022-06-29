"""
Title: Working with preprocessing layers
Authors: Francois Chollet, Mark Omernick
Date created: 2020/07/25
Last modified: 2021/04/23
Description: Overview of how to leverage preprocessing layers to create end-to-end models.
"""
"""
## Keras preprocessing

The Keras preprocessing layers API allows developers to build Keras-native input
processing pipelines. These input processing pipelines can be used as independent
preprocessing code in non-Keras workflows, combined directly with Keras models, and
exported as part of a Keras SavedModel.

With Keras preprocessing layers, you can build and export models that are truly
end-to-end: models that accept raw images or raw structured data as input; models that
handle feature normalization or feature value indexing on their own.
"""

"""
## Available preprocessing

### Text preprocessing

- `tf.keras.layers.TextVectorization`: turns raw strings into an encoded
  representation that can be read by an `Embedding` layer or `Dense` layer.

### Numerical features preprocessing

- `tf.keras.layers.Normalization`: performs feature-wise normalize of
  input features.
- `tf.keras.layers.Discretization`: turns continuous numerical features
  into integer categorical features.

### Categorical features preprocessing

- `tf.keras.layers.CategoryEncoding`: turns integer categorical features
  into one-hot, multi-hot, or count dense representations.
- `tf.keras.layers.Hashing`: performs categorical feature hashing, also known as
  the "hashing trick".
- `tf.keras.layers.StringLookup`: turns string categorical values into an encoded
  representation that can be read by an `Embedding` layer or `Dense` layer.
- `tf.keras.layers.IntegerLookup`: turns integer categorical values into an
  encoded representation that can be read by an `Embedding` layer or `Dense`
  layer.


### Image preprocessing

These layers are for standardizing the inputs of an image model.

- `tf.keras.layers.Resizing`: resizes a batch of images to a target size.
- `tf.keras.layers.Rescaling`: rescales and offsets the values of a batch of
  image (e.g. go from inputs in the `[0, 255]` range to inputs in the `[0, 1]`
  range.
- `tf.keras.layers.CenterCrop`: returns a center crop of a batch of images.

### Image data augmentation

These layers apply random augmentation transforms to a batch of images. They
are only active during training.

- `tf.keras.layers.RandomCrop`
- `tf.keras.layers.RandomFlip`
- `tf.keras.layers.RandomTranslation`
- `tf.keras.layers.RandomRotation`
- `tf.keras.layers.RandomZoom`
- `tf.keras.layers.RandomHeight`
- `tf.keras.layers.RandomWidth`
- `tf.keras.layers.RandomContrast`

"""

"""
## The `adapt()` method

Some preprocessing layers have an internal state that can be computed based on
a sample of the training data. The list of stateful preprocessing layers is:

- `TextVectorization`: holds a mapping between string tokens and integer indices
- `StringLookup` and `IntegerLookup`: hold a mapping between input values and integer
indices.
- `Normalization`: holds the mean and standard deviation of the features.
- `Discretization`: holds information about value bucket boundaries.

Crucially, these layers are **non-trainable**. Their state is not set during training; it
must be set **before training**, either by initializing them from a precomputed constant,
or by "adapting" them on data.

You set the state of a preprocessing layer by exposing it to training data, via the
`adapt()` method:

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

data = np.array(
    [
        [0.1, 0.2, 0.3],
        [0.8, 0.9, 1.0],
        [1.5, 1.6, 1.7],
    ]
)
layer = layers.Normalization()
layer.adapt(data)
normalized_data = layer(data)

print("Features mean: %.2f" % (normalized_data.numpy().mean()))
print("Features std: %.2f" % (normalized_data.numpy().std()))

"""
The `adapt()` method takes either a Numpy array or a `tf.data.Dataset` object. In the
case of `StringLookup` and `TextVectorization`, you can also pass a list of strings:
"""

data = [
    "ξεῖν᾽, ἦ τοι μὲν ὄνειροι ἀμήχανοι ἀκριτόμυθοι",
    "γίγνοντ᾽, οὐδέ τι πάντα τελείεται ἀνθρώποισι.",
    "δοιαὶ γάρ τε πύλαι ἀμενηνῶν εἰσὶν ὀνείρων:",
    "αἱ μὲν γὰρ κεράεσσι τετεύχαται, αἱ δ᾽ ἐλέφαντι:",
    "τῶν οἳ μέν κ᾽ ἔλθωσι διὰ πριστοῦ ἐλέφαντος,",
    "οἵ ῥ᾽ ἐλεφαίρονται, ἔπε᾽ ἀκράαντα φέροντες:",
    "οἱ δὲ διὰ ξεστῶν κεράων ἔλθωσι θύραζε,",
    "οἵ ῥ᾽ ἔτυμα κραίνουσι, βροτῶν ὅτε κέν τις ἴδηται.",
]
layer = layers.TextVectorization()
layer.adapt(data)
vectorized_text = layer(data)
print(vectorized_text)

"""
In addition, adaptable layers always expose an option to directly set state via
constructor arguments or weight assignment. If the intended state values are known at
layer construction time, or are calculated outside of the `adapt()` call, they can be set
without relying on the layer's internal computation. For instance, if external vocabulary
files for the `TextVectorization`, `StringLookup`, or `IntegerLookup` layers already
exist, those can be loaded directly into the lookup tables by passing a path to the
vocabulary file in the layer's constructor arguments.

Here's an example where you instantiate a `StringLookup` layer with precomputed vocabulary:
"""

vocab = ["a", "b", "c", "d"]
data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
layer = layers.StringLookup(vocabulary=vocab)
vectorized_data = layer(data)
print(vectorized_data)

"""
## Preprocessing data before the model or inside the model

There are two ways you could be using preprocessing layers:

**Option 1:** Make them part of the model, like this:

```python
inputs = keras.Input(shape=input_shape)
x = preprocessing_layer(inputs)
outputs = rest_of_the_model(x)
model = keras.Model(inputs, outputs)
```

With this option, preprocessing will happen on device, synchronously with the rest of the
model execution, meaning that it will benefit from GPU acceleration.
If you're training on a GPU, this is the best option for the `Normalization` layer, and for
all image preprocessing and data augmentation layers.

**Option 2:** apply it to your `tf.data.Dataset`, so as to obtain a dataset that yields
batches of preprocessed data, like this:

```python
dataset = dataset.map(lambda x, y: (preprocessing_layer(x), y))
```

With this option, your preprocessing will happen on a CPU, asynchronously, and will be
buffered before going into the model.
In addition, if you call `dataset.prefetch(tf.data.AUTOTUNE)` on your dataset,
the preprocessing will happen efficiently in parallel with training:

```python
dataset = dataset.map(lambda x, y: (preprocessing_layer(x), y))
dataset = dataset.prefetch(tf.data.AUTOTUNE)
model.fit(dataset, ...)
```

This is the best option for `TextVectorization`, and all structured data preprocessing
layers. It can also be a good option if you're training on a CPU and you use image preprocessing
layers.

Note that the `TextVectorization` layer can only be executed on a CPU, as it is mostly a
dictionary lookup operation. Therefore, if you are training your model on a GPU or a TPU,
you should put the `TextVectorization` layer in the `tf.data` pipeline to get the best performance.

**When running on a TPU, you should always place preprocessing layers in the `tf.data` pipeline**
(with the exception of `Normalization` and `Rescaling`, which run fine on a TPU and are commonly
used as the first layer is an image model).
"""

"""
## Benefits of doing preprocessing inside the model at inference time

Even if you go with option 2, you may later want to export an inference-only end-to-end
model that will include the preprocessing layers. The key benefit to doing this is that
**it makes your model portable** and it **helps reduce the
[training/serving skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew)**.

When all data preprocessing is part of the model, other people can load and use your
model without having to be aware of how each feature is expected to be encoded &
normalized. Your inference model will be able to process raw images or raw structured
data, and will not require users of the model to be aware of the details of e.g. the
tokenization scheme used for text, the indexing scheme used for categorical features,
whether image pixel values are normalized to `[-1, +1]` or to `[0, 1]`, etc. This is
especially powerful if you're exporting
your model to another runtime, such as TensorFlow.js: you won't have to
reimplement your preprocessing pipeline in JavaScript.

If you initially put your preprocessing layers in your `tf.data` pipeline,
you can export an inference model that packages the preprocessing.
Simply instantiate a new model that chains
your preprocessing layers and your training model:

```python
inputs = keras.Input(shape=input_shape)
x = preprocessing_layer(inputs)
outputs = training_model(x)
inference_model = keras.Model(inputs, outputs)
```
"""

"""
## Preprocessing during multi-worker training

Preprocessing layers are compatible with the
[tf.distribute](https://www.tensorflow.org/api_docs/python/tf/distribute) API
for running training across multiple machines.

In general, preprocessing layers should be placed inside a `tf.distribute.Strategy.scope()`
and called either inside or before the model as discussed above.

```python
with strategy.scope():
    inputs = keras.Input(shape=input_shape)
    preprocessing_layer = tf.keras.layers.Hashing(10)
    dense_layer = tf.keras.layers.Dense(16)
```

For more details, refer to the _Data preprocessing_ section
of the [Distributed input](https://www.tensorflow.org/tutorials/distribute/input)
tutorial.
"""

"""
## Quick recipes

### Image data augmentation

Note that image data augmentation layers are only active during training (similarly to
the `Dropout` layer).
"""

from tensorflow import keras
from tensorflow.keras import layers

# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Load some data
(x_train, y_train), _ = keras.datasets.cifar10.load_data()
input_shape = x_train.shape[1:]
classes = 10

# Create a tf.data pipeline of augmented images (and their labels)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(16).map(lambda x, y: (data_augmentation(x), y))


# Create a model and train it on the augmented image data
inputs = keras.Input(shape=input_shape)
x = layers.Rescaling(1.0 / 255)(inputs)  # Rescale inputs
outputs = keras.applications.ResNet50(  # Add the rest of the model
    weights=None, input_shape=input_shape, classes=classes
)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
model.fit(train_dataset, steps_per_epoch=5)

"""
You can see a similar setup in action in the example
[image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/).
"""

"""
### Normalizing numerical features
"""

# Load some data
(x_train, y_train), _ = keras.datasets.cifar10.load_data()
x_train = x_train.reshape((len(x_train), -1))
input_shape = x_train.shape[1:]
classes = 10

# Create a Normalization layer and set its internal state using the training data
normalizer = layers.Normalization()
normalizer.adapt(x_train)

# Create a model that include the normalization layer
inputs = keras.Input(shape=input_shape)
x = normalizer(inputs)
outputs = layers.Dense(classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# Train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train)

"""
### Encoding string categorical features via one-hot encoding
"""

# Define some toy data
data = tf.constant([["a"], ["b"], ["c"], ["b"], ["c"], ["a"]])

# Use StringLookup to build an index of the feature values and encode output.
lookup = layers.StringLookup(output_mode="one_hot")
lookup.adapt(data)

# Convert new test data (which includes unknown feature values)
test_data = tf.constant([["a"], ["b"], ["c"], ["d"], ["e"], [""]])
encoded_data = lookup(test_data)
print(encoded_data)

"""
Note that, here, index 0 is reserved for out-of-vocabulary values
(values that were not seen during `adapt()`).

You can see the `StringLookup` in action in the
[Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/)
example.
"""

"""
### Encoding integer categorical features via one-hot encoding
"""

# Define some toy data
data = tf.constant([[10], [20], [20], [10], [30], [0]])

# Use IntegerLookup to build an index of the feature values and encode output.
lookup = layers.IntegerLookup(output_mode="one_hot")
lookup.adapt(data)

# Convert new test data (which includes unknown feature values)
test_data = tf.constant([[10], [10], [20], [50], [60], [0]])
encoded_data = lookup(test_data)
print(encoded_data)

"""
Note that index 0 is reserved for missing values (which you should specify as the value
0), and index 1 is reserved for out-of-vocabulary values (values that were not seen
during `adapt()`). You can configure this by using the `mask_token` and `oov_token`
constructor arguments  of `IntegerLookup`.

You can see the `IntegerLookup` in action in the example
[structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/).
"""

"""
### Applying the hashing trick to an integer categorical feature

If you have a categorical feature that can take many different values (on the order of
10e3 or higher), where each value only appears a few times in the data,
it becomes impractical and ineffective to index and one-hot encode the feature values.
Instead, it can be a good idea to apply the "hashing trick": hash the values to a vector
of fixed size. This keeps the size of the feature space manageable, and removes the need
for explicit indexing.
"""

# Sample data: 10,000 random integers with values between 0 and 100,000
data = np.random.randint(0, 100000, size=(10000, 1))

# Use the Hashing layer to hash the values to the range [0, 64]
hasher = layers.Hashing(num_bins=64, salt=1337)

# Use the CategoryEncoding layer to multi-hot encode the hashed values
encoder = layers.CategoryEncoding(num_tokens=64, output_mode="multi_hot")
encoded_data = encoder(hasher(data))
print(encoded_data.shape)

"""
### Encoding text as a sequence of token indices

This is how you should preprocess text to be passed to an `Embedding` layer.
"""

# Define some text data to adapt the layer
adapt_data = tf.constant(
    [
        "The Brain is wider than the Sky",
        "For put them side by side",
        "The one the other will contain",
        "With ease and You beside",
    ]
)

# Create a TextVectorization layer
text_vectorizer = layers.TextVectorization(output_mode="int")
# Index the vocabulary via `adapt()`
text_vectorizer.adapt(adapt_data)

# Try out the layer
print(
    "Encoded text:\n",
    text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
)

# Create a simple model
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(input_dim=text_vectorizer.vocabulary_size(), output_dim=16)(inputs)
x = layers.GRU(8)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# Create a labeled dataset (which includes unknown tokens)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
)

# Preprocess the string inputs, turning them into int sequences
train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
# Train the model on the int sequences
print("\nTraining model...")
model.compile(optimizer="rmsprop", loss="mse")
model.fit(train_dataset)

# For inference, you can export a model that accepts strings as input
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
outputs = model(x)
end_to_end_model = keras.Model(inputs, outputs)

# Call the end-to-end model on test data (which includes unknown tokens)
print("\nCalling end-to-end model on test string...")
test_data = tf.constant(["The one the other will absorb"])
test_output = end_to_end_model(test_data)
print("Model output:", test_output)

"""
You can see the `TextVectorization` layer in action, combined with an `Embedding` mode,
in the example
[text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch/).

Note that when training such a model, for best performance, you should always
use the `TextVectorization` layer as part of the input pipeline.
"""

"""
### Encoding text as a dense matrix of N-grams with multi-hot encoding

This is how you should preprocess text to be passed to a `Dense` layer.
"""

# Define some text data to adapt the layer
adapt_data = tf.constant(
    [
        "The Brain is wider than the Sky",
        "For put them side by side",
        "The one the other will contain",
        "With ease and You beside",
    ]
)
# Instantiate TextVectorization with "multi_hot" output_mode
# and ngrams=2 (index all bigrams)
text_vectorizer = layers.TextVectorization(output_mode="multi_hot", ngrams=2)
# Index the bigrams via `adapt()`
text_vectorizer.adapt(adapt_data)

# Try out the layer
print(
    "Encoded text:\n",
    text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
)

# Create a simple model
inputs = keras.Input(shape=(text_vectorizer.vocabulary_size(),))
outputs = layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)

# Create a labeled dataset (which includes unknown tokens)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
)

# Preprocess the string inputs, turning them into int sequences
train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
# Train the model on the int sequences
print("\nTraining model...")
model.compile(optimizer="rmsprop", loss="mse")
model.fit(train_dataset)

# For inference, you can export a model that accepts strings as input
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
outputs = model(x)
end_to_end_model = keras.Model(inputs, outputs)

# Call the end-to-end model on test data (which includes unknown tokens)
print("\nCalling end-to-end model on test string...")
test_data = tf.constant(["The one the other will absorb"])
test_output = end_to_end_model(test_data)
print("Model output:", test_output)

"""
### Encoding text as a dense matrix of N-grams with TF-IDF weighting

This is an alternative way of preprocessing text before passing it to a `Dense` layer.
"""

# Define some text data to adapt the layer
adapt_data = tf.constant(
    [
        "The Brain is wider than the Sky",
        "For put them side by side",
        "The one the other will contain",
        "With ease and You beside",
    ]
)
# Instantiate TextVectorization with "tf-idf" output_mode
# (multi-hot with TF-IDF weighting) and ngrams=2 (index all bigrams)
text_vectorizer = layers.TextVectorization(output_mode="tf-idf", ngrams=2)
# Index the bigrams and learn the TF-IDF weights via `adapt()`
text_vectorizer.adapt(adapt_data)

# Try out the layer
print(
    "Encoded text:\n",
    text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
)

# Create a simple model
inputs = keras.Input(shape=(text_vectorizer.vocabulary_size(),))
outputs = layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)

# Create a labeled dataset (which includes unknown tokens)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
)

# Preprocess the string inputs, turning them into int sequences
train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
# Train the model on the int sequences
print("\nTraining model...")
model.compile(optimizer="rmsprop", loss="mse")
model.fit(train_dataset)

# For inference, you can export a model that accepts strings as input
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
outputs = model(x)
end_to_end_model = keras.Model(inputs, outputs)

# Call the end-to-end model on test data (which includes unknown tokens)
print("\nCalling end-to-end model on test string...")
test_data = tf.constant(["The one the other will absorb"])
test_output = end_to_end_model(test_data)
print("Model output:", test_output)


"""
## Important gotchas

### Working with lookup layers with very large vocabularies

You may find yourself working with a very large vocabulary in a `TextVectorization`, a `StringLookup` layer,
or an `IntegerLookup` layer. Typically, a vocabulary larger than 500MB would be considered "very large".

In such a case, for best performance, you should avoid using `adapt()`.
Instead, pre-compute your vocabulary in advance
(you could use Apache Beam or TF Transform for this)
and store it in a file. Then load the vocabulary into the layer at construction
time by passing the file path as the `vocabulary` argument.


### Using lookup layers on a TPU pod or with `ParameterServerStrategy`.

There is an outstanding issue that causes performance to degrade when using
a `TextVectorization`, `StringLookup`, or `IntegerLookup` layer while
training on a TPU pod or on multiple machines via `ParameterServerStrategy`.
This is slated to be fixed in TensorFlow 2.7.

"""
