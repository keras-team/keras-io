# Structured data classification with FeatureSpace

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2022/11/09<br>
**Last modified:** 2022/11/09<br>
**Description:** Classify tabular data in a few lines of code.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/structured_data_classification_with_feature_space.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/structured_data_classification_with_feature_space.py)



---
## Introduction

This example demonstrates how to do structured data classification
(also known as tabular data classification), starting from a raw
CSV file. Our data includes numerical features,
and integer categorical features, and string categorical features.
We will use the utility `keras.utils.FeatureSpace` to index,
preprocess, and encode our features.

The code is adapted from the example
[Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/).
While the previous example managed its own low-level feature preprocessing and
encoding with Keras preprocessing layers, in this example we
delegate everything to `FeatureSpace`, making the workflow
extremely quick and easy.

Note that this example should be run with TensorFlow 2.12 or higher.
Before the release of TensorFlow 2.12, you can use `tf-nightly`.

### The dataset

[Our dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) is provided by the
Cleveland Clinic Foundation for Heart Disease.
It's a CSV file with 303 rows. Each row contains information about a patient (a
**sample**), and each column describes an attribute of the patient (a **feature**). We
use the features to predict whether a patient has a heart disease
(**binary classification**).

Here's the description of each feature:

Column| Description| Feature Type
------------|--------------------|----------------------
Age | Age in years | Numerical
Sex | (1 = male; 0 = female) | Categorical
CP | Chest pain type (0, 1, 2, 3, 4) | Categorical
Trestbpd | Resting blood pressure (in mm Hg on admission) | Numerical
Chol | Serum cholesterol in mg/dl | Numerical
FBS | fasting blood sugar in 120 mg/dl (1 = true; 0 = false) | Categorical
RestECG | Resting electrocardiogram results (0, 1, 2) | Categorical
Thalach | Maximum heart rate achieved | Numerical
Exang | Exercise induced angina (1 = yes; 0 = no) | Categorical
Oldpeak | ST depression induced by exercise relative to rest | Numerical
Slope | Slope of the peak exercise ST segment | Numerical
CA | Number of major vessels (0-3) colored by fluoroscopy | Both numerical & categorical
Thal | 3 = normal; 6 = fixed defect; 7 = reversible defect | Categorical
Target | Diagnosis of heart disease (1 = true; 0 = false) | Target

---
## Setup


```python
import tensorflow as tf
import pandas as pd
from tensorflow import keras
```

---
## Preparing the data

Let's download the data and load it into a Pandas dataframe:


```python
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
```

The dataset includes 303 samples with 14 columns per sample
(13 features, plus the target label):


```python
print(dataframe.shape)
```

<div class="k-default-codeblock">
```
(303, 14)

```
</div>
Here's a preview of a few samples:


```python
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<div class="k-default-codeblock">
```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0</td>
      <td>fixed</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3</td>
      <td>normal</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1</td>
      <td>4</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2</td>
      <td>reversible</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>3</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0</td>
      <td>normal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0</td>
      <td>2</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0</td>
      <td>normal</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The last column, "target", indicates whether the patient
has a heart disease (1) or not (0).

Let's split the data into a training and validation set:


```python
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)
```

<div class="k-default-codeblock">
```
Using 242 samples for training and 61 for validation

```
</div>
Let's generate `tf.data.Dataset` objects for each dataframe:


```python

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)
```

Each `Dataset` yields a tuple `(input, target)` where `input` is a dictionary of features
and `target` is the value `0` or `1`:


```python
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)
```

<div class="k-default-codeblock">
```
Input: {'age': <tf.Tensor: shape=(), dtype=int64, numpy=47>, 'sex': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'cp': <tf.Tensor: shape=(), dtype=int64, numpy=3>, 'trestbps': <tf.Tensor: shape=(), dtype=int64, numpy=130>, 'chol': <tf.Tensor: shape=(), dtype=int64, numpy=253>, 'fbs': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'restecg': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'thalach': <tf.Tensor: shape=(), dtype=int64, numpy=179>, 'exang': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'oldpeak': <tf.Tensor: shape=(), dtype=float64, numpy=0.0>, 'slope': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'ca': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'thal': <tf.Tensor: shape=(), dtype=string, numpy=b'normal'>}
Target: tf.Tensor(0, shape=(), dtype=int64)

```
</div>
Let's batch the datasets:


```python
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
```

---
## Configuring a `FeatureSpace`

To configure how each feature should be preprocessed,
we instantiate a `keras.utils.FeatureSpace`, and we
pass to it a dictionary that maps the name of our features
to a string that describes the feature type.

We have a few "integer categorical" features such as `"FBS"`,
one "string categorical" feature (`"thal"`),
and a few numerical features, which we'd like to normalize
-- except `"age"`, which we'd like to discretize into
a number of bins.

We also use the `crosses` argument
to capture *feature interactions* for some categorical
features, that is to say, create additional features
that represent value co-occurrences for these categorical features.
You can compute feature crosses like this for arbitrary sets of
categorical features -- not just tuples of two features.
Because the resulting co-occurences are hashed
into a fixed-sized vector, you don't need to worry about whether
the co-occurence space is too large.


```python
from keras.utils import FeatureSpace

feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": "integer_categorical",
        "cp": "integer_categorical",
        "fbs": "integer_categorical",
        "restecg": "integer_categorical",
        "exang": "integer_categorical",
        "ca": "integer_categorical",
        # Categorical feature encoded as string
        "thal": "string_categorical",
        # Numerical features to discretize
        "age": "float_discretized",
        # Numerical features to normalize
        "trestbps": "float_normalized",
        "chol": "float_normalized",
        "thalach": "float_normalized",
        "oldpeak": "float_normalized",
        "slope": "float_normalized",
    },
    # We create additional features by hashing
    # value co-occurrences for the
    # following groups of categorical features.
    crosses=[("sex", "age"), ("thal", "ca")],
    # The hashing space for these co-occurrences
    # wil be 32-dimensional.
    crossing_dim=32,
    # Our utility will one-hot encode all categorical
    # features and concat all features into a single
    # vector (one vector per sample).
    output_mode="concat",
)
```

---
## Further customizing a `FeatureSpace`

Specifying the feature type via a string name is quick and easy,
but sometimes you may want to further configure the preprocessing
of each feature. For instance, in our case, our categorical
features don't have a large set of possible values -- it's only
a handful of values per feature (e.g. `1` and `0` for the feature `"FBS"`),
and all possible values are represented in the training set.
As a result, we don't need to reserve an index to represent "out of vocabulary" values
for these features -- which would have been the default behavior.
Below, we just specify `num_oov_indices=0` in each of these features
to tell the feature preprocessor to skip "out of vocabulary" indexing.

Other customizations you have access to include specifying the number of
bins for discretizing features of type `"float_discretized"`,
or the dimensionality of the hashing space for feature crossing.


```python
feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
        "cp": FeatureSpace.integer_categorical(num_oov_indices=0),
        "fbs": FeatureSpace.integer_categorical(num_oov_indices=0),
        "restecg": FeatureSpace.integer_categorical(num_oov_indices=0),
        "exang": FeatureSpace.integer_categorical(num_oov_indices=0),
        "ca": FeatureSpace.integer_categorical(num_oov_indices=0),
        # Categorical feature encoded as string
        "thal": FeatureSpace.string_categorical(num_oov_indices=0),
        # Numerical features to normalize
        "age": FeatureSpace.float_discretized(num_bins=30),
        # Numerical features to normalize
        "trestbps": FeatureSpace.float_normalized(),
        "chol": FeatureSpace.float_normalized(),
        "thalach": FeatureSpace.float_normalized(),
        "oldpeak": FeatureSpace.float_normalized(),
        "slope": FeatureSpace.float_normalized(),
    },
    # Specify feature cross with a custom crossing dim.
    crosses=[
        FeatureSpace.cross(feature_names=("sex", "age"), crossing_dim=64),
        FeatureSpace.cross(
            feature_names=("thal", "ca"),
            crossing_dim=16,
        ),
    ],
    output_mode="concat",
)
```

---
## Adapt the `FeatureSpace` to the training data

Before we start using the `FeatureSpace` to build a model, we have
to adapt it to the training data. During `adapt()`, the `FeatureSpace` will:

- Index the set of possible values for categorical features.
- Compute the mean and variance for numerical features to normalize.
- Compute the value boundaries for the different bins for numerical features to discretize.

Note that `adapt()` should be called on a `tf.data.Dataset` which yields dicts
of feature values -- no labels.


```python
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)
```

At this point, the `FeatureSpace` can be called on a dict of raw feature values, and will return a
single concatenate vector for each sample, combining encoded features and feature crosses.


```python
for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)
```

<div class="k-default-codeblock">
```
preprocessed_x.shape: (32, 138)
preprocessed_x.dtype: <dtype: 'float32'>

```
</div>
---
## Two ways to manage preprocessing: as part of the `tf.data` pipeline, or in the model itself

There are two ways in which you can leverage your `FeatureSpace`:

### Asynchronous preprocessing in `tf.data`

You can make it part of your data pipeline, before the model. This enables asynchronous parallel
preprocessing of the data on CPU before it hits the model. Do this if you're training on GPU or TPU,
or if you want to speed up preprocessing. Usually, this is always the right thing to do during training.

### Synchronous preprocessing in the model

You can make it part of your model. This means that the model will expect dicts of raw feature
values, and the preprocessing batch will be done synchronously (in a blocking manner) before the
rest of the forward pass. Do this if you want to have an end-to-end model that can process
raw feature values -- but keep in mind that your model will only be able to run on CPU,
since most types of feature preprocessing (e.g. string preprocessing) are not GPU or TPU compatible.

Do not do this on GPU / TPU or in performance-sensitive settings. In general, you want to do in-model
preprocessing when you do inference on CPU.

In our case, we will apply the `FeatureSpace` in the tf.data pipeline during training, but we will
do inference with an end-to-end model that includes the `FeatureSpace`.

Let's create a training and validation dataset of preprocessed batches:


```python
preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)
```

---
## Build a model

Time to build a model -- or rather two models:

- A training model that expects preprocessed features (one sample = one vector)
- An inference model that expects raw features (one sample = dict of raw feature values)


```python
dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(32, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation="sigmoid")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
```

---
## Train the model

Let's train our model for 50 epochs. Note that feature preprocessing is happening
as part of the tf.data pipeline, not as part of the model.


```python
training_model.fit(
    preprocessed_train_ds, epochs=20, validation_data=preprocessed_val_ds, verbose=2
)
```

<div class="k-default-codeblock">
```
Epoch 1/20
8/8 - 5s - loss: 0.6999 - accuracy: 0.5413 - val_loss: 0.6163 - val_accuracy: 0.6885 - 5s/epoch - 638ms/step
Epoch 2/20
8/8 - 0s - loss: 0.6044 - accuracy: 0.6364 - val_loss: 0.5635 - val_accuracy: 0.7869 - 268ms/epoch - 34ms/step
Epoch 3/20
8/8 - 0s - loss: 0.5764 - accuracy: 0.6570 - val_loss: 0.5214 - val_accuracy: 0.8197 - 267ms/epoch - 33ms/step
Epoch 4/20
8/8 - 0s - loss: 0.5281 - accuracy: 0.7438 - val_loss: 0.4867 - val_accuracy: 0.8033 - 269ms/epoch - 34ms/step
Epoch 5/20
8/8 - 0s - loss: 0.4858 - accuracy: 0.7727 - val_loss: 0.4587 - val_accuracy: 0.7705 - 268ms/epoch - 34ms/step
Epoch 6/20
8/8 - 0s - loss: 0.4710 - accuracy: 0.7438 - val_loss: 0.4364 - val_accuracy: 0.7705 - 271ms/epoch - 34ms/step
Epoch 7/20
8/8 - 0s - loss: 0.4245 - accuracy: 0.8099 - val_loss: 0.4181 - val_accuracy: 0.7705 - 273ms/epoch - 34ms/step
Epoch 8/20
8/8 - 0s - loss: 0.4261 - accuracy: 0.7645 - val_loss: 0.4043 - val_accuracy: 0.7869 - 269ms/epoch - 34ms/step
Epoch 9/20
8/8 - 0s - loss: 0.4000 - accuracy: 0.7893 - val_loss: 0.3943 - val_accuracy: 0.7869 - 274ms/epoch - 34ms/step
Epoch 10/20
8/8 - 0s - loss: 0.3788 - accuracy: 0.7893 - val_loss: 0.3866 - val_accuracy: 0.7869 - 271ms/epoch - 34ms/step
Epoch 11/20
8/8 - 0s - loss: 0.3612 - accuracy: 0.8347 - val_loss: 0.3809 - val_accuracy: 0.8033 - 268ms/epoch - 33ms/step
Epoch 12/20
8/8 - 0s - loss: 0.3691 - accuracy: 0.8058 - val_loss: 0.3761 - val_accuracy: 0.8033 - 271ms/epoch - 34ms/step
Epoch 13/20
8/8 - 0s - loss: 0.3473 - accuracy: 0.8471 - val_loss: 0.3719 - val_accuracy: 0.7869 - 269ms/epoch - 34ms/step
Epoch 14/20
8/8 - 0s - loss: 0.3590 - accuracy: 0.8264 - val_loss: 0.3682 - val_accuracy: 0.7869 - 275ms/epoch - 34ms/step
Epoch 15/20
8/8 - 0s - loss: 0.3290 - accuracy: 0.8388 - val_loss: 0.3656 - val_accuracy: 0.8033 - 270ms/epoch - 34ms/step
Epoch 16/20
8/8 - 0s - loss: 0.3127 - accuracy: 0.8471 - val_loss: 0.3636 - val_accuracy: 0.8033 - 273ms/epoch - 34ms/step
Epoch 17/20
8/8 - 0s - loss: 0.2991 - accuracy: 0.8843 - val_loss: 0.3623 - val_accuracy: 0.8033 - 275ms/epoch - 34ms/step
Epoch 18/20
8/8 - 0s - loss: 0.3258 - accuracy: 0.8678 - val_loss: 0.3613 - val_accuracy: 0.8033 - 272ms/epoch - 34ms/step
Epoch 19/20
8/8 - 0s - loss: 0.2835 - accuracy: 0.8512 - val_loss: 0.3610 - val_accuracy: 0.8033 - 271ms/epoch - 34ms/step
Epoch 20/20
8/8 - 0s - loss: 0.2700 - accuracy: 0.9050 - val_loss: 0.3613 - val_accuracy: 0.8033 - 269ms/epoch - 34ms/step

<keras.callbacks.History at 0x7f6850106290>

```
</div>
We quickly get to 80% validation accuracy.

---
## Inference on new data with the end-to-end model

Now, we can use our inference model (which includes the `FeatureSpace`)
to make predictions based on dicts of raw features values, as follows:


```python
sample = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = inference_model.predict(input_dict)

print(
    f"This particular patient had a {100 * predictions[0][0]:.2f}% probability "
    "of having a heart disease, as evaluated by our model."
)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 1s 504ms/step
This particular patient had a 56.32% probability of having a heart disease, as evaluated by our model.

```
</div>