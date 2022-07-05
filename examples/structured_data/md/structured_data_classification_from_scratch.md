# Structured data classification from scratch

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/06/09<br>
**Last modified:** 2020/06/09<br>
**Description:** Binary classification of structured data including numerical and categorical features.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/structured_data_classification_from_scratch.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/structured_data_classification_from_scratch.py)



---
## Introduction

This example demonstrates how to do structured data classification, starting from a raw
CSV file. Our data includes both numerical and categorical features. We will use Keras
preprocessing layers to normalize the numerical features and vectorize the categorical
ones.

Note that this example should be run with TensorFlow 2.5 or higher.

### The dataset

[Our dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) is provided by the
Cleveland Clinic Foundation for Heart Disease.
It's a CSV file with 303 rows. Each row contains information about a patient (a
**sample**), and each column describes an attribute of the patient (a **feature**). We
use the features to predict whether a patient has a heart disease (**binary
classification**).

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
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
```

---
## Preparing the data

Let's download the data and load it into a Pandas dataframe:


```python
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)
```

The dataset includes 303 samples with 14 columns per sample (13 features, plus the target
label):


```python
dataframe.shape
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



The last column, "target", indicates whether the patient has a heart disease (1) or not
(0).

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
Input: {'age': <tf.Tensor: shape=(), dtype=int64, numpy=62>, 'sex': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'cp': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 'trestbps': <tf.Tensor: shape=(), dtype=int64, numpy=128>, 'chol': <tf.Tensor: shape=(), dtype=int64, numpy=208>, 'fbs': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'restecg': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 'thalach': <tf.Tensor: shape=(), dtype=int64, numpy=140>, 'exang': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'oldpeak': <tf.Tensor: shape=(), dtype=float64, numpy=0.0>, 'slope': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'ca': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'thal': <tf.Tensor: shape=(), dtype=string, numpy=b'normal'>}
Target: tf.Tensor(0, shape=(), dtype=int64)

```
</div>
Let's batch the datasets:


```python
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
```

---
## Feature preprocessing with Keras layers


The following features are categorical features encoded as integers:

- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `ca`

We will encode these features using **one-hot encoding**. We have two options
here:

 - Use `CategoryEncoding()`, which requires knowing the range of input values
 and will error on input outside the range.
 - Use `IntegerLookup()` which will build a lookup table for inputs and reserve
 an output index for unkown input values.

For this example, we want a simple solution that will handle out of range inputs
at inference, so we will use `IntegerLookup()`.

We also have a categorical feature encoded as a string: `thal`. We will create an
index of all possible features and encode output using the `StringLookup()` layer.

Finally, the following feature are continuous numerical features:

- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`
- `slope`

For each of these features, we will use a `Normalization()` layer to make sure the mean
of each feature is 0 and its standard deviation is 1.

Below, we define 3 utility functions to do the operations:

- `encode_numerical_feature` to apply featurewise normalization to numerical features.
- `encode_string_categorical_feature` to first turn string inputs into integer indices,
then one-hot encode these integer indices.
- `encode_integer_categorical_feature` to one-hot encode integer categorical features.


```python
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

```

---
## Build a model

With this done, we can create our end-to-end model:


```python
# Categorical features encoded as integers
sex = keras.Input(shape=(1,), name="sex", dtype="int64")
cp = keras.Input(shape=(1,), name="cp", dtype="int64")
fbs = keras.Input(shape=(1,), name="fbs", dtype="int64")
restecg = keras.Input(shape=(1,), name="restecg", dtype="int64")
exang = keras.Input(shape=(1,), name="exang", dtype="int64")
ca = keras.Input(shape=(1,), name="ca", dtype="int64")

# Categorical feature encoded as string
thal = keras.Input(shape=(1,), name="thal", dtype="string")

# Numerical features
age = keras.Input(shape=(1,), name="age")
trestbps = keras.Input(shape=(1,), name="trestbps")
chol = keras.Input(shape=(1,), name="chol")
thalach = keras.Input(shape=(1,), name="thalach")
oldpeak = keras.Input(shape=(1,), name="oldpeak")
slope = keras.Input(shape=(1,), name="slope")

all_inputs = [
    sex,
    cp,
    fbs,
    restecg,
    exang,
    ca,
    thal,
    age,
    trestbps,
    chol,
    thalach,
    oldpeak,
    slope,
]

# Integer categorical features
sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)

# String categorical features
thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)

# Numerical features
age_encoded = encode_numerical_feature(age, "age", train_ds)
trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

all_features = layers.concatenate(
    [
        sex_encoded,
        cp_encoded,
        fbs_encoded,
        restecg_encoded,
        exang_encoded,
        slope_encoded,
        ca_encoded,
        thal_encoded,
        age_encoded,
        trestbps_encoded,
        chol_encoded,
        thalach_encoded,
        oldpeak_encoded,
    ]
)
x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
```

Let's visualize our connectivity graph:


```python
# `rankdir='LR'` is to make the graph horizontal.
keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
```

<div class="k-default-codeblock">
```
You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model/model_to_dot to work.

```
</div>
---
## Train the model


```python
model.fit(train_ds, epochs=50, validation_data=val_ds)
```

<div class="k-default-codeblock">
```
Epoch 1/50
8/8 [==============================] - 1s 51ms/step - loss: 0.6024 - accuracy: 0.6901 - val_loss: 0.4522 - val_accuracy: 0.7705
Epoch 2/50
8/8 [==============================] - 0s 3ms/step - loss: 0.5250 - accuracy: 0.7355 - val_loss: 0.4343 - val_accuracy: 0.7541
Epoch 3/50
8/8 [==============================] - 0s 3ms/step - loss: 0.5252 - accuracy: 0.7397 - val_loss: 0.4192 - val_accuracy: 0.7869
Epoch 4/50
8/8 [==============================] - 0s 4ms/step - loss: 0.5083 - accuracy: 0.7603 - val_loss: 0.4078 - val_accuracy: 0.8033
Epoch 5/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4781 - accuracy: 0.7769 - val_loss: 0.3978 - val_accuracy: 0.7869
Epoch 6/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4823 - accuracy: 0.7479 - val_loss: 0.3896 - val_accuracy: 0.7869
Epoch 7/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4707 - accuracy: 0.7397 - val_loss: 0.3835 - val_accuracy: 0.7869
Epoch 8/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4535 - accuracy: 0.7521 - val_loss: 0.3789 - val_accuracy: 0.8033
Epoch 9/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4122 - accuracy: 0.7975 - val_loss: 0.3746 - val_accuracy: 0.8033
Epoch 10/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4396 - accuracy: 0.7851 - val_loss: 0.3711 - val_accuracy: 0.8197
Epoch 11/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4418 - accuracy: 0.7769 - val_loss: 0.3685 - val_accuracy: 0.8197
Epoch 12/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4001 - accuracy: 0.8058 - val_loss: 0.3668 - val_accuracy: 0.8197
Epoch 13/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4129 - accuracy: 0.8017 - val_loss: 0.3658 - val_accuracy: 0.8197
Epoch 14/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3849 - accuracy: 0.7975 - val_loss: 0.3643 - val_accuracy: 0.8197
Epoch 15/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3779 - accuracy: 0.8182 - val_loss: 0.3634 - val_accuracy: 0.8197
Epoch 16/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3674 - accuracy: 0.8306 - val_loss: 0.3624 - val_accuracy: 0.8197
Epoch 17/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3517 - accuracy: 0.8471 - val_loss: 0.3625 - val_accuracy: 0.8197
Epoch 18/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3570 - accuracy: 0.8223 - val_loss: 0.3627 - val_accuracy: 0.8361
Epoch 19/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3489 - accuracy: 0.8306 - val_loss: 0.3627 - val_accuracy: 0.8361
Epoch 20/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3467 - accuracy: 0.8430 - val_loss: 0.3634 - val_accuracy: 0.8361
Epoch 21/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3527 - accuracy: 0.8471 - val_loss: 0.3652 - val_accuracy: 0.8361
Epoch 22/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3386 - accuracy: 0.8471 - val_loss: 0.3656 - val_accuracy: 0.8525
Epoch 23/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3291 - accuracy: 0.8595 - val_loss: 0.3665 - val_accuracy: 0.8525
Epoch 24/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3608 - accuracy: 0.8223 - val_loss: 0.3680 - val_accuracy: 0.8525
Epoch 25/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3494 - accuracy: 0.8554 - val_loss: 0.3688 - val_accuracy: 0.8525
Epoch 26/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3289 - accuracy: 0.8430 - val_loss: 0.3702 - val_accuracy: 0.8361
Epoch 27/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3016 - accuracy: 0.8719 - val_loss: 0.3714 - val_accuracy: 0.8525
Epoch 28/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3299 - accuracy: 0.8554 - val_loss: 0.3721 - val_accuracy: 0.8525
Epoch 29/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3476 - accuracy: 0.8388 - val_loss: 0.3726 - val_accuracy: 0.8525
Epoch 30/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3328 - accuracy: 0.8471 - val_loss: 0.3736 - val_accuracy: 0.8525
Epoch 31/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3203 - accuracy: 0.8471 - val_loss: 0.3741 - val_accuracy: 0.8525
Epoch 32/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3119 - accuracy: 0.8636 - val_loss: 0.3749 - val_accuracy: 0.8525
Epoch 33/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3082 - accuracy: 0.8719 - val_loss: 0.3761 - val_accuracy: 0.8525
Epoch 34/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3025 - accuracy: 0.8719 - val_loss: 0.3772 - val_accuracy: 0.8525
Epoch 35/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2936 - accuracy: 0.8678 - val_loss: 0.3790 - val_accuracy: 0.8525
Epoch 36/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3221 - accuracy: 0.8554 - val_loss: 0.3804 - val_accuracy: 0.8525
Epoch 37/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3063 - accuracy: 0.8595 - val_loss: 0.3813 - val_accuracy: 0.8525
Epoch 38/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2862 - accuracy: 0.8760 - val_loss: 0.3823 - val_accuracy: 0.8525
Epoch 39/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2801 - accuracy: 0.8967 - val_loss: 0.3837 - val_accuracy: 0.8525
Epoch 40/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2891 - accuracy: 0.8636 - val_loss: 0.3842 - val_accuracy: 0.8525
Epoch 41/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2960 - accuracy: 0.8802 - val_loss: 0.3847 - val_accuracy: 0.8525
Epoch 42/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2915 - accuracy: 0.8760 - val_loss: 0.3850 - val_accuracy: 0.8525
Epoch 43/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2604 - accuracy: 0.8926 - val_loss: 0.3855 - val_accuracy: 0.8361
Epoch 44/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2938 - accuracy: 0.8595 - val_loss: 0.3850 - val_accuracy: 0.8361
Epoch 45/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2973 - accuracy: 0.8678 - val_loss: 0.3850 - val_accuracy: 0.8525
Epoch 46/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2987 - accuracy: 0.8884 - val_loss: 0.3859 - val_accuracy: 0.8525
Epoch 47/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2597 - accuracy: 0.8926 - val_loss: 0.3855 - val_accuracy: 0.8525
Epoch 48/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2743 - accuracy: 0.8636 - val_loss: 0.3860 - val_accuracy: 0.8361
Epoch 49/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2872 - accuracy: 0.8802 - val_loss: 0.3868 - val_accuracy: 0.8361
Epoch 50/50
8/8 [==============================] - 0s 3ms/step - loss: 0.2997 - accuracy: 0.8554 - val_loss: 0.3876 - val_accuracy: 0.8361

<keras.callbacks.History at 0x7fdbdeaed590>

```
</div>
We quickly get to 80% validation accuracy.

---
## Inference on new data

To get a prediction for a new sample, you can simply call `model.predict()`. There are
just two things you need to do:

1. wrap scalars into a list so as to have a batch dimension (models only process batches
of data, not single samples)
2. Call `convert_to_tensor` on each feature


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
predictions = model.predict(input_dict)

print(
    "This particular patient had a %.1f percent probability "
    "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],)
)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 274ms/step
This particular patient had a 32.2 percent probability of having a heart disease, as evaluated by our model.

```
</div>
Demo available on HuggingFace.

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/🤗%20Model-Structured%20Data%20Classification-black.svg)](https://huggingface.co/keras-io/structured-data-classification/settings) | [![Generic badge](https://img.shields.io/badge/🤗%20Space-Structured%20Data%20Classification-black.svg)](https://huggingface.co/spaces/keras-io/structured-data-classification) |
