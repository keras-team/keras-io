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
Input: {'age': <tf.Tensor: shape=(), dtype=int64, numpy=62>, 'sex': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'cp': <tf.Tensor: shape=(), dtype=int64, numpy=4>, 'trestbps': <tf.Tensor: shape=(), dtype=int64, numpy=160>, 'chol': <tf.Tensor: shape=(), dtype=int64, numpy=164>, 'fbs': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'restecg': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 'thalach': <tf.Tensor: shape=(), dtype=int64, numpy=145>, 'exang': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'oldpeak': <tf.Tensor: shape=(), dtype=float64, numpy=6.2>, 'slope': <tf.Tensor: shape=(), dtype=int64, numpy=3>, 'ca': <tf.Tensor: shape=(), dtype=int64, numpy=3>, 'thal': <tf.Tensor: shape=(), dtype=string, numpy=b'reversible'>}
Target: tf.Tensor(1, shape=(), dtype=int64)

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
('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')

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
8/8 [==============================] - 1s 35ms/step - loss: 0.7554 - accuracy: 0.5058 - val_loss: 0.6907 - val_accuracy: 0.6393
Epoch 2/50
8/8 [==============================] - 0s 4ms/step - loss: 0.7024 - accuracy: 0.5917 - val_loss: 0.6564 - val_accuracy: 0.7049
Epoch 3/50
8/8 [==============================] - 0s 5ms/step - loss: 0.6661 - accuracy: 0.6249 - val_loss: 0.6252 - val_accuracy: 0.7213
Epoch 4/50
8/8 [==============================] - 0s 4ms/step - loss: 0.6287 - accuracy: 0.7024 - val_loss: 0.5978 - val_accuracy: 0.7377
Epoch 5/50
8/8 [==============================] - 0s 4ms/step - loss: 0.6490 - accuracy: 0.6668 - val_loss: 0.5745 - val_accuracy: 0.7213
Epoch 6/50
8/8 [==============================] - 0s 4ms/step - loss: 0.5906 - accuracy: 0.7570 - val_loss: 0.5550 - val_accuracy: 0.7541
Epoch 7/50
8/8 [==============================] - 0s 4ms/step - loss: 0.5659 - accuracy: 0.7353 - val_loss: 0.5376 - val_accuracy: 0.7869
Epoch 8/50
8/8 [==============================] - 0s 4ms/step - loss: 0.5463 - accuracy: 0.7190 - val_loss: 0.5219 - val_accuracy: 0.7869
Epoch 9/50
8/8 [==============================] - 0s 3ms/step - loss: 0.5498 - accuracy: 0.7106 - val_loss: 0.5082 - val_accuracy: 0.7869
Epoch 10/50
8/8 [==============================] - 0s 4ms/step - loss: 0.5344 - accuracy: 0.7141 - val_loss: 0.4965 - val_accuracy: 0.8033
Epoch 11/50
8/8 [==============================] - 0s 4ms/step - loss: 0.5369 - accuracy: 0.6961 - val_loss: 0.4857 - val_accuracy: 0.8033
Epoch 12/50
8/8 [==============================] - 0s 5ms/step - loss: 0.4920 - accuracy: 0.7948 - val_loss: 0.4757 - val_accuracy: 0.8197
Epoch 13/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4802 - accuracy: 0.7915 - val_loss: 0.4674 - val_accuracy: 0.8197
Epoch 14/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4936 - accuracy: 0.7382 - val_loss: 0.4599 - val_accuracy: 0.8197
Epoch 15/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4956 - accuracy: 0.7907 - val_loss: 0.4538 - val_accuracy: 0.8033
Epoch 16/50
8/8 [==============================] - 0s 5ms/step - loss: 0.4455 - accuracy: 0.7839 - val_loss: 0.4484 - val_accuracy: 0.8033
Epoch 17/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4192 - accuracy: 0.8480 - val_loss: 0.4432 - val_accuracy: 0.8197
Epoch 18/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4265 - accuracy: 0.7966 - val_loss: 0.4393 - val_accuracy: 0.8197
Epoch 19/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4694 - accuracy: 0.8085 - val_loss: 0.4366 - val_accuracy: 0.8197
Epoch 20/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4566 - accuracy: 0.8133 - val_loss: 0.4336 - val_accuracy: 0.8197
Epoch 21/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4060 - accuracy: 0.8351 - val_loss: 0.4314 - val_accuracy: 0.8197
Epoch 22/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4059 - accuracy: 0.8435 - val_loss: 0.4290 - val_accuracy: 0.8197
Epoch 23/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3863 - accuracy: 0.8342 - val_loss: 0.4272 - val_accuracy: 0.8197
Epoch 24/50
8/8 [==============================] - 0s 5ms/step - loss: 0.4222 - accuracy: 0.7998 - val_loss: 0.4260 - val_accuracy: 0.8197
Epoch 25/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3662 - accuracy: 0.8245 - val_loss: 0.4247 - val_accuracy: 0.8033
Epoch 26/50
8/8 [==============================] - 0s 5ms/step - loss: 0.4014 - accuracy: 0.8217 - val_loss: 0.4232 - val_accuracy: 0.8033
Epoch 27/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3935 - accuracy: 0.8375 - val_loss: 0.4219 - val_accuracy: 0.8033
Epoch 28/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4319 - accuracy: 0.8026 - val_loss: 0.4206 - val_accuracy: 0.8197
Epoch 29/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3893 - accuracy: 0.8074 - val_loss: 0.4202 - val_accuracy: 0.8197
Epoch 30/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3437 - accuracy: 0.8605 - val_loss: 0.4200 - val_accuracy: 0.8197
Epoch 31/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3859 - accuracy: 0.8133 - val_loss: 0.4198 - val_accuracy: 0.8197
Epoch 32/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3716 - accuracy: 0.8443 - val_loss: 0.4195 - val_accuracy: 0.8197
Epoch 33/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3691 - accuracy: 0.8217 - val_loss: 0.4198 - val_accuracy: 0.8197
Epoch 34/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3579 - accuracy: 0.8388 - val_loss: 0.4195 - val_accuracy: 0.8197
Epoch 35/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3164 - accuracy: 0.8620 - val_loss: 0.4199 - val_accuracy: 0.8197
Epoch 36/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3276 - accuracy: 0.8433 - val_loss: 0.4210 - val_accuracy: 0.8197
Epoch 37/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3781 - accuracy: 0.8469 - val_loss: 0.4214 - val_accuracy: 0.8197
Epoch 38/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3522 - accuracy: 0.8482 - val_loss: 0.4214 - val_accuracy: 0.8197
Epoch 39/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3988 - accuracy: 0.7981 - val_loss: 0.4216 - val_accuracy: 0.8197
Epoch 40/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3340 - accuracy: 0.8782 - val_loss: 0.4229 - val_accuracy: 0.8197
Epoch 41/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3404 - accuracy: 0.8318 - val_loss: 0.4227 - val_accuracy: 0.8197
Epoch 42/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3005 - accuracy: 0.8533 - val_loss: 0.4225 - val_accuracy: 0.8197
Epoch 43/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3364 - accuracy: 0.8675 - val_loss: 0.4223 - val_accuracy: 0.8197
Epoch 44/50
8/8 [==============================] - 0s 4ms/step - loss: 0.2801 - accuracy: 0.8792 - val_loss: 0.4229 - val_accuracy: 0.8197
Epoch 45/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3463 - accuracy: 0.8487 - val_loss: 0.4237 - val_accuracy: 0.8197
Epoch 46/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3047 - accuracy: 0.8694 - val_loss: 0.4238 - val_accuracy: 0.8197
Epoch 47/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3157 - accuracy: 0.8621 - val_loss: 0.4249 - val_accuracy: 0.8197
Epoch 48/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3048 - accuracy: 0.8557 - val_loss: 0.4251 - val_accuracy: 0.8197
Epoch 49/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3722 - accuracy: 0.8316 - val_loss: 0.4254 - val_accuracy: 0.8197
Epoch 50/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3302 - accuracy: 0.8688 - val_loss: 0.4254 - val_accuracy: 0.8197

<tensorflow.python.keras.callbacks.History at 0x7f1658167ac0>

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
This particular patient had a 18.8 percent probability of having a heart disease, as evaluated by our model.

```
</div>
