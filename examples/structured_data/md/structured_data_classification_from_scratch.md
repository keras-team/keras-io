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

Note that this example should be run with TensorFlow 2.3 or higher, or `tf-nightly`.

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
Input: {'age': <tf.Tensor: shape=(), dtype=int64, numpy=57>, 'sex': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'cp': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'trestbps': <tf.Tensor: shape=(), dtype=int64, numpy=140>, 'chol': <tf.Tensor: shape=(), dtype=int64, numpy=241>, 'fbs': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'restecg': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'thalach': <tf.Tensor: shape=(), dtype=int64, numpy=123>, 'exang': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'oldpeak': <tf.Tensor: shape=(), dtype=float64, numpy=0.2>, 'slope': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'ca': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'thal': <tf.Tensor: shape=(), dtype=string, numpy=b'normal'>}
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

We will encode these features using **one-hot encoding** using the `CategoryEncoding()`
layer.

We also have a categorical feature encoded as a string: `thal`. We will first create an
index of all possible features using the `StringLookup()` layer, then we will one-hot
encode the output indices using a `CategoryEncoding()` layer.

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
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


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


def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
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
sex_encoded = encode_integer_categorical_feature(sex, "sex", train_ds)
cp_encoded = encode_integer_categorical_feature(cp, "cp", train_ds)
fbs_encoded = encode_integer_categorical_feature(fbs, "fbs", train_ds)
restecg_encoded = encode_integer_categorical_feature(restecg, "restecg", train_ds)
exang_encoded = encode_integer_categorical_feature(exang, "exang", train_ds)
ca_encoded = encode_integer_categorical_feature(ca, "ca", train_ds)

# String categorical features
thal_encoded = encode_string_categorical_feature(thal, "thal", train_ds)

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




![png](/img/examples/structured_data/structured_data_classification_from_scratch/structured_data_classification_from_scratch_23_0.png)



---
## Train the model


```python
model.fit(train_ds, epochs=50, validation_data=val_ds)
```

<div class="k-default-codeblock">
```
Epoch 1/50
8/8 [==============================] - 0s 28ms/step - loss: 0.7561 - accuracy: 0.5992 - val_loss: 0.5802 - val_accuracy: 0.6885
Epoch 2/50
8/8 [==============================] - 0s 2ms/step - loss: 0.7136 - accuracy: 0.6322 - val_loss: 0.5368 - val_accuracy: 0.7705
Epoch 3/50
8/8 [==============================] - 0s 3ms/step - loss: 0.6649 - accuracy: 0.6694 - val_loss: 0.5011 - val_accuracy: 0.8197
Epoch 4/50
8/8 [==============================] - 0s 3ms/step - loss: 0.6005 - accuracy: 0.6694 - val_loss: 0.4728 - val_accuracy: 0.8197
Epoch 5/50
8/8 [==============================] - 0s 2ms/step - loss: 0.6148 - accuracy: 0.6983 - val_loss: 0.4503 - val_accuracy: 0.8197
Epoch 6/50
8/8 [==============================] - 0s 2ms/step - loss: 0.5836 - accuracy: 0.6942 - val_loss: 0.4330 - val_accuracy: 0.8197
Epoch 7/50
8/8 [==============================] - 0s 3ms/step - loss: 0.5158 - accuracy: 0.7314 - val_loss: 0.4182 - val_accuracy: 0.8361
Epoch 8/50
8/8 [==============================] - 0s 2ms/step - loss: 0.5251 - accuracy: 0.7479 - val_loss: 0.4067 - val_accuracy: 0.8197
Epoch 9/50
8/8 [==============================] - 0s 2ms/step - loss: 0.5136 - accuracy: 0.7479 - val_loss: 0.3973 - val_accuracy: 0.8197
Epoch 10/50
8/8 [==============================] - 0s 3ms/step - loss: 0.5068 - accuracy: 0.7397 - val_loss: 0.3898 - val_accuracy: 0.8197
Epoch 11/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4652 - accuracy: 0.7479 - val_loss: 0.3832 - val_accuracy: 0.8197
Epoch 12/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4665 - accuracy: 0.7769 - val_loss: 0.3775 - val_accuracy: 0.8033
Epoch 13/50
8/8 [==============================] - 0s 2ms/step - loss: 0.4555 - accuracy: 0.7851 - val_loss: 0.3728 - val_accuracy: 0.7869
Epoch 14/50
8/8 [==============================] - 0s 2ms/step - loss: 0.4347 - accuracy: 0.7975 - val_loss: 0.3691 - val_accuracy: 0.7869
Epoch 15/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4524 - accuracy: 0.7603 - val_loss: 0.3662 - val_accuracy: 0.8033
Epoch 16/50
8/8 [==============================] - 0s 3ms/step - loss: 0.4526 - accuracy: 0.7934 - val_loss: 0.3645 - val_accuracy: 0.8033
Epoch 17/50
8/8 [==============================] - 0s 2ms/step - loss: 0.4029 - accuracy: 0.8223 - val_loss: 0.3630 - val_accuracy: 0.8361
Epoch 18/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3908 - accuracy: 0.8140 - val_loss: 0.3615 - val_accuracy: 0.8361
Epoch 19/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3715 - accuracy: 0.8306 - val_loss: 0.3603 - val_accuracy: 0.8361
Epoch 20/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3736 - accuracy: 0.8264 - val_loss: 0.3590 - val_accuracy: 0.8361
Epoch 21/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3887 - accuracy: 0.8264 - val_loss: 0.3585 - val_accuracy: 0.8361
Epoch 22/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3933 - accuracy: 0.7810 - val_loss: 0.3582 - val_accuracy: 0.8361
Epoch 23/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3740 - accuracy: 0.8471 - val_loss: 0.3575 - val_accuracy: 0.8361
Epoch 24/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3661 - accuracy: 0.8347 - val_loss: 0.3576 - val_accuracy: 0.8361
Epoch 25/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3649 - accuracy: 0.8430 - val_loss: 0.3582 - val_accuracy: 0.8361
Epoch 26/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3339 - accuracy: 0.8554 - val_loss: 0.3581 - val_accuracy: 0.8361
Epoch 27/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3539 - accuracy: 0.8223 - val_loss: 0.3586 - val_accuracy: 0.8361
Epoch 28/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3412 - accuracy: 0.8471 - val_loss: 0.3597 - val_accuracy: 0.8361
Epoch 29/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3243 - accuracy: 0.8554 - val_loss: 0.3611 - val_accuracy: 0.8361
Epoch 30/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3282 - accuracy: 0.8512 - val_loss: 0.3620 - val_accuracy: 0.8361
Epoch 31/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3120 - accuracy: 0.8554 - val_loss: 0.3621 - val_accuracy: 0.8361
Epoch 32/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3153 - accuracy: 0.8760 - val_loss: 0.3626 - val_accuracy: 0.8361
Epoch 33/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3404 - accuracy: 0.8554 - val_loss: 0.3636 - val_accuracy: 0.8361
Epoch 34/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3462 - accuracy: 0.8306 - val_loss: 0.3646 - val_accuracy: 0.8361
Epoch 35/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3055 - accuracy: 0.8636 - val_loss: 0.3660 - val_accuracy: 0.8361
Epoch 36/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3220 - accuracy: 0.8554 - val_loss: 0.3673 - val_accuracy: 0.8361
Epoch 37/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3080 - accuracy: 0.8843 - val_loss: 0.3694 - val_accuracy: 0.8361
Epoch 38/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3105 - accuracy: 0.8471 - val_loss: 0.3702 - val_accuracy: 0.8361
Epoch 39/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3254 - accuracy: 0.8554 - val_loss: 0.3709 - val_accuracy: 0.8361
Epoch 40/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3227 - accuracy: 0.8512 - val_loss: 0.3719 - val_accuracy: 0.8361
Epoch 41/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3240 - accuracy: 0.8636 - val_loss: 0.3730 - val_accuracy: 0.8361
Epoch 42/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3166 - accuracy: 0.8554 - val_loss: 0.3732 - val_accuracy: 0.8361
Epoch 43/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3128 - accuracy: 0.8554 - val_loss: 0.3736 - val_accuracy: 0.8361
Epoch 44/50
8/8 [==============================] - 0s 2ms/step - loss: 0.3010 - accuracy: 0.8719 - val_loss: 0.3722 - val_accuracy: 0.8361
Epoch 45/50
8/8 [==============================] - 0s 2ms/step - loss: 0.2848 - accuracy: 0.8884 - val_loss: 0.3719 - val_accuracy: 0.8361
Epoch 46/50
8/8 [==============================] - 0s 2ms/step - loss: 0.2946 - accuracy: 0.8802 - val_loss: 0.3714 - val_accuracy: 0.8361
Epoch 47/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3013 - accuracy: 0.8719 - val_loss: 0.3720 - val_accuracy: 0.8361
Epoch 48/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3165 - accuracy: 0.8388 - val_loss: 0.3727 - val_accuracy: 0.8361
Epoch 49/50
8/8 [==============================] - 0s 3ms/step - loss: 0.3102 - accuracy: 0.8636 - val_loss: 0.3725 - val_accuracy: 0.8361
Epoch 50/50
8/8 [==============================] - 0s 2ms/step - loss: 0.2908 - accuracy: 0.8678 - val_loss: 0.3724 - val_accuracy: 0.8361

<tensorflow.python.keras.callbacks.History at 0x1500eb510>

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
This particular patient had a 31.0 percent probability of having a heart disease, as evaluated by our model.

```
</div>