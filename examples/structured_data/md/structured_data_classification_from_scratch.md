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
Input: {'age': <tf.Tensor: shape=(), dtype=int64, numpy=63>, 'sex': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'cp': <tf.Tensor: shape=(), dtype=int64, numpy=4>, 'trestbps': <tf.Tensor: shape=(), dtype=int64, numpy=130>, 'chol': <tf.Tensor: shape=(), dtype=int64, numpy=254>, 'fbs': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'restecg': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 'thalach': <tf.Tensor: shape=(), dtype=int64, numpy=147>, 'exang': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'oldpeak': <tf.Tensor: shape=(), dtype=float64, numpy=1.4>, 'slope': <tf.Tensor: shape=(), dtype=int64, numpy=2>, 'ca': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'thal': <tf.Tensor: shape=(), dtype=string, numpy=b'reversible'>}
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




![png](/img/examples/structured_data/structured_data_classification_from_scratch/structured_data_classification_from_scratch_23_0.png)



---
## Train the model


```python
model.fit(train_ds, epochs=50, validation_data=val_ds)
```

<div class="k-default-codeblock">
```
Epoch 1/50
8/8 [==============================] - 1s 38ms/step - loss: 0.6339 - accuracy: 0.6818 - val_loss: 0.5833 - val_accuracy: 0.7213
Epoch 2/50
8/8 [==============================] - 0s 4ms/step - loss: 0.6120 - accuracy: 0.6653 - val_loss: 0.5536 - val_accuracy: 0.7705
Epoch 3/50
8/8 [==============================] - 0s 4ms/step - loss: 0.5803 - accuracy: 0.7025 - val_loss: 0.5281 - val_accuracy: 0.8033
Epoch 4/50
8/8 [==============================] - 0s 5ms/step - loss: 0.5498 - accuracy: 0.7107 - val_loss: 0.5080 - val_accuracy: 0.8197
Epoch 5/50
8/8 [==============================] - 0s 5ms/step - loss: 0.5349 - accuracy: 0.7479 - val_loss: 0.4896 - val_accuracy: 0.8197
Epoch 6/50
8/8 [==============================] - 0s 7ms/step - loss: 0.5360 - accuracy: 0.7273 - val_loss: 0.4722 - val_accuracy: 0.8033
Epoch 7/50
8/8 [==============================] - 0s 5ms/step - loss: 0.4922 - accuracy: 0.7769 - val_loss: 0.4576 - val_accuracy: 0.8197
Epoch 8/50
8/8 [==============================] - 0s 5ms/step - loss: 0.4761 - accuracy: 0.7438 - val_loss: 0.4442 - val_accuracy: 0.8197
Epoch 9/50
8/8 [==============================] - 0s 5ms/step - loss: 0.4671 - accuracy: 0.7893 - val_loss: 0.4322 - val_accuracy: 0.8361
Epoch 10/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4560 - accuracy: 0.7727 - val_loss: 0.4216 - val_accuracy: 0.8197
Epoch 11/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4518 - accuracy: 0.7603 - val_loss: 0.4132 - val_accuracy: 0.8197
Epoch 12/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4423 - accuracy: 0.7934 - val_loss: 0.4057 - val_accuracy: 0.8197
Epoch 13/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4154 - accuracy: 0.8017 - val_loss: 0.3995 - val_accuracy: 0.8033
Epoch 14/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4176 - accuracy: 0.7769 - val_loss: 0.3933 - val_accuracy: 0.8033
Epoch 15/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3971 - accuracy: 0.8140 - val_loss: 0.3882 - val_accuracy: 0.8033
Epoch 16/50
8/8 [==============================] - 0s 4ms/step - loss: 0.4069 - accuracy: 0.8017 - val_loss: 0.3835 - val_accuracy: 0.8033
Epoch 17/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3744 - accuracy: 0.8182 - val_loss: 0.3797 - val_accuracy: 0.8033
Epoch 18/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3963 - accuracy: 0.8099 - val_loss: 0.3762 - val_accuracy: 0.8033
Epoch 19/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3745 - accuracy: 0.8223 - val_loss: 0.3734 - val_accuracy: 0.8033
Epoch 20/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3843 - accuracy: 0.8306 - val_loss: 0.3715 - val_accuracy: 0.8197
Epoch 21/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3565 - accuracy: 0.8471 - val_loss: 0.3697 - val_accuracy: 0.8197
Epoch 22/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3576 - accuracy: 0.8471 - val_loss: 0.3680 - val_accuracy: 0.8197
Epoch 23/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3548 - accuracy: 0.8223 - val_loss: 0.3673 - val_accuracy: 0.8197
Epoch 24/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3698 - accuracy: 0.8512 - val_loss: 0.3669 - val_accuracy: 0.8197
Epoch 25/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3328 - accuracy: 0.8347 - val_loss: 0.3660 - val_accuracy: 0.8197
Epoch 26/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3493 - accuracy: 0.8264 - val_loss: 0.3657 - val_accuracy: 0.8197
Epoch 27/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3679 - accuracy: 0.8306 - val_loss: 0.3650 - val_accuracy: 0.8361
Epoch 28/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3172 - accuracy: 0.8595 - val_loss: 0.3656 - val_accuracy: 0.8361
Epoch 29/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3312 - accuracy: 0.8554 - val_loss: 0.3665 - val_accuracy: 0.8361
Epoch 30/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3226 - accuracy: 0.8554 - val_loss: 0.3666 - val_accuracy: 0.8361
Epoch 31/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3007 - accuracy: 0.8719 - val_loss: 0.3667 - val_accuracy: 0.8361
Epoch 32/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3096 - accuracy: 0.8678 - val_loss: 0.3672 - val_accuracy: 0.8361
Epoch 33/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3293 - accuracy: 0.8306 - val_loss: 0.3673 - val_accuracy: 0.8361
Epoch 34/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3156 - accuracy: 0.8430 - val_loss: 0.3677 - val_accuracy: 0.8361
Epoch 35/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3272 - accuracy: 0.8595 - val_loss: 0.3680 - val_accuracy: 0.8361
Epoch 36/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3194 - accuracy: 0.8471 - val_loss: 0.3685 - val_accuracy: 0.8361
Epoch 37/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3125 - accuracy: 0.8471 - val_loss: 0.3688 - val_accuracy: 0.8361
Epoch 38/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3096 - accuracy: 0.8388 - val_loss: 0.3689 - val_accuracy: 0.8361
Epoch 39/50
8/8 [==============================] - 0s 5ms/step - loss: 0.3096 - accuracy: 0.8512 - val_loss: 0.3696 - val_accuracy: 0.8525
Epoch 40/50
8/8 [==============================] - 0s 5ms/step - loss: 0.2822 - accuracy: 0.8802 - val_loss: 0.3712 - val_accuracy: 0.8525
Epoch 41/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3234 - accuracy: 0.8512 - val_loss: 0.3718 - val_accuracy: 0.8525
Epoch 42/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3283 - accuracy: 0.8802 - val_loss: 0.3716 - val_accuracy: 0.8525
Epoch 43/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3059 - accuracy: 0.8636 - val_loss: 0.3718 - val_accuracy: 0.8525
Epoch 44/50
8/8 [==============================] - 0s 4ms/step - loss: 0.3055 - accuracy: 0.8802 - val_loss: 0.3719 - val_accuracy: 0.8525
Epoch 45/50
8/8 [==============================] - 0s 4ms/step - loss: 0.2874 - accuracy: 0.8719 - val_loss: 0.3729 - val_accuracy: 0.8525
Epoch 46/50
8/8 [==============================] - 0s 4ms/step - loss: 0.2719 - accuracy: 0.8760 - val_loss: 0.3737 - val_accuracy: 0.8361
Epoch 47/50
8/8 [==============================] - 0s 4ms/step - loss: 0.2794 - accuracy: 0.8843 - val_loss: 0.3745 - val_accuracy: 0.8361
Epoch 48/50
8/8 [==============================] - 0s 5ms/step - loss: 0.2940 - accuracy: 0.8802 - val_loss: 0.3747 - val_accuracy: 0.8361
Epoch 49/50
8/8 [==============================] - 0s 4ms/step - loss: 0.2860 - accuracy: 0.8554 - val_loss: 0.3746 - val_accuracy: 0.8525
Epoch 50/50
8/8 [==============================] - 0s 4ms/step - loss: 0.2845 - accuracy: 0.8595 - val_loss: 0.3743 - val_accuracy: 0.8361

<keras.callbacks.History at 0x12eb51e10>

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
This particular patient had a 25.5 percent probability of having a heart disease, as evaluated by our model.

```
</div>