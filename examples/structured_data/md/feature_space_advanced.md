# FeatureSpace advanced use cases

**Author:** [Dimitre Oliveira](https://www.linkedin.com/in/dimitre-oliveira-7a1a0113a/)<br>
**Date created:** 2023/07/01<br>
**Last modified:** 2023/07/01<br>
**Description:** How to use FeatureSpace for advanced preprocessing use cases.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/feature_space_advanced.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/feature_space_advanced.py)



---
## Introduction

This example is an extension of the
[Structured data classification with FeatureSpace](https://keras.io/examples/structured_data/structured_data_classification_with_feature_space/)
code example, and here we will extend it to cover more complex use
cases of the [`keras.utils.FeatureSpace`](https://keras.io/api/utils/feature_space/)
preprocessing utility, like feature hashing, feature crosses, handling missing values and
integrating [Keras preprocessing layers](https://keras.io/api/layers/preprocessing_layers/)
with FeatureSpace.

The general task still is structured data classification (also known as tabular data
classification) using a data that includes numerical features, integer categorical
features, and string categorical features.

### The dataset

[Our dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) is provided by a
Portuguese banking institution.
It's a CSV file with 4119 rows. Each row contains information about marketing campaigns
based on phone calls, and each column describes an attribute of the client. We use the
features to predict whether the client subscribed ('yes') or not ('no') to the product
(bank term deposit).

Here's the description of each feature:

Column| Description| Feature Type
------|------------|-------------
Age | Age of the client | Numerical
Job | Type of job | Categorical
Marital | Marital status | Categorical
Education | Education level of the client | Categorical
Default | Has credit in default? | Categorical
Housing | Has housing loan? | Categorical
Loan | Has personal loan? | Categorical
Contact | Contact communication type | Categorical
Month | Last contact month of year | Categorical
Day_of_week | Last contact day of the week | Categorical
Duration | Last contact duration, in seconds | Numerical
Campaign | Number of contacts performed during this campaign and for this client | Numerical
Pdays | Number of days that passed by after the client was last contacted from a previous campaign | Numerical
Previous | Number of contacts performed before this campaign and for this client | Numerical
Poutcome | Outcome of the previous marketing campaign | Categorical
Emp.var.rate | Employment variation rate | Numerical
Cons.price.idx | Consumer price index | Numerical
Cons.conf.idx | Consumer confidence index | Numerical
Euribor3m | Euribor 3 month rate | Numerical
Nr.employed | Number of employees | Numerical
Y | Has the client subscribed a term deposit? | Target

**Important note regarding the feature `duration`**:  this attribute highly affects the
output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a
call is performed. Also, after the end of the call y is obviously known. Thus, this input
should only be included for benchmark purposes and should be discarded if the intention
is to have a realistic predictive model. For this reason we will drop it.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.utils import FeatureSpace
import pandas as pd
import tensorflow as tf
from pathlib import Path
from zipfile import ZipFile
```

---
## Load the data

Let's download the data and load it into a Pandas dataframe:


```python
data_url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
data_zipped_path = keras.utils.get_file("bank_marketing.zip", data_url, extract=True)
keras_datasets_path = Path(data_zipped_path).parents[0]
with ZipFile(f"{keras_datasets_path}/bank-additional.zip", "r") as zip:
    # Extract files
    zip.extractall(path=keras_datasets_path)

dataframe = pd.read_csv(
    f"{keras_datasets_path}/bank-additional/bank-additional.csv", sep=";"
)
```

We will create a new feature `previously_contacted` to be able to demonstrate some useful
preprocessing techniques, this feature is based on `pdays`. According to the dataset
information if `pdays = 999` it means that the client was not previously contacted, so
let's create a feature to capture that.


```python
# Droping `duration` to avoid target leak
dataframe.drop("duration", axis=1, inplace=True)
# Creating the new feature `previously_contacted`
dataframe["previously_contacted"] = dataframe["pdays"].map(
    lambda x: 0 if x == 999 else 1
)
```

The dataset includes 4119 samples with 21 columns per sample (20 features, plus the
target label), here's a preview of a few samples:


```python
print(f"Dataframe shape: {dataframe.shape}")
print(dataframe.head())
```

<div class="k-default-codeblock">
```
Dataframe shape: (4119, 21)
   age          job  marital          education default  housing     loan  \
0   30  blue-collar  married           basic.9y      no      yes       no   
1   39     services   single        high.school      no       no       no   
2   25     services  married        high.school      no      yes       no   
3   38     services  married           basic.9y      no  unknown  unknown   
4   47       admin.  married  university.degree      no      yes       no   
```
</div>
    
<div class="k-default-codeblock">
```
     contact month day_of_week  ...  pdays  previous     poutcome  \
0   cellular   may         fri  ...    999         0  nonexistent   
1  telephone   may         fri  ...    999         0  nonexistent   
2  telephone   jun         wed  ...    999         0  nonexistent   
3  telephone   jun         fri  ...    999         0  nonexistent   
4   cellular   nov         mon  ...    999         0  nonexistent   
```
</div>
    
<div class="k-default-codeblock">
```
  emp.var.rate  cons.price.idx  cons.conf.idx  euribor3m  nr.employed   y  \
0         -1.8          92.893          -46.2      1.313       5099.1  no   
1          1.1          93.994          -36.4      4.855       5191.0  no   
2          1.4          94.465          -41.8      4.962       5228.1  no   
3          1.4          94.465          -41.8      4.959       5228.1  no   
4         -0.1          93.200          -42.0      4.191       5195.8  no   
```
</div>
    
<div class="k-default-codeblock">
```
  previously_contacted  
0                    0  
1                    0  
2                    0  
3                    0  
4                    0  
```
</div>
    
<div class="k-default-codeblock">
```
[5 rows x 21 columns]

```
</div>
The column, "y", indicates whether the client has subscribed a term deposit or not.

---
## Train/validation split

Let's split the data into a training and validation set:


```python
valid_dataframe = dataframe.sample(frac=0.2, random_state=0)
train_dataframe = dataframe.drop(valid_dataframe.index)

print(
    f"Using {len(train_dataframe)} samples for training and "
    f"{len(valid_dataframe)} for validation"
)
```

<div class="k-default-codeblock">
```
Using 3295 samples for training and 824 for validation

```
</div>
---
## Generating TF datasets

Let's generate
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects
for each dataframe, since our target column `y` is a string we also need to encode it as
an integer to be able to train our model with it. To achieve this we will create a
`StringLookup` layer that will map the strings "no" and "yes" into "0" and "1"
respectively.


```python
label_lookup = keras.layers.StringLookup(
    # the order here is important since the first index will be encoded as 0
    vocabulary=["no", "yes"],
    num_oov_indices=0,
)


def encode_label(x, y):
    encoded_y = label_lookup(y)
    return x, encoded_y


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("y")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.map(encode_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
valid_ds = dataframe_to_dataset(valid_dataframe)
```

Each `Dataset` yields a tuple `(input, target)` where `input` is a dictionary of features
and `target` is the value `0` or `1`:


```python
for x, y in dataframe_to_dataset(train_dataframe).take(1):
    print(f"Input: {x}")
    print(f"Target: {y}")
```

<div class="k-default-codeblock">
```
Input: {'age': <tf.Tensor: shape=(), dtype=int64, numpy=33>, 'job': <tf.Tensor: shape=(), dtype=string, numpy=b'technician'>, 'marital': <tf.Tensor: shape=(), dtype=string, numpy=b'married'>, 'education': <tf.Tensor: shape=(), dtype=string, numpy=b'university.degree'>, 'default': <tf.Tensor: shape=(), dtype=string, numpy=b'unknown'>, 'housing': <tf.Tensor: shape=(), dtype=string, numpy=b'yes'>, 'loan': <tf.Tensor: shape=(), dtype=string, numpy=b'no'>, 'contact': <tf.Tensor: shape=(), dtype=string, numpy=b'cellular'>, 'month': <tf.Tensor: shape=(), dtype=string, numpy=b'aug'>, 'day_of_week': <tf.Tensor: shape=(), dtype=string, numpy=b'tue'>, 'campaign': <tf.Tensor: shape=(), dtype=int64, numpy=1>, 'pdays': <tf.Tensor: shape=(), dtype=int64, numpy=999>, 'previous': <tf.Tensor: shape=(), dtype=int64, numpy=0>, 'poutcome': <tf.Tensor: shape=(), dtype=string, numpy=b'nonexistent'>, 'emp.var.rate': <tf.Tensor: shape=(), dtype=float64, numpy=1.4>, 'cons.price.idx': <tf.Tensor: shape=(), dtype=float64, numpy=93.444>, 'cons.conf.idx': <tf.Tensor: shape=(), dtype=float64, numpy=-36.1>, 'euribor3m': <tf.Tensor: shape=(), dtype=float64, numpy=4.963>, 'nr.employed': <tf.Tensor: shape=(), dtype=float64, numpy=5228.1>, 'previously_contacted': <tf.Tensor: shape=(), dtype=int64, numpy=0>}
Target: 0

```
</div>
---
## Preprocessing

Usually our data is not on the proper or best format for modeling, this is why most of
the time we need to do some kind of preprocessing on the features to make them compatible
with the model or to extract the most of them for the task. We need to do this
preprocessing step for training but but at inference we also need to make sure that the
data goes through the same process, this where a utility like `FeatureSpace` shines, we
can define all the preprocessing once and re-use it at different stages of our system.

Here we will see how to use `FeatureSpace` to perform more complex transformations and
its flexibility, then combine everything together into a single component to preprocess
data for our model.

The `FeatureSpace` utility learns how to process the data by using the `adapt()` function
to learn from it, this requires a dataset containing only feature, so let's create it
together with a utility function to show the preprocessing example in practice:


```python
train_ds_with_no_labels = train_ds.map(lambda x, _: x)


def example_feature_space(dataset, feature_space, feature_names):
    feature_space.adapt(dataset)
    for x in dataset.take(1):
        inputs = {feature_name: x[feature_name] for feature_name in feature_names}
        preprocessed_x = feature_space(inputs)
        print(f"Input: {[{k:v.numpy()} for k, v in inputs.items()]}")
        print(
            f"Preprocessed output: {[{k:v.numpy()} for k, v in preprocessed_x.items()]}"
        )

```

### Feature hashing

**Feature hashing** means hashing or encoding a set of values into a defined number of
bins, in this case we have `campaign` (number of contacts performed during this campaign
and for a client) which is a numerical feature that can assume a varying range of values
and we will hash it into 4 bins, this means that any possible value of the original
feature will be placed into one of those possible 4 bins. The output here can be a
one-hot encoded vector or a single number.


```python
feature_space = FeatureSpace(
    features={
        "campaign": FeatureSpace.integer_hashed(num_bins=4, output_mode="one_hot")
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["campaign"])
```

<div class="k-default-codeblock">
```
Input: [{'campaign': 1}]
Preprocessed output: [{'campaign': array([0., 1., 0., 0.], dtype=float32)}]

```
</div>
**Feature hashing** can also be used for string features.


```python
feature_space = FeatureSpace(
    features={
        "education": FeatureSpace.string_hashed(num_bins=3, output_mode="one_hot")
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["education"])
```

<div class="k-default-codeblock">
```
Input: [{'education': b'basic.9y'}]
Preprocessed output: [{'education': array([0., 1., 0.], dtype=float32)}]

```
</div>
For numerical features we can get a similar behavior by using the `float_discretized`
option, the main difference between this and `integer_hashed` is that with the former we
bin the values while keeping some numerical relationship (close values will likely be
placed at the same bin) while the later (hashing) we cannot guarantee that those numbers
will be hashed into the same bin, it depends on the hashing function.


```python
feature_space = FeatureSpace(
    features={"age": FeatureSpace.float_discretized(num_bins=3, output_mode="one_hot")},
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["age"])
```

<div class="k-default-codeblock">
```
Input: [{'age': 40}]
Preprocessed output: [{'age': array([0., 1., 0.], dtype=float32)}]

```
</div>
### Feature indexing

**Indexing** a string feature essentially means creating a discrete numerical
representation for it, this is especially important for string features since most models
only accept numerical features. This transformation will place the string values into
different categories. The output here can be a one-hot encoded vector or a single number.

Note that by specifying `num_oov_indices=1` we leave one spot at our output vector for
OOV (out of vocabulary) values this is an important tool to handle missing or unseen
values after the training (values that were not seen during the `adapt()` step)


```python
feature_space = FeatureSpace(
    features={
        "default": FeatureSpace.string_categorical(
            num_oov_indices=1, output_mode="one_hot"
        )
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["default"])
```

<div class="k-default-codeblock">
```
Input: [{'default': b'unknown'}]
Preprocessed output: [{'default': array([0., 0., 1., 0.], dtype=float32)}]

```
</div>
We also can do **feature indexing** for integer features, this can be quite important for
some datasets where categorical features are replaced by numbers, for instance features
like `sex` or `gender` where values like (`1 and 0`) do not have a numerical relationship
between them, they are just different categories, this behavior can be perfectly captured
by this transformation.

On this dataset we can use the feature that we created `previously_contacted`. For this
case we want to explicitly set `num_oov_indices=0`, the reason is that we only expect two
possible values for the feature, anything else would be either wrong input or an issue
with the data creation, for this reason we would probably just want the code to throw an
error so that we can be aware of the issue and fix it.


```python
feature_space = FeatureSpace(
    features={
        "previously_contacted": FeatureSpace.integer_categorical(
            num_oov_indices=0, output_mode="one_hot"
        )
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["previously_contacted"])
```

<div class="k-default-codeblock">
```
Input: [{'previously_contacted': 0}]
Preprocessed output: [{'previously_contacted': array([1., 0.], dtype=float32)}]

```
</div>
### Feature crosses (mixing features of diverse types)

With **crosses** we can do feature interactions between an arbitrary number of features
of mixed types as long as they are categorical features, you can think of instead of
having a feature {'age': 20} and another {'job': 'entrepreneur'} we can have
{'age_X_job': 20_entrepreneur}, but with `FeatureSpace` and **crosses** we can apply
specific preprocessing to each individual feature and to the feature cross itself. This
option can be very powerful for specific use cases, here might be a good option since age
combined with job can have different meanings for the banking domain.

We will cross `age` and `job` and hash the combination output of them into a vector
representation of size 8. The output here can be a one-hot encoded vector or a single
number.

Sometimes the combination of multiple features can result into on a super large feature
space, think about crossing someone's ZIP code with its last name, the possibilities
would be in the thousands, that is why the `crossing_dim` parameter is so important it
limits the output dimension of the cross feature.

Note that the combination of possible values of the 6 bins of `age` and the 12 values of
`job` would be 72, so by choosing `crossing_dim = 8` we are choosing to constrain the
output vector.


```python
feature_space = FeatureSpace(
    features={
        "age": FeatureSpace.integer_hashed(num_bins=6, output_mode="one_hot"),
        "job": FeatureSpace.string_categorical(
            num_oov_indices=0, output_mode="one_hot"
        ),
    },
    crosses=[
        FeatureSpace.cross(
            feature_names=("age", "job"),
            crossing_dim=8,
            output_mode="one_hot",
        )
    ],
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["age", "job"])
```

<div class="k-default-codeblock">
```
Input: [{'age': 28}, {'job': b'blue-collar'}]
Preprocessed output: [{'age': array([0., 0., 1., 0., 0., 0.], dtype=float32)}, {'job': array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}, {'age_X_job': array([0., 0., 0., 0., 1., 0., 0., 0.], dtype=float32)}]

```
</div>
### FeatureSpace using a Keras preprocessing layer

To be a really flexible and extensible feature we cannot only rely on those pre-defined
transformation, we must be able to re-use other transformations from the Keras/TensorFlow
ecosystem and customize our own, this is why `FeatureSpace` is also designed to work with
[Keras preprocessing layers](https://keras.io/api/layers/preprocessing_layers/), this way we
can use sophisticated data transformations provided by the framework, you can even create
your own custom Keras preprocessing layers and use it in the same way.

Here we are going to use the
[`keras.layers.TextVectorization`](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/#textvectorization-class)
preprocessing layer to create a TF-IDF
feature from our data. Note that this feature is not a really good use case for TF-IDF,
this is just for demonstration purposes.


```python
custom_layer = keras.layers.TextVectorization(output_mode="tf_idf")

feature_space = FeatureSpace(
    features={
        "education": FeatureSpace.feature(
            preprocessor=custom_layer, dtype="string", output_mode="float"
        )
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["education"])
```

<div class="k-default-codeblock">
```
Input: [{'education': b'university.degree'}]
Preprocessed output: [{'education': array([0.       , 1.4574516, 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       ], dtype=float32)}]

```
</div>
---
## Configuring the final `FeatureSpace`

Now that we know how to use `FeatureSpace` for more complex use cases let's pick the ones
that looks more useful for this task and create the final `FeatureSpace` component.

To configure how each feature should be preprocessed,
we instantiate a `keras.utils.FeatureSpace`, and we
pass to it a dictionary that maps the name of our features
to the feature transformation function.


```python
feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "previously_contacted": FeatureSpace.integer_categorical(num_oov_indices=0),
        # Categorical features encoded as string
        "marital": FeatureSpace.string_categorical(num_oov_indices=0),
        "education": FeatureSpace.string_categorical(num_oov_indices=0),
        "default": FeatureSpace.string_categorical(num_oov_indices=0),
        "housing": FeatureSpace.string_categorical(num_oov_indices=0),
        "loan": FeatureSpace.string_categorical(num_oov_indices=0),
        "contact": FeatureSpace.string_categorical(num_oov_indices=0),
        "month": FeatureSpace.string_categorical(num_oov_indices=0),
        "day_of_week": FeatureSpace.string_categorical(num_oov_indices=0),
        "poutcome": FeatureSpace.string_categorical(num_oov_indices=0),
        # Categorical features to hash and bin
        "job": FeatureSpace.string_hashed(num_bins=3),
        # Numerical features to hash and bin
        "pdays": FeatureSpace.integer_hashed(num_bins=4),
        # Numerical features to normalize and bin
        "age": FeatureSpace.float_discretized(num_bins=4),
        # Numerical features to normalize
        "campaign": FeatureSpace.float_normalized(),
        "previous": FeatureSpace.float_normalized(),
        "emp.var.rate": FeatureSpace.float_normalized(),
        "cons.price.idx": FeatureSpace.float_normalized(),
        "cons.conf.idx": FeatureSpace.float_normalized(),
        "euribor3m": FeatureSpace.float_normalized(),
        "nr.employed": FeatureSpace.float_normalized(),
    },
    # Specify feature cross with a custom crossing dim.
    crosses=[
        FeatureSpace.cross(feature_names=("age", "job"), crossing_dim=8),
        FeatureSpace.cross(feature_names=("housing", "loan"), crossing_dim=6),
        FeatureSpace.cross(
            feature_names=("poutcome", "previously_contacted"), crossing_dim=2
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
- Compute the value boundaries for the different bins for numerical features to
discretize.
- Any other kind of preprocessing required by custom layers.

Note that `adapt()` should be called on a `tf.data.Dataset` which yields dicts
of feature values -- no labels.

But first let's batch the datasets


```python
train_ds = train_ds.batch(32)
valid_ds = valid_ds.batch(32)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)
```

At this point, the `FeatureSpace` can be called on a dict of raw feature values, and
because we set `output_mode="concat"` it will return a single concatenate vector for each
sample, combining encoded features and feature crosses.


```python
for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print(f"preprocessed_x shape: {preprocessed_x.shape}")
    print(f"preprocessed_x sample: \n{preprocessed_x[0]}")
```

<div class="k-default-codeblock">
```
preprocessed_x shape: (32, 77)
preprocessed_x sample: 
[ 0.          1.          0.          0.         -0.19560693  0.95908785
 -0.22542837  1.          0.          0.          1.          0.
  0.          0.          1.          0.          0.          1.
  0.          0.          0.          0.          0.          0.
  0.          0.8486567   0.781508    1.          0.          0.
  0.          0.          1.          1.          0.          0.
  0.          1.          0.          0.          0.          0.
  1.          0.          0.          0.          0.          0.
  0.          0.          0.8400493   0.          0.          1.
  0.          1.          0.          0.         -0.35691845  1.
  0.          0.          0.          0.          0.          0.
  1.          0.          0.          0.          0.          0.
  0.          1.          0.          1.          0.        ]

```
</div>
---
## Saving the `FeatureSpace`

At this point we can choose to save our `FeatureSpace` component, this have many
advantages like re-using it on different experiments that use the same model, saving time
if you need to re-run the preprocessing step, and mainly for model deployment, where by
loading it you can be sure that you will be applying the same preprocessing steps don't
matter the device or environment, this is a great way to reduce
[training/servingskew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew).


```python
feature_space.save("myfeaturespace.keras")
```

---
## Preprocessing with `FeatureSpace` as part of the tf.data pipeline

We will opt to use our component asynchronously by making it part of the tf.data
pipeline, as noted at the
[previous guide](https://keras.io/examples/structured_data/structured_data_classification_with_feature_space/)
This enables asynchronous parallel preprocessing of the data on CPU before it
hits the model. Usually, this is always the right thing to do during training.

Let's create a training and validation dataset of preprocessed batches:


```python
preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

preprocessed_valid_ds = valid_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
```

---
## Model

We will take advantage of our `FeatureSpace` component to build the model, as we want the
model to be compatible with our preprocessing function, let's use the the `FeatureSpace`
feature map as the input of our model.


```python
encoded_features = feature_space.get_encoded_features()
print(encoded_features)
```

<div class="k-default-codeblock">
```
<KerasTensor shape=(None, 77), dtype=float32, sparse=False, name=keras_tensor_56>

```
</div>
This model is quite trivial only for demonstration purposes so don't pay too much
attention to the architecture.


```python
x = keras.layers.Dense(64, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
output = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=encoded_features, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

---
## Training

Let's train our model for 20 epochs. Note that feature preprocessing is happening as part
of the tf.data pipeline, not as part of the model.


```python
model.fit(
    preprocessed_train_ds, validation_data=preprocessed_valid_ds, epochs=20, verbose=2
)
```

<div class="k-default-codeblock">
```
Epoch 1/20
103/103 - 1s - 6ms/step - accuracy: 0.8844 - loss: 0.3453 - val_accuracy: 0.9114 - val_loss: 0.2612
Epoch 2/20
103/103 - 0s - 2ms/step - accuracy: 0.8974 - loss: 0.3010 - val_accuracy: 0.9078 - val_loss: 0.2641
Epoch 3/20
103/103 - 0s - 2ms/step - accuracy: 0.9005 - loss: 0.2863 - val_accuracy: 0.9066 - val_loss: 0.2630
Epoch 4/20
103/103 - 0s - 2ms/step - accuracy: 0.9002 - loss: 0.2925 - val_accuracy: 0.9053 - val_loss: 0.2653
Epoch 5/20
103/103 - 0s - 2ms/step - accuracy: 0.8995 - loss: 0.2893 - val_accuracy: 0.9078 - val_loss: 0.2624
Epoch 6/20
103/103 - 0s - 2ms/step - accuracy: 0.9002 - loss: 0.2866 - val_accuracy: 0.9078 - val_loss: 0.2628
Epoch 7/20
103/103 - 0s - 2ms/step - accuracy: 0.9026 - loss: 0.2868 - val_accuracy: 0.9090 - val_loss: 0.2621
Epoch 8/20
103/103 - 0s - 2ms/step - accuracy: 0.9023 - loss: 0.2802 - val_accuracy: 0.9078 - val_loss: 0.2623
Epoch 9/20
103/103 - 0s - 2ms/step - accuracy: 0.9047 - loss: 0.2743 - val_accuracy: 0.9078 - val_loss: 0.2628
Epoch 10/20
103/103 - 0s - 2ms/step - accuracy: 0.9062 - loss: 0.2761 - val_accuracy: 0.9090 - val_loss: 0.2650
Epoch 11/20
103/103 - 0s - 2ms/step - accuracy: 0.9050 - loss: 0.2729 - val_accuracy: 0.9090 - val_loss: 0.2668
Epoch 12/20
103/103 - 0s - 2ms/step - accuracy: 0.9029 - loss: 0.2699 - val_accuracy: 0.9078 - val_loss: 0.2670
Epoch 13/20
103/103 - 0s - 2ms/step - accuracy: 0.9056 - loss: 0.2671 - val_accuracy: 0.9078 - val_loss: 0.2641
Epoch 14/20
103/103 - 0s - 2ms/step - accuracy: 0.9032 - loss: 0.2750 - val_accuracy: 0.9078 - val_loss: 0.2643
Epoch 15/20
103/103 - 0s - 2ms/step - accuracy: 0.9083 - loss: 0.2650 - val_accuracy: 0.9102 - val_loss: 0.2658
Epoch 16/20
103/103 - 0s - 2ms/step - accuracy: 0.9102 - loss: 0.2593 - val_accuracy: 0.9102 - val_loss: 0.2639
Epoch 17/20
103/103 - 0s - 2ms/step - accuracy: 0.9074 - loss: 0.2719 - val_accuracy: 0.9102 - val_loss: 0.2655
Epoch 18/20
103/103 - 0s - 2ms/step - accuracy: 0.9059 - loss: 0.2655 - val_accuracy: 0.9102 - val_loss: 0.2670
Epoch 19/20
103/103 - 0s - 2ms/step - accuracy: 0.9099 - loss: 0.2650 - val_accuracy: 0.9102 - val_loss: 0.2646
Epoch 20/20
103/103 - 0s - 2ms/step - accuracy: 0.9068 - loss: 0.2624 - val_accuracy: 0.9078 - val_loss: 0.2661

<keras.src.callbacks.history.History at 0x31eac7eb0>

```
</div>
---
## Inference on new data with the end-to-end model

Now, we can build our inference model (which includes the `FeatureSpace`) to make
predictions based on dicts of raw features values, as follows:

### Loading the `FeatureSpace`

First let's load the `FeatureSpace` that we saved a few moment ago, this can be quite
handy if you train a model but want to do inference at different time, possibly using a
different device or environment.


```python
loaded_feature_space = keras.saving.load_model("myfeaturespace.keras")
```

### Building the inference end-to-end model

To build the inference model we need both the feature input map and the preprocessing
encoded Keras tensors.


```python
dict_inputs = loaded_feature_space.get_inputs()
encoded_features = loaded_feature_space.get_encoded_features()
print(encoded_features)

print(dict_inputs)

outputs = model(encoded_features)
inference_model = keras.Model(inputs=dict_inputs, outputs=outputs)

sample = {
    "age": 30,
    "job": "blue-collar",
    "marital": "married",
    "education": "basic.9y",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "fri",
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1,
    "previously_contacted": 0,
}

input_dict = {
    name: keras.ops.convert_to_tensor([value]) for name, value in sample.items()
}
predictions = inference_model.predict(input_dict)

print(
    f"This particular client has a {100 * predictions[0][0]:.2f}% probability "
    "of subscribing a term deposit, as evaluated by our model."
)
```

<div class="k-default-codeblock">
```
<KerasTensor shape=(None, 77), dtype=float32, sparse=False, name=keras_tensor_99>
{'previously_contacted': <KerasTensor shape=(None, 1), dtype=int32, sparse=None, name=previously_contacted>, 'marital': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=marital>, 'education': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=education>, 'default': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=default>, 'housing': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=housing>, 'loan': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=loan>, 'contact': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=contact>, 'month': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=month>, 'day_of_week': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=day_of_week>, 'poutcome': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=poutcome>, 'job': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=job>, 'pdays': <KerasTensor shape=(None, 1), dtype=int32, sparse=None, name=pdays>, 'age': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=age>, 'campaign': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=campaign>, 'previous': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=previous>, 'emp.var.rate': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=emp.var.rate>, 'cons.price.idx': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=cons.price.idx>, 'cons.conf.idx': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=cons.conf.idx>, 'euribor3m': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=euribor3m>, 'nr.employed': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=nr.employed>}
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 126ms/step
This particular client has a 9.60% probability of subscribing a term deposit, as evaluated by our model.

```
</div>