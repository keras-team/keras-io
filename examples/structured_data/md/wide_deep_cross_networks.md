# Structured data learning with Wide, Deep, and Cross networks

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2020/12/31<br>
**Last modified:** 2021/05/05<br>
**Description:** Using Wide & Deep and Deep & Cross networks for structured data classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/wide_deep_cross_networks.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/wide_deep_cross_networks.py)



---
## Introduction

This example demonstrates how to do structured data classification using the two modeling
techniques:

1. [Wide & Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) models
2. [Deep & Cross](https://arxiv.org/abs/1708.05123) models

Note that this example should be run with TensorFlow 2.5 or higher.

---
## The dataset

This example uses the [Covertype](https://archive.ics.uci.edu/ml/datasets/covertype) dataset from the UCI
Machine Learning Repository. The task is to predict forest cover type from cartographic variables.
The dataset includes 506,011 instances with 12 input features: 10 numerical features and 2
categorical features. Each instance is categorized into 1 of 7 classes.

---
## Setup


```python
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

---
## Prepare the data

First, let's load the dataset from the UCI Machine Learning Repository into a Pandas
DataFrame:


```python
data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
)
raw_data = pd.read_csv(data_url, header=None)
print(f"Dataset shape: {raw_data.shape}")
raw_data.head()
```

<div class="k-default-codeblock">
```
Dataset shape: (581012, 55)

```
</div>
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>6279</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>6225</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>6121</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>6211</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>6172</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>



The two categorical features in the dataset are binary-encoded.
We will convert this dataset representation to the typical representation, where each
categorical feature is represented as a single integer value.


```python
soil_type_values = [f"soil_type_{idx+1}" for idx in range(40)]
wilderness_area_values = [f"area_type_{idx+1}" for idx in range(4)]

soil_type = raw_data.loc[:, 14:53].apply(
    lambda x: soil_type_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
)
wilderness_area = raw_data.loc[:, 10:13].apply(
    lambda x: wilderness_area_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
)

CSV_HEADER = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area",
    "Soil_Type",
    "Cover_Type",
]

data = pd.concat(
    [raw_data.loc[:, 0:9], wilderness_area, soil_type, raw_data.loc[:, 54]],
    axis=1,
    ignore_index=True,
)
data.columns = CSV_HEADER

# Convert the target label indices into a range from 0 to 6 (there are 7 labels in total).
data["Cover_Type"] = data["Cover_Type"] - 1

print(f"Dataset shape: {data.shape}")
data.head().T
```

<div class="k-default-codeblock">
```
Dataset shape: (581012, 13)

```
</div>
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Elevation</th>
      <td>2596</td>
      <td>2590</td>
      <td>2804</td>
      <td>2785</td>
      <td>2595</td>
    </tr>
    <tr>
      <th>Aspect</th>
      <td>51</td>
      <td>56</td>
      <td>139</td>
      <td>155</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Slope</th>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>18</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Horizontal_Distance_To_Hydrology</th>
      <td>258</td>
      <td>212</td>
      <td>268</td>
      <td>242</td>
      <td>153</td>
    </tr>
    <tr>
      <th>Vertical_Distance_To_Hydrology</th>
      <td>0</td>
      <td>-6</td>
      <td>65</td>
      <td>118</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Horizontal_Distance_To_Roadways</th>
      <td>510</td>
      <td>390</td>
      <td>3180</td>
      <td>3090</td>
      <td>391</td>
    </tr>
    <tr>
      <th>Hillshade_9am</th>
      <td>221</td>
      <td>220</td>
      <td>234</td>
      <td>238</td>
      <td>220</td>
    </tr>
    <tr>
      <th>Hillshade_Noon</th>
      <td>232</td>
      <td>235</td>
      <td>238</td>
      <td>238</td>
      <td>234</td>
    </tr>
    <tr>
      <th>Hillshade_3pm</th>
      <td>148</td>
      <td>151</td>
      <td>135</td>
      <td>122</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <td>6279</td>
      <td>6225</td>
      <td>6121</td>
      <td>6211</td>
      <td>6172</td>
    </tr>
    <tr>
      <th>Wilderness_Area</th>
      <td>area_type_1</td>
      <td>area_type_1</td>
      <td>area_type_1</td>
      <td>area_type_1</td>
      <td>area_type_1</td>
    </tr>
    <tr>
      <th>Soil_Type</th>
      <td>soil_type_29</td>
      <td>soil_type_29</td>
      <td>soil_type_12</td>
      <td>soil_type_30</td>
      <td>soil_type_29</td>
    </tr>
    <tr>
      <th>Cover_Type</th>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



The shape of the DataFrame shows there are 13 columns per sample
(12 for the features and 1 for the target label).

Let's split the data into training (85%) and test (15%) sets.


```python
train_splits = []
test_splits = []

for _, group_data in data.groupby("Cover_Type"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.85
    train_splits.append(group_data[random_selection])
    test_splits.append(group_data[~random_selection])

train_data = pd.concat(train_splits).sample(frac=1).reset_index(drop=True)
test_data = pd.concat(test_splits).sample(frac=1).reset_index(drop=True)

print(f"Train split size: {len(train_data.index)}")
print(f"Test split size: {len(test_data.index)}")
```

<div class="k-default-codeblock">
```
Train split size: 493323
Test split size: 87689

```
</div>
Next, store the training and test data in separate CSV files.


```python
train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False)
test_data.to_csv(test_data_file, index=False)
```

---
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for reading and parsing
the data into input features, and encoding the input features with respect to their types.


```python
TARGET_FEATURE_NAME = "Cover_Type"

TARGET_FEATURE_LABELS = ["0", "1", "2", "3", "4", "5", "6"]

NUMERIC_FEATURE_NAMES = [
    "Aspect",
    "Elevation",
    "Hillshade_3pm",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Horizontal_Distance_To_Fire_Points",
    "Horizontal_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Slope",
    "Vertical_Distance_To_Hydrology",
]

CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "Soil_Type": list(data["Soil_Type"].unique()),
    "Wilderness_Area": list(data["Wilderness_Area"].unique()),
}

CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

COLUMN_DEFAULTS = [
    [0] if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] else ["NA"]
    for feature_name in CSV_HEADER
]

NUM_CLASSES = len(TARGET_FEATURE_LABELS)
```

---
## Experiment setup

Next, let's define an input function that reads and parses the file, then converts features
and labels into a[`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets)
for training or evaluation.


```python

def get_dataset_from_csv(csv_file_path, batch_size, shuffle=False):

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=True,
        shuffle=shuffle,
    )
    return dataset.cache()

```

Here we configure the parameters and implement the procedure for running a training and
evaluation experiment given a model.


```python
learning_rate = 0.001
dropout_rate = 0.1
batch_size = 265
num_epochs = 50

hidden_units = [32, 32]


def run_experiment(model):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)

    test_dataset = get_dataset_from_csv(test_data_file, batch_size)

    print("Start training the model...")
    history = model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")

    _, accuracy = model.evaluate(test_dataset, verbose=0)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

```

---
## Create model inputs

Now, define the inputs for the models as a dictionary, where the key is the feature name,
and the value is a `keras.layers.Input` tensor with the corresponding feature shape
and data type.


```python

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs

```

---
## Encode features

We create two representations of our input features: sparse and dense:
1. In the **sparse** representation, the categorical features are encoded with one-hot
encoding using the `CategoryEncoding` layer. This representation can be useful for the
model to *memorize* particular feature values to make certain predictions.
2. In the **dense** representation, the categorical features are encoded with
low-dimensional embeddings using the `Embedding` layer. This representation helps
the model to *generalize* well to unseen feature combinations.


```python

from tensorflow.keras.layers import StringLookup


def encode_inputs(inputs, use_embedding=False):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int" if use_embedding else "binary",
            )
            if use_embedding:
                # Convert the string input values into integer indices.
                encoded_feature = lookup(inputs[feature_name])
                embedding_dims = int(math.sqrt(len(vocabulary)))
                # Create an embedding layer with the specified dimensions.
                embedding = layers.Embedding(
                    input_dim=len(vocabulary), output_dim=embedding_dims
                )
                # Convert the index values to embedding representations.
                encoded_feature = embedding(encoded_feature)
            else:
                # Convert the string input values into a one hot encoding.
                encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))
        else:
            # Use the numerical features as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)

        encoded_features.append(encoded_feature)

    all_features = layers.concatenate(encoded_features)
    return all_features

```

---
## Experiment 1: a baseline model

In the first experiment, let's create a multi-layer feed-forward network,
where the categorical features are one-hot encoded.


```python

def create_baseline_model():
    inputs = create_model_inputs()
    features = encode_inputs(inputs)

    for units in hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


baseline_model = create_baseline_model()
keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")
```

<div class="k-default-codeblock">
```
('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')

```
</div>
Let's run it:


```python
run_experiment(baseline_model)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/50
1862/1862 [==============================] - 10s 5ms/step - loss: 0.9208 - sparse_categorical_accuracy: 0.6334
Epoch 2/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.6758 - sparse_categorical_accuracy: 0.7081
Epoch 3/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.6409 - sparse_categorical_accuracy: 0.7225
Epoch 4/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.6209 - sparse_categorical_accuracy: 0.7316
Epoch 5/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.6074 - sparse_categorical_accuracy: 0.7371
Epoch 6/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5975 - sparse_categorical_accuracy: 0.7419
Epoch 7/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5889 - sparse_categorical_accuracy: 0.7458
Epoch 8/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5846 - sparse_categorical_accuracy: 0.7474
Epoch 9/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5810 - sparse_categorical_accuracy: 0.7502
Epoch 10/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5789 - sparse_categorical_accuracy: 0.7502
Epoch 11/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5746 - sparse_categorical_accuracy: 0.7528
Epoch 12/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5718 - sparse_categorical_accuracy: 0.7540
Epoch 13/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5689 - sparse_categorical_accuracy: 0.7551
Epoch 14/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5671 - sparse_categorical_accuracy: 0.7558
Epoch 15/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5650 - sparse_categorical_accuracy: 0.7568
Epoch 16/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5623 - sparse_categorical_accuracy: 0.7577
Epoch 17/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5616 - sparse_categorical_accuracy: 0.7591
Epoch 18/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5583 - sparse_categorical_accuracy: 0.7590
Epoch 19/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5577 - sparse_categorical_accuracy: 0.7593
Epoch 20/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5549 - sparse_categorical_accuracy: 0.7608
Epoch 21/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5564 - sparse_categorical_accuracy: 0.7599
Epoch 22/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5554 - sparse_categorical_accuracy: 0.7606
Epoch 23/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5537 - sparse_categorical_accuracy: 0.7617
Epoch 24/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5518 - sparse_categorical_accuracy: 0.7624
Epoch 25/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5508 - sparse_categorical_accuracy: 0.7618
Epoch 26/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5498 - sparse_categorical_accuracy: 0.7621
Epoch 27/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5497 - sparse_categorical_accuracy: 0.7623
Epoch 28/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5482 - sparse_categorical_accuracy: 0.7645
Epoch 29/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5467 - sparse_categorical_accuracy: 0.7637
Epoch 30/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5469 - sparse_categorical_accuracy: 0.7638
Epoch 31/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5457 - sparse_categorical_accuracy: 0.7641
Epoch 32/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5448 - sparse_categorical_accuracy: 0.7647
Epoch 33/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5440 - sparse_categorical_accuracy: 0.7644
Epoch 34/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5448 - sparse_categorical_accuracy: 0.7653
Epoch 35/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5424 - sparse_categorical_accuracy: 0.7652
Epoch 36/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5416 - sparse_categorical_accuracy: 0.7666
Epoch 37/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5411 - sparse_categorical_accuracy: 0.7663
Epoch 38/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5399 - sparse_categorical_accuracy: 0.7673
Epoch 39/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5410 - sparse_categorical_accuracy: 0.7664
Epoch 40/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5402 - sparse_categorical_accuracy: 0.7668
Epoch 41/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5395 - sparse_categorical_accuracy: 0.7670
Epoch 42/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5382 - sparse_categorical_accuracy: 0.7679
Epoch 43/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5369 - sparse_categorical_accuracy: 0.7680
Epoch 44/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5370 - sparse_categorical_accuracy: 0.7686
Epoch 45/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5358 - sparse_categorical_accuracy: 0.7680
Epoch 46/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5358 - sparse_categorical_accuracy: 0.7698
Epoch 47/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5363 - sparse_categorical_accuracy: 0.7697
Epoch 48/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5349 - sparse_categorical_accuracy: 0.7691
Epoch 49/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5357 - sparse_categorical_accuracy: 0.7691
Epoch 50/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5338 - sparse_categorical_accuracy: 0.7697
Model training finished
Test accuracy: 75.72%

```
</div>
The baseline linear model achieves ~76% test accuracy.

---
## Experiment 2: Wide & Deep model

In the second experiment, we create a Wide & Deep model. The wide part of the model
a linear model, while the deep part of the model is a multi-layer feed-forward network.

Use the sparse representation of the input features in the wide part of the model and the
dense representation of the input features for the deep part of the model.

Note that every input features contributes to both parts of the model with different
representations.


```python

def create_wide_and_deep_model():

    inputs = create_model_inputs()
    wide = encode_inputs(inputs)
    wide = layers.BatchNormalization()(wide)

    deep = encode_inputs(inputs, use_embedding=True)
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([wide, deep])
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


wide_and_deep_model = create_wide_and_deep_model()
keras.utils.plot_model(wide_and_deep_model, show_shapes=True, rankdir="LR")
```

<div class="k-default-codeblock">
```
('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')

```
</div>
Let's run it:


```python
run_experiment(wide_and_deep_model)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/50
1862/1862 [==============================] - 11s 5ms/step - loss: 0.8994 - sparse_categorical_accuracy: 0.6469
Epoch 2/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.6112 - sparse_categorical_accuracy: 0.7350
Epoch 3/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5936 - sparse_categorical_accuracy: 0.7426
Epoch 4/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5814 - sparse_categorical_accuracy: 0.7468
Epoch 5/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5716 - sparse_categorical_accuracy: 0.7517
Epoch 6/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5652 - sparse_categorical_accuracy: 0.7553
Epoch 7/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5595 - sparse_categorical_accuracy: 0.7581
Epoch 8/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5542 - sparse_categorical_accuracy: 0.7600
Epoch 9/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5498 - sparse_categorical_accuracy: 0.7631
Epoch 10/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5459 - sparse_categorical_accuracy: 0.7647
Epoch 11/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5427 - sparse_categorical_accuracy: 0.7655
Epoch 12/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5398 - sparse_categorical_accuracy: 0.7675
Epoch 13/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5360 - sparse_categorical_accuracy: 0.7695
Epoch 14/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5335 - sparse_categorical_accuracy: 0.7697
Epoch 15/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5310 - sparse_categorical_accuracy: 0.7709
Epoch 16/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5289 - sparse_categorical_accuracy: 0.7725
Epoch 17/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5263 - sparse_categorical_accuracy: 0.7739
Epoch 18/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5255 - sparse_categorical_accuracy: 0.7745
Epoch 19/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5235 - sparse_categorical_accuracy: 0.7750
Epoch 20/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5224 - sparse_categorical_accuracy: 0.7757
Epoch 21/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5216 - sparse_categorical_accuracy: 0.7770
Epoch 22/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5205 - sparse_categorical_accuracy: 0.7771
Epoch 23/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5191 - sparse_categorical_accuracy: 0.7769
Epoch 24/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5189 - sparse_categorical_accuracy: 0.7779
Epoch 25/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5166 - sparse_categorical_accuracy: 0.7793
Epoch 26/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5160 - sparse_categorical_accuracy: 0.7794
Epoch 27/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5146 - sparse_categorical_accuracy: 0.7791
Epoch 28/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5136 - sparse_categorical_accuracy: 0.7810
Epoch 29/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5125 - sparse_categorical_accuracy: 0.7809
Epoch 30/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5124 - sparse_categorical_accuracy: 0.7806
Epoch 31/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5112 - sparse_categorical_accuracy: 0.7808
Epoch 32/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5098 - sparse_categorical_accuracy: 0.7822
Epoch 33/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5097 - sparse_categorical_accuracy: 0.7808
Epoch 34/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5094 - sparse_categorical_accuracy: 0.7819
Epoch 35/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5084 - sparse_categorical_accuracy: 0.7823
Epoch 36/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5077 - sparse_categorical_accuracy: 0.7826
Epoch 37/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5067 - sparse_categorical_accuracy: 0.7830
Epoch 38/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5063 - sparse_categorical_accuracy: 0.7834
Epoch 39/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5058 - sparse_categorical_accuracy: 0.7841
Epoch 40/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5047 - sparse_categorical_accuracy: 0.7840
Epoch 41/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5041 - sparse_categorical_accuracy: 0.7848
Epoch 42/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5024 - sparse_categorical_accuracy: 0.7857
Epoch 43/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5020 - sparse_categorical_accuracy: 0.7857
Epoch 44/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5009 - sparse_categorical_accuracy: 0.7865
Epoch 45/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4998 - sparse_categorical_accuracy: 0.7868
Epoch 46/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5000 - sparse_categorical_accuracy: 0.7864
Epoch 47/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4985 - sparse_categorical_accuracy: 0.7876
Epoch 48/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4985 - sparse_categorical_accuracy: 0.7877
Epoch 49/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4979 - sparse_categorical_accuracy: 0.7876
Epoch 50/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4973 - sparse_categorical_accuracy: 0.7881
Model training finished
Test accuracy: 80.69%

```
</div>
The wide and deep model achieves ~79% test accuracy.

---
## Experiment 3: Deep & Cross model

In the third experiment, we create a Deep & Cross model. The deep part of this model
is the same as the deep part created in the previous experiment. The key idea of
the cross part is to apply explicit feature crossing in an efficient way,
where the degree of cross features grows with layer depth.


```python

def create_deep_and_cross_model():

    inputs = create_model_inputs()
    x0 = encode_inputs(inputs, use_embedding=True)

    cross = x0
    for _ in hidden_units:
        units = cross.shape[-1]
        x = layers.Dense(units)(cross)
        cross = x0 * x + cross
    cross = layers.BatchNormalization()(cross)

    deep = x0
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([cross, deep])
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


deep_and_cross_model = create_deep_and_cross_model()
keras.utils.plot_model(deep_and_cross_model, show_shapes=True, rankdir="LR")
```

<div class="k-default-codeblock">
```
('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')

```
</div>
Let's run it:


```python
run_experiment(deep_and_cross_model)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/50
1862/1862 [==============================] - 11s 5ms/step - loss: 0.8585 - sparse_categorical_accuracy: 0.6547
Epoch 2/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5968 - sparse_categorical_accuracy: 0.7424
Epoch 3/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5729 - sparse_categorical_accuracy: 0.7520
Epoch 4/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5610 - sparse_categorical_accuracy: 0.7583
Epoch 5/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5511 - sparse_categorical_accuracy: 0.7623
Epoch 6/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5460 - sparse_categorical_accuracy: 0.7651
Epoch 7/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5408 - sparse_categorical_accuracy: 0.7671
Epoch 8/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5374 - sparse_categorical_accuracy: 0.7695
Epoch 9/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5344 - sparse_categorical_accuracy: 0.7704
Epoch 10/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5310 - sparse_categorical_accuracy: 0.7715
Epoch 11/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5286 - sparse_categorical_accuracy: 0.7725
Epoch 12/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5254 - sparse_categorical_accuracy: 0.7737
Epoch 13/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5249 - sparse_categorical_accuracy: 0.7737
Epoch 14/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5223 - sparse_categorical_accuracy: 0.7752
Epoch 15/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5206 - sparse_categorical_accuracy: 0.7759
Epoch 16/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5187 - sparse_categorical_accuracy: 0.7765
Epoch 17/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5179 - sparse_categorical_accuracy: 0.7772
Epoch 18/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5152 - sparse_categorical_accuracy: 0.7788
Epoch 19/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5145 - sparse_categorical_accuracy: 0.7785
Epoch 20/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5128 - sparse_categorical_accuracy: 0.7800
Epoch 21/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5117 - sparse_categorical_accuracy: 0.7803
Epoch 22/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5105 - sparse_categorical_accuracy: 0.7809
Epoch 23/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5089 - sparse_categorical_accuracy: 0.7813
Epoch 24/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5074 - sparse_categorical_accuracy: 0.7823
Epoch 25/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5061 - sparse_categorical_accuracy: 0.7821
Epoch 26/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5048 - sparse_categorical_accuracy: 0.7832
Epoch 27/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5037 - sparse_categorical_accuracy: 0.7837
Epoch 28/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5017 - sparse_categorical_accuracy: 0.7846
Epoch 29/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.5010 - sparse_categorical_accuracy: 0.7851
Epoch 30/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4991 - sparse_categorical_accuracy: 0.7861
Epoch 31/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4989 - sparse_categorical_accuracy: 0.7849
Epoch 32/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4979 - sparse_categorical_accuracy: 0.7865
Epoch 33/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4961 - sparse_categorical_accuracy: 0.7867
Epoch 34/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4955 - sparse_categorical_accuracy: 0.7871
Epoch 35/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4946 - sparse_categorical_accuracy: 0.7871
Epoch 36/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4946 - sparse_categorical_accuracy: 0.7873
Epoch 37/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4925 - sparse_categorical_accuracy: 0.7877
Epoch 38/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4920 - sparse_categorical_accuracy: 0.7884
Epoch 39/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4910 - sparse_categorical_accuracy: 0.7887
Epoch 40/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4909 - sparse_categorical_accuracy: 0.7883
Epoch 41/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4906 - sparse_categorical_accuracy: 0.7890
Epoch 42/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4883 - sparse_categorical_accuracy: 0.7892
Epoch 43/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4883 - sparse_categorical_accuracy: 0.7896
Epoch 44/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4875 - sparse_categorical_accuracy: 0.7908
Epoch 45/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4866 - sparse_categorical_accuracy: 0.7900
Epoch 46/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4864 - sparse_categorical_accuracy: 0.7902
Epoch 47/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4862 - sparse_categorical_accuracy: 0.7909
Epoch 48/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4849 - sparse_categorical_accuracy: 0.7908
Epoch 49/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4843 - sparse_categorical_accuracy: 0.7910
Epoch 50/50
1862/1862 [==============================] - 5s 3ms/step - loss: 0.4841 - sparse_categorical_accuracy: 0.7921
Model training finished
Test accuracy: 80.61%

```
</div>
The deep and cross model achieves ~81% test accuracy.

---
## Conclusion

You can use Keras Preprocessing Layers to easily handle categorical features
with different encoding mechanisms, including one-hot encoding and feature embedding.
In addition, different model architectures — like wide, deep, and cross networks
— have different advantages, with respect to different dataset properties.
You can explore using them independently or combining them to achieve the best result
for your dataset.
