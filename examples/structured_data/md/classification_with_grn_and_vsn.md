# Classification with Gated Residual and Variable Selection Networks

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/02/10<br>
**Last modified:** 2021/02/10<br>
**Description:** Using Gated Residual and Variable Selection Networks for income level prediction.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/classification_with_grn_and_vsn.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/classification_with_grn_and_vsn.py)



---
## Introduction

This example demonstrates the use of Gated
Residual Networks (GRN) and Variable Selection Networks (VSN), proposed by
Bryan Lim et al. in
[Temporal Fusion Transformers (TFT) for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363),
for structured data classification. GRNs give the flexibility to the model to apply
non-linear processing only where needed. VSNs allow the model to softly remove any
unnecessary noisy inputs which could negatively impact performance.
Together, those techniques help improving the learning capacity of deep neural
network models.

Note that this example implements only the GRN and VSN components described in
in the paper, rather than the whole TFT model, as GRN and VSN can be useful on
their own for structured data learning tasks.


To run the code you need to use TensorFlow 2.3 or higher.

---
## The dataset

This example uses the
[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29)
provided by the
[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The task is binary classification to determine whether a person makes over 50K a year.

The dataset includes ~300K instances with 41 input features: 7 numerical features
and 34 categorical features.

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

First we load the data from the UCI Machine Learning Repository into a Pandas DataFrame.


```python
# Column names.
CSV_HEADER = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_stat",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "instance_weight",
    "migration_code-change_in_msa",
    "migration_code-change_in_reg",
    "migration_code-move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veteran's_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "income_level",
]

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"
data = pd.read_csv(data_url, header=None, names=CSV_HEADER)

test_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz"
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)

print(f"Data shape: {data.shape}")
print(f"Test data shape: {test_data.shape}")

```

<div class="k-default-codeblock">
```
Data shape: (199523, 42)
Test data shape: (99762, 42)

```
</div>
We convert the target column from string to integer.


```python
data["income_level"] = data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)
test_data["income_level"] = test_data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)

```

Then, We split the dataset into train and validation sets.


```python
random_selection = np.random.rand(len(data.index)) <= 0.85
train_data = data[random_selection]
valid_data = data[~random_selection]

```

Finally we store the train and test data splits locally to CSV files.


```python
train_data_file = "train_data.csv"
valid_data_file = "valid_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
valid_data.to_csv(valid_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)
```

---
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for reading and
parsing the data into input features, and encoding the input features with respect
to their types.


```python
# Target feature name.
TARGET_FEATURE_NAME = "income_level"
# Weight column name.
WEIGHT_COLUMN_NAME = "instance_weight"
# Numeric feature names.
NUMERIC_FEATURE_NAMES = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]
# Categorical features and their vocabulary lists.
# Note that we add 'v=' as a prefix to all categorical feature values to make
# sure that they are treated as strings.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    feature_name: sorted([str(value) for value in list(data[feature_name].unique())])
    for feature_name in CSV_HEADER
    if feature_name
    not in list(NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME, TARGET_FEATURE_NAME])
}
# All features names.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(
    CATEGORICAL_FEATURES_WITH_VOCABULARY.keys()
)
# Feature default values.
COLUMN_DEFAULTS = [
    [0.0]
    if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME, WEIGHT_COLUMN_NAME]
    else ["NA"]
    for feature_name in CSV_HEADER
]
```

---
## Create a `tf.data.Dataset` for training and evaluation

We create an input function to read and parse the file, and convert features and
labels into a [`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets) for
training and evaluation.


```python
from tensorflow.keras.layers import StringLookup


def process(features, target):
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # Cast categorical feature values to string.
            features[feature_name] = tf.cast(features[feature_name], tf.dtypes.string)
    # Get the instance weight.
    weight = features.pop(WEIGHT_COLUMN_NAME)
    return features, target, weight


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        shuffle=shuffle,
    ).map(process)

    return dataset

```

---
## Create model inputs


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
## Encode input features

For categorical features, we encode them using `layers.Embedding` using the
`encoding_size` as the embedding dimensions. For the numerical features,
we apply linear transformation using `layers.Dense` to project each feature into
`encoding_size`-dimensional vector. Thus, all the encoded features will have the
same dimensionality.


```python

def encode_inputs(inputs, encoding_size):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            index = StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_index = index(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            embedding_ecoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=encoding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = embedding_ecoder(value_index)
        else:
            # Project the numeric feature to encoding_size using linear transformation.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features

```

---
## Implement the Gated Linear Unit

[Gated Linear Units (GLUs)](https://arxiv.org/abs/1612.08083) provide the
flexibility to suppress input that are not relevant for a given task.


```python

class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

```

---
## Implement the Gated Residual Network

The Gated Residual Network (GRN) works as follows:

1. Applies the nonlinear ELU transformation to the inputs.
2. Applies linear transformation followed by dropout.
4. Applies GLU and adds the original inputs to the output of the GLU to perform skip
(residual) connection.
6. Applies layer normalization and produces the output.


```python

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

```

---
## Implement the Variable Selection Network

The Variable Selection Network (VSN) works as follows:

1. Applies a GRN to each feature individually.
2. Applies a GRN on the concatenation of all the features, followed by a softmax to
produce feature weights.
3. Produces a weighted sum of the output of the individual GRN.

Note that the output of the VSN is [batch_size, encoding_size], regardless of the
number of the input features.


```python

class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelection, self).__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs

```

---
## Create Gated Residual and Variable Selection Networks model


```python

def create_model(encoding_size):
    inputs = create_model_inputs()
    feature_list = encode_inputs(inputs, encoding_size)
    num_features = len(feature_list)

    features = VariableSelection(num_features, encoding_size, dropout_rate)(
        feature_list
    )

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

---
## Compile, train, and evaluate the model


```python
learning_rate = 0.001
dropout_rate = 0.15
batch_size = 265
num_epochs = 20
encoding_size = 16

model = create_model(encoding_size)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)


# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("Start training the model...")
train_dataset = get_dataset_from_csv(
    train_data_file, shuffle=True, batch_size=batch_size
)
valid_dataset = get_dataset_from_csv(valid_data_file, batch_size=batch_size)
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)
print("Model training finished.")

print("Evaluating model performance...")
test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)
_, accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/20
641/641 [==============================] - 26s 22ms/step - loss: 317.7028 - accuracy: 0.9353 - val_loss: 230.1805 - val_accuracy: 0.9497
Epoch 2/20
641/641 [==============================] - 13s 19ms/step - loss: 231.4161 - accuracy: 0.9506 - val_loss: 224.7825 - val_accuracy: 0.9498
Epoch 3/20
641/641 [==============================] - 12s 19ms/step - loss: 226.8173 - accuracy: 0.9503 - val_loss: 223.0818 - val_accuracy: 0.9508
Epoch 4/20
641/641 [==============================] - 13s 19ms/step - loss: 224.1516 - accuracy: 0.9507 - val_loss: 221.8637 - val_accuracy: 0.9509
Epoch 5/20
641/641 [==============================] - 13s 19ms/step - loss: 223.9696 - accuracy: 0.9507 - val_loss: 217.8728 - val_accuracy: 0.9513
Epoch 6/20
641/641 [==============================] - 13s 19ms/step - loss: 220.7267 - accuracy: 0.9508 - val_loss: 220.2448 - val_accuracy: 0.9516
Epoch 7/20
641/641 [==============================] - 13s 19ms/step - loss: 219.7464 - accuracy: 0.9514 - val_loss: 216.4628 - val_accuracy: 0.9516
Epoch 8/20
641/641 [==============================] - 13s 19ms/step - loss: 218.7294 - accuracy: 0.9517 - val_loss: 215.2192 - val_accuracy: 0.9519
Epoch 9/20
641/641 [==============================] - 12s 19ms/step - loss: 218.3938 - accuracy: 0.9516 - val_loss: 217.1790 - val_accuracy: 0.9514
Epoch 10/20
641/641 [==============================] - 13s 19ms/step - loss: 217.2871 - accuracy: 0.9522 - val_loss: 213.4623 - val_accuracy: 0.9523
Epoch 11/20
641/641 [==============================] - 13s 19ms/step - loss: 215.0476 - accuracy: 0.9522 - val_loss: 211.6762 - val_accuracy: 0.9523
Epoch 12/20
641/641 [==============================] - 13s 19ms/step - loss: 213.2402 - accuracy: 0.9527 - val_loss: 212.2001 - val_accuracy: 0.9525
Epoch 13/20
641/641 [==============================] - 13s 20ms/step - loss: 212.8123 - accuracy: 0.9530 - val_loss: 207.9878 - val_accuracy: 0.9538
Epoch 14/20
641/641 [==============================] - 13s 19ms/step - loss: 208.4605 - accuracy: 0.9541 - val_loss: 208.0063 - val_accuracy: 0.9543
Epoch 15/20
641/641 [==============================] - 13s 19ms/step - loss: 211.9185 - accuracy: 0.9533 - val_loss: 208.2112 - val_accuracy: 0.9540
Epoch 16/20
641/641 [==============================] - 13s 19ms/step - loss: 207.7694 - accuracy: 0.9544 - val_loss: 207.3279 - val_accuracy: 0.9547
Epoch 17/20
641/641 [==============================] - 13s 19ms/step - loss: 208.6964 - accuracy: 0.9540 - val_loss: 204.3082 - val_accuracy: 0.9553
Epoch 18/20
641/641 [==============================] - 13s 19ms/step - loss: 207.2199 - accuracy: 0.9547 - val_loss: 206.4799 - val_accuracy: 0.9549
Epoch 19/20
641/641 [==============================] - 13s 19ms/step - loss: 206.7960 - accuracy: 0.9548 - val_loss: 206.0898 - val_accuracy: 0.9555
Epoch 20/20
641/641 [==============================] - 13s 20ms/step - loss: 206.2721 - accuracy: 0.9547 - val_loss: 206.6541 - val_accuracy: 0.9549
Model training finished.
Evaluating model performance...
377/377 [==============================] - 5s 11ms/step - loss: 206.3511 - accuracy: 0.9541
Test accuracy: 95.41%

```
</div>
You should achieve more than 95% accuracy on the test set.

To increase the learning capacity of the model, you can try increasing the
`encoding_size` value, or stacking multiple GRN layers on top of the VSN layer.
This may require to also increase the `dropout_rate` value to avoid overfitting.
