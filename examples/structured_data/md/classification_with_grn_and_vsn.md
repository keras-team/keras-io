# Classification with Gated Residual and Variable Selection Networks

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/02/10<br>
**Last modified:** 2025/01/08<br>
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
import os
import subprocess
import tarfile

os.environ["KERAS_BACKEND"] = "torch"  # or jax, or tensorflow

import numpy as np
import pandas as pd
import keras
from keras import layers
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
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "income_level",
]

data_url = "https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip"
keras.utils.get_file(origin=data_url, extract=True)
```




<div class="k-default-codeblock">
```
'/home/humbulani/.keras/datasets/census+income+kdd.zip'

```
</div>
Determine the downloaded .tar.gz file path and
extract the files from the downloaded .tar.gz file


```python
extracted_path = os.path.join(
    os.path.expanduser("~"), ".keras", "datasets", "census+income+kdd.zip"
)
for root, dirs, files in os.walk(extracted_path):
    for file in files:
        if file.endswith(".tar.gz"):
            tar_gz_path = os.path.join(root, file)
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=root)

train_data_path = os.path.join(
    os.path.expanduser("~"),
    ".keras",
    "datasets",
    "census+income+kdd.zip",
    "census-income.data",
)
test_data_path = os.path.join(
    os.path.expanduser("~"),
    ".keras",
    "datasets",
    "census+income+kdd.zip",
    "census-income.test",
)

data = pd.read_csv(train_data_path, header=None, names=CSV_HEADER)
test_data = pd.read_csv(test_data_path, header=None, names=CSV_HEADER)

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
    (
        [0.0]
        if feature_name
        in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME, WEIGHT_COLUMN_NAME]
        else ["NA"]
    )
    for feature_name in CSV_HEADER
]
```

---
## Create a `tf.data.Dataset` for training and evaluation

We create an input function to read and parse the file, and convert features and
labels into a [`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets) for
training and evaluation.


```python
# Tensorflow required for tf.data.Datasets
import tensorflow as tf


# We process our datasets elements here (categorical) and convert them to indices to avoid this step
# during model training since only tensorflow support strings.
def process(features, target):
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # Cast categorical feature values to string.
            features[feature_name] = tf.cast(features[feature_name], "string")
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            index = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )
            # Convert the string input values into integer indices.
            value_index = index(features[feature_name])
            features[feature_name] = value_index
        else:
            # Do nothing for numerical features
            pass

    # Get the instance weight.
    weight = features.pop(WEIGHT_COLUMN_NAME)
    # Change features from OrderedDict to Dict to match Inputs as they are Dict.
    return dict(features), target, weight


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
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # Make them int64, they are Categorical (whole units)
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="int64"
            )
        else:
            # Make them float32, they are Real numbers
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="float32"
            )
    return inputs

```

---
## Implement the Gated Linear Unit

[Gated Linear Units (GLUs)](https://arxiv.org/abs/1612.08083) provide the
flexibility to suppress input that are not relevant for a given task.


```python

class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

    # Remove build warnings
    def build(self):
        self.built = True

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
        super().__init__()
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

    # Remove build warnings
    def build(self):
        self.built = True

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

For categorical features, we encode them using `layers.Embedding` using the
`encoding_size` as the embedding dimensions. For the numerical features,
we apply linear transformation using `layers.Dense` to project each feature into
`encoding_size`-dimensional vector. Thus, all the encoded features will have the
same dimensionality.


```python

class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super().__init__()
        self.units = units
        # Create an embedding layers with the specified dimensions
        self.embeddings = dict()
        for input_ in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[input_]
            embedding_encoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=self.units, name=input_
            )
            self.embeddings[input_] = embedding_encoder

        # Projection layers for numeric features
        self.proj_layer = dict()
        for input_ in NUMERIC_FEATURE_NAMES:
            proj_layer = layers.Dense(units=self.units)
            self.proj_layer[input_] = proj_layer

        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        concat_inputs = []
        for input_ in inputs:
            if input_ in CATEGORICAL_FEATURES_WITH_VOCABULARY:
                max_index = self.embeddings[input_].input_dim - 1  # Clamp the indices
                # torch had some index errors during embedding hence the clip function
                embedded_feature = self.embeddings[input_](
                    keras.ops.clip(inputs[input_], 0, max_index)
                )
                concat_inputs.append(embedded_feature)
            else:
                # Project the numeric feature to encoding_size using linear transformation.
                proj_feature = keras.ops.expand_dims(inputs[input_], -1)
                proj_feature = self.proj_layer[input_](proj_feature)
                concat_inputs.append(proj_feature)

        v = layers.concatenate(concat_inputs)
        v = self.grn_concat(v)
        v = keras.ops.expand_dims(self.softmax(v), axis=-1)
        x = []
        for idx, input in enumerate(concat_inputs):
            x.append(self.grns[idx](input))
        x = keras.ops.stack(x, axis=1)
        return keras.ops.squeeze(
            keras.ops.matmul(keras.ops.transpose(v, axes=[0, 2, 1]), x), axis=1
        )

    # to remove the build warnings
    def build(self):
        self.built = True

```

---
## Create Gated Residual and Variable Selection Networks model


```python

def create_model(encoding_size):
    inputs = create_model_inputs()
    num_features = len(inputs)
    features = VariableSelection(num_features, encoding_size, dropout_rate)(inputs)
    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    # Functional model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

---
## Compile, train, and evaluate the model


```python
learning_rate = 0.001
dropout_rate = 0.15
batch_size = 265
num_epochs = 20  # may be adjusted to a desired value
encoding_size = 16

model = create_model(encoding_size)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)
```

Let's visualize our connectivity graph:


```python
# `rankdir='LR'` is to make the graph horizontal.
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir="LR")


# Create an early stopping callback.
early_stopping = keras.callbacks.EarlyStopping(
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

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  1s 698ms/step - accuracy: 0.4717 - loss: 1212.3043


  2/Unknown  1s 200ms/step - accuracy: 0.5745 - loss: 1141.6052


  3/Unknown  1s 195ms/step - accuracy: 0.6388 - loss: 1084.4358


  4/Unknown  1s 199ms/step - accuracy: 0.6822 - loss: 1031.0354


  5/Unknown  2s 201ms/step - accuracy: 0.7131 - loss: 986.4984 


  6/Unknown  2s 197ms/step - accuracy: 0.7363 - loss: 947.2644


  7/Unknown  2s 190ms/step - accuracy: 0.7546 - loss: 912.4213


  8/Unknown  2s 188ms/step - accuracy: 0.7698 - loss: 881.4526


  9/Unknown  2s 186ms/step - accuracy: 0.7824 - loss: 853.8523


 10/Unknown  2s 184ms/step - accuracy: 0.7932 - loss: 829.0496


 11/Unknown  3s 183ms/step - accuracy: 0.8022 - loss: 807.4752


 12/Unknown  3s 184ms/step - accuracy: 0.8100 - loss: 788.1222


 13/Unknown  3s 187ms/step - accuracy: 0.8170 - loss: 770.3723


 14/Unknown  3s 187ms/step - accuracy: 0.8233 - loss: 753.6734


 15/Unknown  3s 186ms/step - accuracy: 0.8289 - loss: 737.9523


 16/Unknown  3s 186ms/step - accuracy: 0.8342 - loss: 723.0760


 17/Unknown  4s 186ms/step - accuracy: 0.8389 - loss: 709.2202


 18/Unknown  4s 202ms/step - accuracy: 0.8432 - loss: 696.8585


 19/Unknown  4s 200ms/step - accuracy: 0.8470 - loss: 685.7762


 20/Unknown  4s 198ms/step - accuracy: 0.8505 - loss: 675.3044


 21/Unknown  5s 197ms/step - accuracy: 0.8537 - loss: 665.8409


 22/Unknown  5s 196ms/step - accuracy: 0.8566 - loss: 657.3629


 23/Unknown  5s 195ms/step - accuracy: 0.8593 - loss: 649.5444


 24/Unknown  5s 195ms/step - accuracy: 0.8618 - loss: 642.1780


 25/Unknown  5s 194ms/step - accuracy: 0.8641 - loss: 635.1900


 26/Unknown  6s 195ms/step - accuracy: 0.8662 - loss: 628.5919


 27/Unknown  6s 195ms/step - accuracy: 0.8683 - loss: 622.2363


 28/Unknown  6s 195ms/step - accuracy: 0.8702 - loss: 616.1565


 29/Unknown  6s 194ms/step - accuracy: 0.8720 - loss: 610.3881


 30/Unknown  6s 194ms/step - accuracy: 0.8737 - loss: 604.7990


 31/Unknown  6s 193ms/step - accuracy: 0.8753 - loss: 599.5613


 32/Unknown  7s 194ms/step - accuracy: 0.8769 - loss: 594.4847


 33/Unknown  7s 194ms/step - accuracy: 0.8783 - loss: 589.5745


 34/Unknown  7s 194ms/step - accuracy: 0.8797 - loss: 584.9431


 35/Unknown  7s 194ms/step - accuracy: 0.8810 - loss: 580.5197


 36/Unknown  7s 193ms/step - accuracy: 0.8822 - loss: 576.2609


 37/Unknown  8s 193ms/step - accuracy: 0.8834 - loss: 572.0708


 38/Unknown  8s 194ms/step - accuracy: 0.8845 - loss: 567.9126


 39/Unknown  8s 194ms/step - accuracy: 0.8856 - loss: 563.8269


 40/Unknown  8s 194ms/step - accuracy: 0.8867 - loss: 559.9911


 41/Unknown  8s 194ms/step - accuracy: 0.8877 - loss: 556.2637


 42/Unknown  9s 193ms/step - accuracy: 0.8886 - loss: 552.6080


 43/Unknown  9s 193ms/step - accuracy: 0.8896 - loss: 549.0726


 44/Unknown  9s 193ms/step - accuracy: 0.8905 - loss: 545.6210


 45/Unknown  9s 193ms/step - accuracy: 0.8913 - loss: 542.2662


 46/Unknown  9s 193ms/step - accuracy: 0.8921 - loss: 539.0649


 47/Unknown  10s 193ms/step - accuracy: 0.8929 - loss: 535.9783


 48/Unknown  10s 193ms/step - accuracy: 0.8936 - loss: 532.9994


 49/Unknown  10s 193ms/step - accuracy: 0.8944 - loss: 530.0856


 50/Unknown  10s 193ms/step - accuracy: 0.8951 - loss: 527.2556


 51/Unknown  10s 194ms/step - accuracy: 0.8957 - loss: 524.4853


 52/Unknown  11s 194ms/step - accuracy: 0.8964 - loss: 521.8221


 53/Unknown  11s 194ms/step - accuracy: 0.8970 - loss: 519.2384


 54/Unknown  11s 194ms/step - accuracy: 0.8976 - loss: 516.6887


 55/Unknown  11s 195ms/step - accuracy: 0.8982 - loss: 514.2283


 56/Unknown  11s 195ms/step - accuracy: 0.8987 - loss: 511.8073


 57/Unknown  12s 195ms/step - accuracy: 0.8993 - loss: 509.4113


 58/Unknown  12s 194ms/step - accuracy: 0.8998 - loss: 507.0705


 59/Unknown  12s 194ms/step - accuracy: 0.9004 - loss: 504.7713


 60/Unknown  12s 194ms/step - accuracy: 0.9009 - loss: 502.5121


 61/Unknown  12s 195ms/step - accuracy: 0.9014 - loss: 500.2973


 62/Unknown  13s 195ms/step - accuracy: 0.9019 - loss: 498.1272


 63/Unknown  13s 196ms/step - accuracy: 0.9023 - loss: 496.0018


 64/Unknown  13s 196ms/step - accuracy: 0.9028 - loss: 493.9293


 65/Unknown  13s 197ms/step - accuracy: 0.9032 - loss: 491.9118


 66/Unknown  14s 197ms/step - accuracy: 0.9037 - loss: 489.9484


 67/Unknown  14s 197ms/step - accuracy: 0.9041 - loss: 488.0164


 68/Unknown  14s 197ms/step - accuracy: 0.9045 - loss: 486.1193


 69/Unknown  14s 197ms/step - accuracy: 0.9049 - loss: 484.2630


 70/Unknown  14s 197ms/step - accuracy: 0.9053 - loss: 482.4265


 71/Unknown  14s 197ms/step - accuracy: 0.9057 - loss: 480.6362


 72/Unknown  15s 197ms/step - accuracy: 0.9061 - loss: 478.8780


 73/Unknown  15s 197ms/step - accuracy: 0.9064 - loss: 477.1625


 74/Unknown  15s 197ms/step - accuracy: 0.9068 - loss: 475.4860


 75/Unknown  15s 198ms/step - accuracy: 0.9071 - loss: 473.8222


 76/Unknown  15s 197ms/step - accuracy: 0.9075 - loss: 472.2155


 77/Unknown  16s 197ms/step - accuracy: 0.9078 - loss: 470.6271


 78/Unknown  16s 196ms/step - accuracy: 0.9081 - loss: 469.0505


 79/Unknown  16s 196ms/step - accuracy: 0.9084 - loss: 467.4939


 80/Unknown  16s 196ms/step - accuracy: 0.9087 - loss: 465.9711


 81/Unknown  16s 196ms/step - accuracy: 0.9090 - loss: 464.4900


 82/Unknown  17s 196ms/step - accuracy: 0.9093 - loss: 463.0288


 83/Unknown  17s 196ms/step - accuracy: 0.9096 - loss: 461.5836


 84/Unknown  17s 196ms/step - accuracy: 0.9099 - loss: 460.1690


 85/Unknown  17s 196ms/step - accuracy: 0.9101 - loss: 458.7745


 86/Unknown  17s 196ms/step - accuracy: 0.9104 - loss: 457.3958


 87/Unknown  18s 196ms/step - accuracy: 0.9107 - loss: 456.0403


 88/Unknown  18s 196ms/step - accuracy: 0.9109 - loss: 454.6969


 89/Unknown  18s 195ms/step - accuracy: 0.9112 - loss: 453.3802


 90/Unknown  18s 195ms/step - accuracy: 0.9115 - loss: 452.0960


 91/Unknown  18s 195ms/step - accuracy: 0.9117 - loss: 450.8279


 92/Unknown  18s 195ms/step - accuracy: 0.9119 - loss: 449.5715


 93/Unknown  19s 195ms/step - accuracy: 0.9122 - loss: 448.3546


 94/Unknown  19s 195ms/step - accuracy: 0.9124 - loss: 447.1467


 95/Unknown  19s 195ms/step - accuracy: 0.9126 - loss: 445.9574


 96/Unknown  19s 195ms/step - accuracy: 0.9129 - loss: 444.7892


 97/Unknown  19s 195ms/step - accuracy: 0.9131 - loss: 443.6310


 98/Unknown  20s 195ms/step - accuracy: 0.9133 - loss: 442.5005


 99/Unknown  20s 195ms/step - accuracy: 0.9135 - loss: 441.3832


100/Unknown  20s 195ms/step - accuracy: 0.9137 - loss: 440.2770


101/Unknown  20s 194ms/step - accuracy: 0.9139 - loss: 439.1806


102/Unknown  20s 194ms/step - accuracy: 0.9141 - loss: 438.0936


103/Unknown  21s 194ms/step - accuracy: 0.9144 - loss: 437.0181


104/Unknown  21s 195ms/step - accuracy: 0.9146 - loss: 435.9646


105/Unknown  21s 195ms/step - accuracy: 0.9148 - loss: 434.9277


106/Unknown  21s 195ms/step - accuracy: 0.9150 - loss: 433.8949


107/Unknown  21s 195ms/step - accuracy: 0.9151 - loss: 432.8877


108/Unknown  22s 195ms/step - accuracy: 0.9153 - loss: 431.8965


109/Unknown  22s 195ms/step - accuracy: 0.9155 - loss: 430.9133


110/Unknown  22s 196ms/step - accuracy: 0.9157 - loss: 429.9397


111/Unknown  22s 196ms/step - accuracy: 0.9159 - loss: 428.9818


112/Unknown  22s 196ms/step - accuracy: 0.9161 - loss: 428.0353


113/Unknown  23s 197ms/step - accuracy: 0.9162 - loss: 427.0999


114/Unknown  23s 197ms/step - accuracy: 0.9164 - loss: 426.1697


115/Unknown  23s 197ms/step - accuracy: 0.9166 - loss: 425.2458


116/Unknown  23s 197ms/step - accuracy: 0.9168 - loss: 424.3345


117/Unknown  24s 197ms/step - accuracy: 0.9169 - loss: 423.4386


118/Unknown  24s 197ms/step - accuracy: 0.9171 - loss: 422.5567


119/Unknown  24s 197ms/step - accuracy: 0.9173 - loss: 421.6823


120/Unknown  24s 197ms/step - accuracy: 0.9174 - loss: 420.8182


121/Unknown  24s 197ms/step - accuracy: 0.9176 - loss: 419.9664


122/Unknown  25s 197ms/step - accuracy: 0.9177 - loss: 419.1238


123/Unknown  25s 197ms/step - accuracy: 0.9179 - loss: 418.2940


124/Unknown  25s 197ms/step - accuracy: 0.9181 - loss: 417.4785


125/Unknown  25s 197ms/step - accuracy: 0.9182 - loss: 416.6722


126/Unknown  25s 197ms/step - accuracy: 0.9183 - loss: 415.8714


127/Unknown  26s 197ms/step - accuracy: 0.9185 - loss: 415.0771


128/Unknown  26s 197ms/step - accuracy: 0.9186 - loss: 414.2919


129/Unknown  26s 197ms/step - accuracy: 0.9188 - loss: 413.5163


130/Unknown  26s 197ms/step - accuracy: 0.9189 - loss: 412.7452


131/Unknown  26s 198ms/step - accuracy: 0.9191 - loss: 411.9837


132/Unknown  27s 197ms/step - accuracy: 0.9192 - loss: 411.2362


133/Unknown  27s 198ms/step - accuracy: 0.9193 - loss: 410.4987


134/Unknown  27s 198ms/step - accuracy: 0.9195 - loss: 409.7713


135/Unknown  27s 198ms/step - accuracy: 0.9196 - loss: 409.0497


136/Unknown  27s 198ms/step - accuracy: 0.9197 - loss: 408.3427


137/Unknown  28s 198ms/step - accuracy: 0.9199 - loss: 407.6457


138/Unknown  28s 198ms/step - accuracy: 0.9200 - loss: 406.9554


139/Unknown  28s 198ms/step - accuracy: 0.9201 - loss: 406.2739


140/Unknown  28s 198ms/step - accuracy: 0.9202 - loss: 405.6025


141/Unknown  28s 198ms/step - accuracy: 0.9204 - loss: 404.9378


142/Unknown  29s 198ms/step - accuracy: 0.9205 - loss: 404.2800


143/Unknown  29s 198ms/step - accuracy: 0.9206 - loss: 403.6313


144/Unknown  29s 198ms/step - accuracy: 0.9207 - loss: 402.9886


145/Unknown  29s 198ms/step - accuracy: 0.9208 - loss: 402.3509


146/Unknown  29s 198ms/step - accuracy: 0.9209 - loss: 401.7167


147/Unknown  30s 199ms/step - accuracy: 0.9211 - loss: 401.0935


148/Unknown  30s 199ms/step - accuracy: 0.9212 - loss: 400.4745


149/Unknown  30s 200ms/step - accuracy: 0.9213 - loss: 399.8633


150/Unknown  30s 200ms/step - accuracy: 0.9214 - loss: 399.2570


151/Unknown  31s 200ms/step - accuracy: 0.9215 - loss: 398.6591


152/Unknown  31s 200ms/step - accuracy: 0.9216 - loss: 398.0708


153/Unknown  31s 201ms/step - accuracy: 0.9217 - loss: 397.4871


154/Unknown  31s 201ms/step - accuracy: 0.9218 - loss: 396.9082


155/Unknown  32s 201ms/step - accuracy: 0.9219 - loss: 396.3336


156/Unknown  32s 201ms/step - accuracy: 0.9220 - loss: 395.7641


157/Unknown  32s 202ms/step - accuracy: 0.9222 - loss: 395.1993


158/Unknown  32s 202ms/step - accuracy: 0.9223 - loss: 394.6414


159/Unknown  33s 202ms/step - accuracy: 0.9224 - loss: 394.0871


160/Unknown  33s 202ms/step - accuracy: 0.9225 - loss: 393.5349


161/Unknown  33s 202ms/step - accuracy: 0.9226 - loss: 392.9877


162/Unknown  33s 202ms/step - accuracy: 0.9227 - loss: 392.4430


163/Unknown  33s 202ms/step - accuracy: 0.9228 - loss: 391.9062


164/Unknown  34s 202ms/step - accuracy: 0.9229 - loss: 391.3741


165/Unknown  34s 202ms/step - accuracy: 0.9230 - loss: 390.8450


166/Unknown  34s 202ms/step - accuracy: 0.9231 - loss: 390.3184


167/Unknown  34s 202ms/step - accuracy: 0.9232 - loss: 389.7947


168/Unknown  34s 202ms/step - accuracy: 0.9233 - loss: 389.2769


169/Unknown  35s 201ms/step - accuracy: 0.9234 - loss: 388.7642


170/Unknown  35s 201ms/step - accuracy: 0.9234 - loss: 388.2601


171/Unknown  35s 202ms/step - accuracy: 0.9235 - loss: 387.7563


172/Unknown  35s 202ms/step - accuracy: 0.9236 - loss: 387.2565


173/Unknown  35s 202ms/step - accuracy: 0.9237 - loss: 386.7589


174/Unknown  36s 202ms/step - accuracy: 0.9238 - loss: 386.2676


175/Unknown  36s 202ms/step - accuracy: 0.9239 - loss: 385.7783


176/Unknown  36s 202ms/step - accuracy: 0.9240 - loss: 385.2943


177/Unknown  36s 202ms/step - accuracy: 0.9241 - loss: 384.8183


178/Unknown  36s 202ms/step - accuracy: 0.9242 - loss: 384.3459


179/Unknown  37s 202ms/step - accuracy: 0.9243 - loss: 383.8763


180/Unknown  37s 202ms/step - accuracy: 0.9243 - loss: 383.4090


181/Unknown  37s 202ms/step - accuracy: 0.9244 - loss: 382.9447


182/Unknown  37s 202ms/step - accuracy: 0.9245 - loss: 382.4837


183/Unknown  37s 202ms/step - accuracy: 0.9246 - loss: 382.0276


184/Unknown  38s 202ms/step - accuracy: 0.9247 - loss: 381.5741


185/Unknown  38s 202ms/step - accuracy: 0.9248 - loss: 381.1227


186/Unknown  38s 203ms/step - accuracy: 0.9248 - loss: 380.6739


187/Unknown  38s 203ms/step - accuracy: 0.9249 - loss: 380.2263


188/Unknown  39s 203ms/step - accuracy: 0.9250 - loss: 379.7803


189/Unknown  39s 203ms/step - accuracy: 0.9251 - loss: 379.3362


190/Unknown  39s 203ms/step - accuracy: 0.9252 - loss: 378.8958


191/Unknown  39s 203ms/step - accuracy: 0.9252 - loss: 378.4580


192/Unknown  40s 203ms/step - accuracy: 0.9253 - loss: 378.0284


193/Unknown  40s 203ms/step - accuracy: 0.9254 - loss: 377.6024


194/Unknown  40s 203ms/step - accuracy: 0.9255 - loss: 377.1786


195/Unknown  40s 203ms/step - accuracy: 0.9256 - loss: 376.7578


196/Unknown  40s 203ms/step - accuracy: 0.9256 - loss: 376.3396


197/Unknown  41s 203ms/step - accuracy: 0.9257 - loss: 375.9260


198/Unknown  41s 203ms/step - accuracy: 0.9258 - loss: 375.5150


199/Unknown  41s 203ms/step - accuracy: 0.9259 - loss: 375.1063


200/Unknown  41s 203ms/step - accuracy: 0.9259 - loss: 374.6986


201/Unknown  41s 203ms/step - accuracy: 0.9260 - loss: 374.2932


202/Unknown  42s 203ms/step - accuracy: 0.9261 - loss: 373.8907


203/Unknown  42s 203ms/step - accuracy: 0.9262 - loss: 373.4908


204/Unknown  42s 203ms/step - accuracy: 0.9262 - loss: 373.0950


205/Unknown  42s 203ms/step - accuracy: 0.9263 - loss: 372.7011


206/Unknown  42s 203ms/step - accuracy: 0.9264 - loss: 372.3101


207/Unknown  43s 203ms/step - accuracy: 0.9264 - loss: 371.9205


208/Unknown  43s 203ms/step - accuracy: 0.9265 - loss: 371.5334


209/Unknown  43s 203ms/step - accuracy: 0.9266 - loss: 371.1480


210/Unknown  43s 203ms/step - accuracy: 0.9266 - loss: 370.7651


211/Unknown  43s 203ms/step - accuracy: 0.9267 - loss: 370.3852


212/Unknown  44s 203ms/step - accuracy: 0.9268 - loss: 370.0093


213/Unknown  44s 203ms/step - accuracy: 0.9268 - loss: 369.6351


214/Unknown  44s 204ms/step - accuracy: 0.9269 - loss: 369.2646


215/Unknown  44s 204ms/step - accuracy: 0.9270 - loss: 368.8969


216/Unknown  45s 204ms/step - accuracy: 0.9270 - loss: 368.5307


217/Unknown  45s 204ms/step - accuracy: 0.9271 - loss: 368.1656


218/Unknown  45s 204ms/step - accuracy: 0.9272 - loss: 367.8022


219/Unknown  45s 204ms/step - accuracy: 0.9272 - loss: 367.4400


220/Unknown  45s 204ms/step - accuracy: 0.9273 - loss: 367.0780


221/Unknown  46s 204ms/step - accuracy: 0.9274 - loss: 366.7194


222/Unknown  46s 204ms/step - accuracy: 0.9274 - loss: 366.3614


223/Unknown  46s 204ms/step - accuracy: 0.9275 - loss: 366.0057


224/Unknown  46s 205ms/step - accuracy: 0.9275 - loss: 365.6517


225/Unknown  47s 205ms/step - accuracy: 0.9276 - loss: 365.2998


226/Unknown  47s 205ms/step - accuracy: 0.9277 - loss: 364.9511


227/Unknown  47s 205ms/step - accuracy: 0.9277 - loss: 364.6082


228/Unknown  47s 205ms/step - accuracy: 0.9278 - loss: 364.2671


229/Unknown  47s 205ms/step - accuracy: 0.9278 - loss: 363.9267


230/Unknown  48s 205ms/step - accuracy: 0.9279 - loss: 363.5866


231/Unknown  48s 205ms/step - accuracy: 0.9280 - loss: 363.2502


232/Unknown  48s 205ms/step - accuracy: 0.9280 - loss: 362.9175


233/Unknown  48s 205ms/step - accuracy: 0.9281 - loss: 362.5866


234/Unknown  49s 205ms/step - accuracy: 0.9281 - loss: 362.2563


235/Unknown  49s 205ms/step - accuracy: 0.9282 - loss: 361.9282


236/Unknown  49s 206ms/step - accuracy: 0.9283 - loss: 361.6011


237/Unknown  49s 206ms/step - accuracy: 0.9283 - loss: 361.2760


238/Unknown  50s 207ms/step - accuracy: 0.9284 - loss: 360.9536


239/Unknown  50s 207ms/step - accuracy: 0.9284 - loss: 360.6325


240/Unknown  50s 207ms/step - accuracy: 0.9285 - loss: 360.3125


241/Unknown  51s 208ms/step - accuracy: 0.9285 - loss: 359.9926


242/Unknown  51s 208ms/step - accuracy: 0.9286 - loss: 359.6747


243/Unknown  51s 208ms/step - accuracy: 0.9287 - loss: 359.3593


244/Unknown  51s 209ms/step - accuracy: 0.9287 - loss: 359.0458


245/Unknown  52s 209ms/step - accuracy: 0.9288 - loss: 358.7349


246/Unknown  52s 209ms/step - accuracy: 0.9288 - loss: 358.4268


247/Unknown  52s 209ms/step - accuracy: 0.9289 - loss: 358.1215


248/Unknown  52s 209ms/step - accuracy: 0.9289 - loss: 357.8191


249/Unknown  53s 209ms/step - accuracy: 0.9290 - loss: 357.5194


250/Unknown  53s 209ms/step - accuracy: 0.9290 - loss: 357.2212


251/Unknown  53s 209ms/step - accuracy: 0.9291 - loss: 356.9238


252/Unknown  53s 209ms/step - accuracy: 0.9291 - loss: 356.6286


253/Unknown  53s 209ms/step - accuracy: 0.9292 - loss: 356.3355


254/Unknown  54s 210ms/step - accuracy: 0.9292 - loss: 356.0449


255/Unknown  54s 210ms/step - accuracy: 0.9293 - loss: 355.7557


256/Unknown  54s 210ms/step - accuracy: 0.9293 - loss: 355.4673


257/Unknown  54s 210ms/step - accuracy: 0.9294 - loss: 355.1799


258/Unknown  55s 210ms/step - accuracy: 0.9294 - loss: 354.8956


259/Unknown  55s 210ms/step - accuracy: 0.9295 - loss: 354.6136


260/Unknown  55s 210ms/step - accuracy: 0.9295 - loss: 354.3326


261/Unknown  55s 210ms/step - accuracy: 0.9296 - loss: 354.0539


262/Unknown  56s 210ms/step - accuracy: 0.9296 - loss: 353.7773


263/Unknown  56s 210ms/step - accuracy: 0.9297 - loss: 353.5032


264/Unknown  56s 211ms/step - accuracy: 0.9297 - loss: 353.2297


265/Unknown  56s 211ms/step - accuracy: 0.9298 - loss: 352.9581


266/Unknown  57s 211ms/step - accuracy: 0.9298 - loss: 352.6891


267/Unknown  57s 211ms/step - accuracy: 0.9299 - loss: 352.4228


268/Unknown  57s 211ms/step - accuracy: 0.9299 - loss: 352.1577


269/Unknown  57s 212ms/step - accuracy: 0.9300 - loss: 351.8944


270/Unknown  58s 212ms/step - accuracy: 0.9300 - loss: 351.6331


271/Unknown  58s 212ms/step - accuracy: 0.9300 - loss: 351.3750


272/Unknown  58s 212ms/step - accuracy: 0.9301 - loss: 351.1187


273/Unknown  59s 213ms/step - accuracy: 0.9301 - loss: 350.8644


274/Unknown  59s 213ms/step - accuracy: 0.9302 - loss: 350.6121


275/Unknown  59s 213ms/step - accuracy: 0.9302 - loss: 350.3619


276/Unknown  59s 213ms/step - accuracy: 0.9303 - loss: 350.1118


277/Unknown  60s 213ms/step - accuracy: 0.9303 - loss: 349.8630


278/Unknown  60s 213ms/step - accuracy: 0.9303 - loss: 349.6153


279/Unknown  60s 213ms/step - accuracy: 0.9304 - loss: 349.3688


280/Unknown  60s 213ms/step - accuracy: 0.9304 - loss: 349.1229


281/Unknown  60s 213ms/step - accuracy: 0.9305 - loss: 348.8784


282/Unknown  61s 213ms/step - accuracy: 0.9305 - loss: 348.6351


283/Unknown  61s 213ms/step - accuracy: 0.9306 - loss: 348.3920


284/Unknown  61s 213ms/step - accuracy: 0.9306 - loss: 348.1502


285/Unknown  61s 214ms/step - accuracy: 0.9306 - loss: 347.9100


286/Unknown  62s 214ms/step - accuracy: 0.9307 - loss: 347.6703


287/Unknown  62s 214ms/step - accuracy: 0.9307 - loss: 347.4321


288/Unknown  62s 214ms/step - accuracy: 0.9308 - loss: 347.1949


289/Unknown  62s 214ms/step - accuracy: 0.9308 - loss: 346.9582


290/Unknown  63s 214ms/step - accuracy: 0.9308 - loss: 346.7231


291/Unknown  63s 214ms/step - accuracy: 0.9309 - loss: 346.4895


292/Unknown  63s 214ms/step - accuracy: 0.9309 - loss: 346.2580


293/Unknown  63s 214ms/step - accuracy: 0.9310 - loss: 346.0265


294/Unknown  63s 214ms/step - accuracy: 0.9310 - loss: 345.7953


295/Unknown  64s 214ms/step - accuracy: 0.9310 - loss: 345.5652


296/Unknown  64s 214ms/step - accuracy: 0.9311 - loss: 345.3354


297/Unknown  64s 214ms/step - accuracy: 0.9311 - loss: 345.1051


298/Unknown  64s 214ms/step - accuracy: 0.9312 - loss: 344.8757


299/Unknown  65s 214ms/step - accuracy: 0.9312 - loss: 344.6488


300/Unknown  65s 215ms/step - accuracy: 0.9312 - loss: 344.4228


301/Unknown  65s 215ms/step - accuracy: 0.9313 - loss: 344.1973


302/Unknown  65s 215ms/step - accuracy: 0.9313 - loss: 343.9730


303/Unknown  66s 215ms/step - accuracy: 0.9314 - loss: 343.7490


304/Unknown  66s 215ms/step - accuracy: 0.9314 - loss: 343.5257


305/Unknown  66s 215ms/step - accuracy: 0.9314 - loss: 343.3047


306/Unknown  66s 215ms/step - accuracy: 0.9315 - loss: 343.0848


307/Unknown  66s 215ms/step - accuracy: 0.9315 - loss: 342.8660


308/Unknown  67s 215ms/step - accuracy: 0.9315 - loss: 342.6490


309/Unknown  67s 215ms/step - accuracy: 0.9316 - loss: 342.4344


310/Unknown  67s 215ms/step - accuracy: 0.9316 - loss: 342.2202


311/Unknown  67s 215ms/step - accuracy: 0.9316 - loss: 342.0071


312/Unknown  67s 215ms/step - accuracy: 0.9317 - loss: 341.7946


313/Unknown  68s 215ms/step - accuracy: 0.9317 - loss: 341.5840


314/Unknown  68s 215ms/step - accuracy: 0.9318 - loss: 341.3745


315/Unknown  68s 215ms/step - accuracy: 0.9318 - loss: 341.1663


316/Unknown  68s 215ms/step - accuracy: 0.9318 - loss: 340.9592


317/Unknown  69s 215ms/step - accuracy: 0.9319 - loss: 340.7528


318/Unknown  69s 215ms/step - accuracy: 0.9319 - loss: 340.5479


319/Unknown  69s 215ms/step - accuracy: 0.9319 - loss: 340.3434


320/Unknown  69s 215ms/step - accuracy: 0.9320 - loss: 340.1393


321/Unknown  69s 215ms/step - accuracy: 0.9320 - loss: 339.9351


322/Unknown  70s 215ms/step - accuracy: 0.9320 - loss: 339.7322


323/Unknown  70s 214ms/step - accuracy: 0.9321 - loss: 339.5305


324/Unknown  70s 214ms/step - accuracy: 0.9321 - loss: 339.3291


325/Unknown  70s 214ms/step - accuracy: 0.9321 - loss: 339.1288


326/Unknown  70s 214ms/step - accuracy: 0.9322 - loss: 338.9296


327/Unknown  71s 214ms/step - accuracy: 0.9322 - loss: 338.7308


328/Unknown  71s 214ms/step - accuracy: 0.9322 - loss: 338.5333


329/Unknown  71s 214ms/step - accuracy: 0.9323 - loss: 338.3367


330/Unknown  71s 214ms/step - accuracy: 0.9323 - loss: 338.1408


331/Unknown  71s 214ms/step - accuracy: 0.9323 - loss: 337.9453


332/Unknown  71s 214ms/step - accuracy: 0.9324 - loss: 337.7515


333/Unknown  72s 214ms/step - accuracy: 0.9324 - loss: 337.5584


334/Unknown  72s 214ms/step - accuracy: 0.9324 - loss: 337.3662


335/Unknown  72s 214ms/step - accuracy: 0.9325 - loss: 337.1747


336/Unknown  72s 214ms/step - accuracy: 0.9325 - loss: 336.9847


337/Unknown  72s 214ms/step - accuracy: 0.9325 - loss: 336.7955


338/Unknown  73s 214ms/step - accuracy: 0.9326 - loss: 336.6066


339/Unknown  73s 214ms/step - accuracy: 0.9326 - loss: 336.4176


340/Unknown  73s 214ms/step - accuracy: 0.9326 - loss: 336.2293


341/Unknown  73s 214ms/step - accuracy: 0.9327 - loss: 336.0417


342/Unknown  74s 214ms/step - accuracy: 0.9327 - loss: 335.8549


343/Unknown  74s 213ms/step - accuracy: 0.9327 - loss: 335.6696


344/Unknown  74s 213ms/step - accuracy: 0.9328 - loss: 335.4847


345/Unknown  74s 213ms/step - accuracy: 0.9328 - loss: 335.3005


346/Unknown  74s 214ms/step - accuracy: 0.9328 - loss: 335.1162


347/Unknown  75s 214ms/step - accuracy: 0.9328 - loss: 334.9317


348/Unknown  75s 214ms/step - accuracy: 0.9329 - loss: 334.7472


349/Unknown  75s 214ms/step - accuracy: 0.9329 - loss: 334.5628


350/Unknown  75s 214ms/step - accuracy: 0.9329 - loss: 334.3791


351/Unknown  75s 214ms/step - accuracy: 0.9330 - loss: 334.1966


352/Unknown  76s 214ms/step - accuracy: 0.9330 - loss: 334.0144


353/Unknown  76s 214ms/step - accuracy: 0.9330 - loss: 333.8329


354/Unknown  76s 214ms/step - accuracy: 0.9331 - loss: 333.6525


355/Unknown  76s 214ms/step - accuracy: 0.9331 - loss: 333.4737


356/Unknown  77s 214ms/step - accuracy: 0.9331 - loss: 333.2956


357/Unknown  77s 214ms/step - accuracy: 0.9332 - loss: 333.1186


358/Unknown  77s 214ms/step - accuracy: 0.9332 - loss: 332.9418


359/Unknown  77s 214ms/step - accuracy: 0.9332 - loss: 332.7658


360/Unknown  77s 214ms/step - accuracy: 0.9332 - loss: 332.5904


361/Unknown  78s 214ms/step - accuracy: 0.9333 - loss: 332.4163


362/Unknown  78s 214ms/step - accuracy: 0.9333 - loss: 332.2424


363/Unknown  78s 214ms/step - accuracy: 0.9333 - loss: 332.0692


364/Unknown  78s 214ms/step - accuracy: 0.9334 - loss: 331.8963


365/Unknown  79s 214ms/step - accuracy: 0.9334 - loss: 331.7242


366/Unknown  79s 214ms/step - accuracy: 0.9334 - loss: 331.5526


367/Unknown  79s 214ms/step - accuracy: 0.9334 - loss: 331.3815


368/Unknown  79s 214ms/step - accuracy: 0.9335 - loss: 331.2112


369/Unknown  79s 214ms/step - accuracy: 0.9335 - loss: 331.0415


370/Unknown  80s 214ms/step - accuracy: 0.9335 - loss: 330.8718


371/Unknown  80s 214ms/step - accuracy: 0.9336 - loss: 330.7039


372/Unknown  80s 214ms/step - accuracy: 0.9336 - loss: 330.5366


373/Unknown  80s 214ms/step - accuracy: 0.9336 - loss: 330.3701


374/Unknown  81s 214ms/step - accuracy: 0.9336 - loss: 330.2039


375/Unknown  81s 214ms/step - accuracy: 0.9337 - loss: 330.0381


376/Unknown  81s 214ms/step - accuracy: 0.9337 - loss: 329.8729


377/Unknown  81s 214ms/step - accuracy: 0.9337 - loss: 329.7084


378/Unknown  81s 214ms/step - accuracy: 0.9338 - loss: 329.5450


379/Unknown  82s 214ms/step - accuracy: 0.9338 - loss: 329.3824


380/Unknown  82s 214ms/step - accuracy: 0.9338 - loss: 329.2206


381/Unknown  82s 214ms/step - accuracy: 0.9338 - loss: 329.0592


382/Unknown  82s 214ms/step - accuracy: 0.9339 - loss: 328.8988


383/Unknown  83s 214ms/step - accuracy: 0.9339 - loss: 328.7392


384/Unknown  83s 214ms/step - accuracy: 0.9339 - loss: 328.5800


385/Unknown  83s 214ms/step - accuracy: 0.9339 - loss: 328.4217


386/Unknown  83s 214ms/step - accuracy: 0.9340 - loss: 328.2640


387/Unknown  84s 215ms/step - accuracy: 0.9340 - loss: 328.1079


388/Unknown  84s 215ms/step - accuracy: 0.9340 - loss: 327.9520


389/Unknown  84s 215ms/step - accuracy: 0.9340 - loss: 327.7965


390/Unknown  84s 215ms/step - accuracy: 0.9341 - loss: 327.6415


391/Unknown  84s 215ms/step - accuracy: 0.9341 - loss: 327.4871


392/Unknown  85s 215ms/step - accuracy: 0.9341 - loss: 327.3329


393/Unknown  85s 215ms/step - accuracy: 0.9341 - loss: 327.1793


394/Unknown  85s 215ms/step - accuracy: 0.9342 - loss: 327.0259


395/Unknown  85s 215ms/step - accuracy: 0.9342 - loss: 326.8732


396/Unknown  86s 215ms/step - accuracy: 0.9342 - loss: 326.7209


397/Unknown  86s 215ms/step - accuracy: 0.9342 - loss: 326.5693


398/Unknown  86s 215ms/step - accuracy: 0.9343 - loss: 326.4187


399/Unknown  86s 215ms/step - accuracy: 0.9343 - loss: 326.2686


400/Unknown  87s 216ms/step - accuracy: 0.9343 - loss: 326.1197


401/Unknown  87s 216ms/step - accuracy: 0.9343 - loss: 325.9718


402/Unknown  87s 216ms/step - accuracy: 0.9344 - loss: 325.8240


403/Unknown  88s 216ms/step - accuracy: 0.9344 - loss: 325.6769


404/Unknown  88s 216ms/step - accuracy: 0.9344 - loss: 325.5306


405/Unknown  88s 216ms/step - accuracy: 0.9344 - loss: 325.3849


406/Unknown  88s 216ms/step - accuracy: 0.9345 - loss: 325.2393


407/Unknown  88s 216ms/step - accuracy: 0.9345 - loss: 325.0950


408/Unknown  89s 216ms/step - accuracy: 0.9345 - loss: 324.9518


409/Unknown  89s 216ms/step - accuracy: 0.9345 - loss: 324.8096


410/Unknown  89s 216ms/step - accuracy: 0.9346 - loss: 324.6682


411/Unknown  89s 216ms/step - accuracy: 0.9346 - loss: 324.5275


412/Unknown  90s 216ms/step - accuracy: 0.9346 - loss: 324.3878


413/Unknown  90s 216ms/step - accuracy: 0.9346 - loss: 324.2490


414/Unknown  90s 216ms/step - accuracy: 0.9347 - loss: 324.1107


415/Unknown  90s 216ms/step - accuracy: 0.9347 - loss: 323.9733


416/Unknown  90s 216ms/step - accuracy: 0.9347 - loss: 323.8363


417/Unknown  91s 216ms/step - accuracy: 0.9347 - loss: 323.7005


418/Unknown  91s 216ms/step - accuracy: 0.9347 - loss: 323.5650


419/Unknown  91s 216ms/step - accuracy: 0.9348 - loss: 323.4300


420/Unknown  91s 216ms/step - accuracy: 0.9348 - loss: 323.2954


421/Unknown  92s 216ms/step - accuracy: 0.9348 - loss: 323.1616


422/Unknown  92s 216ms/step - accuracy: 0.9348 - loss: 323.0279


423/Unknown  92s 216ms/step - accuracy: 0.9349 - loss: 322.8943


424/Unknown  92s 217ms/step - accuracy: 0.9349 - loss: 322.7610


425/Unknown  93s 217ms/step - accuracy: 0.9349 - loss: 322.6279


426/Unknown  93s 217ms/step - accuracy: 0.9349 - loss: 322.4956


427/Unknown  93s 217ms/step - accuracy: 0.9349 - loss: 322.3640


428/Unknown  93s 217ms/step - accuracy: 0.9350 - loss: 322.2333


429/Unknown  93s 217ms/step - accuracy: 0.9350 - loss: 322.1030


430/Unknown  94s 217ms/step - accuracy: 0.9350 - loss: 321.9735


431/Unknown  94s 217ms/step - accuracy: 0.9350 - loss: 321.8445


432/Unknown  94s 217ms/step - accuracy: 0.9350 - loss: 321.7156


433/Unknown  95s 217ms/step - accuracy: 0.9351 - loss: 321.5872


434/Unknown  95s 217ms/step - accuracy: 0.9351 - loss: 321.4588


435/Unknown  95s 217ms/step - accuracy: 0.9351 - loss: 321.3307


436/Unknown  95s 217ms/step - accuracy: 0.9351 - loss: 321.2033


437/Unknown  95s 217ms/step - accuracy: 0.9352 - loss: 321.0760


438/Unknown  96s 217ms/step - accuracy: 0.9352 - loss: 320.9493


439/Unknown  96s 217ms/step - accuracy: 0.9352 - loss: 320.8226


440/Unknown  96s 217ms/step - accuracy: 0.9352 - loss: 320.6959


441/Unknown  96s 217ms/step - accuracy: 0.9352 - loss: 320.5698


442/Unknown  97s 218ms/step - accuracy: 0.9353 - loss: 320.4443


443/Unknown  97s 218ms/step - accuracy: 0.9353 - loss: 320.3189


444/Unknown  97s 218ms/step - accuracy: 0.9353 - loss: 320.1938


445/Unknown  97s 218ms/step - accuracy: 0.9353 - loss: 320.0695


446/Unknown  98s 218ms/step - accuracy: 0.9353 - loss: 319.9457


447/Unknown  98s 218ms/step - accuracy: 0.9354 - loss: 319.8219


448/Unknown  98s 218ms/step - accuracy: 0.9354 - loss: 319.6988


449/Unknown  98s 218ms/step - accuracy: 0.9354 - loss: 319.5759


450/Unknown  99s 218ms/step - accuracy: 0.9354 - loss: 319.4533


451/Unknown  99s 218ms/step - accuracy: 0.9354 - loss: 319.3317


452/Unknown  99s 218ms/step - accuracy: 0.9355 - loss: 319.2104


453/Unknown  99s 218ms/step - accuracy: 0.9355 - loss: 319.0900


454/Unknown  100s 218ms/step - accuracy: 0.9355 - loss: 318.9702


455/Unknown  100s 218ms/step - accuracy: 0.9355 - loss: 318.8509


456/Unknown  100s 218ms/step - accuracy: 0.9355 - loss: 318.7319


457/Unknown  100s 218ms/step - accuracy: 0.9356 - loss: 318.6129


458/Unknown  101s 219ms/step - accuracy: 0.9356 - loss: 318.4945


459/Unknown  101s 219ms/step - accuracy: 0.9356 - loss: 318.3764


460/Unknown  101s 219ms/step - accuracy: 0.9356 - loss: 318.2592


461/Unknown  101s 219ms/step - accuracy: 0.9356 - loss: 318.1426


462/Unknown  102s 219ms/step - accuracy: 0.9356 - loss: 318.0262


463/Unknown  102s 219ms/step - accuracy: 0.9357 - loss: 317.9101


464/Unknown  102s 219ms/step - accuracy: 0.9357 - loss: 317.7942


465/Unknown  102s 219ms/step - accuracy: 0.9357 - loss: 317.6788


466/Unknown  102s 219ms/step - accuracy: 0.9357 - loss: 317.5635


467/Unknown  103s 219ms/step - accuracy: 0.9357 - loss: 317.4484


468/Unknown  103s 219ms/step - accuracy: 0.9358 - loss: 317.3335


469/Unknown  103s 219ms/step - accuracy: 0.9358 - loss: 317.2188


470/Unknown  103s 219ms/step - accuracy: 0.9358 - loss: 317.1040


471/Unknown  104s 219ms/step - accuracy: 0.9358 - loss: 316.9898


472/Unknown  104s 219ms/step - accuracy: 0.9358 - loss: 316.8763


473/Unknown  104s 219ms/step - accuracy: 0.9359 - loss: 316.7631


474/Unknown  104s 219ms/step - accuracy: 0.9359 - loss: 316.6501


475/Unknown  105s 219ms/step - accuracy: 0.9359 - loss: 316.5378


476/Unknown  105s 219ms/step - accuracy: 0.9359 - loss: 316.4257


477/Unknown  105s 220ms/step - accuracy: 0.9359 - loss: 316.3138


478/Unknown  105s 220ms/step - accuracy: 0.9359 - loss: 316.2022


479/Unknown  106s 220ms/step - accuracy: 0.9360 - loss: 316.0910


480/Unknown  106s 220ms/step - accuracy: 0.9360 - loss: 315.9806


481/Unknown  106s 220ms/step - accuracy: 0.9360 - loss: 315.8706


482/Unknown  106s 220ms/step - accuracy: 0.9360 - loss: 315.7607


483/Unknown  107s 220ms/step - accuracy: 0.9360 - loss: 315.6510


484/Unknown  107s 220ms/step - accuracy: 0.9361 - loss: 315.5415


485/Unknown  107s 220ms/step - accuracy: 0.9361 - loss: 315.4324


486/Unknown  107s 220ms/step - accuracy: 0.9361 - loss: 315.3241


487/Unknown  108s 220ms/step - accuracy: 0.9361 - loss: 315.2159


488/Unknown  108s 220ms/step - accuracy: 0.9361 - loss: 315.1081


489/Unknown  108s 220ms/step - accuracy: 0.9361 - loss: 315.0006


490/Unknown  108s 220ms/step - accuracy: 0.9362 - loss: 314.8934


491/Unknown  109s 220ms/step - accuracy: 0.9362 - loss: 314.7863


492/Unknown  109s 220ms/step - accuracy: 0.9362 - loss: 314.6796


493/Unknown  109s 220ms/step - accuracy: 0.9362 - loss: 314.5731


494/Unknown  109s 220ms/step - accuracy: 0.9362 - loss: 314.4671


495/Unknown  109s 220ms/step - accuracy: 0.9363 - loss: 314.3610


496/Unknown  110s 220ms/step - accuracy: 0.9363 - loss: 314.2551


497/Unknown  110s 220ms/step - accuracy: 0.9363 - loss: 314.1498


498/Unknown  110s 220ms/step - accuracy: 0.9363 - loss: 314.0451


499/Unknown  110s 220ms/step - accuracy: 0.9363 - loss: 313.9405


500/Unknown  111s 220ms/step - accuracy: 0.9363 - loss: 313.8364


501/Unknown  111s 220ms/step - accuracy: 0.9364 - loss: 313.7324


502/Unknown  111s 220ms/step - accuracy: 0.9364 - loss: 313.6290


503/Unknown  111s 220ms/step - accuracy: 0.9364 - loss: 313.5266


504/Unknown  111s 220ms/step - accuracy: 0.9364 - loss: 313.4240


505/Unknown  112s 220ms/step - accuracy: 0.9364 - loss: 313.3215


506/Unknown  112s 220ms/step - accuracy: 0.9364 - loss: 313.2194


507/Unknown  112s 220ms/step - accuracy: 0.9365 - loss: 313.1174


508/Unknown  112s 220ms/step - accuracy: 0.9365 - loss: 313.0163


509/Unknown  113s 220ms/step - accuracy: 0.9365 - loss: 312.9149


510/Unknown  113s 220ms/step - accuracy: 0.9365 - loss: 312.8137


511/Unknown  113s 220ms/step - accuracy: 0.9365 - loss: 312.7127


512/Unknown  113s 220ms/step - accuracy: 0.9365 - loss: 312.6117


513/Unknown  113s 220ms/step - accuracy: 0.9366 - loss: 312.5108


514/Unknown  114s 220ms/step - accuracy: 0.9366 - loss: 312.4105


515/Unknown  114s 220ms/step - accuracy: 0.9366 - loss: 312.3107


516/Unknown  114s 221ms/step - accuracy: 0.9366 - loss: 312.2112


517/Unknown  115s 221ms/step - accuracy: 0.9366 - loss: 312.1120


518/Unknown  115s 221ms/step - accuracy: 0.9366 - loss: 312.0136


519/Unknown  115s 221ms/step - accuracy: 0.9367 - loss: 311.9154


520/Unknown  115s 221ms/step - accuracy: 0.9367 - loss: 311.8175


521/Unknown  116s 221ms/step - accuracy: 0.9367 - loss: 311.7198


522/Unknown  116s 221ms/step - accuracy: 0.9367 - loss: 311.6223


523/Unknown  116s 221ms/step - accuracy: 0.9367 - loss: 311.5254


524/Unknown  116s 221ms/step - accuracy: 0.9367 - loss: 311.4289


525/Unknown  117s 221ms/step - accuracy: 0.9368 - loss: 311.3327


526/Unknown  117s 221ms/step - accuracy: 0.9368 - loss: 311.2370


527/Unknown  117s 221ms/step - accuracy: 0.9368 - loss: 311.1415


528/Unknown  117s 221ms/step - accuracy: 0.9368 - loss: 311.0460


529/Unknown  118s 221ms/step - accuracy: 0.9368 - loss: 310.9506


530/Unknown  118s 221ms/step - accuracy: 0.9368 - loss: 310.8556


531/Unknown  118s 221ms/step - accuracy: 0.9368 - loss: 310.7611


532/Unknown  118s 221ms/step - accuracy: 0.9369 - loss: 310.6666


533/Unknown  119s 222ms/step - accuracy: 0.9369 - loss: 310.5723


534/Unknown  119s 222ms/step - accuracy: 0.9369 - loss: 310.4783


535/Unknown  119s 222ms/step - accuracy: 0.9369 - loss: 310.3845


536/Unknown  119s 222ms/step - accuracy: 0.9369 - loss: 310.2907


537/Unknown  119s 222ms/step - accuracy: 0.9369 - loss: 310.1972


538/Unknown  120s 222ms/step - accuracy: 0.9370 - loss: 310.1041


539/Unknown  120s 222ms/step - accuracy: 0.9370 - loss: 310.0114


540/Unknown  120s 222ms/step - accuracy: 0.9370 - loss: 309.9187


541/Unknown  120s 222ms/step - accuracy: 0.9370 - loss: 309.8263


542/Unknown  121s 221ms/step - accuracy: 0.9370 - loss: 309.7338


543/Unknown  121s 221ms/step - accuracy: 0.9370 - loss: 309.6414


544/Unknown  121s 222ms/step - accuracy: 0.9370 - loss: 309.5492


545/Unknown  121s 222ms/step - accuracy: 0.9371 - loss: 309.4574


546/Unknown  121s 222ms/step - accuracy: 0.9371 - loss: 309.3658


547/Unknown  122s 222ms/step - accuracy: 0.9371 - loss: 309.2743


548/Unknown  122s 222ms/step - accuracy: 0.9371 - loss: 309.1831


549/Unknown  122s 222ms/step - accuracy: 0.9371 - loss: 309.0924


550/Unknown  122s 222ms/step - accuracy: 0.9371 - loss: 309.0020


551/Unknown  123s 222ms/step - accuracy: 0.9371 - loss: 308.9121


552/Unknown  123s 222ms/step - accuracy: 0.9372 - loss: 308.8221


553/Unknown  123s 222ms/step - accuracy: 0.9372 - loss: 308.7321


554/Unknown  123s 222ms/step - accuracy: 0.9372 - loss: 308.6426


555/Unknown  124s 222ms/step - accuracy: 0.9372 - loss: 308.5534


556/Unknown  124s 222ms/step - accuracy: 0.9372 - loss: 308.4645


557/Unknown  124s 222ms/step - accuracy: 0.9372 - loss: 308.3757


558/Unknown  124s 222ms/step - accuracy: 0.9372 - loss: 308.2873


559/Unknown  125s 222ms/step - accuracy: 0.9373 - loss: 308.1992


560/Unknown  125s 222ms/step - accuracy: 0.9373 - loss: 308.1115


561/Unknown  125s 222ms/step - accuracy: 0.9373 - loss: 308.0239


562/Unknown  125s 222ms/step - accuracy: 0.9373 - loss: 307.9366


563/Unknown  126s 222ms/step - accuracy: 0.9373 - loss: 307.8493


564/Unknown  126s 222ms/step - accuracy: 0.9373 - loss: 307.7624


565/Unknown  126s 222ms/step - accuracy: 0.9373 - loss: 307.6758


566/Unknown  126s 222ms/step - accuracy: 0.9374 - loss: 307.5896


567/Unknown  126s 222ms/step - accuracy: 0.9374 - loss: 307.5038


568/Unknown  127s 222ms/step - accuracy: 0.9374 - loss: 307.4184


569/Unknown  127s 222ms/step - accuracy: 0.9374 - loss: 307.3336


570/Unknown  127s 222ms/step - accuracy: 0.9374 - loss: 307.2491


571/Unknown  127s 222ms/step - accuracy: 0.9374 - loss: 307.1649


572/Unknown  128s 222ms/step - accuracy: 0.9374 - loss: 307.0807


573/Unknown  128s 222ms/step - accuracy: 0.9375 - loss: 306.9967


574/Unknown  128s 223ms/step - accuracy: 0.9375 - loss: 306.9132


575/Unknown  129s 223ms/step - accuracy: 0.9375 - loss: 306.8299


576/Unknown  129s 223ms/step - accuracy: 0.9375 - loss: 306.7468


577/Unknown  129s 223ms/step - accuracy: 0.9375 - loss: 306.6639


578/Unknown  129s 223ms/step - accuracy: 0.9375 - loss: 306.5809


579/Unknown  130s 223ms/step - accuracy: 0.9375 - loss: 306.4984


580/Unknown  130s 223ms/step - accuracy: 0.9376 - loss: 306.4160


581/Unknown  130s 223ms/step - accuracy: 0.9376 - loss: 306.3336


582/Unknown  130s 223ms/step - accuracy: 0.9376 - loss: 306.2515


583/Unknown  131s 223ms/step - accuracy: 0.9376 - loss: 306.1693


584/Unknown  131s 223ms/step - accuracy: 0.9376 - loss: 306.0871


585/Unknown  131s 223ms/step - accuracy: 0.9376 - loss: 306.0051


586/Unknown  131s 223ms/step - accuracy: 0.9376 - loss: 305.9232


587/Unknown  132s 223ms/step - accuracy: 0.9376 - loss: 305.8415


588/Unknown  132s 223ms/step - accuracy: 0.9377 - loss: 305.7600


589/Unknown  132s 223ms/step - accuracy: 0.9377 - loss: 305.6787


590/Unknown  132s 223ms/step - accuracy: 0.9377 - loss: 305.5978


591/Unknown  133s 223ms/step - accuracy: 0.9377 - loss: 305.5169


592/Unknown  133s 223ms/step - accuracy: 0.9377 - loss: 305.4363


593/Unknown  133s 223ms/step - accuracy: 0.9377 - loss: 305.3560


594/Unknown  133s 223ms/step - accuracy: 0.9377 - loss: 305.2758


595/Unknown  133s 223ms/step - accuracy: 0.9378 - loss: 305.1960


596/Unknown  134s 223ms/step - accuracy: 0.9378 - loss: 305.1167


597/Unknown  134s 223ms/step - accuracy: 0.9378 - loss: 305.0374


598/Unknown  134s 223ms/step - accuracy: 0.9378 - loss: 304.9583


599/Unknown  134s 223ms/step - accuracy: 0.9378 - loss: 304.8791


600/Unknown  134s 223ms/step - accuracy: 0.9378 - loss: 304.8002


601/Unknown  135s 223ms/step - accuracy: 0.9378 - loss: 304.7217


602/Unknown  135s 223ms/step - accuracy: 0.9378 - loss: 304.6434


603/Unknown  135s 223ms/step - accuracy: 0.9379 - loss: 304.5656


604/Unknown  135s 223ms/step - accuracy: 0.9379 - loss: 304.4879


605/Unknown  136s 223ms/step - accuracy: 0.9379 - loss: 304.4105


606/Unknown  136s 223ms/step - accuracy: 0.9379 - loss: 304.3331


607/Unknown  136s 223ms/step - accuracy: 0.9379 - loss: 304.2561


608/Unknown  136s 223ms/step - accuracy: 0.9379 - loss: 304.1792


609/Unknown  137s 223ms/step - accuracy: 0.9379 - loss: 304.1025


610/Unknown  137s 223ms/step - accuracy: 0.9379 - loss: 304.0259


611/Unknown  137s 223ms/step - accuracy: 0.9380 - loss: 303.9494


612/Unknown  137s 223ms/step - accuracy: 0.9380 - loss: 303.8730


613/Unknown  137s 223ms/step - accuracy: 0.9380 - loss: 303.7966


614/Unknown  138s 223ms/step - accuracy: 0.9380 - loss: 303.7203


615/Unknown  138s 223ms/step - accuracy: 0.9380 - loss: 303.6440


616/Unknown  138s 223ms/step - accuracy: 0.9380 - loss: 303.5678


617/Unknown  138s 224ms/step - accuracy: 0.9380 - loss: 303.4917


618/Unknown  139s 224ms/step - accuracy: 0.9380 - loss: 303.4161


619/Unknown  139s 224ms/step - accuracy: 0.9380 - loss: 303.3405


620/Unknown  139s 224ms/step - accuracy: 0.9381 - loss: 303.2652


621/Unknown  140s 224ms/step - accuracy: 0.9381 - loss: 303.1904


622/Unknown  140s 224ms/step - accuracy: 0.9381 - loss: 303.1159


623/Unknown  140s 224ms/step - accuracy: 0.9381 - loss: 303.0418


624/Unknown  140s 224ms/step - accuracy: 0.9381 - loss: 302.9678


625/Unknown  141s 224ms/step - accuracy: 0.9381 - loss: 302.8940


626/Unknown  141s 224ms/step - accuracy: 0.9381 - loss: 302.8201


627/Unknown  141s 224ms/step - accuracy: 0.9381 - loss: 302.7465


628/Unknown  141s 224ms/step - accuracy: 0.9382 - loss: 302.6730


629/Unknown  141s 224ms/step - accuracy: 0.9382 - loss: 302.5998


630/Unknown  142s 224ms/step - accuracy: 0.9382 - loss: 302.5268


631/Unknown  142s 224ms/step - accuracy: 0.9382 - loss: 302.4539


632/Unknown  142s 224ms/step - accuracy: 0.9382 - loss: 302.3810


633/Unknown  142s 224ms/step - accuracy: 0.9382 - loss: 302.3086


634/Unknown  143s 224ms/step - accuracy: 0.9382 - loss: 302.2364


635/Unknown  143s 224ms/step - accuracy: 0.9382 - loss: 302.1645


636/Unknown  143s 224ms/step - accuracy: 0.9383 - loss: 302.0930


637/Unknown  143s 224ms/step - accuracy: 0.9383 - loss: 302.0216


638/Unknown  144s 224ms/step - accuracy: 0.9383 - loss: 301.9502


639/Unknown  144s 224ms/step - accuracy: 0.9383 - loss: 301.8791

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()


```
</div>
 639/639 ━━━━━━━━━━━━━━━━━━━━ 160s 249ms/step - accuracy: 0.9383 - loss: 301.8082 - val_accuracy: 0.9485 - val_loss: 235.7996


<div class="k-default-codeblock">
```
Model training finished.
Evaluating model performance...

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  0s 331ms/step - accuracy: 0.9623 - loss: 160.6135


  2/Unknown  0s 119ms/step - accuracy: 0.9557 - loss: 181.4366


  3/Unknown  1s 131ms/step - accuracy: 0.9524 - loss: 198.4659


  4/Unknown  1s 129ms/step - accuracy: 0.9502 - loss: 209.3009


  5/Unknown  1s 133ms/step - accuracy: 0.9499 - loss: 215.6982


  6/Unknown  1s 131ms/step - accuracy: 0.9499 - loss: 219.7466


  7/Unknown  1s 132ms/step - accuracy: 0.9502 - loss: 220.2296


  8/Unknown  1s 132ms/step - accuracy: 0.9504 - loss: 219.6000


  9/Unknown  1s 133ms/step - accuracy: 0.9506 - loss: 218.5403


 10/Unknown  2s 133ms/step - accuracy: 0.9507 - loss: 217.4007


 11/Unknown  2s 134ms/step - accuracy: 0.9507 - loss: 216.4865


 12/Unknown  2s 133ms/step - accuracy: 0.9504 - loss: 215.7090


 13/Unknown  2s 135ms/step - accuracy: 0.9502 - loss: 215.4628


 14/Unknown  2s 135ms/step - accuracy: 0.9500 - loss: 215.0735


 15/Unknown  2s 134ms/step - accuracy: 0.9499 - loss: 214.8078


 16/Unknown  2s 134ms/step - accuracy: 0.9500 - loss: 214.3558


 17/Unknown  2s 134ms/step - accuracy: 0.9500 - loss: 213.9521


 18/Unknown  3s 135ms/step - accuracy: 0.9501 - loss: 213.9012


 19/Unknown  3s 134ms/step - accuracy: 0.9501 - loss: 214.0063


 20/Unknown  3s 135ms/step - accuracy: 0.9501 - loss: 214.2168


 21/Unknown  3s 134ms/step - accuracy: 0.9500 - loss: 214.5657


 22/Unknown  3s 135ms/step - accuracy: 0.9500 - loss: 214.8618


 23/Unknown  3s 134ms/step - accuracy: 0.9500 - loss: 215.1154


 24/Unknown  3s 135ms/step - accuracy: 0.9499 - loss: 215.2906


 25/Unknown  4s 134ms/step - accuracy: 0.9499 - loss: 215.6145


 26/Unknown  4s 135ms/step - accuracy: 0.9499 - loss: 215.8544


 27/Unknown  4s 135ms/step - accuracy: 0.9498 - loss: 216.0591


 28/Unknown  4s 135ms/step - accuracy: 0.9499 - loss: 216.2666


 29/Unknown  4s 135ms/step - accuracy: 0.9499 - loss: 216.4423


 30/Unknown  4s 135ms/step - accuracy: 0.9499 - loss: 216.5613


 31/Unknown  4s 135ms/step - accuracy: 0.9498 - loss: 216.7220


 32/Unknown  5s 135ms/step - accuracy: 0.9498 - loss: 216.8842


 33/Unknown  5s 135ms/step - accuracy: 0.9498 - loss: 217.1658


 34/Unknown  5s 135ms/step - accuracy: 0.9497 - loss: 217.4608


 35/Unknown  5s 135ms/step - accuracy: 0.9496 - loss: 217.7231


 36/Unknown  5s 134ms/step - accuracy: 0.9496 - loss: 217.9504


 37/Unknown  5s 135ms/step - accuracy: 0.9496 - loss: 218.1658


 38/Unknown  5s 134ms/step - accuracy: 0.9495 - loss: 218.3597


 39/Unknown  5s 134ms/step - accuracy: 0.9495 - loss: 218.5269


 40/Unknown  6s 135ms/step - accuracy: 0.9495 - loss: 218.7106


 41/Unknown  6s 135ms/step - accuracy: 0.9494 - loss: 218.8721


 42/Unknown  6s 136ms/step - accuracy: 0.9494 - loss: 218.9924


 43/Unknown  6s 136ms/step - accuracy: 0.9494 - loss: 219.1745


 44/Unknown  6s 137ms/step - accuracy: 0.9494 - loss: 219.3449


 45/Unknown  6s 138ms/step - accuracy: 0.9493 - loss: 219.4943


 46/Unknown  7s 137ms/step - accuracy: 0.9493 - loss: 219.6201


 47/Unknown  7s 138ms/step - accuracy: 0.9493 - loss: 219.7240


 48/Unknown  7s 138ms/step - accuracy: 0.9493 - loss: 219.8335


 49/Unknown  7s 138ms/step - accuracy: 0.9493 - loss: 219.9367


 50/Unknown  7s 138ms/step - accuracy: 0.9493 - loss: 220.0834


 51/Unknown  7s 138ms/step - accuracy: 0.9492 - loss: 220.2067


 52/Unknown  7s 139ms/step - accuracy: 0.9492 - loss: 220.2963


 53/Unknown  8s 138ms/step - accuracy: 0.9492 - loss: 220.3649


 54/Unknown  8s 138ms/step - accuracy: 0.9492 - loss: 220.4462


 55/Unknown  8s 138ms/step - accuracy: 0.9492 - loss: 220.5459


 56/Unknown  8s 138ms/step - accuracy: 0.9492 - loss: 220.6197


 57/Unknown  8s 138ms/step - accuracy: 0.9491 - loss: 220.6727


 58/Unknown  8s 138ms/step - accuracy: 0.9491 - loss: 220.7652


 59/Unknown  8s 138ms/step - accuracy: 0.9491 - loss: 220.8513


 60/Unknown  8s 138ms/step - accuracy: 0.9491 - loss: 220.9392


 61/Unknown  9s 138ms/step - accuracy: 0.9491 - loss: 221.0330


 62/Unknown  9s 137ms/step - accuracy: 0.9491 - loss: 221.1127


 63/Unknown  9s 137ms/step - accuracy: 0.9491 - loss: 221.2177


 64/Unknown  9s 137ms/step - accuracy: 0.9490 - loss: 221.3382


 65/Unknown  9s 137ms/step - accuracy: 0.9490 - loss: 221.4572


 66/Unknown  9s 137ms/step - accuracy: 0.9490 - loss: 221.5552


 67/Unknown  9s 137ms/step - accuracy: 0.9490 - loss: 221.6626


 68/Unknown  9s 136ms/step - accuracy: 0.9490 - loss: 221.7653


 69/Unknown  10s 136ms/step - accuracy: 0.9490 - loss: 221.8680


 70/Unknown  10s 136ms/step - accuracy: 0.9490 - loss: 221.9582


 71/Unknown  10s 136ms/step - accuracy: 0.9489 - loss: 222.0398


 72/Unknown  10s 136ms/step - accuracy: 0.9489 - loss: 222.1409


 73/Unknown  10s 136ms/step - accuracy: 0.9489 - loss: 222.2496


 74/Unknown  10s 136ms/step - accuracy: 0.9489 - loss: 222.3526


 75/Unknown  10s 136ms/step - accuracy: 0.9489 - loss: 222.4433


 76/Unknown  11s 136ms/step - accuracy: 0.9489 - loss: 222.5272


 77/Unknown  11s 136ms/step - accuracy: 0.9489 - loss: 222.6031


 78/Unknown  11s 136ms/step - accuracy: 0.9488 - loss: 222.6857


 79/Unknown  11s 136ms/step - accuracy: 0.9488 - loss: 222.7623


 80/Unknown  11s 136ms/step - accuracy: 0.9488 - loss: 222.8322


 81/Unknown  11s 136ms/step - accuracy: 0.9488 - loss: 222.8963


 82/Unknown  11s 136ms/step - accuracy: 0.9488 - loss: 222.9694


 83/Unknown  11s 136ms/step - accuracy: 0.9488 - loss: 223.0455


 84/Unknown  12s 136ms/step - accuracy: 0.9488 - loss: 223.1209


 85/Unknown  12s 136ms/step - accuracy: 0.9488 - loss: 223.1990


 86/Unknown  12s 136ms/step - accuracy: 0.9488 - loss: 223.2825


 87/Unknown  12s 136ms/step - accuracy: 0.9487 - loss: 223.3633


 88/Unknown  12s 136ms/step - accuracy: 0.9487 - loss: 223.4366


 89/Unknown  12s 136ms/step - accuracy: 0.9487 - loss: 223.5151


 90/Unknown  12s 136ms/step - accuracy: 0.9487 - loss: 223.5958


 91/Unknown  13s 136ms/step - accuracy: 0.9487 - loss: 223.6727


 92/Unknown  13s 137ms/step - accuracy: 0.9487 - loss: 223.7505


 93/Unknown  13s 137ms/step - accuracy: 0.9487 - loss: 223.8250


 94/Unknown  13s 137ms/step - accuracy: 0.9486 - loss: 223.9114


 95/Unknown  13s 137ms/step - accuracy: 0.9486 - loss: 223.9948


 96/Unknown  13s 138ms/step - accuracy: 0.9486 - loss: 224.0807


 97/Unknown  14s 138ms/step - accuracy: 0.9486 - loss: 224.1586


 98/Unknown  14s 138ms/step - accuracy: 0.9486 - loss: 224.2289


 99/Unknown  14s 138ms/step - accuracy: 0.9486 - loss: 224.2979


100/Unknown  14s 138ms/step - accuracy: 0.9486 - loss: 224.3739


101/Unknown  14s 138ms/step - accuracy: 0.9485 - loss: 224.4488


102/Unknown  14s 138ms/step - accuracy: 0.9485 - loss: 224.5210


103/Unknown  14s 139ms/step - accuracy: 0.9485 - loss: 224.5936


104/Unknown  15s 139ms/step - accuracy: 0.9485 - loss: 224.6630


105/Unknown  15s 139ms/step - accuracy: 0.9485 - loss: 224.7316


106/Unknown  15s 139ms/step - accuracy: 0.9485 - loss: 224.8002


107/Unknown  15s 139ms/step - accuracy: 0.9485 - loss: 224.8736


108/Unknown  15s 139ms/step - accuracy: 0.9484 - loss: 224.9466


109/Unknown  15s 138ms/step - accuracy: 0.9484 - loss: 225.0268


110/Unknown  15s 138ms/step - accuracy: 0.9484 - loss: 225.1065


111/Unknown  16s 138ms/step - accuracy: 0.9484 - loss: 225.1895


112/Unknown  16s 139ms/step - accuracy: 0.9484 - loss: 225.2730


113/Unknown  16s 139ms/step - accuracy: 0.9484 - loss: 225.3562


114/Unknown  16s 139ms/step - accuracy: 0.9484 - loss: 225.4317


115/Unknown  16s 139ms/step - accuracy: 0.9484 - loss: 225.5018


116/Unknown  16s 139ms/step - accuracy: 0.9483 - loss: 225.5749


117/Unknown  16s 139ms/step - accuracy: 0.9483 - loss: 225.6508


118/Unknown  17s 139ms/step - accuracy: 0.9483 - loss: 225.7233


119/Unknown  17s 139ms/step - accuracy: 0.9483 - loss: 225.7933


120/Unknown  17s 139ms/step - accuracy: 0.9483 - loss: 225.8627


121/Unknown  17s 139ms/step - accuracy: 0.9483 - loss: 225.9334


122/Unknown  17s 139ms/step - accuracy: 0.9483 - loss: 226.0017


123/Unknown  17s 139ms/step - accuracy: 0.9483 - loss: 226.0612


124/Unknown  17s 139ms/step - accuracy: 0.9483 - loss: 226.1206


125/Unknown  18s 139ms/step - accuracy: 0.9482 - loss: 226.1817


126/Unknown  18s 139ms/step - accuracy: 0.9482 - loss: 226.2397


127/Unknown  18s 139ms/step - accuracy: 0.9482 - loss: 226.2974


128/Unknown  18s 139ms/step - accuracy: 0.9482 - loss: 226.3512


129/Unknown  18s 139ms/step - accuracy: 0.9482 - loss: 226.4005


130/Unknown  18s 139ms/step - accuracy: 0.9482 - loss: 226.4489


131/Unknown  18s 139ms/step - accuracy: 0.9482 - loss: 226.4945


132/Unknown  19s 139ms/step - accuracy: 0.9482 - loss: 226.5418


133/Unknown  19s 139ms/step - accuracy: 0.9482 - loss: 226.5853


134/Unknown  19s 140ms/step - accuracy: 0.9482 - loss: 226.6308


135/Unknown  19s 140ms/step - accuracy: 0.9482 - loss: 226.6727


136/Unknown  19s 140ms/step - accuracy: 0.9482 - loss: 226.7198


137/Unknown  19s 140ms/step - accuracy: 0.9482 - loss: 226.7688


138/Unknown  19s 140ms/step - accuracy: 0.9482 - loss: 226.8152


139/Unknown  20s 140ms/step - accuracy: 0.9482 - loss: 226.8609


140/Unknown  20s 140ms/step - accuracy: 0.9482 - loss: 226.9056


141/Unknown  20s 140ms/step - accuracy: 0.9481 - loss: 226.9498


142/Unknown  20s 140ms/step - accuracy: 0.9481 - loss: 226.9936


143/Unknown  20s 140ms/step - accuracy: 0.9481 - loss: 227.0366


144/Unknown  20s 139ms/step - accuracy: 0.9481 - loss: 227.0785


145/Unknown  20s 139ms/step - accuracy: 0.9481 - loss: 227.1189


146/Unknown  21s 139ms/step - accuracy: 0.9481 - loss: 227.1589


147/Unknown  21s 139ms/step - accuracy: 0.9481 - loss: 227.1976


148/Unknown  21s 140ms/step - accuracy: 0.9481 - loss: 227.2341


149/Unknown  21s 140ms/step - accuracy: 0.9481 - loss: 227.2737


150/Unknown  21s 140ms/step - accuracy: 0.9481 - loss: 227.3114


151/Unknown  21s 140ms/step - accuracy: 0.9481 - loss: 227.3518


152/Unknown  21s 140ms/step - accuracy: 0.9481 - loss: 227.3940


153/Unknown  22s 140ms/step - accuracy: 0.9481 - loss: 227.4359


154/Unknown  22s 140ms/step - accuracy: 0.9481 - loss: 227.4770


155/Unknown  22s 140ms/step - accuracy: 0.9481 - loss: 227.5203


156/Unknown  22s 140ms/step - accuracy: 0.9481 - loss: 227.5634


157/Unknown  22s 140ms/step - accuracy: 0.9481 - loss: 227.6052


158/Unknown  22s 140ms/step - accuracy: 0.9481 - loss: 227.6457


159/Unknown  22s 140ms/step - accuracy: 0.9481 - loss: 227.6858


160/Unknown  23s 140ms/step - accuracy: 0.9481 - loss: 227.7236


161/Unknown  23s 140ms/step - accuracy: 0.9481 - loss: 227.7598


162/Unknown  23s 140ms/step - accuracy: 0.9481 - loss: 227.7944


163/Unknown  23s 140ms/step - accuracy: 0.9481 - loss: 227.8277


164/Unknown  23s 140ms/step - accuracy: 0.9481 - loss: 227.8596


165/Unknown  23s 140ms/step - accuracy: 0.9480 - loss: 227.8896


166/Unknown  23s 140ms/step - accuracy: 0.9480 - loss: 227.9187


167/Unknown  24s 140ms/step - accuracy: 0.9480 - loss: 227.9463


168/Unknown  24s 140ms/step - accuracy: 0.9480 - loss: 227.9726


169/Unknown  24s 140ms/step - accuracy: 0.9480 - loss: 227.9972


170/Unknown  24s 140ms/step - accuracy: 0.9480 - loss: 228.0219


171/Unknown  24s 140ms/step - accuracy: 0.9480 - loss: 228.0469


172/Unknown  24s 140ms/step - accuracy: 0.9480 - loss: 228.0727


173/Unknown  24s 140ms/step - accuracy: 0.9480 - loss: 228.0992


174/Unknown  25s 140ms/step - accuracy: 0.9480 - loss: 228.1232


175/Unknown  25s 140ms/step - accuracy: 0.9480 - loss: 228.1466


176/Unknown  25s 140ms/step - accuracy: 0.9480 - loss: 228.1700


177/Unknown  25s 140ms/step - accuracy: 0.9480 - loss: 228.1928


178/Unknown  25s 140ms/step - accuracy: 0.9480 - loss: 228.2146


179/Unknown  25s 140ms/step - accuracy: 0.9480 - loss: 228.2334


180/Unknown  25s 140ms/step - accuracy: 0.9480 - loss: 228.2502


181/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.2660


182/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.2829


183/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.3001


184/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.3197


185/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.3382


186/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.3550


187/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.3710


188/Unknown  26s 140ms/step - accuracy: 0.9480 - loss: 228.3866


189/Unknown  27s 140ms/step - accuracy: 0.9480 - loss: 228.4023


190/Unknown  27s 140ms/step - accuracy: 0.9480 - loss: 228.4187


191/Unknown  27s 140ms/step - accuracy: 0.9480 - loss: 228.4339


192/Unknown  27s 139ms/step - accuracy: 0.9480 - loss: 228.4476


193/Unknown  27s 139ms/step - accuracy: 0.9480 - loss: 228.4600


194/Unknown  27s 139ms/step - accuracy: 0.9480 - loss: 228.4714


195/Unknown  27s 139ms/step - accuracy: 0.9480 - loss: 228.4833


196/Unknown  27s 139ms/step - accuracy: 0.9480 - loss: 228.4972


197/Unknown  28s 139ms/step - accuracy: 0.9480 - loss: 228.5084


198/Unknown  28s 139ms/step - accuracy: 0.9480 - loss: 228.5233


199/Unknown  28s 139ms/step - accuracy: 0.9480 - loss: 228.5378


200/Unknown  28s 139ms/step - accuracy: 0.9480 - loss: 228.5527


201/Unknown  28s 139ms/step - accuracy: 0.9480 - loss: 228.5660


202/Unknown  28s 139ms/step - accuracy: 0.9480 - loss: 228.5789


203/Unknown  28s 139ms/step - accuracy: 0.9480 - loss: 228.5915


204/Unknown  29s 139ms/step - accuracy: 0.9480 - loss: 228.6026


205/Unknown  29s 139ms/step - accuracy: 0.9480 - loss: 228.6157


206/Unknown  29s 139ms/step - accuracy: 0.9480 - loss: 228.6292


207/Unknown  29s 139ms/step - accuracy: 0.9481 - loss: 228.6418


208/Unknown  29s 139ms/step - accuracy: 0.9481 - loss: 228.6543


209/Unknown  29s 139ms/step - accuracy: 0.9481 - loss: 228.6688


210/Unknown  29s 139ms/step - accuracy: 0.9481 - loss: 228.6819


211/Unknown  29s 139ms/step - accuracy: 0.9481 - loss: 228.6941


212/Unknown  30s 139ms/step - accuracy: 0.9481 - loss: 228.7058


213/Unknown  30s 139ms/step - accuracy: 0.9481 - loss: 228.7174


214/Unknown  30s 139ms/step - accuracy: 0.9481 - loss: 228.7294


215/Unknown  30s 139ms/step - accuracy: 0.9481 - loss: 228.7410


216/Unknown  30s 139ms/step - accuracy: 0.9481 - loss: 228.7537


217/Unknown  30s 139ms/step - accuracy: 0.9481 - loss: 228.7657


218/Unknown  30s 139ms/step - accuracy: 0.9481 - loss: 228.7765


219/Unknown  31s 139ms/step - accuracy: 0.9481 - loss: 228.7861


220/Unknown  31s 139ms/step - accuracy: 0.9481 - loss: 228.7945


221/Unknown  31s 139ms/step - accuracy: 0.9481 - loss: 228.8020


222/Unknown  31s 139ms/step - accuracy: 0.9481 - loss: 228.8082


223/Unknown  31s 139ms/step - accuracy: 0.9481 - loss: 228.8140


224/Unknown  31s 138ms/step - accuracy: 0.9481 - loss: 228.8199


225/Unknown  31s 138ms/step - accuracy: 0.9481 - loss: 228.8271


226/Unknown  31s 138ms/step - accuracy: 0.9481 - loss: 228.8340


227/Unknown  32s 138ms/step - accuracy: 0.9481 - loss: 228.8403


228/Unknown  32s 138ms/step - accuracy: 0.9481 - loss: 228.8456


229/Unknown  32s 138ms/step - accuracy: 0.9481 - loss: 228.8512


230/Unknown  32s 138ms/step - accuracy: 0.9481 - loss: 228.8562


231/Unknown  32s 138ms/step - accuracy: 0.9481 - loss: 228.8611


232/Unknown  32s 138ms/step - accuracy: 0.9481 - loss: 228.8642


233/Unknown  32s 138ms/step - accuracy: 0.9481 - loss: 228.8672


234/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8694


235/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8721


236/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8762


237/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8797


238/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8838


239/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8885


240/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8919


241/Unknown  33s 138ms/step - accuracy: 0.9481 - loss: 228.8947


242/Unknown  34s 138ms/step - accuracy: 0.9481 - loss: 228.8974


243/Unknown  34s 138ms/step - accuracy: 0.9481 - loss: 228.9001


244/Unknown  34s 138ms/step - accuracy: 0.9481 - loss: 228.9023


245/Unknown  34s 138ms/step - accuracy: 0.9481 - loss: 228.9047


246/Unknown  34s 138ms/step - accuracy: 0.9481 - loss: 228.9063


247/Unknown  34s 139ms/step - accuracy: 0.9481 - loss: 228.9082


248/Unknown  35s 139ms/step - accuracy: 0.9481 - loss: 228.9095


249/Unknown  35s 139ms/step - accuracy: 0.9482 - loss: 228.9103


250/Unknown  35s 139ms/step - accuracy: 0.9482 - loss: 228.9118


251/Unknown  35s 139ms/step - accuracy: 0.9482 - loss: 228.9147


252/Unknown  35s 139ms/step - accuracy: 0.9482 - loss: 228.9173


253/Unknown  35s 139ms/step - accuracy: 0.9482 - loss: 228.9200


254/Unknown  36s 139ms/step - accuracy: 0.9482 - loss: 228.9222


255/Unknown  36s 139ms/step - accuracy: 0.9482 - loss: 228.9243


256/Unknown  36s 139ms/step - accuracy: 0.9482 - loss: 228.9268


257/Unknown  36s 139ms/step - accuracy: 0.9482 - loss: 228.9288


258/Unknown  36s 140ms/step - accuracy: 0.9482 - loss: 228.9315


259/Unknown  36s 140ms/step - accuracy: 0.9482 - loss: 228.9344


260/Unknown  36s 140ms/step - accuracy: 0.9482 - loss: 228.9363


261/Unknown  37s 140ms/step - accuracy: 0.9482 - loss: 228.9379


262/Unknown  37s 140ms/step - accuracy: 0.9482 - loss: 228.9381


263/Unknown  37s 140ms/step - accuracy: 0.9482 - loss: 228.9378


264/Unknown  37s 140ms/step - accuracy: 0.9482 - loss: 228.9372


265/Unknown  37s 140ms/step - accuracy: 0.9482 - loss: 228.9366


266/Unknown  37s 140ms/step - accuracy: 0.9482 - loss: 228.9373


267/Unknown  37s 140ms/step - accuracy: 0.9482 - loss: 228.9385


268/Unknown  38s 140ms/step - accuracy: 0.9482 - loss: 228.9391


269/Unknown  38s 140ms/step - accuracy: 0.9482 - loss: 228.9401


270/Unknown  38s 140ms/step - accuracy: 0.9482 - loss: 228.9409


271/Unknown  38s 139ms/step - accuracy: 0.9482 - loss: 228.9415


272/Unknown  38s 139ms/step - accuracy: 0.9482 - loss: 228.9413


273/Unknown  38s 139ms/step - accuracy: 0.9482 - loss: 228.9413


274/Unknown  38s 139ms/step - accuracy: 0.9482 - loss: 228.9406


275/Unknown  38s 139ms/step - accuracy: 0.9482 - loss: 228.9396


276/Unknown  39s 139ms/step - accuracy: 0.9482 - loss: 228.9395


277/Unknown  39s 139ms/step - accuracy: 0.9482 - loss: 228.9393


278/Unknown  39s 139ms/step - accuracy: 0.9482 - loss: 228.9390


279/Unknown  39s 139ms/step - accuracy: 0.9482 - loss: 228.9383


280/Unknown  39s 139ms/step - accuracy: 0.9483 - loss: 228.9368


281/Unknown  39s 139ms/step - accuracy: 0.9483 - loss: 228.9358


282/Unknown  39s 139ms/step - accuracy: 0.9483 - loss: 228.9339


283/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9312


284/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9295


285/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9271


286/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9249


287/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9228


288/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9200


289/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9166


290/Unknown  40s 139ms/step - accuracy: 0.9483 - loss: 228.9136


291/Unknown  41s 139ms/step - accuracy: 0.9483 - loss: 228.9105


292/Unknown  41s 139ms/step - accuracy: 0.9483 - loss: 228.9076


293/Unknown  41s 139ms/step - accuracy: 0.9483 - loss: 228.9045


294/Unknown  41s 139ms/step - accuracy: 0.9483 - loss: 228.9017


295/Unknown  41s 139ms/step - accuracy: 0.9483 - loss: 228.8983


296/Unknown  41s 139ms/step - accuracy: 0.9483 - loss: 228.8967


297/Unknown  41s 139ms/step - accuracy: 0.9483 - loss: 228.8956


298/Unknown  42s 139ms/step - accuracy: 0.9483 - loss: 228.8946


299/Unknown  42s 139ms/step - accuracy: 0.9483 - loss: 228.8931


300/Unknown  42s 139ms/step - accuracy: 0.9483 - loss: 228.8911


301/Unknown  42s 139ms/step - accuracy: 0.9483 - loss: 228.8904


302/Unknown  42s 139ms/step - accuracy: 0.9483 - loss: 228.8897


303/Unknown  42s 139ms/step - accuracy: 0.9483 - loss: 228.8902


304/Unknown  42s 139ms/step - accuracy: 0.9484 - loss: 228.8911


305/Unknown  43s 139ms/step - accuracy: 0.9484 - loss: 228.8920


306/Unknown  43s 139ms/step - accuracy: 0.9484 - loss: 228.8923


307/Unknown  43s 139ms/step - accuracy: 0.9484 - loss: 228.8938


308/Unknown  43s 139ms/step - accuracy: 0.9484 - loss: 228.8946


309/Unknown  43s 139ms/step - accuracy: 0.9484 - loss: 228.8958


310/Unknown  43s 139ms/step - accuracy: 0.9484 - loss: 228.8979


311/Unknown  44s 139ms/step - accuracy: 0.9484 - loss: 228.9006


312/Unknown  44s 139ms/step - accuracy: 0.9484 - loss: 228.9027


313/Unknown  44s 139ms/step - accuracy: 0.9484 - loss: 228.9051


314/Unknown  44s 139ms/step - accuracy: 0.9484 - loss: 228.9074


315/Unknown  44s 140ms/step - accuracy: 0.9484 - loss: 228.9098


316/Unknown  44s 140ms/step - accuracy: 0.9484 - loss: 228.9117


317/Unknown  44s 140ms/step - accuracy: 0.9484 - loss: 228.9142


318/Unknown  45s 140ms/step - accuracy: 0.9484 - loss: 228.9171


319/Unknown  45s 140ms/step - accuracy: 0.9484 - loss: 228.9203


320/Unknown  45s 140ms/step - accuracy: 0.9484 - loss: 228.9232


321/Unknown  45s 140ms/step - accuracy: 0.9484 - loss: 228.9262


322/Unknown  45s 140ms/step - accuracy: 0.9484 - loss: 228.9293


323/Unknown  45s 140ms/step - accuracy: 0.9484 - loss: 228.9324


324/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9353


325/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9387


326/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9429


327/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9469


328/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9508


329/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9543


330/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9582


331/Unknown  46s 140ms/step - accuracy: 0.9484 - loss: 228.9617


332/Unknown  47s 140ms/step - accuracy: 0.9484 - loss: 228.9662


333/Unknown  47s 140ms/step - accuracy: 0.9484 - loss: 228.9707


334/Unknown  47s 140ms/step - accuracy: 0.9484 - loss: 228.9747


335/Unknown  47s 140ms/step - accuracy: 0.9484 - loss: 228.9783


336/Unknown  47s 140ms/step - accuracy: 0.9485 - loss: 228.9814


337/Unknown  47s 140ms/step - accuracy: 0.9485 - loss: 228.9847


338/Unknown  47s 140ms/step - accuracy: 0.9485 - loss: 228.9882


339/Unknown  47s 140ms/step - accuracy: 0.9485 - loss: 228.9913


340/Unknown  48s 139ms/step - accuracy: 0.9485 - loss: 228.9950


341/Unknown  48s 139ms/step - accuracy: 0.9485 - loss: 228.9996


342/Unknown  48s 139ms/step - accuracy: 0.9485 - loss: 229.0038


343/Unknown  48s 139ms/step - accuracy: 0.9485 - loss: 229.0080


344/Unknown  48s 139ms/step - accuracy: 0.9485 - loss: 229.0122


345/Unknown  48s 139ms/step - accuracy: 0.9485 - loss: 229.0163


346/Unknown  48s 139ms/step - accuracy: 0.9485 - loss: 229.0210


347/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0256


348/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0313


349/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0372


350/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0429


351/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0492


352/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0551


353/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0615


354/Unknown  49s 139ms/step - accuracy: 0.9485 - loss: 229.0690


355/Unknown  50s 139ms/step - accuracy: 0.9485 - loss: 229.0767


356/Unknown  50s 139ms/step - accuracy: 0.9485 - loss: 229.0841


357/Unknown  50s 139ms/step - accuracy: 0.9485 - loss: 229.0911


358/Unknown  50s 139ms/step - accuracy: 0.9485 - loss: 229.0985


359/Unknown  50s 139ms/step - accuracy: 0.9485 - loss: 229.1061


360/Unknown  50s 139ms/step - accuracy: 0.9485 - loss: 229.1136


361/Unknown  50s 139ms/step - accuracy: 0.9485 - loss: 229.1216


362/Unknown  51s 139ms/step - accuracy: 0.9485 - loss: 229.1297


363/Unknown  51s 139ms/step - accuracy: 0.9485 - loss: 229.1375


364/Unknown  51s 139ms/step - accuracy: 0.9485 - loss: 229.1453


365/Unknown  51s 139ms/step - accuracy: 0.9485 - loss: 229.1526


366/Unknown  51s 139ms/step - accuracy: 0.9485 - loss: 229.1591


367/Unknown  51s 139ms/step - accuracy: 0.9485 - loss: 229.1653


368/Unknown  51s 139ms/step - accuracy: 0.9485 - loss: 229.1717


369/Unknown  52s 139ms/step - accuracy: 0.9485 - loss: 229.1778


370/Unknown  52s 139ms/step - accuracy: 0.9485 - loss: 229.1837


371/Unknown  52s 139ms/step - accuracy: 0.9485 - loss: 229.1893


372/Unknown  52s 139ms/step - accuracy: 0.9485 - loss: 229.1947


373/Unknown  52s 139ms/step - accuracy: 0.9485 - loss: 229.1999


374/Unknown  52s 139ms/step - accuracy: 0.9485 - loss: 229.2054


375/Unknown  52s 139ms/step - accuracy: 0.9485 - loss: 229.2110


376/Unknown  53s 139ms/step - accuracy: 0.9486 - loss: 229.2163


377/Unknown  53s 139ms/step - accuracy: 0.9486 - loss: 229.2217


```
</div>
 377/377 ━━━━━━━━━━━━━━━━━━━━ 53s 139ms/step - accuracy: 0.9486 - loss: 229.2270


<div class="k-default-codeblock">
```
Test accuracy: 94.94%

```
</div>
You should achieve more than 95% accuracy on the test set.

To increase the learning capacity of the model, you can try increasing the
`encoding_size` value, or stacking multiple GRN layers on top of the VSN layer.
This may require to also increase the `dropout_rate` value to avoid overfitting.

**Example available on HuggingFace**

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Classification%20With%20GRN%20%26%20VSN-red)](https://huggingface.co/keras-io/structured-data-classification-grn-vsn) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Classification%20With%20GRN%20%26%20VSN-red)](https://huggingface.co/spaces/keras-io/structured-data-classification-grn-vsn) |
