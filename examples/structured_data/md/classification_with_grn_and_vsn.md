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

Clean the directory for the downloaded files except the .tar.gz file and
also remove the empty directories


```python
subprocess.run(
    f'find {extracted_path} -type f ! -name "*.tar.gz" -exec rm -f {{}} +',
    shell=True,
    check=True,
)
subprocess.run(
    f"find {extracted_path} -type d -empty -exec rmdir {{}} +", shell=True, check=True
)
```




<div class="k-default-codeblock">
```
CompletedProcess(args='find /home/humbulani/.keras/datasets/census+income+kdd.zip -type d -empty -exec rmdir {} +', returncode=0)

```
</div>
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

        # The reason for each individual backend calculation is that I couldn't find
        # the equivalent keras operation that is backend-agnostic. In the following case there,s
        # a keras.ops.matmul but it was returning errors. I could have used the tensorflow matmul
        # for all backends, but due to jax jit tracing it results in an error.
        def matmul_dependent_on_backend(thsi, v):
            """
            Function for executing matmul for each backend.
            """
            # jax backend
            if keras.backend.backend() == "jax":
                import jax.numpy as jnp

                result = jnp.sum(thsi * v, axis=1)
            elif keras.backend.backend() == "torch":
                result = torch.sum(thsi * v, dim=1)
            # tensorflow backend
            elif keras.backend.backend() == "tensorflow":
                result = keras.ops.squeeze(tf.matmul(thsi, v, transpose_a=True), axis=1)
            # unsupported backend exception
            else:
                raise ValueError(
                    "Unsupported backend: {}".format(keras.backend.backend())
                )
            return result

        # jax backend
        if keras.backend.backend() == "jax":
            # This repetative imports are intentional to force the idea of backend
            # separation
            import jax.numpy as jnp

            result_jax = matmul_dependent_on_backend(v, x)
            return result_jax
        # torch backend
        if keras.backend.backend() == "torch":
            import torch

            result_torch = matmul_dependent_on_backend(v, x)
            return result_torch
        # tensorflow backend
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            result_tf = keras.ops.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
            return result_tf

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
num_epochs = 1  # maybe adjusted to a desired value
encoding_size = 16

model = create_model(encoding_size)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)
```

<div class="k-default-codeblock">
```
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_1', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_2', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_3', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_4', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_5', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_6', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_7', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_8', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_9', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_10', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_11', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_12', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_13', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_14', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_15', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_16', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_17', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_18', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_19', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_20', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_21', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_22', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_23', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_24', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_25', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_26', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_27', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_28', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_29', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_30', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_31', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_32', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_33', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_34', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_35', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_36', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_37', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_38', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(
/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/layers/layer.py:391: UserWarning: `build()` was called on layer 'gated_residual_network_39', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state. This will cause the layer to be marked as built, despite not being actually built, which may cause failures down the line. Make sure to implement a proper `build()` method.
  warnings.warn(

```
</div>
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
  1/Unknown  1s 740ms/step - accuracy: 0.2491 - loss: 1302.2852


  2/Unknown  1s 193ms/step - accuracy: 0.4104 - loss: 1243.1028


  3/Unknown  1s 207ms/step - accuracy: 0.5046 - loss: 1190.0552


  4/Unknown  1s 211ms/step - accuracy: 0.5667 - loss: 1140.0157


  5/Unknown  2s 211ms/step - accuracy: 0.6118 - loss: 1094.2517


  6/Unknown  2s 215ms/step - accuracy: 0.6458 - loss: 1052.5693


  7/Unknown  2s 216ms/step - accuracy: 0.6727 - loss: 1015.1872


  8/Unknown  2s 215ms/step - accuracy: 0.6942 - loss: 982.8596 


  9/Unknown  2s 215ms/step - accuracy: 0.7121 - loss: 953.8212


 10/Unknown  3s 214ms/step - accuracy: 0.7272 - loss: 927.4525


 11/Unknown  3s 214ms/step - accuracy: 0.7400 - loss: 904.0518


 12/Unknown  3s 215ms/step - accuracy: 0.7512 - loss: 882.7589


 13/Unknown  3s 219ms/step - accuracy: 0.7611 - loss: 862.8194


 14/Unknown  4s 219ms/step - accuracy: 0.7700 - loss: 844.1758


 15/Unknown  4s 222ms/step - accuracy: 0.7780 - loss: 826.9224


 16/Unknown  4s 224ms/step - accuracy: 0.7851 - loss: 810.8615


 17/Unknown  4s 224ms/step - accuracy: 0.7916 - loss: 795.8868


 18/Unknown  5s 223ms/step - accuracy: 0.7975 - loss: 782.1332


 19/Unknown  5s 225ms/step - accuracy: 0.8029 - loss: 769.1092


 20/Unknown  5s 226ms/step - accuracy: 0.8079 - loss: 756.8516


 21/Unknown  5s 225ms/step - accuracy: 0.8125 - loss: 745.5027


 22/Unknown  5s 223ms/step - accuracy: 0.8168 - loss: 734.6467


 23/Unknown  6s 222ms/step - accuracy: 0.8207 - loss: 724.4232


 24/Unknown  6s 222ms/step - accuracy: 0.8245 - loss: 714.9254


 25/Unknown  6s 221ms/step - accuracy: 0.8279 - loss: 706.1494


 26/Unknown  6s 220ms/step - accuracy: 0.8311 - loss: 697.8618


 27/Unknown  6s 219ms/step - accuracy: 0.8341 - loss: 690.1976


 28/Unknown  7s 218ms/step - accuracy: 0.8369 - loss: 682.8348


 29/Unknown  7s 218ms/step - accuracy: 0.8395 - loss: 675.9014


 30/Unknown  7s 220ms/step - accuracy: 0.8420 - loss: 669.2024


 31/Unknown  7s 219ms/step - accuracy: 0.8444 - loss: 662.8259


 32/Unknown  8s 219ms/step - accuracy: 0.8467 - loss: 656.7582


 33/Unknown  8s 219ms/step - accuracy: 0.8488 - loss: 650.9562


 34/Unknown  8s 218ms/step - accuracy: 0.8508 - loss: 645.4325


 35/Unknown  8s 218ms/step - accuracy: 0.8527 - loss: 640.0767


 36/Unknown  8s 218ms/step - accuracy: 0.8546 - loss: 634.9040


 37/Unknown  9s 217ms/step - accuracy: 0.8563 - loss: 629.9690


 38/Unknown  9s 216ms/step - accuracy: 0.8580 - loss: 625.1686


 39/Unknown  9s 215ms/step - accuracy: 0.8596 - loss: 620.4742


 40/Unknown  9s 215ms/step - accuracy: 0.8611 - loss: 616.0271


 41/Unknown  9s 215ms/step - accuracy: 0.8625 - loss: 611.7262


 42/Unknown  10s 215ms/step - accuracy: 0.8639 - loss: 607.5671


 43/Unknown  10s 214ms/step - accuracy: 0.8653 - loss: 603.5233


 44/Unknown  10s 214ms/step - accuracy: 0.8666 - loss: 599.6008


 45/Unknown  10s 214ms/step - accuracy: 0.8679 - loss: 595.7900


 46/Unknown  10s 213ms/step - accuracy: 0.8691 - loss: 592.0447


 47/Unknown  11s 214ms/step - accuracy: 0.8702 - loss: 588.3965


 48/Unknown  11s 214ms/step - accuracy: 0.8713 - loss: 584.8365


 49/Unknown  11s 214ms/step - accuracy: 0.8724 - loss: 581.3633


 50/Unknown  11s 215ms/step - accuracy: 0.8735 - loss: 577.9586


 51/Unknown  11s 215ms/step - accuracy: 0.8745 - loss: 574.6376


 52/Unknown  12s 215ms/step - accuracy: 0.8755 - loss: 571.4106


 53/Unknown  12s 215ms/step - accuracy: 0.8764 - loss: 568.2571


 54/Unknown  12s 215ms/step - accuracy: 0.8773 - loss: 565.1942


 55/Unknown  12s 215ms/step - accuracy: 0.8782 - loss: 562.1995


 56/Unknown  13s 215ms/step - accuracy: 0.8791 - loss: 559.2789


 57/Unknown  13s 215ms/step - accuracy: 0.8799 - loss: 556.4360


 58/Unknown  13s 215ms/step - accuracy: 0.8807 - loss: 553.6384


 59/Unknown  13s 215ms/step - accuracy: 0.8815 - loss: 550.8985


 60/Unknown  13s 215ms/step - accuracy: 0.8823 - loss: 548.2059


 61/Unknown  14s 219ms/step - accuracy: 0.8830 - loss: 545.5831


 62/Unknown  14s 218ms/step - accuracy: 0.8838 - loss: 543.0310


 63/Unknown  14s 218ms/step - accuracy: 0.8845 - loss: 540.5276


 64/Unknown  14s 217ms/step - accuracy: 0.8852 - loss: 538.0663


 65/Unknown  15s 217ms/step - accuracy: 0.8858 - loss: 535.6724


 66/Unknown  15s 216ms/step - accuracy: 0.8865 - loss: 533.3412


 67/Unknown  15s 216ms/step - accuracy: 0.8871 - loss: 531.0530


 68/Unknown  15s 215ms/step - accuracy: 0.8877 - loss: 528.8370


 69/Unknown  15s 215ms/step - accuracy: 0.8883 - loss: 526.6666


 70/Unknown  16s 215ms/step - accuracy: 0.8889 - loss: 524.5291


 71/Unknown  16s 215ms/step - accuracy: 0.8894 - loss: 522.4325


 72/Unknown  16s 214ms/step - accuracy: 0.8900 - loss: 520.3626


 73/Unknown  16s 214ms/step - accuracy: 0.8905 - loss: 518.3271


 74/Unknown  16s 214ms/step - accuracy: 0.8910 - loss: 516.3285


 75/Unknown  17s 214ms/step - accuracy: 0.8915 - loss: 514.3654


 76/Unknown  17s 213ms/step - accuracy: 0.8920 - loss: 512.4387


 77/Unknown  17s 213ms/step - accuracy: 0.8925 - loss: 510.5423


 78/Unknown  17s 213ms/step - accuracy: 0.8930 - loss: 508.6753


 79/Unknown  17s 213ms/step - accuracy: 0.8935 - loss: 506.8432


 80/Unknown  18s 213ms/step - accuracy: 0.8939 - loss: 505.0417


 81/Unknown  18s 212ms/step - accuracy: 0.8944 - loss: 503.2594


 82/Unknown  18s 212ms/step - accuracy: 0.8948 - loss: 501.5193


 83/Unknown  18s 212ms/step - accuracy: 0.8953 - loss: 499.8059


 84/Unknown  18s 212ms/step - accuracy: 0.8957 - loss: 498.1246


 85/Unknown  19s 212ms/step - accuracy: 0.8961 - loss: 496.4738


 86/Unknown  19s 213ms/step - accuracy: 0.8965 - loss: 494.8399


 87/Unknown  19s 213ms/step - accuracy: 0.8969 - loss: 493.2329


 88/Unknown  19s 213ms/step - accuracy: 0.8973 - loss: 491.6490


 89/Unknown  19s 213ms/step - accuracy: 0.8977 - loss: 490.0891


 90/Unknown  20s 213ms/step - accuracy: 0.8981 - loss: 488.5493


 91/Unknown  20s 213ms/step - accuracy: 0.8985 - loss: 487.0343


 92/Unknown  20s 213ms/step - accuracy: 0.8988 - loss: 485.5350


 93/Unknown  20s 213ms/step - accuracy: 0.8992 - loss: 484.0580


 94/Unknown  21s 213ms/step - accuracy: 0.8996 - loss: 482.6119


 95/Unknown  21s 213ms/step - accuracy: 0.8999 - loss: 481.1904


 96/Unknown  21s 213ms/step - accuracy: 0.9002 - loss: 479.7946


 97/Unknown  21s 214ms/step - accuracy: 0.9006 - loss: 478.4108


 98/Unknown  21s 214ms/step - accuracy: 0.9009 - loss: 477.0426


 99/Unknown  22s 214ms/step - accuracy: 0.9012 - loss: 475.6963


100/Unknown  22s 214ms/step - accuracy: 0.9016 - loss: 474.3627


101/Unknown  22s 214ms/step - accuracy: 0.9019 - loss: 473.0454


102/Unknown  22s 214ms/step - accuracy: 0.9022 - loss: 471.7478


103/Unknown  22s 213ms/step - accuracy: 0.9025 - loss: 470.4634


104/Unknown  23s 213ms/step - accuracy: 0.9028 - loss: 469.2029


105/Unknown  23s 213ms/step - accuracy: 0.9031 - loss: 467.9564


106/Unknown  23s 213ms/step - accuracy: 0.9034 - loss: 466.7221


107/Unknown  23s 213ms/step - accuracy: 0.9037 - loss: 465.5094


108/Unknown  24s 213ms/step - accuracy: 0.9040 - loss: 464.3122


109/Unknown  24s 213ms/step - accuracy: 0.9042 - loss: 463.1289


110/Unknown  24s 213ms/step - accuracy: 0.9045 - loss: 461.9557


111/Unknown  24s 213ms/step - accuracy: 0.9048 - loss: 460.7968


112/Unknown  24s 213ms/step - accuracy: 0.9050 - loss: 459.6553


113/Unknown  25s 213ms/step - accuracy: 0.9053 - loss: 458.5283


114/Unknown  25s 213ms/step - accuracy: 0.9056 - loss: 457.4111


115/Unknown  25s 213ms/step - accuracy: 0.9058 - loss: 456.3075


116/Unknown  25s 213ms/step - accuracy: 0.9061 - loss: 455.2115


117/Unknown  25s 213ms/step - accuracy: 0.9063 - loss: 454.1252


118/Unknown  26s 213ms/step - accuracy: 0.9066 - loss: 453.0476


119/Unknown  26s 213ms/step - accuracy: 0.9068 - loss: 451.9829


120/Unknown  26s 213ms/step - accuracy: 0.9070 - loss: 450.9324


121/Unknown  26s 213ms/step - accuracy: 0.9073 - loss: 449.8954


122/Unknown  27s 214ms/step - accuracy: 0.9075 - loss: 448.8740


123/Unknown  27s 214ms/step - accuracy: 0.9077 - loss: 447.8640


124/Unknown  27s 214ms/step - accuracy: 0.9080 - loss: 446.8658


125/Unknown  27s 215ms/step - accuracy: 0.9082 - loss: 445.8753


126/Unknown  28s 215ms/step - accuracy: 0.9084 - loss: 444.8939


127/Unknown  28s 215ms/step - accuracy: 0.9086 - loss: 443.9199


128/Unknown  28s 216ms/step - accuracy: 0.9089 - loss: 442.9591


129/Unknown  28s 217ms/step - accuracy: 0.9091 - loss: 442.0043


130/Unknown  29s 217ms/step - accuracy: 0.9093 - loss: 441.0609


131/Unknown  29s 217ms/step - accuracy: 0.9095 - loss: 440.1299


132/Unknown  29s 217ms/step - accuracy: 0.9097 - loss: 439.2089


133/Unknown  29s 217ms/step - accuracy: 0.9099 - loss: 438.2959


134/Unknown  30s 217ms/step - accuracy: 0.9101 - loss: 437.3937


135/Unknown  30s 217ms/step - accuracy: 0.9103 - loss: 436.5073


136/Unknown  30s 218ms/step - accuracy: 0.9105 - loss: 435.6308


137/Unknown  30s 218ms/step - accuracy: 0.9107 - loss: 434.7618


138/Unknown  31s 218ms/step - accuracy: 0.9109 - loss: 433.9018


139/Unknown  31s 219ms/step - accuracy: 0.9111 - loss: 433.0544


140/Unknown  31s 219ms/step - accuracy: 0.9112 - loss: 432.2228


141/Unknown  31s 219ms/step - accuracy: 0.9114 - loss: 431.3983


142/Unknown  32s 219ms/step - accuracy: 0.9116 - loss: 430.5809


143/Unknown  32s 219ms/step - accuracy: 0.9118 - loss: 429.7700


144/Unknown  32s 219ms/step - accuracy: 0.9120 - loss: 428.9709


145/Unknown  32s 219ms/step - accuracy: 0.9121 - loss: 428.1824


146/Unknown  33s 219ms/step - accuracy: 0.9123 - loss: 427.4028


147/Unknown  33s 219ms/step - accuracy: 0.9125 - loss: 426.6298


148/Unknown  33s 219ms/step - accuracy: 0.9126 - loss: 425.8698


149/Unknown  33s 219ms/step - accuracy: 0.9128 - loss: 425.1159


150/Unknown  33s 219ms/step - accuracy: 0.9130 - loss: 424.3755


151/Unknown  34s 219ms/step - accuracy: 0.9131 - loss: 423.6459


152/Unknown  34s 219ms/step - accuracy: 0.9133 - loss: 422.9241


153/Unknown  34s 219ms/step - accuracy: 0.9134 - loss: 422.2071


154/Unknown  34s 220ms/step - accuracy: 0.9136 - loss: 421.4937


155/Unknown  35s 220ms/step - accuracy: 0.9138 - loss: 420.7878


156/Unknown  35s 220ms/step - accuracy: 0.9139 - loss: 420.0875


157/Unknown  35s 220ms/step - accuracy: 0.9141 - loss: 419.3929


158/Unknown  35s 220ms/step - accuracy: 0.9142 - loss: 418.7023


159/Unknown  35s 220ms/step - accuracy: 0.9143 - loss: 418.0189


160/Unknown  36s 220ms/step - accuracy: 0.9145 - loss: 417.3419


161/Unknown  36s 220ms/step - accuracy: 0.9146 - loss: 416.6698


162/Unknown  36s 220ms/step - accuracy: 0.9148 - loss: 416.0050


163/Unknown  36s 220ms/step - accuracy: 0.9149 - loss: 415.3468


164/Unknown  37s 220ms/step - accuracy: 0.9151 - loss: 414.6968


165/Unknown  37s 220ms/step - accuracy: 0.9152 - loss: 414.0533


166/Unknown  37s 220ms/step - accuracy: 0.9153 - loss: 413.4176


167/Unknown  37s 220ms/step - accuracy: 0.9155 - loss: 412.7845


168/Unknown  37s 220ms/step - accuracy: 0.9156 - loss: 412.1593


169/Unknown  38s 220ms/step - accuracy: 0.9157 - loss: 411.5370


170/Unknown  38s 220ms/step - accuracy: 0.9159 - loss: 410.9182


171/Unknown  38s 220ms/step - accuracy: 0.9160 - loss: 410.3065


172/Unknown  38s 221ms/step - accuracy: 0.9161 - loss: 409.7014


173/Unknown  39s 221ms/step - accuracy: 0.9163 - loss: 409.1011


174/Unknown  39s 221ms/step - accuracy: 0.9164 - loss: 408.5077


175/Unknown  39s 222ms/step - accuracy: 0.9165 - loss: 407.9176


176/Unknown  40s 222ms/step - accuracy: 0.9166 - loss: 407.3345


177/Unknown  40s 222ms/step - accuracy: 0.9168 - loss: 406.7563


178/Unknown  40s 222ms/step - accuracy: 0.9169 - loss: 406.1828


179/Unknown  40s 222ms/step - accuracy: 0.9170 - loss: 405.6138


180/Unknown  40s 222ms/step - accuracy: 0.9171 - loss: 405.0493


181/Unknown  41s 222ms/step - accuracy: 0.9172 - loss: 404.4879


182/Unknown  41s 222ms/step - accuracy: 0.9173 - loss: 403.9296


183/Unknown  41s 222ms/step - accuracy: 0.9175 - loss: 403.3773


184/Unknown  41s 222ms/step - accuracy: 0.9176 - loss: 402.8297


185/Unknown  42s 222ms/step - accuracy: 0.9177 - loss: 402.2837


186/Unknown  42s 223ms/step - accuracy: 0.9178 - loss: 401.7392


187/Unknown  42s 223ms/step - accuracy: 0.9179 - loss: 401.1979


188/Unknown  42s 223ms/step - accuracy: 0.9180 - loss: 400.6618


189/Unknown  43s 223ms/step - accuracy: 0.9181 - loss: 400.1364


190/Unknown  43s 223ms/step - accuracy: 0.9182 - loss: 399.6142


191/Unknown  43s 223ms/step - accuracy: 0.9184 - loss: 399.0972


192/Unknown  43s 224ms/step - accuracy: 0.9185 - loss: 398.5863


193/Unknown  44s 224ms/step - accuracy: 0.9186 - loss: 398.0764


194/Unknown  44s 224ms/step - accuracy: 0.9187 - loss: 397.5734


195/Unknown  44s 224ms/step - accuracy: 0.9188 - loss: 397.0740


196/Unknown  44s 224ms/step - accuracy: 0.9189 - loss: 396.5792


197/Unknown  45s 224ms/step - accuracy: 0.9190 - loss: 396.0892


198/Unknown  45s 225ms/step - accuracy: 0.9191 - loss: 395.6027


199/Unknown  45s 225ms/step - accuracy: 0.9192 - loss: 395.1190


200/Unknown  46s 225ms/step - accuracy: 0.9193 - loss: 394.6403


201/Unknown  46s 225ms/step - accuracy: 0.9194 - loss: 394.1669


202/Unknown  46s 225ms/step - accuracy: 0.9195 - loss: 393.6972


203/Unknown  46s 225ms/step - accuracy: 0.9195 - loss: 393.2319


204/Unknown  46s 225ms/step - accuracy: 0.9196 - loss: 392.7684


205/Unknown  47s 225ms/step - accuracy: 0.9197 - loss: 392.3078


206/Unknown  47s 226ms/step - accuracy: 0.9198 - loss: 391.8511


207/Unknown  47s 226ms/step - accuracy: 0.9199 - loss: 391.3967


208/Unknown  48s 226ms/step - accuracy: 0.9200 - loss: 390.9444


209/Unknown  48s 226ms/step - accuracy: 0.9201 - loss: 390.4966


210/Unknown  48s 227ms/step - accuracy: 0.9202 - loss: 390.0508


211/Unknown  48s 227ms/step - accuracy: 0.9203 - loss: 389.6074


212/Unknown  49s 228ms/step - accuracy: 0.9204 - loss: 389.1664


213/Unknown  49s 228ms/step - accuracy: 0.9205 - loss: 388.7271


214/Unknown  49s 229ms/step - accuracy: 0.9205 - loss: 388.2910


215/Unknown  50s 229ms/step - accuracy: 0.9206 - loss: 387.8583


216/Unknown  50s 229ms/step - accuracy: 0.9207 - loss: 387.4301


217/Unknown  50s 229ms/step - accuracy: 0.9208 - loss: 387.0048


218/Unknown  50s 229ms/step - accuracy: 0.9209 - loss: 386.5822


219/Unknown  51s 229ms/step - accuracy: 0.9210 - loss: 386.1640


220/Unknown  51s 229ms/step - accuracy: 0.9211 - loss: 385.7477


221/Unknown  51s 230ms/step - accuracy: 0.9211 - loss: 385.3330


222/Unknown  51s 230ms/step - accuracy: 0.9212 - loss: 384.9213


223/Unknown  52s 229ms/step - accuracy: 0.9213 - loss: 384.5125


224/Unknown  52s 230ms/step - accuracy: 0.9214 - loss: 384.1057


225/Unknown  52s 229ms/step - accuracy: 0.9215 - loss: 383.7001


226/Unknown  52s 229ms/step - accuracy: 0.9215 - loss: 383.2977


227/Unknown  53s 229ms/step - accuracy: 0.9216 - loss: 382.8967


228/Unknown  53s 229ms/step - accuracy: 0.9217 - loss: 382.4982


229/Unknown  53s 229ms/step - accuracy: 0.9218 - loss: 382.1027


230/Unknown  53s 230ms/step - accuracy: 0.9219 - loss: 381.7113


231/Unknown  54s 230ms/step - accuracy: 0.9219 - loss: 381.3225


232/Unknown  54s 230ms/step - accuracy: 0.9220 - loss: 380.9357


233/Unknown  54s 230ms/step - accuracy: 0.9221 - loss: 380.5510


234/Unknown  54s 230ms/step - accuracy: 0.9222 - loss: 380.1692


235/Unknown  55s 230ms/step - accuracy: 0.9222 - loss: 379.7904


236/Unknown  55s 230ms/step - accuracy: 0.9223 - loss: 379.4137


237/Unknown  55s 230ms/step - accuracy: 0.9224 - loss: 379.0397


238/Unknown  55s 230ms/step - accuracy: 0.9225 - loss: 378.6663


239/Unknown  55s 230ms/step - accuracy: 0.9225 - loss: 378.2958


240/Unknown  56s 230ms/step - accuracy: 0.9226 - loss: 377.9276


241/Unknown  56s 230ms/step - accuracy: 0.9227 - loss: 377.5616


242/Unknown  56s 231ms/step - accuracy: 0.9228 - loss: 377.1977


243/Unknown  57s 231ms/step - accuracy: 0.9228 - loss: 376.8351


244/Unknown  57s 231ms/step - accuracy: 0.9229 - loss: 376.4753


245/Unknown  57s 231ms/step - accuracy: 0.9230 - loss: 376.1172


246/Unknown  57s 231ms/step - accuracy: 0.9231 - loss: 375.7617


247/Unknown  58s 231ms/step - accuracy: 0.9231 - loss: 375.4093


248/Unknown  58s 231ms/step - accuracy: 0.9232 - loss: 375.0621


249/Unknown  58s 231ms/step - accuracy: 0.9233 - loss: 374.7164


250/Unknown  58s 231ms/step - accuracy: 0.9233 - loss: 374.3738


251/Unknown  58s 231ms/step - accuracy: 0.9234 - loss: 374.0346


252/Unknown  59s 231ms/step - accuracy: 0.9235 - loss: 373.6974


253/Unknown  59s 231ms/step - accuracy: 0.9235 - loss: 373.3607


254/Unknown  59s 231ms/step - accuracy: 0.9236 - loss: 373.0276


255/Unknown  60s 231ms/step - accuracy: 0.9237 - loss: 372.6956


256/Unknown  60s 232ms/step - accuracy: 0.9237 - loss: 372.3641


257/Unknown  60s 232ms/step - accuracy: 0.9238 - loss: 372.0338


258/Unknown  60s 232ms/step - accuracy: 0.9239 - loss: 371.7052


259/Unknown  61s 232ms/step - accuracy: 0.9239 - loss: 371.3809


260/Unknown  61s 232ms/step - accuracy: 0.9240 - loss: 371.0571


261/Unknown  61s 232ms/step - accuracy: 0.9241 - loss: 370.7343


262/Unknown  61s 232ms/step - accuracy: 0.9241 - loss: 370.4119


263/Unknown  62s 232ms/step - accuracy: 0.9242 - loss: 370.0924


264/Unknown  62s 233ms/step - accuracy: 0.9243 - loss: 369.7755


265/Unknown  62s 233ms/step - accuracy: 0.9243 - loss: 369.4610


266/Unknown  62s 233ms/step - accuracy: 0.9244 - loss: 369.1483


267/Unknown  63s 233ms/step - accuracy: 0.9244 - loss: 368.8368


268/Unknown  63s 233ms/step - accuracy: 0.9245 - loss: 368.5270


269/Unknown  63s 233ms/step - accuracy: 0.9246 - loss: 368.2200


270/Unknown  64s 233ms/step - accuracy: 0.9246 - loss: 367.9145


271/Unknown  64s 233ms/step - accuracy: 0.9247 - loss: 367.6098


272/Unknown  64s 233ms/step - accuracy: 0.9247 - loss: 367.3058


273/Unknown  64s 233ms/step - accuracy: 0.9248 - loss: 367.0038


274/Unknown  64s 233ms/step - accuracy: 0.9249 - loss: 366.7027


275/Unknown  65s 233ms/step - accuracy: 0.9249 - loss: 366.4044


276/Unknown  65s 233ms/step - accuracy: 0.9250 - loss: 366.1067


277/Unknown  65s 233ms/step - accuracy: 0.9250 - loss: 365.8106


278/Unknown  65s 233ms/step - accuracy: 0.9251 - loss: 365.5167


279/Unknown  66s 233ms/step - accuracy: 0.9252 - loss: 365.2255


280/Unknown  66s 233ms/step - accuracy: 0.9252 - loss: 364.9359


281/Unknown  66s 233ms/step - accuracy: 0.9253 - loss: 364.6480


282/Unknown  66s 233ms/step - accuracy: 0.9253 - loss: 364.3615


283/Unknown  67s 234ms/step - accuracy: 0.9254 - loss: 364.0771


284/Unknown  67s 234ms/step - accuracy: 0.9254 - loss: 363.7952


285/Unknown  67s 234ms/step - accuracy: 0.9255 - loss: 363.5163


286/Unknown  67s 234ms/step - accuracy: 0.9255 - loss: 363.2393


287/Unknown  68s 234ms/step - accuracy: 0.9256 - loss: 362.9632


288/Unknown  68s 234ms/step - accuracy: 0.9257 - loss: 362.6883


289/Unknown  68s 234ms/step - accuracy: 0.9257 - loss: 362.4147


290/Unknown  68s 234ms/step - accuracy: 0.9258 - loss: 362.1423


291/Unknown  69s 234ms/step - accuracy: 0.9258 - loss: 361.8702


292/Unknown  69s 234ms/step - accuracy: 0.9259 - loss: 361.5998


293/Unknown  69s 234ms/step - accuracy: 0.9259 - loss: 361.3298


294/Unknown  69s 234ms/step - accuracy: 0.9260 - loss: 361.0607


295/Unknown  70s 234ms/step - accuracy: 0.9260 - loss: 360.7924


296/Unknown  70s 234ms/step - accuracy: 0.9261 - loss: 360.5264


297/Unknown  70s 234ms/step - accuracy: 0.9261 - loss: 360.2625


298/Unknown  70s 234ms/step - accuracy: 0.9262 - loss: 359.9996


299/Unknown  70s 234ms/step - accuracy: 0.9262 - loss: 359.7378


300/Unknown  71s 234ms/step - accuracy: 0.9263 - loss: 359.4759


301/Unknown  71s 234ms/step - accuracy: 0.9263 - loss: 359.2154


302/Unknown  71s 234ms/step - accuracy: 0.9264 - loss: 358.9561


303/Unknown  71s 234ms/step - accuracy: 0.9264 - loss: 358.6978


304/Unknown  72s 234ms/step - accuracy: 0.9265 - loss: 358.4405


305/Unknown  72s 234ms/step - accuracy: 0.9265 - loss: 358.1843


306/Unknown  72s 234ms/step - accuracy: 0.9266 - loss: 357.9290


307/Unknown  72s 234ms/step - accuracy: 0.9266 - loss: 357.6744


308/Unknown  73s 234ms/step - accuracy: 0.9267 - loss: 357.4214


309/Unknown  73s 234ms/step - accuracy: 0.9267 - loss: 357.1702


310/Unknown  73s 234ms/step - accuracy: 0.9268 - loss: 356.9197


311/Unknown  73s 234ms/step - accuracy: 0.9268 - loss: 356.6696


312/Unknown  74s 234ms/step - accuracy: 0.9269 - loss: 356.4204


313/Unknown  74s 234ms/step - accuracy: 0.9269 - loss: 356.1724


314/Unknown  74s 234ms/step - accuracy: 0.9270 - loss: 355.9252


315/Unknown  74s 234ms/step - accuracy: 0.9270 - loss: 355.6787


316/Unknown  75s 234ms/step - accuracy: 0.9271 - loss: 355.4333


317/Unknown  75s 235ms/step - accuracy: 0.9271 - loss: 355.1894


318/Unknown  75s 235ms/step - accuracy: 0.9272 - loss: 354.9465


319/Unknown  75s 235ms/step - accuracy: 0.9272 - loss: 354.7051


320/Unknown  76s 235ms/step - accuracy: 0.9273 - loss: 354.4644


321/Unknown  76s 235ms/step - accuracy: 0.9273 - loss: 354.2247


322/Unknown  76s 235ms/step - accuracy: 0.9274 - loss: 353.9866


323/Unknown  76s 235ms/step - accuracy: 0.9274 - loss: 353.7493


324/Unknown  77s 235ms/step - accuracy: 0.9275 - loss: 353.5126


325/Unknown  77s 235ms/step - accuracy: 0.9275 - loss: 353.2766


326/Unknown  77s 235ms/step - accuracy: 0.9276 - loss: 353.0421


327/Unknown  77s 235ms/step - accuracy: 0.9276 - loss: 352.8081


328/Unknown  78s 235ms/step - accuracy: 0.9276 - loss: 352.5762


329/Unknown  78s 235ms/step - accuracy: 0.9277 - loss: 352.3457


330/Unknown  78s 235ms/step - accuracy: 0.9277 - loss: 352.1166


331/Unknown  78s 235ms/step - accuracy: 0.9278 - loss: 351.8886


332/Unknown  79s 235ms/step - accuracy: 0.9278 - loss: 351.6612


333/Unknown  79s 235ms/step - accuracy: 0.9279 - loss: 351.4354


334/Unknown  79s 235ms/step - accuracy: 0.9279 - loss: 351.2099


335/Unknown  79s 235ms/step - accuracy: 0.9280 - loss: 350.9852


336/Unknown  80s 235ms/step - accuracy: 0.9280 - loss: 350.7610


337/Unknown  80s 235ms/step - accuracy: 0.9280 - loss: 350.5387


338/Unknown  80s 235ms/step - accuracy: 0.9281 - loss: 350.3170


339/Unknown  80s 235ms/step - accuracy: 0.9281 - loss: 350.0962


340/Unknown  81s 236ms/step - accuracy: 0.9282 - loss: 349.8758


341/Unknown  81s 236ms/step - accuracy: 0.9282 - loss: 349.6569


342/Unknown  81s 236ms/step - accuracy: 0.9283 - loss: 349.4397


343/Unknown  81s 236ms/step - accuracy: 0.9283 - loss: 349.2237


344/Unknown  82s 236ms/step - accuracy: 0.9283 - loss: 349.0085


345/Unknown  82s 236ms/step - accuracy: 0.9284 - loss: 348.7947


346/Unknown  82s 236ms/step - accuracy: 0.9284 - loss: 348.5816


347/Unknown  82s 236ms/step - accuracy: 0.9285 - loss: 348.3696


348/Unknown  83s 236ms/step - accuracy: 0.9285 - loss: 348.1585


349/Unknown  83s 236ms/step - accuracy: 0.9285 - loss: 347.9480


350/Unknown  83s 236ms/step - accuracy: 0.9286 - loss: 347.7393


351/Unknown  83s 236ms/step - accuracy: 0.9286 - loss: 347.5312


352/Unknown  84s 236ms/step - accuracy: 0.9287 - loss: 347.3236


353/Unknown  84s 237ms/step - accuracy: 0.9287 - loss: 347.1163


354/Unknown  84s 237ms/step - accuracy: 0.9287 - loss: 346.9100


355/Unknown  85s 237ms/step - accuracy: 0.9288 - loss: 346.7040


356/Unknown  85s 237ms/step - accuracy: 0.9288 - loss: 346.5000


357/Unknown  85s 237ms/step - accuracy: 0.9289 - loss: 346.2967


358/Unknown  85s 236ms/step - accuracy: 0.9289 - loss: 346.0948


359/Unknown  85s 236ms/step - accuracy: 0.9289 - loss: 345.8932


360/Unknown  86s 236ms/step - accuracy: 0.9290 - loss: 345.6926


361/Unknown  86s 236ms/step - accuracy: 0.9290 - loss: 345.4924


362/Unknown  86s 236ms/step - accuracy: 0.9291 - loss: 345.2931


363/Unknown  86s 236ms/step - accuracy: 0.9291 - loss: 345.0950


364/Unknown  87s 236ms/step - accuracy: 0.9291 - loss: 344.8981


365/Unknown  87s 236ms/step - accuracy: 0.9292 - loss: 344.7014


366/Unknown  87s 236ms/step - accuracy: 0.9292 - loss: 344.5053


367/Unknown  87s 237ms/step - accuracy: 0.9292 - loss: 344.3105


368/Unknown  88s 237ms/step - accuracy: 0.9293 - loss: 344.1166


369/Unknown  88s 237ms/step - accuracy: 0.9293 - loss: 343.9235


370/Unknown  88s 237ms/step - accuracy: 0.9294 - loss: 343.7310


371/Unknown  88s 237ms/step - accuracy: 0.9294 - loss: 343.5392


372/Unknown  89s 237ms/step - accuracy: 0.9294 - loss: 343.3481


373/Unknown  89s 237ms/step - accuracy: 0.9295 - loss: 343.1579


374/Unknown  89s 237ms/step - accuracy: 0.9295 - loss: 342.9688


375/Unknown  89s 237ms/step - accuracy: 0.9295 - loss: 342.7797


376/Unknown  89s 237ms/step - accuracy: 0.9296 - loss: 342.5912


377/Unknown  90s 237ms/step - accuracy: 0.9296 - loss: 342.4032


378/Unknown  90s 237ms/step - accuracy: 0.9297 - loss: 342.2162


379/Unknown  90s 237ms/step - accuracy: 0.9297 - loss: 342.0300


380/Unknown  91s 237ms/step - accuracy: 0.9297 - loss: 341.8448


381/Unknown  91s 237ms/step - accuracy: 0.9298 - loss: 341.6605


382/Unknown  91s 237ms/step - accuracy: 0.9298 - loss: 341.4775


383/Unknown  91s 237ms/step - accuracy: 0.9298 - loss: 341.2947


384/Unknown  92s 237ms/step - accuracy: 0.9299 - loss: 341.1129


385/Unknown  92s 237ms/step - accuracy: 0.9299 - loss: 340.9320


386/Unknown  92s 237ms/step - accuracy: 0.9299 - loss: 340.7514


387/Unknown  92s 237ms/step - accuracy: 0.9300 - loss: 340.5714


388/Unknown  93s 238ms/step - accuracy: 0.9300 - loss: 340.3924


389/Unknown  93s 238ms/step - accuracy: 0.9300 - loss: 340.2137


390/Unknown  93s 237ms/step - accuracy: 0.9301 - loss: 340.0357


391/Unknown  93s 237ms/step - accuracy: 0.9301 - loss: 339.8592


392/Unknown  94s 237ms/step - accuracy: 0.9301 - loss: 339.6831


393/Unknown  94s 237ms/step - accuracy: 0.9302 - loss: 339.5081


394/Unknown  94s 237ms/step - accuracy: 0.9302 - loss: 339.3347


395/Unknown  94s 237ms/step - accuracy: 0.9302 - loss: 339.1619


396/Unknown  94s 237ms/step - accuracy: 0.9303 - loss: 338.9900


397/Unknown  95s 237ms/step - accuracy: 0.9303 - loss: 338.8185


398/Unknown  95s 237ms/step - accuracy: 0.9303 - loss: 338.6484


399/Unknown  95s 237ms/step - accuracy: 0.9304 - loss: 338.4797


400/Unknown  95s 237ms/step - accuracy: 0.9304 - loss: 338.3118


401/Unknown  96s 237ms/step - accuracy: 0.9304 - loss: 338.1445


402/Unknown  96s 238ms/step - accuracy: 0.9305 - loss: 337.9777


403/Unknown  96s 238ms/step - accuracy: 0.9305 - loss: 337.8118


404/Unknown  96s 238ms/step - accuracy: 0.9305 - loss: 337.6463


405/Unknown  97s 238ms/step - accuracy: 0.9306 - loss: 337.4813


406/Unknown  97s 238ms/step - accuracy: 0.9306 - loss: 337.3167


407/Unknown  97s 238ms/step - accuracy: 0.9306 - loss: 337.1526


408/Unknown  98s 238ms/step - accuracy: 0.9307 - loss: 336.9886


409/Unknown  98s 238ms/step - accuracy: 0.9307 - loss: 336.8252


410/Unknown  98s 238ms/step - accuracy: 0.9307 - loss: 336.6620


411/Unknown  98s 238ms/step - accuracy: 0.9308 - loss: 336.4990


412/Unknown  99s 238ms/step - accuracy: 0.9308 - loss: 336.3363


413/Unknown  99s 238ms/step - accuracy: 0.9308 - loss: 336.1741


414/Unknown  99s 238ms/step - accuracy: 0.9309 - loss: 336.0125


415/Unknown  99s 238ms/step - accuracy: 0.9309 - loss: 335.8519


416/Unknown  100s 238ms/step - accuracy: 0.9309 - loss: 335.6918


417/Unknown  100s 238ms/step - accuracy: 0.9309 - loss: 335.5317


418/Unknown  100s 238ms/step - accuracy: 0.9310 - loss: 335.3719


419/Unknown  100s 238ms/step - accuracy: 0.9310 - loss: 335.2126


420/Unknown  101s 238ms/step - accuracy: 0.9310 - loss: 335.0541


421/Unknown  101s 238ms/step - accuracy: 0.9311 - loss: 334.8961


422/Unknown  101s 238ms/step - accuracy: 0.9311 - loss: 334.7386


423/Unknown  101s 238ms/step - accuracy: 0.9311 - loss: 334.5819


424/Unknown  101s 238ms/step - accuracy: 0.9312 - loss: 334.4256


425/Unknown  102s 238ms/step - accuracy: 0.9312 - loss: 334.2702


426/Unknown  102s 238ms/step - accuracy: 0.9312 - loss: 334.1153


427/Unknown  102s 238ms/step - accuracy: 0.9312 - loss: 333.9609


428/Unknown  102s 238ms/step - accuracy: 0.9313 - loss: 333.8071


429/Unknown  103s 238ms/step - accuracy: 0.9313 - loss: 333.6533


430/Unknown  103s 238ms/step - accuracy: 0.9313 - loss: 333.5001


431/Unknown  103s 238ms/step - accuracy: 0.9314 - loss: 333.3471


432/Unknown  103s 238ms/step - accuracy: 0.9314 - loss: 333.1947


433/Unknown  104s 238ms/step - accuracy: 0.9314 - loss: 333.0432


434/Unknown  104s 238ms/step - accuracy: 0.9315 - loss: 332.8920


435/Unknown  104s 238ms/step - accuracy: 0.9315 - loss: 332.7414


436/Unknown  104s 238ms/step - accuracy: 0.9315 - loss: 332.5910


437/Unknown  105s 239ms/step - accuracy: 0.9315 - loss: 332.4410


438/Unknown  105s 239ms/step - accuracy: 0.9316 - loss: 332.2921


439/Unknown  105s 239ms/step - accuracy: 0.9316 - loss: 332.1440


440/Unknown  106s 239ms/step - accuracy: 0.9316 - loss: 331.9966


441/Unknown  106s 239ms/step - accuracy: 0.9317 - loss: 331.8502


442/Unknown  106s 239ms/step - accuracy: 0.9317 - loss: 331.7041


443/Unknown  106s 239ms/step - accuracy: 0.9317 - loss: 331.5587


444/Unknown  107s 239ms/step - accuracy: 0.9317 - loss: 331.4137


445/Unknown  107s 239ms/step - accuracy: 0.9318 - loss: 331.2690


446/Unknown  107s 239ms/step - accuracy: 0.9318 - loss: 331.1252


447/Unknown  107s 239ms/step - accuracy: 0.9318 - loss: 330.9824


448/Unknown  108s 239ms/step - accuracy: 0.9318 - loss: 330.8399


449/Unknown  108s 239ms/step - accuracy: 0.9319 - loss: 330.6978


450/Unknown  108s 239ms/step - accuracy: 0.9319 - loss: 330.5566


451/Unknown  108s 239ms/step - accuracy: 0.9319 - loss: 330.4161


452/Unknown  109s 240ms/step - accuracy: 0.9320 - loss: 330.2763


453/Unknown  109s 240ms/step - accuracy: 0.9320 - loss: 330.1370


454/Unknown  109s 240ms/step - accuracy: 0.9320 - loss: 329.9986


455/Unknown  110s 240ms/step - accuracy: 0.9320 - loss: 329.8606


456/Unknown  110s 240ms/step - accuracy: 0.9321 - loss: 329.7228


457/Unknown  110s 240ms/step - accuracy: 0.9321 - loss: 329.5859


458/Unknown  110s 240ms/step - accuracy: 0.9321 - loss: 329.4496


459/Unknown  110s 240ms/step - accuracy: 0.9321 - loss: 329.3138


460/Unknown  111s 240ms/step - accuracy: 0.9322 - loss: 329.1786


461/Unknown  111s 240ms/step - accuracy: 0.9322 - loss: 329.0436


462/Unknown  111s 240ms/step - accuracy: 0.9322 - loss: 328.9090


463/Unknown  112s 240ms/step - accuracy: 0.9322 - loss: 328.7749


464/Unknown  112s 240ms/step - accuracy: 0.9323 - loss: 328.6411


465/Unknown  112s 240ms/step - accuracy: 0.9323 - loss: 328.5077


466/Unknown  112s 240ms/step - accuracy: 0.9323 - loss: 328.3747


467/Unknown  113s 240ms/step - accuracy: 0.9323 - loss: 328.2422


468/Unknown  113s 240ms/step - accuracy: 0.9324 - loss: 328.1100


469/Unknown  113s 240ms/step - accuracy: 0.9324 - loss: 327.9782


470/Unknown  113s 240ms/step - accuracy: 0.9324 - loss: 327.8466


471/Unknown  114s 240ms/step - accuracy: 0.9324 - loss: 327.7153


472/Unknown  114s 240ms/step - accuracy: 0.9325 - loss: 327.5845


473/Unknown  114s 240ms/step - accuracy: 0.9325 - loss: 327.4537


474/Unknown  114s 240ms/step - accuracy: 0.9325 - loss: 327.3231


475/Unknown  115s 240ms/step - accuracy: 0.9325 - loss: 327.1929


476/Unknown  115s 240ms/step - accuracy: 0.9326 - loss: 327.0634


477/Unknown  115s 240ms/step - accuracy: 0.9326 - loss: 326.9344


478/Unknown  115s 240ms/step - accuracy: 0.9326 - loss: 326.8064


479/Unknown  116s 240ms/step - accuracy: 0.9326 - loss: 326.6785


480/Unknown  116s 240ms/step - accuracy: 0.9327 - loss: 326.5513


481/Unknown  116s 240ms/step - accuracy: 0.9327 - loss: 326.4247


482/Unknown  116s 240ms/step - accuracy: 0.9327 - loss: 326.2985


483/Unknown  117s 241ms/step - accuracy: 0.9327 - loss: 326.1726


484/Unknown  117s 241ms/step - accuracy: 0.9328 - loss: 326.0472


485/Unknown  117s 241ms/step - accuracy: 0.9328 - loss: 325.9221


486/Unknown  117s 241ms/step - accuracy: 0.9328 - loss: 325.7975


487/Unknown  118s 241ms/step - accuracy: 0.9328 - loss: 325.6735


488/Unknown  118s 241ms/step - accuracy: 0.9329 - loss: 325.5500


489/Unknown  118s 241ms/step - accuracy: 0.9329 - loss: 325.4270


490/Unknown  119s 241ms/step - accuracy: 0.9329 - loss: 325.3045


491/Unknown  119s 241ms/step - accuracy: 0.9329 - loss: 325.1823


492/Unknown  119s 241ms/step - accuracy: 0.9330 - loss: 325.0604


493/Unknown  119s 241ms/step - accuracy: 0.9330 - loss: 324.9387


494/Unknown  119s 241ms/step - accuracy: 0.9330 - loss: 324.8173


495/Unknown  120s 241ms/step - accuracy: 0.9330 - loss: 324.6962


496/Unknown  120s 241ms/step - accuracy: 0.9330 - loss: 324.5757


497/Unknown  120s 241ms/step - accuracy: 0.9331 - loss: 324.4557


498/Unknown  121s 241ms/step - accuracy: 0.9331 - loss: 324.3361


499/Unknown  121s 241ms/step - accuracy: 0.9331 - loss: 324.2168


500/Unknown  121s 241ms/step - accuracy: 0.9331 - loss: 324.0978


501/Unknown  121s 241ms/step - accuracy: 0.9332 - loss: 323.9789


502/Unknown  122s 241ms/step - accuracy: 0.9332 - loss: 323.8604


503/Unknown  122s 241ms/step - accuracy: 0.9332 - loss: 323.7420


504/Unknown  122s 241ms/step - accuracy: 0.9332 - loss: 323.6241


505/Unknown  122s 241ms/step - accuracy: 0.9333 - loss: 323.5069


506/Unknown  122s 241ms/step - accuracy: 0.9333 - loss: 323.3900


507/Unknown  123s 241ms/step - accuracy: 0.9333 - loss: 323.2732


508/Unknown  123s 241ms/step - accuracy: 0.9333 - loss: 323.1570


509/Unknown  123s 241ms/step - accuracy: 0.9333 - loss: 323.0411


510/Unknown  124s 241ms/step - accuracy: 0.9334 - loss: 322.9258


511/Unknown  124s 241ms/step - accuracy: 0.9334 - loss: 322.8106


512/Unknown  124s 241ms/step - accuracy: 0.9334 - loss: 322.6954


513/Unknown  124s 241ms/step - accuracy: 0.9334 - loss: 322.5804


514/Unknown  125s 241ms/step - accuracy: 0.9334 - loss: 322.4655


515/Unknown  125s 241ms/step - accuracy: 0.9335 - loss: 322.3508


516/Unknown  125s 242ms/step - accuracy: 0.9335 - loss: 322.2363


517/Unknown  125s 242ms/step - accuracy: 0.9335 - loss: 322.1223


518/Unknown  126s 242ms/step - accuracy: 0.9335 - loss: 322.0087


519/Unknown  126s 242ms/step - accuracy: 0.9336 - loss: 321.8951


520/Unknown  126s 242ms/step - accuracy: 0.9336 - loss: 321.7819


521/Unknown  126s 242ms/step - accuracy: 0.9336 - loss: 321.6687


522/Unknown  127s 242ms/step - accuracy: 0.9336 - loss: 321.5559


523/Unknown  127s 242ms/step - accuracy: 0.9336 - loss: 321.4433


524/Unknown  127s 242ms/step - accuracy: 0.9337 - loss: 321.3311


525/Unknown  127s 241ms/step - accuracy: 0.9337 - loss: 321.2190


526/Unknown  128s 241ms/step - accuracy: 0.9337 - loss: 321.1076


527/Unknown  128s 241ms/step - accuracy: 0.9337 - loss: 320.9963


528/Unknown  128s 241ms/step - accuracy: 0.9337 - loss: 320.8853


529/Unknown  128s 241ms/step - accuracy: 0.9338 - loss: 320.7745


530/Unknown  128s 241ms/step - accuracy: 0.9338 - loss: 320.6637


531/Unknown  129s 241ms/step - accuracy: 0.9338 - loss: 320.5531


532/Unknown  129s 241ms/step - accuracy: 0.9338 - loss: 320.4430


533/Unknown  129s 241ms/step - accuracy: 0.9339 - loss: 320.3329


534/Unknown  129s 241ms/step - accuracy: 0.9339 - loss: 320.2232


535/Unknown  130s 241ms/step - accuracy: 0.9339 - loss: 320.1136


536/Unknown  130s 241ms/step - accuracy: 0.9339 - loss: 320.0045


537/Unknown  130s 241ms/step - accuracy: 0.9339 - loss: 319.8954


538/Unknown  130s 242ms/step - accuracy: 0.9340 - loss: 319.7864


539/Unknown  131s 242ms/step - accuracy: 0.9340 - loss: 319.6782


540/Unknown  131s 242ms/step - accuracy: 0.9340 - loss: 319.5701


541/Unknown  131s 242ms/step - accuracy: 0.9340 - loss: 319.4627


542/Unknown  132s 242ms/step - accuracy: 0.9340 - loss: 319.3557


543/Unknown  132s 242ms/step - accuracy: 0.9341 - loss: 319.2490


544/Unknown  132s 242ms/step - accuracy: 0.9341 - loss: 319.1425


545/Unknown  132s 242ms/step - accuracy: 0.9341 - loss: 319.0364


546/Unknown  133s 242ms/step - accuracy: 0.9341 - loss: 318.9309


547/Unknown  133s 242ms/step - accuracy: 0.9341 - loss: 318.8256


548/Unknown  133s 242ms/step - accuracy: 0.9342 - loss: 318.7207


549/Unknown  133s 242ms/step - accuracy: 0.9342 - loss: 318.6164


550/Unknown  134s 242ms/step - accuracy: 0.9342 - loss: 318.5123


551/Unknown  134s 242ms/step - accuracy: 0.9342 - loss: 318.4086


552/Unknown  134s 242ms/step - accuracy: 0.9342 - loss: 318.3053


553/Unknown  134s 242ms/step - accuracy: 0.9343 - loss: 318.2019


554/Unknown  135s 242ms/step - accuracy: 0.9343 - loss: 318.0987


555/Unknown  135s 242ms/step - accuracy: 0.9343 - loss: 317.9957


556/Unknown  135s 242ms/step - accuracy: 0.9343 - loss: 317.8931


557/Unknown  136s 242ms/step - accuracy: 0.9343 - loss: 317.7906


558/Unknown  136s 242ms/step - accuracy: 0.9344 - loss: 317.6887


559/Unknown  136s 242ms/step - accuracy: 0.9344 - loss: 317.5871


560/Unknown  136s 242ms/step - accuracy: 0.9344 - loss: 317.4857


561/Unknown  137s 242ms/step - accuracy: 0.9344 - loss: 317.3849


562/Unknown  137s 243ms/step - accuracy: 0.9344 - loss: 317.2843


563/Unknown  137s 243ms/step - accuracy: 0.9344 - loss: 317.1841


564/Unknown  137s 243ms/step - accuracy: 0.9345 - loss: 317.0841


565/Unknown  138s 243ms/step - accuracy: 0.9345 - loss: 316.9843


566/Unknown  138s 243ms/step - accuracy: 0.9345 - loss: 316.8847


567/Unknown  138s 243ms/step - accuracy: 0.9345 - loss: 316.7856


568/Unknown  138s 243ms/step - accuracy: 0.9345 - loss: 316.6870


569/Unknown  139s 243ms/step - accuracy: 0.9346 - loss: 316.5887


570/Unknown  139s 243ms/step - accuracy: 0.9346 - loss: 316.4907


571/Unknown  139s 243ms/step - accuracy: 0.9346 - loss: 316.3928


572/Unknown  139s 243ms/step - accuracy: 0.9346 - loss: 316.2950


573/Unknown  140s 243ms/step - accuracy: 0.9346 - loss: 316.1977


574/Unknown  140s 243ms/step - accuracy: 0.9347 - loss: 316.1008


575/Unknown  140s 243ms/step - accuracy: 0.9347 - loss: 316.0045


576/Unknown  140s 243ms/step - accuracy: 0.9347 - loss: 315.9082


577/Unknown  141s 243ms/step - accuracy: 0.9347 - loss: 315.8120


578/Unknown  141s 243ms/step - accuracy: 0.9347 - loss: 315.7160


579/Unknown  141s 243ms/step - accuracy: 0.9347 - loss: 315.6203


580/Unknown  142s 243ms/step - accuracy: 0.9348 - loss: 315.5247


581/Unknown  142s 243ms/step - accuracy: 0.9348 - loss: 315.4293


582/Unknown  142s 243ms/step - accuracy: 0.9348 - loss: 315.3342


583/Unknown  142s 243ms/step - accuracy: 0.9348 - loss: 315.2393


584/Unknown  143s 243ms/step - accuracy: 0.9348 - loss: 315.1447


585/Unknown  143s 243ms/step - accuracy: 0.9348 - loss: 315.0504


586/Unknown  143s 243ms/step - accuracy: 0.9349 - loss: 314.9563


587/Unknown  143s 243ms/step - accuracy: 0.9349 - loss: 314.8626


588/Unknown  144s 243ms/step - accuracy: 0.9349 - loss: 314.7693


589/Unknown  144s 243ms/step - accuracy: 0.9349 - loss: 314.6760


590/Unknown  144s 243ms/step - accuracy: 0.9349 - loss: 314.5832


591/Unknown  144s 243ms/step - accuracy: 0.9349 - loss: 314.4905


592/Unknown  145s 244ms/step - accuracy: 0.9350 - loss: 314.3980


593/Unknown  145s 244ms/step - accuracy: 0.9350 - loss: 314.3056


594/Unknown  145s 244ms/step - accuracy: 0.9350 - loss: 314.2133


595/Unknown  145s 244ms/step - accuracy: 0.9350 - loss: 314.1214


596/Unknown  146s 244ms/step - accuracy: 0.9350 - loss: 314.0297


597/Unknown  146s 244ms/step - accuracy: 0.9351 - loss: 313.9380


598/Unknown  146s 244ms/step - accuracy: 0.9351 - loss: 313.8465


599/Unknown  147s 244ms/step - accuracy: 0.9351 - loss: 313.7552


600/Unknown  147s 244ms/step - accuracy: 0.9351 - loss: 313.6645


601/Unknown  147s 244ms/step - accuracy: 0.9351 - loss: 313.5738


602/Unknown  147s 244ms/step - accuracy: 0.9351 - loss: 313.4837


603/Unknown  148s 244ms/step - accuracy: 0.9352 - loss: 313.3939


604/Unknown  148s 244ms/step - accuracy: 0.9352 - loss: 313.3044


605/Unknown  148s 244ms/step - accuracy: 0.9352 - loss: 313.2152


606/Unknown  148s 244ms/step - accuracy: 0.9352 - loss: 313.1262


607/Unknown  149s 244ms/step - accuracy: 0.9352 - loss: 313.0374


608/Unknown  149s 244ms/step - accuracy: 0.9352 - loss: 312.9488


609/Unknown  149s 244ms/step - accuracy: 0.9352 - loss: 312.8601


610/Unknown  149s 244ms/step - accuracy: 0.9353 - loss: 312.7717


611/Unknown  150s 244ms/step - accuracy: 0.9353 - loss: 312.6833


612/Unknown  150s 244ms/step - accuracy: 0.9353 - loss: 312.5952


613/Unknown  150s 244ms/step - accuracy: 0.9353 - loss: 312.5075


614/Unknown  150s 244ms/step - accuracy: 0.9353 - loss: 312.4199


615/Unknown  151s 244ms/step - accuracy: 0.9353 - loss: 312.3324


616/Unknown  151s 244ms/step - accuracy: 0.9354 - loss: 312.2452


617/Unknown  151s 245ms/step - accuracy: 0.9354 - loss: 312.1583


618/Unknown  152s 245ms/step - accuracy: 0.9354 - loss: 312.0714


619/Unknown  152s 245ms/step - accuracy: 0.9354 - loss: 311.9846


620/Unknown  152s 245ms/step - accuracy: 0.9354 - loss: 311.8981


621/Unknown  152s 245ms/step - accuracy: 0.9354 - loss: 311.8117


622/Unknown  153s 245ms/step - accuracy: 0.9355 - loss: 311.7256


623/Unknown  153s 245ms/step - accuracy: 0.9355 - loss: 311.6397


624/Unknown  153s 245ms/step - accuracy: 0.9355 - loss: 311.5540


625/Unknown  153s 245ms/step - accuracy: 0.9355 - loss: 311.4685


626/Unknown  154s 245ms/step - accuracy: 0.9355 - loss: 311.3833


627/Unknown  154s 245ms/step - accuracy: 0.9355 - loss: 311.2987


628/Unknown  154s 244ms/step - accuracy: 0.9356 - loss: 311.2146


629/Unknown  154s 244ms/step - accuracy: 0.9356 - loss: 311.1306


630/Unknown  155s 244ms/step - accuracy: 0.9356 - loss: 311.0466


631/Unknown  155s 244ms/step - accuracy: 0.9356 - loss: 310.9627


632/Unknown  155s 244ms/step - accuracy: 0.9356 - loss: 310.8792


633/Unknown  155s 244ms/step - accuracy: 0.9356 - loss: 310.7963


634/Unknown  155s 244ms/step - accuracy: 0.9356 - loss: 310.7135


635/Unknown  156s 244ms/step - accuracy: 0.9357 - loss: 310.6309


636/Unknown  156s 244ms/step - accuracy: 0.9357 - loss: 310.5486


637/Unknown  156s 244ms/step - accuracy: 0.9357 - loss: 310.4668


638/Unknown  157s 245ms/step - accuracy: 0.9357 - loss: 310.3852


639/Unknown  157s 245ms/step - accuracy: 0.9357 - loss: 310.3038


640/Unknown  157s 245ms/step - accuracy: 0.9357 - loss: 310.2225

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()


```
</div>
 640/640 ━━━━━━━━━━━━━━━━━━━━ 174s 271ms/step - accuracy: 0.9358 - loss: 310.1415 - val_accuracy: 0.9498 - val_loss: 227.1810


<div class="k-default-codeblock">
```
Model training finished.
Evaluating model performance...

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  0s 341ms/step - accuracy: 0.9509 - loss: 162.4458


  2/Unknown  0s 144ms/step - accuracy: 0.9443 - loss: 181.0830


  3/Unknown  1s 142ms/step - accuracy: 0.9419 - loss: 196.8291


  4/Unknown  1s 146ms/step - accuracy: 0.9404 - loss: 206.0489


  5/Unknown  1s 142ms/step - accuracy: 0.9407 - loss: 211.7518


  6/Unknown  1s 145ms/step - accuracy: 0.9415 - loss: 215.5974


  7/Unknown  1s 146ms/step - accuracy: 0.9426 - loss: 216.1428


  8/Unknown  1s 145ms/step - accuracy: 0.9435 - loss: 215.5485


  9/Unknown  2s 145ms/step - accuracy: 0.9441 - loss: 214.7674


 10/Unknown  2s 144ms/step - accuracy: 0.9446 - loss: 213.8222


 11/Unknown  2s 144ms/step - accuracy: 0.9452 - loss: 213.1922


 12/Unknown  2s 144ms/step - accuracy: 0.9455 - loss: 212.5974


 13/Unknown  2s 144ms/step - accuracy: 0.9457 - loss: 212.4554


 14/Unknown  2s 144ms/step - accuracy: 0.9459 - loss: 212.1609


 15/Unknown  2s 145ms/step - accuracy: 0.9461 - loss: 212.0100


 16/Unknown  3s 145ms/step - accuracy: 0.9464 - loss: 211.6899


 17/Unknown  3s 145ms/step - accuracy: 0.9466 - loss: 211.4328


 18/Unknown  3s 145ms/step - accuracy: 0.9468 - loss: 211.5108


 19/Unknown  3s 144ms/step - accuracy: 0.9469 - loss: 211.7577


 20/Unknown  3s 144ms/step - accuracy: 0.9470 - loss: 212.0558


 21/Unknown  3s 144ms/step - accuracy: 0.9470 - loss: 212.4518


 22/Unknown  3s 144ms/step - accuracy: 0.9471 - loss: 212.7789


 23/Unknown  4s 144ms/step - accuracy: 0.9471 - loss: 213.0560


 24/Unknown  4s 144ms/step - accuracy: 0.9471 - loss: 213.2825


 25/Unknown  4s 144ms/step - accuracy: 0.9471 - loss: 213.6518


 26/Unknown  4s 144ms/step - accuracy: 0.9471 - loss: 213.9233


 27/Unknown  4s 144ms/step - accuracy: 0.9472 - loss: 214.1503


 28/Unknown  4s 144ms/step - accuracy: 0.9472 - loss: 214.3847


 29/Unknown  4s 144ms/step - accuracy: 0.9473 - loss: 214.5778


 30/Unknown  5s 144ms/step - accuracy: 0.9473 - loss: 214.7275


 31/Unknown  5s 145ms/step - accuracy: 0.9473 - loss: 214.8989


 32/Unknown  5s 146ms/step - accuracy: 0.9474 - loss: 215.0651


 33/Unknown  5s 147ms/step - accuracy: 0.9474 - loss: 215.3561


 34/Unknown  5s 148ms/step - accuracy: 0.9474 - loss: 215.6748


 35/Unknown  5s 149ms/step - accuracy: 0.9473 - loss: 215.9542


 36/Unknown  6s 149ms/step - accuracy: 0.9473 - loss: 216.1981


 37/Unknown  6s 149ms/step - accuracy: 0.9473 - loss: 216.4192


 38/Unknown  6s 149ms/step - accuracy: 0.9473 - loss: 216.6240


 39/Unknown  6s 150ms/step - accuracy: 0.9473 - loss: 216.7979


 40/Unknown  6s 150ms/step - accuracy: 0.9474 - loss: 216.9769


 41/Unknown  6s 150ms/step - accuracy: 0.9474 - loss: 217.1346


 42/Unknown  7s 150ms/step - accuracy: 0.9474 - loss: 217.2511


 43/Unknown  7s 150ms/step - accuracy: 0.9474 - loss: 217.4261


 44/Unknown  7s 150ms/step - accuracy: 0.9474 - loss: 217.5872


 45/Unknown  7s 150ms/step - accuracy: 0.9474 - loss: 217.7351


 46/Unknown  7s 150ms/step - accuracy: 0.9474 - loss: 217.8584


 47/Unknown  7s 150ms/step - accuracy: 0.9474 - loss: 217.9560


 48/Unknown  7s 149ms/step - accuracy: 0.9475 - loss: 218.0626


 49/Unknown  7s 149ms/step - accuracy: 0.9475 - loss: 218.1622


 50/Unknown  8s 148ms/step - accuracy: 0.9475 - loss: 218.3029


 51/Unknown  8s 148ms/step - accuracy: 0.9475 - loss: 218.4238


 52/Unknown  8s 148ms/step - accuracy: 0.9475 - loss: 218.5109


 53/Unknown  8s 147ms/step - accuracy: 0.9476 - loss: 218.5741


 54/Unknown  8s 147ms/step - accuracy: 0.9476 - loss: 218.6540


 55/Unknown  8s 147ms/step - accuracy: 0.9476 - loss: 218.7485


 56/Unknown  8s 146ms/step - accuracy: 0.9476 - loss: 218.8147


 57/Unknown  9s 146ms/step - accuracy: 0.9477 - loss: 218.8586


 58/Unknown  9s 146ms/step - accuracy: 0.9477 - loss: 218.9425


 59/Unknown  9s 146ms/step - accuracy: 0.9477 - loss: 219.0194


 60/Unknown  9s 146ms/step - accuracy: 0.9477 - loss: 219.0956


 61/Unknown  9s 146ms/step - accuracy: 0.9477 - loss: 219.1801


 62/Unknown  9s 146ms/step - accuracy: 0.9478 - loss: 219.2492


 63/Unknown  9s 146ms/step - accuracy: 0.9478 - loss: 219.3433


 64/Unknown  10s 146ms/step - accuracy: 0.9478 - loss: 219.4519


 65/Unknown  10s 146ms/step - accuracy: 0.9478 - loss: 219.5580


 66/Unknown  10s 146ms/step - accuracy: 0.9478 - loss: 219.6452


 67/Unknown  10s 145ms/step - accuracy: 0.9478 - loss: 219.7370


 68/Unknown  10s 145ms/step - accuracy: 0.9478 - loss: 219.8202


 69/Unknown  10s 145ms/step - accuracy: 0.9478 - loss: 219.9068


 70/Unknown  10s 145ms/step - accuracy: 0.9479 - loss: 219.9808


 71/Unknown  11s 145ms/step - accuracy: 0.9479 - loss: 220.0469


 72/Unknown  11s 145ms/step - accuracy: 0.9479 - loss: 220.1314


 73/Unknown  11s 145ms/step - accuracy: 0.9479 - loss: 220.2233


 74/Unknown  11s 145ms/step - accuracy: 0.9479 - loss: 220.3074


 75/Unknown  11s 145ms/step - accuracy: 0.9479 - loss: 220.3792


 76/Unknown  11s 145ms/step - accuracy: 0.9479 - loss: 220.4452


 77/Unknown  11s 145ms/step - accuracy: 0.9480 - loss: 220.5054


 78/Unknown  12s 145ms/step - accuracy: 0.9480 - loss: 220.5729


 79/Unknown  12s 145ms/step - accuracy: 0.9480 - loss: 220.6319


 80/Unknown  12s 146ms/step - accuracy: 0.9480 - loss: 220.6857


 81/Unknown  12s 146ms/step - accuracy: 0.9480 - loss: 220.7341


 82/Unknown  12s 147ms/step - accuracy: 0.9480 - loss: 220.7907


 83/Unknown  12s 147ms/step - accuracy: 0.9480 - loss: 220.8492


 84/Unknown  13s 147ms/step - accuracy: 0.9480 - loss: 220.9068


 85/Unknown  13s 147ms/step - accuracy: 0.9480 - loss: 220.9686


 86/Unknown  13s 147ms/step - accuracy: 0.9480 - loss: 221.0368


 87/Unknown  13s 147ms/step - accuracy: 0.9480 - loss: 221.1018


 88/Unknown  13s 147ms/step - accuracy: 0.9480 - loss: 221.1624


 89/Unknown  13s 148ms/step - accuracy: 0.9480 - loss: 221.2261


 90/Unknown  13s 148ms/step - accuracy: 0.9481 - loss: 221.2927


 91/Unknown  14s 148ms/step - accuracy: 0.9481 - loss: 221.3554


 92/Unknown  14s 148ms/step - accuracy: 0.9481 - loss: 221.4201


 93/Unknown  14s 148ms/step - accuracy: 0.9481 - loss: 221.4830


 94/Unknown  14s 148ms/step - accuracy: 0.9481 - loss: 221.5580


 95/Unknown  14s 148ms/step - accuracy: 0.9481 - loss: 221.6310


 96/Unknown  14s 148ms/step - accuracy: 0.9481 - loss: 221.7074


 97/Unknown  15s 147ms/step - accuracy: 0.9481 - loss: 221.7751


 98/Unknown  15s 147ms/step - accuracy: 0.9481 - loss: 221.8373


 99/Unknown  15s 148ms/step - accuracy: 0.9481 - loss: 221.9001


100/Unknown  15s 148ms/step - accuracy: 0.9481 - loss: 221.9738


101/Unknown  15s 148ms/step - accuracy: 0.9481 - loss: 222.0465


102/Unknown  15s 148ms/step - accuracy: 0.9480 - loss: 222.1164


103/Unknown  15s 148ms/step - accuracy: 0.9480 - loss: 222.1863


104/Unknown  16s 148ms/step - accuracy: 0.9480 - loss: 222.2534


105/Unknown  16s 148ms/step - accuracy: 0.9480 - loss: 222.3202


106/Unknown  16s 148ms/step - accuracy: 0.9480 - loss: 222.3874


107/Unknown  16s 148ms/step - accuracy: 0.9480 - loss: 222.4603


108/Unknown  16s 148ms/step - accuracy: 0.9480 - loss: 222.5335


109/Unknown  16s 148ms/step - accuracy: 0.9480 - loss: 222.6129


110/Unknown  17s 149ms/step - accuracy: 0.9480 - loss: 222.6919


111/Unknown  17s 148ms/step - accuracy: 0.9480 - loss: 222.7742


112/Unknown  17s 148ms/step - accuracy: 0.9480 - loss: 222.8567


113/Unknown  17s 149ms/step - accuracy: 0.9480 - loss: 222.9396


114/Unknown  17s 148ms/step - accuracy: 0.9480 - loss: 223.0149


115/Unknown  17s 148ms/step - accuracy: 0.9480 - loss: 223.0847


116/Unknown  17s 148ms/step - accuracy: 0.9480 - loss: 223.1570


117/Unknown  18s 148ms/step - accuracy: 0.9480 - loss: 223.2323


118/Unknown  18s 148ms/step - accuracy: 0.9480 - loss: 223.3047


119/Unknown  18s 148ms/step - accuracy: 0.9480 - loss: 223.3757


120/Unknown  18s 148ms/step - accuracy: 0.9480 - loss: 223.4460


121/Unknown  18s 148ms/step - accuracy: 0.9480 - loss: 223.5164


122/Unknown  18s 147ms/step - accuracy: 0.9480 - loss: 223.5857


123/Unknown  18s 147ms/step - accuracy: 0.9480 - loss: 223.6464


124/Unknown  18s 147ms/step - accuracy: 0.9480 - loss: 223.7074


125/Unknown  19s 147ms/step - accuracy: 0.9480 - loss: 223.7708


126/Unknown  19s 147ms/step - accuracy: 0.9480 - loss: 223.8305


127/Unknown  19s 147ms/step - accuracy: 0.9480 - loss: 223.8893


128/Unknown  19s 147ms/step - accuracy: 0.9480 - loss: 223.9453


129/Unknown  19s 147ms/step - accuracy: 0.9480 - loss: 223.9971


130/Unknown  19s 147ms/step - accuracy: 0.9480 - loss: 224.0474


131/Unknown  19s 147ms/step - accuracy: 0.9480 - loss: 224.0945


132/Unknown  20s 147ms/step - accuracy: 0.9480 - loss: 224.1432


133/Unknown  20s 147ms/step - accuracy: 0.9480 - loss: 224.1884


134/Unknown  20s 147ms/step - accuracy: 0.9480 - loss: 224.2346


135/Unknown  20s 147ms/step - accuracy: 0.9480 - loss: 224.2775


136/Unknown  20s 147ms/step - accuracy: 0.9480 - loss: 224.3277


137/Unknown  20s 147ms/step - accuracy: 0.9480 - loss: 224.3793


138/Unknown  20s 147ms/step - accuracy: 0.9480 - loss: 224.4278


139/Unknown  21s 147ms/step - accuracy: 0.9481 - loss: 224.4756


140/Unknown  21s 147ms/step - accuracy: 0.9481 - loss: 224.5227


141/Unknown  21s 147ms/step - accuracy: 0.9481 - loss: 224.5697


142/Unknown  21s 147ms/step - accuracy: 0.9481 - loss: 224.6159


143/Unknown  21s 147ms/step - accuracy: 0.9481 - loss: 224.6616


144/Unknown  21s 146ms/step - accuracy: 0.9481 - loss: 224.7067


145/Unknown  21s 146ms/step - accuracy: 0.9481 - loss: 224.7498


146/Unknown  22s 146ms/step - accuracy: 0.9481 - loss: 224.7929


147/Unknown  22s 146ms/step - accuracy: 0.9481 - loss: 224.8350


148/Unknown  22s 147ms/step - accuracy: 0.9481 - loss: 224.8750


149/Unknown  22s 147ms/step - accuracy: 0.9481 - loss: 224.9186


150/Unknown  22s 147ms/step - accuracy: 0.9481 - loss: 224.9600


151/Unknown  22s 147ms/step - accuracy: 0.9481 - loss: 225.0039


152/Unknown  23s 147ms/step - accuracy: 0.9481 - loss: 225.0496


153/Unknown  23s 147ms/step - accuracy: 0.9481 - loss: 225.0955


154/Unknown  23s 148ms/step - accuracy: 0.9481 - loss: 225.1402


155/Unknown  23s 148ms/step - accuracy: 0.9481 - loss: 225.1870


156/Unknown  23s 148ms/step - accuracy: 0.9481 - loss: 225.2341


157/Unknown  23s 148ms/step - accuracy: 0.9481 - loss: 225.2801


158/Unknown  24s 148ms/step - accuracy: 0.9481 - loss: 225.3243


159/Unknown  24s 148ms/step - accuracy: 0.9481 - loss: 225.3679


160/Unknown  24s 148ms/step - accuracy: 0.9481 - loss: 225.4094


161/Unknown  24s 148ms/step - accuracy: 0.9481 - loss: 225.4495


162/Unknown  24s 148ms/step - accuracy: 0.9481 - loss: 225.4878


163/Unknown  24s 148ms/step - accuracy: 0.9481 - loss: 225.5246


164/Unknown  25s 148ms/step - accuracy: 0.9481 - loss: 225.5602


165/Unknown  25s 149ms/step - accuracy: 0.9481 - loss: 225.5937


166/Unknown  25s 149ms/step - accuracy: 0.9481 - loss: 225.6266


167/Unknown  25s 149ms/step - accuracy: 0.9481 - loss: 225.6585


168/Unknown  25s 148ms/step - accuracy: 0.9481 - loss: 225.6897


169/Unknown  25s 148ms/step - accuracy: 0.9481 - loss: 225.7194


170/Unknown  25s 148ms/step - accuracy: 0.9481 - loss: 225.7488


171/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.7782


172/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.8078


173/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.8382


174/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.8664


175/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.8942


176/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.9219


177/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.9492


178/Unknown  26s 148ms/step - accuracy: 0.9481 - loss: 225.9751


179/Unknown  27s 148ms/step - accuracy: 0.9482 - loss: 225.9979


180/Unknown  27s 148ms/step - accuracy: 0.9482 - loss: 226.0186


181/Unknown  27s 147ms/step - accuracy: 0.9482 - loss: 226.0383


182/Unknown  27s 147ms/step - accuracy: 0.9482 - loss: 226.0596


183/Unknown  27s 147ms/step - accuracy: 0.9482 - loss: 226.0812


184/Unknown  27s 147ms/step - accuracy: 0.9482 - loss: 226.1047


185/Unknown  27s 147ms/step - accuracy: 0.9482 - loss: 226.1278


186/Unknown  28s 147ms/step - accuracy: 0.9482 - loss: 226.1494


187/Unknown  28s 147ms/step - accuracy: 0.9482 - loss: 226.1703


188/Unknown  28s 147ms/step - accuracy: 0.9482 - loss: 226.1906


189/Unknown  28s 147ms/step - accuracy: 0.9482 - loss: 226.2107


190/Unknown  28s 147ms/step - accuracy: 0.9482 - loss: 226.2321


191/Unknown  28s 147ms/step - accuracy: 0.9482 - loss: 226.2524


192/Unknown  28s 147ms/step - accuracy: 0.9482 - loss: 226.2710


193/Unknown  29s 147ms/step - accuracy: 0.9482 - loss: 226.2888


194/Unknown  29s 147ms/step - accuracy: 0.9482 - loss: 226.3056


195/Unknown  29s 147ms/step - accuracy: 0.9482 - loss: 226.3227


196/Unknown  29s 147ms/step - accuracy: 0.9482 - loss: 226.3410


197/Unknown  29s 147ms/step - accuracy: 0.9482 - loss: 226.3565


198/Unknown  29s 147ms/step - accuracy: 0.9482 - loss: 226.3754


199/Unknown  29s 147ms/step - accuracy: 0.9482 - loss: 226.3936


200/Unknown  30s 147ms/step - accuracy: 0.9482 - loss: 226.4121


201/Unknown  30s 147ms/step - accuracy: 0.9482 - loss: 226.4291


202/Unknown  30s 147ms/step - accuracy: 0.9483 - loss: 226.4454


203/Unknown  30s 147ms/step - accuracy: 0.9483 - loss: 226.4612


204/Unknown  30s 147ms/step - accuracy: 0.9483 - loss: 226.4754


205/Unknown  30s 147ms/step - accuracy: 0.9483 - loss: 226.4914


206/Unknown  31s 148ms/step - accuracy: 0.9483 - loss: 226.5079


207/Unknown  31s 148ms/step - accuracy: 0.9483 - loss: 226.5235


208/Unknown  31s 148ms/step - accuracy: 0.9483 - loss: 226.5387


209/Unknown  31s 148ms/step - accuracy: 0.9483 - loss: 226.5556


210/Unknown  31s 148ms/step - accuracy: 0.9483 - loss: 226.5714


211/Unknown  31s 148ms/step - accuracy: 0.9483 - loss: 226.5859


212/Unknown  32s 148ms/step - accuracy: 0.9483 - loss: 226.5997


213/Unknown  32s 148ms/step - accuracy: 0.9483 - loss: 226.6133


214/Unknown  32s 148ms/step - accuracy: 0.9483 - loss: 226.6275


215/Unknown  32s 148ms/step - accuracy: 0.9483 - loss: 226.6414


216/Unknown  32s 148ms/step - accuracy: 0.9483 - loss: 226.6565


217/Unknown  32s 148ms/step - accuracy: 0.9483 - loss: 226.6711


218/Unknown  32s 148ms/step - accuracy: 0.9483 - loss: 226.6844


219/Unknown  33s 148ms/step - accuracy: 0.9483 - loss: 226.6964


220/Unknown  33s 148ms/step - accuracy: 0.9483 - loss: 226.7072


221/Unknown  33s 148ms/step - accuracy: 0.9483 - loss: 226.7171


222/Unknown  33s 148ms/step - accuracy: 0.9483 - loss: 226.7257


223/Unknown  33s 148ms/step - accuracy: 0.9484 - loss: 226.7337


224/Unknown  33s 148ms/step - accuracy: 0.9484 - loss: 226.7415


225/Unknown  33s 148ms/step - accuracy: 0.9484 - loss: 226.7505


226/Unknown  34s 148ms/step - accuracy: 0.9484 - loss: 226.7594


227/Unknown  34s 148ms/step - accuracy: 0.9484 - loss: 226.7673


228/Unknown  34s 148ms/step - accuracy: 0.9484 - loss: 226.7739


229/Unknown  34s 148ms/step - accuracy: 0.9484 - loss: 226.7809


230/Unknown  34s 148ms/step - accuracy: 0.9484 - loss: 226.7871


231/Unknown  34s 148ms/step - accuracy: 0.9484 - loss: 226.7928


232/Unknown  35s 148ms/step - accuracy: 0.9484 - loss: 226.7966


233/Unknown  35s 148ms/step - accuracy: 0.9484 - loss: 226.8005


234/Unknown  35s 148ms/step - accuracy: 0.9484 - loss: 226.8036


235/Unknown  35s 148ms/step - accuracy: 0.9484 - loss: 226.8075


236/Unknown  35s 148ms/step - accuracy: 0.9484 - loss: 226.8128


237/Unknown  35s 148ms/step - accuracy: 0.9484 - loss: 226.8175


238/Unknown  36s 148ms/step - accuracy: 0.9484 - loss: 226.8228


239/Unknown  36s 148ms/step - accuracy: 0.9484 - loss: 226.8287


240/Unknown  36s 148ms/step - accuracy: 0.9485 - loss: 226.8334


241/Unknown  36s 148ms/step - accuracy: 0.9485 - loss: 226.8374


242/Unknown  36s 148ms/step - accuracy: 0.9485 - loss: 226.8412


243/Unknown  36s 148ms/step - accuracy: 0.9485 - loss: 226.8454


244/Unknown  36s 148ms/step - accuracy: 0.9485 - loss: 226.8489


245/Unknown  37s 148ms/step - accuracy: 0.9485 - loss: 226.8528


246/Unknown  37s 148ms/step - accuracy: 0.9485 - loss: 226.8559


247/Unknown  37s 148ms/step - accuracy: 0.9485 - loss: 226.8593


248/Unknown  37s 148ms/step - accuracy: 0.9485 - loss: 226.8624


249/Unknown  37s 148ms/step - accuracy: 0.9485 - loss: 226.8650


250/Unknown  37s 148ms/step - accuracy: 0.9485 - loss: 226.8684


251/Unknown  37s 148ms/step - accuracy: 0.9485 - loss: 226.8727


252/Unknown  38s 148ms/step - accuracy: 0.9485 - loss: 226.8768


253/Unknown  38s 149ms/step - accuracy: 0.9485 - loss: 226.8811


254/Unknown  38s 149ms/step - accuracy: 0.9485 - loss: 226.8850


255/Unknown  38s 149ms/step - accuracy: 0.9485 - loss: 226.8888


256/Unknown  38s 149ms/step - accuracy: 0.9485 - loss: 226.8931


257/Unknown  38s 149ms/step - accuracy: 0.9485 - loss: 226.8969


258/Unknown  39s 149ms/step - accuracy: 0.9485 - loss: 226.9015


259/Unknown  39s 149ms/step - accuracy: 0.9485 - loss: 226.9061


260/Unknown  39s 149ms/step - accuracy: 0.9485 - loss: 226.9099


261/Unknown  39s 149ms/step - accuracy: 0.9485 - loss: 226.9135


262/Unknown  39s 149ms/step - accuracy: 0.9486 - loss: 226.9156


263/Unknown  39s 149ms/step - accuracy: 0.9486 - loss: 226.9174


264/Unknown  39s 149ms/step - accuracy: 0.9486 - loss: 226.9189


265/Unknown  40s 149ms/step - accuracy: 0.9486 - loss: 226.9205


266/Unknown  40s 149ms/step - accuracy: 0.9486 - loss: 226.9237


267/Unknown  40s 149ms/step - accuracy: 0.9486 - loss: 226.9278


268/Unknown  40s 149ms/step - accuracy: 0.9486 - loss: 226.9312


269/Unknown  40s 149ms/step - accuracy: 0.9486 - loss: 226.9352


270/Unknown  40s 149ms/step - accuracy: 0.9486 - loss: 226.9391


271/Unknown  41s 149ms/step - accuracy: 0.9486 - loss: 226.9431


272/Unknown  41s 149ms/step - accuracy: 0.9486 - loss: 226.9461


273/Unknown  41s 149ms/step - accuracy: 0.9486 - loss: 226.9491


274/Unknown  41s 149ms/step - accuracy: 0.9486 - loss: 226.9513


275/Unknown  41s 149ms/step - accuracy: 0.9486 - loss: 226.9532


276/Unknown  41s 150ms/step - accuracy: 0.9486 - loss: 226.9559


277/Unknown  42s 150ms/step - accuracy: 0.9486 - loss: 226.9587


278/Unknown  42s 150ms/step - accuracy: 0.9486 - loss: 226.9614


279/Unknown  42s 150ms/step - accuracy: 0.9486 - loss: 226.9636


280/Unknown  42s 150ms/step - accuracy: 0.9486 - loss: 226.9648


281/Unknown  42s 150ms/step - accuracy: 0.9486 - loss: 226.9664


282/Unknown  43s 150ms/step - accuracy: 0.9487 - loss: 226.9669


283/Unknown  43s 150ms/step - accuracy: 0.9487 - loss: 226.9666


284/Unknown  43s 150ms/step - accuracy: 0.9487 - loss: 226.9671


285/Unknown  43s 151ms/step - accuracy: 0.9487 - loss: 226.9668


286/Unknown  43s 151ms/step - accuracy: 0.9487 - loss: 226.9667


287/Unknown  43s 150ms/step - accuracy: 0.9487 - loss: 226.9665


288/Unknown  44s 151ms/step - accuracy: 0.9487 - loss: 226.9657


289/Unknown  44s 151ms/step - accuracy: 0.9487 - loss: 226.9642


290/Unknown  44s 151ms/step - accuracy: 0.9487 - loss: 226.9632


291/Unknown  44s 151ms/step - accuracy: 0.9487 - loss: 226.9620


292/Unknown  44s 151ms/step - accuracy: 0.9487 - loss: 226.9610


293/Unknown  44s 151ms/step - accuracy: 0.9487 - loss: 226.9598


294/Unknown  44s 151ms/step - accuracy: 0.9487 - loss: 226.9589


295/Unknown  45s 150ms/step - accuracy: 0.9487 - loss: 226.9574


296/Unknown  45s 150ms/step - accuracy: 0.9487 - loss: 226.9580


297/Unknown  45s 150ms/step - accuracy: 0.9487 - loss: 226.9589


298/Unknown  45s 150ms/step - accuracy: 0.9487 - loss: 226.9600


299/Unknown  45s 150ms/step - accuracy: 0.9487 - loss: 226.9605


300/Unknown  45s 150ms/step - accuracy: 0.9487 - loss: 226.9607


301/Unknown  45s 150ms/step - accuracy: 0.9488 - loss: 226.9622


302/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9636


303/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9660


304/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9686


305/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9712


306/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9734


307/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9763


308/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9788


309/Unknown  46s 150ms/step - accuracy: 0.9488 - loss: 226.9816


310/Unknown  47s 150ms/step - accuracy: 0.9488 - loss: 226.9850


311/Unknown  47s 150ms/step - accuracy: 0.9488 - loss: 226.9886


312/Unknown  47s 150ms/step - accuracy: 0.9488 - loss: 226.9916


313/Unknown  47s 150ms/step - accuracy: 0.9488 - loss: 226.9948


314/Unknown  47s 150ms/step - accuracy: 0.9488 - loss: 226.9977


315/Unknown  47s 150ms/step - accuracy: 0.9488 - loss: 227.0007


316/Unknown  47s 150ms/step - accuracy: 0.9488 - loss: 227.0032


317/Unknown  48s 150ms/step - accuracy: 0.9488 - loss: 227.0062


318/Unknown  48s 150ms/step - accuracy: 0.9488 - loss: 227.0096


319/Unknown  48s 150ms/step - accuracy: 0.9488 - loss: 227.0135


320/Unknown  48s 150ms/step - accuracy: 0.9488 - loss: 227.0171


321/Unknown  48s 150ms/step - accuracy: 0.9488 - loss: 227.0207


322/Unknown  48s 150ms/step - accuracy: 0.9488 - loss: 227.0244


323/Unknown  48s 150ms/step - accuracy: 0.9488 - loss: 227.0280


324/Unknown  49s 149ms/step - accuracy: 0.9488 - loss: 227.0314


325/Unknown  49s 149ms/step - accuracy: 0.9489 - loss: 227.0350


326/Unknown  49s 149ms/step - accuracy: 0.9489 - loss: 227.0395


327/Unknown  49s 149ms/step - accuracy: 0.9489 - loss: 227.0438


328/Unknown  49s 149ms/step - accuracy: 0.9489 - loss: 227.0479


329/Unknown  49s 149ms/step - accuracy: 0.9489 - loss: 227.0516


330/Unknown  50s 150ms/step - accuracy: 0.9489 - loss: 227.0557


331/Unknown  50s 150ms/step - accuracy: 0.9489 - loss: 227.0595


332/Unknown  50s 150ms/step - accuracy: 0.9489 - loss: 227.0644


333/Unknown  50s 150ms/step - accuracy: 0.9489 - loss: 227.0693


334/Unknown  50s 150ms/step - accuracy: 0.9489 - loss: 227.0736


335/Unknown  50s 150ms/step - accuracy: 0.9489 - loss: 227.0776


336/Unknown  51s 150ms/step - accuracy: 0.9489 - loss: 227.0809


337/Unknown  51s 150ms/step - accuracy: 0.9489 - loss: 227.0846


338/Unknown  51s 150ms/step - accuracy: 0.9489 - loss: 227.0884


339/Unknown  51s 150ms/step - accuracy: 0.9489 - loss: 227.0918


340/Unknown  51s 150ms/step - accuracy: 0.9489 - loss: 227.0958


341/Unknown  51s 150ms/step - accuracy: 0.9489 - loss: 227.1006


342/Unknown  52s 150ms/step - accuracy: 0.9489 - loss: 227.1051


343/Unknown  52s 150ms/step - accuracy: 0.9489 - loss: 227.1095


344/Unknown  52s 150ms/step - accuracy: 0.9489 - loss: 227.1141


345/Unknown  52s 150ms/step - accuracy: 0.9489 - loss: 227.1185


346/Unknown  52s 150ms/step - accuracy: 0.9489 - loss: 227.1235


347/Unknown  52s 150ms/step - accuracy: 0.9489 - loss: 227.1284


348/Unknown  52s 150ms/step - accuracy: 0.9489 - loss: 227.1345


349/Unknown  53s 150ms/step - accuracy: 0.9489 - loss: 227.1408


350/Unknown  53s 150ms/step - accuracy: 0.9489 - loss: 227.1469


351/Unknown  53s 150ms/step - accuracy: 0.9489 - loss: 227.1535


352/Unknown  53s 150ms/step - accuracy: 0.9490 - loss: 227.1598


353/Unknown  53s 150ms/step - accuracy: 0.9490 - loss: 227.1664


354/Unknown  53s 150ms/step - accuracy: 0.9490 - loss: 227.1744


355/Unknown  53s 150ms/step - accuracy: 0.9490 - loss: 227.1824


356/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.1902


357/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.1976


358/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.2054


359/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.2136


360/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.2215


361/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.2299


362/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.2383


363/Unknown  54s 150ms/step - accuracy: 0.9490 - loss: 227.2464


364/Unknown  55s 150ms/step - accuracy: 0.9490 - loss: 227.2545


365/Unknown  55s 150ms/step - accuracy: 0.9490 - loss: 227.2623


366/Unknown  55s 150ms/step - accuracy: 0.9490 - loss: 227.2691


367/Unknown  55s 150ms/step - accuracy: 0.9490 - loss: 227.2757


368/Unknown  55s 149ms/step - accuracy: 0.9490 - loss: 227.2823


369/Unknown  55s 149ms/step - accuracy: 0.9490 - loss: 227.2889


370/Unknown  55s 149ms/step - accuracy: 0.9490 - loss: 227.2951


371/Unknown  56s 149ms/step - accuracy: 0.9490 - loss: 227.3009


372/Unknown  56s 149ms/step - accuracy: 0.9490 - loss: 227.3066


373/Unknown  56s 149ms/step - accuracy: 0.9490 - loss: 227.3122


374/Unknown  56s 149ms/step - accuracy: 0.9490 - loss: 227.3181


375/Unknown  56s 149ms/step - accuracy: 0.9490 - loss: 227.3240


376/Unknown  56s 149ms/step - accuracy: 0.9490 - loss: 227.3297


377/Unknown  56s 149ms/step - accuracy: 0.9490 - loss: 227.3355


```
</div>
 377/377 ━━━━━━━━━━━━━━━━━━━━ 56s 149ms/step - accuracy: 0.9490 - loss: 227.3412


<div class="k-default-codeblock">
```
Test accuracy: 95.0%

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
