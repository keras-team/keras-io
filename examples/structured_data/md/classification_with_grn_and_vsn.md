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
                vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[input_]
                # Create an embedding layer with the specified dimensions
                embedding_encoder = layers.Embedding(
                    input_dim=len(vocabulary), output_dim=self.units
                )
                max_index = len(vocabulary) - 1  # Clamp the indices
                # torch had some index errors during embedding hence the clip function
                embedded_feature = embedding_encoder(
                    keras.ops.clip(inputs[input_], 0, max_index)
                )
                concat_inputs.append(embedded_feature)
            else:
                # Project the numeric feature to encoding_size using linear transformation.
                proj_feature = keras.ops.expand_dims(inputs[input_], -1)
                proj_feature = layers.Dense(units=self.units)(proj_feature)
                concat_inputs.append(proj_feature)

        v = layers.concatenate(concat_inputs)
        v = self.grn_concat(v)
        v = keras.ops.expand_dims(self.softmax(v), axis=-1)
        x = []
        for idx, input in enumerate(concat_inputs):
            x.append(self.grns[idx](input))
        x = keras.ops.stack(x, axis=1)

        # The reason for each individual backend calculation is that I couldn't find
        # the equivalent keras operation that is backend-agnostic. In the following case there's
        # a keras.ops.matmul but it was returning errors. I could have used the tensorflow matmul
        # for all backends, but due to jax jit tracing it results in an error.

        def matmul_dependent_on_backend(tensor_1, tensor_2):
            """
            Function for executing matmul for each backend.
            """
            # jax backend
            if keras.backend.backend() == "jax":
                import jax.numpy as jnp

                result = jnp.sum(tensor_1 * tensor_2, axis=1)
            elif keras.backend.backend() == "torch":
                import torch

                result = torch.sum(tensor_1 * tensor_2, dim=1)
            # tensorflow backend
            elif keras.backend.backend() == "tensorflow":
                import tensorflow as tf

                result = keras.ops.squeeze(
                    tf.matmul(tensor_1, tensor_2, transpose_a=True), axis=1
                )
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
  1/Unknown  1s 639ms/step - accuracy: 0.7547 - loss: 1135.6592


  2/Unknown  1s 213ms/step - accuracy: 0.7679 - loss: 1144.7844


  3/Unknown  1s 212ms/step - accuracy: 0.7874 - loss: 1133.4303


  4/Unknown  1s 217ms/step - accuracy: 0.8052 - loss: 1119.3180


  5/Unknown  2s 218ms/step - accuracy: 0.8182 - loss: 1108.6187


  6/Unknown  2s 219ms/step - accuracy: 0.8294 - loss: 1094.4551


  7/Unknown  2s 219ms/step - accuracy: 0.8383 - loss: 1078.6390


  8/Unknown  2s 216ms/step - accuracy: 0.8457 - loss: 1060.9896


  9/Unknown  2s 218ms/step - accuracy: 0.8520 - loss: 1043.9601


 10/Unknown  3s 219ms/step - accuracy: 0.8575 - loss: 1026.8148


 11/Unknown  3s 220ms/step - accuracy: 0.8621 - loss: 1010.5887


 12/Unknown  3s 220ms/step - accuracy: 0.8662 - loss: 994.9163 


 13/Unknown  3s 220ms/step - accuracy: 0.8698 - loss: 980.0228


 14/Unknown  4s 221ms/step - accuracy: 0.8729 - loss: 965.6760


 15/Unknown  4s 222ms/step - accuracy: 0.8756 - loss: 952.5342


 16/Unknown  4s 222ms/step - accuracy: 0.8780 - loss: 939.8143


 17/Unknown  4s 224ms/step - accuracy: 0.8802 - loss: 927.9307


 18/Unknown  4s 225ms/step - accuracy: 0.8822 - loss: 916.5122


 19/Unknown  5s 227ms/step - accuracy: 0.8841 - loss: 905.6166


 20/Unknown  5s 228ms/step - accuracy: 0.8858 - loss: 895.1278


 21/Unknown  5s 242ms/step - accuracy: 0.8874 - loss: 885.1747


 22/Unknown  6s 243ms/step - accuracy: 0.8889 - loss: 875.6932


 23/Unknown  6s 243ms/step - accuracy: 0.8902 - loss: 866.5015


 24/Unknown  6s 244ms/step - accuracy: 0.8915 - loss: 857.5644


 25/Unknown  6s 244ms/step - accuracy: 0.8927 - loss: 848.9833


 26/Unknown  7s 245ms/step - accuracy: 0.8939 - loss: 840.5500


 27/Unknown  7s 245ms/step - accuracy: 0.8950 - loss: 832.3251


 28/Unknown  7s 246ms/step - accuracy: 0.8960 - loss: 824.5741


 29/Unknown  8s 246ms/step - accuracy: 0.8970 - loss: 817.0550


 30/Unknown  8s 246ms/step - accuracy: 0.8980 - loss: 809.6518


 31/Unknown  8s 247ms/step - accuracy: 0.8989 - loss: 802.6484


 32/Unknown  8s 248ms/step - accuracy: 0.8998 - loss: 795.8282


 33/Unknown  9s 250ms/step - accuracy: 0.9007 - loss: 789.2944


 34/Unknown  9s 251ms/step - accuracy: 0.9014 - loss: 783.0054


 35/Unknown  9s 251ms/step - accuracy: 0.9022 - loss: 777.0065


 36/Unknown  9s 252ms/step - accuracy: 0.9028 - loss: 771.2643


 37/Unknown  10s 252ms/step - accuracy: 0.9035 - loss: 765.6718


 38/Unknown  10s 253ms/step - accuracy: 0.9041 - loss: 760.2097


 39/Unknown  10s 252ms/step - accuracy: 0.9047 - loss: 754.9798


 40/Unknown  10s 253ms/step - accuracy: 0.9053 - loss: 750.0705


 41/Unknown  11s 253ms/step - accuracy: 0.9059 - loss: 745.2705


 42/Unknown  11s 253ms/step - accuracy: 0.9064 - loss: 740.6479


 43/Unknown  11s 253ms/step - accuracy: 0.9069 - loss: 736.0775


 44/Unknown  12s 254ms/step - accuracy: 0.9074 - loss: 731.6031


 45/Unknown  12s 254ms/step - accuracy: 0.9079 - loss: 727.2579


 46/Unknown  12s 255ms/step - accuracy: 0.9083 - loss: 723.0585


 47/Unknown  12s 255ms/step - accuracy: 0.9088 - loss: 719.0982


 48/Unknown  13s 255ms/step - accuracy: 0.9092 - loss: 715.2946


 49/Unknown  13s 256ms/step - accuracy: 0.9096 - loss: 711.5944


 50/Unknown  13s 256ms/step - accuracy: 0.9100 - loss: 707.9474


 51/Unknown  13s 257ms/step - accuracy: 0.9104 - loss: 704.3686


 52/Unknown  14s 257ms/step - accuracy: 0.9107 - loss: 700.8705


 53/Unknown  14s 257ms/step - accuracy: 0.9111 - loss: 697.4860


 54/Unknown  14s 258ms/step - accuracy: 0.9115 - loss: 694.1683


 55/Unknown  15s 258ms/step - accuracy: 0.9118 - loss: 690.9406


 56/Unknown  15s 258ms/step - accuracy: 0.9122 - loss: 687.7889


 57/Unknown  15s 259ms/step - accuracy: 0.9125 - loss: 684.7002


 58/Unknown  15s 259ms/step - accuracy: 0.9128 - loss: 681.6891


 59/Unknown  16s 259ms/step - accuracy: 0.9131 - loss: 678.7246


 60/Unknown  16s 260ms/step - accuracy: 0.9135 - loss: 675.8397


 61/Unknown  16s 260ms/step - accuracy: 0.9138 - loss: 672.9956


 62/Unknown  17s 260ms/step - accuracy: 0.9141 - loss: 670.1965


 63/Unknown  17s 260ms/step - accuracy: 0.9144 - loss: 667.4491


 64/Unknown  17s 261ms/step - accuracy: 0.9147 - loss: 664.7303


 65/Unknown  17s 261ms/step - accuracy: 0.9149 - loss: 662.0557


 66/Unknown  18s 261ms/step - accuracy: 0.9152 - loss: 659.4619


 67/Unknown  18s 262ms/step - accuracy: 0.9155 - loss: 656.9384


 68/Unknown  18s 262ms/step - accuracy: 0.9157 - loss: 654.4771


 69/Unknown  18s 262ms/step - accuracy: 0.9160 - loss: 652.0395


 70/Unknown  19s 262ms/step - accuracy: 0.9162 - loss: 649.6675


 71/Unknown  19s 262ms/step - accuracy: 0.9165 - loss: 647.3254


 72/Unknown  19s 262ms/step - accuracy: 0.9167 - loss: 645.0142


 73/Unknown  20s 263ms/step - accuracy: 0.9170 - loss: 642.7188


 74/Unknown  20s 263ms/step - accuracy: 0.9172 - loss: 640.4702


 75/Unknown  20s 263ms/step - accuracy: 0.9174 - loss: 638.2885


 76/Unknown  20s 264ms/step - accuracy: 0.9176 - loss: 636.1548


 77/Unknown  21s 265ms/step - accuracy: 0.9178 - loss: 634.0768


 78/Unknown  21s 266ms/step - accuracy: 0.9181 - loss: 632.0367


 79/Unknown  21s 266ms/step - accuracy: 0.9183 - loss: 630.0468


 80/Unknown  22s 267ms/step - accuracy: 0.9185 - loss: 628.0881


 81/Unknown  22s 267ms/step - accuracy: 0.9186 - loss: 626.1548


 82/Unknown  22s 268ms/step - accuracy: 0.9188 - loss: 624.2523


 83/Unknown  23s 268ms/step - accuracy: 0.9190 - loss: 622.3786


 84/Unknown  23s 268ms/step - accuracy: 0.9192 - loss: 620.5410


 85/Unknown  23s 268ms/step - accuracy: 0.9194 - loss: 618.7263


 86/Unknown  23s 268ms/step - accuracy: 0.9196 - loss: 616.9473


 87/Unknown  24s 268ms/step - accuracy: 0.9198 - loss: 615.2016


 88/Unknown  24s 268ms/step - accuracy: 0.9199 - loss: 613.4753


 89/Unknown  24s 268ms/step - accuracy: 0.9201 - loss: 611.7766


 90/Unknown  24s 268ms/step - accuracy: 0.9203 - loss: 610.1014


 91/Unknown  25s 268ms/step - accuracy: 0.9204 - loss: 608.4407


 92/Unknown  25s 268ms/step - accuracy: 0.9206 - loss: 606.8138


 93/Unknown  25s 268ms/step - accuracy: 0.9208 - loss: 605.2163


 94/Unknown  26s 268ms/step - accuracy: 0.9209 - loss: 603.6697


 95/Unknown  26s 268ms/step - accuracy: 0.9211 - loss: 602.1389


 96/Unknown  26s 268ms/step - accuracy: 0.9212 - loss: 600.6275


 97/Unknown  26s 269ms/step - accuracy: 0.9214 - loss: 599.1346


 98/Unknown  27s 269ms/step - accuracy: 0.9215 - loss: 597.6561


 99/Unknown  27s 269ms/step - accuracy: 0.9217 - loss: 596.2009


100/Unknown  27s 269ms/step - accuracy: 0.9218 - loss: 594.7636


101/Unknown  28s 270ms/step - accuracy: 0.9220 - loss: 593.3494


102/Unknown  28s 271ms/step - accuracy: 0.9221 - loss: 591.9595


103/Unknown  28s 271ms/step - accuracy: 0.9222 - loss: 590.5878


104/Unknown  29s 271ms/step - accuracy: 0.9224 - loss: 589.2469


105/Unknown  29s 272ms/step - accuracy: 0.9225 - loss: 587.9120


106/Unknown  29s 272ms/step - accuracy: 0.9226 - loss: 586.5919


107/Unknown  29s 272ms/step - accuracy: 0.9228 - loss: 585.2885


108/Unknown  30s 272ms/step - accuracy: 0.9229 - loss: 584.0063


109/Unknown  30s 272ms/step - accuracy: 0.9230 - loss: 582.7457


110/Unknown  30s 272ms/step - accuracy: 0.9232 - loss: 581.4992


111/Unknown  31s 272ms/step - accuracy: 0.9233 - loss: 580.2630


112/Unknown  31s 272ms/step - accuracy: 0.9234 - loss: 579.0372


113/Unknown  31s 272ms/step - accuracy: 0.9235 - loss: 577.8276


114/Unknown  31s 271ms/step - accuracy: 0.9237 - loss: 576.6309


115/Unknown  32s 271ms/step - accuracy: 0.9238 - loss: 575.4511


116/Unknown  32s 272ms/step - accuracy: 0.9239 - loss: 574.2883


117/Unknown  32s 272ms/step - accuracy: 0.9240 - loss: 573.1392


118/Unknown  32s 272ms/step - accuracy: 0.9241 - loss: 571.9974


119/Unknown  33s 272ms/step - accuracy: 0.9242 - loss: 570.8646


120/Unknown  33s 272ms/step - accuracy: 0.9244 - loss: 569.7486


121/Unknown  33s 272ms/step - accuracy: 0.9245 - loss: 568.6423


122/Unknown  34s 272ms/step - accuracy: 0.9246 - loss: 567.5450


123/Unknown  34s 272ms/step - accuracy: 0.9247 - loss: 566.4605


124/Unknown  34s 272ms/step - accuracy: 0.9248 - loss: 565.3865


125/Unknown  34s 273ms/step - accuracy: 0.9249 - loss: 564.3419


126/Unknown  35s 273ms/step - accuracy: 0.9250 - loss: 563.3077


127/Unknown  35s 274ms/step - accuracy: 0.9251 - loss: 562.2865


128/Unknown  35s 274ms/step - accuracy: 0.9252 - loss: 561.2808


129/Unknown  36s 274ms/step - accuracy: 0.9253 - loss: 560.2839


130/Unknown  36s 275ms/step - accuracy: 0.9254 - loss: 559.2988


131/Unknown  36s 275ms/step - accuracy: 0.9255 - loss: 558.3257


132/Unknown  37s 275ms/step - accuracy: 0.9256 - loss: 557.3645


133/Unknown  37s 275ms/step - accuracy: 0.9257 - loss: 556.4144


134/Unknown  37s 275ms/step - accuracy: 0.9258 - loss: 555.4783


135/Unknown  37s 275ms/step - accuracy: 0.9258 - loss: 554.5511


136/Unknown  38s 275ms/step - accuracy: 0.9259 - loss: 553.6387


137/Unknown  38s 275ms/step - accuracy: 0.9260 - loss: 552.7291


138/Unknown  38s 275ms/step - accuracy: 0.9261 - loss: 551.8307


139/Unknown  39s 276ms/step - accuracy: 0.9262 - loss: 550.9406


140/Unknown  39s 276ms/step - accuracy: 0.9263 - loss: 550.0619


141/Unknown  39s 276ms/step - accuracy: 0.9264 - loss: 549.1920


142/Unknown  40s 276ms/step - accuracy: 0.9264 - loss: 548.3396


143/Unknown  40s 276ms/step - accuracy: 0.9265 - loss: 547.4969


144/Unknown  40s 276ms/step - accuracy: 0.9266 - loss: 546.6680


145/Unknown  40s 276ms/step - accuracy: 0.9267 - loss: 545.8489


146/Unknown  41s 276ms/step - accuracy: 0.9267 - loss: 545.0404


147/Unknown  41s 277ms/step - accuracy: 0.9268 - loss: 544.2534


148/Unknown  41s 277ms/step - accuracy: 0.9269 - loss: 543.4774


149/Unknown  42s 277ms/step - accuracy: 0.9269 - loss: 542.7190


150/Unknown  42s 277ms/step - accuracy: 0.9270 - loss: 541.9668


151/Unknown  42s 277ms/step - accuracy: 0.9271 - loss: 541.2206


152/Unknown  43s 278ms/step - accuracy: 0.9271 - loss: 540.4850


153/Unknown  43s 278ms/step - accuracy: 0.9272 - loss: 539.7543


154/Unknown  43s 278ms/step - accuracy: 0.9273 - loss: 539.0289


155/Unknown  43s 278ms/step - accuracy: 0.9273 - loss: 538.3131


156/Unknown  44s 278ms/step - accuracy: 0.9274 - loss: 537.6083


157/Unknown  44s 278ms/step - accuracy: 0.9274 - loss: 536.9111


158/Unknown  44s 278ms/step - accuracy: 0.9275 - loss: 536.2197


159/Unknown  45s 278ms/step - accuracy: 0.9276 - loss: 535.5397


160/Unknown  45s 278ms/step - accuracy: 0.9276 - loss: 534.8652


161/Unknown  45s 278ms/step - accuracy: 0.9277 - loss: 534.1959


162/Unknown  45s 279ms/step - accuracy: 0.9277 - loss: 533.5316


163/Unknown  46s 279ms/step - accuracy: 0.9278 - loss: 532.8730


164/Unknown  46s 279ms/step - accuracy: 0.9278 - loss: 532.2230


165/Unknown  46s 279ms/step - accuracy: 0.9279 - loss: 531.5815


166/Unknown  47s 279ms/step - accuracy: 0.9279 - loss: 530.9472


167/Unknown  47s 279ms/step - accuracy: 0.9280 - loss: 530.3171


168/Unknown  47s 279ms/step - accuracy: 0.9280 - loss: 529.6996


169/Unknown  48s 280ms/step - accuracy: 0.9281 - loss: 529.0839


170/Unknown  48s 280ms/step - accuracy: 0.9281 - loss: 528.4716


171/Unknown  48s 280ms/step - accuracy: 0.9282 - loss: 527.8640


172/Unknown  49s 280ms/step - accuracy: 0.9282 - loss: 527.2624


173/Unknown  49s 280ms/step - accuracy: 0.9283 - loss: 526.6663


174/Unknown  49s 280ms/step - accuracy: 0.9283 - loss: 526.0778


175/Unknown  49s 280ms/step - accuracy: 0.9284 - loss: 525.4918


176/Unknown  50s 280ms/step - accuracy: 0.9284 - loss: 524.9106


177/Unknown  50s 281ms/step - accuracy: 0.9285 - loss: 524.3377


178/Unknown  50s 281ms/step - accuracy: 0.9285 - loss: 523.7676


179/Unknown  51s 281ms/step - accuracy: 0.9286 - loss: 523.2013


180/Unknown  51s 281ms/step - accuracy: 0.9286 - loss: 522.6400


181/Unknown  51s 281ms/step - accuracy: 0.9287 - loss: 522.0832


182/Unknown  51s 281ms/step - accuracy: 0.9287 - loss: 521.5319


183/Unknown  52s 281ms/step - accuracy: 0.9287 - loss: 520.9847


184/Unknown  52s 280ms/step - accuracy: 0.9288 - loss: 520.4411


185/Unknown  52s 280ms/step - accuracy: 0.9288 - loss: 519.9037


186/Unknown  52s 280ms/step - accuracy: 0.9289 - loss: 519.3732


187/Unknown  53s 280ms/step - accuracy: 0.9289 - loss: 518.8501


188/Unknown  53s 280ms/step - accuracy: 0.9290 - loss: 518.3302


189/Unknown  53s 280ms/step - accuracy: 0.9290 - loss: 517.8130


190/Unknown  54s 280ms/step - accuracy: 0.9290 - loss: 517.2994


191/Unknown  54s 280ms/step - accuracy: 0.9291 - loss: 516.7885


192/Unknown  54s 280ms/step - accuracy: 0.9291 - loss: 516.2802


193/Unknown  54s 280ms/step - accuracy: 0.9292 - loss: 515.7775


194/Unknown  55s 280ms/step - accuracy: 0.9292 - loss: 515.2805


195/Unknown  55s 280ms/step - accuracy: 0.9292 - loss: 514.7891


196/Unknown  55s 280ms/step - accuracy: 0.9293 - loss: 514.3018


197/Unknown  56s 280ms/step - accuracy: 0.9293 - loss: 513.8198


198/Unknown  56s 280ms/step - accuracy: 0.9293 - loss: 513.3423


199/Unknown  56s 280ms/step - accuracy: 0.9294 - loss: 512.8683


200/Unknown  57s 281ms/step - accuracy: 0.9294 - loss: 512.3986


201/Unknown  57s 281ms/step - accuracy: 0.9295 - loss: 511.9317


202/Unknown  57s 281ms/step - accuracy: 0.9295 - loss: 511.4658


203/Unknown  57s 281ms/step - accuracy: 0.9295 - loss: 511.0026


204/Unknown  58s 281ms/step - accuracy: 0.9296 - loss: 510.5465


205/Unknown  58s 281ms/step - accuracy: 0.9296 - loss: 510.0919


206/Unknown  58s 281ms/step - accuracy: 0.9296 - loss: 509.6407


207/Unknown  59s 281ms/step - accuracy: 0.9297 - loss: 509.1948


208/Unknown  59s 281ms/step - accuracy: 0.9297 - loss: 508.7514


209/Unknown  59s 281ms/step - accuracy: 0.9297 - loss: 508.3121


210/Unknown  59s 282ms/step - accuracy: 0.9298 - loss: 507.8748


211/Unknown  60s 282ms/step - accuracy: 0.9298 - loss: 507.4425


212/Unknown  60s 282ms/step - accuracy: 0.9298 - loss: 507.0140


213/Unknown  60s 282ms/step - accuracy: 0.9299 - loss: 506.5887


214/Unknown  61s 282ms/step - accuracy: 0.9299 - loss: 506.1693


215/Unknown  61s 282ms/step - accuracy: 0.9299 - loss: 505.7507


216/Unknown  61s 282ms/step - accuracy: 0.9299 - loss: 505.3338


217/Unknown  62s 282ms/step - accuracy: 0.9300 - loss: 504.9218


218/Unknown  62s 282ms/step - accuracy: 0.9300 - loss: 504.5124


219/Unknown  62s 282ms/step - accuracy: 0.9300 - loss: 504.1049


220/Unknown  62s 282ms/step - accuracy: 0.9301 - loss: 503.7035


221/Unknown  63s 282ms/step - accuracy: 0.9301 - loss: 503.3031


222/Unknown  63s 282ms/step - accuracy: 0.9301 - loss: 502.9056


223/Unknown  63s 282ms/step - accuracy: 0.9301 - loss: 502.5111


224/Unknown  64s 283ms/step - accuracy: 0.9302 - loss: 502.1182


225/Unknown  64s 283ms/step - accuracy: 0.9302 - loss: 501.7288


226/Unknown  64s 283ms/step - accuracy: 0.9302 - loss: 501.3393


227/Unknown  65s 283ms/step - accuracy: 0.9303 - loss: 500.9521


228/Unknown  65s 283ms/step - accuracy: 0.9303 - loss: 500.5659


229/Unknown  65s 283ms/step - accuracy: 0.9303 - loss: 500.1812


230/Unknown  65s 283ms/step - accuracy: 0.9303 - loss: 499.8016


231/Unknown  66s 283ms/step - accuracy: 0.9304 - loss: 499.4240


232/Unknown  66s 283ms/step - accuracy: 0.9304 - loss: 499.0496


233/Unknown  66s 283ms/step - accuracy: 0.9304 - loss: 498.6773


234/Unknown  67s 284ms/step - accuracy: 0.9304 - loss: 498.3081


235/Unknown  67s 284ms/step - accuracy: 0.9305 - loss: 497.9410


236/Unknown  67s 284ms/step - accuracy: 0.9305 - loss: 497.5768


237/Unknown  68s 285ms/step - accuracy: 0.9305 - loss: 497.2149


238/Unknown  68s 285ms/step - accuracy: 0.9306 - loss: 496.8535


239/Unknown  69s 285ms/step - accuracy: 0.9306 - loss: 496.4950


240/Unknown  69s 285ms/step - accuracy: 0.9306 - loss: 496.1360


241/Unknown  69s 285ms/step - accuracy: 0.9306 - loss: 495.7780


242/Unknown  69s 285ms/step - accuracy: 0.9307 - loss: 495.4213


243/Unknown  70s 285ms/step - accuracy: 0.9307 - loss: 495.0667


244/Unknown  70s 285ms/step - accuracy: 0.9307 - loss: 494.7135


245/Unknown  70s 285ms/step - accuracy: 0.9307 - loss: 494.3597


246/Unknown  71s 285ms/step - accuracy: 0.9307 - loss: 494.0078


247/Unknown  71s 285ms/step - accuracy: 0.9308 - loss: 493.6620


248/Unknown  71s 285ms/step - accuracy: 0.9308 - loss: 493.3164


249/Unknown  71s 286ms/step - accuracy: 0.9308 - loss: 492.9716


250/Unknown  72s 286ms/step - accuracy: 0.9308 - loss: 492.6284


251/Unknown  72s 286ms/step - accuracy: 0.9309 - loss: 492.2888


252/Unknown  72s 286ms/step - accuracy: 0.9309 - loss: 491.9524


253/Unknown  73s 286ms/step - accuracy: 0.9309 - loss: 491.6182


254/Unknown  73s 286ms/step - accuracy: 0.9309 - loss: 491.2857


255/Unknown  73s 286ms/step - accuracy: 0.9310 - loss: 490.9565


256/Unknown  74s 286ms/step - accuracy: 0.9310 - loss: 490.6284


257/Unknown  74s 286ms/step - accuracy: 0.9310 - loss: 490.3038


258/Unknown  74s 287ms/step - accuracy: 0.9310 - loss: 489.9812


259/Unknown  75s 287ms/step - accuracy: 0.9310 - loss: 489.6599


260/Unknown  75s 287ms/step - accuracy: 0.9311 - loss: 489.3393


261/Unknown  75s 287ms/step - accuracy: 0.9311 - loss: 489.0207


262/Unknown  76s 287ms/step - accuracy: 0.9311 - loss: 488.7037


263/Unknown  76s 288ms/step - accuracy: 0.9311 - loss: 488.3870


264/Unknown  76s 288ms/step - accuracy: 0.9312 - loss: 488.0719


265/Unknown  77s 288ms/step - accuracy: 0.9312 - loss: 487.7573


266/Unknown  77s 288ms/step - accuracy: 0.9312 - loss: 487.4449


267/Unknown  77s 288ms/step - accuracy: 0.9312 - loss: 487.1332


268/Unknown  77s 288ms/step - accuracy: 0.9312 - loss: 486.8221


269/Unknown  78s 288ms/step - accuracy: 0.9313 - loss: 486.5128


270/Unknown  78s 288ms/step - accuracy: 0.9313 - loss: 486.2052


271/Unknown  78s 288ms/step - accuracy: 0.9313 - loss: 485.8982


272/Unknown  79s 288ms/step - accuracy: 0.9313 - loss: 485.5928


273/Unknown  79s 288ms/step - accuracy: 0.9313 - loss: 485.2877


274/Unknown  79s 288ms/step - accuracy: 0.9314 - loss: 484.9843


275/Unknown  80s 288ms/step - accuracy: 0.9314 - loss: 484.6812


276/Unknown  80s 288ms/step - accuracy: 0.9314 - loss: 484.3784


277/Unknown  80s 288ms/step - accuracy: 0.9314 - loss: 484.0761


278/Unknown  80s 288ms/step - accuracy: 0.9314 - loss: 483.7755


279/Unknown  81s 288ms/step - accuracy: 0.9315 - loss: 483.4770


280/Unknown  81s 288ms/step - accuracy: 0.9315 - loss: 483.1796


281/Unknown  81s 289ms/step - accuracy: 0.9315 - loss: 482.8820


282/Unknown  82s 289ms/step - accuracy: 0.9315 - loss: 482.5845


283/Unknown  82s 289ms/step - accuracy: 0.9315 - loss: 482.2882


284/Unknown  82s 289ms/step - accuracy: 0.9316 - loss: 481.9939


285/Unknown  83s 289ms/step - accuracy: 0.9316 - loss: 481.7016


286/Unknown  83s 289ms/step - accuracy: 0.9316 - loss: 481.4112


287/Unknown  83s 290ms/step - accuracy: 0.9316 - loss: 481.1231


288/Unknown  84s 290ms/step - accuracy: 0.9316 - loss: 480.8353


289/Unknown  84s 290ms/step - accuracy: 0.9316 - loss: 480.5485


290/Unknown  84s 290ms/step - accuracy: 0.9317 - loss: 480.2634


291/Unknown  85s 290ms/step - accuracy: 0.9317 - loss: 479.9796


292/Unknown  85s 290ms/step - accuracy: 0.9317 - loss: 479.6985


293/Unknown  85s 290ms/step - accuracy: 0.9317 - loss: 479.4188


294/Unknown  86s 290ms/step - accuracy: 0.9317 - loss: 479.1406


295/Unknown  86s 290ms/step - accuracy: 0.9317 - loss: 478.8636


296/Unknown  86s 290ms/step - accuracy: 0.9318 - loss: 478.5882


297/Unknown  87s 290ms/step - accuracy: 0.9318 - loss: 478.3139


298/Unknown  87s 291ms/step - accuracy: 0.9318 - loss: 478.0403


299/Unknown  87s 291ms/step - accuracy: 0.9318 - loss: 477.7664


300/Unknown  88s 291ms/step - accuracy: 0.9318 - loss: 477.4944


301/Unknown  88s 291ms/step - accuracy: 0.9318 - loss: 477.2241


302/Unknown  88s 291ms/step - accuracy: 0.9319 - loss: 476.9537


303/Unknown  88s 291ms/step - accuracy: 0.9319 - loss: 476.6845


304/Unknown  89s 291ms/step - accuracy: 0.9319 - loss: 476.4153


305/Unknown  89s 291ms/step - accuracy: 0.9319 - loss: 476.1466


306/Unknown  89s 291ms/step - accuracy: 0.9319 - loss: 475.8818


307/Unknown  90s 291ms/step - accuracy: 0.9319 - loss: 475.6189


308/Unknown  90s 291ms/step - accuracy: 0.9320 - loss: 475.3565


309/Unknown  90s 291ms/step - accuracy: 0.9320 - loss: 475.0963


310/Unknown  91s 291ms/step - accuracy: 0.9320 - loss: 474.8365


311/Unknown  91s 291ms/step - accuracy: 0.9320 - loss: 474.5764


312/Unknown  91s 291ms/step - accuracy: 0.9320 - loss: 474.3172


313/Unknown  91s 291ms/step - accuracy: 0.9320 - loss: 474.0599


314/Unknown  92s 291ms/step - accuracy: 0.9321 - loss: 473.8053


315/Unknown  92s 291ms/step - accuracy: 0.9321 - loss: 473.5508


316/Unknown  92s 291ms/step - accuracy: 0.9321 - loss: 473.2971


317/Unknown  93s 291ms/step - accuracy: 0.9321 - loss: 473.0444


318/Unknown  93s 291ms/step - accuracy: 0.9321 - loss: 472.7928


319/Unknown  93s 291ms/step - accuracy: 0.9321 - loss: 472.5423


320/Unknown  94s 292ms/step - accuracy: 0.9322 - loss: 472.2928


321/Unknown  94s 292ms/step - accuracy: 0.9322 - loss: 472.0435


322/Unknown  94s 292ms/step - accuracy: 0.9322 - loss: 471.7943


323/Unknown  95s 292ms/step - accuracy: 0.9322 - loss: 471.5467


324/Unknown  95s 292ms/step - accuracy: 0.9322 - loss: 471.2997


325/Unknown  95s 292ms/step - accuracy: 0.9322 - loss: 471.0542


326/Unknown  96s 293ms/step - accuracy: 0.9322 - loss: 470.8102


327/Unknown  96s 293ms/step - accuracy: 0.9323 - loss: 470.5675


328/Unknown  96s 292ms/step - accuracy: 0.9323 - loss: 470.3263


329/Unknown  97s 292ms/step - accuracy: 0.9323 - loss: 470.0857


330/Unknown  97s 292ms/step - accuracy: 0.9323 - loss: 469.8463


331/Unknown  97s 292ms/step - accuracy: 0.9323 - loss: 469.6079


332/Unknown  97s 292ms/step - accuracy: 0.9323 - loss: 469.3716


333/Unknown  98s 292ms/step - accuracy: 0.9323 - loss: 469.1367


334/Unknown  98s 292ms/step - accuracy: 0.9324 - loss: 468.9023


335/Unknown  98s 292ms/step - accuracy: 0.9324 - loss: 468.6692


336/Unknown  99s 293ms/step - accuracy: 0.9324 - loss: 468.4366


337/Unknown  99s 293ms/step - accuracy: 0.9324 - loss: 468.2042


338/Unknown  99s 293ms/step - accuracy: 0.9324 - loss: 467.9723


339/Unknown  100s 293ms/step - accuracy: 0.9324 - loss: 467.7420


340/Unknown  100s 293ms/step - accuracy: 0.9324 - loss: 467.5124


341/Unknown  100s 293ms/step - accuracy: 0.9325 - loss: 467.2834


342/Unknown  100s 293ms/step - accuracy: 0.9325 - loss: 467.0563


343/Unknown  101s 293ms/step - accuracy: 0.9325 - loss: 466.8293


344/Unknown  101s 293ms/step - accuracy: 0.9325 - loss: 466.6034


345/Unknown  101s 293ms/step - accuracy: 0.9325 - loss: 466.3770


346/Unknown  102s 293ms/step - accuracy: 0.9325 - loss: 466.1513


347/Unknown  102s 294ms/step - accuracy: 0.9325 - loss: 465.9265


348/Unknown  103s 294ms/step - accuracy: 0.9325 - loss: 465.7036


349/Unknown  103s 294ms/step - accuracy: 0.9326 - loss: 465.4817


350/Unknown  103s 294ms/step - accuracy: 0.9326 - loss: 465.2596


351/Unknown  104s 294ms/step - accuracy: 0.9326 - loss: 465.0396


352/Unknown  104s 294ms/step - accuracy: 0.9326 - loss: 464.8197


353/Unknown  104s 295ms/step - accuracy: 0.9326 - loss: 464.6001


354/Unknown  105s 295ms/step - accuracy: 0.9326 - loss: 464.3814


355/Unknown  105s 295ms/step - accuracy: 0.9326 - loss: 464.1634


356/Unknown  105s 295ms/step - accuracy: 0.9326 - loss: 463.9456


357/Unknown  106s 296ms/step - accuracy: 0.9327 - loss: 463.7287


358/Unknown  106s 296ms/step - accuracy: 0.9327 - loss: 463.5123


359/Unknown  107s 296ms/step - accuracy: 0.9327 - loss: 463.2966


360/Unknown  107s 296ms/step - accuracy: 0.9327 - loss: 463.0821


361/Unknown  107s 296ms/step - accuracy: 0.9327 - loss: 462.8690


362/Unknown  107s 296ms/step - accuracy: 0.9327 - loss: 462.6569


363/Unknown  108s 296ms/step - accuracy: 0.9327 - loss: 462.4464


364/Unknown  108s 296ms/step - accuracy: 0.9327 - loss: 462.2366


365/Unknown  108s 296ms/step - accuracy: 0.9328 - loss: 462.0275


366/Unknown  109s 296ms/step - accuracy: 0.9328 - loss: 461.8186


367/Unknown  109s 296ms/step - accuracy: 0.9328 - loss: 461.6106


368/Unknown  109s 296ms/step - accuracy: 0.9328 - loss: 461.4036


369/Unknown  110s 296ms/step - accuracy: 0.9328 - loss: 461.1969


370/Unknown  110s 296ms/step - accuracy: 0.9328 - loss: 460.9911


371/Unknown  110s 296ms/step - accuracy: 0.9328 - loss: 460.7857


372/Unknown  111s 297ms/step - accuracy: 0.9328 - loss: 460.5810


373/Unknown  111s 297ms/step - accuracy: 0.9329 - loss: 460.3773


374/Unknown  111s 297ms/step - accuracy: 0.9329 - loss: 460.1751


375/Unknown  112s 297ms/step - accuracy: 0.9329 - loss: 459.9739


376/Unknown  112s 297ms/step - accuracy: 0.9329 - loss: 459.7736


377/Unknown  112s 297ms/step - accuracy: 0.9329 - loss: 459.5740


378/Unknown  113s 297ms/step - accuracy: 0.9329 - loss: 459.3742


379/Unknown  113s 297ms/step - accuracy: 0.9329 - loss: 459.1753


380/Unknown  113s 297ms/step - accuracy: 0.9329 - loss: 458.9761


381/Unknown  113s 297ms/step - accuracy: 0.9329 - loss: 458.7773


382/Unknown  114s 297ms/step - accuracy: 0.9330 - loss: 458.5791


383/Unknown  114s 297ms/step - accuracy: 0.9330 - loss: 458.3817


384/Unknown  114s 297ms/step - accuracy: 0.9330 - loss: 458.1854


385/Unknown  115s 297ms/step - accuracy: 0.9330 - loss: 457.9907


386/Unknown  115s 297ms/step - accuracy: 0.9330 - loss: 457.7969


387/Unknown  115s 297ms/step - accuracy: 0.9330 - loss: 457.6028


388/Unknown  116s 297ms/step - accuracy: 0.9330 - loss: 457.4084


389/Unknown  116s 297ms/step - accuracy: 0.9330 - loss: 457.2144


390/Unknown  116s 297ms/step - accuracy: 0.9330 - loss: 457.0205


391/Unknown  116s 297ms/step - accuracy: 0.9331 - loss: 456.8265


392/Unknown  117s 297ms/step - accuracy: 0.9331 - loss: 456.6332


393/Unknown  117s 297ms/step - accuracy: 0.9331 - loss: 456.4408


394/Unknown  118s 298ms/step - accuracy: 0.9331 - loss: 456.2486


395/Unknown  118s 298ms/step - accuracy: 0.9331 - loss: 456.0572


396/Unknown  118s 298ms/step - accuracy: 0.9331 - loss: 455.8668


397/Unknown  119s 298ms/step - accuracy: 0.9331 - loss: 455.6776


398/Unknown  119s 298ms/step - accuracy: 0.9331 - loss: 455.4887


399/Unknown  119s 298ms/step - accuracy: 0.9331 - loss: 455.2997


400/Unknown  119s 298ms/step - accuracy: 0.9331 - loss: 455.1118


401/Unknown  120s 298ms/step - accuracy: 0.9332 - loss: 454.9251


402/Unknown  120s 298ms/step - accuracy: 0.9332 - loss: 454.7390


403/Unknown  121s 298ms/step - accuracy: 0.9332 - loss: 454.5541


404/Unknown  121s 298ms/step - accuracy: 0.9332 - loss: 454.3698


405/Unknown  121s 298ms/step - accuracy: 0.9332 - loss: 454.1861


406/Unknown  122s 299ms/step - accuracy: 0.9332 - loss: 454.0026


407/Unknown  122s 299ms/step - accuracy: 0.9332 - loss: 453.8191


408/Unknown  122s 299ms/step - accuracy: 0.9332 - loss: 453.6364


409/Unknown  122s 299ms/step - accuracy: 0.9332 - loss: 453.4547


410/Unknown  123s 299ms/step - accuracy: 0.9333 - loss: 453.2735


411/Unknown  123s 298ms/step - accuracy: 0.9333 - loss: 453.0928


412/Unknown  123s 298ms/step - accuracy: 0.9333 - loss: 452.9133


413/Unknown  124s 298ms/step - accuracy: 0.9333 - loss: 452.7352


414/Unknown  124s 298ms/step - accuracy: 0.9333 - loss: 452.5580


415/Unknown  124s 298ms/step - accuracy: 0.9333 - loss: 452.3810


416/Unknown  125s 298ms/step - accuracy: 0.9333 - loss: 452.2045


417/Unknown  125s 298ms/step - accuracy: 0.9333 - loss: 452.0283


418/Unknown  125s 298ms/step - accuracy: 0.9333 - loss: 451.8531


419/Unknown  125s 298ms/step - accuracy: 0.9333 - loss: 451.6781


420/Unknown  126s 299ms/step - accuracy: 0.9334 - loss: 451.5038


421/Unknown  126s 299ms/step - accuracy: 0.9334 - loss: 451.3297


422/Unknown  126s 299ms/step - accuracy: 0.9334 - loss: 451.1558


423/Unknown  127s 299ms/step - accuracy: 0.9334 - loss: 450.9830


424/Unknown  127s 299ms/step - accuracy: 0.9334 - loss: 450.8108


425/Unknown  127s 299ms/step - accuracy: 0.9334 - loss: 450.6389


426/Unknown  128s 299ms/step - accuracy: 0.9334 - loss: 450.4671


427/Unknown  128s 299ms/step - accuracy: 0.9334 - loss: 450.2961


428/Unknown  129s 300ms/step - accuracy: 0.9334 - loss: 450.1259


429/Unknown  129s 300ms/step - accuracy: 0.9334 - loss: 449.9558


430/Unknown  129s 300ms/step - accuracy: 0.9334 - loss: 449.7865


431/Unknown  130s 300ms/step - accuracy: 0.9335 - loss: 449.6180


432/Unknown  130s 300ms/step - accuracy: 0.9335 - loss: 449.4497


433/Unknown  130s 300ms/step - accuracy: 0.9335 - loss: 449.2816


434/Unknown  131s 300ms/step - accuracy: 0.9335 - loss: 449.1138


435/Unknown  131s 300ms/step - accuracy: 0.9335 - loss: 448.9468


436/Unknown  131s 300ms/step - accuracy: 0.9335 - loss: 448.7806


437/Unknown  131s 300ms/step - accuracy: 0.9335 - loss: 448.6151


438/Unknown  132s 300ms/step - accuracy: 0.9335 - loss: 448.4496


439/Unknown  132s 300ms/step - accuracy: 0.9335 - loss: 448.2848


440/Unknown  132s 300ms/step - accuracy: 0.9335 - loss: 448.1208


441/Unknown  133s 300ms/step - accuracy: 0.9335 - loss: 447.9579


442/Unknown  133s 300ms/step - accuracy: 0.9336 - loss: 447.7962


443/Unknown  133s 300ms/step - accuracy: 0.9336 - loss: 447.6350


444/Unknown  134s 300ms/step - accuracy: 0.9336 - loss: 447.4741


445/Unknown  134s 301ms/step - accuracy: 0.9336 - loss: 447.3135


446/Unknown  134s 301ms/step - accuracy: 0.9336 - loss: 447.1530


447/Unknown  135s 301ms/step - accuracy: 0.9336 - loss: 446.9935


448/Unknown  135s 301ms/step - accuracy: 0.9336 - loss: 446.8344


449/Unknown  135s 301ms/step - accuracy: 0.9336 - loss: 446.6755


450/Unknown  136s 301ms/step - accuracy: 0.9336 - loss: 446.5173


451/Unknown  136s 301ms/step - accuracy: 0.9336 - loss: 446.3598


452/Unknown  136s 301ms/step - accuracy: 0.9336 - loss: 446.2028


453/Unknown  137s 301ms/step - accuracy: 0.9336 - loss: 446.0462


454/Unknown  137s 301ms/step - accuracy: 0.9337 - loss: 445.8899


455/Unknown  137s 301ms/step - accuracy: 0.9337 - loss: 445.7338


456/Unknown  138s 301ms/step - accuracy: 0.9337 - loss: 445.5780


457/Unknown  138s 301ms/step - accuracy: 0.9337 - loss: 445.4228


458/Unknown  138s 301ms/step - accuracy: 0.9337 - loss: 445.2682


459/Unknown  139s 302ms/step - accuracy: 0.9337 - loss: 445.1138


460/Unknown  139s 302ms/step - accuracy: 0.9337 - loss: 444.9601


461/Unknown  139s 302ms/step - accuracy: 0.9337 - loss: 444.8067


462/Unknown  140s 302ms/step - accuracy: 0.9337 - loss: 444.6537


463/Unknown  140s 302ms/step - accuracy: 0.9337 - loss: 444.5009


464/Unknown  140s 302ms/step - accuracy: 0.9337 - loss: 444.3487


465/Unknown  141s 302ms/step - accuracy: 0.9337 - loss: 444.1962


466/Unknown  141s 302ms/step - accuracy: 0.9338 - loss: 444.0451


467/Unknown  141s 302ms/step - accuracy: 0.9338 - loss: 443.8947


468/Unknown  142s 302ms/step - accuracy: 0.9338 - loss: 443.7449


469/Unknown  142s 302ms/step - accuracy: 0.9338 - loss: 443.5953


470/Unknown  142s 302ms/step - accuracy: 0.9338 - loss: 443.4461


471/Unknown  143s 302ms/step - accuracy: 0.9338 - loss: 443.2972


472/Unknown  143s 302ms/step - accuracy: 0.9338 - loss: 443.1489


473/Unknown  143s 302ms/step - accuracy: 0.9338 - loss: 443.0009


474/Unknown  144s 302ms/step - accuracy: 0.9338 - loss: 442.8532


475/Unknown  144s 302ms/step - accuracy: 0.9338 - loss: 442.7062


476/Unknown  144s 303ms/step - accuracy: 0.9338 - loss: 442.5600


477/Unknown  145s 303ms/step - accuracy: 0.9338 - loss: 442.4141


478/Unknown  145s 303ms/step - accuracy: 0.9338 - loss: 442.2683


479/Unknown  145s 303ms/step - accuracy: 0.9339 - loss: 442.1230


480/Unknown  146s 303ms/step - accuracy: 0.9339 - loss: 441.9781


481/Unknown  146s 303ms/step - accuracy: 0.9339 - loss: 441.8338


482/Unknown  146s 303ms/step - accuracy: 0.9339 - loss: 441.6898


483/Unknown  147s 303ms/step - accuracy: 0.9339 - loss: 441.5460


484/Unknown  147s 303ms/step - accuracy: 0.9339 - loss: 441.4024


485/Unknown  147s 303ms/step - accuracy: 0.9339 - loss: 441.2589


486/Unknown  148s 303ms/step - accuracy: 0.9339 - loss: 441.1156


487/Unknown  148s 303ms/step - accuracy: 0.9339 - loss: 440.9724


488/Unknown  148s 303ms/step - accuracy: 0.9339 - loss: 440.8297


489/Unknown  149s 303ms/step - accuracy: 0.9339 - loss: 440.6873


490/Unknown  149s 303ms/step - accuracy: 0.9339 - loss: 440.5454


491/Unknown  149s 303ms/step - accuracy: 0.9339 - loss: 440.4040


492/Unknown  149s 303ms/step - accuracy: 0.9340 - loss: 440.2630


493/Unknown  150s 303ms/step - accuracy: 0.9340 - loss: 440.1223


494/Unknown  150s 303ms/step - accuracy: 0.9340 - loss: 439.9818


495/Unknown  150s 303ms/step - accuracy: 0.9340 - loss: 439.8419


496/Unknown  151s 303ms/step - accuracy: 0.9340 - loss: 439.7021


497/Unknown  151s 303ms/step - accuracy: 0.9340 - loss: 439.5632


498/Unknown  151s 303ms/step - accuracy: 0.9340 - loss: 439.4245


499/Unknown  152s 303ms/step - accuracy: 0.9340 - loss: 439.2861


500/Unknown  152s 303ms/step - accuracy: 0.9340 - loss: 439.1480


501/Unknown  152s 303ms/step - accuracy: 0.9340 - loss: 439.0100


502/Unknown  153s 303ms/step - accuracy: 0.9340 - loss: 438.8721


503/Unknown  153s 303ms/step - accuracy: 0.9340 - loss: 438.7347


504/Unknown  153s 304ms/step - accuracy: 0.9340 - loss: 438.5977


505/Unknown  154s 304ms/step - accuracy: 0.9340 - loss: 438.4611


506/Unknown  154s 304ms/step - accuracy: 0.9341 - loss: 438.3248


507/Unknown  154s 304ms/step - accuracy: 0.9341 - loss: 438.1890


508/Unknown  155s 304ms/step - accuracy: 0.9341 - loss: 438.0539


509/Unknown  155s 304ms/step - accuracy: 0.9341 - loss: 437.9191


510/Unknown  155s 304ms/step - accuracy: 0.9341 - loss: 437.7845


511/Unknown  156s 304ms/step - accuracy: 0.9341 - loss: 437.6505


512/Unknown  156s 304ms/step - accuracy: 0.9341 - loss: 437.5170


513/Unknown  156s 304ms/step - accuracy: 0.9341 - loss: 437.3838


514/Unknown  157s 304ms/step - accuracy: 0.9341 - loss: 437.2512


515/Unknown  157s 304ms/step - accuracy: 0.9341 - loss: 437.1193


516/Unknown  157s 304ms/step - accuracy: 0.9341 - loss: 436.9874


517/Unknown  158s 304ms/step - accuracy: 0.9341 - loss: 436.8558


518/Unknown  158s 304ms/step - accuracy: 0.9341 - loss: 436.7245


519/Unknown  158s 305ms/step - accuracy: 0.9341 - loss: 436.5931


520/Unknown  159s 305ms/step - accuracy: 0.9342 - loss: 436.4619


521/Unknown  159s 305ms/step - accuracy: 0.9342 - loss: 436.3311


522/Unknown  159s 305ms/step - accuracy: 0.9342 - loss: 436.2004


523/Unknown  160s 305ms/step - accuracy: 0.9342 - loss: 436.0700


524/Unknown  160s 305ms/step - accuracy: 0.9342 - loss: 435.9399


525/Unknown  160s 305ms/step - accuracy: 0.9342 - loss: 435.8101


526/Unknown  161s 305ms/step - accuracy: 0.9342 - loss: 435.6806


527/Unknown  161s 305ms/step - accuracy: 0.9342 - loss: 435.5518


528/Unknown  161s 305ms/step - accuracy: 0.9342 - loss: 435.4231


529/Unknown  162s 305ms/step - accuracy: 0.9342 - loss: 435.2952


530/Unknown  162s 305ms/step - accuracy: 0.9342 - loss: 435.1673


531/Unknown  163s 305ms/step - accuracy: 0.9342 - loss: 435.0397


532/Unknown  163s 305ms/step - accuracy: 0.9342 - loss: 434.9123


533/Unknown  163s 305ms/step - accuracy: 0.9342 - loss: 434.7850


534/Unknown  163s 305ms/step - accuracy: 0.9342 - loss: 434.6579


535/Unknown  164s 306ms/step - accuracy: 0.9343 - loss: 434.5314


536/Unknown  164s 306ms/step - accuracy: 0.9343 - loss: 434.4052


537/Unknown  164s 306ms/step - accuracy: 0.9343 - loss: 434.2791


538/Unknown  165s 306ms/step - accuracy: 0.9343 - loss: 434.1533


539/Unknown  165s 306ms/step - accuracy: 0.9343 - loss: 434.0278


540/Unknown  165s 306ms/step - accuracy: 0.9343 - loss: 433.9026


541/Unknown  166s 306ms/step - accuracy: 0.9343 - loss: 433.7784


542/Unknown  166s 306ms/step - accuracy: 0.9343 - loss: 433.6542


543/Unknown  166s 306ms/step - accuracy: 0.9343 - loss: 433.5302


544/Unknown  167s 306ms/step - accuracy: 0.9343 - loss: 433.4064


545/Unknown  167s 306ms/step - accuracy: 0.9343 - loss: 433.2828


546/Unknown  167s 306ms/step - accuracy: 0.9343 - loss: 433.1595


547/Unknown  168s 306ms/step - accuracy: 0.9343 - loss: 433.0361


548/Unknown  168s 306ms/step - accuracy: 0.9343 - loss: 432.9130


549/Unknown  168s 306ms/step - accuracy: 0.9343 - loss: 432.7905


550/Unknown  169s 306ms/step - accuracy: 0.9343 - loss: 432.6687


551/Unknown  169s 306ms/step - accuracy: 0.9344 - loss: 432.5470


552/Unknown  169s 306ms/step - accuracy: 0.9344 - loss: 432.4257


553/Unknown  170s 306ms/step - accuracy: 0.9344 - loss: 432.3044


554/Unknown  170s 306ms/step - accuracy: 0.9344 - loss: 432.1833


555/Unknown  170s 306ms/step - accuracy: 0.9344 - loss: 432.0628


556/Unknown  171s 306ms/step - accuracy: 0.9344 - loss: 431.9423


557/Unknown  171s 307ms/step - accuracy: 0.9344 - loss: 431.8222


558/Unknown  171s 307ms/step - accuracy: 0.9344 - loss: 431.7024


559/Unknown  172s 307ms/step - accuracy: 0.9344 - loss: 431.5831


560/Unknown  172s 307ms/step - accuracy: 0.9344 - loss: 431.4641


561/Unknown  172s 307ms/step - accuracy: 0.9344 - loss: 431.3452


562/Unknown  173s 307ms/step - accuracy: 0.9344 - loss: 431.2266


563/Unknown  173s 307ms/step - accuracy: 0.9344 - loss: 431.1085


564/Unknown  173s 307ms/step - accuracy: 0.9344 - loss: 430.9907


565/Unknown  174s 307ms/step - accuracy: 0.9344 - loss: 430.8736


566/Unknown  174s 307ms/step - accuracy: 0.9344 - loss: 430.7569


567/Unknown  174s 307ms/step - accuracy: 0.9344 - loss: 430.6405


568/Unknown  175s 307ms/step - accuracy: 0.9344 - loss: 430.5246


569/Unknown  175s 307ms/step - accuracy: 0.9345 - loss: 430.4088


570/Unknown  175s 307ms/step - accuracy: 0.9345 - loss: 430.2935


571/Unknown  176s 307ms/step - accuracy: 0.9345 - loss: 430.1787


572/Unknown  176s 307ms/step - accuracy: 0.9345 - loss: 430.0643


573/Unknown  176s 307ms/step - accuracy: 0.9345 - loss: 429.9501


574/Unknown  177s 307ms/step - accuracy: 0.9345 - loss: 429.8360


575/Unknown  177s 307ms/step - accuracy: 0.9345 - loss: 429.7223


576/Unknown  177s 307ms/step - accuracy: 0.9345 - loss: 429.6088


577/Unknown  178s 307ms/step - accuracy: 0.9345 - loss: 429.4955


578/Unknown  178s 307ms/step - accuracy: 0.9345 - loss: 429.3824


579/Unknown  178s 307ms/step - accuracy: 0.9345 - loss: 429.2694


580/Unknown  179s 307ms/step - accuracy: 0.9345 - loss: 429.1570


581/Unknown  179s 307ms/step - accuracy: 0.9345 - loss: 429.0447


582/Unknown  179s 307ms/step - accuracy: 0.9345 - loss: 428.9325


583/Unknown  180s 307ms/step - accuracy: 0.9345 - loss: 428.8205


584/Unknown  180s 307ms/step - accuracy: 0.9345 - loss: 428.7086


585/Unknown  180s 307ms/step - accuracy: 0.9345 - loss: 428.5968


586/Unknown  181s 307ms/step - accuracy: 0.9345 - loss: 428.4850


587/Unknown  181s 308ms/step - accuracy: 0.9345 - loss: 428.3734


588/Unknown  181s 308ms/step - accuracy: 0.9346 - loss: 428.2621


589/Unknown  182s 308ms/step - accuracy: 0.9346 - loss: 428.1508


590/Unknown  182s 308ms/step - accuracy: 0.9346 - loss: 428.0396


591/Unknown  182s 308ms/step - accuracy: 0.9346 - loss: 427.9288


592/Unknown  182s 308ms/step - accuracy: 0.9346 - loss: 427.8181


593/Unknown  183s 308ms/step - accuracy: 0.9346 - loss: 427.7074


594/Unknown  183s 308ms/step - accuracy: 0.9346 - loss: 427.5972


595/Unknown  183s 308ms/step - accuracy: 0.9346 - loss: 427.4879


596/Unknown  184s 308ms/step - accuracy: 0.9346 - loss: 427.3787


597/Unknown  184s 308ms/step - accuracy: 0.9346 - loss: 427.2694


598/Unknown  184s 308ms/step - accuracy: 0.9346 - loss: 427.1603


599/Unknown  185s 308ms/step - accuracy: 0.9346 - loss: 427.0517


600/Unknown  185s 308ms/step - accuracy: 0.9346 - loss: 426.9435


601/Unknown  185s 308ms/step - accuracy: 0.9346 - loss: 426.8353


602/Unknown  186s 308ms/step - accuracy: 0.9346 - loss: 426.7278


603/Unknown  186s 308ms/step - accuracy: 0.9346 - loss: 426.6205


604/Unknown  186s 308ms/step - accuracy: 0.9346 - loss: 426.5135


605/Unknown  187s 308ms/step - accuracy: 0.9346 - loss: 426.4069


606/Unknown  187s 308ms/step - accuracy: 0.9346 - loss: 426.3006


607/Unknown  187s 308ms/step - accuracy: 0.9346 - loss: 426.1946


608/Unknown  188s 308ms/step - accuracy: 0.9347 - loss: 426.0891


609/Unknown  188s 308ms/step - accuracy: 0.9347 - loss: 425.9837


610/Unknown  188s 308ms/step - accuracy: 0.9347 - loss: 425.8788


611/Unknown  189s 308ms/step - accuracy: 0.9347 - loss: 425.7745


612/Unknown  189s 308ms/step - accuracy: 0.9347 - loss: 425.6705


613/Unknown  189s 308ms/step - accuracy: 0.9347 - loss: 425.5667


614/Unknown  190s 308ms/step - accuracy: 0.9347 - loss: 425.4628


615/Unknown  190s 308ms/step - accuracy: 0.9347 - loss: 425.3591


616/Unknown  190s 308ms/step - accuracy: 0.9347 - loss: 425.2557


617/Unknown  191s 308ms/step - accuracy: 0.9347 - loss: 425.1524


618/Unknown  191s 308ms/step - accuracy: 0.9347 - loss: 425.0492


619/Unknown  191s 308ms/step - accuracy: 0.9347 - loss: 424.9461


620/Unknown  192s 309ms/step - accuracy: 0.9347 - loss: 424.8428


621/Unknown  192s 309ms/step - accuracy: 0.9347 - loss: 424.7397


622/Unknown  192s 309ms/step - accuracy: 0.9347 - loss: 424.6367


623/Unknown  193s 309ms/step - accuracy: 0.9347 - loss: 424.5339


624/Unknown  193s 309ms/step - accuracy: 0.9347 - loss: 424.4314


625/Unknown  193s 309ms/step - accuracy: 0.9347 - loss: 424.3293


626/Unknown  194s 309ms/step - accuracy: 0.9347 - loss: 424.2273


627/Unknown  194s 309ms/step - accuracy: 0.9347 - loss: 424.1257


628/Unknown  194s 309ms/step - accuracy: 0.9347 - loss: 424.0243


629/Unknown  195s 309ms/step - accuracy: 0.9347 - loss: 423.9233


630/Unknown  195s 309ms/step - accuracy: 0.9347 - loss: 423.8223


631/Unknown  195s 309ms/step - accuracy: 0.9348 - loss: 423.7213


632/Unknown  195s 309ms/step - accuracy: 0.9348 - loss: 423.6205


633/Unknown  196s 309ms/step - accuracy: 0.9348 - loss: 423.5200


634/Unknown  196s 309ms/step - accuracy: 0.9348 - loss: 423.4196


635/Unknown  196s 309ms/step - accuracy: 0.9348 - loss: 423.3194


636/Unknown  197s 309ms/step - accuracy: 0.9348 - loss: 423.2194


637/Unknown  197s 309ms/step - accuracy: 0.9348 - loss: 423.1197


638/Unknown  197s 309ms/step - accuracy: 0.9348 - loss: 423.0203


639/Unknown  198s 309ms/step - accuracy: 0.9348 - loss: 422.9211


640/Unknown  198s 309ms/step - accuracy: 0.9348 - loss: 422.8223

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()


```
</div>
 640/640 ━━━━━━━━━━━━━━━━━━━━ 224s 350ms/step - accuracy: 0.9348 - loss: 422.7237 - val_accuracy: 0.9369 - val_loss: 320.7682


<div class="k-default-codeblock">
```
Model training finished.
Evaluating model performance...

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  0s 454ms/step - accuracy: 0.9283 - loss: 244.9023


  2/Unknown  1s 198ms/step - accuracy: 0.9311 - loss: 263.8576


  3/Unknown  1s 204ms/step - accuracy: 0.9323 - loss: 275.0815


  4/Unknown  1s 209ms/step - accuracy: 0.9322 - loss: 285.0687


  5/Unknown  1s 211ms/step - accuracy: 0.9328 - loss: 290.7856


  6/Unknown  2s 213ms/step - accuracy: 0.9335 - loss: 294.0614


  7/Unknown  2s 218ms/step - accuracy: 0.9340 - loss: 295.1135


  8/Unknown  2s 223ms/step - accuracy: 0.9344 - loss: 294.9874


  9/Unknown  2s 227ms/step - accuracy: 0.9347 - loss: 295.5774


 10/Unknown  3s 230ms/step - accuracy: 0.9350 - loss: 296.0671


 11/Unknown  3s 228ms/step - accuracy: 0.9352 - loss: 296.4334


 12/Unknown  3s 228ms/step - accuracy: 0.9355 - loss: 296.5976


 13/Unknown  3s 229ms/step - accuracy: 0.9358 - loss: 296.8359


 14/Unknown  3s 228ms/step - accuracy: 0.9360 - loss: 296.8992


 15/Unknown  4s 228ms/step - accuracy: 0.9363 - loss: 297.1031


 16/Unknown  4s 228ms/step - accuracy: 0.9365 - loss: 297.1199


 17/Unknown  4s 227ms/step - accuracy: 0.9367 - loss: 297.1887


 18/Unknown  4s 227ms/step - accuracy: 0.9369 - loss: 297.4390


 19/Unknown  5s 226ms/step - accuracy: 0.9370 - loss: 297.9974


 20/Unknown  5s 225ms/step - accuracy: 0.9370 - loss: 298.6873


 21/Unknown  5s 226ms/step - accuracy: 0.9370 - loss: 299.6551


 22/Unknown  5s 226ms/step - accuracy: 0.9370 - loss: 300.4757


 23/Unknown  5s 226ms/step - accuracy: 0.9370 - loss: 301.2497


 24/Unknown  6s 226ms/step - accuracy: 0.9370 - loss: 301.9757


 25/Unknown  6s 227ms/step - accuracy: 0.9370 - loss: 302.7468


 26/Unknown  6s 227ms/step - accuracy: 0.9369 - loss: 303.3629


 27/Unknown  6s 228ms/step - accuracy: 0.9370 - loss: 303.8804


 28/Unknown  7s 228ms/step - accuracy: 0.9370 - loss: 304.4332


 29/Unknown  7s 229ms/step - accuracy: 0.9370 - loss: 304.9322


 30/Unknown  7s 229ms/step - accuracy: 0.9370 - loss: 305.3882


 31/Unknown  7s 229ms/step - accuracy: 0.9370 - loss: 305.8696


 32/Unknown  8s 229ms/step - accuracy: 0.9369 - loss: 306.2744


 33/Unknown  8s 230ms/step - accuracy: 0.9369 - loss: 306.7179


 34/Unknown  8s 230ms/step - accuracy: 0.9369 - loss: 307.2096


 35/Unknown  8s 230ms/step - accuracy: 0.9369 - loss: 307.6509


 36/Unknown  9s 230ms/step - accuracy: 0.9368 - loss: 308.0346


 37/Unknown  9s 230ms/step - accuracy: 0.9368 - loss: 308.4092


 38/Unknown  9s 230ms/step - accuracy: 0.9368 - loss: 308.7528


 39/Unknown  9s 230ms/step - accuracy: 0.9368 - loss: 309.0463


 40/Unknown  9s 230ms/step - accuracy: 0.9368 - loss: 309.3176


 41/Unknown  10s 230ms/step - accuracy: 0.9368 - loss: 309.5844


 42/Unknown  10s 231ms/step - accuracy: 0.9369 - loss: 309.7717


 43/Unknown  10s 231ms/step - accuracy: 0.9369 - loss: 309.9764


 44/Unknown  10s 231ms/step - accuracy: 0.9369 - loss: 310.1483


 45/Unknown  11s 232ms/step - accuracy: 0.9369 - loss: 310.2879


 46/Unknown  11s 232ms/step - accuracy: 0.9369 - loss: 310.4070


 47/Unknown  11s 232ms/step - accuracy: 0.9369 - loss: 310.4993


 48/Unknown  11s 233ms/step - accuracy: 0.9369 - loss: 310.6008


 49/Unknown  12s 233ms/step - accuracy: 0.9370 - loss: 310.6914


 50/Unknown  12s 233ms/step - accuracy: 0.9370 - loss: 310.8468


 51/Unknown  12s 233ms/step - accuracy: 0.9370 - loss: 310.9822


 52/Unknown  12s 233ms/step - accuracy: 0.9370 - loss: 311.0945


 53/Unknown  13s 234ms/step - accuracy: 0.9370 - loss: 311.1777


 54/Unknown  13s 233ms/step - accuracy: 0.9370 - loss: 311.2862


 55/Unknown  13s 234ms/step - accuracy: 0.9370 - loss: 311.4064


 56/Unknown  13s 234ms/step - accuracy: 0.9370 - loss: 311.4765


 57/Unknown  14s 234ms/step - accuracy: 0.9371 - loss: 311.5206


 58/Unknown  14s 234ms/step - accuracy: 0.9371 - loss: 311.6007


 59/Unknown  14s 235ms/step - accuracy: 0.9371 - loss: 311.6716


 60/Unknown  14s 236ms/step - accuracy: 0.9371 - loss: 311.7329


 61/Unknown  15s 238ms/step - accuracy: 0.9371 - loss: 311.8119


 62/Unknown  15s 238ms/step - accuracy: 0.9371 - loss: 311.8692


 63/Unknown  15s 239ms/step - accuracy: 0.9371 - loss: 311.9624


 64/Unknown  16s 239ms/step - accuracy: 0.9371 - loss: 312.0652


 65/Unknown  16s 239ms/step - accuracy: 0.9371 - loss: 312.1604


 66/Unknown  16s 239ms/step - accuracy: 0.9371 - loss: 312.2377


 67/Unknown  16s 239ms/step - accuracy: 0.9371 - loss: 312.3071


 68/Unknown  16s 239ms/step - accuracy: 0.9371 - loss: 312.3609


 69/Unknown  17s 239ms/step - accuracy: 0.9371 - loss: 312.4107


 70/Unknown  17s 239ms/step - accuracy: 0.9371 - loss: 312.4532


 71/Unknown  17s 238ms/step - accuracy: 0.9371 - loss: 312.4868


 72/Unknown  17s 238ms/step - accuracy: 0.9371 - loss: 312.5398


 73/Unknown  18s 238ms/step - accuracy: 0.9371 - loss: 312.6050


 74/Unknown  18s 237ms/step - accuracy: 0.9371 - loss: 312.6606


 75/Unknown  18s 238ms/step - accuracy: 0.9371 - loss: 312.7076


 76/Unknown  18s 238ms/step - accuracy: 0.9371 - loss: 312.7575


 77/Unknown  19s 238ms/step - accuracy: 0.9371 - loss: 312.7997


 78/Unknown  19s 237ms/step - accuracy: 0.9371 - loss: 312.8430


 79/Unknown  19s 237ms/step - accuracy: 0.9371 - loss: 312.8756


 80/Unknown  19s 238ms/step - accuracy: 0.9371 - loss: 312.8968


 81/Unknown  19s 238ms/step - accuracy: 0.9371 - loss: 312.9210


 82/Unknown  20s 238ms/step - accuracy: 0.9371 - loss: 312.9413


 83/Unknown  20s 237ms/step - accuracy: 0.9371 - loss: 312.9521


 84/Unknown  20s 237ms/step - accuracy: 0.9371 - loss: 312.9711


 85/Unknown  20s 238ms/step - accuracy: 0.9371 - loss: 312.9870


 86/Unknown  21s 238ms/step - accuracy: 0.9371 - loss: 313.0185


 87/Unknown  21s 238ms/step - accuracy: 0.9371 - loss: 313.0474


 88/Unknown  21s 238ms/step - accuracy: 0.9371 - loss: 313.0713


 89/Unknown  21s 238ms/step - accuracy: 0.9371 - loss: 313.0957


 90/Unknown  22s 238ms/step - accuracy: 0.9371 - loss: 313.1271


 91/Unknown  22s 238ms/step - accuracy: 0.9371 - loss: 313.1564


 92/Unknown  22s 238ms/step - accuracy: 0.9371 - loss: 313.1984


 93/Unknown  22s 238ms/step - accuracy: 0.9371 - loss: 313.2350


 94/Unknown  23s 238ms/step - accuracy: 0.9371 - loss: 313.2901


 95/Unknown  23s 237ms/step - accuracy: 0.9371 - loss: 313.3413


 96/Unknown  23s 237ms/step - accuracy: 0.9371 - loss: 313.3984


 97/Unknown  23s 237ms/step - accuracy: 0.9371 - loss: 313.4404


 98/Unknown  23s 237ms/step - accuracy: 0.9371 - loss: 313.4749


 99/Unknown  24s 237ms/step - accuracy: 0.9371 - loss: 313.5033


100/Unknown  24s 237ms/step - accuracy: 0.9370 - loss: 313.5537


101/Unknown  24s 237ms/step - accuracy: 0.9370 - loss: 313.5986


102/Unknown  24s 237ms/step - accuracy: 0.9370 - loss: 313.6464


103/Unknown  25s 237ms/step - accuracy: 0.9370 - loss: 313.7014


104/Unknown  25s 237ms/step - accuracy: 0.9370 - loss: 313.7552


105/Unknown  25s 237ms/step - accuracy: 0.9370 - loss: 313.8083


106/Unknown  25s 237ms/step - accuracy: 0.9370 - loss: 313.8546


107/Unknown  26s 237ms/step - accuracy: 0.9370 - loss: 313.9135


108/Unknown  26s 237ms/step - accuracy: 0.9370 - loss: 313.9725


109/Unknown  26s 237ms/step - accuracy: 0.9370 - loss: 314.0344


110/Unknown  26s 237ms/step - accuracy: 0.9370 - loss: 314.0972


111/Unknown  27s 237ms/step - accuracy: 0.9370 - loss: 314.1661


112/Unknown  27s 237ms/step - accuracy: 0.9370 - loss: 314.2336


113/Unknown  27s 237ms/step - accuracy: 0.9370 - loss: 314.3030


114/Unknown  27s 237ms/step - accuracy: 0.9369 - loss: 314.3631


115/Unknown  27s 237ms/step - accuracy: 0.9369 - loss: 314.4212


116/Unknown  28s 237ms/step - accuracy: 0.9369 - loss: 314.4800


117/Unknown  28s 237ms/step - accuracy: 0.9369 - loss: 314.5418


118/Unknown  28s 237ms/step - accuracy: 0.9369 - loss: 314.6005


119/Unknown  28s 237ms/step - accuracy: 0.9369 - loss: 314.6577


120/Unknown  29s 237ms/step - accuracy: 0.9369 - loss: 314.7148


121/Unknown  29s 237ms/step - accuracy: 0.9369 - loss: 314.7736


122/Unknown  29s 237ms/step - accuracy: 0.9369 - loss: 314.8324


123/Unknown  29s 237ms/step - accuracy: 0.9369 - loss: 314.8822


124/Unknown  30s 237ms/step - accuracy: 0.9369 - loss: 314.9319


125/Unknown  30s 236ms/step - accuracy: 0.9369 - loss: 314.9872


126/Unknown  30s 236ms/step - accuracy: 0.9369 - loss: 315.0437


127/Unknown  30s 236ms/step - accuracy: 0.9369 - loss: 315.1023


128/Unknown  30s 236ms/step - accuracy: 0.9369 - loss: 315.1588


129/Unknown  31s 236ms/step - accuracy: 0.9369 - loss: 315.2117


130/Unknown  31s 236ms/step - accuracy: 0.9369 - loss: 315.2639


131/Unknown  31s 236ms/step - accuracy: 0.9369 - loss: 315.3077


132/Unknown  31s 236ms/step - accuracy: 0.9369 - loss: 315.3511


133/Unknown  32s 236ms/step - accuracy: 0.9369 - loss: 315.3907


134/Unknown  32s 236ms/step - accuracy: 0.9369 - loss: 315.4317


135/Unknown  32s 236ms/step - accuracy: 0.9369 - loss: 315.4686


136/Unknown  32s 236ms/step - accuracy: 0.9369 - loss: 315.5135


137/Unknown  33s 236ms/step - accuracy: 0.9369 - loss: 315.5602


138/Unknown  33s 236ms/step - accuracy: 0.9369 - loss: 315.6026


139/Unknown  33s 236ms/step - accuracy: 0.9369 - loss: 315.6417


140/Unknown  33s 236ms/step - accuracy: 0.9369 - loss: 315.6794


141/Unknown  34s 236ms/step - accuracy: 0.9369 - loss: 315.7147


142/Unknown  34s 236ms/step - accuracy: 0.9369 - loss: 315.7485


143/Unknown  34s 236ms/step - accuracy: 0.9369 - loss: 315.7847


144/Unknown  34s 236ms/step - accuracy: 0.9369 - loss: 315.8179


145/Unknown  34s 236ms/step - accuracy: 0.9369 - loss: 315.8486


146/Unknown  35s 236ms/step - accuracy: 0.9369 - loss: 315.8745


147/Unknown  35s 236ms/step - accuracy: 0.9369 - loss: 315.8998


148/Unknown  35s 237ms/step - accuracy: 0.9369 - loss: 315.9251


149/Unknown  36s 237ms/step - accuracy: 0.9369 - loss: 315.9571


150/Unknown  36s 237ms/step - accuracy: 0.9369 - loss: 315.9861


151/Unknown  36s 238ms/step - accuracy: 0.9369 - loss: 316.0147


152/Unknown  36s 238ms/step - accuracy: 0.9369 - loss: 316.0457


153/Unknown  37s 238ms/step - accuracy: 0.9369 - loss: 316.0818


154/Unknown  37s 239ms/step - accuracy: 0.9369 - loss: 316.1190


155/Unknown  37s 239ms/step - accuracy: 0.9369 - loss: 316.1610


156/Unknown  37s 239ms/step - accuracy: 0.9369 - loss: 316.2017


157/Unknown  38s 239ms/step - accuracy: 0.9369 - loss: 316.2386


158/Unknown  38s 239ms/step - accuracy: 0.9369 - loss: 316.2743


159/Unknown  38s 239ms/step - accuracy: 0.9369 - loss: 316.3090


160/Unknown  38s 239ms/step - accuracy: 0.9369 - loss: 316.3393


161/Unknown  39s 239ms/step - accuracy: 0.9369 - loss: 316.3685


162/Unknown  39s 238ms/step - accuracy: 0.9369 - loss: 316.3964


163/Unknown  39s 238ms/step - accuracy: 0.9369 - loss: 316.4256


164/Unknown  39s 238ms/step - accuracy: 0.9369 - loss: 316.4547


165/Unknown  40s 238ms/step - accuracy: 0.9369 - loss: 316.4827


166/Unknown  40s 238ms/step - accuracy: 0.9369 - loss: 316.5098


167/Unknown  40s 238ms/step - accuracy: 0.9369 - loss: 316.5354


168/Unknown  40s 238ms/step - accuracy: 0.9369 - loss: 316.5593


169/Unknown  40s 238ms/step - accuracy: 0.9369 - loss: 316.5794


170/Unknown  41s 238ms/step - accuracy: 0.9369 - loss: 316.5972


171/Unknown  41s 238ms/step - accuracy: 0.9369 - loss: 316.6163


172/Unknown  41s 238ms/step - accuracy: 0.9369 - loss: 316.6369


173/Unknown  41s 238ms/step - accuracy: 0.9369 - loss: 316.6581


174/Unknown  42s 238ms/step - accuracy: 0.9369 - loss: 316.6765


175/Unknown  42s 238ms/step - accuracy: 0.9369 - loss: 316.6960


176/Unknown  42s 238ms/step - accuracy: 0.9369 - loss: 316.7166


177/Unknown  42s 238ms/step - accuracy: 0.9369 - loss: 316.7375


178/Unknown  43s 238ms/step - accuracy: 0.9369 - loss: 316.7581


179/Unknown  43s 238ms/step - accuracy: 0.9369 - loss: 316.7736


180/Unknown  43s 238ms/step - accuracy: 0.9369 - loss: 316.7855


181/Unknown  43s 238ms/step - accuracy: 0.9369 - loss: 316.7971


182/Unknown  44s 238ms/step - accuracy: 0.9369 - loss: 316.8113


183/Unknown  44s 238ms/step - accuracy: 0.9369 - loss: 316.8255


184/Unknown  44s 238ms/step - accuracy: 0.9369 - loss: 316.8394


185/Unknown  44s 238ms/step - accuracy: 0.9369 - loss: 316.8533


186/Unknown  44s 238ms/step - accuracy: 0.9369 - loss: 316.8657


187/Unknown  45s 238ms/step - accuracy: 0.9369 - loss: 316.8781


188/Unknown  45s 238ms/step - accuracy: 0.9369 - loss: 316.8895


189/Unknown  45s 238ms/step - accuracy: 0.9369 - loss: 316.9013


190/Unknown  45s 238ms/step - accuracy: 0.9369 - loss: 316.9127


191/Unknown  46s 238ms/step - accuracy: 0.9369 - loss: 316.9218


192/Unknown  46s 238ms/step - accuracy: 0.9369 - loss: 316.9285


193/Unknown  46s 238ms/step - accuracy: 0.9369 - loss: 316.9337


194/Unknown  46s 238ms/step - accuracy: 0.9369 - loss: 316.9380


195/Unknown  47s 238ms/step - accuracy: 0.9369 - loss: 316.9421


196/Unknown  47s 238ms/step - accuracy: 0.9370 - loss: 316.9471


197/Unknown  47s 238ms/step - accuracy: 0.9370 - loss: 316.9490


198/Unknown  47s 237ms/step - accuracy: 0.9370 - loss: 316.9529


199/Unknown  47s 237ms/step - accuracy: 0.9370 - loss: 316.9548


200/Unknown  48s 237ms/step - accuracy: 0.9370 - loss: 316.9601


201/Unknown  48s 237ms/step - accuracy: 0.9370 - loss: 316.9623


202/Unknown  48s 237ms/step - accuracy: 0.9370 - loss: 316.9635


203/Unknown  48s 237ms/step - accuracy: 0.9370 - loss: 316.9644


204/Unknown  49s 237ms/step - accuracy: 0.9370 - loss: 316.9641


205/Unknown  49s 237ms/step - accuracy: 0.9370 - loss: 316.9671


206/Unknown  49s 237ms/step - accuracy: 0.9370 - loss: 316.9708


207/Unknown  49s 237ms/step - accuracy: 0.9370 - loss: 316.9719


208/Unknown  50s 237ms/step - accuracy: 0.9370 - loss: 316.9739


209/Unknown  50s 237ms/step - accuracy: 0.9370 - loss: 316.9771


210/Unknown  50s 237ms/step - accuracy: 0.9370 - loss: 316.9793


211/Unknown  50s 237ms/step - accuracy: 0.9370 - loss: 316.9799


212/Unknown  51s 237ms/step - accuracy: 0.9370 - loss: 316.9795


213/Unknown  51s 237ms/step - accuracy: 0.9370 - loss: 316.9814


214/Unknown  51s 237ms/step - accuracy: 0.9371 - loss: 316.9827


215/Unknown  51s 237ms/step - accuracy: 0.9371 - loss: 316.9838


216/Unknown  51s 237ms/step - accuracy: 0.9371 - loss: 316.9861


217/Unknown  52s 237ms/step - accuracy: 0.9371 - loss: 316.9871


218/Unknown  52s 238ms/step - accuracy: 0.9371 - loss: 316.9863


219/Unknown  52s 238ms/step - accuracy: 0.9371 - loss: 316.9843


220/Unknown  53s 238ms/step - accuracy: 0.9371 - loss: 316.9799


221/Unknown  53s 238ms/step - accuracy: 0.9371 - loss: 316.9756


222/Unknown  53s 238ms/step - accuracy: 0.9371 - loss: 316.9699


223/Unknown  53s 238ms/step - accuracy: 0.9371 - loss: 316.9641


224/Unknown  53s 238ms/step - accuracy: 0.9371 - loss: 316.9575


225/Unknown  54s 238ms/step - accuracy: 0.9371 - loss: 316.9523


226/Unknown  54s 238ms/step - accuracy: 0.9371 - loss: 316.9461


227/Unknown  54s 238ms/step - accuracy: 0.9371 - loss: 316.9384


228/Unknown  54s 238ms/step - accuracy: 0.9371 - loss: 316.9294


229/Unknown  55s 238ms/step - accuracy: 0.9371 - loss: 316.9204


230/Unknown  55s 237ms/step - accuracy: 0.9371 - loss: 316.9109


231/Unknown  55s 237ms/step - accuracy: 0.9372 - loss: 316.9017


232/Unknown  55s 237ms/step - accuracy: 0.9372 - loss: 316.8900


233/Unknown  55s 237ms/step - accuracy: 0.9372 - loss: 316.8779


234/Unknown  56s 237ms/step - accuracy: 0.9372 - loss: 316.8656


235/Unknown  56s 237ms/step - accuracy: 0.9372 - loss: 316.8537


236/Unknown  56s 237ms/step - accuracy: 0.9372 - loss: 316.8433


237/Unknown  56s 237ms/step - accuracy: 0.9372 - loss: 316.8315


238/Unknown  57s 237ms/step - accuracy: 0.9372 - loss: 316.8193


239/Unknown  57s 237ms/step - accuracy: 0.9372 - loss: 316.8091


240/Unknown  57s 237ms/step - accuracy: 0.9372 - loss: 316.7974


241/Unknown  57s 237ms/step - accuracy: 0.9372 - loss: 316.7842


242/Unknown  58s 237ms/step - accuracy: 0.9372 - loss: 316.7721


243/Unknown  58s 237ms/step - accuracy: 0.9372 - loss: 316.7608


244/Unknown  58s 237ms/step - accuracy: 0.9372 - loss: 316.7492


245/Unknown  58s 237ms/step - accuracy: 0.9372 - loss: 316.7386


246/Unknown  59s 237ms/step - accuracy: 0.9372 - loss: 316.7275


247/Unknown  59s 237ms/step - accuracy: 0.9372 - loss: 316.7159


248/Unknown  59s 237ms/step - accuracy: 0.9373 - loss: 316.7039


249/Unknown  59s 237ms/step - accuracy: 0.9373 - loss: 316.6928


250/Unknown  60s 238ms/step - accuracy: 0.9373 - loss: 316.6821


251/Unknown  60s 238ms/step - accuracy: 0.9373 - loss: 316.6718


252/Unknown  60s 238ms/step - accuracy: 0.9373 - loss: 316.6607


253/Unknown  60s 238ms/step - accuracy: 0.9373 - loss: 316.6497


254/Unknown  61s 238ms/step - accuracy: 0.9373 - loss: 316.6387


255/Unknown  61s 238ms/step - accuracy: 0.9373 - loss: 316.6275


256/Unknown  61s 238ms/step - accuracy: 0.9373 - loss: 316.6164


257/Unknown  61s 238ms/step - accuracy: 0.9373 - loss: 316.6047


258/Unknown  62s 238ms/step - accuracy: 0.9373 - loss: 316.5946


259/Unknown  62s 238ms/step - accuracy: 0.9373 - loss: 316.5848


260/Unknown  62s 237ms/step - accuracy: 0.9373 - loss: 316.5742


261/Unknown  62s 237ms/step - accuracy: 0.9373 - loss: 316.5627


262/Unknown  62s 237ms/step - accuracy: 0.9373 - loss: 316.5490


263/Unknown  63s 237ms/step - accuracy: 0.9373 - loss: 316.5363


264/Unknown  63s 237ms/step - accuracy: 0.9373 - loss: 316.5237


265/Unknown  63s 237ms/step - accuracy: 0.9374 - loss: 316.5107


266/Unknown  63s 237ms/step - accuracy: 0.9374 - loss: 316.4989


267/Unknown  64s 237ms/step - accuracy: 0.9374 - loss: 316.4873


268/Unknown  64s 237ms/step - accuracy: 0.9374 - loss: 316.4750


269/Unknown  64s 237ms/step - accuracy: 0.9374 - loss: 316.4644


270/Unknown  64s 237ms/step - accuracy: 0.9374 - loss: 316.4539


271/Unknown  64s 237ms/step - accuracy: 0.9374 - loss: 316.4436


272/Unknown  65s 237ms/step - accuracy: 0.9374 - loss: 316.4321


273/Unknown  65s 237ms/step - accuracy: 0.9374 - loss: 316.4208


274/Unknown  65s 237ms/step - accuracy: 0.9374 - loss: 316.4090


275/Unknown  65s 237ms/step - accuracy: 0.9374 - loss: 316.3964


276/Unknown  66s 237ms/step - accuracy: 0.9374 - loss: 316.3851


277/Unknown  66s 237ms/step - accuracy: 0.9374 - loss: 316.3748


278/Unknown  66s 237ms/step - accuracy: 0.9374 - loss: 316.3647


279/Unknown  66s 237ms/step - accuracy: 0.9374 - loss: 316.3557


280/Unknown  67s 237ms/step - accuracy: 0.9374 - loss: 316.3452


281/Unknown  67s 237ms/step - accuracy: 0.9374 - loss: 316.3349


282/Unknown  67s 237ms/step - accuracy: 0.9375 - loss: 316.3237


283/Unknown  67s 237ms/step - accuracy: 0.9375 - loss: 316.3120


284/Unknown  67s 237ms/step - accuracy: 0.9375 - loss: 316.3006


285/Unknown  68s 237ms/step - accuracy: 0.9375 - loss: 316.2882


286/Unknown  68s 237ms/step - accuracy: 0.9375 - loss: 316.2762


287/Unknown  68s 237ms/step - accuracy: 0.9375 - loss: 316.2653


288/Unknown  68s 237ms/step - accuracy: 0.9375 - loss: 316.2543


289/Unknown  69s 237ms/step - accuracy: 0.9375 - loss: 316.2430


290/Unknown  69s 237ms/step - accuracy: 0.9375 - loss: 316.2317


291/Unknown  69s 237ms/step - accuracy: 0.9375 - loss: 316.2207


292/Unknown  69s 237ms/step - accuracy: 0.9375 - loss: 316.2094


293/Unknown  70s 237ms/step - accuracy: 0.9375 - loss: 316.1977


294/Unknown  70s 237ms/step - accuracy: 0.9375 - loss: 316.1866


295/Unknown  70s 236ms/step - accuracy: 0.9375 - loss: 316.1754


296/Unknown  70s 236ms/step - accuracy: 0.9375 - loss: 316.1661


297/Unknown  70s 236ms/step - accuracy: 0.9375 - loss: 316.1575


298/Unknown  71s 236ms/step - accuracy: 0.9375 - loss: 316.1480


299/Unknown  71s 236ms/step - accuracy: 0.9375 - loss: 316.1383


300/Unknown  71s 236ms/step - accuracy: 0.9375 - loss: 316.1286


301/Unknown  71s 236ms/step - accuracy: 0.9375 - loss: 316.1200


302/Unknown  72s 236ms/step - accuracy: 0.9375 - loss: 316.1111


303/Unknown  72s 236ms/step - accuracy: 0.9376 - loss: 316.1031


304/Unknown  72s 236ms/step - accuracy: 0.9376 - loss: 316.0949


305/Unknown  72s 236ms/step - accuracy: 0.9376 - loss: 316.0871


306/Unknown  73s 236ms/step - accuracy: 0.9376 - loss: 316.0789


307/Unknown  73s 236ms/step - accuracy: 0.9376 - loss: 316.0715


308/Unknown  73s 236ms/step - accuracy: 0.9376 - loss: 316.0636


309/Unknown  73s 236ms/step - accuracy: 0.9376 - loss: 316.0563


310/Unknown  73s 236ms/step - accuracy: 0.9376 - loss: 316.0502


311/Unknown  74s 236ms/step - accuracy: 0.9376 - loss: 316.0448


312/Unknown  74s 236ms/step - accuracy: 0.9376 - loss: 316.0383


313/Unknown  74s 236ms/step - accuracy: 0.9376 - loss: 316.0314


314/Unknown  74s 236ms/step - accuracy: 0.9376 - loss: 316.0244


315/Unknown  75s 236ms/step - accuracy: 0.9376 - loss: 316.0172


316/Unknown  75s 236ms/step - accuracy: 0.9376 - loss: 316.0094


317/Unknown  75s 236ms/step - accuracy: 0.9376 - loss: 316.0017


318/Unknown  75s 236ms/step - accuracy: 0.9376 - loss: 315.9943


319/Unknown  76s 236ms/step - accuracy: 0.9376 - loss: 315.9870


320/Unknown  76s 236ms/step - accuracy: 0.9376 - loss: 315.9799


321/Unknown  76s 236ms/step - accuracy: 0.9376 - loss: 315.9724


322/Unknown  76s 236ms/step - accuracy: 0.9376 - loss: 315.9654


323/Unknown  76s 236ms/step - accuracy: 0.9376 - loss: 315.9579


324/Unknown  77s 236ms/step - accuracy: 0.9376 - loss: 315.9504


325/Unknown  77s 236ms/step - accuracy: 0.9377 - loss: 315.9435


326/Unknown  77s 236ms/step - accuracy: 0.9377 - loss: 315.9373


327/Unknown  77s 236ms/step - accuracy: 0.9377 - loss: 315.9312


328/Unknown  77s 235ms/step - accuracy: 0.9377 - loss: 315.9245


329/Unknown  78s 235ms/step - accuracy: 0.9377 - loss: 315.9179


330/Unknown  78s 235ms/step - accuracy: 0.9377 - loss: 315.9118


331/Unknown  78s 235ms/step - accuracy: 0.9377 - loss: 315.9054


332/Unknown  78s 235ms/step - accuracy: 0.9377 - loss: 315.9005


333/Unknown  79s 235ms/step - accuracy: 0.9377 - loss: 315.8957


334/Unknown  79s 235ms/step - accuracy: 0.9377 - loss: 315.8907


335/Unknown  79s 235ms/step - accuracy: 0.9377 - loss: 315.8854


336/Unknown  79s 235ms/step - accuracy: 0.9377 - loss: 315.8794


337/Unknown  80s 235ms/step - accuracy: 0.9377 - loss: 315.8741


338/Unknown  80s 235ms/step - accuracy: 0.9377 - loss: 315.8690


339/Unknown  80s 235ms/step - accuracy: 0.9377 - loss: 315.8629


340/Unknown  80s 235ms/step - accuracy: 0.9377 - loss: 315.8569


341/Unknown  80s 235ms/step - accuracy: 0.9377 - loss: 315.8515


342/Unknown  81s 235ms/step - accuracy: 0.9377 - loss: 315.8463


343/Unknown  81s 235ms/step - accuracy: 0.9377 - loss: 315.8414


344/Unknown  81s 235ms/step - accuracy: 0.9377 - loss: 315.8365


345/Unknown  81s 235ms/step - accuracy: 0.9377 - loss: 315.8317


346/Unknown  82s 235ms/step - accuracy: 0.9377 - loss: 315.8274


347/Unknown  82s 235ms/step - accuracy: 0.9377 - loss: 315.8230


348/Unknown  82s 235ms/step - accuracy: 0.9377 - loss: 315.8196


349/Unknown  82s 235ms/step - accuracy: 0.9377 - loss: 315.8168


350/Unknown  83s 235ms/step - accuracy: 0.9377 - loss: 315.8138


351/Unknown  83s 235ms/step - accuracy: 0.9377 - loss: 315.8109


352/Unknown  83s 235ms/step - accuracy: 0.9377 - loss: 315.8079


353/Unknown  83s 235ms/step - accuracy: 0.9377 - loss: 315.8058


354/Unknown  83s 235ms/step - accuracy: 0.9377 - loss: 315.8061


355/Unknown  84s 235ms/step - accuracy: 0.9377 - loss: 315.8066


356/Unknown  84s 235ms/step - accuracy: 0.9378 - loss: 315.8063


357/Unknown  84s 235ms/step - accuracy: 0.9378 - loss: 315.8054


358/Unknown  84s 235ms/step - accuracy: 0.9378 - loss: 315.8055


359/Unknown  85s 235ms/step - accuracy: 0.9378 - loss: 315.8056


360/Unknown  85s 235ms/step - accuracy: 0.9378 - loss: 315.8053


361/Unknown  85s 235ms/step - accuracy: 0.9378 - loss: 315.8054


362/Unknown  85s 235ms/step - accuracy: 0.9378 - loss: 315.8049


363/Unknown  86s 235ms/step - accuracy: 0.9378 - loss: 315.8039


364/Unknown  86s 235ms/step - accuracy: 0.9378 - loss: 315.8022


365/Unknown  86s 235ms/step - accuracy: 0.9378 - loss: 315.8002


366/Unknown  86s 235ms/step - accuracy: 0.9378 - loss: 315.7970


367/Unknown  87s 235ms/step - accuracy: 0.9378 - loss: 315.7933


368/Unknown  87s 235ms/step - accuracy: 0.9378 - loss: 315.7896


369/Unknown  87s 235ms/step - accuracy: 0.9378 - loss: 315.7862


370/Unknown  87s 235ms/step - accuracy: 0.9378 - loss: 315.7821


371/Unknown  88s 235ms/step - accuracy: 0.9378 - loss: 315.7777


372/Unknown  88s 235ms/step - accuracy: 0.9378 - loss: 315.7731


373/Unknown  88s 236ms/step - accuracy: 0.9378 - loss: 315.7687


374/Unknown  88s 236ms/step - accuracy: 0.9378 - loss: 315.7640


375/Unknown  89s 236ms/step - accuracy: 0.9378 - loss: 315.7592


376/Unknown  89s 236ms/step - accuracy: 0.9378 - loss: 315.7550


377/Unknown  89s 236ms/step - accuracy: 0.9378 - loss: 315.7512


```
</div>
 377/377 ━━━━━━━━━━━━━━━━━━━━ 89s 236ms/step - accuracy: 0.9378 - loss: 315.7473


<div class="k-default-codeblock">
```
Test accuracy: 93.87%

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
