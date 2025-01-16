"""
Title: Classification with Gated Residual and Variable Selection Networks
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/02/10
Last modified: 2025/01/08
Description: Using Gated Residual and Variable Selection Networks for income level prediction.
Accelerator: GPU
Converted to Keras 3 by: [Sitam Meur](https://github.com/sitamgithub-MSIT) and made backend-agnostic by: [Humbulani Ndou](https://github.com/Humbulani1234)
"""

"""
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
"""

"""
## The dataset

This example uses the
[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29)
provided by the
[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The task is binary classification to determine whether a person makes over 50K a year.

The dataset includes ~300K instances with 41 input features: 7 numerical features
and 34 categorical features.
"""

"""
## Setup
"""

import os
import subprocess
import tarfile

os.environ["KERAS_BACKEND"] = "torch"  # or jax, or tensorflow

import numpy as np
import pandas as pd
import keras
from keras import layers

"""
## Prepare the data

First we load the data from the UCI Machine Learning Repository into a Pandas DataFrame.
"""

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

"""
Determine the downloaded .tar.gz file path and
extract the files from the downloaded .tar.gz file
"""

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


"""
We convert the target column from string to integer.
"""

data["income_level"] = data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)
test_data["income_level"] = test_data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)


"""
Then, We split the dataset into train and validation sets.
"""

random_selection = np.random.rand(len(data.index)) <= 0.85
train_data = data[random_selection]
valid_data = data[~random_selection]


"""
Finally we store the train and test data splits locally to CSV files.
"""

train_data_file = "train_data.csv"
valid_data_file = "valid_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
valid_data.to_csv(valid_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)

"""
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for reading and
parsing the data into input features, and encoding the input features with respect
to their types.
"""

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

"""
## Create a `tf.data.Dataset` for training and evaluation

We create an input function to read and parse the file, and convert features and
labels into a [`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets) for
training and evaluation.
"""

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


"""
## Create model inputs
"""


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


"""
## Implement the Gated Linear Unit

[Gated Linear Units (GLUs)](https://arxiv.org/abs/1612.08083) provide the
flexibility to suppress input that are not relevant for a given task.
"""


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


"""
## Implement the Gated Residual Network

The Gated Residual Network (GRN) works as follows:

1. Applies the nonlinear ELU transformation to the inputs.
2. Applies linear transformation followed by dropout.
4. Applies GLU and adds the original inputs to the output of the GLU to perform skip
(residual) connection.
6. Applies layer normalization and produces the output.
"""


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


"""
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

"""


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


"""
## Create Gated Residual and Variable Selection Networks model
"""


def create_model(encoding_size):
    inputs = create_model_inputs()
    num_features = len(inputs)
    features = VariableSelection(num_features, encoding_size, dropout_rate)(inputs)
    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    # Functional model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


"""
## Compile, train, and evaluate the model
"""

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

"""
Let's visualize our connectivity graph:
"""

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

"""
You should achieve more than 95% accuracy on the test set.

To increase the learning capacity of the model, you can try increasing the
`encoding_size` value, or stacking multiple GRN layers on top of the VSN layer.
This may require to also increase the `dropout_rate` value to avoid overfitting.
"""

"""
**Example available on HuggingFace**

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Classification%20With%20GRN%20%26%20VSN-red)](https://huggingface.co/keras-io/structured-data-classification-grn-vsn) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Classification%20With%20GRN%20%26%20VSN-red)](https://huggingface.co/spaces/keras-io/structured-data-classification-grn-vsn) |
"""
