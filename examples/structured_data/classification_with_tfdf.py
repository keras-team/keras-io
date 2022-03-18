"""
Title: Classification with TensorFlow Decision Forests
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2022/01/25
Last modified: 2022/01/25
Description: Using TensorFlow Decision Forests for structured data classification.
"""

"""
## Introduction

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests)
is a collection of state-of-the-art algorithms of Decision Forest models
that are compatible with Keras APIs.
The models include [Random Forests](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel),
[Gradient Boosted Trees](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel),
and [CART](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/CartModel),
and can be used for regression, classification, and ranking task.
For a beginner's guide to TensorFlow Decision Forests,
please refer to this [tutorial](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab).


This example uses Gradient Boosted Trees model in binary classification of
structured data, and covers the following scenarios:

1. Build a decision forests model by specifying the input feature usage.
2. Implement a custom *Binary Target encoder* as a [Keras Preprocessing layer](https://keras.io/api/layers/preprocessing_layers/)
to encode the categorical features with respect to their target value co-occurrences,
and then use the encoded features to build a decision forests model.
3. Encode the categorical features as [embeddings](https://keras.io/api/layers/core_layers/embedding),
train these embeddings in a simple NN model, and then use the
trained embeddings as inputs to build decision forests model.

This example uses TensorFlow 2.7 or higher,
as well as [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests),
which you can install using the following command:

```python
pip install -U tensorflow_decision_forests
```
"""

"""
## Setup
"""

import math
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf

"""
## Prepare the data

This example uses the
[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29)
provided by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The task is binary classification to determine whether a person makes over 50K a year.

The dataset includes ~300K instances with 41 input features: 7 numerical features
and 34 categorical features.

First we load the data from the UCI Machine Learning Repository into a Pandas DataFrame.
"""

BASE_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income"
CSV_HEADER = [
    l.decode("utf-8").split(":")[0].replace(" ", "_")
    for l in urllib.request.urlopen(f"{BASE_PATH}.names")
    if not l.startswith(b"|")
][2:]
CSV_HEADER.append("income_level")

train_data = pd.read_csv(
    f"{BASE_PATH}.data.gz",
    header=None,
    names=CSV_HEADER,
)
test_data = pd.read_csv(
    f"{BASE_PATH}.test.gz",
    header=None,
    names=CSV_HEADER,
)

"""
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for encoding
the input features with respect to their types.
"""

# Target column name.
TARGET_COLUMN_NAME = "income_level"
# The labels of the target columns.
TARGET_LABELS = [" - 50000.", " 50000+."]
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
CATEGORICAL_FEATURE_NAMES = [
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
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
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "migration_code-change_in_msa",
    "migration_code-change_in_reg",
    "migration_code-move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veteran's_admin",
    "veterans_benefits",
    "year",
]


"""
Now we perform basic data preparation.
"""


def prepare_dataframe(dataframe):
    # Convert the target labels from string to integer.
    dataframe[TARGET_COLUMN_NAME] = dataframe[TARGET_COLUMN_NAME].map(
        TARGET_LABELS.index
    )
    # Cast the categorical features to string.
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        dataframe[feature_name] = dataframe[feature_name].astype(str)


prepare_dataframe(train_data)
prepare_dataframe(test_data)

"""
Now let's show the shapes of the training and test dataframes, and display some instances.
"""

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(train_data.head().T)

"""
## Configure hyperparameters

You can find all the parameters of the Gradient Boosted Tree model in the
[documentation](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel)
"""

# Maximum number of decision trees. The effective number of trained trees can be smaller if early stopping is enabled.
NUM_TREES = 250
# Minimum number of examples in a node.
MIN_EXAMPLES = 6
# Maximum depth of the tree. max_depth=1 means that all trees will be roots.
MAX_DEPTH = 5
# Ratio of the dataset (sampling without replacement) used to train individual trees for the random sampling method.
SUBSAMPLE = 0.65
# Control the sampling of the datasets used to train individual trees.
SAMPLING_METHOD = "RANDOM"
# Ratio of the training dataset used to monitor the training. Require to be >0 if early stopping is enabled.
VALIDATION_RATIO = 0.1

"""
## Implement a training and evaluation procedure

The `run_experiment()` method is responsible loading the train and test datasets,
training a given model, and evaluating the trained model.

Note that when training a Decision Forests model, only one epoch is needed to
read the full dataset. Any extra steps will result in unnecessary slower training.
Therefore, the default `num_epochs=1` is used in the `run_experiment()` method.
"""


def run_experiment(model, train_data, test_data, num_epochs=1, batch_size=None):

    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        train_data, label=TARGET_COLUMN_NAME, weight=WEIGHT_COLUMN_NAME
    )
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data, label=TARGET_COLUMN_NAME, weight=WEIGHT_COLUMN_NAME
    )

    model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


"""
## Experiment 1: Decision Forests with raw features
"""

"""
### Specify model input feature usages

You can attach semantics to each feature to control how it is used by the model.
If not specified, the semantics are inferred from the representation type.
It is recommended to specify the [feature usages](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/FeatureUsage)
explicitly to avoid incorrect inferred semantics is incorrect.
For example, a categorical value identifier (integer) will be be inferred as numerical,
while it is semantically categorical.

For numerical features, you can set the `discretized` parameters to the number
of buckets by which the numerical feature should be discretized.
This makes the training faster but may lead to worse models.
"""


def specify_feature_usages():
    feature_usages = []

    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(
            name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL
        )
        feature_usages.append(feature_usage)

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(
            name=feature_name, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL
        )
        feature_usages.append(feature_usage)

    return feature_usages


"""
### Create a Gradient Boosted Trees model

When compiling a decision forests model, you may only provide extra evaluation metrics.
The loss is specified in the model construction,
and the optimizer is irrelevant to decision forests models.
"""


def create_gbt_model():
    # See all the model parameters in https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel
    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        features=specify_feature_usages(),
        exclude_non_specified_features=True,
        num_trees=NUM_TREES,
        max_depth=MAX_DEPTH,
        min_examples=MIN_EXAMPLES,
        subsample=SUBSAMPLE,
        validation_ratio=VALIDATION_RATIO,
        task=tfdf.keras.Task.CLASSIFICATION,
    )

    gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    return gbt_model


"""
### Train and evaluate the model
"""

gbt_model = create_gbt_model()
run_experiment(gbt_model, train_data, test_data)

"""
### Inspect the model

The `model.summary()` method will display several types of information about
your decision trees model, model type, task, input features, and feature importance.
"""

print(gbt_model.summary())

"""
## Experiment 2: Decision Forests with target encoding

[Target encoding](https://dl.acm.org/doi/10.1145/507533.507538) is a common preprocessing
technique for categorical features that convert them into numerical features.
Using categorical features with high cardinality as-is may lead to overfitting.
Target encoding aims to replace each categorical feature value with one or more
numerical values that represent its co-occurrence with the target labels.

More precisely, given a categorical feature, the binary target encoder in this example
will produce three new numerical features:

1. `positive_frequency`: How many times each feature value occurred with a positive target label.
2. `negative_frequency`: How many times each feature value occurred with a negative target label.
3. `positive_probability`: The probability that the target label is positive,
given the feature value, which is computed as
`positive_frequency / (positive_frequency + negative_frequency + correction)`.
The `correction` term is added in to make the division more stable for rare categorical values.
The default value for `correction` is 1.0.



Note that target encoding is effective with models that cannot automatically
learn dense representations to categorical features, such as decision forests
or kernel methods. If neural network models are used, its recommended to
encode categorical features as embeddings.
"""

"""
### Implement Binary Target Encoder

For simplicity, we assume that the inputs for the `adapt` and `call` methods
are in the expected data types and shapes, so no validation logic is added.

It is recommended to pass the `vocabulary_size` of the categorical feature to the
`BinaryTargetEncoding` constructor. If not specified, it will be computed during
the `adapt()` method execution.
"""


class BinaryTargetEncoding(layers.Layer):
    def __init__(self, vocabulary_size=None, correction=1.0, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.correction = correction

    def adapt(self, data):
        # data is expected to be an integer numpy array to a Tensor shape [num_exmples, 2].
        # This contains feature values for a given feature in the dataset, and target values.

        # Convert the data to a tensor.
        data = tf.convert_to_tensor(data)
        # Separate the feature values and target values
        feature_values = tf.cast(data[:, 0], tf.dtypes.int32)
        target_values = tf.cast(data[:, 1], tf.dtypes.bool)

        # Compute the vocabulary_size of not specified.
        if self.vocabulary_size is None:
            self.vocabulary_size = tf.unique(feature_values).y.shape[0]

        # Filter the data where the target label is positive.
        positive_indices = tf.where(condition=target_values)
        postive_feature_values = tf.gather_nd(
            params=feature_values, indices=positive_indices
        )
        # Compute how many times each feature value occurred with a positive target label.
        positive_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(postive_feature_values.shape[0], 1), dtype=tf.dtypes.float64
            ),
            segment_ids=postive_feature_values,
            num_segments=self.vocabulary_size,
        )

        # Filter the data where the target label is negative.
        negative_indices = tf.where(condition=tf.math.logical_not(target_values))
        negative_feature_values = tf.gather_nd(
            params=feature_values, indices=negative_indices
        )
        # Compute how many times each feature value occurred with a negative target label.
        negative_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(negative_feature_values.shape[0], 1), dtype=tf.dtypes.float64
            ),
            segment_ids=negative_feature_values,
            num_segments=self.vocabulary_size,
        )
        # Compute positive probability for the input feature values.
        positive_probability = positive_frequency / (
            positive_frequency + negative_frequency + self.correction
        )
        # Concatenate the computed statistics for traget_encoding.
        target_encoding_statistics = tf.cast(
            tf.concat(
                [positive_frequency, negative_frequency, positive_probability], axis=1
            ),
            dtype=tf.dtypes.float32,
        )
        self.target_encoding_statistics = tf.constant(target_encoding_statistics)

    def call(self, inputs):
        # inputs is expected to be an integer numpy array to a Tensor shape [num_exmples, 1].
        # This includes the feature values for a given feature in the dataset.

        # Raise an error if the target encoding statistics are not computed.
        if self.target_encoding_statistics == None:
            raise ValueError(
                f"You need to call the adapt method to compute target encoding statistics."
            )

        # Convert the inputs to a tensor.
        inputs = tf.convert_to_tensor(inputs)
        # Cast the inputs int64 a tensor.
        inputs = tf.cast(inputs, tf.dtypes.int64)
        # Lookup target encoding statistics for the input feature values.
        target_encoding_statistics = tf.cast(
            tf.gather_nd(self.target_encoding_statistics, inputs),
            dtype=tf.dtypes.float32,
        )
        return target_encoding_statistics


"""
Let's test the binary target encoder
"""

data = tf.constant(
    [
        [0, 1],
        [2, 0],
        [0, 1],
        [1, 1],
        [1, 1],
        [2, 0],
        [1, 0],
        [0, 1],
        [2, 1],
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 1],
        [1, 1],
        [1, 1],
        [2, 0],
        [1, 0],
        [0, 1],
        [2, 0],
    ]
)

binary_target_encoder = BinaryTargetEncoding()
binary_target_encoder.adapt(data)
print(binary_target_encoder([[0], [1], [2]]))

"""
### Create model inputs
"""


def create_model_inputs():
    inputs = {}

    for feature_name in NUMERIC_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.string
        )

    return inputs


"""
### Implement a feature encoding with target encoding
"""


def create_target_encoder():
    inputs = create_model_inputs()
    target_values = train_data[[TARGET_COLUMN_NAME]].to_numpy()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Get the vocabulary of the categorical feature.
            vocabulary = sorted(
                [str(value) for value in list(train_data[feature_name].unique())]
            )
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_indices = lookup(inputs[feature_name])
            # Prepare the data to adapt the target encoding.
            print("### Adapting target encoding for:", feature_name)
            feature_values = train_data[[feature_name]].to_numpy().astype(str)
            feature_value_indices = lookup(feature_values)
            data = tf.concat([feature_value_indices, target_values], axis=1)
            feature_encoder = BinaryTargetEncoding()
            feature_encoder.adapt(data)
            # Convert the feature value indices to target encoding representations.
            encoded_feature = feature_encoder(tf.expand_dims(value_indices, -1))
        else:
            # Expand the dimensions of the numerical input feature and use it as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # Add the encoded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features = tf.concat(encoded_features, axis=1)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)


"""
### Create a Gradient Boosted Trees model with a preprocessor

In this scenario, we use the target encoding as a preprocessor for the Gradient Boosted Tree model,
and let the model infer semantics of the input features.
"""


def create_gbt_with_preprocessor(preprocessor):

    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        preprocessing=preprocessor,
        num_trees=NUM_TREES,
        max_depth=MAX_DEPTH,
        min_examples=MIN_EXAMPLES,
        subsample=SUBSAMPLE,
        validation_ratio=VALIDATION_RATIO,
        task=tfdf.keras.Task.CLASSIFICATION,
    )

    gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])

    return gbt_model


"""
### Train and evaluate the model
"""

gbt_model = create_gbt_with_preprocessor(create_target_encoder())
run_experiment(gbt_model, train_data, test_data)

"""
## Experiment 3: Decision Forests with trained embeddings

In this scenario, we build an encoder model that codes the categorical
features to embeddings, where the size of the embedding for a given categorical
feature is the square root to the size of its vocabulary.

We train these embeddings in a simple NN model through backpropagation.
After the embedding encoder is trained, we used it as a preprocessor to the
input features of a Gradient Boosted Tree model.

Note that the embeddings and a decision forest model cannot be trained
synergically in one phase, since decision forest models do not train with backpropagation.
Rather, embeddings has to be trained in an initial phase,
and then used as static inputs to the decision forest model.
"""

"""
### Implement feature encoding with embeddings
"""


def create_embedding_encoder(size=None):
    inputs = create_model_inputs()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Get the vocabulary of the categorical feature.
            vocabulary = sorted(
                [str(value) for value in list(train_data[feature_name].unique())]
            )
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_index = lookup(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            vocabulary_size = len(vocabulary)
            embedding_size = int(math.sqrt(vocabulary_size))
            feature_encoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = feature_encoder(value_index)
        else:
            # Expand the dimensions of the numerical input feature and use it as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # Add the encoded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features = layers.concatenate(encoded_features, axis=1)
    # Apply dropout.
    encoded_features = layers.Dropout(rate=0.25)(encoded_features)
    # Perform non-linearity projection.
    encoded_features = layers.Dense(
        units=size if size else encoded_features.shape[-1], activation="gelu"
    )(encoded_features)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)


"""
### Build an NN model to train the embeddings
"""


def create_nn_model(encoder):
    inputs = create_model_inputs()
    embeddings = encoder(inputs)
    output = layers.Dense(units=1, activation="sigmoid")(embeddings)

    nn_model = keras.Model(inputs=inputs, outputs=output)
    nn_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy("accuracy")],
    )
    return nn_model


embedding_encoder = create_embedding_encoder(size=64)
run_experiment(
    create_nn_model(embedding_encoder),
    train_data,
    test_data,
    num_epochs=5,
    batch_size=256,
)

"""
### Train and evaluate a Gradient Boosted Tree model with embeddings
"""

gbt_model = create_gbt_with_preprocessor(embedding_encoder)
run_experiment(gbt_model, train_data, test_data)

"""
## Concluding remarks

TensorFlow Decision Forests provide powerful models, especially with structured data.
In our experiments, the Gradient Boosted Tree model achieved 95.79% test accuracy.
When using the target encoding with categorical feature, the same model achieved 95.81% test accuracy.
When pretraining embeddings to be used as inputs to the Gradient Boosted Tree model,
we achieved 95.82% test accuracy.

Decision Forests can be used with Neural Networks, either by
1) using Neural Networks to learn useful representation of the input data,
and then using Decision Forests for the supervised learning task, or by
2) creating an ensemble of both Decision Forests and Neural Network models.

Note that TensorFlow Decision Forests does not (yet) support hardware accelerators.
All training and inference is done on the CPU.
Besides, Decision Forests require a finite dataset that fits in memory
for their training procedures. However, there are diminishing returns
for increasing the size of the dataset, and Decision Forests algorithms
arguably need fewer examples for convergence than large Neural Network models.
"""
