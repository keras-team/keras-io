"""
Title: Classification with TensorFlow Decision Forests
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2022/01/31
Last modified: 2022/01/31
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

This example uses Gradient Boosted Trees model in binary classification of
structured data, and covers the following scenarios:

1. Build a decision forests model by specifying the input feature usage.
2. Implement a custom *Binary Target encoder* as a [Keras Preprocessing layer](https://keras.io/api/layers/preprocessing_layers/)
to encode the categorical features with respect to their target value co-occurrences,
and then use the encoded features to build a decision forests model.
3. Encode the categorical features as [embddings](https://keras.io/api/layers/core_layers/embedding),
train these embeddings in a simple linear model, and then use the
trained embeddings as inputs to build decision forests model.

This example uses TensorFlow 2.7 or higher,
as well as [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests),
which you can install using the following command:

``` python
pip install -U tensorflow_decision_forests
```
"""

"""
## Setup
"""

import math
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

train_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz",
    header=None,
    names=CSV_HEADER,
)

test_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz",
    header=None,
    names=CSV_HEADER,
)

"""
We convert the target column from string to integer.
"""

train_data["income_level"] = train_data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)
test_data["income_level"] = test_data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)

"""
Now let's show the shapes of the training and test dataframes, and display some instances.
"""

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
train_data.head().T

"""
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for encoding
the input features with respect to their types.
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
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    feature_name: sorted(
        [str(value) for value in list(train_data[feature_name].unique())]
    )
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

"""
## Configure hyperparameters

You can find all the parameters of the Gradient Boosted Tree model in the
[documentation](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel)
"""

GROWING_STRATEGY = "BEST_FIRST_GLOBAL"
NUM_TREES = 250
MIN_EXAMPLES = 6
MAX_DEPTH = 5
SUBSAMPLE = 0.65
SAMPLING_METHOD = "RANDOM"
VALIDATION_RATIO = 0.1

"""
## Implement a training and evaluation procedure
"""


def fix_datatypes(features):
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            if features[feature_name].dtype != tf.dtypes.string:
                # Convert categorical feature values to string.
                features[feature_name] = tf.strings.as_string(features[feature_name])
    return features


def run_experiment(model, train_data, test_data, num_epochs=1, batch_size=None):

    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        train_data, label=TARGET_FEATURE_NAME, weight=WEIGHT_COLUMN_NAME
    ).map(lambda features, target, weight: (fix_datatypes(features), target, weight))
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data, label=TARGET_FEATURE_NAME, weight=WEIGHT_COLUMN_NAME
    ).map(lambda features, target, weight: (fix_datatypes(features), target, weight))

    history = model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history


"""
## Create model inputs
"""


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


"""
## Experiment 1: Decision Forests with raw features
"""

"""
### Specify model input feature usages

You can attache semantics to each feature to control how it is used by the model.
If not specified, the semantics is inferred from the representation type.
It is recommended to specify the [feature usages](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/FeatureUsage)
explicitly to avoid incorrect inferred semantics is incorrect.
For example, a categorical value identifier (integer) will be be inferred as numerical,
while it is semantically categorical.

For numerical features, you can set the `discretized` parameters to the number
of buckets by which the numerical feature should be discretized.
This makes the training faster but may lead to worse models.
"""


def specify_feature_usages(inputs):
    feature_usages = []

    for feature_name in inputs:
        if inputs[feature_name].dtype == tf.dtypes.float32:
            feature_usage = tfdf.keras.FeatureUsage(
                name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL
            )
        else:
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
    feature_usages = specify_feature_usages(create_model_inputs())
    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        features=feature_usages,
        exclude_non_specified_features=True,
        growing_strategy=GROWING_STRATEGY,
        num_trees=NUM_TREES,
        max_depth=MAX_DEPTH,
        min_examples=MIN_EXAMPLES,
        subsample=SUBSAMPLE,
        validation_ratio=VALIDATION_RATIO,
        task=tfdf.keras.Task.CLASSIFICATION,
        loss="DEFAULT",
    )

    gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    return gbt_model


"""
### Train and evaluate the model
"""

gbt_model = create_gbt_model()
history = run_experiment(gbt_model, train_data, test_data)

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
numerical values that repesent its co-occerence with the target labels.

More preciely, given a categorical feature, the binary target encoder in this example
will produce three new numerical features:

1. `positive_frequency`: How many times each feature value occured with a positive target label.
2. `negative_frequency`: How many times each feature value occured with a negative target label.
3. `positive_probability`: The probability that the target label is possitive,
given the feature value, which is computed as
`positive_frequency / (positive_frequency + negative_frequency)`.


Note that target encoding is effective with models that cannot automatically
learn dense repesentations to categorical features, such as decision forests
or kernel methods. If neural network models are used, its recommended to
encode categorical features as embeddings.
"""

"""
### Implement Binary Target Encoder
"""

from keras.engine import base_preprocessing_layer


class BinaryTargetEncoding(base_preprocessing_layer.PreprocessingLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell("BinaryTargetEncoding").set(
            True
        )

    def adapt(self, data):
        # data is expected to be a integer numpy array ot a Tensor shape [num_exmples, 2].
        # This contain feature values for a given feature in the dataset, and target values.

        self.compile()
        self.reset_state()

        # Validate that the data has the expected rank.
        if data.shape.rank != 2:
            raise ValueError(f"Data is expected to be of rank 2, got {data.shape.rank}")
        # Validate the dimensions of the second axis.
        if data.shape[-1] != 2:
            raise ValueError(
                f"The second axis of data is expected to have 2 dimensions, got {data.shape[-1]}"
            )
        # Validate that the data has the expected shape.
        if data.dtype not in [tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int64]:
            raise ValueError(
                f"Data is expected to be a Tensor of type integer, got {data.dtype}"
            )
        # Separate the feature values and target values
        data = self._standardize_inputs(data)
        feature_values = data[:, 0]
        target_values = data[:, 1]
        # Validate that the target has only 0 and 1 values.
        unique_target_values = tf.sort(tf.unique(target_values).y)
        if not tf.math.reduce_all(
            tf.math.equal(
                unique_target_values,
                tf.constant([0, 1], dtype=unique_target_values.dtype),
            )
        ):
            raise ValueError(
                f"Target values should be either 0 or 1, got {unique_target_values.numpy()}"
            )

        target_values = tf.cast(target_values, tf.dtypes.bool)

        print("Target encoding: Computing unique feature values...")
        # Get feature vocabulary.
        unique_feature_values = tf.sort(tf.unique(feature_values).y)

        print(
            "Target encoding: Computing frequencies for feature values with positive tatgets..."
        )
        # Filter the data where target label is positive.
        positive_indices = tf.where(condition=target_values)
        postive_feature_values = tf.gather_nd(
            params=feature_values, indices=positive_indices
        )
        # Compute how many time each feature value occured with a positive target label.
        positive_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(postive_feature_values.shape[0], 1), dtype=tf.dtypes.int32
            ),
            segment_ids=postive_feature_values,
            num_segments=unique_feature_values.shape[0],
        )

        print(
            "Target encoding: Computing frequencies for feature values with negative tatgets..."
        )
        # Filter the data where target label is negative.
        negative_indices = tf.where(condition=tf.math.logical_not(target_values))
        negative_feature_values = tf.gather_nd(
            params=feature_values, indices=negative_indices
        )
        # Compute how many time each feature value occured with a negative target label.
        negative_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(negative_feature_values.shape[0], 1), dtype=tf.dtypes.int32
            ),
            segment_ids=negative_feature_values,
            num_segments=unique_feature_values.shape[0],
        )

        print("Target encoding: Storing target encoding statistics...")
        self.positive_frequency_lookup = tf.constant(positive_frequency)
        self.negative_frequency_lookup = tf.constant(negative_frequency)

        self._is_adapted = True

    def reset_state(self):
        if self.built:
            self.positive_frequency_lookup = None
            self.negative_frequency_lookup = None

    def call(self, inputs):
        # data is expected to be a integer numpy array ot a Tensor shape [num_exmples, 1].
        # This include the feature values for a given feature in the dataset.

        # Raise an error of the target encoding statistics are not computed.
        if (
            self.positive_frequency_lookup == None
            or self.negative_frequency_lookup == None
        ):
            raise ValueError(
                f"You need to call the adapt method to compute target encoding statistics."
            )

        inputs = self._standardize_inputs(inputs)

        # Fix and validate inputs shape.
        if inputs.shape.rank == 1:
            inputs = tf.expand_dims(inputs, axis=-1)
        elif inputs.shape.rank > 2:
            raise ValueError(
                f"inputs is expected to be of rank 2, got {data.shape.rank}"
            )
        # Validate the dimensions of the second axis.
        if inputs.shape[-1] != 1:
            raise ValueError(
                f"The second axis of inputs is expected to have 1 dimension, got {data.shape[-1]}"
            )
        # Lookup positive frequencies for the input feature values.
        positive_fequency = tf.cast(
            tf.gather_nd(self.positive_frequency_lookup, inputs),
            dtype=tf.dtypes.float32,
        )
        # Lookup negative frequencies for the input feature values.
        negative_fequency = tf.cast(
            tf.gather_nd(self.negative_frequency_lookup, inputs),
            dtype=tf.dtypes.float32,
        )
        # Compute positive probability for the input feature values.
        positive_probability = positive_fequency / (
            positive_fequency + negative_fequency
        )
        # Concatenate and return the looked-up statistics.
        return layers.concatenate(
            [positive_fequency, negative_fequency, positive_probability], axis=1
        )

    def compute_output_shape(self, input_shape):
        # The output shape is expected to be [num_examplesm 3].
        return tf.TensorShape([input_shape[0], 3])

    def compute_output_signature(self, input_spec):
        output_shape = self.compute_output_shape(input_spec.shape.as_list())
        return tf.TensorSpec(shape=output_shape, dtype=tf.int64)

    def _standardize_inputs(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if inputs.dtype != tf.dtypes.int64:
            inputs = tf.cast(inputs, tf.dtypes.int64)
        return inputs


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
print(binary_target_encoder([0, 1, 2]))

"""
### Implement a feature encoding with target encoding
"""


def create_target_encoder():
    inputs = create_model_inputs()
    target_values = train_data[[TARGET_FEATURE_NAME]].to_numpy()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
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
            data = layers.concatenate([feature_value_indices, target_values], axis=1)
            feature_encoder = BinaryTargetEncoding()
            feature_encoder.adapt(data)
            # Convert the feature value indices to target encoding representations.
            encoded_feature = feature_encoder(value_indices)
        else:
            # Expand the dimensions of the numerical input feature and use it as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # Add the encodded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features = layers.concatenate(encoded_features, axis=1)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)


"""
### Create a Gradient Boosted Trees model with a preprocessor

In this scenario, we use the target encoding as a preprocessor for the Gradiente Boosted Tree model,
and let the model infer semantics of the input features.
"""


def create_gbt_with_preprocessor(preprocessor):

    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        preprocessing=preprocessor,
        growing_strategy=GROWING_STRATEGY,
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
history = run_experiment(gbt_model, train_data, test_data)

"""
## Experiment 3: Decision Forests with trained embeddings

In this scenario, we build an encoder model that codes the categorical
features to embeddings, where the size of the embedding for a given categorical
feature is the square root to the size of its vocabulary.

We train these embeddings in a simple linear model through backprobagation.
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


def create_embedding_encoder():
    inputs = create_model_inputs()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
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
        # Add the encodded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features = layers.concatenate(encoded_features, axis=1)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)


"""
### Build a linear model to train the embeddings
"""


def create_linear_model(encoder):
    inputs = create_model_inputs()
    linear_output = layers.Dense(units=1, activation="sigmoid")(encoder(inputs))

    linear_model = keras.Model(inputs=inputs, outputs=linear_output)
    linear_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy("accuracy")],
    )
    return linear_model


embedding_encoder = create_embedding_encoder()
history = run_experiment(
    create_linear_model(embedding_encoder),
    train_data,
    test_data,
    num_epochs=3,
    batch_size=256,
)

"""
### Train and evaluate a Gradient Boosted Tree model with embeddings
"""

gbt_model = create_gbt_with_preprocessor(embedding_encoder)
history = run_experiment(gbt_model, train_data, test_data)

"""
## Concluding remarks

TensorFlow Decision Forests provide powerful models, especially with structured data.
Decision Forests can be used with Neural Networks, either by
1) using Neural Networks to learn useful repesentation of the input data,
and then using Decision Forests for the supervised learning task, or by
2) creating an ensemble of both Decision Forests and Neural Network models.

Note that TensorFlow Decision Forests does not (yet) support hardware accelerators.
All training and inference is done on the CPU.
Besides, Decision Forests require a finite dataset that fits in memory
for their training procedures. However, there are diminishing returns
for increasing the size of the dataset, and Decision Forests algorithms
arguably need fewer examples for convergence than large Neural Network models.
"""
