"""
Title: Advanced techniques for structured data
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2020/12/31
Last modified: 2020/12/31
Description: Exploring advanced modeling techniques for structured data.
"""

"""
## Introduction

This example demonstrates how to do structured data classification using advanced techniques like:
1. [Wide & Deep](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) models
2. [Deep & Cross](https://arxiv.org/abs/1708.05123) models

The example covers using [tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column)
to handle input features, and encoding categorical features using one-hot encoding and
embedding respesentation.


Note that this example should be run with TensorFlow 2.3 or higher.
"""

"""
## The dataset

This example uses the [covertype](https://archive.ics.uci.edu/ml/datasets/covertype) from UCI
Machine Learning Repository. The task is to Predict forest cover type from cartographic variables.
The dataset includes 506,011 intances with 12 input features, 10 of which numerical, and the other 2 are
categorical. The target class has 7 labels.
"""

"""
## Setup
"""

import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import math
from sklearn.model_selection import train_test_split

"""
## Prepare the data

First, download the data from the UCI Machine Learning Repository to a pandas Dataframe:
"""

data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
)
raw_data = pd.read_csv(data_url, header=None)
print(f"Dataset shape: {raw_data.shape}")
raw_data.head()

"""
The two catrogircal features in the dataset, as well as the target feature are binary encoded.
We will this dataset repesentation to the typical repesentation, where each categorical feature is
repesented as on column.
"""

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
data["Cover_Type"] = data["Cover_Type"] - 1

print(f"Dataset shape: {data.shape}")
data.head().T

"""
You can see now the dataframe has 13 columns per sample (12 features, plus the target label).

Let's split the data into training and testing splits, with 85% and 15% of the instance, respectively.
"""

train_data, test_data = train_test_split(
    data, train_size=0.85, stratify=data["Cover_Type"]
)

print(f"Train split size: {len(train_data.index)}")
print(f"Test split size: {len(test_data.index)}")

"""
Now we store the train and test data splits to CSV files.
"""

train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False)
test_data.to_csv(test_data_file, index=False)

"""
## Define dataset metadata
"""

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

"""
Second, let's download the train and test data files

We can now load the data in Pandas Dataframes. The training data split includes 431,010 samples,
while the test data split includes 75,001 samples, with 13 columns per sample (12 features, plus the
target label):
"""

"""
## Experiment setup

We create an input function to read and parse the file, and convert features and labels into a
[`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets) for training or evaluation.
"""


def get_dataset_from_csv(csv_file_path, batch_size, num_epochs=None, shuffle=False):

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=num_epochs,
        header=True,
        shuffle=shuffle,
    )

    return dataset.cache()


"""
Here we configure the parameters and implement the procedure for running a training and evaluation
experiment given a model.
"""

train_size = 493860
learning_rate = 0.001
dropout_rate = 0.1
batch_size = 265
num_epochs = 50

train_steps_per_epoch = train_size // batch_size
hidden_units = [32, 32]


def run_experiment(model):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)

    test_dataset = get_dataset_from_csv(test_data_file, batch_size, num_epochs=1)

    print("Start training the model...")
    history = model.fit(
        train_dataset, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch
    )
    print("Model training finished")

    _, accuracy = model.evaluate(test_dataset, verbose=0)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


"""
## Create model inputs

We define the inputs for our models as a dictionary, where the key is the feature name, and the value
is a `keras.layers.Input` tensor with the corresponsing feature shape and data type.
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
## Create feature columns

We create two representations of our input features: sparse and dense:
1. In the **sparse** representation, the categorical features are encoded with one-hot encoding,
using [tf.feature_columns.indicator_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column).
This representation can be useful for the model to *memorize* particular feature values to make certain
predictions.
2. In the **dense** representation, the categorical features are encoded with low-dimensional embeddings,
using the [tf.feature_column.embedding_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column).
This representation helps the model to *generalize* well to unseen feature combinations.

These feature columns are converted to a Keras dense layer using the
[tf.keras.layers.DenseFeatures](https://www.tensorflow.org/api_docs/python/tf/keras/layers/DenseFeatures).
"""


def create_sparse_feature_columns():
    feature_columns = []
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            # Create a numeric feature column.
            feature_column = tf.feature_column.numeric_column(feature_name)
        else:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a categorical feature column.
            feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
                feature_name, vocabulary_list=vocabulary
            )
            # Encode the categorical feature with one-hot representation.
            feature_column = tf.feature_column.indicator_column(feature_column)
        feature_columns.append(feature_column)

    # Create a crossed feature to capture the interaction between Soil_Type and Wilderness_Area.
    num_values = int(
        len(CATEGORICAL_FEATURES_WITH_VOCABULARY["Soil_Type"])
        * len(CATEGORICAL_FEATURES_WITH_VOCABULARY["Wilderness_Area"])
    )
    soil_type_X_wilderness_area = tf.feature_column.crossed_column(
        ["Soil_Type", "Wilderness_Area"], hash_bucket_size=num_values
    )
    # Encode the crossed feature with one-hot representation.
    soil_type_X_wilderness_area_onehot = tf.feature_column.indicator_column(
        soil_type_X_wilderness_area
    )
    feature_columns.append(soil_type_X_wilderness_area_onehot)

    return feature_columns


def create_dense_feature_columns():
    feature_columns = []
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            # Create a numeric feature column.
            feature_column = tf.feature_column.numeric_column(feature_name)
        else:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a categorical feature column.
            feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
                feature_name, vocabulary_list=vocabulary
            )
            # Set the embedding dimensions to the square root of the vocabulary size.
            embedding_dims = int(math.sqrt(len(vocabulary)))
            # Encode the categorical feature with embedding representation.
            feature_column = tf.feature_column.embedding_column(
                feature_column, dimension=embedding_dims
            )
        feature_columns.append(feature_column)

    # Create a crossed feature to capture the interaction between Soil_Type and Wilderness_Area.
    num_values = int(
        len(CATEGORICAL_FEATURES_WITH_VOCABULARY["Soil_Type"])
        * len(CATEGORICAL_FEATURES_WITH_VOCABULARY["Wilderness_Area"])
    )
    soil_type_X_wilderness_area = tf.feature_column.crossed_column(
        ["Soil_Type", "Wilderness_Area"], hash_bucket_size=num_values
    )
    # Encode the crossed feature with embedding representation.
    soil_type_X_wilderness_area_embedding = tf.feature_column.embedding_column(
        soil_type_X_wilderness_area, dimension=int(math.sqrt(num_values))
    )
    feature_columns.append(soil_type_X_wilderness_area_embedding)

    return feature_columns


"""
## Experiment 1: Baseline model

In first experiment, we create a multi-layer feed-forward network, where the categorical features are
one-hot encoded.
"""


def create_baseline_model():
    inputs = create_model_inputs()
    sparse_feature_columns = create_sparse_feature_columns()
    features = layers.DenseFeatures(sparse_feature_columns)(inputs)

    for units in hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


baseline_model = create_baseline_model()
keras.utils.plot_model(baseline_model, show_shapes=True)

run_experiment(baseline_model)

"""
The baseline linear model acheives ~76.4% test accuracy.
"""

"""
## Experiment 2: wide & deep model

In the second experiment, we create a wide and deep model. The wide part of the model a linear model,
while the deep part of the model is a multi-layer feed-forward network.

In the wide part of the model, we use the sparse representation of the input features, while we use the
dense repesentation of the input features for the deep part of the model.

Note that every input features contributes to both parts of the model with different representations.
"""


def create_wide_and_deep_model():

    inputs = create_model_inputs()

    sparse_feature_columns = create_sparse_feature_columns()
    wide = layers.DenseFeatures(sparse_feature_columns)(inputs)
    wide = layers.BatchNormalization()(wide)

    dense_feature_columns = create_dense_feature_columns()
    deep = layers.DenseFeatures(dense_feature_columns)(inputs)
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

keras.utils.plot_model(wide_and_deep_model, show_shapes=True)

run_experiment(wide_and_deep_model)

"""
The wide and deep model acheives ~79.8% test accuracy.
"""

"""
## Experiment 3: deep & cross model

In the third experiment, we create a deep and cross model. The deep part of this model is the same as
the deep part created in the previous experiment. The key idea of the cross part is to apply explicit feature
crossing in an efficient way, where the degree of cross features grows with layer depth.
"""


def create_deep_and_cross_model():

    inputs = create_model_inputs()
    dense_feature_columns = create_dense_feature_columns()

    x0 = layers.DenseFeatures(dense_feature_columns)(inputs)

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
    model = keras.Model(inputs=inputs, outputs=outputs, name="dnc-classifier")

    return model


deep_and_cross_model = create_deep_and_cross_model()
keras.utils.plot_model(deep_and_cross_model, show_shapes=True)

run_experiment(deep_and_cross_model)

"""
The deep and cross model acheives ~82.7% test accuracy.
"""

"""
## Conclusion

You can use the `tf.feature_column` API with your `Keras` models to easily handle input categorical
features with different encoding mechanisms, including one-hot encoding and low-dimensional embedding.
In addition, model architectures, like wide, deep, and cross networks, have different advantages,
with respect to the various dataset properties. You can explore using them independently or combining
them to achieve the best result for your dataset.
"""
