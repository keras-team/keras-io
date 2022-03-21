"""
Title: Feature Preprocessing with a custom Feature Preprocessor layer.
Author: [Fernando Nieuwveldt](https://www.linkedin.com/in/fernandonieuwveldt/)
Date created: 2022/03/21
Last modified: 2022/03/21
Description: Feature Preprocessing with a custom Feature Preprocessor layer.
"""
"""
# Feature Preprocessing with a custom Feature Preprocessor layer for an end to end training pipeline.
"""

"""
Often for structured data problems we use multiple libraries for preprocessing
or feature engineering. We might have a full ML training pipeline consisting of
different libraries for example Pandas for reading data and also feature engineeering,
sklearn for encoding features for example OneHot encoding and Normalization. The
estimator might be an sklearn classifier, xgboost or it can for example be a Keras model.
In the latter case, we would end up with artifacts for feature engineering and encoding
and different artifacts for the saved model. The pipeline is also disconnected and
an extra step is needed to feed encoded data to the Keras model. For this step the data
can be mapped from a dataframe to something like tf.data.Datasets type or numpy array
before feeding it to a Keras model.

In this example we will implement feature preprocessing natively with
Keras by implementing a custom Keras layer for all feature preprocessing steps.This layer
utilizes Keras preprocessing layers. For constructing our model we will use the Functional API.
"""

"""
## Setup
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    StringLookup,
    IntegerLookup,
    Normalization,
    Embedding,
    Discretization,
    Reshape,
)
import pandas as pd

"""
## Load data

For this example we wil use the income dataset. This dataset contains a mixture of
numerical and categorical features some with high cardinality.
"""

CSV_HEADER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

data_frame = pd.read_csv(data_url, header=None, names=CSV_HEADER)
labels = data_frame.pop("income_bracket")
labels_binary = 1.0 * (labels == " >50K")

print(f"Train dataset shape: {data_frame.shape}")

"""
## Split features into different feature groups
"""

NUMERIC_FEATURES = [
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "fnlwgt",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "marital_status",
    "relationship",
    "gender",
    "education",
    "occupation",
]

DISCRETIZE_FEATURES = ["age"]

EMBEDDING_FEATURES = ["native_country"]

# split data frame into different feature groups
data_dict = {
    "numerical_features": data_frame[NUMERIC_FEATURES].values,
    "categorical_features": data_frame[CATEGORICAL_FEATURES].values,
    "discretize_features": data_frame[DISCRETIZE_FEATURES].values,
    "embedding_features": data_frame[EMBEDDING_FEATURES].values,
}

"""
## Create Feature Preprocessing layer

In this section we will implement a custom Keras layer for preprocessing. We have an adapt
method that will update our preprocessing layer's states and apply them in the forward pass. We have
one feature that has high cardinality and we will use embedding for this feature.

"""


class FeaturePreprocessor(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(FeaturePreprocessor, self).__init__(*args, **kwargs)
        # set preprocessing layers
        self.normalizer = Normalization()
        self.categoric_string_lookup = StringLookup()
        self.integer_lookup = IntegerLookup(output_mode="binary")
        self.discretizer = Discretization(num_bins=10)
        self.embedding_string_lookup = StringLookup()

    def adapt(self, data={}):
        """Update layer states"""
        self.normalizer.adapt(data["numerical_features"])

        self.categoric_string_lookup.adapt(data["categorical_features"])

        # apply integer lookup on string lookup layer output
        self.integer_lookup.adapt(
            self.categoric_string_lookup(data["categorical_features"])
        )

        self.discretizer.adapt(data["discretize_features"])

        self.embedding_string_lookup.adapt(data["embedding_features"])

        # create sequential model for string and integer lookup layers
        self.categoric_encoding = tf.keras.models.Sequential(
            [self.categoric_string_lookup, self.integer_lookup]
        )

        vocabulary_size = self.embedding_string_lookup.vocabulary_size()
        embedding_size = int(tf.math.sqrt(vocabulary_size))
        # create sequential model for embedding feature
        self.embedding = tf.keras.models.Sequential(
            [
                self.embedding_string_lookup,
                Embedding(input_dim=vocabulary_size, output_dim=embedding_size),
                Reshape((-1,)),
            ]
        )

    def call(self, inputs={}):
        """Apply adapted layers"""
        numerical_features = self.normalizer(inputs["numerical_features"])

        categorical_features = self.categoric_encoding(inputs["categorical_features"])

        discretize_features = tf.cast(
            self.discretizer(inputs["discretize_features"]), dtype=tf.float32
        )

        embedding_features = self.embedding(inputs["embedding_features"])

        return tf.keras.layers.concatenate(
            [
                numerical_features,
                categorical_features,
                discretize_features,
                embedding_features,
            ]
        )


"""
## Compile and fit model

We now have a feature preprocessing layer. Lets put everything together and fit the model:
"""

inputs = {
    "numerical_features": tf.keras.Input(
        shape=(len(NUMERIC_FEATURES),), dtype=tf.float32, name="numeric_layer"
    ),
    "categorical_features": tf.keras.Input(
        shape=(len(CATEGORICAL_FEATURES),), dtype=tf.string, name="categoric_layer"
    ),
    "discretize_features": tf.keras.Input(
        shape=(len(DISCRETIZE_FEATURES),), dtype=tf.float32, name="discretize_layer"
    ),
    "embedding_features": tf.keras.Input(
        shape=(len(EMBEDDING_FEATURES),), dtype=tf.string, name="embedding_layer"
    ),
}

# update preprocessing layer states
preprocessing_layer = FeaturePreprocessor()
preprocessing_layer.adapt(data_dict)

# rest of model
x = preprocessing_layer(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.fit(data_dict, labels_binary.values, epochs=10, validation_split=0.25)
