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
or feature engineering. We might have a ML training pipeline consisting of
different libraries for example Pandas for reading data and also feature engineeering,
sklearn for encoding features for example OneHot encoding and Normalization. The
estimator might be an sklearn classifier, xgboost or it can for example be a Keras model.
In the latter case, we would end up with artifacts for feature engineering and encoding
and different artifacts for the saved model. The pipeline is also disconnected and
an extra step is needed to feed encoded data to the Keras model. For this step the data
can be mapped from a dataframe to something like tf.data.Datasets type or numpy array
before feeding it to a Keras model.

In this example we will implement feature preprocessing natively with
Keras by implementing a custom Keras layer for all feature preprocessing steps. This layer
utilizes Keras preprocessing layers. For constructing our model we will use the Functional API.
"""

"""
## Setup
"""
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

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

NUMERICAL_FEATURES = [
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

FEATURES_TO_DISCRETIZE = ["age"]

FEATURES_TO_EMBED = ["native_country"]

# split data frame into different feature groups
categorical_features = {
    feature: data_frame[[feature]].values for feature in CATEGORICAL_FEATURES
}
numerical_features = {
    feature: data_frame[[feature]].values for feature in NUMERICAL_FEATURES
}
discretize_features = {
    feature: data_frame[[feature]].values for feature in FEATURES_TO_DISCRETIZE
}
embedding_features = {
    feature: data_frame[[feature]].values for feature in FEATURES_TO_EMBED
}

data_dict = {
    **numerical_features,
    **categorical_features,
    **discretize_features,
    **embedding_features,
}

categorical_features_input = {
    feature: tf.keras.Input(shape=(1,), dtype=tf.string, name=feature)
    for feature in CATEGORICAL_FEATURES
}

numerical_features_input = {
    feature: tf.keras.Input(shape=(1,), dtype=tf.float32, name=feature)
    for feature in NUMERICAL_FEATURES
}

discretize_features_input = {
    feature: tf.keras.Input(shape=(1,), dtype=tf.float32, name=feature)
    for feature in FEATURES_TO_DISCRETIZE
}

embedding_features_input = {
    feature: tf.keras.Input(shape=(1,), dtype=tf.string, name=feature)
    for feature in FEATURES_TO_EMBED
}

input_dict = {
    **numerical_features_input,
    **categorical_features_input,
    **discretize_features_input,
    **embedding_features_input,
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
        self.normalizer = {
            feature: layers.Normalization() for feature in NUMERICAL_FEATURES
        }
        self.categoric_string_lookup = {
            feature: layers.StringLookup() for feature in CATEGORICAL_FEATURES
        }
        self.integer_lookup = {
            feature: layers.IntegerLookup(output_mode="binary")
            for feature in CATEGORICAL_FEATURES
        }
        self.discretizer = {
            feature: layers.Discretization(num_bins=10)
            for feature in FEATURES_TO_DISCRETIZE
        }
        self.embedding_string_lookup = {
            feature: layers.StringLookup() for feature in FEATURES_TO_EMBED
        }

    def adapt(self, data):
        """Update layer states"""
        for feature, preprocessor in self.normalizer.items():
            preprocessor.adapt(data[feature])

        for feature, preprocessor in self.discretizer.items():
            preprocessor.adapt(data[feature])

        for feature, preprocessor in self.categoric_string_lookup.items():
            preprocessor.adapt(data[feature])
            self.integer_lookup[feature].adapt(preprocessor(data[feature]))

        for feature, preprocessor in self.embedding_string_lookup.items():
            preprocessor.adapt(data[feature])

        vocabulary_size = preprocessor.vocabulary_size()
        embedding_size = int(tf.math.sqrt(float(vocabulary_size)))

        # create sequential model for the embedding feature
        self.embedding = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
                self.embedding_string_lookup[FEATURES_TO_EMBED[0]],
                layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_size),
                layers.Reshape((-1,)),
            ]
        )

    def call(self, inputs={}):
        """Apply adapted layers"""
        numerical_features = apply_preprocessor(inputs, self.normalizer)

        categorical_features = {
            feature: self.categoric_string_lookup[feature](inputs[feature])
            for feature in CATEGORICAL_FEATURES
        }
        categorical_features = apply_preprocessor(
            categorical_features, self.integer_lookup
        )

        embedding_features = self.embedding(inputs["native_country"])

        discretize_features = apply_preprocessor(inputs, self.discretizer)
        discretize_features = tf.cast(discretize_features, dtype=tf.float32)

        return tf.keras.layers.concatenate(
            [
                numerical_features,
                categorical_features,
                discretize_features,
                embedding_features,
            ]
        )


def apply_preprocessor(inputs, preprocessor_dict):
    return tf.keras.layers.concatenate(
        [
            preprocessor(inputs[feature])
            for feature, preprocessor in preprocessor_dict.items()
        ]
    )


"""
## Compile and fit model

We now have a feature preprocessing layer. Lets put everything together and fit the model:
"""

# update preprocessing layer states
preprocessing_layer = FeaturePreprocessor()
preprocessing_layer.adapt(data_dict)

# rest of model
x = preprocessing_layer(input_dict)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=input_dict, outputs=outputs)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.fit(data_dict, labels_binary.values, epochs=10, validation_split=0.25)
