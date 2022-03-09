"""
Title: Feature Engineering using Keras Functional API and custom Feature layer
Author: [Fernando Nieuwveldt](https://www.linkedin.com/in/fernandonieuwveldt/)
Date created: 2022/02/26
Last modified: 2022/02/26
Description: Feature Engineering as part of your network architecture.
"""
"""
# Feature Engineering using Keras custom Feature Layer for an end to end training pipeline.
"""

"""
Often for structured data problems we end up using multiple libraries for preprocessing
or feature engineering. We can go as far as having a full ML training pipeline using
different libraries for example Pandas for reading data and also feature engineeering,
sklearn for encoding features for example OneHot encoding and Normalization. The
estimator might be an sklearn classifier, xgboost or it can for example be a Keras model.
In the latter case, we would end up with artifacts for feature engineering and encoding
and different artifacts for the saved model. The pipeline is also disconnected and
an extra step is needed to feed encoded data to the Keras model. For this step the data
can be mapped from a dataframe to something like tf.data.Datasets type or numpy array
before feeding it to a Keras model.

In this example we will consider implementing a training pipeline natively with
Keras/Tensorflow. From loading data with tf.data and implementing a custom Keras layer for all feature engineering steps.
These engineered features will be stateless. We will end up with a training pipeline where
feature engineering will be part of the network architecture and can be persisted and loaded
for inference as standalone.

Steps we will follow:
* Load data with tf.data
* Create Input layer
* Create feature layer by subclassing Layer class
* Train model

For constructing our model we will use the Functional API.
"""

"""
## Setup
"""

import tensorflow as tf
from keras.utils.vis_utils import plot_model

"""
## Load data with tf.data

We will use the heart disease dataset. Lets read in the data:
"""

heart_dir = tf.keras.utils.get_file(
    "heart.csv",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/heart.csv",
)

dataset = tf.data.experimental.make_csv_dataset(
    heart_dir, batch_size=64, label_name="target", num_epochs=10
)

binary_features = ["sex", "fbs", "exang"]
numeric_features = [
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "slope",
    "cp",
    "restecg",
    "ca",
]
categoric_features = ["thal"]

"""
## Create Input Layer

We first create a dictionary with feature name and data type mapper and than create a dict of Input objects.
"""

dtype_mapper = {
    "age": tf.float32,
    "sex": tf.float32,
    "cp": tf.float32,
    "trestbps": tf.float32,
    "chol": tf.float32,
    "fbs": tf.float32,
    "restecg": tf.float32,
    "thalach": tf.float32,
    "exang": tf.float32,
    "oldpeak": tf.float32,
    "slope": tf.float32,
    "ca": tf.float32,
    "thal": tf.string,
}


def create_inputs(data_type_mapper):
    """Create model inputs

    Args:
        data_type_mapper (dict): Dictionary with feature as key and dtype as value
                                 For example {'age': tf.float32, ...}
    Returns:
        (dict): Keras inputs for each feature
    """
    return {
        feature: tf.keras.Input(shape=(1,), name=feature, dtype=dtype)
        for feature, dtype in data_type_mapper.items()
    }


feature_layer_inputs = create_inputs(dtype_mapper)


"""
## Engineered Features: Custom Keras layer

For feature engineering we will be using a custom Keras layer. When all the features are defined we will concatenate all these into one layer.
"""


class FeatureLayer(tf.keras.layers.Layer):
    """Custom Layer for Feature engineering steps
    """

    def __init__(self, *args, **kwargs):
        super(FeatureLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        age_and_gender = tf.cast(
            tf.math.logical_and(inputs["age"] > 50, inputs["sex"] == 1),
            dtype=tf.float32,
        )

        thal_fixed_category = tf.cast(inputs["thal"] == "fixed", dtype=tf.float32)
        thal_reversible_category = tf.cast(
            inputs["thal"] == "reversible", dtype=tf.float32
        )
        thal_normal_category = tf.cast(inputs["thal"] == "normal", dtype=tf.float32)

        trest_chol_ratio = inputs["trestbps"] / inputs["chol"]
        trest_cross_thalach = inputs["trestbps"] * inputs["thalach"]

        # concat all newly created features into one layer
        feature_list = [
            thal_fixed_category,
            thal_reversible_category,
            thal_normal_category,
            age_and_gender,
            trest_chol_ratio,
            trest_cross_thalach,
        ]

        engineered_feature_layer = tf.keras.layers.concatenate(feature_list)

        numeric_feature_layer = tf.keras.layers.concatenate(
            [inputs[feature] for feature in numeric_features]
        )

        binary_feature_layer = tf.keras.layers.concatenate(
            [inputs[feature] for feature in binary_features]
        )

        # Add the rest of features into final feature layer
        feature_layer = tf.keras.layers.concatenate(
            [engineered_feature_layer, numeric_feature_layer, binary_feature_layer]
        )

        return feature_layer


"""
## Compile and fit model

Our last step is to create and fit our Keras model. For this example we will use a
simple model architecture.
"""

# setup model, this is basically Logistic regression
feature_layer = FeatureLayer(name="feature_layer")(feature_layer_inputs)
x = tf.keras.layers.BatchNormalization(name="batch_norm")(feature_layer)
output = tf.keras.layers.Dense(1, activation="sigmoid", name="target")(x)
model = tf.keras.Model(inputs=feature_layer_inputs, outputs=output)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

"""
## Visualize full network

Lets visualize our complete model graph.
"""

plot_model(model, to_file="complete_model_graph.png")

"""
# Train Model and Save model
"""

model.fit(dataset, epochs=10)

# save model
tf.keras.models.save_model(model, "saved_model")

"""
# Load model and predict on raw data

Below we can see that we load in and predict on raw data. Since our feature engineering is part
of the network we have a complete inference pipeline included in the architecture. This model can
be deployed as a standalone endpoint with no need to first transform feature before feeding it to your
model endpoint.
"""

# load model for inference
loaded_model = tf.keras.models.load_model("saved_model")

dict(zip(loaded_model.metrics_names, loaded_model.evaluate(dataset)))
