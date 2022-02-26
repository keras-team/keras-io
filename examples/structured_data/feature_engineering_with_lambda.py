"""
Title: Feature Engineering using Keras Funcional API and Lambda layers
Author: [Fernando Nieuwveldt](https://www.linkedin.com/in/fernandonieuwveldt/)
Date created: 2022/02/26
Last modified: 2022/02/26
Description: Feature Engineering as part of your network architecture.
"""
"""
# Feature Engineering using Keras Lambda Layers for an end to end training pipeline.
"""

"""
Often for structured data problems we end up using multiple libraries for preprocessing
or feature engineering. We can go as far as having a full ML training pipeline using
different libraries for example Pandas for reading data and also feature engineeering,
sklearn for encoding features for example OneHot encoding and Normalization. The
estimator might be an sklearn classifier, xgboost or it can for example be a Keras model.
In the latter case, we would end up with artifacts for feature engineering and encoding
and also different artifacts for the saved model. The pipeline is also disconnected and
an extra step is needed to feed encoded data to the Keras model. For this step the data
can be mapped from a dataframe to something like tf.data.Datasets type or numpy array
before feeding it to a Keras model.

In this example we will consider implementing a training pipeline natively with
Keras/Tensorflow. From loading data with tf.data, Lambda layers for feature engineering.
These engineered features will be stateless. We will end up with a training pipeline where
feature engineering will be part of the network architecture and can be persisted and loaded
for inference as standalone.

Steps we will follow:
- Load data with tf.data
- Create Input layer
- Create feature layer using Lambda layers
- Train model

For constructing our model we will the Functional API. As we build our netwrk we will visualize model.
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

We first create a dictionary with feature name and type mapper and than create a dict of Input objects.
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
## Visualize Input layer

As our network evolves we will visualize it at different construction points.
"""

input_layer_graph = tf.keras.Model(
    inputs=feature_layer_inputs, outputs=feature_layer_inputs
)
plot_model(input_layer_graph)

"""
## Engineered Features: Custom functions for Lambda Layers

For feature engineering we will be using Keras Lambda layers. Below are the custom functions that will be used
transform our features. The categoric feature "thal" we will apply One hot encoding manually to illustrate how it can
be done with Lambda layers.
"""


def ratio(x):
    """compute the ratio between two numeric features"""
    return x[0] / x[1]


def cross_feature(x):
    """compute the crossing of two features"""
    return tf.cast(x[0] * x[1], dtype=tf.float32)


def age_and_gender(x):
    """check if age gt 50 and if gender is male"""
    return tf.cast(tf.math.logical_and(x[0] > 50, x[1] == 1), dtype=tf.float32)


def is_fixed(x):
    """encode categoric feature if value is equal to fixed"""
    return tf.cast(x == "fixed", dtype=tf.float32)


def is_reversible(x):
    """encode categoric feature if value is equal to fixed"""
    return tf.cast(x == "reversible", dtype=tf.float32)


def is_normal(x):
    """encode categoric feature if value is equal to fixed"""
    return tf.cast(x == "normal", dtype=tf.float32)


"""
## Feature engineering

Now that we have functions for our engineered features, we can start building feature layer. When
all the layers are defined we will concatenate all these layers.
"""

is_fixed = tf.keras.layers.Lambda(is_fixed, name="is_fixed")(
    feature_layer_inputs["thal"]
)

is_normal = tf.keras.layers.Lambda(is_normal, name="is_normal")(
    feature_layer_inputs["thal"]
)

is_reversible = tf.keras.layers.Lambda(is_reversible, name="is_reversible")(
    feature_layer_inputs["thal"]
)

age_and_gender = tf.keras.layers.Lambda(age_and_gender, name="age_and_gender")(
    (feature_layer_inputs["age"], feature_layer_inputs["sex"])
)

trest_chol_ratio = tf.keras.layers.Lambda(ratio, name="trest_chol_ratio")(
    (feature_layer_inputs["trestbps"], feature_layer_inputs["chol"])
)

trest_cross_thalach = tf.keras.layers.Lambda(cross_feature, name="trest_cross_thalach")(
    (feature_layer_inputs["trestbps"], feature_layer_inputs["thalach"])
)

# concat all newly created features into one layer
feature_list = [
    is_fixed,
    is_normal,
    is_reversible,
    age_and_gender,
    trest_chol_ratio,
    trest_cross_thalach,
]

lambda_feature_layer = tf.keras.layers.concatenate(
    feature_list, name="lambda_feature_layer"
)

"""
## Visualize Lambda feature Layer
"""

feature_graph = tf.keras.Model(
    inputs=feature_layer_inputs, outputs=lambda_feature_layer
)
plot_model(feature_graph)

"""
## Combine all features

All our engineered feature layers are created and we can now combine it with our other
features into one complete feature layer.
"""

numeric_feature_layer = tf.keras.layers.concatenate(
    [feature_layer_inputs[feature] for feature in numeric_features],
    name="numeric_feature_layer",
)

binary_feature_layer = tf.keras.layers.concatenate(
    [feature_layer_inputs[feature] for feature in binary_features],
    name="binary_feature_layer",
)

# Add the rest of features
feature_layer = tf.keras.layers.concatenate(
    [lambda_feature_layer, numeric_feature_layer, binary_feature_layer],
    name="feature_layer",
)

"""
## Visualize full Feature Layer

We can now view our complete Feature layer.
"""

full_feature_graph = tf.keras.Model(inputs=feature_layer_inputs, outputs=feature_layer)
plot_model(full_feature_graph)

"""
## Compile and fit model

Our last step is to create and fit our Keras model. For this example we will use a
simple model architecture.
"""

# setup model, this is basically Logistic regression
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

plot_model(model)

"""
# Train Model and Save model
"""

model.fit(dataset, epochs=10)

# save model
tf.keras.models.save_model(model, "lambda_layered_model")

"""
# Load model and predict on raw data

Below we can see that we load in and predict on raw data. Since our feature engineering is part
of the network we have a complete inference pipeline included in the architecture. This model can
be deployed as a standalone endpoint with no need to first transform feature before feeding it to your
model endpoint.
"""

# load model for inference
loaded_model = tf.keras.models.load_model("lambda_layered_model")

dict(zip(loaded_model.metrics_names, loaded_model.evaluate(dataset)))
