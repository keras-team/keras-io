"""
Title: Feature Preprocessing with a custom Feature Preprocessor layer.
Author: [Fernando Nieuwveldt](https://www.linkedin.com/in/fernandonieuwveldt/)
Date created: 2023/08/27
Last modified: 2023/08/27
Description: Feature Preprocessing with a custom Feature Preprocessor layer.
"""
"""
# Feature Preprocessing with a custom Feature Preprocessor layer for an end to end training and inference pipeline.
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
utilizes Keras preprocessing layers. For constructing our model we will use the Functional API. At the end we will
apply the common pattern to preprocess the data first before passing it to the model.
"""

"""
## Setup
"""
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

"""
## Preparing the data

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

dataframe = pd.read_csv(data_url, header=None, names=CSV_HEADER)

"""
Let's split the data into training and validation sets:
"""

validation_dataframe = dataframe.sample(frac=0.2, random_state=42)
train_dataframe = dataframe.drop(validation_dataframe.index)

train_labels = train_dataframe.pop("income_bracket")
train_labels_binary = 1.0 * (train_labels == " >50K")

validation_labels = validation_dataframe.pop("income_bracket")
validation_labels_binary = 1.0 * (validation_labels == " >50K")

"""
We can print out the shape of data for both training and validation sets. We have a total of 14 features.
"""

print(f"Total samples in the training set: {train_dataframe.shape}")
print(f"Total samples in the validation set: {validation_dataframe.shape}")

"""
## Split features into different feature groups

Features in the income dataset are preprocessed based on their characteristics. NUMERICAL_FEATURES
are scaled to prevent undue influence from large numerical ranges. CATEGORICAL_FEATURES are transformed (e.g., one-hot encoded)
to make them machine-readable. "age" is DISCRETIZED to capture patterns across age groups rather than specific ages. High-cardinality "native_country"
uses embeddings to efficiently represent relationships in a compact space, enhancing the model's learning capability.
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

"""
## Split data frame into different feature groups

Here we will create the different data mappings. First we split the features into their different groups. Thereafter we
create our input dictionary for the input layer.
"""

categorical_features = {
    feature: train_dataframe[[feature]].values for feature in CATEGORICAL_FEATURES
}
numerical_features = {
    feature: train_dataframe[[feature]].values for feature in NUMERICAL_FEATURES
}
discretize_features = {
    feature: train_dataframe[[feature]].values for feature in FEATURES_TO_DISCRETIZE
}
embedding_features = {
    feature: train_dataframe[[feature]].values for feature in FEATURES_TO_EMBED
}

train_data_dict = {
    **numerical_features,
    **categorical_features,
    **discretize_features,
    **embedding_features,
}

"""
Below we will create input layers for all the features.
"""

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

"""
Lets combine all our features with their different data types into one input dict.
"""

input_dict = {
    **numerical_features_input,
    **categorical_features_input,
    **discretize_features_input,
    **embedding_features_input,
}


"""
## Create Feature Preprocessing layer

In this section we will implement a custom Keras layer for preprocessing. We have an adapt
method that will update our preprocessing layer's states which we will apply in the forward pass.

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

    def apply_preprocessor(self, inputs, preprocessor_dict):
        """Helper function to apply preprocessing layers in the forward pass"""
        return tf.keras.layers.concatenate(
            [
                preprocessor(inputs[feature])
                for feature, preprocessor in preprocessor_dict.items()
            ]
        )

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
        numerical_features = self.apply_preprocessor(inputs, self.normalizer)

        categorical_features = {
            feature: self.categoric_string_lookup[feature](inputs[feature])
            for feature in CATEGORICAL_FEATURES
        }
        categorical_features = self.apply_preprocessor(
            categorical_features, self.integer_lookup
        )

        embedding_features = self.embedding(inputs[FEATURES_TO_EMBED[0]])

        discretize_features = self.apply_preprocessor(inputs, self.discretizer)
        discretize_features = tf.cast(discretize_features, dtype=tf.float32)

        return tf.keras.layers.concatenate(
            [
                numerical_features,
                categorical_features,
                discretize_features,
                embedding_features,
            ]
        )


"""
## Speed up computation

To save computational time we will apply the preprocessing layer before we pass the data to the model. The preprocessing layer does not need to be part of the forward 
and can be applied once off.
"""

# update preprocessing layer states
preprocessing_layer = FeaturePreprocessor()
preprocessing_layer.adapt(train_data_dict)

preprocessed_inputs = preprocessing_layer(input_dict)

preprocessed_data_set = preprocessing_layer(train_data_dict)
preprocessing_model = tf.keras.Model(input_dict, preprocessed_inputs)

x = tf.keras.layers.Dense(64, activation="relu")(preprocessed_inputs)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
training_model = tf.keras.Model(inputs=preprocessed_inputs, outputs=outputs)

"""
## Compile and fit model

Lets put everything together and fit the model:
"""

training_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

training_model.fit(
    preprocessing_model(train_data_dict), train_labels_binary.values, epochs=10
)

"""
## Create model for inference

We need one last step to create a model that can be used for inference. Since we splitted the model into a preprocessing and training step we can’t save training_model as is.
We need to combine the preprocessing and training into one model. Let’s create our inference model. Finally we will save the model and load it for inference
"""

inference_model = tf.keras.Model(input_dict, training_model(preprocessed_inputs))
inference_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

tf.keras.models.save_model(model=inference_model, filepath="model")
saved_inference_model = tf.keras.models.load_model("model")

"""
## Model evaluation

Lets run inference on both the training and evaluation set and compare results. We can see that the metrics on the training data is
similar to the validation set.
"""

print(f"Evaluation on the training dataset")
dict(
    zip(
        saved_inference_model.metrics_names,
        saved_inference_model.evaluate(train_data_dict, train_labels_binary.values),
    )
)

"""
Now let's apply the model on unseen data. In the `evaluate` method to the model we pass the validation data in the 
form of a dictionary.
"""

print(f"Evaluation on the validation dataset")
dict(
    zip(
        saved_inference_model.metrics_names,
        saved_inference_model.evaluate(
            dict(validation_dataframe), validation_labels_binary.values
        ),
    )
)
