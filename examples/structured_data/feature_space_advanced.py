"""
Title: FeatureSpace avanced use cases
Author: [Dimitre Oliveira](https://www.linkedin.com/in/dimitre-oliveira-7a1a0113a/)
Date created: 2023/06/27
Last modified: 2023/06/27
Description: How to use FeatureSpace for advanced preprocessing use cases.
Accelerator: None
"""
"""
## Introduction

This example is an extension of the [Structured data classification with
FeatureSpace](https://keras.io/examples/structured_data/structured_data_classification_wit
h_feature_space/) code example, and here we will extend it to cover more complex use
cases of the [`keras.utils.FeatureSpace`](https://keras.io/api/utils/feature_space/)
preprocessing utility, like feature hashing, feature crosses, handling missing values and
integrating [Keras preprocessing layers](https://keras.io/guides/preprocessing_layers/)
with FeatureSpace.

The general task still is structured data classification (also known as tabular data
classification) using a data that includes numerical features, integer categorical
features, and string categorical features.
"""

"""
### The dataset

[Our dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) is provided by the
Cleveland Clinic Foundation for Heart Disease.
It's a CSV file with 303 rows. Each row contains information about a patient (a
**sample**), and each column describes an attribute of the patient (a **feature**). We
use the features to predict whether a patient has a heart disease
(**binary classification**).

Here's the description of each feature:

Column| Description| Feature Type
------------|--------------------|----------------------
Age | Age in years | Numerical
Sex | (1 = male; 0 = female) | Categorical
CP | Chest pain type (0, 1, 2, 3, 4) | Categorical
Trestbpd | Resting blood pressure (in mm Hg on admission) | Numerical
Chol | Serum cholesterol in mg/dl | Numerical
FBS | fasting blood sugar in 120 mg/dl (1 = true; 0 = false) | Categorical
RestECG | Resting electrocardiogram results (0, 1, 2) | Categorical
Thalach | Maximum heart rate achieved | Numerical
Exang | Exercise induced angina (1 = yes; 0 = no) | Categorical
Oldpeak | ST depression induced by exercise relative to rest | Numerical
Slope | Slope of the peak exercise ST segment | Numerical
CA | Number of major vessels (0-3) colored by fluoroscopy | Both numerical & categorical
Thal | 3 = normal; 6 = fixed defect; 7 = reversible defect | Categorical
Target | Diagnosis of heart disease (1 = true; 0 = false) | Target
"""

"""
## Setup
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import FeatureSpace

"""
## Load the data

Let's download the data and load it into a Pandas dataframe:
"""

data_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(data_url)

"""
The dataset includes 303 samples with 14 columns per sample (13 features, plus the target
label), here's a preview of a few samples:
"""

print(f"Dataframe shape: {dataframe.shape}")
display(dataframe.head())

"""
The last column, "target", indicates whether the patient has a heart disease (1) or not
(0).
"""

"""
## Train/validation split

Let's split the data into a training and validation set:
"""

valid_dataframe = dataframe.sample(frac=0.2, random_state=42)
train_dataframe = dataframe.drop(valid_dataframe.index)

print(
    f"Using {len(train_dataframe)} samples for training and {len(valid_dataframe)} for validation"
)

"""
## Generating TF datasets

Let's generate
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects
for each dataframe:
"""


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
valid_ds = dataframe_to_dataset(valid_dataframe)

"""
Each `Dataset` yields a tuple `(input, target)` where `input` is a dictionary of features
and `target` is the value `0` or `1`:
"""

for x, y in dataframe_to_dataset(train_dataframe).take(1):
    print(f"Input: {x}")
    print(f"Target: {y}")

"""
## Preprocessing

Usually our data is not on the proper or best format for modeling, this is why most of
the time we need to do some kind of preprocessing on the features to make them compatible
with the model or to extract the most of them for the task. We need to do this
preprocessing step for training but but at inference we also need to make sure that the
data goes through the same process, this where a utility like `FeatureSpace` shines, we
can define all the preprocessing once and re-use it at different stages of our system.

Here we will see how to use `FeatureSpace` to perform more complex transformations and
its flexibility, then combine everything together into a single component to preprocess
data for our model.
"""

"""
The `FeatureSpace` utility learns how to process the data by using the `adapt()` function
to learn from it, this requires a dataset containing only feature, so let's create it
together with a utility function to show the preprocessing example in practice:
"""

train_ds_with_no_labels = train_ds.map(lambda x, _: x)


def example_feature_space(dataset, feature_space, feature_names):
    feature_space.adapt(dataset)
    for x in dataset.take(1):
        inputs = {feature_name: x[feature_name] for feature_name in feature_names}
        preprocessed_x = feature_space(inputs)
        print(f"Input: {[{k:v.numpy()} for k, v in inputs.items()]}")
        print(
            f"Preprocessed output: {[{k:v.numpy()} for k, v in preprocessed_x.items()]}"
        )


"""
### Feature hashing
"""

"""
**Feature hashing** means hashing or encoding a set of values into a defined number of
bins, in this case we have `thalach` (Maximum heart rate achieved) which is a numerical
feature that can assume a varying range of values and we will hash it into 4 bins, this
means that any possible value of the original feature will be placed into one of those
possible 5 bins. The output here can be a one-hot encoded vector or a single number.
"""

feature_space = FeatureSpace(
    features={
        "thalach": FeatureSpace.integer_hashed(num_bins=4, output_mode="one_hot")
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["thalach"])

"""
**Feature hashing** can also be used for string features.
"""

feature_space = FeatureSpace(
    features={"thal": FeatureSpace.string_hashed(num_bins=2, output_mode="one_hot")},
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["thal"])

"""
For numerical features we can get a similar behavior by using the `float_discretized`
option, the main difference between this and `integer_hashed` is that with the former we
bin the values while keeping some numerical relationship (close values will likely be
placed at the same bin) while the later (hashing) we cannot guarantee that those numbers
will be hashed into the same bin, it depends on the hashing function.
"""

feature_space = FeatureSpace(
    features={"age": FeatureSpace.float_discretized(num_bins=3, output_mode="one_hot")},
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["age"])

"""
### Feature indexing
"""

"""
**Indexing** a string feature essentially means creating a discrete numerical
representation for it, this is especially important for string features since most models
only accept numerical features. This transformation will place the string values into
different categories. The output here can be a one-hot encoded vector or a single number.

Note that by specifying `num_oov_indices=1` we leave one spot at our output vector for
OOV (out of vocabulary) values this is an important tool to handle missing or unseen
values after the training (values that were not seen during the `adapt()` step)
"""

feature_space = FeatureSpace(
    features={
        "thal": FeatureSpace.string_categorical(
            num_oov_indices=1, output_mode="one_hot"
        )
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["thal"])

"""
We also can do **feature indexing** for integer features, this is quite important for
features like `sex` where values like (`1 and 0`) do not have a numerical relationship
between them, they are just different categories, this behavior can be perfectly captured
by this transformation.

For this case we want to explicitly set `num_oov_indices=0` for the feature `sex` the
reason is that we only expect two possible values for this feature anything else would
either wrong input or and issue with the data creation, for this reason we would probably
just want the code to throw an error so that we can be aware of the issue and fix it.
"""

feature_space = FeatureSpace(
    features={
        "sex": FeatureSpace.integer_categorical(
            num_oov_indices=0, output_mode="one_hot"
        )
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["sex"])

"""
### Feature crosses (mixing features of diverse types)

With **crosses** we can do feature interactions between an arbitrary number of features
of mixed types as long as they are categorical features, you can think of instead of
having a feature {'age': 20} and another {'sex': 0} we can have {'age_X_sex': 20_0}, but
with `FeatureSpace` and **crosses** we can apply specific preprocessing to each
individual feature and to the feature cross itself. This option can be very powerful for
specific use cases, here might be a good option since age combined with gender can have
different meanings for the health domain.

We will cross `age` and `sex` and hash the combination output of them into a vector
representation of size 8. The output here can be a one-hot encoded vector or a single
number.

Sometimes the combination of multiple features can result into on a super large feature
space, think about crossing someone's ZIP code with its last name, the possibilities
would be in the thousands, that is why the `crossing_dim` parameter is so important it
limits the output dimension of the cross feature.

Note that the combination of possible values of the 6 bins of `age` and the 2 values of
`sex` would be 12, so by choosing `crossing_dim` we are choosing to constrain a little
bit the output vector.
"""

feature_space = FeatureSpace(
    features={
        "age": FeatureSpace.integer_hashed(num_bins=6, output_mode="one_hot"),
        "sex": FeatureSpace.integer_categorical(
            num_oov_indices=0, output_mode="one_hot"
        ),
    },
    crosses=[
        FeatureSpace.cross(
            feature_names=("age", "sex"),
            crossing_dim=8,
            output_mode="one_hot",
        )
    ],
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["age", "sex"])

"""
### FeatureSpace using a Keras preprocessing layer

To be a really flexible and extensible feature we cannot only rely on those pre-defined
transformation, we must be able to re-use other transformations from the Keras/TensorFlow
ecosystem and customize our own, this is why `FeatureSpace` is also designed to work with
[Keras preprocessing layers](https://keras.io/guides/preprocessing_layers/), this way we
can use sophisticated data transformations provided by the framework, you can even create
your own custom Keras preprocessing layers and use it in the same way.

Here we are going to use the
[`tf.keras.layers.TextVectorization`](https://keras.io/api/layers/preprocessing_layers/tex
t/text_vectorization/#textvectorization-class) preprocessing layer to create a TF-IDF
feature from our data.
"""

custom_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")

feature_space = FeatureSpace(
    features={
        "thal": FeatureSpace.feature(
            preprocessor=custom_layer, dtype="string", output_mode="float"
        )
    },
    output_mode="dict",
)
example_feature_space(train_ds_with_no_labels, feature_space, ["thal"])

"""
## Configuring the final `FeatureSpace`

Now that we know how to use `FeatureSpace` for more complex use cases let's pick the ones
that looks more useful for this task and create the final `FeatureSpace` component.

To configure how each feature should be preprocessed,
we instantiate a `keras.utils.FeatureSpace`, and we
pass to it a dictionary that maps the name of our features
to the feature transformation function.

"""

feature_space = FeatureSpace(
    features={
        # Categorical features encoded as integers
        "sex": FeatureSpace.integer_categorical(num_oov_indices=0),
        "cp": FeatureSpace.integer_categorical(num_oov_indices=0),
        "fbs": FeatureSpace.integer_categorical(num_oov_indices=0),
        "restecg": FeatureSpace.integer_categorical(num_oov_indices=0),
        "exang": FeatureSpace.integer_categorical(num_oov_indices=0),
        "ca": FeatureSpace.integer_categorical(num_oov_indices=0),
        # Categorical features encoded as string
        "thal": FeatureSpace.string_categorical(num_oov_indices=1),
        # Numerical features to hash and bin
        "age": FeatureSpace.integer_hashed(num_bins=6),
        "thalach": FeatureSpace.integer_hashed(num_bins=4),
        # Numerical features to normalize
        "trestbps": FeatureSpace.float_normalized(),
        "chol": FeatureSpace.float_normalized(),
        "oldpeak": FeatureSpace.float_normalized(),
        "slope": FeatureSpace.float_normalized(),
    },
    # Specify feature cross with a custom crossing dim.
    crosses=[
        FeatureSpace.cross(
            feature_names=("age", "sex"),
            crossing_dim=8,
            output_mode="one_hot",
        )
    ],
    output_mode="concat",
)

"""
## Adapt the `FeatureSpace` to the training data

Before we start using the `FeatureSpace` to build a model, we have
to adapt it to the training data. During `adapt()`, the `FeatureSpace` will:

- Index the set of possible values for categorical features.
- Compute the mean and variance for numerical features to normalize.
- Compute the value boundaries for the different bins for numerical features to
discretize.
- Any other kind of preprocessing required by custom layers.

Note that `adapt()` should be called on a `tf.data.Dataset` which yields dicts
of feature values -- no labels.

But first let's batch the datasets
"""

train_ds = train_ds.batch(32)
valid_ds = valid_ds.batch(32)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

"""
At this point, the `FeatureSpace` can be called on a dict of raw feature values, and
because we set `output_mode="concat"` it will return a single concatenate vector for each
sample, combining encoded features and feature crosses.
"""

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print(f"preprocessed_x shape: {preprocessed_x.shape}")
    print(f"preprocessed_x sample: \n{preprocessed_x[0]}")

"""
## Saving the `FeatureSpace`

At this point we can choose to save our `FeatureSpace` component, this have many
advantages like re-using it on different experiments that use the same model, saving time
if you need to re-run the preprocessing step, and mainly for model deployment, where by
loading it you can be sure that you will be applying the same preprocessing steps don't
matter the device or environment, this is a great way to reduce [training/serving
skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew).
skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew).
"""

feature_space.save("myfeaturespace.keras")

"""
## Preprocessing with `FeatureSpace` as part of the tf.data pipeline

We will opt to use our component asynchronously by making it part of the tf.data
pipeline, as noted at the [previous
guide](https://keras.io/examples/structured_data/structured_data_classification_with_featu
re_space/) This enables asynchronous parallel preprocessing of the data on CPU before it
hits the model. Usually, this is always the right thing to do during training.

Let's create a training and validation dataset of preprocessed batches:
"""

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

preprocessed_valid_ds = valid_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

"""
## Model

We will take advantage of our `FeatureSpace` component to build the model, as we want the
model to be compatible with our preprocessing function, let's use the the `FeatureSpace`
feature map as the input of our model.
"""

encoded_features = feature_space.get_encoded_features()
print(encoded_features)

"""
This model is quite trivial only for demonstration purposes so don't pay too much
attention to the architecture.
"""

x = tf.keras.layers.Dense(32, activation="relu")(encoded_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=encoded_features, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

"""
## Training

Let's train our model for 50 epochs. Note that feature preprocessing is happening as part
of the tf.data pipeline, not as part of the model.
"""

model.fit(
    preprocessed_train_ds, validation_data=preprocessed_valid_ds, epochs=20, verbose=2
)

"""
## Inference on new data with the end-to-end model

Now, we can build our inference model (which includes the `FeatureSpace`) to make
predictions based on dicts of raw features values, as follows:
"""

"""
### Loading the `FeatureSpace`

First let's load the `FeatureSpace` that we saved a few moment ago, this can be quite
handy if you train a model but want to do inference at different time, possibly using a
different device or environment.
"""

loaded_feature_space = tf.keras.models.load_model("myfeaturespace.keras")

"""
### Building the inference end-to-end model

To build the inference model we need both the feature input map and the preprocessing
encoded Keras tensors.
"""

dict_inputs = loaded_feature_space.get_inputs()
encoded_features = loaded_feature_space.get_encoded_features()
print(encoded_features)

dict_inputs

outputs = model(encoded_features)
inference_model = tf.keras.Model(inputs=dict_inputs, outputs=outputs)

sample = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = inference_model.predict(input_dict)

print(
    f"This particular patient had a {100 * predictions[0][0]:.2f}% probability "
    "of having a heart disease, as evaluated by our model."
)
