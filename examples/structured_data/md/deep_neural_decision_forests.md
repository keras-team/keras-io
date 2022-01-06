# Classification with Neural Decision Forests

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/01/15<br>
**Last modified:** 2021/01/15<br>
**Description:** How to train differentiable decision trees for end-to-end learning in deep neural networks.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/deep_neural_decision_forests.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/deep_neural_decision_forests.py)



---
## Introduction

This example provides an implementation of the
[Deep Neural Decision Forest](https://ieeexplore.ieee.org/document/7410529)
model introduced by P. Kontschieder et al. for structured data classification.
It demonstrates how to build a stochastic and differentiable decision tree model,
train it end-to-end, and unify decision trees with deep representation learning.

---
## The dataset

This example uses the
[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)
provided by the
[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The task is binary classification
to predict whether a person is likely to be making over USD 50,000 a year.

The dataset includes 48,842 instances with 14 input features (such as age, work class, education, occupation, and so on): 5 numerical features
and 9 categorical features.

---
## Setup


```python
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import math
```

---
## Prepare the data


```python
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

train_data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
train_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)

test_data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")
```

<div class="k-default-codeblock">
```
Train dataset shape: (32561, 15)
Test dataset shape: (16282, 15)

```
</div>
Remove the first record (because it is not a valid data example) and a trailing
'dot' in the class labels.


```python
test_data = test_data[1:]
test_data.income_bracket = test_data.income_bracket.apply(
    lambda value: value.replace(".", "")
)
```

We store the training and test data splits locally as CSV files.


```python
train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)
```

---
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for reading and parsing
and encoding input features.


```python
# A list of the numerical feature names.
NUMERIC_FEATURE_NAMES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
# A dictionary of the categorical features and their vocabulary.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "workclass": sorted(list(train_data["workclass"].unique())),
    "education": sorted(list(train_data["education"].unique())),
    "marital_status": sorted(list(train_data["marital_status"].unique())),
    "occupation": sorted(list(train_data["occupation"].unique())),
    "relationship": sorted(list(train_data["relationship"].unique())),
    "race": sorted(list(train_data["race"].unique())),
    "gender": sorted(list(train_data["gender"].unique())),
    "native_country": sorted(list(train_data["native_country"].unique())),
}
# A list of the columns to ignore from the dataset.
IGNORE_COLUMN_NAMES = ["fnlwgt"]
# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
# A list of all the input features.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# A list of column default values for each feature.
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES + IGNORE_COLUMN_NAMES else ["NA"]
    for feature_name in CSV_HEADER
]
# The name of the target feature.
TARGET_FEATURE_NAME = "income_bracket"
# A list of the labels of the target features.
TARGET_LABELS = [" <=50K", " >50K"]
```

---
## Create `tf.data.Dataset` objects for training and validation

We create an input function to read and parse the file, and convert features and labels
into a [`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets)
for training and validation. We also preprocess the input by mapping the target label
to an index.


```python
from tensorflow.keras.layers import StringLookup

target_label_lookup = StringLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        na_value="?",
        shuffle=shuffle,
    ).map(lambda features, target: (features, target_label_lookup(target)))
    return dataset.cache()

```

---
## Create model inputs


```python

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

```

---
## Encode input features


```python

def encode_inputs(inputs):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token, nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and num_oov_indices to 0.
            lookup = StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_index = lookup(inputs[feature_name])
            embedding_dims = int(math.sqrt(lookup.vocabulary_size()))
            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=lookup.vocabulary_size(), output_dim=embedding_dims
            )
            # Convert the index values to embedding representations.
            encoded_feature = embedding(value_index)
        else:
            # Use the numerical features as-is.
            encoded_feature = inputs[feature_name]
            if inputs[feature_name].shape[-1] is None:
                encoded_feature = tf.expand_dims(encoded_feature, -1)

        encoded_features.append(encoded_feature)

    encoded_features = layers.concatenate(encoded_features)
    return encoded_features

```

---
## Deep Neural Decision Tree

A neural decision tree model has two sets of weights to learn. The first set is `pi`,
which represents the probability distribution of the classes in the tree leaves.
The second set is the weights of the routing layer `decision_fn`, which represents the probability
of going to each leave. The forward pass of the model works as follows:

1. The model expects input `features` as a single vector encoding all the features of an instance
in the batch. This vector can be generated from a Convolution Neural Network (CNN) applied to images
or dense transformations applied to structured data features.
2. The model first applies a `used_features_mask` to randomly select a subset of input features to use.
3. Then, the model computes the probabilities (`mu`) for the input instances to reach the tree leaves
by iteratively performing a *stochastic* routing throughout the tree levels.
4. Finally, the probabilities of reaching the leaves are combined by the class probabilities at the
leaves to produce the final `outputs`.


```python

class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]

        # Initialize the weights of the classes in leaves.
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
            ),
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]

        # Apply the feature mask to the input features.
        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]
        # Concatenate the routing probabilities with their complements.
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]

        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                :, begin_idx:end_idx, :
            ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = keras.activations.softmax(self.pi)  # [num_leaves, num_classes]
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs

```

---
## Deep Neural Decision Forest

The neural decision forest model consists of a set of neural decision trees that are
trained simultaneously. The output of the forest model is the average outputs of its trees.


```python

class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionForest, self).__init__()
        self.ensemble = []
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, num_classes])

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs

```

Finally, let's set up the code that will train and evaluate the model.


```python
learning_rate = 0.01
batch_size = 265
num_epochs = 10
hidden_units = [64, 64]


def run_experiment(model):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("Start training the model...")
    train_dataset = get_dataset_from_csv(
        train_data_file, shuffle=True, batch_size=batch_size
    )

    model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")

    print("Evaluating the model on the test data...")
    test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)

    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

```

---
## Experiment 1: train a decision tree model

In this experiment, we train a single neural decision tree model
where we use all input features.


```python
num_trees = 10
depth = 10
used_features_rate = 1.0
num_classes = len(TARGET_LABELS)


def create_tree_model():
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    tree = NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)

    outputs = tree(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


tree_model = create_tree_model()
run_experiment(tree_model)

```

<div class="k-default-codeblock">
```

123/123 [==============================] - 3s 9ms/step - loss: 0.5326 - sparse_categorical_accuracy: 0.7838
Epoch 2/10
123/123 [==============================] - 1s 9ms/step - loss: 0.3406 - sparse_categorical_accuracy: 0.8469
Epoch 3/10
123/123 [==============================] - 1s 9ms/step - loss: 0.3254 - sparse_categorical_accuracy: 0.8499
Epoch 4/10
123/123 [==============================] - 1s 9ms/step - loss: 0.3188 - sparse_categorical_accuracy: 0.8539
Epoch 5/10
123/123 [==============================] - 1s 9ms/step - loss: 0.3137 - sparse_categorical_accuracy: 0.8573
Epoch 6/10
123/123 [==============================] - 1s 9ms/step - loss: 0.3091 - sparse_categorical_accuracy: 0.8581
Epoch 7/10
123/123 [==============================] - 1s 9ms/step - loss: 0.3039 - sparse_categorical_accuracy: 0.8596
Epoch 8/10
123/123 [==============================] - 1s 9ms/step - loss: 0.2991 - sparse_categorical_accuracy: 0.8633
Epoch 9/10
123/123 [==============================] - 1s 9ms/step - loss: 0.2935 - sparse_categorical_accuracy: 0.8667
Epoch 10/10
123/123 [==============================] - 1s 9ms/step - loss: 0.2877 - sparse_categorical_accuracy: 0.8708
Model training finished
Evaluating the model on the test data...
62/62 [==============================] - 1s 5ms/step - loss: 0.3314 - sparse_categorical_accuracy: 0.8471
Test accuracy: 84.71%

```
</div>
---
## Experiment 2: train a forest model

In this experiment, we train a neural decision forest with `num_trees` trees
where each tree uses randomly selected 50% of the input features. You can control the number
of features to be used in each tree by setting the `used_features_rate` variable.
In addition, we set the depth to 5 instead of 10 compared to the previous experiment.


```python
num_trees = 25
depth = 5
used_features_rate = 0.5


def create_forest_model():
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    forest_model = NeuralDecisionForest(
        num_trees, depth, num_features, used_features_rate, num_classes
    )

    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


forest_model = create_forest_model()

run_experiment(forest_model)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/10
123/123 [==============================] - 9s 7ms/step - loss: 0.5523 - sparse_categorical_accuracy: 0.7872
Epoch 2/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3435 - sparse_categorical_accuracy: 0.8465
Epoch 3/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3260 - sparse_categorical_accuracy: 0.8514
Epoch 4/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3197 - sparse_categorical_accuracy: 0.8533
Epoch 5/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3160 - sparse_categorical_accuracy: 0.8535
Epoch 6/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3133 - sparse_categorical_accuracy: 0.8545
Epoch 7/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3110 - sparse_categorical_accuracy: 0.8556
Epoch 8/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3088 - sparse_categorical_accuracy: 0.8559
Epoch 9/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3066 - sparse_categorical_accuracy: 0.8573
Epoch 10/10
123/123 [==============================] - 1s 6ms/step - loss: 0.3048 - sparse_categorical_accuracy: 0.8573
Model training finished
Evaluating the model on the test data...
62/62 [==============================] - 2s 5ms/step - loss: 0.3140 - sparse_categorical_accuracy: 0.8533
Test accuracy: 85.33%

```
</div>
