# Structured data learning with TabTransformer

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2022/01/18<br>
**Last modified:** 2022/01/18<br>
**Description:** Using contextual embeddings for structured data classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/tabtransformer.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/tabtransformer.py)



---
## Introduction

This example demonstrates how to do structured data classification using
[TabTransformer](https://arxiv.org/abs/2012.06678), a deep tabular data modeling
architecture for supervised and semi-supervised learning.
The TabTransformer is built upon self-attention based Transformers.
The Transformer layers transform the embeddings of categorical features
into robust contextual embeddings to achieve higher predictive accuracy.

This example should be run with TensorFlow 2.7 or higher,
as well as [TensorFlow Addons](https://www.tensorflow.org/addons/overview),
which can be installed using the following command:

```python
pip install -U tensorflow-addons
```

---
## Setup


```python
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
```

<div class="k-default-codeblock">
```
2022-01-18 18:54:01.877364: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-01-18 18:54:01.877410: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

```
</div>
---
## Prepare the data

This example uses the
[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)
provided by the
[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The task is binary classification
to predict whether a person is likely to be making over USD 50,000 a year.

The dataset includes 48,842 instances with 14 input features: 5 numerical features and 9 categorical features.

First, let's load the dataset from the UCI Machine Learning Repository into a Pandas
DataFrame:


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
Remove the first record (because it is not a valid data example) and a trailing 'dot' in the class labels.


```python
test_data = test_data[1:]
test_data.income_bracket = test_data.income_bracket.apply(
    lambda value: value.replace(".", "")
)
```

Now we store the training and test data in separate CSV files.


```python
train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)
```

---
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for reading and parsing
the data into input features, and encoding the input features with respect to their types.


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
# Name of the column to be used as instances weight.
WEIGHT_COLUMN_NAME = "fnlwgt"
# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
# A list of all the input features.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# A list of column default values for each feature.
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME] else ["NA"]
    for feature_name in CSV_HEADER
]
# The name of the target feature.
TARGET_FEATURE_NAME = "income_bracket"
# A list of the labels of the target features.
TARGET_LABELS = [" <=50K", " >50K"]
```

---
## Configure the hyperparameters

The hyperparameters includes model architecture and training configurations.


```python
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 265
NUM_EPOCHS = 15

NUM_TRANSFORMER_BLOCKS = 3  # Number of transformer blocks.
NUM_HEADS = 4  # Number of attention heads.
EMBEDDING_DIMS = 16  # Embedding dimensions of the categorical features.
MLP_HIDDEN_UNITS_FACTORS = [
    2,
    1,
]  # MLP hidden layer units, as factors of the number of inputs.
NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.
```

---
## Implement data reading pipeline

We define an input function that reads and parses the file, then converts features
and labels into a[`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets)
for training or evaluation.


```python
target_label_lookup = layers.StringLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)


def prepare_example(features, target):
    target_index = target_label_lookup(target)
    weights = features.pop(WEIGHT_COLUMN_NAME)
    return features, target_index, weights


def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
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
    ).map(prepare_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return dataset.cache()

```

<div class="k-default-codeblock">
```
2022-01-18 18:54:12.579096: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-01-18 18:54:12.579159: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-01-18 18:54:12.579200: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (keras-notebooks): /proc/driver/nvidia/version does not exist
2022-01-18 18:54:12.580891: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

```
</div>
---
## Implement a training and evaluation procedure


```python

def run_experiment(
    model,
    train_data_file,
    test_data_file,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_data_file, batch_size)

    print("Start training the model...")
    history = model.fit(
        train_dataset, epochs=num_epochs, validation_data=validation_dataset
    )
    print("Model training finished")

    _, accuracy = model.evaluate(validation_dataset, verbose=0)

    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")

    return history

```

---
## Create model inputs

Now, define the inputs for the models as a dictionary, where the key is the feature name,
and the value is a `keras.layers.Input` tensor with the corresponding feature shape
and data type.


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
## Encode features

The `encode_inputs` method returns `encoded_categorical_feature_list` and `numerical_feature_list`.
We encode the categorical features as embeddings, using a fixed `embedding_dims` for all the features,
regardless their vocabulary sizes. This is required for the Transformer model.


```python

def encode_inputs(inputs, embedding_dims):

    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:

            # Get the vocabulary of the categorical feature.
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]

            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Convert the string input values into integer indices.
            encoded_feature = lookup(inputs[feature_name])

            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:

            # Use the numerical features as-is.
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list

```

---
## Implement an MLP block


```python

def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)

```

---
## Experiment 1: a baseline model

In the first experiment, we create a simple multi-layer feed-forward network.


```python

def create_baseline_model(
    embedding_dims, num_mlp_blocks, mlp_hidden_units_factors, dropout_rate
):

    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Concatenate all features.
    features = layers.concatenate(
        encoded_categorical_feature_list + numerical_feature_list
    )
    # Compute Feedforward layer units.
    feedforward_units = [features.shape[-1]]

    # Create several feedforwad layers with skip connections.
    for layer_idx in range(num_mlp_blocks):
        features = create_mlp(
            hidden_units=feedforward_units,
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{layer_idx}",
        )(features)

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


baseline_model = create_baseline_model(
    embedding_dims=EMBEDDING_DIMS,
    num_mlp_blocks=NUM_MLP_BLOCKS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", baseline_model.count_params())
keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")
```

<div class="k-default-codeblock">
```
Total model weights: 109629
('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')

```
</div>
Let's train and evaluate the baseline model:


```python
history = run_experiment(
    model=baseline_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/15
123/123 [==============================] - 7s 27ms/step - loss: 110574.9141 - accuracy: 0.7469 - val_loss: 95106.3047 - val_accuracy: 0.7812
Epoch 2/15
123/123 [==============================] - 2s 15ms/step - loss: 90679.1172 - accuracy: 0.7677 - val_loss: 85083.6250 - val_accuracy: 0.7831
Epoch 3/15
123/123 [==============================] - 2s 15ms/step - loss: 75590.9531 - accuracy: 0.7925 - val_loss: 68293.9844 - val_accuracy: 0.8179
Epoch 4/15
123/123 [==============================] - 2s 15ms/step - loss: 72426.5391 - accuracy: 0.8017 - val_loss: 69626.7812 - val_accuracy: 0.8183
Epoch 5/15
123/123 [==============================] - 2s 15ms/step - loss: 70548.9766 - accuracy: 0.8044 - val_loss: 78782.6875 - val_accuracy: 0.8059
Epoch 6/15
123/123 [==============================] - 2s 15ms/step - loss: 69423.2812 - accuracy: 0.8056 - val_loss: 68385.0859 - val_accuracy: 0.8188
Epoch 7/15
123/123 [==============================] - 2s 15ms/step - loss: 68915.7656 - accuracy: 0.8113 - val_loss: 67443.1797 - val_accuracy: 0.8216
Epoch 8/15
123/123 [==============================] - 2s 15ms/step - loss: 68597.8906 - accuracy: 0.8120 - val_loss: 68740.5938 - val_accuracy: 0.8187
Epoch 9/15
123/123 [==============================] - 2s 15ms/step - loss: 68021.6328 - accuracy: 0.8118 - val_loss: 66513.7031 - val_accuracy: 0.8205
Epoch 10/15
123/123 [==============================] - 2s 15ms/step - loss: 67422.9531 - accuracy: 0.8150 - val_loss: 67441.1094 - val_accuracy: 0.8212
Epoch 11/15
123/123 [==============================] - 2s 15ms/step - loss: 67292.8047 - accuracy: 0.8149 - val_loss: 67310.7031 - val_accuracy: 0.8206
Epoch 12/15
123/123 [==============================] - 2s 15ms/step - loss: 67039.5391 - accuracy: 0.8152 - val_loss: 66347.3359 - val_accuracy: 0.8211
Epoch 13/15
123/123 [==============================] - 2s 14ms/step - loss: 66747.0234 - accuracy: 0.8167 - val_loss: 66547.9453 - val_accuracy: 0.8205
Epoch 14/15
123/123 [==============================] - 2s 15ms/step - loss: 66576.5938 - accuracy: 0.8193 - val_loss: 64635.7031 - val_accuracy: 0.8237
Epoch 15/15
123/123 [==============================] - 2s 15ms/step - loss: 65734.0938 - accuracy: 0.8219 - val_loss: 65115.3750 - val_accuracy: 0.8312
Model training finished
Validation accuracy: 83.12%

```
</div>
The baseline linear model achieves ~81% validation accuracy.

---
## Experiment 2: TabTransformer

The TabTransformer architecture works as follows:

1. All the categorical features are encoded as embeddings, using the same `embedding_dims`.
This means that each value in each categorical feature will have its own embedding vector.
2. A column embedding, one embedding vector for each categorical feature, is added (point-wise) to the categorical feature embedding.
3. The embedded categorical features are fed into a stack of Transformer blocks.
Each Transformer block consists of a multi-head self-attention layer followed by a feed-forward layer.
3. The outputs of the final Transformer layer, which are the *contextual embeddings* of the categorical features,
are concatenated with the input numerical features, and fed into a final MLP block.
4. A `softmax` classifer is applied at the end of the model.

The [paper](https://arxiv.org/abs/2012.06678) discusses both addition and concatenation of the column embedding in the
*Appendix: Experiment and Model Details* section.
The architecture of TabTransformer is shown below, as presented in the paper.

<img src="https://raw.githubusercontent.com/keras-team/keras-io/master/examples/structured_data/img/tabtransformer/tabtransformer.png" width="500"/>

```python

def create_tabtransformer_classifier(
    num_transformer_blocks,
    num_heads,
    embedding_dims,
    mlp_hidden_units_factors,
    dropout_rate,
    use_column_embedding=False,
):

    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Stack categorical feature embeddings for the Tansformer.
    encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
    # Concatenate numerical features.
    numerical_features = layers.concatenate(numerical_feature_list)

    # Add column embedding to categorical feature embeddings.
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=embedding_dims
        )
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(
            column_indices
        )

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)
        # Skip connection 1.
        x = layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, encoded_categorical_features]
        )
        # Layer normalization 1.
        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
        # Feedforward.
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{block_idx}",
        )(x)
        # Skip connection 2.
        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        # Layer normalization 2.
        encoded_categorical_features = layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Flatten the "contextualized" embeddings of the categorical features.
    categorical_features = layers.Flatten()(encoded_categorical_features)
    # Apply layer normalization to the numerical features.
    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
    # Prepare the input for the final MLP block.
    features = layers.concatenate([categorical_features, numerical_features])

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


tabtransformer_model = create_tabtransformer_classifier(
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS,
    embedding_dims=EMBEDDING_DIMS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print("Total model weights:", tabtransformer_model.count_params())
keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir="LR")
```

<div class="k-default-codeblock">
```
Total model weights: 87479
('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')

```
</div>
Let's train and evaluate the TabTransformer model:


```python
history = run_experiment(
    model=tabtransformer_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/15
123/123 [==============================] - 14s 63ms/step - loss: 79315.3516 - accuracy: 0.7970 - val_loss: 66115.7969 - val_accuracy: 0.8406
Epoch 2/15
123/123 [==============================] - 7s 53ms/step - loss: 68751.2109 - accuracy: 0.8253 - val_loss: 65475.8320 - val_accuracy: 0.8397
Epoch 3/15
123/123 [==============================] - 7s 54ms/step - loss: 66452.5312 - accuracy: 0.8297 - val_loss: 62136.4180 - val_accuracy: 0.8463
Epoch 4/15
123/123 [==============================] - 7s 53ms/step - loss: 64246.3477 - accuracy: 0.8366 - val_loss: 62125.8516 - val_accuracy: 0.8452
Epoch 5/15
123/123 [==============================] - 6s 53ms/step - loss: 63572.6055 - accuracy: 0.8387 - val_loss: 61939.7930 - val_accuracy: 0.8447
Epoch 6/15
123/123 [==============================] - 6s 52ms/step - loss: 63012.8477 - accuracy: 0.8415 - val_loss: 62016.4570 - val_accuracy: 0.8454
Epoch 7/15
123/123 [==============================] - 6s 52ms/step - loss: 62265.0547 - accuracy: 0.8424 - val_loss: 61894.2852 - val_accuracy: 0.8471
Epoch 8/15
123/123 [==============================] - 6s 52ms/step - loss: 62148.0352 - accuracy: 0.8424 - val_loss: 61682.7070 - val_accuracy: 0.8422
Epoch 9/15
123/123 [==============================] - 7s 53ms/step - loss: 61794.2773 - accuracy: 0.8441 - val_loss: 62334.7891 - val_accuracy: 0.8421
Epoch 10/15
123/123 [==============================] - 7s 54ms/step - loss: 61479.3867 - accuracy: 0.8439 - val_loss: 62580.0625 - val_accuracy: 0.8418
Epoch 11/15
123/123 [==============================] - 7s 54ms/step - loss: 61184.4883 - accuracy: 0.8462 - val_loss: 61866.2031 - val_accuracy: 0.8456
Epoch 12/15
123/123 [==============================] - 7s 54ms/step - loss: 61241.9336 - accuracy: 0.8448 - val_loss: 61736.5547 - val_accuracy: 0.8431
Epoch 13/15
123/123 [==============================] - 7s 53ms/step - loss: 61207.5625 - accuracy: 0.8453 - val_loss: 61739.3633 - val_accuracy: 0.8433
Epoch 14/15
123/123 [==============================] - 6s 53ms/step - loss: 61024.6758 - accuracy: 0.8450 - val_loss: 61468.6406 - val_accuracy: 0.8450
Epoch 15/15
123/123 [==============================] - 6s 53ms/step - loss: 60983.3984 - accuracy: 0.8455 - val_loss: 62017.6562 - val_accuracy: 0.8422
Model training finished
Validation accuracy: 84.22%

```
</div>
The TabTransformer model achieves ~85% validation accuracy.
Note that, with the default parameter configurations, both the baseline and the TabTransformer
have similar number of trainable weights: 109,629 and 92,151 respectively, and both use the same training hyperparameters.

---
## Conclusion

TabTransformer significantly outperforms MLP and recent
deep networks for tabular data while matching the performance of tree-based ensemble models.
TabTransformer can be learned in end-to-end supervised training using labeled examples.
For a scenario where there are a few labeled examples and a large number of unlabeled
examples, a pre-training procedure can be employed to train the Transformer layers using unlabeled data.
This is followed by fine-tuning of the pre-trained Transformer layers along with
the top MLP layer using the labeled data.
