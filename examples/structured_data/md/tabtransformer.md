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

```
</div>
    
![png](/img/examples/structured_data/tabtransformer/tabtransformer_24_1.png)
    



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
123/123 [==============================] - 6s 25ms/step - loss: 110178.8203 - accuracy: 0.7478 - val_loss: 92703.0859 - val_accuracy: 0.7825
Epoch 2/15
123/123 [==============================] - 2s 14ms/step - loss: 90979.8125 - accuracy: 0.7675 - val_loss: 71798.9219 - val_accuracy: 0.8001
Epoch 3/15
123/123 [==============================] - 2s 14ms/step - loss: 77226.5547 - accuracy: 0.7902 - val_loss: 68581.0312 - val_accuracy: 0.8168
Epoch 4/15
123/123 [==============================] - 2s 14ms/step - loss: 72652.2422 - accuracy: 0.8004 - val_loss: 70084.0469 - val_accuracy: 0.7974
Epoch 5/15
123/123 [==============================] - 2s 14ms/step - loss: 71207.9375 - accuracy: 0.8033 - val_loss: 66552.1719 - val_accuracy: 0.8130
Epoch 6/15
123/123 [==============================] - 2s 14ms/step - loss: 69321.4375 - accuracy: 0.8091 - val_loss: 65837.0469 - val_accuracy: 0.8149
Epoch 7/15
123/123 [==============================] - 2s 14ms/step - loss: 68839.3359 - accuracy: 0.8099 - val_loss: 65613.0156 - val_accuracy: 0.8187
Epoch 8/15
123/123 [==============================] - 2s 14ms/step - loss: 68126.7344 - accuracy: 0.8124 - val_loss: 66155.8594 - val_accuracy: 0.8108
Epoch 9/15
123/123 [==============================] - 2s 14ms/step - loss: 67768.9844 - accuracy: 0.8147 - val_loss: 66705.8047 - val_accuracy: 0.8230
Epoch 10/15
123/123 [==============================] - 2s 14ms/step - loss: 67482.5859 - accuracy: 0.8151 - val_loss: 65668.3672 - val_accuracy: 0.8143
Epoch 11/15
123/123 [==============================] - 2s 14ms/step - loss: 66792.6875 - accuracy: 0.8181 - val_loss: 66536.3828 - val_accuracy: 0.8233
Epoch 12/15
123/123 [==============================] - 2s 14ms/step - loss: 65610.4531 - accuracy: 0.8229 - val_loss: 70377.7266 - val_accuracy: 0.8256
Epoch 13/15
123/123 [==============================] - 2s 14ms/step - loss: 63930.2500 - accuracy: 0.8282 - val_loss: 68294.8516 - val_accuracy: 0.8289
Epoch 14/15
123/123 [==============================] - 2s 14ms/step - loss: 63420.1562 - accuracy: 0.8323 - val_loss: 63050.5859 - val_accuracy: 0.8324
Epoch 15/15
123/123 [==============================] - 2s 14ms/step - loss: 62619.4531 - accuracy: 0.8345 - val_loss: 66933.7500 - val_accuracy: 0.8277
Model training finished
Validation accuracy: 82.77%

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

```
</div>
    
![png](/img/examples/structured_data/tabtransformer/tabtransformer_29_1.png)
    



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
123/123 [==============================] - 13s 61ms/step - loss: 82503.1641 - accuracy: 0.7944 - val_loss: 64260.2305 - val_accuracy: 0.8421
Epoch 2/15
123/123 [==============================] - 6s 51ms/step - loss: 68677.9375 - accuracy: 0.8251 - val_loss: 63819.8633 - val_accuracy: 0.8389
Epoch 3/15
123/123 [==============================] - 6s 51ms/step - loss: 66703.8984 - accuracy: 0.8301 - val_loss: 63052.8789 - val_accuracy: 0.8428
Epoch 4/15
123/123 [==============================] - 6s 51ms/step - loss: 65287.8672 - accuracy: 0.8342 - val_loss: 61593.1484 - val_accuracy: 0.8451
Epoch 5/15
123/123 [==============================] - 6s 52ms/step - loss: 63968.8594 - accuracy: 0.8379 - val_loss: 61385.4531 - val_accuracy: 0.8442
Epoch 6/15
123/123 [==============================] - 6s 51ms/step - loss: 63645.7812 - accuracy: 0.8394 - val_loss: 61332.3281 - val_accuracy: 0.8447
Epoch 7/15
123/123 [==============================] - 6s 51ms/step - loss: 62778.6055 - accuracy: 0.8412 - val_loss: 61342.5352 - val_accuracy: 0.8461
Epoch 8/15
123/123 [==============================] - 6s 51ms/step - loss: 62815.6992 - accuracy: 0.8398 - val_loss: 61220.8242 - val_accuracy: 0.8460
Epoch 9/15
123/123 [==============================] - 6s 52ms/step - loss: 62191.1016 - accuracy: 0.8416 - val_loss: 61055.9102 - val_accuracy: 0.8452
Epoch 10/15
123/123 [==============================] - 6s 51ms/step - loss: 61992.1602 - accuracy: 0.8439 - val_loss: 61251.8047 - val_accuracy: 0.8441
Epoch 11/15
123/123 [==============================] - 6s 50ms/step - loss: 61745.1289 - accuracy: 0.8429 - val_loss: 61364.7695 - val_accuracy: 0.8445
Epoch 12/15
123/123 [==============================] - 6s 51ms/step - loss: 61696.3477 - accuracy: 0.8445 - val_loss: 61074.3594 - val_accuracy: 0.8450
Epoch 13/15
123/123 [==============================] - 6s 51ms/step - loss: 61569.1719 - accuracy: 0.8436 - val_loss: 61844.9688 - val_accuracy: 0.8456
Epoch 14/15
123/123 [==============================] - 6s 51ms/step - loss: 61343.0898 - accuracy: 0.8445 - val_loss: 61702.8828 - val_accuracy: 0.8455
Epoch 15/15
123/123 [==============================] - 6s 51ms/step - loss: 61355.0547 - accuracy: 0.8454 - val_loss: 61272.2852 - val_accuracy: 0.8455
Model training finished
Validation accuracy: 84.55%

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

Example available on HuggingFace.

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/🤗%20Model-TabTransformer-black.svg)](https://huggingface.co/keras-io/tab_transformer) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-TabTransformer-black.svg)](https://huggingface.co/spaces/keras-io/TabTransformer_Classification) |
