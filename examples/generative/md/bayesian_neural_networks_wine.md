# Probabilistic Bayesian Neural Networks

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/01/15<br>
**Last modified:** 2021/01/15<br>
**Description:** Building probabilistic Bayesian neural network models with TensorFlow Probability.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/bayesian_neural_networks_wine.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/bayesian_neural_networks_wine.py)



---
## Introduction

Taking a probabilistic approach to deep learning allows to account for *uncertainty*,
so that models can assign less levels of confidence to incorrect predictions.
Sources of uncertainty can be found in the data, due to measurement error or
noise in the labels, or the model, due to insufficient data availability for
the model to learn effectively.


This example demonstrates how to build basic probabilistic Bayesian neural networks
to account for these two types of uncertainty.
We use [TensorFlow Probability](https://www.tensorflow.org/probability) library,
which is compatible with Keras API.

This example requires TensorFlow 2.3 or higher.
You can install Tensorflow Probability using the following command:

```python
pip install tensorflow-probability
```

---
## The dataset

We use the [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
dataset, which is available in the [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/wine_quality).
We use the red wine subset, which contains 4,898 examples.
The dataset has 11numerical physicochemical features of the wine, and the task
is to predict the wine quality, which is a score between 0 and 10.
In this example, we treat this as a regression task.

You can install TensorFlow Datasets using the following command:

```python
pip install tensorflow-datasets
```

---
## Setup


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
```

---
## Create training and evaluation datasets

Here, we load the `wine_quality` dataset using `tfds.load()`, and we convert
the target feature to float. Then, we shuffle the dataset and split it into
training and test sets. We take the first `train_size` examples as the train
split, and the rest as the test split.


```python

def get_train_and_test_splits(train_size, batch_size=1):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset = (
        tfds.load(name="wine_quality", as_supervised=True, split="train")
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=dataset_size)
        .cache()
    )
    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)

    return train_dataset, test_dataset

```

---
## Compile, train, and evaluate the model


```python
hidden_units = [8, 8]
learning_rate = 0.001


def run_experiment(model, loss, train_dataset, test_dataset):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

```

---
## Create model inputs


```python
FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

```

---
## Experiment 1: standard neural network

We create a standard deterministic neural network model as a baseline.


```python

def create_baseline_model():
    inputs = create_model_inputs()
    input_values = [value for _, value in sorted(inputs.items())]
    features = keras.layers.concatenate(input_values)
    features = layers.BatchNormalization()(features)

    # Create hidden layers with deterministic weights using the Dense layer.
    for units in hidden_units:
        features = layers.Dense(units, activation="sigmoid")(features)
    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

Let's split the wine dataset into training and test sets, with 85% and 15% of
the examples, respectively.


```python
dataset_size = 4898
batch_size = 256
train_size = int(dataset_size * 0.85)
train_dataset, test_dataset = get_train_and_test_splits(train_size, batch_size)
```

Now let's train the baseline model. We use the `MeanSquaredError`
as the loss function.


```python
num_epochs = 100
mse_loss = keras.losses.MeanSquaredError()
baseline_model = create_baseline_model()
run_experiment(baseline_model, mse_loss, train_dataset, test_dataset)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/100
17/17 [==============================] - 1s 53ms/step - loss: 37.5710 - root_mean_squared_error: 6.1294 - val_loss: 35.6750 - val_root_mean_squared_error: 5.9729
Epoch 2/100
17/17 [==============================] - 0s 7ms/step - loss: 35.5154 - root_mean_squared_error: 5.9594 - val_loss: 34.2430 - val_root_mean_squared_error: 5.8518
Epoch 3/100
17/17 [==============================] - 0s 7ms/step - loss: 33.9975 - root_mean_squared_error: 5.8307 - val_loss: 32.8003 - val_root_mean_squared_error: 5.7272
Epoch 4/100
17/17 [==============================] - 0s 12ms/step - loss: 32.5928 - root_mean_squared_error: 5.7090 - val_loss: 31.3385 - val_root_mean_squared_error: 5.5981
Epoch 5/100
17/17 [==============================] - 0s 7ms/step - loss: 30.8914 - root_mean_squared_error: 5.5580 - val_loss: 29.8659 - val_root_mean_squared_error: 5.4650
Epoch 6/100
17/17 [==============================] - 0s 7ms/step - loss: 29.5307 - root_mean_squared_error: 5.4342 - val_loss: 28.3900 - val_root_mean_squared_error: 5.3282
Epoch 7/100
17/17 [==============================] - 0s 6ms/step - loss: 28.2115 - root_mean_squared_error: 5.3113 - val_loss: 26.9246 - val_root_mean_squared_error: 5.1889
Epoch 8/100
17/17 [==============================] - 0s 7ms/step - loss: 26.4299 - root_mean_squared_error: 5.1410 - val_loss: 25.4681 - val_root_mean_squared_error: 5.0466
Epoch 9/100
17/17 [==============================] - 0s 7ms/step - loss: 25.0831 - root_mean_squared_error: 5.0083 - val_loss: 24.0296 - val_root_mean_squared_error: 4.9020
Epoch 10/100
17/17 [==============================] - 0s 6ms/step - loss: 23.7685 - root_mean_squared_error: 4.8752 - val_loss: 22.6224 - val_root_mean_squared_error: 4.7563
Epoch 11/100
17/17 [==============================] - 0s 6ms/step - loss: 22.4562 - root_mean_squared_error: 4.7388 - val_loss: 21.2548 - val_root_mean_squared_error: 4.6103
Epoch 12/100
17/17 [==============================] - 0s 7ms/step - loss: 21.1921 - root_mean_squared_error: 4.6034 - val_loss: 19.9361 - val_root_mean_squared_error: 4.4650
Epoch 13/100
17/17 [==============================] - 0s 6ms/step - loss: 19.9356 - root_mean_squared_error: 4.4649 - val_loss: 18.6669 - val_root_mean_squared_error: 4.3205
Epoch 14/100
17/17 [==============================] - 0s 12ms/step - loss: 18.7415 - root_mean_squared_error: 4.3291 - val_loss: 17.4475 - val_root_mean_squared_error: 4.1770
Epoch 15/100
17/17 [==============================] - 0s 7ms/step - loss: 17.4381 - root_mean_squared_error: 4.1759 - val_loss: 16.2859 - val_root_mean_squared_error: 4.0356
Epoch 16/100
17/17 [==============================] - 0s 7ms/step - loss: 16.4205 - root_mean_squared_error: 4.0521 - val_loss: 15.1778 - val_root_mean_squared_error: 3.8959
Epoch 17/100
17/17 [==============================] - 0s 7ms/step - loss: 15.4181 - root_mean_squared_error: 3.9264 - val_loss: 14.1279 - val_root_mean_squared_error: 3.7587
Epoch 18/100
17/17 [==============================] - 0s 7ms/step - loss: 14.1984 - root_mean_squared_error: 3.7680 - val_loss: 13.1276 - val_root_mean_squared_error: 3.6232
Epoch 19/100
17/17 [==============================] - 0s 7ms/step - loss: 13.2537 - root_mean_squared_error: 3.6404 - val_loss: 12.1697 - val_root_mean_squared_error: 3.4885
Epoch 20/100
17/17 [==============================] - 0s 6ms/step - loss: 12.2955 - root_mean_squared_error: 3.5064 - val_loss: 11.2556 - val_root_mean_squared_error: 3.3549
Epoch 21/100
17/17 [==============================] - 0s 7ms/step - loss: 11.3546 - root_mean_squared_error: 3.3695 - val_loss: 10.3749 - val_root_mean_squared_error: 3.2210
Epoch 22/100
17/17 [==============================] - 0s 7ms/step - loss: 10.3501 - root_mean_squared_error: 3.2171 - val_loss: 9.5317 - val_root_mean_squared_error: 3.0873
Epoch 23/100
17/17 [==============================] - 0s 7ms/step - loss: 9.6030 - root_mean_squared_error: 3.0988 - val_loss: 8.7294 - val_root_mean_squared_error: 2.9545
Epoch 24/100
17/17 [==============================] - 0s 7ms/step - loss: 8.6998 - root_mean_squared_error: 2.9493 - val_loss: 7.9622 - val_root_mean_squared_error: 2.8217
Epoch 25/100
17/17 [==============================] - 0s 8ms/step - loss: 8.0382 - root_mean_squared_error: 2.8351 - val_loss: 7.2247 - val_root_mean_squared_error: 2.6879
Epoch 26/100
17/17 [==============================] - 0s 7ms/step - loss: 7.2271 - root_mean_squared_error: 2.6882 - val_loss: 6.5203 - val_root_mean_squared_error: 2.5535
Epoch 27/100
17/17 [==============================] - 0s 7ms/step - loss: 6.4049 - root_mean_squared_error: 2.5308 - val_loss: 5.8537 - val_root_mean_squared_error: 2.4194
Epoch 28/100
17/17 [==============================] - 0s 7ms/step - loss: 5.8823 - root_mean_squared_error: 2.4252 - val_loss: 5.2296 - val_root_mean_squared_error: 2.2868
Epoch 29/100
17/17 [==============================] - 0s 7ms/step - loss: 5.2445 - root_mean_squared_error: 2.2900 - val_loss: 4.6449 - val_root_mean_squared_error: 2.1552
Epoch 30/100
17/17 [==============================] - 0s 12ms/step - loss: 4.6201 - root_mean_squared_error: 2.1493 - val_loss: 4.1010 - val_root_mean_squared_error: 2.0251
Epoch 31/100
17/17 [==============================] - 0s 6ms/step - loss: 4.1470 - root_mean_squared_error: 2.0362 - val_loss: 3.5964 - val_root_mean_squared_error: 1.8964
Epoch 32/100
17/17 [==============================] - 0s 6ms/step - loss: 3.5799 - root_mean_squared_error: 1.8920 - val_loss: 3.1314 - val_root_mean_squared_error: 1.7696
Epoch 33/100
17/17 [==============================] - 0s 6ms/step - loss: 3.0863 - root_mean_squared_error: 1.7567 - val_loss: 2.7145 - val_root_mean_squared_error: 1.6476
Epoch 34/100
17/17 [==============================] - 0s 6ms/step - loss: 2.6680 - root_mean_squared_error: 1.6333 - val_loss: 2.3401 - val_root_mean_squared_error: 1.5297
Epoch 35/100
17/17 [==============================] - 0s 6ms/step - loss: 2.3241 - root_mean_squared_error: 1.5244 - val_loss: 2.0120 - val_root_mean_squared_error: 1.4185
Epoch 36/100
17/17 [==============================] - 0s 6ms/step - loss: 1.9857 - root_mean_squared_error: 1.4091 - val_loss: 1.7239 - val_root_mean_squared_error: 1.3130
Epoch 37/100
17/17 [==============================] - 0s 6ms/step - loss: 1.6724 - root_mean_squared_error: 1.2932 - val_loss: 1.4767 - val_root_mean_squared_error: 1.2152
Epoch 38/100
17/17 [==============================] - 0s 6ms/step - loss: 1.4694 - root_mean_squared_error: 1.2121 - val_loss: 1.2731 - val_root_mean_squared_error: 1.1283
Epoch 39/100
17/17 [==============================] - 0s 6ms/step - loss: 1.2701 - root_mean_squared_error: 1.1267 - val_loss: 1.1105 - val_root_mean_squared_error: 1.0538
Epoch 40/100
17/17 [==============================] - 0s 6ms/step - loss: 1.1102 - root_mean_squared_error: 1.0536 - val_loss: 0.9843 - val_root_mean_squared_error: 0.9921
Epoch 41/100
17/17 [==============================] - 0s 6ms/step - loss: 0.9705 - root_mean_squared_error: 0.9850 - val_loss: 0.8924 - val_root_mean_squared_error: 0.9447
Epoch 42/100
17/17 [==============================] - 0s 12ms/step - loss: 0.8842 - root_mean_squared_error: 0.9402 - val_loss: 0.8379 - val_root_mean_squared_error: 0.9153
Epoch 43/100
17/17 [==============================] - 0s 6ms/step - loss: 0.8289 - root_mean_squared_error: 0.9104 - val_loss: 0.8082 - val_root_mean_squared_error: 0.8990
Epoch 44/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7621 - root_mean_squared_error: 0.8728 - val_loss: 0.8023 - val_root_mean_squared_error: 0.8957
Epoch 45/100
17/17 [==============================] - 0s 7ms/step - loss: 0.7786 - root_mean_squared_error: 0.8824 - val_loss: 0.8043 - val_root_mean_squared_error: 0.8968
Epoch 46/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7909 - root_mean_squared_error: 0.8892 - val_loss: 0.8039 - val_root_mean_squared_error: 0.8966
Epoch 47/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7895 - root_mean_squared_error: 0.8885 - val_loss: 0.8047 - val_root_mean_squared_error: 0.8970
Epoch 48/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7666 - root_mean_squared_error: 0.8754 - val_loss: 0.8048 - val_root_mean_squared_error: 0.8971
Epoch 49/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7703 - root_mean_squared_error: 0.8775 - val_loss: 0.8026 - val_root_mean_squared_error: 0.8959
Epoch 50/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7951 - root_mean_squared_error: 0.8916 - val_loss: 0.8031 - val_root_mean_squared_error: 0.8962
Epoch 51/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7927 - root_mean_squared_error: 0.8903 - val_loss: 0.8020 - val_root_mean_squared_error: 0.8956
Epoch 52/100
17/17 [==============================] - 0s 6ms/step - loss: 0.8006 - root_mean_squared_error: 0.8946 - val_loss: 0.8014 - val_root_mean_squared_error: 0.8952
Epoch 53/100
17/17 [==============================] - 0s 11ms/step - loss: 0.7770 - root_mean_squared_error: 0.8814 - val_loss: 0.8008 - val_root_mean_squared_error: 0.8949
Epoch 54/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7553 - root_mean_squared_error: 0.8690 - val_loss: 0.8013 - val_root_mean_squared_error: 0.8951
Epoch 55/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7950 - root_mean_squared_error: 0.8914 - val_loss: 0.8009 - val_root_mean_squared_error: 0.8949
Epoch 56/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7868 - root_mean_squared_error: 0.8870 - val_loss: 0.8002 - val_root_mean_squared_error: 0.8946
Epoch 57/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7671 - root_mean_squared_error: 0.8758 - val_loss: 0.7998 - val_root_mean_squared_error: 0.8943
Epoch 58/100
17/17 [==============================] - 0s 6ms/step - loss: 0.8018 - root_mean_squared_error: 0.8953 - val_loss: 0.7994 - val_root_mean_squared_error: 0.8941
Epoch 59/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7856 - root_mean_squared_error: 0.8863 - val_loss: 0.7982 - val_root_mean_squared_error: 0.8934
Epoch 60/100
17/17 [==============================] - 0s 7ms/step - loss: 0.7989 - root_mean_squared_error: 0.8937 - val_loss: 0.7952 - val_root_mean_squared_error: 0.8917
Epoch 61/100
17/17 [==============================] - 0s 10ms/step - loss: 0.7610 - root_mean_squared_error: 0.8722 - val_loss: 0.7933 - val_root_mean_squared_error: 0.8907
Epoch 62/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7534 - root_mean_squared_error: 0.8679 - val_loss: 0.7927 - val_root_mean_squared_error: 0.8903
Epoch 63/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7590 - root_mean_squared_error: 0.8712 - val_loss: 0.7903 - val_root_mean_squared_error: 0.8890
Epoch 64/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7588 - root_mean_squared_error: 0.8711 - val_loss: 0.7883 - val_root_mean_squared_error: 0.8879
Epoch 65/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7717 - root_mean_squared_error: 0.8784 - val_loss: 0.7833 - val_root_mean_squared_error: 0.8851
Epoch 66/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7651 - root_mean_squared_error: 0.8747 - val_loss: 0.7802 - val_root_mean_squared_error: 0.8833
Epoch 67/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7420 - root_mean_squared_error: 0.8612 - val_loss: 0.7780 - val_root_mean_squared_error: 0.8820
Epoch 68/100
17/17 [==============================] - 0s 15ms/step - loss: 0.7521 - root_mean_squared_error: 0.8672 - val_loss: 0.7762 - val_root_mean_squared_error: 0.8810
Epoch 69/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7641 - root_mean_squared_error: 0.8741 - val_loss: 0.7727 - val_root_mean_squared_error: 0.8790
Epoch 70/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7677 - root_mean_squared_error: 0.8761 - val_loss: 0.7736 - val_root_mean_squared_error: 0.8796
Epoch 71/100
17/17 [==============================] - 0s 7ms/step - loss: 0.7665 - root_mean_squared_error: 0.8754 - val_loss: 0.7679 - val_root_mean_squared_error: 0.8763
Epoch 72/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7634 - root_mean_squared_error: 0.8737 - val_loss: 0.7670 - val_root_mean_squared_error: 0.8758
Epoch 73/100
17/17 [==============================] - 0s 7ms/step - loss: 0.7510 - root_mean_squared_error: 0.8665 - val_loss: 0.7667 - val_root_mean_squared_error: 0.8756
Epoch 74/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7528 - root_mean_squared_error: 0.8674 - val_loss: 0.7633 - val_root_mean_squared_error: 0.8737
Epoch 75/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7498 - root_mean_squared_error: 0.8658 - val_loss: 0.7595 - val_root_mean_squared_error: 0.8715
Epoch 76/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7464 - root_mean_squared_error: 0.8638 - val_loss: 0.7568 - val_root_mean_squared_error: 0.8700
Epoch 77/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7289 - root_mean_squared_error: 0.8537 - val_loss: 0.7554 - val_root_mean_squared_error: 0.8691
Epoch 78/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7349 - root_mean_squared_error: 0.8572 - val_loss: 0.7527 - val_root_mean_squared_error: 0.8676
Epoch 79/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7500 - root_mean_squared_error: 0.8659 - val_loss: 0.7491 - val_root_mean_squared_error: 0.8655
Epoch 80/100
17/17 [==============================] - 0s 7ms/step - loss: 0.7214 - root_mean_squared_error: 0.8493 - val_loss: 0.7466 - val_root_mean_squared_error: 0.8641
Epoch 81/100
17/17 [==============================] - 0s 14ms/step - loss: 0.7365 - root_mean_squared_error: 0.8580 - val_loss: 0.7463 - val_root_mean_squared_error: 0.8639
Epoch 82/100
17/17 [==============================] - 0s 8ms/step - loss: 0.7273 - root_mean_squared_error: 0.8528 - val_loss: 0.7422 - val_root_mean_squared_error: 0.8615
Epoch 83/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7437 - root_mean_squared_error: 0.8623 - val_loss: 0.7400 - val_root_mean_squared_error: 0.8602
Epoch 84/100
17/17 [==============================] - 0s 10ms/step - loss: 0.7103 - root_mean_squared_error: 0.8427 - val_loss: 0.7350 - val_root_mean_squared_error: 0.8573
Epoch 85/100
17/17 [==============================] - 0s 9ms/step - loss: 0.7200 - root_mean_squared_error: 0.8485 - val_loss: 0.7319 - val_root_mean_squared_error: 0.8555
Epoch 86/100
17/17 [==============================] - 0s 7ms/step - loss: 0.7213 - root_mean_squared_error: 0.8492 - val_loss: 0.7282 - val_root_mean_squared_error: 0.8533
Epoch 87/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7151 - root_mean_squared_error: 0.8456 - val_loss: 0.7289 - val_root_mean_squared_error: 0.8538
Epoch 88/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7120 - root_mean_squared_error: 0.8438 - val_loss: 0.7209 - val_root_mean_squared_error: 0.8491
Epoch 89/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7055 - root_mean_squared_error: 0.8399 - val_loss: 0.7181 - val_root_mean_squared_error: 0.8474
Epoch 90/100
17/17 [==============================] - 0s 7ms/step - loss: 0.6871 - root_mean_squared_error: 0.8289 - val_loss: 0.7135 - val_root_mean_squared_error: 0.8447
Epoch 91/100
17/17 [==============================] - 0s 12ms/step - loss: 0.7138 - root_mean_squared_error: 0.8448 - val_loss: 0.7093 - val_root_mean_squared_error: 0.8422
Epoch 92/100
17/17 [==============================] - 0s 6ms/step - loss: 0.7003 - root_mean_squared_error: 0.8368 - val_loss: 0.7061 - val_root_mean_squared_error: 0.8403
Epoch 93/100
17/17 [==============================] - 0s 6ms/step - loss: 0.6813 - root_mean_squared_error: 0.8252 - val_loss: 0.7016 - val_root_mean_squared_error: 0.8376
Epoch 94/100
17/17 [==============================] - 0s 7ms/step - loss: 0.6920 - root_mean_squared_error: 0.8318 - val_loss: 0.6943 - val_root_mean_squared_error: 0.8332
Epoch 95/100
17/17 [==============================] - 0s 6ms/step - loss: 0.6927 - root_mean_squared_error: 0.8322 - val_loss: 0.6901 - val_root_mean_squared_error: 0.8307
Epoch 96/100
17/17 [==============================] - 0s 6ms/step - loss: 0.6929 - root_mean_squared_error: 0.8323 - val_loss: 0.6866 - val_root_mean_squared_error: 0.8286
Epoch 97/100
17/17 [==============================] - 0s 6ms/step - loss: 0.6582 - root_mean_squared_error: 0.8112 - val_loss: 0.6797 - val_root_mean_squared_error: 0.8244
Epoch 98/100
17/17 [==============================] - 0s 6ms/step - loss: 0.6733 - root_mean_squared_error: 0.8205 - val_loss: 0.6740 - val_root_mean_squared_error: 0.8210
Epoch 99/100
17/17 [==============================] - 0s 7ms/step - loss: 0.6623 - root_mean_squared_error: 0.8138 - val_loss: 0.6713 - val_root_mean_squared_error: 0.8193
Epoch 100/100
17/17 [==============================] - 0s 6ms/step - loss: 0.6522 - root_mean_squared_error: 0.8075 - val_loss: 0.6666 - val_root_mean_squared_error: 0.8165
Model training finished.
Train RMSE: 0.809
Evaluating model performance...
Test RMSE: 0.816

```
</div>
We take a sample from the test set use the model to obtain predictions for them.
Note that since the baseline model is deterministic, we get a single a
*point estimate* prediction for each test example, with no information about the
uncertainty of the model nor the prediction.


```python
sample = 10
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[
    0
]

predicted = baseline_model(examples).numpy()
for idx in range(sample):
    print(f"Predicted: {round(float(predicted[idx][0]), 1)} - Actual: {targets[idx]}")
```

<div class="k-default-codeblock">
```
Predicted: 6.0 - Actual: 6.0
Predicted: 6.2 - Actual: 6.0
Predicted: 5.8 - Actual: 7.0
Predicted: 6.0 - Actual: 5.0
Predicted: 5.7 - Actual: 5.0
Predicted: 6.2 - Actual: 7.0
Predicted: 5.6 - Actual: 5.0
Predicted: 6.2 - Actual: 6.0
Predicted: 6.2 - Actual: 6.0
Predicted: 6.2 - Actual: 7.0

```
</div>
---
## Experiment 2: Bayesian neural network (BNN)

The object of the Bayesian approach for modeling neural networks is to capture
the *epistemic uncertainty*, which is uncertainty about the model fitness,
due to limited training data.

The idea is that, instead of learning specific weight (and bias) *values* in the
neural network, the Bayesian approach learns weight *distributions*
- from which we can sample to produce an output for a given input -
to encode weight uncertainty.

Thus, we need to define prior and the posterior distributions of these weights,
and the training process is to learn the parameters of these distributions.


```python
# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

```

We use the `tfp.layers.DenseVariational` layer instead of the standard
`keras.layers.Dense` layer in the neural network model.


```python

def create_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)

    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

The epistemic uncertainty can be reduced as we increase the size of the
training data. That is, the more data the BNN model sees, the more it is certain
about its estimates for the weights (distribution parameters).
Let's test this behaviour by training the BNN model on a small subset of
the training set, and then on the full training set, to compare the output variances.

### Train BNN  with a small training subset.


```python
num_epochs = 500
train_sample_size = int(train_size * 0.3)
small_train_dataset = train_dataset.unbatch().take(train_sample_size).batch(batch_size)

bnn_model_small = create_bnn_model(train_sample_size)
run_experiment(bnn_model_small, mse_loss, small_train_dataset, test_dataset)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:From /Users/khalidsalama/Technology/python-venvs/keras-env/lib/python3.7/site-packages/tensorflow/python/ops/linalg/linear_operator_lower_triangular.py:167: calling LinearOperator.__init__ (from tensorflow.python.ops.linalg.linear_operator) with graph_parents is deprecated and will be removed in a future version.
Instructions for updating:
Do not pass `graph_parents`.  They will  no longer be used.

WARNING:tensorflow:From /Users/khalidsalama/Technology/python-venvs/keras-env/lib/python3.7/site-packages/tensorflow/python/ops/linalg/linear_operator_lower_triangular.py:167: calling LinearOperator.__init__ (from tensorflow.python.ops.linalg.linear_operator) with graph_parents is deprecated and will be removed in a future version.
Instructions for updating:
Do not pass `graph_parents`.  They will  no longer be used.

Start training the model...
Epoch 1/500
5/5 [==============================] - 2s 123ms/step - loss: 34.5497 - root_mean_squared_error: 5.8764 - val_loss: 37.1164 - val_root_mean_squared_error: 6.0910
Epoch 2/500
5/5 [==============================] - 0s 28ms/step - loss: 36.0738 - root_mean_squared_error: 6.0007 - val_loss: 31.7373 - val_root_mean_squared_error: 5.6322
Epoch 3/500
5/5 [==============================] - 0s 29ms/step - loss: 33.3177 - root_mean_squared_error: 5.7700 - val_loss: 36.2135 - val_root_mean_squared_error: 6.0164
Epoch 4/500
5/5 [==============================] - 0s 30ms/step - loss: 35.1247 - root_mean_squared_error: 5.9232 - val_loss: 35.6158 - val_root_mean_squared_error: 5.9663
Epoch 5/500
5/5 [==============================] - 0s 23ms/step - loss: 34.7653 - root_mean_squared_error: 5.8936 - val_loss: 34.3038 - val_root_mean_squared_error: 5.8556
Epoch 6/500
5/5 [==============================] - 0s 28ms/step - loss: 36.5552 - root_mean_squared_error: 6.0443 - val_loss: 32.0486 - val_root_mean_squared_error: 5.6596
Epoch 7/500
5/5 [==============================] - 0s 51ms/step - loss: 30.8287 - root_mean_squared_error: 5.5508 - val_loss: 31.3516 - val_root_mean_squared_error: 5.5980
Epoch 8/500
5/5 [==============================] - 0s 25ms/step - loss: 34.5313 - root_mean_squared_error: 5.8741 - val_loss: 30.3839 - val_root_mean_squared_error: 5.5107
Epoch 9/500
5/5 [==============================] - 0s 25ms/step - loss: 32.8506 - root_mean_squared_error: 5.7300 - val_loss: 32.6105 - val_root_mean_squared_error: 5.7092
Epoch 10/500
5/5 [==============================] - 0s 30ms/step - loss: 32.9074 - root_mean_squared_error: 5.7338 - val_loss: 34.9973 - val_root_mean_squared_error: 5.9145
Epoch 11/500
5/5 [==============================] - 0s 24ms/step - loss: 35.5673 - root_mean_squared_error: 5.9614 - val_loss: 32.2414 - val_root_mean_squared_error: 5.6770
Epoch 12/500
5/5 [==============================] - 0s 24ms/step - loss: 33.3083 - root_mean_squared_error: 5.7689 - val_loss: 30.8746 - val_root_mean_squared_error: 5.5553
Epoch 13/500
5/5 [==============================] - 0s 25ms/step - loss: 31.7328 - root_mean_squared_error: 5.6317 - val_loss: 30.7058 - val_root_mean_squared_error: 5.5397
Epoch 14/500
5/5 [==============================] - 0s 24ms/step - loss: 32.7779 - root_mean_squared_error: 5.7237 - val_loss: 30.2540 - val_root_mean_squared_error: 5.4991
Epoch 15/500
5/5 [==============================] - 0s 23ms/step - loss: 32.0536 - root_mean_squared_error: 5.6592 - val_loss: 27.5936 - val_root_mean_squared_error: 5.2519
Epoch 16/500
5/5 [==============================] - 0s 25ms/step - loss: 29.5777 - root_mean_squared_error: 5.4366 - val_loss: 32.8287 - val_root_mean_squared_error: 5.7285
Epoch 17/500
5/5 [==============================] - 0s 23ms/step - loss: 28.3097 - root_mean_squared_error: 5.3189 - val_loss: 29.1598 - val_root_mean_squared_error: 5.3987
Epoch 18/500
5/5 [==============================] - 0s 23ms/step - loss: 30.3494 - root_mean_squared_error: 5.5072 - val_loss: 28.5926 - val_root_mean_squared_error: 5.3455
Epoch 19/500
5/5 [==============================] - 0s 23ms/step - loss: 26.9082 - root_mean_squared_error: 5.1851 - val_loss: 28.0350 - val_root_mean_squared_error: 5.2931
Epoch 20/500
5/5 [==============================] - 0s 24ms/step - loss: 30.1764 - root_mean_squared_error: 5.4876 - val_loss: 29.0993 - val_root_mean_squared_error: 5.3929
Epoch 21/500
5/5 [==============================] - 0s 29ms/step - loss: 29.0847 - root_mean_squared_error: 5.3910 - val_loss: 27.6059 - val_root_mean_squared_error: 5.2527
Epoch 22/500
5/5 [==============================] - 0s 27ms/step - loss: 29.5264 - root_mean_squared_error: 5.4320 - val_loss: 29.1853 - val_root_mean_squared_error: 5.4009
Epoch 23/500
5/5 [==============================] - 0s 24ms/step - loss: 29.1371 - root_mean_squared_error: 5.3966 - val_loss: 24.8898 - val_root_mean_squared_error: 4.9879
Epoch 24/500
5/5 [==============================] - 0s 28ms/step - loss: 25.8451 - root_mean_squared_error: 5.0811 - val_loss: 28.0819 - val_root_mean_squared_error: 5.2980
Epoch 25/500
5/5 [==============================] - 0s 27ms/step - loss: 27.1873 - root_mean_squared_error: 5.2120 - val_loss: 23.8516 - val_root_mean_squared_error: 4.8820
Epoch 26/500
5/5 [==============================] - 0s 49ms/step - loss: 27.0771 - root_mean_squared_error: 5.2017 - val_loss: 25.7502 - val_root_mean_squared_error: 5.0734
Epoch 27/500
5/5 [==============================] - 0s 29ms/step - loss: 28.3087 - root_mean_squared_error: 5.3186 - val_loss: 28.3822 - val_root_mean_squared_error: 5.3264
Epoch 28/500
5/5 [==============================] - 0s 25ms/step - loss: 22.8255 - root_mean_squared_error: 4.7744 - val_loss: 25.9716 - val_root_mean_squared_error: 5.0948
Epoch 29/500
5/5 [==============================] - 0s 24ms/step - loss: 26.0508 - root_mean_squared_error: 5.1005 - val_loss: 24.9538 - val_root_mean_squared_error: 4.9938
Epoch 30/500
5/5 [==============================] - 0s 23ms/step - loss: 24.5965 - root_mean_squared_error: 4.9583 - val_loss: 24.0356 - val_root_mean_squared_error: 4.9008
Epoch 31/500
5/5 [==============================] - 0s 24ms/step - loss: 23.7836 - root_mean_squared_error: 4.8747 - val_loss: 25.1406 - val_root_mean_squared_error: 5.0129
Epoch 32/500
5/5 [==============================] - 0s 25ms/step - loss: 26.3945 - root_mean_squared_error: 5.1348 - val_loss: 24.6489 - val_root_mean_squared_error: 4.9630
Epoch 33/500
5/5 [==============================] - 0s 24ms/step - loss: 23.4255 - root_mean_squared_error: 4.8368 - val_loss: 29.2355 - val_root_mean_squared_error: 5.4056
Epoch 34/500
5/5 [==============================] - 0s 23ms/step - loss: 24.5091 - root_mean_squared_error: 4.9474 - val_loss: 23.0982 - val_root_mean_squared_error: 4.8042
Epoch 35/500
5/5 [==============================] - 0s 23ms/step - loss: 22.9675 - root_mean_squared_error: 4.7883 - val_loss: 21.5173 - val_root_mean_squared_error: 4.6370
Epoch 36/500
5/5 [==============================] - 0s 23ms/step - loss: 23.5021 - root_mean_squared_error: 4.8468 - val_loss: 21.8308 - val_root_mean_squared_error: 4.6709
Epoch 37/500
5/5 [==============================] - 0s 23ms/step - loss: 21.1085 - root_mean_squared_error: 4.5921 - val_loss: 21.0316 - val_root_mean_squared_error: 4.5840
Epoch 38/500
5/5 [==============================] - 0s 23ms/step - loss: 22.2651 - root_mean_squared_error: 4.7161 - val_loss: 22.6428 - val_root_mean_squared_error: 4.7564
Epoch 39/500
5/5 [==============================] - 0s 22ms/step - loss: 21.0495 - root_mean_squared_error: 4.5846 - val_loss: 22.0930 - val_root_mean_squared_error: 4.6993
Epoch 40/500
5/5 [==============================] - 0s 23ms/step - loss: 23.2005 - root_mean_squared_error: 4.8144 - val_loss: 19.5605 - val_root_mean_squared_error: 4.4209
Epoch 41/500
5/5 [==============================] - 0s 49ms/step - loss: 21.9923 - root_mean_squared_error: 4.6873 - val_loss: 20.4386 - val_root_mean_squared_error: 4.5191
Epoch 42/500
5/5 [==============================] - 0s 23ms/step - loss: 20.8150 - root_mean_squared_error: 4.5602 - val_loss: 21.1297 - val_root_mean_squared_error: 4.5950
Epoch 43/500
5/5 [==============================] - 0s 23ms/step - loss: 21.4027 - root_mean_squared_error: 4.6244 - val_loss: 19.8054 - val_root_mean_squared_error: 4.4487
Epoch 44/500
5/5 [==============================] - 0s 26ms/step - loss: 19.4238 - root_mean_squared_error: 4.4045 - val_loss: 18.5104 - val_root_mean_squared_error: 4.3000
Epoch 45/500
5/5 [==============================] - 0s 22ms/step - loss: 17.9783 - root_mean_squared_error: 4.2372 - val_loss: 19.1582 - val_root_mean_squared_error: 4.3753
Epoch 46/500
5/5 [==============================] - 0s 23ms/step - loss: 24.9891 - root_mean_squared_error: 4.9956 - val_loss: 18.6654 - val_root_mean_squared_error: 4.3179
Epoch 47/500
5/5 [==============================] - 0s 22ms/step - loss: 20.7825 - root_mean_squared_error: 4.5561 - val_loss: 17.8539 - val_root_mean_squared_error: 4.2236
Epoch 48/500
5/5 [==============================] - 0s 23ms/step - loss: 18.7044 - root_mean_squared_error: 4.3222 - val_loss: 17.6113 - val_root_mean_squared_error: 4.1949
Epoch 49/500
5/5 [==============================] - 0s 22ms/step - loss: 19.0035 - root_mean_squared_error: 4.3561 - val_loss: 17.8298 - val_root_mean_squared_error: 4.2209
Epoch 50/500
5/5 [==============================] - 0s 22ms/step - loss: 19.7652 - root_mean_squared_error: 4.4439 - val_loss: 16.7403 - val_root_mean_squared_error: 4.0897
Epoch 51/500
5/5 [==============================] - 0s 24ms/step - loss: 18.5954 - root_mean_squared_error: 4.3090 - val_loss: 16.8347 - val_root_mean_squared_error: 4.1010
Epoch 52/500
5/5 [==============================] - 0s 23ms/step - loss: 17.0763 - root_mean_squared_error: 4.1300 - val_loss: 18.6582 - val_root_mean_squared_error: 4.3175
Epoch 53/500
5/5 [==============================] - 0s 50ms/step - loss: 17.5386 - root_mean_squared_error: 4.1852 - val_loss: 15.9178 - val_root_mean_squared_error: 3.9875
Epoch 54/500
5/5 [==============================] - 0s 24ms/step - loss: 18.7728 - root_mean_squared_error: 4.3307 - val_loss: 16.2876 - val_root_mean_squared_error: 4.0335
Epoch 55/500
5/5 [==============================] - 0s 24ms/step - loss: 18.2994 - root_mean_squared_error: 4.2755 - val_loss: 16.1559 - val_root_mean_squared_error: 4.0168
Epoch 56/500
5/5 [==============================] - 0s 25ms/step - loss: 16.9655 - root_mean_squared_error: 4.1167 - val_loss: 15.3627 - val_root_mean_squared_error: 3.9169
Epoch 57/500
5/5 [==============================] - 0s 23ms/step - loss: 18.3151 - root_mean_squared_error: 4.2774 - val_loss: 15.0077 - val_root_mean_squared_error: 3.8717
Epoch 58/500
5/5 [==============================] - 0s 25ms/step - loss: 16.0278 - root_mean_squared_error: 4.0016 - val_loss: 17.4293 - val_root_mean_squared_error: 4.1727
Epoch 59/500
5/5 [==============================] - 0s 23ms/step - loss: 16.5838 - root_mean_squared_error: 4.0700 - val_loss: 16.1102 - val_root_mean_squared_error: 4.0112
Epoch 60/500
5/5 [==============================] - 0s 26ms/step - loss: 15.8593 - root_mean_squared_error: 3.9800 - val_loss: 16.9057 - val_root_mean_squared_error: 4.1096
Epoch 61/500
5/5 [==============================] - 0s 28ms/step - loss: 16.5961 - root_mean_squared_error: 4.0695 - val_loss: 15.1462 - val_root_mean_squared_error: 3.8895
Epoch 62/500
5/5 [==============================] - 0s 28ms/step - loss: 16.1716 - root_mean_squared_error: 4.0183 - val_loss: 14.4138 - val_root_mean_squared_error: 3.7940
Epoch 63/500
5/5 [==============================] - 0s 23ms/step - loss: 14.9571 - root_mean_squared_error: 3.8637 - val_loss: 14.4786 - val_root_mean_squared_error: 3.8025
Epoch 64/500
5/5 [==============================] - 0s 23ms/step - loss: 16.2354 - root_mean_squared_error: 4.0237 - val_loss: 13.8646 - val_root_mean_squared_error: 3.7203
Epoch 65/500
5/5 [==============================] - 0s 26ms/step - loss: 13.3850 - root_mean_squared_error: 3.6544 - val_loss: 14.3994 - val_root_mean_squared_error: 3.7920
Epoch 66/500
5/5 [==============================] - 0s 28ms/step - loss: 13.4995 - root_mean_squared_error: 3.6697 - val_loss: 13.0720 - val_root_mean_squared_error: 3.6132
Epoch 67/500
5/5 [==============================] - 0s 26ms/step - loss: 13.9107 - root_mean_squared_error: 3.7265 - val_loss: 13.6829 - val_root_mean_squared_error: 3.6965
Epoch 68/500
5/5 [==============================] - 0s 24ms/step - loss: 13.9577 - root_mean_squared_error: 3.7338 - val_loss: 13.0406 - val_root_mean_squared_error: 3.6086
Epoch 69/500
5/5 [==============================] - 0s 28ms/step - loss: 12.3432 - root_mean_squared_error: 3.5103 - val_loss: 13.4267 - val_root_mean_squared_error: 3.6613
Epoch 70/500
5/5 [==============================] - 0s 29ms/step - loss: 12.9993 - root_mean_squared_error: 3.6030 - val_loss: 12.9957 - val_root_mean_squared_error: 3.6020
Epoch 71/500
5/5 [==============================] - 0s 48ms/step - loss: 13.0387 - root_mean_squared_error: 3.6080 - val_loss: 11.9408 - val_root_mean_squared_error: 3.4530
Epoch 72/500
5/5 [==============================] - 0s 25ms/step - loss: 13.2321 - root_mean_squared_error: 3.6322 - val_loss: 11.9622 - val_root_mean_squared_error: 3.4564
Epoch 73/500
5/5 [==============================] - 0s 24ms/step - loss: 11.8500 - root_mean_squared_error: 3.4397 - val_loss: 13.5543 - val_root_mean_squared_error: 3.6791
Epoch 74/500
5/5 [==============================] - 0s 26ms/step - loss: 12.1866 - root_mean_squared_error: 3.4860 - val_loss: 11.7750 - val_root_mean_squared_error: 3.4291
Epoch 75/500
5/5 [==============================] - 0s 25ms/step - loss: 11.6645 - root_mean_squared_error: 3.4111 - val_loss: 11.2046 - val_root_mean_squared_error: 3.3441
Epoch 76/500
5/5 [==============================] - 0s 24ms/step - loss: 11.5377 - root_mean_squared_error: 3.3933 - val_loss: 9.8681 - val_root_mean_squared_error: 3.1376
Epoch 77/500
5/5 [==============================] - 0s 25ms/step - loss: 12.1589 - root_mean_squared_error: 3.4838 - val_loss: 12.1295 - val_root_mean_squared_error: 3.4800
Epoch 78/500
5/5 [==============================] - 0s 24ms/step - loss: 11.5987 - root_mean_squared_error: 3.4007 - val_loss: 10.3823 - val_root_mean_squared_error: 3.2184
Epoch 79/500
5/5 [==============================] - 0s 25ms/step - loss: 9.5063 - root_mean_squared_error: 3.0798 - val_loss: 9.2236 - val_root_mean_squared_error: 3.0333
Epoch 80/500
5/5 [==============================] - 0s 23ms/step - loss: 10.1585 - root_mean_squared_error: 3.1832 - val_loss: 8.8941 - val_root_mean_squared_error: 2.9785
Epoch 81/500
5/5 [==============================] - 0s 28ms/step - loss: 9.7241 - root_mean_squared_error: 3.1144 - val_loss: 10.5314 - val_root_mean_squared_error: 3.2427
Epoch 82/500
5/5 [==============================] - 0s 24ms/step - loss: 11.0123 - root_mean_squared_error: 3.3149 - val_loss: 11.2796 - val_root_mean_squared_error: 3.3558
Epoch 83/500
5/5 [==============================] - 0s 26ms/step - loss: 10.8654 - root_mean_squared_error: 3.2912 - val_loss: 9.6790 - val_root_mean_squared_error: 3.1080
Epoch 84/500
5/5 [==============================] - 0s 24ms/step - loss: 9.4513 - root_mean_squared_error: 3.0704 - val_loss: 9.0130 - val_root_mean_squared_error: 2.9990
Epoch 85/500
5/5 [==============================] - 0s 25ms/step - loss: 10.6010 - root_mean_squared_error: 3.2520 - val_loss: 10.9773 - val_root_mean_squared_error: 3.3104
Epoch 86/500
5/5 [==============================] - 0s 51ms/step - loss: 10.8176 - root_mean_squared_error: 3.2839 - val_loss: 8.8422 - val_root_mean_squared_error: 2.9698
Epoch 87/500
5/5 [==============================] - 0s 24ms/step - loss: 9.6039 - root_mean_squared_error: 3.0930 - val_loss: 8.4190 - val_root_mean_squared_error: 2.8985
Epoch 88/500
5/5 [==============================] - 0s 25ms/step - loss: 9.2164 - root_mean_squared_error: 3.0314 - val_loss: 9.7586 - val_root_mean_squared_error: 3.1209
Epoch 89/500
5/5 [==============================] - 0s 24ms/step - loss: 9.5353 - root_mean_squared_error: 3.0842 - val_loss: 9.0446 - val_root_mean_squared_error: 3.0050
Epoch 90/500
5/5 [==============================] - 0s 28ms/step - loss: 7.3943 - root_mean_squared_error: 2.7134 - val_loss: 8.8956 - val_root_mean_squared_error: 2.9786
Epoch 91/500
5/5 [==============================] - 0s 24ms/step - loss: 10.8292 - root_mean_squared_error: 3.2840 - val_loss: 7.4228 - val_root_mean_squared_error: 2.7201
Epoch 92/500
5/5 [==============================] - 0s 23ms/step - loss: 7.9797 - root_mean_squared_error: 2.8199 - val_loss: 7.7927 - val_root_mean_squared_error: 2.7874
Epoch 93/500
5/5 [==============================] - 0s 31ms/step - loss: 7.8615 - root_mean_squared_error: 2.7981 - val_loss: 8.0942 - val_root_mean_squared_error: 2.8413
Epoch 94/500
5/5 [==============================] - 0s 27ms/step - loss: 7.5888 - root_mean_squared_error: 2.7507 - val_loss: 7.0144 - val_root_mean_squared_error: 2.6439
Epoch 95/500
5/5 [==============================] - 0s 24ms/step - loss: 7.3221 - root_mean_squared_error: 2.6991 - val_loss: 8.8086 - val_root_mean_squared_error: 2.9646
Epoch 96/500
5/5 [==============================] - 0s 24ms/step - loss: 6.9442 - root_mean_squared_error: 2.6307 - val_loss: 6.5821 - val_root_mean_squared_error: 2.5613
Epoch 97/500
5/5 [==============================] - 0s 24ms/step - loss: 6.8026 - root_mean_squared_error: 2.6029 - val_loss: 7.7956 - val_root_mean_squared_error: 2.7885
Epoch 98/500
5/5 [==============================] - 0s 49ms/step - loss: 7.8946 - root_mean_squared_error: 2.8057 - val_loss: 5.9105 - val_root_mean_squared_error: 2.4259
Epoch 99/500
5/5 [==============================] - 0s 23ms/step - loss: 6.8761 - root_mean_squared_error: 2.6167 - val_loss: 7.9098 - val_root_mean_squared_error: 2.8085
Epoch 100/500
5/5 [==============================] - 0s 24ms/step - loss: 6.0116 - root_mean_squared_error: 2.4469 - val_loss: 6.5543 - val_root_mean_squared_error: 2.5561
Epoch 101/500
5/5 [==============================] - 0s 24ms/step - loss: 7.0128 - root_mean_squared_error: 2.6436 - val_loss: 5.6065 - val_root_mean_squared_error: 2.3629
Epoch 102/500
5/5 [==============================] - 0s 25ms/step - loss: 5.4275 - root_mean_squared_error: 2.3246 - val_loss: 6.3063 - val_root_mean_squared_error: 2.5070
Epoch 103/500
5/5 [==============================] - 0s 25ms/step - loss: 5.4632 - root_mean_squared_error: 2.3311 - val_loss: 6.7699 - val_root_mean_squared_error: 2.5975
Epoch 104/500
5/5 [==============================] - 0s 25ms/step - loss: 6.0006 - root_mean_squared_error: 2.4446 - val_loss: 5.8965 - val_root_mean_squared_error: 2.4240
Epoch 105/500
5/5 [==============================] - 0s 22ms/step - loss: 5.5511 - root_mean_squared_error: 2.3518 - val_loss: 4.9193 - val_root_mean_squared_error: 2.2131
Epoch 106/500
5/5 [==============================] - 0s 25ms/step - loss: 5.5903 - root_mean_squared_error: 2.3585 - val_loss: 5.1943 - val_root_mean_squared_error: 2.2729
Epoch 107/500
5/5 [==============================] - 0s 23ms/step - loss: 6.5486 - root_mean_squared_error: 2.5533 - val_loss: 4.9375 - val_root_mean_squared_error: 2.2168
Epoch 108/500
5/5 [==============================] - 0s 26ms/step - loss: 5.2350 - root_mean_squared_error: 2.2822 - val_loss: 5.4530 - val_root_mean_squared_error: 2.3301
Epoch 109/500
5/5 [==============================] - 0s 26ms/step - loss: 5.1418 - root_mean_squared_error: 2.2622 - val_loss: 5.2209 - val_root_mean_squared_error: 2.2808
Epoch 110/500
5/5 [==============================] - 0s 24ms/step - loss: 5.2639 - root_mean_squared_error: 2.2890 - val_loss: 4.4983 - val_root_mean_squared_error: 2.1149
Epoch 111/500
5/5 [==============================] - 0s 25ms/step - loss: 4.7013 - root_mean_squared_error: 2.1607 - val_loss: 4.7354 - val_root_mean_squared_error: 2.1712
Epoch 112/500
5/5 [==============================] - 0s 25ms/step - loss: 5.1581 - root_mean_squared_error: 2.2635 - val_loss: 4.9248 - val_root_mean_squared_error: 2.2136
Epoch 113/500
5/5 [==============================] - 0s 25ms/step - loss: 5.4640 - root_mean_squared_error: 2.3308 - val_loss: 4.0738 - val_root_mean_squared_error: 2.0122
Epoch 114/500
5/5 [==============================] - 0s 25ms/step - loss: 4.3722 - root_mean_squared_error: 2.0852 - val_loss: 4.4029 - val_root_mean_squared_error: 2.0919
Epoch 115/500
5/5 [==============================] - 0s 24ms/step - loss: 4.5255 - root_mean_squared_error: 2.1208 - val_loss: 3.5650 - val_root_mean_squared_error: 1.8808
Epoch 116/500
5/5 [==============================] - 0s 49ms/step - loss: 4.1281 - root_mean_squared_error: 2.0260 - val_loss: 3.6151 - val_root_mean_squared_error: 1.8946
Epoch 117/500
5/5 [==============================] - 0s 26ms/step - loss: 3.7002 - root_mean_squared_error: 1.9170 - val_loss: 3.4885 - val_root_mean_squared_error: 1.8612
Epoch 118/500
5/5 [==============================] - 0s 24ms/step - loss: 3.8082 - root_mean_squared_error: 1.9452 - val_loss: 3.6706 - val_root_mean_squared_error: 1.9089
Epoch 119/500
5/5 [==============================] - 0s 23ms/step - loss: 4.2802 - root_mean_squared_error: 2.0626 - val_loss: 3.4468 - val_root_mean_squared_error: 1.8492
Epoch 120/500
5/5 [==============================] - 0s 24ms/step - loss: 3.8495 - root_mean_squared_error: 1.9542 - val_loss: 3.7354 - val_root_mean_squared_error: 1.9270
Epoch 121/500
5/5 [==============================] - 0s 25ms/step - loss: 3.1704 - root_mean_squared_error: 1.7736 - val_loss: 3.2792 - val_root_mean_squared_error: 1.8025
Epoch 122/500
5/5 [==============================] - 0s 23ms/step - loss: 2.9830 - root_mean_squared_error: 1.7193 - val_loss: 2.8080 - val_root_mean_squared_error: 1.6674
Epoch 123/500
5/5 [==============================] - 0s 29ms/step - loss: 3.0630 - root_mean_squared_error: 1.7425 - val_loss: 2.7252 - val_root_mean_squared_error: 1.6431
Epoch 124/500
5/5 [==============================] - 0s 30ms/step - loss: 3.0082 - root_mean_squared_error: 1.7263 - val_loss: 3.2043 - val_root_mean_squared_error: 1.7834
Epoch 125/500
5/5 [==============================] - 0s 24ms/step - loss: 3.0392 - root_mean_squared_error: 1.7356 - val_loss: 2.7903 - val_root_mean_squared_error: 1.6618
Epoch 126/500
5/5 [==============================] - 0s 25ms/step - loss: 3.0374 - root_mean_squared_error: 1.7347 - val_loss: 3.0553 - val_root_mean_squared_error: 1.7413
Epoch 127/500
5/5 [==============================] - 0s 28ms/step - loss: 2.8584 - root_mean_squared_error: 1.6831 - val_loss: 3.7921 - val_root_mean_squared_error: 1.9414
Epoch 128/500
5/5 [==============================] - 0s 24ms/step - loss: 2.5141 - root_mean_squared_error: 1.5762 - val_loss: 2.9707 - val_root_mean_squared_error: 1.7158
Epoch 129/500
5/5 [==============================] - 0s 25ms/step - loss: 2.3010 - root_mean_squared_error: 1.5074 - val_loss: 2.8559 - val_root_mean_squared_error: 1.6823
Epoch 130/500
5/5 [==============================] - 0s 23ms/step - loss: 2.4480 - root_mean_squared_error: 1.5539 - val_loss: 2.2670 - val_root_mean_squared_error: 1.4965
Epoch 131/500
5/5 [==============================] - 0s 50ms/step - loss: 2.1190 - root_mean_squared_error: 1.4460 - val_loss: 2.1149 - val_root_mean_squared_error: 1.4438
Epoch 132/500
5/5 [==============================] - 0s 25ms/step - loss: 2.4441 - root_mean_squared_error: 1.5553 - val_loss: 2.1703 - val_root_mean_squared_error: 1.4644
Epoch 133/500
5/5 [==============================] - 0s 24ms/step - loss: 2.1058 - root_mean_squared_error: 1.4416 - val_loss: 1.8681 - val_root_mean_squared_error: 1.3561
Epoch 134/500
5/5 [==============================] - 0s 23ms/step - loss: 2.5839 - root_mean_squared_error: 1.6000 - val_loss: 1.7737 - val_root_mean_squared_error: 1.3235
Epoch 135/500
5/5 [==============================] - 0s 23ms/step - loss: 2.0477 - root_mean_squared_error: 1.4202 - val_loss: 1.8301 - val_root_mean_squared_error: 1.3419
Epoch 136/500
5/5 [==============================] - 0s 28ms/step - loss: 1.7564 - root_mean_squared_error: 1.3158 - val_loss: 2.5573 - val_root_mean_squared_error: 1.5918
Epoch 137/500
5/5 [==============================] - 0s 23ms/step - loss: 1.9168 - root_mean_squared_error: 1.3754 - val_loss: 1.9317 - val_root_mean_squared_error: 1.3816
Epoch 138/500
5/5 [==============================] - 0s 24ms/step - loss: 1.7714 - root_mean_squared_error: 1.3205 - val_loss: 1.5942 - val_root_mean_squared_error: 1.2520
Epoch 139/500
5/5 [==============================] - 0s 23ms/step - loss: 1.7672 - root_mean_squared_error: 1.3199 - val_loss: 1.6904 - val_root_mean_squared_error: 1.2887
Epoch 140/500
5/5 [==============================] - 0s 23ms/step - loss: 1.8368 - root_mean_squared_error: 1.3444 - val_loss: 1.9556 - val_root_mean_squared_error: 1.3897
Epoch 141/500
5/5 [==============================] - 0s 23ms/step - loss: 1.4516 - root_mean_squared_error: 1.1929 - val_loss: 1.4540 - val_root_mean_squared_error: 1.1947
Epoch 142/500
5/5 [==============================] - 0s 23ms/step - loss: 1.6615 - root_mean_squared_error: 1.2798 - val_loss: 1.5250 - val_root_mean_squared_error: 1.2224
Epoch 143/500
5/5 [==============================] - 0s 48ms/step - loss: 1.3321 - root_mean_squared_error: 1.1395 - val_loss: 1.4206 - val_root_mean_squared_error: 1.1786
Epoch 144/500
5/5 [==============================] - 0s 22ms/step - loss: 1.1701 - root_mean_squared_error: 1.0681 - val_loss: 1.3812 - val_root_mean_squared_error: 1.1640
Epoch 145/500
5/5 [==============================] - 0s 23ms/step - loss: 1.5458 - root_mean_squared_error: 1.2301 - val_loss: 1.3235 - val_root_mean_squared_error: 1.1391
Epoch 146/500
5/5 [==============================] - 0s 23ms/step - loss: 1.5435 - root_mean_squared_error: 1.2315 - val_loss: 1.2045 - val_root_mean_squared_error: 1.0840
Epoch 147/500
5/5 [==============================] - 0s 25ms/step - loss: 1.0523 - root_mean_squared_error: 1.0093 - val_loss: 1.0540 - val_root_mean_squared_error: 1.0134
Epoch 148/500
5/5 [==============================] - 0s 23ms/step - loss: 1.0261 - root_mean_squared_error: 0.9991 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0627
Epoch 149/500
5/5 [==============================] - 0s 23ms/step - loss: 1.2111 - root_mean_squared_error: 1.0871 - val_loss: 1.2300 - val_root_mean_squared_error: 1.0975
Epoch 150/500
5/5 [==============================] - 0s 22ms/step - loss: 1.2639 - root_mean_squared_error: 1.1117 - val_loss: 1.1289 - val_root_mean_squared_error: 1.0526
Epoch 151/500
5/5 [==============================] - 0s 24ms/step - loss: 1.3348 - root_mean_squared_error: 1.1426 - val_loss: 1.0639 - val_root_mean_squared_error: 1.0200
Epoch 152/500
5/5 [==============================] - 0s 23ms/step - loss: 0.9300 - root_mean_squared_error: 0.9481 - val_loss: 1.1141 - val_root_mean_squared_error: 1.0422
Epoch 153/500
5/5 [==============================] - 0s 23ms/step - loss: 0.9623 - root_mean_squared_error: 0.9653 - val_loss: 1.0172 - val_root_mean_squared_error: 0.9947
Epoch 154/500
5/5 [==============================] - 0s 23ms/step - loss: 0.9852 - root_mean_squared_error: 0.9783 - val_loss: 0.9282 - val_root_mean_squared_error: 0.9483
Epoch 155/500
5/5 [==============================] - 0s 24ms/step - loss: 0.9206 - root_mean_squared_error: 0.9452 - val_loss: 0.9001 - val_root_mean_squared_error: 0.9345
Epoch 156/500
5/5 [==============================] - 0s 23ms/step - loss: 1.0529 - root_mean_squared_error: 1.0143 - val_loss: 0.9266 - val_root_mean_squared_error: 0.9481
Epoch 157/500
5/5 [==============================] - 0s 28ms/step - loss: 0.9866 - root_mean_squared_error: 0.9757 - val_loss: 0.8818 - val_root_mean_squared_error: 0.9250
Epoch 158/500
5/5 [==============================] - 0s 29ms/step - loss: 0.9407 - root_mean_squared_error: 0.9533 - val_loss: 0.9068 - val_root_mean_squared_error: 0.9348
Epoch 159/500
5/5 [==============================] - 0s 28ms/step - loss: 0.8764 - root_mean_squared_error: 0.9201 - val_loss: 0.8394 - val_root_mean_squared_error: 0.8996
Epoch 160/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8423 - root_mean_squared_error: 0.9005 - val_loss: 0.9926 - val_root_mean_squared_error: 0.9809
Epoch 161/500
5/5 [==============================] - 0s 25ms/step - loss: 0.9209 - root_mean_squared_error: 0.9441 - val_loss: 0.9548 - val_root_mean_squared_error: 0.9631
Epoch 162/500
5/5 [==============================] - 0s 53ms/step - loss: 0.8409 - root_mean_squared_error: 0.8986 - val_loss: 0.7863 - val_root_mean_squared_error: 0.8691
Epoch 163/500
5/5 [==============================] - 0s 28ms/step - loss: 0.8465 - root_mean_squared_error: 0.9037 - val_loss: 0.8578 - val_root_mean_squared_error: 0.9128
Epoch 164/500
5/5 [==============================] - 0s 28ms/step - loss: 1.0945 - root_mean_squared_error: 1.0295 - val_loss: 0.7973 - val_root_mean_squared_error: 0.8771
Epoch 165/500
5/5 [==============================] - 0s 25ms/step - loss: 0.8100 - root_mean_squared_error: 0.8847 - val_loss: 0.7997 - val_root_mean_squared_error: 0.8799
Epoch 166/500
5/5 [==============================] - 0s 29ms/step - loss: 0.7677 - root_mean_squared_error: 0.8590 - val_loss: 0.8497 - val_root_mean_squared_error: 0.9044
Epoch 167/500
5/5 [==============================] - 0s 26ms/step - loss: 0.8083 - root_mean_squared_error: 0.8821 - val_loss: 0.8104 - val_root_mean_squared_error: 0.8830
Epoch 168/500
5/5 [==============================] - 0s 28ms/step - loss: 0.8437 - root_mean_squared_error: 0.9018 - val_loss: 0.8208 - val_root_mean_squared_error: 0.8897
Epoch 169/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7993 - root_mean_squared_error: 0.8795 - val_loss: 0.8277 - val_root_mean_squared_error: 0.8944
Epoch 170/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8578 - root_mean_squared_error: 0.9107 - val_loss: 0.7953 - val_root_mean_squared_error: 0.8759
Epoch 171/500
5/5 [==============================] - 0s 23ms/step - loss: 0.8132 - root_mean_squared_error: 0.8851 - val_loss: 0.8133 - val_root_mean_squared_error: 0.8865
Epoch 172/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8388 - root_mean_squared_error: 0.8973 - val_loss: 0.8120 - val_root_mean_squared_error: 0.8856
Epoch 173/500
5/5 [==============================] - 0s 26ms/step - loss: 0.8318 - root_mean_squared_error: 0.8975 - val_loss: 0.8447 - val_root_mean_squared_error: 0.9026
Epoch 174/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7910 - root_mean_squared_error: 0.8739 - val_loss: 0.8007 - val_root_mean_squared_error: 0.8776
Epoch 175/500
5/5 [==============================] - 0s 27ms/step - loss: 0.8191 - root_mean_squared_error: 0.8897 - val_loss: 0.8421 - val_root_mean_squared_error: 0.9023
Epoch 176/500
5/5 [==============================] - 0s 49ms/step - loss: 0.8122 - root_mean_squared_error: 0.8854 - val_loss: 0.8001 - val_root_mean_squared_error: 0.8793
Epoch 177/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8937 - root_mean_squared_error: 0.9269 - val_loss: 0.8157 - val_root_mean_squared_error: 0.8891
Epoch 178/500
5/5 [==============================] - 0s 25ms/step - loss: 0.8506 - root_mean_squared_error: 0.9041 - val_loss: 0.7962 - val_root_mean_squared_error: 0.8749
Epoch 179/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8242 - root_mean_squared_error: 0.8905 - val_loss: 0.7926 - val_root_mean_squared_error: 0.8747
Epoch 180/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7267 - root_mean_squared_error: 0.8366 - val_loss: 0.8144 - val_root_mean_squared_error: 0.8861
Epoch 181/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8167 - root_mean_squared_error: 0.8867 - val_loss: 0.8432 - val_root_mean_squared_error: 0.9051
Epoch 182/500
5/5 [==============================] - 0s 25ms/step - loss: 0.8154 - root_mean_squared_error: 0.8872 - val_loss: 0.8372 - val_root_mean_squared_error: 0.9018
Epoch 183/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7893 - root_mean_squared_error: 0.8712 - val_loss: 0.7816 - val_root_mean_squared_error: 0.8698
Epoch 184/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8039 - root_mean_squared_error: 0.8806 - val_loss: 0.7965 - val_root_mean_squared_error: 0.8771
Epoch 185/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7799 - root_mean_squared_error: 0.8650 - val_loss: 0.7970 - val_root_mean_squared_error: 0.8774
Epoch 186/500
5/5 [==============================] - 0s 26ms/step - loss: 0.8292 - root_mean_squared_error: 0.8942 - val_loss: 0.8043 - val_root_mean_squared_error: 0.8831
Epoch 187/500
5/5 [==============================] - 0s 29ms/step - loss: 0.8029 - root_mean_squared_error: 0.8778 - val_loss: 0.8145 - val_root_mean_squared_error: 0.8839
Epoch 188/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7981 - root_mean_squared_error: 0.8775 - val_loss: 0.7802 - val_root_mean_squared_error: 0.8676
Epoch 189/500
5/5 [==============================] - 0s 50ms/step - loss: 0.7970 - root_mean_squared_error: 0.8774 - val_loss: 0.7976 - val_root_mean_squared_error: 0.8761
Epoch 190/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7405 - root_mean_squared_error: 0.8431 - val_loss: 0.7691 - val_root_mean_squared_error: 0.8610
Epoch 191/500
5/5 [==============================] - 0s 23ms/step - loss: 0.8645 - root_mean_squared_error: 0.9145 - val_loss: 0.7639 - val_root_mean_squared_error: 0.8579
Epoch 192/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7683 - root_mean_squared_error: 0.8597 - val_loss: 0.7644 - val_root_mean_squared_error: 0.8576
Epoch 193/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7167 - root_mean_squared_error: 0.8291 - val_loss: 0.8145 - val_root_mean_squared_error: 0.8862
Epoch 194/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7789 - root_mean_squared_error: 0.8661 - val_loss: 0.8025 - val_root_mean_squared_error: 0.8780
Epoch 195/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7479 - root_mean_squared_error: 0.8499 - val_loss: 0.7768 - val_root_mean_squared_error: 0.8636
Epoch 196/500
5/5 [==============================] - 0s 26ms/step - loss: 0.7265 - root_mean_squared_error: 0.8365 - val_loss: 0.7940 - val_root_mean_squared_error: 0.8755
Epoch 197/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8197 - root_mean_squared_error: 0.8889 - val_loss: 0.7937 - val_root_mean_squared_error: 0.8763
Epoch 198/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7608 - root_mean_squared_error: 0.8578 - val_loss: 0.8331 - val_root_mean_squared_error: 0.8962
Epoch 199/500
5/5 [==============================] - 0s 25ms/step - loss: 0.8011 - root_mean_squared_error: 0.8791 - val_loss: 0.7862 - val_root_mean_squared_error: 0.8678
Epoch 200/500
5/5 [==============================] - 0s 25ms/step - loss: 0.8080 - root_mean_squared_error: 0.8834 - val_loss: 0.7610 - val_root_mean_squared_error: 0.8561
Epoch 201/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7939 - root_mean_squared_error: 0.8734 - val_loss: 0.8291 - val_root_mean_squared_error: 0.8972
Epoch 202/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7268 - root_mean_squared_error: 0.8385 - val_loss: 0.8059 - val_root_mean_squared_error: 0.8823
Epoch 203/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7283 - root_mean_squared_error: 0.8368 - val_loss: 0.7729 - val_root_mean_squared_error: 0.8624
Epoch 204/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8576 - root_mean_squared_error: 0.9099 - val_loss: 0.7689 - val_root_mean_squared_error: 0.8625
Epoch 205/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8417 - root_mean_squared_error: 0.9041 - val_loss: 0.7476 - val_root_mean_squared_error: 0.8473
Epoch 206/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7707 - root_mean_squared_error: 0.8619 - val_loss: 0.7783 - val_root_mean_squared_error: 0.8671
Epoch 207/500
5/5 [==============================] - 0s 46ms/step - loss: 0.7557 - root_mean_squared_error: 0.8524 - val_loss: 0.7857 - val_root_mean_squared_error: 0.8717
Epoch 208/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7820 - root_mean_squared_error: 0.8691 - val_loss: 0.7714 - val_root_mean_squared_error: 0.8634
Epoch 209/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8418 - root_mean_squared_error: 0.9022 - val_loss: 0.7394 - val_root_mean_squared_error: 0.8440
Epoch 210/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7295 - root_mean_squared_error: 0.8390 - val_loss: 0.7565 - val_root_mean_squared_error: 0.8524
Epoch 211/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7619 - root_mean_squared_error: 0.8570 - val_loss: 0.7692 - val_root_mean_squared_error: 0.8616
Epoch 212/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7506 - root_mean_squared_error: 0.8503 - val_loss: 0.7815 - val_root_mean_squared_error: 0.8668
Epoch 213/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7751 - root_mean_squared_error: 0.8635 - val_loss: 0.7475 - val_root_mean_squared_error: 0.8484
Epoch 214/500
5/5 [==============================] - 0s 23ms/step - loss: 0.8638 - root_mean_squared_error: 0.9151 - val_loss: 0.7849 - val_root_mean_squared_error: 0.8703
Epoch 215/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7738 - root_mean_squared_error: 0.8632 - val_loss: 0.7959 - val_root_mean_squared_error: 0.8774
Epoch 216/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8304 - root_mean_squared_error: 0.8951 - val_loss: 0.7724 - val_root_mean_squared_error: 0.8639
Epoch 217/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7390 - root_mean_squared_error: 0.8449 - val_loss: 0.7639 - val_root_mean_squared_error: 0.8593
Epoch 218/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6980 - root_mean_squared_error: 0.8179 - val_loss: 0.7427 - val_root_mean_squared_error: 0.8455
Epoch 219/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7427 - root_mean_squared_error: 0.8424 - val_loss: 0.7792 - val_root_mean_squared_error: 0.8678
Epoch 220/500
5/5 [==============================] - 0s 27ms/step - loss: 0.8164 - root_mean_squared_error: 0.8860 - val_loss: 0.7272 - val_root_mean_squared_error: 0.8332
Epoch 221/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7912 - root_mean_squared_error: 0.8720 - val_loss: 0.7275 - val_root_mean_squared_error: 0.8376
Epoch 222/500
5/5 [==============================] - 0s 51ms/step - loss: 0.8228 - root_mean_squared_error: 0.8919 - val_loss: 0.7485 - val_root_mean_squared_error: 0.8509
Epoch 223/500
5/5 [==============================] - 0s 25ms/step - loss: 0.8092 - root_mean_squared_error: 0.8857 - val_loss: 0.7568 - val_root_mean_squared_error: 0.8537
Epoch 224/500
5/5 [==============================] - 0s 23ms/step - loss: 0.8271 - root_mean_squared_error: 0.8942 - val_loss: 0.7364 - val_root_mean_squared_error: 0.8412
Epoch 225/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7333 - root_mean_squared_error: 0.8413 - val_loss: 0.7679 - val_root_mean_squared_error: 0.8591
Epoch 226/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7393 - root_mean_squared_error: 0.8450 - val_loss: 0.7247 - val_root_mean_squared_error: 0.8364
Epoch 227/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6748 - root_mean_squared_error: 0.8039 - val_loss: 0.7312 - val_root_mean_squared_error: 0.8387
Epoch 228/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7311 - root_mean_squared_error: 0.8400 - val_loss: 0.7805 - val_root_mean_squared_error: 0.8657
Epoch 229/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7548 - root_mean_squared_error: 0.8531 - val_loss: 0.7258 - val_root_mean_squared_error: 0.8358
Epoch 230/500
5/5 [==============================] - 0s 21ms/step - loss: 0.7415 - root_mean_squared_error: 0.8451 - val_loss: 0.7924 - val_root_mean_squared_error: 0.8708
Epoch 231/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6691 - root_mean_squared_error: 0.8002 - val_loss: 0.7403 - val_root_mean_squared_error: 0.8453
Epoch 232/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7372 - root_mean_squared_error: 0.8410 - val_loss: 0.7428 - val_root_mean_squared_error: 0.8454
Epoch 233/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7712 - root_mean_squared_error: 0.8620 - val_loss: 0.7161 - val_root_mean_squared_error: 0.8287
Epoch 234/500
5/5 [==============================] - 0s 49ms/step - loss: 0.7626 - root_mean_squared_error: 0.8565 - val_loss: 0.8201 - val_root_mean_squared_error: 0.8902
Epoch 235/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7941 - root_mean_squared_error: 0.8767 - val_loss: 0.7443 - val_root_mean_squared_error: 0.8471
Epoch 236/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7564 - root_mean_squared_error: 0.8534 - val_loss: 0.7774 - val_root_mean_squared_error: 0.8671
Epoch 237/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7528 - root_mean_squared_error: 0.8496 - val_loss: 0.7432 - val_root_mean_squared_error: 0.8460
Epoch 238/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7079 - root_mean_squared_error: 0.8247 - val_loss: 0.7340 - val_root_mean_squared_error: 0.8380
Epoch 239/500
5/5 [==============================] - 0s 26ms/step - loss: 0.7300 - root_mean_squared_error: 0.8374 - val_loss: 0.7220 - val_root_mean_squared_error: 0.8322
Epoch 240/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7547 - root_mean_squared_error: 0.8549 - val_loss: 0.7624 - val_root_mean_squared_error: 0.8581
Epoch 241/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6694 - root_mean_squared_error: 0.8010 - val_loss: 0.7355 - val_root_mean_squared_error: 0.8384
Epoch 242/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7242 - root_mean_squared_error: 0.8367 - val_loss: 0.7731 - val_root_mean_squared_error: 0.8628
Epoch 243/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7459 - root_mean_squared_error: 0.8472 - val_loss: 0.7190 - val_root_mean_squared_error: 0.8331
Epoch 244/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6912 - root_mean_squared_error: 0.8138 - val_loss: 0.7832 - val_root_mean_squared_error: 0.8686
Epoch 245/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7481 - root_mean_squared_error: 0.8473 - val_loss: 0.7630 - val_root_mean_squared_error: 0.8573
Epoch 246/500
5/5 [==============================] - 0s 27ms/step - loss: 0.6978 - root_mean_squared_error: 0.8160 - val_loss: 0.7327 - val_root_mean_squared_error: 0.8413
Epoch 247/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6919 - root_mean_squared_error: 0.8135 - val_loss: 0.7156 - val_root_mean_squared_error: 0.8291
Epoch 248/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6894 - root_mean_squared_error: 0.8128 - val_loss: 0.7740 - val_root_mean_squared_error: 0.8619
Epoch 249/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7616 - root_mean_squared_error: 0.8552 - val_loss: 0.7760 - val_root_mean_squared_error: 0.8663
Epoch 250/500
5/5 [==============================] - 0s 22ms/step - loss: 0.8407 - root_mean_squared_error: 0.9018 - val_loss: 0.7372 - val_root_mean_squared_error: 0.8428
Epoch 251/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7895 - root_mean_squared_error: 0.8726 - val_loss: 0.7285 - val_root_mean_squared_error: 0.8380
Epoch 252/500
5/5 [==============================] - 0s 51ms/step - loss: 0.7048 - root_mean_squared_error: 0.8207 - val_loss: 0.7689 - val_root_mean_squared_error: 0.8609
Epoch 253/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7606 - root_mean_squared_error: 0.8559 - val_loss: 0.7764 - val_root_mean_squared_error: 0.8694
Epoch 254/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7185 - root_mean_squared_error: 0.8336 - val_loss: 0.7594 - val_root_mean_squared_error: 0.8553
Epoch 255/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6937 - root_mean_squared_error: 0.8166 - val_loss: 0.7762 - val_root_mean_squared_error: 0.8677
Epoch 256/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7527 - root_mean_squared_error: 0.8514 - val_loss: 0.7772 - val_root_mean_squared_error: 0.8633
Epoch 257/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6744 - root_mean_squared_error: 0.8049 - val_loss: 0.7088 - val_root_mean_squared_error: 0.8255
Epoch 258/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6624 - root_mean_squared_error: 0.7957 - val_loss: 0.7535 - val_root_mean_squared_error: 0.8537
Epoch 259/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7274 - root_mean_squared_error: 0.8369 - val_loss: 0.7559 - val_root_mean_squared_error: 0.8524
Epoch 260/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6586 - root_mean_squared_error: 0.7920 - val_loss: 0.7221 - val_root_mean_squared_error: 0.8336
Epoch 261/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7214 - root_mean_squared_error: 0.8335 - val_loss: 0.7118 - val_root_mean_squared_error: 0.8278
Epoch 262/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7559 - root_mean_squared_error: 0.8521 - val_loss: 0.7124 - val_root_mean_squared_error: 0.8250
Epoch 263/500
5/5 [==============================] - 0s 26ms/step - loss: 0.7975 - root_mean_squared_error: 0.8792 - val_loss: 0.7326 - val_root_mean_squared_error: 0.8392
Epoch 264/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7621 - root_mean_squared_error: 0.8567 - val_loss: 0.6924 - val_root_mean_squared_error: 0.8132
Epoch 265/500
5/5 [==============================] - 0s 24ms/step - loss: 0.8164 - root_mean_squared_error: 0.8901 - val_loss: 0.7495 - val_root_mean_squared_error: 0.8500
Epoch 266/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7177 - root_mean_squared_error: 0.8287 - val_loss: 0.7154 - val_root_mean_squared_error: 0.8301
Epoch 267/500
5/5 [==============================] - 0s 52ms/step - loss: 0.6838 - root_mean_squared_error: 0.8128 - val_loss: 0.7271 - val_root_mean_squared_error: 0.8357
Epoch 268/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6907 - root_mean_squared_error: 0.8147 - val_loss: 0.7028 - val_root_mean_squared_error: 0.8238
Epoch 269/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7429 - root_mean_squared_error: 0.8468 - val_loss: 0.7082 - val_root_mean_squared_error: 0.8251
Epoch 270/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7156 - root_mean_squared_error: 0.8269 - val_loss: 0.7003 - val_root_mean_squared_error: 0.8185
Epoch 271/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6744 - root_mean_squared_error: 0.8023 - val_loss: 0.7447 - val_root_mean_squared_error: 0.8472
Epoch 272/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7493 - root_mean_squared_error: 0.8494 - val_loss: 0.7058 - val_root_mean_squared_error: 0.8220
Epoch 273/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6362 - root_mean_squared_error: 0.7798 - val_loss: 0.7115 - val_root_mean_squared_error: 0.8264
Epoch 274/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7507 - root_mean_squared_error: 0.8522 - val_loss: 0.7288 - val_root_mean_squared_error: 0.8364
Epoch 275/500
5/5 [==============================] - 0s 30ms/step - loss: 0.6771 - root_mean_squared_error: 0.8072 - val_loss: 0.7186 - val_root_mean_squared_error: 0.8306
Epoch 276/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7164 - root_mean_squared_error: 0.8300 - val_loss: 0.7539 - val_root_mean_squared_error: 0.8537
Epoch 277/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6832 - root_mean_squared_error: 0.8085 - val_loss: 0.7163 - val_root_mean_squared_error: 0.8313
Epoch 278/500
5/5 [==============================] - 0s 21ms/step - loss: 0.6483 - root_mean_squared_error: 0.7866 - val_loss: 0.7756 - val_root_mean_squared_error: 0.8670
Epoch 279/500
5/5 [==============================] - 0s 52ms/step - loss: 0.7171 - root_mean_squared_error: 0.8302 - val_loss: 0.7686 - val_root_mean_squared_error: 0.8603
Epoch 280/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6970 - root_mean_squared_error: 0.8184 - val_loss: 0.6715 - val_root_mean_squared_error: 0.8021
Epoch 281/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7498 - root_mean_squared_error: 0.8500 - val_loss: 0.6808 - val_root_mean_squared_error: 0.8086
Epoch 282/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7235 - root_mean_squared_error: 0.8343 - val_loss: 0.6793 - val_root_mean_squared_error: 0.8059
Epoch 283/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7404 - root_mean_squared_error: 0.8436 - val_loss: 0.6993 - val_root_mean_squared_error: 0.8210
Epoch 284/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7441 - root_mean_squared_error: 0.8441 - val_loss: 0.6685 - val_root_mean_squared_error: 0.8000
Epoch 285/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6994 - root_mean_squared_error: 0.8199 - val_loss: 0.7613 - val_root_mean_squared_error: 0.8571
Epoch 286/500
5/5 [==============================] - 0s 23ms/step - loss: 0.8160 - root_mean_squared_error: 0.8900 - val_loss: 0.7254 - val_root_mean_squared_error: 0.8336
Epoch 287/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6605 - root_mean_squared_error: 0.7941 - val_loss: 0.6856 - val_root_mean_squared_error: 0.8132
Epoch 288/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6544 - root_mean_squared_error: 0.7953 - val_loss: 0.6673 - val_root_mean_squared_error: 0.7995
Epoch 289/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7139 - root_mean_squared_error: 0.8273 - val_loss: 0.6888 - val_root_mean_squared_error: 0.8143
Epoch 290/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6896 - root_mean_squared_error: 0.8103 - val_loss: 0.7122 - val_root_mean_squared_error: 0.8271
Epoch 291/500
5/5 [==============================] - 0s 30ms/step - loss: 0.6413 - root_mean_squared_error: 0.7829 - val_loss: 0.7196 - val_root_mean_squared_error: 0.8319
Epoch 292/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7062 - root_mean_squared_error: 0.8226 - val_loss: 0.6775 - val_root_mean_squared_error: 0.8066
Epoch 293/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6935 - root_mean_squared_error: 0.8134 - val_loss: 0.6493 - val_root_mean_squared_error: 0.7882
Epoch 294/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6277 - root_mean_squared_error: 0.7751 - val_loss: 0.6772 - val_root_mean_squared_error: 0.8056
Epoch 295/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7432 - root_mean_squared_error: 0.8469 - val_loss: 0.7124 - val_root_mean_squared_error: 0.8295
Epoch 296/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6976 - root_mean_squared_error: 0.8182 - val_loss: 0.7118 - val_root_mean_squared_error: 0.8290
Epoch 297/500
5/5 [==============================] - 0s 51ms/step - loss: 0.7564 - root_mean_squared_error: 0.8544 - val_loss: 0.8055 - val_root_mean_squared_error: 0.8823
Epoch 298/500
5/5 [==============================] - 0s 26ms/step - loss: 0.7469 - root_mean_squared_error: 0.8484 - val_loss: 0.7155 - val_root_mean_squared_error: 0.8307
Epoch 299/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6460 - root_mean_squared_error: 0.7854 - val_loss: 0.6947 - val_root_mean_squared_error: 0.8177
Epoch 300/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7251 - root_mean_squared_error: 0.8329 - val_loss: 0.7230 - val_root_mean_squared_error: 0.8339
Epoch 301/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7180 - root_mean_squared_error: 0.8318 - val_loss: 0.7314 - val_root_mean_squared_error: 0.8388
Epoch 302/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6883 - root_mean_squared_error: 0.8149 - val_loss: 0.7025 - val_root_mean_squared_error: 0.8225
Epoch 303/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7122 - root_mean_squared_error: 0.8271 - val_loss: 0.7275 - val_root_mean_squared_error: 0.8356
Epoch 304/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6439 - root_mean_squared_error: 0.7849 - val_loss: 0.6779 - val_root_mean_squared_error: 0.8078
Epoch 305/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7199 - root_mean_squared_error: 0.8306 - val_loss: 0.7096 - val_root_mean_squared_error: 0.8244
Epoch 306/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6864 - root_mean_squared_error: 0.8133 - val_loss: 0.7185 - val_root_mean_squared_error: 0.8297
Epoch 307/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6548 - root_mean_squared_error: 0.7914 - val_loss: 0.6945 - val_root_mean_squared_error: 0.8193
Epoch 308/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6788 - root_mean_squared_error: 0.8079 - val_loss: 0.6673 - val_root_mean_squared_error: 0.7976
Epoch 309/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7204 - root_mean_squared_error: 0.8324 - val_loss: 0.6690 - val_root_mean_squared_error: 0.8013
Epoch 310/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6927 - root_mean_squared_error: 0.8136 - val_loss: 0.7401 - val_root_mean_squared_error: 0.8441
Epoch 311/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7054 - root_mean_squared_error: 0.8209 - val_loss: 0.7119 - val_root_mean_squared_error: 0.8237
Epoch 312/500
5/5 [==============================] - 0s 51ms/step - loss: 0.7186 - root_mean_squared_error: 0.8305 - val_loss: 0.6801 - val_root_mean_squared_error: 0.8085
Epoch 313/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7315 - root_mean_squared_error: 0.8362 - val_loss: 0.6644 - val_root_mean_squared_error: 0.7969
Epoch 314/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7217 - root_mean_squared_error: 0.8339 - val_loss: 0.7076 - val_root_mean_squared_error: 0.8243
Epoch 315/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6926 - root_mean_squared_error: 0.8160 - val_loss: 0.6762 - val_root_mean_squared_error: 0.8038
Epoch 316/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7188 - root_mean_squared_error: 0.8297 - val_loss: 0.6881 - val_root_mean_squared_error: 0.8108
Epoch 317/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6969 - root_mean_squared_error: 0.8169 - val_loss: 0.6981 - val_root_mean_squared_error: 0.8183
Epoch 318/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6659 - root_mean_squared_error: 0.8003 - val_loss: 0.7295 - val_root_mean_squared_error: 0.8370
Epoch 319/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7042 - root_mean_squared_error: 0.8249 - val_loss: 0.6996 - val_root_mean_squared_error: 0.8170
Epoch 320/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7121 - root_mean_squared_error: 0.8276 - val_loss: 0.7173 - val_root_mean_squared_error: 0.8311
Epoch 321/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7089 - root_mean_squared_error: 0.8254 - val_loss: 0.6441 - val_root_mean_squared_error: 0.7845
Epoch 322/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7077 - root_mean_squared_error: 0.8232 - val_loss: 0.6828 - val_root_mean_squared_error: 0.8079
Epoch 323/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6486 - root_mean_squared_error: 0.7870 - val_loss: 0.6958 - val_root_mean_squared_error: 0.8160
Epoch 324/500
5/5 [==============================] - 0s 52ms/step - loss: 0.7725 - root_mean_squared_error: 0.8626 - val_loss: 0.6914 - val_root_mean_squared_error: 0.8156
Epoch 325/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6871 - root_mean_squared_error: 0.8105 - val_loss: 0.6671 - val_root_mean_squared_error: 0.8009
Epoch 326/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7224 - root_mean_squared_error: 0.8343 - val_loss: 0.6768 - val_root_mean_squared_error: 0.8044
Epoch 327/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6915 - root_mean_squared_error: 0.8148 - val_loss: 0.7427 - val_root_mean_squared_error: 0.8465
Epoch 328/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7080 - root_mean_squared_error: 0.8262 - val_loss: 0.6833 - val_root_mean_squared_error: 0.8113
Epoch 329/500
5/5 [==============================] - 0s 21ms/step - loss: 0.6816 - root_mean_squared_error: 0.8076 - val_loss: 0.6864 - val_root_mean_squared_error: 0.8102
Epoch 330/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6512 - root_mean_squared_error: 0.7856 - val_loss: 0.6571 - val_root_mean_squared_error: 0.7935
Epoch 331/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6780 - root_mean_squared_error: 0.8057 - val_loss: 0.6960 - val_root_mean_squared_error: 0.8176
Epoch 332/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6567 - root_mean_squared_error: 0.7941 - val_loss: 0.7158 - val_root_mean_squared_error: 0.8276
Epoch 333/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6901 - root_mean_squared_error: 0.8136 - val_loss: 0.7536 - val_root_mean_squared_error: 0.8513
Epoch 334/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7403 - root_mean_squared_error: 0.8447 - val_loss: 0.7058 - val_root_mean_squared_error: 0.8211
Epoch 335/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7262 - root_mean_squared_error: 0.8347 - val_loss: 0.7129 - val_root_mean_squared_error: 0.8296
Epoch 336/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7000 - root_mean_squared_error: 0.8203 - val_loss: 0.6782 - val_root_mean_squared_error: 0.8045
Epoch 337/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6679 - root_mean_squared_error: 0.8008 - val_loss: 0.6719 - val_root_mean_squared_error: 0.8031
Epoch 338/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6790 - root_mean_squared_error: 0.8044 - val_loss: 0.6837 - val_root_mean_squared_error: 0.8118
Epoch 339/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6546 - root_mean_squared_error: 0.7914 - val_loss: 0.6586 - val_root_mean_squared_error: 0.7937
Epoch 340/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6533 - root_mean_squared_error: 0.7906 - val_loss: 0.6633 - val_root_mean_squared_error: 0.7970
Epoch 341/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6378 - root_mean_squared_error: 0.7806 - val_loss: 0.6999 - val_root_mean_squared_error: 0.8195
Epoch 342/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6722 - root_mean_squared_error: 0.8030 - val_loss: 0.6799 - val_root_mean_squared_error: 0.8086
Epoch 343/500
5/5 [==============================] - 0s 51ms/step - loss: 0.6846 - root_mean_squared_error: 0.8088 - val_loss: 0.6723 - val_root_mean_squared_error: 0.8002
Epoch 344/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7113 - root_mean_squared_error: 0.8265 - val_loss: 0.7015 - val_root_mean_squared_error: 0.8236
Epoch 345/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6934 - root_mean_squared_error: 0.8151 - val_loss: 0.6536 - val_root_mean_squared_error: 0.7920
Epoch 346/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6692 - root_mean_squared_error: 0.8013 - val_loss: 0.6804 - val_root_mean_squared_error: 0.8073
Epoch 347/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7500 - root_mean_squared_error: 0.8481 - val_loss: 0.6521 - val_root_mean_squared_error: 0.7898
Epoch 348/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7302 - root_mean_squared_error: 0.8393 - val_loss: 0.6892 - val_root_mean_squared_error: 0.8120
Epoch 349/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6997 - root_mean_squared_error: 0.8227 - val_loss: 0.7224 - val_root_mean_squared_error: 0.8339
Epoch 350/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6755 - root_mean_squared_error: 0.8027 - val_loss: 0.7060 - val_root_mean_squared_error: 0.8245
Epoch 351/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6674 - root_mean_squared_error: 0.8009 - val_loss: 0.7052 - val_root_mean_squared_error: 0.8235
Epoch 352/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7115 - root_mean_squared_error: 0.8267 - val_loss: 0.7205 - val_root_mean_squared_error: 0.8350
Epoch 353/500
5/5 [==============================] - 0s 28ms/step - loss: 0.7211 - root_mean_squared_error: 0.8322 - val_loss: 0.6810 - val_root_mean_squared_error: 0.8094
Epoch 354/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6717 - root_mean_squared_error: 0.8060 - val_loss: 0.6790 - val_root_mean_squared_error: 0.8075
Epoch 355/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7165 - root_mean_squared_error: 0.8305 - val_loss: 0.6914 - val_root_mean_squared_error: 0.8136
Epoch 356/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6534 - root_mean_squared_error: 0.7932 - val_loss: 0.7125 - val_root_mean_squared_error: 0.8274
Epoch 357/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7291 - root_mean_squared_error: 0.8361 - val_loss: 0.6969 - val_root_mean_squared_error: 0.8163
Epoch 358/500
5/5 [==============================] - 0s 49ms/step - loss: 0.6929 - root_mean_squared_error: 0.8130 - val_loss: 0.6746 - val_root_mean_squared_error: 0.8054
Epoch 359/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6606 - root_mean_squared_error: 0.7957 - val_loss: 0.6951 - val_root_mean_squared_error: 0.8179
Epoch 360/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6277 - root_mean_squared_error: 0.7731 - val_loss: 0.6621 - val_root_mean_squared_error: 0.7967
Epoch 361/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6940 - root_mean_squared_error: 0.8135 - val_loss: 0.6961 - val_root_mean_squared_error: 0.8185
Epoch 362/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6373 - root_mean_squared_error: 0.7782 - val_loss: 0.6535 - val_root_mean_squared_error: 0.7895
Epoch 363/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7273 - root_mean_squared_error: 0.8382 - val_loss: 0.6582 - val_root_mean_squared_error: 0.7929
Epoch 364/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7115 - root_mean_squared_error: 0.8270 - val_loss: 0.6824 - val_root_mean_squared_error: 0.8100
Epoch 365/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7151 - root_mean_squared_error: 0.8279 - val_loss: 0.6839 - val_root_mean_squared_error: 0.8089
Epoch 366/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7447 - root_mean_squared_error: 0.8459 - val_loss: 0.7030 - val_root_mean_squared_error: 0.8199
Epoch 367/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6911 - root_mean_squared_error: 0.8127 - val_loss: 0.6449 - val_root_mean_squared_error: 0.7870
Epoch 368/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6745 - root_mean_squared_error: 0.8025 - val_loss: 0.7237 - val_root_mean_squared_error: 0.8336
Epoch 369/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6731 - root_mean_squared_error: 0.8021 - val_loss: 0.6790 - val_root_mean_squared_error: 0.8104
Epoch 370/500
5/5 [==============================] - 0s 51ms/step - loss: 0.6618 - root_mean_squared_error: 0.7941 - val_loss: 0.6618 - val_root_mean_squared_error: 0.7963
Epoch 371/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7355 - root_mean_squared_error: 0.8425 - val_loss: 0.7035 - val_root_mean_squared_error: 0.8219
Epoch 372/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6687 - root_mean_squared_error: 0.8017 - val_loss: 0.7205 - val_root_mean_squared_error: 0.8309
Epoch 373/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6458 - root_mean_squared_error: 0.7837 - val_loss: 0.6572 - val_root_mean_squared_error: 0.7943
Epoch 374/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6265 - root_mean_squared_error: 0.7709 - val_loss: 0.6935 - val_root_mean_squared_error: 0.8130
Epoch 375/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7675 - root_mean_squared_error: 0.8589 - val_loss: 0.6627 - val_root_mean_squared_error: 0.7949
Epoch 376/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7467 - root_mean_squared_error: 0.8490 - val_loss: 0.7403 - val_root_mean_squared_error: 0.8438
Epoch 377/500
5/5 [==============================] - 0s 26ms/step - loss: 0.7062 - root_mean_squared_error: 0.8216 - val_loss: 0.6824 - val_root_mean_squared_error: 0.8096
Epoch 378/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7009 - root_mean_squared_error: 0.8202 - val_loss: 0.6797 - val_root_mean_squared_error: 0.8073
Epoch 379/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6947 - root_mean_squared_error: 0.8166 - val_loss: 0.6724 - val_root_mean_squared_error: 0.8048
Epoch 380/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7222 - root_mean_squared_error: 0.8346 - val_loss: 0.6669 - val_root_mean_squared_error: 0.8001
Epoch 381/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7200 - root_mean_squared_error: 0.8299 - val_loss: 0.6634 - val_root_mean_squared_error: 0.7965
Epoch 382/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6457 - root_mean_squared_error: 0.7848 - val_loss: 0.6450 - val_root_mean_squared_error: 0.7855
Epoch 383/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6982 - root_mean_squared_error: 0.8182 - val_loss: 0.6571 - val_root_mean_squared_error: 0.7937
Epoch 384/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6841 - root_mean_squared_error: 0.8093 - val_loss: 0.6481 - val_root_mean_squared_error: 0.7886
Epoch 385/500
5/5 [==============================] - 0s 28ms/step - loss: 0.7473 - root_mean_squared_error: 0.8468 - val_loss: 0.6969 - val_root_mean_squared_error: 0.8169
Epoch 386/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7950 - root_mean_squared_error: 0.8734 - val_loss: 0.6604 - val_root_mean_squared_error: 0.7943
Epoch 387/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6658 - root_mean_squared_error: 0.7988 - val_loss: 0.6513 - val_root_mean_squared_error: 0.7891
Epoch 388/500
5/5 [==============================] - 0s 50ms/step - loss: 0.6939 - root_mean_squared_error: 0.8182 - val_loss: 0.6667 - val_root_mean_squared_error: 0.7970
Epoch 389/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6707 - root_mean_squared_error: 0.7997 - val_loss: 0.6681 - val_root_mean_squared_error: 0.8008
Epoch 390/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7574 - root_mean_squared_error: 0.8541 - val_loss: 0.6842 - val_root_mean_squared_error: 0.8113
Epoch 391/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6784 - root_mean_squared_error: 0.8051 - val_loss: 0.6687 - val_root_mean_squared_error: 0.8008
Epoch 392/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7107 - root_mean_squared_error: 0.8277 - val_loss: 0.6581 - val_root_mean_squared_error: 0.7944
Epoch 393/500
5/5 [==============================] - 0s 27ms/step - loss: 0.6950 - root_mean_squared_error: 0.8138 - val_loss: 0.6317 - val_root_mean_squared_error: 0.7797
Epoch 394/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6680 - root_mean_squared_error: 0.7990 - val_loss: 0.6615 - val_root_mean_squared_error: 0.7996
Epoch 395/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6928 - root_mean_squared_error: 0.8127 - val_loss: 0.6917 - val_root_mean_squared_error: 0.8141
Epoch 396/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6651 - root_mean_squared_error: 0.7950 - val_loss: 0.6708 - val_root_mean_squared_error: 0.8036
Epoch 397/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6715 - root_mean_squared_error: 0.8013 - val_loss: 0.6737 - val_root_mean_squared_error: 0.8034
Epoch 398/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7624 - root_mean_squared_error: 0.8557 - val_loss: 0.6910 - val_root_mean_squared_error: 0.8147
Epoch 399/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6304 - root_mean_squared_error: 0.7764 - val_loss: 0.6698 - val_root_mean_squared_error: 0.8008
Epoch 400/500
5/5 [==============================] - 0s 25ms/step - loss: 0.7246 - root_mean_squared_error: 0.8332 - val_loss: 0.6707 - val_root_mean_squared_error: 0.8019
Epoch 401/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6873 - root_mean_squared_error: 0.8129 - val_loss: 0.6785 - val_root_mean_squared_error: 0.8077
Epoch 402/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6383 - root_mean_squared_error: 0.7791 - val_loss: 0.6800 - val_root_mean_squared_error: 0.8043
Epoch 403/500
5/5 [==============================] - 0s 49ms/step - loss: 0.7334 - root_mean_squared_error: 0.8406 - val_loss: 0.7436 - val_root_mean_squared_error: 0.8489
Epoch 404/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6223 - root_mean_squared_error: 0.7702 - val_loss: 0.7088 - val_root_mean_squared_error: 0.8271
Epoch 405/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6346 - root_mean_squared_error: 0.7806 - val_loss: 0.6329 - val_root_mean_squared_error: 0.7782
Epoch 406/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6206 - root_mean_squared_error: 0.7688 - val_loss: 0.6601 - val_root_mean_squared_error: 0.7950
Epoch 407/500
5/5 [==============================] - 0s 27ms/step - loss: 0.6754 - root_mean_squared_error: 0.8042 - val_loss: 0.6721 - val_root_mean_squared_error: 0.8000
Epoch 408/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6660 - root_mean_squared_error: 0.7977 - val_loss: 0.6708 - val_root_mean_squared_error: 0.8021
Epoch 409/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7287 - root_mean_squared_error: 0.8383 - val_loss: 0.6905 - val_root_mean_squared_error: 0.8112
Epoch 410/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6636 - root_mean_squared_error: 0.7988 - val_loss: 0.6687 - val_root_mean_squared_error: 0.8056
Epoch 411/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6483 - root_mean_squared_error: 0.7880 - val_loss: 0.6700 - val_root_mean_squared_error: 0.8021
Epoch 412/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6903 - root_mean_squared_error: 0.8165 - val_loss: 0.6488 - val_root_mean_squared_error: 0.7883
Epoch 413/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6810 - root_mean_squared_error: 0.8091 - val_loss: 0.6745 - val_root_mean_squared_error: 0.8036
Epoch 414/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6605 - root_mean_squared_error: 0.7956 - val_loss: 0.7393 - val_root_mean_squared_error: 0.8413
Epoch 415/500
5/5 [==============================] - 0s 50ms/step - loss: 0.7289 - root_mean_squared_error: 0.8360 - val_loss: 0.6585 - val_root_mean_squared_error: 0.7942
Epoch 416/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6593 - root_mean_squared_error: 0.7952 - val_loss: 0.6952 - val_root_mean_squared_error: 0.8161
Epoch 417/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6184 - root_mean_squared_error: 0.7677 - val_loss: 0.6687 - val_root_mean_squared_error: 0.8011
Epoch 418/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6959 - root_mean_squared_error: 0.8162 - val_loss: 0.6444 - val_root_mean_squared_error: 0.7838
Epoch 419/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6922 - root_mean_squared_error: 0.8147 - val_loss: 0.6719 - val_root_mean_squared_error: 0.8036
Epoch 420/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6976 - root_mean_squared_error: 0.8165 - val_loss: 0.6308 - val_root_mean_squared_error: 0.7748
Epoch 421/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6752 - root_mean_squared_error: 0.8009 - val_loss: 0.6706 - val_root_mean_squared_error: 0.8000
Epoch 422/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6117 - root_mean_squared_error: 0.7631 - val_loss: 0.6457 - val_root_mean_squared_error: 0.7877
Epoch 423/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6667 - root_mean_squared_error: 0.8011 - val_loss: 0.6822 - val_root_mean_squared_error: 0.8100
Epoch 424/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7146 - root_mean_squared_error: 0.8276 - val_loss: 0.6677 - val_root_mean_squared_error: 0.8007
Epoch 425/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6167 - root_mean_squared_error: 0.7650 - val_loss: 0.6768 - val_root_mean_squared_error: 0.8040
Epoch 426/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6793 - root_mean_squared_error: 0.8041 - val_loss: 0.6565 - val_root_mean_squared_error: 0.7951
Epoch 427/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6693 - root_mean_squared_error: 0.8007 - val_loss: 0.6635 - val_root_mean_squared_error: 0.7966
Epoch 428/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6098 - root_mean_squared_error: 0.7632 - val_loss: 0.6617 - val_root_mean_squared_error: 0.7967
Epoch 429/500
5/5 [==============================] - 0s 23ms/step - loss: 0.5998 - root_mean_squared_error: 0.7570 - val_loss: 0.6431 - val_root_mean_squared_error: 0.7858
Epoch 430/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6912 - root_mean_squared_error: 0.8152 - val_loss: 0.6594 - val_root_mean_squared_error: 0.7938
Epoch 431/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6972 - root_mean_squared_error: 0.8183 - val_loss: 0.6476 - val_root_mean_squared_error: 0.7866
Epoch 432/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7235 - root_mean_squared_error: 0.8332 - val_loss: 0.6863 - val_root_mean_squared_error: 0.8096
Epoch 433/500
5/5 [==============================] - 0s 49ms/step - loss: 0.6658 - root_mean_squared_error: 0.7983 - val_loss: 0.7312 - val_root_mean_squared_error: 0.8372
Epoch 434/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6906 - root_mean_squared_error: 0.8136 - val_loss: 0.6784 - val_root_mean_squared_error: 0.8044
Epoch 435/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6538 - root_mean_squared_error: 0.7912 - val_loss: 0.6394 - val_root_mean_squared_error: 0.7831
Epoch 436/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6781 - root_mean_squared_error: 0.8061 - val_loss: 0.6322 - val_root_mean_squared_error: 0.7775
Epoch 437/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6764 - root_mean_squared_error: 0.8041 - val_loss: 0.6605 - val_root_mean_squared_error: 0.7938
Epoch 438/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6336 - root_mean_squared_error: 0.7781 - val_loss: 0.6325 - val_root_mean_squared_error: 0.7764
Epoch 439/500
5/5 [==============================] - 0s 25ms/step - loss: 0.6574 - root_mean_squared_error: 0.7919 - val_loss: 0.6597 - val_root_mean_squared_error: 0.7948
Epoch 440/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7291 - root_mean_squared_error: 0.8364 - val_loss: 0.7193 - val_root_mean_squared_error: 0.8335
Epoch 441/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7064 - root_mean_squared_error: 0.8238 - val_loss: 0.6604 - val_root_mean_squared_error: 0.7954
Epoch 442/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6831 - root_mean_squared_error: 0.8101 - val_loss: 0.6583 - val_root_mean_squared_error: 0.7922
Epoch 443/500
5/5 [==============================] - 0s 21ms/step - loss: 0.6780 - root_mean_squared_error: 0.8075 - val_loss: 0.6593 - val_root_mean_squared_error: 0.7964
Epoch 444/500
5/5 [==============================] - 0s 26ms/step - loss: 0.6776 - root_mean_squared_error: 0.8053 - val_loss: 0.6756 - val_root_mean_squared_error: 0.8051
Epoch 445/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7038 - root_mean_squared_error: 0.8196 - val_loss: 0.6896 - val_root_mean_squared_error: 0.8142
Epoch 446/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6949 - root_mean_squared_error: 0.8152 - val_loss: 0.6618 - val_root_mean_squared_error: 0.7968
Epoch 447/500
5/5 [==============================] - 0s 24ms/step - loss: 0.7053 - root_mean_squared_error: 0.8233 - val_loss: 0.6188 - val_root_mean_squared_error: 0.7679
Epoch 448/500
5/5 [==============================] - 0s 52ms/step - loss: 0.5599 - root_mean_squared_error: 0.7313 - val_loss: 0.6729 - val_root_mean_squared_error: 0.8027
Epoch 449/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6209 - root_mean_squared_error: 0.7705 - val_loss: 0.6575 - val_root_mean_squared_error: 0.7951
Epoch 450/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6385 - root_mean_squared_error: 0.7801 - val_loss: 0.6457 - val_root_mean_squared_error: 0.7887
Epoch 451/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7040 - root_mean_squared_error: 0.8222 - val_loss: 0.6504 - val_root_mean_squared_error: 0.7879
Epoch 452/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6897 - root_mean_squared_error: 0.8132 - val_loss: 0.6534 - val_root_mean_squared_error: 0.7896
Epoch 453/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6447 - root_mean_squared_error: 0.7864 - val_loss: 0.6787 - val_root_mean_squared_error: 0.8073
Epoch 454/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6449 - root_mean_squared_error: 0.7858 - val_loss: 0.6989 - val_root_mean_squared_error: 0.8188
Epoch 455/500
5/5 [==============================] - 0s 26ms/step - loss: 0.7286 - root_mean_squared_error: 0.8376 - val_loss: 0.6351 - val_root_mean_squared_error: 0.7776
Epoch 456/500
5/5 [==============================] - 0s 23ms/step - loss: 0.8276 - root_mean_squared_error: 0.8908 - val_loss: 0.6382 - val_root_mean_squared_error: 0.7783
Epoch 457/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7226 - root_mean_squared_error: 0.8334 - val_loss: 0.6628 - val_root_mean_squared_error: 0.7985
Epoch 458/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6253 - root_mean_squared_error: 0.7704 - val_loss: 0.6818 - val_root_mean_squared_error: 0.8086
Epoch 459/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6864 - root_mean_squared_error: 0.8106 - val_loss: 0.6379 - val_root_mean_squared_error: 0.7814
Epoch 460/500
5/5 [==============================] - 0s 49ms/step - loss: 0.6145 - root_mean_squared_error: 0.7655 - val_loss: 0.6575 - val_root_mean_squared_error: 0.7912
Epoch 461/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6658 - root_mean_squared_error: 0.7969 - val_loss: 0.6579 - val_root_mean_squared_error: 0.7901
Epoch 462/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6814 - root_mean_squared_error: 0.8079 - val_loss: 0.6620 - val_root_mean_squared_error: 0.7936
Epoch 463/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6393 - root_mean_squared_error: 0.7800 - val_loss: 0.6852 - val_root_mean_squared_error: 0.8107
Epoch 464/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6327 - root_mean_squared_error: 0.7786 - val_loss: 0.6678 - val_root_mean_squared_error: 0.8004
Epoch 465/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6284 - root_mean_squared_error: 0.7722 - val_loss: 0.6552 - val_root_mean_squared_error: 0.7946
Epoch 466/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6527 - root_mean_squared_error: 0.7897 - val_loss: 0.6611 - val_root_mean_squared_error: 0.7944
Epoch 467/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6673 - root_mean_squared_error: 0.7988 - val_loss: 0.6883 - val_root_mean_squared_error: 0.8146
Epoch 468/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6583 - root_mean_squared_error: 0.7962 - val_loss: 0.6466 - val_root_mean_squared_error: 0.7878
Epoch 469/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6531 - root_mean_squared_error: 0.7883 - val_loss: 0.6836 - val_root_mean_squared_error: 0.8100
Epoch 470/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6244 - root_mean_squared_error: 0.7718 - val_loss: 0.6570 - val_root_mean_squared_error: 0.7944
Epoch 471/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6676 - root_mean_squared_error: 0.7960 - val_loss: 0.6355 - val_root_mean_squared_error: 0.7803
Epoch 472/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6555 - root_mean_squared_error: 0.7910 - val_loss: 0.6836 - val_root_mean_squared_error: 0.8078
Epoch 473/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6626 - root_mean_squared_error: 0.7949 - val_loss: 0.6839 - val_root_mean_squared_error: 0.8068
Epoch 474/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6608 - root_mean_squared_error: 0.7966 - val_loss: 0.7414 - val_root_mean_squared_error: 0.8443
Epoch 475/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6802 - root_mean_squared_error: 0.8064 - val_loss: 0.6605 - val_root_mean_squared_error: 0.7978
Epoch 476/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6220 - root_mean_squared_error: 0.7708 - val_loss: 0.6452 - val_root_mean_squared_error: 0.7849
Epoch 477/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6397 - root_mean_squared_error: 0.7802 - val_loss: 0.6720 - val_root_mean_squared_error: 0.8021
Epoch 478/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6989 - root_mean_squared_error: 0.8189 - val_loss: 0.6355 - val_root_mean_squared_error: 0.7792
Epoch 479/500
5/5 [==============================] - 0s 51ms/step - loss: 0.6608 - root_mean_squared_error: 0.7935 - val_loss: 0.6511 - val_root_mean_squared_error: 0.7861
Epoch 480/500
5/5 [==============================] - 0s 23ms/step - loss: 0.7235 - root_mean_squared_error: 0.8344 - val_loss: 0.6626 - val_root_mean_squared_error: 0.7954
Epoch 481/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6621 - root_mean_squared_error: 0.7964 - val_loss: 0.6952 - val_root_mean_squared_error: 0.8179
Epoch 482/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6750 - root_mean_squared_error: 0.8012 - val_loss: 0.6360 - val_root_mean_squared_error: 0.7812
Epoch 483/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6302 - root_mean_squared_error: 0.7755 - val_loss: 0.6410 - val_root_mean_squared_error: 0.7851
Epoch 484/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6613 - root_mean_squared_error: 0.7940 - val_loss: 0.6846 - val_root_mean_squared_error: 0.8090
Epoch 485/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7568 - root_mean_squared_error: 0.8528 - val_loss: 0.6580 - val_root_mean_squared_error: 0.7945
Epoch 486/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6584 - root_mean_squared_error: 0.7945 - val_loss: 0.6288 - val_root_mean_squared_error: 0.7745
Epoch 487/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7020 - root_mean_squared_error: 0.8208 - val_loss: 0.6567 - val_root_mean_squared_error: 0.7916
Epoch 488/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6255 - root_mean_squared_error: 0.7747 - val_loss: 0.6511 - val_root_mean_squared_error: 0.7888
Epoch 489/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6627 - root_mean_squared_error: 0.7978 - val_loss: 0.6641 - val_root_mean_squared_error: 0.7972
Epoch 490/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6547 - root_mean_squared_error: 0.7904 - val_loss: 0.6809 - val_root_mean_squared_error: 0.8058
Epoch 491/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6844 - root_mean_squared_error: 0.8091 - val_loss: 0.6478 - val_root_mean_squared_error: 0.7862
Epoch 492/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6692 - root_mean_squared_error: 0.8027 - val_loss: 0.6736 - val_root_mean_squared_error: 0.8031
Epoch 493/500
5/5 [==============================] - 0s 49ms/step - loss: 0.6969 - root_mean_squared_error: 0.8178 - val_loss: 0.6516 - val_root_mean_squared_error: 0.7888
Epoch 494/500
5/5 [==============================] - 0s 22ms/step - loss: 0.7148 - root_mean_squared_error: 0.8288 - val_loss: 0.6612 - val_root_mean_squared_error: 0.7938
Epoch 495/500
5/5 [==============================] - 0s 24ms/step - loss: 0.6978 - root_mean_squared_error: 0.8162 - val_loss: 0.6258 - val_root_mean_squared_error: 0.7723
Epoch 496/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6448 - root_mean_squared_error: 0.7858 - val_loss: 0.6372 - val_root_mean_squared_error: 0.7808
Epoch 497/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6871 - root_mean_squared_error: 0.8121 - val_loss: 0.6437 - val_root_mean_squared_error: 0.7825
Epoch 498/500
5/5 [==============================] - 0s 23ms/step - loss: 0.6213 - root_mean_squared_error: 0.7690 - val_loss: 0.6581 - val_root_mean_squared_error: 0.7922
Epoch 499/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6604 - root_mean_squared_error: 0.7913 - val_loss: 0.6522 - val_root_mean_squared_error: 0.7908
Epoch 500/500
5/5 [==============================] - 0s 22ms/step - loss: 0.6190 - root_mean_squared_error: 0.7678 - val_loss: 0.6734 - val_root_mean_squared_error: 0.8037
Model training finished.
Train RMSE: 0.805
Evaluating model performance...
Test RMSE: 0.801

```
</div>
Since we have trained a BNN model, the model produces a different output each time
we call it with the same input, since each time a new set of weights are sampled
from the distributions to construct the network and produce an output.
The less certain the mode weights are, the more variability (wider range) we will
see in the outputs of the same inputs.


```python

def compute_predictions(model, iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    for idx in range(sample):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
            f"Actual: {targets[idx]}"
        )


compute_predictions(bnn_model_small)
```

<div class="k-default-codeblock">
```
Predictions mean: 5.63, min: 4.92, max: 6.15, range: 1.23 - Actual: 6.0
Predictions mean: 6.35, min: 6.01, max: 6.54, range: 0.53 - Actual: 6.0
Predictions mean: 5.65, min: 4.84, max: 6.25, range: 1.41 - Actual: 7.0
Predictions mean: 5.74, min: 5.21, max: 6.25, range: 1.04 - Actual: 5.0
Predictions mean: 5.99, min: 5.26, max: 6.29, range: 1.03 - Actual: 5.0
Predictions mean: 6.26, min: 6.01, max: 6.47, range: 0.46 - Actual: 7.0
Predictions mean: 5.28, min: 4.73, max: 5.86, range: 1.12 - Actual: 5.0
Predictions mean: 6.34, min: 6.06, max: 6.53, range: 0.47 - Actual: 6.0
Predictions mean: 6.23, min: 5.91, max: 6.44, range: 0.53 - Actual: 6.0
Predictions mean: 6.33, min: 6.05, max: 6.54, range: 0.48 - Actual: 7.0

```
</div>
### Train BNN  with the whole training set.


```python
num_epochs = 500
bnn_model_full = create_bnn_model(train_size)
run_experiment(bnn_model_full, mse_loss, train_dataset, test_dataset)

compute_predictions(bnn_model_full)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/500
17/17 [==============================] - 2s 32ms/step - loss: 25.4811 - root_mean_squared_error: 5.0465 - val_loss: 23.8428 - val_root_mean_squared_error: 4.8824
Epoch 2/500
17/17 [==============================] - 0s 7ms/step - loss: 23.0849 - root_mean_squared_error: 4.8040 - val_loss: 24.1269 - val_root_mean_squared_error: 4.9115
Epoch 3/500
17/17 [==============================] - 0s 7ms/step - loss: 22.5191 - root_mean_squared_error: 4.7449 - val_loss: 23.3312 - val_root_mean_squared_error: 4.8297
Epoch 4/500
17/17 [==============================] - 0s 7ms/step - loss: 22.9571 - root_mean_squared_error: 4.7896 - val_loss: 24.4072 - val_root_mean_squared_error: 4.9399
Epoch 5/500
17/17 [==============================] - 0s 6ms/step - loss: 21.4049 - root_mean_squared_error: 4.6245 - val_loss: 21.1895 - val_root_mean_squared_error: 4.6027
Epoch 6/500
17/17 [==============================] - 0s 6ms/step - loss: 19.2838 - root_mean_squared_error: 4.3901 - val_loss: 18.1539 - val_root_mean_squared_error: 4.2602
Epoch 7/500
17/17 [==============================] - 0s 8ms/step - loss: 18.7080 - root_mean_squared_error: 4.3231 - val_loss: 16.3230 - val_root_mean_squared_error: 4.0396
Epoch 8/500
17/17 [==============================] - 0s 8ms/step - loss: 17.6290 - root_mean_squared_error: 4.1977 - val_loss: 16.5583 - val_root_mean_squared_error: 4.0686
Epoch 9/500
17/17 [==============================] - 0s 7ms/step - loss: 17.3524 - root_mean_squared_error: 4.1646 - val_loss: 14.7747 - val_root_mean_squared_error: 3.8434
Epoch 10/500
17/17 [==============================] - 0s 6ms/step - loss: 14.9073 - root_mean_squared_error: 3.8602 - val_loss: 16.7577 - val_root_mean_squared_error: 4.0931
Epoch 11/500
17/17 [==============================] - 0s 7ms/step - loss: 14.4963 - root_mean_squared_error: 3.8058 - val_loss: 12.5314 - val_root_mean_squared_error: 3.5394
Epoch 12/500
17/17 [==============================] - 0s 6ms/step - loss: 14.4477 - root_mean_squared_error: 3.8000 - val_loss: 14.8935 - val_root_mean_squared_error: 3.8586
Epoch 13/500
17/17 [==============================] - 0s 7ms/step - loss: 12.4730 - root_mean_squared_error: 3.5303 - val_loss: 10.6793 - val_root_mean_squared_error: 3.2672
Epoch 14/500
17/17 [==============================] - 0s 7ms/step - loss: 12.1715 - root_mean_squared_error: 3.4864 - val_loss: 11.7965 - val_root_mean_squared_error: 3.4339
Epoch 15/500
17/17 [==============================] - 0s 7ms/step - loss: 10.4368 - root_mean_squared_error: 3.2292 - val_loss: 9.1492 - val_root_mean_squared_error: 3.0239
Epoch 16/500
17/17 [==============================] - 0s 7ms/step - loss: 8.8438 - root_mean_squared_error: 2.9698 - val_loss: 10.2119 - val_root_mean_squared_error: 3.1947
Epoch 17/500
17/17 [==============================] - 0s 15ms/step - loss: 10.0310 - root_mean_squared_error: 3.1655 - val_loss: 8.8562 - val_root_mean_squared_error: 2.9751
Epoch 18/500
17/17 [==============================] - 0s 6ms/step - loss: 8.1578 - root_mean_squared_error: 2.8537 - val_loss: 6.3204 - val_root_mean_squared_error: 2.5129
Epoch 19/500
17/17 [==============================] - 0s 6ms/step - loss: 7.9699 - root_mean_squared_error: 2.8203 - val_loss: 6.8128 - val_root_mean_squared_error: 2.6093
Epoch 20/500
17/17 [==============================] - 0s 7ms/step - loss: 6.5304 - root_mean_squared_error: 2.5515 - val_loss: 5.7017 - val_root_mean_squared_error: 2.3866
Epoch 21/500
17/17 [==============================] - 0s 7ms/step - loss: 6.9045 - root_mean_squared_error: 2.6236 - val_loss: 5.4992 - val_root_mean_squared_error: 2.3440
Epoch 22/500
17/17 [==============================] - 0s 6ms/step - loss: 6.1354 - root_mean_squared_error: 2.4716 - val_loss: 5.5341 - val_root_mean_squared_error: 2.3513
Epoch 23/500
17/17 [==============================] - 0s 6ms/step - loss: 4.6087 - root_mean_squared_error: 2.1413 - val_loss: 3.6785 - val_root_mean_squared_error: 1.9163
Epoch 24/500
17/17 [==============================] - 0s 7ms/step - loss: 3.9958 - root_mean_squared_error: 1.9943 - val_loss: 3.4025 - val_root_mean_squared_error: 1.8432
Epoch 25/500
17/17 [==============================] - 0s 7ms/step - loss: 3.5342 - root_mean_squared_error: 1.8749 - val_loss: 2.8847 - val_root_mean_squared_error: 1.6968
Epoch 26/500
17/17 [==============================] - 0s 7ms/step - loss: 2.3471 - root_mean_squared_error: 1.5270 - val_loss: 1.9949 - val_root_mean_squared_error: 1.4100
Epoch 27/500
17/17 [==============================] - 0s 7ms/step - loss: 2.4082 - root_mean_squared_error: 1.5485 - val_loss: 2.2110 - val_root_mean_squared_error: 1.4845
Epoch 28/500
17/17 [==============================] - 0s 6ms/step - loss: 2.6516 - root_mean_squared_error: 1.6253 - val_loss: 1.7143 - val_root_mean_squared_error: 1.3068
Epoch 29/500
17/17 [==============================] - 0s 7ms/step - loss: 1.9386 - root_mean_squared_error: 1.3858 - val_loss: 1.8892 - val_root_mean_squared_error: 1.3722
Epoch 30/500
17/17 [==============================] - 0s 7ms/step - loss: 1.3866 - root_mean_squared_error: 1.1685 - val_loss: 1.4181 - val_root_mean_squared_error: 1.1882
Epoch 31/500
17/17 [==============================] - 0s 7ms/step - loss: 1.2736 - root_mean_squared_error: 1.1230 - val_loss: 0.9443 - val_root_mean_squared_error: 0.9679
Epoch 32/500
17/17 [==============================] - 0s 6ms/step - loss: 1.1876 - root_mean_squared_error: 1.0809 - val_loss: 0.8534 - val_root_mean_squared_error: 0.9197
Epoch 33/500
17/17 [==============================] - 0s 7ms/step - loss: 0.9041 - root_mean_squared_error: 0.9465 - val_loss: 0.8062 - val_root_mean_squared_error: 0.8930
Epoch 34/500
17/17 [==============================] - 0s 15ms/step - loss: 0.9396 - root_mean_squared_error: 0.9653 - val_loss: 1.2465 - val_root_mean_squared_error: 1.1138
Epoch 35/500
17/17 [==============================] - 0s 8ms/step - loss: 0.7741 - root_mean_squared_error: 0.8754 - val_loss: 0.8585 - val_root_mean_squared_error: 0.9225
Epoch 36/500
17/17 [==============================] - 0s 7ms/step - loss: 0.8769 - root_mean_squared_error: 0.9323 - val_loss: 0.8419 - val_root_mean_squared_error: 0.9132
Epoch 37/500
17/17 [==============================] - 0s 7ms/step - loss: 0.8034 - root_mean_squared_error: 0.8920 - val_loss: 0.7569 - val_root_mean_squared_error: 0.8662
Epoch 38/500
17/17 [==============================] - 0s 8ms/step - loss: 0.8000 - root_mean_squared_error: 0.8899 - val_loss: 0.7685 - val_root_mean_squared_error: 0.8719
Epoch 39/500
17/17 [==============================] - 0s 8ms/step - loss: 0.8210 - root_mean_squared_error: 0.9015 - val_loss: 0.7486 - val_root_mean_squared_error: 0.8605
Epoch 40/500
17/17 [==============================] - 0s 6ms/step - loss: 0.7703 - root_mean_squared_error: 0.8730 - val_loss: 0.7937 - val_root_mean_squared_error: 0.8864
Epoch 41/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7822 - root_mean_squared_error: 0.8798 - val_loss: 0.7712 - val_root_mean_squared_error: 0.8742
Epoch 42/500
17/17 [==============================] - 0s 8ms/step - loss: 0.7368 - root_mean_squared_error: 0.8531 - val_loss: 0.7595 - val_root_mean_squared_error: 0.8673
Epoch 43/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7663 - root_mean_squared_error: 0.8707 - val_loss: 0.7773 - val_root_mean_squared_error: 0.8767
Epoch 44/500
17/17 [==============================] - 0s 6ms/step - loss: 0.8201 - root_mean_squared_error: 0.9003 - val_loss: 0.7967 - val_root_mean_squared_error: 0.8885
Epoch 45/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7218 - root_mean_squared_error: 0.8446 - val_loss: 0.7979 - val_root_mean_squared_error: 0.8890
Epoch 46/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7432 - root_mean_squared_error: 0.8570 - val_loss: 0.7028 - val_root_mean_squared_error: 0.8340
Epoch 47/500
17/17 [==============================] - 0s 14ms/step - loss: 0.7969 - root_mean_squared_error: 0.8869 - val_loss: 0.7667 - val_root_mean_squared_error: 0.8704
Epoch 48/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7684 - root_mean_squared_error: 0.8718 - val_loss: 0.7098 - val_root_mean_squared_error: 0.8371
Epoch 49/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7249 - root_mean_squared_error: 0.8465 - val_loss: 0.7585 - val_root_mean_squared_error: 0.8663
Epoch 50/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7199 - root_mean_squared_error: 0.8430 - val_loss: 0.7528 - val_root_mean_squared_error: 0.8632
Epoch 51/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7268 - root_mean_squared_error: 0.8473 - val_loss: 0.7388 - val_root_mean_squared_error: 0.8552
Epoch 52/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7488 - root_mean_squared_error: 0.8598 - val_loss: 0.7873 - val_root_mean_squared_error: 0.8826
Epoch 53/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7344 - root_mean_squared_error: 0.8520 - val_loss: 0.7363 - val_root_mean_squared_error: 0.8529
Epoch 54/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7607 - root_mean_squared_error: 0.8665 - val_loss: 0.6676 - val_root_mean_squared_error: 0.8113
Epoch 55/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7382 - root_mean_squared_error: 0.8540 - val_loss: 0.7588 - val_root_mean_squared_error: 0.8662
Epoch 56/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7413 - root_mean_squared_error: 0.8556 - val_loss: 0.7111 - val_root_mean_squared_error: 0.8381
Epoch 57/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7131 - root_mean_squared_error: 0.8388 - val_loss: 0.6964 - val_root_mean_squared_error: 0.8285
Epoch 58/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7426 - root_mean_squared_error: 0.8561 - val_loss: 0.7690 - val_root_mean_squared_error: 0.8721
Epoch 59/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6891 - root_mean_squared_error: 0.8243 - val_loss: 0.7290 - val_root_mean_squared_error: 0.8491
Epoch 60/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7234 - root_mean_squared_error: 0.8449 - val_loss: 0.7089 - val_root_mean_squared_error: 0.8357
Epoch 61/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6999 - root_mean_squared_error: 0.8308 - val_loss: 0.6579 - val_root_mean_squared_error: 0.8055
Epoch 62/500
17/17 [==============================] - 0s 8ms/step - loss: 0.7279 - root_mean_squared_error: 0.8469 - val_loss: 0.7262 - val_root_mean_squared_error: 0.8471
Epoch 63/500
17/17 [==============================] - 0s 8ms/step - loss: 0.7166 - root_mean_squared_error: 0.8408 - val_loss: 0.6732 - val_root_mean_squared_error: 0.8145
Epoch 64/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6770 - root_mean_squared_error: 0.8170 - val_loss: 0.6825 - val_root_mean_squared_error: 0.8216
Epoch 65/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7059 - root_mean_squared_error: 0.8344 - val_loss: 0.7204 - val_root_mean_squared_error: 0.8425
Epoch 66/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6967 - root_mean_squared_error: 0.8288 - val_loss: 0.6962 - val_root_mean_squared_error: 0.8289
Epoch 67/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7359 - root_mean_squared_error: 0.8518 - val_loss: 0.7152 - val_root_mean_squared_error: 0.8411
Epoch 68/500
17/17 [==============================] - 0s 15ms/step - loss: 0.7078 - root_mean_squared_error: 0.8358 - val_loss: 0.7306 - val_root_mean_squared_error: 0.8493
Epoch 69/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6964 - root_mean_squared_error: 0.8283 - val_loss: 0.7078 - val_root_mean_squared_error: 0.8357
Epoch 70/500
17/17 [==============================] - 0s 7ms/step - loss: 0.7130 - root_mean_squared_error: 0.8389 - val_loss: 0.6738 - val_root_mean_squared_error: 0.8154
Epoch 71/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6977 - root_mean_squared_error: 0.8295 - val_loss: 0.7203 - val_root_mean_squared_error: 0.8423
Epoch 72/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6664 - root_mean_squared_error: 0.8102 - val_loss: 0.6722 - val_root_mean_squared_error: 0.8126
Epoch 73/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6743 - root_mean_squared_error: 0.8151 - val_loss: 0.7051 - val_root_mean_squared_error: 0.8344
Epoch 74/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6712 - root_mean_squared_error: 0.8132 - val_loss: 0.7162 - val_root_mean_squared_error: 0.8408
Epoch 75/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6740 - root_mean_squared_error: 0.8144 - val_loss: 0.6757 - val_root_mean_squared_error: 0.8171
Epoch 76/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6725 - root_mean_squared_error: 0.8146 - val_loss: 0.6784 - val_root_mean_squared_error: 0.8187
Epoch 77/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6438 - root_mean_squared_error: 0.7957 - val_loss: 0.6656 - val_root_mean_squared_error: 0.8094
Epoch 78/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6683 - root_mean_squared_error: 0.8113 - val_loss: 0.6316 - val_root_mean_squared_error: 0.7875
Epoch 79/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6644 - root_mean_squared_error: 0.8089 - val_loss: 0.7148 - val_root_mean_squared_error: 0.8391
Epoch 80/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6844 - root_mean_squared_error: 0.8208 - val_loss: 0.7066 - val_root_mean_squared_error: 0.8341
Epoch 81/500
17/17 [==============================] - 0s 6ms/step - loss: 0.7099 - root_mean_squared_error: 0.8363 - val_loss: 0.7006 - val_root_mean_squared_error: 0.8313
Epoch 82/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6632 - root_mean_squared_error: 0.8081 - val_loss: 0.6509 - val_root_mean_squared_error: 0.8009
Epoch 83/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6638 - root_mean_squared_error: 0.8080 - val_loss: 0.6525 - val_root_mean_squared_error: 0.8015
Epoch 84/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6565 - root_mean_squared_error: 0.8038 - val_loss: 0.6282 - val_root_mean_squared_error: 0.7860
Epoch 85/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6519 - root_mean_squared_error: 0.8004 - val_loss: 0.6606 - val_root_mean_squared_error: 0.8069
Epoch 86/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6498 - root_mean_squared_error: 0.7995 - val_loss: 0.6954 - val_root_mean_squared_error: 0.8287
Epoch 87/500
17/17 [==============================] - 0s 6ms/step - loss: 0.7198 - root_mean_squared_error: 0.8415 - val_loss: 0.6520 - val_root_mean_squared_error: 0.8007
Epoch 88/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6696 - root_mean_squared_error: 0.8117 - val_loss: 0.6607 - val_root_mean_squared_error: 0.8059
Epoch 89/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6324 - root_mean_squared_error: 0.7881 - val_loss: 0.6774 - val_root_mean_squared_error: 0.8169
Epoch 90/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6853 - root_mean_squared_error: 0.8219 - val_loss: 0.6645 - val_root_mean_squared_error: 0.8087
Epoch 91/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6745 - root_mean_squared_error: 0.8149 - val_loss: 0.6890 - val_root_mean_squared_error: 0.8242
Epoch 92/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6504 - root_mean_squared_error: 0.7997 - val_loss: 0.6506 - val_root_mean_squared_error: 0.8005
Epoch 93/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6978 - root_mean_squared_error: 0.8291 - val_loss: 0.6454 - val_root_mean_squared_error: 0.7976
Epoch 94/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6528 - root_mean_squared_error: 0.8012 - val_loss: 0.6376 - val_root_mean_squared_error: 0.7915
Epoch 95/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6560 - root_mean_squared_error: 0.8033 - val_loss: 0.6384 - val_root_mean_squared_error: 0.7918
Epoch 96/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6382 - root_mean_squared_error: 0.7921 - val_loss: 0.6298 - val_root_mean_squared_error: 0.7865
Epoch 97/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6413 - root_mean_squared_error: 0.7940 - val_loss: 0.6228 - val_root_mean_squared_error: 0.7815
Epoch 98/500
17/17 [==============================] - 0s 15ms/step - loss: 0.6207 - root_mean_squared_error: 0.7802 - val_loss: 0.6310 - val_root_mean_squared_error: 0.7877
Epoch 99/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6556 - root_mean_squared_error: 0.8031 - val_loss: 0.6377 - val_root_mean_squared_error: 0.7912
Epoch 100/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6623 - root_mean_squared_error: 0.8070 - val_loss: 0.6252 - val_root_mean_squared_error: 0.7830
Epoch 101/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6245 - root_mean_squared_error: 0.7831 - val_loss: 0.6489 - val_root_mean_squared_error: 0.7976
Epoch 102/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6341 - root_mean_squared_error: 0.7894 - val_loss: 0.6125 - val_root_mean_squared_error: 0.7764
Epoch 103/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6057 - root_mean_squared_error: 0.7712 - val_loss: 0.6559 - val_root_mean_squared_error: 0.8032
Epoch 104/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6467 - root_mean_squared_error: 0.7967 - val_loss: 0.6380 - val_root_mean_squared_error: 0.7917
Epoch 105/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6421 - root_mean_squared_error: 0.7944 - val_loss: 0.6499 - val_root_mean_squared_error: 0.7992
Epoch 106/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6549 - root_mean_squared_error: 0.8024 - val_loss: 0.6636 - val_root_mean_squared_error: 0.8086
Epoch 107/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6320 - root_mean_squared_error: 0.7883 - val_loss: 0.6316 - val_root_mean_squared_error: 0.7879
Epoch 108/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6271 - root_mean_squared_error: 0.7848 - val_loss: 0.6549 - val_root_mean_squared_error: 0.8024
Epoch 109/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6650 - root_mean_squared_error: 0.8087 - val_loss: 0.6203 - val_root_mean_squared_error: 0.7802
Epoch 110/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6490 - root_mean_squared_error: 0.7983 - val_loss: 0.6411 - val_root_mean_squared_error: 0.7927
Epoch 111/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6529 - root_mean_squared_error: 0.8010 - val_loss: 0.6151 - val_root_mean_squared_error: 0.7764
Epoch 112/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6393 - root_mean_squared_error: 0.7922 - val_loss: 0.6644 - val_root_mean_squared_error: 0.8090
Epoch 113/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6441 - root_mean_squared_error: 0.7951 - val_loss: 0.6530 - val_root_mean_squared_error: 0.8016
Epoch 114/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6468 - root_mean_squared_error: 0.7971 - val_loss: 0.6830 - val_root_mean_squared_error: 0.8194
Epoch 115/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6450 - root_mean_squared_error: 0.7961 - val_loss: 0.6200 - val_root_mean_squared_error: 0.7807
Epoch 116/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6408 - root_mean_squared_error: 0.7927 - val_loss: 0.6326 - val_root_mean_squared_error: 0.7882
Epoch 117/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6176 - root_mean_squared_error: 0.7784 - val_loss: 0.6505 - val_root_mean_squared_error: 0.8000
Epoch 118/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6214 - root_mean_squared_error: 0.7812 - val_loss: 0.6480 - val_root_mean_squared_error: 0.7975
Epoch 119/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6470 - root_mean_squared_error: 0.7972 - val_loss: 0.6115 - val_root_mean_squared_error: 0.7740
Epoch 120/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6318 - root_mean_squared_error: 0.7876 - val_loss: 0.6466 - val_root_mean_squared_error: 0.7966
Epoch 121/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6160 - root_mean_squared_error: 0.7774 - val_loss: 0.6977 - val_root_mean_squared_error: 0.8282
Epoch 122/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6458 - root_mean_squared_error: 0.7967 - val_loss: 0.6203 - val_root_mean_squared_error: 0.7806
Epoch 123/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6457 - root_mean_squared_error: 0.7961 - val_loss: 0.6386 - val_root_mean_squared_error: 0.7921
Epoch 124/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6251 - root_mean_squared_error: 0.7838 - val_loss: 0.6131 - val_root_mean_squared_error: 0.7755
Epoch 125/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6192 - root_mean_squared_error: 0.7796 - val_loss: 0.6151 - val_root_mean_squared_error: 0.7773
Epoch 126/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6320 - root_mean_squared_error: 0.7875 - val_loss: 0.6221 - val_root_mean_squared_error: 0.7808
Epoch 127/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6300 - root_mean_squared_error: 0.7864 - val_loss: 0.6410 - val_root_mean_squared_error: 0.7943
Epoch 128/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6452 - root_mean_squared_error: 0.7957 - val_loss: 0.7021 - val_root_mean_squared_error: 0.8319
Epoch 129/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5982 - root_mean_squared_error: 0.7655 - val_loss: 0.6129 - val_root_mean_squared_error: 0.7752
Epoch 130/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6284 - root_mean_squared_error: 0.7850 - val_loss: 0.6383 - val_root_mean_squared_error: 0.7920
Epoch 131/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6138 - root_mean_squared_error: 0.7767 - val_loss: 0.6096 - val_root_mean_squared_error: 0.7738
Epoch 132/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6356 - root_mean_squared_error: 0.7899 - val_loss: 0.6477 - val_root_mean_squared_error: 0.7975
Epoch 133/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6258 - root_mean_squared_error: 0.7835 - val_loss: 0.6242 - val_root_mean_squared_error: 0.7822
Epoch 134/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6253 - root_mean_squared_error: 0.7827 - val_loss: 0.6451 - val_root_mean_squared_error: 0.7958
Epoch 135/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6241 - root_mean_squared_error: 0.7819 - val_loss: 0.5958 - val_root_mean_squared_error: 0.7637
Epoch 136/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6289 - root_mean_squared_error: 0.7855 - val_loss: 0.6245 - val_root_mean_squared_error: 0.7831
Epoch 137/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5913 - root_mean_squared_error: 0.7611 - val_loss: 0.6111 - val_root_mean_squared_error: 0.7741
Epoch 138/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6529 - root_mean_squared_error: 0.8010 - val_loss: 0.6274 - val_root_mean_squared_error: 0.7841
Epoch 139/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6310 - root_mean_squared_error: 0.7865 - val_loss: 0.6275 - val_root_mean_squared_error: 0.7858
Epoch 140/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6147 - root_mean_squared_error: 0.7764 - val_loss: 0.6265 - val_root_mean_squared_error: 0.7837
Epoch 141/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6393 - root_mean_squared_error: 0.7917 - val_loss: 0.6231 - val_root_mean_squared_error: 0.7814
Epoch 142/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6432 - root_mean_squared_error: 0.7947 - val_loss: 0.6159 - val_root_mean_squared_error: 0.7767
Epoch 143/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6130 - root_mean_squared_error: 0.7751 - val_loss: 0.6089 - val_root_mean_squared_error: 0.7723
Epoch 144/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6270 - root_mean_squared_error: 0.7842 - val_loss: 0.6123 - val_root_mean_squared_error: 0.7742
Epoch 145/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6177 - root_mean_squared_error: 0.7782 - val_loss: 0.6363 - val_root_mean_squared_error: 0.7894
Epoch 146/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6169 - root_mean_squared_error: 0.7776 - val_loss: 0.6232 - val_root_mean_squared_error: 0.7816
Epoch 147/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5952 - root_mean_squared_error: 0.7631 - val_loss: 0.6079 - val_root_mean_squared_error: 0.7725
Epoch 148/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6091 - root_mean_squared_error: 0.7730 - val_loss: 0.6275 - val_root_mean_squared_error: 0.7850
Epoch 149/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6265 - root_mean_squared_error: 0.7840 - val_loss: 0.6681 - val_root_mean_squared_error: 0.8095
Epoch 150/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6302 - root_mean_squared_error: 0.7863 - val_loss: 0.6182 - val_root_mean_squared_error: 0.7775
Epoch 151/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6205 - root_mean_squared_error: 0.7800 - val_loss: 0.6328 - val_root_mean_squared_error: 0.7878
Epoch 152/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6306 - root_mean_squared_error: 0.7866 - val_loss: 0.6592 - val_root_mean_squared_error: 0.8049
Epoch 153/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6070 - root_mean_squared_error: 0.7713 - val_loss: 0.6229 - val_root_mean_squared_error: 0.7821
Epoch 154/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6173 - root_mean_squared_error: 0.7778 - val_loss: 0.6185 - val_root_mean_squared_error: 0.7787
Epoch 155/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6303 - root_mean_squared_error: 0.7861 - val_loss: 0.6353 - val_root_mean_squared_error: 0.7901
Epoch 156/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6289 - root_mean_squared_error: 0.7853 - val_loss: 0.6091 - val_root_mean_squared_error: 0.7722
Epoch 157/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6411 - root_mean_squared_error: 0.7927 - val_loss: 0.6229 - val_root_mean_squared_error: 0.7812
Epoch 158/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6559 - root_mean_squared_error: 0.8021 - val_loss: 0.6090 - val_root_mean_squared_error: 0.7728
Epoch 159/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6125 - root_mean_squared_error: 0.7750 - val_loss: 0.6055 - val_root_mean_squared_error: 0.7703
Epoch 160/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6273 - root_mean_squared_error: 0.7841 - val_loss: 0.6250 - val_root_mean_squared_error: 0.7830
Epoch 161/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6421 - root_mean_squared_error: 0.7935 - val_loss: 0.6344 - val_root_mean_squared_error: 0.7883
Epoch 162/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6150 - root_mean_squared_error: 0.7765 - val_loss: 0.6188 - val_root_mean_squared_error: 0.7795
Epoch 163/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6298 - root_mean_squared_error: 0.7859 - val_loss: 0.6662 - val_root_mean_squared_error: 0.8086
Epoch 164/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6100 - root_mean_squared_error: 0.7727 - val_loss: 0.6358 - val_root_mean_squared_error: 0.7899
Epoch 165/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6211 - root_mean_squared_error: 0.7802 - val_loss: 0.6093 - val_root_mean_squared_error: 0.7722
Epoch 166/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6145 - root_mean_squared_error: 0.7755 - val_loss: 0.6161 - val_root_mean_squared_error: 0.7766
Epoch 167/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5985 - root_mean_squared_error: 0.7658 - val_loss: 0.6350 - val_root_mean_squared_error: 0.7894
Epoch 168/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5986 - root_mean_squared_error: 0.7658 - val_loss: 0.6032 - val_root_mean_squared_error: 0.7683
Epoch 169/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6324 - root_mean_squared_error: 0.7869 - val_loss: 0.6483 - val_root_mean_squared_error: 0.7985
Epoch 170/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6201 - root_mean_squared_error: 0.7799 - val_loss: 0.6292 - val_root_mean_squared_error: 0.7854
Epoch 171/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6026 - root_mean_squared_error: 0.7683 - val_loss: 0.6003 - val_root_mean_squared_error: 0.7676
Epoch 172/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6116 - root_mean_squared_error: 0.7738 - val_loss: 0.5924 - val_root_mean_squared_error: 0.7611
Epoch 173/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6396 - root_mean_squared_error: 0.7923 - val_loss: 0.6097 - val_root_mean_squared_error: 0.7729
Epoch 174/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6203 - root_mean_squared_error: 0.7791 - val_loss: 0.6404 - val_root_mean_squared_error: 0.7924
Epoch 175/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6071 - root_mean_squared_error: 0.7712 - val_loss: 0.6234 - val_root_mean_squared_error: 0.7806
Epoch 176/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6176 - root_mean_squared_error: 0.7781 - val_loss: 0.6160 - val_root_mean_squared_error: 0.7765
Epoch 177/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6241 - root_mean_squared_error: 0.7815 - val_loss: 0.6022 - val_root_mean_squared_error: 0.7674
Epoch 178/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6258 - root_mean_squared_error: 0.7832 - val_loss: 0.6168 - val_root_mean_squared_error: 0.7770
Epoch 179/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6198 - root_mean_squared_error: 0.7786 - val_loss: 0.6225 - val_root_mean_squared_error: 0.7813
Epoch 180/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5986 - root_mean_squared_error: 0.7652 - val_loss: 0.6253 - val_root_mean_squared_error: 0.7831
Epoch 181/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6367 - root_mean_squared_error: 0.7900 - val_loss: 0.6306 - val_root_mean_squared_error: 0.7852
Epoch 182/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6151 - root_mean_squared_error: 0.7760 - val_loss: 0.6516 - val_root_mean_squared_error: 0.8004
Epoch 183/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6104 - root_mean_squared_error: 0.7735 - val_loss: 0.6248 - val_root_mean_squared_error: 0.7833
Epoch 184/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6207 - root_mean_squared_error: 0.7801 - val_loss: 0.6019 - val_root_mean_squared_error: 0.7670
Epoch 185/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6113 - root_mean_squared_error: 0.7736 - val_loss: 0.6109 - val_root_mean_squared_error: 0.7743
Epoch 186/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6123 - root_mean_squared_error: 0.7745 - val_loss: 0.6030 - val_root_mean_squared_error: 0.7680
Epoch 187/500
17/17 [==============================] - 0s 16ms/step - loss: 0.6180 - root_mean_squared_error: 0.7783 - val_loss: 0.6111 - val_root_mean_squared_error: 0.7733
Epoch 188/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6010 - root_mean_squared_error: 0.7675 - val_loss: 0.6092 - val_root_mean_squared_error: 0.7720
Epoch 189/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5892 - root_mean_squared_error: 0.7595 - val_loss: 0.6231 - val_root_mean_squared_error: 0.7813
Epoch 190/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5961 - root_mean_squared_error: 0.7638 - val_loss: 0.6017 - val_root_mean_squared_error: 0.7677
Epoch 191/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6331 - root_mean_squared_error: 0.7875 - val_loss: 0.6403 - val_root_mean_squared_error: 0.7919
Epoch 192/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5967 - root_mean_squared_error: 0.7633 - val_loss: 0.6328 - val_root_mean_squared_error: 0.7872
Epoch 193/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6309 - root_mean_squared_error: 0.7863 - val_loss: 0.6279 - val_root_mean_squared_error: 0.7845
Epoch 194/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6351 - root_mean_squared_error: 0.7892 - val_loss: 0.6094 - val_root_mean_squared_error: 0.7718
Epoch 195/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6439 - root_mean_squared_error: 0.7943 - val_loss: 0.6019 - val_root_mean_squared_error: 0.7677
Epoch 196/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6394 - root_mean_squared_error: 0.7915 - val_loss: 0.6308 - val_root_mean_squared_error: 0.7857
Epoch 197/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6206 - root_mean_squared_error: 0.7794 - val_loss: 0.6010 - val_root_mean_squared_error: 0.7671
Epoch 198/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6133 - root_mean_squared_error: 0.7745 - val_loss: 0.6032 - val_root_mean_squared_error: 0.7695
Epoch 199/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6033 - root_mean_squared_error: 0.7685 - val_loss: 0.6091 - val_root_mean_squared_error: 0.7720
Epoch 200/500
17/17 [==============================] - 0s 15ms/step - loss: 0.6210 - root_mean_squared_error: 0.7798 - val_loss: 0.6111 - val_root_mean_squared_error: 0.7735
Epoch 201/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6074 - root_mean_squared_error: 0.7707 - val_loss: 0.6390 - val_root_mean_squared_error: 0.7922
Epoch 202/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6002 - root_mean_squared_error: 0.7668 - val_loss: 0.6354 - val_root_mean_squared_error: 0.7888
Epoch 203/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6027 - root_mean_squared_error: 0.7679 - val_loss: 0.5971 - val_root_mean_squared_error: 0.7643
Epoch 204/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6133 - root_mean_squared_error: 0.7748 - val_loss: 0.6042 - val_root_mean_squared_error: 0.7690
Epoch 205/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6253 - root_mean_squared_error: 0.7828 - val_loss: 0.5867 - val_root_mean_squared_error: 0.7574
Epoch 206/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6313 - root_mean_squared_error: 0.7861 - val_loss: 0.5919 - val_root_mean_squared_error: 0.7605
Epoch 207/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6301 - root_mean_squared_error: 0.7859 - val_loss: 0.6512 - val_root_mean_squared_error: 0.7992
Epoch 208/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5999 - root_mean_squared_error: 0.7663 - val_loss: 0.6127 - val_root_mean_squared_error: 0.7746
Epoch 209/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6088 - root_mean_squared_error: 0.7718 - val_loss: 0.6267 - val_root_mean_squared_error: 0.7843
Epoch 210/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6036 - root_mean_squared_error: 0.7685 - val_loss: 0.6575 - val_root_mean_squared_error: 0.8033
Epoch 211/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5999 - root_mean_squared_error: 0.7662 - val_loss: 0.6232 - val_root_mean_squared_error: 0.7811
Epoch 212/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6038 - root_mean_squared_error: 0.7683 - val_loss: 0.6066 - val_root_mean_squared_error: 0.7705
Epoch 213/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6313 - root_mean_squared_error: 0.7863 - val_loss: 0.6151 - val_root_mean_squared_error: 0.7753
Epoch 214/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5939 - root_mean_squared_error: 0.7624 - val_loss: 0.5897 - val_root_mean_squared_error: 0.7601
Epoch 215/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6081 - root_mean_squared_error: 0.7712 - val_loss: 0.6457 - val_root_mean_squared_error: 0.7959
Epoch 216/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6311 - root_mean_squared_error: 0.7858 - val_loss: 0.6197 - val_root_mean_squared_error: 0.7781
Epoch 217/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6073 - root_mean_squared_error: 0.7712 - val_loss: 0.5987 - val_root_mean_squared_error: 0.7664
Epoch 218/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6339 - root_mean_squared_error: 0.7875 - val_loss: 0.6120 - val_root_mean_squared_error: 0.7735
Epoch 219/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5916 - root_mean_squared_error: 0.7601 - val_loss: 0.5876 - val_root_mean_squared_error: 0.7573
Epoch 220/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6176 - root_mean_squared_error: 0.7776 - val_loss: 0.6004 - val_root_mean_squared_error: 0.7666
Epoch 221/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6165 - root_mean_squared_error: 0.7767 - val_loss: 0.6088 - val_root_mean_squared_error: 0.7728
Epoch 222/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6398 - root_mean_squared_error: 0.7918 - val_loss: 0.6021 - val_root_mean_squared_error: 0.7677
Epoch 223/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6158 - root_mean_squared_error: 0.7763 - val_loss: 0.6057 - val_root_mean_squared_error: 0.7714
Epoch 224/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6192 - root_mean_squared_error: 0.7784 - val_loss: 0.6076 - val_root_mean_squared_error: 0.7714
Epoch 225/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6180 - root_mean_squared_error: 0.7776 - val_loss: 0.6159 - val_root_mean_squared_error: 0.7762
Epoch 226/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6197 - root_mean_squared_error: 0.7798 - val_loss: 0.6536 - val_root_mean_squared_error: 0.8003
Epoch 227/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6210 - root_mean_squared_error: 0.7807 - val_loss: 0.6004 - val_root_mean_squared_error: 0.7669
Epoch 228/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6206 - root_mean_squared_error: 0.7789 - val_loss: 0.6201 - val_root_mean_squared_error: 0.7790
Epoch 229/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6153 - root_mean_squared_error: 0.7762 - val_loss: 0.6034 - val_root_mean_squared_error: 0.7679
Epoch 230/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6122 - root_mean_squared_error: 0.7742 - val_loss: 0.5961 - val_root_mean_squared_error: 0.7631
Epoch 231/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5927 - root_mean_squared_error: 0.7612 - val_loss: 0.5957 - val_root_mean_squared_error: 0.7632
Epoch 232/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6114 - root_mean_squared_error: 0.7732 - val_loss: 0.6350 - val_root_mean_squared_error: 0.7890
Epoch 233/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6148 - root_mean_squared_error: 0.7756 - val_loss: 0.6230 - val_root_mean_squared_error: 0.7813
Epoch 234/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6204 - root_mean_squared_error: 0.7788 - val_loss: 0.6086 - val_root_mean_squared_error: 0.7713
Epoch 235/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6007 - root_mean_squared_error: 0.7664 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7635
Epoch 236/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6081 - root_mean_squared_error: 0.7711 - val_loss: 0.6035 - val_root_mean_squared_error: 0.7687
Epoch 237/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6132 - root_mean_squared_error: 0.7749 - val_loss: 0.5985 - val_root_mean_squared_error: 0.7648
Epoch 238/500
17/17 [==============================] - 0s 15ms/step - loss: 0.6195 - root_mean_squared_error: 0.7785 - val_loss: 0.6014 - val_root_mean_squared_error: 0.7673
Epoch 239/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6005 - root_mean_squared_error: 0.7659 - val_loss: 0.6202 - val_root_mean_squared_error: 0.7793
Epoch 240/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5956 - root_mean_squared_error: 0.7628 - val_loss: 0.6212 - val_root_mean_squared_error: 0.7797
Epoch 241/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6377 - root_mean_squared_error: 0.7903 - val_loss: 0.6228 - val_root_mean_squared_error: 0.7798
Epoch 242/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6299 - root_mean_squared_error: 0.7853 - val_loss: 0.5920 - val_root_mean_squared_error: 0.7612
Epoch 243/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6274 - root_mean_squared_error: 0.7837 - val_loss: 0.6029 - val_root_mean_squared_error: 0.7685
Epoch 244/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6354 - root_mean_squared_error: 0.7891 - val_loss: 0.6223 - val_root_mean_squared_error: 0.7798
Epoch 245/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6463 - root_mean_squared_error: 0.7956 - val_loss: 0.5952 - val_root_mean_squared_error: 0.7624
Epoch 246/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6052 - root_mean_squared_error: 0.7693 - val_loss: 0.6119 - val_root_mean_squared_error: 0.7742
Epoch 247/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5941 - root_mean_squared_error: 0.7616 - val_loss: 0.5972 - val_root_mean_squared_error: 0.7645
Epoch 248/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6040 - root_mean_squared_error: 0.7680 - val_loss: 0.6120 - val_root_mean_squared_error: 0.7741
Epoch 249/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6147 - root_mean_squared_error: 0.7759 - val_loss: 0.6041 - val_root_mean_squared_error: 0.7689
Epoch 250/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6397 - root_mean_squared_error: 0.7914 - val_loss: 0.6068 - val_root_mean_squared_error: 0.7699
Epoch 251/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6096 - root_mean_squared_error: 0.7722 - val_loss: 0.6003 - val_root_mean_squared_error: 0.7659
Epoch 252/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5797 - root_mean_squared_error: 0.7526 - val_loss: 0.5960 - val_root_mean_squared_error: 0.7630
Epoch 253/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6264 - root_mean_squared_error: 0.7831 - val_loss: 0.6078 - val_root_mean_squared_error: 0.7708
Epoch 254/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5831 - root_mean_squared_error: 0.7543 - val_loss: 0.5969 - val_root_mean_squared_error: 0.7646
Epoch 255/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5860 - root_mean_squared_error: 0.7576 - val_loss: 0.6419 - val_root_mean_squared_error: 0.7932
Epoch 256/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6290 - root_mean_squared_error: 0.7847 - val_loss: 0.6092 - val_root_mean_squared_error: 0.7714
Epoch 257/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6214 - root_mean_squared_error: 0.7797 - val_loss: 0.5967 - val_root_mean_squared_error: 0.7639
Epoch 258/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6317 - root_mean_squared_error: 0.7864 - val_loss: 0.6090 - val_root_mean_squared_error: 0.7719
Epoch 259/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5978 - root_mean_squared_error: 0.7636 - val_loss: 0.6240 - val_root_mean_squared_error: 0.7827
Epoch 260/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6248 - root_mean_squared_error: 0.7818 - val_loss: 0.5984 - val_root_mean_squared_error: 0.7643
Epoch 261/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6344 - root_mean_squared_error: 0.7877 - val_loss: 0.6031 - val_root_mean_squared_error: 0.7672
Epoch 262/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6055 - root_mean_squared_error: 0.7693 - val_loss: 0.6150 - val_root_mean_squared_error: 0.7759
Epoch 263/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6080 - root_mean_squared_error: 0.7712 - val_loss: 0.6008 - val_root_mean_squared_error: 0.7662
Epoch 264/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6296 - root_mean_squared_error: 0.7849 - val_loss: 0.6131 - val_root_mean_squared_error: 0.7733
Epoch 265/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6042 - root_mean_squared_error: 0.7692 - val_loss: 0.6185 - val_root_mean_squared_error: 0.7781
Epoch 266/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6034 - root_mean_squared_error: 0.7684 - val_loss: 0.6042 - val_root_mean_squared_error: 0.7681
Epoch 267/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6070 - root_mean_squared_error: 0.7703 - val_loss: 0.6636 - val_root_mean_squared_error: 0.8064
Epoch 268/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6228 - root_mean_squared_error: 0.7807 - val_loss: 0.6201 - val_root_mean_squared_error: 0.7787
Epoch 269/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6016 - root_mean_squared_error: 0.7671 - val_loss: 0.6174 - val_root_mean_squared_error: 0.7780
Epoch 270/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5976 - root_mean_squared_error: 0.7638 - val_loss: 0.6294 - val_root_mean_squared_error: 0.7850
Epoch 271/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6156 - root_mean_squared_error: 0.7764 - val_loss: 0.6254 - val_root_mean_squared_error: 0.7822
Epoch 272/500
17/17 [==============================] - 0s 15ms/step - loss: 0.5977 - root_mean_squared_error: 0.7646 - val_loss: 0.5992 - val_root_mean_squared_error: 0.7647
Epoch 273/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6125 - root_mean_squared_error: 0.7739 - val_loss: 0.6064 - val_root_mean_squared_error: 0.7701
Epoch 274/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6462 - root_mean_squared_error: 0.7954 - val_loss: 0.5992 - val_root_mean_squared_error: 0.7648
Epoch 275/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6186 - root_mean_squared_error: 0.7778 - val_loss: 0.5868 - val_root_mean_squared_error: 0.7576
Epoch 276/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5845 - root_mean_squared_error: 0.7557 - val_loss: 0.6236 - val_root_mean_squared_error: 0.7821
Epoch 277/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6105 - root_mean_squared_error: 0.7727 - val_loss: 0.5981 - val_root_mean_squared_error: 0.7650
Epoch 278/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6279 - root_mean_squared_error: 0.7834 - val_loss: 0.6088 - val_root_mean_squared_error: 0.7714
Epoch 279/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6293 - root_mean_squared_error: 0.7853 - val_loss: 0.6024 - val_root_mean_squared_error: 0.7674
Epoch 280/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6096 - root_mean_squared_error: 0.7717 - val_loss: 0.5954 - val_root_mean_squared_error: 0.7626
Epoch 281/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6008 - root_mean_squared_error: 0.7659 - val_loss: 0.5952 - val_root_mean_squared_error: 0.7623
Epoch 282/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6062 - root_mean_squared_error: 0.7704 - val_loss: 0.5954 - val_root_mean_squared_error: 0.7623
Epoch 283/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6265 - root_mean_squared_error: 0.7827 - val_loss: 0.5906 - val_root_mean_squared_error: 0.7601
Epoch 284/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5936 - root_mean_squared_error: 0.7612 - val_loss: 0.6005 - val_root_mean_squared_error: 0.7657
Epoch 285/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6227 - root_mean_squared_error: 0.7807 - val_loss: 0.5903 - val_root_mean_squared_error: 0.7595
Epoch 286/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6129 - root_mean_squared_error: 0.7738 - val_loss: 0.5996 - val_root_mean_squared_error: 0.7653
Epoch 287/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6246 - root_mean_squared_error: 0.7818 - val_loss: 0.6197 - val_root_mean_squared_error: 0.7779
Epoch 288/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6366 - root_mean_squared_error: 0.7891 - val_loss: 0.5889 - val_root_mean_squared_error: 0.7582
Epoch 289/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6448 - root_mean_squared_error: 0.7941 - val_loss: 0.5873 - val_root_mean_squared_error: 0.7579
Epoch 290/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5960 - root_mean_squared_error: 0.7626 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7632
Epoch 291/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6196 - root_mean_squared_error: 0.7782 - val_loss: 0.6216 - val_root_mean_squared_error: 0.7813
Epoch 292/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6053 - root_mean_squared_error: 0.7696 - val_loss: 0.5929 - val_root_mean_squared_error: 0.7605
Epoch 293/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6027 - root_mean_squared_error: 0.7672 - val_loss: 0.5876 - val_root_mean_squared_error: 0.7581
Epoch 294/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5904 - root_mean_squared_error: 0.7592 - val_loss: 0.6084 - val_root_mean_squared_error: 0.7712
Epoch 295/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6075 - root_mean_squared_error: 0.7708 - val_loss: 0.5854 - val_root_mean_squared_error: 0.7564
Epoch 296/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6111 - root_mean_squared_error: 0.7725 - val_loss: 0.6202 - val_root_mean_squared_error: 0.7786
Epoch 297/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5904 - root_mean_squared_error: 0.7593 - val_loss: 0.6099 - val_root_mean_squared_error: 0.7723
Epoch 298/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6252 - root_mean_squared_error: 0.7816 - val_loss: 0.6031 - val_root_mean_squared_error: 0.7683
Epoch 299/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6202 - root_mean_squared_error: 0.7790 - val_loss: 0.6160 - val_root_mean_squared_error: 0.7764
Epoch 300/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5797 - root_mean_squared_error: 0.7525 - val_loss: 0.6233 - val_root_mean_squared_error: 0.7808
Epoch 301/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6157 - root_mean_squared_error: 0.7756 - val_loss: 0.6162 - val_root_mean_squared_error: 0.7768
Epoch 302/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6075 - root_mean_squared_error: 0.7707 - val_loss: 0.6491 - val_root_mean_squared_error: 0.7969
Epoch 303/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5950 - root_mean_squared_error: 0.7623 - val_loss: 0.6149 - val_root_mean_squared_error: 0.7750
Epoch 304/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6227 - root_mean_squared_error: 0.7806 - val_loss: 0.5884 - val_root_mean_squared_error: 0.7586
Epoch 305/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6125 - root_mean_squared_error: 0.7735 - val_loss: 0.6271 - val_root_mean_squared_error: 0.7837
Epoch 306/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5919 - root_mean_squared_error: 0.7604 - val_loss: 0.5983 - val_root_mean_squared_error: 0.7653
Epoch 307/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6042 - root_mean_squared_error: 0.7687 - val_loss: 0.6120 - val_root_mean_squared_error: 0.7730
Epoch 308/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6201 - root_mean_squared_error: 0.7787 - val_loss: 0.5862 - val_root_mean_squared_error: 0.7563
Epoch 309/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5929 - root_mean_squared_error: 0.7602 - val_loss: 0.6259 - val_root_mean_squared_error: 0.7829
Epoch 310/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5845 - root_mean_squared_error: 0.7556 - val_loss: 0.6049 - val_root_mean_squared_error: 0.7688
Epoch 311/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6100 - root_mean_squared_error: 0.7722 - val_loss: 0.5889 - val_root_mean_squared_error: 0.7590
Epoch 312/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6028 - root_mean_squared_error: 0.7672 - val_loss: 0.6295 - val_root_mean_squared_error: 0.7846
Epoch 313/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6009 - root_mean_squared_error: 0.7663 - val_loss: 0.6240 - val_root_mean_squared_error: 0.7814
Epoch 314/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5961 - root_mean_squared_error: 0.7635 - val_loss: 0.6065 - val_root_mean_squared_error: 0.7701
Epoch 315/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5933 - root_mean_squared_error: 0.7610 - val_loss: 0.6005 - val_root_mean_squared_error: 0.7664
Epoch 316/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6127 - root_mean_squared_error: 0.7734 - val_loss: 0.6085 - val_root_mean_squared_error: 0.7707
Epoch 317/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6102 - root_mean_squared_error: 0.7726 - val_loss: 0.6066 - val_root_mean_squared_error: 0.7695
Epoch 318/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6124 - root_mean_squared_error: 0.7734 - val_loss: 0.5840 - val_root_mean_squared_error: 0.7558
Epoch 319/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5769 - root_mean_squared_error: 0.7499 - val_loss: 0.6194 - val_root_mean_squared_error: 0.7786
Epoch 320/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6275 - root_mean_squared_error: 0.7832 - val_loss: 0.6108 - val_root_mean_squared_error: 0.7716
Epoch 321/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6013 - root_mean_squared_error: 0.7664 - val_loss: 0.5918 - val_root_mean_squared_error: 0.7595
Epoch 322/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5959 - root_mean_squared_error: 0.7624 - val_loss: 0.5869 - val_root_mean_squared_error: 0.7579
Epoch 323/500
17/17 [==============================] - 0s 14ms/step - loss: 0.5801 - root_mean_squared_error: 0.7525 - val_loss: 0.5930 - val_root_mean_squared_error: 0.7610
Epoch 324/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6149 - root_mean_squared_error: 0.7754 - val_loss: 0.6057 - val_root_mean_squared_error: 0.7687
Epoch 325/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6302 - root_mean_squared_error: 0.7851 - val_loss: 0.6320 - val_root_mean_squared_error: 0.7864
Epoch 326/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6397 - root_mean_squared_error: 0.7907 - val_loss: 0.5924 - val_root_mean_squared_error: 0.7608
Epoch 327/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5906 - root_mean_squared_error: 0.7599 - val_loss: 0.6427 - val_root_mean_squared_error: 0.7932
Epoch 328/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5958 - root_mean_squared_error: 0.7627 - val_loss: 0.6012 - val_root_mean_squared_error: 0.7668
Epoch 329/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6037 - root_mean_squared_error: 0.7676 - val_loss: 0.6317 - val_root_mean_squared_error: 0.7858
Epoch 330/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5985 - root_mean_squared_error: 0.7642 - val_loss: 0.5915 - val_root_mean_squared_error: 0.7597
Epoch 331/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6208 - root_mean_squared_error: 0.7788 - val_loss: 0.6193 - val_root_mean_squared_error: 0.7792
Epoch 332/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5969 - root_mean_squared_error: 0.7636 - val_loss: 0.5965 - val_root_mean_squared_error: 0.7636
Epoch 333/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5783 - root_mean_squared_error: 0.7508 - val_loss: 0.6237 - val_root_mean_squared_error: 0.7803
Epoch 334/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6057 - root_mean_squared_error: 0.7685 - val_loss: 0.6094 - val_root_mean_squared_error: 0.7709
Epoch 335/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6091 - root_mean_squared_error: 0.7712 - val_loss: 0.6153 - val_root_mean_squared_error: 0.7747
Epoch 336/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6017 - root_mean_squared_error: 0.7660 - val_loss: 0.5883 - val_root_mean_squared_error: 0.7576
Epoch 337/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6108 - root_mean_squared_error: 0.7720 - val_loss: 0.6196 - val_root_mean_squared_error: 0.7781
Epoch 338/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6142 - root_mean_squared_error: 0.7749 - val_loss: 0.5963 - val_root_mean_squared_error: 0.7630
Epoch 339/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6174 - root_mean_squared_error: 0.7765 - val_loss: 0.6311 - val_root_mean_squared_error: 0.7863
Epoch 340/500
17/17 [==============================] - 0s 15ms/step - loss: 0.5940 - root_mean_squared_error: 0.7610 - val_loss: 0.5993 - val_root_mean_squared_error: 0.7642
Epoch 341/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6060 - root_mean_squared_error: 0.7689 - val_loss: 0.6026 - val_root_mean_squared_error: 0.7665
Epoch 342/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5977 - root_mean_squared_error: 0.7637 - val_loss: 0.6480 - val_root_mean_squared_error: 0.7970
Epoch 343/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6257 - root_mean_squared_error: 0.7818 - val_loss: 0.6010 - val_root_mean_squared_error: 0.7662
Epoch 344/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6057 - root_mean_squared_error: 0.7690 - val_loss: 0.6172 - val_root_mean_squared_error: 0.7773
Epoch 345/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5789 - root_mean_squared_error: 0.7516 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7626
Epoch 346/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6183 - root_mean_squared_error: 0.7771 - val_loss: 0.6002 - val_root_mean_squared_error: 0.7656
Epoch 347/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5998 - root_mean_squared_error: 0.7651 - val_loss: 0.6092 - val_root_mean_squared_error: 0.7711
Epoch 348/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6003 - root_mean_squared_error: 0.7655 - val_loss: 0.5929 - val_root_mean_squared_error: 0.7606
Epoch 349/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5925 - root_mean_squared_error: 0.7607 - val_loss: 0.6191 - val_root_mean_squared_error: 0.7788
Epoch 350/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5989 - root_mean_squared_error: 0.7646 - val_loss: 0.5997 - val_root_mean_squared_error: 0.7651
Epoch 351/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6292 - root_mean_squared_error: 0.7845 - val_loss: 0.6151 - val_root_mean_squared_error: 0.7758
Epoch 352/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6140 - root_mean_squared_error: 0.7745 - val_loss: 0.6603 - val_root_mean_squared_error: 0.8033
Epoch 353/500
17/17 [==============================] - 0s 15ms/step - loss: 0.5985 - root_mean_squared_error: 0.7639 - val_loss: 0.5969 - val_root_mean_squared_error: 0.7633
Epoch 354/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6095 - root_mean_squared_error: 0.7714 - val_loss: 0.6210 - val_root_mean_squared_error: 0.7795
Epoch 355/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6194 - root_mean_squared_error: 0.7779 - val_loss: 0.5887 - val_root_mean_squared_error: 0.7568
Epoch 356/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6018 - root_mean_squared_error: 0.7664 - val_loss: 0.5997 - val_root_mean_squared_error: 0.7647
Epoch 357/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6017 - root_mean_squared_error: 0.7662 - val_loss: 0.5870 - val_root_mean_squared_error: 0.7563
Epoch 358/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5794 - root_mean_squared_error: 0.7515 - val_loss: 0.6120 - val_root_mean_squared_error: 0.7745
Epoch 359/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6175 - root_mean_squared_error: 0.7766 - val_loss: 0.5897 - val_root_mean_squared_error: 0.7584
Epoch 360/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6267 - root_mean_squared_error: 0.7826 - val_loss: 0.5888 - val_root_mean_squared_error: 0.7588
Epoch 361/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6005 - root_mean_squared_error: 0.7656 - val_loss: 0.5997 - val_root_mean_squared_error: 0.7648
Epoch 362/500
17/17 [==============================] - 0s 9ms/step - loss: 0.5889 - root_mean_squared_error: 0.7577 - val_loss: 0.6018 - val_root_mean_squared_error: 0.7662
Epoch 363/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6066 - root_mean_squared_error: 0.7699 - val_loss: 0.6220 - val_root_mean_squared_error: 0.7804
Epoch 364/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5999 - root_mean_squared_error: 0.7651 - val_loss: 0.5920 - val_root_mean_squared_error: 0.7604
Epoch 365/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6089 - root_mean_squared_error: 0.7712 - val_loss: 0.5968 - val_root_mean_squared_error: 0.7639
Epoch 366/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5895 - root_mean_squared_error: 0.7585 - val_loss: 0.6019 - val_root_mean_squared_error: 0.7677
Epoch 367/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6460 - root_mean_squared_error: 0.7946 - val_loss: 0.5895 - val_root_mean_squared_error: 0.7578
Epoch 368/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6418 - root_mean_squared_error: 0.7921 - val_loss: 0.6121 - val_root_mean_squared_error: 0.7737
Epoch 369/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6014 - root_mean_squared_error: 0.7658 - val_loss: 0.6056 - val_root_mean_squared_error: 0.7696
Epoch 370/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5892 - root_mean_squared_error: 0.7583 - val_loss: 0.5916 - val_root_mean_squared_error: 0.7596
Epoch 371/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5971 - root_mean_squared_error: 0.7634 - val_loss: 0.6114 - val_root_mean_squared_error: 0.7726
Epoch 372/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6086 - root_mean_squared_error: 0.7707 - val_loss: 0.5994 - val_root_mean_squared_error: 0.7646
Epoch 373/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6091 - root_mean_squared_error: 0.7713 - val_loss: 0.6527 - val_root_mean_squared_error: 0.7997
Epoch 374/500
17/17 [==============================] - 0s 15ms/step - loss: 0.6040 - root_mean_squared_error: 0.7682 - val_loss: 0.6160 - val_root_mean_squared_error: 0.7759
Epoch 375/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5897 - root_mean_squared_error: 0.7583 - val_loss: 0.6122 - val_root_mean_squared_error: 0.7726
Epoch 376/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6159 - root_mean_squared_error: 0.7754 - val_loss: 0.5931 - val_root_mean_squared_error: 0.7615
Epoch 377/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6246 - root_mean_squared_error: 0.7812 - val_loss: 0.6176 - val_root_mean_squared_error: 0.7773
Epoch 378/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6368 - root_mean_squared_error: 0.7883 - val_loss: 0.5963 - val_root_mean_squared_error: 0.7617
Epoch 379/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6028 - root_mean_squared_error: 0.7672 - val_loss: 0.6186 - val_root_mean_squared_error: 0.7770
Epoch 380/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5954 - root_mean_squared_error: 0.7622 - val_loss: 0.6086 - val_root_mean_squared_error: 0.7710
Epoch 381/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5977 - root_mean_squared_error: 0.7634 - val_loss: 0.6194 - val_root_mean_squared_error: 0.7780
Epoch 382/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6018 - root_mean_squared_error: 0.7661 - val_loss: 0.5854 - val_root_mean_squared_error: 0.7550
Epoch 383/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5958 - root_mean_squared_error: 0.7627 - val_loss: 0.5928 - val_root_mean_squared_error: 0.7618
Epoch 384/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6164 - root_mean_squared_error: 0.7757 - val_loss: 0.6002 - val_root_mean_squared_error: 0.7648
Epoch 385/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5974 - root_mean_squared_error: 0.7636 - val_loss: 0.5974 - val_root_mean_squared_error: 0.7650
Epoch 386/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5901 - root_mean_squared_error: 0.7585 - val_loss: 0.5932 - val_root_mean_squared_error: 0.7616
Epoch 387/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6090 - root_mean_squared_error: 0.7712 - val_loss: 0.6353 - val_root_mean_squared_error: 0.7881
Epoch 388/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6054 - root_mean_squared_error: 0.7687 - val_loss: 0.6115 - val_root_mean_squared_error: 0.7723
Epoch 389/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6085 - root_mean_squared_error: 0.7703 - val_loss: 0.5999 - val_root_mean_squared_error: 0.7664
Epoch 390/500
17/17 [==============================] - 0s 9ms/step - loss: 0.5914 - root_mean_squared_error: 0.7593 - val_loss: 0.5927 - val_root_mean_squared_error: 0.7609
Epoch 391/500
17/17 [==============================] - 0s 14ms/step - loss: 0.6028 - root_mean_squared_error: 0.7669 - val_loss: 0.6017 - val_root_mean_squared_error: 0.7664
Epoch 392/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5746 - root_mean_squared_error: 0.7472 - val_loss: 0.5827 - val_root_mean_squared_error: 0.7529
Epoch 393/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5860 - root_mean_squared_error: 0.7565 - val_loss: 0.5982 - val_root_mean_squared_error: 0.7642
Epoch 394/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5775 - root_mean_squared_error: 0.7502 - val_loss: 0.5848 - val_root_mean_squared_error: 0.7549
Epoch 395/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6263 - root_mean_squared_error: 0.7821 - val_loss: 0.5880 - val_root_mean_squared_error: 0.7577
Epoch 396/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5905 - root_mean_squared_error: 0.7591 - val_loss: 0.5838 - val_root_mean_squared_error: 0.7551
Epoch 397/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6235 - root_mean_squared_error: 0.7805 - val_loss: 0.5867 - val_root_mean_squared_error: 0.7558
Epoch 398/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6025 - root_mean_squared_error: 0.7660 - val_loss: 0.6180 - val_root_mean_squared_error: 0.7760
Epoch 399/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6350 - root_mean_squared_error: 0.7872 - val_loss: 0.6213 - val_root_mean_squared_error: 0.7786
Epoch 400/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6041 - root_mean_squared_error: 0.7680 - val_loss: 0.6142 - val_root_mean_squared_error: 0.7739
Epoch 401/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5994 - root_mean_squared_error: 0.7647 - val_loss: 0.5942 - val_root_mean_squared_error: 0.7611
Epoch 402/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6174 - root_mean_squared_error: 0.7759 - val_loss: 0.6061 - val_root_mean_squared_error: 0.7686
Epoch 403/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5882 - root_mean_squared_error: 0.7579 - val_loss: 0.5882 - val_root_mean_squared_error: 0.7567
Epoch 404/500
17/17 [==============================] - 0s 15ms/step - loss: 0.5902 - root_mean_squared_error: 0.7588 - val_loss: 0.5856 - val_root_mean_squared_error: 0.7553
Epoch 405/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6037 - root_mean_squared_error: 0.7675 - val_loss: 0.6109 - val_root_mean_squared_error: 0.7722
Epoch 406/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6091 - root_mean_squared_error: 0.7710 - val_loss: 0.6071 - val_root_mean_squared_error: 0.7690
Epoch 407/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6052 - root_mean_squared_error: 0.7680 - val_loss: 0.5951 - val_root_mean_squared_error: 0.7618
Epoch 408/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6153 - root_mean_squared_error: 0.7747 - val_loss: 0.6026 - val_root_mean_squared_error: 0.7670
Epoch 409/500
17/17 [==============================] - 0s 9ms/step - loss: 0.6168 - root_mean_squared_error: 0.7761 - val_loss: 0.5897 - val_root_mean_squared_error: 0.7578
Epoch 410/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5897 - root_mean_squared_error: 0.7580 - val_loss: 0.5951 - val_root_mean_squared_error: 0.7614
Epoch 411/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6224 - root_mean_squared_error: 0.7792 - val_loss: 0.5999 - val_root_mean_squared_error: 0.7644
Epoch 412/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6073 - root_mean_squared_error: 0.7699 - val_loss: 0.5986 - val_root_mean_squared_error: 0.7641
Epoch 413/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5948 - root_mean_squared_error: 0.7616 - val_loss: 0.5885 - val_root_mean_squared_error: 0.7579
Epoch 414/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5930 - root_mean_squared_error: 0.7599 - val_loss: 0.6134 - val_root_mean_squared_error: 0.7745
Epoch 415/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6161 - root_mean_squared_error: 0.7752 - val_loss: 0.5944 - val_root_mean_squared_error: 0.7615
Epoch 416/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6054 - root_mean_squared_error: 0.7685 - val_loss: 0.5884 - val_root_mean_squared_error: 0.7571
Epoch 417/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6218 - root_mean_squared_error: 0.7789 - val_loss: 0.5895 - val_root_mean_squared_error: 0.7583
Epoch 418/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6094 - root_mean_squared_error: 0.7709 - val_loss: 0.5928 - val_root_mean_squared_error: 0.7606
Epoch 419/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5762 - root_mean_squared_error: 0.7494 - val_loss: 0.5968 - val_root_mean_squared_error: 0.7629
Epoch 420/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6115 - root_mean_squared_error: 0.7723 - val_loss: 0.5982 - val_root_mean_squared_error: 0.7643
Epoch 421/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6416 - root_mean_squared_error: 0.7918 - val_loss: 0.5936 - val_root_mean_squared_error: 0.7610
Epoch 422/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5779 - root_mean_squared_error: 0.7504 - val_loss: 0.6062 - val_root_mean_squared_error: 0.7684
Epoch 423/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5848 - root_mean_squared_error: 0.7547 - val_loss: 0.6075 - val_root_mean_squared_error: 0.7711
Epoch 424/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6076 - root_mean_squared_error: 0.7701 - val_loss: 0.5845 - val_root_mean_squared_error: 0.7540
Epoch 425/500
17/17 [==============================] - 0s 15ms/step - loss: 0.5999 - root_mean_squared_error: 0.7648 - val_loss: 0.5922 - val_root_mean_squared_error: 0.7596
Epoch 426/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5820 - root_mean_squared_error: 0.7522 - val_loss: 0.6012 - val_root_mean_squared_error: 0.7658
Epoch 427/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5932 - root_mean_squared_error: 0.7608 - val_loss: 0.5936 - val_root_mean_squared_error: 0.7611
Epoch 428/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5957 - root_mean_squared_error: 0.7621 - val_loss: 0.6008 - val_root_mean_squared_error: 0.7648
Epoch 429/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6087 - root_mean_squared_error: 0.7706 - val_loss: 0.5910 - val_root_mean_squared_error: 0.7595
Epoch 430/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5985 - root_mean_squared_error: 0.7639 - val_loss: 0.5958 - val_root_mean_squared_error: 0.7623
Epoch 431/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6192 - root_mean_squared_error: 0.7774 - val_loss: 0.5887 - val_root_mean_squared_error: 0.7580
Epoch 432/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5921 - root_mean_squared_error: 0.7598 - val_loss: 0.5916 - val_root_mean_squared_error: 0.7599
Epoch 433/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5943 - root_mean_squared_error: 0.7616 - val_loss: 0.5927 - val_root_mean_squared_error: 0.7606
Epoch 434/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5891 - root_mean_squared_error: 0.7573 - val_loss: 0.5941 - val_root_mean_squared_error: 0.7614
Epoch 435/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5901 - root_mean_squared_error: 0.7588 - val_loss: 0.5899 - val_root_mean_squared_error: 0.7579
Epoch 436/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5918 - root_mean_squared_error: 0.7598 - val_loss: 0.6159 - val_root_mean_squared_error: 0.7762
Epoch 437/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5904 - root_mean_squared_error: 0.7587 - val_loss: 0.5941 - val_root_mean_squared_error: 0.7605
Epoch 438/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6069 - root_mean_squared_error: 0.7697 - val_loss: 0.6156 - val_root_mean_squared_error: 0.7749
Epoch 439/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5956 - root_mean_squared_error: 0.7619 - val_loss: 0.6232 - val_root_mean_squared_error: 0.7799
Epoch 440/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5906 - root_mean_squared_error: 0.7590 - val_loss: 0.6021 - val_root_mean_squared_error: 0.7658
Epoch 441/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6069 - root_mean_squared_error: 0.7691 - val_loss: 0.6010 - val_root_mean_squared_error: 0.7661
Epoch 442/500
17/17 [==============================] - 0s 15ms/step - loss: 0.5850 - root_mean_squared_error: 0.7546 - val_loss: 0.5892 - val_root_mean_squared_error: 0.7573
Epoch 443/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5962 - root_mean_squared_error: 0.7622 - val_loss: 0.6017 - val_root_mean_squared_error: 0.7666
Epoch 444/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6066 - root_mean_squared_error: 0.7692 - val_loss: 0.5908 - val_root_mean_squared_error: 0.7591
Epoch 445/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6138 - root_mean_squared_error: 0.7736 - val_loss: 0.5894 - val_root_mean_squared_error: 0.7573
Epoch 446/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6007 - root_mean_squared_error: 0.7646 - val_loss: 0.5851 - val_root_mean_squared_error: 0.7546
Epoch 447/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6057 - root_mean_squared_error: 0.7686 - val_loss: 0.6087 - val_root_mean_squared_error: 0.7707
Epoch 448/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5903 - root_mean_squared_error: 0.7583 - val_loss: 0.5978 - val_root_mean_squared_error: 0.7638
Epoch 449/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6103 - root_mean_squared_error: 0.7713 - val_loss: 0.6037 - val_root_mean_squared_error: 0.7670
Epoch 450/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6089 - root_mean_squared_error: 0.7707 - val_loss: 0.6132 - val_root_mean_squared_error: 0.7734
Epoch 451/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5799 - root_mean_squared_error: 0.7511 - val_loss: 0.5895 - val_root_mean_squared_error: 0.7577
Epoch 452/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5936 - root_mean_squared_error: 0.7609 - val_loss: 0.6005 - val_root_mean_squared_error: 0.7650
Epoch 453/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5955 - root_mean_squared_error: 0.7623 - val_loss: 0.5939 - val_root_mean_squared_error: 0.7619
Epoch 454/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6062 - root_mean_squared_error: 0.7688 - val_loss: 0.5896 - val_root_mean_squared_error: 0.7579
Epoch 455/500
17/17 [==============================] - 0s 14ms/step - loss: 0.5837 - root_mean_squared_error: 0.7542 - val_loss: 0.5910 - val_root_mean_squared_error: 0.7591
Epoch 456/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5827 - root_mean_squared_error: 0.7541 - val_loss: 0.5971 - val_root_mean_squared_error: 0.7633
Epoch 457/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5910 - root_mean_squared_error: 0.7594 - val_loss: 0.5922 - val_root_mean_squared_error: 0.7596
Epoch 458/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6183 - root_mean_squared_error: 0.7770 - val_loss: 0.5889 - val_root_mean_squared_error: 0.7578
Epoch 459/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6071 - root_mean_squared_error: 0.7693 - val_loss: 0.5922 - val_root_mean_squared_error: 0.7607
Epoch 460/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6001 - root_mean_squared_error: 0.7651 - val_loss: 0.6161 - val_root_mean_squared_error: 0.7754
Epoch 461/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5811 - root_mean_squared_error: 0.7522 - val_loss: 0.6205 - val_root_mean_squared_error: 0.7776
Epoch 462/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5819 - root_mean_squared_error: 0.7524 - val_loss: 0.5909 - val_root_mean_squared_error: 0.7601
Epoch 463/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6029 - root_mean_squared_error: 0.7672 - val_loss: 0.6106 - val_root_mean_squared_error: 0.7709
Epoch 464/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6316 - root_mean_squared_error: 0.7853 - val_loss: 0.6012 - val_root_mean_squared_error: 0.7660
Epoch 465/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6150 - root_mean_squared_error: 0.7741 - val_loss: 0.5878 - val_root_mean_squared_error: 0.7558
Epoch 466/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6124 - root_mean_squared_error: 0.7726 - val_loss: 0.6086 - val_root_mean_squared_error: 0.7706
Epoch 467/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6032 - root_mean_squared_error: 0.7674 - val_loss: 0.6020 - val_root_mean_squared_error: 0.7666
Epoch 468/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6021 - root_mean_squared_error: 0.7666 - val_loss: 0.5998 - val_root_mean_squared_error: 0.7641
Epoch 469/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5847 - root_mean_squared_error: 0.7544 - val_loss: 0.5971 - val_root_mean_squared_error: 0.7626
Epoch 470/500
17/17 [==============================] - 0s 9ms/step - loss: 0.6063 - root_mean_squared_error: 0.7685 - val_loss: 0.5842 - val_root_mean_squared_error: 0.7543
Epoch 471/500
17/17 [==============================] - 0s 6ms/step - loss: 0.6104 - root_mean_squared_error: 0.7716 - val_loss: 0.5893 - val_root_mean_squared_error: 0.7578
Epoch 472/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6029 - root_mean_squared_error: 0.7665 - val_loss: 0.5973 - val_root_mean_squared_error: 0.7635
Epoch 473/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5775 - root_mean_squared_error: 0.7499 - val_loss: 0.5891 - val_root_mean_squared_error: 0.7577
Epoch 474/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5837 - root_mean_squared_error: 0.7539 - val_loss: 0.5819 - val_root_mean_squared_error: 0.7518
Epoch 475/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5995 - root_mean_squared_error: 0.7642 - val_loss: 0.5947 - val_root_mean_squared_error: 0.7605
Epoch 476/500
17/17 [==============================] - 0s 14ms/step - loss: 0.5963 - root_mean_squared_error: 0.7621 - val_loss: 0.6001 - val_root_mean_squared_error: 0.7649
Epoch 477/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6257 - root_mean_squared_error: 0.7811 - val_loss: 0.5871 - val_root_mean_squared_error: 0.7559
Epoch 478/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6040 - root_mean_squared_error: 0.7670 - val_loss: 0.5822 - val_root_mean_squared_error: 0.7531
Epoch 479/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5979 - root_mean_squared_error: 0.7632 - val_loss: 0.5914 - val_root_mean_squared_error: 0.7586
Epoch 480/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6220 - root_mean_squared_error: 0.7787 - val_loss: 0.5836 - val_root_mean_squared_error: 0.7545
Epoch 481/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6018 - root_mean_squared_error: 0.7656 - val_loss: 0.6078 - val_root_mean_squared_error: 0.7686
Epoch 482/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6236 - root_mean_squared_error: 0.7801 - val_loss: 0.5821 - val_root_mean_squared_error: 0.7531
Epoch 483/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6038 - root_mean_squared_error: 0.7677 - val_loss: 0.5888 - val_root_mean_squared_error: 0.7572
Epoch 484/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6195 - root_mean_squared_error: 0.7774 - val_loss: 0.5861 - val_root_mean_squared_error: 0.7548
Epoch 485/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5996 - root_mean_squared_error: 0.7643 - val_loss: 0.6005 - val_root_mean_squared_error: 0.7655
Epoch 486/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6171 - root_mean_squared_error: 0.7756 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7651
Epoch 487/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5994 - root_mean_squared_error: 0.7645 - val_loss: 0.6064 - val_root_mean_squared_error: 0.7689
Epoch 488/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6117 - root_mean_squared_error: 0.7719 - val_loss: 0.5963 - val_root_mean_squared_error: 0.7628
Epoch 489/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6100 - root_mean_squared_error: 0.7712 - val_loss: 0.6076 - val_root_mean_squared_error: 0.7696
Epoch 490/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6120 - root_mean_squared_error: 0.7727 - val_loss: 0.5962 - val_root_mean_squared_error: 0.7629
Epoch 491/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6074 - root_mean_squared_error: 0.7697 - val_loss: 0.5956 - val_root_mean_squared_error: 0.7619
Epoch 492/500
17/17 [==============================] - 0s 8ms/step - loss: 0.5871 - root_mean_squared_error: 0.7563 - val_loss: 0.6002 - val_root_mean_squared_error: 0.7640
Epoch 493/500
17/17 [==============================] - 0s 15ms/step - loss: 0.6152 - root_mean_squared_error: 0.7741 - val_loss: 0.5874 - val_root_mean_squared_error: 0.7571
Epoch 494/500
17/17 [==============================] - 0s 8ms/step - loss: 0.6071 - root_mean_squared_error: 0.7693 - val_loss: 0.5848 - val_root_mean_squared_error: 0.7543
Epoch 495/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5799 - root_mean_squared_error: 0.7511 - val_loss: 0.5902 - val_root_mean_squared_error: 0.7572
Epoch 496/500
17/17 [==============================] - 0s 6ms/step - loss: 0.5926 - root_mean_squared_error: 0.7603 - val_loss: 0.5961 - val_root_mean_squared_error: 0.7616
Epoch 497/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5928 - root_mean_squared_error: 0.7595 - val_loss: 0.5916 - val_root_mean_squared_error: 0.7595
Epoch 498/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6115 - root_mean_squared_error: 0.7715 - val_loss: 0.5869 - val_root_mean_squared_error: 0.7558
Epoch 499/500
17/17 [==============================] - 0s 7ms/step - loss: 0.6044 - root_mean_squared_error: 0.7673 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7645
Epoch 500/500
17/17 [==============================] - 0s 7ms/step - loss: 0.5853 - root_mean_squared_error: 0.7550 - val_loss: 0.5999 - val_root_mean_squared_error: 0.7651
Model training finished.
Train RMSE: 0.762
Evaluating model performance...
Test RMSE: 0.759
Predictions mean: 5.41, min: 5.06, max: 5.9, range: 0.84 - Actual: 6.0
Predictions mean: 6.5, min: 6.16, max: 6.61, range: 0.44 - Actual: 6.0
Predictions mean: 5.59, min: 4.96, max: 6.0, range: 1.04 - Actual: 7.0
Predictions mean: 5.67, min: 5.25, max: 6.01, range: 0.76 - Actual: 5.0
Predictions mean: 6.02, min: 5.68, max: 6.39, range: 0.71 - Actual: 5.0
Predictions mean: 6.35, min: 6.11, max: 6.52, range: 0.41 - Actual: 7.0
Predictions mean: 5.21, min: 4.85, max: 5.68, range: 0.83 - Actual: 5.0
Predictions mean: 6.53, min: 6.35, max: 6.64, range: 0.28 - Actual: 6.0
Predictions mean: 6.3, min: 6.05, max: 6.47, range: 0.42 - Actual: 6.0
Predictions mean: 6.44, min: 6.19, max: 6.59, range: 0.4 - Actual: 7.0

```
</div>
Notice that the model trained with the full training dataset shows smaller range
(uncertainty) in the prediction values for the same inputs, compared to the model
trained with a subset of the training dataset.

---
## Experiment 3: probabilistic Bayesian neural network

So far, the output of the standard and the Bayesian NN models that we built is
deterministic, that is, produces a point estimate as a prediction for a given example.
We can create a probabilistic NN by letting the model output a distribution.
In this case, the model captures the *aleatoric uncertainty* as well,
which is due to irreducible noise in the data, or to the stochastic nature of the
process generating the data.

In this example, we model the output as a `IndependentNormal` distribution,
with learnable mean and variance parameters. If the task was classification,
we would have used `IndependentBernoulli` with binary classes, and `OneHotCategorical`
with multiple classes, to model distribution of the model output.


```python

def create_probablistic_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

Since the output of the model is a distribution, rather than a point estimate,
we use the [negative loglikelihood](https://en.wikipedia.org/wiki/Likelihood_function)
as our loss function to compute how likely to see the true data (targets) from the
estimated distribution produced by the model.


```python

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


num_epochs = 1000
prob_bnn_model = create_probablistic_bnn_model(train_size)
run_experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset)
```

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/1000
17/17 [==============================] - 2s 36ms/step - loss: 11.2378 - root_mean_squared_error: 6.6758 - val_loss: 8.5554 - val_root_mean_squared_error: 6.6240
Epoch 2/1000
17/17 [==============================] - 0s 7ms/step - loss: 11.8285 - root_mean_squared_error: 6.5718 - val_loss: 8.2138 - val_root_mean_squared_error: 6.5256
Epoch 3/1000
17/17 [==============================] - 0s 7ms/step - loss: 8.8566 - root_mean_squared_error: 6.5369 - val_loss: 5.8749 - val_root_mean_squared_error: 6.3394
Epoch 4/1000
17/17 [==============================] - 0s 7ms/step - loss: 7.8191 - root_mean_squared_error: 6.3981 - val_loss: 7.6224 - val_root_mean_squared_error: 6.4473
Epoch 5/1000
17/17 [==============================] - 0s 7ms/step - loss: 6.2598 - root_mean_squared_error: 6.4613 - val_loss: 5.9415 - val_root_mean_squared_error: 6.3466
Epoch 6/1000
17/17 [==============================] - 0s 7ms/step - loss: 6.0987 - root_mean_squared_error: 6.4901 - val_loss: 6.0386 - val_root_mean_squared_error: 6.3686
Epoch 7/1000
17/17 [==============================] - 0s 7ms/step - loss: 5.5819 - root_mean_squared_error: 6.4010 - val_loss: 4.9589 - val_root_mean_squared_error: 6.2099
Epoch 8/1000
17/17 [==============================] - 0s 7ms/step - loss: 5.4525 - root_mean_squared_error: 6.4817 - val_loss: 6.8042 - val_root_mean_squared_error: 6.0617
Epoch 9/1000
17/17 [==============================] - 0s 7ms/step - loss: 4.7016 - root_mean_squared_error: 6.1980 - val_loss: 4.5854 - val_root_mean_squared_error: 6.1331
Epoch 10/1000
17/17 [==============================] - 0s 6ms/step - loss: 4.3992 - root_mean_squared_error: 6.2870 - val_loss: 5.4277 - val_root_mean_squared_error: 5.9639
Epoch 11/1000
17/17 [==============================] - 0s 7ms/step - loss: 4.7261 - root_mean_squared_error: 6.1046 - val_loss: 3.8970 - val_root_mean_squared_error: 6.2413
Epoch 12/1000
17/17 [==============================] - 0s 6ms/step - loss: 4.0523 - root_mean_squared_error: 6.1176 - val_loss: 4.0278 - val_root_mean_squared_error: 6.2741
Epoch 13/1000
17/17 [==============================] - 0s 7ms/step - loss: 4.0525 - root_mean_squared_error: 6.1399 - val_loss: 3.9712 - val_root_mean_squared_error: 6.1504
Epoch 14/1000
17/17 [==============================] - 0s 6ms/step - loss: 3.9146 - root_mean_squared_error: 5.9999 - val_loss: 3.5438 - val_root_mean_squared_error: 6.2295
Epoch 15/1000
17/17 [==============================] - 0s 16ms/step - loss: 3.5343 - root_mean_squared_error: 5.9969 - val_loss: 3.6160 - val_root_mean_squared_error: 6.0013
Epoch 16/1000
17/17 [==============================] - 0s 7ms/step - loss: 3.4089 - root_mean_squared_error: 6.0077 - val_loss: 3.1442 - val_root_mean_squared_error: 5.8765
Epoch 17/1000
17/17 [==============================] - 0s 6ms/step - loss: 3.3385 - root_mean_squared_error: 6.0883 - val_loss: 3.1682 - val_root_mean_squared_error: 5.6954
Epoch 18/1000
17/17 [==============================] - 0s 6ms/step - loss: 3.1420 - root_mean_squared_error: 6.0263 - val_loss: 3.1506 - val_root_mean_squared_error: 5.7370
Epoch 19/1000
17/17 [==============================] - 0s 6ms/step - loss: 3.1645 - root_mean_squared_error: 6.1128 - val_loss: 2.9942 - val_root_mean_squared_error: 6.2751
Epoch 20/1000
17/17 [==============================] - 0s 7ms/step - loss: 3.0444 - root_mean_squared_error: 5.9666 - val_loss: 2.9360 - val_root_mean_squared_error: 5.8790
Epoch 21/1000
17/17 [==============================] - 0s 7ms/step - loss: 3.0090 - root_mean_squared_error: 5.8281 - val_loss: 2.9882 - val_root_mean_squared_error: 5.6285
Epoch 22/1000
17/17 [==============================] - 0s 6ms/step - loss: 3.0276 - root_mean_squared_error: 5.7490 - val_loss: 2.8758 - val_root_mean_squared_error: 5.9208
Epoch 23/1000
17/17 [==============================] - 0s 6ms/step - loss: 2.8940 - root_mean_squared_error: 5.8924 - val_loss: 2.8509 - val_root_mean_squared_error: 5.8735
Epoch 24/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.8709 - root_mean_squared_error: 5.6038 - val_loss: 2.8469 - val_root_mean_squared_error: 5.6682
Epoch 25/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.8351 - root_mean_squared_error: 5.6712 - val_loss: 2.7831 - val_root_mean_squared_error: 5.4349
Epoch 26/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.8740 - root_mean_squared_error: 5.5970 - val_loss: 2.7714 - val_root_mean_squared_error: 5.7378
Epoch 27/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.7579 - root_mean_squared_error: 5.5881 - val_loss: 2.7504 - val_root_mean_squared_error: 5.2322
Epoch 28/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.7298 - root_mean_squared_error: 5.3991 - val_loss: 2.7191 - val_root_mean_squared_error: 5.2280
Epoch 29/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.7106 - root_mean_squared_error: 5.3116 - val_loss: 2.6726 - val_root_mean_squared_error: 5.2263
Epoch 30/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.6840 - root_mean_squared_error: 5.1170 - val_loss: 2.6427 - val_root_mean_squared_error: 4.8465
Epoch 31/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.6488 - root_mean_squared_error: 5.0541 - val_loss: 2.6070 - val_root_mean_squared_error: 4.9453
Epoch 32/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.6122 - root_mean_squared_error: 4.9675 - val_loss: 2.5821 - val_root_mean_squared_error: 4.7760
Epoch 33/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.6005 - root_mean_squared_error: 4.7939 - val_loss: 2.5303 - val_root_mean_squared_error: 4.6228
Epoch 34/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.5341 - root_mean_squared_error: 4.6320 - val_loss: 2.5073 - val_root_mean_squared_error: 4.1938
Epoch 35/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.5092 - root_mean_squared_error: 4.4484 - val_loss: 2.4568 - val_root_mean_squared_error: 4.4054
Epoch 36/1000
17/17 [==============================] - 0s 8ms/step - loss: 2.4803 - root_mean_squared_error: 4.4221 - val_loss: 2.4045 - val_root_mean_squared_error: 4.2555
Epoch 37/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.4184 - root_mean_squared_error: 4.1667 - val_loss: 2.3895 - val_root_mean_squared_error: 3.9923
Epoch 38/1000
17/17 [==============================] - 0s 15ms/step - loss: 2.3945 - root_mean_squared_error: 3.9842 - val_loss: 2.3463 - val_root_mean_squared_error: 4.0221
Epoch 39/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.3549 - root_mean_squared_error: 3.9801 - val_loss: 2.3241 - val_root_mean_squared_error: 3.7825
Epoch 40/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.3028 - root_mean_squared_error: 3.7566 - val_loss: 2.2388 - val_root_mean_squared_error: 3.4149
Epoch 41/1000
17/17 [==============================] - 0s 8ms/step - loss: 2.2767 - root_mean_squared_error: 3.6482 - val_loss: 2.2171 - val_root_mean_squared_error: 3.4705
Epoch 42/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.2197 - root_mean_squared_error: 3.3953 - val_loss: 2.1720 - val_root_mean_squared_error: 3.3142
Epoch 43/1000
17/17 [==============================] - 0s 6ms/step - loss: 2.1648 - root_mean_squared_error: 3.3274 - val_loss: 2.1094 - val_root_mean_squared_error: 2.9487
Epoch 44/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.1003 - root_mean_squared_error: 3.1417 - val_loss: 2.0644 - val_root_mean_squared_error: 2.9427
Epoch 45/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.0687 - root_mean_squared_error: 3.0065 - val_loss: 1.9902 - val_root_mean_squared_error: 2.9885
Epoch 46/1000
17/17 [==============================] - 0s 7ms/step - loss: 2.0124 - root_mean_squared_error: 2.8658 - val_loss: 1.9283 - val_root_mean_squared_error: 2.6447
Epoch 47/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.9436 - root_mean_squared_error: 2.5917 - val_loss: 1.9076 - val_root_mean_squared_error: 2.5451
Epoch 48/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.8873 - root_mean_squared_error: 2.4896 - val_loss: 1.8217 - val_root_mean_squared_error: 2.3413
Epoch 49/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.8233 - root_mean_squared_error: 2.4170 - val_loss: 1.7237 - val_root_mean_squared_error: 2.0586
Epoch 50/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.7795 - root_mean_squared_error: 2.2608 - val_loss: 1.7422 - val_root_mean_squared_error: 2.2960
Epoch 51/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.7283 - root_mean_squared_error: 2.1138 - val_loss: 1.6616 - val_root_mean_squared_error: 2.0372
Epoch 52/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.6634 - root_mean_squared_error: 2.0577 - val_loss: 1.6208 - val_root_mean_squared_error: 1.9096
Epoch 53/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.6020 - root_mean_squared_error: 1.8898 - val_loss: 1.5293 - val_root_mean_squared_error: 1.7964
Epoch 54/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.5437 - root_mean_squared_error: 1.8196 - val_loss: 1.4718 - val_root_mean_squared_error: 1.6636
Epoch 55/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.4934 - root_mean_squared_error: 1.6812 - val_loss: 1.4815 - val_root_mean_squared_error: 1.7225
Epoch 56/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.4226 - root_mean_squared_error: 1.5844 - val_loss: 1.4410 - val_root_mean_squared_error: 1.5321
Epoch 57/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.3964 - root_mean_squared_error: 1.4721 - val_loss: 1.3670 - val_root_mean_squared_error: 1.5063
Epoch 58/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.3646 - root_mean_squared_error: 1.4776 - val_loss: 1.3830 - val_root_mean_squared_error: 1.4543
Epoch 59/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.3287 - root_mean_squared_error: 1.3935 - val_loss: 1.3150 - val_root_mean_squared_error: 1.3399
Epoch 60/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2957 - root_mean_squared_error: 1.3059 - val_loss: 1.3045 - val_root_mean_squared_error: 1.2721
Epoch 61/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2911 - root_mean_squared_error: 1.3167 - val_loss: 1.3050 - val_root_mean_squared_error: 1.3444
Epoch 62/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2897 - root_mean_squared_error: 1.2974 - val_loss: 1.3762 - val_root_mean_squared_error: 1.3909
Epoch 63/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2906 - root_mean_squared_error: 1.2594 - val_loss: 1.3308 - val_root_mean_squared_error: 1.2587
Epoch 64/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2873 - root_mean_squared_error: 1.2371 - val_loss: 1.3247 - val_root_mean_squared_error: 1.2556
Epoch 65/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2936 - root_mean_squared_error: 1.2357 - val_loss: 1.3001 - val_root_mean_squared_error: 1.2808
Epoch 66/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2774 - root_mean_squared_error: 1.2208 - val_loss: 1.3008 - val_root_mean_squared_error: 1.2579
Epoch 67/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2785 - root_mean_squared_error: 1.1956 - val_loss: 1.3136 - val_root_mean_squared_error: 1.1549
Epoch 68/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.3022 - root_mean_squared_error: 1.2697 - val_loss: 1.2897 - val_root_mean_squared_error: 1.2557
Epoch 69/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2783 - root_mean_squared_error: 1.2133 - val_loss: 1.2994 - val_root_mean_squared_error: 1.1848
Epoch 70/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2748 - root_mean_squared_error: 1.2468 - val_loss: 1.2487 - val_root_mean_squared_error: 1.2234
Epoch 71/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.3044 - root_mean_squared_error: 1.2584 - val_loss: 1.2914 - val_root_mean_squared_error: 1.2204
Epoch 72/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.3003 - root_mean_squared_error: 1.2246 - val_loss: 1.2990 - val_root_mean_squared_error: 1.3182
Epoch 73/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2931 - root_mean_squared_error: 1.2326 - val_loss: 1.2694 - val_root_mean_squared_error: 1.2426
Epoch 74/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2812 - root_mean_squared_error: 1.2638 - val_loss: 1.2904 - val_root_mean_squared_error: 1.2861
Epoch 75/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2839 - root_mean_squared_error: 1.2589 - val_loss: 1.2607 - val_root_mean_squared_error: 1.2206
Epoch 76/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.3011 - root_mean_squared_error: 1.2677 - val_loss: 1.3174 - val_root_mean_squared_error: 1.1763
Epoch 77/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2659 - root_mean_squared_error: 1.2432 - val_loss: 1.2220 - val_root_mean_squared_error: 1.1466
Epoch 78/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2543 - root_mean_squared_error: 1.1910 - val_loss: 1.2699 - val_root_mean_squared_error: 1.2280
Epoch 79/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2747 - root_mean_squared_error: 1.2245 - val_loss: 1.2364 - val_root_mean_squared_error: 1.2088
Epoch 80/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2818 - root_mean_squared_error: 1.1950 - val_loss: 1.2331 - val_root_mean_squared_error: 1.2324
Epoch 81/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2795 - root_mean_squared_error: 1.2225 - val_loss: 1.3843 - val_root_mean_squared_error: 1.1531
Epoch 82/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2589 - root_mean_squared_error: 1.2104 - val_loss: 1.2576 - val_root_mean_squared_error: 1.1132
Epoch 83/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2724 - root_mean_squared_error: 1.2520 - val_loss: 1.2602 - val_root_mean_squared_error: 1.2762
Epoch 84/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2808 - root_mean_squared_error: 1.1789 - val_loss: 1.2554 - val_root_mean_squared_error: 1.2119
Epoch 85/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2690 - root_mean_squared_error: 1.1932 - val_loss: 1.2757 - val_root_mean_squared_error: 1.2473
Epoch 86/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2570 - root_mean_squared_error: 1.1678 - val_loss: 1.2473 - val_root_mean_squared_error: 1.1636
Epoch 87/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2660 - root_mean_squared_error: 1.2344 - val_loss: 1.2851 - val_root_mean_squared_error: 1.1793
Epoch 88/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.2462 - root_mean_squared_error: 1.1596 - val_loss: 1.2959 - val_root_mean_squared_error: 1.2687
Epoch 89/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.2542 - root_mean_squared_error: 1.1579 - val_loss: 1.2456 - val_root_mean_squared_error: 1.1803
Epoch 90/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.2420 - root_mean_squared_error: 1.1570 - val_loss: 1.3422 - val_root_mean_squared_error: 1.1635
Epoch 91/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.2424 - root_mean_squared_error: 1.1632 - val_loss: 1.2530 - val_root_mean_squared_error: 1.2598
Epoch 92/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.2426 - root_mean_squared_error: 1.1834 - val_loss: 1.2561 - val_root_mean_squared_error: 1.2399
Epoch 93/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.2327 - root_mean_squared_error: 1.1593 - val_loss: 1.2793 - val_root_mean_squared_error: 1.1470
Epoch 94/1000
17/17 [==============================] - 0s 10ms/step - loss: 1.2521 - root_mean_squared_error: 1.2406 - val_loss: 1.2537 - val_root_mean_squared_error: 1.1976
Epoch 95/1000
17/17 [==============================] - 0s 18ms/step - loss: 1.2430 - root_mean_squared_error: 1.1688 - val_loss: 1.2378 - val_root_mean_squared_error: 1.1282
Epoch 96/1000
17/17 [==============================] - 0s 10ms/step - loss: 1.2877 - root_mean_squared_error: 1.2104 - val_loss: 1.2516 - val_root_mean_squared_error: 1.1541
Epoch 97/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.2151 - root_mean_squared_error: 1.1574 - val_loss: 1.2308 - val_root_mean_squared_error: 1.1588
Epoch 98/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2205 - root_mean_squared_error: 1.1469 - val_loss: 1.2497 - val_root_mean_squared_error: 1.1888
Epoch 99/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.2730 - root_mean_squared_error: 1.2070 - val_loss: 1.2372 - val_root_mean_squared_error: 1.1904
Epoch 100/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2336 - root_mean_squared_error: 1.1824 - val_loss: 1.2166 - val_root_mean_squared_error: 1.1228
Epoch 101/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.2513 - root_mean_squared_error: 1.1504 - val_loss: 1.1961 - val_root_mean_squared_error: 1.1779
Epoch 102/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2268 - root_mean_squared_error: 1.1350 - val_loss: 1.2320 - val_root_mean_squared_error: 1.2296
Epoch 103/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2353 - root_mean_squared_error: 1.1633 - val_loss: 1.2737 - val_root_mean_squared_error: 1.1560
Epoch 104/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2274 - root_mean_squared_error: 1.1508 - val_loss: 1.2072 - val_root_mean_squared_error: 1.1489
Epoch 105/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2090 - root_mean_squared_error: 1.1233 - val_loss: 1.2444 - val_root_mean_squared_error: 1.2486
Epoch 106/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.2113 - root_mean_squared_error: 1.1468 - val_loss: 1.1935 - val_root_mean_squared_error: 1.1627
Epoch 107/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2160 - root_mean_squared_error: 1.1503 - val_loss: 1.2111 - val_root_mean_squared_error: 1.1754
Epoch 108/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2202 - root_mean_squared_error: 1.1738 - val_loss: 1.2164 - val_root_mean_squared_error: 1.0748
Epoch 109/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2199 - root_mean_squared_error: 1.1490 - val_loss: 1.2519 - val_root_mean_squared_error: 1.1357
Epoch 110/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2033 - root_mean_squared_error: 1.1536 - val_loss: 1.2120 - val_root_mean_squared_error: 1.1373
Epoch 111/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2568 - root_mean_squared_error: 1.2004 - val_loss: 1.1910 - val_root_mean_squared_error: 1.1829
Epoch 112/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2276 - root_mean_squared_error: 1.1676 - val_loss: 1.2467 - val_root_mean_squared_error: 1.1483
Epoch 113/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.2188 - root_mean_squared_error: 1.1603 - val_loss: 1.1857 - val_root_mean_squared_error: 1.2108
Epoch 114/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2702 - root_mean_squared_error: 1.1428 - val_loss: 1.2118 - val_root_mean_squared_error: 1.1270
Epoch 115/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2277 - root_mean_squared_error: 1.1581 - val_loss: 1.2043 - val_root_mean_squared_error: 1.2401
Epoch 116/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2152 - root_mean_squared_error: 1.1246 - val_loss: 1.2443 - val_root_mean_squared_error: 1.2347
Epoch 117/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2342 - root_mean_squared_error: 1.1575 - val_loss: 1.2665 - val_root_mean_squared_error: 1.1618
Epoch 118/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2368 - root_mean_squared_error: 1.1527 - val_loss: 1.1930 - val_root_mean_squared_error: 1.1209
Epoch 119/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2304 - root_mean_squared_error: 1.1323 - val_loss: 1.2043 - val_root_mean_squared_error: 1.2092
Epoch 120/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2048 - root_mean_squared_error: 1.1507 - val_loss: 1.2133 - val_root_mean_squared_error: 1.1843
Epoch 121/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2200 - root_mean_squared_error: 1.1359 - val_loss: 1.2080 - val_root_mean_squared_error: 1.1563
Epoch 122/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2218 - root_mean_squared_error: 1.1505 - val_loss: 1.2139 - val_root_mean_squared_error: 1.1204
Epoch 123/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2188 - root_mean_squared_error: 1.1340 - val_loss: 1.1840 - val_root_mean_squared_error: 1.0908
Epoch 124/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1951 - root_mean_squared_error: 1.1377 - val_loss: 1.1972 - val_root_mean_squared_error: 1.1349
Epoch 125/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2256 - root_mean_squared_error: 1.1396 - val_loss: 1.1921 - val_root_mean_squared_error: 1.1700
Epoch 126/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2264 - root_mean_squared_error: 1.1708 - val_loss: 1.1985 - val_root_mean_squared_error: 1.1199
Epoch 127/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1955 - root_mean_squared_error: 1.1245 - val_loss: 1.1869 - val_root_mean_squared_error: 1.0550
Epoch 128/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2293 - root_mean_squared_error: 1.1699 - val_loss: 1.2393 - val_root_mean_squared_error: 1.2133
Epoch 129/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.2475 - root_mean_squared_error: 1.1631 - val_loss: 1.1792 - val_root_mean_squared_error: 1.1564
Epoch 130/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2092 - root_mean_squared_error: 1.1150 - val_loss: 1.1965 - val_root_mean_squared_error: 1.1132
Epoch 131/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2124 - root_mean_squared_error: 1.1416 - val_loss: 1.2150 - val_root_mean_squared_error: 1.2224
Epoch 132/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1929 - root_mean_squared_error: 1.1203 - val_loss: 1.2019 - val_root_mean_squared_error: 1.0970
Epoch 133/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2575 - root_mean_squared_error: 1.1356 - val_loss: 1.1901 - val_root_mean_squared_error: 1.1212
Epoch 134/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2102 - root_mean_squared_error: 1.1294 - val_loss: 1.2233 - val_root_mean_squared_error: 1.0950
Epoch 135/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1882 - root_mean_squared_error: 1.1617 - val_loss: 1.1982 - val_root_mean_squared_error: 1.1402
Epoch 136/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2498 - root_mean_squared_error: 1.1490 - val_loss: 1.1815 - val_root_mean_squared_error: 1.0911
Epoch 137/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2039 - root_mean_squared_error: 1.1328 - val_loss: 1.2317 - val_root_mean_squared_error: 1.1201
Epoch 138/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2047 - root_mean_squared_error: 1.1164 - val_loss: 1.1934 - val_root_mean_squared_error: 1.1235
Epoch 139/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2165 - root_mean_squared_error: 1.1469 - val_loss: 1.2170 - val_root_mean_squared_error: 1.1903
Epoch 140/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2145 - root_mean_squared_error: 1.1357 - val_loss: 1.2186 - val_root_mean_squared_error: 1.0883
Epoch 141/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2193 - root_mean_squared_error: 1.1429 - val_loss: 1.2192 - val_root_mean_squared_error: 1.1608
Epoch 142/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1952 - root_mean_squared_error: 1.1260 - val_loss: 1.1788 - val_root_mean_squared_error: 1.1093
Epoch 143/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1973 - root_mean_squared_error: 1.1164 - val_loss: 1.1675 - val_root_mean_squared_error: 1.1359
Epoch 144/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2037 - root_mean_squared_error: 1.1279 - val_loss: 1.1872 - val_root_mean_squared_error: 1.1309
Epoch 145/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1720 - root_mean_squared_error: 1.0729 - val_loss: 1.2009 - val_root_mean_squared_error: 1.1653
Epoch 146/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2099 - root_mean_squared_error: 1.1197 - val_loss: 1.1831 - val_root_mean_squared_error: 1.1277
Epoch 147/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2319 - root_mean_squared_error: 1.1616 - val_loss: 1.1912 - val_root_mean_squared_error: 1.1419
Epoch 148/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1760 - root_mean_squared_error: 1.1291 - val_loss: 1.2184 - val_root_mean_squared_error: 1.0950
Epoch 149/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1951 - root_mean_squared_error: 1.1409 - val_loss: 1.1899 - val_root_mean_squared_error: 1.1287
Epoch 150/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2093 - root_mean_squared_error: 1.1458 - val_loss: 1.1960 - val_root_mean_squared_error: 1.1575
Epoch 151/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1766 - root_mean_squared_error: 1.1061 - val_loss: 1.1933 - val_root_mean_squared_error: 1.0854
Epoch 152/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1955 - root_mean_squared_error: 1.1007 - val_loss: 1.2003 - val_root_mean_squared_error: 1.1315
Epoch 153/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2181 - root_mean_squared_error: 1.1281 - val_loss: 1.1893 - val_root_mean_squared_error: 1.1472
Epoch 154/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2075 - root_mean_squared_error: 1.1103 - val_loss: 1.2351 - val_root_mean_squared_error: 1.1373
Epoch 155/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2123 - root_mean_squared_error: 1.1300 - val_loss: 1.1949 - val_root_mean_squared_error: 1.1229
Epoch 156/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1881 - root_mean_squared_error: 1.0853 - val_loss: 1.1824 - val_root_mean_squared_error: 1.1421
Epoch 157/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1916 - root_mean_squared_error: 1.1197 - val_loss: 1.2079 - val_root_mean_squared_error: 1.1647
Epoch 158/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1733 - root_mean_squared_error: 1.1443 - val_loss: 1.2213 - val_root_mean_squared_error: 1.1052
Epoch 159/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2051 - root_mean_squared_error: 1.1373 - val_loss: 1.1730 - val_root_mean_squared_error: 1.0555
Epoch 160/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2136 - root_mean_squared_error: 1.1300 - val_loss: 1.1812 - val_root_mean_squared_error: 1.1292
Epoch 161/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1848 - root_mean_squared_error: 1.0776 - val_loss: 1.1770 - val_root_mean_squared_error: 1.1607
Epoch 162/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1892 - root_mean_squared_error: 1.1201 - val_loss: 1.1948 - val_root_mean_squared_error: 1.1561
Epoch 163/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2020 - root_mean_squared_error: 1.1464 - val_loss: 1.2564 - val_root_mean_squared_error: 1.2153
Epoch 164/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1803 - root_mean_squared_error: 1.1263 - val_loss: 1.2212 - val_root_mean_squared_error: 1.1674
Epoch 165/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2112 - root_mean_squared_error: 1.1184 - val_loss: 1.1810 - val_root_mean_squared_error: 1.1156
Epoch 166/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2106 - root_mean_squared_error: 1.1246 - val_loss: 1.1957 - val_root_mean_squared_error: 1.1272
Epoch 167/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1890 - root_mean_squared_error: 1.0853 - val_loss: 1.1831 - val_root_mean_squared_error: 1.1485
Epoch 168/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2118 - root_mean_squared_error: 1.1227 - val_loss: 1.1997 - val_root_mean_squared_error: 1.0971
Epoch 169/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2263 - root_mean_squared_error: 1.1457 - val_loss: 1.1705 - val_root_mean_squared_error: 1.1093
Epoch 170/1000
17/17 [==============================] - 0s 14ms/step - loss: 1.2050 - root_mean_squared_error: 1.1334 - val_loss: 1.2030 - val_root_mean_squared_error: 1.0974
Epoch 171/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1898 - root_mean_squared_error: 1.1114 - val_loss: 1.1795 - val_root_mean_squared_error: 1.0999
Epoch 172/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1781 - root_mean_squared_error: 1.0871 - val_loss: 1.2483 - val_root_mean_squared_error: 1.1037
Epoch 173/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2013 - root_mean_squared_error: 1.1266 - val_loss: 1.1861 - val_root_mean_squared_error: 1.1123
Epoch 174/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1847 - root_mean_squared_error: 1.1122 - val_loss: 1.1985 - val_root_mean_squared_error: 1.0747
Epoch 175/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1771 - root_mean_squared_error: 1.1232 - val_loss: 1.1842 - val_root_mean_squared_error: 1.0836
Epoch 176/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1986 - root_mean_squared_error: 1.1419 - val_loss: 1.1900 - val_root_mean_squared_error: 1.0951
Epoch 177/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1890 - root_mean_squared_error: 1.1131 - val_loss: 1.1476 - val_root_mean_squared_error: 1.1174
Epoch 178/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2058 - root_mean_squared_error: 1.1102 - val_loss: 1.1636 - val_root_mean_squared_error: 1.1137
Epoch 179/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1880 - root_mean_squared_error: 1.1276 - val_loss: 1.1942 - val_root_mean_squared_error: 1.1049
Epoch 180/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1993 - root_mean_squared_error: 1.1239 - val_loss: 1.1761 - val_root_mean_squared_error: 1.1373
Epoch 181/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1709 - root_mean_squared_error: 1.0825 - val_loss: 1.1361 - val_root_mean_squared_error: 1.0720
Epoch 182/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1885 - root_mean_squared_error: 1.0742 - val_loss: 1.1890 - val_root_mean_squared_error: 1.0894
Epoch 183/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1922 - root_mean_squared_error: 1.1326 - val_loss: 1.1631 - val_root_mean_squared_error: 1.0957
Epoch 184/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1610 - root_mean_squared_error: 1.1056 - val_loss: 1.1973 - val_root_mean_squared_error: 1.1111
Epoch 185/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1958 - root_mean_squared_error: 1.1275 - val_loss: 1.1605 - val_root_mean_squared_error: 1.0616
Epoch 186/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1795 - root_mean_squared_error: 1.1134 - val_loss: 1.2015 - val_root_mean_squared_error: 1.1034
Epoch 187/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1915 - root_mean_squared_error: 1.0982 - val_loss: 1.1839 - val_root_mean_squared_error: 1.0911
Epoch 188/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1607 - root_mean_squared_error: 1.0841 - val_loss: 1.1833 - val_root_mean_squared_error: 1.1393
Epoch 189/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1904 - root_mean_squared_error: 1.1377 - val_loss: 1.1703 - val_root_mean_squared_error: 1.0720
Epoch 190/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1932 - root_mean_squared_error: 1.0975 - val_loss: 1.2062 - val_root_mean_squared_error: 1.1707
Epoch 191/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2020 - root_mean_squared_error: 1.1216 - val_loss: 1.1501 - val_root_mean_squared_error: 1.1034
Epoch 192/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1871 - root_mean_squared_error: 1.0865 - val_loss: 1.1690 - val_root_mean_squared_error: 1.0843
Epoch 193/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1684 - root_mean_squared_error: 1.1242 - val_loss: 1.1683 - val_root_mean_squared_error: 1.1093
Epoch 194/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1833 - root_mean_squared_error: 1.1186 - val_loss: 1.1540 - val_root_mean_squared_error: 1.0785
Epoch 195/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1819 - root_mean_squared_error: 1.1227 - val_loss: 1.1861 - val_root_mean_squared_error: 1.1171
Epoch 196/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1978 - root_mean_squared_error: 1.1531 - val_loss: 1.1560 - val_root_mean_squared_error: 1.1058
Epoch 197/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1894 - root_mean_squared_error: 1.1206 - val_loss: 1.1646 - val_root_mean_squared_error: 1.0916
Epoch 198/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1858 - root_mean_squared_error: 1.0975 - val_loss: 1.1704 - val_root_mean_squared_error: 1.0379
Epoch 199/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2077 - root_mean_squared_error: 1.1269 - val_loss: 1.2023 - val_root_mean_squared_error: 1.1006
Epoch 200/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1857 - root_mean_squared_error: 1.1024 - val_loss: 1.1607 - val_root_mean_squared_error: 1.1348
Epoch 201/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1991 - root_mean_squared_error: 1.1306 - val_loss: 1.1964 - val_root_mean_squared_error: 1.0963
Epoch 202/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1803 - root_mean_squared_error: 1.1018 - val_loss: 1.1929 - val_root_mean_squared_error: 1.0793
Epoch 203/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1745 - root_mean_squared_error: 1.1128 - val_loss: 1.1835 - val_root_mean_squared_error: 1.0657
Epoch 204/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1985 - root_mean_squared_error: 1.1219 - val_loss: 1.1679 - val_root_mean_squared_error: 1.1403
Epoch 205/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2140 - root_mean_squared_error: 1.1172 - val_loss: 1.1653 - val_root_mean_squared_error: 1.1153
Epoch 206/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1843 - root_mean_squared_error: 1.1079 - val_loss: 1.1737 - val_root_mean_squared_error: 1.1224
Epoch 207/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1829 - root_mean_squared_error: 1.0805 - val_loss: 1.1818 - val_root_mean_squared_error: 1.1198
Epoch 208/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1857 - root_mean_squared_error: 1.1026 - val_loss: 1.1748 - val_root_mean_squared_error: 1.0937
Epoch 209/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2006 - root_mean_squared_error: 1.1048 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0826
Epoch 210/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1853 - root_mean_squared_error: 1.1157 - val_loss: 1.1752 - val_root_mean_squared_error: 1.0802
Epoch 211/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1778 - root_mean_squared_error: 1.1348 - val_loss: 1.1697 - val_root_mean_squared_error: 1.0864
Epoch 212/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1984 - root_mean_squared_error: 1.1319 - val_loss: 1.2117 - val_root_mean_squared_error: 1.1147
Epoch 213/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1825 - root_mean_squared_error: 1.1116 - val_loss: 1.1976 - val_root_mean_squared_error: 1.1295
Epoch 214/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1611 - root_mean_squared_error: 1.1107 - val_loss: 1.1633 - val_root_mean_squared_error: 1.0857
Epoch 215/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1873 - root_mean_squared_error: 1.1153 - val_loss: 1.1446 - val_root_mean_squared_error: 1.0490
Epoch 216/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1921 - root_mean_squared_error: 1.1129 - val_loss: 1.1885 - val_root_mean_squared_error: 1.0949
Epoch 217/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1496 - root_mean_squared_error: 1.1120 - val_loss: 1.2003 - val_root_mean_squared_error: 1.1164
Epoch 218/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1953 - root_mean_squared_error: 1.1254 - val_loss: 1.1754 - val_root_mean_squared_error: 1.1199
Epoch 219/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1920 - root_mean_squared_error: 1.1369 - val_loss: 1.1550 - val_root_mean_squared_error: 1.0693
Epoch 220/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1752 - root_mean_squared_error: 1.1051 - val_loss: 1.1948 - val_root_mean_squared_error: 1.0942
Epoch 221/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1748 - root_mean_squared_error: 1.0991 - val_loss: 1.1666 - val_root_mean_squared_error: 1.0877
Epoch 222/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1809 - root_mean_squared_error: 1.1341 - val_loss: 1.1667 - val_root_mean_squared_error: 1.1405
Epoch 223/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1605 - root_mean_squared_error: 1.0893 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0493
Epoch 224/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1846 - root_mean_squared_error: 1.1036 - val_loss: 1.1939 - val_root_mean_squared_error: 1.1539
Epoch 225/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1708 - root_mean_squared_error: 1.1090 - val_loss: 1.1946 - val_root_mean_squared_error: 1.0918
Epoch 226/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1914 - root_mean_squared_error: 1.1307 - val_loss: 1.1555 - val_root_mean_squared_error: 1.0604
Epoch 227/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1889 - root_mean_squared_error: 1.1370 - val_loss: 1.1613 - val_root_mean_squared_error: 1.1074
Epoch 228/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1595 - root_mean_squared_error: 1.0637 - val_loss: 1.1960 - val_root_mean_squared_error: 1.0881
Epoch 229/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1810 - root_mean_squared_error: 1.1077 - val_loss: 1.1958 - val_root_mean_squared_error: 1.1195
Epoch 230/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1978 - root_mean_squared_error: 1.0757 - val_loss: 1.2355 - val_root_mean_squared_error: 1.1362
Epoch 231/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1771 - root_mean_squared_error: 1.0715 - val_loss: 1.2613 - val_root_mean_squared_error: 1.1745
Epoch 232/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1575 - root_mean_squared_error: 1.0955 - val_loss: 1.1848 - val_root_mean_squared_error: 1.1703
Epoch 233/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1656 - root_mean_squared_error: 1.0834 - val_loss: 1.1489 - val_root_mean_squared_error: 1.0833
Epoch 234/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1897 - root_mean_squared_error: 1.1035 - val_loss: 1.1738 - val_root_mean_squared_error: 1.1347
Epoch 235/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1872 - root_mean_squared_error: 1.1227 - val_loss: 1.1674 - val_root_mean_squared_error: 1.0889
Epoch 236/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1737 - root_mean_squared_error: 1.0981 - val_loss: 1.1721 - val_root_mean_squared_error: 1.0904
Epoch 237/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1646 - root_mean_squared_error: 1.0757 - val_loss: 1.1630 - val_root_mean_squared_error: 1.0591
Epoch 238/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1774 - root_mean_squared_error: 1.0984 - val_loss: 1.1950 - val_root_mean_squared_error: 1.1136
Epoch 239/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1790 - root_mean_squared_error: 1.0847 - val_loss: 1.1720 - val_root_mean_squared_error: 1.1149
Epoch 240/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1887 - root_mean_squared_error: 1.1374 - val_loss: 1.1726 - val_root_mean_squared_error: 1.0875
Epoch 241/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1626 - root_mean_squared_error: 1.0883 - val_loss: 1.1711 - val_root_mean_squared_error: 1.0989
Epoch 242/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1926 - root_mean_squared_error: 1.0799 - val_loss: 1.1895 - val_root_mean_squared_error: 1.1096
Epoch 243/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1695 - root_mean_squared_error: 1.1123 - val_loss: 1.1976 - val_root_mean_squared_error: 1.0709
Epoch 244/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1798 - root_mean_squared_error: 1.0794 - val_loss: 1.1915 - val_root_mean_squared_error: 1.1561
Epoch 245/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1747 - root_mean_squared_error: 1.1028 - val_loss: 1.1634 - val_root_mean_squared_error: 1.1043
Epoch 246/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1612 - root_mean_squared_error: 1.1053 - val_loss: 1.2462 - val_root_mean_squared_error: 1.1677
Epoch 247/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1879 - root_mean_squared_error: 1.0988 - val_loss: 1.1715 - val_root_mean_squared_error: 1.0945
Epoch 248/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1808 - root_mean_squared_error: 1.1063 - val_loss: 1.1850 - val_root_mean_squared_error: 1.1036
Epoch 249/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2077 - root_mean_squared_error: 1.0912 - val_loss: 1.1811 - val_root_mean_squared_error: 1.1123
Epoch 250/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1737 - root_mean_squared_error: 1.0728 - val_loss: 1.1569 - val_root_mean_squared_error: 1.1307
Epoch 251/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1833 - root_mean_squared_error: 1.1069 - val_loss: 1.1955 - val_root_mean_squared_error: 1.1350
Epoch 252/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1746 - root_mean_squared_error: 1.0928 - val_loss: 1.1687 - val_root_mean_squared_error: 1.0441
Epoch 253/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1651 - root_mean_squared_error: 1.0646 - val_loss: 1.1736 - val_root_mean_squared_error: 1.1561
Epoch 254/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1685 - root_mean_squared_error: 1.1009 - val_loss: 1.1546 - val_root_mean_squared_error: 1.1596
Epoch 255/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1594 - root_mean_squared_error: 1.1026 - val_loss: 1.1463 - val_root_mean_squared_error: 1.0895
Epoch 256/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1752 - root_mean_squared_error: 1.0914 - val_loss: 1.1546 - val_root_mean_squared_error: 1.0881
Epoch 257/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1648 - root_mean_squared_error: 1.0903 - val_loss: 1.1731 - val_root_mean_squared_error: 1.1133
Epoch 258/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1933 - root_mean_squared_error: 1.1339 - val_loss: 1.1729 - val_root_mean_squared_error: 1.1477
Epoch 259/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1746 - root_mean_squared_error: 1.0876 - val_loss: 1.1552 - val_root_mean_squared_error: 1.1218
Epoch 260/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1620 - root_mean_squared_error: 1.0859 - val_loss: 1.1835 - val_root_mean_squared_error: 1.0940
Epoch 261/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1761 - root_mean_squared_error: 1.1129 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0719
Epoch 262/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1822 - root_mean_squared_error: 1.0845 - val_loss: 1.1720 - val_root_mean_squared_error: 1.0874
Epoch 263/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1582 - root_mean_squared_error: 1.0799 - val_loss: 1.1668 - val_root_mean_squared_error: 1.1496
Epoch 264/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1612 - root_mean_squared_error: 1.0875 - val_loss: 1.1485 - val_root_mean_squared_error: 1.1107
Epoch 265/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1849 - root_mean_squared_error: 1.1086 - val_loss: 1.1840 - val_root_mean_squared_error: 1.0984
Epoch 266/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1595 - root_mean_squared_error: 1.0878 - val_loss: 1.1668 - val_root_mean_squared_error: 1.0781
Epoch 267/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1803 - root_mean_squared_error: 1.1152 - val_loss: 1.1662 - val_root_mean_squared_error: 1.0641
Epoch 268/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1642 - root_mean_squared_error: 1.0997 - val_loss: 1.1821 - val_root_mean_squared_error: 1.1124
Epoch 269/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1448 - root_mean_squared_error: 1.0887 - val_loss: 1.2022 - val_root_mean_squared_error: 1.1344
Epoch 270/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1891 - root_mean_squared_error: 1.1247 - val_loss: 1.1543 - val_root_mean_squared_error: 1.1006
Epoch 271/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1625 - root_mean_squared_error: 1.0678 - val_loss: 1.1730 - val_root_mean_squared_error: 1.0822
Epoch 272/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1445 - root_mean_squared_error: 1.0985 - val_loss: 1.1612 - val_root_mean_squared_error: 1.1014
Epoch 273/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1632 - root_mean_squared_error: 1.0998 - val_loss: 1.1759 - val_root_mean_squared_error: 1.1210
Epoch 274/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1605 - root_mean_squared_error: 1.0696 - val_loss: 1.1621 - val_root_mean_squared_error: 1.0599
Epoch 275/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.0864 - val_loss: 1.1540 - val_root_mean_squared_error: 1.0990
Epoch 276/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1819 - root_mean_squared_error: 1.0759 - val_loss: 1.1677 - val_root_mean_squared_error: 1.0775
Epoch 277/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1463 - root_mean_squared_error: 1.1065 - val_loss: 1.1670 - val_root_mean_squared_error: 1.1011
Epoch 278/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1564 - root_mean_squared_error: 1.0726 - val_loss: 1.1696 - val_root_mean_squared_error: 1.1327
Epoch 279/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1512 - root_mean_squared_error: 1.0915 - val_loss: 1.1585 - val_root_mean_squared_error: 1.0481
Epoch 280/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1731 - root_mean_squared_error: 1.0870 - val_loss: 1.1612 - val_root_mean_squared_error: 1.1131
Epoch 281/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1760 - root_mean_squared_error: 1.1031 - val_loss: 1.1686 - val_root_mean_squared_error: 1.0316
Epoch 282/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1515 - root_mean_squared_error: 1.0816 - val_loss: 1.1615 - val_root_mean_squared_error: 1.1235
Epoch 283/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1683 - root_mean_squared_error: 1.0962 - val_loss: 1.1488 - val_root_mean_squared_error: 1.0644
Epoch 284/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1866 - root_mean_squared_error: 1.1093 - val_loss: 1.1873 - val_root_mean_squared_error: 1.0724
Epoch 285/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1918 - root_mean_squared_error: 1.1159 - val_loss: 1.1664 - val_root_mean_squared_error: 1.0934
Epoch 286/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1887 - root_mean_squared_error: 1.1189 - val_loss: 1.1664 - val_root_mean_squared_error: 1.0797
Epoch 287/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1659 - root_mean_squared_error: 1.0758 - val_loss: 1.1812 - val_root_mean_squared_error: 1.0660
Epoch 288/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1971 - root_mean_squared_error: 1.1059 - val_loss: 1.1585 - val_root_mean_squared_error: 1.0747
Epoch 289/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1478 - root_mean_squared_error: 1.0753 - val_loss: 1.1626 - val_root_mean_squared_error: 1.1538
Epoch 290/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1615 - root_mean_squared_error: 1.1194 - val_loss: 1.1852 - val_root_mean_squared_error: 1.0507
Epoch 291/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1764 - root_mean_squared_error: 1.1070 - val_loss: 1.1699 - val_root_mean_squared_error: 1.1351
Epoch 292/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1647 - root_mean_squared_error: 1.1187 - val_loss: 1.1667 - val_root_mean_squared_error: 1.1012
Epoch 293/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1578 - root_mean_squared_error: 1.0695 - val_loss: 1.1656 - val_root_mean_squared_error: 1.0745
Epoch 294/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1757 - root_mean_squared_error: 1.0911 - val_loss: 1.1602 - val_root_mean_squared_error: 1.1156
Epoch 295/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1664 - root_mean_squared_error: 1.0919 - val_loss: 1.1638 - val_root_mean_squared_error: 1.1635
Epoch 296/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1925 - root_mean_squared_error: 1.1225 - val_loss: 1.1669 - val_root_mean_squared_error: 1.0964
Epoch 297/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1690 - root_mean_squared_error: 1.0986 - val_loss: 1.1731 - val_root_mean_squared_error: 1.0547
Epoch 298/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1682 - root_mean_squared_error: 1.0650 - val_loss: 1.1762 - val_root_mean_squared_error: 1.0469
Epoch 299/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1485 - root_mean_squared_error: 1.0429 - val_loss: 1.1895 - val_root_mean_squared_error: 1.0964
Epoch 300/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1760 - root_mean_squared_error: 1.0923 - val_loss: 1.1792 - val_root_mean_squared_error: 1.1484
Epoch 301/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1540 - root_mean_squared_error: 1.1005 - val_loss: 1.1772 - val_root_mean_squared_error: 1.1186
Epoch 302/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1591 - root_mean_squared_error: 1.0953 - val_loss: 1.1494 - val_root_mean_squared_error: 1.0869
Epoch 303/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1898 - root_mean_squared_error: 1.1113 - val_loss: 1.1523 - val_root_mean_squared_error: 1.1291
Epoch 304/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1691 - root_mean_squared_error: 1.0894 - val_loss: 1.1577 - val_root_mean_squared_error: 1.1203
Epoch 305/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1499 - root_mean_squared_error: 1.1235 - val_loss: 1.1774 - val_root_mean_squared_error: 1.0813
Epoch 306/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1692 - root_mean_squared_error: 1.0775 - val_loss: 1.1660 - val_root_mean_squared_error: 1.0657
Epoch 307/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1624 - root_mean_squared_error: 1.0896 - val_loss: 1.1565 - val_root_mean_squared_error: 1.0434
Epoch 308/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1587 - root_mean_squared_error: 1.0889 - val_loss: 1.1645 - val_root_mean_squared_error: 1.0701
Epoch 309/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1924 - root_mean_squared_error: 1.0953 - val_loss: 1.1814 - val_root_mean_squared_error: 1.0974
Epoch 310/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2044 - root_mean_squared_error: 1.1102 - val_loss: 1.1636 - val_root_mean_squared_error: 1.0579
Epoch 311/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1766 - root_mean_squared_error: 1.0789 - val_loss: 1.1757 - val_root_mean_squared_error: 1.0838
Epoch 312/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1640 - root_mean_squared_error: 1.0792 - val_loss: 1.1646 - val_root_mean_squared_error: 1.1157
Epoch 313/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1856 - root_mean_squared_error: 1.0880 - val_loss: 1.1912 - val_root_mean_squared_error: 1.0936
Epoch 314/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1839 - root_mean_squared_error: 1.0934 - val_loss: 1.1743 - val_root_mean_squared_error: 1.0665
Epoch 315/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1926 - root_mean_squared_error: 1.0752 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0840
Epoch 316/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1671 - root_mean_squared_error: 1.0960 - val_loss: 1.1628 - val_root_mean_squared_error: 1.1135
Epoch 317/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1769 - root_mean_squared_error: 1.1124 - val_loss: 1.1822 - val_root_mean_squared_error: 1.0881
Epoch 318/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1733 - root_mean_squared_error: 1.0885 - val_loss: 1.1500 - val_root_mean_squared_error: 1.0908
Epoch 319/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1708 - root_mean_squared_error: 1.1124 - val_loss: 1.1711 - val_root_mean_squared_error: 1.0698
Epoch 320/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1609 - root_mean_squared_error: 1.0593 - val_loss: 1.1654 - val_root_mean_squared_error: 1.1188
Epoch 321/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1842 - root_mean_squared_error: 1.0914 - val_loss: 1.2095 - val_root_mean_squared_error: 1.1310
Epoch 322/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1609 - root_mean_squared_error: 1.0785 - val_loss: 1.1739 - val_root_mean_squared_error: 1.1200
Epoch 323/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1726 - root_mean_squared_error: 1.1085 - val_loss: 1.1581 - val_root_mean_squared_error: 1.0662
Epoch 324/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1567 - root_mean_squared_error: 1.0934 - val_loss: 1.1790 - val_root_mean_squared_error: 1.0734
Epoch 325/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1705 - root_mean_squared_error: 1.0974 - val_loss: 1.1695 - val_root_mean_squared_error: 1.0777
Epoch 326/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1788 - root_mean_squared_error: 1.1047 - val_loss: 1.1684 - val_root_mean_squared_error: 1.1146
Epoch 327/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1556 - root_mean_squared_error: 1.0692 - val_loss: 1.1687 - val_root_mean_squared_error: 1.0784
Epoch 328/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1773 - root_mean_squared_error: 1.0855 - val_loss: 1.1643 - val_root_mean_squared_error: 1.0533
Epoch 329/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1742 - root_mean_squared_error: 1.0817 - val_loss: 1.1480 - val_root_mean_squared_error: 1.0992
Epoch 330/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1637 - root_mean_squared_error: 1.0783 - val_loss: 1.1750 - val_root_mean_squared_error: 1.0996
Epoch 331/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1577 - root_mean_squared_error: 1.0841 - val_loss: 1.1504 - val_root_mean_squared_error: 1.1367
Epoch 332/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1713 - root_mean_squared_error: 1.0650 - val_loss: 1.1470 - val_root_mean_squared_error: 1.0852
Epoch 333/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1592 - root_mean_squared_error: 1.0885 - val_loss: 1.1598 - val_root_mean_squared_error: 1.0638
Epoch 334/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1438 - root_mean_squared_error: 1.0854 - val_loss: 1.1669 - val_root_mean_squared_error: 1.1113
Epoch 335/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1692 - root_mean_squared_error: 1.0980 - val_loss: 1.1610 - val_root_mean_squared_error: 1.1076
Epoch 336/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1603 - root_mean_squared_error: 1.0780 - val_loss: 1.2025 - val_root_mean_squared_error: 1.1167
Epoch 337/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1712 - root_mean_squared_error: 1.0760 - val_loss: 1.1717 - val_root_mean_squared_error: 1.1020
Epoch 338/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1708 - root_mean_squared_error: 1.1014 - val_loss: 1.1441 - val_root_mean_squared_error: 1.0885
Epoch 339/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1681 - root_mean_squared_error: 1.1046 - val_loss: 1.1603 - val_root_mean_squared_error: 1.0880
Epoch 340/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1833 - root_mean_squared_error: 1.1178 - val_loss: 1.1529 - val_root_mean_squared_error: 1.0863
Epoch 341/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1734 - root_mean_squared_error: 1.1214 - val_loss: 1.1515 - val_root_mean_squared_error: 1.0919
Epoch 342/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1730 - root_mean_squared_error: 1.0787 - val_loss: 1.1548 - val_root_mean_squared_error: 1.0920
Epoch 343/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1609 - root_mean_squared_error: 1.0715 - val_loss: 1.1470 - val_root_mean_squared_error: 1.1187
Epoch 344/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1838 - root_mean_squared_error: 1.0998 - val_loss: 1.1536 - val_root_mean_squared_error: 1.0836
Epoch 345/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1723 - root_mean_squared_error: 1.0783 - val_loss: 1.1448 - val_root_mean_squared_error: 1.0685
Epoch 346/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1618 - root_mean_squared_error: 1.0987 - val_loss: 1.1482 - val_root_mean_squared_error: 1.0609
Epoch 347/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1598 - root_mean_squared_error: 1.0890 - val_loss: 1.1911 - val_root_mean_squared_error: 1.0929
Epoch 348/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1846 - root_mean_squared_error: 1.0925 - val_loss: 1.1550 - val_root_mean_squared_error: 1.0809
Epoch 349/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1718 - root_mean_squared_error: 1.0904 - val_loss: 1.1583 - val_root_mean_squared_error: 1.0867
Epoch 350/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1445 - root_mean_squared_error: 1.0665 - val_loss: 1.1565 - val_root_mean_squared_error: 1.0797
Epoch 351/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1593 - root_mean_squared_error: 1.0852 - val_loss: 1.1653 - val_root_mean_squared_error: 1.0715
Epoch 352/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1485 - root_mean_squared_error: 1.1051 - val_loss: 1.1649 - val_root_mean_squared_error: 1.1223
Epoch 353/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1617 - root_mean_squared_error: 1.0881 - val_loss: 1.1797 - val_root_mean_squared_error: 1.1274
Epoch 354/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1733 - root_mean_squared_error: 1.0857 - val_loss: 1.1588 - val_root_mean_squared_error: 1.0965
Epoch 355/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1682 - root_mean_squared_error: 1.0854 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0560
Epoch 356/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1325 - root_mean_squared_error: 1.0575 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0988
Epoch 357/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1703 - root_mean_squared_error: 1.0693 - val_loss: 1.1642 - val_root_mean_squared_error: 1.1213
Epoch 358/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1772 - root_mean_squared_error: 1.0868 - val_loss: 1.1610 - val_root_mean_squared_error: 1.1202
Epoch 359/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1743 - root_mean_squared_error: 1.0522 - val_loss: 1.1535 - val_root_mean_squared_error: 1.0652
Epoch 360/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1741 - root_mean_squared_error: 1.0998 - val_loss: 1.1612 - val_root_mean_squared_error: 1.0666
Epoch 361/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1555 - root_mean_squared_error: 1.0677 - val_loss: 1.1660 - val_root_mean_squared_error: 1.0824
Epoch 362/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1561 - root_mean_squared_error: 1.0686 - val_loss: 1.1699 - val_root_mean_squared_error: 1.1156
Epoch 363/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1818 - root_mean_squared_error: 1.0999 - val_loss: 1.1776 - val_root_mean_squared_error: 1.1283
Epoch 364/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1606 - root_mean_squared_error: 1.1004 - val_loss: 1.1638 - val_root_mean_squared_error: 1.1019
Epoch 365/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1695 - root_mean_squared_error: 1.0672 - val_loss: 1.1477 - val_root_mean_squared_error: 1.0467
Epoch 366/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1705 - root_mean_squared_error: 1.0894 - val_loss: 1.1691 - val_root_mean_squared_error: 1.0504
Epoch 367/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1623 - root_mean_squared_error: 1.0541 - val_loss: 1.1815 - val_root_mean_squared_error: 1.0694
Epoch 368/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1511 - root_mean_squared_error: 1.0954 - val_loss: 1.1461 - val_root_mean_squared_error: 1.0897
Epoch 369/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1769 - root_mean_squared_error: 1.0781 - val_loss: 1.1454 - val_root_mean_squared_error: 1.0635
Epoch 370/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1718 - root_mean_squared_error: 1.1040 - val_loss: 1.1649 - val_root_mean_squared_error: 1.1214
Epoch 371/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1694 - root_mean_squared_error: 1.1147 - val_loss: 1.1441 - val_root_mean_squared_error: 1.0617
Epoch 372/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1650 - root_mean_squared_error: 1.0827 - val_loss: 1.1643 - val_root_mean_squared_error: 1.1035
Epoch 373/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1635 - root_mean_squared_error: 1.0924 - val_loss: 1.1454 - val_root_mean_squared_error: 1.1251
Epoch 374/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1837 - root_mean_squared_error: 1.0932 - val_loss: 1.1783 - val_root_mean_squared_error: 1.0826
Epoch 375/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1849 - root_mean_squared_error: 1.1176 - val_loss: 1.1800 - val_root_mean_squared_error: 1.0610
Epoch 376/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1655 - root_mean_squared_error: 1.0602 - val_loss: 1.1634 - val_root_mean_squared_error: 1.0873
Epoch 377/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1519 - root_mean_squared_error: 1.0850 - val_loss: 1.1609 - val_root_mean_squared_error: 1.0255
Epoch 378/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1549 - root_mean_squared_error: 1.0874 - val_loss: 1.1600 - val_root_mean_squared_error: 1.0981
Epoch 379/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1676 - root_mean_squared_error: 1.0702 - val_loss: 1.1889 - val_root_mean_squared_error: 1.0996
Epoch 380/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1596 - root_mean_squared_error: 1.0802 - val_loss: 1.1624 - val_root_mean_squared_error: 1.0631
Epoch 381/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1704 - root_mean_squared_error: 1.1070 - val_loss: 1.1666 - val_root_mean_squared_error: 1.0809
Epoch 382/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1558 - root_mean_squared_error: 1.0786 - val_loss: 1.1647 - val_root_mean_squared_error: 1.0642
Epoch 383/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1744 - root_mean_squared_error: 1.1090 - val_loss: 1.1772 - val_root_mean_squared_error: 1.0729
Epoch 384/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1762 - root_mean_squared_error: 1.1140 - val_loss: 1.1633 - val_root_mean_squared_error: 1.0647
Epoch 385/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1648 - root_mean_squared_error: 1.1077 - val_loss: 1.1635 - val_root_mean_squared_error: 1.0557
Epoch 386/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1608 - root_mean_squared_error: 1.0980 - val_loss: 1.1466 - val_root_mean_squared_error: 1.0383
Epoch 387/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1744 - root_mean_squared_error: 1.0958 - val_loss: 1.2083 - val_root_mean_squared_error: 1.1020
Epoch 388/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1616 - root_mean_squared_error: 1.0911 - val_loss: 1.1492 - val_root_mean_squared_error: 1.0783
Epoch 389/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1881 - root_mean_squared_error: 1.0861 - val_loss: 1.1640 - val_root_mean_squared_error: 1.0896
Epoch 390/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1698 - root_mean_squared_error: 1.0784 - val_loss: 1.1605 - val_root_mean_squared_error: 1.1496
Epoch 391/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1557 - root_mean_squared_error: 1.0588 - val_loss: 1.1601 - val_root_mean_squared_error: 1.1125
Epoch 392/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1628 - root_mean_squared_error: 1.0975 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0636
Epoch 393/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.2097 - root_mean_squared_error: 1.1291 - val_loss: 1.1780 - val_root_mean_squared_error: 1.0411
Epoch 394/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1656 - root_mean_squared_error: 1.0878 - val_loss: 1.1559 - val_root_mean_squared_error: 1.1048
Epoch 395/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1752 - root_mean_squared_error: 1.0867 - val_loss: 1.1879 - val_root_mean_squared_error: 1.0917
Epoch 396/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1641 - root_mean_squared_error: 1.0698 - val_loss: 1.1648 - val_root_mean_squared_error: 1.0806
Epoch 397/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1521 - root_mean_squared_error: 1.1069 - val_loss: 1.1552 - val_root_mean_squared_error: 1.0632
Epoch 398/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1534 - root_mean_squared_error: 1.0863 - val_loss: 1.1490 - val_root_mean_squared_error: 1.1275
Epoch 399/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1843 - root_mean_squared_error: 1.1108 - val_loss: 1.1626 - val_root_mean_squared_error: 1.1148
Epoch 400/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1547 - root_mean_squared_error: 1.0913 - val_loss: 1.1737 - val_root_mean_squared_error: 1.1100
Epoch 401/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1658 - root_mean_squared_error: 1.0660 - val_loss: 1.1557 - val_root_mean_squared_error: 1.0977
Epoch 402/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1619 - root_mean_squared_error: 1.1060 - val_loss: 1.1890 - val_root_mean_squared_error: 1.0604
Epoch 403/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1925 - root_mean_squared_error: 1.0774 - val_loss: 1.1573 - val_root_mean_squared_error: 1.1118
Epoch 404/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1568 - root_mean_squared_error: 1.0667 - val_loss: 1.1809 - val_root_mean_squared_error: 1.0423
Epoch 405/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1716 - root_mean_squared_error: 1.0714 - val_loss: 1.1617 - val_root_mean_squared_error: 1.1223
Epoch 406/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1720 - root_mean_squared_error: 1.0862 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0598
Epoch 407/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1537 - root_mean_squared_error: 1.0845 - val_loss: 1.1675 - val_root_mean_squared_error: 1.0831
Epoch 408/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1595 - root_mean_squared_error: 1.0684 - val_loss: 1.1716 - val_root_mean_squared_error: 1.0386
Epoch 409/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1593 - root_mean_squared_error: 1.0826 - val_loss: 1.1564 - val_root_mean_squared_error: 1.1078
Epoch 410/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1696 - root_mean_squared_error: 1.0963 - val_loss: 1.1730 - val_root_mean_squared_error: 1.1118
Epoch 411/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1953 - root_mean_squared_error: 1.0886 - val_loss: 1.1491 - val_root_mean_squared_error: 1.0739
Epoch 412/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1430 - root_mean_squared_error: 1.0679 - val_loss: 1.1637 - val_root_mean_squared_error: 1.0908
Epoch 413/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1847 - root_mean_squared_error: 1.1096 - val_loss: 1.1458 - val_root_mean_squared_error: 1.0831
Epoch 414/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1630 - root_mean_squared_error: 1.0788 - val_loss: 1.1511 - val_root_mean_squared_error: 1.0474
Epoch 415/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1726 - root_mean_squared_error: 1.0912 - val_loss: 1.1570 - val_root_mean_squared_error: 1.1569
Epoch 416/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1749 - root_mean_squared_error: 1.0714 - val_loss: 1.1697 - val_root_mean_squared_error: 1.0994
Epoch 417/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1472 - root_mean_squared_error: 1.0988 - val_loss: 1.1800 - val_root_mean_squared_error: 1.0806
Epoch 418/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1641 - root_mean_squared_error: 1.1020 - val_loss: 1.1627 - val_root_mean_squared_error: 1.0784
Epoch 419/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1781 - root_mean_squared_error: 1.1056 - val_loss: 1.1674 - val_root_mean_squared_error: 1.0673
Epoch 420/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1370 - root_mean_squared_error: 1.0774 - val_loss: 1.1666 - val_root_mean_squared_error: 1.0919
Epoch 421/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1640 - root_mean_squared_error: 1.0857 - val_loss: 1.1687 - val_root_mean_squared_error: 1.0754
Epoch 422/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1628 - root_mean_squared_error: 1.0847 - val_loss: 1.1589 - val_root_mean_squared_error: 1.0890
Epoch 423/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1649 - root_mean_squared_error: 1.0895 - val_loss: 1.1536 - val_root_mean_squared_error: 1.1505
Epoch 424/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1799 - root_mean_squared_error: 1.0964 - val_loss: 1.1648 - val_root_mean_squared_error: 1.1318
Epoch 425/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1945 - root_mean_squared_error: 1.0793 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0794
Epoch 426/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1692 - root_mean_squared_error: 1.0814 - val_loss: 1.1707 - val_root_mean_squared_error: 1.0863
Epoch 427/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1401 - root_mean_squared_error: 1.0740 - val_loss: 1.1615 - val_root_mean_squared_error: 1.0740
Epoch 428/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1698 - root_mean_squared_error: 1.0994 - val_loss: 1.1806 - val_root_mean_squared_error: 1.0703
Epoch 429/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1620 - root_mean_squared_error: 1.0793 - val_loss: 1.1531 - val_root_mean_squared_error: 1.0531
Epoch 430/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1549 - root_mean_squared_error: 1.0719 - val_loss: 1.1736 - val_root_mean_squared_error: 1.1164
Epoch 431/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1711 - root_mean_squared_error: 1.0799 - val_loss: 1.1767 - val_root_mean_squared_error: 1.0991
Epoch 432/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1555 - root_mean_squared_error: 1.0780 - val_loss: 1.1537 - val_root_mean_squared_error: 1.0718
Epoch 433/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1665 - root_mean_squared_error: 1.1100 - val_loss: 1.1684 - val_root_mean_squared_error: 1.0840
Epoch 434/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1558 - root_mean_squared_error: 1.1136 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0909
Epoch 435/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1578 - root_mean_squared_error: 1.1008 - val_loss: 1.1773 - val_root_mean_squared_error: 1.0846
Epoch 436/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1670 - root_mean_squared_error: 1.0851 - val_loss: 1.1560 - val_root_mean_squared_error: 1.1129
Epoch 437/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1789 - root_mean_squared_error: 1.1014 - val_loss: 1.1627 - val_root_mean_squared_error: 1.0729
Epoch 438/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1729 - root_mean_squared_error: 1.0826 - val_loss: 1.1542 - val_root_mean_squared_error: 1.1085
Epoch 439/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1543 - root_mean_squared_error: 1.0806 - val_loss: 1.1561 - val_root_mean_squared_error: 1.0654
Epoch 440/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1841 - root_mean_squared_error: 1.0956 - val_loss: 1.1597 - val_root_mean_squared_error: 1.0488
Epoch 441/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1698 - root_mean_squared_error: 1.0760 - val_loss: 1.1715 - val_root_mean_squared_error: 1.0983
Epoch 442/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1728 - root_mean_squared_error: 1.0979 - val_loss: 1.1506 - val_root_mean_squared_error: 1.0344
Epoch 443/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1580 - root_mean_squared_error: 1.1133 - val_loss: 1.1672 - val_root_mean_squared_error: 1.0884
Epoch 444/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1908 - root_mean_squared_error: 1.1157 - val_loss: 1.1515 - val_root_mean_squared_error: 1.0894
Epoch 445/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1764 - root_mean_squared_error: 1.1006 - val_loss: 1.1556 - val_root_mean_squared_error: 1.1157
Epoch 446/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1752 - root_mean_squared_error: 1.0844 - val_loss: 1.1594 - val_root_mean_squared_error: 1.0525
Epoch 447/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1569 - root_mean_squared_error: 1.0812 - val_loss: 1.1750 - val_root_mean_squared_error: 1.0501
Epoch 448/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1772 - root_mean_squared_error: 1.1307 - val_loss: 1.1445 - val_root_mean_squared_error: 1.0741
Epoch 449/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1617 - root_mean_squared_error: 1.0922 - val_loss: 1.1670 - val_root_mean_squared_error: 1.0516
Epoch 450/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1534 - root_mean_squared_error: 1.0663 - val_loss: 1.1666 - val_root_mean_squared_error: 1.1015
Epoch 451/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1707 - root_mean_squared_error: 1.1039 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0868
Epoch 452/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1621 - root_mean_squared_error: 1.0753 - val_loss: 1.1413 - val_root_mean_squared_error: 1.0517
Epoch 453/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1427 - root_mean_squared_error: 1.0932 - val_loss: 1.1399 - val_root_mean_squared_error: 1.0565
Epoch 454/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1528 - root_mean_squared_error: 1.0747 - val_loss: 1.1658 - val_root_mean_squared_error: 1.0999
Epoch 455/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1600 - root_mean_squared_error: 1.0871 - val_loss: 1.1715 - val_root_mean_squared_error: 1.0784
Epoch 456/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1792 - root_mean_squared_error: 1.1357 - val_loss: 1.1573 - val_root_mean_squared_error: 1.1104
Epoch 457/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1621 - root_mean_squared_error: 1.1029 - val_loss: 1.1477 - val_root_mean_squared_error: 1.0821
Epoch 458/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1661 - root_mean_squared_error: 1.0925 - val_loss: 1.1526 - val_root_mean_squared_error: 1.0794
Epoch 459/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1532 - root_mean_squared_error: 1.0789 - val_loss: 1.1525 - val_root_mean_squared_error: 1.0712
Epoch 460/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1423 - root_mean_squared_error: 1.0740 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0881
Epoch 461/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1787 - root_mean_squared_error: 1.1160 - val_loss: 1.1580 - val_root_mean_squared_error: 0.9963
Epoch 462/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1785 - root_mean_squared_error: 1.0788 - val_loss: 1.1543 - val_root_mean_squared_error: 1.0935
Epoch 463/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1766 - root_mean_squared_error: 1.1029 - val_loss: 1.1662 - val_root_mean_squared_error: 1.0539
Epoch 464/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1731 - root_mean_squared_error: 1.0698 - val_loss: 1.1687 - val_root_mean_squared_error: 1.1419
Epoch 465/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1786 - root_mean_squared_error: 1.0942 - val_loss: 1.1715 - val_root_mean_squared_error: 1.1271
Epoch 466/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1904 - root_mean_squared_error: 1.0958 - val_loss: 1.1552 - val_root_mean_squared_error: 1.0469
Epoch 467/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1778 - root_mean_squared_error: 1.0786 - val_loss: 1.1526 - val_root_mean_squared_error: 1.0233
Epoch 468/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1527 - root_mean_squared_error: 1.0573 - val_loss: 1.1547 - val_root_mean_squared_error: 1.0549
Epoch 469/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1679 - root_mean_squared_error: 1.0822 - val_loss: 1.1480 - val_root_mean_squared_error: 1.0876
Epoch 470/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1704 - root_mean_squared_error: 1.0922 - val_loss: 1.1670 - val_root_mean_squared_error: 1.1073
Epoch 471/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1793 - root_mean_squared_error: 1.1133 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0858
Epoch 472/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1513 - root_mean_squared_error: 1.0705 - val_loss: 1.1605 - val_root_mean_squared_error: 1.1007
Epoch 473/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1594 - root_mean_squared_error: 1.0941 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0915
Epoch 474/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1634 - root_mean_squared_error: 1.0887 - val_loss: 1.1615 - val_root_mean_squared_error: 1.0510
Epoch 475/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1553 - root_mean_squared_error: 1.0820 - val_loss: 1.1663 - val_root_mean_squared_error: 1.0774
Epoch 476/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1387 - root_mean_squared_error: 1.0790 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0600
Epoch 477/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1788 - root_mean_squared_error: 1.0933 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0697
Epoch 478/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1724 - root_mean_squared_error: 1.0908 - val_loss: 1.1350 - val_root_mean_squared_error: 1.0705
Epoch 479/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1494 - root_mean_squared_error: 1.0762 - val_loss: 1.1771 - val_root_mean_squared_error: 1.0509
Epoch 480/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1581 - root_mean_squared_error: 1.1075 - val_loss: 1.1567 - val_root_mean_squared_error: 1.1301
Epoch 481/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1942 - root_mean_squared_error: 1.1168 - val_loss: 1.1550 - val_root_mean_squared_error: 1.0931
Epoch 482/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1851 - root_mean_squared_error: 1.0990 - val_loss: 1.1433 - val_root_mean_squared_error: 1.0700
Epoch 483/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1592 - root_mean_squared_error: 1.1013 - val_loss: 1.1515 - val_root_mean_squared_error: 1.0907
Epoch 484/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1578 - root_mean_squared_error: 1.0821 - val_loss: 1.1680 - val_root_mean_squared_error: 1.0743
Epoch 485/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1695 - root_mean_squared_error: 1.0700 - val_loss: 1.1592 - val_root_mean_squared_error: 1.0755
Epoch 486/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1664 - root_mean_squared_error: 1.0840 - val_loss: 1.1696 - val_root_mean_squared_error: 1.0709
Epoch 487/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1517 - root_mean_squared_error: 1.0313 - val_loss: 1.1575 - val_root_mean_squared_error: 1.1214
Epoch 488/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1503 - root_mean_squared_error: 1.0563 - val_loss: 1.1718 - val_root_mean_squared_error: 1.0507
Epoch 489/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1875 - root_mean_squared_error: 1.1058 - val_loss: 1.1585 - val_root_mean_squared_error: 1.0693
Epoch 490/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1623 - root_mean_squared_error: 1.0576 - val_loss: 1.1608 - val_root_mean_squared_error: 1.0662
Epoch 491/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1851 - root_mean_squared_error: 1.0956 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0574
Epoch 492/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1589 - root_mean_squared_error: 1.0448 - val_loss: 1.1626 - val_root_mean_squared_error: 1.0930
Epoch 493/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1715 - root_mean_squared_error: 1.1035 - val_loss: 1.1567 - val_root_mean_squared_error: 1.0723
Epoch 494/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1344 - root_mean_squared_error: 1.0536 - val_loss: 1.1726 - val_root_mean_squared_error: 1.0900
Epoch 495/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1731 - root_mean_squared_error: 1.1015 - val_loss: 1.1555 - val_root_mean_squared_error: 1.0978
Epoch 496/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1498 - root_mean_squared_error: 1.0615 - val_loss: 1.1498 - val_root_mean_squared_error: 1.0713
Epoch 497/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1300 - root_mean_squared_error: 1.0601 - val_loss: 1.1511 - val_root_mean_squared_error: 1.0998
Epoch 498/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1681 - root_mean_squared_error: 1.1134 - val_loss: 1.1692 - val_root_mean_squared_error: 1.1091
Epoch 499/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1552 - root_mean_squared_error: 1.0694 - val_loss: 1.1565 - val_root_mean_squared_error: 1.0677
Epoch 500/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1626 - root_mean_squared_error: 1.0813 - val_loss: 1.1615 - val_root_mean_squared_error: 1.0579
Epoch 501/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1494 - root_mean_squared_error: 1.0805 - val_loss: 1.1723 - val_root_mean_squared_error: 1.1371
Epoch 502/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1664 - root_mean_squared_error: 1.0967 - val_loss: 1.1616 - val_root_mean_squared_error: 1.0826
Epoch 503/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1589 - root_mean_squared_error: 1.0533 - val_loss: 1.1790 - val_root_mean_squared_error: 1.1231
Epoch 504/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1723 - root_mean_squared_error: 1.0737 - val_loss: 1.1644 - val_root_mean_squared_error: 1.0839
Epoch 505/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1563 - root_mean_squared_error: 1.0997 - val_loss: 1.1552 - val_root_mean_squared_error: 1.0735
Epoch 506/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1645 - root_mean_squared_error: 1.0708 - val_loss: 1.1485 - val_root_mean_squared_error: 1.0803
Epoch 507/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1544 - root_mean_squared_error: 1.0668 - val_loss: 1.1478 - val_root_mean_squared_error: 1.1023
Epoch 508/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1549 - root_mean_squared_error: 1.0824 - val_loss: 1.1406 - val_root_mean_squared_error: 1.0220
Epoch 509/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1561 - root_mean_squared_error: 1.0497 - val_loss: 1.1444 - val_root_mean_squared_error: 1.0999
Epoch 510/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1662 - root_mean_squared_error: 1.0918 - val_loss: 1.1470 - val_root_mean_squared_error: 1.0137
Epoch 511/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1520 - root_mean_squared_error: 1.0847 - val_loss: 1.1617 - val_root_mean_squared_error: 1.0705
Epoch 512/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1689 - root_mean_squared_error: 1.0769 - val_loss: 1.1533 - val_root_mean_squared_error: 1.0312
Epoch 513/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1453 - root_mean_squared_error: 1.0526 - val_loss: 1.1509 - val_root_mean_squared_error: 1.0680
Epoch 514/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1679 - root_mean_squared_error: 1.0913 - val_loss: 1.1671 - val_root_mean_squared_error: 1.0610
Epoch 515/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1623 - root_mean_squared_error: 1.0717 - val_loss: 1.1632 - val_root_mean_squared_error: 1.0021
Epoch 516/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1399 - root_mean_squared_error: 1.0275 - val_loss: 1.1772 - val_root_mean_squared_error: 1.0567
Epoch 517/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1730 - root_mean_squared_error: 1.0913 - val_loss: 1.1648 - val_root_mean_squared_error: 1.0584
Epoch 518/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1502 - root_mean_squared_error: 1.0887 - val_loss: 1.1521 - val_root_mean_squared_error: 1.0495
Epoch 519/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1897 - root_mean_squared_error: 1.0948 - val_loss: 1.1532 - val_root_mean_squared_error: 1.0371
Epoch 520/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1468 - root_mean_squared_error: 1.0685 - val_loss: 1.1664 - val_root_mean_squared_error: 1.0871
Epoch 521/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1555 - root_mean_squared_error: 1.1006 - val_loss: 1.1809 - val_root_mean_squared_error: 1.0573
Epoch 522/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1625 - root_mean_squared_error: 1.0980 - val_loss: 1.1566 - val_root_mean_squared_error: 1.0576
Epoch 523/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1525 - root_mean_squared_error: 1.0700 - val_loss: 1.1678 - val_root_mean_squared_error: 1.1050
Epoch 524/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1507 - root_mean_squared_error: 1.0591 - val_loss: 1.1726 - val_root_mean_squared_error: 1.0936
Epoch 525/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1507 - root_mean_squared_error: 1.0472 - val_loss: 1.1465 - val_root_mean_squared_error: 1.0600
Epoch 526/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1628 - root_mean_squared_error: 1.0847 - val_loss: 1.1546 - val_root_mean_squared_error: 1.0993
Epoch 527/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1408 - root_mean_squared_error: 1.0737 - val_loss: 1.1676 - val_root_mean_squared_error: 1.1235
Epoch 528/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1596 - root_mean_squared_error: 1.0654 - val_loss: 1.1656 - val_root_mean_squared_error: 1.0776
Epoch 529/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1919 - root_mean_squared_error: 1.0693 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0658
Epoch 530/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1439 - root_mean_squared_error: 1.0845 - val_loss: 1.1664 - val_root_mean_squared_error: 1.0350
Epoch 531/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1776 - root_mean_squared_error: 1.0689 - val_loss: 1.1663 - val_root_mean_squared_error: 1.0855
Epoch 532/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1549 - root_mean_squared_error: 1.0655 - val_loss: 1.1637 - val_root_mean_squared_error: 1.0877
Epoch 533/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1883 - root_mean_squared_error: 1.0809 - val_loss: 1.1639 - val_root_mean_squared_error: 1.1018
Epoch 534/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1811 - root_mean_squared_error: 1.1071 - val_loss: 1.1565 - val_root_mean_squared_error: 1.0752
Epoch 535/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1633 - root_mean_squared_error: 1.0829 - val_loss: 1.1652 - val_root_mean_squared_error: 1.0957
Epoch 536/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1573 - root_mean_squared_error: 1.0845 - val_loss: 1.1606 - val_root_mean_squared_error: 1.1357
Epoch 537/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1678 - root_mean_squared_error: 1.0866 - val_loss: 1.1443 - val_root_mean_squared_error: 1.0985
Epoch 538/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1553 - root_mean_squared_error: 1.0888 - val_loss: 1.1553 - val_root_mean_squared_error: 1.0699
Epoch 539/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1499 - root_mean_squared_error: 1.0913 - val_loss: 1.1633 - val_root_mean_squared_error: 1.1146
Epoch 540/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1474 - root_mean_squared_error: 1.0417 - val_loss: 1.1562 - val_root_mean_squared_error: 1.0777
Epoch 541/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1580 - root_mean_squared_error: 1.0826 - val_loss: 1.1499 - val_root_mean_squared_error: 1.0735
Epoch 542/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1821 - root_mean_squared_error: 1.0870 - val_loss: 1.1589 - val_root_mean_squared_error: 1.0858
Epoch 543/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.0862 - val_loss: 1.1583 - val_root_mean_squared_error: 1.0613
Epoch 544/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1644 - root_mean_squared_error: 1.1023 - val_loss: 1.1485 - val_root_mean_squared_error: 1.0526
Epoch 545/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1572 - root_mean_squared_error: 1.0437 - val_loss: 1.1594 - val_root_mean_squared_error: 1.0755
Epoch 546/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1708 - root_mean_squared_error: 1.1099 - val_loss: 1.1470 - val_root_mean_squared_error: 1.0541
Epoch 547/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1641 - root_mean_squared_error: 1.0843 - val_loss: 1.1434 - val_root_mean_squared_error: 1.0816
Epoch 548/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1466 - root_mean_squared_error: 1.0738 - val_loss: 1.1551 - val_root_mean_squared_error: 1.0856
Epoch 549/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1655 - root_mean_squared_error: 1.0629 - val_loss: 1.1571 - val_root_mean_squared_error: 1.0516
Epoch 550/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1674 - root_mean_squared_error: 1.0974 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0264
Epoch 551/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1432 - root_mean_squared_error: 1.0841 - val_loss: 1.1521 - val_root_mean_squared_error: 1.0976
Epoch 552/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1513 - root_mean_squared_error: 1.0891 - val_loss: 1.1782 - val_root_mean_squared_error: 1.1002
Epoch 553/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1822 - root_mean_squared_error: 1.1014 - val_loss: 1.1504 - val_root_mean_squared_error: 1.1261
Epoch 554/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1876 - root_mean_squared_error: 1.0883 - val_loss: 1.1496 - val_root_mean_squared_error: 1.0757
Epoch 555/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1749 - root_mean_squared_error: 1.0705 - val_loss: 1.1497 - val_root_mean_squared_error: 1.0554
Epoch 556/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1638 - root_mean_squared_error: 1.0596 - val_loss: 1.1619 - val_root_mean_squared_error: 1.0889
Epoch 557/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1413 - root_mean_squared_error: 1.0502 - val_loss: 1.2233 - val_root_mean_squared_error: 1.1197
Epoch 558/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1511 - root_mean_squared_error: 1.0979 - val_loss: 1.1817 - val_root_mean_squared_error: 1.0976
Epoch 559/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1536 - root_mean_squared_error: 1.0878 - val_loss: 1.1617 - val_root_mean_squared_error: 1.1214
Epoch 560/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1721 - root_mean_squared_error: 1.1253 - val_loss: 1.1614 - val_root_mean_squared_error: 1.0857
Epoch 561/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1462 - root_mean_squared_error: 1.0648 - val_loss: 1.1744 - val_root_mean_squared_error: 1.1356
Epoch 562/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1541 - root_mean_squared_error: 1.0740 - val_loss: 1.1679 - val_root_mean_squared_error: 1.0957
Epoch 563/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1630 - root_mean_squared_error: 1.0710 - val_loss: 1.1643 - val_root_mean_squared_error: 1.0731
Epoch 564/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1554 - root_mean_squared_error: 1.0691 - val_loss: 1.1653 - val_root_mean_squared_error: 1.0903
Epoch 565/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1477 - root_mean_squared_error: 1.0828 - val_loss: 1.1587 - val_root_mean_squared_error: 1.0643
Epoch 566/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1534 - root_mean_squared_error: 1.0922 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0587
Epoch 567/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1698 - root_mean_squared_error: 1.1047 - val_loss: 1.1581 - val_root_mean_squared_error: 1.1239
Epoch 568/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1876 - root_mean_squared_error: 1.0928 - val_loss: 1.1613 - val_root_mean_squared_error: 1.0427
Epoch 569/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1457 - root_mean_squared_error: 1.0780 - val_loss: 1.1981 - val_root_mean_squared_error: 1.1336
Epoch 570/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1659 - root_mean_squared_error: 1.0846 - val_loss: 1.1438 - val_root_mean_squared_error: 1.1027
Epoch 571/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1496 - root_mean_squared_error: 1.0646 - val_loss: 1.1626 - val_root_mean_squared_error: 1.0604
Epoch 572/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1453 - root_mean_squared_error: 1.0515 - val_loss: 1.1540 - val_root_mean_squared_error: 1.0945
Epoch 573/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1582 - root_mean_squared_error: 1.0772 - val_loss: 1.1534 - val_root_mean_squared_error: 1.0582
Epoch 574/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1756 - root_mean_squared_error: 1.0813 - val_loss: 1.1606 - val_root_mean_squared_error: 1.0830
Epoch 575/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1676 - root_mean_squared_error: 1.0965 - val_loss: 1.1528 - val_root_mean_squared_error: 1.0494
Epoch 576/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1662 - root_mean_squared_error: 1.0818 - val_loss: 1.1838 - val_root_mean_squared_error: 1.0764
Epoch 577/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1388 - root_mean_squared_error: 1.0560 - val_loss: 1.1596 - val_root_mean_squared_error: 1.1070
Epoch 578/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1748 - root_mean_squared_error: 1.1030 - val_loss: 1.1537 - val_root_mean_squared_error: 1.0252
Epoch 579/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1748 - root_mean_squared_error: 1.0754 - val_loss: 1.1660 - val_root_mean_squared_error: 1.0687
Epoch 580/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1555 - root_mean_squared_error: 1.0976 - val_loss: 1.1472 - val_root_mean_squared_error: 1.0532
Epoch 581/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1640 - root_mean_squared_error: 1.0818 - val_loss: 1.1581 - val_root_mean_squared_error: 1.0669
Epoch 582/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1567 - root_mean_squared_error: 1.0826 - val_loss: 1.1546 - val_root_mean_squared_error: 1.1138
Epoch 583/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1825 - root_mean_squared_error: 1.1009 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0669
Epoch 584/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1659 - root_mean_squared_error: 1.0655 - val_loss: 1.1606 - val_root_mean_squared_error: 1.1164
Epoch 585/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1614 - root_mean_squared_error: 1.0922 - val_loss: 1.1467 - val_root_mean_squared_error: 1.0683
Epoch 586/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1635 - root_mean_squared_error: 1.0840 - val_loss: 1.1585 - val_root_mean_squared_error: 1.0962
Epoch 587/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1686 - root_mean_squared_error: 1.0790 - val_loss: 1.1803 - val_root_mean_squared_error: 1.0784
Epoch 588/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1659 - root_mean_squared_error: 1.0769 - val_loss: 1.1553 - val_root_mean_squared_error: 1.0330
Epoch 589/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1816 - root_mean_squared_error: 1.0785 - val_loss: 1.1573 - val_root_mean_squared_error: 1.1101
Epoch 590/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1487 - root_mean_squared_error: 1.0527 - val_loss: 1.1498 - val_root_mean_squared_error: 1.0585
Epoch 591/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1439 - root_mean_squared_error: 1.0642 - val_loss: 1.1578 - val_root_mean_squared_error: 1.0273
Epoch 592/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1341 - root_mean_squared_error: 1.0614 - val_loss: 1.1749 - val_root_mean_squared_error: 1.0954
Epoch 593/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1555 - root_mean_squared_error: 1.0771 - val_loss: 1.1416 - val_root_mean_squared_error: 1.0829
Epoch 594/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1569 - root_mean_squared_error: 1.0734 - val_loss: 1.1796 - val_root_mean_squared_error: 1.0619
Epoch 595/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1428 - root_mean_squared_error: 1.0713 - val_loss: 1.1585 - val_root_mean_squared_error: 1.0805
Epoch 596/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1641 - root_mean_squared_error: 1.0824 - val_loss: 1.1509 - val_root_mean_squared_error: 1.0940
Epoch 597/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1612 - root_mean_squared_error: 1.0671 - val_loss: 1.1489 - val_root_mean_squared_error: 1.0989
Epoch 598/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1614 - root_mean_squared_error: 1.1092 - val_loss: 1.1733 - val_root_mean_squared_error: 1.0960
Epoch 599/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1431 - root_mean_squared_error: 1.0506 - val_loss: 1.1748 - val_root_mean_squared_error: 1.0811
Epoch 600/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1642 - root_mean_squared_error: 1.0830 - val_loss: 1.1661 - val_root_mean_squared_error: 1.0745
Epoch 601/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1696 - root_mean_squared_error: 1.0881 - val_loss: 1.1499 - val_root_mean_squared_error: 1.0342
Epoch 602/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1756 - root_mean_squared_error: 1.0612 - val_loss: 1.1571 - val_root_mean_squared_error: 1.0770
Epoch 603/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1710 - root_mean_squared_error: 1.0907 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0667
Epoch 604/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1632 - root_mean_squared_error: 1.0569 - val_loss: 1.1461 - val_root_mean_squared_error: 1.0748
Epoch 605/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1770 - root_mean_squared_error: 1.0857 - val_loss: 1.1469 - val_root_mean_squared_error: 1.0886
Epoch 606/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1719 - root_mean_squared_error: 1.0727 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0734
Epoch 607/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1664 - root_mean_squared_error: 1.0818 - val_loss: 1.1436 - val_root_mean_squared_error: 1.0523
Epoch 608/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1409 - root_mean_squared_error: 1.0520 - val_loss: 1.1578 - val_root_mean_squared_error: 1.0789
Epoch 609/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1516 - root_mean_squared_error: 1.0656 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0852
Epoch 610/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1488 - root_mean_squared_error: 1.0629 - val_loss: 1.1678 - val_root_mean_squared_error: 1.0512
Epoch 611/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1672 - root_mean_squared_error: 1.0558 - val_loss: 1.1675 - val_root_mean_squared_error: 1.1078
Epoch 612/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1515 - root_mean_squared_error: 1.0878 - val_loss: 1.1507 - val_root_mean_squared_error: 1.1201
Epoch 613/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1598 - root_mean_squared_error: 1.0587 - val_loss: 1.1503 - val_root_mean_squared_error: 1.0990
Epoch 614/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1565 - root_mean_squared_error: 1.0863 - val_loss: 1.1549 - val_root_mean_squared_error: 1.0551
Epoch 615/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1847 - root_mean_squared_error: 1.1200 - val_loss: 1.1565 - val_root_mean_squared_error: 1.0821
Epoch 616/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1529 - root_mean_squared_error: 1.0789 - val_loss: 1.1400 - val_root_mean_squared_error: 1.0281
Epoch 617/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1809 - root_mean_squared_error: 1.0778 - val_loss: 1.1708 - val_root_mean_squared_error: 1.0259
Epoch 618/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1782 - root_mean_squared_error: 1.0660 - val_loss: 1.1610 - val_root_mean_squared_error: 1.0902
Epoch 619/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1774 - root_mean_squared_error: 1.0930 - val_loss: 1.1701 - val_root_mean_squared_error: 1.0378
Epoch 620/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1684 - root_mean_squared_error: 1.0854 - val_loss: 1.1768 - val_root_mean_squared_error: 1.0216
Epoch 621/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1457 - root_mean_squared_error: 1.0531 - val_loss: 1.1517 - val_root_mean_squared_error: 1.0616
Epoch 622/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1452 - root_mean_squared_error: 1.0633 - val_loss: 1.1481 - val_root_mean_squared_error: 1.0387
Epoch 623/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1494 - root_mean_squared_error: 1.0759 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0953
Epoch 624/1000
17/17 [==============================] - 0s 17ms/step - loss: 1.1666 - root_mean_squared_error: 1.0918 - val_loss: 1.1545 - val_root_mean_squared_error: 1.0580
Epoch 625/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1613 - root_mean_squared_error: 1.0621 - val_loss: 1.1809 - val_root_mean_squared_error: 1.0311
Epoch 626/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1636 - root_mean_squared_error: 1.0763 - val_loss: 1.1426 - val_root_mean_squared_error: 1.0789
Epoch 627/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1619 - root_mean_squared_error: 1.0589 - val_loss: 1.1457 - val_root_mean_squared_error: 1.0750
Epoch 628/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1609 - root_mean_squared_error: 1.1008 - val_loss: 1.1597 - val_root_mean_squared_error: 1.1101
Epoch 629/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1499 - root_mean_squared_error: 1.0730 - val_loss: 1.1531 - val_root_mean_squared_error: 1.0904
Epoch 630/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1497 - root_mean_squared_error: 1.0792 - val_loss: 1.1594 - val_root_mean_squared_error: 1.0692
Epoch 631/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1557 - root_mean_squared_error: 1.0703 - val_loss: 1.1631 - val_root_mean_squared_error: 1.0672
Epoch 632/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1437 - root_mean_squared_error: 1.0634 - val_loss: 1.1511 - val_root_mean_squared_error: 1.0547
Epoch 633/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1695 - root_mean_squared_error: 1.0972 - val_loss: 1.1537 - val_root_mean_squared_error: 1.0720
Epoch 634/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1748 - root_mean_squared_error: 1.0846 - val_loss: 1.1477 - val_root_mean_squared_error: 1.1116
Epoch 635/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1566 - root_mean_squared_error: 1.0685 - val_loss: 1.1621 - val_root_mean_squared_error: 1.0298
Epoch 636/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1489 - root_mean_squared_error: 1.0593 - val_loss: 1.1542 - val_root_mean_squared_error: 1.1333
Epoch 637/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1483 - root_mean_squared_error: 1.0837 - val_loss: 1.1631 - val_root_mean_squared_error: 1.0768
Epoch 638/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1807 - root_mean_squared_error: 1.1210 - val_loss: 1.1426 - val_root_mean_squared_error: 1.0535
Epoch 639/1000
17/17 [==============================] - 0s 17ms/step - loss: 1.1296 - root_mean_squared_error: 1.0241 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0484
Epoch 640/1000
17/17 [==============================] - 0s 10ms/step - loss: 1.1461 - root_mean_squared_error: 1.0823 - val_loss: 1.1392 - val_root_mean_squared_error: 1.0746
Epoch 641/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1559 - root_mean_squared_error: 1.0784 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0652
Epoch 642/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1776 - root_mean_squared_error: 1.0762 - val_loss: 1.1362 - val_root_mean_squared_error: 1.0828
Epoch 643/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1552 - root_mean_squared_error: 1.1044 - val_loss: 1.1557 - val_root_mean_squared_error: 1.0815
Epoch 644/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1660 - root_mean_squared_error: 1.0773 - val_loss: 1.1688 - val_root_mean_squared_error: 1.0772
Epoch 645/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1709 - root_mean_squared_error: 1.0737 - val_loss: 1.1514 - val_root_mean_squared_error: 1.0782
Epoch 646/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1440 - root_mean_squared_error: 1.0829 - val_loss: 1.1525 - val_root_mean_squared_error: 1.0633
Epoch 647/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1514 - root_mean_squared_error: 1.0867 - val_loss: 1.1826 - val_root_mean_squared_error: 1.0921
Epoch 648/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1511 - root_mean_squared_error: 1.0649 - val_loss: 1.1603 - val_root_mean_squared_error: 1.0662
Epoch 649/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1363 - root_mean_squared_error: 1.0633 - val_loss: 1.1669 - val_root_mean_squared_error: 1.0951
Epoch 650/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1637 - root_mean_squared_error: 1.0573 - val_loss: 1.1613 - val_root_mean_squared_error: 1.0868
Epoch 651/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1709 - root_mean_squared_error: 1.1016 - val_loss: 1.1497 - val_root_mean_squared_error: 1.0414
Epoch 652/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1744 - root_mean_squared_error: 1.0546 - val_loss: 1.1733 - val_root_mean_squared_error: 1.0621
Epoch 653/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1691 - root_mean_squared_error: 1.0782 - val_loss: 1.1736 - val_root_mean_squared_error: 1.0636
Epoch 654/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1370 - root_mean_squared_error: 1.0749 - val_loss: 1.1396 - val_root_mean_squared_error: 1.0503
Epoch 655/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1426 - root_mean_squared_error: 1.0594 - val_loss: 1.1719 - val_root_mean_squared_error: 1.1077
Epoch 656/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1558 - root_mean_squared_error: 1.0355 - val_loss: 1.1464 - val_root_mean_squared_error: 1.0943
Epoch 657/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1341 - root_mean_squared_error: 1.0550 - val_loss: 1.1703 - val_root_mean_squared_error: 1.1033
Epoch 658/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1514 - root_mean_squared_error: 1.0981 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0718
Epoch 659/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1543 - root_mean_squared_error: 1.0652 - val_loss: 1.1849 - val_root_mean_squared_error: 1.0415
Epoch 660/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1443 - root_mean_squared_error: 1.0543 - val_loss: 1.1689 - val_root_mean_squared_error: 1.0995
Epoch 661/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1886 - root_mean_squared_error: 1.0961 - val_loss: 1.1691 - val_root_mean_squared_error: 1.0925
Epoch 662/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1629 - root_mean_squared_error: 1.0944 - val_loss: 1.1450 - val_root_mean_squared_error: 1.0895
Epoch 663/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1503 - root_mean_squared_error: 1.0478 - val_loss: 1.1584 - val_root_mean_squared_error: 1.1030
Epoch 664/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1415 - root_mean_squared_error: 1.0891 - val_loss: 1.1626 - val_root_mean_squared_error: 1.1048
Epoch 665/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1745 - root_mean_squared_error: 1.1052 - val_loss: 1.1543 - val_root_mean_squared_error: 1.0885
Epoch 666/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1521 - root_mean_squared_error: 1.0668 - val_loss: 1.1521 - val_root_mean_squared_error: 1.1214
Epoch 667/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1458 - root_mean_squared_error: 1.0732 - val_loss: 1.1586 - val_root_mean_squared_error: 1.0686
Epoch 668/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1507 - root_mean_squared_error: 1.0674 - val_loss: 1.1587 - val_root_mean_squared_error: 1.1141
Epoch 669/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1532 - root_mean_squared_error: 1.0605 - val_loss: 1.1609 - val_root_mean_squared_error: 1.1102
Epoch 670/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1738 - root_mean_squared_error: 1.0675 - val_loss: 1.1724 - val_root_mean_squared_error: 1.1287
Epoch 671/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1863 - root_mean_squared_error: 1.0762 - val_loss: 1.1450 - val_root_mean_squared_error: 1.0558
Epoch 672/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1554 - root_mean_squared_error: 1.0563 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0668
Epoch 673/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1551 - root_mean_squared_error: 1.0720 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0550
Epoch 674/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1490 - root_mean_squared_error: 1.0628 - val_loss: 1.1478 - val_root_mean_squared_error: 1.0705
Epoch 675/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1555 - root_mean_squared_error: 1.0833 - val_loss: 1.1668 - val_root_mean_squared_error: 1.0639
Epoch 676/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1543 - root_mean_squared_error: 1.0922 - val_loss: 1.1510 - val_root_mean_squared_error: 1.0403
Epoch 677/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1629 - root_mean_squared_error: 1.0836 - val_loss: 1.1479 - val_root_mean_squared_error: 1.0431
Epoch 678/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1750 - root_mean_squared_error: 1.0939 - val_loss: 1.1618 - val_root_mean_squared_error: 1.1008
Epoch 679/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1607 - root_mean_squared_error: 1.0687 - val_loss: 1.1689 - val_root_mean_squared_error: 1.0452
Epoch 680/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1546 - root_mean_squared_error: 1.0659 - val_loss: 1.1629 - val_root_mean_squared_error: 1.1008
Epoch 681/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1442 - root_mean_squared_error: 1.0386 - val_loss: 1.1594 - val_root_mean_squared_error: 1.0428
Epoch 682/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1529 - root_mean_squared_error: 1.0894 - val_loss: 1.1627 - val_root_mean_squared_error: 1.0732
Epoch 683/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1511 - root_mean_squared_error: 1.0838 - val_loss: 1.1569 - val_root_mean_squared_error: 1.1128
Epoch 684/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1633 - root_mean_squared_error: 1.0528 - val_loss: 1.1629 - val_root_mean_squared_error: 1.0316
Epoch 685/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1442 - root_mean_squared_error: 1.0689 - val_loss: 1.1585 - val_root_mean_squared_error: 1.1007
Epoch 686/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1633 - root_mean_squared_error: 1.0669 - val_loss: 1.1456 - val_root_mean_squared_error: 1.0061
Epoch 687/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1603 - root_mean_squared_error: 1.0467 - val_loss: 1.1545 - val_root_mean_squared_error: 1.0214
Epoch 688/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1690 - root_mean_squared_error: 1.0948 - val_loss: 1.1683 - val_root_mean_squared_error: 1.0078
Epoch 689/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1428 - root_mean_squared_error: 1.0675 - val_loss: 1.1831 - val_root_mean_squared_error: 1.1137
Epoch 690/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1467 - root_mean_squared_error: 1.0578 - val_loss: 1.1490 - val_root_mean_squared_error: 1.0577
Epoch 691/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1362 - root_mean_squared_error: 1.0549 - val_loss: 1.1438 - val_root_mean_squared_error: 1.0212
Epoch 692/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1720 - root_mean_squared_error: 1.1066 - val_loss: 1.1485 - val_root_mean_squared_error: 1.0708
Epoch 693/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1699 - root_mean_squared_error: 1.0773 - val_loss: 1.1478 - val_root_mean_squared_error: 1.0762
Epoch 694/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.2091 - root_mean_squared_error: 1.1026 - val_loss: 1.1525 - val_root_mean_squared_error: 1.0795
Epoch 695/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1549 - root_mean_squared_error: 1.0846 - val_loss: 1.1663 - val_root_mean_squared_error: 1.0997
Epoch 696/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1457 - root_mean_squared_error: 1.0779 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0783
Epoch 697/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1350 - root_mean_squared_error: 1.0610 - val_loss: 1.1469 - val_root_mean_squared_error: 1.0957
Epoch 698/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1685 - root_mean_squared_error: 1.0876 - val_loss: 1.1646 - val_root_mean_squared_error: 1.0735
Epoch 699/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1440 - root_mean_squared_error: 1.0470 - val_loss: 1.1749 - val_root_mean_squared_error: 1.0356
Epoch 700/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1405 - root_mean_squared_error: 1.0830 - val_loss: 1.1521 - val_root_mean_squared_error: 1.0629
Epoch 701/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1563 - root_mean_squared_error: 1.0993 - val_loss: 1.1540 - val_root_mean_squared_error: 1.0517
Epoch 702/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1538 - root_mean_squared_error: 1.0643 - val_loss: 1.1663 - val_root_mean_squared_error: 1.0638
Epoch 703/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1572 - root_mean_squared_error: 1.0848 - val_loss: 1.1394 - val_root_mean_squared_error: 1.0089
Epoch 704/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1618 - root_mean_squared_error: 1.0775 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0696
Epoch 705/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1551 - root_mean_squared_error: 1.0727 - val_loss: 1.1634 - val_root_mean_squared_error: 1.0767
Epoch 706/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1679 - root_mean_squared_error: 1.1097 - val_loss: 1.1656 - val_root_mean_squared_error: 1.1059
Epoch 707/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1644 - root_mean_squared_error: 1.0816 - val_loss: 1.1603 - val_root_mean_squared_error: 1.0403
Epoch 708/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1707 - root_mean_squared_error: 1.0744 - val_loss: 1.1603 - val_root_mean_squared_error: 1.0708
Epoch 709/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1368 - root_mean_squared_error: 1.0794 - val_loss: 1.1489 - val_root_mean_squared_error: 1.0863
Epoch 710/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1491 - root_mean_squared_error: 1.0742 - val_loss: 1.1504 - val_root_mean_squared_error: 1.0845
Epoch 711/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1730 - root_mean_squared_error: 1.1115 - val_loss: 1.1506 - val_root_mean_squared_error: 1.0635
Epoch 712/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1564 - root_mean_squared_error: 1.0526 - val_loss: 1.1473 - val_root_mean_squared_error: 1.1279
Epoch 713/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1470 - root_mean_squared_error: 1.0484 - val_loss: 1.1486 - val_root_mean_squared_error: 1.0438
Epoch 714/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1521 - root_mean_squared_error: 1.0575 - val_loss: 1.1555 - val_root_mean_squared_error: 1.0361
Epoch 715/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1485 - root_mean_squared_error: 1.0840 - val_loss: 1.1419 - val_root_mean_squared_error: 1.0830
Epoch 716/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1479 - root_mean_squared_error: 1.0833 - val_loss: 1.1843 - val_root_mean_squared_error: 1.1126
Epoch 717/1000
17/17 [==============================] - 0s 6ms/step - loss: 1.1530 - root_mean_squared_error: 1.0871 - val_loss: 1.1496 - val_root_mean_squared_error: 1.0912
Epoch 718/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1545 - root_mean_squared_error: 1.0661 - val_loss: 1.1531 - val_root_mean_squared_error: 1.0655
Epoch 719/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1571 - root_mean_squared_error: 1.0682 - val_loss: 1.1502 - val_root_mean_squared_error: 1.0971
Epoch 720/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1459 - root_mean_squared_error: 1.0567 - val_loss: 1.1673 - val_root_mean_squared_error: 1.0819
Epoch 721/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1666 - root_mean_squared_error: 1.1062 - val_loss: 1.1545 - val_root_mean_squared_error: 1.0482
Epoch 722/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1441 - root_mean_squared_error: 1.0680 - val_loss: 1.1474 - val_root_mean_squared_error: 1.0845
Epoch 723/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1270 - root_mean_squared_error: 1.0602 - val_loss: 1.1632 - val_root_mean_squared_error: 1.0631
Epoch 724/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1544 - root_mean_squared_error: 1.0645 - val_loss: 1.1588 - val_root_mean_squared_error: 1.0693
Epoch 725/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1718 - root_mean_squared_error: 1.0894 - val_loss: 1.1522 - val_root_mean_squared_error: 1.0830
Epoch 726/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1478 - root_mean_squared_error: 1.0540 - val_loss: 1.1446 - val_root_mean_squared_error: 1.0510
Epoch 727/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1536 - root_mean_squared_error: 1.0327 - val_loss: 1.1694 - val_root_mean_squared_error: 1.1186
Epoch 728/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1727 - root_mean_squared_error: 1.1152 - val_loss: 1.1539 - val_root_mean_squared_error: 1.0498
Epoch 729/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1407 - root_mean_squared_error: 1.0682 - val_loss: 1.1516 - val_root_mean_squared_error: 1.1231
Epoch 730/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1684 - root_mean_squared_error: 1.0797 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0615
Epoch 731/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1471 - root_mean_squared_error: 1.0812 - val_loss: 1.1633 - val_root_mean_squared_error: 1.0972
Epoch 732/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1573 - root_mean_squared_error: 1.0576 - val_loss: 1.1635 - val_root_mean_squared_error: 1.0936
Epoch 733/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1601 - root_mean_squared_error: 1.0804 - val_loss: 1.1637 - val_root_mean_squared_error: 1.0801
Epoch 734/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1616 - root_mean_squared_error: 1.0498 - val_loss: 1.1403 - val_root_mean_squared_error: 1.0510
Epoch 735/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1683 - root_mean_squared_error: 1.0631 - val_loss: 1.1590 - val_root_mean_squared_error: 1.0535
Epoch 736/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1577 - root_mean_squared_error: 1.0652 - val_loss: 1.1532 - val_root_mean_squared_error: 1.0657
Epoch 737/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1513 - root_mean_squared_error: 1.0990 - val_loss: 1.1671 - val_root_mean_squared_error: 1.0549
Epoch 738/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1454 - root_mean_squared_error: 1.0821 - val_loss: 1.1537 - val_root_mean_squared_error: 1.0603
Epoch 739/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1415 - root_mean_squared_error: 1.0411 - val_loss: 1.1513 - val_root_mean_squared_error: 1.0507
Epoch 740/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1412 - root_mean_squared_error: 1.0708 - val_loss: 1.1554 - val_root_mean_squared_error: 1.0688
Epoch 741/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1419 - root_mean_squared_error: 1.0633 - val_loss: 1.1572 - val_root_mean_squared_error: 1.0151
Epoch 742/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1676 - root_mean_squared_error: 1.0641 - val_loss: 1.1632 - val_root_mean_squared_error: 1.0863
Epoch 743/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1621 - root_mean_squared_error: 1.0823 - val_loss: 1.1640 - val_root_mean_squared_error: 1.0994
Epoch 744/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1430 - root_mean_squared_error: 1.0871 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0432
Epoch 745/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1559 - root_mean_squared_error: 1.0851 - val_loss: 1.1503 - val_root_mean_squared_error: 1.0797
Epoch 746/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1563 - root_mean_squared_error: 1.0898 - val_loss: 1.1566 - val_root_mean_squared_error: 1.1207
Epoch 747/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1586 - root_mean_squared_error: 1.0623 - val_loss: 1.1712 - val_root_mean_squared_error: 1.1315
Epoch 748/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1763 - root_mean_squared_error: 1.1001 - val_loss: 1.1716 - val_root_mean_squared_error: 1.0874
Epoch 749/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1588 - root_mean_squared_error: 1.0938 - val_loss: 1.1594 - val_root_mean_squared_error: 1.0651
Epoch 750/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1476 - root_mean_squared_error: 1.0663 - val_loss: 1.1583 - val_root_mean_squared_error: 1.0723
Epoch 751/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1826 - root_mean_squared_error: 1.1032 - val_loss: 1.1549 - val_root_mean_squared_error: 1.0862
Epoch 752/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1622 - root_mean_squared_error: 1.0865 - val_loss: 1.1421 - val_root_mean_squared_error: 1.0306
Epoch 753/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1506 - root_mean_squared_error: 1.0471 - val_loss: 1.1665 - val_root_mean_squared_error: 1.1012
Epoch 754/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1478 - root_mean_squared_error: 1.0584 - val_loss: 1.1639 - val_root_mean_squared_error: 1.1013
Epoch 755/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1466 - root_mean_squared_error: 1.0494 - val_loss: 1.1509 - val_root_mean_squared_error: 1.1267
Epoch 756/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1428 - root_mean_squared_error: 1.0803 - val_loss: 1.1598 - val_root_mean_squared_error: 1.0467
Epoch 757/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1673 - root_mean_squared_error: 1.0739 - val_loss: 1.1514 - val_root_mean_squared_error: 1.0570
Epoch 758/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1623 - root_mean_squared_error: 1.0670 - val_loss: 1.1395 - val_root_mean_squared_error: 1.0169
Epoch 759/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1484 - root_mean_squared_error: 1.0713 - val_loss: 1.1507 - val_root_mean_squared_error: 1.0982
Epoch 760/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1602 - root_mean_squared_error: 1.0773 - val_loss: 1.1494 - val_root_mean_squared_error: 1.0708
Epoch 761/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1393 - root_mean_squared_error: 1.0606 - val_loss: 1.1462 - val_root_mean_squared_error: 1.0339
Epoch 762/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1497 - root_mean_squared_error: 1.0756 - val_loss: 1.1420 - val_root_mean_squared_error: 1.0770
Epoch 763/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1625 - root_mean_squared_error: 1.0728 - val_loss: 1.1566 - val_root_mean_squared_error: 1.0690
Epoch 764/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.0844 - val_loss: 1.1530 - val_root_mean_squared_error: 1.1062
Epoch 765/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1462 - root_mean_squared_error: 1.0485 - val_loss: 1.1744 - val_root_mean_squared_error: 1.0777
Epoch 766/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1562 - root_mean_squared_error: 1.0728 - val_loss: 1.1593 - val_root_mean_squared_error: 1.0451
Epoch 767/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1369 - root_mean_squared_error: 1.0630 - val_loss: 1.1590 - val_root_mean_squared_error: 1.0406
Epoch 768/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1505 - root_mean_squared_error: 1.0538 - val_loss: 1.1546 - val_root_mean_squared_error: 1.0828
Epoch 769/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1389 - root_mean_squared_error: 1.0640 - val_loss: 1.1577 - val_root_mean_squared_error: 1.0981
Epoch 770/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1578 - root_mean_squared_error: 1.0662 - val_loss: 1.1460 - val_root_mean_squared_error: 1.0626
Epoch 771/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1675 - root_mean_squared_error: 1.0817 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0406
Epoch 772/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1543 - root_mean_squared_error: 1.0680 - val_loss: 1.1490 - val_root_mean_squared_error: 1.0890
Epoch 773/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1682 - root_mean_squared_error: 1.0845 - val_loss: 1.1540 - val_root_mean_squared_error: 1.1046
Epoch 774/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1585 - root_mean_squared_error: 1.1008 - val_loss: 1.1818 - val_root_mean_squared_error: 1.0455
Epoch 775/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1665 - root_mean_squared_error: 1.0429 - val_loss: 1.1548 - val_root_mean_squared_error: 1.0597
Epoch 776/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1545 - root_mean_squared_error: 1.0824 - val_loss: 1.1626 - val_root_mean_squared_error: 1.0953
Epoch 777/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1569 - root_mean_squared_error: 1.0806 - val_loss: 1.1640 - val_root_mean_squared_error: 1.1173
Epoch 778/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1731 - root_mean_squared_error: 1.0797 - val_loss: 1.1588 - val_root_mean_squared_error: 1.0463
Epoch 779/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1406 - root_mean_squared_error: 1.0597 - val_loss: 1.1451 - val_root_mean_squared_error: 1.0885
Epoch 780/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1568 - root_mean_squared_error: 1.0902 - val_loss: 1.1542 - val_root_mean_squared_error: 1.0787
Epoch 781/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1504 - root_mean_squared_error: 1.0655 - val_loss: 1.1610 - val_root_mean_squared_error: 1.0604
Epoch 782/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1611 - root_mean_squared_error: 1.0629 - val_loss: 1.1616 - val_root_mean_squared_error: 1.0841
Epoch 783/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1373 - root_mean_squared_error: 1.0804 - val_loss: 1.1639 - val_root_mean_squared_error: 1.0744
Epoch 784/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1411 - root_mean_squared_error: 1.0535 - val_loss: 1.1570 - val_root_mean_squared_error: 1.1242
Epoch 785/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1406 - root_mean_squared_error: 1.0582 - val_loss: 1.1429 - val_root_mean_squared_error: 1.0309
Epoch 786/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1417 - root_mean_squared_error: 1.0604 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0469
Epoch 787/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1556 - root_mean_squared_error: 1.0926 - val_loss: 1.1691 - val_root_mean_squared_error: 1.0894
Epoch 788/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1380 - root_mean_squared_error: 1.0733 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0746
Epoch 789/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1560 - root_mean_squared_error: 1.0723 - val_loss: 1.1524 - val_root_mean_squared_error: 1.0254
Epoch 790/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1439 - root_mean_squared_error: 1.0620 - val_loss: 1.1629 - val_root_mean_squared_error: 1.1046
Epoch 791/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1459 - root_mean_squared_error: 1.0391 - val_loss: 1.1570 - val_root_mean_squared_error: 1.0022
Epoch 792/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1534 - root_mean_squared_error: 1.0534 - val_loss: 1.1583 - val_root_mean_squared_error: 1.0697
Epoch 793/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.0959 - val_loss: 1.1667 - val_root_mean_squared_error: 1.0600
Epoch 794/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1492 - root_mean_squared_error: 1.0498 - val_loss: 1.1627 - val_root_mean_squared_error: 1.0702
Epoch 795/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1702 - root_mean_squared_error: 1.0689 - val_loss: 1.1596 - val_root_mean_squared_error: 1.0549
Epoch 796/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1511 - root_mean_squared_error: 1.0844 - val_loss: 1.1428 - val_root_mean_squared_error: 1.0361
Epoch 797/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1581 - root_mean_squared_error: 1.0534 - val_loss: 1.1632 - val_root_mean_squared_error: 1.0987
Epoch 798/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1473 - root_mean_squared_error: 1.0691 - val_loss: 1.1565 - val_root_mean_squared_error: 1.0829
Epoch 799/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1617 - root_mean_squared_error: 1.0816 - val_loss: 1.1563 - val_root_mean_squared_error: 1.0533
Epoch 800/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1626 - root_mean_squared_error: 1.0663 - val_loss: 1.1628 - val_root_mean_squared_error: 1.0900
Epoch 801/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1472 - root_mean_squared_error: 1.0564 - val_loss: 1.1404 - val_root_mean_squared_error: 1.0740
Epoch 802/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1490 - root_mean_squared_error: 1.0669 - val_loss: 1.1539 - val_root_mean_squared_error: 1.1528
Epoch 803/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1694 - root_mean_squared_error: 1.0843 - val_loss: 1.1679 - val_root_mean_squared_error: 1.0708
Epoch 804/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1391 - root_mean_squared_error: 1.0404 - val_loss: 1.1462 - val_root_mean_squared_error: 1.0396
Epoch 805/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1581 - root_mean_squared_error: 1.0733 - val_loss: 1.1623 - val_root_mean_squared_error: 1.0897
Epoch 806/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1510 - root_mean_squared_error: 1.0777 - val_loss: 1.1449 - val_root_mean_squared_error: 1.0609
Epoch 807/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1547 - root_mean_squared_error: 1.0753 - val_loss: 1.1452 - val_root_mean_squared_error: 1.0623
Epoch 808/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1256 - root_mean_squared_error: 1.0463 - val_loss: 1.1742 - val_root_mean_squared_error: 1.0418
Epoch 809/1000
17/17 [==============================] - 0s 17ms/step - loss: 1.1597 - root_mean_squared_error: 1.0625 - val_loss: 1.1456 - val_root_mean_squared_error: 1.0860
Epoch 810/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1271 - root_mean_squared_error: 1.0539 - val_loss: 1.1641 - val_root_mean_squared_error: 1.0593
Epoch 811/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1444 - root_mean_squared_error: 1.0594 - val_loss: 1.1494 - val_root_mean_squared_error: 1.0612
Epoch 812/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1441 - root_mean_squared_error: 1.0523 - val_loss: 1.1500 - val_root_mean_squared_error: 1.0866
Epoch 813/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1684 - root_mean_squared_error: 1.0968 - val_loss: 1.1585 - val_root_mean_squared_error: 1.0747
Epoch 814/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1320 - root_mean_squared_error: 1.0617 - val_loss: 1.1505 - val_root_mean_squared_error: 1.0829
Epoch 815/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1586 - root_mean_squared_error: 1.0611 - val_loss: 1.1539 - val_root_mean_squared_error: 1.0924
Epoch 816/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1479 - root_mean_squared_error: 1.0599 - val_loss: 1.1521 - val_root_mean_squared_error: 1.0800
Epoch 817/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.0900 - val_loss: 1.1491 - val_root_mean_squared_error: 1.0925
Epoch 818/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1601 - root_mean_squared_error: 1.0907 - val_loss: 1.1510 - val_root_mean_squared_error: 1.1083
Epoch 819/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1582 - root_mean_squared_error: 1.0686 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0929
Epoch 820/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1384 - root_mean_squared_error: 1.0629 - val_loss: 1.1456 - val_root_mean_squared_error: 1.0562
Epoch 821/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1404 - root_mean_squared_error: 1.0518 - val_loss: 1.1521 - val_root_mean_squared_error: 1.0569
Epoch 822/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1535 - root_mean_squared_error: 1.0583 - val_loss: 1.1480 - val_root_mean_squared_error: 1.0805
Epoch 823/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1496 - root_mean_squared_error: 1.0775 - val_loss: 1.1658 - val_root_mean_squared_error: 1.0212
Epoch 824/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1431 - root_mean_squared_error: 1.0520 - val_loss: 1.1748 - val_root_mean_squared_error: 1.0632
Epoch 825/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1415 - root_mean_squared_error: 1.0433 - val_loss: 1.1462 - val_root_mean_squared_error: 1.0235
Epoch 826/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1432 - root_mean_squared_error: 1.0420 - val_loss: 1.1637 - val_root_mean_squared_error: 1.0512
Epoch 827/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1384 - root_mean_squared_error: 1.0702 - val_loss: 1.1489 - val_root_mean_squared_error: 1.0714
Epoch 828/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1680 - root_mean_squared_error: 1.0659 - val_loss: 1.1553 - val_root_mean_squared_error: 1.1109
Epoch 829/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1619 - root_mean_squared_error: 1.0726 - val_loss: 1.1695 - val_root_mean_squared_error: 1.0531
Epoch 830/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1527 - root_mean_squared_error: 1.0782 - val_loss: 1.1470 - val_root_mean_squared_error: 1.0731
Epoch 831/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1533 - root_mean_squared_error: 1.0876 - val_loss: 1.1479 - val_root_mean_squared_error: 1.0625
Epoch 832/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1767 - root_mean_squared_error: 1.0949 - val_loss: 1.1436 - val_root_mean_squared_error: 1.0784
Epoch 833/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1510 - root_mean_squared_error: 1.0814 - val_loss: 1.1713 - val_root_mean_squared_error: 1.0917
Epoch 834/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1291 - root_mean_squared_error: 1.0510 - val_loss: 1.1708 - val_root_mean_squared_error: 1.0825
Epoch 835/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1348 - root_mean_squared_error: 1.0532 - val_loss: 1.1519 - val_root_mean_squared_error: 1.0547
Epoch 836/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1623 - root_mean_squared_error: 1.0781 - val_loss: 1.1481 - val_root_mean_squared_error: 1.0617
Epoch 837/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1598 - root_mean_squared_error: 1.0756 - val_loss: 1.1614 - val_root_mean_squared_error: 1.0597
Epoch 838/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1497 - root_mean_squared_error: 1.0567 - val_loss: 1.1511 - val_root_mean_squared_error: 1.0996
Epoch 839/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1357 - root_mean_squared_error: 1.0512 - val_loss: 1.1692 - val_root_mean_squared_error: 1.1078
Epoch 840/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1466 - root_mean_squared_error: 1.0768 - val_loss: 1.1717 - val_root_mean_squared_error: 1.0462
Epoch 841/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1583 - root_mean_squared_error: 1.0877 - val_loss: 1.1449 - val_root_mean_squared_error: 1.0309
Epoch 842/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1649 - root_mean_squared_error: 1.0739 - val_loss: 1.1548 - val_root_mean_squared_error: 1.0742
Epoch 843/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1538 - root_mean_squared_error: 1.0706 - val_loss: 1.1611 - val_root_mean_squared_error: 1.0650
Epoch 844/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1630 - root_mean_squared_error: 1.0696 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0589
Epoch 845/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1394 - root_mean_squared_error: 1.0644 - val_loss: 1.1764 - val_root_mean_squared_error: 1.0475
Epoch 846/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1556 - root_mean_squared_error: 1.0762 - val_loss: 1.1624 - val_root_mean_squared_error: 1.1453
Epoch 847/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1825 - root_mean_squared_error: 1.1081 - val_loss: 1.1427 - val_root_mean_squared_error: 1.0270
Epoch 848/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1569 - root_mean_squared_error: 1.0687 - val_loss: 1.1578 - val_root_mean_squared_error: 1.0624
Epoch 849/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1673 - root_mean_squared_error: 1.0876 - val_loss: 1.1528 - val_root_mean_squared_error: 1.0926
Epoch 850/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1574 - root_mean_squared_error: 1.0672 - val_loss: 1.1538 - val_root_mean_squared_error: 1.0370
Epoch 851/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1529 - root_mean_squared_error: 1.0600 - val_loss: 1.1527 - val_root_mean_squared_error: 1.0266
Epoch 852/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1524 - root_mean_squared_error: 1.0748 - val_loss: 1.1450 - val_root_mean_squared_error: 1.0611
Epoch 853/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1474 - root_mean_squared_error: 1.0662 - val_loss: 1.1537 - val_root_mean_squared_error: 1.0211
Epoch 854/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1525 - root_mean_squared_error: 1.1039 - val_loss: 1.1526 - val_root_mean_squared_error: 1.0293
Epoch 855/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1474 - root_mean_squared_error: 1.0637 - val_loss: 1.1449 - val_root_mean_squared_error: 1.0489
Epoch 856/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1684 - root_mean_squared_error: 1.0762 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0315
Epoch 857/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1581 - root_mean_squared_error: 1.0926 - val_loss: 1.1562 - val_root_mean_squared_error: 1.0440
Epoch 858/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1476 - root_mean_squared_error: 1.0654 - val_loss: 1.1449 - val_root_mean_squared_error: 1.0614
Epoch 859/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1250 - root_mean_squared_error: 1.0461 - val_loss: 1.1508 - val_root_mean_squared_error: 1.0934
Epoch 860/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1728 - root_mean_squared_error: 1.0940 - val_loss: 1.1556 - val_root_mean_squared_error: 1.0938
Epoch 861/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1659 - root_mean_squared_error: 1.0736 - val_loss: 1.1571 - val_root_mean_squared_error: 1.0659
Epoch 862/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1498 - root_mean_squared_error: 1.0440 - val_loss: 1.1656 - val_root_mean_squared_error: 1.0696
Epoch 863/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1739 - root_mean_squared_error: 1.0882 - val_loss: 1.1656 - val_root_mean_squared_error: 1.0842
Epoch 864/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1665 - root_mean_squared_error: 1.0668 - val_loss: 1.1618 - val_root_mean_squared_error: 1.0827
Epoch 865/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1457 - root_mean_squared_error: 1.0611 - val_loss: 1.1530 - val_root_mean_squared_error: 1.0570
Epoch 866/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1401 - root_mean_squared_error: 1.0589 - val_loss: 1.1501 - val_root_mean_squared_error: 1.0803
Epoch 867/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1656 - root_mean_squared_error: 1.0834 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0201
Epoch 868/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1639 - root_mean_squared_error: 1.0851 - val_loss: 1.1546 - val_root_mean_squared_error: 1.0209
Epoch 869/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1474 - root_mean_squared_error: 1.0606 - val_loss: 1.1611 - val_root_mean_squared_error: 1.0764
Epoch 870/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1412 - root_mean_squared_error: 1.0447 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0748
Epoch 871/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1593 - root_mean_squared_error: 1.0926 - val_loss: 1.1496 - val_root_mean_squared_error: 1.0700
Epoch 872/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1466 - root_mean_squared_error: 1.0660 - val_loss: 1.1619 - val_root_mean_squared_error: 1.0771
Epoch 873/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1576 - root_mean_squared_error: 1.0799 - val_loss: 1.1516 - val_root_mean_squared_error: 1.0901
Epoch 874/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1439 - root_mean_squared_error: 1.0684 - val_loss: 1.1563 - val_root_mean_squared_error: 1.0396
Epoch 875/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1508 - root_mean_squared_error: 1.0741 - val_loss: 1.1553 - val_root_mean_squared_error: 1.1042
Epoch 876/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1518 - root_mean_squared_error: 1.0613 - val_loss: 1.1426 - val_root_mean_squared_error: 1.0721
Epoch 877/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1336 - root_mean_squared_error: 1.0503 - val_loss: 1.1504 - val_root_mean_squared_error: 1.0382
Epoch 878/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1583 - root_mean_squared_error: 1.0973 - val_loss: 1.1543 - val_root_mean_squared_error: 1.0306
Epoch 879/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1647 - root_mean_squared_error: 1.0950 - val_loss: 1.1667 - val_root_mean_squared_error: 1.0930
Epoch 880/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1547 - root_mean_squared_error: 1.0798 - val_loss: 1.1453 - val_root_mean_squared_error: 1.0625
Epoch 881/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1677 - root_mean_squared_error: 1.0907 - val_loss: 1.1484 - val_root_mean_squared_error: 1.0751
Epoch 882/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1437 - root_mean_squared_error: 1.0622 - val_loss: 1.1608 - val_root_mean_squared_error: 1.0740
Epoch 883/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1523 - root_mean_squared_error: 1.0512 - val_loss: 1.1487 - val_root_mean_squared_error: 1.0874
Epoch 884/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1463 - root_mean_squared_error: 1.0695 - val_loss: 1.1547 - val_root_mean_squared_error: 1.0890
Epoch 885/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1556 - root_mean_squared_error: 1.0779 - val_loss: 1.1457 - val_root_mean_squared_error: 1.0758
Epoch 886/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1698 - root_mean_squared_error: 1.0734 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0935
Epoch 887/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1528 - root_mean_squared_error: 1.0638 - val_loss: 1.1420 - val_root_mean_squared_error: 1.0871
Epoch 888/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1515 - root_mean_squared_error: 1.0722 - val_loss: 1.1490 - val_root_mean_squared_error: 1.0605
Epoch 889/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1396 - root_mean_squared_error: 1.0541 - val_loss: 1.1508 - val_root_mean_squared_error: 1.0830
Epoch 890/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1463 - root_mean_squared_error: 1.0846 - val_loss: 1.1551 - val_root_mean_squared_error: 1.0437
Epoch 891/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1522 - root_mean_squared_error: 1.0649 - val_loss: 1.1583 - val_root_mean_squared_error: 1.0398
Epoch 892/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1612 - root_mean_squared_error: 1.0731 - val_loss: 1.1512 - val_root_mean_squared_error: 1.0788
Epoch 893/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1584 - root_mean_squared_error: 1.0723 - val_loss: 1.1518 - val_root_mean_squared_error: 1.0712
Epoch 894/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1470 - root_mean_squared_error: 1.0396 - val_loss: 1.1600 - val_root_mean_squared_error: 1.1049
Epoch 895/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1557 - root_mean_squared_error: 1.0892 - val_loss: 1.1463 - val_root_mean_squared_error: 1.0755
Epoch 896/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1425 - root_mean_squared_error: 1.0530 - val_loss: 1.1621 - val_root_mean_squared_error: 1.1156
Epoch 897/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1675 - root_mean_squared_error: 1.0883 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0633
Epoch 898/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1400 - root_mean_squared_error: 1.0346 - val_loss: 1.1545 - val_root_mean_squared_error: 1.1070
Epoch 899/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1477 - root_mean_squared_error: 1.0613 - val_loss: 1.1463 - val_root_mean_squared_error: 1.0464
Epoch 900/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1413 - root_mean_squared_error: 1.0638 - val_loss: 1.1643 - val_root_mean_squared_error: 1.0594
Epoch 901/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1501 - root_mean_squared_error: 1.0465 - val_loss: 1.1681 - val_root_mean_squared_error: 1.0948
Epoch 902/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1499 - root_mean_squared_error: 1.0573 - val_loss: 1.1336 - val_root_mean_squared_error: 1.1003
Epoch 903/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1600 - root_mean_squared_error: 1.0729 - val_loss: 1.1541 - val_root_mean_squared_error: 1.0617
Epoch 904/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1518 - root_mean_squared_error: 1.0724 - val_loss: 1.1553 - val_root_mean_squared_error: 1.0674
Epoch 905/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1551 - root_mean_squared_error: 1.0629 - val_loss: 1.1483 - val_root_mean_squared_error: 1.0351
Epoch 906/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1723 - root_mean_squared_error: 1.0799 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0586
Epoch 907/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1610 - root_mean_squared_error: 1.0680 - val_loss: 1.1505 - val_root_mean_squared_error: 1.0786
Epoch 908/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1562 - root_mean_squared_error: 1.0686 - val_loss: 1.1698 - val_root_mean_squared_error: 1.0550
Epoch 909/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1564 - root_mean_squared_error: 1.0618 - val_loss: 1.1419 - val_root_mean_squared_error: 1.0663
Epoch 910/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1395 - root_mean_squared_error: 1.0524 - val_loss: 1.1374 - val_root_mean_squared_error: 1.0479
Epoch 911/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1419 - root_mean_squared_error: 1.0688 - val_loss: 1.1453 - val_root_mean_squared_error: 1.0429
Epoch 912/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1538 - root_mean_squared_error: 1.0688 - val_loss: 1.1581 - val_root_mean_squared_error: 1.0579
Epoch 913/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1500 - root_mean_squared_error: 1.0783 - val_loss: 1.1493 - val_root_mean_squared_error: 1.0617
Epoch 914/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1378 - root_mean_squared_error: 1.0583 - val_loss: 1.1465 - val_root_mean_squared_error: 1.1010
Epoch 915/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1315 - root_mean_squared_error: 1.0718 - val_loss: 1.1461 - val_root_mean_squared_error: 1.0390
Epoch 916/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1376 - root_mean_squared_error: 1.0738 - val_loss: 1.1545 - val_root_mean_squared_error: 1.0918
Epoch 917/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1464 - root_mean_squared_error: 1.0508 - val_loss: 1.1473 - val_root_mean_squared_error: 1.0590
Epoch 918/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1372 - root_mean_squared_error: 1.0647 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0784
Epoch 919/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1607 - root_mean_squared_error: 1.0749 - val_loss: 1.1629 - val_root_mean_squared_error: 1.0932
Epoch 920/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1569 - root_mean_squared_error: 1.0515 - val_loss: 1.1480 - val_root_mean_squared_error: 1.0496
Epoch 921/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1293 - root_mean_squared_error: 1.0700 - val_loss: 1.1532 - val_root_mean_squared_error: 1.0711
Epoch 922/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1524 - root_mean_squared_error: 1.0503 - val_loss: 1.1517 - val_root_mean_squared_error: 1.1328
Epoch 923/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1588 - root_mean_squared_error: 1.0524 - val_loss: 1.1488 - val_root_mean_squared_error: 1.0764
Epoch 924/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1326 - root_mean_squared_error: 1.0645 - val_loss: 1.1641 - val_root_mean_squared_error: 1.0658
Epoch 925/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1450 - root_mean_squared_error: 1.0624 - val_loss: 1.1651 - val_root_mean_squared_error: 1.0312
Epoch 926/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1410 - root_mean_squared_error: 1.0780 - val_loss: 1.1480 - val_root_mean_squared_error: 1.0778
Epoch 927/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1305 - root_mean_squared_error: 1.0355 - val_loss: 1.1465 - val_root_mean_squared_error: 1.1035
Epoch 928/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1543 - root_mean_squared_error: 1.0826 - val_loss: 1.1510 - val_root_mean_squared_error: 1.1022
Epoch 929/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1305 - root_mean_squared_error: 1.0489 - val_loss: 1.1705 - val_root_mean_squared_error: 1.0796
Epoch 930/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1464 - root_mean_squared_error: 1.0499 - val_loss: 1.1443 - val_root_mean_squared_error: 1.0287
Epoch 931/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1408 - root_mean_squared_error: 1.0617 - val_loss: 1.1554 - val_root_mean_squared_error: 1.1039
Epoch 932/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1256 - root_mean_squared_error: 1.0488 - val_loss: 1.1543 - val_root_mean_squared_error: 1.0307
Epoch 933/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1459 - root_mean_squared_error: 1.0749 - val_loss: 1.1674 - val_root_mean_squared_error: 1.0748
Epoch 934/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1506 - root_mean_squared_error: 1.0721 - val_loss: 1.1567 - val_root_mean_squared_error: 1.0792
Epoch 935/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1381 - root_mean_squared_error: 1.0647 - val_loss: 1.1451 - val_root_mean_squared_error: 1.0948
Epoch 936/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1377 - root_mean_squared_error: 1.0781 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0616
Epoch 937/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1457 - root_mean_squared_error: 1.0686 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0492
Epoch 938/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1348 - root_mean_squared_error: 1.0894 - val_loss: 1.1616 - val_root_mean_squared_error: 1.0816
Epoch 939/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1548 - root_mean_squared_error: 1.0938 - val_loss: 1.1595 - val_root_mean_squared_error: 1.1148
Epoch 940/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1555 - root_mean_squared_error: 1.0653 - val_loss: 1.1628 - val_root_mean_squared_error: 1.0849
Epoch 941/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1712 - root_mean_squared_error: 1.0599 - val_loss: 1.1555 - val_root_mean_squared_error: 1.0740
Epoch 942/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1592 - root_mean_squared_error: 1.0713 - val_loss: 1.1539 - val_root_mean_squared_error: 1.0653
Epoch 943/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1600 - root_mean_squared_error: 1.0810 - val_loss: 1.1521 - val_root_mean_squared_error: 1.0934
Epoch 944/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1385 - root_mean_squared_error: 1.0309 - val_loss: 1.1528 - val_root_mean_squared_error: 1.0593
Epoch 945/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1330 - root_mean_squared_error: 1.0415 - val_loss: 1.1565 - val_root_mean_squared_error: 1.1075
Epoch 946/1000
17/17 [==============================] - 0s 16ms/step - loss: 1.1427 - root_mean_squared_error: 1.0696 - val_loss: 1.1490 - val_root_mean_squared_error: 1.0491
Epoch 947/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1416 - root_mean_squared_error: 1.0759 - val_loss: 1.1485 - val_root_mean_squared_error: 1.0408
Epoch 948/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1396 - root_mean_squared_error: 1.0446 - val_loss: 1.1617 - val_root_mean_squared_error: 1.0695
Epoch 949/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.1080 - val_loss: 1.1451 - val_root_mean_squared_error: 1.0676
Epoch 950/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1441 - root_mean_squared_error: 1.0791 - val_loss: 1.1597 - val_root_mean_squared_error: 1.1097
Epoch 951/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1441 - root_mean_squared_error: 1.0562 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0873
Epoch 952/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1550 - root_mean_squared_error: 1.1026 - val_loss: 1.1547 - val_root_mean_squared_error: 1.0873
Epoch 953/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1458 - root_mean_squared_error: 1.0623 - val_loss: 1.1422 - val_root_mean_squared_error: 1.1142
Epoch 954/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1298 - root_mean_squared_error: 1.0635 - val_loss: 1.1656 - val_root_mean_squared_error: 1.1026
Epoch 955/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1594 - root_mean_squared_error: 1.0945 - val_loss: 1.1479 - val_root_mean_squared_error: 1.0901
Epoch 956/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1490 - root_mean_squared_error: 1.0501 - val_loss: 1.1636 - val_root_mean_squared_error: 1.0577
Epoch 957/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1607 - root_mean_squared_error: 1.0737 - val_loss: 1.1480 - val_root_mean_squared_error: 1.0629
Epoch 958/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1769 - root_mean_squared_error: 1.0720 - val_loss: 1.1397 - val_root_mean_squared_error: 1.0253
Epoch 959/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1587 - root_mean_squared_error: 1.0748 - val_loss: 1.1536 - val_root_mean_squared_error: 1.1078
Epoch 960/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1669 - root_mean_squared_error: 1.0358 - val_loss: 1.1445 - val_root_mean_squared_error: 1.0778
Epoch 961/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1409 - root_mean_squared_error: 1.0642 - val_loss: 1.1431 - val_root_mean_squared_error: 1.0546
Epoch 962/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1356 - root_mean_squared_error: 1.0588 - val_loss: 1.1637 - val_root_mean_squared_error: 1.1019
Epoch 963/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1359 - root_mean_squared_error: 1.0672 - val_loss: 1.1462 - val_root_mean_squared_error: 1.0070
Epoch 964/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1581 - root_mean_squared_error: 1.1096 - val_loss: 1.1488 - val_root_mean_squared_error: 1.1210
Epoch 965/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1484 - root_mean_squared_error: 1.0616 - val_loss: 1.1458 - val_root_mean_squared_error: 1.0751
Epoch 966/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1392 - root_mean_squared_error: 1.0789 - val_loss: 1.1564 - val_root_mean_squared_error: 1.0439
Epoch 967/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1548 - root_mean_squared_error: 1.0772 - val_loss: 1.1512 - val_root_mean_squared_error: 1.0639
Epoch 968/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1411 - root_mean_squared_error: 1.0760 - val_loss: 1.1533 - val_root_mean_squared_error: 1.0760
Epoch 969/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1468 - root_mean_squared_error: 1.0465 - val_loss: 1.1621 - val_root_mean_squared_error: 1.1124
Epoch 970/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1445 - root_mean_squared_error: 1.0420 - val_loss: 1.1514 - val_root_mean_squared_error: 1.0734
Epoch 971/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1461 - root_mean_squared_error: 1.0648 - val_loss: 1.1454 - val_root_mean_squared_error: 1.0599
Epoch 972/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1673 - root_mean_squared_error: 1.0783 - val_loss: 1.1561 - val_root_mean_squared_error: 1.0985
Epoch 973/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1517 - root_mean_squared_error: 1.0756 - val_loss: 1.1552 - val_root_mean_squared_error: 1.0859
Epoch 974/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1332 - root_mean_squared_error: 1.0417 - val_loss: 1.1444 - val_root_mean_squared_error: 1.0575
Epoch 975/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1500 - root_mean_squared_error: 1.0494 - val_loss: 1.1598 - val_root_mean_squared_error: 1.0814
Epoch 976/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1666 - root_mean_squared_error: 1.0560 - val_loss: 1.1473 - val_root_mean_squared_error: 1.0445
Epoch 977/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1618 - root_mean_squared_error: 1.0435 - val_loss: 1.1553 - val_root_mean_squared_error: 1.0342
Epoch 978/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1889 - root_mean_squared_error: 1.0884 - val_loss: 1.1455 - val_root_mean_squared_error: 1.0760
Epoch 979/1000
17/17 [==============================] - 0s 15ms/step - loss: 1.1358 - root_mean_squared_error: 1.0589 - val_loss: 1.1593 - val_root_mean_squared_error: 1.0761
Epoch 980/1000
17/17 [==============================] - 0s 8ms/step - loss: 1.1345 - root_mean_squared_error: 1.0498 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0980
Epoch 981/1000
17/17 [==============================] - 0s 9ms/step - loss: 1.1380 - root_mean_squared_error: 1.0522 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0847
Epoch 982/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1531 - root_mean_squared_error: 1.0702 - val_loss: 1.1529 - val_root_mean_squared_error: 1.0891
Epoch 983/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1278 - root_mean_squared_error: 1.0645 - val_loss: 1.1716 - val_root_mean_squared_error: 1.0977
Epoch 984/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1446 - root_mean_squared_error: 1.0777 - val_loss: 1.1639 - val_root_mean_squared_error: 1.0794
Epoch 985/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1485 - root_mean_squared_error: 1.0803 - val_loss: 1.1577 - val_root_mean_squared_error: 1.0671
Epoch 986/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1411 - root_mean_squared_error: 1.0583 - val_loss: 1.1530 - val_root_mean_squared_error: 1.0710
Epoch 987/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1501 - root_mean_squared_error: 1.0641 - val_loss: 1.1505 - val_root_mean_squared_error: 1.0194
Epoch 988/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1531 - root_mean_squared_error: 1.0557 - val_loss: 1.1436 - val_root_mean_squared_error: 1.0386
Epoch 989/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1470 - root_mean_squared_error: 1.0672 - val_loss: 1.1586 - val_root_mean_squared_error: 1.1176
Epoch 990/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1530 - root_mean_squared_error: 1.0759 - val_loss: 1.1617 - val_root_mean_squared_error: 1.0887
Epoch 991/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1323 - root_mean_squared_error: 1.0504 - val_loss: 1.1512 - val_root_mean_squared_error: 1.1049
Epoch 992/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1469 - root_mean_squared_error: 1.0519 - val_loss: 1.1478 - val_root_mean_squared_error: 1.0986
Epoch 993/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1242 - root_mean_squared_error: 1.0397 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0479
Epoch 994/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1640 - root_mean_squared_error: 1.0745 - val_loss: 1.1464 - val_root_mean_squared_error: 1.0402
Epoch 995/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1323 - root_mean_squared_error: 1.0431 - val_loss: 1.1553 - val_root_mean_squared_error: 1.1060
Epoch 996/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.0686 - val_loss: 1.1554 - val_root_mean_squared_error: 1.0370
Epoch 997/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1351 - root_mean_squared_error: 1.0628 - val_loss: 1.1472 - val_root_mean_squared_error: 1.0813
Epoch 998/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1324 - root_mean_squared_error: 1.0858 - val_loss: 1.1527 - val_root_mean_squared_error: 1.0578
Epoch 999/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1591 - root_mean_squared_error: 1.0801 - val_loss: 1.1483 - val_root_mean_squared_error: 1.0442
Epoch 1000/1000
17/17 [==============================] - 0s 7ms/step - loss: 1.1402 - root_mean_squared_error: 1.0554 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0389
Model training finished.
Train RMSE: 1.068
Evaluating model performance...
Test RMSE: 1.068

```
</div>
Now let's produce an output from the model given the test examples.
The output is now a distribution, and we can use its mean and variance
to compute the confidence intervals (CI) of the prediction.


```python
prediction_distribution = prob_bnn_model(examples)
prediction_mean = prediction_distribution.mean().numpy().tolist()
prediction_stdv = prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
prediction_stdv = prediction_stdv.tolist()

for idx in range(sample):
    print(
        f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
        f"stddev: {round(prediction_stdv[idx][0], 2)}, "
        f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
        f" - Actual: {targets[idx]}"
    )
```

<div class="k-default-codeblock">
```
Prediction mean: 5.29, stddev: 0.66, 95% CI: [6.58 - 4.0] - Actual: 6.0
Prediction mean: 6.49, stddev: 0.81, 95% CI: [8.08 - 4.89] - Actual: 6.0
Prediction mean: 5.85, stddev: 0.7, 95% CI: [7.22 - 4.48] - Actual: 7.0
Prediction mean: 5.59, stddev: 0.69, 95% CI: [6.95 - 4.24] - Actual: 5.0
Prediction mean: 6.37, stddev: 0.87, 95% CI: [8.07 - 4.67] - Actual: 5.0
Prediction mean: 6.34, stddev: 0.78, 95% CI: [7.87 - 4.81] - Actual: 7.0
Prediction mean: 5.14, stddev: 0.65, 95% CI: [6.4 - 3.87] - Actual: 5.0
Prediction mean: 6.49, stddev: 0.81, 95% CI: [8.09 - 4.89] - Actual: 6.0
Prediction mean: 6.25, stddev: 0.77, 95% CI: [7.76 - 4.74] - Actual: 6.0
Prediction mean: 6.39, stddev: 0.78, 95% CI: [7.92 - 4.85] - Actual: 7.0

```
</div>