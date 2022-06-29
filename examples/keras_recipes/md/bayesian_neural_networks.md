# Probabilistic Bayesian Neural Networks

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/01/15<br>
**Last modified:** 2021/01/15<br>
**Description:** Building probabilistic Bayesian neural network models with TensorFlow Probability.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team\keras-io\blob\master\examples\keras_recipes/ipynb/bayesian_neural_networks.ipynb)  <span class="k-dot">�</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team\keras-io\blob\master\examples\keras_recipes/bayesian_neural_networks.py)



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

<div class="k-default-codeblock">
```
[1mDownloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to ~\tensorflow_datasets\wine_quality\white\1.0.0...[0m
```
</div>
    


<div class="k-default-codeblock">
```
Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...: 0 examples [00:00, ? examples/s]

Shuffling ~\tensorflow_datasets\wine_quality\white\1.0.0.incompleteLKW91L\wine_quality-train.tfrecord*...:   0…

[1mDataset wine_quality downloaded and prepared to ~\tensorflow_datasets\wine_quality\white\1.0.0. Subsequent calls will reuse this data.[0m
```
</div>
    

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
17/17 [==============================] - 1s 14ms/step - loss: 35.5860 - root_mean_squared_error: 5.9654 - val_loss: 33.0292 - val_root_mean_squared_error: 5.7471
Epoch 2/100
17/17 [==============================] - 0s 3ms/step - loss: 33.7436 - root_mean_squared_error: 5.8089 - val_loss: 31.3336 - val_root_mean_squared_error: 5.5976
Epoch 3/100
17/17 [==============================] - 0s 2ms/step - loss: 32.2078 - root_mean_squared_error: 5.6752 - val_loss: 29.8722 - val_root_mean_squared_error: 5.4655
Epoch 4/100
17/17 [==============================] - 0s 3ms/step - loss: 30.7526 - root_mean_squared_error: 5.5455 - val_loss: 28.5134 - val_root_mean_squared_error: 5.3398
Epoch 5/100
17/17 [==============================] - 0s 3ms/step - loss: 29.3498 - root_mean_squared_error: 5.4175 - val_loss: 27.2294 - val_root_mean_squared_error: 5.2182
Epoch 6/100
17/17 [==============================] - 0s 3ms/step - loss: 27.9862 - root_mean_squared_error: 5.2902 - val_loss: 25.9754 - val_root_mean_squared_error: 5.0966
Epoch 7/100
17/17 [==============================] - 0s 3ms/step - loss: 26.6470 - root_mean_squared_error: 5.1621 - val_loss: 24.7588 - val_root_mean_squared_error: 4.9758
Epoch 8/100
17/17 [==============================] - 0s 3ms/step - loss: 25.3238 - root_mean_squared_error: 5.0323 - val_loss: 23.5576 - val_root_mean_squared_error: 4.8536
Epoch 9/100
17/17 [==============================] - 0s 3ms/step - loss: 24.0192 - root_mean_squared_error: 4.9009 - val_loss: 22.3791 - val_root_mean_squared_error: 4.7307
Epoch 10/100
17/17 [==============================] - 0s 3ms/step - loss: 22.7357 - root_mean_squared_error: 4.7682 - val_loss: 21.2072 - val_root_mean_squared_error: 4.6051
Epoch 11/100
17/17 [==============================] - 0s 2ms/step - loss: 21.4646 - root_mean_squared_error: 4.6330 - val_loss: 20.0460 - val_root_mean_squared_error: 4.4773
Epoch 12/100
17/17 [==============================] - 0s 3ms/step - loss: 20.2093 - root_mean_squared_error: 4.4955 - val_loss: 18.8931 - val_root_mean_squared_error: 4.3466
Epoch 13/100
17/17 [==============================] - 0s 3ms/step - loss: 18.9731 - root_mean_squared_error: 4.3558 - val_loss: 17.7550 - val_root_mean_squared_error: 4.2137
Epoch 14/100
17/17 [==============================] - 0s 3ms/step - loss: 17.7627 - root_mean_squared_error: 4.2146 - val_loss: 16.6309 - val_root_mean_squared_error: 4.0781
Epoch 15/100
17/17 [==============================] - 0s 3ms/step - loss: 16.5788 - root_mean_squared_error: 4.0717 - val_loss: 15.5295 - val_root_mean_squared_error: 3.9407
Epoch 16/100
17/17 [==============================] - 0s 3ms/step - loss: 15.4303 - root_mean_squared_error: 3.9281 - val_loss: 14.4522 - val_root_mean_squared_error: 3.8016
Epoch 17/100
17/17 [==============================] - 0s 3ms/step - loss: 14.3183 - root_mean_squared_error: 3.7840 - val_loss: 13.4037 - val_root_mean_squared_error: 3.6611
Epoch 18/100
17/17 [==============================] - 0s 2ms/step - loss: 13.2449 - root_mean_squared_error: 3.6393 - val_loss: 12.3897 - val_root_mean_squared_error: 3.5199
Epoch 19/100
17/17 [==============================] - 0s 3ms/step - loss: 12.2174 - root_mean_squared_error: 3.4953 - val_loss: 11.4166 - val_root_mean_squared_error: 3.3788
Epoch 20/100
17/17 [==============================] - 0s 2ms/step - loss: 11.2381 - root_mean_squared_error: 3.3523 - val_loss: 10.4879 - val_root_mean_squared_error: 3.2385
Epoch 21/100
17/17 [==============================] - 0s 2ms/step - loss: 10.3063 - root_mean_squared_error: 3.2103 - val_loss: 9.5980 - val_root_mean_squared_error: 3.0981
Epoch 22/100
17/17 [==============================] - 0s 3ms/step - loss: 9.4194 - root_mean_squared_error: 3.0691 - val_loss: 8.7597 - val_root_mean_squared_error: 2.9597
Epoch 23/100
17/17 [==============================] - 0s 2ms/step - loss: 8.5841 - root_mean_squared_error: 2.9299 - val_loss: 7.9609 - val_root_mean_squared_error: 2.8215
Epoch 24/100
17/17 [==============================] - 0s 3ms/step - loss: 7.7929 - root_mean_squared_error: 2.7916 - val_loss: 7.2087 - val_root_mean_squared_error: 2.6849
Epoch 25/100
17/17 [==============================] - 0s 2ms/step - loss: 7.0491 - root_mean_squared_error: 2.6550 - val_loss: 6.5002 - val_root_mean_squared_error: 2.5495
Epoch 26/100
17/17 [==============================] - 0s 2ms/step - loss: 6.3512 - root_mean_squared_error: 2.5202 - val_loss: 5.8409 - val_root_mean_squared_error: 2.4168
Epoch 27/100
17/17 [==============================] - 0s 3ms/step - loss: 5.6987 - root_mean_squared_error: 2.3872 - val_loss: 5.2240 - val_root_mean_squared_error: 2.2856
Epoch 28/100
17/17 [==============================] - 0s 2ms/step - loss: 5.0913 - root_mean_squared_error: 2.2564 - val_loss: 4.6510 - val_root_mean_squared_error: 2.1566
Epoch 29/100
17/17 [==============================] - 0s 2ms/step - loss: 4.5287 - root_mean_squared_error: 2.1281 - val_loss: 4.1234 - val_root_mean_squared_error: 2.0306
Epoch 30/100
17/17 [==============================] - 0s 3ms/step - loss: 4.0080 - root_mean_squared_error: 2.0020 - val_loss: 3.6341 - val_root_mean_squared_error: 1.9063
Epoch 31/100
17/17 [==============================] - 0s 2ms/step - loss: 3.5284 - root_mean_squared_error: 1.8784 - val_loss: 3.1844 - val_root_mean_squared_error: 1.7845
Epoch 32/100
17/17 [==============================] - 0s 3ms/step - loss: 3.0870 - root_mean_squared_error: 1.7570 - val_loss: 2.7748 - val_root_mean_squared_error: 1.6658
Epoch 33/100
17/17 [==============================] - 0s 3ms/step - loss: 2.6872 - root_mean_squared_error: 1.6393 - val_loss: 2.4109 - val_root_mean_squared_error: 1.5527
Epoch 34/100
17/17 [==============================] - 0s 3ms/step - loss: 2.3280 - root_mean_squared_error: 1.5258 - val_loss: 2.0793 - val_root_mean_squared_error: 1.4420
Epoch 35/100
17/17 [==============================] - 0s 2ms/step - loss: 2.0061 - root_mean_squared_error: 1.4164 - val_loss: 1.7880 - val_root_mean_squared_error: 1.3372
Epoch 36/100
17/17 [==============================] - 0s 2ms/step - loss: 1.7220 - root_mean_squared_error: 1.3122 - val_loss: 1.5375 - val_root_mean_squared_error: 1.2400
Epoch 37/100
17/17 [==============================] - 0s 3ms/step - loss: 1.4802 - root_mean_squared_error: 1.2166 - val_loss: 1.3271 - val_root_mean_squared_error: 1.1520
Epoch 38/100
17/17 [==============================] - 0s 2ms/step - loss: 1.2760 - root_mean_squared_error: 1.1296 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0744
Epoch 39/100
17/17 [==============================] - 0s 2ms/step - loss: 1.1087 - root_mean_squared_error: 1.0530 - val_loss: 1.0152 - val_root_mean_squared_error: 1.0076
Epoch 40/100
17/17 [==============================] - 0s 3ms/step - loss: 0.9781 - root_mean_squared_error: 0.9890 - val_loss: 0.9132 - val_root_mean_squared_error: 0.9556
Epoch 41/100
17/17 [==============================] - 0s 2ms/step - loss: 0.8838 - root_mean_squared_error: 0.9401 - val_loss: 0.8478 - val_root_mean_squared_error: 0.9208
Epoch 42/100
17/17 [==============================] - 0s 2ms/step - loss: 0.8226 - root_mean_squared_error: 0.9070 - val_loss: 0.8105 - val_root_mean_squared_error: 0.9003
Epoch 43/100
17/17 [==============================] - 0s 3ms/step - loss: 0.7895 - root_mean_squared_error: 0.8885 - val_loss: 0.7976 - val_root_mean_squared_error: 0.8931
Epoch 44/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7780 - root_mean_squared_error: 0.8821 - val_loss: 0.7935 - val_root_mean_squared_error: 0.8908
Epoch 45/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7736 - root_mean_squared_error: 0.8795 - val_loss: 0.7917 - val_root_mean_squared_error: 0.8898
Epoch 46/100
17/17 [==============================] - 0s 3ms/step - loss: 0.7710 - root_mean_squared_error: 0.8780 - val_loss: 0.7883 - val_root_mean_squared_error: 0.8879
Epoch 47/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7673 - root_mean_squared_error: 0.8760 - val_loss: 0.7834 - val_root_mean_squared_error: 0.8851
Epoch 48/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7628 - root_mean_squared_error: 0.8734 - val_loss: 0.7773 - val_root_mean_squared_error: 0.8817
Epoch 49/100
17/17 [==============================] - 0s 3ms/step - loss: 0.7571 - root_mean_squared_error: 0.8701 - val_loss: 0.7696 - val_root_mean_squared_error: 0.8772
Epoch 50/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7497 - root_mean_squared_error: 0.8658 - val_loss: 0.7606 - val_root_mean_squared_error: 0.8721
Epoch 51/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7426 - root_mean_squared_error: 0.8618 - val_loss: 0.7511 - val_root_mean_squared_error: 0.8667
Epoch 52/100
17/17 [==============================] - 0s 3ms/step - loss: 0.7325 - root_mean_squared_error: 0.8558 - val_loss: 0.7426 - val_root_mean_squared_error: 0.8617
Epoch 53/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7216 - root_mean_squared_error: 0.8495 - val_loss: 0.7306 - val_root_mean_squared_error: 0.8548
Epoch 54/100
17/17 [==============================] - 0s 2ms/step - loss: 0.7086 - root_mean_squared_error: 0.8418 - val_loss: 0.7156 - val_root_mean_squared_error: 0.8459
Epoch 55/100
17/17 [==============================] - 0s 3ms/step - loss: 0.6981 - root_mean_squared_error: 0.8355 - val_loss: 0.7032 - val_root_mean_squared_error: 0.8386
Epoch 56/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6841 - root_mean_squared_error: 0.8271 - val_loss: 0.6905 - val_root_mean_squared_error: 0.8309
Epoch 57/100
17/17 [==============================] - 0s 3ms/step - loss: 0.6721 - root_mean_squared_error: 0.8198 - val_loss: 0.6770 - val_root_mean_squared_error: 0.8228
Epoch 58/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6606 - root_mean_squared_error: 0.8128 - val_loss: 0.6650 - val_root_mean_squared_error: 0.8155
Epoch 59/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6511 - root_mean_squared_error: 0.8069 - val_loss: 0.6550 - val_root_mean_squared_error: 0.8093
Epoch 60/100
17/17 [==============================] - 0s 3ms/step - loss: 0.6418 - root_mean_squared_error: 0.8011 - val_loss: 0.6461 - val_root_mean_squared_error: 0.8038
Epoch 61/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6349 - root_mean_squared_error: 0.7968 - val_loss: 0.6384 - val_root_mean_squared_error: 0.7990
Epoch 62/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6278 - root_mean_squared_error: 0.7924 - val_loss: 0.6315 - val_root_mean_squared_error: 0.7947
Epoch 63/100
17/17 [==============================] - 0s 3ms/step - loss: 0.6216 - root_mean_squared_error: 0.7884 - val_loss: 0.6262 - val_root_mean_squared_error: 0.7913
Epoch 64/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6171 - root_mean_squared_error: 0.7856 - val_loss: 0.6249 - val_root_mean_squared_error: 0.7905
Epoch 65/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6148 - root_mean_squared_error: 0.7841 - val_loss: 0.6162 - val_root_mean_squared_error: 0.7850
Epoch 66/100
17/17 [==============================] - 0s 3ms/step - loss: 0.6076 - root_mean_squared_error: 0.7795 - val_loss: 0.6120 - val_root_mean_squared_error: 0.7823
Epoch 67/100
17/17 [==============================] - 0s 2ms/step - loss: 0.6028 - root_mean_squared_error: 0.7764 - val_loss: 0.6061 - val_root_mean_squared_error: 0.7785
Epoch 68/100
17/17 [==============================] - 0s 3ms/step - loss: 0.6037 - root_mean_squared_error: 0.7770 - val_loss: 0.6036 - val_root_mean_squared_error: 0.7769
Epoch 69/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5999 - root_mean_squared_error: 0.7746 - val_loss: 0.5989 - val_root_mean_squared_error: 0.7739
Epoch 70/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5949 - root_mean_squared_error: 0.7713 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7723
Epoch 71/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5956 - root_mean_squared_error: 0.7717 - val_loss: 0.5932 - val_root_mean_squared_error: 0.7702
Epoch 72/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5920 - root_mean_squared_error: 0.7694 - val_loss: 0.5900 - val_root_mean_squared_error: 0.7681
Epoch 73/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5912 - root_mean_squared_error: 0.7689 - val_loss: 0.5886 - val_root_mean_squared_error: 0.7672
Epoch 74/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5866 - root_mean_squared_error: 0.7659 - val_loss: 0.5852 - val_root_mean_squared_error: 0.7650
Epoch 75/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5855 - root_mean_squared_error: 0.7652 - val_loss: 0.5835 - val_root_mean_squared_error: 0.7638
Epoch 76/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5835 - root_mean_squared_error: 0.7639 - val_loss: 0.5813 - val_root_mean_squared_error: 0.7624
Epoch 77/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5837 - root_mean_squared_error: 0.7640 - val_loss: 0.5797 - val_root_mean_squared_error: 0.7614
Epoch 78/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5821 - root_mean_squared_error: 0.7629 - val_loss: 0.5782 - val_root_mean_squared_error: 0.7604
Epoch 79/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5790 - root_mean_squared_error: 0.7609 - val_loss: 0.5758 - val_root_mean_squared_error: 0.7588
Epoch 80/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5766 - root_mean_squared_error: 0.7594 - val_loss: 0.5749 - val_root_mean_squared_error: 0.7582
Epoch 81/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5755 - root_mean_squared_error: 0.7586 - val_loss: 0.5724 - val_root_mean_squared_error: 0.7566
Epoch 82/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5749 - root_mean_squared_error: 0.7582 - val_loss: 0.5708 - val_root_mean_squared_error: 0.7555
Epoch 83/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5732 - root_mean_squared_error: 0.7571 - val_loss: 0.5692 - val_root_mean_squared_error: 0.7545
Epoch 84/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5741 - root_mean_squared_error: 0.7577 - val_loss: 0.5678 - val_root_mean_squared_error: 0.7535
Epoch 85/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5716 - root_mean_squared_error: 0.7560 - val_loss: 0.5669 - val_root_mean_squared_error: 0.7529
Epoch 86/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5704 - root_mean_squared_error: 0.7553 - val_loss: 0.5648 - val_root_mean_squared_error: 0.7516
Epoch 87/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5697 - root_mean_squared_error: 0.7548 - val_loss: 0.5656 - val_root_mean_squared_error: 0.7521
Epoch 88/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5669 - root_mean_squared_error: 0.7529 - val_loss: 0.5624 - val_root_mean_squared_error: 0.7500
Epoch 89/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5669 - root_mean_squared_error: 0.7529 - val_loss: 0.5610 - val_root_mean_squared_error: 0.7490
Epoch 90/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5654 - root_mean_squared_error: 0.7519 - val_loss: 0.5613 - val_root_mean_squared_error: 0.7492
Epoch 91/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5670 - root_mean_squared_error: 0.7530 - val_loss: 0.5602 - val_root_mean_squared_error: 0.7485
Epoch 92/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5647 - root_mean_squared_error: 0.7515 - val_loss: 0.5579 - val_root_mean_squared_error: 0.7469
Epoch 93/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5642 - root_mean_squared_error: 0.7511 - val_loss: 0.5559 - val_root_mean_squared_error: 0.7456
Epoch 94/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5621 - root_mean_squared_error: 0.7497 - val_loss: 0.5548 - val_root_mean_squared_error: 0.7449
Epoch 95/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5608 - root_mean_squared_error: 0.7489 - val_loss: 0.5536 - val_root_mean_squared_error: 0.7441
Epoch 96/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5621 - root_mean_squared_error: 0.7498 - val_loss: 0.5530 - val_root_mean_squared_error: 0.7437
Epoch 97/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5602 - root_mean_squared_error: 0.7485 - val_loss: 0.5515 - val_root_mean_squared_error: 0.7426
Epoch 98/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5599 - root_mean_squared_error: 0.7482 - val_loss: 0.5505 - val_root_mean_squared_error: 0.7420
Epoch 99/100
17/17 [==============================] - 0s 3ms/step - loss: 0.5579 - root_mean_squared_error: 0.7469 - val_loss: 0.5490 - val_root_mean_squared_error: 0.7409
Epoch 100/100
17/17 [==============================] - 0s 2ms/step - loss: 0.5592 - root_mean_squared_error: 0.7478 - val_loss: 0.5484 - val_root_mean_squared_error: 0.7405
Model training finished.
Train RMSE: 0.746
Evaluating model performance...
Test RMSE: 0.741
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
Predicted: 6.1 - Actual: 6.0
Predicted: 4.9 - Actual: 4.0
Predicted: 5.9 - Actual: 5.0
Predicted: 6.5 - Actual: 7.0
Predicted: 5.7 - Actual: 5.0
Predicted: 6.1 - Actual: 6.0
Predicted: 5.4 - Actual: 6.0
Predicted: 6.5 - Actual: 6.0
Predicted: 6.5 - Actual: 7.0
Predicted: 6.5 - Actual: 6.0
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
WARNING:tensorflow:From C:\Users\conno\anaconda3\envs\bayes\lib\site-packages\tensorflow_probability\python\distributions\distribution.py:346: calling MultivariateNormalDiag.__init__ (from tensorflow_probability.python.distributions.mvn_diag) with scale_identity_multiplier is deprecated and will be removed after 2020-01-01.
Instructions for updating:
`scale_identity_multiplier` is deprecated; please combine it into `scale_diag` directly instead.
```
</div>
    

<div class="k-default-codeblock">
```
WARNING:tensorflow:From C:\Users\conno\anaconda3\envs\bayes\lib\site-packages\tensorflow_probability\python\distributions\distribution.py:346: calling MultivariateNormalDiag.__init__ (from tensorflow_probability.python.distributions.mvn_diag) with scale_identity_multiplier is deprecated and will be removed after 2020-01-01.
Instructions for updating:
`scale_identity_multiplier` is deprecated; please combine it into `scale_diag` directly instead.
```
</div>
    

<div class="k-default-codeblock">
```
Start training the model...
Epoch 1/500
5/5 [==============================] - 2s 239ms/step - loss: 46.2377 - root_mean_squared_error: 6.7989 - val_loss: 49.0011 - val_root_mean_squared_error: 6.9989
Epoch 2/500
5/5 [==============================] - 0s 8ms/step - loss: 52.9262 - root_mean_squared_error: 7.2742 - val_loss: 46.8293 - val_root_mean_squared_error: 6.8422
Epoch 3/500
5/5 [==============================] - 0s 8ms/step - loss: 45.8535 - root_mean_squared_error: 6.7705 - val_loss: 42.9995 - val_root_mean_squared_error: 6.5564
Epoch 4/500
5/5 [==============================] - 0s 9ms/step - loss: 45.6427 - root_mean_squared_error: 6.7549 - val_loss: 43.5259 - val_root_mean_squared_error: 6.5963
Epoch 5/500
5/5 [==============================] - 0s 8ms/step - loss: 48.2844 - root_mean_squared_error: 6.9476 - val_loss: 41.7103 - val_root_mean_squared_error: 6.4572
Epoch 6/500
5/5 [==============================] - 0s 8ms/step - loss: 40.1781 - root_mean_squared_error: 6.3374 - val_loss: 43.2038 - val_root_mean_squared_error: 6.5718
Epoch 7/500
5/5 [==============================] - 0s 9ms/step - loss: 46.1981 - root_mean_squared_error: 6.7959 - val_loss: 46.4081 - val_root_mean_squared_error: 6.8111
Epoch 8/500
5/5 [==============================] - 0s 7ms/step - loss: 40.5498 - root_mean_squared_error: 6.3665 - val_loss: 40.5166 - val_root_mean_squared_error: 6.3641
Epoch 9/500
5/5 [==============================] - 0s 7ms/step - loss: 38.6053 - root_mean_squared_error: 6.2121 - val_loss: 48.2229 - val_root_mean_squared_error: 6.9436
Epoch 10/500
5/5 [==============================] - 0s 8ms/step - loss: 46.0120 - root_mean_squared_error: 6.7821 - val_loss: 43.8349 - val_root_mean_squared_error: 6.6199
Epoch 11/500
5/5 [==============================] - 0s 7ms/step - loss: 37.6656 - root_mean_squared_error: 6.1360 - val_loss: 41.7939 - val_root_mean_squared_error: 6.4637
Epoch 12/500
5/5 [==============================] - 0s 6ms/step - loss: 41.5022 - root_mean_squared_error: 6.4411 - val_loss: 33.9450 - val_root_mean_squared_error: 5.8248
Epoch 13/500
5/5 [==============================] - 0s 8ms/step - loss: 36.3389 - root_mean_squared_error: 6.0271 - val_loss: 43.2833 - val_root_mean_squared_error: 6.5777
Epoch 14/500
5/5 [==============================] - 0s 8ms/step - loss: 38.2426 - root_mean_squared_error: 6.1829 - val_loss: 36.0395 - val_root_mean_squared_error: 6.0022
Epoch 15/500
5/5 [==============================] - 0s 6ms/step - loss: 43.7655 - root_mean_squared_error: 6.6143 - val_loss: 38.7248 - val_root_mean_squared_error: 6.2220
Epoch 16/500
5/5 [==============================] - 0s 8ms/step - loss: 41.8891 - root_mean_squared_error: 6.4712 - val_loss: 42.7908 - val_root_mean_squared_error: 6.5402
Epoch 17/500
5/5 [==============================] - 0s 8ms/step - loss: 36.5762 - root_mean_squared_error: 6.0465 - val_loss: 40.9752 - val_root_mean_squared_error: 6.3998
Epoch 18/500
5/5 [==============================] - 0s 7ms/step - loss: 39.5787 - root_mean_squared_error: 6.2899 - val_loss: 33.3126 - val_root_mean_squared_error: 5.7702
Epoch 19/500
5/5 [==============================] - 0s 8ms/step - loss: 37.0878 - root_mean_squared_error: 6.0889 - val_loss: 37.7592 - val_root_mean_squared_error: 6.1436
Epoch 20/500
5/5 [==============================] - 0s 8ms/step - loss: 37.2262 - root_mean_squared_error: 6.1001 - val_loss: 40.1233 - val_root_mean_squared_error: 6.3329
Epoch 21/500
5/5 [==============================] - 0s 8ms/step - loss: 34.2142 - root_mean_squared_error: 5.8479 - val_loss: 37.8172 - val_root_mean_squared_error: 6.1485
Epoch 22/500
5/5 [==============================] - 0s 9ms/step - loss: 35.8846 - root_mean_squared_error: 5.9889 - val_loss: 37.3976 - val_root_mean_squared_error: 6.1140
Epoch 23/500
5/5 [==============================] - 0s 7ms/step - loss: 35.7685 - root_mean_squared_error: 5.9795 - val_loss: 35.8500 - val_root_mean_squared_error: 5.9866
Epoch 24/500
5/5 [==============================] - 0s 7ms/step - loss: 35.0936 - root_mean_squared_error: 5.9225 - val_loss: 37.6272 - val_root_mean_squared_error: 6.1331
Epoch 25/500
5/5 [==============================] - 0s 8ms/step - loss: 33.8082 - root_mean_squared_error: 5.8134 - val_loss: 33.2308 - val_root_mean_squared_error: 5.7629
Epoch 26/500
5/5 [==============================] - 0s 7ms/step - loss: 35.7079 - root_mean_squared_error: 5.9745 - val_loss: 32.6228 - val_root_mean_squared_error: 5.7103
Epoch 27/500
5/5 [==============================] - 0s 7ms/step - loss: 31.8274 - root_mean_squared_error: 5.6405 - val_loss: 41.7901 - val_root_mean_squared_error: 6.4635
Epoch 28/500
5/5 [==============================] - 0s 8ms/step - loss: 33.6900 - root_mean_squared_error: 5.8029 - val_loss: 32.6052 - val_root_mean_squared_error: 5.7087
Epoch 29/500
5/5 [==============================] - 0s 7ms/step - loss: 32.2443 - root_mean_squared_error: 5.6770 - val_loss: 31.6094 - val_root_mean_squared_error: 5.6208
Epoch 30/500
5/5 [==============================] - 0s 7ms/step - loss: 35.2510 - root_mean_squared_error: 5.9360 - val_loss: 40.0530 - val_root_mean_squared_error: 6.3277
Epoch 31/500
5/5 [==============================] - 0s 8ms/step - loss: 33.7461 - root_mean_squared_error: 5.8077 - val_loss: 32.2815 - val_root_mean_squared_error: 5.6804
Epoch 32/500
5/5 [==============================] - 0s 8ms/step - loss: 35.6961 - root_mean_squared_error: 5.9733 - val_loss: 34.9137 - val_root_mean_squared_error: 5.9075
Epoch 33/500
5/5 [==============================] - 0s 7ms/step - loss: 33.1791 - root_mean_squared_error: 5.7588 - val_loss: 31.5960 - val_root_mean_squared_error: 5.6193
Epoch 34/500
5/5 [==============================] - 0s 8ms/step - loss: 29.6632 - root_mean_squared_error: 5.4451 - val_loss: 32.7611 - val_root_mean_squared_error: 5.7225
Epoch 35/500
5/5 [==============================] - 0s 7ms/step - loss: 32.3230 - root_mean_squared_error: 5.6839 - val_loss: 29.7752 - val_root_mean_squared_error: 5.4546
Epoch 36/500
5/5 [==============================] - 0s 8ms/step - loss: 27.7547 - root_mean_squared_error: 5.2666 - val_loss: 28.9397 - val_root_mean_squared_error: 5.3779
Epoch 37/500
5/5 [==============================] - 0s 8ms/step - loss: 31.1692 - root_mean_squared_error: 5.5815 - val_loss: 31.2919 - val_root_mean_squared_error: 5.5922
Epoch 38/500
5/5 [==============================] - 0s 8ms/step - loss: 29.1104 - root_mean_squared_error: 5.3939 - val_loss: 31.3540 - val_root_mean_squared_error: 5.5979
Epoch 39/500
5/5 [==============================] - 0s 7ms/step - loss: 29.3911 - root_mean_squared_error: 5.4197 - val_loss: 26.2233 - val_root_mean_squared_error: 5.1191
Epoch 40/500
5/5 [==============================] - 0s 8ms/step - loss: 28.9432 - root_mean_squared_error: 5.3785 - val_loss: 27.2295 - val_root_mean_squared_error: 5.2165
Epoch 41/500
5/5 [==============================] - 0s 7ms/step - loss: 28.0697 - root_mean_squared_error: 5.2965 - val_loss: 26.9286 - val_root_mean_squared_error: 5.1875
Epoch 42/500
5/5 [==============================] - 0s 8ms/step - loss: 28.3805 - root_mean_squared_error: 5.3259 - val_loss: 24.8901 - val_root_mean_squared_error: 4.9871
Epoch 43/500
5/5 [==============================] - 0s 8ms/step - loss: 29.7593 - root_mean_squared_error: 5.4538 - val_loss: 25.3729 - val_root_mean_squared_error: 5.0354
Epoch 44/500
5/5 [==============================] - 0s 7ms/step - loss: 26.9240 - root_mean_squared_error: 5.1874 - val_loss: 26.4675 - val_root_mean_squared_error: 5.1428
Epoch 45/500
5/5 [==============================] - 0s 7ms/step - loss: 25.6454 - root_mean_squared_error: 5.0624 - val_loss: 34.3410 - val_root_mean_squared_error: 5.8592
Epoch 46/500
5/5 [==============================] - 0s 8ms/step - loss: 27.3551 - root_mean_squared_error: 5.2287 - val_loss: 27.9540 - val_root_mean_squared_error: 5.2853
Epoch 47/500
5/5 [==============================] - 0s 8ms/step - loss: 26.9219 - root_mean_squared_error: 5.1871 - val_loss: 28.1336 - val_root_mean_squared_error: 5.3024
Epoch 48/500
5/5 [==============================] - 0s 8ms/step - loss: 24.9465 - root_mean_squared_error: 4.9929 - val_loss: 26.6770 - val_root_mean_squared_error: 5.1633
Epoch 49/500
5/5 [==============================] - 0s 7ms/step - loss: 25.7313 - root_mean_squared_error: 5.0708 - val_loss: 27.7764 - val_root_mean_squared_error: 5.2687
Epoch 50/500
5/5 [==============================] - 0s 7ms/step - loss: 27.4524 - root_mean_squared_error: 5.2377 - val_loss: 24.8870 - val_root_mean_squared_error: 4.9874
Epoch 51/500
5/5 [==============================] - 0s 8ms/step - loss: 25.6330 - root_mean_squared_error: 5.0611 - val_loss: 24.5337 - val_root_mean_squared_error: 4.9518
Epoch 52/500
5/5 [==============================] - 0s 8ms/step - loss: 22.3847 - root_mean_squared_error: 4.7295 - val_loss: 28.3979 - val_root_mean_squared_error: 5.3275
Epoch 53/500
5/5 [==============================] - 0s 8ms/step - loss: 26.0658 - root_mean_squared_error: 5.1039 - val_loss: 24.6880 - val_root_mean_squared_error: 4.9667
Epoch 54/500
5/5 [==============================] - 0s 8ms/step - loss: 24.3243 - root_mean_squared_error: 4.9304 - val_loss: 21.6574 - val_root_mean_squared_error: 4.6516
Epoch 55/500
5/5 [==============================] - 0s 7ms/step - loss: 23.8808 - root_mean_squared_error: 4.8852 - val_loss: 24.5205 - val_root_mean_squared_error: 4.9502
Epoch 56/500
5/5 [==============================] - 0s 7ms/step - loss: 22.2468 - root_mean_squared_error: 4.7145 - val_loss: 21.8014 - val_root_mean_squared_error: 4.6672
Epoch 57/500
5/5 [==============================] - 0s 7ms/step - loss: 25.1863 - root_mean_squared_error: 5.0170 - val_loss: 23.1018 - val_root_mean_squared_error: 4.8051
Epoch 58/500
5/5 [==============================] - 0s 7ms/step - loss: 24.2818 - root_mean_squared_error: 4.9260 - val_loss: 23.1634 - val_root_mean_squared_error: 4.8113
Epoch 59/500
5/5 [==============================] - 0s 8ms/step - loss: 22.4691 - root_mean_squared_error: 4.7382 - val_loss: 20.9803 - val_root_mean_squared_error: 4.5782
Epoch 60/500
5/5 [==============================] - 0s 7ms/step - loss: 22.6180 - root_mean_squared_error: 4.7541 - val_loss: 22.3755 - val_root_mean_squared_error: 4.7284
Epoch 61/500
5/5 [==============================] - 0s 7ms/step - loss: 22.3317 - root_mean_squared_error: 4.7237 - val_loss: 20.2962 - val_root_mean_squared_error: 4.5030
Epoch 62/500
5/5 [==============================] - 0s 9ms/step - loss: 22.7876 - root_mean_squared_error: 4.7717 - val_loss: 24.2449 - val_root_mean_squared_error: 4.9221
Epoch 63/500
5/5 [==============================] - 0s 8ms/step - loss: 22.6093 - root_mean_squared_error: 4.7530 - val_loss: 23.4267 - val_root_mean_squared_error: 4.8381
Epoch 64/500
5/5 [==============================] - 0s 8ms/step - loss: 21.8746 - root_mean_squared_error: 4.6753 - val_loss: 19.1240 - val_root_mean_squared_error: 4.3709
Epoch 65/500
5/5 [==============================] - 0s 9ms/step - loss: 22.7078 - root_mean_squared_error: 4.7637 - val_loss: 19.6240 - val_root_mean_squared_error: 4.4276
Epoch 66/500
5/5 [==============================] - 0s 8ms/step - loss: 21.4172 - root_mean_squared_error: 4.6261 - val_loss: 20.6224 - val_root_mean_squared_error: 4.5390
Epoch 67/500
5/5 [==============================] - 0s 9ms/step - loss: 20.5307 - root_mean_squared_error: 4.5290 - val_loss: 18.0356 - val_root_mean_squared_error: 4.2441
Epoch 68/500
5/5 [==============================] - 0s 7ms/step - loss: 18.4069 - root_mean_squared_error: 4.2877 - val_loss: 18.2721 - val_root_mean_squared_error: 4.2726
Epoch 69/500
5/5 [==============================] - 0s 7ms/step - loss: 18.6000 - root_mean_squared_error: 4.3106 - val_loss: 18.9493 - val_root_mean_squared_error: 4.3508
Epoch 70/500
5/5 [==============================] - 0s 6ms/step - loss: 18.8891 - root_mean_squared_error: 4.3438 - val_loss: 21.2702 - val_root_mean_squared_error: 4.6099
Epoch 71/500
5/5 [==============================] - 0s 8ms/step - loss: 21.2569 - root_mean_squared_error: 4.6087 - val_loss: 17.9836 - val_root_mean_squared_error: 4.2378
Epoch 72/500
5/5 [==============================] - 0s 7ms/step - loss: 18.1872 - root_mean_squared_error: 4.2621 - val_loss: 20.2043 - val_root_mean_squared_error: 4.4932
Epoch 73/500
5/5 [==============================] - 0s 7ms/step - loss: 20.1426 - root_mean_squared_error: 4.4864 - val_loss: 19.9946 - val_root_mean_squared_error: 4.4696
Epoch 74/500
5/5 [==============================] - 0s 8ms/step - loss: 18.5614 - root_mean_squared_error: 4.3062 - val_loss: 15.9425 - val_root_mean_squared_error: 3.9898
Epoch 75/500
5/5 [==============================] - 0s 8ms/step - loss: 18.9486 - root_mean_squared_error: 4.3510 - val_loss: 16.6702 - val_root_mean_squared_error: 4.0809
Epoch 76/500
5/5 [==============================] - 0s 9ms/step - loss: 18.5912 - root_mean_squared_error: 4.3094 - val_loss: 16.3551 - val_root_mean_squared_error: 4.0422
Epoch 77/500
5/5 [==============================] - 0s 8ms/step - loss: 16.4029 - root_mean_squared_error: 4.0470 - val_loss: 17.1655 - val_root_mean_squared_error: 4.1408
Epoch 78/500
5/5 [==============================] - 0s 7ms/step - loss: 17.5889 - root_mean_squared_error: 4.1917 - val_loss: 15.9467 - val_root_mean_squared_error: 3.9906
Epoch 79/500
5/5 [==============================] - 0s 10ms/step - loss: 18.1798 - root_mean_squared_error: 4.2614 - val_loss: 17.8881 - val_root_mean_squared_error: 4.2271
Epoch 80/500
5/5 [==============================] - 0s 8ms/step - loss: 17.0017 - root_mean_squared_error: 4.1214 - val_loss: 15.5239 - val_root_mean_squared_error: 3.9373
Epoch 81/500
5/5 [==============================] - 0s 7ms/step - loss: 17.2094 - root_mean_squared_error: 4.1458 - val_loss: 15.5859 - val_root_mean_squared_error: 3.9451
Epoch 82/500
5/5 [==============================] - 0s 9ms/step - loss: 17.2572 - root_mean_squared_error: 4.1522 - val_loss: 15.9025 - val_root_mean_squared_error: 3.9850
Epoch 83/500
5/5 [==============================] - 0s 8ms/step - loss: 15.9265 - root_mean_squared_error: 3.9883 - val_loss: 15.4678 - val_root_mean_squared_error: 3.9303
Epoch 84/500
5/5 [==============================] - 0s 7ms/step - loss: 14.8140 - root_mean_squared_error: 3.8462 - val_loss: 13.8966 - val_root_mean_squared_error: 3.7247
Epoch 85/500
5/5 [==============================] - 0s 9ms/step - loss: 15.6210 - root_mean_squared_error: 3.9501 - val_loss: 14.3627 - val_root_mean_squared_error: 3.7870
Epoch 86/500
5/5 [==============================] - 0s 8ms/step - loss: 15.5299 - root_mean_squared_error: 3.9383 - val_loss: 14.4062 - val_root_mean_squared_error: 3.7929
Epoch 87/500
5/5 [==============================] - 0s 10ms/step - loss: 13.3381 - root_mean_squared_error: 3.6486 - val_loss: 13.6614 - val_root_mean_squared_error: 3.6936
Epoch 88/500
5/5 [==============================] - 0s 8ms/step - loss: 14.3805 - root_mean_squared_error: 3.7891 - val_loss: 13.5505 - val_root_mean_squared_error: 3.6779
Epoch 89/500
5/5 [==============================] - 0s 8ms/step - loss: 14.1007 - root_mean_squared_error: 3.7522 - val_loss: 14.8413 - val_root_mean_squared_error: 3.8502
Epoch 90/500
5/5 [==============================] - 0s 8ms/step - loss: 16.7443 - root_mean_squared_error: 4.0894 - val_loss: 12.9235 - val_root_mean_squared_error: 3.5912
Epoch 91/500
5/5 [==============================] - 0s 8ms/step - loss: 13.0323 - root_mean_squared_error: 3.6066 - val_loss: 14.4730 - val_root_mean_squared_error: 3.8016
Epoch 92/500
5/5 [==============================] - 0s 8ms/step - loss: 13.1593 - root_mean_squared_error: 3.6242 - val_loss: 12.9546 - val_root_mean_squared_error: 3.5959
Epoch 93/500
5/5 [==============================] - 0s 9ms/step - loss: 14.5558 - root_mean_squared_error: 3.8125 - val_loss: 13.4988 - val_root_mean_squared_error: 3.6712
Epoch 94/500
5/5 [==============================] - 0s 8ms/step - loss: 13.7224 - root_mean_squared_error: 3.7014 - val_loss: 12.7883 - val_root_mean_squared_error: 3.5728
Epoch 95/500
5/5 [==============================] - 0s 9ms/step - loss: 14.0358 - root_mean_squared_error: 3.7434 - val_loss: 13.3544 - val_root_mean_squared_error: 3.6522
Epoch 96/500
5/5 [==============================] - 0s 8ms/step - loss: 12.5854 - root_mean_squared_error: 3.5445 - val_loss: 14.8026 - val_root_mean_squared_error: 3.8445
Epoch 97/500
5/5 [==============================] - 0s 7ms/step - loss: 13.0674 - root_mean_squared_error: 3.6117 - val_loss: 11.6989 - val_root_mean_squared_error: 3.4167
Epoch 98/500
5/5 [==============================] - 0s 8ms/step - loss: 12.2989 - root_mean_squared_error: 3.5037 - val_loss: 11.1329 - val_root_mean_squared_error: 3.3335
Epoch 99/500
5/5 [==============================] - 0s 8ms/step - loss: 12.7089 - root_mean_squared_error: 3.5620 - val_loss: 13.2025 - val_root_mean_squared_error: 3.6301
Epoch 100/500
5/5 [==============================] - 0s 7ms/step - loss: 14.2369 - root_mean_squared_error: 3.7706 - val_loss: 12.3462 - val_root_mean_squared_error: 3.5100
Epoch 101/500
5/5 [==============================] - 0s 7ms/step - loss: 12.5098 - root_mean_squared_error: 3.5335 - val_loss: 11.0389 - val_root_mean_squared_error: 3.3192
Epoch 102/500
5/5 [==============================] - 0s 8ms/step - loss: 12.4746 - root_mean_squared_error: 3.5287 - val_loss: 11.4848 - val_root_mean_squared_error: 3.3854
Epoch 103/500
5/5 [==============================] - 0s 8ms/step - loss: 11.5089 - root_mean_squared_error: 3.3890 - val_loss: 10.6527 - val_root_mean_squared_error: 3.2606
Epoch 104/500
5/5 [==============================] - 0s 8ms/step - loss: 11.9449 - root_mean_squared_error: 3.4526 - val_loss: 12.9800 - val_root_mean_squared_error: 3.5995
Epoch 105/500
5/5 [==============================] - 0s 7ms/step - loss: 12.3183 - root_mean_squared_error: 3.5066 - val_loss: 10.1739 - val_root_mean_squared_error: 3.1858
Epoch 106/500
5/5 [==============================] - 0s 7ms/step - loss: 11.0529 - root_mean_squared_error: 3.3209 - val_loss: 11.1035 - val_root_mean_squared_error: 3.3288
Epoch 107/500
5/5 [==============================] - 0s 9ms/step - loss: 10.7807 - root_mean_squared_error: 3.2799 - val_loss: 12.9830 - val_root_mean_squared_error: 3.6010
Epoch 108/500
5/5 [==============================] - 0s 8ms/step - loss: 10.4866 - root_mean_squared_error: 3.2349 - val_loss: 10.3904 - val_root_mean_squared_error: 3.2202
Epoch 109/500
5/5 [==============================] - 0s 7ms/step - loss: 10.6603 - root_mean_squared_error: 3.2617 - val_loss: 10.7658 - val_root_mean_squared_error: 3.2776
Epoch 110/500
5/5 [==============================] - 0s 8ms/step - loss: 10.6952 - root_mean_squared_error: 3.2664 - val_loss: 10.1145 - val_root_mean_squared_error: 3.1765
Epoch 111/500
5/5 [==============================] - 0s 8ms/step - loss: 10.2908 - root_mean_squared_error: 3.2035 - val_loss: 9.2185 - val_root_mean_squared_error: 3.0327
Epoch 112/500
5/5 [==============================] - 0s 7ms/step - loss: 9.3915 - root_mean_squared_error: 3.0602 - val_loss: 9.2944 - val_root_mean_squared_error: 3.0445
Epoch 113/500
5/5 [==============================] - 0s 7ms/step - loss: 10.5705 - root_mean_squared_error: 3.2478 - val_loss: 9.8069 - val_root_mean_squared_error: 3.1284
Epoch 114/500
5/5 [==============================] - 0s 7ms/step - loss: 9.5622 - root_mean_squared_error: 3.0883 - val_loss: 10.0188 - val_root_mean_squared_error: 3.1613
Epoch 115/500
5/5 [==============================] - 0s 7ms/step - loss: 9.3348 - root_mean_squared_error: 3.0511 - val_loss: 8.6187 - val_root_mean_squared_error: 2.9308
Epoch 116/500
5/5 [==============================] - 0s 8ms/step - loss: 9.8692 - root_mean_squared_error: 3.1375 - val_loss: 8.6462 - val_root_mean_squared_error: 2.9358
Epoch 117/500
5/5 [==============================] - 0s 8ms/step - loss: 10.0576 - root_mean_squared_error: 3.1679 - val_loss: 8.4217 - val_root_mean_squared_error: 2.8980
Epoch 118/500
5/5 [==============================] - 0s 7ms/step - loss: 9.1009 - root_mean_squared_error: 3.0129 - val_loss: 9.4706 - val_root_mean_squared_error: 3.0735
Epoch 119/500
5/5 [==============================] - 0s 9ms/step - loss: 9.7467 - root_mean_squared_error: 3.1188 - val_loss: 9.5122 - val_root_mean_squared_error: 3.0804
Epoch 120/500
5/5 [==============================] - 0s 7ms/step - loss: 9.8879 - root_mean_squared_error: 3.1410 - val_loss: 8.1646 - val_root_mean_squared_error: 2.8535
Epoch 121/500
5/5 [==============================] - 0s 8ms/step - loss: 8.7129 - root_mean_squared_error: 2.9474 - val_loss: 7.9832 - val_root_mean_squared_error: 2.8215
Epoch 122/500
5/5 [==============================] - 0s 8ms/step - loss: 8.1263 - root_mean_squared_error: 2.8457 - val_loss: 7.7345 - val_root_mean_squared_error: 2.7767
Epoch 123/500
5/5 [==============================] - 0s 7ms/step - loss: 8.0503 - root_mean_squared_error: 2.8327 - val_loss: 8.8093 - val_root_mean_squared_error: 2.9642
Epoch 124/500
5/5 [==============================] - 0s 7ms/step - loss: 7.6386 - root_mean_squared_error: 2.7596 - val_loss: 7.4452 - val_root_mean_squared_error: 2.7240
Epoch 125/500
5/5 [==============================] - 0s 8ms/step - loss: 8.0457 - root_mean_squared_error: 2.8317 - val_loss: 7.7812 - val_root_mean_squared_error: 2.7845
Epoch 126/500
5/5 [==============================] - 0s 8ms/step - loss: 8.0535 - root_mean_squared_error: 2.8336 - val_loss: 9.3565 - val_root_mean_squared_error: 3.0554
Epoch 127/500
5/5 [==============================] - 0s 7ms/step - loss: 8.2439 - root_mean_squared_error: 2.8671 - val_loss: 7.6753 - val_root_mean_squared_error: 2.7655
Epoch 128/500
5/5 [==============================] - 0s 8ms/step - loss: 7.8935 - root_mean_squared_error: 2.8044 - val_loss: 7.5042 - val_root_mean_squared_error: 2.7349
Epoch 129/500
5/5 [==============================] - 0s 8ms/step - loss: 7.8573 - root_mean_squared_error: 2.7988 - val_loss: 6.9096 - val_root_mean_squared_error: 2.6232
Epoch 130/500
5/5 [==============================] - 0s 7ms/step - loss: 6.7604 - root_mean_squared_error: 2.5951 - val_loss: 7.2327 - val_root_mean_squared_error: 2.6844
Epoch 131/500
5/5 [==============================] - 0s 8ms/step - loss: 7.1849 - root_mean_squared_error: 2.6755 - val_loss: 7.3409 - val_root_mean_squared_error: 2.7049
Epoch 132/500
5/5 [==============================] - 0s 8ms/step - loss: 7.2442 - root_mean_squared_error: 2.6870 - val_loss: 6.0521 - val_root_mean_squared_error: 2.4552
Epoch 133/500
5/5 [==============================] - 0s 7ms/step - loss: 6.3105 - root_mean_squared_error: 2.5067 - val_loss: 6.6750 - val_root_mean_squared_error: 2.5793
Epoch 134/500
5/5 [==============================] - 0s 8ms/step - loss: 7.7914 - root_mean_squared_error: 2.7870 - val_loss: 6.8932 - val_root_mean_squared_error: 2.6210
Epoch 135/500
5/5 [==============================] - 0s 7ms/step - loss: 6.7076 - root_mean_squared_error: 2.5850 - val_loss: 5.8227 - val_root_mean_squared_error: 2.4077
Epoch 136/500
5/5 [==============================] - 0s 7ms/step - loss: 6.1051 - root_mean_squared_error: 2.4650 - val_loss: 5.6785 - val_root_mean_squared_error: 2.3771
Epoch 137/500
5/5 [==============================] - 0s 8ms/step - loss: 6.6653 - root_mean_squared_error: 2.5771 - val_loss: 6.2303 - val_root_mean_squared_error: 2.4907
Epoch 138/500
5/5 [==============================] - 0s 7ms/step - loss: 6.4688 - root_mean_squared_error: 2.5383 - val_loss: 6.4154 - val_root_mean_squared_error: 2.5272
Epoch 139/500
5/5 [==============================] - 0s 9ms/step - loss: 6.3343 - root_mean_squared_error: 2.5117 - val_loss: 5.4067 - val_root_mean_squared_error: 2.3186
Epoch 140/500
5/5 [==============================] - 0s 8ms/step - loss: 6.0890 - root_mean_squared_error: 2.4622 - val_loss: 5.7272 - val_root_mean_squared_error: 2.3878
Epoch 141/500
5/5 [==============================] - 0s 7ms/step - loss: 5.8547 - root_mean_squared_error: 2.4147 - val_loss: 5.5547 - val_root_mean_squared_error: 2.3515
Epoch 142/500
5/5 [==============================] - 0s 8ms/step - loss: 5.8994 - root_mean_squared_error: 2.4236 - val_loss: 4.8616 - val_root_mean_squared_error: 2.1984
Epoch 143/500
5/5 [==============================] - 0s 7ms/step - loss: 5.0084 - root_mean_squared_error: 2.2319 - val_loss: 5.0371 - val_root_mean_squared_error: 2.2388
Epoch 144/500
5/5 [==============================] - 0s 6ms/step - loss: 4.9666 - root_mean_squared_error: 2.2216 - val_loss: 4.8042 - val_root_mean_squared_error: 2.1852
Epoch 145/500
5/5 [==============================] - 0s 8ms/step - loss: 5.2728 - root_mean_squared_error: 2.2904 - val_loss: 5.2731 - val_root_mean_squared_error: 2.2905
Epoch 146/500
5/5 [==============================] - 0s 7ms/step - loss: 5.6613 - root_mean_squared_error: 2.3738 - val_loss: 5.1354 - val_root_mean_squared_error: 2.2593
Epoch 147/500
5/5 [==============================] - 0s 6ms/step - loss: 5.1640 - root_mean_squared_error: 2.2671 - val_loss: 4.5920 - val_root_mean_squared_error: 2.1360
Epoch 148/500
5/5 [==============================] - 0s 8ms/step - loss: 4.9738 - root_mean_squared_error: 2.2243 - val_loss: 4.3932 - val_root_mean_squared_error: 2.0881
Epoch 149/500
5/5 [==============================] - 0s 7ms/step - loss: 4.1795 - root_mean_squared_error: 2.0372 - val_loss: 3.9302 - val_root_mean_squared_error: 1.9743
Epoch 150/500
5/5 [==============================] - 0s 7ms/step - loss: 4.8671 - root_mean_squared_error: 2.1996 - val_loss: 4.4557 - val_root_mean_squared_error: 2.1041
Epoch 151/500
5/5 [==============================] - 0s 8ms/step - loss: 4.7503 - root_mean_squared_error: 2.1735 - val_loss: 4.4462 - val_root_mean_squared_error: 2.1028
Epoch 152/500
5/5 [==============================] - 0s 8ms/step - loss: 4.3167 - root_mean_squared_error: 2.0710 - val_loss: 4.0744 - val_root_mean_squared_error: 2.0106
Epoch 153/500
5/5 [==============================] - 0s 7ms/step - loss: 4.5176 - root_mean_squared_error: 2.1178 - val_loss: 4.2886 - val_root_mean_squared_error: 2.0648
Epoch 154/500
5/5 [==============================] - 0s 8ms/step - loss: 4.0666 - root_mean_squared_error: 2.0097 - val_loss: 3.7240 - val_root_mean_squared_error: 1.9219
Epoch 155/500
5/5 [==============================] - 0s 7ms/step - loss: 3.8771 - root_mean_squared_error: 1.9604 - val_loss: 4.1124 - val_root_mean_squared_error: 2.0211
Epoch 156/500
5/5 [==============================] - 0s 7ms/step - loss: 4.1810 - root_mean_squared_error: 2.0379 - val_loss: 4.2141 - val_root_mean_squared_error: 2.0459
Epoch 157/500
5/5 [==============================] - 0s 8ms/step - loss: 4.0281 - root_mean_squared_error: 1.9989 - val_loss: 3.5129 - val_root_mean_squared_error: 1.8655
Epoch 158/500
5/5 [==============================] - 0s 8ms/step - loss: 3.4092 - root_mean_squared_error: 1.8393 - val_loss: 3.6255 - val_root_mean_squared_error: 1.8963
Epoch 159/500
5/5 [==============================] - 0s 7ms/step - loss: 3.3143 - root_mean_squared_error: 1.8121 - val_loss: 3.9437 - val_root_mean_squared_error: 1.9785
Epoch 160/500
5/5 [==============================] - 0s 9ms/step - loss: 3.3143 - root_mean_squared_error: 1.8120 - val_loss: 3.3045 - val_root_mean_squared_error: 1.8103
Epoch 161/500
5/5 [==============================] - 0s 7ms/step - loss: 3.7391 - root_mean_squared_error: 1.9263 - val_loss: 2.9063 - val_root_mean_squared_error: 1.6967
Epoch 162/500
5/5 [==============================] - 0s 7ms/step - loss: 3.3922 - root_mean_squared_error: 1.8323 - val_loss: 3.0792 - val_root_mean_squared_error: 1.7464
Epoch 163/500
5/5 [==============================] - 0s 8ms/step - loss: 3.2664 - root_mean_squared_error: 1.7990 - val_loss: 3.4557 - val_root_mean_squared_error: 1.8511
Epoch 164/500
5/5 [==============================] - 0s 7ms/step - loss: 3.1036 - root_mean_squared_error: 1.7535 - val_loss: 2.8895 - val_root_mean_squared_error: 1.6910
Epoch 165/500
5/5 [==============================] - 0s 7ms/step - loss: 3.0393 - root_mean_squared_error: 1.7342 - val_loss: 3.2136 - val_root_mean_squared_error: 1.7843
Epoch 166/500
5/5 [==============================] - 0s 8ms/step - loss: 2.6852 - root_mean_squared_error: 1.6282 - val_loss: 2.4742 - val_root_mean_squared_error: 1.5623
Epoch 167/500
5/5 [==============================] - 0s 7ms/step - loss: 3.2876 - root_mean_squared_error: 1.8062 - val_loss: 3.0855 - val_root_mean_squared_error: 1.7472
Epoch 168/500
5/5 [==============================] - 0s 7ms/step - loss: 3.1321 - root_mean_squared_error: 1.7609 - val_loss: 2.8482 - val_root_mean_squared_error: 1.6789
Epoch 169/500
5/5 [==============================] - 0s 8ms/step - loss: 2.6814 - root_mean_squared_error: 1.6280 - val_loss: 2.4223 - val_root_mean_squared_error: 1.5461
Epoch 170/500
5/5 [==============================] - 0s 8ms/step - loss: 2.9074 - root_mean_squared_error: 1.6956 - val_loss: 2.9819 - val_root_mean_squared_error: 1.7185
Epoch 171/500
5/5 [==============================] - 0s 7ms/step - loss: 2.5451 - root_mean_squared_error: 1.5864 - val_loss: 2.3025 - val_root_mean_squared_error: 1.5067
Epoch 172/500
5/5 [==============================] - 0s 8ms/step - loss: 2.5708 - root_mean_squared_error: 1.5933 - val_loss: 2.4211 - val_root_mean_squared_error: 1.5457
Epoch 173/500
5/5 [==============================] - 0s 7ms/step - loss: 2.4471 - root_mean_squared_error: 1.5547 - val_loss: 2.0389 - val_root_mean_squared_error: 1.4150
Epoch 174/500
5/5 [==============================] - 0s 7ms/step - loss: 2.2351 - root_mean_squared_error: 1.4843 - val_loss: 1.7456 - val_root_mean_squared_error: 1.3106
Epoch 175/500
5/5 [==============================] - 0s 8ms/step - loss: 2.0772 - root_mean_squared_error: 1.4309 - val_loss: 2.0395 - val_root_mean_squared_error: 1.4173
Epoch 176/500
5/5 [==============================] - 0s 7ms/step - loss: 2.0001 - root_mean_squared_error: 1.4017 - val_loss: 2.0027 - val_root_mean_squared_error: 1.4011
Epoch 177/500
5/5 [==============================] - 0s 7ms/step - loss: 2.3164 - root_mean_squared_error: 1.5125 - val_loss: 2.0450 - val_root_mean_squared_error: 1.4195
Epoch 178/500
5/5 [==============================] - 0s 8ms/step - loss: 1.9574 - root_mean_squared_error: 1.3878 - val_loss: 1.6294 - val_root_mean_squared_error: 1.2647
Epoch 179/500
5/5 [==============================] - 0s 7ms/step - loss: 1.6588 - root_mean_squared_error: 1.2763 - val_loss: 1.9774 - val_root_mean_squared_error: 1.3939
Epoch 180/500
5/5 [==============================] - 0s 7ms/step - loss: 1.7278 - root_mean_squared_error: 1.3024 - val_loss: 1.9368 - val_root_mean_squared_error: 1.3786
Epoch 181/500
5/5 [==============================] - 0s 8ms/step - loss: 1.9194 - root_mean_squared_error: 1.3742 - val_loss: 1.5600 - val_root_mean_squared_error: 1.2370
Epoch 182/500
5/5 [==============================] - 0s 7ms/step - loss: 1.7109 - root_mean_squared_error: 1.2971 - val_loss: 1.7722 - val_root_mean_squared_error: 1.3197
Epoch 183/500
5/5 [==============================] - 0s 7ms/step - loss: 1.7066 - root_mean_squared_error: 1.2949 - val_loss: 2.1218 - val_root_mean_squared_error: 1.4463
Epoch 184/500
5/5 [==============================] - 0s 8ms/step - loss: 1.6016 - root_mean_squared_error: 1.2511 - val_loss: 1.6546 - val_root_mean_squared_error: 1.2737
Epoch 185/500
5/5 [==============================] - 0s 7ms/step - loss: 1.3690 - root_mean_squared_error: 1.1555 - val_loss: 1.6435 - val_root_mean_squared_error: 1.2687
Epoch 186/500
5/5 [==============================] - 0s 7ms/step - loss: 1.5218 - root_mean_squared_error: 1.2202 - val_loss: 1.5625 - val_root_mean_squared_error: 1.2372
Epoch 187/500
5/5 [==============================] - 0s 8ms/step - loss: 1.8070 - root_mean_squared_error: 1.3323 - val_loss: 1.9401 - val_root_mean_squared_error: 1.3825
Epoch 188/500
5/5 [==============================] - 0s 8ms/step - loss: 1.3281 - root_mean_squared_error: 1.1407 - val_loss: 1.5691 - val_root_mean_squared_error: 1.2364
Epoch 189/500
5/5 [==============================] - 0s 8ms/step - loss: 1.3169 - root_mean_squared_error: 1.1330 - val_loss: 1.4483 - val_root_mean_squared_error: 1.1894
Epoch 190/500
5/5 [==============================] - 0s 9ms/step - loss: 1.4866 - root_mean_squared_error: 1.2057 - val_loss: 1.2282 - val_root_mean_squared_error: 1.0921
Epoch 191/500
5/5 [==============================] - 0s 7ms/step - loss: 1.3782 - root_mean_squared_error: 1.1596 - val_loss: 1.1190 - val_root_mean_squared_error: 1.0418
Epoch 192/500
5/5 [==============================] - 0s 7ms/step - loss: 1.4427 - root_mean_squared_error: 1.1890 - val_loss: 1.2618 - val_root_mean_squared_error: 1.1074
Epoch 193/500
5/5 [==============================] - 0s 8ms/step - loss: 1.0772 - root_mean_squared_error: 1.0200 - val_loss: 1.2445 - val_root_mean_squared_error: 1.1012
Epoch 194/500
5/5 [==============================] - 0s 7ms/step - loss: 1.2535 - root_mean_squared_error: 1.1045 - val_loss: 1.1080 - val_root_mean_squared_error: 1.0378
Epoch 195/500
5/5 [==============================] - 0s 8ms/step - loss: 0.9622 - root_mean_squared_error: 0.9648 - val_loss: 1.0185 - val_root_mean_squared_error: 0.9909
Epoch 196/500
5/5 [==============================] - 0s 8ms/step - loss: 1.1461 - root_mean_squared_error: 1.0549 - val_loss: 1.1735 - val_root_mean_squared_error: 1.0681
Epoch 197/500
5/5 [==============================] - 0s 8ms/step - loss: 1.0108 - root_mean_squared_error: 0.9887 - val_loss: 1.3042 - val_root_mean_squared_error: 1.1256
Epoch 198/500
5/5 [==============================] - 0s 8ms/step - loss: 1.4147 - root_mean_squared_error: 1.1770 - val_loss: 1.1107 - val_root_mean_squared_error: 1.0397
Epoch 199/500
5/5 [==============================] - 0s 8ms/step - loss: 0.9955 - root_mean_squared_error: 0.9802 - val_loss: 0.9723 - val_root_mean_squared_error: 0.9701
Epoch 200/500
5/5 [==============================] - 0s 7ms/step - loss: 1.0123 - root_mean_squared_error: 0.9891 - val_loss: 0.9592 - val_root_mean_squared_error: 0.9633
Epoch 201/500
5/5 [==============================] - 0s 8ms/step - loss: 1.1365 - root_mean_squared_error: 1.0492 - val_loss: 1.0563 - val_root_mean_squared_error: 1.0117
Epoch 202/500
5/5 [==============================] - 0s 9ms/step - loss: 0.9651 - root_mean_squared_error: 0.9654 - val_loss: 0.9475 - val_root_mean_squared_error: 0.9547
Epoch 203/500
5/5 [==============================] - 0s 8ms/step - loss: 1.0471 - root_mean_squared_error: 1.0070 - val_loss: 0.8681 - val_root_mean_squared_error: 0.9128
Epoch 204/500
5/5 [==============================] - 0s 7ms/step - loss: 0.9531 - root_mean_squared_error: 0.9573 - val_loss: 0.9093 - val_root_mean_squared_error: 0.9353
Epoch 205/500
5/5 [==============================] - 0s 10ms/step - loss: 0.9929 - root_mean_squared_error: 0.9787 - val_loss: 0.8707 - val_root_mean_squared_error: 0.9154
Epoch 206/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8833 - root_mean_squared_error: 0.9230 - val_loss: 0.9810 - val_root_mean_squared_error: 0.9752
Epoch 207/500
5/5 [==============================] - 0s 8ms/step - loss: 0.9496 - root_mean_squared_error: 0.9553 - val_loss: 0.8528 - val_root_mean_squared_error: 0.9065
Epoch 208/500
5/5 [==============================] - 0s 9ms/step - loss: 0.8704 - root_mean_squared_error: 0.9150 - val_loss: 0.9010 - val_root_mean_squared_error: 0.9318
Epoch 209/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8947 - root_mean_squared_error: 0.9289 - val_loss: 0.8612 - val_root_mean_squared_error: 0.9093
Epoch 210/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8210 - root_mean_squared_error: 0.8861 - val_loss: 0.9138 - val_root_mean_squared_error: 0.9377
Epoch 211/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8074 - root_mean_squared_error: 0.8815 - val_loss: 0.8727 - val_root_mean_squared_error: 0.9172
Epoch 212/500
5/5 [==============================] - 0s 7ms/step - loss: 0.9145 - root_mean_squared_error: 0.9378 - val_loss: 0.9894 - val_root_mean_squared_error: 0.9774
Epoch 213/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8471 - root_mean_squared_error: 0.9017 - val_loss: 0.8623 - val_root_mean_squared_error: 0.9138
Epoch 214/500
5/5 [==============================] - 0s 8ms/step - loss: 0.9194 - root_mean_squared_error: 0.9395 - val_loss: 0.9762 - val_root_mean_squared_error: 0.9724
Epoch 215/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8298 - root_mean_squared_error: 0.8936 - val_loss: 0.8450 - val_root_mean_squared_error: 0.9025
Epoch 216/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8298 - root_mean_squared_error: 0.8924 - val_loss: 0.8339 - val_root_mean_squared_error: 0.8936
Epoch 217/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7561 - root_mean_squared_error: 0.8492 - val_loss: 0.8833 - val_root_mean_squared_error: 0.9210
Epoch 218/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8694 - root_mean_squared_error: 0.9148 - val_loss: 0.8643 - val_root_mean_squared_error: 0.9122
Epoch 219/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8652 - root_mean_squared_error: 0.9131 - val_loss: 0.7927 - val_root_mean_squared_error: 0.8729
Epoch 220/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8154 - root_mean_squared_error: 0.8873 - val_loss: 0.8357 - val_root_mean_squared_error: 0.8967
Epoch 221/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8024 - root_mean_squared_error: 0.8784 - val_loss: 0.8347 - val_root_mean_squared_error: 0.8941
Epoch 222/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8150 - root_mean_squared_error: 0.8827 - val_loss: 0.8459 - val_root_mean_squared_error: 0.9005
Epoch 223/500
5/5 [==============================] - 0s 9ms/step - loss: 0.8222 - root_mean_squared_error: 0.8905 - val_loss: 0.7911 - val_root_mean_squared_error: 0.8723
Epoch 224/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7916 - root_mean_squared_error: 0.8715 - val_loss: 0.8076 - val_root_mean_squared_error: 0.8804
Epoch 225/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7810 - root_mean_squared_error: 0.8662 - val_loss: 0.8394 - val_root_mean_squared_error: 0.8949
Epoch 226/500
5/5 [==============================] - 0s 9ms/step - loss: 0.8835 - root_mean_squared_error: 0.9240 - val_loss: 0.8092 - val_root_mean_squared_error: 0.8840
Epoch 227/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7962 - root_mean_squared_error: 0.8707 - val_loss: 0.8084 - val_root_mean_squared_error: 0.8797
Epoch 228/500
5/5 [==============================] - 0s 6ms/step - loss: 0.7863 - root_mean_squared_error: 0.8674 - val_loss: 0.8203 - val_root_mean_squared_error: 0.8890
Epoch 229/500
5/5 [==============================] - 0s 6ms/step - loss: 0.8944 - root_mean_squared_error: 0.9271 - val_loss: 0.8871 - val_root_mean_squared_error: 0.9245
Epoch 230/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8957 - root_mean_squared_error: 0.9293 - val_loss: 0.7887 - val_root_mean_squared_error: 0.8714
Epoch 231/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8096 - root_mean_squared_error: 0.8814 - val_loss: 0.8407 - val_root_mean_squared_error: 0.8990
Epoch 232/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8129 - root_mean_squared_error: 0.8830 - val_loss: 0.8125 - val_root_mean_squared_error: 0.8837
Epoch 233/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7982 - root_mean_squared_error: 0.8752 - val_loss: 0.8250 - val_root_mean_squared_error: 0.8895
Epoch 234/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8536 - root_mean_squared_error: 0.9085 - val_loss: 0.9317 - val_root_mean_squared_error: 0.9495
Epoch 235/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8111 - root_mean_squared_error: 0.8849 - val_loss: 0.8122 - val_root_mean_squared_error: 0.8856
Epoch 236/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8089 - root_mean_squared_error: 0.8813 - val_loss: 0.9191 - val_root_mean_squared_error: 0.9414
Epoch 237/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7942 - root_mean_squared_error: 0.8741 - val_loss: 0.8237 - val_root_mean_squared_error: 0.8902
Epoch 238/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7509 - root_mean_squared_error: 0.8477 - val_loss: 0.8359 - val_root_mean_squared_error: 0.8979
Epoch 239/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8076 - root_mean_squared_error: 0.8812 - val_loss: 0.8295 - val_root_mean_squared_error: 0.8935
Epoch 240/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8114 - root_mean_squared_error: 0.8826 - val_loss: 0.8221 - val_root_mean_squared_error: 0.8910
Epoch 241/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8199 - root_mean_squared_error: 0.8887 - val_loss: 0.8368 - val_root_mean_squared_error: 0.8978
Epoch 242/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8133 - root_mean_squared_error: 0.8849 - val_loss: 0.8200 - val_root_mean_squared_error: 0.8910
Epoch 243/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8209 - root_mean_squared_error: 0.8883 - val_loss: 0.8213 - val_root_mean_squared_error: 0.8902
Epoch 244/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7531 - root_mean_squared_error: 0.8494 - val_loss: 0.7979 - val_root_mean_squared_error: 0.8770
Epoch 245/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7664 - root_mean_squared_error: 0.8559 - val_loss: 0.7947 - val_root_mean_squared_error: 0.8758
Epoch 246/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8474 - root_mean_squared_error: 0.9027 - val_loss: 0.8045 - val_root_mean_squared_error: 0.8808
Epoch 247/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7988 - root_mean_squared_error: 0.8765 - val_loss: 0.8464 - val_root_mean_squared_error: 0.9049
Epoch 248/500
5/5 [==============================] - 0s 9ms/step - loss: 0.8008 - root_mean_squared_error: 0.8761 - val_loss: 0.8179 - val_root_mean_squared_error: 0.8867
Epoch 249/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7784 - root_mean_squared_error: 0.8640 - val_loss: 0.7743 - val_root_mean_squared_error: 0.8622
Epoch 250/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7345 - root_mean_squared_error: 0.8394 - val_loss: 0.7821 - val_root_mean_squared_error: 0.8655
Epoch 251/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8320 - root_mean_squared_error: 0.8939 - val_loss: 0.7880 - val_root_mean_squared_error: 0.8706
Epoch 252/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7759 - root_mean_squared_error: 0.8635 - val_loss: 0.7971 - val_root_mean_squared_error: 0.8753
Epoch 253/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7642 - root_mean_squared_error: 0.8568 - val_loss: 0.8008 - val_root_mean_squared_error: 0.8789
Epoch 254/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8063 - root_mean_squared_error: 0.8822 - val_loss: 0.7557 - val_root_mean_squared_error: 0.8538
Epoch 255/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7863 - root_mean_squared_error: 0.8713 - val_loss: 0.8181 - val_root_mean_squared_error: 0.8887
Epoch 256/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8474 - root_mean_squared_error: 0.9033 - val_loss: 0.7677 - val_root_mean_squared_error: 0.8600
Epoch 257/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8019 - root_mean_squared_error: 0.8774 - val_loss: 0.8154 - val_root_mean_squared_error: 0.8870
Epoch 258/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7761 - root_mean_squared_error: 0.8646 - val_loss: 0.8126 - val_root_mean_squared_error: 0.8840
Epoch 259/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7731 - root_mean_squared_error: 0.8637 - val_loss: 0.7768 - val_root_mean_squared_error: 0.8665
Epoch 260/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8288 - root_mean_squared_error: 0.8928 - val_loss: 0.8168 - val_root_mean_squared_error: 0.8864
Epoch 261/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7911 - root_mean_squared_error: 0.8734 - val_loss: 0.7687 - val_root_mean_squared_error: 0.8602
Epoch 262/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7401 - root_mean_squared_error: 0.8443 - val_loss: 0.7805 - val_root_mean_squared_error: 0.8673
Epoch 263/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7512 - root_mean_squared_error: 0.8493 - val_loss: 0.7786 - val_root_mean_squared_error: 0.8681
Epoch 264/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7873 - root_mean_squared_error: 0.8696 - val_loss: 0.8144 - val_root_mean_squared_error: 0.8834
Epoch 265/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7971 - root_mean_squared_error: 0.8736 - val_loss: 0.7728 - val_root_mean_squared_error: 0.8627
Epoch 266/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7620 - root_mean_squared_error: 0.8557 - val_loss: 0.8114 - val_root_mean_squared_error: 0.8841
Epoch 267/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7506 - root_mean_squared_error: 0.8506 - val_loss: 0.7918 - val_root_mean_squared_error: 0.8737
Epoch 268/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7891 - root_mean_squared_error: 0.8732 - val_loss: 0.7868 - val_root_mean_squared_error: 0.8701
Epoch 269/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7890 - root_mean_squared_error: 0.8716 - val_loss: 0.7468 - val_root_mean_squared_error: 0.8460
Epoch 270/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7969 - root_mean_squared_error: 0.8765 - val_loss: 0.8166 - val_root_mean_squared_error: 0.8884
Epoch 271/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7869 - root_mean_squared_error: 0.8708 - val_loss: 0.7566 - val_root_mean_squared_error: 0.8515
Epoch 272/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7843 - root_mean_squared_error: 0.8694 - val_loss: 0.7692 - val_root_mean_squared_error: 0.8615
Epoch 273/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7406 - root_mean_squared_error: 0.8419 - val_loss: 0.7837 - val_root_mean_squared_error: 0.8719
Epoch 274/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7337 - root_mean_squared_error: 0.8402 - val_loss: 0.7663 - val_root_mean_squared_error: 0.8578
Epoch 275/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7252 - root_mean_squared_error: 0.8345 - val_loss: 0.7676 - val_root_mean_squared_error: 0.8598
Epoch 276/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7121 - root_mean_squared_error: 0.8264 - val_loss: 0.7744 - val_root_mean_squared_error: 0.8648
Epoch 277/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7825 - root_mean_squared_error: 0.8697 - val_loss: 0.7781 - val_root_mean_squared_error: 0.8672
Epoch 278/500
5/5 [==============================] - 0s 6ms/step - loss: 0.7872 - root_mean_squared_error: 0.8716 - val_loss: 0.7327 - val_root_mean_squared_error: 0.8400
Epoch 279/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7959 - root_mean_squared_error: 0.8750 - val_loss: 0.7501 - val_root_mean_squared_error: 0.8465
Epoch 280/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7957 - root_mean_squared_error: 0.8750 - val_loss: 0.7376 - val_root_mean_squared_error: 0.8423
Epoch 281/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7330 - root_mean_squared_error: 0.8390 - val_loss: 0.7472 - val_root_mean_squared_error: 0.8449
Epoch 282/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8238 - root_mean_squared_error: 0.8935 - val_loss: 0.8130 - val_root_mean_squared_error: 0.8849
Epoch 283/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8167 - root_mean_squared_error: 0.8855 - val_loss: 0.7906 - val_root_mean_squared_error: 0.8719
Epoch 284/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7826 - root_mean_squared_error: 0.8684 - val_loss: 0.7470 - val_root_mean_squared_error: 0.8458
Epoch 285/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7489 - root_mean_squared_error: 0.8509 - val_loss: 0.7678 - val_root_mean_squared_error: 0.8592
Epoch 286/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8022 - root_mean_squared_error: 0.8795 - val_loss: 0.7766 - val_root_mean_squared_error: 0.8647
Epoch 287/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8191 - root_mean_squared_error: 0.8883 - val_loss: 0.7484 - val_root_mean_squared_error: 0.8490
Epoch 288/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7670 - root_mean_squared_error: 0.8583 - val_loss: 0.7411 - val_root_mean_squared_error: 0.8470
Epoch 289/500
5/5 [==============================] - 0s 7ms/step - loss: 0.8169 - root_mean_squared_error: 0.8889 - val_loss: 0.7513 - val_root_mean_squared_error: 0.8529
Epoch 290/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7756 - root_mean_squared_error: 0.8655 - val_loss: 0.7123 - val_root_mean_squared_error: 0.8255
Epoch 291/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7638 - root_mean_squared_error: 0.8573 - val_loss: 0.8171 - val_root_mean_squared_error: 0.8880
Epoch 292/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7540 - root_mean_squared_error: 0.8519 - val_loss: 0.7485 - val_root_mean_squared_error: 0.8495
Epoch 293/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7940 - root_mean_squared_error: 0.8754 - val_loss: 0.7306 - val_root_mean_squared_error: 0.8389
Epoch 294/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7948 - root_mean_squared_error: 0.8745 - val_loss: 0.7733 - val_root_mean_squared_error: 0.8630
Epoch 295/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7208 - root_mean_squared_error: 0.8326 - val_loss: 0.7376 - val_root_mean_squared_error: 0.8430
Epoch 296/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7401 - root_mean_squared_error: 0.8460 - val_loss: 0.7499 - val_root_mean_squared_error: 0.8532
Epoch 297/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7035 - root_mean_squared_error: 0.8236 - val_loss: 0.7145 - val_root_mean_squared_error: 0.8299
Epoch 298/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7208 - root_mean_squared_error: 0.8346 - val_loss: 0.7509 - val_root_mean_squared_error: 0.8494
Epoch 299/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7560 - root_mean_squared_error: 0.8539 - val_loss: 0.7447 - val_root_mean_squared_error: 0.8464
Epoch 300/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7553 - root_mean_squared_error: 0.8528 - val_loss: 0.7489 - val_root_mean_squared_error: 0.8478
Epoch 301/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7750 - root_mean_squared_error: 0.8656 - val_loss: 0.7667 - val_root_mean_squared_error: 0.8587
Epoch 302/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7304 - root_mean_squared_error: 0.8373 - val_loss: 0.7564 - val_root_mean_squared_error: 0.8525
Epoch 303/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7718 - root_mean_squared_error: 0.8624 - val_loss: 0.7768 - val_root_mean_squared_error: 0.8673
Epoch 304/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6965 - root_mean_squared_error: 0.8156 - val_loss: 0.7437 - val_root_mean_squared_error: 0.8451
Epoch 305/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7086 - root_mean_squared_error: 0.8267 - val_loss: 0.7150 - val_root_mean_squared_error: 0.8285
Epoch 306/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7133 - root_mean_squared_error: 0.8267 - val_loss: 0.7051 - val_root_mean_squared_error: 0.8234
Epoch 307/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7249 - root_mean_squared_error: 0.8350 - val_loss: 0.7339 - val_root_mean_squared_error: 0.8389
Epoch 308/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7171 - root_mean_squared_error: 0.8294 - val_loss: 0.7413 - val_root_mean_squared_error: 0.8429
Epoch 309/500
5/5 [==============================] - 0s 8ms/step - loss: 0.8003 - root_mean_squared_error: 0.8806 - val_loss: 0.7712 - val_root_mean_squared_error: 0.8603
Epoch 310/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7788 - root_mean_squared_error: 0.8664 - val_loss: 0.7407 - val_root_mean_squared_error: 0.8418
Epoch 311/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7360 - root_mean_squared_error: 0.8411 - val_loss: 0.7340 - val_root_mean_squared_error: 0.8393
Epoch 312/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7292 - root_mean_squared_error: 0.8378 - val_loss: 0.7230 - val_root_mean_squared_error: 0.8359
Epoch 313/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7407 - root_mean_squared_error: 0.8455 - val_loss: 0.7880 - val_root_mean_squared_error: 0.8710
Epoch 314/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7556 - root_mean_squared_error: 0.8542 - val_loss: 0.7572 - val_root_mean_squared_error: 0.8533
Epoch 315/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7441 - root_mean_squared_error: 0.8469 - val_loss: 0.7923 - val_root_mean_squared_error: 0.8756
Epoch 316/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7124 - root_mean_squared_error: 0.8278 - val_loss: 0.7203 - val_root_mean_squared_error: 0.8310
Epoch 317/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7914 - root_mean_squared_error: 0.8753 - val_loss: 0.7160 - val_root_mean_squared_error: 0.8290
Epoch 318/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7438 - root_mean_squared_error: 0.8447 - val_loss: 0.7534 - val_root_mean_squared_error: 0.8498
Epoch 319/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6989 - root_mean_squared_error: 0.8187 - val_loss: 0.7101 - val_root_mean_squared_error: 0.8301
Epoch 320/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6864 - root_mean_squared_error: 0.8097 - val_loss: 0.6991 - val_root_mean_squared_error: 0.8201
Epoch 321/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6687 - root_mean_squared_error: 0.8010 - val_loss: 0.7088 - val_root_mean_squared_error: 0.8243
Epoch 322/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7642 - root_mean_squared_error: 0.8580 - val_loss: 0.7205 - val_root_mean_squared_error: 0.8337
Epoch 323/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7320 - root_mean_squared_error: 0.8400 - val_loss: 0.7678 - val_root_mean_squared_error: 0.8591
Epoch 324/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7363 - root_mean_squared_error: 0.8429 - val_loss: 0.7313 - val_root_mean_squared_error: 0.8362
Epoch 325/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7017 - root_mean_squared_error: 0.8202 - val_loss: 0.7115 - val_root_mean_squared_error: 0.8263
Epoch 326/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7147 - root_mean_squared_error: 0.8292 - val_loss: 0.7458 - val_root_mean_squared_error: 0.8461
Epoch 327/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7020 - root_mean_squared_error: 0.8203 - val_loss: 0.7090 - val_root_mean_squared_error: 0.8264
Epoch 328/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7295 - root_mean_squared_error: 0.8399 - val_loss: 0.7125 - val_root_mean_squared_error: 0.8258
Epoch 329/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6957 - root_mean_squared_error: 0.8162 - val_loss: 0.8002 - val_root_mean_squared_error: 0.8789
Epoch 330/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7047 - root_mean_squared_error: 0.8239 - val_loss: 0.7405 - val_root_mean_squared_error: 0.8446
Epoch 331/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7419 - root_mean_squared_error: 0.8437 - val_loss: 0.6980 - val_root_mean_squared_error: 0.8171
Epoch 332/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7251 - root_mean_squared_error: 0.8345 - val_loss: 0.7383 - val_root_mean_squared_error: 0.8403
Epoch 333/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7374 - root_mean_squared_error: 0.8401 - val_loss: 0.7114 - val_root_mean_squared_error: 0.8250
Epoch 334/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6987 - root_mean_squared_error: 0.8157 - val_loss: 0.8161 - val_root_mean_squared_error: 0.8877
Epoch 335/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6895 - root_mean_squared_error: 0.8123 - val_loss: 0.7491 - val_root_mean_squared_error: 0.8483
Epoch 336/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7274 - root_mean_squared_error: 0.8353 - val_loss: 0.7191 - val_root_mean_squared_error: 0.8304
Epoch 337/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6853 - root_mean_squared_error: 0.8086 - val_loss: 0.7156 - val_root_mean_squared_error: 0.8331
Epoch 338/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6991 - root_mean_squared_error: 0.8190 - val_loss: 0.7005 - val_root_mean_squared_error: 0.8194
Epoch 339/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6935 - root_mean_squared_error: 0.8162 - val_loss: 0.7180 - val_root_mean_squared_error: 0.8291
Epoch 340/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7260 - root_mean_squared_error: 0.8364 - val_loss: 0.7421 - val_root_mean_squared_error: 0.8448
Epoch 341/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6774 - root_mean_squared_error: 0.8050 - val_loss: 0.7157 - val_root_mean_squared_error: 0.8281
Epoch 342/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6683 - root_mean_squared_error: 0.8003 - val_loss: 0.7348 - val_root_mean_squared_error: 0.8409
Epoch 343/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6974 - root_mean_squared_error: 0.8175 - val_loss: 0.7008 - val_root_mean_squared_error: 0.8211
Epoch 344/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7632 - root_mean_squared_error: 0.8577 - val_loss: 0.7016 - val_root_mean_squared_error: 0.8201
Epoch 345/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7096 - root_mean_squared_error: 0.8249 - val_loss: 0.7031 - val_root_mean_squared_error: 0.8217
Epoch 346/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7927 - root_mean_squared_error: 0.8755 - val_loss: 0.6874 - val_root_mean_squared_error: 0.8131
Epoch 347/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7181 - root_mean_squared_error: 0.8315 - val_loss: 0.7176 - val_root_mean_squared_error: 0.8309
Epoch 348/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7624 - root_mean_squared_error: 0.8566 - val_loss: 0.6897 - val_root_mean_squared_error: 0.8135
Epoch 349/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7508 - root_mean_squared_error: 0.8495 - val_loss: 0.7204 - val_root_mean_squared_error: 0.8319
Epoch 350/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7126 - root_mean_squared_error: 0.8301 - val_loss: 0.7422 - val_root_mean_squared_error: 0.8464
Epoch 351/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7421 - root_mean_squared_error: 0.8446 - val_loss: 0.7397 - val_root_mean_squared_error: 0.8440
Epoch 352/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7830 - root_mean_squared_error: 0.8678 - val_loss: 0.7178 - val_root_mean_squared_error: 0.8287
Epoch 353/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6575 - root_mean_squared_error: 0.7940 - val_loss: 0.7094 - val_root_mean_squared_error: 0.8287
Epoch 354/500
5/5 [==============================] - 0s 6ms/step - loss: 0.6726 - root_mean_squared_error: 0.8022 - val_loss: 0.7503 - val_root_mean_squared_error: 0.8510
Epoch 355/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6854 - root_mean_squared_error: 0.8097 - val_loss: 0.6776 - val_root_mean_squared_error: 0.8077
Epoch 356/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7194 - root_mean_squared_error: 0.8308 - val_loss: 0.6715 - val_root_mean_squared_error: 0.8018
Epoch 357/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7261 - root_mean_squared_error: 0.8366 - val_loss: 0.6878 - val_root_mean_squared_error: 0.8119
Epoch 358/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7302 - root_mean_squared_error: 0.8394 - val_loss: 0.7274 - val_root_mean_squared_error: 0.8370
Epoch 359/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6644 - root_mean_squared_error: 0.7983 - val_loss: 0.7131 - val_root_mean_squared_error: 0.8253
Epoch 360/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7406 - root_mean_squared_error: 0.8436 - val_loss: 0.7442 - val_root_mean_squared_error: 0.8479
Epoch 361/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7089 - root_mean_squared_error: 0.8258 - val_loss: 0.7080 - val_root_mean_squared_error: 0.8235
Epoch 362/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7021 - root_mean_squared_error: 0.8213 - val_loss: 0.6935 - val_root_mean_squared_error: 0.8172
Epoch 363/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6944 - root_mean_squared_error: 0.8172 - val_loss: 0.7031 - val_root_mean_squared_error: 0.8228
Epoch 364/500
5/5 [==============================] - 0s 6ms/step - loss: 0.6872 - root_mean_squared_error: 0.8110 - val_loss: 0.7183 - val_root_mean_squared_error: 0.8327
Epoch 365/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6893 - root_mean_squared_error: 0.8144 - val_loss: 0.7024 - val_root_mean_squared_error: 0.8210
Epoch 366/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6865 - root_mean_squared_error: 0.8133 - val_loss: 0.6702 - val_root_mean_squared_error: 0.8052
Epoch 367/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7070 - root_mean_squared_error: 0.8247 - val_loss: 0.6828 - val_root_mean_squared_error: 0.8079
Epoch 368/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7585 - root_mean_squared_error: 0.8560 - val_loss: 0.7130 - val_root_mean_squared_error: 0.8304
Epoch 369/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7285 - root_mean_squared_error: 0.8393 - val_loss: 0.6798 - val_root_mean_squared_error: 0.8100
Epoch 370/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7143 - root_mean_squared_error: 0.8276 - val_loss: 0.6982 - val_root_mean_squared_error: 0.8172
Epoch 371/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6500 - root_mean_squared_error: 0.7893 - val_loss: 0.7203 - val_root_mean_squared_error: 0.8322
Epoch 372/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7337 - root_mean_squared_error: 0.8389 - val_loss: 0.6784 - val_root_mean_squared_error: 0.8065
Epoch 373/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6927 - root_mean_squared_error: 0.8141 - val_loss: 0.6716 - val_root_mean_squared_error: 0.8050
Epoch 374/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6775 - root_mean_squared_error: 0.8060 - val_loss: 0.6861 - val_root_mean_squared_error: 0.8110
Epoch 375/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7299 - root_mean_squared_error: 0.8378 - val_loss: 0.6821 - val_root_mean_squared_error: 0.8096
Epoch 376/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6939 - root_mean_squared_error: 0.8170 - val_loss: 0.6946 - val_root_mean_squared_error: 0.8152
Epoch 377/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6995 - root_mean_squared_error: 0.8184 - val_loss: 0.7201 - val_root_mean_squared_error: 0.8329
Epoch 378/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6919 - root_mean_squared_error: 0.8162 - val_loss: 0.6602 - val_root_mean_squared_error: 0.7944
Epoch 379/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6976 - root_mean_squared_error: 0.8203 - val_loss: 0.6527 - val_root_mean_squared_error: 0.7895
Epoch 380/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7642 - root_mean_squared_error: 0.8576 - val_loss: 0.7263 - val_root_mean_squared_error: 0.8363
Epoch 381/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6331 - root_mean_squared_error: 0.7774 - val_loss: 0.7265 - val_root_mean_squared_error: 0.8389
Epoch 382/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6758 - root_mean_squared_error: 0.8044 - val_loss: 0.6534 - val_root_mean_squared_error: 0.7887
Epoch 383/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7618 - root_mean_squared_error: 0.8569 - val_loss: 0.6716 - val_root_mean_squared_error: 0.8009
Epoch 384/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7189 - root_mean_squared_error: 0.8312 - val_loss: 0.7329 - val_root_mean_squared_error: 0.8405
Epoch 385/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7034 - root_mean_squared_error: 0.8207 - val_loss: 0.6868 - val_root_mean_squared_error: 0.8121
Epoch 386/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7017 - root_mean_squared_error: 0.8217 - val_loss: 0.7018 - val_root_mean_squared_error: 0.8192
Epoch 387/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7023 - root_mean_squared_error: 0.8205 - val_loss: 0.7149 - val_root_mean_squared_error: 0.8288
Epoch 388/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6908 - root_mean_squared_error: 0.8148 - val_loss: 0.7186 - val_root_mean_squared_error: 0.8300
Epoch 389/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6225 - root_mean_squared_error: 0.7731 - val_loss: 0.7360 - val_root_mean_squared_error: 0.8412
Epoch 390/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6956 - root_mean_squared_error: 0.8188 - val_loss: 0.7197 - val_root_mean_squared_error: 0.8355
Epoch 391/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7217 - root_mean_squared_error: 0.8342 - val_loss: 0.6622 - val_root_mean_squared_error: 0.7957
Epoch 392/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6877 - root_mean_squared_error: 0.8119 - val_loss: 0.6844 - val_root_mean_squared_error: 0.8114
Epoch 393/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6620 - root_mean_squared_error: 0.7972 - val_loss: 0.6726 - val_root_mean_squared_error: 0.8039
Epoch 394/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6957 - root_mean_squared_error: 0.8177 - val_loss: 0.6664 - val_root_mean_squared_error: 0.7972
Epoch 395/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7091 - root_mean_squared_error: 0.8266 - val_loss: 0.7088 - val_root_mean_squared_error: 0.8258
Epoch 396/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7130 - root_mean_squared_error: 0.8286 - val_loss: 0.7436 - val_root_mean_squared_error: 0.8464
Epoch 397/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6527 - root_mean_squared_error: 0.7888 - val_loss: 0.7029 - val_root_mean_squared_error: 0.8221
Epoch 398/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6792 - root_mean_squared_error: 0.8062 - val_loss: 0.6859 - val_root_mean_squared_error: 0.8107
Epoch 399/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6944 - root_mean_squared_error: 0.8155 - val_loss: 0.7171 - val_root_mean_squared_error: 0.8314
Epoch 400/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6460 - root_mean_squared_error: 0.7868 - val_loss: 0.7132 - val_root_mean_squared_error: 0.8270
Epoch 401/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7219 - root_mean_squared_error: 0.8346 - val_loss: 0.6635 - val_root_mean_squared_error: 0.7984
Epoch 402/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7167 - root_mean_squared_error: 0.8296 - val_loss: 0.6990 - val_root_mean_squared_error: 0.8182
Epoch 403/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7334 - root_mean_squared_error: 0.8398 - val_loss: 0.7469 - val_root_mean_squared_error: 0.8483
Epoch 404/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6651 - root_mean_squared_error: 0.7970 - val_loss: 0.6819 - val_root_mean_squared_error: 0.8108
Epoch 405/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7258 - root_mean_squared_error: 0.8342 - val_loss: 0.7053 - val_root_mean_squared_error: 0.8235
Epoch 406/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6943 - root_mean_squared_error: 0.8151 - val_loss: 0.7223 - val_root_mean_squared_error: 0.8327
Epoch 407/500
5/5 [==============================] - 0s 6ms/step - loss: 0.6978 - root_mean_squared_error: 0.8170 - val_loss: 0.6857 - val_root_mean_squared_error: 0.8094
Epoch 408/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7000 - root_mean_squared_error: 0.8206 - val_loss: 0.7152 - val_root_mean_squared_error: 0.8290
Epoch 409/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7116 - root_mean_squared_error: 0.8271 - val_loss: 0.6564 - val_root_mean_squared_error: 0.7936
Epoch 410/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7864 - root_mean_squared_error: 0.8719 - val_loss: 0.6866 - val_root_mean_squared_error: 0.8119
Epoch 411/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7114 - root_mean_squared_error: 0.8257 - val_loss: 0.7374 - val_root_mean_squared_error: 0.8426
Epoch 412/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6788 - root_mean_squared_error: 0.8073 - val_loss: 0.7052 - val_root_mean_squared_error: 0.8212
Epoch 413/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6538 - root_mean_squared_error: 0.7887 - val_loss: 0.7179 - val_root_mean_squared_error: 0.8291
Epoch 414/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7083 - root_mean_squared_error: 0.8251 - val_loss: 0.6630 - val_root_mean_squared_error: 0.7960
Epoch 415/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6994 - root_mean_squared_error: 0.8203 - val_loss: 0.6983 - val_root_mean_squared_error: 0.8165
Epoch 416/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6767 - root_mean_squared_error: 0.8071 - val_loss: 0.6393 - val_root_mean_squared_error: 0.7818
Epoch 417/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6909 - root_mean_squared_error: 0.8148 - val_loss: 0.6858 - val_root_mean_squared_error: 0.8115
Epoch 418/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7318 - root_mean_squared_error: 0.8393 - val_loss: 0.7354 - val_root_mean_squared_error: 0.8417
Epoch 419/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7075 - root_mean_squared_error: 0.8237 - val_loss: 0.6553 - val_root_mean_squared_error: 0.7922
Epoch 420/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6756 - root_mean_squared_error: 0.8077 - val_loss: 0.6678 - val_root_mean_squared_error: 0.7993
Epoch 421/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7111 - root_mean_squared_error: 0.8277 - val_loss: 0.6789 - val_root_mean_squared_error: 0.8080
Epoch 422/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6938 - root_mean_squared_error: 0.8164 - val_loss: 0.6902 - val_root_mean_squared_error: 0.8167
Epoch 423/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6903 - root_mean_squared_error: 0.8133 - val_loss: 0.6777 - val_root_mean_squared_error: 0.8081
Epoch 424/500
5/5 [==============================] - 0s 6ms/step - loss: 0.6586 - root_mean_squared_error: 0.7940 - val_loss: 0.6836 - val_root_mean_squared_error: 0.8091
Epoch 425/500
5/5 [==============================] - 0s 9ms/step - loss: 0.7230 - root_mean_squared_error: 0.8326 - val_loss: 0.6586 - val_root_mean_squared_error: 0.7934
Epoch 426/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6808 - root_mean_squared_error: 0.8074 - val_loss: 0.6630 - val_root_mean_squared_error: 0.7986
Epoch 427/500
5/5 [==============================] - 0s 10ms/step - loss: 0.7123 - root_mean_squared_error: 0.8266 - val_loss: 0.6728 - val_root_mean_squared_error: 0.8024
Epoch 428/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7027 - root_mean_squared_error: 0.8223 - val_loss: 0.6857 - val_root_mean_squared_error: 0.8104
Epoch 429/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6651 - root_mean_squared_error: 0.7968 - val_loss: 0.6946 - val_root_mean_squared_error: 0.8163
Epoch 430/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7096 - root_mean_squared_error: 0.8267 - val_loss: 0.6563 - val_root_mean_squared_error: 0.7931
Epoch 431/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6854 - root_mean_squared_error: 0.8114 - val_loss: 0.6559 - val_root_mean_squared_error: 0.7944
Epoch 432/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6460 - root_mean_squared_error: 0.7864 - val_loss: 0.6494 - val_root_mean_squared_error: 0.7902
Epoch 433/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6171 - root_mean_squared_error: 0.7664 - val_loss: 0.7009 - val_root_mean_squared_error: 0.8199
Epoch 434/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6444 - root_mean_squared_error: 0.7850 - val_loss: 0.6582 - val_root_mean_squared_error: 0.7944
Epoch 435/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6725 - root_mean_squared_error: 0.8020 - val_loss: 0.6449 - val_root_mean_squared_error: 0.7853
Epoch 436/500
5/5 [==============================] - 0s 10ms/step - loss: 0.7071 - root_mean_squared_error: 0.8226 - val_loss: 0.6765 - val_root_mean_squared_error: 0.8049
Epoch 437/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6530 - root_mean_squared_error: 0.7900 - val_loss: 0.6650 - val_root_mean_squared_error: 0.7985
Epoch 438/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6936 - root_mean_squared_error: 0.8153 - val_loss: 0.7160 - val_root_mean_squared_error: 0.8270
Epoch 439/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6925 - root_mean_squared_error: 0.8143 - val_loss: 0.6482 - val_root_mean_squared_error: 0.7874
Epoch 440/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6541 - root_mean_squared_error: 0.7894 - val_loss: 0.6750 - val_root_mean_squared_error: 0.8040
Epoch 441/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7203 - root_mean_squared_error: 0.8340 - val_loss: 0.6939 - val_root_mean_squared_error: 0.8151
Epoch 442/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6744 - root_mean_squared_error: 0.8038 - val_loss: 0.6813 - val_root_mean_squared_error: 0.8068
Epoch 443/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6744 - root_mean_squared_error: 0.8043 - val_loss: 0.7422 - val_root_mean_squared_error: 0.8458
Epoch 444/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6595 - root_mean_squared_error: 0.7923 - val_loss: 0.6674 - val_root_mean_squared_error: 0.7988
Epoch 445/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6203 - root_mean_squared_error: 0.7679 - val_loss: 0.7034 - val_root_mean_squared_error: 0.8213
Epoch 446/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6711 - root_mean_squared_error: 0.8013 - val_loss: 0.7416 - val_root_mean_squared_error: 0.8472
Epoch 447/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7121 - root_mean_squared_error: 0.8288 - val_loss: 0.6672 - val_root_mean_squared_error: 0.8005
Epoch 448/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6744 - root_mean_squared_error: 0.8025 - val_loss: 0.6686 - val_root_mean_squared_error: 0.7992
Epoch 449/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6177 - root_mean_squared_error: 0.7671 - val_loss: 0.6627 - val_root_mean_squared_error: 0.7980
Epoch 450/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6364 - root_mean_squared_error: 0.7789 - val_loss: 0.7457 - val_root_mean_squared_error: 0.8469
Epoch 451/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7037 - root_mean_squared_error: 0.8227 - val_loss: 0.7475 - val_root_mean_squared_error: 0.8485
Epoch 452/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6681 - root_mean_squared_error: 0.8001 - val_loss: 0.6609 - val_root_mean_squared_error: 0.7987
Epoch 453/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6773 - root_mean_squared_error: 0.8078 - val_loss: 0.7432 - val_root_mean_squared_error: 0.8482
Epoch 454/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7262 - root_mean_squared_error: 0.8372 - val_loss: 0.6497 - val_root_mean_squared_error: 0.7865
Epoch 455/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7066 - root_mean_squared_error: 0.8226 - val_loss: 0.6469 - val_root_mean_squared_error: 0.7877
Epoch 456/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6499 - root_mean_squared_error: 0.7885 - val_loss: 0.6596 - val_root_mean_squared_error: 0.7932
Epoch 457/500
5/5 [==============================] - 0s 6ms/step - loss: 0.6497 - root_mean_squared_error: 0.7892 - val_loss: 0.6502 - val_root_mean_squared_error: 0.7875
Epoch 458/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6492 - root_mean_squared_error: 0.7881 - val_loss: 0.6428 - val_root_mean_squared_error: 0.7829
Epoch 459/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6401 - root_mean_squared_error: 0.7840 - val_loss: 0.6679 - val_root_mean_squared_error: 0.8010
Epoch 460/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7352 - root_mean_squared_error: 0.8421 - val_loss: 0.6870 - val_root_mean_squared_error: 0.8104
Epoch 461/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6759 - root_mean_squared_error: 0.8034 - val_loss: 0.6756 - val_root_mean_squared_error: 0.8059
Epoch 462/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7137 - root_mean_squared_error: 0.8284 - val_loss: 0.6897 - val_root_mean_squared_error: 0.8139
Epoch 463/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6505 - root_mean_squared_error: 0.7886 - val_loss: 0.6374 - val_root_mean_squared_error: 0.7787
Epoch 464/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6830 - root_mean_squared_error: 0.8085 - val_loss: 0.6516 - val_root_mean_squared_error: 0.7869
Epoch 465/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6938 - root_mean_squared_error: 0.8177 - val_loss: 0.6871 - val_root_mean_squared_error: 0.8117
Epoch 466/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6159 - root_mean_squared_error: 0.7661 - val_loss: 0.6832 - val_root_mean_squared_error: 0.8109
Epoch 467/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6758 - root_mean_squared_error: 0.8051 - val_loss: 0.6680 - val_root_mean_squared_error: 0.7987
Epoch 468/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7117 - root_mean_squared_error: 0.8275 - val_loss: 0.6832 - val_root_mean_squared_error: 0.8083
Epoch 469/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6822 - root_mean_squared_error: 0.8090 - val_loss: 0.6550 - val_root_mean_squared_error: 0.7927
Epoch 470/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6880 - root_mean_squared_error: 0.8115 - val_loss: 0.6446 - val_root_mean_squared_error: 0.7867
Epoch 471/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6507 - root_mean_squared_error: 0.7890 - val_loss: 0.6768 - val_root_mean_squared_error: 0.8041
Epoch 472/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6368 - root_mean_squared_error: 0.7797 - val_loss: 0.6621 - val_root_mean_squared_error: 0.7961
Epoch 473/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6495 - root_mean_squared_error: 0.7872 - val_loss: 0.6588 - val_root_mean_squared_error: 0.7951
Epoch 474/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6838 - root_mean_squared_error: 0.8091 - val_loss: 0.6736 - val_root_mean_squared_error: 0.8030
Epoch 475/500
5/5 [==============================] - 0s 8ms/step - loss: 0.7204 - root_mean_squared_error: 0.8347 - val_loss: 0.6888 - val_root_mean_squared_error: 0.8151
Epoch 476/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7249 - root_mean_squared_error: 0.8357 - val_loss: 0.6436 - val_root_mean_squared_error: 0.7846
Epoch 477/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6700 - root_mean_squared_error: 0.8017 - val_loss: 0.6429 - val_root_mean_squared_error: 0.7863
Epoch 478/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6669 - root_mean_squared_error: 0.7963 - val_loss: 0.6564 - val_root_mean_squared_error: 0.7928
Epoch 479/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6610 - root_mean_squared_error: 0.7965 - val_loss: 0.6655 - val_root_mean_squared_error: 0.7966
Epoch 480/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6351 - root_mean_squared_error: 0.7802 - val_loss: 0.6433 - val_root_mean_squared_error: 0.7840
Epoch 481/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7008 - root_mean_squared_error: 0.8180 - val_loss: 0.6826 - val_root_mean_squared_error: 0.8088
Epoch 482/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6522 - root_mean_squared_error: 0.7897 - val_loss: 0.6533 - val_root_mean_squared_error: 0.7915
Epoch 483/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6808 - root_mean_squared_error: 0.8084 - val_loss: 0.6606 - val_root_mean_squared_error: 0.7957
Epoch 484/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6618 - root_mean_squared_error: 0.7938 - val_loss: 0.6640 - val_root_mean_squared_error: 0.8012
Epoch 485/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6664 - root_mean_squared_error: 0.7982 - val_loss: 0.6765 - val_root_mean_squared_error: 0.8058
Epoch 486/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6842 - root_mean_squared_error: 0.8075 - val_loss: 0.6386 - val_root_mean_squared_error: 0.7818
Epoch 487/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6382 - root_mean_squared_error: 0.7803 - val_loss: 0.6568 - val_root_mean_squared_error: 0.7925
Epoch 488/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6911 - root_mean_squared_error: 0.8153 - val_loss: 0.6552 - val_root_mean_squared_error: 0.7910
Epoch 489/500
5/5 [==============================] - 0s 7ms/step - loss: 0.7275 - root_mean_squared_error: 0.8359 - val_loss: 0.6533 - val_root_mean_squared_error: 0.7880
Epoch 490/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6877 - root_mean_squared_error: 0.8131 - val_loss: 0.6505 - val_root_mean_squared_error: 0.7900
Epoch 491/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6579 - root_mean_squared_error: 0.7913 - val_loss: 0.6722 - val_root_mean_squared_error: 0.8071
Epoch 492/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6672 - root_mean_squared_error: 0.7979 - val_loss: 0.6780 - val_root_mean_squared_error: 0.8060
Epoch 493/500
5/5 [==============================] - 0s 9ms/step - loss: 0.6595 - root_mean_squared_error: 0.7952 - val_loss: 0.6847 - val_root_mean_squared_error: 0.8120
Epoch 494/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6801 - root_mean_squared_error: 0.8077 - val_loss: 0.6472 - val_root_mean_squared_error: 0.7863
Epoch 495/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6960 - root_mean_squared_error: 0.8181 - val_loss: 0.6890 - val_root_mean_squared_error: 0.8126
Epoch 496/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6788 - root_mean_squared_error: 0.8059 - val_loss: 0.6458 - val_root_mean_squared_error: 0.7856
Epoch 497/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6689 - root_mean_squared_error: 0.7997 - val_loss: 0.6553 - val_root_mean_squared_error: 0.7921
Epoch 498/500
5/5 [==============================] - 0s 7ms/step - loss: 0.6714 - root_mean_squared_error: 0.8021 - val_loss: 0.6629 - val_root_mean_squared_error: 0.7974
Epoch 499/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6612 - root_mean_squared_error: 0.7957 - val_loss: 0.6555 - val_root_mean_squared_error: 0.7953
Epoch 500/500
5/5 [==============================] - 0s 8ms/step - loss: 0.6925 - root_mean_squared_error: 0.8170 - val_loss: 0.6365 - val_root_mean_squared_error: 0.7787
Model training finished.
Train RMSE: 0.796
Evaluating model performance...
Test RMSE: 0.82
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
Predictions mean: 6.11, min: 5.58, max: 6.36, range: 0.78 - Actual: 6.0
Predictions mean: 5.25, min: 4.45, max: 5.99, range: 1.54 - Actual: 4.0
Predictions mean: 5.97, min: 5.38, max: 6.29, range: 0.92 - Actual: 5.0
Predictions mean: 6.29, min: 6.01, max: 6.44, range: 0.44 - Actual: 7.0
Predictions mean: 6.0, min: 5.41, max: 6.34, range: 0.93 - Actual: 5.0
Predictions mean: 6.14, min: 5.59, max: 6.39, range: 0.8 - Actual: 6.0
Predictions mean: 5.68, min: 4.94, max: 6.17, range: 1.23 - Actual: 6.0
Predictions mean: 6.3, min: 5.9, max: 6.45, range: 0.55 - Actual: 6.0
Predictions mean: 6.2, min: 5.75, max: 6.45, range: 0.71 - Actual: 7.0
Predictions mean: 6.31, min: 5.94, max: 6.47, range: 0.52 - Actual: 6.0
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
17/17 [==============================] - 1s 22ms/step - loss: 28.5738 - root_mean_squared_error: 5.3451 - val_loss: 27.6907 - val_root_mean_squared_error: 5.2617
Epoch 2/500
17/17 [==============================] - 0s 3ms/step - loss: 25.5625 - root_mean_squared_error: 5.0555 - val_loss: 25.2499 - val_root_mean_squared_error: 5.0245
Epoch 3/500
17/17 [==============================] - 0s 3ms/step - loss: 24.7044 - root_mean_squared_error: 4.9699 - val_loss: 26.6671 - val_root_mean_squared_error: 5.1636
Epoch 4/500
17/17 [==============================] - 0s 3ms/step - loss: 24.4043 - root_mean_squared_error: 4.9396 - val_loss: 19.0605 - val_root_mean_squared_error: 4.3652
Epoch 5/500
17/17 [==============================] - 0s 3ms/step - loss: 24.1436 - root_mean_squared_error: 4.9132 - val_loss: 18.3716 - val_root_mean_squared_error: 4.2857
Epoch 6/500
17/17 [==============================] - 0s 2ms/step - loss: 21.0608 - root_mean_squared_error: 4.5887 - val_loss: 19.0667 - val_root_mean_squared_error: 4.3658
Epoch 7/500
17/17 [==============================] - 0s 3ms/step - loss: 20.3242 - root_mean_squared_error: 4.5077 - val_loss: 25.5117 - val_root_mean_squared_error: 5.0505
Epoch 8/500
17/17 [==============================] - 0s 3ms/step - loss: 18.4351 - root_mean_squared_error: 4.2931 - val_loss: 16.8163 - val_root_mean_squared_error: 4.1002
Epoch 9/500
17/17 [==============================] - 0s 3ms/step - loss: 17.7227 - root_mean_squared_error: 4.2093 - val_loss: 16.5683 - val_root_mean_squared_error: 4.0698
Epoch 10/500
17/17 [==============================] - 0s 3ms/step - loss: 17.2048 - root_mean_squared_error: 4.1473 - val_loss: 18.4072 - val_root_mean_squared_error: 4.2898
Epoch 11/500
17/17 [==============================] - 0s 3ms/step - loss: 15.5533 - root_mean_squared_error: 3.9432 - val_loss: 14.8016 - val_root_mean_squared_error: 3.8466
Epoch 12/500
17/17 [==============================] - 0s 3ms/step - loss: 14.9682 - root_mean_squared_error: 3.8682 - val_loss: 13.7221 - val_root_mean_squared_error: 3.7037
Epoch 13/500
17/17 [==============================] - 0s 3ms/step - loss: 14.0424 - root_mean_squared_error: 3.7466 - val_loss: 14.4401 - val_root_mean_squared_error: 3.7995
Epoch 14/500
17/17 [==============================] - 0s 3ms/step - loss: 13.1498 - root_mean_squared_error: 3.6255 - val_loss: 11.4653 - val_root_mean_squared_error: 3.3853
Epoch 15/500
17/17 [==============================] - 0s 3ms/step - loss: 11.5501 - root_mean_squared_error: 3.3978 - val_loss: 11.7298 - val_root_mean_squared_error: 3.4241
Epoch 16/500
17/17 [==============================] - 0s 3ms/step - loss: 11.4142 - root_mean_squared_error: 3.3777 - val_loss: 11.1638 - val_root_mean_squared_error: 3.3404
Epoch 17/500
17/17 [==============================] - 0s 3ms/step - loss: 10.4584 - root_mean_squared_error: 3.2332 - val_loss: 10.1527 - val_root_mean_squared_error: 3.1855
Epoch 18/500
17/17 [==============================] - 0s 3ms/step - loss: 9.9638 - root_mean_squared_error: 3.1556 - val_loss: 10.8206 - val_root_mean_squared_error: 3.2888
Epoch 19/500
17/17 [==============================] - 0s 3ms/step - loss: 9.5702 - root_mean_squared_error: 3.0927 - val_loss: 8.7598 - val_root_mean_squared_error: 2.9587
Epoch 20/500
17/17 [==============================] - 0s 3ms/step - loss: 8.4816 - root_mean_squared_error: 2.9114 - val_loss: 6.6505 - val_root_mean_squared_error: 2.5778
Epoch 21/500
17/17 [==============================] - 0s 3ms/step - loss: 8.0186 - root_mean_squared_error: 2.8308 - val_loss: 6.2340 - val_root_mean_squared_error: 2.4956
Epoch 22/500
17/17 [==============================] - 0s 3ms/step - loss: 7.5044 - root_mean_squared_error: 2.7385 - val_loss: 7.0815 - val_root_mean_squared_error: 2.6599
Epoch 23/500
17/17 [==============================] - 0s 3ms/step - loss: 6.4198 - root_mean_squared_error: 2.5327 - val_loss: 6.1665 - val_root_mean_squared_error: 2.4820
Epoch 24/500
17/17 [==============================] - 0s 3ms/step - loss: 6.6303 - root_mean_squared_error: 2.5738 - val_loss: 4.6622 - val_root_mean_squared_error: 2.1578
Epoch 25/500
17/17 [==============================] - 0s 3ms/step - loss: 5.5424 - root_mean_squared_error: 2.3531 - val_loss: 4.8293 - val_root_mean_squared_error: 2.1965
Epoch 26/500
17/17 [==============================] - 0s 2ms/step - loss: 5.3318 - root_mean_squared_error: 2.3079 - val_loss: 4.7150 - val_root_mean_squared_error: 2.1700
Epoch 27/500
17/17 [==============================] - 0s 3ms/step - loss: 3.8897 - root_mean_squared_error: 1.9708 - val_loss: 2.8917 - val_root_mean_squared_error: 1.6988
Epoch 28/500
17/17 [==============================] - 0s 3ms/step - loss: 3.8182 - root_mean_squared_error: 1.9525 - val_loss: 2.9499 - val_root_mean_squared_error: 1.7155
Epoch 29/500
17/17 [==============================] - 0s 3ms/step - loss: 3.3346 - root_mean_squared_error: 1.8244 - val_loss: 3.5701 - val_root_mean_squared_error: 1.8876
Epoch 30/500
17/17 [==============================] - 0s 3ms/step - loss: 2.5465 - root_mean_squared_error: 1.5937 - val_loss: 2.2015 - val_root_mean_squared_error: 1.4812
Epoch 31/500
17/17 [==============================] - 0s 2ms/step - loss: 2.4023 - root_mean_squared_error: 1.5481 - val_loss: 2.6177 - val_root_mean_squared_error: 1.6161
Epoch 32/500
17/17 [==============================] - 0s 3ms/step - loss: 2.2647 - root_mean_squared_error: 1.5028 - val_loss: 1.6816 - val_root_mean_squared_error: 1.2943
Epoch 33/500
17/17 [==============================] - 0s 3ms/step - loss: 1.7956 - root_mean_squared_error: 1.3375 - val_loss: 1.5518 - val_root_mean_squared_error: 1.2433
Epoch 34/500
17/17 [==============================] - 0s 3ms/step - loss: 1.2435 - root_mean_squared_error: 1.1119 - val_loss: 1.1264 - val_root_mean_squared_error: 1.0579
Epoch 35/500
17/17 [==============================] - 0s 2ms/step - loss: 1.1736 - root_mean_squared_error: 1.0800 - val_loss: 1.1706 - val_root_mean_squared_error: 1.0790
Epoch 36/500
17/17 [==============================] - 0s 2ms/step - loss: 1.0909 - root_mean_squared_error: 1.0411 - val_loss: 1.0823 - val_root_mean_squared_error: 1.0375
Epoch 37/500
17/17 [==============================] - 0s 3ms/step - loss: 0.9696 - root_mean_squared_error: 0.9809 - val_loss: 0.8075 - val_root_mean_squared_error: 0.8949
Epoch 38/500
17/17 [==============================] - 0s 3ms/step - loss: 1.0157 - root_mean_squared_error: 1.0044 - val_loss: 0.9420 - val_root_mean_squared_error: 0.9672
Epoch 39/500
17/17 [==============================] - 0s 3ms/step - loss: 0.8322 - root_mean_squared_error: 0.9082 - val_loss: 0.8628 - val_root_mean_squared_error: 0.9240
Epoch 40/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7929 - root_mean_squared_error: 0.8860 - val_loss: 0.8987 - val_root_mean_squared_error: 0.9436
Epoch 41/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7980 - root_mean_squared_error: 0.8893 - val_loss: 0.7314 - val_root_mean_squared_error: 0.8515
Epoch 42/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7881 - root_mean_squared_error: 0.8836 - val_loss: 0.8106 - val_root_mean_squared_error: 0.8962
Epoch 43/500
17/17 [==============================] - 0s 2ms/step - loss: 0.8635 - root_mean_squared_error: 0.9254 - val_loss: 0.7599 - val_root_mean_squared_error: 0.8672
Epoch 44/500
17/17 [==============================] - 0s 3ms/step - loss: 0.8222 - root_mean_squared_error: 0.9024 - val_loss: 0.7688 - val_root_mean_squared_error: 0.8722
Epoch 45/500
17/17 [==============================] - 0s 3ms/step - loss: 0.8044 - root_mean_squared_error: 0.8928 - val_loss: 0.8370 - val_root_mean_squared_error: 0.9102
Epoch 46/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7909 - root_mean_squared_error: 0.8852 - val_loss: 0.7898 - val_root_mean_squared_error: 0.8843
Epoch 47/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7781 - root_mean_squared_error: 0.8778 - val_loss: 0.7945 - val_root_mean_squared_error: 0.8869
Epoch 48/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7846 - root_mean_squared_error: 0.8814 - val_loss: 0.7565 - val_root_mean_squared_error: 0.8651
Epoch 49/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7640 - root_mean_squared_error: 0.8695 - val_loss: 0.7308 - val_root_mean_squared_error: 0.8503
Epoch 50/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7490 - root_mean_squared_error: 0.8606 - val_loss: 0.7316 - val_root_mean_squared_error: 0.8503
Epoch 51/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7633 - root_mean_squared_error: 0.8690 - val_loss: 0.7718 - val_root_mean_squared_error: 0.8736
Epoch 52/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7413 - root_mean_squared_error: 0.8561 - val_loss: 0.7474 - val_root_mean_squared_error: 0.8598
Epoch 53/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7819 - root_mean_squared_error: 0.8799 - val_loss: 0.7969 - val_root_mean_squared_error: 0.8889
Epoch 54/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7457 - root_mean_squared_error: 0.8587 - val_loss: 0.7728 - val_root_mean_squared_error: 0.8742
Epoch 55/500
17/17 [==============================] - 0s 2ms/step - loss: 0.7614 - root_mean_squared_error: 0.8679 - val_loss: 0.7351 - val_root_mean_squared_error: 0.8528
Epoch 56/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7492 - root_mean_squared_error: 0.8608 - val_loss: 0.7074 - val_root_mean_squared_error: 0.8370
Epoch 57/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7448 - root_mean_squared_error: 0.8584 - val_loss: 0.8005 - val_root_mean_squared_error: 0.8896
Epoch 58/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7577 - root_mean_squared_error: 0.8658 - val_loss: 0.7571 - val_root_mean_squared_error: 0.8656
Epoch 59/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7609 - root_mean_squared_error: 0.8676 - val_loss: 0.7262 - val_root_mean_squared_error: 0.8480
Epoch 60/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7332 - root_mean_squared_error: 0.8512 - val_loss: 0.6973 - val_root_mean_squared_error: 0.8297
Epoch 61/500
17/17 [==============================] - 0s 2ms/step - loss: 0.7247 - root_mean_squared_error: 0.8462 - val_loss: 0.7129 - val_root_mean_squared_error: 0.8397
Epoch 62/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7480 - root_mean_squared_error: 0.8601 - val_loss: 0.6817 - val_root_mean_squared_error: 0.8205
Epoch 63/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7457 - root_mean_squared_error: 0.8587 - val_loss: 0.7785 - val_root_mean_squared_error: 0.8770
Epoch 64/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7189 - root_mean_squared_error: 0.8428 - val_loss: 0.7365 - val_root_mean_squared_error: 0.8521
Epoch 65/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7343 - root_mean_squared_error: 0.8519 - val_loss: 0.7567 - val_root_mean_squared_error: 0.8646
Epoch 66/500
17/17 [==============================] - 0s 2ms/step - loss: 0.7167 - root_mean_squared_error: 0.8417 - val_loss: 0.6936 - val_root_mean_squared_error: 0.8275
Epoch 67/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7052 - root_mean_squared_error: 0.8349 - val_loss: 0.7106 - val_root_mean_squared_error: 0.8373
Epoch 68/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7152 - root_mean_squared_error: 0.8407 - val_loss: 0.7273 - val_root_mean_squared_error: 0.8480
Epoch 69/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7132 - root_mean_squared_error: 0.8394 - val_loss: 0.7258 - val_root_mean_squared_error: 0.8463
Epoch 70/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6991 - root_mean_squared_error: 0.8311 - val_loss: 0.7345 - val_root_mean_squared_error: 0.8511
Epoch 71/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6969 - root_mean_squared_error: 0.8293 - val_loss: 0.6991 - val_root_mean_squared_error: 0.8309
Epoch 72/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7015 - root_mean_squared_error: 0.8322 - val_loss: 0.7493 - val_root_mean_squared_error: 0.8596
Epoch 73/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7649 - root_mean_squared_error: 0.8696 - val_loss: 0.7025 - val_root_mean_squared_error: 0.8326
Epoch 74/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6905 - root_mean_squared_error: 0.8253 - val_loss: 0.6538 - val_root_mean_squared_error: 0.8029
Epoch 75/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7004 - root_mean_squared_error: 0.8315 - val_loss: 0.7409 - val_root_mean_squared_error: 0.8551
Epoch 76/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6934 - root_mean_squared_error: 0.8272 - val_loss: 0.6958 - val_root_mean_squared_error: 0.8285
Epoch 77/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6927 - root_mean_squared_error: 0.8266 - val_loss: 0.7784 - val_root_mean_squared_error: 0.8770
Epoch 78/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6721 - root_mean_squared_error: 0.8140 - val_loss: 0.7061 - val_root_mean_squared_error: 0.8355
Epoch 79/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7000 - root_mean_squared_error: 0.8314 - val_loss: 0.6812 - val_root_mean_squared_error: 0.8209
Epoch 80/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7096 - root_mean_squared_error: 0.8370 - val_loss: 0.7052 - val_root_mean_squared_error: 0.8342
Epoch 81/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6934 - root_mean_squared_error: 0.8266 - val_loss: 0.6884 - val_root_mean_squared_error: 0.8238
Epoch 82/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7044 - root_mean_squared_error: 0.8340 - val_loss: 0.7043 - val_root_mean_squared_error: 0.8335
Epoch 83/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7094 - root_mean_squared_error: 0.8367 - val_loss: 0.6920 - val_root_mean_squared_error: 0.8260
Epoch 84/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6998 - root_mean_squared_error: 0.8312 - val_loss: 0.6877 - val_root_mean_squared_error: 0.8235
Epoch 85/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6942 - root_mean_squared_error: 0.8274 - val_loss: 0.6947 - val_root_mean_squared_error: 0.8285
Epoch 86/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6778 - root_mean_squared_error: 0.8175 - val_loss: 0.6620 - val_root_mean_squared_error: 0.8076
Epoch 87/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6856 - root_mean_squared_error: 0.8221 - val_loss: 0.6508 - val_root_mean_squared_error: 0.8006
Epoch 88/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6871 - root_mean_squared_error: 0.8230 - val_loss: 0.6493 - val_root_mean_squared_error: 0.7998
Epoch 89/500
17/17 [==============================] - 0s 3ms/step - loss: 0.7030 - root_mean_squared_error: 0.8329 - val_loss: 0.6469 - val_root_mean_squared_error: 0.7983
Epoch 90/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6630 - root_mean_squared_error: 0.8082 - val_loss: 0.6753 - val_root_mean_squared_error: 0.8158
Epoch 91/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6901 - root_mean_squared_error: 0.8249 - val_loss: 0.6485 - val_root_mean_squared_error: 0.7993
Epoch 92/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6700 - root_mean_squared_error: 0.8125 - val_loss: 0.6657 - val_root_mean_squared_error: 0.8100
Epoch 93/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6741 - root_mean_squared_error: 0.8150 - val_loss: 0.6330 - val_root_mean_squared_error: 0.7896
Epoch 94/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6747 - root_mean_squared_error: 0.8155 - val_loss: 0.6664 - val_root_mean_squared_error: 0.8104
Epoch 95/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6538 - root_mean_squared_error: 0.8027 - val_loss: 0.6424 - val_root_mean_squared_error: 0.7945
Epoch 96/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6716 - root_mean_squared_error: 0.8134 - val_loss: 0.6553 - val_root_mean_squared_error: 0.8027
Epoch 97/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6619 - root_mean_squared_error: 0.8072 - val_loss: 0.6620 - val_root_mean_squared_error: 0.8070
Epoch 98/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6685 - root_mean_squared_error: 0.8117 - val_loss: 0.6460 - val_root_mean_squared_error: 0.7972
Epoch 99/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6457 - root_mean_squared_error: 0.7971 - val_loss: 0.6486 - val_root_mean_squared_error: 0.8000
Epoch 100/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6634 - root_mean_squared_error: 0.8083 - val_loss: 0.7634 - val_root_mean_squared_error: 0.8682
Epoch 101/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6717 - root_mean_squared_error: 0.8138 - val_loss: 0.6584 - val_root_mean_squared_error: 0.8059
Epoch 102/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6577 - root_mean_squared_error: 0.8045 - val_loss: 0.6543 - val_root_mean_squared_error: 0.8022
Epoch 103/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6661 - root_mean_squared_error: 0.8099 - val_loss: 0.6436 - val_root_mean_squared_error: 0.7951
Epoch 104/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6643 - root_mean_squared_error: 0.8089 - val_loss: 0.6448 - val_root_mean_squared_error: 0.7963
Epoch 105/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6734 - root_mean_squared_error: 0.8142 - val_loss: 0.6588 - val_root_mean_squared_error: 0.8050
Epoch 106/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6558 - root_mean_squared_error: 0.8035 - val_loss: 0.6380 - val_root_mean_squared_error: 0.7920
Epoch 107/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6543 - root_mean_squared_error: 0.8026 - val_loss: 0.6717 - val_root_mean_squared_error: 0.8132
Epoch 108/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6560 - root_mean_squared_error: 0.8035 - val_loss: 0.6703 - val_root_mean_squared_error: 0.8126
Epoch 109/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6475 - root_mean_squared_error: 0.7980 - val_loss: 0.6658 - val_root_mean_squared_error: 0.8093
Epoch 110/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6515 - root_mean_squared_error: 0.8009 - val_loss: 0.6460 - val_root_mean_squared_error: 0.7972
Epoch 111/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6529 - root_mean_squared_error: 0.8018 - val_loss: 0.6581 - val_root_mean_squared_error: 0.8054
Epoch 112/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6410 - root_mean_squared_error: 0.7942 - val_loss: 0.6591 - val_root_mean_squared_error: 0.8052
Epoch 113/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6503 - root_mean_squared_error: 0.7998 - val_loss: 0.6335 - val_root_mean_squared_error: 0.7889
Epoch 114/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6372 - root_mean_squared_error: 0.7917 - val_loss: 0.6143 - val_root_mean_squared_error: 0.7768
Epoch 115/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6483 - root_mean_squared_error: 0.7983 - val_loss: 0.7078 - val_root_mean_squared_error: 0.8355
Epoch 116/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6614 - root_mean_squared_error: 0.8071 - val_loss: 0.6190 - val_root_mean_squared_error: 0.7806
Epoch 117/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6317 - root_mean_squared_error: 0.7881 - val_loss: 0.6357 - val_root_mean_squared_error: 0.7903
Epoch 118/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6284 - root_mean_squared_error: 0.7861 - val_loss: 0.6826 - val_root_mean_squared_error: 0.8200
Epoch 119/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6321 - root_mean_squared_error: 0.7884 - val_loss: 0.7019 - val_root_mean_squared_error: 0.8316
Epoch 120/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6450 - root_mean_squared_error: 0.7964 - val_loss: 0.6141 - val_root_mean_squared_error: 0.7762
Epoch 121/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6433 - root_mean_squared_error: 0.7951 - val_loss: 0.6344 - val_root_mean_squared_error: 0.7897
Epoch 122/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6575 - root_mean_squared_error: 0.8045 - val_loss: 0.6564 - val_root_mean_squared_error: 0.8032
Epoch 123/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6258 - root_mean_squared_error: 0.7842 - val_loss: 0.6205 - val_root_mean_squared_error: 0.7820
Epoch 124/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6392 - root_mean_squared_error: 0.7927 - val_loss: 0.6890 - val_root_mean_squared_error: 0.8244
Epoch 125/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6718 - root_mean_squared_error: 0.8127 - val_loss: 0.6264 - val_root_mean_squared_error: 0.7851
Epoch 126/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6313 - root_mean_squared_error: 0.7876 - val_loss: 0.6492 - val_root_mean_squared_error: 0.7993
Epoch 127/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6477 - root_mean_squared_error: 0.7984 - val_loss: 0.6351 - val_root_mean_squared_error: 0.7904
Epoch 128/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6298 - root_mean_squared_error: 0.7869 - val_loss: 0.6417 - val_root_mean_squared_error: 0.7941
Epoch 129/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6344 - root_mean_squared_error: 0.7894 - val_loss: 0.6136 - val_root_mean_squared_error: 0.7768
Epoch 130/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6259 - root_mean_squared_error: 0.7844 - val_loss: 0.6252 - val_root_mean_squared_error: 0.7833
Epoch 131/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6324 - root_mean_squared_error: 0.7883 - val_loss: 0.6262 - val_root_mean_squared_error: 0.7837
Epoch 132/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6668 - root_mean_squared_error: 0.8101 - val_loss: 0.6180 - val_root_mean_squared_error: 0.7783
Epoch 133/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6419 - root_mean_squared_error: 0.7944 - val_loss: 0.6218 - val_root_mean_squared_error: 0.7817
Epoch 134/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6363 - root_mean_squared_error: 0.7906 - val_loss: 0.6684 - val_root_mean_squared_error: 0.8110
Epoch 135/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6246 - root_mean_squared_error: 0.7832 - val_loss: 0.6195 - val_root_mean_squared_error: 0.7797
Epoch 136/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6470 - root_mean_squared_error: 0.7975 - val_loss: 0.6129 - val_root_mean_squared_error: 0.7748
Epoch 137/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6510 - root_mean_squared_error: 0.8002 - val_loss: 0.6303 - val_root_mean_squared_error: 0.7870
Epoch 138/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6213 - root_mean_squared_error: 0.7812 - val_loss: 0.6339 - val_root_mean_squared_error: 0.7884
Epoch 139/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6315 - root_mean_squared_error: 0.7874 - val_loss: 0.6114 - val_root_mean_squared_error: 0.7754
Epoch 140/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6299 - root_mean_squared_error: 0.7871 - val_loss: 0.6338 - val_root_mean_squared_error: 0.7891
Epoch 141/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6325 - root_mean_squared_error: 0.7882 - val_loss: 0.6188 - val_root_mean_squared_error: 0.7794
Epoch 142/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6367 - root_mean_squared_error: 0.7912 - val_loss: 0.6406 - val_root_mean_squared_error: 0.7931
Epoch 143/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6330 - root_mean_squared_error: 0.7886 - val_loss: 0.6082 - val_root_mean_squared_error: 0.7725
Epoch 144/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6291 - root_mean_squared_error: 0.7861 - val_loss: 0.6225 - val_root_mean_squared_error: 0.7819
Epoch 145/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6459 - root_mean_squared_error: 0.7968 - val_loss: 0.6086 - val_root_mean_squared_error: 0.7724
Epoch 146/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6409 - root_mean_squared_error: 0.7939 - val_loss: 0.6297 - val_root_mean_squared_error: 0.7861
Epoch 147/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6250 - root_mean_squared_error: 0.7833 - val_loss: 0.6008 - val_root_mean_squared_error: 0.7673
Epoch 148/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6295 - root_mean_squared_error: 0.7865 - val_loss: 0.6705 - val_root_mean_squared_error: 0.8121
Epoch 149/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6277 - root_mean_squared_error: 0.7853 - val_loss: 0.6241 - val_root_mean_squared_error: 0.7829
Epoch 150/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6354 - root_mean_squared_error: 0.7898 - val_loss: 0.6181 - val_root_mean_squared_error: 0.7795
Epoch 151/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6281 - root_mean_squared_error: 0.7858 - val_loss: 0.6314 - val_root_mean_squared_error: 0.7873
Epoch 152/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6373 - root_mean_squared_error: 0.7918 - val_loss: 0.6283 - val_root_mean_squared_error: 0.7866
Epoch 153/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6237 - root_mean_squared_error: 0.7825 - val_loss: 0.6348 - val_root_mean_squared_error: 0.7894
Epoch 154/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6262 - root_mean_squared_error: 0.7839 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7657
Epoch 155/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6236 - root_mean_squared_error: 0.7822 - val_loss: 0.6191 - val_root_mean_squared_error: 0.7799
Epoch 156/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6166 - root_mean_squared_error: 0.7779 - val_loss: 0.6445 - val_root_mean_squared_error: 0.7962
Epoch 157/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6299 - root_mean_squared_error: 0.7863 - val_loss: 0.6080 - val_root_mean_squared_error: 0.7724
Epoch 158/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6390 - root_mean_squared_error: 0.7924 - val_loss: 0.6075 - val_root_mean_squared_error: 0.7725
Epoch 159/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6249 - root_mean_squared_error: 0.7828 - val_loss: 0.6144 - val_root_mean_squared_error: 0.7767
Epoch 160/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6276 - root_mean_squared_error: 0.7851 - val_loss: 0.6207 - val_root_mean_squared_error: 0.7812
Epoch 161/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6236 - root_mean_squared_error: 0.7827 - val_loss: 0.6222 - val_root_mean_squared_error: 0.7810
Epoch 162/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6260 - root_mean_squared_error: 0.7839 - val_loss: 0.6050 - val_root_mean_squared_error: 0.7709
Epoch 163/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6316 - root_mean_squared_error: 0.7875 - val_loss: 0.6058 - val_root_mean_squared_error: 0.7703
Epoch 164/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6393 - root_mean_squared_error: 0.7923 - val_loss: 0.6021 - val_root_mean_squared_error: 0.7678
Epoch 165/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6260 - root_mean_squared_error: 0.7842 - val_loss: 0.6293 - val_root_mean_squared_error: 0.7859
Epoch 166/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6326 - root_mean_squared_error: 0.7884 - val_loss: 0.6216 - val_root_mean_squared_error: 0.7818
Epoch 167/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6255 - root_mean_squared_error: 0.7837 - val_loss: 0.6138 - val_root_mean_squared_error: 0.7754
Epoch 168/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6282 - root_mean_squared_error: 0.7854 - val_loss: 0.6315 - val_root_mean_squared_error: 0.7877
Epoch 169/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6148 - root_mean_squared_error: 0.7766 - val_loss: 0.6099 - val_root_mean_squared_error: 0.7735
Epoch 170/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6137 - root_mean_squared_error: 0.7763 - val_loss: 0.6506 - val_root_mean_squared_error: 0.7990
Epoch 171/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6325 - root_mean_squared_error: 0.7882 - val_loss: 0.6230 - val_root_mean_squared_error: 0.7826
Epoch 172/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6273 - root_mean_squared_error: 0.7849 - val_loss: 0.6093 - val_root_mean_squared_error: 0.7734
Epoch 173/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6243 - root_mean_squared_error: 0.7824 - val_loss: 0.6075 - val_root_mean_squared_error: 0.7714
Epoch 174/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6185 - root_mean_squared_error: 0.7788 - val_loss: 0.6616 - val_root_mean_squared_error: 0.8060
Epoch 175/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6197 - root_mean_squared_error: 0.7798 - val_loss: 0.5990 - val_root_mean_squared_error: 0.7668
Epoch 176/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6411 - root_mean_squared_error: 0.7936 - val_loss: 0.6121 - val_root_mean_squared_error: 0.7752
Epoch 177/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6303 - root_mean_squared_error: 0.7867 - val_loss: 0.6189 - val_root_mean_squared_error: 0.7790
Epoch 178/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6340 - root_mean_squared_error: 0.7885 - val_loss: 0.6316 - val_root_mean_squared_error: 0.7875
Epoch 179/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6435 - root_mean_squared_error: 0.7949 - val_loss: 0.5934 - val_root_mean_squared_error: 0.7629
Epoch 180/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6224 - root_mean_squared_error: 0.7814 - val_loss: 0.6250 - val_root_mean_squared_error: 0.7824
Epoch 181/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6415 - root_mean_squared_error: 0.7939 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7644
Epoch 182/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6269 - root_mean_squared_error: 0.7844 - val_loss: 0.6082 - val_root_mean_squared_error: 0.7724
Epoch 183/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6132 - root_mean_squared_error: 0.7760 - val_loss: 0.6122 - val_root_mean_squared_error: 0.7755
Epoch 184/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6273 - root_mean_squared_error: 0.7845 - val_loss: 0.6334 - val_root_mean_squared_error: 0.7890
Epoch 185/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6337 - root_mean_squared_error: 0.7887 - val_loss: 0.6053 - val_root_mean_squared_error: 0.7700
Epoch 186/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6346 - root_mean_squared_error: 0.7892 - val_loss: 0.6310 - val_root_mean_squared_error: 0.7875
Epoch 187/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6241 - root_mean_squared_error: 0.7828 - val_loss: 0.6025 - val_root_mean_squared_error: 0.7690
Epoch 188/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6344 - root_mean_squared_error: 0.7888 - val_loss: 0.5956 - val_root_mean_squared_error: 0.7640
Epoch 189/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6063 - root_mean_squared_error: 0.7711 - val_loss: 0.6512 - val_root_mean_squared_error: 0.7997
Epoch 190/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6222 - root_mean_squared_error: 0.7813 - val_loss: 0.6220 - val_root_mean_squared_error: 0.7810
Epoch 191/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6140 - root_mean_squared_error: 0.7758 - val_loss: 0.6105 - val_root_mean_squared_error: 0.7731
Epoch 192/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6286 - root_mean_squared_error: 0.7851 - val_loss: 0.6083 - val_root_mean_squared_error: 0.7720
Epoch 193/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6116 - root_mean_squared_error: 0.7745 - val_loss: 0.6020 - val_root_mean_squared_error: 0.7684
Epoch 194/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6220 - root_mean_squared_error: 0.7811 - val_loss: 0.6180 - val_root_mean_squared_error: 0.7799
Epoch 195/500
17/17 [==============================] - 0s 4ms/step - loss: 0.6083 - root_mean_squared_error: 0.7722 - val_loss: 0.6646 - val_root_mean_squared_error: 0.8076
Epoch 196/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6202 - root_mean_squared_error: 0.7801 - val_loss: 0.6079 - val_root_mean_squared_error: 0.7715
Epoch 197/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6192 - root_mean_squared_error: 0.7792 - val_loss: 0.6133 - val_root_mean_squared_error: 0.7757
Epoch 198/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6239 - root_mean_squared_error: 0.7826 - val_loss: 0.6183 - val_root_mean_squared_error: 0.7782
Epoch 199/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6223 - root_mean_squared_error: 0.7812 - val_loss: 0.6310 - val_root_mean_squared_error: 0.7877
Epoch 200/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6371 - root_mean_squared_error: 0.7908 - val_loss: 0.6552 - val_root_mean_squared_error: 0.8033
Epoch 201/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6164 - root_mean_squared_error: 0.7777 - val_loss: 0.6032 - val_root_mean_squared_error: 0.7685
Epoch 202/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6108 - root_mean_squared_error: 0.7740 - val_loss: 0.6223 - val_root_mean_squared_error: 0.7804
Epoch 203/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6147 - root_mean_squared_error: 0.7765 - val_loss: 0.6394 - val_root_mean_squared_error: 0.7926
Epoch 204/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6143 - root_mean_squared_error: 0.7757 - val_loss: 0.6461 - val_root_mean_squared_error: 0.7969
Epoch 205/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6236 - root_mean_squared_error: 0.7821 - val_loss: 0.6580 - val_root_mean_squared_error: 0.8035
Epoch 206/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6187 - root_mean_squared_error: 0.7789 - val_loss: 0.5949 - val_root_mean_squared_error: 0.7631
Epoch 207/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6188 - root_mean_squared_error: 0.7789 - val_loss: 0.6358 - val_root_mean_squared_error: 0.7905
Epoch 208/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6055 - root_mean_squared_error: 0.7705 - val_loss: 0.5923 - val_root_mean_squared_error: 0.7618
Epoch 209/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6418 - root_mean_squared_error: 0.7937 - val_loss: 0.6056 - val_root_mean_squared_error: 0.7705
Epoch 210/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6266 - root_mean_squared_error: 0.7843 - val_loss: 0.6504 - val_root_mean_squared_error: 0.7985
Epoch 211/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6267 - root_mean_squared_error: 0.7842 - val_loss: 0.6060 - val_root_mean_squared_error: 0.7704
Epoch 212/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6166 - root_mean_squared_error: 0.7776 - val_loss: 0.6155 - val_root_mean_squared_error: 0.7772
Epoch 213/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6285 - root_mean_squared_error: 0.7853 - val_loss: 0.6162 - val_root_mean_squared_error: 0.7779
Epoch 214/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6107 - root_mean_squared_error: 0.7740 - val_loss: 0.6279 - val_root_mean_squared_error: 0.7851
Epoch 215/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6188 - root_mean_squared_error: 0.7790 - val_loss: 0.6199 - val_root_mean_squared_error: 0.7794
Epoch 216/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6254 - root_mean_squared_error: 0.7832 - val_loss: 0.6190 - val_root_mean_squared_error: 0.7786
Epoch 217/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6223 - root_mean_squared_error: 0.7815 - val_loss: 0.6174 - val_root_mean_squared_error: 0.7780
Epoch 218/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6176 - root_mean_squared_error: 0.7783 - val_loss: 0.6151 - val_root_mean_squared_error: 0.7761
Epoch 219/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6283 - root_mean_squared_error: 0.7851 - val_loss: 0.5976 - val_root_mean_squared_error: 0.7650
Epoch 220/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6109 - root_mean_squared_error: 0.7739 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7648
Epoch 221/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6081 - root_mean_squared_error: 0.7716 - val_loss: 0.6008 - val_root_mean_squared_error: 0.7675
Epoch 222/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6099 - root_mean_squared_error: 0.7729 - val_loss: 0.6052 - val_root_mean_squared_error: 0.7702
Epoch 223/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6151 - root_mean_squared_error: 0.7765 - val_loss: 0.6307 - val_root_mean_squared_error: 0.7865
Epoch 224/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6129 - root_mean_squared_error: 0.7746 - val_loss: 0.5939 - val_root_mean_squared_error: 0.7624
Epoch 225/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6273 - root_mean_squared_error: 0.7843 - val_loss: 0.6182 - val_root_mean_squared_error: 0.7781
Epoch 226/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6146 - root_mean_squared_error: 0.7762 - val_loss: 0.6062 - val_root_mean_squared_error: 0.7712
Epoch 227/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6162 - root_mean_squared_error: 0.7774 - val_loss: 0.5997 - val_root_mean_squared_error: 0.7656
Epoch 228/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6077 - root_mean_squared_error: 0.7715 - val_loss: 0.6247 - val_root_mean_squared_error: 0.7820
Epoch 229/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6173 - root_mean_squared_error: 0.7781 - val_loss: 0.6127 - val_root_mean_squared_error: 0.7760
Epoch 230/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6179 - root_mean_squared_error: 0.7780 - val_loss: 0.6052 - val_root_mean_squared_error: 0.7696
Epoch 231/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6180 - root_mean_squared_error: 0.7781 - val_loss: 0.6023 - val_root_mean_squared_error: 0.7679
Epoch 232/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6027 - root_mean_squared_error: 0.7679 - val_loss: 0.6250 - val_root_mean_squared_error: 0.7826
Epoch 233/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6155 - root_mean_squared_error: 0.7767 - val_loss: 0.6151 - val_root_mean_squared_error: 0.7760
Epoch 234/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6029 - root_mean_squared_error: 0.7684 - val_loss: 0.6366 - val_root_mean_squared_error: 0.7903
Epoch 235/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6093 - root_mean_squared_error: 0.7729 - val_loss: 0.6089 - val_root_mean_squared_error: 0.7724
Epoch 236/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6132 - root_mean_squared_error: 0.7752 - val_loss: 0.6038 - val_root_mean_squared_error: 0.7689
Epoch 237/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6131 - root_mean_squared_error: 0.7751 - val_loss: 0.6110 - val_root_mean_squared_error: 0.7732
Epoch 238/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6260 - root_mean_squared_error: 0.7835 - val_loss: 0.6351 - val_root_mean_squared_error: 0.7891
Epoch 239/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6213 - root_mean_squared_error: 0.7805 - val_loss: 0.6263 - val_root_mean_squared_error: 0.7836
Epoch 240/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6263 - root_mean_squared_error: 0.7836 - val_loss: 0.6087 - val_root_mean_squared_error: 0.7724
Epoch 241/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6075 - root_mean_squared_error: 0.7711 - val_loss: 0.6030 - val_root_mean_squared_error: 0.7686
Epoch 242/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6316 - root_mean_squared_error: 0.7869 - val_loss: 0.6012 - val_root_mean_squared_error: 0.7675
Epoch 243/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6092 - root_mean_squared_error: 0.7723 - val_loss: 0.6189 - val_root_mean_squared_error: 0.7786
Epoch 244/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6281 - root_mean_squared_error: 0.7847 - val_loss: 0.6188 - val_root_mean_squared_error: 0.7799
Epoch 245/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6235 - root_mean_squared_error: 0.7816 - val_loss: 0.6137 - val_root_mean_squared_error: 0.7752
Epoch 246/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6164 - root_mean_squared_error: 0.7768 - val_loss: 0.6305 - val_root_mean_squared_error: 0.7872
Epoch 247/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6267 - root_mean_squared_error: 0.7835 - val_loss: 0.5914 - val_root_mean_squared_error: 0.7606
Epoch 248/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6130 - root_mean_squared_error: 0.7747 - val_loss: 0.6418 - val_root_mean_squared_error: 0.7927
Epoch 249/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6159 - root_mean_squared_error: 0.7764 - val_loss: 0.5972 - val_root_mean_squared_error: 0.7650
Epoch 250/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6096 - root_mean_squared_error: 0.7727 - val_loss: 0.6331 - val_root_mean_squared_error: 0.7881
Epoch 251/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6071 - root_mean_squared_error: 0.7710 - val_loss: 0.6099 - val_root_mean_squared_error: 0.7722
Epoch 252/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6069 - root_mean_squared_error: 0.7708 - val_loss: 0.6243 - val_root_mean_squared_error: 0.7816
Epoch 253/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6161 - root_mean_squared_error: 0.7768 - val_loss: 0.6307 - val_root_mean_squared_error: 0.7871
Epoch 254/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6123 - root_mean_squared_error: 0.7741 - val_loss: 0.6157 - val_root_mean_squared_error: 0.7766
Epoch 255/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6210 - root_mean_squared_error: 0.7800 - val_loss: 0.6097 - val_root_mean_squared_error: 0.7717
Epoch 256/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6176 - root_mean_squared_error: 0.7782 - val_loss: 0.6136 - val_root_mean_squared_error: 0.7752
Epoch 257/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6086 - root_mean_squared_error: 0.7719 - val_loss: 0.5912 - val_root_mean_squared_error: 0.7599
Epoch 258/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6101 - root_mean_squared_error: 0.7729 - val_loss: 0.6138 - val_root_mean_squared_error: 0.7757
Epoch 259/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6102 - root_mean_squared_error: 0.7731 - val_loss: 0.6012 - val_root_mean_squared_error: 0.7679
Epoch 260/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6150 - root_mean_squared_error: 0.7764 - val_loss: 0.6086 - val_root_mean_squared_error: 0.7709
Epoch 261/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6132 - root_mean_squared_error: 0.7748 - val_loss: 0.5959 - val_root_mean_squared_error: 0.7639
Epoch 262/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6101 - root_mean_squared_error: 0.7728 - val_loss: 0.6166 - val_root_mean_squared_error: 0.7767
Epoch 263/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6064 - root_mean_squared_error: 0.7707 - val_loss: 0.6161 - val_root_mean_squared_error: 0.7766
Epoch 264/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6266 - root_mean_squared_error: 0.7830 - val_loss: 0.5905 - val_root_mean_squared_error: 0.7597
Epoch 265/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6074 - root_mean_squared_error: 0.7712 - val_loss: 0.5837 - val_root_mean_squared_error: 0.7557
Epoch 266/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6098 - root_mean_squared_error: 0.7726 - val_loss: 0.6212 - val_root_mean_squared_error: 0.7807
Epoch 267/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6009 - root_mean_squared_error: 0.7670 - val_loss: 0.6041 - val_root_mean_squared_error: 0.7684
Epoch 268/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6117 - root_mean_squared_error: 0.7740 - val_loss: 0.5876 - val_root_mean_squared_error: 0.7584
Epoch 269/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6054 - root_mean_squared_error: 0.7702 - val_loss: 0.6223 - val_root_mean_squared_error: 0.7806
Epoch 270/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6090 - root_mean_squared_error: 0.7721 - val_loss: 0.6397 - val_root_mean_squared_error: 0.7914
Epoch 271/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6091 - root_mean_squared_error: 0.7723 - val_loss: 0.5925 - val_root_mean_squared_error: 0.7609
Epoch 272/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6056 - root_mean_squared_error: 0.7699 - val_loss: 0.6020 - val_root_mean_squared_error: 0.7673
Epoch 273/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6059 - root_mean_squared_error: 0.7700 - val_loss: 0.6188 - val_root_mean_squared_error: 0.7783
Epoch 274/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6276 - root_mean_squared_error: 0.7844 - val_loss: 0.5953 - val_root_mean_squared_error: 0.7633
Epoch 275/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6113 - root_mean_squared_error: 0.7735 - val_loss: 0.5934 - val_root_mean_squared_error: 0.7614
Epoch 276/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6097 - root_mean_squared_error: 0.7728 - val_loss: 0.6029 - val_root_mean_squared_error: 0.7681
Epoch 277/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6127 - root_mean_squared_error: 0.7744 - val_loss: 0.5942 - val_root_mean_squared_error: 0.7617
Epoch 278/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6021 - root_mean_squared_error: 0.7677 - val_loss: 0.6305 - val_root_mean_squared_error: 0.7856
Epoch 279/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6136 - root_mean_squared_error: 0.7751 - val_loss: 0.6046 - val_root_mean_squared_error: 0.7686
Epoch 280/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6007 - root_mean_squared_error: 0.7670 - val_loss: 0.6032 - val_root_mean_squared_error: 0.7675
Epoch 281/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6137 - root_mean_squared_error: 0.7750 - val_loss: 0.6320 - val_root_mean_squared_error: 0.7857
Epoch 282/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6030 - root_mean_squared_error: 0.7681 - val_loss: 0.6494 - val_root_mean_squared_error: 0.7973
Epoch 283/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6074 - root_mean_squared_error: 0.7713 - val_loss: 0.5901 - val_root_mean_squared_error: 0.7598
Epoch 284/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6139 - root_mean_squared_error: 0.7749 - val_loss: 0.6326 - val_root_mean_squared_error: 0.7863
Epoch 285/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6041 - root_mean_squared_error: 0.7689 - val_loss: 0.6032 - val_root_mean_squared_error: 0.7681
Epoch 286/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6091 - root_mean_squared_error: 0.7724 - val_loss: 0.6142 - val_root_mean_squared_error: 0.7757
Epoch 287/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6135 - root_mean_squared_error: 0.7750 - val_loss: 0.6041 - val_root_mean_squared_error: 0.7682
Epoch 288/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6051 - root_mean_squared_error: 0.7693 - val_loss: 0.6033 - val_root_mean_squared_error: 0.7688
Epoch 289/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6123 - root_mean_squared_error: 0.7740 - val_loss: 0.6082 - val_root_mean_squared_error: 0.7707
Epoch 290/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6101 - root_mean_squared_error: 0.7725 - val_loss: 0.6074 - val_root_mean_squared_error: 0.7703
Epoch 291/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5999 - root_mean_squared_error: 0.7660 - val_loss: 0.6118 - val_root_mean_squared_error: 0.7742
Epoch 292/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6066 - root_mean_squared_error: 0.7705 - val_loss: 0.6081 - val_root_mean_squared_error: 0.7705
Epoch 293/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6053 - root_mean_squared_error: 0.7694 - val_loss: 0.6131 - val_root_mean_squared_error: 0.7745
Epoch 294/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6246 - root_mean_squared_error: 0.7821 - val_loss: 0.6128 - val_root_mean_squared_error: 0.7743
Epoch 295/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6155 - root_mean_squared_error: 0.7765 - val_loss: 0.5942 - val_root_mean_squared_error: 0.7621
Epoch 296/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6006 - root_mean_squared_error: 0.7671 - val_loss: 0.6121 - val_root_mean_squared_error: 0.7737
Epoch 297/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6157 - root_mean_squared_error: 0.7765 - val_loss: 0.6203 - val_root_mean_squared_error: 0.7793
Epoch 298/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6170 - root_mean_squared_error: 0.7771 - val_loss: 0.5958 - val_root_mean_squared_error: 0.7630
Epoch 299/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6162 - root_mean_squared_error: 0.7767 - val_loss: 0.6081 - val_root_mean_squared_error: 0.7713
Epoch 300/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6178 - root_mean_squared_error: 0.7779 - val_loss: 0.6254 - val_root_mean_squared_error: 0.7824
Epoch 301/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6110 - root_mean_squared_error: 0.7732 - val_loss: 0.6216 - val_root_mean_squared_error: 0.7802
Epoch 302/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6082 - root_mean_squared_error: 0.7710 - val_loss: 0.5842 - val_root_mean_squared_error: 0.7548
Epoch 303/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6027 - root_mean_squared_error: 0.7677 - val_loss: 0.6034 - val_root_mean_squared_error: 0.7674
Epoch 304/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6151 - root_mean_squared_error: 0.7759 - val_loss: 0.6243 - val_root_mean_squared_error: 0.7829
Epoch 305/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5990 - root_mean_squared_error: 0.7653 - val_loss: 0.6295 - val_root_mean_squared_error: 0.7846
Epoch 306/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6206 - root_mean_squared_error: 0.7793 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7656
Epoch 307/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6036 - root_mean_squared_error: 0.7683 - val_loss: 0.5945 - val_root_mean_squared_error: 0.7620
Epoch 308/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6022 - root_mean_squared_error: 0.7673 - val_loss: 0.6073 - val_root_mean_squared_error: 0.7707
Epoch 309/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6003 - root_mean_squared_error: 0.7661 - val_loss: 0.6109 - val_root_mean_squared_error: 0.7731
Epoch 310/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6120 - root_mean_squared_error: 0.7743 - val_loss: 0.5914 - val_root_mean_squared_error: 0.7601
Epoch 311/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6068 - root_mean_squared_error: 0.7703 - val_loss: 0.5974 - val_root_mean_squared_error: 0.7646
Epoch 312/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6095 - root_mean_squared_error: 0.7720 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7659
Epoch 313/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6011 - root_mean_squared_error: 0.7670 - val_loss: 0.6001 - val_root_mean_squared_error: 0.7664
Epoch 314/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6138 - root_mean_squared_error: 0.7749 - val_loss: 0.6054 - val_root_mean_squared_error: 0.7690
Epoch 315/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6011 - root_mean_squared_error: 0.7670 - val_loss: 0.6013 - val_root_mean_squared_error: 0.7668
Epoch 316/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6157 - root_mean_squared_error: 0.7760 - val_loss: 0.6191 - val_root_mean_squared_error: 0.7783
Epoch 317/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6112 - root_mean_squared_error: 0.7733 - val_loss: 0.6262 - val_root_mean_squared_error: 0.7837
Epoch 318/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6012 - root_mean_squared_error: 0.7670 - val_loss: 0.6277 - val_root_mean_squared_error: 0.7843
Epoch 319/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6197 - root_mean_squared_error: 0.7783 - val_loss: 0.5978 - val_root_mean_squared_error: 0.7645
Epoch 320/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6127 - root_mean_squared_error: 0.7739 - val_loss: 0.5882 - val_root_mean_squared_error: 0.7584
Epoch 321/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5999 - root_mean_squared_error: 0.7659 - val_loss: 0.6065 - val_root_mean_squared_error: 0.7705
Epoch 322/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6172 - root_mean_squared_error: 0.7772 - val_loss: 0.5855 - val_root_mean_squared_error: 0.7560
Epoch 323/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6125 - root_mean_squared_error: 0.7744 - val_loss: 0.6138 - val_root_mean_squared_error: 0.7754
Epoch 324/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6102 - root_mean_squared_error: 0.7726 - val_loss: 0.6011 - val_root_mean_squared_error: 0.7666
Epoch 325/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6034 - root_mean_squared_error: 0.7680 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7657
Epoch 326/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6002 - root_mean_squared_error: 0.7658 - val_loss: 0.6360 - val_root_mean_squared_error: 0.7900
Epoch 327/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5974 - root_mean_squared_error: 0.7644 - val_loss: 0.5937 - val_root_mean_squared_error: 0.7620
Epoch 328/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6116 - root_mean_squared_error: 0.7733 - val_loss: 0.6291 - val_root_mean_squared_error: 0.7846
Epoch 329/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6045 - root_mean_squared_error: 0.7687 - val_loss: 0.6067 - val_root_mean_squared_error: 0.7702
Epoch 330/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6184 - root_mean_squared_error: 0.7779 - val_loss: 0.5841 - val_root_mean_squared_error: 0.7553
Epoch 331/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6035 - root_mean_squared_error: 0.7686 - val_loss: 0.5989 - val_root_mean_squared_error: 0.7642
Epoch 332/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6092 - root_mean_squared_error: 0.7720 - val_loss: 0.6436 - val_root_mean_squared_error: 0.7939
Epoch 333/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6043 - root_mean_squared_error: 0.7691 - val_loss: 0.6267 - val_root_mean_squared_error: 0.7839
Epoch 334/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6062 - root_mean_squared_error: 0.7697 - val_loss: 0.6072 - val_root_mean_squared_error: 0.7702
Epoch 335/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6082 - root_mean_squared_error: 0.7714 - val_loss: 0.5942 - val_root_mean_squared_error: 0.7622
Epoch 336/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6090 - root_mean_squared_error: 0.7717 - val_loss: 0.6208 - val_root_mean_squared_error: 0.7796
Epoch 337/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6034 - root_mean_squared_error: 0.7680 - val_loss: 0.6428 - val_root_mean_squared_error: 0.7934
Epoch 338/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6011 - root_mean_squared_error: 0.7668 - val_loss: 0.6398 - val_root_mean_squared_error: 0.7923
Epoch 339/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6084 - root_mean_squared_error: 0.7713 - val_loss: 0.6040 - val_root_mean_squared_error: 0.7696
Epoch 340/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6128 - root_mean_squared_error: 0.7742 - val_loss: 0.6271 - val_root_mean_squared_error: 0.7835
Epoch 341/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6043 - root_mean_squared_error: 0.7685 - val_loss: 0.5963 - val_root_mean_squared_error: 0.7639
Epoch 342/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6125 - root_mean_squared_error: 0.7741 - val_loss: 0.6655 - val_root_mean_squared_error: 0.8080
Epoch 343/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6043 - root_mean_squared_error: 0.7693 - val_loss: 0.5994 - val_root_mean_squared_error: 0.7659
Epoch 344/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6135 - root_mean_squared_error: 0.7747 - val_loss: 0.6104 - val_root_mean_squared_error: 0.7724
Epoch 345/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6005 - root_mean_squared_error: 0.7662 - val_loss: 0.5936 - val_root_mean_squared_error: 0.7613
Epoch 346/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6031 - root_mean_squared_error: 0.7678 - val_loss: 0.5996 - val_root_mean_squared_error: 0.7649
Epoch 347/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5987 - root_mean_squared_error: 0.7649 - val_loss: 0.6065 - val_root_mean_squared_error: 0.7706
Epoch 348/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6015 - root_mean_squared_error: 0.7666 - val_loss: 0.5836 - val_root_mean_squared_error: 0.7551
Epoch 349/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6040 - root_mean_squared_error: 0.7685 - val_loss: 0.6331 - val_root_mean_squared_error: 0.7873
Epoch 350/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6012 - root_mean_squared_error: 0.7668 - val_loss: 0.6230 - val_root_mean_squared_error: 0.7798
Epoch 351/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6008 - root_mean_squared_error: 0.7660 - val_loss: 0.5872 - val_root_mean_squared_error: 0.7569
Epoch 352/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6013 - root_mean_squared_error: 0.7669 - val_loss: 0.5968 - val_root_mean_squared_error: 0.7637
Epoch 353/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6073 - root_mean_squared_error: 0.7704 - val_loss: 0.5853 - val_root_mean_squared_error: 0.7562
Epoch 354/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6090 - root_mean_squared_error: 0.7721 - val_loss: 0.6307 - val_root_mean_squared_error: 0.7859
Epoch 355/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6105 - root_mean_squared_error: 0.7725 - val_loss: 0.6011 - val_root_mean_squared_error: 0.7675
Epoch 356/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6049 - root_mean_squared_error: 0.7692 - val_loss: 0.6029 - val_root_mean_squared_error: 0.7677
Epoch 357/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5981 - root_mean_squared_error: 0.7647 - val_loss: 0.5989 - val_root_mean_squared_error: 0.7658
Epoch 358/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6064 - root_mean_squared_error: 0.7703 - val_loss: 0.5948 - val_root_mean_squared_error: 0.7630
Epoch 359/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6140 - root_mean_squared_error: 0.7749 - val_loss: 0.6064 - val_root_mean_squared_error: 0.7705
Epoch 360/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6112 - root_mean_squared_error: 0.7732 - val_loss: 0.6102 - val_root_mean_squared_error: 0.7722
Epoch 361/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6061 - root_mean_squared_error: 0.7699 - val_loss: 0.5919 - val_root_mean_squared_error: 0.7612
Epoch 362/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6129 - root_mean_squared_error: 0.7743 - val_loss: 0.6037 - val_root_mean_squared_error: 0.7683
Epoch 363/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6132 - root_mean_squared_error: 0.7743 - val_loss: 0.5955 - val_root_mean_squared_error: 0.7638
Epoch 364/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5977 - root_mean_squared_error: 0.7638 - val_loss: 0.6115 - val_root_mean_squared_error: 0.7741
Epoch 365/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6110 - root_mean_squared_error: 0.7730 - val_loss: 0.6124 - val_root_mean_squared_error: 0.7736
Epoch 366/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6130 - root_mean_squared_error: 0.7741 - val_loss: 0.6160 - val_root_mean_squared_error: 0.7764
Epoch 367/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6081 - root_mean_squared_error: 0.7709 - val_loss: 0.6070 - val_root_mean_squared_error: 0.7704
Epoch 368/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6104 - root_mean_squared_error: 0.7725 - val_loss: 0.5928 - val_root_mean_squared_error: 0.7602
Epoch 369/500
17/17 [==============================] - 0s 4ms/step - loss: 0.6044 - root_mean_squared_error: 0.7686 - val_loss: 0.6322 - val_root_mean_squared_error: 0.7867
Epoch 370/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6025 - root_mean_squared_error: 0.7674 - val_loss: 0.5915 - val_root_mean_squared_error: 0.7598
Epoch 371/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6143 - root_mean_squared_error: 0.7750 - val_loss: 0.5955 - val_root_mean_squared_error: 0.7622
Epoch 372/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6063 - root_mean_squared_error: 0.7696 - val_loss: 0.5959 - val_root_mean_squared_error: 0.7639
Epoch 373/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6034 - root_mean_squared_error: 0.7679 - val_loss: 0.5832 - val_root_mean_squared_error: 0.7548
Epoch 374/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6042 - root_mean_squared_error: 0.7683 - val_loss: 0.6047 - val_root_mean_squared_error: 0.7685
Epoch 375/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6182 - root_mean_squared_error: 0.7780 - val_loss: 0.6271 - val_root_mean_squared_error: 0.7831
Epoch 376/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6010 - root_mean_squared_error: 0.7663 - val_loss: 0.6093 - val_root_mean_squared_error: 0.7720
Epoch 377/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6032 - root_mean_squared_error: 0.7680 - val_loss: 0.6004 - val_root_mean_squared_error: 0.7663
Epoch 378/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6010 - root_mean_squared_error: 0.7665 - val_loss: 0.6131 - val_root_mean_squared_error: 0.7743
Epoch 379/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5996 - root_mean_squared_error: 0.7652 - val_loss: 0.5982 - val_root_mean_squared_error: 0.7650
Epoch 380/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6103 - root_mean_squared_error: 0.7724 - val_loss: 0.6123 - val_root_mean_squared_error: 0.7748
Epoch 381/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6041 - root_mean_squared_error: 0.7685 - val_loss: 0.5945 - val_root_mean_squared_error: 0.7610
Epoch 382/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6069 - root_mean_squared_error: 0.7704 - val_loss: 0.5916 - val_root_mean_squared_error: 0.7608
Epoch 383/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5957 - root_mean_squared_error: 0.7626 - val_loss: 0.6003 - val_root_mean_squared_error: 0.7649
Epoch 384/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6009 - root_mean_squared_error: 0.7663 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7658
Epoch 385/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5989 - root_mean_squared_error: 0.7648 - val_loss: 0.6363 - val_root_mean_squared_error: 0.7889
Epoch 386/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6026 - root_mean_squared_error: 0.7673 - val_loss: 0.5986 - val_root_mean_squared_error: 0.7647
Epoch 387/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5986 - root_mean_squared_error: 0.7647 - val_loss: 0.5929 - val_root_mean_squared_error: 0.7608
Epoch 388/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6036 - root_mean_squared_error: 0.7678 - val_loss: 0.6029 - val_root_mean_squared_error: 0.7678
Epoch 389/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6064 - root_mean_squared_error: 0.7696 - val_loss: 0.5847 - val_root_mean_squared_error: 0.7555
Epoch 390/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5942 - root_mean_squared_error: 0.7615 - val_loss: 0.5937 - val_root_mean_squared_error: 0.7613
Epoch 391/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6182 - root_mean_squared_error: 0.7775 - val_loss: 0.6153 - val_root_mean_squared_error: 0.7758
Epoch 392/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5981 - root_mean_squared_error: 0.7642 - val_loss: 0.5990 - val_root_mean_squared_error: 0.7650
Epoch 393/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5996 - root_mean_squared_error: 0.7653 - val_loss: 0.6377 - val_root_mean_squared_error: 0.7897
Epoch 394/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6120 - root_mean_squared_error: 0.7736 - val_loss: 0.5972 - val_root_mean_squared_error: 0.7641
Epoch 395/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6111 - root_mean_squared_error: 0.7726 - val_loss: 0.6156 - val_root_mean_squared_error: 0.7759
Epoch 396/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6021 - root_mean_squared_error: 0.7670 - val_loss: 0.5964 - val_root_mean_squared_error: 0.7630
Epoch 397/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5990 - root_mean_squared_error: 0.7649 - val_loss: 0.6020 - val_root_mean_squared_error: 0.7672
Epoch 398/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6250 - root_mean_squared_error: 0.7820 - val_loss: 0.6388 - val_root_mean_squared_error: 0.7913
Epoch 399/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6034 - root_mean_squared_error: 0.7675 - val_loss: 0.6016 - val_root_mean_squared_error: 0.7667
Epoch 400/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6045 - root_mean_squared_error: 0.7683 - val_loss: 0.6092 - val_root_mean_squared_error: 0.7711
Epoch 401/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6047 - root_mean_squared_error: 0.7687 - val_loss: 0.6077 - val_root_mean_squared_error: 0.7707
Epoch 402/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6052 - root_mean_squared_error: 0.7690 - val_loss: 0.5992 - val_root_mean_squared_error: 0.7652
Epoch 403/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6055 - root_mean_squared_error: 0.7691 - val_loss: 0.5972 - val_root_mean_squared_error: 0.7636
Epoch 404/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6024 - root_mean_squared_error: 0.7669 - val_loss: 0.5938 - val_root_mean_squared_error: 0.7613
Epoch 405/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6050 - root_mean_squared_error: 0.7686 - val_loss: 0.6029 - val_root_mean_squared_error: 0.7672
Epoch 406/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5980 - root_mean_squared_error: 0.7643 - val_loss: 0.5886 - val_root_mean_squared_error: 0.7571
Epoch 407/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5987 - root_mean_squared_error: 0.7647 - val_loss: 0.6024 - val_root_mean_squared_error: 0.7674
Epoch 408/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5971 - root_mean_squared_error: 0.7634 - val_loss: 0.5975 - val_root_mean_squared_error: 0.7637
Epoch 409/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6139 - root_mean_squared_error: 0.7745 - val_loss: 0.6014 - val_root_mean_squared_error: 0.7663
Epoch 410/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6020 - root_mean_squared_error: 0.7670 - val_loss: 0.6048 - val_root_mean_squared_error: 0.7690
Epoch 411/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6035 - root_mean_squared_error: 0.7679 - val_loss: 0.6108 - val_root_mean_squared_error: 0.7713
Epoch 412/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6031 - root_mean_squared_error: 0.7674 - val_loss: 0.5934 - val_root_mean_squared_error: 0.7615
Epoch 413/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5983 - root_mean_squared_error: 0.7641 - val_loss: 0.5906 - val_root_mean_squared_error: 0.7590
Epoch 414/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6033 - root_mean_squared_error: 0.7681 - val_loss: 0.6200 - val_root_mean_squared_error: 0.7775
Epoch 415/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6020 - root_mean_squared_error: 0.7668 - val_loss: 0.5966 - val_root_mean_squared_error: 0.7644
Epoch 416/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6012 - root_mean_squared_error: 0.7661 - val_loss: 0.5919 - val_root_mean_squared_error: 0.7606
Epoch 417/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5992 - root_mean_squared_error: 0.7650 - val_loss: 0.5952 - val_root_mean_squared_error: 0.7618
Epoch 418/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6014 - root_mean_squared_error: 0.7662 - val_loss: 0.5971 - val_root_mean_squared_error: 0.7646
Epoch 419/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6036 - root_mean_squared_error: 0.7679 - val_loss: 0.5927 - val_root_mean_squared_error: 0.7604
Epoch 420/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5986 - root_mean_squared_error: 0.7643 - val_loss: 0.6090 - val_root_mean_squared_error: 0.7711
Epoch 421/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6038 - root_mean_squared_error: 0.7680 - val_loss: 0.5930 - val_root_mean_squared_error: 0.7602
Epoch 422/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6079 - root_mean_squared_error: 0.7701 - val_loss: 0.6159 - val_root_mean_squared_error: 0.7763
Epoch 423/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5915 - root_mean_squared_error: 0.7595 - val_loss: 0.5920 - val_root_mean_squared_error: 0.7598
Epoch 424/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5933 - root_mean_squared_error: 0.7613 - val_loss: 0.6094 - val_root_mean_squared_error: 0.7715
Epoch 425/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6055 - root_mean_squared_error: 0.7690 - val_loss: 0.5889 - val_root_mean_squared_error: 0.7575
Epoch 426/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5938 - root_mean_squared_error: 0.7615 - val_loss: 0.6338 - val_root_mean_squared_error: 0.7883
Epoch 427/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6027 - root_mean_squared_error: 0.7669 - val_loss: 0.5994 - val_root_mean_squared_error: 0.7652
Epoch 428/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5900 - root_mean_squared_error: 0.7589 - val_loss: 0.6272 - val_root_mean_squared_error: 0.7832
Epoch 429/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6096 - root_mean_squared_error: 0.7718 - val_loss: 0.6469 - val_root_mean_squared_error: 0.7958
Epoch 430/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5988 - root_mean_squared_error: 0.7648 - val_loss: 0.5999 - val_root_mean_squared_error: 0.7650
Epoch 431/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6080 - root_mean_squared_error: 0.7706 - val_loss: 0.6050 - val_root_mean_squared_error: 0.7693
Epoch 432/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5903 - root_mean_squared_error: 0.7590 - val_loss: 0.6303 - val_root_mean_squared_error: 0.7847
Epoch 433/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6121 - root_mean_squared_error: 0.7733 - val_loss: 0.6058 - val_root_mean_squared_error: 0.7692
Epoch 434/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6017 - root_mean_squared_error: 0.7668 - val_loss: 0.5755 - val_root_mean_squared_error: 0.7487
Epoch 435/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6012 - root_mean_squared_error: 0.7662 - val_loss: 0.6022 - val_root_mean_squared_error: 0.7676
Epoch 436/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5933 - root_mean_squared_error: 0.7609 - val_loss: 0.5922 - val_root_mean_squared_error: 0.7607
Epoch 437/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6156 - root_mean_squared_error: 0.7754 - val_loss: 0.5909 - val_root_mean_squared_error: 0.7602
Epoch 438/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6090 - root_mean_squared_error: 0.7711 - val_loss: 0.6103 - val_root_mean_squared_error: 0.7725
Epoch 439/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5993 - root_mean_squared_error: 0.7646 - val_loss: 0.5958 - val_root_mean_squared_error: 0.7632
Epoch 440/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5982 - root_mean_squared_error: 0.7638 - val_loss: 0.5969 - val_root_mean_squared_error: 0.7633
Epoch 441/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6037 - root_mean_squared_error: 0.7679 - val_loss: 0.5942 - val_root_mean_squared_error: 0.7611
Epoch 442/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6086 - root_mean_squared_error: 0.7713 - val_loss: 0.6197 - val_root_mean_squared_error: 0.7778
Epoch 443/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6024 - root_mean_squared_error: 0.7674 - val_loss: 0.5859 - val_root_mean_squared_error: 0.7558
Epoch 444/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6043 - root_mean_squared_error: 0.7684 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7658
Epoch 445/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5997 - root_mean_squared_error: 0.7655 - val_loss: 0.6098 - val_root_mean_squared_error: 0.7719
Epoch 446/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6039 - root_mean_squared_error: 0.7677 - val_loss: 0.5932 - val_root_mean_squared_error: 0.7611
Epoch 447/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6026 - root_mean_squared_error: 0.7669 - val_loss: 0.6057 - val_root_mean_squared_error: 0.7687
Epoch 448/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6068 - root_mean_squared_error: 0.7700 - val_loss: 0.5941 - val_root_mean_squared_error: 0.7611
Epoch 449/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6029 - root_mean_squared_error: 0.7673 - val_loss: 0.5960 - val_root_mean_squared_error: 0.7620
Epoch 450/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5937 - root_mean_squared_error: 0.7613 - val_loss: 0.6035 - val_root_mean_squared_error: 0.7675
Epoch 451/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6073 - root_mean_squared_error: 0.7701 - val_loss: 0.5791 - val_root_mean_squared_error: 0.7513
Epoch 452/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6050 - root_mean_squared_error: 0.7685 - val_loss: 0.5967 - val_root_mean_squared_error: 0.7631
Epoch 453/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6014 - root_mean_squared_error: 0.7660 - val_loss: 0.6110 - val_root_mean_squared_error: 0.7727
Epoch 454/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6018 - root_mean_squared_error: 0.7666 - val_loss: 0.5935 - val_root_mean_squared_error: 0.7607
Epoch 455/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5975 - root_mean_squared_error: 0.7633 - val_loss: 0.6047 - val_root_mean_squared_error: 0.7689
Epoch 456/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6127 - root_mean_squared_error: 0.7734 - val_loss: 0.5880 - val_root_mean_squared_error: 0.7585
Epoch 457/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5983 - root_mean_squared_error: 0.7644 - val_loss: 0.6348 - val_root_mean_squared_error: 0.7873
Epoch 458/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5990 - root_mean_squared_error: 0.7646 - val_loss: 0.5980 - val_root_mean_squared_error: 0.7636
Epoch 459/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6027 - root_mean_squared_error: 0.7669 - val_loss: 0.6045 - val_root_mean_squared_error: 0.7681
Epoch 460/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6022 - root_mean_squared_error: 0.7666 - val_loss: 0.5924 - val_root_mean_squared_error: 0.7597
Epoch 461/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5988 - root_mean_squared_error: 0.7643 - val_loss: 0.6040 - val_root_mean_squared_error: 0.7674
Epoch 462/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6089 - root_mean_squared_error: 0.7709 - val_loss: 0.5923 - val_root_mean_squared_error: 0.7604
Epoch 463/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6004 - root_mean_squared_error: 0.7655 - val_loss: 0.5980 - val_root_mean_squared_error: 0.7648
Epoch 464/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6071 - root_mean_squared_error: 0.7702 - val_loss: 0.5832 - val_root_mean_squared_error: 0.7535
Epoch 465/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5993 - root_mean_squared_error: 0.7651 - val_loss: 0.6023 - val_root_mean_squared_error: 0.7663
Epoch 466/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5976 - root_mean_squared_error: 0.7639 - val_loss: 0.6040 - val_root_mean_squared_error: 0.7683
Epoch 467/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5980 - root_mean_squared_error: 0.7638 - val_loss: 0.6038 - val_root_mean_squared_error: 0.7672
Epoch 468/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6028 - root_mean_squared_error: 0.7670 - val_loss: 0.5997 - val_root_mean_squared_error: 0.7664
Epoch 469/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5907 - root_mean_squared_error: 0.7589 - val_loss: 0.5923 - val_root_mean_squared_error: 0.7597
Epoch 470/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5980 - root_mean_squared_error: 0.7641 - val_loss: 0.6084 - val_root_mean_squared_error: 0.7706
Epoch 471/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5976 - root_mean_squared_error: 0.7635 - val_loss: 0.6359 - val_root_mean_squared_error: 0.7889
Epoch 472/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5969 - root_mean_squared_error: 0.7630 - val_loss: 0.5851 - val_root_mean_squared_error: 0.7554
Epoch 473/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6062 - root_mean_squared_error: 0.7691 - val_loss: 0.5960 - val_root_mean_squared_error: 0.7628
Epoch 474/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6081 - root_mean_squared_error: 0.7704 - val_loss: 0.6074 - val_root_mean_squared_error: 0.7699
Epoch 475/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5975 - root_mean_squared_error: 0.7636 - val_loss: 0.6193 - val_root_mean_squared_error: 0.7771
Epoch 476/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6003 - root_mean_squared_error: 0.7655 - val_loss: 0.5998 - val_root_mean_squared_error: 0.7660
Epoch 477/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6000 - root_mean_squared_error: 0.7650 - val_loss: 0.5914 - val_root_mean_squared_error: 0.7606
Epoch 478/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6034 - root_mean_squared_error: 0.7676 - val_loss: 0.5977 - val_root_mean_squared_error: 0.7631
Epoch 479/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5990 - root_mean_squared_error: 0.7643 - val_loss: 0.6223 - val_root_mean_squared_error: 0.7798
Epoch 480/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6076 - root_mean_squared_error: 0.7701 - val_loss: 0.6167 - val_root_mean_squared_error: 0.7758
Epoch 481/500
17/17 [==============================] - 0s 2ms/step - loss: 0.5973 - root_mean_squared_error: 0.7636 - val_loss: 0.5917 - val_root_mean_squared_error: 0.7592
Epoch 482/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6039 - root_mean_squared_error: 0.7676 - val_loss: 0.6142 - val_root_mean_squared_error: 0.7747
Epoch 483/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6023 - root_mean_squared_error: 0.7664 - val_loss: 0.6049 - val_root_mean_squared_error: 0.7675
Epoch 484/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6049 - root_mean_squared_error: 0.7681 - val_loss: 0.5860 - val_root_mean_squared_error: 0.7561
Epoch 485/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6033 - root_mean_squared_error: 0.7673 - val_loss: 0.5895 - val_root_mean_squared_error: 0.7583
Epoch 486/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6021 - root_mean_squared_error: 0.7668 - val_loss: 0.6050 - val_root_mean_squared_error: 0.7674
Epoch 487/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6140 - root_mean_squared_error: 0.7741 - val_loss: 0.5891 - val_root_mean_squared_error: 0.7583
Epoch 488/500
17/17 [==============================] - 0s 2ms/step - loss: 0.6042 - root_mean_squared_error: 0.7678 - val_loss: 0.5857 - val_root_mean_squared_error: 0.7552
Epoch 489/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6045 - root_mean_squared_error: 0.7680 - val_loss: 0.6030 - val_root_mean_squared_error: 0.7668
Epoch 490/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6002 - root_mean_squared_error: 0.7651 - val_loss: 0.6100 - val_root_mean_squared_error: 0.7723
Epoch 491/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5970 - root_mean_squared_error: 0.7632 - val_loss: 0.5860 - val_root_mean_squared_error: 0.7549
Epoch 492/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6093 - root_mean_squared_error: 0.7714 - val_loss: 0.5932 - val_root_mean_squared_error: 0.7611
Epoch 493/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6031 - root_mean_squared_error: 0.7671 - val_loss: 0.5945 - val_root_mean_squared_error: 0.7613
Epoch 494/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5988 - root_mean_squared_error: 0.7644 - val_loss: 0.5814 - val_root_mean_squared_error: 0.7534
Epoch 495/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6024 - root_mean_squared_error: 0.7663 - val_loss: 0.5893 - val_root_mean_squared_error: 0.7581
Epoch 496/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6098 - root_mean_squared_error: 0.7716 - val_loss: 0.5925 - val_root_mean_squared_error: 0.7602
Epoch 497/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5977 - root_mean_squared_error: 0.7639 - val_loss: 0.5981 - val_root_mean_squared_error: 0.7631
Epoch 498/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5983 - root_mean_squared_error: 0.7643 - val_loss: 0.5945 - val_root_mean_squared_error: 0.7607
Epoch 499/500
17/17 [==============================] - 0s 3ms/step - loss: 0.6028 - root_mean_squared_error: 0.7670 - val_loss: 0.6029 - val_root_mean_squared_error: 0.7672
Epoch 500/500
17/17 [==============================] - 0s 3ms/step - loss: 0.5973 - root_mean_squared_error: 0.7636 - val_loss: 0.6011 - val_root_mean_squared_error: 0.7663
Model training finished.
Train RMSE: 0.767
Evaluating model performance...
Test RMSE: 0.761
Predictions mean: 6.18, min: 5.72, max: 6.37, range: 0.65 - Actual: 6.0
Predictions mean: 5.15, min: 4.82, max: 5.68, range: 0.86 - Actual: 4.0
Predictions mean: 5.99, min: 5.63, max: 6.29, range: 0.66 - Actual: 5.0
Predictions mean: 6.51, min: 6.28, max: 6.62, range: 0.34 - Actual: 7.0
Predictions mean: 5.79, min: 5.47, max: 6.13, range: 0.67 - Actual: 5.0
Predictions mean: 6.15, min: 5.66, max: 6.35, range: 0.69 - Actual: 6.0
Predictions mean: 5.31, min: 5.01, max: 5.69, range: 0.68 - Actual: 6.0
Predictions mean: 6.47, min: 6.28, max: 6.59, range: 0.31 - Actual: 6.0
Predictions mean: 6.44, min: 6.17, max: 6.62, range: 0.45 - Actual: 7.0
Predictions mean: 6.54, min: 6.34, max: 6.63, range: 0.3 - Actual: 6.0
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

    # Create a probabilisticÃ¥ output (Normal distribution), and use the `Dense` layer
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
17/17 [==============================] - 2s 24ms/step - loss: 608.0859 - root_mean_squared_error: 6.0532 - val_loss: 783.1953 - val_root_mean_squared_error: 5.9762
Epoch 2/1000
17/17 [==============================] - 0s 3ms/step - loss: 468.2290 - root_mean_squared_error: 5.9296 - val_loss: 317.6672 - val_root_mean_squared_error: 5.7193
Epoch 3/1000
17/17 [==============================] - 0s 3ms/step - loss: 304.8195 - root_mean_squared_error: 5.8999 - val_loss: 272.0102 - val_root_mean_squared_error: 5.9631
Epoch 4/1000
17/17 [==============================] - 0s 3ms/step - loss: 312.9174 - root_mean_squared_error: 5.8328 - val_loss: 373.8178 - val_root_mean_squared_error: 5.8386
Epoch 5/1000
17/17 [==============================] - 0s 3ms/step - loss: 258.8027 - root_mean_squared_error: 5.7936 - val_loss: 237.3600 - val_root_mean_squared_error: 5.8606
Epoch 6/1000
17/17 [==============================] - 0s 3ms/step - loss: 211.4161 - root_mean_squared_error: 5.7636 - val_loss: 182.0104 - val_root_mean_squared_error: 5.5959
Epoch 7/1000
17/17 [==============================] - 0s 3ms/step - loss: 206.5159 - root_mean_squared_error: 5.7355 - val_loss: 158.0384 - val_root_mean_squared_error: 5.5645
Epoch 8/1000
17/17 [==============================] - 0s 3ms/step - loss: 144.5974 - root_mean_squared_error: 5.7358 - val_loss: 149.8738 - val_root_mean_squared_error: 5.4934
Epoch 9/1000
17/17 [==============================] - 0s 3ms/step - loss: 154.5544 - root_mean_squared_error: 5.6423 - val_loss: 106.9331 - val_root_mean_squared_error: 5.4058
Epoch 10/1000
17/17 [==============================] - 0s 3ms/step - loss: 110.2705 - root_mean_squared_error: 5.5864 - val_loss: 107.5287 - val_root_mean_squared_error: 5.6671
Epoch 11/1000
17/17 [==============================] - 0s 3ms/step - loss: 94.7189 - root_mean_squared_error: 5.6163 - val_loss: 89.6229 - val_root_mean_squared_error: 5.3544
Epoch 12/1000
17/17 [==============================] - 0s 3ms/step - loss: 79.6590 - root_mean_squared_error: 5.5766 - val_loss: 107.9557 - val_root_mean_squared_error: 5.4254
Epoch 13/1000
17/17 [==============================] - 0s 3ms/step - loss: 76.3482 - root_mean_squared_error: 5.4890 - val_loss: 61.0483 - val_root_mean_squared_error: 5.3026
Epoch 14/1000
17/17 [==============================] - 0s 3ms/step - loss: 66.3037 - root_mean_squared_error: 5.4483 - val_loss: 44.5011 - val_root_mean_squared_error: 5.5012
Epoch 15/1000
17/17 [==============================] - 0s 3ms/step - loss: 51.7069 - root_mean_squared_error: 5.4428 - val_loss: 74.5570 - val_root_mean_squared_error: 5.2234
Epoch 16/1000
17/17 [==============================] - 0s 3ms/step - loss: 48.3532 - root_mean_squared_error: 5.3955 - val_loss: 51.1887 - val_root_mean_squared_error: 5.4369
Epoch 17/1000
17/17 [==============================] - 0s 3ms/step - loss: 43.4584 - root_mean_squared_error: 5.3604 - val_loss: 42.9819 - val_root_mean_squared_error: 5.1786
Epoch 18/1000
17/17 [==============================] - 0s 3ms/step - loss: 36.3355 - root_mean_squared_error: 5.2832 - val_loss: 33.8545 - val_root_mean_squared_error: 5.1524
Epoch 19/1000
17/17 [==============================] - 0s 3ms/step - loss: 28.8905 - root_mean_squared_error: 5.3963 - val_loss: 29.1823 - val_root_mean_squared_error: 5.2778
Epoch 20/1000
17/17 [==============================] - 0s 3ms/step - loss: 27.0190 - root_mean_squared_error: 5.2930 - val_loss: 28.1456 - val_root_mean_squared_error: 5.1313
Epoch 21/1000
17/17 [==============================] - 0s 3ms/step - loss: 22.6092 - root_mean_squared_error: 5.2539 - val_loss: 20.7683 - val_root_mean_squared_error: 5.2715
Epoch 22/1000
17/17 [==============================] - 0s 3ms/step - loss: 20.1324 - root_mean_squared_error: 5.1641 - val_loss: 20.8474 - val_root_mean_squared_error: 4.9262
Epoch 23/1000
17/17 [==============================] - 0s 3ms/step - loss: 18.9927 - root_mean_squared_error: 5.0763 - val_loss: 15.8366 - val_root_mean_squared_error: 5.2296
Epoch 24/1000
17/17 [==============================] - 0s 3ms/step - loss: 18.0008 - root_mean_squared_error: 5.0368 - val_loss: 17.6986 - val_root_mean_squared_error: 4.9921
Epoch 25/1000
17/17 [==============================] - 0s 2ms/step - loss: 16.7585 - root_mean_squared_error: 5.0273 - val_loss: 14.3036 - val_root_mean_squared_error: 4.7533
Epoch 26/1000
17/17 [==============================] - 0s 3ms/step - loss: 14.1434 - root_mean_squared_error: 4.9532 - val_loss: 12.7188 - val_root_mean_squared_error: 5.0815
Epoch 27/1000
17/17 [==============================] - 0s 3ms/step - loss: 12.9412 - root_mean_squared_error: 4.9997 - val_loss: 11.8630 - val_root_mean_squared_error: 4.9074
Epoch 28/1000
17/17 [==============================] - 0s 3ms/step - loss: 11.2111 - root_mean_squared_error: 4.9333 - val_loss: 11.6796 - val_root_mean_squared_error: 4.8763
Epoch 29/1000
17/17 [==============================] - 0s 3ms/step - loss: 10.3731 - root_mean_squared_error: 4.8005 - val_loss: 9.5903 - val_root_mean_squared_error: 4.7450
Epoch 30/1000
17/17 [==============================] - 0s 3ms/step - loss: 9.6243 - root_mean_squared_error: 4.8211 - val_loss: 8.2387 - val_root_mean_squared_error: 4.6173
Epoch 31/1000
17/17 [==============================] - 0s 3ms/step - loss: 8.3145 - root_mean_squared_error: 4.7252 - val_loss: 7.4255 - val_root_mean_squared_error: 4.7375
Epoch 32/1000
17/17 [==============================] - 0s 3ms/step - loss: 7.5226 - root_mean_squared_error: 4.7049 - val_loss: 6.8698 - val_root_mean_squared_error: 4.4199
Epoch 33/1000
17/17 [==============================] - 0s 3ms/step - loss: 6.6118 - root_mean_squared_error: 4.5927 - val_loss: 6.6083 - val_root_mean_squared_error: 4.4738
Epoch 34/1000
17/17 [==============================] - 0s 2ms/step - loss: 6.2551 - root_mean_squared_error: 4.4541 - val_loss: 7.0293 - val_root_mean_squared_error: 4.9604
Epoch 35/1000
17/17 [==============================] - 0s 3ms/step - loss: 5.6321 - root_mean_squared_error: 4.3852 - val_loss: 5.6696 - val_root_mean_squared_error: 4.3697
Epoch 36/1000
17/17 [==============================] - 0s 3ms/step - loss: 5.1815 - root_mean_squared_error: 4.3240 - val_loss: 4.6707 - val_root_mean_squared_error: 4.2194
Epoch 37/1000
17/17 [==============================] - 0s 3ms/step - loss: 4.7883 - root_mean_squared_error: 4.3198 - val_loss: 4.5038 - val_root_mean_squared_error: 4.2827
Epoch 38/1000
17/17 [==============================] - 0s 3ms/step - loss: 4.8705 - root_mean_squared_error: 4.3070 - val_loss: 4.4088 - val_root_mean_squared_error: 4.2383
Epoch 39/1000
17/17 [==============================] - 0s 3ms/step - loss: 4.3279 - root_mean_squared_error: 4.2186 - val_loss: 4.0680 - val_root_mean_squared_error: 4.0137
Epoch 40/1000
17/17 [==============================] - 0s 2ms/step - loss: 3.7565 - root_mean_squared_error: 4.1073 - val_loss: 3.5630 - val_root_mean_squared_error: 3.9382
Epoch 41/1000
17/17 [==============================] - 0s 3ms/step - loss: 3.4001 - root_mean_squared_error: 3.9490 - val_loss: 3.2632 - val_root_mean_squared_error: 3.8589
Epoch 42/1000
17/17 [==============================] - 0s 3ms/step - loss: 3.3533 - root_mean_squared_error: 3.9207 - val_loss: 3.7039 - val_root_mean_squared_error: 4.1430
Epoch 43/1000
17/17 [==============================] - 0s 3ms/step - loss: 3.2957 - root_mean_squared_error: 3.9588 - val_loss: 3.1608 - val_root_mean_squared_error: 3.8895
Epoch 44/1000
17/17 [==============================] - 0s 3ms/step - loss: 3.0041 - root_mean_squared_error: 3.8399 - val_loss: 3.2161 - val_root_mean_squared_error: 3.9576
Epoch 45/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.8778 - root_mean_squared_error: 3.7951 - val_loss: 2.8013 - val_root_mean_squared_error: 3.6066
Epoch 46/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.6906 - root_mean_squared_error: 3.6795 - val_loss: 2.8276 - val_root_mean_squared_error: 3.7370
Epoch 47/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.5539 - root_mean_squared_error: 3.5756 - val_loss: 2.3982 - val_root_mean_squared_error: 3.4026
Epoch 48/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.4840 - root_mean_squared_error: 3.5497 - val_loss: 2.2977 - val_root_mean_squared_error: 3.2707
Epoch 49/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.2835 - root_mean_squared_error: 3.2755 - val_loss: 2.3527 - val_root_mean_squared_error: 3.3310
Epoch 50/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.3639 - root_mean_squared_error: 3.3830 - val_loss: 2.3065 - val_root_mean_squared_error: 3.4290
Epoch 51/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.1807 - root_mean_squared_error: 3.0637 - val_loss: 2.1769 - val_root_mean_squared_error: 3.1781
Epoch 52/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.3532 - root_mean_squared_error: 3.3288 - val_loss: 2.2097 - val_root_mean_squared_error: 3.2176
Epoch 53/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.2076 - root_mean_squared_error: 3.1037 - val_loss: 2.2127 - val_root_mean_squared_error: 3.1866
Epoch 54/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.2148 - root_mean_squared_error: 3.1117 - val_loss: 2.0074 - val_root_mean_squared_error: 2.9055
Epoch 55/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.0816 - root_mean_squared_error: 2.8850 - val_loss: 2.0071 - val_root_mean_squared_error: 2.8121
Epoch 56/1000
17/17 [==============================] - 0s 3ms/step - loss: 2.0420 - root_mean_squared_error: 2.8147 - val_loss: 1.8973 - val_root_mean_squared_error: 2.5101
Epoch 57/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.9642 - root_mean_squared_error: 2.6485 - val_loss: 2.0049 - val_root_mean_squared_error: 2.6011
Epoch 58/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.8633 - root_mean_squared_error: 2.4396 - val_loss: 1.8241 - val_root_mean_squared_error: 2.3691
Epoch 59/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.8665 - root_mean_squared_error: 2.4400 - val_loss: 1.7115 - val_root_mean_squared_error: 2.1169
Epoch 60/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.8168 - root_mean_squared_error: 2.2877 - val_loss: 1.6876 - val_root_mean_squared_error: 2.1206
Epoch 61/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.7651 - root_mean_squared_error: 2.2203 - val_loss: 1.6988 - val_root_mean_squared_error: 2.1233
Epoch 62/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.6727 - root_mean_squared_error: 1.9780 - val_loss: 1.6658 - val_root_mean_squared_error: 2.0822
Epoch 63/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.6638 - root_mean_squared_error: 2.0140 - val_loss: 1.5485 - val_root_mean_squared_error: 1.8139
Epoch 64/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.5563 - root_mean_squared_error: 1.7876 - val_loss: 1.5716 - val_root_mean_squared_error: 1.8578
Epoch 65/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.5657 - root_mean_squared_error: 1.8176 - val_loss: 1.5857 - val_root_mean_squared_error: 1.7956
Epoch 66/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.5090 - root_mean_squared_error: 1.7039 - val_loss: 1.6917 - val_root_mean_squared_error: 1.8580
Epoch 67/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.4956 - root_mean_squared_error: 1.6658 - val_loss: 1.3815 - val_root_mean_squared_error: 1.5016
Epoch 68/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.4585 - root_mean_squared_error: 1.5614 - val_loss: 1.3816 - val_root_mean_squared_error: 1.5205
Epoch 69/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.4510 - root_mean_squared_error: 1.5881 - val_loss: 1.3958 - val_root_mean_squared_error: 1.5456
Epoch 70/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.4497 - root_mean_squared_error: 1.4968 - val_loss: 1.3808 - val_root_mean_squared_error: 1.4376
Epoch 71/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.4250 - root_mean_squared_error: 1.4666 - val_loss: 1.4022 - val_root_mean_squared_error: 1.4333
Epoch 72/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.3920 - root_mean_squared_error: 1.4308 - val_loss: 1.3145 - val_root_mean_squared_error: 1.2464
Epoch 73/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3775 - root_mean_squared_error: 1.3835 - val_loss: 1.5836 - val_root_mean_squared_error: 1.5770
Epoch 74/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3757 - root_mean_squared_error: 1.3314 - val_loss: 1.3384 - val_root_mean_squared_error: 1.3083
Epoch 75/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.3654 - root_mean_squared_error: 1.3668 - val_loss: 1.2846 - val_root_mean_squared_error: 1.2600
Epoch 76/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3368 - root_mean_squared_error: 1.3219 - val_loss: 1.3490 - val_root_mean_squared_error: 1.3239
Epoch 77/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.4028 - root_mean_squared_error: 1.3673 - val_loss: 1.4149 - val_root_mean_squared_error: 1.3986
Epoch 78/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3073 - root_mean_squared_error: 1.2942 - val_loss: 1.4106 - val_root_mean_squared_error: 1.3036
Epoch 79/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3320 - root_mean_squared_error: 1.2992 - val_loss: 1.3340 - val_root_mean_squared_error: 1.3315
Epoch 80/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.3135 - root_mean_squared_error: 1.3068 - val_loss: 1.3593 - val_root_mean_squared_error: 1.2520
Epoch 81/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3344 - root_mean_squared_error: 1.2596 - val_loss: 1.3209 - val_root_mean_squared_error: 1.2081
Epoch 82/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3315 - root_mean_squared_error: 1.2797 - val_loss: 1.3221 - val_root_mean_squared_error: 1.3191
Epoch 83/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3119 - root_mean_squared_error: 1.2725 - val_loss: 1.3335 - val_root_mean_squared_error: 1.3560
Epoch 84/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3254 - root_mean_squared_error: 1.2816 - val_loss: 1.2617 - val_root_mean_squared_error: 1.2610
Epoch 85/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3112 - root_mean_squared_error: 1.2519 - val_loss: 1.2758 - val_root_mean_squared_error: 1.2947
Epoch 86/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2990 - root_mean_squared_error: 1.2361 - val_loss: 1.3846 - val_root_mean_squared_error: 1.3068
Epoch 87/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3095 - root_mean_squared_error: 1.2467 - val_loss: 1.3252 - val_root_mean_squared_error: 1.3611
Epoch 88/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3113 - root_mean_squared_error: 1.2739 - val_loss: 1.3132 - val_root_mean_squared_error: 1.2689
Epoch 89/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3109 - root_mean_squared_error: 1.2418 - val_loss: 1.2700 - val_root_mean_squared_error: 1.2684
Epoch 90/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3022 - root_mean_squared_error: 1.2471 - val_loss: 1.2803 - val_root_mean_squared_error: 1.2024
Epoch 91/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2775 - root_mean_squared_error: 1.2168 - val_loss: 1.2765 - val_root_mean_squared_error: 1.2054
Epoch 92/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2886 - root_mean_squared_error: 1.2250 - val_loss: 1.2854 - val_root_mean_squared_error: 1.1540
Epoch 93/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2906 - root_mean_squared_error: 1.2338 - val_loss: 1.3091 - val_root_mean_squared_error: 1.2749
Epoch 94/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2860 - root_mean_squared_error: 1.2297 - val_loss: 1.3128 - val_root_mean_squared_error: 1.1606
Epoch 95/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2860 - root_mean_squared_error: 1.2002 - val_loss: 1.2774 - val_root_mean_squared_error: 1.2199
Epoch 96/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.3174 - root_mean_squared_error: 1.2779 - val_loss: 1.2667 - val_root_mean_squared_error: 1.2778
Epoch 97/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2669 - root_mean_squared_error: 1.2321 - val_loss: 1.2567 - val_root_mean_squared_error: 1.2276
Epoch 98/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.3196 - root_mean_squared_error: 1.2643 - val_loss: 1.2627 - val_root_mean_squared_error: 1.2699
Epoch 99/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2968 - root_mean_squared_error: 1.2496 - val_loss: 1.2311 - val_root_mean_squared_error: 1.1586
Epoch 100/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2678 - root_mean_squared_error: 1.1816 - val_loss: 1.2980 - val_root_mean_squared_error: 1.3181
Epoch 101/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2585 - root_mean_squared_error: 1.2041 - val_loss: 1.2708 - val_root_mean_squared_error: 1.2474
Epoch 102/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2720 - root_mean_squared_error: 1.2148 - val_loss: 1.2521 - val_root_mean_squared_error: 1.1612
Epoch 103/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2510 - root_mean_squared_error: 1.2025 - val_loss: 1.2310 - val_root_mean_squared_error: 1.2108
Epoch 104/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2884 - root_mean_squared_error: 1.1996 - val_loss: 1.2396 - val_root_mean_squared_error: 1.1456
Epoch 105/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2627 - root_mean_squared_error: 1.2033 - val_loss: 1.2494 - val_root_mean_squared_error: 1.1918
Epoch 106/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2895 - root_mean_squared_error: 1.2337 - val_loss: 1.3180 - val_root_mean_squared_error: 1.1845
Epoch 107/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2617 - root_mean_squared_error: 1.2040 - val_loss: 1.3150 - val_root_mean_squared_error: 1.2273
Epoch 108/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2822 - root_mean_squared_error: 1.2144 - val_loss: 1.2765 - val_root_mean_squared_error: 1.2778
Epoch 109/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2703 - root_mean_squared_error: 1.2184 - val_loss: 1.2243 - val_root_mean_squared_error: 1.2106
Epoch 110/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2513 - root_mean_squared_error: 1.1855 - val_loss: 1.2794 - val_root_mean_squared_error: 1.2051
Epoch 111/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2387 - root_mean_squared_error: 1.1771 - val_loss: 1.2685 - val_root_mean_squared_error: 1.1852
Epoch 112/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2958 - root_mean_squared_error: 1.2331 - val_loss: 1.2746 - val_root_mean_squared_error: 1.2584
Epoch 113/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2493 - root_mean_squared_error: 1.2045 - val_loss: 1.2743 - val_root_mean_squared_error: 1.1811
Epoch 114/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2688 - root_mean_squared_error: 1.1922 - val_loss: 1.2125 - val_root_mean_squared_error: 1.1762
Epoch 115/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2558 - root_mean_squared_error: 1.1826 - val_loss: 1.2275 - val_root_mean_squared_error: 1.1325
Epoch 116/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2680 - root_mean_squared_error: 1.2233 - val_loss: 1.2475 - val_root_mean_squared_error: 1.1720
Epoch 117/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2381 - root_mean_squared_error: 1.1857 - val_loss: 1.3026 - val_root_mean_squared_error: 1.2070
Epoch 118/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2586 - root_mean_squared_error: 1.1744 - val_loss: 1.3026 - val_root_mean_squared_error: 1.2169
Epoch 119/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2716 - root_mean_squared_error: 1.2057 - val_loss: 1.2332 - val_root_mean_squared_error: 1.2096
Epoch 120/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2418 - root_mean_squared_error: 1.1713 - val_loss: 1.2310 - val_root_mean_squared_error: 1.2367
Epoch 121/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2426 - root_mean_squared_error: 1.1826 - val_loss: 1.2102 - val_root_mean_squared_error: 1.1474
Epoch 122/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2268 - root_mean_squared_error: 1.1597 - val_loss: 1.2399 - val_root_mean_squared_error: 1.2015
Epoch 123/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2469 - root_mean_squared_error: 1.1724 - val_loss: 1.2366 - val_root_mean_squared_error: 1.1964
Epoch 124/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2264 - root_mean_squared_error: 1.1783 - val_loss: 1.2268 - val_root_mean_squared_error: 1.1512
Epoch 125/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2349 - root_mean_squared_error: 1.1849 - val_loss: 1.2417 - val_root_mean_squared_error: 1.1122
Epoch 126/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2443 - root_mean_squared_error: 1.1853 - val_loss: 1.2485 - val_root_mean_squared_error: 1.1464
Epoch 127/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2283 - root_mean_squared_error: 1.1676 - val_loss: 1.2129 - val_root_mean_squared_error: 1.1799
Epoch 128/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2199 - root_mean_squared_error: 1.1384 - val_loss: 1.2473 - val_root_mean_squared_error: 1.1359
Epoch 129/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2272 - root_mean_squared_error: 1.1515 - val_loss: 1.2661 - val_root_mean_squared_error: 1.2059
Epoch 130/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2343 - root_mean_squared_error: 1.1536 - val_loss: 1.2322 - val_root_mean_squared_error: 1.1544
Epoch 131/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2344 - root_mean_squared_error: 1.1844 - val_loss: 1.2410 - val_root_mean_squared_error: 1.1679
Epoch 132/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2152 - root_mean_squared_error: 1.1577 - val_loss: 1.2894 - val_root_mean_squared_error: 1.0868
Epoch 133/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2248 - root_mean_squared_error: 1.1609 - val_loss: 1.2368 - val_root_mean_squared_error: 1.2167
Epoch 134/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2286 - root_mean_squared_error: 1.1671 - val_loss: 1.2372 - val_root_mean_squared_error: 1.1599
Epoch 135/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2343 - root_mean_squared_error: 1.1634 - val_loss: 1.2232 - val_root_mean_squared_error: 1.1737
Epoch 136/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2139 - root_mean_squared_error: 1.1453 - val_loss: 1.2118 - val_root_mean_squared_error: 1.1402
Epoch 137/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2220 - root_mean_squared_error: 1.1621 - val_loss: 1.2505 - val_root_mean_squared_error: 1.1622
Epoch 138/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2385 - root_mean_squared_error: 1.1708 - val_loss: 1.2353 - val_root_mean_squared_error: 1.1396
Epoch 139/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2137 - root_mean_squared_error: 1.1229 - val_loss: 1.2276 - val_root_mean_squared_error: 1.2055
Epoch 140/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2319 - root_mean_squared_error: 1.1588 - val_loss: 1.1771 - val_root_mean_squared_error: 1.0899
Epoch 141/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2331 - root_mean_squared_error: 1.1540 - val_loss: 1.2286 - val_root_mean_squared_error: 1.2043
Epoch 142/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2319 - root_mean_squared_error: 1.1597 - val_loss: 1.2443 - val_root_mean_squared_error: 1.1145
Epoch 143/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2322 - root_mean_squared_error: 1.1661 - val_loss: 1.2056 - val_root_mean_squared_error: 1.1573
Epoch 144/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2226 - root_mean_squared_error: 1.1400 - val_loss: 1.3128 - val_root_mean_squared_error: 1.2015
Epoch 145/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2083 - root_mean_squared_error: 1.1469 - val_loss: 1.2359 - val_root_mean_squared_error: 1.1682
Epoch 146/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2331 - root_mean_squared_error: 1.1555 - val_loss: 1.2508 - val_root_mean_squared_error: 1.2122
Epoch 147/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2003 - root_mean_squared_error: 1.1412 - val_loss: 1.1942 - val_root_mean_squared_error: 1.1048
Epoch 148/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2206 - root_mean_squared_error: 1.1433 - val_loss: 1.2074 - val_root_mean_squared_error: 1.0891
Epoch 149/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2174 - root_mean_squared_error: 1.1600 - val_loss: 1.2050 - val_root_mean_squared_error: 1.1321
Epoch 150/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2145 - root_mean_squared_error: 1.1420 - val_loss: 1.1902 - val_root_mean_squared_error: 1.1619
Epoch 151/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2043 - root_mean_squared_error: 1.1278 - val_loss: 1.1960 - val_root_mean_squared_error: 1.1190
Epoch 152/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2069 - root_mean_squared_error: 1.1332 - val_loss: 1.2709 - val_root_mean_squared_error: 1.1523
Epoch 153/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2078 - root_mean_squared_error: 1.1530 - val_loss: 1.1936 - val_root_mean_squared_error: 1.1113
Epoch 154/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1993 - root_mean_squared_error: 1.1099 - val_loss: 1.2047 - val_root_mean_squared_error: 1.1662
Epoch 155/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2137 - root_mean_squared_error: 1.1313 - val_loss: 1.1999 - val_root_mean_squared_error: 1.1548
Epoch 156/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2088 - root_mean_squared_error: 1.1454 - val_loss: 1.2070 - val_root_mean_squared_error: 1.1403
Epoch 157/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.2113 - root_mean_squared_error: 1.1532 - val_loss: 1.1980 - val_root_mean_squared_error: 1.1089
Epoch 158/1000
17/17 [==============================] - 0s 4ms/step - loss: 1.2021 - root_mean_squared_error: 1.1087 - val_loss: 1.1972 - val_root_mean_squared_error: 1.1343
Epoch 159/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1994 - root_mean_squared_error: 1.1319 - val_loss: 1.1809 - val_root_mean_squared_error: 1.1482
Epoch 160/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2040 - root_mean_squared_error: 1.1477 - val_loss: 1.1966 - val_root_mean_squared_error: 1.1876
Epoch 161/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2097 - root_mean_squared_error: 1.1206 - val_loss: 1.1991 - val_root_mean_squared_error: 1.1363
Epoch 162/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2141 - root_mean_squared_error: 1.1244 - val_loss: 1.1947 - val_root_mean_squared_error: 1.1384
Epoch 163/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2108 - root_mean_squared_error: 1.1401 - val_loss: 1.2115 - val_root_mean_squared_error: 1.1382
Epoch 164/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1988 - root_mean_squared_error: 1.1268 - val_loss: 1.2173 - val_root_mean_squared_error: 1.0910
Epoch 165/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2031 - root_mean_squared_error: 1.1467 - val_loss: 1.2069 - val_root_mean_squared_error: 1.0988
Epoch 166/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2158 - root_mean_squared_error: 1.1375 - val_loss: 1.1970 - val_root_mean_squared_error: 1.1245
Epoch 167/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1857 - root_mean_squared_error: 1.1122 - val_loss: 1.1758 - val_root_mean_squared_error: 1.1255
Epoch 168/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2035 - root_mean_squared_error: 1.1240 - val_loss: 1.2072 - val_root_mean_squared_error: 1.1315
Epoch 169/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1950 - root_mean_squared_error: 1.1135 - val_loss: 1.1952 - val_root_mean_squared_error: 1.1436
Epoch 170/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2057 - root_mean_squared_error: 1.1211 - val_loss: 1.2022 - val_root_mean_squared_error: 1.0830
Epoch 171/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2020 - root_mean_squared_error: 1.1498 - val_loss: 1.2053 - val_root_mean_squared_error: 1.1686
Epoch 172/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2271 - root_mean_squared_error: 1.1533 - val_loss: 1.1956 - val_root_mean_squared_error: 1.1645
Epoch 173/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2053 - root_mean_squared_error: 1.1253 - val_loss: 1.2109 - val_root_mean_squared_error: 1.0570
Epoch 174/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1972 - root_mean_squared_error: 1.1327 - val_loss: 1.1961 - val_root_mean_squared_error: 1.1060
Epoch 175/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2044 - root_mean_squared_error: 1.1112 - val_loss: 1.1865 - val_root_mean_squared_error: 1.1788
Epoch 176/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2065 - root_mean_squared_error: 1.1361 - val_loss: 1.2203 - val_root_mean_squared_error: 1.1034
Epoch 177/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2003 - root_mean_squared_error: 1.1331 - val_loss: 1.1841 - val_root_mean_squared_error: 1.1695
Epoch 178/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2030 - root_mean_squared_error: 1.1309 - val_loss: 1.1961 - val_root_mean_squared_error: 1.1144
Epoch 179/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1988 - root_mean_squared_error: 1.1384 - val_loss: 1.1976 - val_root_mean_squared_error: 1.1114
Epoch 180/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2044 - root_mean_squared_error: 1.1323 - val_loss: 1.2330 - val_root_mean_squared_error: 1.1252
Epoch 181/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1870 - root_mean_squared_error: 1.1195 - val_loss: 1.1761 - val_root_mean_squared_error: 1.0902
Epoch 182/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1890 - root_mean_squared_error: 1.1090 - val_loss: 1.2010 - val_root_mean_squared_error: 1.0821
Epoch 183/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1968 - root_mean_squared_error: 1.1293 - val_loss: 1.2527 - val_root_mean_squared_error: 1.1629
Epoch 184/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1941 - root_mean_squared_error: 1.1486 - val_loss: 1.1908 - val_root_mean_squared_error: 1.1250
Epoch 185/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1982 - root_mean_squared_error: 1.1213 - val_loss: 1.1695 - val_root_mean_squared_error: 1.1152
Epoch 186/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2117 - root_mean_squared_error: 1.1222 - val_loss: 1.1809 - val_root_mean_squared_error: 1.1244
Epoch 187/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1838 - root_mean_squared_error: 1.1159 - val_loss: 1.2134 - val_root_mean_squared_error: 1.1125
Epoch 188/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1960 - root_mean_squared_error: 1.1230 - val_loss: 1.1748 - val_root_mean_squared_error: 1.1183
Epoch 189/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1936 - root_mean_squared_error: 1.1264 - val_loss: 1.1783 - val_root_mean_squared_error: 1.1459
Epoch 190/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1852 - root_mean_squared_error: 1.1252 - val_loss: 1.2232 - val_root_mean_squared_error: 1.0870
Epoch 191/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1822 - root_mean_squared_error: 1.1017 - val_loss: 1.1820 - val_root_mean_squared_error: 1.1084
Epoch 192/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1956 - root_mean_squared_error: 1.0993 - val_loss: 1.2030 - val_root_mean_squared_error: 1.1152
Epoch 193/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1951 - root_mean_squared_error: 1.1317 - val_loss: 1.1786 - val_root_mean_squared_error: 1.1206
Epoch 194/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1878 - root_mean_squared_error: 1.1139 - val_loss: 1.1808 - val_root_mean_squared_error: 1.1509
Epoch 195/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1953 - root_mean_squared_error: 1.1141 - val_loss: 1.1812 - val_root_mean_squared_error: 1.1259
Epoch 196/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1836 - root_mean_squared_error: 1.1154 - val_loss: 1.2042 - val_root_mean_squared_error: 1.1150
Epoch 197/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1950 - root_mean_squared_error: 1.1193 - val_loss: 1.1786 - val_root_mean_squared_error: 1.1198
Epoch 198/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2006 - root_mean_squared_error: 1.1017 - val_loss: 1.1755 - val_root_mean_squared_error: 1.1494
Epoch 199/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1881 - root_mean_squared_error: 1.1154 - val_loss: 1.1750 - val_root_mean_squared_error: 1.1179
Epoch 200/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1825 - root_mean_squared_error: 1.0889 - val_loss: 1.2338 - val_root_mean_squared_error: 1.0859
Epoch 201/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1979 - root_mean_squared_error: 1.1164 - val_loss: 1.1885 - val_root_mean_squared_error: 1.0941
Epoch 202/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2105 - root_mean_squared_error: 1.1315 - val_loss: 1.1962 - val_root_mean_squared_error: 1.1206
Epoch 203/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2140 - root_mean_squared_error: 1.1214 - val_loss: 1.1799 - val_root_mean_squared_error: 1.1363
Epoch 204/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1932 - root_mean_squared_error: 1.1356 - val_loss: 1.1685 - val_root_mean_squared_error: 1.0688
Epoch 205/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2073 - root_mean_squared_error: 1.1285 - val_loss: 1.2114 - val_root_mean_squared_error: 1.0875
Epoch 206/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1852 - root_mean_squared_error: 1.1078 - val_loss: 1.1839 - val_root_mean_squared_error: 1.0762
Epoch 207/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1885 - root_mean_squared_error: 1.1436 - val_loss: 1.1885 - val_root_mean_squared_error: 1.1268
Epoch 208/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1809 - root_mean_squared_error: 1.1086 - val_loss: 1.1604 - val_root_mean_squared_error: 1.1629
Epoch 209/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1899 - root_mean_squared_error: 1.1154 - val_loss: 1.2356 - val_root_mean_squared_error: 1.1483
Epoch 210/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1906 - root_mean_squared_error: 1.0888 - val_loss: 1.1739 - val_root_mean_squared_error: 1.0804
Epoch 211/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1939 - root_mean_squared_error: 1.1189 - val_loss: 1.1752 - val_root_mean_squared_error: 1.0724
Epoch 212/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1965 - root_mean_squared_error: 1.1240 - val_loss: 1.2089 - val_root_mean_squared_error: 1.1501
Epoch 213/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1870 - root_mean_squared_error: 1.1076 - val_loss: 1.1709 - val_root_mean_squared_error: 1.0547
Epoch 214/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1883 - root_mean_squared_error: 1.1094 - val_loss: 1.2159 - val_root_mean_squared_error: 1.1501
Epoch 215/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1755 - root_mean_squared_error: 1.1120 - val_loss: 1.1714 - val_root_mean_squared_error: 1.0883
Epoch 216/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1814 - root_mean_squared_error: 1.1048 - val_loss: 1.1725 - val_root_mean_squared_error: 1.0946
Epoch 217/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1926 - root_mean_squared_error: 1.1188 - val_loss: 1.1862 - val_root_mean_squared_error: 1.1736
Epoch 218/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1941 - root_mean_squared_error: 1.1139 - val_loss: 1.2013 - val_root_mean_squared_error: 1.0529
Epoch 219/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1836 - root_mean_squared_error: 1.0927 - val_loss: 1.1798 - val_root_mean_squared_error: 1.1234
Epoch 220/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2001 - root_mean_squared_error: 1.1089 - val_loss: 1.1633 - val_root_mean_squared_error: 1.0881
Epoch 221/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1758 - root_mean_squared_error: 1.0984 - val_loss: 1.1595 - val_root_mean_squared_error: 1.0902
Epoch 222/1000
17/17 [==============================] - 0s 4ms/step - loss: 1.1730 - root_mean_squared_error: 1.1045 - val_loss: 1.1717 - val_root_mean_squared_error: 1.0807
Epoch 223/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.2013 - root_mean_squared_error: 1.1246 - val_loss: 1.1905 - val_root_mean_squared_error: 1.0984
Epoch 224/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1796 - root_mean_squared_error: 1.1008 - val_loss: 1.1761 - val_root_mean_squared_error: 1.0657
Epoch 225/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1878 - root_mean_squared_error: 1.1114 - val_loss: 1.1641 - val_root_mean_squared_error: 1.1222
Epoch 226/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1740 - root_mean_squared_error: 1.1060 - val_loss: 1.1748 - val_root_mean_squared_error: 1.0719
Epoch 227/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1941 - root_mean_squared_error: 1.1098 - val_loss: 1.1623 - val_root_mean_squared_error: 1.0975
Epoch 228/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1934 - root_mean_squared_error: 1.1070 - val_loss: 1.1785 - val_root_mean_squared_error: 1.0811
Epoch 229/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1854 - root_mean_squared_error: 1.1080 - val_loss: 1.1750 - val_root_mean_squared_error: 1.0582
Epoch 230/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1956 - root_mean_squared_error: 1.1070 - val_loss: 1.1625 - val_root_mean_squared_error: 1.1547
Epoch 231/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1873 - root_mean_squared_error: 1.1221 - val_loss: 1.1917 - val_root_mean_squared_error: 1.1583
Epoch 232/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1784 - root_mean_squared_error: 1.0974 - val_loss: 1.1838 - val_root_mean_squared_error: 1.1371
Epoch 233/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1875 - root_mean_squared_error: 1.1062 - val_loss: 1.1721 - val_root_mean_squared_error: 1.1275
Epoch 234/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1771 - root_mean_squared_error: 1.1109 - val_loss: 1.1663 - val_root_mean_squared_error: 1.1024
Epoch 235/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1988 - root_mean_squared_error: 1.0961 - val_loss: 1.1709 - val_root_mean_squared_error: 1.0817
Epoch 236/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1834 - root_mean_squared_error: 1.1039 - val_loss: 1.1895 - val_root_mean_squared_error: 1.0993
Epoch 237/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1794 - root_mean_squared_error: 1.0748 - val_loss: 1.1571 - val_root_mean_squared_error: 1.0600
Epoch 238/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1859 - root_mean_squared_error: 1.1210 - val_loss: 1.1898 - val_root_mean_squared_error: 1.1183
Epoch 239/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1827 - root_mean_squared_error: 1.1056 - val_loss: 1.2100 - val_root_mean_squared_error: 1.0896
Epoch 240/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1949 - root_mean_squared_error: 1.1192 - val_loss: 1.1581 - val_root_mean_squared_error: 1.0688
Epoch 241/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1870 - root_mean_squared_error: 1.0979 - val_loss: 1.1845 - val_root_mean_squared_error: 1.1396
Epoch 242/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1812 - root_mean_squared_error: 1.1070 - val_loss: 1.1708 - val_root_mean_squared_error: 1.1427
Epoch 243/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1891 - root_mean_squared_error: 1.1216 - val_loss: 1.1758 - val_root_mean_squared_error: 1.1132
Epoch 244/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1750 - root_mean_squared_error: 1.0933 - val_loss: 1.2238 - val_root_mean_squared_error: 1.0855
Epoch 245/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1835 - root_mean_squared_error: 1.1171 - val_loss: 1.1631 - val_root_mean_squared_error: 1.1003
Epoch 246/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1844 - root_mean_squared_error: 1.1278 - val_loss: 1.1816 - val_root_mean_squared_error: 1.1223
Epoch 247/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1677 - root_mean_squared_error: 1.0940 - val_loss: 1.2113 - val_root_mean_squared_error: 1.0913
Epoch 248/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1766 - root_mean_squared_error: 1.0830 - val_loss: 1.1612 - val_root_mean_squared_error: 1.0696
Epoch 249/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1821 - root_mean_squared_error: 1.0978 - val_loss: 1.1530 - val_root_mean_squared_error: 1.0959
Epoch 250/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1761 - root_mean_squared_error: 1.0905 - val_loss: 1.1713 - val_root_mean_squared_error: 1.1143
Epoch 251/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1773 - root_mean_squared_error: 1.1079 - val_loss: 1.1593 - val_root_mean_squared_error: 1.1192
Epoch 252/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1806 - root_mean_squared_error: 1.1069 - val_loss: 1.1899 - val_root_mean_squared_error: 1.1024
Epoch 253/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1896 - root_mean_squared_error: 1.1152 - val_loss: 1.1679 - val_root_mean_squared_error: 1.1096
Epoch 254/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1847 - root_mean_squared_error: 1.1044 - val_loss: 1.1850 - val_root_mean_squared_error: 1.0860
Epoch 255/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1931 - root_mean_squared_error: 1.1195 - val_loss: 1.1674 - val_root_mean_squared_error: 1.1229
Epoch 256/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1787 - root_mean_squared_error: 1.1029 - val_loss: 1.1801 - val_root_mean_squared_error: 1.0965
Epoch 257/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1754 - root_mean_squared_error: 1.1139 - val_loss: 1.1860 - val_root_mean_squared_error: 1.1375
Epoch 258/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1900 - root_mean_squared_error: 1.1208 - val_loss: 1.1608 - val_root_mean_squared_error: 1.0595
Epoch 259/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1906 - root_mean_squared_error: 1.1084 - val_loss: 1.1755 - val_root_mean_squared_error: 1.1283
Epoch 260/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1847 - root_mean_squared_error: 1.1029 - val_loss: 1.1760 - val_root_mean_squared_error: 1.1344
Epoch 261/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1681 - root_mean_squared_error: 1.1229 - val_loss: 1.1827 - val_root_mean_squared_error: 1.1066
Epoch 262/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1754 - root_mean_squared_error: 1.0819 - val_loss: 1.1679 - val_root_mean_squared_error: 1.0899
Epoch 263/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1892 - root_mean_squared_error: 1.1103 - val_loss: 1.2027 - val_root_mean_squared_error: 1.1901
Epoch 264/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1801 - root_mean_squared_error: 1.1088 - val_loss: 1.1714 - val_root_mean_squared_error: 1.1088
Epoch 265/1000
17/17 [==============================] - 0s 4ms/step - loss: 1.1791 - root_mean_squared_error: 1.1005 - val_loss: 1.1753 - val_root_mean_squared_error: 1.0993
Epoch 266/1000
17/17 [==============================] - 0s 4ms/step - loss: 1.1813 - root_mean_squared_error: 1.1025 - val_loss: 1.1992 - val_root_mean_squared_error: 1.1207
Epoch 267/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1869 - root_mean_squared_error: 1.1021 - val_loss: 1.1817 - val_root_mean_squared_error: 1.1422
Epoch 268/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1762 - root_mean_squared_error: 1.0999 - val_loss: 1.1676 - val_root_mean_squared_error: 1.1018
Epoch 269/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1781 - root_mean_squared_error: 1.1067 - val_loss: 1.1866 - val_root_mean_squared_error: 1.1098
Epoch 270/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1816 - root_mean_squared_error: 1.0900 - val_loss: 1.1818 - val_root_mean_squared_error: 1.1237
Epoch 271/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1722 - root_mean_squared_error: 1.0973 - val_loss: 1.1839 - val_root_mean_squared_error: 1.1169
Epoch 272/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1852 - root_mean_squared_error: 1.1120 - val_loss: 1.1674 - val_root_mean_squared_error: 1.1165
Epoch 273/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1744 - root_mean_squared_error: 1.1005 - val_loss: 1.1714 - val_root_mean_squared_error: 1.1082
Epoch 274/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1815 - root_mean_squared_error: 1.1034 - val_loss: 1.1925 - val_root_mean_squared_error: 1.1133
Epoch 275/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1812 - root_mean_squared_error: 1.1125 - val_loss: 1.1553 - val_root_mean_squared_error: 1.1033
Epoch 276/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1926 - root_mean_squared_error: 1.0943 - val_loss: 1.1811 - val_root_mean_squared_error: 1.0823
Epoch 277/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1880 - root_mean_squared_error: 1.1046 - val_loss: 1.1900 - val_root_mean_squared_error: 1.0880
Epoch 278/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1795 - root_mean_squared_error: 1.1052 - val_loss: 1.1907 - val_root_mean_squared_error: 1.0761
Epoch 279/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1745 - root_mean_squared_error: 1.0767 - val_loss: 1.2060 - val_root_mean_squared_error: 1.1412
Epoch 280/1000
17/17 [==============================] - 0s 4ms/step - loss: 1.1727 - root_mean_squared_error: 1.1058 - val_loss: 1.1851 - val_root_mean_squared_error: 1.1265
Epoch 281/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1694 - root_mean_squared_error: 1.0940 - val_loss: 1.2064 - val_root_mean_squared_error: 1.1837
Epoch 282/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1798 - root_mean_squared_error: 1.0894 - val_loss: 1.1667 - val_root_mean_squared_error: 1.1039
Epoch 283/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1789 - root_mean_squared_error: 1.0884 - val_loss: 1.1662 - val_root_mean_squared_error: 1.0773
Epoch 284/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1865 - root_mean_squared_error: 1.0849 - val_loss: 1.1867 - val_root_mean_squared_error: 1.1221
Epoch 285/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1770 - root_mean_squared_error: 1.1046 - val_loss: 1.1931 - val_root_mean_squared_error: 1.0942
Epoch 286/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1816 - root_mean_squared_error: 1.0999 - val_loss: 1.1606 - val_root_mean_squared_error: 1.0707
Epoch 287/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1790 - root_mean_squared_error: 1.1030 - val_loss: 1.1685 - val_root_mean_squared_error: 1.0536
Epoch 288/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1815 - root_mean_squared_error: 1.1174 - val_loss: 1.1631 - val_root_mean_squared_error: 1.1152
Epoch 289/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1717 - root_mean_squared_error: 1.0930 - val_loss: 1.1766 - val_root_mean_squared_error: 1.0862
Epoch 290/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1916 - root_mean_squared_error: 1.1123 - val_loss: 1.1555 - val_root_mean_squared_error: 1.1453
Epoch 291/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1698 - root_mean_squared_error: 1.0926 - val_loss: 1.1677 - val_root_mean_squared_error: 1.1091
Epoch 292/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1932 - root_mean_squared_error: 1.0989 - val_loss: 1.1584 - val_root_mean_squared_error: 1.1154
Epoch 293/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1720 - root_mean_squared_error: 1.1109 - val_loss: 1.1957 - val_root_mean_squared_error: 1.0847
Epoch 294/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1791 - root_mean_squared_error: 1.0973 - val_loss: 1.1662 - val_root_mean_squared_error: 1.0626
Epoch 295/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1812 - root_mean_squared_error: 1.1037 - val_loss: 1.1610 - val_root_mean_squared_error: 1.1183
Epoch 296/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1764 - root_mean_squared_error: 1.1065 - val_loss: 1.1699 - val_root_mean_squared_error: 1.1365
Epoch 297/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1719 - root_mean_squared_error: 1.1105 - val_loss: 1.1739 - val_root_mean_squared_error: 1.0867
Epoch 298/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1865 - root_mean_squared_error: 1.1157 - val_loss: 1.1856 - val_root_mean_squared_error: 1.1058
Epoch 299/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1723 - root_mean_squared_error: 1.0973 - val_loss: 1.1685 - val_root_mean_squared_error: 1.0942
Epoch 300/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1843 - root_mean_squared_error: 1.1066 - val_loss: 1.1785 - val_root_mean_squared_error: 1.0745
Epoch 301/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1929 - root_mean_squared_error: 1.1113 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0878
Epoch 302/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1672 - root_mean_squared_error: 1.0847 - val_loss: 1.1583 - val_root_mean_squared_error: 1.0970
Epoch 303/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1846 - root_mean_squared_error: 1.1297 - val_loss: 1.2013 - val_root_mean_squared_error: 1.0991
Epoch 304/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1724 - root_mean_squared_error: 1.0899 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0781
Epoch 305/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1663 - root_mean_squared_error: 1.1007 - val_loss: 1.1885 - val_root_mean_squared_error: 1.0875
Epoch 306/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1873 - root_mean_squared_error: 1.1214 - val_loss: 1.1785 - val_root_mean_squared_error: 1.0805
Epoch 307/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1776 - root_mean_squared_error: 1.1130 - val_loss: 1.1681 - val_root_mean_squared_error: 1.0684
Epoch 308/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1732 - root_mean_squared_error: 1.0900 - val_loss: 1.1719 - val_root_mean_squared_error: 1.1420
Epoch 309/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1757 - root_mean_squared_error: 1.0945 - val_loss: 1.1503 - val_root_mean_squared_error: 1.0932
Epoch 310/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1717 - root_mean_squared_error: 1.0961 - val_loss: 1.1868 - val_root_mean_squared_error: 1.0741
Epoch 311/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1768 - root_mean_squared_error: 1.0796 - val_loss: 1.2142 - val_root_mean_squared_error: 1.1338
Epoch 312/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1565 - root_mean_squared_error: 1.0982 - val_loss: 1.1530 - val_root_mean_squared_error: 1.0731
Epoch 313/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1700 - root_mean_squared_error: 1.0880 - val_loss: 1.1648 - val_root_mean_squared_error: 1.0764
Epoch 314/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1719 - root_mean_squared_error: 1.0989 - val_loss: 1.1604 - val_root_mean_squared_error: 1.0712
Epoch 315/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1827 - root_mean_squared_error: 1.1023 - val_loss: 1.1522 - val_root_mean_squared_error: 1.1010
Epoch 316/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1747 - root_mean_squared_error: 1.0969 - val_loss: 1.1582 - val_root_mean_squared_error: 1.1162
Epoch 317/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1811 - root_mean_squared_error: 1.0973 - val_loss: 1.1617 - val_root_mean_squared_error: 1.1087
Epoch 318/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1734 - root_mean_squared_error: 1.0991 - val_loss: 1.1638 - val_root_mean_squared_error: 1.0716
Epoch 319/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1778 - root_mean_squared_error: 1.1154 - val_loss: 1.1927 - val_root_mean_squared_error: 1.1222
Epoch 320/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1749 - root_mean_squared_error: 1.1075 - val_loss: 1.1713 - val_root_mean_squared_error: 1.0946
Epoch 321/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1789 - root_mean_squared_error: 1.1034 - val_loss: 1.1637 - val_root_mean_squared_error: 1.0844
Epoch 322/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1788 - root_mean_squared_error: 1.0893 - val_loss: 1.1707 - val_root_mean_squared_error: 1.0841
Epoch 323/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1729 - root_mean_squared_error: 1.1059 - val_loss: 1.1640 - val_root_mean_squared_error: 1.0323
Epoch 324/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1722 - root_mean_squared_error: 1.1104 - val_loss: 1.1632 - val_root_mean_squared_error: 1.0051
Epoch 325/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1672 - root_mean_squared_error: 1.0880 - val_loss: 1.1652 - val_root_mean_squared_error: 1.1002
Epoch 326/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1753 - root_mean_squared_error: 1.0728 - val_loss: 1.1579 - val_root_mean_squared_error: 1.1219
Epoch 327/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1738 - root_mean_squared_error: 1.1076 - val_loss: 1.1405 - val_root_mean_squared_error: 1.0366
Epoch 328/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1698 - root_mean_squared_error: 1.0960 - val_loss: 1.1732 - val_root_mean_squared_error: 1.1188
Epoch 329/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1753 - root_mean_squared_error: 1.0721 - val_loss: 1.2076 - val_root_mean_squared_error: 1.0804
Epoch 330/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1792 - root_mean_squared_error: 1.0892 - val_loss: 1.1661 - val_root_mean_squared_error: 1.0857
Epoch 331/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1808 - root_mean_squared_error: 1.0978 - val_loss: 1.1977 - val_root_mean_squared_error: 1.0983
Epoch 332/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1764 - root_mean_squared_error: 1.0971 - val_loss: 1.2271 - val_root_mean_squared_error: 1.1190
Epoch 333/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1757 - root_mean_squared_error: 1.0973 - val_loss: 1.1628 - val_root_mean_squared_error: 1.1055
Epoch 334/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1714 - root_mean_squared_error: 1.0938 - val_loss: 1.1573 - val_root_mean_squared_error: 1.0604
Epoch 335/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1629 - root_mean_squared_error: 1.0725 - val_loss: 1.1791 - val_root_mean_squared_error: 1.1020
Epoch 336/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1759 - root_mean_squared_error: 1.0936 - val_loss: 1.1585 - val_root_mean_squared_error: 1.1145
Epoch 337/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1880 - root_mean_squared_error: 1.0883 - val_loss: 1.1685 - val_root_mean_squared_error: 1.0795
Epoch 338/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1732 - root_mean_squared_error: 1.0811 - val_loss: 1.1717 - val_root_mean_squared_error: 1.1147
Epoch 339/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1756 - root_mean_squared_error: 1.1103 - val_loss: 1.1556 - val_root_mean_squared_error: 1.0803
Epoch 340/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1791 - root_mean_squared_error: 1.0988 - val_loss: 1.1736 - val_root_mean_squared_error: 1.0999
Epoch 341/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1691 - root_mean_squared_error: 1.0791 - val_loss: 1.1616 - val_root_mean_squared_error: 1.1008
Epoch 342/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1731 - root_mean_squared_error: 1.0805 - val_loss: 1.2623 - val_root_mean_squared_error: 1.1364
Epoch 343/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1691 - root_mean_squared_error: 1.0833 - val_loss: 1.1983 - val_root_mean_squared_error: 1.1732
Epoch 344/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1736 - root_mean_squared_error: 1.0748 - val_loss: 1.1802 - val_root_mean_squared_error: 1.1361
Epoch 345/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1686 - root_mean_squared_error: 1.1087 - val_loss: 1.1753 - val_root_mean_squared_error: 1.0949
Epoch 346/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1784 - root_mean_squared_error: 1.0838 - val_loss: 1.1689 - val_root_mean_squared_error: 1.0907
Epoch 347/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1762 - root_mean_squared_error: 1.0970 - val_loss: 1.1912 - val_root_mean_squared_error: 1.1770
Epoch 348/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1861 - root_mean_squared_error: 1.1059 - val_loss: 1.1672 - val_root_mean_squared_error: 1.0806
Epoch 349/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1751 - root_mean_squared_error: 1.0962 - val_loss: 1.1694 - val_root_mean_squared_error: 1.1181
Epoch 350/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1824 - root_mean_squared_error: 1.1135 - val_loss: 1.1779 - val_root_mean_squared_error: 1.1387
Epoch 351/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1784 - root_mean_squared_error: 1.1000 - val_loss: 1.1634 - val_root_mean_squared_error: 1.0839
Epoch 352/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1739 - root_mean_squared_error: 1.1106 - val_loss: 1.1689 - val_root_mean_squared_error: 1.0541
Epoch 353/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1753 - root_mean_squared_error: 1.0940 - val_loss: 1.1686 - val_root_mean_squared_error: 1.0845
Epoch 354/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1698 - root_mean_squared_error: 1.0907 - val_loss: 1.1465 - val_root_mean_squared_error: 1.0831
Epoch 355/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1788 - root_mean_squared_error: 1.0940 - val_loss: 1.1634 - val_root_mean_squared_error: 1.0784
Epoch 356/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1770 - root_mean_squared_error: 1.1010 - val_loss: 1.1626 - val_root_mean_squared_error: 1.1213
Epoch 357/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1598 - root_mean_squared_error: 1.1013 - val_loss: 1.1763 - val_root_mean_squared_error: 1.0891
Epoch 358/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1695 - root_mean_squared_error: 1.0806 - val_loss: 1.1712 - val_root_mean_squared_error: 1.0983
Epoch 359/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1821 - root_mean_squared_error: 1.0928 - val_loss: 1.1661 - val_root_mean_squared_error: 1.1581
Epoch 360/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1746 - root_mean_squared_error: 1.0955 - val_loss: 1.1634 - val_root_mean_squared_error: 1.1176
Epoch 361/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1683 - root_mean_squared_error: 1.0807 - val_loss: 1.1603 - val_root_mean_squared_error: 1.1007
Epoch 362/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1700 - root_mean_squared_error: 1.0847 - val_loss: 1.1769 - val_root_mean_squared_error: 1.1347
Epoch 363/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1663 - root_mean_squared_error: 1.1063 - val_loss: 1.1728 - val_root_mean_squared_error: 1.1502
Epoch 364/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1689 - root_mean_squared_error: 1.0966 - val_loss: 1.1642 - val_root_mean_squared_error: 1.1320
Epoch 365/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1755 - root_mean_squared_error: 1.1060 - val_loss: 1.1605 - val_root_mean_squared_error: 1.0892
Epoch 366/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1651 - root_mean_squared_error: 1.0650 - val_loss: 1.1573 - val_root_mean_squared_error: 1.0864
Epoch 367/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1689 - root_mean_squared_error: 1.0762 - val_loss: 1.1877 - val_root_mean_squared_error: 1.1264
Epoch 368/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1665 - root_mean_squared_error: 1.0914 - val_loss: 1.1616 - val_root_mean_squared_error: 1.0985
Epoch 369/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1639 - root_mean_squared_error: 1.0855 - val_loss: 1.1600 - val_root_mean_squared_error: 1.0856
Epoch 370/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1789 - root_mean_squared_error: 1.0885 - val_loss: 1.1736 - val_root_mean_squared_error: 1.1177
Epoch 371/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1686 - root_mean_squared_error: 1.0848 - val_loss: 1.2187 - val_root_mean_squared_error: 1.1342
Epoch 372/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1758 - root_mean_squared_error: 1.0999 - val_loss: 1.1726 - val_root_mean_squared_error: 1.0541
Epoch 373/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1743 - root_mean_squared_error: 1.0972 - val_loss: 1.1708 - val_root_mean_squared_error: 1.0565
Epoch 374/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1753 - root_mean_squared_error: 1.0852 - val_loss: 1.1659 - val_root_mean_squared_error: 1.1198
Epoch 375/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1657 - root_mean_squared_error: 1.0775 - val_loss: 1.1693 - val_root_mean_squared_error: 1.1181
Epoch 376/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1639 - root_mean_squared_error: 1.0695 - val_loss: 1.1910 - val_root_mean_squared_error: 1.0852
Epoch 377/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1782 - root_mean_squared_error: 1.0845 - val_loss: 1.1732 - val_root_mean_squared_error: 1.0821
Epoch 378/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1727 - root_mean_squared_error: 1.0972 - val_loss: 1.1648 - val_root_mean_squared_error: 1.0947
Epoch 379/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1717 - root_mean_squared_error: 1.0759 - val_loss: 1.1545 - val_root_mean_squared_error: 1.0969
Epoch 380/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1707 - root_mean_squared_error: 1.0993 - val_loss: 1.1727 - val_root_mean_squared_error: 1.0730
Epoch 381/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1675 - root_mean_squared_error: 1.0883 - val_loss: 1.1512 - val_root_mean_squared_error: 1.0553
Epoch 382/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1684 - root_mean_squared_error: 1.0773 - val_loss: 1.1756 - val_root_mean_squared_error: 1.0898
Epoch 383/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1759 - root_mean_squared_error: 1.0924 - val_loss: 1.1819 - val_root_mean_squared_error: 1.0838
Epoch 384/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1706 - root_mean_squared_error: 1.0836 - val_loss: 1.1804 - val_root_mean_squared_error: 1.1004
Epoch 385/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1646 - root_mean_squared_error: 1.0815 - val_loss: 1.1653 - val_root_mean_squared_error: 1.1181
Epoch 386/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1686 - root_mean_squared_error: 1.0897 - val_loss: 1.1738 - val_root_mean_squared_error: 1.1065
Epoch 387/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1681 - root_mean_squared_error: 1.0872 - val_loss: 1.1729 - val_root_mean_squared_error: 1.0917
Epoch 388/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1784 - root_mean_squared_error: 1.0848 - val_loss: 1.1767 - val_root_mean_squared_error: 1.0866
Epoch 389/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1708 - root_mean_squared_error: 1.0997 - val_loss: 1.1510 - val_root_mean_squared_error: 1.1075
Epoch 390/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1647 - root_mean_squared_error: 1.0769 - val_loss: 1.1790 - val_root_mean_squared_error: 1.0076
Epoch 391/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1734 - root_mean_squared_error: 1.0938 - val_loss: 1.1921 - val_root_mean_squared_error: 1.1396
Epoch 392/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1626 - root_mean_squared_error: 1.0768 - val_loss: 1.1704 - val_root_mean_squared_error: 1.0527
Epoch 393/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1726 - root_mean_squared_error: 1.0981 - val_loss: 1.1788 - val_root_mean_squared_error: 1.0989
Epoch 394/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1681 - root_mean_squared_error: 1.0775 - val_loss: 1.1593 - val_root_mean_squared_error: 1.0953
Epoch 395/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1744 - root_mean_squared_error: 1.0830 - val_loss: 1.1572 - val_root_mean_squared_error: 1.0841
Epoch 396/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1724 - root_mean_squared_error: 1.0960 - val_loss: 1.1524 - val_root_mean_squared_error: 1.0938
Epoch 397/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1718 - root_mean_squared_error: 1.0889 - val_loss: 1.1784 - val_root_mean_squared_error: 1.0879
Epoch 398/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1847 - root_mean_squared_error: 1.1005 - val_loss: 1.1705 - val_root_mean_squared_error: 1.0258
Epoch 399/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1642 - root_mean_squared_error: 1.0878 - val_loss: 1.1542 - val_root_mean_squared_error: 1.0682
Epoch 400/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1661 - root_mean_squared_error: 1.0670 - val_loss: 1.1824 - val_root_mean_squared_error: 1.0845
Epoch 401/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1647 - root_mean_squared_error: 1.0938 - val_loss: 1.1685 - val_root_mean_squared_error: 1.0917
Epoch 402/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1770 - root_mean_squared_error: 1.1008 - val_loss: 1.1661 - val_root_mean_squared_error: 1.0501
Epoch 403/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1758 - root_mean_squared_error: 1.0886 - val_loss: 1.1882 - val_root_mean_squared_error: 1.1270
Epoch 404/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1636 - root_mean_squared_error: 1.0843 - val_loss: 1.1658 - val_root_mean_squared_error: 1.0839
Epoch 405/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1647 - root_mean_squared_error: 1.0907 - val_loss: 1.1779 - val_root_mean_squared_error: 1.0836
Epoch 406/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1652 - root_mean_squared_error: 1.0854 - val_loss: 1.1738 - val_root_mean_squared_error: 1.0919
Epoch 407/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1705 - root_mean_squared_error: 1.0717 - val_loss: 1.1624 - val_root_mean_squared_error: 1.0633
Epoch 408/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1803 - root_mean_squared_error: 1.1033 - val_loss: 1.1831 - val_root_mean_squared_error: 1.0626
Epoch 409/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1686 - root_mean_squared_error: 1.1094 - val_loss: 1.1652 - val_root_mean_squared_error: 1.0914
Epoch 410/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1769 - root_mean_squared_error: 1.0925 - val_loss: 1.1641 - val_root_mean_squared_error: 1.0778
Epoch 411/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1716 - root_mean_squared_error: 1.1038 - val_loss: 1.1865 - val_root_mean_squared_error: 1.1222
Epoch 412/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1704 - root_mean_squared_error: 1.1064 - val_loss: 1.1703 - val_root_mean_squared_error: 1.0909
Epoch 413/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1719 - root_mean_squared_error: 1.0990 - val_loss: 1.1687 - val_root_mean_squared_error: 1.0573
Epoch 414/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1768 - root_mean_squared_error: 1.0738 - val_loss: 1.1648 - val_root_mean_squared_error: 1.1085
Epoch 415/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1681 - root_mean_squared_error: 1.1003 - val_loss: 1.1619 - val_root_mean_squared_error: 1.1077
Epoch 416/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1782 - root_mean_squared_error: 1.1030 - val_loss: 1.1806 - val_root_mean_squared_error: 1.1070
Epoch 417/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1695 - root_mean_squared_error: 1.0867 - val_loss: 1.1617 - val_root_mean_squared_error: 1.0535
Epoch 418/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1716 - root_mean_squared_error: 1.0929 - val_loss: 1.1596 - val_root_mean_squared_error: 1.0528
Epoch 419/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1699 - root_mean_squared_error: 1.0812 - val_loss: 1.1715 - val_root_mean_squared_error: 1.0859
Epoch 420/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1671 - root_mean_squared_error: 1.0918 - val_loss: 1.1585 - val_root_mean_squared_error: 1.1063
Epoch 421/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1770 - root_mean_squared_error: 1.0893 - val_loss: 1.1562 - val_root_mean_squared_error: 1.0617
Epoch 422/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1612 - root_mean_squared_error: 1.0789 - val_loss: 1.1804 - val_root_mean_squared_error: 1.1026
Epoch 423/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1695 - root_mean_squared_error: 1.0701 - val_loss: 1.1871 - val_root_mean_squared_error: 1.1195
Epoch 424/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1768 - root_mean_squared_error: 1.0899 - val_loss: 1.1679 - val_root_mean_squared_error: 1.0632
Epoch 425/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1652 - root_mean_squared_error: 1.0630 - val_loss: 1.1563 - val_root_mean_squared_error: 1.0697
Epoch 426/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1692 - root_mean_squared_error: 1.1058 - val_loss: 1.2034 - val_root_mean_squared_error: 1.1043
Epoch 427/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1708 - root_mean_squared_error: 1.0862 - val_loss: 1.1596 - val_root_mean_squared_error: 1.0739
Epoch 428/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1756 - root_mean_squared_error: 1.0905 - val_loss: 1.1613 - val_root_mean_squared_error: 1.0620
Epoch 429/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1715 - root_mean_squared_error: 1.1041 - val_loss: 1.1568 - val_root_mean_squared_error: 1.1137
Epoch 430/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1687 - root_mean_squared_error: 1.0896 - val_loss: 1.1547 - val_root_mean_squared_error: 1.0189
Epoch 431/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1679 - root_mean_squared_error: 1.0886 - val_loss: 1.1506 - val_root_mean_squared_error: 1.0983
Epoch 432/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1603 - root_mean_squared_error: 1.0653 - val_loss: 1.1672 - val_root_mean_squared_error: 1.1578
Epoch 433/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1665 - root_mean_squared_error: 1.0871 - val_loss: 1.1676 - val_root_mean_squared_error: 1.1380
Epoch 434/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1674 - root_mean_squared_error: 1.0670 - val_loss: 1.1602 - val_root_mean_squared_error: 1.1053
Epoch 435/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1705 - root_mean_squared_error: 1.1024 - val_loss: 1.1838 - val_root_mean_squared_error: 1.1176
Epoch 436/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1651 - root_mean_squared_error: 1.0924 - val_loss: 1.1624 - val_root_mean_squared_error: 1.0791
Epoch 437/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1656 - root_mean_squared_error: 1.0827 - val_loss: 1.1666 - val_root_mean_squared_error: 1.0661
Epoch 438/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1713 - root_mean_squared_error: 1.0894 - val_loss: 1.1519 - val_root_mean_squared_error: 1.1289
Epoch 439/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1724 - root_mean_squared_error: 1.0910 - val_loss: 1.1782 - val_root_mean_squared_error: 1.1119
Epoch 440/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1703 - root_mean_squared_error: 1.0902 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0795
Epoch 441/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1761 - root_mean_squared_error: 1.0876 - val_loss: 1.1572 - val_root_mean_squared_error: 1.1205
Epoch 442/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1692 - root_mean_squared_error: 1.0889 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0826
Epoch 443/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1659 - root_mean_squared_error: 1.0993 - val_loss: 1.1589 - val_root_mean_squared_error: 1.0993
Epoch 444/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1599 - root_mean_squared_error: 1.0800 - val_loss: 1.1424 - val_root_mean_squared_error: 1.0841
Epoch 445/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1623 - root_mean_squared_error: 1.0928 - val_loss: 1.1835 - val_root_mean_squared_error: 1.1164
Epoch 446/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1703 - root_mean_squared_error: 1.0762 - val_loss: 1.1752 - val_root_mean_squared_error: 1.0957
Epoch 447/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1595 - root_mean_squared_error: 1.0974 - val_loss: 1.1584 - val_root_mean_squared_error: 1.1045
Epoch 448/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1740 - root_mean_squared_error: 1.1196 - val_loss: 1.1726 - val_root_mean_squared_error: 1.1422
Epoch 449/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1658 - root_mean_squared_error: 1.0868 - val_loss: 1.1557 - val_root_mean_squared_error: 1.1024
Epoch 450/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1668 - root_mean_squared_error: 1.0838 - val_loss: 1.1559 - val_root_mean_squared_error: 1.1100
Epoch 451/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1767 - root_mean_squared_error: 1.0929 - val_loss: 1.1599 - val_root_mean_squared_error: 1.1228
Epoch 452/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1720 - root_mean_squared_error: 1.0846 - val_loss: 1.1712 - val_root_mean_squared_error: 1.0963
Epoch 453/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1622 - root_mean_squared_error: 1.0807 - val_loss: 1.1564 - val_root_mean_squared_error: 1.1052
Epoch 454/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1582 - root_mean_squared_error: 1.0972 - val_loss: 1.1582 - val_root_mean_squared_error: 1.0809
Epoch 455/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1665 - root_mean_squared_error: 1.0806 - val_loss: 1.1673 - val_root_mean_squared_error: 1.0544
Epoch 456/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1690 - root_mean_squared_error: 1.0870 - val_loss: 1.1680 - val_root_mean_squared_error: 1.0896
Epoch 457/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1662 - root_mean_squared_error: 1.0922 - val_loss: 1.1567 - val_root_mean_squared_error: 1.0678
Epoch 458/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1566 - root_mean_squared_error: 1.0765 - val_loss: 1.1496 - val_root_mean_squared_error: 1.1563
Epoch 459/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1705 - root_mean_squared_error: 1.0902 - val_loss: 1.1554 - val_root_mean_squared_error: 1.0868
Epoch 460/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1655 - root_mean_squared_error: 1.0787 - val_loss: 1.1480 - val_root_mean_squared_error: 1.0645
Epoch 461/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1644 - root_mean_squared_error: 1.0734 - val_loss: 1.1769 - val_root_mean_squared_error: 1.0424
Epoch 462/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1653 - root_mean_squared_error: 1.0808 - val_loss: 1.1576 - val_root_mean_squared_error: 1.0561
Epoch 463/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1689 - root_mean_squared_error: 1.0582 - val_loss: 1.1526 - val_root_mean_squared_error: 1.0705
Epoch 464/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1666 - root_mean_squared_error: 1.0991 - val_loss: 1.1784 - val_root_mean_squared_error: 1.0741
Epoch 465/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1662 - root_mean_squared_error: 1.0764 - val_loss: 1.1731 - val_root_mean_squared_error: 1.0979
Epoch 466/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1665 - root_mean_squared_error: 1.0890 - val_loss: 1.1718 - val_root_mean_squared_error: 1.0511
Epoch 467/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1715 - root_mean_squared_error: 1.0875 - val_loss: 1.1689 - val_root_mean_squared_error: 1.1004
Epoch 468/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1647 - root_mean_squared_error: 1.0857 - val_loss: 1.1538 - val_root_mean_squared_error: 1.0836
Epoch 469/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1631 - root_mean_squared_error: 1.0960 - val_loss: 1.2036 - val_root_mean_squared_error: 1.1128
Epoch 470/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1725 - root_mean_squared_error: 1.1096 - val_loss: 1.1650 - val_root_mean_squared_error: 1.0752
Epoch 471/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1784 - root_mean_squared_error: 1.1003 - val_loss: 1.1776 - val_root_mean_squared_error: 1.1289
Epoch 472/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1676 - root_mean_squared_error: 1.1160 - val_loss: 1.1762 - val_root_mean_squared_error: 1.1146
Epoch 473/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1717 - root_mean_squared_error: 1.0874 - val_loss: 1.1774 - val_root_mean_squared_error: 1.1411
Epoch 474/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1705 - root_mean_squared_error: 1.1010 - val_loss: 1.1604 - val_root_mean_squared_error: 1.1134
Epoch 475/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1600 - root_mean_squared_error: 1.0790 - val_loss: 1.2343 - val_root_mean_squared_error: 1.1322
Epoch 476/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1675 - root_mean_squared_error: 1.0867 - val_loss: 1.1799 - val_root_mean_squared_error: 1.0792
Epoch 477/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1713 - root_mean_squared_error: 1.0944 - val_loss: 1.1670 - val_root_mean_squared_error: 1.1038
Epoch 478/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1583 - root_mean_squared_error: 1.0667 - val_loss: 1.1648 - val_root_mean_squared_error: 1.0984
Epoch 479/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1670 - root_mean_squared_error: 1.0783 - val_loss: 1.1652 - val_root_mean_squared_error: 1.1091
Epoch 480/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1678 - root_mean_squared_error: 1.0919 - val_loss: 1.1642 - val_root_mean_squared_error: 1.0974
Epoch 481/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1659 - root_mean_squared_error: 1.0933 - val_loss: 1.1906 - val_root_mean_squared_error: 1.0984
Epoch 482/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1694 - root_mean_squared_error: 1.0751 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0602
Epoch 483/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1583 - root_mean_squared_error: 1.0583 - val_loss: 1.1753 - val_root_mean_squared_error: 1.1047
Epoch 484/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1805 - root_mean_squared_error: 1.0748 - val_loss: 1.1829 - val_root_mean_squared_error: 1.1210
Epoch 485/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1696 - root_mean_squared_error: 1.0846 - val_loss: 1.1591 - val_root_mean_squared_error: 1.0724
Epoch 486/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1603 - root_mean_squared_error: 1.0751 - val_loss: 1.1506 - val_root_mean_squared_error: 1.0920
Epoch 487/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1629 - root_mean_squared_error: 1.0972 - val_loss: 1.1634 - val_root_mean_squared_error: 1.1002
Epoch 488/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1666 - root_mean_squared_error: 1.0899 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0928
Epoch 489/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1684 - root_mean_squared_error: 1.0901 - val_loss: 1.1651 - val_root_mean_squared_error: 1.0744
Epoch 490/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1614 - root_mean_squared_error: 1.0743 - val_loss: 1.1604 - val_root_mean_squared_error: 1.0661
Epoch 491/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1667 - root_mean_squared_error: 1.0834 - val_loss: 1.1663 - val_root_mean_squared_error: 1.1573
Epoch 492/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1631 - root_mean_squared_error: 1.0803 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0497
Epoch 493/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1614 - root_mean_squared_error: 1.0893 - val_loss: 1.1605 - val_root_mean_squared_error: 1.1008
Epoch 494/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1802 - root_mean_squared_error: 1.0821 - val_loss: 1.1515 - val_root_mean_squared_error: 1.0958
Epoch 495/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1638 - root_mean_squared_error: 1.0898 - val_loss: 1.1418 - val_root_mean_squared_error: 1.0824
Epoch 496/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1613 - root_mean_squared_error: 1.0917 - val_loss: 1.1649 - val_root_mean_squared_error: 1.0826
Epoch 497/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1635 - root_mean_squared_error: 1.1002 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0959
Epoch 498/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1695 - root_mean_squared_error: 1.0932 - val_loss: 1.1500 - val_root_mean_squared_error: 1.0759
Epoch 499/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1745 - root_mean_squared_error: 1.0980 - val_loss: 1.1670 - val_root_mean_squared_error: 1.1195
Epoch 500/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1812 - root_mean_squared_error: 1.1109 - val_loss: 1.1536 - val_root_mean_squared_error: 1.1128
Epoch 501/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1773 - root_mean_squared_error: 1.0978 - val_loss: 1.1603 - val_root_mean_squared_error: 1.0862
Epoch 502/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1660 - root_mean_squared_error: 1.0866 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0935
Epoch 503/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1664 - root_mean_squared_error: 1.0948 - val_loss: 1.1497 - val_root_mean_squared_error: 1.0988
Epoch 504/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1710 - root_mean_squared_error: 1.1016 - val_loss: 1.1629 - val_root_mean_squared_error: 1.0912
Epoch 505/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1692 - root_mean_squared_error: 1.1035 - val_loss: 1.1614 - val_root_mean_squared_error: 1.0702
Epoch 506/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1625 - root_mean_squared_error: 1.0640 - val_loss: 1.1624 - val_root_mean_squared_error: 1.0905
Epoch 507/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1730 - root_mean_squared_error: 1.0960 - val_loss: 1.1760 - val_root_mean_squared_error: 1.1229
Epoch 508/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1749 - root_mean_squared_error: 1.0958 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0776
Epoch 509/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1655 - root_mean_squared_error: 1.0967 - val_loss: 1.1694 - val_root_mean_squared_error: 1.1444
Epoch 510/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1774 - root_mean_squared_error: 1.0772 - val_loss: 1.1622 - val_root_mean_squared_error: 1.0934
Epoch 511/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1811 - root_mean_squared_error: 1.1084 - val_loss: 1.1610 - val_root_mean_squared_error: 1.1022
Epoch 512/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1675 - root_mean_squared_error: 1.0791 - val_loss: 1.1550 - val_root_mean_squared_error: 1.0553
Epoch 513/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1638 - root_mean_squared_error: 1.0996 - val_loss: 1.1655 - val_root_mean_squared_error: 1.0888
Epoch 514/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1606 - root_mean_squared_error: 1.0672 - val_loss: 1.1531 - val_root_mean_squared_error: 1.0920
Epoch 515/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1657 - root_mean_squared_error: 1.0948 - val_loss: 1.1661 - val_root_mean_squared_error: 1.1225
Epoch 516/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1661 - root_mean_squared_error: 1.0861 - val_loss: 1.1546 - val_root_mean_squared_error: 1.0940
Epoch 517/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1631 - root_mean_squared_error: 1.0731 - val_loss: 1.1843 - val_root_mean_squared_error: 1.1182
Epoch 518/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1646 - root_mean_squared_error: 1.1047 - val_loss: 1.1570 - val_root_mean_squared_error: 1.0628
Epoch 519/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1673 - root_mean_squared_error: 1.0879 - val_loss: 1.1526 - val_root_mean_squared_error: 1.0591
Epoch 520/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1698 - root_mean_squared_error: 1.0978 - val_loss: 1.1526 - val_root_mean_squared_error: 1.0885
Epoch 521/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1646 - root_mean_squared_error: 1.0839 - val_loss: 1.1587 - val_root_mean_squared_error: 1.0266
Epoch 522/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1632 - root_mean_squared_error: 1.0826 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0836
Epoch 523/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1652 - root_mean_squared_error: 1.0766 - val_loss: 1.1677 - val_root_mean_squared_error: 1.0909
Epoch 524/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1729 - root_mean_squared_error: 1.0869 - val_loss: 1.1557 - val_root_mean_squared_error: 1.1164
Epoch 525/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1664 - root_mean_squared_error: 1.0961 - val_loss: 1.1631 - val_root_mean_squared_error: 1.0330
Epoch 526/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1634 - root_mean_squared_error: 1.0912 - val_loss: 1.1663 - val_root_mean_squared_error: 1.1055
Epoch 527/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1724 - root_mean_squared_error: 1.0784 - val_loss: 1.1684 - val_root_mean_squared_error: 1.1164
Epoch 528/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1655 - root_mean_squared_error: 1.0921 - val_loss: 1.1617 - val_root_mean_squared_error: 1.0652
Epoch 529/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1738 - root_mean_squared_error: 1.0724 - val_loss: 1.1631 - val_root_mean_squared_error: 1.0568
Epoch 530/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1581 - root_mean_squared_error: 1.0793 - val_loss: 1.1596 - val_root_mean_squared_error: 1.1034
Epoch 531/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1710 - root_mean_squared_error: 1.1033 - val_loss: 1.1540 - val_root_mean_squared_error: 1.0510
Epoch 532/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1688 - root_mean_squared_error: 1.0964 - val_loss: 1.1577 - val_root_mean_squared_error: 1.0944
Epoch 533/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1625 - root_mean_squared_error: 1.0820 - val_loss: 1.1536 - val_root_mean_squared_error: 1.0732
Epoch 534/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1688 - root_mean_squared_error: 1.0953 - val_loss: 1.1701 - val_root_mean_squared_error: 1.0768
Epoch 535/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1717 - root_mean_squared_error: 1.0909 - val_loss: 1.1745 - val_root_mean_squared_error: 1.1172
Epoch 536/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1577 - root_mean_squared_error: 1.0851 - val_loss: 1.1679 - val_root_mean_squared_error: 1.1072
Epoch 537/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1697 - root_mean_squared_error: 1.0928 - val_loss: 1.1629 - val_root_mean_squared_error: 1.0734
Epoch 538/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1677 - root_mean_squared_error: 1.0763 - val_loss: 1.1705 - val_root_mean_squared_error: 1.0871
Epoch 539/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1669 - root_mean_squared_error: 1.0615 - val_loss: 1.1900 - val_root_mean_squared_error: 1.0971
Epoch 540/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1747 - root_mean_squared_error: 1.0962 - val_loss: 1.1603 - val_root_mean_squared_error: 1.1224
Epoch 541/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1611 - root_mean_squared_error: 1.0842 - val_loss: 1.1726 - val_root_mean_squared_error: 1.0909
Epoch 542/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1785 - root_mean_squared_error: 1.1058 - val_loss: 1.1758 - val_root_mean_squared_error: 1.1171
Epoch 543/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1666 - root_mean_squared_error: 1.0688 - val_loss: 1.1812 - val_root_mean_squared_error: 1.0906
Epoch 544/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1627 - root_mean_squared_error: 1.0886 - val_loss: 1.1633 - val_root_mean_squared_error: 1.0685
Epoch 545/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1623 - root_mean_squared_error: 1.0699 - val_loss: 1.1623 - val_root_mean_squared_error: 1.0443
Epoch 546/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1612 - root_mean_squared_error: 1.0853 - val_loss: 1.1630 - val_root_mean_squared_error: 1.0620
Epoch 547/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1641 - root_mean_squared_error: 1.0753 - val_loss: 1.1612 - val_root_mean_squared_error: 1.0559
Epoch 548/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1631 - root_mean_squared_error: 1.0928 - val_loss: 1.1755 - val_root_mean_squared_error: 1.0745
Epoch 549/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1675 - root_mean_squared_error: 1.0736 - val_loss: 1.1739 - val_root_mean_squared_error: 1.0760
Epoch 550/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1628 - root_mean_squared_error: 1.0738 - val_loss: 1.1773 - val_root_mean_squared_error: 1.0640
Epoch 551/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1750 - root_mean_squared_error: 1.0775 - val_loss: 1.1466 - val_root_mean_squared_error: 1.0315
Epoch 552/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1688 - root_mean_squared_error: 1.1022 - val_loss: 1.1510 - val_root_mean_squared_error: 1.0579
Epoch 553/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1641 - root_mean_squared_error: 1.0841 - val_loss: 1.1763 - val_root_mean_squared_error: 1.0385
Epoch 554/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1639 - root_mean_squared_error: 1.0981 - val_loss: 1.1852 - val_root_mean_squared_error: 1.1045
Epoch 555/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1745 - root_mean_squared_error: 1.0892 - val_loss: 1.1651 - val_root_mean_squared_error: 1.1131
Epoch 556/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1694 - root_mean_squared_error: 1.1070 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0909
Epoch 557/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1594 - root_mean_squared_error: 1.0980 - val_loss: 1.1648 - val_root_mean_squared_error: 1.0976
Epoch 558/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1674 - root_mean_squared_error: 1.0859 - val_loss: 1.1775 - val_root_mean_squared_error: 1.0640
Epoch 559/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1587 - root_mean_squared_error: 1.0794 - val_loss: 1.1741 - val_root_mean_squared_error: 1.1114
Epoch 560/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1602 - root_mean_squared_error: 1.0884 - val_loss: 1.1581 - val_root_mean_squared_error: 1.1220
Epoch 561/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1578 - root_mean_squared_error: 1.0740 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0791
Epoch 562/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1651 - root_mean_squared_error: 1.0892 - val_loss: 1.1676 - val_root_mean_squared_error: 1.0816
Epoch 563/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1531 - root_mean_squared_error: 1.0594 - val_loss: 1.1813 - val_root_mean_squared_error: 1.0603
Epoch 564/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1796 - root_mean_squared_error: 1.0968 - val_loss: 1.1566 - val_root_mean_squared_error: 1.0481
Epoch 565/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1639 - root_mean_squared_error: 1.0838 - val_loss: 1.1572 - val_root_mean_squared_error: 1.1217
Epoch 566/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1631 - root_mean_squared_error: 1.0726 - val_loss: 1.1458 - val_root_mean_squared_error: 1.0473
Epoch 567/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1579 - root_mean_squared_error: 1.0873 - val_loss: 1.1705 - val_root_mean_squared_error: 1.0739
Epoch 568/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1612 - root_mean_squared_error: 1.0720 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0877
Epoch 569/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1552 - root_mean_squared_error: 1.0593 - val_loss: 1.1601 - val_root_mean_squared_error: 1.0859
Epoch 570/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1637 - root_mean_squared_error: 1.0862 - val_loss: 1.1654 - val_root_mean_squared_error: 1.0727
Epoch 571/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1772 - root_mean_squared_error: 1.0680 - val_loss: 1.1552 - val_root_mean_squared_error: 1.0357
Epoch 572/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1713 - root_mean_squared_error: 1.1116 - val_loss: 1.1642 - val_root_mean_squared_error: 1.0674
Epoch 573/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1612 - root_mean_squared_error: 1.0691 - val_loss: 1.1561 - val_root_mean_squared_error: 1.0872
Epoch 574/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1625 - root_mean_squared_error: 1.1002 - val_loss: 1.1483 - val_root_mean_squared_error: 1.0697
Epoch 575/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1639 - root_mean_squared_error: 1.0842 - val_loss: 1.1570 - val_root_mean_squared_error: 1.1079
Epoch 576/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1630 - root_mean_squared_error: 1.0833 - val_loss: 1.1715 - val_root_mean_squared_error: 1.0850
Epoch 577/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1754 - root_mean_squared_error: 1.1003 - val_loss: 1.1627 - val_root_mean_squared_error: 1.0764
Epoch 578/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1632 - root_mean_squared_error: 1.0745 - val_loss: 1.1895 - val_root_mean_squared_error: 1.1021
Epoch 579/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1584 - root_mean_squared_error: 1.0928 - val_loss: 1.1452 - val_root_mean_squared_error: 1.0772
Epoch 580/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1609 - root_mean_squared_error: 1.0800 - val_loss: 1.1589 - val_root_mean_squared_error: 1.0418
Epoch 581/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1572 - root_mean_squared_error: 1.0572 - val_loss: 1.1627 - val_root_mean_squared_error: 1.0425
Epoch 582/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1612 - root_mean_squared_error: 1.0930 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0770
Epoch 583/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1672 - root_mean_squared_error: 1.0982 - val_loss: 1.1721 - val_root_mean_squared_error: 1.0969
Epoch 584/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1626 - root_mean_squared_error: 1.0885 - val_loss: 1.1638 - val_root_mean_squared_error: 1.0964
Epoch 585/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1628 - root_mean_squared_error: 1.0806 - val_loss: 1.1668 - val_root_mean_squared_error: 1.1173
Epoch 586/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1550 - root_mean_squared_error: 1.0735 - val_loss: 1.1473 - val_root_mean_squared_error: 1.0757
Epoch 587/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1724 - root_mean_squared_error: 1.0970 - val_loss: 1.1714 - val_root_mean_squared_error: 1.0875
Epoch 588/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1659 - root_mean_squared_error: 1.0758 - val_loss: 1.1480 - val_root_mean_squared_error: 1.1119
Epoch 589/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1602 - root_mean_squared_error: 1.0845 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0825
Epoch 590/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1612 - root_mean_squared_error: 1.0857 - val_loss: 1.1822 - val_root_mean_squared_error: 1.0629
Epoch 591/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1667 - root_mean_squared_error: 1.1082 - val_loss: 1.1600 - val_root_mean_squared_error: 1.1347
Epoch 592/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1656 - root_mean_squared_error: 1.0864 - val_loss: 1.1879 - val_root_mean_squared_error: 1.1289
Epoch 593/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1653 - root_mean_squared_error: 1.0860 - val_loss: 1.1591 - val_root_mean_squared_error: 1.0656
Epoch 594/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1636 - root_mean_squared_error: 1.0864 - val_loss: 1.1533 - val_root_mean_squared_error: 1.0783
Epoch 595/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1644 - root_mean_squared_error: 1.1030 - val_loss: 1.1648 - val_root_mean_squared_error: 1.0786
Epoch 596/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1677 - root_mean_squared_error: 1.0874 - val_loss: 1.1676 - val_root_mean_squared_error: 1.0957
Epoch 597/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1609 - root_mean_squared_error: 1.0884 - val_loss: 1.1558 - val_root_mean_squared_error: 1.0870
Epoch 598/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1663 - root_mean_squared_error: 1.0925 - val_loss: 1.1419 - val_root_mean_squared_error: 1.0427
Epoch 599/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1605 - root_mean_squared_error: 1.0781 - val_loss: 1.1511 - val_root_mean_squared_error: 1.1142
Epoch 600/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1770 - root_mean_squared_error: 1.0984 - val_loss: 1.1612 - val_root_mean_squared_error: 1.0894
Epoch 601/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1631 - root_mean_squared_error: 1.0942 - val_loss: 1.1473 - val_root_mean_squared_error: 1.0965
Epoch 602/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1563 - root_mean_squared_error: 1.0841 - val_loss: 1.1472 - val_root_mean_squared_error: 1.1222
Epoch 603/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1609 - root_mean_squared_error: 1.0814 - val_loss: 1.1628 - val_root_mean_squared_error: 1.0586
Epoch 604/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1605 - root_mean_squared_error: 1.0811 - val_loss: 1.1466 - val_root_mean_squared_error: 1.0939
Epoch 605/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1623 - root_mean_squared_error: 1.0715 - val_loss: 1.1592 - val_root_mean_squared_error: 1.0855
Epoch 606/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1699 - root_mean_squared_error: 1.1107 - val_loss: 1.1631 - val_root_mean_squared_error: 1.0869
Epoch 607/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1629 - root_mean_squared_error: 1.0728 - val_loss: 1.1608 - val_root_mean_squared_error: 1.0719
Epoch 608/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1611 - root_mean_squared_error: 1.0848 - val_loss: 1.1535 - val_root_mean_squared_error: 1.0468
Epoch 609/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1671 - root_mean_squared_error: 1.0627 - val_loss: 1.1672 - val_root_mean_squared_error: 1.0930
Epoch 610/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1807 - root_mean_squared_error: 1.1134 - val_loss: 1.1831 - val_root_mean_squared_error: 1.1166
Epoch 611/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1713 - root_mean_squared_error: 1.0806 - val_loss: 1.1618 - val_root_mean_squared_error: 1.1265
Epoch 612/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1673 - root_mean_squared_error: 1.1024 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0841
Epoch 613/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1659 - root_mean_squared_error: 1.0995 - val_loss: 1.1780 - val_root_mean_squared_error: 1.1041
Epoch 614/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1621 - root_mean_squared_error: 1.0859 - val_loss: 1.1924 - val_root_mean_squared_error: 1.0976
Epoch 615/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1598 - root_mean_squared_error: 1.0995 - val_loss: 1.1776 - val_root_mean_squared_error: 1.1023
Epoch 616/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1647 - root_mean_squared_error: 1.0825 - val_loss: 1.1652 - val_root_mean_squared_error: 1.0893
Epoch 617/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1705 - root_mean_squared_error: 1.0940 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0818
Epoch 618/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1649 - root_mean_squared_error: 1.0910 - val_loss: 1.1566 - val_root_mean_squared_error: 1.0845
Epoch 619/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1653 - root_mean_squared_error: 1.0687 - val_loss: 1.1479 - val_root_mean_squared_error: 1.0723
Epoch 620/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1665 - root_mean_squared_error: 1.1077 - val_loss: 1.1651 - val_root_mean_squared_error: 1.0704
Epoch 621/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1632 - root_mean_squared_error: 1.0874 - val_loss: 1.1603 - val_root_mean_squared_error: 1.0679
Epoch 622/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1712 - root_mean_squared_error: 1.0730 - val_loss: 1.1595 - val_root_mean_squared_error: 1.0819
Epoch 623/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1648 - root_mean_squared_error: 1.1005 - val_loss: 1.1422 - val_root_mean_squared_error: 1.0911
Epoch 624/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1675 - root_mean_squared_error: 1.0922 - val_loss: 1.1446 - val_root_mean_squared_error: 1.1369
Epoch 625/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1581 - root_mean_squared_error: 1.0870 - val_loss: 1.1625 - val_root_mean_squared_error: 1.0943
Epoch 626/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1627 - root_mean_squared_error: 1.0584 - val_loss: 1.1462 - val_root_mean_squared_error: 1.0529
Epoch 627/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1557 - root_mean_squared_error: 1.0741 - val_loss: 1.1398 - val_root_mean_squared_error: 1.0198
Epoch 628/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1620 - root_mean_squared_error: 1.0790 - val_loss: 1.1863 - val_root_mean_squared_error: 1.0467
Epoch 629/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1644 - root_mean_squared_error: 1.0924 - val_loss: 1.1651 - val_root_mean_squared_error: 1.0871
Epoch 630/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1559 - root_mean_squared_error: 1.0781 - val_loss: 1.1712 - val_root_mean_squared_error: 1.1008
Epoch 631/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1650 - root_mean_squared_error: 1.0733 - val_loss: 1.1671 - val_root_mean_squared_error: 1.0992
Epoch 632/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1543 - root_mean_squared_error: 1.0664 - val_loss: 1.1696 - val_root_mean_squared_error: 1.0778
Epoch 633/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1681 - root_mean_squared_error: 1.0954 - val_loss: 1.1422 - val_root_mean_squared_error: 1.0702
Epoch 634/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1667 - root_mean_squared_error: 1.0799 - val_loss: 1.1665 - val_root_mean_squared_error: 1.0558
Epoch 635/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1701 - root_mean_squared_error: 1.0736 - val_loss: 1.1823 - val_root_mean_squared_error: 1.0666
Epoch 636/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1666 - root_mean_squared_error: 1.0888 - val_loss: 1.1659 - val_root_mean_squared_error: 1.0769
Epoch 637/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1590 - root_mean_squared_error: 1.0824 - val_loss: 1.1459 - val_root_mean_squared_error: 1.0464
Epoch 638/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1674 - root_mean_squared_error: 1.0867 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0242
Epoch 639/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1574 - root_mean_squared_error: 1.0719 - val_loss: 1.1704 - val_root_mean_squared_error: 1.0933
Epoch 640/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1546 - root_mean_squared_error: 1.0634 - val_loss: 1.1700 - val_root_mean_squared_error: 1.0994
Epoch 641/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1570 - root_mean_squared_error: 1.0761 - val_loss: 1.1718 - val_root_mean_squared_error: 1.0891
Epoch 642/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1663 - root_mean_squared_error: 1.0785 - val_loss: 1.1693 - val_root_mean_squared_error: 1.1142
Epoch 643/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1652 - root_mean_squared_error: 1.0567 - val_loss: 1.1634 - val_root_mean_squared_error: 1.0836
Epoch 644/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1617 - root_mean_squared_error: 1.0734 - val_loss: 1.1600 - val_root_mean_squared_error: 1.0960
Epoch 645/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1635 - root_mean_squared_error: 1.0896 - val_loss: 1.1563 - val_root_mean_squared_error: 1.0642
Epoch 646/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1671 - root_mean_squared_error: 1.0998 - val_loss: 1.1548 - val_root_mean_squared_error: 1.0816
Epoch 647/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1655 - root_mean_squared_error: 1.0782 - val_loss: 1.1586 - val_root_mean_squared_error: 1.0921
Epoch 648/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1566 - root_mean_squared_error: 1.0845 - val_loss: 1.1414 - val_root_mean_squared_error: 1.1061
Epoch 649/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1644 - root_mean_squared_error: 1.0823 - val_loss: 1.1564 - val_root_mean_squared_error: 1.0997
Epoch 650/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1723 - root_mean_squared_error: 1.0590 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0718
Epoch 651/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1610 - root_mean_squared_error: 1.0792 - val_loss: 1.1605 - val_root_mean_squared_error: 1.1253
Epoch 652/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1673 - root_mean_squared_error: 1.0900 - val_loss: 1.1699 - val_root_mean_squared_error: 1.0517
Epoch 653/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1597 - root_mean_squared_error: 1.0681 - val_loss: 1.1452 - val_root_mean_squared_error: 1.0999
Epoch 654/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1610 - root_mean_squared_error: 1.0584 - val_loss: 1.1690 - val_root_mean_squared_error: 1.1265
Epoch 655/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1629 - root_mean_squared_error: 1.0650 - val_loss: 1.1903 - val_root_mean_squared_error: 1.1045
Epoch 656/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1587 - root_mean_squared_error: 1.0628 - val_loss: 1.1546 - val_root_mean_squared_error: 1.0615
Epoch 657/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1613 - root_mean_squared_error: 1.0677 - val_loss: 1.1666 - val_root_mean_squared_error: 1.0937
Epoch 658/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1656 - root_mean_squared_error: 1.0667 - val_loss: 1.1725 - val_root_mean_squared_error: 1.0770
Epoch 659/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1637 - root_mean_squared_error: 1.0587 - val_loss: 1.1571 - val_root_mean_squared_error: 1.0964
Epoch 660/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1592 - root_mean_squared_error: 1.0816 - val_loss: 1.1837 - val_root_mean_squared_error: 1.0678
Epoch 661/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1612 - root_mean_squared_error: 1.0655 - val_loss: 1.1623 - val_root_mean_squared_error: 1.0922
Epoch 662/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1564 - root_mean_squared_error: 1.0713 - val_loss: 1.1594 - val_root_mean_squared_error: 1.0834
Epoch 663/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1688 - root_mean_squared_error: 1.0793 - val_loss: 1.1629 - val_root_mean_squared_error: 1.0844
Epoch 664/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1520 - root_mean_squared_error: 1.0671 - val_loss: 1.1690 - val_root_mean_squared_error: 1.0864
Epoch 665/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1827 - root_mean_squared_error: 1.1018 - val_loss: 1.1507 - val_root_mean_squared_error: 1.0817
Epoch 666/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1625 - root_mean_squared_error: 1.0975 - val_loss: 1.1593 - val_root_mean_squared_error: 1.0851
Epoch 667/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1689 - root_mean_squared_error: 1.0764 - val_loss: 1.1484 - val_root_mean_squared_error: 1.0728
Epoch 668/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1628 - root_mean_squared_error: 1.0943 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0867
Epoch 669/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1598 - root_mean_squared_error: 1.0754 - val_loss: 1.1797 - val_root_mean_squared_error: 1.0887
Epoch 670/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1656 - root_mean_squared_error: 1.0893 - val_loss: 1.1628 - val_root_mean_squared_error: 1.0752
Epoch 671/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1651 - root_mean_squared_error: 1.0787 - val_loss: 1.1721 - val_root_mean_squared_error: 1.0862
Epoch 672/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1625 - root_mean_squared_error: 1.0829 - val_loss: 1.1455 - val_root_mean_squared_error: 1.0466
Epoch 673/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1721 - root_mean_squared_error: 1.0723 - val_loss: 1.1582 - val_root_mean_squared_error: 1.0988
Epoch 674/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1711 - root_mean_squared_error: 1.0810 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0410
Epoch 675/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1604 - root_mean_squared_error: 1.0660 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0584
Epoch 676/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1670 - root_mean_squared_error: 1.0855 - val_loss: 1.1623 - val_root_mean_squared_error: 1.0518
Epoch 677/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1679 - root_mean_squared_error: 1.0891 - val_loss: 1.1757 - val_root_mean_squared_error: 1.1123
Epoch 678/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1637 - root_mean_squared_error: 1.0882 - val_loss: 1.1659 - val_root_mean_squared_error: 1.1115
Epoch 679/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1637 - root_mean_squared_error: 1.0910 - val_loss: 1.1562 - val_root_mean_squared_error: 1.0806
Epoch 680/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1545 - root_mean_squared_error: 1.0718 - val_loss: 1.1669 - val_root_mean_squared_error: 1.0759
Epoch 681/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1657 - root_mean_squared_error: 1.0691 - val_loss: 1.1449 - val_root_mean_squared_error: 1.1291
Epoch 682/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1623 - root_mean_squared_error: 1.0602 - val_loss: 1.1633 - val_root_mean_squared_error: 1.0602
Epoch 683/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1609 - root_mean_squared_error: 1.0789 - val_loss: 1.1576 - val_root_mean_squared_error: 1.0937
Epoch 684/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1561 - root_mean_squared_error: 1.0604 - val_loss: 1.1695 - val_root_mean_squared_error: 1.0807
Epoch 685/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1569 - root_mean_squared_error: 1.0758 - val_loss: 1.1702 - val_root_mean_squared_error: 1.1258
Epoch 686/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1597 - root_mean_squared_error: 1.0920 - val_loss: 1.1577 - val_root_mean_squared_error: 1.0617
Epoch 687/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1653 - root_mean_squared_error: 1.0800 - val_loss: 1.1725 - val_root_mean_squared_error: 1.1236
Epoch 688/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1702 - root_mean_squared_error: 1.0739 - val_loss: 1.1795 - val_root_mean_squared_error: 1.0684
Epoch 689/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1646 - root_mean_squared_error: 1.0934 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0806
Epoch 690/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1578 - root_mean_squared_error: 1.0793 - val_loss: 1.1545 - val_root_mean_squared_error: 1.0953
Epoch 691/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1611 - root_mean_squared_error: 1.1009 - val_loss: 1.1540 - val_root_mean_squared_error: 1.0872
Epoch 692/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1580 - root_mean_squared_error: 1.0661 - val_loss: 1.1718 - val_root_mean_squared_error: 1.1084
Epoch 693/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1562 - root_mean_squared_error: 1.0936 - val_loss: 1.1640 - val_root_mean_squared_error: 1.1470
Epoch 694/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1580 - root_mean_squared_error: 1.0836 - val_loss: 1.1568 - val_root_mean_squared_error: 1.0386
Epoch 695/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1591 - root_mean_squared_error: 1.0654 - val_loss: 1.1522 - val_root_mean_squared_error: 1.0505
Epoch 696/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1761 - root_mean_squared_error: 1.0770 - val_loss: 1.1610 - val_root_mean_squared_error: 1.0558
Epoch 697/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1578 - root_mean_squared_error: 1.0929 - val_loss: 1.1628 - val_root_mean_squared_error: 1.0757
Epoch 698/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1626 - root_mean_squared_error: 1.0685 - val_loss: 1.1532 - val_root_mean_squared_error: 1.0940
Epoch 699/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1587 - root_mean_squared_error: 1.0885 - val_loss: 1.1685 - val_root_mean_squared_error: 1.0692
Epoch 700/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1621 - root_mean_squared_error: 1.0629 - val_loss: 1.1715 - val_root_mean_squared_error: 1.0946
Epoch 701/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1689 - root_mean_squared_error: 1.1026 - val_loss: 1.1735 - val_root_mean_squared_error: 1.1221
Epoch 702/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1614 - root_mean_squared_error: 1.0675 - val_loss: 1.1746 - val_root_mean_squared_error: 1.0887
Epoch 703/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1564 - root_mean_squared_error: 1.0867 - val_loss: 1.1661 - val_root_mean_squared_error: 1.1082
Epoch 704/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1727 - root_mean_squared_error: 1.0888 - val_loss: 1.1753 - val_root_mean_squared_error: 1.0885
Epoch 705/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1556 - root_mean_squared_error: 1.0940 - val_loss: 1.1597 - val_root_mean_squared_error: 1.0917
Epoch 706/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1603 - root_mean_squared_error: 1.0829 - val_loss: 1.1540 - val_root_mean_squared_error: 1.0475
Epoch 707/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1527 - root_mean_squared_error: 1.0962 - val_loss: 1.1505 - val_root_mean_squared_error: 1.0672
Epoch 708/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1587 - root_mean_squared_error: 1.0789 - val_loss: 1.1831 - val_root_mean_squared_error: 1.1216
Epoch 709/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1665 - root_mean_squared_error: 1.1025 - val_loss: 1.1535 - val_root_mean_squared_error: 1.0670
Epoch 710/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1631 - root_mean_squared_error: 1.0742 - val_loss: 1.1602 - val_root_mean_squared_error: 1.0641
Epoch 711/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1549 - root_mean_squared_error: 1.0556 - val_loss: 1.1507 - val_root_mean_squared_error: 1.0661
Epoch 712/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1651 - root_mean_squared_error: 1.0701 - val_loss: 1.1519 - val_root_mean_squared_error: 1.1049
Epoch 713/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1622 - root_mean_squared_error: 1.0920 - val_loss: 1.1553 - val_root_mean_squared_error: 1.0723
Epoch 714/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1607 - root_mean_squared_error: 1.0641 - val_loss: 1.1613 - val_root_mean_squared_error: 1.0876
Epoch 715/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1608 - root_mean_squared_error: 1.0797 - val_loss: 1.1476 - val_root_mean_squared_error: 1.0561
Epoch 716/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1542 - root_mean_squared_error: 1.0654 - val_loss: 1.1490 - val_root_mean_squared_error: 1.0711
Epoch 717/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1554 - root_mean_squared_error: 1.0587 - val_loss: 1.1551 - val_root_mean_squared_error: 1.0620
Epoch 718/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1565 - root_mean_squared_error: 1.0694 - val_loss: 1.1582 - val_root_mean_squared_error: 1.0694
Epoch 719/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1661 - root_mean_squared_error: 1.0819 - val_loss: 1.1676 - val_root_mean_squared_error: 1.1044
Epoch 720/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1625 - root_mean_squared_error: 1.0837 - val_loss: 1.1525 - val_root_mean_squared_error: 1.0652
Epoch 721/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1672 - root_mean_squared_error: 1.0733 - val_loss: 1.1497 - val_root_mean_squared_error: 1.0965
Epoch 722/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1589 - root_mean_squared_error: 1.0796 - val_loss: 1.1576 - val_root_mean_squared_error: 1.0908
Epoch 723/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1700 - root_mean_squared_error: 1.0842 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0563
Epoch 724/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1534 - root_mean_squared_error: 1.0680 - val_loss: 1.1484 - val_root_mean_squared_error: 1.0643
Epoch 725/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1557 - root_mean_squared_error: 1.0693 - val_loss: 1.1510 - val_root_mean_squared_error: 1.0857
Epoch 726/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1550 - root_mean_squared_error: 1.0833 - val_loss: 1.1566 - val_root_mean_squared_error: 1.0982
Epoch 727/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1585 - root_mean_squared_error: 1.0689 - val_loss: 1.1557 - val_root_mean_squared_error: 1.0961
Epoch 728/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1714 - root_mean_squared_error: 1.0747 - val_loss: 1.1664 - val_root_mean_squared_error: 1.0124
Epoch 729/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1656 - root_mean_squared_error: 1.0817 - val_loss: 1.1612 - val_root_mean_squared_error: 1.0783
Epoch 730/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1602 - root_mean_squared_error: 1.0811 - val_loss: 1.1634 - val_root_mean_squared_error: 1.0973
Epoch 731/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1544 - root_mean_squared_error: 1.0734 - val_loss: 1.1770 - val_root_mean_squared_error: 1.0560
Epoch 732/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1628 - root_mean_squared_error: 1.0812 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0008
Epoch 733/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1558 - root_mean_squared_error: 1.0739 - val_loss: 1.1444 - val_root_mean_squared_error: 1.1087
Epoch 734/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1655 - root_mean_squared_error: 1.0773 - val_loss: 1.1487 - val_root_mean_squared_error: 1.0868
Epoch 735/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1598 - root_mean_squared_error: 1.0897 - val_loss: 1.1634 - val_root_mean_squared_error: 1.0699
Epoch 736/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1576 - root_mean_squared_error: 1.0929 - val_loss: 1.1666 - val_root_mean_squared_error: 1.1203
Epoch 737/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1596 - root_mean_squared_error: 1.0694 - val_loss: 1.1550 - val_root_mean_squared_error: 1.1193
Epoch 738/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1663 - root_mean_squared_error: 1.0739 - val_loss: 1.1571 - val_root_mean_squared_error: 1.0735
Epoch 739/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1611 - root_mean_squared_error: 1.0792 - val_loss: 1.1645 - val_root_mean_squared_error: 1.1120
Epoch 740/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1655 - root_mean_squared_error: 1.0859 - val_loss: 1.1618 - val_root_mean_squared_error: 1.0638
Epoch 741/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1598 - root_mean_squared_error: 1.0773 - val_loss: 1.1655 - val_root_mean_squared_error: 1.0691
Epoch 742/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1582 - root_mean_squared_error: 1.0853 - val_loss: 1.1746 - val_root_mean_squared_error: 1.0488
Epoch 743/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1616 - root_mean_squared_error: 1.0813 - val_loss: 1.1550 - val_root_mean_squared_error: 1.0638
Epoch 744/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1611 - root_mean_squared_error: 1.0786 - val_loss: 1.1598 - val_root_mean_squared_error: 1.0731
Epoch 745/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1699 - root_mean_squared_error: 1.1046 - val_loss: 1.1643 - val_root_mean_squared_error: 1.0723
Epoch 746/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1569 - root_mean_squared_error: 1.0641 - val_loss: 1.1700 - val_root_mean_squared_error: 1.1204
Epoch 747/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1620 - root_mean_squared_error: 1.0833 - val_loss: 1.1665 - val_root_mean_squared_error: 1.0848
Epoch 748/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1577 - root_mean_squared_error: 1.0837 - val_loss: 1.1498 - val_root_mean_squared_error: 1.0498
Epoch 749/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1755 - root_mean_squared_error: 1.0891 - val_loss: 1.1594 - val_root_mean_squared_error: 1.1160
Epoch 750/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1526 - root_mean_squared_error: 1.0759 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0792
Epoch 751/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1667 - root_mean_squared_error: 1.0778 - val_loss: 1.1587 - val_root_mean_squared_error: 1.0883
Epoch 752/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1619 - root_mean_squared_error: 1.0808 - val_loss: 1.1661 - val_root_mean_squared_error: 1.0734
Epoch 753/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1623 - root_mean_squared_error: 1.0742 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0818
Epoch 754/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1581 - root_mean_squared_error: 1.0658 - val_loss: 1.1900 - val_root_mean_squared_error: 1.1120
Epoch 755/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1610 - root_mean_squared_error: 1.0806 - val_loss: 1.1611 - val_root_mean_squared_error: 1.0749
Epoch 756/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1574 - root_mean_squared_error: 1.0745 - val_loss: 1.1570 - val_root_mean_squared_error: 1.0954
Epoch 757/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1585 - root_mean_squared_error: 1.0746 - val_loss: 1.1576 - val_root_mean_squared_error: 1.0770
Epoch 758/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1611 - root_mean_squared_error: 1.0659 - val_loss: 1.1612 - val_root_mean_squared_error: 1.0956
Epoch 759/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1568 - root_mean_squared_error: 1.0760 - val_loss: 1.1491 - val_root_mean_squared_error: 1.0506
Epoch 760/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1532 - root_mean_squared_error: 1.0717 - val_loss: 1.1619 - val_root_mean_squared_error: 1.0858
Epoch 761/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1560 - root_mean_squared_error: 1.0927 - val_loss: 1.1561 - val_root_mean_squared_error: 1.0315
Epoch 762/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1674 - root_mean_squared_error: 1.0958 - val_loss: 1.1678 - val_root_mean_squared_error: 1.0886
Epoch 763/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1602 - root_mean_squared_error: 1.0819 - val_loss: 1.1525 - val_root_mean_squared_error: 1.0657
Epoch 764/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1574 - root_mean_squared_error: 1.0739 - val_loss: 1.1403 - val_root_mean_squared_error: 1.0980
Epoch 765/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1676 - root_mean_squared_error: 1.0840 - val_loss: 1.1688 - val_root_mean_squared_error: 1.0660
Epoch 766/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1694 - root_mean_squared_error: 1.0867 - val_loss: 1.1559 - val_root_mean_squared_error: 1.1132
Epoch 767/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1578 - root_mean_squared_error: 1.0844 - val_loss: 1.1535 - val_root_mean_squared_error: 1.0934
Epoch 768/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1619 - root_mean_squared_error: 1.0793 - val_loss: 1.1616 - val_root_mean_squared_error: 1.0708
Epoch 769/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1658 - root_mean_squared_error: 1.0904 - val_loss: 1.1515 - val_root_mean_squared_error: 1.0888
Epoch 770/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1537 - root_mean_squared_error: 1.0781 - val_loss: 1.1631 - val_root_mean_squared_error: 1.1074
Epoch 771/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1511 - root_mean_squared_error: 1.0543 - val_loss: 1.1662 - val_root_mean_squared_error: 1.0687
Epoch 772/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1520 - root_mean_squared_error: 1.0940 - val_loss: 1.1527 - val_root_mean_squared_error: 1.0762
Epoch 773/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1604 - root_mean_squared_error: 1.0801 - val_loss: 1.1762 - val_root_mean_squared_error: 1.1023
Epoch 774/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1634 - root_mean_squared_error: 1.0764 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0891
Epoch 775/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1595 - root_mean_squared_error: 1.0672 - val_loss: 1.1527 - val_root_mean_squared_error: 1.0620
Epoch 776/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1590 - root_mean_squared_error: 1.0877 - val_loss: 1.1567 - val_root_mean_squared_error: 1.0817
Epoch 777/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1620 - root_mean_squared_error: 1.0972 - val_loss: 1.1662 - val_root_mean_squared_error: 1.1095
Epoch 778/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1597 - root_mean_squared_error: 1.0679 - val_loss: 1.1569 - val_root_mean_squared_error: 1.0881
Epoch 779/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1539 - root_mean_squared_error: 1.0516 - val_loss: 1.1617 - val_root_mean_squared_error: 1.0867
Epoch 780/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1677 - root_mean_squared_error: 1.0678 - val_loss: 1.1718 - val_root_mean_squared_error: 1.0995
Epoch 781/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1604 - root_mean_squared_error: 1.0919 - val_loss: 1.1611 - val_root_mean_squared_error: 1.0751
Epoch 782/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1540 - root_mean_squared_error: 1.0682 - val_loss: 1.1554 - val_root_mean_squared_error: 1.1092
Epoch 783/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1715 - root_mean_squared_error: 1.0662 - val_loss: 1.1549 - val_root_mean_squared_error: 1.0863
Epoch 784/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1572 - root_mean_squared_error: 1.0985 - val_loss: 1.1436 - val_root_mean_squared_error: 1.0527
Epoch 785/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1630 - root_mean_squared_error: 1.0932 - val_loss: 1.1643 - val_root_mean_squared_error: 1.0967
Epoch 786/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1615 - root_mean_squared_error: 1.1004 - val_loss: 1.1709 - val_root_mean_squared_error: 1.0604
Epoch 787/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1598 - root_mean_squared_error: 1.0774 - val_loss: 1.1678 - val_root_mean_squared_error: 1.1025
Epoch 788/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1678 - root_mean_squared_error: 1.0950 - val_loss: 1.1640 - val_root_mean_squared_error: 1.0763
Epoch 789/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1595 - root_mean_squared_error: 1.0716 - val_loss: 1.1422 - val_root_mean_squared_error: 1.0350
Epoch 790/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1616 - root_mean_squared_error: 1.0732 - val_loss: 1.1565 - val_root_mean_squared_error: 1.0749
Epoch 791/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1569 - root_mean_squared_error: 1.0758 - val_loss: 1.1740 - val_root_mean_squared_error: 1.0549
Epoch 792/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1632 - root_mean_squared_error: 1.0918 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0773
Epoch 793/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1616 - root_mean_squared_error: 1.0838 - val_loss: 1.1574 - val_root_mean_squared_error: 1.0420
Epoch 794/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1707 - root_mean_squared_error: 1.0904 - val_loss: 1.1542 - val_root_mean_squared_error: 1.0957
Epoch 795/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1659 - root_mean_squared_error: 1.0883 - val_loss: 1.1554 - val_root_mean_squared_error: 1.0635
Epoch 796/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1621 - root_mean_squared_error: 1.0628 - val_loss: 1.1444 - val_root_mean_squared_error: 1.0802
Epoch 797/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1570 - root_mean_squared_error: 1.0559 - val_loss: 1.1601 - val_root_mean_squared_error: 1.0405
Epoch 798/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1588 - root_mean_squared_error: 1.0571 - val_loss: 1.1581 - val_root_mean_squared_error: 1.1228
Epoch 799/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1648 - root_mean_squared_error: 1.0867 - val_loss: 1.1618 - val_root_mean_squared_error: 1.0875
Epoch 800/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1557 - root_mean_squared_error: 1.0894 - val_loss: 1.1697 - val_root_mean_squared_error: 1.0589
Epoch 801/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1549 - root_mean_squared_error: 1.0775 - val_loss: 1.1492 - val_root_mean_squared_error: 1.0589
Epoch 802/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1602 - root_mean_squared_error: 1.0799 - val_loss: 1.1545 - val_root_mean_squared_error: 1.0264
Epoch 803/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1546 - root_mean_squared_error: 1.0538 - val_loss: 1.1667 - val_root_mean_squared_error: 1.0886
Epoch 804/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1599 - root_mean_squared_error: 1.0845 - val_loss: 1.1598 - val_root_mean_squared_error: 1.1165
Epoch 805/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1524 - root_mean_squared_error: 1.0835 - val_loss: 1.1598 - val_root_mean_squared_error: 1.0749
Epoch 806/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1527 - root_mean_squared_error: 1.0836 - val_loss: 1.1679 - val_root_mean_squared_error: 1.0951
Epoch 807/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1732 - root_mean_squared_error: 1.0872 - val_loss: 1.1629 - val_root_mean_squared_error: 1.1070
Epoch 808/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1561 - root_mean_squared_error: 1.0747 - val_loss: 1.1655 - val_root_mean_squared_error: 1.0866
Epoch 809/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1604 - root_mean_squared_error: 1.0632 - val_loss: 1.1611 - val_root_mean_squared_error: 0.9967
Epoch 810/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1612 - root_mean_squared_error: 1.0815 - val_loss: 1.1503 - val_root_mean_squared_error: 1.0049
Epoch 811/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1627 - root_mean_squared_error: 1.0784 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0890
Epoch 812/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1604 - root_mean_squared_error: 1.0780 - val_loss: 1.1703 - val_root_mean_squared_error: 1.0731
Epoch 813/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1502 - root_mean_squared_error: 1.0803 - val_loss: 1.1534 - val_root_mean_squared_error: 1.0476
Epoch 814/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1587 - root_mean_squared_error: 1.0770 - val_loss: 1.1429 - val_root_mean_squared_error: 1.0227
Epoch 815/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1590 - root_mean_squared_error: 1.0586 - val_loss: 1.1597 - val_root_mean_squared_error: 1.0611
Epoch 816/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1533 - root_mean_squared_error: 1.0696 - val_loss: 1.1569 - val_root_mean_squared_error: 1.0786
Epoch 817/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1539 - root_mean_squared_error: 1.0761 - val_loss: 1.1502 - val_root_mean_squared_error: 1.0973
Epoch 818/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1566 - root_mean_squared_error: 1.0797 - val_loss: 1.1538 - val_root_mean_squared_error: 1.0606
Epoch 819/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1614 - root_mean_squared_error: 1.0629 - val_loss: 1.1741 - val_root_mean_squared_error: 1.0204
Epoch 820/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1594 - root_mean_squared_error: 1.0704 - val_loss: 1.1635 - val_root_mean_squared_error: 1.0974
Epoch 821/1000
17/17 [==============================] - 0s 4ms/step - loss: 1.1531 - root_mean_squared_error: 1.0602 - val_loss: 1.1469 - val_root_mean_squared_error: 1.0822
Epoch 822/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1566 - root_mean_squared_error: 1.0564 - val_loss: 1.1443 - val_root_mean_squared_error: 1.0548
Epoch 823/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1566 - root_mean_squared_error: 1.0664 - val_loss: 1.1564 - val_root_mean_squared_error: 1.0836
Epoch 824/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1551 - root_mean_squared_error: 1.0728 - val_loss: 1.1535 - val_root_mean_squared_error: 1.0854
Epoch 825/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1591 - root_mean_squared_error: 1.0790 - val_loss: 1.1455 - val_root_mean_squared_error: 1.0446
Epoch 826/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1577 - root_mean_squared_error: 1.0701 - val_loss: 1.1573 - val_root_mean_squared_error: 1.0635
Epoch 827/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1538 - root_mean_squared_error: 1.0607 - val_loss: 1.1590 - val_root_mean_squared_error: 1.0666
Epoch 828/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1580 - root_mean_squared_error: 1.0921 - val_loss: 1.1524 - val_root_mean_squared_error: 1.0995
Epoch 829/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1528 - root_mean_squared_error: 1.0800 - val_loss: 1.1577 - val_root_mean_squared_error: 1.0136
Epoch 830/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1558 - root_mean_squared_error: 1.0558 - val_loss: 1.1864 - val_root_mean_squared_error: 1.0924
Epoch 831/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1514 - root_mean_squared_error: 1.0903 - val_loss: 1.1562 - val_root_mean_squared_error: 1.0749
Epoch 832/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1598 - root_mean_squared_error: 1.0882 - val_loss: 1.1508 - val_root_mean_squared_error: 1.0767
Epoch 833/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1642 - root_mean_squared_error: 1.0838 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0577
Epoch 834/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1569 - root_mean_squared_error: 1.0699 - val_loss: 1.1607 - val_root_mean_squared_error: 1.0742
Epoch 835/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1611 - root_mean_squared_error: 1.0708 - val_loss: 1.1688 - val_root_mean_squared_error: 1.0778
Epoch 836/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1483 - root_mean_squared_error: 1.0745 - val_loss: 1.1506 - val_root_mean_squared_error: 1.0471
Epoch 837/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1536 - root_mean_squared_error: 1.0572 - val_loss: 1.1506 - val_root_mean_squared_error: 1.0578
Epoch 838/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1555 - root_mean_squared_error: 1.0830 - val_loss: 1.1584 - val_root_mean_squared_error: 1.0667
Epoch 839/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1670 - root_mean_squared_error: 1.0729 - val_loss: 1.1692 - val_root_mean_squared_error: 1.0608
Epoch 840/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1577 - root_mean_squared_error: 1.0890 - val_loss: 1.1551 - val_root_mean_squared_error: 1.1297
Epoch 841/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1547 - root_mean_squared_error: 1.0852 - val_loss: 1.1570 - val_root_mean_squared_error: 1.0508
Epoch 842/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1582 - root_mean_squared_error: 1.0875 - val_loss: 1.1626 - val_root_mean_squared_error: 1.0761
Epoch 843/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1602 - root_mean_squared_error: 1.0883 - val_loss: 1.1604 - val_root_mean_squared_error: 1.1105
Epoch 844/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1557 - root_mean_squared_error: 1.0831 - val_loss: 1.1442 - val_root_mean_squared_error: 1.0645
Epoch 845/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1617 - root_mean_squared_error: 1.0719 - val_loss: 1.1693 - val_root_mean_squared_error: 1.1072
Epoch 846/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1534 - root_mean_squared_error: 1.0828 - val_loss: 1.1577 - val_root_mean_squared_error: 1.0630
Epoch 847/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1593 - root_mean_squared_error: 1.0866 - val_loss: 1.1714 - val_root_mean_squared_error: 1.1106
Epoch 848/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1570 - root_mean_squared_error: 1.0858 - val_loss: 1.1549 - val_root_mean_squared_error: 1.1029
Epoch 849/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1523 - root_mean_squared_error: 1.0810 - val_loss: 1.1792 - val_root_mean_squared_error: 1.1096
Epoch 850/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1589 - root_mean_squared_error: 1.0879 - val_loss: 1.1604 - val_root_mean_squared_error: 1.0236
Epoch 851/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1511 - root_mean_squared_error: 1.0503 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0669
Epoch 852/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1570 - root_mean_squared_error: 1.0683 - val_loss: 1.1506 - val_root_mean_squared_error: 1.0644
Epoch 853/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1555 - root_mean_squared_error: 1.0589 - val_loss: 1.1536 - val_root_mean_squared_error: 1.0448
Epoch 854/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1594 - root_mean_squared_error: 1.0739 - val_loss: 1.1786 - val_root_mean_squared_error: 1.0670
Epoch 855/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1618 - root_mean_squared_error: 1.0923 - val_loss: 1.1700 - val_root_mean_squared_error: 1.1184
Epoch 856/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1621 - root_mean_squared_error: 1.0841 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0715
Epoch 857/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1545 - root_mean_squared_error: 1.0690 - val_loss: 1.1444 - val_root_mean_squared_error: 1.0588
Epoch 858/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1521 - root_mean_squared_error: 1.0590 - val_loss: 1.1628 - val_root_mean_squared_error: 1.0502
Epoch 859/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1476 - root_mean_squared_error: 1.0854 - val_loss: 1.1566 - val_root_mean_squared_error: 1.0971
Epoch 860/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1608 - root_mean_squared_error: 1.0796 - val_loss: 1.1824 - val_root_mean_squared_error: 1.0971
Epoch 861/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1491 - root_mean_squared_error: 1.0484 - val_loss: 1.1570 - val_root_mean_squared_error: 1.0297
Epoch 862/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1582 - root_mean_squared_error: 1.0765 - val_loss: 1.1604 - val_root_mean_squared_error: 1.1239
Epoch 863/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1604 - root_mean_squared_error: 1.0852 - val_loss: 1.1541 - val_root_mean_squared_error: 1.0325
Epoch 864/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1636 - root_mean_squared_error: 1.0695 - val_loss: 1.1562 - val_root_mean_squared_error: 1.0447
Epoch 865/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1615 - root_mean_squared_error: 1.0685 - val_loss: 1.1577 - val_root_mean_squared_error: 1.0729
Epoch 866/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1537 - root_mean_squared_error: 1.0770 - val_loss: 1.1623 - val_root_mean_squared_error: 1.1042
Epoch 867/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1586 - root_mean_squared_error: 1.0869 - val_loss: 1.1698 - val_root_mean_squared_error: 1.0793
Epoch 868/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1592 - root_mean_squared_error: 1.0667 - val_loss: 1.1674 - val_root_mean_squared_error: 1.1509
Epoch 869/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1611 - root_mean_squared_error: 1.0868 - val_loss: 1.1746 - val_root_mean_squared_error: 1.0713
Epoch 870/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1454 - root_mean_squared_error: 1.0803 - val_loss: 1.1471 - val_root_mean_squared_error: 1.0904
Epoch 871/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1625 - root_mean_squared_error: 1.0806 - val_loss: 1.1628 - val_root_mean_squared_error: 1.0958
Epoch 872/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1583 - root_mean_squared_error: 1.0825 - val_loss: 1.1449 - val_root_mean_squared_error: 1.0490
Epoch 873/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1543 - root_mean_squared_error: 1.0671 - val_loss: 1.1573 - val_root_mean_squared_error: 1.0505
Epoch 874/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1530 - root_mean_squared_error: 1.0941 - val_loss: 1.1586 - val_root_mean_squared_error: 1.0986
Epoch 875/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1671 - root_mean_squared_error: 1.0849 - val_loss: 1.1528 - val_root_mean_squared_error: 1.0778
Epoch 876/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1560 - root_mean_squared_error: 1.0806 - val_loss: 1.1412 - val_root_mean_squared_error: 1.0673
Epoch 877/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1603 - root_mean_squared_error: 1.0767 - val_loss: 1.1697 - val_root_mean_squared_error: 1.0817
Epoch 878/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1621 - root_mean_squared_error: 1.0841 - val_loss: 1.1589 - val_root_mean_squared_error: 1.0939
Epoch 879/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1561 - root_mean_squared_error: 1.0779 - val_loss: 1.1774 - val_root_mean_squared_error: 1.1123
Epoch 880/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1566 - root_mean_squared_error: 1.0700 - val_loss: 1.1422 - val_root_mean_squared_error: 1.0276
Epoch 881/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1678 - root_mean_squared_error: 1.0728 - val_loss: 1.1457 - val_root_mean_squared_error: 1.0400
Epoch 882/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1545 - root_mean_squared_error: 1.0638 - val_loss: 1.1642 - val_root_mean_squared_error: 1.1042
Epoch 883/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1572 - root_mean_squared_error: 1.0779 - val_loss: 1.1498 - val_root_mean_squared_error: 1.0421
Epoch 884/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1540 - root_mean_squared_error: 1.0911 - val_loss: 1.1621 - val_root_mean_squared_error: 1.0628
Epoch 885/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1535 - root_mean_squared_error: 1.0741 - val_loss: 1.1563 - val_root_mean_squared_error: 1.0889
Epoch 886/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1532 - root_mean_squared_error: 1.0993 - val_loss: 1.1466 - val_root_mean_squared_error: 1.0660
Epoch 887/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1554 - root_mean_squared_error: 1.0801 - val_loss: 1.1469 - val_root_mean_squared_error: 1.0704
Epoch 888/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1581 - root_mean_squared_error: 1.0824 - val_loss: 1.1494 - val_root_mean_squared_error: 1.0610
Epoch 889/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1582 - root_mean_squared_error: 1.0981 - val_loss: 1.1569 - val_root_mean_squared_error: 1.0616
Epoch 890/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1565 - root_mean_squared_error: 1.0702 - val_loss: 1.1507 - val_root_mean_squared_error: 1.0829
Epoch 891/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1550 - root_mean_squared_error: 1.0882 - val_loss: 1.1600 - val_root_mean_squared_error: 1.1011
Epoch 892/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1545 - root_mean_squared_error: 1.0638 - val_loss: 1.1665 - val_root_mean_squared_error: 1.0662
Epoch 893/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1571 - root_mean_squared_error: 1.0652 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0726
Epoch 894/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1580 - root_mean_squared_error: 1.0807 - val_loss: 1.1475 - val_root_mean_squared_error: 1.0540
Epoch 895/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1565 - root_mean_squared_error: 1.0798 - val_loss: 1.1582 - val_root_mean_squared_error: 1.0754
Epoch 896/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1587 - root_mean_squared_error: 1.0600 - val_loss: 1.1609 - val_root_mean_squared_error: 1.0561
Epoch 897/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1513 - root_mean_squared_error: 1.0549 - val_loss: 1.1598 - val_root_mean_squared_error: 1.0631
Epoch 898/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1680 - root_mean_squared_error: 1.1011 - val_loss: 1.1712 - val_root_mean_squared_error: 1.1187
Epoch 899/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1602 - root_mean_squared_error: 1.0873 - val_loss: 1.1488 - val_root_mean_squared_error: 1.0935
Epoch 900/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1578 - root_mean_squared_error: 1.0782 - val_loss: 1.1585 - val_root_mean_squared_error: 1.0790
Epoch 901/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1538 - root_mean_squared_error: 1.0533 - val_loss: 1.1691 - val_root_mean_squared_error: 1.0409
Epoch 902/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1486 - root_mean_squared_error: 1.0682 - val_loss: 1.1571 - val_root_mean_squared_error: 1.1020
Epoch 903/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1538 - root_mean_squared_error: 1.0683 - val_loss: 1.1513 - val_root_mean_squared_error: 1.0656
Epoch 904/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1595 - root_mean_squared_error: 1.0667 - val_loss: 1.1604 - val_root_mean_squared_error: 1.0582
Epoch 905/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1635 - root_mean_squared_error: 1.0828 - val_loss: 1.1518 - val_root_mean_squared_error: 1.1046
Epoch 906/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1501 - root_mean_squared_error: 1.0773 - val_loss: 1.1520 - val_root_mean_squared_error: 1.0653
Epoch 907/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1600 - root_mean_squared_error: 1.0698 - val_loss: 1.1859 - val_root_mean_squared_error: 1.1162
Epoch 908/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1647 - root_mean_squared_error: 1.0820 - val_loss: 1.1529 - val_root_mean_squared_error: 1.1067
Epoch 909/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1561 - root_mean_squared_error: 1.0757 - val_loss: 1.1674 - val_root_mean_squared_error: 1.0708
Epoch 910/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1547 - root_mean_squared_error: 1.0790 - val_loss: 1.1670 - val_root_mean_squared_error: 1.1244
Epoch 911/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1546 - root_mean_squared_error: 1.0717 - val_loss: 1.1497 - val_root_mean_squared_error: 1.0488
Epoch 912/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1514 - root_mean_squared_error: 1.0803 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0738
Epoch 913/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1533 - root_mean_squared_error: 1.0612 - val_loss: 1.1423 - val_root_mean_squared_error: 1.0436
Epoch 914/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1539 - root_mean_squared_error: 1.0588 - val_loss: 1.1838 - val_root_mean_squared_error: 1.0922
Epoch 915/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1596 - root_mean_squared_error: 1.0539 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0660
Epoch 916/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1597 - root_mean_squared_error: 1.0723 - val_loss: 1.1382 - val_root_mean_squared_error: 1.0900
Epoch 917/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1514 - root_mean_squared_error: 1.0719 - val_loss: 1.1712 - val_root_mean_squared_error: 1.1009
Epoch 918/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1539 - root_mean_squared_error: 1.0772 - val_loss: 1.1536 - val_root_mean_squared_error: 1.0906
Epoch 919/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1574 - root_mean_squared_error: 1.0692 - val_loss: 1.1689 - val_root_mean_squared_error: 1.0662
Epoch 920/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1542 - root_mean_squared_error: 1.0712 - val_loss: 1.1643 - val_root_mean_squared_error: 1.0753
Epoch 921/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1525 - root_mean_squared_error: 1.0776 - val_loss: 1.1722 - val_root_mean_squared_error: 1.0890
Epoch 922/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1616 - root_mean_squared_error: 1.0940 - val_loss: 1.1556 - val_root_mean_squared_error: 1.0719
Epoch 923/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1509 - root_mean_squared_error: 1.0772 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0634
Epoch 924/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1519 - root_mean_squared_error: 1.0677 - val_loss: 1.1528 - val_root_mean_squared_error: 1.1041
Epoch 925/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1541 - root_mean_squared_error: 1.0761 - val_loss: 1.1581 - val_root_mean_squared_error: 1.0448
Epoch 926/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1534 - root_mean_squared_error: 1.0684 - val_loss: 1.1623 - val_root_mean_squared_error: 1.0877
Epoch 927/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1503 - root_mean_squared_error: 1.0780 - val_loss: 1.1575 - val_root_mean_squared_error: 1.0817
Epoch 928/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1640 - root_mean_squared_error: 1.0782 - val_loss: 1.1492 - val_root_mean_squared_error: 1.0996
Epoch 929/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1535 - root_mean_squared_error: 1.0620 - val_loss: 1.1477 - val_root_mean_squared_error: 1.0387
Epoch 930/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1564 - root_mean_squared_error: 1.0913 - val_loss: 1.1490 - val_root_mean_squared_error: 1.0827
Epoch 931/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1531 - root_mean_squared_error: 1.0767 - val_loss: 1.1426 - val_root_mean_squared_error: 1.0768
Epoch 932/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1472 - root_mean_squared_error: 1.0534 - val_loss: 1.1628 - val_root_mean_squared_error: 1.1103
Epoch 933/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1511 - root_mean_squared_error: 1.0796 - val_loss: 1.1571 - val_root_mean_squared_error: 1.0896
Epoch 934/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1570 - root_mean_squared_error: 1.0759 - val_loss: 1.1503 - val_root_mean_squared_error: 1.0374
Epoch 935/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1601 - root_mean_squared_error: 1.0827 - val_loss: 1.1580 - val_root_mean_squared_error: 1.0773
Epoch 936/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1531 - root_mean_squared_error: 1.0675 - val_loss: 1.1635 - val_root_mean_squared_error: 1.1202
Epoch 937/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1569 - root_mean_squared_error: 1.0778 - val_loss: 1.1466 - val_root_mean_squared_error: 1.0417
Epoch 938/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1557 - root_mean_squared_error: 1.0616 - val_loss: 1.1446 - val_root_mean_squared_error: 1.0622
Epoch 939/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1487 - root_mean_squared_error: 1.0603 - val_loss: 1.1665 - val_root_mean_squared_error: 1.0219
Epoch 940/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1594 - root_mean_squared_error: 1.0707 - val_loss: 1.1544 - val_root_mean_squared_error: 1.0743
Epoch 941/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1564 - root_mean_squared_error: 1.0726 - val_loss: 1.1479 - val_root_mean_squared_error: 1.0455
Epoch 942/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1481 - root_mean_squared_error: 1.0644 - val_loss: 1.1541 - val_root_mean_squared_error: 1.0698
Epoch 943/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1628 - root_mean_squared_error: 1.0714 - val_loss: 1.1698 - val_root_mean_squared_error: 1.0812
Epoch 944/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1590 - root_mean_squared_error: 1.0730 - val_loss: 1.1643 - val_root_mean_squared_error: 1.1394
Epoch 945/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1579 - root_mean_squared_error: 1.0974 - val_loss: 1.1561 - val_root_mean_squared_error: 1.0315
Epoch 946/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1564 - root_mean_squared_error: 1.0809 - val_loss: 1.1436 - val_root_mean_squared_error: 1.0306
Epoch 947/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1534 - root_mean_squared_error: 1.0831 - val_loss: 1.1469 - val_root_mean_squared_error: 1.0707
Epoch 948/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1505 - root_mean_squared_error: 1.0828 - val_loss: 1.1805 - val_root_mean_squared_error: 1.1009
Epoch 949/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1528 - root_mean_squared_error: 1.0681 - val_loss: 1.1444 - val_root_mean_squared_error: 1.0601
Epoch 950/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1521 - root_mean_squared_error: 1.0603 - val_loss: 1.1632 - val_root_mean_squared_error: 1.0444
Epoch 951/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1549 - root_mean_squared_error: 1.0580 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0578
Epoch 952/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1582 - root_mean_squared_error: 1.0610 - val_loss: 1.1469 - val_root_mean_squared_error: 1.1447
Epoch 953/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1543 - root_mean_squared_error: 1.0594 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0710
Epoch 954/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1596 - root_mean_squared_error: 1.0752 - val_loss: 1.1493 - val_root_mean_squared_error: 1.0478
Epoch 955/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1541 - root_mean_squared_error: 1.0698 - val_loss: 1.1483 - val_root_mean_squared_error: 1.0894
Epoch 956/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1578 - root_mean_squared_error: 1.0743 - val_loss: 1.1498 - val_root_mean_squared_error: 1.1012
Epoch 957/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1578 - root_mean_squared_error: 1.0776 - val_loss: 1.1523 - val_root_mean_squared_error: 1.0735
Epoch 958/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1532 - root_mean_squared_error: 1.0957 - val_loss: 1.1528 - val_root_mean_squared_error: 1.0379
Epoch 959/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1568 - root_mean_squared_error: 1.0879 - val_loss: 1.1521 - val_root_mean_squared_error: 1.0544
Epoch 960/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1552 - root_mean_squared_error: 1.0815 - val_loss: 1.1441 - val_root_mean_squared_error: 1.0641
Epoch 961/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1505 - root_mean_squared_error: 1.0829 - val_loss: 1.1396 - val_root_mean_squared_error: 1.0649
Epoch 962/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1550 - root_mean_squared_error: 1.0887 - val_loss: 1.1554 - val_root_mean_squared_error: 1.0825
Epoch 963/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1535 - root_mean_squared_error: 1.0693 - val_loss: 1.1570 - val_root_mean_squared_error: 1.0449
Epoch 964/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1572 - root_mean_squared_error: 1.0628 - val_loss: 1.1576 - val_root_mean_squared_error: 1.0816
Epoch 965/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1455 - root_mean_squared_error: 1.0575 - val_loss: 1.1635 - val_root_mean_squared_error: 1.0915
Epoch 966/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1630 - root_mean_squared_error: 1.0709 - val_loss: 1.1482 - val_root_mean_squared_error: 1.0878
Epoch 967/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1474 - root_mean_squared_error: 1.0750 - val_loss: 1.1509 - val_root_mean_squared_error: 1.1075
Epoch 968/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1677 - root_mean_squared_error: 1.0841 - val_loss: 1.1725 - val_root_mean_squared_error: 1.0555
Epoch 969/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1587 - root_mean_squared_error: 1.0893 - val_loss: 1.1557 - val_root_mean_squared_error: 1.0959
Epoch 970/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1543 - root_mean_squared_error: 1.0607 - val_loss: 1.1534 - val_root_mean_squared_error: 1.0272
Epoch 971/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1540 - root_mean_squared_error: 1.0465 - val_loss: 1.1726 - val_root_mean_squared_error: 1.1018
Epoch 972/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1591 - root_mean_squared_error: 1.0645 - val_loss: 1.1606 - val_root_mean_squared_error: 1.1088
Epoch 973/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1565 - root_mean_squared_error: 1.0864 - val_loss: 1.1605 - val_root_mean_squared_error: 1.0813
Epoch 974/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1581 - root_mean_squared_error: 1.0819 - val_loss: 1.1604 - val_root_mean_squared_error: 1.0767
Epoch 975/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1546 - root_mean_squared_error: 1.0806 - val_loss: 1.1562 - val_root_mean_squared_error: 1.0843
Epoch 976/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1464 - root_mean_squared_error: 1.0725 - val_loss: 1.1700 - val_root_mean_squared_error: 1.0860
Epoch 977/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1615 - root_mean_squared_error: 1.0967 - val_loss: 1.1578 - val_root_mean_squared_error: 1.1233
Epoch 978/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1587 - root_mean_squared_error: 1.0822 - val_loss: 1.1499 - val_root_mean_squared_error: 1.1038
Epoch 979/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1507 - root_mean_squared_error: 1.0633 - val_loss: 1.1384 - val_root_mean_squared_error: 1.0679
Epoch 980/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1528 - root_mean_squared_error: 1.0630 - val_loss: 1.1574 - val_root_mean_squared_error: 1.0826
Epoch 981/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1577 - root_mean_squared_error: 1.0750 - val_loss: 1.1534 - val_root_mean_squared_error: 1.0835
Epoch 982/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1568 - root_mean_squared_error: 1.0859 - val_loss: 1.1549 - val_root_mean_squared_error: 1.1044
Epoch 983/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1536 - root_mean_squared_error: 1.0754 - val_loss: 1.1655 - val_root_mean_squared_error: 1.1283
Epoch 984/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1580 - root_mean_squared_error: 1.0721 - val_loss: 1.1525 - val_root_mean_squared_error: 1.0515
Epoch 985/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1549 - root_mean_squared_error: 1.0650 - val_loss: 1.1570 - val_root_mean_squared_error: 1.0654
Epoch 986/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1593 - root_mean_squared_error: 1.0755 - val_loss: 1.1408 - val_root_mean_squared_error: 1.0505
Epoch 987/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1431 - root_mean_squared_error: 1.0655 - val_loss: 1.1599 - val_root_mean_squared_error: 1.0884
Epoch 988/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1512 - root_mean_squared_error: 1.0655 - val_loss: 1.1601 - val_root_mean_squared_error: 1.0671
Epoch 989/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1513 - root_mean_squared_error: 1.0693 - val_loss: 1.1564 - val_root_mean_squared_error: 1.0609
Epoch 990/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1556 - root_mean_squared_error: 1.0710 - val_loss: 1.1591 - val_root_mean_squared_error: 1.0956
Epoch 991/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1593 - root_mean_squared_error: 1.0685 - val_loss: 1.1457 - val_root_mean_squared_error: 1.0729
Epoch 992/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1534 - root_mean_squared_error: 1.0723 - val_loss: 1.1785 - val_root_mean_squared_error: 1.0594
Epoch 993/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1574 - root_mean_squared_error: 1.0937 - val_loss: 1.1487 - val_root_mean_squared_error: 1.1002
Epoch 994/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1519 - root_mean_squared_error: 1.0700 - val_loss: 1.1764 - val_root_mean_squared_error: 1.1213
Epoch 995/1000
17/17 [==============================] - 0s 3ms/step - loss: 1.1570 - root_mean_squared_error: 1.0782 - val_loss: 1.1630 - val_root_mean_squared_error: 1.0611
Epoch 996/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1533 - root_mean_squared_error: 1.0748 - val_loss: 1.1555 - val_root_mean_squared_error: 1.1129
Epoch 997/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1519 - root_mean_squared_error: 1.0802 - val_loss: 1.1625 - val_root_mean_squared_error: 1.0620
Epoch 998/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1489 - root_mean_squared_error: 1.0847 - val_loss: 1.1560 - val_root_mean_squared_error: 1.0709
Epoch 999/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1582 - root_mean_squared_error: 1.0569 - val_loss: 1.1482 - val_root_mean_squared_error: 1.0917
Epoch 1000/1000
17/17 [==============================] - 0s 2ms/step - loss: 1.1502 - root_mean_squared_error: 1.0777 - val_loss: 1.1559 - val_root_mean_squared_error: 1.0803
Model training finished.
Train RMSE: 1.07
Evaluating model performance...
Test RMSE: 1.078
```
</div>
    

Now let's produce an output from the model given the test examples.
The output is now a distribution, and we can use its mean and variance
to compute the confidence intervals (CI) of the prediction.


```python
prediction_distribution = prob_bnn_model(examples)
prediction_mean = prediction_distribution.mean().numpy().tolist()
prediction_stdv = prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean Â± (1.96 * stdv)
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
Prediction mean: 6.2, stddev: 0.78, 95% CI: [7.74 - 4.66] - Actual: 6.0
Prediction mean: 5.07, stddev: 0.71, 95% CI: [6.46 - 3.68] - Actual: 4.0
Prediction mean: 5.99, stddev: 0.74, 95% CI: [7.44 - 4.53] - Actual: 5.0
Prediction mean: 6.45, stddev: 0.81, 95% CI: [8.04 - 4.85] - Actual: 7.0
Prediction mean: 5.72, stddev: 0.73, 95% CI: [7.15 - 4.3] - Actual: 5.0
Prediction mean: 6.14, stddev: 0.79, 95% CI: [7.7 - 4.59] - Actual: 6.0
Prediction mean: 5.21, stddev: 0.72, 95% CI: [6.62 - 3.8] - Actual: 6.0
Prediction mean: 6.3, stddev: 0.81, 95% CI: [7.89 - 4.71] - Actual: 6.0
Prediction mean: 6.29, stddev: 0.81, 95% CI: [7.89 - 4.7] - Actual: 7.0
Prediction mean: 6.46, stddev: 0.81, 95% CI: [8.05 - 4.86] - Actual: 6.0
```
</div>
    

**Example available on HuggingFace**
| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model%3A%20-Bayesian%20Neural%20Networks-black.svg)](https://huggingface.co/keras-io/ProbabalisticBayesianModel-Wine) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces%3A-Bayesian%20Neural%20Networks-black.svg)](https://huggingface.co/spaces/keras-io/ProbabilisticBayesianNetwork) |
