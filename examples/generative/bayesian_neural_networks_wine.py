"""
Title: Probabilistic Bayesian Neural Networks
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/15
Last modified: 2021/01/15
Description: Building probabilistic Bayesian neural network models with TensorFlow Probability.
"""

"""
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
"""

"""
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
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

"""
## Create training and evaluation datasets

Here, we load the `wine_quality` dataset using `tfds.load()`, and we convert
the target feature to float. Then, we shuffle the dataset and split it into
training and test sets. We take the first `train_size` examples as the train
split, and the rest as the test split.
"""


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


"""
## Compile, train, and evaluate the model
"""

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


"""
## Create model inputs
"""

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


"""
## Experiment 1: standard neural network

We create a standard deterministic neural network model as a baseline.
"""


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


"""
Let's split the wine dataset into training and test sets, with 85% and 15% of
the examples, respectively.
"""

dataset_size = 4898
batch_size = 256
train_size = int(dataset_size * 0.85)
train_dataset, test_dataset = get_train_and_test_splits(train_size, batch_size)

"""
Now let's train the baseline model. We use the `MeanSquaredError`
as the loss function.
"""

num_epochs = 100
mse_loss = keras.losses.MeanSquaredError()
baseline_model = create_baseline_model()
run_experiment(baseline_model, mse_loss, train_dataset, test_dataset)

"""
We take a sample from the test set use the model to obtain predictions for them.
Note that since the baseline model is deterministic, we get a single a
*point estimate* prediction for each test example, with no information about the
uncertainty of the model nor the prediction.
"""

sample = 10
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[
    0
]

predicted = baseline_model(examples).numpy()
for idx in range(sample):
    print(f"Predicted: {round(float(predicted[idx][0]), 1)} - Actual: {targets[idx]}")

"""
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
"""

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


"""
We use the `tfp.layers.DenseVariational` layer instead of the standard
`keras.layers.Dense` layer in the neural network model.
"""


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


"""
The epistemic uncertainty can be reduced as we increase the size of the
training data. That is, the more data the BNN model sees, the more it is certain
about its estimates for the weights (distribution parameters).
Let's test this behaviour by training the BNN model on a small subset of
the training set, and then on the full training set, to compare the output variances.
"""

"""
### Train BNN  with a small training subset.
"""

num_epochs = 500
train_sample_size = int(train_size * 0.3)
small_train_dataset = train_dataset.unbatch().take(train_sample_size).batch(batch_size)

bnn_model_small = create_bnn_model(train_sample_size)
run_experiment(bnn_model_small, mse_loss, small_train_dataset, test_dataset)

"""
Since we have trained a BNN model, the model produces a different output each time
we call it with the same input, since each time a new set of weights are sampled
from the distributions to construct the network and produce an output.
The less certain the mode weights are, the more variability (wider range) we will
see in the outputs of the same inputs.
"""


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

"""
### Train BNN  with the whole training set.
"""

num_epochs = 500
bnn_model_full = create_bnn_model(train_size)
run_experiment(bnn_model_full, mse_loss, train_dataset, test_dataset)

compute_predictions(bnn_model_full)

"""
Notice that the model trained with the full training dataset shows smaller range
(uncertainty) in the prediction values for the same inputs, compared to the model
trained with a subset of the training dataset.
"""

"""
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
"""


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


"""
Since the output of the model is a distribution, rather than a point estimate,
we use the [negative loglikelihood](https://en.wikipedia.org/wiki/Likelihood_function)
as our loss function to compute how likely to see the true data (targets) from the
estimated distribution produced by the model.
"""


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


num_epochs = 1000
prob_bnn_model = create_probablistic_bnn_model(train_size)
run_experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset)

"""
Now let's produce an output from the model given the test examples.
The output is now a distribution, and we can use its mean and variance
to compute the confidence intervals (CI) of the prediction.
"""

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
