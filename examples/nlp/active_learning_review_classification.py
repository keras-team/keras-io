"""
Title: Review Classification using Active Learning
Author: [Darshan Deshpande](https://twitter.com/getdarshan)
Date created: 2021/10/29
Last modified: 2021/10/29
Description: Demonstrating the advantages of active learning through review classification.
"""
"""
## Introduction

With the growth of data-centric Machine Learning, Active Learning has grown in popularity
amongst businesses and researchers. Active Learning is a technique of progressively
training ML models so that the resultant model requires lesser amount of training data to
achieve competitive scores.

This tutorial highlights a simple example of how active learning works by demonstrating a
ratio-based sampling strategy that results in lower overall false positive and negative
rates as compared to a model trained on the full dataset.
"""

"""
## Importing required libraries
"""

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

"""
#Loading and preprocessing the data

We will be using the IMDB reviews dataset for our experiments. This dataset has 50,000
reviews in total, including training and testing splits. We will merge these splits and
sample our own, balanced training, validation and testing sets.
"""

dataset = tfds.load("imdb_reviews", split="train + test", as_supervised=True)
print("Total examples: ", len(dataset))

"""
The basic concept of active learning involves labelling a subset of data at first. For
the ratio sampling technique that we will be using, we will need well balanced training,
validation and testing splits.
"""

# Splitting the dataset into two datasets with 0 and 1 labels
zeros_dataset = dataset.filter(lambda x, y: y == 0)
ones_dataset = dataset.filter(lambda x, y: y == 1)

# Initial split will contain 5000 samples each for validation and testing splits
# and 15000 samples for training

val_split = 2500
test_split = 2500
train_samples = 7500

# Creating the training, validation and testing datasets
val_dataset = zeros_dataset.take(val_split).concatenate(ones_dataset.take(val_split))
test_dataset = (
    zeros_dataset.skip(val_split)
    .take(test_split)
    .concatenate(ones_dataset.skip(val_split).take(test_split))
)
train_dataset = (
    zeros_dataset.skip(val_split + test_split)
    .take(train_samples)
    .concatenate(ones_dataset.skip(val_split + test_split).take(train_samples))
)

print(f"Initial training set size: {train_samples*2}")
print(f"Validation set size: {val_split*2}")
print(f"Testing set size: {test_split*2}")

# Dumping the remaining samples in their respective data pools
pool_zeros = zeros_dataset.skip(val_split + test_split + train_samples)
pool_ones = ones_dataset.skip(val_split + test_split + train_samples)

print(f"Unlabeled zeros pool: {25000-val_split-test_split-train_samples}")
print(f"Unlabeled ones pool: {25000-val_split-test_split-train_samples}")

"""
### Fitting the TextVectorization layer

Since we are working with textual data, we will need to encode the text as vectors which
would then be passed through an Embedding layer. To make this tokenization process
faster, we use the `map()` function with it's parallelization functionality.
"""

vectorizer = layers.TextVectorization(
    3000,
    output_sequence_length=150,
)
vectorizer.adapt(
    train_dataset.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).batch(256)
)


def vectorize_text(text, label):
    text = vectorizer(text)
    return text, label


train_dataset = train_dataset.map(
    vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
pool_zeros = pool_zeros.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
pool_ones = pool_ones.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(256).map(
    vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
)
test_dataset = test_dataset.batch(256).map(
    vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
)

"""
## Creating Helper Functions
"""

# Helper function for appending new history objects with older ones
def append_history(losses, val_losses, acc, val_acc, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    acc = acc + history.history["accuracy"]
    val_acc = val_acc + history.history["val_accuracy"]
    return losses, val_losses, acc, val_acc


# Plotter function
def plot_history(losses, val_losses, accuracies, val_accuracies):
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(["train_accuracy", "val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


"""
## Creating the Model
"""


def create_model():
    model = keras.models.Sequential()
    model.add(layers.Embedding(3000, 128))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(20, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()
    return model


"""
## Training on the entire dataset
"""

# Sampling the full train dataset to train on
full_train = (
    train_dataset.concatenate(pool_ones).concatenate(pool_zeros).cache().shuffle(20000)
)

model = create_model()
model.compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics=[
        "accuracy",
        keras.metrics.FalseNegatives(),
        keras.metrics.FalsePositives(),
    ],
)

# We will save the best model at every epoch and load the best one for evaluation on the
test set
history = model.fit(
    full_train.batch(256),
    epochs=20,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=4, verbose=1),
        keras.callbacks.ModelCheckpoint(
            "FullModelCheckpoint.h5", verbose=1, save_best_only=True
        ),
    ],
)

plot_history(
    history.history["loss"],
    history.history["val_loss"],
    history.history["accuracy"],
    history.history["val_accuracy"],
)

# Loading the best checkpoint
model = keras.models.load_model("FullModelCheckpoint.h5")

print("-" * 100)
print(
    "Test set evaluation: ", model.evaluate(test_dataset, verbose=0, return_dict=True)
)
print("-" * 100)

"""
## Training via Active Learning
A general pipeline that we follow when performing active learning is demonstrated below:

![Active Learning](https://i.imgur.com/dmNKusp.png)

The pipeline can be summarized in five parts:

1. Sample and annotate a small, balanced training dataset
2. Train the model on this small subset
3. Evaluate the model on a balanced testing set
4. If the model satisfies the business criteria, deploy it in a real time setting
5. If it doesn't pass the criteria, sample a few more samples according to the ratio of
false positives and negatives, add them to the training set and repeat from step 2 till
the model passes the tests or till all available data is exhausted.
"""


def train_small_models(
    train_dataset,
    pool_zeros,
    pool_ones,
    val_dataset,
    test_dataset,
    iters=3,
    sampling_size=5000,
):

    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies = [], [], [], []
    dataset_len = train_samples * 2

    model = create_model()
    # We will monitor the false positives and false negatives predicted by our model
    # These will decide the subsequent sampling ratio for every active learning loop
    model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=[
            "accuracy",
            keras.metrics.FalseNegatives(),
            keras.metrics.FalsePositives(),
        ],
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        "Checkpoint.h5", save_best_only=True, verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=4, verbose=1)

    print(f"Starting to train with {dataset_len} samples")
    # Initial fit with a small subset of the training set
    history = model.fit(
        train_dataset.cache().shuffle(20000).batch(256),
        epochs=20,
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping],
    )

    # Appending history
    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )

    for iter_n in range(iters):

        # Getting predictions from previously trained model
        predictions = model.predict(test_dataset)

        # Generating labels from the output probabilities
        rounded = tf.where(tf.greater(predictions, 0.5), 1, 0)

        # Evaluating the number of zeros and ones incorrrectly classified
        _, _, counter_zeros, counter_ones = model.evaluate(test_dataset, verbose=0)

        print("-" * 100)
        print(
f"Number of zeros incorrectly classified: {counter_zeros}, Number of ones incorrectly
classified: {counter_ones}"
        )

        # This technique of active learning demonstrates ratio based sampling where
# Number of ones/zeros to sample = Number of ones/zeros incorrectly classified / Total
incorrectly classified
        if counter_zeros != 0 and counter_ones != 0:
            total = counter_zeros + counter_ones
            sample_ratio_ones, sample_ratio_zeros = (
                counter_ones / total,
                counter_zeros / total,
            )
# In the case where all samples are correctly predicted, we can sample both classes
equally
        else:
            sample_ratio_ones, sample_ratio_zeros = 0.5, 0.5

        print(
f"Sample ratio ones: {sample_ratio_ones}, Sample ratio zeros:
{sample_ratio_zeros}"
        )

        # Sample the required number of ones and zeros
        sampled_dataset = pool_zeros.take(
            int(sample_ratio_zeros * sampling_size)
        ).concatenate(pool_ones.take(int(sample_ratio_ones * sampling_size)))

        # Skip the sampled data points so as to avoid repetition
        pool_zeros = pool_zeros.skip(int(sample_ratio_zeros * sampling_size))
        pool_ones = pool_ones.skip(int(sample_ratio_ones * sampling_size))

        # Concatenating the train_dataset with the sampled_dataset
        train_dataset = train_dataset.concatenate(sampled_dataset).prefetch(
            tf.data.AUTOTUNE
        )

        dataset_len += sampling_size
        print(f"Starting training with {dataset_len} samples")
        print("-" * 100)

        # We recompile the model to reset the optimizer states and retrain the model
        model.compile(
            loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=[
                "accuracy",
                keras.metrics.FalseNegatives(),
                keras.metrics.FalsePositives(),
            ],
        )
        history = model.fit(
            train_dataset.cache().shuffle(20000).batch(256),
            validation_data=val_dataset,
            epochs=20,
            callbacks=[
                checkpoint,
                keras.callbacks.EarlyStopping(patience=4, verbose=1),
            ],
        )

        # Appending the history
        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

        # Loading the best model from this training loop
        model = keras.models.load_model("Checkpoint.h5")

    # Plotting the overall history and evaluating the final model
    plot_history(losses, val_losses, accuracies, val_accuracies)
    print("-" * 100)
    print(
        "Test set evaluation: ",
        model.evaluate(test_dataset, verbose=0, return_dict=True),
    )
    print("-" * 100)

    return model


al_model = train_small_models(
    train_dataset, pool_zeros, pool_ones, val_dataset, test_dataset
)

"""
## Conclusion

Active Learning is a growing area of research. This example demonstrates the benefits of
using Active Learning from a business standpoint, as it eliminates the need to annotate
large amounts of data and thus saves money.

The following are some noteworthy observations from this example:

1. We only require 30,000 samples to reach the same (if not better) scores as the model
trained on the full datatset. This means that in a real life setting, we save the effort
required for annotating 10,000 images!
2. The number of false negatives and false positives are well balanced at the end of the
training as compared to the skewed ratio obtained from the full training. This makes the
model slightly more useful in real life scenarios where both the labels hold equal
importance.

For further reading about the types of sampling ratios, training techniques or available
open source libraries/implementations, you can refer to the resources below:

1. Wikipedia: [Query strategies for active
learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)#Query_strategies)
2. [modAL](https://github.com/modAL-python/modAL): A Modular Active Learning framework
3. Google's unofficial [active learning playground](https://github.com/google/active-learning)
"""
