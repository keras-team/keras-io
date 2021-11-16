# Review Classification using Active Learning

**Author:** [Darshan Deshpande](https://twitter.com/getdarshan)<br>
**Date created:** 2021/10/29<br>
**Last modified:** 2021/10/29<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/active_learning_review_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/active_learning_review_classification.py)


**Description:** Demonstrating the advantages of active learning through review classification.

---
## Introduction

With the growth of data-centric Machine Learning, Active Learning has grown in popularity
amongst businesses and researchers. Active Learning seeks to progressively
train ML models so that the resultant model requires lesser amount of training data to
achieve competitive scores.

The structure of an Active Learning pipeline involves a classifier and an oracle. The
oracle is an annotator that cleans, selects, labels the data, and feeds it to the model
when required. The oracle is a trained individual or a group of individuals that
ensure consistency in labeling of new data.

The process starts with annotating a small subset of the full dataset and training an
initial model. The best model checkpoint is saved and then tested on a balanced test
set. The test set must be carefully sampled because the full training process will be
dependent on it. Once we have the initial evaluation scores, the oracle is tasked with
labeling more samples; the number of data points to be sampled is usually determined by
the business requirements. After that, the newly sampled data is added to the training
set, and the training procedure repeats. This cycle continues until either an
acceptable score is reached or some other business metric is met.

This tutorial provides a basic demonstration of how Active Learning works by
demonstrating a ratio-based (least confidence) sampling strategy that results in lower
overall false positive and negative rates when compared to a model trained on the entire
dataset. This sampling falls under the domain of *uncertanity sampling*, in which new
datasets are sampled based on the uncertanity that the model outputs for the
corresponding label. In our example, we compare our model's false positive and false
negative rates and annotate the new data based on their ratio.

Some other sampling techniques include:

1. [Committee sampling](https://www.researchgate.net/publication/51909346_Committee-Based_Sample_Selection_for_Probabilistic_Classifiers):
Using multiple models to vote for the best data points to be sampled
2. [Entropy reduction](https://www.researchgate.net/publication/51909346_Committee-Based_Sample_Selection_for_Probabilistic_Classifiers):
Sampling according to an entropy threshold, selecting more of the samples that produce the highest entropy score.
3. [Minimum margin based sampling](https://arxiv.org/abs/1906.00025v1):
Selects data points closest to the decision boundary

---
## Importing required libraries


```python
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import re
import string

tfds.disable_progress_bar()
```

---
## Loading and preprocessing the data

We will be using the IMDB reviews dataset for our experiments. This dataset has 50,000
reviews in total, including training and testing splits. We will merge these splits and
sample our own, balanced training, validation and testing sets.


```python
dataset = tfds.load(
    "imdb_reviews",
    split="train + test",
    as_supervised=True,
    batch_size=-1,
    shuffle_files=False,
)
reviews, labels = tfds.as_numpy(dataset)

print("Total examples:", reviews.shape[0])
```

<div class="k-default-codeblock">
```
Total examples: 50000

```
</div>
Active learning starts with labeling a subset of data.
For the ratio sampling technique that we will be using, we will need well-balanced training,
validation and testing splits.


```python
val_split = 2500
test_split = 2500
train_split = 7500

# Separating the negative and positive samples for manual stratification
x_positives, y_positives = reviews[labels == 1], labels[labels == 1]
x_negatives, y_negatives = reviews[labels == 0], labels[labels == 0]

# Creating training, validation and testing splits
x_val, y_val = (
    tf.concat((x_positives[:val_split], x_negatives[:val_split]), 0),
    tf.concat((y_positives[:val_split], y_negatives[:val_split]), 0),
)
x_test, y_test = (
    tf.concat(
        (
            x_positives[val_split : val_split + test_split],
            x_negatives[val_split : val_split + test_split],
        ),
        0,
    ),
    tf.concat(
        (
            y_positives[val_split : val_split + test_split],
            y_negatives[val_split : val_split + test_split],
        ),
        0,
    ),
)
x_train, y_train = (
    tf.concat(
        (
            x_positives[val_split + test_split : val_split + test_split + train_split],
            x_negatives[val_split + test_split : val_split + test_split + train_split],
        ),
        0,
    ),
    tf.concat(
        (
            y_positives[val_split + test_split : val_split + test_split + train_split],
            y_negatives[val_split + test_split : val_split + test_split + train_split],
        ),
        0,
    ),
)

# Remaining pool of samples are stored separately. These are only labeled as and when required
x_pool_positives, y_pool_positives = (
    x_positives[val_split + test_split + train_split :],
    y_positives[val_split + test_split + train_split :],
)
x_pool_negatives, y_pool_negatives = (
    x_negatives[val_split + test_split + train_split :],
    y_negatives[val_split + test_split + train_split :],
)

# Creating TF Datasets for faster prefetching and parallelization
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

pool_negatives = tf.data.Dataset.from_tensor_slices(
    (x_pool_negatives, y_pool_negatives)
)
pool_positives = tf.data.Dataset.from_tensor_slices(
    (x_pool_positives, y_pool_positives)
)

print(f"Initial training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Testing set size: {len(test_dataset)}")
print(f"Unlabeled negative pool: {len(pool_negatives)}")
print(f"Unlabeled positive pool: {len(pool_positives)}")
```

<div class="k-default-codeblock">
```
Initial training set size: 15000
Validation set size: 5000
Testing set size: 5000
Unlabeled negative pool: 12500
Unlabeled positive pool: 12500

```
</div>
### Fitting the `TextVectorization` layer

Since we are working with text data, we will need to encode the text strings as vectors which
would then be passed through an `Embedding` layer. To make this tokenization process
faster, we use the `map()` function with its parallelization functionality.


```python

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


vectorizer = layers.TextVectorization(
    3000, standardize=custom_standardization, output_sequence_length=150
)
# Adapting the dataset
vectorizer.adapt(
    train_dataset.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).batch(256)
)


def vectorize_text(text, label):
    text = vectorizer(text)
    return text, label


train_dataset = train_dataset.map(
    vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
pool_negatives = pool_negatives.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
pool_positives = pool_positives.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(256).map(
    vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
)
test_dataset = test_dataset.batch(256).map(
    vectorize_text, num_parallel_calls=tf.data.AUTOTUNE
)
```

---
## Creating Helper Functions


```python
# Helper function for merging new history objects with older ones
def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["binary_accuracy"]
    val_accuracy = val_accuracy + history.history["val_binary_accuracy"]
    return losses, val_losses, accuracy, val_accuracy


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

```

---
## Creating the Model

We create a small bidirectional LSTM model. When using Active Learning, you should make sure
that the model architecture is capable of overfitting to the initial data.
Overfitting gives a strong hint that the model will have enough capacity for
future, unseen data.


```python

def create_model():
    model = keras.models.Sequential(
        [
            layers.Input(shape=(150,)),
            layers.Embedding(input_dim=3000, output_dim=128),
            layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
            layers.GlobalMaxPool1D(),
            layers.Dense(20, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.summary()
    return model

```

---
## Training on the entire dataset

To show the effectiveness of Active Learning, we will first train the model on the entire
dataset containing 40,000 labeled samples. This model will be used for comparison later.


```python

def train_full_model(full_train_dataset, val_dataset, test_dataset):
    model = create_model()
    model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.FalseNegatives(),
            keras.metrics.FalsePositives(),
        ],
    )

    # We will save the best model at every epoch and load the best one for evaluation on the test set
    history = model.fit(
        full_train_dataset.batch(256),
        epochs=20,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, verbose=1),
            keras.callbacks.ModelCheckpoint(
                "FullModelCheckpoint.h5", verbose=1, save_best_only=True
            ),
        ],
    )

    # Plot history
    plot_history(
        history.history["loss"],
        history.history["val_loss"],
        history.history["binary_accuracy"],
        history.history["val_binary_accuracy"],
    )

    # Loading the best checkpoint
    model = keras.models.load_model("FullModelCheckpoint.h5")

    print("-" * 100)
    print(
        "Test set evaluation: ",
        model.evaluate(test_dataset, verbose=0, return_dict=True),
    )
    print("-" * 100)
    return model


# Sampling the full train dataset to train on
full_train_dataset = (
    train_dataset.concatenate(pool_positives)
    .concatenate(pool_negatives)
    .cache()
    .shuffle(20000)
)

# Training the full model
full_dataset_model = train_full_model(full_train_dataset, val_dataset, test_dataset)
```

<div class="k-default-codeblock">
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 150, 128)          384000    
                                                                 
 bidirectional (Bidirectiona  (None, 150, 64)          41216     
 l)                                                              
                                                                 
 global_max_pooling1d (Globa  (None, 64)               0         
 lMaxPooling1D)                                                  
                                                                 
 dense (Dense)               (None, 20)                1300      
                                                                 
 dropout (Dropout)           (None, 20)                0         
                                                                 
 dense_1 (Dense)             (None, 1)                 21        
                                                                 
=================================================================
Total params: 426,537
Trainable params: 426,537
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
156/157 [============================>.] - ETA: 0s - loss: 0.5150 - binary_accuracy: 0.7615 - false_negatives: 3314.0000 - false_positives: 6210.0000
Epoch 00001: val_loss improved from inf to 0.47791, saving model to FullModelCheckpoint.h5
157/157 [==============================] - 25s 103ms/step - loss: 0.5148 - binary_accuracy: 0.7617 - false_negatives: 3316.0000 - false_positives: 6217.0000 - val_loss: 0.4779 - val_binary_accuracy: 0.7858 - val_false_negatives: 970.0000 - val_false_positives: 101.0000
Epoch 2/20
156/157 [============================>.] - ETA: 0s - loss: 0.3659 - binary_accuracy: 0.8500 - false_negatives: 2833.0000 - false_positives: 3158.0000
Epoch 00002: val_loss improved from 0.47791 to 0.35345, saving model to FullModelCheckpoint.h5
157/157 [==============================] - 9s 59ms/step - loss: 0.3656 - binary_accuracy: 0.8501 - false_negatives: 2836.0000 - false_positives: 3159.0000 - val_loss: 0.3535 - val_binary_accuracy: 0.8502 - val_false_negatives: 363.0000 - val_false_positives: 386.0000
Epoch 3/20
156/157 [============================>.] - ETA: 0s - loss: 0.3319 - binary_accuracy: 0.8653 - false_negatives: 2507.0000 - false_positives: 2873.0000
Epoch 00003: val_loss improved from 0.35345 to 0.33150, saving model to FullModelCheckpoint.h5
157/157 [==============================] - 9s 55ms/step - loss: 0.3319 - binary_accuracy: 0.8652 - false_negatives: 2512.0000 - false_positives: 2878.0000 - val_loss: 0.3315 - val_binary_accuracy: 0.8576 - val_false_negatives: 423.0000 - val_false_positives: 289.0000
Epoch 4/20
156/157 [============================>.] - ETA: 0s - loss: 0.3130 - binary_accuracy: 0.8764 - false_negatives: 2398.0000 - false_positives: 2538.0000
Epoch 00004: val_loss did not improve from 0.33150
157/157 [==============================] - 9s 55ms/step - loss: 0.3129 - binary_accuracy: 0.8763 - false_negatives: 2404.0000 - false_positives: 2542.0000 - val_loss: 0.3328 - val_binary_accuracy: 0.8586 - val_false_negatives: 263.0000 - val_false_positives: 444.0000
Epoch 5/20
156/157 [============================>.] - ETA: 0s - loss: 0.2918 - binary_accuracy: 0.8867 - false_negatives: 2141.0000 - false_positives: 2385.0000
Epoch 00005: val_loss did not improve from 0.33150
157/157 [==============================] - 9s 55ms/step - loss: 0.2917 - binary_accuracy: 0.8867 - false_negatives: 2143.0000 - false_positives: 2388.0000 - val_loss: 0.3762 - val_binary_accuracy: 0.8468 - val_false_negatives: 476.0000 - val_false_positives: 290.0000
Epoch 6/20
156/157 [============================>.] - ETA: 0s - loss: 0.2819 - binary_accuracy: 0.8901 - false_negatives: 2112.0000 - false_positives: 2277.0000
Epoch 00006: val_loss did not improve from 0.33150
157/157 [==============================] - 9s 55ms/step - loss: 0.2819 - binary_accuracy: 0.8902 - false_negatives: 2112.0000 - false_positives: 2282.0000 - val_loss: 0.4018 - val_binary_accuracy: 0.8312 - val_false_negatives: 694.0000 - val_false_positives: 150.0000
Epoch 7/20
156/157 [============================>.] - ETA: 0s - loss: 0.2650 - binary_accuracy: 0.8992 - false_negatives: 1902.0000 - false_positives: 2122.0000
Epoch 00007: val_loss improved from 0.33150 to 0.32843, saving model to FullModelCheckpoint.h5
157/157 [==============================] - 9s 55ms/step - loss: 0.2649 - binary_accuracy: 0.8992 - false_negatives: 1908.0000 - false_positives: 2123.0000 - val_loss: 0.3284 - val_binary_accuracy: 0.8578 - val_false_negatives: 274.0000 - val_false_positives: 437.0000
Epoch 8/20
157/157 [==============================] - ETA: 0s - loss: 0.2508 - binary_accuracy: 0.9051 - false_negatives: 1821.0000 - false_positives: 1974.0000
Epoch 00008: val_loss did not improve from 0.32843
157/157 [==============================] - 9s 55ms/step - loss: 0.2508 - binary_accuracy: 0.9051 - false_negatives: 1821.0000 - false_positives: 1974.0000 - val_loss: 0.4806 - val_binary_accuracy: 0.8194 - val_false_negatives: 788.0000 - val_false_positives: 115.0000
Epoch 9/20
156/157 [============================>.] - ETA: 0s - loss: 0.2377 - binary_accuracy: 0.9112 - false_negatives: 1771.0000 - false_positives: 1775.0000
Epoch 00009: val_loss did not improve from 0.32843
157/157 [==============================] - 9s 54ms/step - loss: 0.2378 - binary_accuracy: 0.9112 - false_negatives: 1775.0000 - false_positives: 1777.0000 - val_loss: 0.3378 - val_binary_accuracy: 0.8562 - val_false_negatives: 335.0000 - val_false_positives: 384.0000
Epoch 10/20
156/157 [============================>.] - ETA: 0s - loss: 0.2209 - binary_accuracy: 0.9195 - false_negatives: 1591.0000 - false_positives: 1623.0000
Epoch 00010: val_loss did not improve from 0.32843
157/157 [==============================] - 9s 55ms/step - loss: 0.2211 - binary_accuracy: 0.9195 - false_negatives: 1594.0000 - false_positives: 1627.0000 - val_loss: 0.3475 - val_binary_accuracy: 0.8556 - val_false_negatives: 425.0000 - val_false_positives: 297.0000
Epoch 11/20
156/157 [============================>.] - ETA: 0s - loss: 0.2060 - binary_accuracy: 0.9251 - false_negatives: 1512.0000 - false_positives: 1479.0000
Epoch 00011: val_loss did not improve from 0.32843
157/157 [==============================] - 9s 55ms/step - loss: 0.2061 - binary_accuracy: 0.9251 - false_negatives: 1517.0000 - false_positives: 1479.0000 - val_loss: 0.3823 - val_binary_accuracy: 0.8522 - val_false_negatives: 276.0000 - val_false_positives: 463.0000
Epoch 00011: early stopping

```
</div>
![png](/img/examples/nlp/active_learning_review_classification/active_learning_review_classification_15_1.png)



![png](/img/examples/nlp/active_learning_review_classification/active_learning_review_classification_15_2.png)


<div class="k-default-codeblock">
```
----------------------------------------------------------------------------------------------------
Test set evaluation:  {'loss': 0.34183189272880554, 'binary_accuracy': 0.8579999804496765, 'false_negatives': 295.0, 'false_positives': 415.0}
----------------------------------------------------------------------------------------------------

```
</div>
---
## Training via Active Learning

The general process we follow when performing Active Learning is demonstrated below:

![Active Learning](https://i.imgur.com/dmNKusp.png)

The pipeline can be summarized in five parts:

1. Sample and annotate a small, balanced training dataset
2. Train the model on this small subset
3. Evaluate the model on a balanced testing set
4. If the model satisfies the business criteria, deploy it in a real time setting
5. If it doesn't pass the criteria, sample a few more samples according to the ratio of
false positives and negatives, add them to the training set and repeat from step 2 till
the model passes the tests or till all available data is exhausted.

For the code below, we will perform sampling using the following formula:<br/>

![Ratio Sampling](https://i.imgur.com/LyZEiZL.png)

Active Learning techniques use callbacks extensively for progress tracking. We will be
using model checkpointing and early stopping for this example. The `patience` parameter
for Early Stopping can help minimize overfitting and the time required. We have set it
`patience=4` for now but since the model is robust, we can increase the patience level if
desired.

Note: We are not loading the checkpoint after the first training iteration. In my
experience working on Active Learning techniques, this helps the model probe the
newly formed loss landscape. Even if the model fails to improve in the second iteration,
we will still gain insight about the possible future false positive and negative rates.
This will help us sample a better set in the next iteration where the model will have a
greater chance to improve.


```python

def train_active_learning_models(
    train_dataset,
    pool_negatives,
    pool_positives,
    val_dataset,
    test_dataset,
    num_iterations=3,
    sampling_size=5000,
):

    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies = [], [], [], []

    model = create_model()
    # We will monitor the false positives and false negatives predicted by our model
    # These will decide the subsequent sampling ratio for every Active Learning loop
    model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.FalseNegatives(),
            keras.metrics.FalsePositives(),
        ],
    )

    # Defining checkpoints.
    # The checkpoint callback is reused throughout the training since it only saves the best overall model.
    checkpoint = keras.callbacks.ModelCheckpoint(
        "AL_Model.h5", save_best_only=True, verbose=1
    )
    # Here, patience is set to 4. This can be set higher if desired.
    early_stopping = keras.callbacks.EarlyStopping(patience=4, verbose=1)

    print(f"Starting to train with {len(train_dataset)} samples")
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

    for iteration in range(num_iterations):
        # Getting predictions from previously trained model
        predictions = model.predict(test_dataset)

        # Generating labels from the output probabilities
        rounded = tf.where(tf.greater(predictions, 0.5), 1, 0)

        # Evaluating the number of zeros and ones incorrrectly classified
        _, _, false_negatives, false_positives = model.evaluate(test_dataset, verbose=0)

        print("-" * 100)
        print(
            f"Number of zeros incorrectly classified: {false_negatives}, Number of ones incorrectly classified: {false_positives}"
        )

        # This technique of Active Learning demonstrates ratio based sampling where
        # Number of ones/zeros to sample = Number of ones/zeros incorrectly classified / Total incorrectly classified
        if false_negatives != 0 and false_positives != 0:
            total = false_negatives + false_positives
            sample_ratio_ones, sample_ratio_zeros = (
                false_positives / total,
                false_negatives / total,
            )
        # In the case where all samples are correctly predicted, we can sample both classes equally
        else:
            sample_ratio_ones, sample_ratio_zeros = 0.5, 0.5

        print(
            f"Sample ratio for positives: {sample_ratio_ones}, Sample ratio for negatives:{sample_ratio_zeros}"
        )

        # Sample the required number of ones and zeros
        sampled_dataset = pool_negatives.take(
            int(sample_ratio_zeros * sampling_size)
        ).concatenate(pool_positives.take(int(sample_ratio_ones * sampling_size)))

        # Skip the sampled data points to avoid repetition of sample
        pool_negatives = pool_negatives.skip(int(sample_ratio_zeros * sampling_size))
        pool_positives = pool_positives.skip(int(sample_ratio_ones * sampling_size))

        # Concatenating the train_dataset with the sampled_dataset
        train_dataset = train_dataset.concatenate(sampled_dataset).prefetch(
            tf.data.AUTOTUNE
        )

        print(f"Starting training with {len(train_dataset)} samples")
        print("-" * 100)

        # We recompile the model to reset the optimizer states and retrain the model
        model.compile(
            loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=[
                keras.metrics.BinaryAccuracy(),
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
        model = keras.models.load_model("AL_Model.h5")

    # Plotting the overall history and evaluating the final model
    plot_history(losses, val_losses, accuracies, val_accuracies)
    print("-" * 100)
    print(
        "Test set evaluation: ",
        model.evaluate(test_dataset, verbose=0, return_dict=True),
    )
    print("-" * 100)

    return model


active_learning_model = train_active_learning_models(
    train_dataset, pool_negatives, pool_positives, val_dataset, test_dataset
)
```

<div class="k-default-codeblock">
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, 150, 128)          384000    
                                                                 
 bidirectional_1 (Bidirectio  (None, 150, 64)          41216     
 nal)                                                            
                                                                 
 global_max_pooling1d_1 (Glo  (None, 64)               0         
 balMaxPooling1D)                                                
                                                                 
 dense_2 (Dense)             (None, 20)                1300      
                                                                 
 dropout_1 (Dropout)         (None, 20)                0         
                                                                 
 dense_3 (Dense)             (None, 1)                 21        
                                                                 
=================================================================
Total params: 426,537
Trainable params: 426,537
Non-trainable params: 0
_________________________________________________________________
Starting to train with 15000 samples
Epoch 1/20
59/59 [==============================] - ETA: 0s - loss: 0.6235 - binary_accuracy: 0.6679 - false_negatives_1: 3111.0000 - false_positives_1: 1870.0000
Epoch 00001: val_loss improved from inf to 0.43017, saving model to AL_Model.h5
59/59 [==============================] - 13s 87ms/step - loss: 0.6235 - binary_accuracy: 0.6679 - false_negatives_1: 3111.0000 - false_positives_1: 1870.0000 - val_loss: 0.4302 - val_binary_accuracy: 0.8286 - val_false_negatives_1: 513.0000 - val_false_positives_1: 344.0000
Epoch 2/20
58/59 [============================>.] - ETA: 0s - loss: 0.4381 - binary_accuracy: 0.8232 - false_negatives_1: 1412.0000 - false_positives_1: 1213.0000
Epoch 00002: val_loss improved from 0.43017 to 0.40090, saving model to AL_Model.h5
59/59 [==============================] - 4s 64ms/step - loss: 0.4373 - binary_accuracy: 0.8235 - false_negatives_1: 1423.0000 - false_positives_1: 1225.0000 - val_loss: 0.4009 - val_binary_accuracy: 0.8248 - val_false_negatives_1: 674.0000 - val_false_positives_1: 202.0000
Epoch 3/20
58/59 [============================>.] - ETA: 0s - loss: 0.3810 - binary_accuracy: 0.8544 - false_negatives_1: 1115.0000 - false_positives_1: 1047.0000
Epoch 00003: val_loss improved from 0.40090 to 0.36085, saving model to AL_Model.h5
59/59 [==============================] - 4s 61ms/step - loss: 0.3805 - binary_accuracy: 0.8545 - false_negatives_1: 1123.0000 - false_positives_1: 1060.0000 - val_loss: 0.3608 - val_binary_accuracy: 0.8408 - val_false_negatives_1: 231.0000 - val_false_positives_1: 565.0000
Epoch 4/20
58/59 [============================>.] - ETA: 0s - loss: 0.3436 - binary_accuracy: 0.8647 - false_negatives_1: 995.0000 - false_positives_1: 1014.0000
Epoch 00004: val_loss improved from 0.36085 to 0.35469, saving model to AL_Model.h5
59/59 [==============================] - 4s 61ms/step - loss: 0.3428 - binary_accuracy: 0.8654 - false_negatives_1: 999.0000 - false_positives_1: 1020.0000 - val_loss: 0.3547 - val_binary_accuracy: 0.8452 - val_false_negatives_1: 266.0000 - val_false_positives_1: 508.0000
Epoch 5/20
58/59 [============================>.] - ETA: 0s - loss: 0.3166 - binary_accuracy: 0.8834 - false_negatives_1: 835.0000 - false_positives_1: 897.0000
Epoch 00005: val_loss did not improve from 0.35469
59/59 [==============================] - 4s 60ms/step - loss: 0.3163 - binary_accuracy: 0.8835 - false_negatives_1: 839.0000 - false_positives_1: 908.0000 - val_loss: 0.3554 - val_binary_accuracy: 0.8508 - val_false_negatives_1: 382.0000 - val_false_positives_1: 364.0000
Epoch 6/20
58/59 [============================>.] - ETA: 0s - loss: 0.2935 - binary_accuracy: 0.8944 - false_negatives_1: 757.0000 - false_positives_1: 811.0000
Epoch 00006: val_loss did not improve from 0.35469
59/59 [==============================] - 4s 60ms/step - loss: 0.2938 - binary_accuracy: 0.8945 - false_negatives_1: 765.0000 - false_positives_1: 818.0000 - val_loss: 0.3718 - val_binary_accuracy: 0.8458 - val_false_negatives_1: 345.0000 - val_false_positives_1: 426.0000
Epoch 7/20
58/59 [============================>.] - ETA: 0s - loss: 0.2794 - binary_accuracy: 0.9003 - false_negatives_1: 732.0000 - false_positives_1: 748.0000
Epoch 00007: val_loss did not improve from 0.35469
59/59 [==============================] - 3s 59ms/step - loss: 0.2797 - binary_accuracy: 0.9001 - false_negatives_1: 749.0000 - false_positives_1: 749.0000 - val_loss: 0.3825 - val_binary_accuracy: 0.8406 - val_false_negatives_1: 228.0000 - val_false_positives_1: 569.0000
Epoch 8/20
58/59 [============================>.] - ETA: 0s - loss: 0.2526 - binary_accuracy: 0.9147 - false_negatives_1: 620.0000 - false_positives_1: 647.0000
Epoch 00008: val_loss did not improve from 0.35469
59/59 [==============================] - 4s 60ms/step - loss: 0.2561 - binary_accuracy: 0.9134 - false_negatives_1: 620.0000 - false_positives_1: 679.0000 - val_loss: 0.4109 - val_binary_accuracy: 0.8258 - val_false_negatives_1: 622.0000 - val_false_positives_1: 249.0000
Epoch 00008: early stopping
----------------------------------------------------------------------------------------------------
Number of zeros incorrectly classified: 665.0, Number of ones incorrectly classified: 234.0
Sample ratio for positives: 0.26028921023359286, Sample ratio for negatives:0.7397107897664071
Starting training with 19999 samples
----------------------------------------------------------------------------------------------------
Epoch 1/20
78/79 [============================>.] - ETA: 0s - loss: 0.2955 - binary_accuracy: 0.8902 - false_negatives_2: 1091.0000 - false_positives_2: 1101.0000
Epoch 00001: val_loss did not improve from 0.35469
79/79 [==============================] - 15s 83ms/step - loss: 0.2956 - binary_accuracy: 0.8901 - false_negatives_2: 1095.0000 - false_positives_2: 1102.0000 - val_loss: 0.4136 - val_binary_accuracy: 0.8238 - val_false_negatives_2: 156.0000 - val_false_positives_2: 725.0000
Epoch 2/20
78/79 [============================>.] - ETA: 0s - loss: 0.2657 - binary_accuracy: 0.9047 - false_negatives_2: 953.0000 - false_positives_2: 949.0000
Epoch 00002: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 61ms/step - loss: 0.2659 - binary_accuracy: 0.9047 - false_negatives_2: 954.0000 - false_positives_2: 951.0000 - val_loss: 0.4079 - val_binary_accuracy: 0.8386 - val_false_negatives_2: 510.0000 - val_false_positives_2: 297.0000
Epoch 3/20
78/79 [============================>.] - ETA: 0s - loss: 0.2475 - binary_accuracy: 0.9126 - false_negatives_2: 892.0000 - false_positives_2: 854.0000
Epoch 00003: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 58ms/step - loss: 0.2474 - binary_accuracy: 0.9126 - false_negatives_2: 893.0000 - false_positives_2: 855.0000 - val_loss: 0.4207 - val_binary_accuracy: 0.8364 - val_false_negatives_2: 228.0000 - val_false_positives_2: 590.0000
Epoch 4/20
78/79 [============================>.] - ETA: 0s - loss: 0.2319 - binary_accuracy: 0.9193 - false_negatives_2: 805.0000 - false_positives_2: 807.0000
Epoch 00004: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 57ms/step - loss: 0.2319 - binary_accuracy: 0.9192 - false_negatives_2: 807.0000 - false_positives_2: 808.0000 - val_loss: 0.4080 - val_binary_accuracy: 0.8310 - val_false_negatives_2: 264.0000 - val_false_positives_2: 581.0000
Epoch 5/20
78/79 [============================>.] - ETA: 0s - loss: 0.2133 - binary_accuracy: 0.9260 - false_negatives_2: 728.0000 - false_positives_2: 750.0000
Epoch 00005: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 57ms/step - loss: 0.2133 - binary_accuracy: 0.9259 - false_negatives_2: 729.0000 - false_positives_2: 752.0000 - val_loss: 0.4054 - val_binary_accuracy: 0.8394 - val_false_negatives_2: 371.0000 - val_false_positives_2: 432.0000
Epoch 6/20
78/79 [============================>.] - ETA: 0s - loss: 0.1982 - binary_accuracy: 0.9361 - false_negatives_2: 639.0000 - false_positives_2: 636.0000
Epoch 00006: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 57ms/step - loss: 0.1980 - binary_accuracy: 0.9362 - false_negatives_2: 639.0000 - false_positives_2: 636.0000 - val_loss: 0.5185 - val_binary_accuracy: 0.8284 - val_false_negatives_2: 590.0000 - val_false_positives_2: 268.0000
Epoch 7/20
78/79 [============================>.] - ETA: 0s - loss: 0.1887 - binary_accuracy: 0.9409 - false_negatives_2: 606.0000 - false_positives_2: 575.0000
Epoch 00007: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 57ms/step - loss: 0.1886 - binary_accuracy: 0.9408 - false_negatives_2: 606.0000 - false_positives_2: 577.0000 - val_loss: 0.6881 - val_binary_accuracy: 0.7886 - val_false_negatives_2: 893.0000 - val_false_positives_2: 164.0000
Epoch 8/20
78/79 [============================>.] - ETA: 0s - loss: 0.1778 - binary_accuracy: 0.9443 - false_negatives_2: 575.0000 - false_positives_2: 538.0000
Epoch 00008: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 57ms/step - loss: 0.1776 - binary_accuracy: 0.9443 - false_negatives_2: 575.0000 - false_positives_2: 538.0000 - val_loss: 0.5921 - val_binary_accuracy: 0.8244 - val_false_negatives_2: 634.0000 - val_false_positives_2: 244.0000
Epoch 9/20
78/79 [============================>.] - ETA: 0s - loss: 0.1598 - binary_accuracy: 0.9505 - false_negatives_2: 507.0000 - false_positives_2: 481.0000
Epoch 00009: val_loss did not improve from 0.35469
79/79 [==============================] - 5s 57ms/step - loss: 0.1597 - binary_accuracy: 0.9506 - false_negatives_2: 507.0000 - false_positives_2: 481.0000 - val_loss: 0.5393 - val_binary_accuracy: 0.8214 - val_false_negatives_2: 542.0000 - val_false_positives_2: 351.0000
Epoch 00009: early stopping
----------------------------------------------------------------------------------------------------
Number of zeros incorrectly classified: 270.0, Number of ones incorrectly classified: 498.0
Sample ratio for positives: 0.6484375, Sample ratio for negatives:0.3515625
Starting training with 24998 samples
----------------------------------------------------------------------------------------------------
Epoch 1/20
97/98 [============================>.] - ETA: 0s - loss: 0.3554 - binary_accuracy: 0.8609 - false_negatives_3: 1714.0000 - false_positives_3: 1739.0000
Epoch 00001: val_loss improved from 0.35469 to 0.34182, saving model to AL_Model.h5
98/98 [==============================] - 17s 82ms/step - loss: 0.3548 - binary_accuracy: 0.8613 - false_negatives_3: 1720.0000 - false_positives_3: 1748.0000 - val_loss: 0.3418 - val_binary_accuracy: 0.8528 - val_false_negatives_3: 369.0000 - val_false_positives_3: 367.0000
Epoch 2/20
97/98 [============================>.] - ETA: 0s - loss: 0.3176 - binary_accuracy: 0.8785 - false_negatives_3: 1473.0000 - false_positives_3: 1544.0000
Epoch 00002: val_loss did not improve from 0.34182
98/98 [==============================] - 6s 56ms/step - loss: 0.3179 - binary_accuracy: 0.8784 - false_negatives_3: 1479.0000 - false_positives_3: 1560.0000 - val_loss: 0.4785 - val_binary_accuracy: 0.8102 - val_false_negatives_3: 793.0000 - val_false_positives_3: 156.0000
Epoch 3/20
97/98 [============================>.] - ETA: 0s - loss: 0.2986 - binary_accuracy: 0.8893 - false_negatives_3: 1353.0000 - false_positives_3: 1396.0000
Epoch 00003: val_loss did not improve from 0.34182
98/98 [==============================] - 5s 56ms/step - loss: 0.2985 - binary_accuracy: 0.8893 - false_negatives_3: 1366.0000 - false_positives_3: 1402.0000 - val_loss: 0.3473 - val_binary_accuracy: 0.8542 - val_false_negatives_3: 340.0000 - val_false_positives_3: 389.0000
Epoch 4/20
97/98 [============================>.] - ETA: 0s - loss: 0.2822 - binary_accuracy: 0.8970 - false_negatives_3: 1253.0000 - false_positives_3: 1305.0000
Epoch 00004: val_loss did not improve from 0.34182
98/98 [==============================] - 6s 56ms/step - loss: 0.2820 - binary_accuracy: 0.8971 - false_negatives_3: 1257.0000 - false_positives_3: 1316.0000 - val_loss: 0.3849 - val_binary_accuracy: 0.8386 - val_false_negatives_3: 537.0000 - val_false_positives_3: 270.0000
Epoch 5/20
97/98 [============================>.] - ETA: 0s - loss: 0.2666 - binary_accuracy: 0.9047 - false_negatives_3: 1130.0000 - false_positives_3: 1237.0000
Epoch 00005: val_loss did not improve from 0.34182
98/98 [==============================] - 6s 56ms/step - loss: 0.2666 - binary_accuracy: 0.9048 - false_negatives_3: 1142.0000 - false_positives_3: 1238.0000 - val_loss: 0.3731 - val_binary_accuracy: 0.8444 - val_false_negatives_3: 251.0000 - val_false_positives_3: 527.0000
Epoch 00005: early stopping
----------------------------------------------------------------------------------------------------
Number of zeros incorrectly classified: 392.0, Number of ones incorrectly classified: 356.0
Sample ratio for positives: 0.47593582887700536, Sample ratio for negatives:0.5240641711229946
Starting training with 29997 samples
----------------------------------------------------------------------------------------------------
Epoch 1/20
117/118 [============================>.] - ETA: 0s - loss: 0.3345 - binary_accuracy: 0.8720 - false_negatives_4: 1835.0000 - false_positives_4: 1998.0000
Epoch 00001: val_loss did not improve from 0.34182
118/118 [==============================] - 20s 96ms/step - loss: 0.3343 - binary_accuracy: 0.8722 - false_negatives_4: 1835.0000 - false_positives_4: 1999.0000 - val_loss: 0.3478 - val_binary_accuracy: 0.8488 - val_false_negatives_4: 250.0000 - val_false_positives_4: 506.0000
Epoch 2/20
117/118 [============================>.] - ETA: 0s - loss: 0.3061 - binary_accuracy: 0.8842 - false_negatives_4: 1667.0000 - false_positives_4: 1801.0000
Epoch 00002: val_loss improved from 0.34182 to 0.33779, saving model to AL_Model.h5
118/118 [==============================] - 7s 56ms/step - loss: 0.3059 - binary_accuracy: 0.8843 - false_negatives_4: 1670.0000 - false_positives_4: 1802.0000 - val_loss: 0.3378 - val_binary_accuracy: 0.8534 - val_false_negatives_4: 335.0000 - val_false_positives_4: 398.0000
Epoch 3/20
117/118 [============================>.] - ETA: 0s - loss: 0.2923 - binary_accuracy: 0.8921 - false_negatives_4: 1626.0000 - false_positives_4: 1607.0000
Epoch 00003: val_loss did not improve from 0.33779
118/118 [==============================] - 7s 56ms/step - loss: 0.2923 - binary_accuracy: 0.8921 - false_negatives_4: 1626.0000 - false_positives_4: 1611.0000 - val_loss: 0.3413 - val_binary_accuracy: 0.8486 - val_false_negatives_4: 269.0000 - val_false_positives_4: 488.0000
Epoch 4/20
117/118 [============================>.] - ETA: 0s - loss: 0.2746 - binary_accuracy: 0.8997 - false_negatives_4: 1459.0000 - false_positives_4: 1546.0000
Epoch 00004: val_loss did not improve from 0.33779
118/118 [==============================] - 7s 55ms/step - loss: 0.2746 - binary_accuracy: 0.8996 - false_negatives_4: 1465.0000 - false_positives_4: 1546.0000 - val_loss: 0.3810 - val_binary_accuracy: 0.8326 - val_false_negatives_4: 169.0000 - val_false_positives_4: 668.0000
Epoch 5/20
117/118 [============================>.] - ETA: 0s - loss: 0.2598 - binary_accuracy: 0.9066 - false_negatives_4: 1336.0000 - false_positives_4: 1462.0000
Epoch 00005: val_loss did not improve from 0.33779
118/118 [==============================] - 7s 56ms/step - loss: 0.2597 - binary_accuracy: 0.9066 - false_negatives_4: 1337.0000 - false_positives_4: 1465.0000 - val_loss: 0.4038 - val_binary_accuracy: 0.8332 - val_false_negatives_4: 643.0000 - val_false_positives_4: 191.0000
Epoch 6/20
117/118 [============================>.] - ETA: 0s - loss: 0.2461 - binary_accuracy: 0.9132 - false_negatives_4: 1263.0000 - false_positives_4: 1337.0000
Epoch 00006: val_loss did not improve from 0.33779
118/118 [==============================] - 7s 55ms/step - loss: 0.2462 - binary_accuracy: 0.9132 - false_negatives_4: 1263.0000 - false_positives_4: 1341.0000 - val_loss: 0.3546 - val_binary_accuracy: 0.8500 - val_false_negatives_4: 359.0000 - val_false_positives_4: 391.0000
Epoch 00006: early stopping

```
</div>
![png](/img/examples/nlp/active_learning_review_classification/active_learning_review_classification_17_1.png)



![png](/img/examples/nlp/active_learning_review_classification/active_learning_review_classification_17_2.png)


<div class="k-default-codeblock">
```
----------------------------------------------------------------------------------------------------
Test set evaluation:  {'loss': 0.34248775243759155, 'binary_accuracy': 0.854200005531311, 'false_negatives_4': 348.0, 'false_positives_4': 381.0}
----------------------------------------------------------------------------------------------------

```
</div>
---
## Conclusion

Active Learning is a growing area of research. This example demonstrates the cost-efficiency
benefits of using Active Learning, as it eliminates the need to annotate large amounts of
data, saving resources.

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

1. [Active Learning Literature Survey](http://burrsettles.com/pub/settles.activelearning.pdf) (Burr Settles, 2010).
2. [modAL](https://github.com/modAL-python/modAL): A Modular Active Learning framework.
3. Google's unofficial [Active Learning playground](https://github.com/google/active-learning).
