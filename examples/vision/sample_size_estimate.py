"""
Title: Estimating required sample size for model training
Author: [JacoVerster](https://twitter.com/JacoVerster)
Date created: 2021/05/20
Last modified: 2021/05/21
Description: Estimate the number of samples required to reach a desired model accuracy.
"""

"""
# Introduction
In most real-world use cases, the image data required to train a deep learning model is
limited, this is especially true in the medical imaging domain where dataset creation is
costly. The first question that usually comes up is "how many images will we need to
train a good enough machine learning model".

Statistical approaches can be used to try and calculate the size of an optimum training
set, but this is difficult. In many cases a small set of data is available and then it is
useful to consider the “behaviour” of the model when subjected to varying training sample
sizes. A model-based sample size determination method can be used to estimate the optimum
number of images needed to arrive at a scientifically valid sample size that would give
the required model performance.

A systematic review of [Sample-Size Determination Methodologies](https://bit.ly/3f53LSs)
by Balki et al. provides examples of several sample-size determination methods. In this
example, a balanced subsampling scheme is used to determine the optimum sample size for
our model. This is done by selecting a random subsample consisting of Y number of images
and training the model using the subsample. The model is then evaluated on an independent
test set. This process is repeated N times for each subsample with replacement to allow
for the construction of a mean and confidence interval for the observed performance.

This example requires TensorFlow 2.4 or higher.
"""

"""
# Setup
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Define seed and fixed variables
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
AUTO = tf.data.AUTOTUNE

"""
# Load TensorFlow dataset

We'll be using the [TF Flowers dataset](https://bit.ly/34FQxWc).
"""

# Specify dataset parameters
dataset_name = "tf_flowers"
batch_size = 64
image_size = (224, 224)

# Load data from tfds and split 10% off for a test set
(train_data, test_data), ds_info = tfds.load(
    dataset_name,
    split=["train[:90%]", "train[90%:]"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Extract number of classes and list of class names
num_train_samples = train_data.cardinality().numpy()
num_classes = ds_info.features["label"].num_classes
class_names = ds_info.features["label"].names

print(f"Number of training samples: {num_train_samples}")
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

"""
# Prepare test set for training
"""

test_ds = (
    test_data.map(
        lambda img, lab: (tf.image.resize(img, image_size), lab),
        num_parallel_calls=AUTO,
    )  # Resize images
    .map(
        lambda img, lab: (img, tf.one_hot(lab, num_classes)), num_parallel_calls=AUTO
    )  # Convert to one-hot labels
    .cache()
    .batch(batch_size)
    .prefetch(AUTO)
)

# Extract true labels from the test set (to be used for model evaluation later)
y_true = np.empty((0, num_classes))
for _, label in test_ds.unbatch().as_numpy_iterator():
    y_true = np.vstack((y_true, label))
y_true_indices = np.argmax(y_true, axis=1)

# Count number of test samples
num_test_samples = len(y_true_indices)
print("Test set length:", num_test_samples)

# Plot images from the test set
image_batch, label_batch = next(iter(test_ds))

plt.figure(figsize=(16, 12))
for n in range(30):
    ax = plt.subplot(5, 6, n + 1)
    plt.imshow(image_batch[n].numpy().astype("int32"))
    plt.title(np.array(class_names)[label_batch[n].numpy() == True][0])
    plt.axis("off")

"""
# Define model building & train functions

We create a few convenience functions to build a transfer-learning model, compile and
train it and unfreeze layers for fine-tuning.
"""


def build_model(num_classes, img_size=image_size[0], top_dropout=0.3):
    """Creates a Keras MobileNetV2 model without the top layer using imagenet
        weights, adding new custom top layers.

    Arguments:
        num_classes: Int, number of classese to use in the softmax layer.
        img_size: Int, square size of input images (defaults is 224).
        top_dropout: Int, value for dropout layer (defaults is 0.3).

    Returns:
        Uncompiled Keras model.
    """

    # Create input and pre-processing layers for MobileNetV2
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = preprocessing.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    model = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_tensor=x
    )

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.Dropout(top_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    print("Trainable weights:", len(model.trainable_weights))
    print("Non_trainable weights:", len(model.non_trainable_weights))
    return model


def compile_and_train(
    model,
    train_ds,
    val_ds,
    metrics=[keras.metrics.AUC(name="auc"), "acc"],
    optimizer=keras.optimizers.Adam(),
    patience=5,
    epochs=5,
):
    """Compiles and trains a Keras model using EarlyStopping callback on
        'val_auc' (requires at minimum 'auc' as a metric. Compiles using
        categorical_crossentropy loss with optimizer and metrics of choice.

    Arguments:
        model: Uncompiled Keras model.
        train_ds: tf.data.Dataset, trainig dataset.
        val_ds: tf.data.Dataset, validation dataset.
        class_weights: Dict, weights per class.
        metrics: Keras/TF metrics, requires at least 'auc' metric(defaults is
                [keras.metrics.AUC(name='auc'), 'acc']).
        optimizer: Keras/TF optimizer (defaults is keras.optimizers.Adam()).
        patience: Int, epochsfor EarlyStopping patience (defaults is 5).
        epochs: Int, number of epochs to train (default is 5).

    Returns:
        Training history for trained Keras model.
    """

    stopper = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        min_delta=0,
        patience=patience,
        verbose=1,
        restore_best_weights=True,
    )

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    history = model.fit(
        train_ds, epochs=epochs, validation_data=val_ds, callbacks=[stopper]
    )
    return history


def unfreeze(model, block_name, verbose=0):
    """Unfreezes Keras model layers.

    Arguments:
        model: Keras model.
        block_name: Str, layer name for example block_name = 'block4'.
                    Checks if supplied string is in the layer name.
        verbose: Int, 0 means silent, 1 prints out layers trainability status.

    Returns:
        Keras model with all layers after (and including) the specified
        block_name to trainable, excluding BatchNormalization layers.
    """

    # Set the whole model trainable and
    model.trainable = True
    set_trainable = False

    for layer in model.layers:
        if block_name in layer.name:
            set_trainable = True
        if set_trainable and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            if verbose == 1:
                print(layer.name, "trainable")
        else:
            layer.trainable = False
            if verbose == 1:
                print(layer.name, "NOT trainable")
    print("Trainable weights:", len(model.trainable_weights))
    print("Non-trainable weights:", len(model.non_trainable_weights))
    return model


"""
# Define augmentation function

Define image augmentation using keras preprocessing layers.
"""

image_augmentation = keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(height_factor=(-0.1, -0), fill_mode="constant"),
        preprocessing.RandomContrast(factor=0.1),
    ],
)

"""
# Define iterative subsampling functions

To iteratively train a model over several subsample sets we need to create some functions
for splitting the dataset and training the model.

"""


def create_subsets(dataset, fraction):
    """Creates a subset from a dataset based on a specified fraction. It is
        divided into training (90%) and validation (10%) sets.

    Arguments:
        dataset: tf.data.Dataset.
        fraction: Int, number between 0 and 1.

    Returns:
        Training and validations datasets with number of subsamples.
    """

    ds_subset = dataset.take(int(num_train_samples * fraction))

    # Calculate subset length
    num_subsamples = ds_subset.cardinality().numpy()
    print("Subset length:", num_subsamples)

    # Split off 90% for training and 10% for validation
    train_val_split = int(num_subsamples * 0.9)
    train_ds = ds_subset.take(train_val_split)
    print("Number of training samples:", train_ds.cardinality().numpy())

    val_ds = ds_subset.skip(train_val_split)
    print("Number of validation samples:", val_ds.cardinality().numpy())

    # Prepare the training set for training, apply random augmentations
    train_ds = (
        train_ds.map(
            lambda img, lab: (tf.image.resize(img, image_size), lab),
            num_parallel_calls=AUTO,
        )
        .map(
            lambda img, lab: (img, tf.one_hot(lab, num_classes)),
            num_parallel_calls=AUTO,
        )
        .batch(batch_size)  # First batch images before augmenting
        .map(
            lambda img, lab: (image_augmentation(img), lab), num_parallel_calls=AUTO,
        )  # Augment images
        .cache()
        .prefetch(AUTO)
    )

    # Prepare validation set for training, only resize and one-hot encode
    val_ds = (
        val_ds.map(
            lambda img, lab: (tf.image.resize(img, image_size), lab),
            num_parallel_calls=AUTO,
        )
        .map(
            lambda img, lab: (img, tf.one_hot(lab, num_classes)),
            num_parallel_calls=AUTO,
        )
        .cache()
        .batch(batch_size)
        .prefetch(AUTO)
    )
    return train_ds, val_ds, num_subsamples


def train_model(train_ds, val_ds, test_ds):
    """Builds a model, trains only the top layers for 10 epochs. Unfreezes
        deeper layers train for 20 more epochs. Calculates model accuracy.

    Arguments:
        train_ds: tf.data.Dataset, trainig dataset.
        val_ds: tf.data.Dataset, validation dataset.
        test_ds: tf.data.Dataset, test dataset.

    Returns:
        Model accuracy.
    """

    model = build_model(num_classes)

    # Compile and train top layers
    history = compile_and_train(
        model,
        train_ds,
        val_ds,
        metrics=[keras.metrics.AUC(name="auc"), "acc"],
        optimizer=keras.optimizers.Adam(),
        patience=3,
        epochs=10,
    )

    # Unfreeze model from block 6 onwards
    model = unfreeze(model, "block_10")

    # Compile and train for 20 epochs with a lower learning rate
    fine_tune_epochs = 20
    total_epochs = history.epoch[-1] + fine_tune_epochs

    history_fine = compile_and_train(
        model,
        train_ds,
        val_ds,
        metrics=[keras.metrics.AUC(name="auc"), "acc"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        patience=5,
        epochs=total_epochs,
    )

    # Calculate model accuracy on the test set
    y_pred = model.predict(test_ds)
    y_pred_indices = np.argmax(y_pred, axis=1)
    acc = np.sum(np.equal(y_pred_indices, y_true_indices)) / num_test_samples
    return acc


"""
Let's create a small subset and plot a few images from the training set. These should now
be augmented.
"""

# Test image augmentations on a subset (10% of training data)
train_ds, val_ds, num_samples = create_subsets(train_data, 0.1)

# Plot images from the training set
image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(16, 12))
for n in range(30):
    ax = plt.subplot(5, 6, n + 1)
    plt.imshow(image_batch[n].numpy().astype("int32"))
    plt.title(np.array(class_names)[label_batch[n].numpy() == True][0])
    plt.axis("off")

"""
# Train models iteratively

Now that we have model building functions and supporting iterative functions we can train
the model over several subsample splits.

- We select the subsample splits as 5%, 10%, 25% and 50% of the downloaded dataset. We
pretend that only 50% of the actual data is available at present.
- We train the model 5 times from scratch at each split and record the accuracy values.

Note that this trains 20 models and will take some time. Make sure you have a GPU runtime
active.

To keep this example lightweight, sample data from a previous training run is provided.
"""


def train_iteratively(
    train_data, sample_splits=[0.05, 0.1, 0.25, 0.5], iter_per_split=5
):
    # Train all the sample models and calculate accuracy
    overall_accuracy = []
    sample_sizes = []

    for fraction in sample_splits:
        print(f"Fraqction split: {fraction}")
        # Repeat training 3 times for each sample size
        sample_accuracy = []
        for i in range(iter_per_split):
            print(f"Run {i+1} out of {iter_per_split}:")
            # Create fraction subsets, train model and calculate accuracy
            # train_ds, val_ds, sub_len, class_len = create_subsets(train_df, fraction)
            train_ds, val_ds, num_samples = create_subsets(train_data, fraction)
            accuracy = train_model(train_ds, val_ds, test_ds)
            print(f"Accuracy: {accuracy}")
            sample_accuracy.append(accuracy)
        overall_accuracy.append(sample_accuracy)
        sample_sizes.append(num_samples)
    return overall_accuracy, sample_sizes


# Running the above function produces the following outputs
overall_accuracy = [
    [0.82016348773, 0.74659400544, 0.80108991825, 0.84468664850, 0.82288828337],
    [0.86103542234, 0.87738419618, 0.85013623978, 0.89373297002, 0.89100817438],
    [0.89100817438, 0.92370572207, 0.88555858310, 0.91008174386, 0.89100817438],
    [0.89373297002, 0.93732970027, 0.91280653950, 0.87193460490, 0.91280653950],
]

sample_sizes = [165, 330, 825, 1651]

"""
# Learning curve

We now plot the learning curve by fitting an exponential curve through the mean accuracy
points. We use TF to fit an exponential function through the data.

We then extrapolate the learning curve to the predict the accuracy of a model trained on
the whole training set.
"""

# The x-values, mean accuracy and errors can be extracted
x = sample_sizes
mean_acc = [np.mean(i) for i in overall_accuracy]
error = [np.std(i) for i in overall_accuracy]

# Define the exponential and cost functions


def exp_func(x, a, b):
    return a * x ** b


def squared_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# Define variables, learning rate and number of epochs for fitting with TF
a = tf.Variable(0.6)  # Use smart guesses for the weights to speed up fitting
b = tf.Variable(0.05)
learning_rate = 0.01
training_epochs = 2000

# Fit the exponential function to the data
for epoch in range(training_epochs):
    with tf.GradientTape() as g:
        y_pred = exp_func(x, a, b)
        cost_function = squared_error(y_pred, mean_acc)
    # Get gradients and compute adjusted weights
    gradients = g.gradient(cost_function, [a, b])
    a.assign_sub(gradients[0] * learning_rate)
    b.assign_sub(gradients[1] * learning_rate)
print(f"Curve fit weights: a = {a.numpy()} and b = {b.numpy()}.")

# We can now estimate the accuracy for the whole training set using exp_func
max_acc = exp_func(num_train_samples, a, b)

# Print predicted x value and append to plot values
print(f"A model accuracy of {max_acc} is predicted for {num_train_samples} samples.")
x_cont = np.linspace(x[0], num_train_samples, 100)

# Build the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
ax.plot(x_cont, exp_func(x_cont, a, b), "r-", label="Fitted exponential curve.")
ax.set_ylabel("Model clasification accuracy.", fontsize=12)
ax.set_xlabel("Training sample size.", fontsize=12)
ax.set_xticks(np.append(x, num_train_samples))
ax.set_yticks(np.append(mean_acc, max_acc))
ax.set_xticklabels(list(np.append(x, num_train_samples)), rotation=90, fontsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
ax.legend(loc=(0.75, 0.75), fontsize=10)
ax.xaxis.grid(True)
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

"""
The mean absolute error (MAE) can be calculated for curve fit to see how well it fits
the data. The lower the error the better the fit.
"""

mae = np.mean([np.abs(mean_acc[i] - exp_func(x[i], a, b)) for i in range(len(x))])
print(f"The mean absolute error for the curve fit is {np.round(mae, 4)}.")

"""
From the extrapolated curve we can see that 3303 images will yield approximately an
accuracy of about 95%.

Now, let's use all the data (3303 images) and train the model to see if our prediction
was accurate!
"""

# Now train the model with full dataset to get the actual accuracy
train_ds, val_ds, num_samples = create_subsets(train_data, 1)
accuracy = train_model(train_ds, val_ds, test_ds)
print(f"A model accuracy of {accuracy} is reached on {num_samples} images!")

"""
# Conclusion

We see that a model accuracy of about 94-96%* is reached using 3303 images. This is quite
close to our estimate!

Even though we used only 50% of the dataset (1651 images) we were able to model the training
behaviour of our model and predict the model accuracy for a given amount of images. This same
methodology can be used to predict the amount of images needed to reach a desired accuracy.
This is very useful when a smaller set of data is available, and it has been shown that
convergence on a deep learning model is possible, but more images are needed. The image count
prediction can be used to plan and budget for further image collection initiatives.

*Note: repeat results may vary due to randomness.
"""
