"""
Title: Model-based sample size determination
Author: [JacoVerster](https://twitter.com/JacoVerster)
Date created: 2021/05/20
Last modified: 2021/05/21
Description: Estimate the number of images required to reach a desired model accuracy.
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

A systematic review of [Sample-Size Determination Methodologies for Machine Learning in
Medical Imaging Research](https://bit.ly/3f53LSs) by Balki et al. provides examples of
several sample-size determination methods. In this example, a
balanced subsampling scheme is used to determine the optimum sample size for our model.
This is done by selecting a random subsample consisting of Y number of images and
training the model using the subsample. The model is then evaluated on an independent
test set. This process is repeated N times for each subsample with replacement to allow
for the construction of a mean and confidence interval for the observed performance.

This example requires TensorFlow 2.4 or higher and the latest version of
[imgaug](https://imgaug.readthedocs.io/).
"""

"""
# Setup
"""

"""shell
!pip install imgaug==0.4.0
"""


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers
print("Tensorflow version", tf.__version__)
print("Keras version", keras.__version__)

# Define fixed variables and seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
AUTO = tf.data.AUTOTUNE

"""
# Load TensorFlow dataset

We'll be using the [TensorFlow Flowers
dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers).
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
nr_train_samples = train_data.cardinality().numpy()
num_classes = ds_info.features["label"].num_classes
class_names = ds_info.features["label"].names

print(f"Nr. of training samples: {nr_train_samples}")
print(f"Nr. of classes: {num_classes}")
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


def build_model(model_name, num_classes, img_size=image_size[0], top_dropout=0.3):
    """
    Creates a Keras application model without the top layer using imagenet
    weights, adding the following custom layers:
      - GlobalAveragePooling2D(name="avg_pool")
      - Dropout(top_dropout), defaults to 0.3
      - Dense(num_classes, activation="softmax", name="serve_out")

    Arguments
    ---------
    model_name:     "VGG16", "ResNet50", "EfficientNetB0", etc. as per Keras
                    applications: https://keras.io/api/applications/
    num_classes:    Number of classese to use in the softmax layer.
    img_size    =   Square size of input images, defaults to 224x224.
    top_dropout =   Sets value for dropout layer, defaults to 0.3.

    """

    # Define function to resize and pre-process images inside the model
    def resize_and_preprocess(x):
        x = tf.image.resize(x, [img_size, img_size])

        # Select preprocessing mode based on the model specified
        # (includes only a few of the Keras applications)
        caffe_modes = ["VGG", "ResNet5", "ResNet1"]
        tf_modes = ["MobileNet", "Xception", "NASNet", "Inception"]
        if any(x in model_name for x in caffe_modes):
            mode = "caffe"
        elif any(x in model_name for x in tf_modes):
            mode = "tf"
        else:
            mode = None

        # Apply preprocessing (if applicable). EfficientNet has no preprocessing.
        if mode == "tf":
            # Divide by max/2 pixel value
            x = tf.divide(x, [127.5, 127.5, 127.5])
            x = tf.subtract(x, [1, 1, 1])  # Minus one to normalise to [-1, 1]
        elif mode == "caffe":
            x = tf.reverse(x, axis=[-1])  # Swop channels RGB -> BGR
            # Subtract mean value
            x = tf.subtract(x, [103.939, 116.779, 123.68])
        return x

    # Create input and pre-processing layers and import pre-trained model
    inputs = layers.Input(shape=(img_size, img_size, 3), name="serve_in")
    x = layers.Lambda(resize_and_preprocess)(inputs)
    model = getattr(tf.keras.applications, model_name)(
        include_top=False, weights="imagenet", input_tensor=x
    )

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.Dropout(top_dropout)(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="serve_out")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)

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
    """
    Compiles and trains a Keras model using EarlyStopping callback that monitors
    'val_auc' (so requires at minimum 'auc' as a metric. Compiles using
    categorical_crossentropy loss with optimizer and metrics of choice.

    Arguments
    ---------
    model:          Uncompiled Keras model class.
    train_ds:       Trainig dataset in tf.data format.
    val_ds:         Validation dataset in tf.data format.
    class_weights:  Dictionary of weights per class.
    metrics:        Keras/TF metrics, requires at least 'auc' metric.
                    Defaults to [keras.metrics.AUC(name='auc'), 'acc'].
    optimizer:      Keras/TF optimizer, defaults to keras.optimizers.Adam().
    patience:       EarlyStopping patience, defaults to 5 epochs.
    epochs:         Number of epochs to train, defaults to 5.

    """

    stopper = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        min_delta=0,
        patience=patience,
        verbose=1,
        restore_best_weights=True,
    )

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=metrics)

    history = model.fit(
        train_ds, epochs=epochs, validation_data=val_ds, callbacks=[stopper]
    )
    return history


def unfreeze(model, block_name, verbose=0):
    """
    Unfreezes Keras model layers. Sets all layers, except BatchNormalization,
    after (and including) the specified block_name to trainable.

    Arguments
    ---------
    model:        Keras model.
    block_name:   String with layer name, for example block_name = 'block4'.
                  Checks if supplied string is in the layer name.
    verbose:      Int: 0 means silent, 1 prints out layers trainability status.

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

We'll use the RandAugment augmentation as shown in a recent Keras code example:
[RandAugment for Image Classification for Improved
Robustness](https://https://keras.io/examples/vision/randaugment/).
"""

rand_aug = iaa.RandAugment(n=3, m=7)


def rand_augment(images):
    """
    Randomly augments a batch of image tensors using imgaug library.

    Arguments
    ---------
    images:       TensorFlow tensors to augment, must be batched

    """
    # Convert back to uint8 and convert tensor to numpy values
    images = tf.cast(images, tf.uint8)
    images = rand_aug(images=images.numpy())

    return images


"""
# Define iterative subsampling functions

To iteratively train a model over several subsample sets we need to create some functions
for splitting the dataset and training the model.

"""


def create_subsets(dataset, percentage):
    """
    Creates a subset from a dataset based on a specified percentage. This subset
    is then divided into training (90%) and validation (10%) sets. Output contains
    the training set, validation set and number of subsamples.

    Arguments
    ---------
    dataset:      Dataset in tf.data format.
    percentage:   Number between 0 and 100, excluding limits (0 > x > 100).

    """

    ds_subset = dataset.take(int(nr_train_samples * percentage))

    # Calculate subset length
    nr_of_subsamples = ds_subset.cardinality().numpy()
    print("Subset length:", nr_of_subsamples)

    # Split off 90% for training and 10% for validation
    train_val_split = int(nr_of_subsamples * 0.9)
    train_ds = ds_subset.take(train_val_split)
    print("Nr of training samples:", train_ds.cardinality().numpy())

    val_ds = ds_subset.skip(train_val_split)
    print("Nr of validation samples:", val_ds.cardinality().numpy())

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
            lambda img, lab: (
                tf.py_function(rand_augment, [img], [tf.float32])[0],
                lab,
            ),
            num_parallel_calls=AUTO,
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
    return train_ds, val_ds, nr_of_subsamples


def train_model(train_ds, val_ds, test_ds):
    """
    1) Builds a VGG19 model.
    2) Trains only the top layers for 10 epochs.
    3) Unfreezes from block 4 onwards.
    4) Trains the deeper layers for 20 more epochs.
    """

    model = build_model("VGG19", num_classes)

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

    # Unfreeze model from block 4 onwards
    model = unfreeze(model, "block4")

    # Compile and train for 20 epochs with a lower learning rate
    fine_tune_epochs = 20
    total_epochs = history.epoch[-1] + fine_tune_epochs

    history_fine = compile_and_train(
        model,
        train_ds,
        val_ds,
        metrics=[keras.metrics.AUC(name="auc"), "acc"],
        optimizer=keras.optimizers.Adam(lr=1e-4),
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
be augmented using RandAug.
"""

# Test image augmentations on a subset (5% of training data)
train_ds, val_ds, nr_of_samples = create_subsets(train_data, 0.1)

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

To keep this example lightweight, sample data from a previous training run is provided
after the commented out code below.
"""

# # Train all the sample models and calculate accuracy
# sample_splits = [0.05, 0.1, 0.25, 0.5]
# iter_per_split = 5
# overall_accuracy = []
# sample_sizes = []

# for percentage in sample_splits:
#     print(f"Percentage split: {percentage}")
#     # Repeat training 3 times for each sample size
#     sample_accuracy = []
#     for i in range(iter_per_split):
#         print(f"Run {i+1} out of {iter_per_split}:")
#         # Create percentage subsets, train model and calculate accuracy
#         # train_ds, val_ds, sub_len, class_len = create_subsets(train_df, percentage)
#         train_ds, val_ds, nr_of_samples = create_subsets(
#             train_data, percentage)
#         accuracy = train_model(train_ds, val_ds, test_ds)
#         print(f"Accuracy: {accuracy}")
#         sample_accuracy.append(accuracy)

#     overall_accuracy.append(sample_accuracy)
#     sample_sizes.append(nr_of_samples)
#     # smallest_class_sizes.append(class_len)

#     # Clear session and reset random seeds
#     tf.keras.backend.clear_session()
#     tf.random.set_seed(seed)
#     np.random.seed(seed)


# Sample data
overall_accuracy = [
    [0.71662125, 0.77111717, 0.73024523, 0.75476839, 0.47411444],
    [0.80926431, 0.67302452, 0.66757493, 0.74386921, 0.79291553],
    [0.82288828, 0.82561308, 0.85286104, 0.83378747, 0.8119891],
    [0.89645777, 0.90463215, 0.84468665, 0.88283379, 0.88555858],
]

sample_sizes = [165, 330, 825, 1651]

"""
# Learning curve

We now plot the learning curve by fitting an exponential curve through the mean accuracy
points.

We then extrapolate the learning curve to the desired model accuracy of 95% to estimate
the number of images required for the training set that would produce this accuracy.
"""

# The x-values, mean accuracy and errors can be extracted
x_pos = sample_sizes
mean_acc = [np.mean(i) for i in overall_accuracy]
error = [np.std(i) for i in overall_accuracy]

# Define an exponential function


def func(x, a, b):
    """Return values from an exponential function."""
    return a * x ** b


# Fit an exponential curve through the data points
popt, pcov = curve_fit(func, x_pos, mean_acc)

# Extrapolate the fitted curve to predict the x value for the desired accuracy.
# Take the largest sample size and add 100 until the desired accuracy is reached.
desired_acc = 0.95
x_pred = max(x_pos)
accuracy = func(x_pred, *popt)
while accuracy < desired_acc:
    x_pred = x_pred + 100
    accuracy = func(x_pred, *popt)

# Print predicted x value and append to plot values
print(f"We predict that {x_pred} samples will produce {desired_acc} accuracy.")
x_plot = np.append(x_pos, x_pred)

# Create smooth x values for plot range
lin_x = np.linspace(x_plot[0], x_plot[-1], 100)

# Build the plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(x_pos, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
ax.plot(lin_x, func(lin_x, *popt), "r-", label="Fitted exponential curve.")
ax.set_ylabel("Model clasification accuracy.", fontsize=12)
ax.set_xlabel("Training sample size.", fontsize=12)
ax.set_xticks(x_plot)
ax.set_xticklabels([str(x) for x in x_plot], rotation=90, fontsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
ax.legend(loc=(0.75, 0.75), fontsize=10)
ax.xaxis.grid(True)
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

"""
We also calculate the standard deviation error for curve fit to see how well it fits the
data. The lower the error the better the fit.
"""

std_dev = np.std([mean_acc[i] - func(x_pos[i], *popt)
                  for i in range(len(x_pos))])
print(
    f"The standard deviation error for the curve fit is {np.round(std_dev, 4)}.")

"""
From the extrapolated curve we can see that we require an estimated 3151 images to reach
95% accuracy.

Now, let's use all the data (3269 images) and train the model to see if our prediction
was accurate!
"""

# Now train the model with full dataset to get the actual accuracy
train_ds, val_ds, nr_of_samples = create_subsets(train_data, 0.99)
accuracy = train_model(train_ds, val_ds, test_ds)
print(f"A model accuracy of {accuracy} is reached on {nr_of_samples} images!")

"""
# Conclusion

We see that a model accuracy of about 91-94%* is reached using 3269 images. This is quite
close to our estimation!

Even though we used only 50% of the dataset (1651 images) we were able to model the
training behaviour of our model and predict the number of images we would need to reach
the desired accuracy. This is very useful when a smaller set of data is available and it
has been shown that convergence on a deep learning model is possible but more images are
needed. The image count prediction can be used to plan and budget for further image
collection initiatives.

*Note: repeat results may vary due to randomness.
"""
