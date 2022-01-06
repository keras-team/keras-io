# Estimating required sample size for model training

**Author:** [JacoVerster](https://twitter.com/JacoVerster)<br>
**Date created:** 2021/05/20<br>
**Last modified:** 2021/06/06<br>
**Description:** Modeling the relationship between training set size and model accuracy.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/sample_size_estimate.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/sample_size_estimate.py)



## Introduction

In many real-world scenarios, the amount image data available to train a deep learning model is
limited. This is especially true in the medical imaging domain, where dataset creation is
costly. One of the first questions that usually comes up when approaching a new problem is:
**"how many images will we need to train a good enough machine learning model?"**

In most cases, a small set of samples is available, and we can use it to model the relationship
between training data size and model performance. Such a model can be used to estimate the optimal
number of images needed to arrive at a sample size that would achieve the required model performance.

A systematic review of
[Sample-Size Determination Methodologies](https://www.researchgate.net/publication/335779941_Sample-Size_Determination_Methodologies_for_Machine_Learning_in_Medical_Imaging_Research_A_Systematic_Review)
by Balki et al. provides examples of several sample-size determination methods. In this
example, a balanced subsampling scheme is used to determine the optimal sample size for
our model. This is done by selecting a random subsample consisting of Y number of images
and training the model using the subsample. The model is then evaluated on an independent
test set. This process is repeated N times for each subsample with replacement to allow
for the construction of a mean and confidence interval for the observed performance.

## Setup


```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# Define seed and fixed variables
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
AUTO = tf.data.AUTOTUNE
```

## Load TensorFlow dataset and convert to NumPy arrays

We'll be using the [TF Flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers).


```python
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
num_classes = ds_info.features["label"].num_classes
class_names = ds_info.features["label"].names

print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Convert datasets to NumPy arrays
def dataset_to_array(dataset, image_size, num_classes):
    images, labels = [], []
    for img, lab in dataset.as_numpy_iterator():
        images.append(tf.image.resize(img, image_size).numpy())
        labels.append(tf.one_hot(lab, num_classes))
    return np.array(images), np.array(labels)


img_train, label_train = dataset_to_array(train_data, image_size, num_classes)
img_test, label_test = dataset_to_array(test_data, image_size, num_classes)

num_train_samples = len(img_train)
print(f"Number of training samples: {num_train_samples}")
```

<div class="k-default-codeblock">
```
Number of classes: 5
Class names: ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
Number of training samples: 3303

```
</div>
## Plot a few examples from the test set


```python
plt.figure(figsize=(16, 12))
for n in range(30):
    ax = plt.subplot(5, 6, n + 1)
    plt.imshow(img_test[n].astype("uint8"))
    plt.title(np.array(class_names)[label_test[n] == True][0])
    plt.axis("off")
```


    
![png](/img/examples/keras_recipes/sample_size_estimate/sample_size_estimate_7_0.png)
    


## Augmentation

Define image augmentation using keras preprocessing layers and apply them to the training set.


```python
# Define image augmentation model
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomZoom(height_factor=(-0.1, -0)),
        layers.RandomContrast(factor=0.1),
    ],
)

# Apply the augmentations to the training images and plot a few examples
img_train = image_augmentation(img_train).numpy()

plt.figure(figsize=(16, 12))
for n in range(30):
    ax = plt.subplot(5, 6, n + 1)
    plt.imshow(img_train[n].astype("uint8"))
    plt.title(np.array(class_names)[label_train[n] == True][0])
    plt.axis("off")
```


    
![png](/img/examples/keras_recipes/sample_size_estimate/sample_size_estimate_9_0.png)
    


## Define model building & training functions

We create a few convenience functions to build a transfer-learning model, compile and
train it and unfreeze layers for fine-tuning.


```python

def build_model(num_classes, img_size=image_size[0], top_dropout=0.3):
    """Creates a classifier based on pre-trained MobileNetV2.

    Arguments:
        num_classes: Int, number of classese to use in the softmax layer.
        img_size: Int, square size of input images (defaults is 224).
        top_dropout: Int, value for dropout layer (defaults is 0.3).

    Returns:
        Uncompiled Keras model.
    """

    # Create input and pre-processing layers for MobileNetV2
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    model = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_tensor=x
    )

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.Dropout(top_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    print("Trainable weights:", len(model.trainable_weights))
    print("Non_trainable weights:", len(model.non_trainable_weights))
    return model


def compile_and_train(
    model,
    training_data,
    training_labels,
    metrics=[keras.metrics.AUC(name="auc"), "acc"],
    optimizer=keras.optimizers.Adam(),
    patience=5,
    epochs=5,
):
    """Compiles and trains the model.

    Arguments:
        model: Uncompiled Keras model.
        training_data: NumPy Array, trainig data.
        training_labels: NumPy Array, trainig labels.
        metrics: Keras/TF metrics, requires at least 'auc' metric (default is
                `[keras.metrics.AUC(name='auc'), 'acc']`).
        optimizer: Keras/TF optimizer (defaults is `keras.optimizers.Adam()).
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
        x=training_data,
        y=training_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[stopper],
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

    # Unfreeze from block_name onwards
    set_trainable = False

    for layer in model.layers:
        if block_name in layer.name:
            set_trainable = True
        if set_trainable and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            if verbose == 1:
                print(layer.name, "trainable")
        else:
            if verbose == 1:
                print(layer.name, "NOT trainable")
    print("Trainable weights:", len(model.trainable_weights))
    print("Non-trainable weights:", len(model.non_trainable_weights))
    return model

```

## Define iterative training function

To train a model over several subsample sets we need to create an iterative training function.


```python

def train_model(training_data, training_labels):
    """Trains the model as follows:

    - Trains only the top layers for 10 epochs.
    - Unfreezes deeper layers.
    - Train for 20 more epochs.

    Arguments:
        training_data: NumPy Array, trainig data.
        training_labels: NumPy Array, trainig labels.

    Returns:
        Model accuracy.
    """

    model = build_model(num_classes)

    # Compile and train top layers
    history = compile_and_train(
        model,
        training_data,
        training_labels,
        metrics=[keras.metrics.AUC(name="auc"), "acc"],
        optimizer=keras.optimizers.Adam(),
        patience=3,
        epochs=10,
    )

    # Unfreeze model from block 10 onwards
    model = unfreeze(model, "block_10")

    # Compile and train for 20 epochs with a lower learning rate
    fine_tune_epochs = 20
    total_epochs = history.epoch[-1] + fine_tune_epochs

    history_fine = compile_and_train(
        model,
        training_data,
        training_labels,
        metrics=[keras.metrics.AUC(name="auc"), "acc"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        patience=5,
        epochs=total_epochs,
    )

    # Calculate model accuracy on the test set
    _, _, acc = model.evaluate(img_test, label_test)
    return np.round(acc, 4)

```

## Train models iteratively

Now that we have model building functions and supporting iterative functions we can train
the model over several subsample splits.

- We select the subsample splits as 5%, 10%, 25% and 50% of the downloaded dataset. We
pretend that only 50% of the actual data is available at present.
- We train the model 5 times from scratch at each split and record the accuracy values.

Note that this trains 20 models and will take some time. Make sure you have a GPU runtime
active.

To keep this example lightweight, sample data from a previous training run is provided.


```python

def train_iteratively(sample_splits=[0.05, 0.1, 0.25, 0.5], iter_per_split=5):
    """Trains a model iteratively over several sample splits.

    Arguments:
        sample_splits: List/NumPy array, contains fractions of the trainins set
                        to train over.
        iter_per_split: Int, number of times to train a model per sample split.

    Returns:
        Training accuracy for all splits and iterations and the number of samples
        used for training at each split.
    """
    # Train all the sample models and calculate accuracy
    train_acc = []
    sample_sizes = []

    for fraction in sample_splits:
        print(f"Fraction split: {fraction}")
        # Repeat training 3 times for each sample size
        sample_accuracy = []
        num_samples = int(num_train_samples * fraction)
        for i in range(iter_per_split):
            print(f"Run {i+1} out of {iter_per_split}:")
            # Create fractional subsets
            rand_idx = np.random.randint(num_train_samples, size=num_samples)
            train_img_subset = img_train[rand_idx, :]
            train_label_subset = label_train[rand_idx, :]
            # Train model and calculate accuracy
            accuracy = train_model(train_img_subset, train_label_subset)
            print(f"Accuracy: {accuracy}")
            sample_accuracy.append(accuracy)
        train_acc.append(sample_accuracy)
        sample_sizes.append(num_samples)
    return train_acc, sample_sizes


# Running the above function produces the following outputs
train_acc = [
    [0.8202, 0.7466, 0.8011, 0.8447, 0.8229],
    [0.861, 0.8774, 0.8501, 0.8937, 0.891],
    [0.891, 0.9237, 0.8856, 0.9101, 0.891],
    [0.8937, 0.9373, 0.9128, 0.8719, 0.9128],
]

sample_sizes = [165, 330, 825, 1651]
```

## Learning curve

We now plot the learning curve by fitting an exponential curve through the mean accuracy
points. We use TF to fit an exponential function through the data.

We then extrapolate the learning curve to the predict the accuracy of a model trained on
the whole training set.


```python

def fit_and_predict(train_acc, sample_sizes, pred_sample_size):
    """Fits a learning curve to model training accuracy results.

    Arguments:
        train_acc: List/Numpy Array, training accuracy for all model
                    training splits and iterations.
        sample_sizes: List/Numpy array, number of samples used for training at
                    each split.
        pred_sample_size: Int, sample size to predict model accuracy based on
                        fitted learning curve.
    """
    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    # Define mean squared error cost and exponential curve fit functions
    mse = keras.losses.MeanSquaredError()

    def exp_func(x, a, b):
        return a * x ** b

    # Define variables, learning rate and number of epochs for fitting with TF
    a = tf.Variable(0.0)
    b = tf.Variable(0.0)
    learning_rate = 0.01
    training_epochs = 5000

    # Fit the exponential function to the data
    for epoch in range(training_epochs):
        with tf.GradientTape() as tape:
            y_pred = exp_func(x, a, b)
            cost_function = mse(y_pred, mean_acc)
        # Get gradients and compute adjusted weights
        gradients = tape.gradient(cost_function, [a, b])
        a.assign_sub(gradients[0] * learning_rate)
        b.assign_sub(gradients[1] * learning_rate)
    print(f"Curve fit weights: a = {a.numpy()} and b = {b.numpy()}.")

    # We can now estimate the accuracy for pred_sample_size
    max_acc = exp_func(pred_sample_size, a, b).numpy()

    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.")
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
    ax.plot(x_cont, exp_func(x_cont, a, b), "r-", label="Fitted exponential curve.")
    ax.set_ylabel("Model clasification accuracy.", fontsize=12)
    ax.set_xlabel("Training sample size.", fontsize=12)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
    ax.legend(loc=(0.75, 0.75), fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

    # The mean absolute error (MAE) is calculated for curve fit to see how well
    # it fits the data. The lower the error the better the fit.
    mae = keras.losses.MeanAbsoluteError()
    print(f"The mae for the curve fit is {mae(mean_acc, exp_func(x, a, b)).numpy()}.")


# We use the whole training set to predict the model accuracy
fit_and_predict(train_acc, sample_sizes, pred_sample_size=num_train_samples)
```

<div class="k-default-codeblock">
```
Curve fit weights: a = 0.6445642113685608 and b = 0.0480974055826664.
A model accuracy of 0.9517360925674438 is predicted for 3303 samples.

```
</div>
    
![png](/img/examples/keras_recipes/sample_size_estimate/sample_size_estimate_17_1.png)
    


<div class="k-default-codeblock">
```
The mae for the curve fit is 0.016098812222480774.

```
</div>
From the extrapolated curve we can see that 3303 images will yield an estimated
accuracy of about 95%.

Now, let's use all the data (3303 images) and train the model to see if our prediction
was accurate!


```python
# Now train the model with full dataset to get the actual accuracy
accuracy = train_model(img_train, label_train)
print(f"A model accuracy of {accuracy} is reached on {num_train_samples} images!")
```

<div class="k-default-codeblock">
```

Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
9412608/9406464 [==============================] - 0s 0us/step
Trainable weights: 2
Non_trainable weights: 260
Epoch 1/10
47/47 [==============================] - 34s 88ms/step - loss: 1.0756 - auc: 0.8513 - acc: 0.5821 - val_loss: 0.4947 - val_auc: 0.9761 - val_acc: 0.8429
Epoch 2/10
47/47 [==============================] - 3s 67ms/step - loss: 0.5470 - auc: 0.9629 - acc: 0.8022 - val_loss: 0.3746 - val_auc: 0.9854 - val_acc: 0.8882
Epoch 3/10
47/47 [==============================] - 3s 66ms/step - loss: 0.4495 - auc: 0.9744 - acc: 0.8445 - val_loss: 0.3474 - val_auc: 0.9861 - val_acc: 0.8882
Epoch 4/10
47/47 [==============================] - 3s 66ms/step - loss: 0.3914 - auc: 0.9802 - acc: 0.8647 - val_loss: 0.3171 - val_auc: 0.9882 - val_acc: 0.8912
Epoch 5/10
47/47 [==============================] - 3s 73ms/step - loss: 0.3631 - auc: 0.9832 - acc: 0.8681 - val_loss: 0.2983 - val_auc: 0.9895 - val_acc: 0.9003
Epoch 6/10
47/47 [==============================] - 3s 67ms/step - loss: 0.3242 - auc: 0.9867 - acc: 0.8856 - val_loss: 0.2915 - val_auc: 0.9898 - val_acc: 0.9003
Epoch 7/10
47/47 [==============================] - 3s 73ms/step - loss: 0.3016 - auc: 0.9883 - acc: 0.8930 - val_loss: 0.2912 - val_auc: 0.9895 - val_acc: 0.9033
Epoch 8/10
47/47 [==============================] - 3s 66ms/step - loss: 0.2765 - auc: 0.9906 - acc: 0.9017 - val_loss: 0.2824 - val_auc: 0.9900 - val_acc: 0.9033
Epoch 9/10
47/47 [==============================] - 3s 66ms/step - loss: 0.2721 - auc: 0.9907 - acc: 0.9028 - val_loss: 0.2804 - val_auc: 0.9899 - val_acc: 0.9033
Epoch 10/10
47/47 [==============================] - 3s 65ms/step - loss: 0.2564 - auc: 0.9914 - acc: 0.9098 - val_loss: 0.2913 - val_auc: 0.9891 - val_acc: 0.8973
Trainable weights: 24
Non-trainable weights: 238
Epoch 1/29
47/47 [==============================] - 9s 112ms/step - loss: 0.3316 - auc: 0.9850 - acc: 0.8789 - val_loss: 0.2392 - val_auc: 0.9915 - val_acc: 0.9033
Epoch 2/29
47/47 [==============================] - 4s 93ms/step - loss: 0.1497 - auc: 0.9966 - acc: 0.9478 - val_loss: 0.2797 - val_auc: 0.9906 - val_acc: 0.8731
Epoch 3/29
47/47 [==============================] - 4s 94ms/step - loss: 0.0981 - auc: 0.9982 - acc: 0.9640 - val_loss: 0.1795 - val_auc: 0.9960 - val_acc: 0.9366
Epoch 4/29
47/47 [==============================] - 4s 94ms/step - loss: 0.0652 - auc: 0.9990 - acc: 0.9788 - val_loss: 0.2161 - val_auc: 0.9924 - val_acc: 0.9275
Epoch 5/29
47/47 [==============================] - 4s 94ms/step - loss: 0.0327 - auc: 0.9999 - acc: 0.9896 - val_loss: 0.2161 - val_auc: 0.9919 - val_acc: 0.9517
Epoch 6/29
47/47 [==============================] - 4s 94ms/step - loss: 0.0269 - auc: 0.9999 - acc: 0.9923 - val_loss: 0.2485 - val_auc: 0.9894 - val_acc: 0.9335
Epoch 7/29
47/47 [==============================] - 4s 94ms/step - loss: 0.0174 - auc: 0.9998 - acc: 0.9956 - val_loss: 0.2692 - val_auc: 0.9871 - val_acc: 0.9215
Epoch 8/29
47/47 [==============================] - 4s 94ms/step - loss: 0.0253 - auc: 0.9999 - acc: 0.9913 - val_loss: 0.2645 - val_auc: 0.9864 - val_acc: 0.9275
Restoring model weights from the end of the best epoch.
Epoch 00008: early stopping
12/12 [==============================] - 1s 39ms/step - loss: 0.1378 - auc: 0.9975 - acc: 0.9537
A model accuracy of 0.9537 is reached on 3303 images!

```
</div>
## Conclusion

We see that a model accuracy of about 94-96%* is reached using 3303 images. This is quite
close to our estimate!

Even though we used only 50% of the dataset (1651 images) we were able to model the training
behaviour of our model and predict the model accuracy for a given amount of images. This same
methodology can be used to predict the amount of images needed to reach a desired accuracy.
This is very useful when a smaller set of data is available, and it has been shown that
convergence on a deep learning model is possible, but more images are needed. The image count
prediction can be used to plan and budget for further image collection initiatives.
