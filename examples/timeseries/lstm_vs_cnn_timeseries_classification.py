"""
Title: Timeseries Classification with LSTM and CNN Models
Author: [Georgios Sklavounakos] (https://github.com/gsklavounakos)
Date created: 2025/06/05
Last modified: 2025/06/05
Description: Comparing LSTM and 1D CNN models for timeseries classification on the FordA dataset from the UCR/UEA archive.
Accelerator: GPU
"""

"""
## Introduction

This example demonstrates how to perform timeseries classification using two deep learning models:
a Long Short-Term Memory (LSTM) model, which processes sequences using recurrent layers, and a 1D Convolutional
Neural Network (CNN) model, which uses convolutions to detect local temporal patterns. We use the FordA dataset
from the [UCR/UEA archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), which contains univariate
timeseries of engine noise measurements for binary classification (fault detection). This example compares the
performance of both models under identical training conditions.

The workflow includes:
- Loading and preprocessing the FordA dataset
- Building and training LSTM and CNN models
- Evaluating and comparing their performance
- Visualizing training metrics
"""

"""
## Setup
"""

import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

"""
## Load the Data: FordA Dataset

### Dataset Description

The FordA dataset contains 3601 training instances and 1320 testing instances, each a univariate timeseries of
500 timesteps representing engine noise measurements. The task is to classify whether a fault is present (label 1)
or not (label -1). The data is z-normalized (mean=0, std=1) and sourced from the UCR/UEA archive. For details, see
[the dataset description](http://www.j-wichard.de/publications/FordPaper.pdf).
"""

def readucr(filename):
    """Read UCR timeseries dataset from a TSV file."""
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0].astype(int)
    x = data[:, 1:].astype(np.float32)
    return x, y

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

"""
## Visualize the Data

We plot one example timeseries from each class to understand the data's characteristics.
"""

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure(figsize=(8, 4))
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label=f"class {c}")
plt.title("Sample Time Series from Each Class")
plt.xlabel("Timestep")
plt.ylabel("Amplitude")
plt.legend(loc="best")
plt.show()
plt.close()

"""
## Preprocess the Data

The timeseries are already z-normalized and of fixed length (500 timesteps). We reshape the data to
`(samples, timesteps, 1)` to represent univariate timeseries as multivariate with one channel, enabling
compatibility with both LSTM and CNN models. We also standardize labels to 0 and 1 for binary classification.
"""

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Standardize labels: map {-1, 1} to {0, 1}
y_train = np.where(y_train == -1, 0, y_train)
y_test = np.where(y_test == -1, 0, y_test)

# Shuffle training data for validation split
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

num_classes = len(np.unique(y_train))

"""
## Build the Models

We define two models:
1. **LSTM Model**: A recurrent model with two LSTM layers for sequential processing, with dropout for regularization.
2. **CNN Model**: A convolutional model with 1D convolutions, batch normalization, max-pooling, and global average pooling,
   inspired by [this paper](https://arxiv.org/abs/1611.06455).

Both models use similar hyperparameters for a fair comparison.
"""

def build_lstm_model(input_shape, num_classes):
    """Build an LSTM-based model for timeseries classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def build_cnn_model(input_shape, num_classes):
    """Build a 1D CNN-based model for timeseries classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=7, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

input_shape = x_train.shape[1:]
lstm_model = build_lstm_model(input_shape, num_classes)
cnn_model = build_cnn_model(input_shape, num_classes)

"""
## Train the Models

We train both models with identical settings: Adam optimizer, sparse categorical crossentropy loss,
and early stopping to prevent overfitting. We save the best model weights based on validation loss
and reload them for evaluation.
"""

epochs = 100
batch_size = 32

def train_model(model, model_name, x_train, y_train, x_test, y_test):
    """Train and evaluate a model, return history and test metrics."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                f"best_{model_name}_model.keras",
                save_best_only=True,
                monitor="val_loss"
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # Load best model for evaluation
    best_model = keras.models.load_model(f"best_{model_name}_model.keras")
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
    
    return history, test_acc, test_loss

# Train LSTM model
print("Training LSTM model...")
lstm_history, lstm_test_acc, lstm_test_loss = train_model(lstm_model, "LSTM", x_train, y_train, x_test, y_test)

# Train CNN model
print("Training CNN model...")
cnn_history, cnn_test_acc, cnn_test_loss = train_model(cnn_model, "CNN", x_train, y_train, x_test, y_test)

"""
## Visualize Training Metrics

We plot the training and validation accuracy and loss for both models to compare their performance.
"""

def plot_training_metrics(histories, model_names):
    """Plot training and validation accuracy/loss for multiple models."""
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    for history, name in zip(histories, model_names):
        plt.plot(history.history["sparse_categorical_accuracy"], label=f"{name} Train")
        plt.plot(history.history["val_sparse_categorical_accuracy"], linestyle="--", label=f"{name} Val")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    for history, name in zip(histories, model_names):
        plt.plot(history.history["loss"], label=f"{name} Train")
        plt.plot(history.history["val_loss"], linestyle="--", label=f"{name} Val")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_metrics([lstm_history, cnn_history], ["LSTM", "CNN"])

"""
## Evaluate and Compare Models

We compare the test accuracy and loss of both models to assess their performance.
"""

print("\nModel Comparison:")
print(f"LSTM Test Accuracy: {lstm_test_acc:.4f}, Loss: {lstm_test_loss:.4f}")
print(f"CNN Test Accuracy: {cnn_test_acc:.4f}, Loss: {cnn_test_loss:.4f}")

"""
## Conclusions

This example compared an LSTM-based model and a 1D CNN-based model for timeseries classification
on the FordA dataset. The LSTM model leverages sequential dependencies, while the CNN model captures
local temporal patterns through convolutions. The CNN often converges faster due to its robust architecture
with batch normalization and global pooling, while the LSTM may better handle long-term dependencies.

To improve performance, consider:
- Tuning hyperparameters (e.g., number of layers, units, or kernel sizes) using Keras Tuner.
- Adding further regularization (e.g., L2 regularization) to prevent overfitting.
- Experimenting with hybrid architectures combining LSTM and CNN layers.
- Using data augmentation techniques for timeseries, such as jittering or scaling.
"""