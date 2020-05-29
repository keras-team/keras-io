"""
Title: Regression Model on Boston Housing Dataset
Author: [Robert](https://www.linkedin.com/in/robert1995/)
Date created: 2019/05/30
Last modified: 2020/05/30
Description: Perform regression model on Boston Housing Dataset.
"""
"""
## Introduction
The Boston Housing Dataset is provided by keras. The details about the features can be
found here
https://keras.io/api/datasets/boston_housing/

The steps in the model are as follows:

1. Load and Split Dataset
2. Build and Fit Model
3. Visualize Train and Validation Errors
4. Make Predictions
"""

import pandas
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

"""
## Load and Split Dataset
"""

# The dataset is splitted by 70:30 proportion
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.3, seed=42
)

print("Train Dimension:", x_train.shape)
print("Test Dimension:", x_test.shape)

"""
## Build and Fit Model
"""


def build_model():
    model = keras.Sequential(
        [
            # input_shape is number of features
            layers.Dense(32, activation="relu", input_shape=[x_train.shape[1]]),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    # The loss function is mean squared error
    # The metrics is Mean Absolute Error and Mean Squared Error
    model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])
    return model


# Build the model
model = build_model()
model.summary()

# Use early stopping to break the training if the val_loss doesn't changed in 10 epochs
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)


history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=100,
    verbose=0,
    validation_split=0.3,
    callbacks=[early_stop],
)

"""
## Visualize Train and Validation Loss
"""

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper right")
plt.show()

# Test Loss
loss, mae, mse = model.evaluate(x_test, y_test, verbose=2)

"""
## Make Prediction
"""

prediction = model.predict(x_test).flatten()
prediction

plt.axes(aspect="equal")
plt.scatter(y_test, prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()


"""
The model seems to work well. The prediction is reasonably well. Let's see the residual
distribution
"""

error = prediction - y_test
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()

"""
The residual distribution is not quite gaussion. This might cause by the outlier or the
small size of the dataset. But overall the model performs quite well.
"""
