"""
Title: Timeseries forecasting with LSTM for temperature prediction
Author: [Falak Shah](https://github.com/falaktheoptimist)
Date created: 2020/06/28
Last modified: 2020/06/28
Description: Predict average next day temperature using LSTM based model.
"""

"""
## Introduction
This script demonstrates how you can use an LSTM model for time series
prediction. Usage of timeseries_dataset_from_array is explained for
preparing input data for the model.
"""

"""
## Setup
This example requires TensorFlow 2.3 or higher.
"""


from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import os
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import tensorflow as tf

"""
## Fetch the data

We will use the [Austin weather data from ](https://www.kaggle.com/grubenm/austin-weather)
dataset. It provides timeseries of weather related information for Austin.

We will try to predict the Average temperature for next day using temperature
and date/ month features. The simplicity of this dataset allows us to
demonstrate the time series use case of LSTM effectively.
"""

tf.keras.utils.get_file(
    "/tmp/austin_weather.csv",
    "https://gist.githubusercontent.com/falaktheoptimist/de854aa32393600bfedd38d35419124e/raw/15fa2f9431910fddb74aecf6dd562db80425979a/austin_weather.csv",
)

# Load the data
df = pd.read_csv("/tmp/austin_weather.csv")
df["Date"] = pd.to_datetime(df["Date"])
print(df.head(5))

"""
## Feature generation

Since weather information is seasonal, features indicating the month
and day of the month would be useful for prediction. Creating those additional
features from the date.
"""
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day

"""
## Visualizing the data
"""
fig, ax = plt.subplots(figsize=[16, 9])
ax.plot("Date", "TempAvgF", data=df)
ax.grid(True)
ax.set_xlabel("Date")
ax.set_ylabel("Avg. Temperature (F)")
ax.set_title("Austin temperature pattern")

# Monthly x axis labels
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter("%Y-%m")
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate(rotation=90)

"""
## Input data preparation

Given weather information till day `d`, The target is the temperature at day d+1
The `timeseries_dataset_from_array` that we'll use below gathers samples
starting from day `d` to day `d + sequence_length - 1` in a sample.
So, corresponding target is temperature on day `d + sequence_length`
We use a `sequence_length` of 10.
"""
data_df = df[["Date", "TempAvgF", "month", "day"]]
data_df["Target"] = df["TempAvgF"].shift(-10)
data_df.dropna(inplace=True)
print(data_df.head())

# Train and validation split
data_train = data_df[data_df["Date"].dt.year < 2016]
data_validation = data_df[data_df["Date"].dt.year > 2016]

# Extract the required features and target from the dataframe
feature_list = ["TempAvgF", "month", "day"]
X_train = data_train[feature_list].values
y_train = data_train["Target"].values
X_validation = data_validation[feature_list].values
y_validation = data_validation["Target"].values

"""
Use `timeseries_dataset_from_array` for creating `tf.data.dataset` object from
the array. It creates batches of (batch_size, sequence_length, num_features)
shape used in training. And corresponding (batch_size, 1) target.
Feel free to increase stride to 10 if you do not want any
overlap across samples
"""
dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_train, y_train, sequence_length=10, sequence_stride=1
)
dataset_validation = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_validation,
    y_validation,
    sequence_length=10,
    batch_size=X_validation.shape[0] - 10,
    sequence_stride=1,
)

# A sneak peek at the data
for batch in dataset_train:
    inputs, targets = batch
    break
inp_arr = inputs.numpy()
print("Input shape:  {}".format(inp_arr.shape))
print("Target shape: {}".format(targets.numpy().shape))

print("Sample of input:")
print(inp_arr[0], "\n\n", inp_arr[1])

"""
 Note how the last temperature on the last timestamp of sample 2 (11th day)
 is the target for sample 1 which uses temperatures for day 1-10 as inputs.
"""
print("Targets: \n {}".format(targets))

"""
## Build the LSTM based model

Single LSTM layer followed by a dense layer.
"""
inputs = layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = layers.LSTM(20, return_sequences=False)(inputs)
outputs = layers.Dense(1)(lstm_out)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_ts")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
print(model.summary())

"""
## Train the model
"""
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=5
)
modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/tmp/lstm_ckpt", save_weights_only=True, save_best_only=True
)
history = model.fit(
    dataset_train,
    epochs=200,
    validation_data=dataset_validation,
    callbacks=[es_callback, modelckpt_callback],
)

"""
## Training and validation losses
"""
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

"""
## Visualize results on validation set
Load all validation data and predict
"""
input_validation, target_validation = next(iter(dataset_validation))
output_validation = model.predict(input_validation)

# Plot overlay plots for comparison
plt.figure(figsize=[12, 8])
plt.plot(target_validation.numpy())
plt.plot(output_validation)
plt.legend(["Target", "Predicted"])
plt.title("Predictions and ground truth on validation data")
