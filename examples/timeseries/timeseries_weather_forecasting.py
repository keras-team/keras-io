"""
Title: Timeseries forecasting for weather prediction

**Authors:**

 - [Prabhanshu Attri](https://prabhanshu.com/github),
 - [Yashika Sharma](https://github.com/yashika51),
 - [Kristi Takach](https://github.com/ktakattack),
 - [Falak Shah](https://github.com/falaktheoptimist)
<br>

**Date created:** 2020/06/23  <br>
**Last modified:** 2020/06/30  <br>
**Description:** This notebook demonstrates how to do timeseries forecasting
using a LSTM model.
"""

"""
## Setup


As of 25/6/20, `timeseries_dataset_from_array()` is available with TensorFlow nightly.

If the following cell doesn't work, try uninstalling Tensorflow and Keras using
this command.

```
!pip uninstall tf-nightly keras -y
```

# install tf-nightly for timeseries_dataset_from_array
!pip install 'tf-nightly==2.3.0.dev20200623'
!pip install 'keras==2.4.0'

**Note**: Please restart the runtime using the button above in order to load
installed Tensorflow and Keras versions.

Let's do the necessary imports in the next cell.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

"""
## Climate Data Time-Series

We would be using Jena Climate dataset recorded by the [Max Planck Institute
for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/). The dataset consists
of 14 features such as temperature, pressure, humidity etc, recorded once per
10 minutes. <br/><br/>

**Location**: Weather Station, Max Planck Institute for Biogeochemistry
in Jena, Germany

**Time-frame Considered**: Jan 10, 2009 - December 31, 2016 <br/><br/>


The table below shows the column names, their value formats and description.

Index| Features      |Format             |Description
-----|---------------|-------------------|-----------------------
1    |Date Time      |01.01.2009 00:10:00|Date-time reference
2    |p (mbar)       |996.52             |The pascal SI derived unit of pressure used to
     |               |                   |quantify internal pressure. Meteorological
     |               |                   |reports typically state atmospheric pressure
     |				 |					 |in millibars.
3    |T (degC)       |-8.02              |Temperature in Celsius
4    |Tpot (K)       |265.4              |Temperature in Kelvin
5    |Tdew (degC)    |-8.9               |Temperature in Celsius relative to humidity.
	 |				 |					 |Dew Point is a measure of the absolute amount
	 |				 |					 |of water in the air, the DP is the temperature
	 |				 |					 |at which the air cannot hold all the moisture in
	 |				 |					 |it and water condenses. 
6    |rh (%)         |93.3               |Relative Humidity is a measure of how saturated
	 |				 |					 |the air is with water vapor, the %RH determines
	 |				 |					 |the amount of water contained within collection
	 |				 |					 |objects. 
7    |VPmax (mbar)   |3.33               |Saturation vapor pressure
8    |VPact (mbar)   |3.11               |Vapor pressure
9    |VPdef (mbar)   |0.22               |Vapor pressure deficit
10   |sh (g/kg)      |1.94               |Specific humidity
11   |H2OC (mmol/mol)|3.12               |Water vapor concentration
12   |rho (g/m ** 3) |1307.75            |Airtight
13   |wv (m/s)       |1.03               |Wind speed 
14   |max. wv (m/s)  |1.75               |Maximum wind speed
15   |wd (deg)       |152.3              |Wind direction in degrees
"""

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

zip_path = tf.keras.utils.get_file(
    origin=uri, fname="jena_climate_2009_2016.csv.zip", extract=True
)
csv_path, _ = os.path.splitext(zip_path)

# checking out the first 5 rows of the dataset
df = pd.read_csv(csv_path)
df = df.iloc[:-1]
df.head()

"""
## Raw Visualizations

We are visualizing the features against time to check the sequence and patterns.
"""

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date Time"


def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()


show_raw_visualization(df)

"""
This heat map shows the correlation between different features.
"""

corr = df.corr()
sns.heatmap(corr)

"""
The input data will include pressure, temperature (in Celsius) and specific humidity.
The below line graphs show each feature graphed by hour, month and year.
"""


def show_time_based_visualizations(data, idx):
    selected_feature = feature_keys[idx]
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k"
    )

    dataframe = data.copy()
    dataframe.index = pd.to_datetime(data[date_time_key])

    x_data = [dataframe.index.hour, dataframe.index.month, dataframe.index.year]
    x_labels = ["Hour of Day", "Month", "Year"]

    for i in range(3):
        selected_ax = axes[i]
        sns.lineplot(x=x_data[i], y=dataframe[selected_feature], ax=selected_ax)
        selected_ax.set(xlabel=x_labels[i], ylabel=titles[idx])
    plt.tight_layout()
    plt.show()
    return


# plot pressure
show_time_based_visualizations(df, 0)
# plot temperature
show_time_based_visualizations(df, 1)
# plot specific humidity
show_time_based_visualizations(df, 8)

"""
## Data Preprocessing

Here we are picking 300000 data points for training. The observation is recorded every
10 mins, that means 6 times in an hour. Thus STEP is equal to 6. We are tracking data
from past 720 timestamps(i.e. 720/6=120 hours) and this will be used to predict the
temperature after 72 timestamps(i.e. 76/6=12 hours).

Raw data is normalized using a z score formula. Since every feature has values with
varying ranges, normalization is done to confine the values in a range of [0,1] before
training a neural network.
It is done by subtracting the mean and dividing by the standard deviation of each feature
"""

train_split = 300000
past = 720
future = 72
step = 6
learning_rate = 0.005
validation_steps = 50

batch_size = 256
buffer_size = 10000
evaluation_interval = 200
epochs = 10


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


"""
We will be using selecting few parameters from the dataset to avoid complete memory(RAM)
usage. Moreover, we can see from the correlation heatmap, few parameters like
Relative Humidity and Specific Humidity are redundant.
"""

feature_idx = [0, 1, 5, 7, 8, 10, 11]
print("The selected parameters are:", ", ".join([titles[i] for i in feature_idx]))
selected_features = [feature_keys[i] for i in feature_idx]
features = df[selected_features]
features.index = df[date_time_key]
features.head()

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

# training dataset
start = past + future
end = start + train_split

x_train = train_data[[i for i in range(7)]].values
y_train = features.iloc[start:end][[1]]

sequence_length = int(past / step)

dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


# validation dataset
start = train_split + past + future
end = len(val_data) - past - future

x_val = val_data.iloc[:end][[i for i in range(7)]].values
y_val = features.iloc[start:][[1]]

dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train:
    inputs, targets = batch
    break

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

"""
## Training
"""

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32, return_sequences=False)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")


model.summary()

"""
Saving a checkpoint loads the weights from a trained model so that we don't have to train the
model again and again whenever new runtime is started.
"""

path_checkpoint = "model_checkpoint.keras"
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=5
)

modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

"""
We can visualize the loss with the function below. After one point the loss stops
decreasing.
"""


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")

"""
## Prediction

The trained model above is now able to make predictions for 5 sets of values from
validation set.
"""


def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


for x, y in dataset_val.take(5):
    show_plot(
        [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
        12,
        "Single Step Prediction",
    )
