"""
Title: EEG Signal Classifiation
Author: [Ayyuce Demirbas](https://twitter.com/demirbasayyuce)
Date created: 2022/03/14
Last modified: 2022/03/14
Description: A simple EEG signal classification project.
"""
"""

"""

"""
Download the eeg-data.csv file from
https://www.kaggle.com/wpncrh/classifying-tasks-using-eeg-data-w-tensorflow-nn/data
"""

"""
Import pandas and read the file
"""

import pandas as pd

df = pd.read_csv("eeg-data.csv")

import json

df["eeg_power"] = df.eeg_power.map(json.loads)

"""
We drop the unnecessary columns
"""

df.drop("Unnamed: 0", 1, inplace=True)
df.drop("indra_time", 1, inplace=True)
df.drop("browser_latency", 1, inplace=True)
df.drop("reading_time", 1, inplace=True)
df.drop("attention_esense", 1, inplace=True)
df.drop("meditation_esense", 1, inplace=True)
df.drop("raw_values", 1, inplace=True)
df.drop("signal_quality", 1, inplace=True)
df.drop("createdAt", 1, inplace=True)
df.drop("updatedAt", 1, inplace=True)

df.head()

"""
Print the unique label names, these are the target classes
"""

pd.unique(df.label.values)

len(pd.unique(df.label.values))

"""
We have 68 classes, so we need 68 units at the last layer of the model
"""

"""
Electroencephalography (EEG) power represents amount of activity in certain frequency
bands of the signal while coherence between different electrodes reflects the degree to
which connections are present across brain regions. [1]
"""

type(df["eeg_power"][0])

"""
The eeg_power values are lists, we need to convert them to numpy arrays
"""

import numpy as np

len(df["eeg_power"])

len(df["eeg_power"][0])

"""
We need to define an array with shape = (30013, 8) 
"""

data = np.empty(shape=(30013, 8))

"""
We convert all of the data to numpy array and add them into our empty numpy array named
data:
"""

for i in range(30013):
    element = df.at[i, "eeg_power"] = np.asarray(df["eeg_power"][i])
    data[i] = element

"""
We must convert labels to a numpy array
"""

targets = df["label"].to_numpy()

"""
We use the label encoding technique for handling our target values. 
"""

import sklearn.preprocessing as preprocessing

le = preprocessing.LabelEncoder()
le.fit(targets)

targets = le.transform(targets)

targets

"""
Or, if you want to use one-hot encoding, uncomment and use these lines:
"""

# oh = preprocessing.OneHotEncoder(handle_unknown='ignore')
# targets = oh.fit_transform(df[['label']]).toarray()

"""
Our encoded unique values are:
"""

np.unique(targets)

"""
Normalize the target data (labels):
"""

targets = (targets - targets.mean(axis=0)) / targets.std()

targets

"""
Split the data into train and test sets:
"""

from sklearn.model_selection import train_test_split

data_train, data_test, targets_train, targets_test = train_test_split(
    data,
    targets,
    test_size=0.3,
    shuffle=True,
)

"""
Print the shapes:
"""

print(data_train.shape)
print(data_test.shape)
print(targets_train.shape)
print(targets_test.shape)

"""
Build the model:
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

dropout_rate = 0.3
weight_decay = 1e-5


model = Sequential(
    [
        Dense(
            128,
            kernel_regularizer=regularizers.l2(weight_decay),
            activation="relu",
            input_shape=(data_train.shape[1],),
        ),
        Dropout(dropout_rate),
        Dense(128, kernel_regularizer=regularizers.l2(weight_decay), activation="relu"),
        Dropout(dropout_rate),
        Dense(128, kernel_regularizer=regularizers.l2(weight_decay), activation="relu"),
        Dropout(dropout_rate),
        Dense(128, kernel_regularizer=regularizers.l2(weight_decay), activation="relu"),
        Dropout(dropout_rate),
        Dense(128, kernel_regularizer=regularizers.l2(weight_decay), activation="relu"),
        Dropout(dropout_rate),
        Dense(128, kernel_regularizer=regularizers.l2(weight_decay), activation="relu"),
        Dropout(dropout_rate),
        Dense(68),
    ]
)

"""
Print the model summary:
"""

model.summary()

"""
Compile the model

optimizer = adam

loss = mean squared error, computes the mean squared error between labels and predictions.

metrics = mean absolute error, computes the mean absolute error between labels and
predictions
"""

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

"""
Train the model:
"""

history = model.fit(
    data_train, targets_train, epochs=100, validation_split=0.2, batch_size=64
)

"""
We evaluate the model on the test set:
"""

model.evaluate(data_test, targets_test)

"""
Plot the training and validation loss
"""

import matplotlib.pyplot as plt

#%matplotlib inline

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss vs. epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")
plt.show()

"""
##References

[1]
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5766131/#:~:text=Electroencephalography%20(EEG)%20power%20represents%20amount,across%20brain%20regions%20%5B1%5D.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5766131/#:~:text=Electroencephalography%20(EEG)%20power%20represents%20amount,across%20brain%20regions%20%5B1%5D.
"""
