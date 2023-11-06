# Electroencephalogram Signal Classification for action identification

**Author:** [Suvaditya Mukherjee](https://github.com/suvadityamuk)<br>
**Date created:** 2022/11/03<br>
**Last modified:** 2022/11/05<br>
**Description:** Training a Convolutional model to classify EEG signals produced by exposure to certain stimuli.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/eeg_signal_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/eeg_signal_classification.py)



---
## Introduction

The following example explores how we can make a Convolution-based Neural Network to
perform classification on Electroencephalogram signals captured when subjects were
exposed to different stimuli.
We train a model from scratch since such signal-classification models are fairly scarce
in pre-trained format.
The data we use is sourced from the UC Berkeley-Biosense Lab where the data was collected
from 15 subjects at the same time.
Our process is as follows:

- Load the [UC Berkeley-Biosense Synchronized Brainwave Dataset](https://www.kaggle.com/datasets/berkeley-biosense/synchronized-brainwave-dataset)
- Visualize random samples from the data
- Pre-process, collate and scale the data to finally make a `tf.data.Dataset`
- Prepare class weights in order to tackle major imbalances
- Create a Conv1D and Dense-based model to perform classification
- Define callbacks and hyperparameters
- Train the model
- Plot metrics from History and perform evaluation

This example needs the following external dependencies (Gdown, Scikit-learn, Pandas,
Numpy, Matplotlib). You can install it via the following commands.

Gdown is an external package used to download large files from Google Drive. To know
more, you can refer to its [PyPi page here](https://pypi.org/project/gdown)

---
## Setup and Data Downloads

First, lets install our dependencies:


```python
!pip install gdown -q
!pip install sklearn -q
!pip install pandas -q
!pip install numpy -q
!pip install matplotlib -q
```

Next, lets download our dataset.
The gdown package makes it easy to download the data from Google Drive:


```python
!gdown 1V5B7Bt6aJm0UHbR7cRKBEK8jx7lYPVuX
!# gdown will download eeg-data.csv onto the local drive for use. Total size of
!# eeg-data.csv is 105.7 MB
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn import preprocessing, model_selection
import random

QUALITY_THRESHOLD = 128
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2
```
<div class="k-default-codeblock">
```
Downloading...
From: https://drive.google.com/uc?id=1V5B7Bt6aJm0UHbR7cRKBEK8jx7lYPVuX
To: /home/suvaditya/Programming/personal/oss/keras-io/scripts/tmp_8705318/eeg-data.csv
100%|████████████████████████████████████████| 106M/106M [00:05<00:00, 17.7MB/s]

```
</div>
---
## Read data from `eeg-data.csv`

We use the Pandas library to read the `eeg-data.csv` file and display the first 5 rows
using the `.head()` command


```python
eeg = pd.read_csv("eeg-data.csv")
```

We remove unlabeled samples from our dataset as they do not contribute to the model. We
also perform a `.drop()` operation on the columns that are not required for training data
preparation


```python
unlabeled_eeg = eeg[eeg["label"] == "unlabeled"]
eeg = eeg.loc[eeg["label"] != "unlabeled"]
eeg = eeg.loc[eeg["label"] != "everyone paired"]

eeg.drop(
    [
        "indra_time",
        "Unnamed: 0",
        "browser_latency",
        "reading_time",
        "attention_esense",
        "meditation_esense",
        "updatedAt",
        "createdAt",
    ],
    axis=1,
    inplace=True,
)

eeg.reset_index(drop=True, inplace=True)
eeg.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>eeg_power</th>
      <th>raw_values</th>
      <th>signal_quality</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>[56887.0, 45471.0, 20074.0, 5359.0, 22594.0, 7...</td>
      <td>[99.0, 96.0, 91.0, 89.0, 91.0, 89.0, 87.0, 93....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>[11626.0, 60301.0, 5805.0, 15729.0, 4448.0, 33...</td>
      <td>[23.0, 40.0, 64.0, 89.0, 86.0, 33.0, -14.0, -1...</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>[15777.0, 33461.0, 21385.0, 44193.0, 11741.0, ...</td>
      <td>[41.0, 26.0, 16.0, 20.0, 34.0, 51.0, 56.0, 55....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>[311822.0, 44739.0, 19000.0, 19100.0, 2650.0, ...</td>
      <td>[208.0, 198.0, 122.0, 84.0, 161.0, 249.0, 216....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[687393.0, 10289.0, 2942.0, 9874.0, 1059.0, 29...</td>
      <td>[129.0, 133.0, 114.0, 105.0, 101.0, 109.0, 99....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
  </tbody>
</table>


In the data, the samples recorded are given a score from 0 to 128 based on how
well-calibrated the sensor was (0 being best, 200 being worst). We filter the values
based on an arbitrary cutoff limit of 128.


```python

def convert_string_data_to_values(value_string):
    str_list = json.loads(value_string)
    return str_list


eeg["raw_values"] = eeg["raw_values"].apply(convert_string_data_to_values)

eeg = eeg.loc[eeg["signal_quality"] < QUALITY_THRESHOLD]
print(eeg.shape)
eeg.head()
```

<div class="k-default-codeblock">
```
(9954, 5)

```
</div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>eeg_power</th>
      <th>raw_values</th>
      <th>signal_quality</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>[56887.0, 45471.0, 20074.0, 5359.0, 22594.0, 7...</td>
      <td>[99.0, 96.0, 91.0, 89.0, 91.0, 89.0, 87.0, 93....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>[11626.0, 60301.0, 5805.0, 15729.0, 4448.0, 33...</td>
      <td>[23.0, 40.0, 64.0, 89.0, 86.0, 33.0, -14.0, -1...</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>[15777.0, 33461.0, 21385.0, 44193.0, 11741.0, ...</td>
      <td>[41.0, 26.0, 16.0, 20.0, 34.0, 51.0, 56.0, 55....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>[311822.0, 44739.0, 19000.0, 19100.0, 2650.0, ...</td>
      <td>[208.0, 198.0, 122.0, 84.0, 161.0, 249.0, 216....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[687393.0, 10289.0, 2942.0, 9874.0, 1059.0, 29...</td>
      <td>[129.0, 133.0, 114.0, 105.0, 101.0, 109.0, 99....</td>
      <td>0</td>
      <td>blinkInstruction</td>
    </tr>
  </tbody>
</table>


---
## Visualize one random sample from the data

We visualize one sample from the data to understand how the stimulus-induced signal looks
like


```python

def view_eeg_plot(idx):
    data = eeg.loc[idx, "raw_values"]
    plt.plot(data)
    plt.title(f"Sample random plot")
    plt.show()


view_eeg_plot(7)
```


    
![png](/img/examples/timeseries/eeg_signal_classification/eeg_signal_15_0.png)
    


---
## Pre-process and collate data

There are a total of 67 different labels present in the data, where there are numbered
sub-labels. We collate them under a single label as per their numbering and replace them
in the data itself. Following this process, we perform simple Label encoding to get them
in an integer format.


```python
print("Before replacing labels")
print(eeg["label"].unique(), "\n")
print(len(eeg["label"].unique()), "\n")


eeg.replace(
    {
        "label": {
            "blink1": "blink",
            "blink2": "blink",
            "blink3": "blink",
            "blink4": "blink",
            "blink5": "blink",
            "math1": "math",
            "math2": "math",
            "math3": "math",
            "math4": "math",
            "math5": "math",
            "math6": "math",
            "math7": "math",
            "math8": "math",
            "math9": "math",
            "math10": "math",
            "math11": "math",
            "math12": "math",
            "thinkOfItems-ver1": "thinkOfItems",
            "thinkOfItems-ver2": "thinkOfItems",
            "video-ver1": "video",
            "video-ver2": "video",
            "thinkOfItemsInstruction-ver1": "thinkOfItemsInstruction",
            "thinkOfItemsInstruction-ver2": "thinkOfItemsInstruction",
            "colorRound1-1": "colorRound1",
            "colorRound1-2": "colorRound1",
            "colorRound1-3": "colorRound1",
            "colorRound1-4": "colorRound1",
            "colorRound1-5": "colorRound1",
            "colorRound1-6": "colorRound1",
            "colorRound2-1": "colorRound2",
            "colorRound2-2": "colorRound2",
            "colorRound2-3": "colorRound2",
            "colorRound2-4": "colorRound2",
            "colorRound2-5": "colorRound2",
            "colorRound2-6": "colorRound2",
            "colorRound3-1": "colorRound3",
            "colorRound3-2": "colorRound3",
            "colorRound3-3": "colorRound3",
            "colorRound3-4": "colorRound3",
            "colorRound3-5": "colorRound3",
            "colorRound3-6": "colorRound3",
            "colorRound4-1": "colorRound4",
            "colorRound4-2": "colorRound4",
            "colorRound4-3": "colorRound4",
            "colorRound4-4": "colorRound4",
            "colorRound4-5": "colorRound4",
            "colorRound4-6": "colorRound4",
            "colorRound5-1": "colorRound5",
            "colorRound5-2": "colorRound5",
            "colorRound5-3": "colorRound5",
            "colorRound5-4": "colorRound5",
            "colorRound5-5": "colorRound5",
            "colorRound5-6": "colorRound5",
            "colorInstruction1": "colorInstruction",
            "colorInstruction2": "colorInstruction",
            "readyRound1": "readyRound",
            "readyRound2": "readyRound",
            "readyRound3": "readyRound",
            "readyRound4": "readyRound",
            "readyRound5": "readyRound",
            "colorRound1": "colorRound",
            "colorRound2": "colorRound",
            "colorRound3": "colorRound",
            "colorRound4": "colorRound",
            "colorRound5": "colorRound",
        }
    },
    inplace=True,
)

print("After replacing labels")
print(eeg["label"].unique())
print(len(eeg["label"].unique()))

le = preprocessing.LabelEncoder()  # Generates a look-up table
le.fit(eeg["label"])
eeg["label"] = le.transform(eeg["label"])
```

<div class="k-default-codeblock">
```
Before replacing labels
['blinkInstruction' 'blink1' 'blink2' 'blink3' 'blink4' 'blink5'
 'relaxInstruction' 'relax' 'mathInstruction' 'math1' 'math2' 'math3'
 'math4' 'math5' 'math6' 'math7' 'math8' 'math9' 'math10' 'math11'
 'math12' 'musicInstruction' 'music' 'videoInstruction' 'video-ver1'
 'thinkOfItemsInstruction-ver1' 'thinkOfItems-ver1' 'colorInstruction1'
 'colorInstruction2' 'readyRound1' 'colorRound1-1' 'colorRound1-2'
 'colorRound1-3' 'colorRound1-4' 'colorRound1-5' 'colorRound1-6'
 'readyRound2' 'colorRound2-1' 'colorRound2-2' 'colorRound2-3'
 'colorRound2-4' 'colorRound2-5' 'colorRound2-6' 'readyRound3'
 'colorRound3-1' 'colorRound3-2' 'colorRound3-3' 'colorRound3-4'
 'colorRound3-5' 'colorRound3-6' 'readyRound4' 'colorRound4-1'
 'colorRound4-2' 'colorRound4-3' 'colorRound4-4' 'colorRound4-5'
 'colorRound4-6' 'readyRound5' 'colorRound5-1' 'colorRound5-2'
 'colorRound5-3' 'colorRound5-4' 'colorRound5-5' 'colorRound5-6'
 'video-ver2' 'thinkOfItemsInstruction-ver2' 'thinkOfItems-ver2'] 
```
</div>
    
<div class="k-default-codeblock">
```
67 
```
</div>
    
<div class="k-default-codeblock">
```
After replacing labels
['blinkInstruction' 'blink' 'relaxInstruction' 'relax' 'mathInstruction'
 'math' 'musicInstruction' 'music' 'videoInstruction' 'video'
 'thinkOfItemsInstruction' 'thinkOfItems' 'colorInstruction' 'readyRound'
 'colorRound1' 'colorRound2' 'colorRound3' 'colorRound4' 'colorRound5']
19

```
</div>
We extract the number of unique classes present in the data


```python
num_classes = len(eeg["label"].unique())
print(num_classes)
```

<div class="k-default-codeblock">
```
19

```
</div>
We now visualize the number of samples present in each class using a Bar plot.


```python
plt.bar(range(num_classes), eeg["label"].value_counts())
plt.title("Number of samples per class")
plt.show()
```


    
![png](/img/examples/timeseries/eeg_signal_classification/eeg_signal_22_0.png)
    


---
## Scale and split data

We perform a simple Min-Max scaling to bring the value-range between 0 and 1. We do not
use Standard Scaling as the data does not follow a Gaussian distribution.


```python
scaler = preprocessing.MinMaxScaler()
series_list = [
    scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in eeg["raw_values"]
]

labels_list = [i for i in eeg["label"]]
```

We now create a Train-test split with a 15% holdout set. Following this, we reshape the
data to create a sequence of length 512. We also convert the labels from their current
label-encoded form to a one-hot encoding to enable use of several different
`keras.metrics` functions.


```python
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    series_list, labels_list, test_size=0.15, random_state=42, shuffle=True
)

print(
    f"Length of x_train : {len(x_train)}\nLength of x_test : {len(x_test)}\nLength of y_train : {len(y_train)}\nLength of y_test : {len(y_test)}"
)

x_train = np.asarray(x_train).astype(np.float32).reshape(-1, 512, 1)
y_train = np.asarray(y_train).astype(np.float32).reshape(-1, 1)
y_train = keras.utils.to_categorical(y_train)

x_test = np.asarray(x_test).astype(np.float32).reshape(-1, 512, 1)
y_test = np.asarray(y_test).astype(np.float32).reshape(-1, 1)
y_test = keras.utils.to_categorical(y_test)
```

<div class="k-default-codeblock">
```
Length of x_train : 8460
Length of x_test : 1494
Length of y_train : 8460
Length of y_test : 1494

```
</div>
---
## Prepare `tf.data.Dataset`

We now create a `tf.data.Dataset` from this data to prepare it for training. We also
shuffle and batch the data for use later.


```python
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
```

---
## Make Class Weights using Naive method

As we can see from the plot of number of samples per class, the dataset is imbalanced.
Hence, we **calculate weights for each class** to make sure that the model is trained in
a fair manner without preference to any specific class due to greater number of samples.

We use a naive method to calculate these weights, finding an **inverse proportion** of
each class and using that as the weight.


```python
vals_dict = {}
for i in eeg["label"]:
    if i in vals_dict.keys():
        vals_dict[i] += 1
    else:
        vals_dict[i] = 1
total = sum(vals_dict.values())

# Formula used - Naive method where
# weight = 1 - (no. of samples present / total no. of samples)
# So more the samples, lower the weight

weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
print(weight_dict)
```

<div class="k-default-codeblock">
```
{1: 0.9872413100261201, 0: 0.975989551938919, 14: 0.9841269841269842, 13: 0.9061683745228049, 9: 0.9838255977496484, 8: 0.9059674502712477, 11: 0.9847297568816556, 10: 0.9063692987743621, 18: 0.9838255977496484, 17: 0.9057665260196905, 16: 0.9373116335141651, 15: 0.9065702230259193, 2: 0.9211372312638135, 12: 0.9525818766325096, 3: 0.9245529435402853, 4: 0.943841671689773, 5: 0.9641350210970464, 6: 0.981514968856741, 7: 0.9443439823186659}

```
</div>
---
## Define simple function to plot all the metrics present in a `keras.callbacks.History`
object


```python

def plot_history_metrics(history: keras.callbacks.History):
    total_plots = len(history.history)
    cols = total_plots // 2

    rows = total_plots // cols

    if total_plots % cols != 0:
        rows += 1

    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    plt.show()

```

---
## Define function to generate Convolutional model


```python

def create_model():
    input_layer = keras.Input(shape=(512, 1))

    x = layers.Conv1D(
        filters=32, kernel_size=3, strides=2, activation="relu", padding="same"
    )(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=128, kernel_size=5, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=256, kernel_size=5, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=512, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=1024, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        2048, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        1024, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=input_layer, outputs=output_layer)

```

---
## Get Model summary


```python
conv_model = create_model()

print(conv_model.summary())
```

<div class="k-default-codeblock">
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 512, 1)]          0         
                                                                 
 conv1d (Conv1D)             (None, 256, 32)           128       
                                                                 
 batch_normalization (BatchN  (None, 256, 32)          128       
 ormalization)                                                   
                                                                 
 conv1d_1 (Conv1D)           (None, 128, 64)           6208      
                                                                 
 batch_normalization_1 (Batc  (None, 128, 64)          256       
 hNormalization)                                                 
                                                                 
 conv1d_2 (Conv1D)           (None, 64, 128)           41088     
                                                                 
 batch_normalization_2 (Batc  (None, 64, 128)          512       
 hNormalization)                                                 
                                                                 
 conv1d_3 (Conv1D)           (None, 32, 256)           164096    
                                                                 
 batch_normalization_3 (Batc  (None, 32, 256)          1024      
 hNormalization)                                                 
                                                                 
 conv1d_4 (Conv1D)           (None, 16, 512)           918016    
                                                                 
 batch_normalization_4 (Batc  (None, 16, 512)          2048      
 hNormalization)                                                 
                                                                 
 conv1d_5 (Conv1D)           (None, 8, 1024)           3671040   
                                                                 
 batch_normalization_5 (Batc  (None, 8, 1024)          4096      
 hNormalization)                                                 
                                                                 
 dropout (Dropout)           (None, 8, 1024)           0         
                                                                 
 flatten (Flatten)           (None, 8192)              0         
                                                                 
 dense (Dense)               (None, 4096)              33558528  
                                                                 
 dropout_1 (Dropout)         (None, 4096)              0         
                                                                 
 dense_1 (Dense)             (None, 2048)              8390656   
                                                                 
 dropout_2 (Dropout)         (None, 2048)              0         
                                                                 
 dense_2 (Dense)             (None, 1024)              2098176   
                                                                 
 dropout_3 (Dropout)         (None, 1024)              0         
                                                                 
 dense_3 (Dense)             (None, 128)               131200    
                                                                 
 dense_4 (Dense)             (None, 19)                2451      
                                                                 
=================================================================
Total params: 48,989,651
Trainable params: 48,985,619
Non-trainable params: 4,032
_________________________________________________________________
None

```
</div>
---
## Define callbacks, optimizer, loss and metrics

We set the number of epochs at 30 after performing extensive experimentation. It was seen
that this was the optimal number, after performing Early-Stopping analysis as well.
We define a Model Checkpoint callback to make sure that we only get the best model
weights.
We also define a ReduceLROnPlateau as there were several cases found during
experimentation where the loss stagnated after a certain point. On the other hand, a
direct LRScheduler was found to be too aggressive in its decay.


```python
epochs = 30

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_top_k_categorical_accuracy",
        factor=0.2,
        patience=2,
        min_lr=0.000001,
    ),
]

optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
loss = keras.losses.CategoricalCrossentropy()
```

---
## Compile model and call `model.fit()`

We use the `Adam` optimizer since it is commonly considered the best choice for
preliminary training, and was found to be the best optimizer.
We use `CategoricalCrossentropy` as the loss as our labels are in a one-hot-encoded form.

We define the `TopKCategoricalAccuracy(k=3)`, `AUC`, `Precision` and `Recall` metrics to
further aid in understanding the model better.


```python
conv_model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        keras.metrics.TopKCategoricalAccuracy(k=3),
        keras.metrics.AUC(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
    ],
)

conv_model_history = conv_model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=test_dataset,
    class_weight=weight_dict,
)
```

<div class="k-default-codeblock">
```
Epoch 1/30
133/133 [==============================] - 11s 63ms/step - loss: 12.8625 - top_k_categorical_accuracy: 0.2747 - auc: 0.6495 - precision: 0.0806 - recall: 5.9102e-04 - val_loss: 3.9201 - val_top_k_categorical_accuracy: 0.2610 - val_auc: 0.6191 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 0.0010
Epoch 2/30
133/133 [==============================] - 8s 58ms/step - loss: 3.1074 - top_k_categorical_accuracy: 0.3106 - auc: 0.6959 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 3.1456 - val_top_k_categorical_accuracy: 0.2610 - val_auc: 0.6480 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 0.0010
Epoch 3/30
133/133 [==============================] - 8s 58ms/step - loss: 2.7097 - top_k_categorical_accuracy: 0.3190 - auc: 0.7112 - precision: 0.3000 - recall: 3.5461e-04 - val_loss: 3.1122 - val_top_k_categorical_accuracy: 0.2222 - val_auc: 0.6147 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 0.0010
Epoch 4/30
133/133 [==============================] - 8s 59ms/step - loss: 2.5204 - top_k_categorical_accuracy: 0.3603 - auc: 0.7473 - precision: 0.4706 - recall: 0.0019 - val_loss: 2.9930 - val_top_k_categorical_accuracy: 0.2697 - val_auc: 0.6289 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 2.0000e-04
Epoch 5/30
133/133 [==============================] - 8s 59ms/step - loss: 2.3726 - top_k_categorical_accuracy: 0.4090 - auc: 0.7879 - precision: 0.5238 - recall: 0.0052 - val_loss: 3.0241 - val_top_k_categorical_accuracy: 0.3039 - val_auc: 0.6718 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 2.0000e-04
Epoch 6/30
133/133 [==============================] - 8s 60ms/step - loss: 2.2426 - top_k_categorical_accuracy: 0.4736 - auc: 0.8214 - precision: 0.5230 - recall: 0.0108 - val_loss: 3.4078 - val_top_k_categorical_accuracy: 0.3112 - val_auc: 0.6652 - val_precision: 0.1053 - val_recall: 0.0013 - lr: 2.0000e-04
Epoch 7/30
133/133 [==============================] - 8s 59ms/step - loss: 2.1656 - top_k_categorical_accuracy: 0.5063 - auc: 0.8391 - precision: 0.5207 - recall: 0.0104 - val_loss: 3.4816 - val_top_k_categorical_accuracy: 0.2871 - val_auc: 0.6571 - val_precision: 0.1944 - val_recall: 0.0047 - lr: 2.0000e-04
Epoch 8/30
133/133 [==============================] - 8s 59ms/step - loss: 2.0417 - top_k_categorical_accuracy: 0.5721 - auc: 0.8654 - precision: 0.5032 - recall: 0.0184 - val_loss: 3.8548 - val_top_k_categorical_accuracy: 0.2979 - val_auc: 0.6512 - val_precision: 0.2308 - val_recall: 0.0100 - lr: 2.0000e-04
Epoch 9/30
133/133 [==============================] - 8s 59ms/step - loss: 1.8322 - top_k_categorical_accuracy: 0.6704 - auc: 0.9003 - precision: 0.5526 - recall: 0.0242 - val_loss: 3.9738 - val_top_k_categorical_accuracy: 0.2972 - val_auc: 0.6630 - val_precision: 0.3333 - val_recall: 6.6934e-04 - lr: 4.0000e-05
Epoch 10/30
133/133 [==============================] - 8s 59ms/step - loss: 1.6351 - top_k_categorical_accuracy: 0.7643 - auc: 0.9261 - precision: 0.5908 - recall: 0.0473 - val_loss: 4.4714 - val_top_k_categorical_accuracy: 0.2945 - val_auc: 0.6523 - val_precision: 0.1944 - val_recall: 0.0047 - lr: 4.0000e-05
Epoch 11/30
133/133 [==============================] - 8s 59ms/step - loss: 1.4918 - top_k_categorical_accuracy: 0.8257 - auc: 0.9418 - precision: 0.6353 - recall: 0.0861 - val_loss: 4.5518 - val_top_k_categorical_accuracy: 0.3153 - val_auc: 0.6513 - val_precision: 0.1930 - val_recall: 0.0074 - lr: 8.0000e-06
Epoch 12/30
133/133 [==============================] - 8s 58ms/step - loss: 1.4434 - top_k_categorical_accuracy: 0.8461 - auc: 0.9467 - precision: 0.6641 - recall: 0.1024 - val_loss: 4.6866 - val_top_k_categorical_accuracy: 0.3099 - val_auc: 0.6517 - val_precision: 0.1944 - val_recall: 0.0094 - lr: 8.0000e-06
Epoch 13/30
133/133 [==============================] - 8s 58ms/step - loss: 1.4034 - top_k_categorical_accuracy: 0.8583 - auc: 0.9503 - precision: 0.6641 - recall: 0.1234 - val_loss: 4.8320 - val_top_k_categorical_accuracy: 0.3186 - val_auc: 0.6484 - val_precision: 0.1682 - val_recall: 0.0120 - lr: 8.0000e-06
Epoch 14/30
133/133 [==============================] - 8s 59ms/step - loss: 1.3726 - top_k_categorical_accuracy: 0.8743 - auc: 0.9530 - precision: 0.6652 - recall: 0.1416 - val_loss: 4.9853 - val_top_k_categorical_accuracy: 0.3166 - val_auc: 0.6464 - val_precision: 0.1679 - val_recall: 0.0147 - lr: 8.0000e-06
Epoch 15/30
133/133 [==============================] - 8s 59ms/step - loss: 1.3389 - top_k_categorical_accuracy: 0.8800 - auc: 0.9556 - precision: 0.6668 - recall: 0.1587 - val_loss: 5.1136 - val_top_k_categorical_accuracy: 0.3220 - val_auc: 0.6452 - val_precision: 0.1491 - val_recall: 0.0161 - lr: 8.0000e-06
Epoch 16/30
133/133 [==============================] - 8s 59ms/step - loss: 1.3102 - top_k_categorical_accuracy: 0.8909 - auc: 0.9581 - precision: 0.6685 - recall: 0.1819 - val_loss: 5.2552 - val_top_k_categorical_accuracy: 0.3246 - val_auc: 0.6442 - val_precision: 0.1717 - val_recall: 0.0228 - lr: 8.0000e-06
Epoch 17/30
133/133 [==============================] - 8s 60ms/step - loss: 1.2734 - top_k_categorical_accuracy: 0.9048 - auc: 0.9609 - precision: 0.6883 - recall: 0.2078 - val_loss: 5.4066 - val_top_k_categorical_accuracy: 0.3226 - val_auc: 0.6419 - val_precision: 0.1799 - val_recall: 0.0288 - lr: 8.0000e-06
Epoch 18/30
133/133 [==============================] - 8s 59ms/step - loss: 1.2415 - top_k_categorical_accuracy: 0.9092 - auc: 0.9633 - precision: 0.6850 - recall: 0.2247 - val_loss: 5.5507 - val_top_k_categorical_accuracy: 0.3213 - val_auc: 0.6390 - val_precision: 0.1707 - val_recall: 0.0328 - lr: 8.0000e-06
Epoch 19/30
133/133 [==============================] - 8s 59ms/step - loss: 1.2207 - top_k_categorical_accuracy: 0.9155 - auc: 0.9650 - precision: 0.7097 - recall: 0.2482 - val_loss: 5.5749 - val_top_k_categorical_accuracy: 0.3186 - val_auc: 0.6396 - val_precision: 0.1736 - val_recall: 0.0335 - lr: 1.6000e-06
Epoch 20/30
133/133 [==============================] - 8s 62ms/step - loss: 1.2110 - top_k_categorical_accuracy: 0.9186 - auc: 0.9657 - precision: 0.7066 - recall: 0.2491 - val_loss: 5.6133 - val_top_k_categorical_accuracy: 0.3173 - val_auc: 0.6400 - val_precision: 0.1650 - val_recall: 0.0328 - lr: 1.6000e-06
Epoch 21/30
133/133 [==============================] - 8s 59ms/step - loss: 1.2038 - top_k_categorical_accuracy: 0.9181 - auc: 0.9663 - precision: 0.7131 - recall: 0.2595 - val_loss: 5.6321 - val_top_k_categorical_accuracy: 0.3173 - val_auc: 0.6397 - val_precision: 0.1577 - val_recall: 0.0315 - lr: 1.0000e-06
Epoch 22/30
133/133 [==============================] - 8s 59ms/step - loss: 1.1998 - top_k_categorical_accuracy: 0.9229 - auc: 0.9666 - precision: 0.6866 - recall: 0.2538 - val_loss: 5.6516 - val_top_k_categorical_accuracy: 0.3193 - val_auc: 0.6403 - val_precision: 0.1589 - val_recall: 0.0321 - lr: 1.0000e-06
Epoch 23/30
133/133 [==============================] - 8s 60ms/step - loss: 1.1948 - top_k_categorical_accuracy: 0.9206 - auc: 0.9668 - precision: 0.7076 - recall: 0.2606 - val_loss: 5.6690 - val_top_k_categorical_accuracy: 0.3199 - val_auc: 0.6397 - val_precision: 0.1607 - val_recall: 0.0328 - lr: 1.0000e-06
Epoch 24/30
133/133 [==============================] - 7s 51ms/step - loss: 1.1970 - top_k_categorical_accuracy: 0.9233 - auc: 0.9666 - precision: 0.6933 - recall: 0.2595 - val_loss: 5.6968 - val_top_k_categorical_accuracy: 0.3206 - val_auc: 0.6391 - val_precision: 0.1608 - val_recall: 0.0335 - lr: 1.0000e-06
Epoch 25/30
133/133 [==============================] - 8s 60ms/step - loss: 1.1919 - top_k_categorical_accuracy: 0.9240 - auc: 0.9670 - precision: 0.7035 - recall: 0.2636 - val_loss: 5.7125 - val_top_k_categorical_accuracy: 0.3199 - val_auc: 0.6398 - val_precision: 0.1546 - val_recall: 0.0328 - lr: 1.0000e-06
Epoch 26/30
133/133 [==============================] - 8s 59ms/step - loss: 1.1896 - top_k_categorical_accuracy: 0.9227 - auc: 0.9671 - precision: 0.6973 - recall: 0.2661 - val_loss: 5.7305 - val_top_k_categorical_accuracy: 0.3206 - val_auc: 0.6388 - val_precision: 0.1567 - val_recall: 0.0335 - lr: 1.0000e-06
Epoch 27/30
133/133 [==============================] - 8s 59ms/step - loss: 1.1820 - top_k_categorical_accuracy: 0.9234 - auc: 0.9677 - precision: 0.7033 - recall: 0.2690 - val_loss: 5.7487 - val_top_k_categorical_accuracy: 0.3226 - val_auc: 0.6381 - val_precision: 0.1610 - val_recall: 0.0348 - lr: 1.0000e-06
Epoch 28/30
133/133 [==============================] - 8s 59ms/step - loss: 1.1780 - top_k_categorical_accuracy: 0.9251 - auc: 0.9680 - precision: 0.7105 - recall: 0.2701 - val_loss: 5.7676 - val_top_k_categorical_accuracy: 0.3193 - val_auc: 0.6391 - val_precision: 0.1560 - val_recall: 0.0341 - lr: 1.0000e-06
Epoch 29/30
133/133 [==============================] - 8s 58ms/step - loss: 1.1737 - top_k_categorical_accuracy: 0.9275 - auc: 0.9684 - precision: 0.7024 - recall: 0.2765 - val_loss: 5.7901 - val_top_k_categorical_accuracy: 0.3193 - val_auc: 0.6391 - val_precision: 0.1518 - val_recall: 0.0341 - lr: 1.0000e-06
Epoch 30/30
133/133 [==============================] - 8s 58ms/step - loss: 1.1725 - top_k_categorical_accuracy: 0.9252 - auc: 0.9685 - precision: 0.7119 - recall: 0.2780 - val_loss: 5.8028 - val_top_k_categorical_accuracy: 0.3199 - val_auc: 0.6382 - val_precision: 0.1450 - val_recall: 0.0328 - lr: 1.0000e-06

```
</div>
---
## Visualize model metrics during training

We use the function defined above to see model metrics during training.


```python
plot_history_metrics(conv_model_history)
```


    
![png](/img/examples/timeseries/eeg_signal_classification/eeg_signal_48_0.png)
    


---
## Evaluate model on test data


```python
loss, accuracy, auc, precision, recall = conv_model.evaluate(test_dataset)
print(f"Loss : {loss}")
print(f"Top 3 Categorical Accuracy : {accuracy}")
print(f"Area under the Curve (ROC) : {auc}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")


def view_evaluated_eeg_plots(model):
    start_index = random.randint(10, len(eeg))
    end_index = start_index + 11
    data = eeg.loc[start_index:end_index, "raw_values"]
    data_array = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in data]
    data_array = [np.asarray(data_array).astype(np.float32).reshape(-1, 512, 1)]
    original_labels = eeg.loc[start_index:end_index, "label"]
    predicted_labels = np.argmax(model.predict(data_array, verbose=0), axis=1)
    original_labels = [
        le.inverse_transform(np.array(label).reshape(-1))[0]
        for label in original_labels
    ]
    predicted_labels = [
        le.inverse_transform(np.array(label).reshape(-1))[0]
        for label in predicted_labels
    ]
    total_plots = 12
    cols = total_plots // 3
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1
    pos = range(1, total_plots + 1)
    fig = plt.figure(figsize=(20, 10))
    for i, (plot_data, og_label, pred_label) in enumerate(
        zip(data, original_labels, predicted_labels)
    ):
        plt.subplot(rows, cols, pos[i])
        plt.plot(plot_data)
        plt.title(f"Actual Label : {og_label}\nPredicted Label : {pred_label}")
        fig.subplots_adjust(hspace=0.5)
    plt.show()


view_evaluated_eeg_plots(conv_model)
```

<div class="k-default-codeblock">
```
24/24 [==============================] - 0s 9ms/step - loss: 5.8028 - top_k_categorical_accuracy: 0.3199 - auc: 0.6382 - precision: 0.1450 - recall: 0.0328
Loss : 5.802786827087402
Top 3 Categorical Accuracy : 0.31994643807411194
Area under the Curve (ROC) : 0.6381803750991821
Precision : 0.14497041702270508
Recall : 0.032797858119010925

```
</div>
    
![png](/img/examples/timeseries/eeg_signal_classification/eeg_signal_50_1.png)
    

