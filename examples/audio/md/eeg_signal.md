# Electroencephalogram Signal Classification for action identification

**Author:** [Suvaditya Mukherjee](https://github.com/suvadityamuk)<br>
**Date created:** 2022/11/03<br>
**Last modified:** 2022/11/05<br>
**Description:** Training a Convolutional model to classify EEG signals produced by exposure to certain stimuli.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/audio/ipynb/eeg_signal.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/audio/eeg_signal.py)



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
!pip install gdown
!pip install sklearn
!pip install pandas
!pip install numpy
!pip install matplotlib
```

<div class="k-default-codeblock">
```
Requirement already satisfied: gdown in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (4.5.3)
Requirement already satisfied: six in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from gdown) (1.16.0)
Requirement already satisfied: filelock in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from gdown) (3.8.0)
Requirement already satisfied: beautifulsoup4 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from gdown) (4.11.1)
Requirement already satisfied: requests[socks] in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from gdown) (2.27.1)
Requirement already satisfied: tqdm in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from gdown) (4.64.0)
Requirement already satisfied: soupsieve>1.2 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from beautifulsoup4->gdown) (2.3.2.post1)
Requirement already satisfied: certifi>=2017.4.17 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from requests[socks]->gdown) (2022.5.18.1)
Requirement already satisfied: idna<4,>=2.5 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from requests[socks]->gdown) (3.3)
Requirement already satisfied: charset-normalizer~=2.0.0 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from requests[socks]->gdown) (2.0.12)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from requests[socks]->gdown) (1.26.9)
Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from requests[socks]->gdown) (1.7.1)
Requirement already satisfied: sklearn in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (0.0)
Requirement already satisfied: scikit-learn in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from sklearn) (1.1.2)
Requirement already satisfied: scipy>=1.3.2 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from scikit-learn->sklearn) (1.9.0)
Requirement already satisfied: numpy>=1.17.3 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from scikit-learn->sklearn) (1.23.2)
Requirement already satisfied: threadpoolctl>=2.0.0 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from scikit-learn->sklearn) (3.1.0)
Requirement already satisfied: joblib>=1.0.0 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from scikit-learn->sklearn) (1.1.0)
Requirement already satisfied: pandas in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (1.4.3)
Requirement already satisfied: pytz>=2020.1 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from pandas) (2022.1)
Requirement already satisfied: numpy>=1.18.5 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from pandas) (1.23.2)
Requirement already satisfied: python-dateutil>=2.8.1 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from pandas) (2.8.2)
Requirement already satisfied: six>=1.5 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)
Requirement already satisfied: numpy in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (1.23.2)
Requirement already satisfied: matplotlib in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (3.5.2)
Requirement already satisfied: python-dateutil>=2.7 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (2.8.2)
Requirement already satisfied: kiwisolver>=1.0.1 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (1.4.3)
Requirement already satisfied: pillow>=6.2.0 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (9.1.1)
Requirement already satisfied: pyparsing>=2.2.1 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (3.0.9)
Requirement already satisfied: fonttools>=4.22.0 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (4.33.3)
Requirement already satisfied: packaging>=20.0 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (21.3)
Requirement already satisfied: cycler>=0.10 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (0.11.0)
Requirement already satisfied: numpy>=1.17 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from matplotlib) (1.23.2)
Requirement already satisfied: six>=1.5 in /home/suvaditya/Programming/personal/oss/oss-ml/lib/python3.8/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)

```
</div>
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
To: /home/suvaditya/Programming/personal/oss/keras-io/scripts/tmp_5852548/eeg-data.csv
100%|████████████████████████████████████████| 106M/106M [00:06<00:00, 17.3MB/s]

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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<div class="k-default-codeblock">
```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</div>
</style>
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
</div>



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
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<div class="k-default-codeblock">
```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</div>
</style>
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
</div>



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


    
![png](/img/examples/audio/eeg_signal/eeg_signal_15_0.png)
    


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


    
![png](/img/examples/audio/eeg_signal/eeg_signal_22_0.png)
    


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
133/133 [==============================] - 13s 63ms/step - loss: 13.7334 - top_k_categorical_accuracy: 0.2678 - auc: 0.6498 - precision: 0.1259 - recall: 0.0020 - val_loss: 4.1816 - val_top_k_categorical_accuracy: 0.2610 - val_auc: 0.6638 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 0.0010
Epoch 2/30
133/133 [==============================] - 8s 58ms/step - loss: 3.3075 - top_k_categorical_accuracy: 0.3199 - auc: 0.6986 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 3.2463 - val_top_k_categorical_accuracy: 0.2329 - val_auc: 0.6395 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 0.0010
Epoch 3/30
133/133 [==============================] - 8s 58ms/step - loss: 2.7907 - top_k_categorical_accuracy: 0.3383 - auc: 0.7165 - precision: 0.2000 - recall: 1.1820e-04 - val_loss: 3.0565 - val_top_k_categorical_accuracy: 0.2510 - val_auc: 0.6351 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 0.0010
Epoch 4/30
133/133 [==============================] - 8s 57ms/step - loss: 2.5643 - top_k_categorical_accuracy: 0.3846 - auc: 0.7574 - precision: 0.7143 - recall: 0.0012 - val_loss: 3.0693 - val_top_k_categorical_accuracy: 0.2791 - val_auc: 0.6383 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - lr: 2.0000e-04
Epoch 5/30
133/133 [==============================] - 8s 58ms/step - loss: 2.3815 - top_k_categorical_accuracy: 0.4480 - auc: 0.8014 - precision: 0.6000 - recall: 0.0050 - val_loss: 3.1906 - val_top_k_categorical_accuracy: 0.3166 - val_auc: 0.6714 - val_precision: 0.2500 - val_recall: 6.6934e-04 - lr: 2.0000e-04
Epoch 6/30
133/133 [==============================] - 8s 58ms/step - loss: 2.1999 - top_k_categorical_accuracy: 0.5260 - auc: 0.8424 - precision: 0.6138 - recall: 0.0242 - val_loss: 3.4796 - val_top_k_categorical_accuracy: 0.2932 - val_auc: 0.6534 - val_precision: 0.1795 - val_recall: 0.0047 - lr: 2.0000e-04
Epoch 7/30
133/133 [==============================] - 8s 59ms/step - loss: 2.0204 - top_k_categorical_accuracy: 0.6294 - auc: 0.8766 - precision: 0.6336 - recall: 0.0695 - val_loss: 3.7237 - val_top_k_categorical_accuracy: 0.3092 - val_auc: 0.6488 - val_precision: 0.2143 - val_recall: 0.0161 - lr: 2.0000e-04
Epoch 8/30
133/133 [==============================] - 8s 58ms/step - loss: 1.7604 - top_k_categorical_accuracy: 0.7287 - auc: 0.9148 - precision: 0.7522 - recall: 0.1327 - val_loss: 4.1081 - val_top_k_categorical_accuracy: 0.3199 - val_auc: 0.6592 - val_precision: 0.2000 - val_recall: 0.0341 - lr: 4.0000e-05
Epoch 9/30
133/133 [==============================] - 8s 58ms/step - loss: 1.4884 - top_k_categorical_accuracy: 0.8099 - auc: 0.9450 - precision: 0.7830 - recall: 0.2572 - val_loss: 4.6625 - val_top_k_categorical_accuracy: 0.3273 - val_auc: 0.6527 - val_precision: 0.2061 - val_recall: 0.0589 - lr: 4.0000e-05
Epoch 10/30
133/133 [==============================] - 8s 58ms/step - loss: 1.2871 - top_k_categorical_accuracy: 0.8560 - auc: 0.9620 - precision: 0.8197 - recall: 0.3837 - val_loss: 5.2381 - val_top_k_categorical_accuracy: 0.3307 - val_auc: 0.6487 - val_precision: 0.1834 - val_recall: 0.0723 - lr: 4.0000e-05
Epoch 11/30
133/133 [==============================] - 8s 58ms/step - loss: 1.1167 - top_k_categorical_accuracy: 0.8909 - auc: 0.9740 - precision: 0.8458 - recall: 0.4959 - val_loss: 5.7839 - val_top_k_categorical_accuracy: 0.3333 - val_auc: 0.6414 - val_precision: 0.1778 - val_recall: 0.0857 - lr: 4.0000e-05
Epoch 12/30
133/133 [==============================] - 8s 59ms/step - loss: 0.9559 - top_k_categorical_accuracy: 0.9226 - auc: 0.9831 - precision: 0.8802 - recall: 0.6002 - val_loss: 6.4426 - val_top_k_categorical_accuracy: 0.3280 - val_auc: 0.6380 - val_precision: 0.1824 - val_recall: 0.1071 - lr: 4.0000e-05
Epoch 13/30
133/133 [==============================] - 8s 59ms/step - loss: 0.8390 - top_k_categorical_accuracy: 0.9447 - auc: 0.9885 - precision: 0.8976 - recall: 0.6726 - val_loss: 7.0075 - val_top_k_categorical_accuracy: 0.3226 - val_auc: 0.6295 - val_precision: 0.1764 - val_recall: 0.1151 - lr: 4.0000e-05
Epoch 14/30
133/133 [==============================] - 8s 59ms/step - loss: 0.7354 - top_k_categorical_accuracy: 0.9635 - auc: 0.9926 - precision: 0.9145 - recall: 0.7219 - val_loss: 7.0575 - val_top_k_categorical_accuracy: 0.3226 - val_auc: 0.6327 - val_precision: 0.1898 - val_recall: 0.1225 - lr: 8.0000e-06
Epoch 15/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6792 - top_k_categorical_accuracy: 0.9713 - auc: 0.9946 - precision: 0.9336 - recall: 0.7558 - val_loss: 7.1773 - val_top_k_categorical_accuracy: 0.3266 - val_auc: 0.6329 - val_precision: 0.1919 - val_recall: 0.1265 - lr: 8.0000e-06
Epoch 16/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6428 - top_k_categorical_accuracy: 0.9743 - auc: 0.9957 - precision: 0.9456 - recall: 0.7749 - val_loss: 7.1511 - val_top_k_categorical_accuracy: 0.3273 - val_auc: 0.6343 - val_precision: 0.1938 - val_recall: 0.1258 - lr: 1.6000e-06
Epoch 17/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6380 - top_k_categorical_accuracy: 0.9766 - auc: 0.9958 - precision: 0.9490 - recall: 0.7786 - val_loss: 7.1815 - val_top_k_categorical_accuracy: 0.3266 - val_auc: 0.6345 - val_precision: 0.1925 - val_recall: 0.1272 - lr: 1.6000e-06
Epoch 18/30
133/133 [==============================] - 8s 58ms/step - loss: 0.6313 - top_k_categorical_accuracy: 0.9766 - auc: 0.9959 - precision: 0.9499 - recall: 0.7845 - val_loss: 7.1852 - val_top_k_categorical_accuracy: 0.3286 - val_auc: 0.6338 - val_precision: 0.1884 - val_recall: 0.1238 - lr: 1.0000e-06
Epoch 19/30
133/133 [==============================] - 8s 58ms/step - loss: 0.6276 - top_k_categorical_accuracy: 0.9765 - auc: 0.9960 - precision: 0.9499 - recall: 0.7851 - val_loss: 7.2069 - val_top_k_categorical_accuracy: 0.3300 - val_auc: 0.6344 - val_precision: 0.1886 - val_recall: 0.1245 - lr: 1.0000e-06
Epoch 20/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6243 - top_k_categorical_accuracy: 0.9758 - auc: 0.9961 - precision: 0.9515 - recall: 0.7865 - val_loss: 7.2225 - val_top_k_categorical_accuracy: 0.3286 - val_auc: 0.6341 - val_precision: 0.1874 - val_recall: 0.1232 - lr: 1.0000e-06
Epoch 21/30
133/133 [==============================] - 8s 58ms/step - loss: 0.6236 - top_k_categorical_accuracy: 0.9770 - auc: 0.9961 - precision: 0.9492 - recall: 0.7861 - val_loss: 7.2446 - val_top_k_categorical_accuracy: 0.3286 - val_auc: 0.6339 - val_precision: 0.1891 - val_recall: 0.1252 - lr: 1.0000e-06
Epoch 22/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6174 - top_k_categorical_accuracy: 0.9772 - auc: 0.9963 - precision: 0.9493 - recall: 0.7878 - val_loss: 7.2727 - val_top_k_categorical_accuracy: 0.3273 - val_auc: 0.6337 - val_precision: 0.1899 - val_recall: 0.1265 - lr: 1.0000e-06
Epoch 23/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6129 - top_k_categorical_accuracy: 0.9786 - auc: 0.9964 - precision: 0.9523 - recall: 0.7866 - val_loss: 7.2914 - val_top_k_categorical_accuracy: 0.3280 - val_auc: 0.6334 - val_precision: 0.1892 - val_recall: 0.1265 - lr: 1.0000e-06
Epoch 24/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6074 - top_k_categorical_accuracy: 0.9794 - auc: 0.9965 - precision: 0.9580 - recall: 0.8000 - val_loss: 7.3214 - val_top_k_categorical_accuracy: 0.3293 - val_auc: 0.6335 - val_precision: 0.1900 - val_recall: 0.1272 - lr: 1.0000e-06
Epoch 25/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6064 - top_k_categorical_accuracy: 0.9784 - auc: 0.9965 - precision: 0.9538 - recall: 0.7929 - val_loss: 7.3337 - val_top_k_categorical_accuracy: 0.3280 - val_auc: 0.6333 - val_precision: 0.1882 - val_recall: 0.1258 - lr: 1.0000e-06
Epoch 26/30
133/133 [==============================] - 8s 59ms/step - loss: 0.6006 - top_k_categorical_accuracy: 0.9794 - auc: 0.9967 - precision: 0.9554 - recall: 0.8002 - val_loss: 7.3597 - val_top_k_categorical_accuracy: 0.3280 - val_auc: 0.6323 - val_precision: 0.1890 - val_recall: 0.1265 - lr: 1.0000e-06
Epoch 27/30
133/133 [==============================] - 8s 58ms/step - loss: 0.5970 - top_k_categorical_accuracy: 0.9790 - auc: 0.9966 - precision: 0.9582 - recall: 0.8051 - val_loss: 7.3698 - val_top_k_categorical_accuracy: 0.3280 - val_auc: 0.6317 - val_precision: 0.1874 - val_recall: 0.1252 - lr: 1.0000e-06
Epoch 28/30
133/133 [==============================] - 8s 59ms/step - loss: 0.5937 - top_k_categorical_accuracy: 0.9793 - auc: 0.9968 - precision: 0.9601 - recall: 0.8051 - val_loss: 7.3871 - val_top_k_categorical_accuracy: 0.3300 - val_auc: 0.6314 - val_precision: 0.1873 - val_recall: 0.1258 - lr: 1.0000e-06
Epoch 29/30
133/133 [==============================] - 8s 59ms/step - loss: 0.5920 - top_k_categorical_accuracy: 0.9805 - auc: 0.9968 - precision: 0.9585 - recall: 0.8061 - val_loss: 7.4167 - val_top_k_categorical_accuracy: 0.3293 - val_auc: 0.6318 - val_precision: 0.1865 - val_recall: 0.1258 - lr: 1.0000e-06
Epoch 30/30
133/133 [==============================] - 8s 59ms/step - loss: 0.5898 - top_k_categorical_accuracy: 0.9805 - auc: 0.9969 - precision: 0.9597 - recall: 0.8099 - val_loss: 7.4371 - val_top_k_categorical_accuracy: 0.3286 - val_auc: 0.6319 - val_precision: 0.1858 - val_recall: 0.1258 - lr: 1.0000e-06

```
</div>
---
## Visualize model metrics during training

We use the function defined above to see model metrics during training.


```python
plot_history_metrics(conv_model_history)
```


    
![png](/img/examples/audio/eeg_signal/eeg_signal_48_0.png)
    


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
24/24 [==============================] - 0s 9ms/step - loss: 7.4371 - top_k_categorical_accuracy: 0.3286 - auc: 0.6319 - precision: 0.1858 - recall: 0.1258
Loss : 7.437110424041748
Top 3 Categorical Accuracy : 0.3286479115486145
Area under the Curve (ROC) : 0.6319202780723572
Precision : 0.18577075004577637
Recall : 0.12583668529987335

```
</div>
    
![png](/img/examples/audio/eeg_signal/eeg_signal_50_1.png)
    

