# Audio Classification with the STFTSpectrogram layer

**Author:** [Mostafa M. Amin](https://mostafa-amin.com)<br>
**Date created:** 2024/10/04<br>
**Last modified:** 2024/10/04<br>
**Description:** Introducing the `STFTSpectrogram` layer to extract spectrograms for audio classification.


<div class='example_version_banner keras_3'>ⓘ This example uses Keras 3</div>
<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/audio/ipynb/stft.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/audio/stft.py)



---
## Introduction

Preprocessing audio as spectrograms is an essential step in the vast majority
of audio-based applications. Spectrograms represent the frequency content of a
signal over time, are widely used for this purpose. In this tutorial, we'll
demonstrate how to use the `STFTSpectrogram` layer in Keras to convert raw
audio waveforms into spectrograms **within the model**. We'll then feed
these spectrograms into an LSTM network followed by Dense layers to perform
audio classification on the Speech Commands dataset.

We will:

- Load the ESC-10 dataset.
- Preprocess the raw audio waveforms and generate spectrograms using
   `STFTSpectrogram`.
- Build two models, one using spectrograms as 1D signals and the other is using
   as images (2D signals) with a pretrained image model.
- Train and evaluate the models.

---
## Setup

### Importing the necessary libraries


```python
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile
import tensorflow as tf
from keras import layers
from scipy.signal import resample

np.random.seed(41)
tf.random.set_seed(41)
```

### Define some variables


```python
BASE_DATA_DIR = "./datasets/esc-50_extracted/ESC-50-master/"
BATCH_SIZE = 16
NUM_CLASSES = 10
EPOCHS = 200
SAMPLE_RATE = 16000
```

---
## Download and Preprocess the ESC-10 Dataset

We'll use the Dataset for Environmental Sound Classification dataset (ESC-10).
This dataset consists of five-second .wav files of environmental sounds.

### Download and Extract the dataset


```python
keras.utils.get_file(
    "esc-50.zip",
    "https://github.com/karoldvl/ESC-50/archive/master.zip",
    cache_dir="./",
    cache_subdir="datasets",
    extract=True,
)
```




    './datasets/esc-50_extracted'



### Read the CSV file


```python
pd_data = pd.read_csv(os.path.join(BASE_DATA_DIR, "meta", "esc50.csv"))
# filter ESC-50 to ESC-10 and reassign the targets
pd_data = pd_data[pd_data["esc10"]]
targets = sorted(pd_data["target"].unique().tolist())
assert len(targets) == NUM_CLASSES
old_target_to_new_target = {old: new for new, old in enumerate(targets)}
pd_data["target"] = pd_data["target"].map(lambda t: old_target_to_new_target[t])
pd_data
```





  <div id="df-28fa47d3-aab6-44fb-a592-a8baf16d5123" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>fold</th>
      <th>target</th>
      <th>category</th>
      <th>esc10</th>
      <th>src_file</th>
      <th>take</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1-100032-A-0.wav</td>
      <td>1</td>
      <td>0</td>
      <td>dog</td>
      <td>True</td>
      <td>100032</td>
      <td>A</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1-110389-A-0.wav</td>
      <td>1</td>
      <td>0</td>
      <td>dog</td>
      <td>True</td>
      <td>110389</td>
      <td>A</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1-116765-A-41.wav</td>
      <td>1</td>
      <td>9</td>
      <td>chainsaw</td>
      <td>True</td>
      <td>116765</td>
      <td>A</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1-17150-A-12.wav</td>
      <td>1</td>
      <td>4</td>
      <td>crackling_fire</td>
      <td>True</td>
      <td>17150</td>
      <td>A</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1-172649-A-40.wav</td>
      <td>1</td>
      <td>8</td>
      <td>helicopter</td>
      <td>True</td>
      <td>172649</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1876</th>
      <td>5-233160-A-1.wav</td>
      <td>5</td>
      <td>1</td>
      <td>rooster</td>
      <td>True</td>
      <td>233160</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1888</th>
      <td>5-234879-A-1.wav</td>
      <td>5</td>
      <td>1</td>
      <td>rooster</td>
      <td>True</td>
      <td>234879</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1889</th>
      <td>5-234879-B-1.wav</td>
      <td>5</td>
      <td>1</td>
      <td>rooster</td>
      <td>True</td>
      <td>234879</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>5-235671-A-38.wav</td>
      <td>5</td>
      <td>7</td>
      <td>clock_tick</td>
      <td>True</td>
      <td>235671</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>5-9032-A-0.wav</td>
      <td>5</td>
      <td>0</td>
      <td>dog</td>
      <td>True</td>
      <td>9032</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 7 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-28fa47d3-aab6-44fb-a592-a8baf16d5123')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-28fa47d3-aab6-44fb-a592-a8baf16d5123 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-28fa47d3-aab6-44fb-a592-a8baf16d5123');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-bafc7d67-95d3-48c8-8a4a-1219f83d582e">
  <button class="colab-df-quickchart" onclick="quickchart('df-bafc7d67-95d3-48c8-8a4a-1219f83d582e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-bafc7d67-95d3-48c8-8a4a-1219f83d582e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_d26ec336-5e58-46ec-8a2a-33fb10cd44b4">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('pd_data')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d26ec336-5e58-46ec-8a2a-33fb10cd44b4 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('pd_data');
      }
      })();
    </script>
  </div>

    </div>
  </div>




### Define functions to read and preprocess the WAV files


```python
def read_wav_file(path, target_sr=SAMPLE_RATE):
    sr, wav = scipy.io.wavfile.read(os.path.join(BASE_DATA_DIR, "audio", path))
    wav = wav.astype(np.float32) / 32768.0  # normalize to [-1, 1]
    num_samples = int(len(wav) * target_sr / sr)  # resample to 16 kHz
    wav = resample(wav, num_samples)
    return wav[:, None]  # Add a channel dimension (of size 1)
```

Create a function that uses the `STFTSpectrogram` to compute a spectrogram,
then plots it.


```python
def plot_single_spectrogram(sample_wav_data):
    spectrogram = layers.STFTSpectrogram(
        mode="log",
        frame_length=SAMPLE_RATE * 20 // 1000,
        frame_step=SAMPLE_RATE * 5 // 1000,
        fft_length=1024,
        trainable=False,
    )(sample_wav_data[None, ...]).numpy()[0]

    # Plot the spectrogram
    plt.imshow(spectrogram.T, origin="lower")
    plt.title("Single Channel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()
```

Create a function that uses the `STFTSpectrogram` to compute three
spectrograms with multiple bandwidths, then aligns them as an image
with different channels, to get a multi-bandwith spectrogram, then plots the spectrogram.


```python
def plot_multi_bandwidth_spectrogram(sample_wav_data):
    # All spectrograms must use the same `fft_length`, `frame_step`, and
    # `padding="same"` in order to produce spectrograms with identical shapes,
    # hence aligning them together. `expand_dims` ensures that the shapes are
    # compatible with image models.

    spectrograms = np.concatenate(
        [
            layers.STFTSpectrogram(
                mode="log",
                frame_length=SAMPLE_RATE * x // 1000,
                frame_step=SAMPLE_RATE * 5 // 1000,
                fft_length=1024,
                padding="same",
                expand_dims=True,
            )(sample_wav_data[None, ...]).numpy()[0, ...]
            for x in [5, 10, 20]
        ],
        axis=-1,
    ).transpose([1, 0, 2])

    # normalize each color channel for better viewing
    mn = spectrograms.min(axis=(0, 1), keepdims=True)
    mx = spectrograms.max(axis=(0, 1), keepdims=True)
    spectrograms = (spectrograms - mn) / (mx - mn)

    plt.imshow(spectrograms, origin="lower")
    plt.title("Multi-bandwidth Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()
```

Demonstrate a sample wav file.


```python
sample_wav_data = read_wav_file(pd_data["filename"].tolist()[52])
plt.plot(sample_wav_data[:, 0])
plt.show()
```


    
![png](https://github.com/mostafa-mahmoud/keras-io/blob/master/examples/audio/img/stft/raw_audio.png)
    


Plot a Spectrogram


```python
plot_single_spectrogram(sample_wav_data)
```


    
![png](https://github.com/mostafa-mahmoud/keras-io/blob/master/examples/audio/img/stft/spectrogram.png)
    


Plot a multi-bandwidth spectrogram


```python
plot_multi_bandwidth_spectrogram(sample_wav_data)
```


    
![png](https://github.com/mostafa-mahmoud/keras-io/blob/master/examples/audio/img/stft/multiband_spectrogram.png)
    


### Define functions to construct a TF Dataset


```python
def read_dataset(df, folds, batch_size, shuffle=True):
    msk = df["fold"].isin(folds)
    filenames = df["filename"][msk]
    targets = df["target"][msk].values
    waves = np.array(
        [read_wav_file(fil) for fil in filenames], dtype=np.float32
    )
    ds = tf.data.Dataset.from_tensor_slices((waves, targets))
    if shuffle:
        ds = ds.shuffle(1024, seed=41)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
```

---
## Training the Models

In this tutorial we demonstrate the different usecases of the `STFTSpectrogram`
layer.

The first model will use a non-trainable `STFTSpectrogram` layer, so it is
intended purely for preprocessing. Additionally, the model will use 1D signals,
hence it make use of Conv1D layers.

The second model will use a trainable `STFTSpectrogram` layer with the
`expand_dims` option, which expands the shapes to be compatible with image
models.

### Create the 1D model

1. Create a non-trainable spectrograms, extracting a 1D time signal.
2. Apply `Conv1D` layers with `LayerNormalization` simialar to the
   classic VGG design.
4. Apply global maximum pooling to have fixed set of features.
5. Add `Dense` layers to make the final predictions based on the features.


```python
model1d = keras.Sequential(
    [
        layers.InputLayer((None, 1)),
        layers.STFTSpectrogram(
            mode="log",
            frame_length=SAMPLE_RATE * 40 // 1000,
            frame_step=SAMPLE_RATE * 15 // 1000,
            trainable=False,
        ),
        layers.Conv1D(64, 64, activation="relu"),
        layers.Conv1D(128, 16, activation="relu"),
        layers.LayerNormalization(),
        layers.MaxPooling1D(4),
        layers.Conv1D(128, 8, activation="relu"),
        layers.Conv1D(256, 8, activation="relu"),
        layers.Conv1D(512, 4, activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(0.5),
        layers.GlobalMaxPooling1D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ],
    name="model_1d_non_trainble_stft",
)
model1d.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model1d.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_1d_non_trainble_stft"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ stft_spectrogram_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">STFTSpectrogram</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">513</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">656,640</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">2,101,312</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">131,200</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ layer_normalization                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">131,200</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">524,800</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ layer_normalization_1                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │           <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">LayerNormalization</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_max_pooling1d                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalMaxPooling1D</span>)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">131,328</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │          <span style="color: #00af00; text-decoration-color: #00af00">65,792</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">2,570</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,008,522</span> (15.29 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,351,882</span> (12.79 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">656,640</span> (2.50 MB)
</pre>



Create the datasets.


```python
train_ds = read_dataset(pd_data, [1, 2, 3], BATCH_SIZE)
valid_ds = read_dataset(pd_data, [4], BATCH_SIZE)
```

Train the model and restore the best weights.


```python
history_model1d = model1d.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EPOCHS,
            restore_best_weights=True,
        )
    ],
)
```

    Epoch 1/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 156ms/step - accuracy: 0.1245 - loss: 4.2444 - val_accuracy: 0.1000 - val_loss: 2.4278
    Epoch 2/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.0806 - loss: 2.7835 - val_accuracy: 0.1000 - val_loss: 2.3543
    Epoch 3/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.0888 - loss: 2.6195 - val_accuracy: 0.1000 - val_loss: 2.3541
    Epoch 4/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.1115 - loss: 2.5092 - val_accuracy: 0.1000 - val_loss: 2.3471
    Epoch 5/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.1409 - loss: 2.4205 - val_accuracy: 0.1000 - val_loss: 2.3239
    Epoch 6/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.1781 - loss: 2.3260 - val_accuracy: 0.1000 - val_loss: 2.3006
    Epoch 7/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.1547 - loss: 2.3376 - val_accuracy: 0.1250 - val_loss: 2.2977
    Epoch 8/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.1800 - loss: 2.2921 - val_accuracy: 0.1625 - val_loss: 2.3067
    Epoch 9/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.1284 - loss: 2.3478 - val_accuracy: 0.1500 - val_loss: 2.2944
    Epoch 10/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.1629 - loss: 2.1633 - val_accuracy: 0.1500 - val_loss: 2.2766
    Epoch 11/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.1842 - loss: 2.3069 - val_accuracy: 0.1625 - val_loss: 2.2705
    Epoch 12/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.2572 - loss: 2.1478 - val_accuracy: 0.1625 - val_loss: 2.2422
    Epoch 13/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.2231 - loss: 2.1545 - val_accuracy: 0.1750 - val_loss: 2.2360
    Epoch 14/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.2034 - loss: 2.1187 - val_accuracy: 0.1500 - val_loss: 2.2403
    Epoch 15/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.2206 - loss: 2.0991 - val_accuracy: 0.1500 - val_loss: 2.2049
    Epoch 16/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.2611 - loss: 2.0664 - val_accuracy: 0.1750 - val_loss: 2.1872
    Epoch 17/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.3196 - loss: 1.9365 - val_accuracy: 0.2000 - val_loss: 2.1703
    Epoch 18/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.2120 - loss: 2.0215 - val_accuracy: 0.1625 - val_loss: 2.1758
    Epoch 19/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.2534 - loss: 1.9584 - val_accuracy: 0.1750 - val_loss: 2.1369
    Epoch 20/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.2796 - loss: 1.9447 - val_accuracy: 0.1750 - val_loss: 2.1204
    Epoch 21/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.3088 - loss: 1.9246 - val_accuracy: 0.1625 - val_loss: 2.1485
    Epoch 22/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.3104 - loss: 1.9013 - val_accuracy: 0.2750 - val_loss: 2.0732
    Epoch 23/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.3541 - loss: 1.7609 - val_accuracy: 0.2875 - val_loss: 2.0588
    Epoch 24/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.4227 - loss: 1.7060 - val_accuracy: 0.2250 - val_loss: 2.0552
    Epoch 25/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.3697 - loss: 1.7722 - val_accuracy: 0.2500 - val_loss: 2.0429
    Epoch 26/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.3378 - loss: 1.7445 - val_accuracy: 0.3250 - val_loss: 1.9986
    Epoch 27/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.3996 - loss: 1.6541 - val_accuracy: 0.3375 - val_loss: 1.9875
    Epoch 28/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.4262 - loss: 1.7056 - val_accuracy: 0.3500 - val_loss: 1.9778
    Epoch 29/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.3665 - loss: 1.7461 - val_accuracy: 0.3750 - val_loss: 1.9156
    Epoch 30/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.4146 - loss: 1.5785 - val_accuracy: 0.3875 - val_loss: 1.9187
    Epoch 31/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.4244 - loss: 1.5860 - val_accuracy: 0.3750 - val_loss: 1.8981
    Epoch 32/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.4807 - loss: 1.4774 - val_accuracy: 0.3750 - val_loss: 1.8986
    Epoch 33/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.4324 - loss: 1.5736 - val_accuracy: 0.4750 - val_loss: 1.8538
    Epoch 34/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.4532 - loss: 1.5245 - val_accuracy: 0.3375 - val_loss: 1.8551
    Epoch 35/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.4329 - loss: 1.6343 - val_accuracy: 0.4500 - val_loss: 1.8211
    Epoch 36/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.4115 - loss: 1.5483 - val_accuracy: 0.4625 - val_loss: 1.8416
    Epoch 37/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5484 - loss: 1.3587 - val_accuracy: 0.5125 - val_loss: 1.8110
    Epoch 38/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5446 - loss: 1.3416 - val_accuracy: 0.5125 - val_loss: 1.7468
    Epoch 39/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5121 - loss: 1.3676 - val_accuracy: 0.4500 - val_loss: 1.7626
    Epoch 40/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5061 - loss: 1.4286 - val_accuracy: 0.3875 - val_loss: 1.7960
    Epoch 41/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.5215 - loss: 1.2668 - val_accuracy: 0.5250 - val_loss: 1.7217
    Epoch 42/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.5479 - loss: 1.2947 - val_accuracy: 0.5500 - val_loss: 1.7184
    Epoch 43/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6094 - loss: 1.1735 - val_accuracy: 0.5625 - val_loss: 1.7282
    Epoch 44/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.5600 - loss: 1.2402 - val_accuracy: 0.5875 - val_loss: 1.6925
    Epoch 45/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.5832 - loss: 1.2894 - val_accuracy: 0.6375 - val_loss: 1.6058
    Epoch 46/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5657 - loss: 1.2921 - val_accuracy: 0.6500 - val_loss: 1.6680
    Epoch 47/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6184 - loss: 1.1524 - val_accuracy: 0.5625 - val_loss: 1.6323
    Epoch 48/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5814 - loss: 1.1617 - val_accuracy: 0.6000 - val_loss: 1.6096
    Epoch 49/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.5875 - loss: 1.1061 - val_accuracy: 0.6125 - val_loss: 1.5366
    Epoch 50/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5496 - loss: 1.1619 - val_accuracy: 0.5625 - val_loss: 1.5659
    Epoch 51/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6688 - loss: 1.0273 - val_accuracy: 0.5750 - val_loss: 1.5680
    Epoch 52/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.5799 - loss: 1.1538 - val_accuracy: 0.5500 - val_loss: 1.6440
    Epoch 53/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6026 - loss: 1.1530 - val_accuracy: 0.4875 - val_loss: 1.6001
    Epoch 54/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.5592 - loss: 1.3008 - val_accuracy: 0.5875 - val_loss: 1.5333
    Epoch 55/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.6584 - loss: 1.0050 - val_accuracy: 0.6875 - val_loss: 1.5033
    Epoch 56/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6798 - loss: 0.9936 - val_accuracy: 0.5625 - val_loss: 1.5382
    Epoch 57/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6642 - loss: 0.9778 - val_accuracy: 0.5625 - val_loss: 1.6258
    Epoch 58/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.5908 - loss: 1.1110 - val_accuracy: 0.5625 - val_loss: 1.4781
    Epoch 59/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.6739 - loss: 0.8819 - val_accuracy: 0.6875 - val_loss: 1.5324
    Epoch 60/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.6736 - loss: 1.0863 - val_accuracy: 0.5750 - val_loss: 1.4473
    Epoch 61/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.6527 - loss: 0.9804 - val_accuracy: 0.7125 - val_loss: 1.4108
    Epoch 62/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6857 - loss: 0.9003 - val_accuracy: 0.6125 - val_loss: 1.4286
    Epoch 63/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6316 - loss: 1.0357 - val_accuracy: 0.7000 - val_loss: 1.3977
    Epoch 64/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.6451 - loss: 0.9713 - val_accuracy: 0.6625 - val_loss: 1.3920
    Epoch 65/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6904 - loss: 0.8858 - val_accuracy: 0.6750 - val_loss: 1.4193
    Epoch 66/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6804 - loss: 0.8756 - val_accuracy: 0.6625 - val_loss: 1.4281
    Epoch 67/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.7116 - loss: 0.8166 - val_accuracy: 0.6875 - val_loss: 1.3488
    Epoch 68/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7220 - loss: 0.7609 - val_accuracy: 0.5625 - val_loss: 1.4119
    Epoch 69/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7494 - loss: 0.7637 - val_accuracy: 0.7000 - val_loss: 1.3010
    Epoch 70/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7424 - loss: 0.7919 - val_accuracy: 0.7000 - val_loss: 1.3206
    Epoch 71/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7279 - loss: 0.7351 - val_accuracy: 0.7000 - val_loss: 1.3087
    Epoch 72/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7845 - loss: 0.7194 - val_accuracy: 0.5375 - val_loss: 1.3698
    Epoch 73/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7264 - loss: 0.8170 - val_accuracy: 0.6000 - val_loss: 1.3532
    Epoch 74/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7908 - loss: 0.7581 - val_accuracy: 0.5875 - val_loss: 1.4209
    Epoch 75/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.6864 - loss: 0.8250 - val_accuracy: 0.6500 - val_loss: 1.3502
    Epoch 76/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 0.7347 - loss: 0.7305 - val_accuracy: 0.7250 - val_loss: 1.2865
    Epoch 77/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7604 - loss: 0.7048 - val_accuracy: 0.7250 - val_loss: 1.3332
    Epoch 78/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.7594 - loss: 0.6649 - val_accuracy: 0.7000 - val_loss: 1.2792
    Epoch 79/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.7290 - loss: 0.7345 - val_accuracy: 0.6875 - val_loss: 1.2711
    Epoch 80/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7297 - loss: 0.7764 - val_accuracy: 0.6000 - val_loss: 1.3184
    Epoch 81/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7691 - loss: 0.6801 - val_accuracy: 0.6375 - val_loss: 1.2750
    Epoch 82/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7892 - loss: 0.5682 - val_accuracy: 0.6875 - val_loss: 1.2757
    Epoch 83/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8194 - loss: 0.5687 - val_accuracy: 0.6625 - val_loss: 1.2782
    Epoch 84/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.7501 - loss: 0.6861 - val_accuracy: 0.6500 - val_loss: 1.2174
    Epoch 85/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7697 - loss: 0.6788 - val_accuracy: 0.6500 - val_loss: 1.2307
    Epoch 86/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8330 - loss: 0.5290 - val_accuracy: 0.7375 - val_loss: 1.1837
    Epoch 87/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7902 - loss: 0.5991 - val_accuracy: 0.7125 - val_loss: 1.1900
    Epoch 88/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8043 - loss: 0.5234 - val_accuracy: 0.7250 - val_loss: 1.1571
    Epoch 89/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8698 - loss: 0.4276 - val_accuracy: 0.6875 - val_loss: 1.2687
    Epoch 90/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7847 - loss: 0.5585 - val_accuracy: 0.7000 - val_loss: 1.1325
    Epoch 91/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7939 - loss: 0.5400 - val_accuracy: 0.6000 - val_loss: 1.3001
    Epoch 92/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7346 - loss: 0.6950 - val_accuracy: 0.6375 - val_loss: 1.2272
    Epoch 93/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7552 - loss: 0.7079 - val_accuracy: 0.6625 - val_loss: 1.3120
    Epoch 94/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.7922 - loss: 0.5768 - val_accuracy: 0.7125 - val_loss: 1.1539
    Epoch 95/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8382 - loss: 0.5269 - val_accuracy: 0.6875 - val_loss: 1.1578
    Epoch 96/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8132 - loss: 0.5083 - val_accuracy: 0.7125 - val_loss: 1.2243
    Epoch 97/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.7645 - loss: 0.6122 - val_accuracy: 0.7500 - val_loss: 1.1170
    Epoch 98/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8144 - loss: 0.5027 - val_accuracy: 0.7125 - val_loss: 1.1149
    Epoch 99/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8433 - loss: 0.5107 - val_accuracy: 0.7000 - val_loss: 1.2307
    Epoch 100/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8003 - loss: 0.5118 - val_accuracy: 0.6625 - val_loss: 1.1481
    Epoch 101/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8529 - loss: 0.5549 - val_accuracy: 0.7250 - val_loss: 1.0904
    Epoch 102/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8556 - loss: 0.4658 - val_accuracy: 0.6500 - val_loss: 1.1937
    Epoch 103/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8372 - loss: 0.5354 - val_accuracy: 0.7250 - val_loss: 1.0752
    Epoch 104/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8706 - loss: 0.4202 - val_accuracy: 0.7125 - val_loss: 1.1652
    Epoch 105/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8313 - loss: 0.4406 - val_accuracy: 0.7625 - val_loss: 1.0402
    Epoch 106/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8843 - loss: 0.3304 - val_accuracy: 0.7125 - val_loss: 1.0639
    Epoch 107/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8255 - loss: 0.4714 - val_accuracy: 0.7500 - val_loss: 1.0506
    Epoch 108/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.8357 - loss: 0.4495 - val_accuracy: 0.6500 - val_loss: 1.1300
    Epoch 109/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8516 - loss: 0.4337 - val_accuracy: 0.6625 - val_loss: 1.0926
    Epoch 110/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8595 - loss: 0.3821 - val_accuracy: 0.7500 - val_loss: 1.0060
    Epoch 111/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9013 - loss: 0.3532 - val_accuracy: 0.7375 - val_loss: 1.0593
    Epoch 112/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9196 - loss: 0.3679 - val_accuracy: 0.7000 - val_loss: 1.1048
    Epoch 113/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8616 - loss: 0.3405 - val_accuracy: 0.7875 - val_loss: 0.9890
    Epoch 114/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8668 - loss: 0.3663 - val_accuracy: 0.7375 - val_loss: 0.9889
    Epoch 115/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8662 - loss: 0.3801 - val_accuracy: 0.7000 - val_loss: 1.0562
    Epoch 116/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8786 - loss: 0.3273 - val_accuracy: 0.7625 - val_loss: 1.0371
    Epoch 117/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8816 - loss: 0.3685 - val_accuracy: 0.7625 - val_loss: 1.0140
    Epoch 118/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8854 - loss: 0.2800 - val_accuracy: 0.7250 - val_loss: 1.0308
    Epoch 119/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8678 - loss: 0.3703 - val_accuracy: 0.8000 - val_loss: 0.9636
    Epoch 120/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8406 - loss: 0.3641 - val_accuracy: 0.7375 - val_loss: 0.9721
    Epoch 121/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9191 - loss: 0.2665 - val_accuracy: 0.6875 - val_loss: 1.0395
    Epoch 122/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9072 - loss: 0.3009 - val_accuracy: 0.7875 - val_loss: 0.9262
    Epoch 123/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9251 - loss: 0.3178 - val_accuracy: 0.7375 - val_loss: 0.9866
    Epoch 124/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8653 - loss: 0.3785 - val_accuracy: 0.6875 - val_loss: 1.0291
    Epoch 125/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9045 - loss: 0.3111 - val_accuracy: 0.7125 - val_loss: 0.9345
    Epoch 126/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8839 - loss: 0.3579 - val_accuracy: 0.7250 - val_loss: 0.9299
    Epoch 127/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9146 - loss: 0.2813 - val_accuracy: 0.7125 - val_loss: 0.9586
    Epoch 128/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8877 - loss: 0.3315 - val_accuracy: 0.7125 - val_loss: 0.9665
    Epoch 129/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9010 - loss: 0.3063 - val_accuracy: 0.6750 - val_loss: 1.0592
    Epoch 130/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9112 - loss: 0.2796 - val_accuracy: 0.7750 - val_loss: 0.8969
    Epoch 131/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8697 - loss: 0.3572 - val_accuracy: 0.7000 - val_loss: 0.9880
    Epoch 132/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8259 - loss: 0.4261 - val_accuracy: 0.6625 - val_loss: 1.0524
    Epoch 133/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8531 - loss: 0.4237 - val_accuracy: 0.7250 - val_loss: 0.9656
    Epoch 134/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9407 - loss: 0.1865 - val_accuracy: 0.7625 - val_loss: 0.9342
    Epoch 135/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9189 - loss: 0.2394 - val_accuracy: 0.7625 - val_loss: 0.8912
    Epoch 136/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9122 - loss: 0.2626 - val_accuracy: 0.7375 - val_loss: 0.9067
    Epoch 137/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9026 - loss: 0.2640 - val_accuracy: 0.7125 - val_loss: 0.9370
    Epoch 138/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9174 - loss: 0.2604 - val_accuracy: 0.7250 - val_loss: 1.0577
    Epoch 139/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9232 - loss: 0.2842 - val_accuracy: 0.7500 - val_loss: 0.9899
    Epoch 140/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9556 - loss: 0.2127 - val_accuracy: 0.7375 - val_loss: 0.8684
    Epoch 141/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9271 - loss: 0.2382 - val_accuracy: 0.7125 - val_loss: 0.9507
    Epoch 142/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9211 - loss: 0.2564 - val_accuracy: 0.7250 - val_loss: 0.9363
    Epoch 143/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9402 - loss: 0.2097 - val_accuracy: 0.6875 - val_loss: 1.0072
    Epoch 144/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9362 - loss: 0.1988 - val_accuracy: 0.7250 - val_loss: 1.0202
    Epoch 145/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8824 - loss: 0.3077 - val_accuracy: 0.7250 - val_loss: 0.9403
    Epoch 146/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9459 - loss: 0.2285 - val_accuracy: 0.7875 - val_loss: 0.8283
    Epoch 147/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9374 - loss: 0.2152 - val_accuracy: 0.7500 - val_loss: 0.8698
    Epoch 148/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9318 - loss: 0.2208 - val_accuracy: 0.7500 - val_loss: 0.8737
    Epoch 149/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9556 - loss: 0.1720 - val_accuracy: 0.7000 - val_loss: 0.9449
    Epoch 150/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9591 - loss: 0.2218 - val_accuracy: 0.7250 - val_loss: 0.9200
    Epoch 151/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9445 - loss: 0.1897 - val_accuracy: 0.7000 - val_loss: 1.0287
    Epoch 152/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9416 - loss: 0.2399 - val_accuracy: 0.7375 - val_loss: 0.8801
    Epoch 153/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9339 - loss: 0.2180 - val_accuracy: 0.7250 - val_loss: 0.8141
    Epoch 154/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9667 - loss: 0.1312 - val_accuracy: 0.7500 - val_loss: 0.8385
    Epoch 155/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9475 - loss: 0.1373 - val_accuracy: 0.7375 - val_loss: 0.8242
    Epoch 156/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9548 - loss: 0.1734 - val_accuracy: 0.7500 - val_loss: 0.8838
    Epoch 157/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9176 - loss: 0.2404 - val_accuracy: 0.7250 - val_loss: 0.8494
    Epoch 158/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9167 - loss: 0.2424 - val_accuracy: 0.7875 - val_loss: 0.8152
    Epoch 159/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9185 - loss: 0.2523 - val_accuracy: 0.7125 - val_loss: 0.8421
    Epoch 160/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9355 - loss: 0.2263 - val_accuracy: 0.7125 - val_loss: 0.8865
    Epoch 161/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9379 - loss: 0.2025 - val_accuracy: 0.7625 - val_loss: 0.7982
    Epoch 162/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9536 - loss: 0.1795 - val_accuracy: 0.7000 - val_loss: 0.9263
    Epoch 163/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9680 - loss: 0.1502 - val_accuracy: 0.7500 - val_loss: 0.8247
    Epoch 164/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9774 - loss: 0.1383 - val_accuracy: 0.7000 - val_loss: 0.8442
    Epoch 165/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.8993 - loss: 0.2467 - val_accuracy: 0.7500 - val_loss: 0.7719
    Epoch 166/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9585 - loss: 0.1505 - val_accuracy: 0.7250 - val_loss: 0.8167
    Epoch 167/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9526 - loss: 0.1608 - val_accuracy: 0.7250 - val_loss: 0.9205
    Epoch 168/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8710 - loss: 0.3986 - val_accuracy: 0.7250 - val_loss: 0.9304
    Epoch 169/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9287 - loss: 0.2472 - val_accuracy: 0.7250 - val_loss: 0.8753
    Epoch 170/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9806 - loss: 0.1289 - val_accuracy: 0.7125 - val_loss: 0.8512
    Epoch 171/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9439 - loss: 0.1761 - val_accuracy: 0.7875 - val_loss: 0.7551
    Epoch 172/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9767 - loss: 0.0996 - val_accuracy: 0.7375 - val_loss: 0.8713
    Epoch 173/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9679 - loss: 0.1126 - val_accuracy: 0.7250 - val_loss: 0.7779
    Epoch 174/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9514 - loss: 0.1694 - val_accuracy: 0.7500 - val_loss: 0.7914
    Epoch 175/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9345 - loss: 0.2099 - val_accuracy: 0.7375 - val_loss: 0.8459
    Epoch 176/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9645 - loss: 0.1776 - val_accuracy: 0.7375 - val_loss: 0.8384
    Epoch 177/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9374 - loss: 0.1946 - val_accuracy: 0.6750 - val_loss: 0.9231
    Epoch 178/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9698 - loss: 0.1160 - val_accuracy: 0.6875 - val_loss: 1.0010
    Epoch 179/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9370 - loss: 0.1938 - val_accuracy: 0.7125 - val_loss: 1.1130
    Epoch 180/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9769 - loss: 0.1525 - val_accuracy: 0.7125 - val_loss: 1.0235
    Epoch 181/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9392 - loss: 0.1836 - val_accuracy: 0.7125 - val_loss: 0.9246
    Epoch 182/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9541 - loss: 0.1493 - val_accuracy: 0.7500 - val_loss: 0.8284
    Epoch 183/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9258 - loss: 0.1721 - val_accuracy: 0.7250 - val_loss: 0.8203
    Epoch 184/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9648 - loss: 0.1357 - val_accuracy: 0.7250 - val_loss: 0.8922
    Epoch 185/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9543 - loss: 0.1509 - val_accuracy: 0.7375 - val_loss: 0.8921
    Epoch 186/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9778 - loss: 0.1099 - val_accuracy: 0.7375 - val_loss: 0.8261
    Epoch 187/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9622 - loss: 0.1298 - val_accuracy: 0.7375 - val_loss: 0.8475
    Epoch 188/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9757 - loss: 0.1117 - val_accuracy: 0.7250 - val_loss: 0.8500
    Epoch 189/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9577 - loss: 0.1148 - val_accuracy: 0.7500 - val_loss: 0.7524
    Epoch 190/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9637 - loss: 0.1133 - val_accuracy: 0.7375 - val_loss: 0.7990
    Epoch 191/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9788 - loss: 0.1059 - val_accuracy: 0.7000 - val_loss: 0.8601
    Epoch 192/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9670 - loss: 0.1287 - val_accuracy: 0.7500 - val_loss: 0.8350
    Epoch 193/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9686 - loss: 0.1103 - val_accuracy: 0.7375 - val_loss: 0.7967
    Epoch 194/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9741 - loss: 0.1083 - val_accuracy: 0.7250 - val_loss: 0.8237
    Epoch 195/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9673 - loss: 0.1268 - val_accuracy: 0.7500 - val_loss: 0.7768
    Epoch 196/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9733 - loss: 0.1293 - val_accuracy: 0.7500 - val_loss: 0.8068
    Epoch 197/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9810 - loss: 0.0844 - val_accuracy: 0.7250 - val_loss: 0.8169
    Epoch 198/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9871 - loss: 0.0763 - val_accuracy: 0.7625 - val_loss: 0.7554
    Epoch 199/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9912 - loss: 0.0694 - val_accuracy: 0.7500 - val_loss: 0.8077
    Epoch 200/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9868 - loss: 0.0900 - val_accuracy: 0.7125 - val_loss: 0.7910


### Create the 2D model

1. Create three spectrograms with multiple band-widths from the raw input.
2. Concatenate the three spectrograms to have three channels.
3. Load `MobileNet` and set the weights from the weights trained on `ImageNet`.
4. Apply global maximum pooling to have fixed set of features.
5. Add `Dense` layers to make the final predictions based on the features.


```python
input = layers.Input((None, 1))
spectrograms = [
    layers.STFTSpectrogram(
        mode="log",
        frame_length=SAMPLE_RATE * x // 1000,
        frame_step=SAMPLE_RATE * 15 // 1000,
        fft_length=2048,
        padding="same",
        expand_dims=True,
        # trainable=True,  # trainable by default
    )(input)
    for x in [30, 40, 50]
]

multi_spectrograms = layers.Concatenate(axis=-1)(spectrograms)

img_model_weights_no_top = keras.applications.MobileNet(
    weights="imagenet"
).get_weights()[:-2]

img_model = keras.applications.MobileNet(
    weights=None,
    include_top=False,
    pooling="max",
)
output = img_model(multi_spectrograms)
img_model.set_weights(img_model_weights_no_top)

output = layers.Dropout(0.5)(output)
output = layers.Dense(256, activation="relu")(output)
output = layers.Dense(256, activation="relu")(output)
output = layers.Dense(NUM_CLASSES, activation="softmax")(output)
model2d = keras.Model(input, output, name="model_2d_trainble_stft")

model2d.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model2d.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "model_2d_trainble_stft"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)              </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">        Param # </span>┃<span style="font-weight: bold"> Connected to           </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_1             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)        │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)              │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ stft_spectrogram_5        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1025</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">984,000</span> │ input_layer_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">STFTSpectrogram</span>)         │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ stft_spectrogram_6        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1025</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)  │      <span style="color: #00af00; text-decoration-color: #00af00">1,312,000</span> │ input_layer_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">STFTSpectrogram</span>)         │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ stft_spectrogram_7        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1025</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)  │      <span style="color: #00af00; text-decoration-color: #00af00">1,640,000</span> │ input_layer_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">STFTSpectrogram</span>)         │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ concatenate (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1025</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)  │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ stft_spectrogram_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                           │                        │                │ stft_spectrogram_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
│                           │                        │                │ stft_spectrogram_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ mobilenet_1.00_None       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │      <span style="color: #00af00; text-decoration-color: #00af00">3,228,864</span> │ concatenate[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)              │                        │                │                        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │              <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ mobilenet_1.00_None[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │ dropout_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]        │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">65,792</span> │ dense_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├───────────────────────────┼────────────────────────┼────────────────┼────────────────────────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │          <span style="color: #00af00; text-decoration-color: #00af00">2,570</span> │ dense_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
└───────────────────────────┴────────────────────────┴────────────────┴────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,495,626</span> (28.59 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,473,738</span> (28.51 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,888</span> (85.50 KB)
</pre>



Create the datasets.


```python
train_ds = read_dataset(pd_data, [1, 2, 3], BATCH_SIZE)
valid_ds = read_dataset(pd_data, [4], BATCH_SIZE)
```

Train the model and restore the best weights.


```python
history_model2d = model2d.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EPOCHS,
            restore_best_weights=True,
        )
    ],
)
```

    Epoch 1/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 241ms/step - accuracy: 0.0772 - loss: 10.0748 - val_accuracy: 0.1125 - val_loss: 3.4215
    Epoch 2/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 56ms/step - accuracy: 0.1752 - loss: 5.3220 - val_accuracy: 0.0750 - val_loss: 3.8230
    Epoch 3/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.1838 - loss: 4.8451 - val_accuracy: 0.0625 - val_loss: 3.2156
    Epoch 4/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.1830 - loss: 4.0179 - val_accuracy: 0.1750 - val_loss: 2.5628
    Epoch 5/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.1856 - loss: 3.8230 - val_accuracy: 0.2500 - val_loss: 2.0747
    Epoch 6/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.2231 - loss: 3.3651 - val_accuracy: 0.2125 - val_loss: 2.2284
    Epoch 7/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.2456 - loss: 3.5056 - val_accuracy: 0.2375 - val_loss: 2.1704
    Epoch 8/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.2829 - loss: 2.6851 - val_accuracy: 0.2750 - val_loss: 1.9332
    Epoch 9/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.3317 - loss: 2.7608 - val_accuracy: 0.2250 - val_loss: 1.9706
    Epoch 10/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.3782 - loss: 2.2747 - val_accuracy: 0.3500 - val_loss: 1.7094
    Epoch 11/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.3352 - loss: 2.3634 - val_accuracy: 0.3000 - val_loss: 1.6620
    Epoch 12/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.4165 - loss: 2.0768 - val_accuracy: 0.3625 - val_loss: 1.4914
    Epoch 13/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.5090 - loss: 1.5401 - val_accuracy: 0.3750 - val_loss: 1.5994
    Epoch 14/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.4981 - loss: 1.6282 - val_accuracy: 0.4875 - val_loss: 1.3522
    Epoch 15/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.4556 - loss: 1.8024 - val_accuracy: 0.5875 - val_loss: 1.1055
    Epoch 16/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.5468 - loss: 1.3591 - val_accuracy: 0.5625 - val_loss: 1.1655
    Epoch 17/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.4804 - loss: 1.6241 - val_accuracy: 0.6625 - val_loss: 1.0041
    Epoch 18/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.5663 - loss: 1.2215 - val_accuracy: 0.6875 - val_loss: 0.9314
    Epoch 19/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.6244 - loss: 1.3624 - val_accuracy: 0.6250 - val_loss: 1.0186
    Epoch 20/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.6304 - loss: 1.0465 - val_accuracy: 0.7125 - val_loss: 0.8449
    Epoch 21/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.5547 - loss: 1.3271 - val_accuracy: 0.7750 - val_loss: 0.7789
    Epoch 22/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.6104 - loss: 1.2843 - val_accuracy: 0.7250 - val_loss: 0.8555
    Epoch 23/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.7531 - loss: 0.8091 - val_accuracy: 0.7250 - val_loss: 0.7905
    Epoch 24/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.6975 - loss: 0.8802 - val_accuracy: 0.8000 - val_loss: 0.6515
    Epoch 25/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.7240 - loss: 0.7285 - val_accuracy: 0.8125 - val_loss: 0.6149
    Epoch 26/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.8093 - loss: 0.6332 - val_accuracy: 0.8000 - val_loss: 0.5753
    Epoch 27/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.7431 - loss: 0.7749 - val_accuracy: 0.8875 - val_loss: 0.4763
    Epoch 28/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.7394 - loss: 0.6632 - val_accuracy: 0.8750 - val_loss: 0.4819
    Epoch 29/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.8196 - loss: 0.5819 - val_accuracy: 0.9000 - val_loss: 0.4198
    Epoch 30/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.8328 - loss: 0.4982 - val_accuracy: 0.9000 - val_loss: 0.4261
    Epoch 31/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.8401 - loss: 0.5029 - val_accuracy: 0.8750 - val_loss: 0.4263
    Epoch 32/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.7893 - loss: 0.6699 - val_accuracy: 0.8750 - val_loss: 0.4196
    Epoch 33/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.8039 - loss: 0.5666 - val_accuracy: 0.8500 - val_loss: 0.4357
    Epoch 34/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.8805 - loss: 0.4499 - val_accuracy: 0.8125 - val_loss: 0.4730
    Epoch 35/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.8835 - loss: 0.3691 - val_accuracy: 0.9250 - val_loss: 0.3386
    Epoch 36/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.8860 - loss: 0.3629 - val_accuracy: 0.9000 - val_loss: 0.3530
    Epoch 37/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.8985 - loss: 0.3046 - val_accuracy: 0.9000 - val_loss: 0.3883
    Epoch 38/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 0.9061 - loss: 0.2937 - val_accuracy: 0.9000 - val_loss: 0.3497
    Epoch 39/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.8779 - loss: 0.2889 - val_accuracy: 0.8750 - val_loss: 0.3350
    Epoch 40/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.9069 - loss: 0.2309 - val_accuracy: 0.9000 - val_loss: 0.2529
    Epoch 41/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9093 - loss: 0.2495 - val_accuracy: 0.8875 - val_loss: 0.3495
    Epoch 42/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9337 - loss: 0.1973 - val_accuracy: 0.9125 - val_loss: 0.2986
    Epoch 43/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.8957 - loss: 0.2724 - val_accuracy: 0.9125 - val_loss: 0.3048
    Epoch 44/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9239 - loss: 0.2497 - val_accuracy: 0.9125 - val_loss: 0.2958
    Epoch 45/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 59ms/step - accuracy: 0.9299 - loss: 0.2440 - val_accuracy: 0.8750 - val_loss: 0.3564
    Epoch 46/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9454 - loss: 0.1340 - val_accuracy: 0.8875 - val_loss: 0.2642
    Epoch 47/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9366 - loss: 0.1625 - val_accuracy: 0.8750 - val_loss: 0.2580
    Epoch 48/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9205 - loss: 0.2051 - val_accuracy: 0.8875 - val_loss: 0.2658
    Epoch 49/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9063 - loss: 0.2548 - val_accuracy: 0.9125 - val_loss: 0.2548
    Epoch 50/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.9301 - loss: 0.2034 - val_accuracy: 0.9250 - val_loss: 0.2041
    Epoch 51/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9275 - loss: 0.2189 - val_accuracy: 0.9250 - val_loss: 0.2208
    Epoch 52/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9554 - loss: 0.1417 - val_accuracy: 0.9000 - val_loss: 0.2802
    Epoch 53/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9575 - loss: 0.1930 - val_accuracy: 0.9000 - val_loss: 0.2934
    Epoch 54/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9661 - loss: 0.1227 - val_accuracy: 0.9125 - val_loss: 0.3053
    Epoch 55/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9689 - loss: 0.0847 - val_accuracy: 0.9125 - val_loss: 0.2769
    Epoch 56/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9219 - loss: 0.1821 - val_accuracy: 0.9000 - val_loss: 0.2513
    Epoch 57/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9694 - loss: 0.1105 - val_accuracy: 0.8875 - val_loss: 0.3085
    Epoch 58/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9567 - loss: 0.1781 - val_accuracy: 0.8875 - val_loss: 0.3107
    Epoch 59/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9705 - loss: 0.0815 - val_accuracy: 0.8875 - val_loss: 0.2790
    Epoch 60/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9707 - loss: 0.1056 - val_accuracy: 0.9375 - val_loss: 0.2997
    Epoch 61/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9797 - loss: 0.0578 - val_accuracy: 0.9375 - val_loss: 0.3195
    Epoch 62/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9598 - loss: 0.1054 - val_accuracy: 0.8875 - val_loss: 0.3326
    Epoch 63/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9853 - loss: 0.0367 - val_accuracy: 0.8875 - val_loss: 0.3362
    Epoch 64/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9495 - loss: 0.1097 - val_accuracy: 0.8875 - val_loss: 0.3407
    Epoch 65/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9782 - loss: 0.0582 - val_accuracy: 0.8750 - val_loss: 0.3544
    Epoch 66/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9787 - loss: 0.0990 - val_accuracy: 0.9125 - val_loss: 0.3045
    Epoch 67/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9973 - loss: 0.0442 - val_accuracy: 0.8750 - val_loss: 0.3278
    Epoch 68/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9831 - loss: 0.0547 - val_accuracy: 0.9125 - val_loss: 0.3073
    Epoch 69/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9934 - loss: 0.0493 - val_accuracy: 0.9125 - val_loss: 0.2853
    Epoch 70/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9751 - loss: 0.0767 - val_accuracy: 0.9125 - val_loss: 0.2797
    Epoch 71/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9596 - loss: 0.0671 - val_accuracy: 0.9250 - val_loss: 0.2401
    Epoch 72/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9694 - loss: 0.0734 - val_accuracy: 0.9125 - val_loss: 0.3236
    Epoch 73/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9373 - loss: 0.1989 - val_accuracy: 0.9250 - val_loss: 0.3073
    Epoch 74/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9969 - loss: 0.0237 - val_accuracy: 0.9125 - val_loss: 0.3161
    Epoch 75/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9958 - loss: 0.0372 - val_accuracy: 0.9250 - val_loss: 0.3126
    Epoch 76/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9943 - loss: 0.0249 - val_accuracy: 0.9000 - val_loss: 0.3145
    Epoch 77/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9957 - loss: 0.0433 - val_accuracy: 0.9000 - val_loss: 0.3318
    Epoch 78/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9854 - loss: 0.0319 - val_accuracy: 0.8875 - val_loss: 0.3469
    Epoch 79/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9913 - loss: 0.0352 - val_accuracy: 0.8875 - val_loss: 0.3538
    Epoch 80/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9960 - loss: 0.0180 - val_accuracy: 0.8875 - val_loss: 0.4080
    Epoch 81/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9719 - loss: 0.0926 - val_accuracy: 0.8875 - val_loss: 0.3289
    Epoch 82/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9642 - loss: 0.0865 - val_accuracy: 0.9125 - val_loss: 0.2783
    Epoch 83/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9753 - loss: 0.0405 - val_accuracy: 0.9250 - val_loss: 0.3091
    Epoch 84/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9961 - loss: 0.0189 - val_accuracy: 0.9000 - val_loss: 0.3357
    Epoch 85/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9957 - loss: 0.0290 - val_accuracy: 0.8875 - val_loss: 0.4122
    Epoch 86/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9912 - loss: 0.0283 - val_accuracy: 0.8875 - val_loss: 0.5165
    Epoch 87/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9832 - loss: 0.0300 - val_accuracy: 0.8625 - val_loss: 0.4759
    Epoch 88/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9889 - loss: 0.0327 - val_accuracy: 0.8625 - val_loss: 0.4597
    Epoch 89/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9922 - loss: 0.0308 - val_accuracy: 0.8750 - val_loss: 0.6098
    Epoch 90/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9934 - loss: 0.0223 - val_accuracy: 0.8625 - val_loss: 0.5787
    Epoch 91/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9774 - loss: 0.0397 - val_accuracy: 0.8625 - val_loss: 0.6253
    Epoch 92/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9992 - loss: 0.0160 - val_accuracy: 0.8375 - val_loss: 0.6546
    Epoch 93/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9938 - loss: 0.0258 - val_accuracy: 0.8375 - val_loss: 0.5290
    Epoch 94/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9927 - loss: 0.0459 - val_accuracy: 0.8250 - val_loss: 0.5061
    Epoch 95/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9784 - loss: 0.0330 - val_accuracy: 0.8500 - val_loss: 0.3513
    Epoch 96/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9754 - loss: 0.0654 - val_accuracy: 0.8125 - val_loss: 0.5994
    Epoch 97/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0079 - val_accuracy: 0.8375 - val_loss: 0.5918
    Epoch 98/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9934 - loss: 0.0269 - val_accuracy: 0.8125 - val_loss: 0.6964
    Epoch 99/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9789 - loss: 0.0671 - val_accuracy: 0.8375 - val_loss: 0.5500
    Epoch 100/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9935 - loss: 0.0244 - val_accuracy: 0.8375 - val_loss: 0.5278
    Epoch 101/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0098 - val_accuracy: 0.8375 - val_loss: 0.5453
    Epoch 102/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9906 - loss: 0.0151 - val_accuracy: 0.8625 - val_loss: 0.5868
    Epoch 103/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9883 - loss: 0.0307 - val_accuracy: 0.8750 - val_loss: 0.4981
    Epoch 104/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9926 - loss: 0.0317 - val_accuracy: 0.8625 - val_loss: 0.4333
    Epoch 105/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9686 - loss: 0.1032 - val_accuracy: 0.8625 - val_loss: 0.3988
    Epoch 106/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0159 - val_accuracy: 0.9000 - val_loss: 0.3234
    Epoch 107/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9921 - loss: 0.0335 - val_accuracy: 0.9250 - val_loss: 0.2830
    Epoch 108/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9929 - loss: 0.0213 - val_accuracy: 0.9000 - val_loss: 0.3106
    Epoch 109/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0135 - val_accuracy: 0.9000 - val_loss: 0.4439
    Epoch 110/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9986 - loss: 0.0109 - val_accuracy: 0.9000 - val_loss: 0.4254
    Epoch 111/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0036 - val_accuracy: 0.9000 - val_loss: 0.4273
    Epoch 112/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9989 - loss: 0.0138 - val_accuracy: 0.8875 - val_loss: 0.4412
    Epoch 113/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9864 - loss: 0.0426 - val_accuracy: 0.8250 - val_loss: 0.6279
    Epoch 114/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9661 - loss: 0.0747 - val_accuracy: 0.8500 - val_loss: 0.4518
    Epoch 115/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 0.9789 - loss: 0.0558 - val_accuracy: 0.8625 - val_loss: 0.4124
    Epoch 116/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 0.9957 - loss: 0.0168 - val_accuracy: 0.8875 - val_loss: 0.3987
    Epoch 117/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9915 - loss: 0.0221 - val_accuracy: 0.8875 - val_loss: 0.4118
    Epoch 118/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9986 - loss: 0.0175 - val_accuracy: 0.9125 - val_loss: 0.3454
    Epoch 119/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 0.9923 - loss: 0.0223 - val_accuracy: 0.9000 - val_loss: 0.2743
    Epoch 120/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9899 - loss: 0.0176 - val_accuracy: 0.8875 - val_loss: 0.3684
    Epoch 121/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9992 - loss: 0.0063 - val_accuracy: 0.8625 - val_loss: 0.4809
    Epoch 122/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9594 - loss: 0.0860 - val_accuracy: 0.8500 - val_loss: 0.5523
    Epoch 123/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9982 - loss: 0.0102 - val_accuracy: 0.8750 - val_loss: 0.4670
    Epoch 124/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9736 - loss: 0.0752 - val_accuracy: 0.9125 - val_loss: 0.4640
    Epoch 125/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9989 - loss: 0.0120 - val_accuracy: 0.9125 - val_loss: 0.4506
    Epoch 126/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9878 - loss: 0.0404 - val_accuracy: 0.9125 - val_loss: 0.3384
    Epoch 127/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9969 - loss: 0.0069 - val_accuracy: 0.9250 - val_loss: 0.2696
    Epoch 128/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9978 - loss: 0.0086 - val_accuracy: 0.9125 - val_loss: 0.2604
    Epoch 129/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9868 - loss: 0.0320 - val_accuracy: 0.9375 - val_loss: 0.2580
    Epoch 130/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 0.9884 - loss: 0.0233 - val_accuracy: 0.9375 - val_loss: 0.2284
    Epoch 131/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9931 - loss: 0.0161 - val_accuracy: 0.9500 - val_loss: 0.2568
    Epoch 132/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0049 - val_accuracy: 0.9375 - val_loss: 0.2414
    Epoch 133/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9820 - loss: 0.0392 - val_accuracy: 0.9125 - val_loss: 0.3750
    Epoch 134/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 0.9982 - loss: 0.0040 - val_accuracy: 0.9250 - val_loss: 0.3631
    Epoch 135/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9981 - loss: 0.0075 - val_accuracy: 0.9375 - val_loss: 0.3204
    Epoch 136/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0107 - val_accuracy: 0.9125 - val_loss: 0.3899
    Epoch 137/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 0.9926 - loss: 0.0198 - val_accuracy: 0.9000 - val_loss: 0.4519
    Epoch 138/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9939 - loss: 0.0133 - val_accuracy: 0.9125 - val_loss: 0.3867
    Epoch 139/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 1.0000 - loss: 0.0051 - val_accuracy: 0.9000 - val_loss: 0.3504
    Epoch 140/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 1.0000 - loss: 0.0056 - val_accuracy: 0.9000 - val_loss: 0.3603
    Epoch 141/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 1.0000 - loss: 0.0019 - val_accuracy: 0.9000 - val_loss: 0.3859
    Epoch 142/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9971 - loss: 0.0122 - val_accuracy: 0.8875 - val_loss: 0.3531
    Epoch 143/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0123 - val_accuracy: 0.8875 - val_loss: 0.3246
    Epoch 144/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9984 - loss: 0.0070 - val_accuracy: 0.9000 - val_loss: 0.3378
    Epoch 145/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0029 - val_accuracy: 0.9000 - val_loss: 0.3518
    Epoch 146/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9907 - loss: 0.0150 - val_accuracy: 0.9000 - val_loss: 0.3707
    Epoch 147/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0027 - val_accuracy: 0.8875 - val_loss: 0.3692
    Epoch 148/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9926 - loss: 0.0130 - val_accuracy: 0.9125 - val_loss: 0.3239
    Epoch 149/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9885 - loss: 0.0169 - val_accuracy: 0.9000 - val_loss: 0.3157
    Epoch 150/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0018 - val_accuracy: 0.9000 - val_loss: 0.3370
    Epoch 151/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0034 - val_accuracy: 0.9000 - val_loss: 0.3482
    Epoch 152/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0084 - val_accuracy: 0.9000 - val_loss: 0.3282
    Epoch 153/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 7.1933e-04 - val_accuracy: 0.9000 - val_loss: 0.3260
    Epoch 154/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0025 - val_accuracy: 0.9000 - val_loss: 0.3172
    Epoch 155/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9870 - loss: 0.0168 - val_accuracy: 0.9250 - val_loss: 0.3019
    Epoch 156/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 1.0000 - loss: 0.0020 - val_accuracy: 0.9375 - val_loss: 0.3361
    Epoch 157/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9982 - loss: 0.0123 - val_accuracy: 0.9250 - val_loss: 0.3319
    Epoch 158/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 54ms/step - accuracy: 1.0000 - loss: 0.0050 - val_accuracy: 0.9125 - val_loss: 0.3063
    Epoch 159/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0027 - val_accuracy: 0.9250 - val_loss: 0.2939
    Epoch 160/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0063 - val_accuracy: 0.9250 - val_loss: 0.2858
    Epoch 161/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9963 - loss: 0.0085 - val_accuracy: 0.9000 - val_loss: 0.3987
    Epoch 162/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9969 - loss: 0.0059 - val_accuracy: 0.8875 - val_loss: 0.4951
    Epoch 163/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0064 - val_accuracy: 0.8875 - val_loss: 0.3813
    Epoch 164/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0029 - val_accuracy: 0.9000 - val_loss: 0.3151
    Epoch 165/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9890 - loss: 0.0168 - val_accuracy: 0.8875 - val_loss: 0.3726
    Epoch 166/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9949 - loss: 0.0134 - val_accuracy: 0.8875 - val_loss: 0.4562
    Epoch 167/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9982 - loss: 0.0090 - val_accuracy: 0.8875 - val_loss: 0.3808
    Epoch 168/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0049 - val_accuracy: 0.9000 - val_loss: 0.3403
    Epoch 169/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9978 - loss: 0.0082 - val_accuracy: 0.9000 - val_loss: 0.4163
    Epoch 170/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9949 - loss: 0.0205 - val_accuracy: 0.9125 - val_loss: 0.3201
    Epoch 171/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0056 - val_accuracy: 0.9000 - val_loss: 0.3134
    Epoch 172/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0013 - val_accuracy: 0.8875 - val_loss: 0.2984
    Epoch 173/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9986 - loss: 0.0069 - val_accuracy: 0.8875 - val_loss: 0.2955
    Epoch 174/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0026 - val_accuracy: 0.9000 - val_loss: 0.3088
    Epoch 175/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9899 - loss: 0.0128 - val_accuracy: 0.9125 - val_loss: 0.3128
    Epoch 176/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9957 - loss: 0.0105 - val_accuracy: 0.9000 - val_loss: 0.4046
    Epoch 177/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9938 - loss: 0.0173 - val_accuracy: 0.9250 - val_loss: 0.3463
    Epoch 178/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0075 - val_accuracy: 0.9250 - val_loss: 0.3115
    Epoch 179/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9933 - loss: 0.0123 - val_accuracy: 0.9250 - val_loss: 0.2709
    Epoch 180/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9868 - loss: 0.0278 - val_accuracy: 0.9125 - val_loss: 0.3692
    Epoch 181/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 0.9250 - val_loss: 0.3713
    Epoch 182/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0052 - val_accuracy: 0.9375 - val_loss: 0.3373
    Epoch 183/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0026 - val_accuracy: 0.9250 - val_loss: 0.3887
    Epoch 184/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0037 - val_accuracy: 0.9125 - val_loss: 0.3894
    Epoch 185/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0023 - val_accuracy: 0.9250 - val_loss: 0.3724
    Epoch 186/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9812 - loss: 0.0613 - val_accuracy: 0.8875 - val_loss: 0.3160
    Epoch 187/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 0.8875 - val_loss: 0.3470
    Epoch 188/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0035 - val_accuracy: 0.8750 - val_loss: 0.2881
    Epoch 189/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9907 - loss: 0.0236 - val_accuracy: 0.8625 - val_loss: 0.2981
    Epoch 190/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0018 - val_accuracy: 0.8625 - val_loss: 0.3023
    Epoch 191/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0079 - val_accuracy: 0.8875 - val_loss: 0.2972
    Epoch 192/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9868 - loss: 0.0136 - val_accuracy: 0.8875 - val_loss: 0.3565
    Epoch 193/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 4.5110e-04 - val_accuracy: 0.8875 - val_loss: 0.3770
    Epoch 194/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 58ms/step - accuracy: 1.0000 - loss: 8.7789e-04 - val_accuracy: 0.9125 - val_loss: 0.3755
    Epoch 195/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 1.0000 - loss: 0.0014 - val_accuracy: 0.9125 - val_loss: 0.3466
    Epoch 196/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9653 - loss: 0.0880 - val_accuracy: 0.9125 - val_loss: 0.2321
    Epoch 197/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9971 - loss: 0.0222 - val_accuracy: 0.8875 - val_loss: 0.6022
    Epoch 198/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9900 - loss: 0.0499 - val_accuracy: 0.9125 - val_loss: 0.2179
    Epoch 199/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9895 - loss: 0.0396 - val_accuracy: 0.8625 - val_loss: 0.5256
    Epoch 200/200
    [1m15/15[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 55ms/step - accuracy: 0.9814 - loss: 0.1108 - val_accuracy: 0.9000 - val_loss: 0.3664


### Plot Training History


```python
epochs_range = range(EPOCHS)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(
    epochs_range,
    history_model1d.history["accuracy"],
    label="Training Accuracy,1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model1d.history["val_accuracy"],
    label="Validation Accuracy, 1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["accuracy"],
    label="Training Accuracy, 2D model with trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["val_accuracy"],
    label="Validation Accuracy, 2D model with trainable STFT",
)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(
    epochs_range,
    history_model1d.history["loss"],
    label="Training Loss,1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model1d.history["val_loss"],
    label="Validation Loss, 1D model with non-trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["loss"],
    label="Training Loss, 2D model with trainable STFT",
)
plt.plot(
    epochs_range,
    history_model2d.history["val_loss"],
    label="Validation Loss, 2D model with trainable STFT",
)
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
```


    
![png](https://github.com/mostafa-mahmoud/keras-io/blob/master/examples/audio/img/stft/training.png)
    


### Evaluate on Test Data


Running the models on the test set.


```python
test_ds = read_dataset(pd_data, [5], BATCH_SIZE, False)
```


```python
_, test_acc = model1d.evaluate(test_ds)
print(
    f"1D model wit non-trainable STFT -> Test Accuracy: {test_acc * 100:.2f}%"
)
```

    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.8780 - loss: 0.5934
    1D model wit non-trainable STFT -> Test Accuracy: 85.00%



```python
_, test_acc = model2d.evaluate(test_ds)
print(f"2D model with trainable STFT -> Test Accuracy: {test_acc * 100:.2f}%")
```

    [1m5/5[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 15ms/step - accuracy: 0.9137 - loss: 0.4167
    2D model with trainable STFT -> Test Accuracy: 90.00%



