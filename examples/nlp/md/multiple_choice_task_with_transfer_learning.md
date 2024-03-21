# MultipleChoice Task with Transfer Learning

**Author:** Md Awsafur Rahman<br>
**Date created:** 2023/09/14<br>
**Last modified:** 2024/01/10<br>
**Description:** Use pre-trained nlp models for multiplechoice task.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/multiple_choice_task_with_transfer_learning.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/multiple_choice_task_with_transfer_learning.py)



---
## Introduction

In this example, we will demonstrate how to perform the **MultipleChoice** task by
finetuning pre-trained DebertaV3 model. In this task, several candidate answers are
provided along with a context and the model is trained to select the correct answer
unlike question answering. We will use SWAG dataset to demonstrate this example.

---
## Setup


```python

import keras
import keras_nlp
import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
```

---
## Dataset
In this example we'll use **SWAG** dataset for multiplechoice task.


```python
!wget "https://github.com/rowanz/swagaf/archive/refs/heads/master.zip" -O swag.zip
!unzip -q swag.zip
```

<div class="k-default-codeblock">
```
--2024-01-11 01:43:38--  https://github.com/rowanz/swagaf/archive/refs/heads/master.zip
Resolving github.com (github.com)... 140.82.112.4
Connecting to github.com (github.com)|140.82.112.4|:443... 

connected.

HTTP request sent, awaiting response... 

302 Found
Location: https://codeload.github.com/rowanz/swagaf/zip/refs/heads/master [following]
--2024-01-11 01:43:38--  https://codeload.github.com/rowanz/swagaf/zip/refs/heads/master
Resolving codeload.github.com (codeload.github.com)... 140.82.113.9
Connecting to codeload.github.com (codeload.github.com)|140.82.113.9|:443... 

connected.

HTTP request sent, awaiting response... 

200 OK
Length: unspecified [application/zip]
Saving to: ‘swag.zip’
```
</div>
    
    
swag.zip                [<=>                 ]       0  --.-KB/s               

    
swag.zip                [ <=>                ]  84.24K   229KB/s               

    
swag.zip                [  <=>               ]   1.46M  2.44MB/s               

    
swag.zip                [   <=>              ]   2.20M  1.48MB/s               

    
swag.zip                [    <=>             ]   8.59M  3.16MB/s               

    
swag.zip                [     <=>            ]  15.40M  4.96MB/s               

    
swag.zip                [      <=>           ]  17.54M  5.05MB/s               
swag.zip                [       <=>          ]  19.94M  5.71MB/s    in 3.5s    
    
<div class="k-default-codeblock">
```
2024-01-11 01:43:42 (5.06 MB/s) - ‘swag.zip’ saved [20905751]
```
</div>
    



```python
!ls swagaf-master/data
```

<div class="k-default-codeblock">
```
README.md  test.csv  train.csv	train_full.csv	val.csv  val_full.csv

```
</div>
---
## Configuration


```python

class CFG:
    preset = "deberta_v3_extra_small_en"  # Name of pretrained models
    sequence_length = 200  # Input sequence length
    seed = 42  # Random seed
    epochs = 5  # Training epochs
    batch_size = 8  # Batch size
    augment = True  # Augmentation (Shuffle Options)

```

---
## Reproducibility
Sets value for random seed to produce similar result in each run.


```python
keras.utils.set_random_seed(CFG.seed)

```

---
## Meta Data
* **train.csv** - will be used for training.
* `sent1` and `sent2`: these fields show how a sentence starts, and if you put the two
together, you get the `startphrase` field.
* `ending_<i>`: suggests a possible ending for how a sentence can end, but only one of
them is correct.
    * `label`: identifies the correct sentence ending.

* **val.csv** - similar to `train.csv` but will be used for validation.


```python
# Train data
train_df = pd.read_csv(
    "swagaf-master/data/train.csv", index_col=0
)  # Read CSV file into a DataFrame
train_df = train_df.sample(frac=0.02)
print("# Train Data: {:,}".format(len(train_df)))

# Valid data
valid_df = pd.read_csv(
    "swagaf-master/data/val.csv", index_col=0
)  # Read CSV file into a DataFrame
valid_df = valid_df.sample(frac=0.02)
print("# Valid Data: {:,}".format(len(valid_df)))
```

<div class="k-default-codeblock">
```
# Train Data: 1,471
# Valid Data: 400

```
</div>
---
## Contextualize Options

Our approach entails furnishing the model with question and answer pairs, as opposed to
employing a single question for all five options. In practice, this signifies that for
the five options, we will supply the model with the same set of five questions combined
with each respective answer choice (e.g., `(Q + A)`, `(Q + B)`, and so on). This analogy
draws parallels to the practice of revisiting a question multiple times during an exam to
promote a deeper understanding of the problem at hand.

> Notably, in the context of SWAG dataset, question is the start of a sentence and
options are possible ending of that sentence.


```python

# Define a function to create options based on the prompt and choices
def make_options(row):
    row["options"] = [
        f"{row.startphrase}\n{row.ending0}",  # Option 0
        f"{row.startphrase}\n{row.ending1}",  # Option 1
        f"{row.startphrase}\n{row.ending2}",  # Option 2
        f"{row.startphrase}\n{row.ending3}",
    ]  # Option 3
    return row

```

Apply the `make_options` function to each row of the dataframe


```python
train_df = train_df.apply(make_options, axis=1)
valid_df = valid_df.apply(make_options, axis=1)
```

---
## Preprocessing

**What it does:** The preprocessor takes input strings and transforms them into a
dictionary (`token_ids`, `padding_mask`) containing preprocessed tensors. This process
starts with tokenization, where input strings are converted into sequences of token IDs.

**Why it's important:** Initially, raw text data is complex and challenging for modeling
due to its high dimensionality. By converting text into a compact set of tokens, such as
transforming `"The quick brown fox"` into `["the", "qu", "##ick", "br", "##own", "fox"]`,
we simplify the data. Many models rely on special tokens and additional tensors to
understand input. These tokens help divide input and identify padding, among other tasks.
Making all sequences the same length through padding boosts computational efficiency,
making subsequent steps smoother.

Explore the following pages to access the available preprocessing and tokenizer layers in
**KerasNLP**:
- [Preprocessing](https://keras.io/api/keras_nlp/preprocessing_layers/)
- [Tokenizers](https://keras.io/api/keras_nlp/tokenizers/)


```python
preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
    preset=CFG.preset,  # Name of the model
    sequence_length=CFG.sequence_length,  # Max sequence length, will be padded if shorter
)
```

Now, let's examine what the output shape of the preprocessing layer looks like. The
output shape of the layer can be represented as $(num\_choices, sequence\_length)$.


```python
outs = preprocessor(train_df.options.iloc[0])  # Process options for the first row

# Display the shape of each processed output
for k, v in outs.items():
    print(k, ":", v.shape)
```

<div class="k-default-codeblock">
```
token_ids : (4, 200)
padding_mask : (4, 200)

```
</div>
We'll use the `preprocessing_fn` function to transform each text option using the
`dataset.map(preprocessing_fn)` method.


```python

def preprocess_fn(text, label=None):
    text = preprocessor(text)  # Preprocess text
    return (
        (text, label) if label is not None else text
    )  # Return processed text and label if available

```

---
## Augmentation

In this notebook, we'll experiment with an interesting augmentation technique,
`option_shuffle`. Since we're providing the model with one option at a time, we can
introduce a shuffle to the order of options. For instance, options `[A, C, E, D, B]`
would be rearranged as `[D, B, A, E, C]`. This practice will help the model focus on the
content of the options themselves, rather than being influenced by their positions.

**Note:** Even though `option_shuffle` function is written in pure
tensorflow, it can be used with any backend (e.g. JAX, PyTorch) as it is only used
in `tf.data.Dataset` pipeline which is compatible with Keras 3 routines.


```python

def option_shuffle(options, labels, prob=0.50, seed=None):
    if tf.random.uniform([]) > prob:  # Shuffle probability check
        return options, labels
    # Shuffle indices of options and labels in the same order
    indices = tf.random.shuffle(tf.range(tf.shape(options)[0]), seed=seed)
    # Shuffle options and labels
    options = tf.gather(options, indices)
    labels = tf.gather(labels, indices)
    return options, labels

```

In the following function, we'll merge all augmentation functions to apply to the text.
These augmentations will be applied to the data using the `dataset.map(augment_fn)`
approach.


```python

def augment_fn(text, label=None):
    text, label = option_shuffle(text, label, prob=0.5)  # Shuffle the options
    return (text, label) if label is not None else text

```

---
## DataLoader

The code below sets up a robust data flow pipeline using `tf.data.Dataset` for data
processing. Notable aspects of `tf.data` include its ability to simplify pipeline
construction and represent components in sequences.

To learn more about `tf.data`, refer to this
[documentation](https://www.tensorflow.org/guide/data).


```python

def build_dataset(
    texts,
    labels=None,
    batch_size=32,
    cache=False,
    augment=False,
    repeat=False,
    shuffle=1024,
):
    AUTO = tf.data.AUTOTUNE  # AUTOTUNE option
    slices = (
        (texts,)
        if labels is None
        else (texts, keras.utils.to_categorical(labels, num_classes=4))
    )  # Create slices
    ds = tf.data.Dataset.from_tensor_slices(slices)  # Create dataset from slices
    ds = ds.cache() if cache else ds  # Cache dataset if enabled
    if augment:  # Apply augmentation if enabled
        ds = ds.map(augment_fn, num_parallel_calls=AUTO)
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTO)  # Map preprocessing function
    ds = ds.repeat() if repeat else ds  # Repeat dataset if enabled
    opt = tf.data.Options()  # Create dataset options
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)  # Shuffle dataset if enabled
        opt.experimental_deterministic = False
    ds = ds.with_options(opt)  # Set dataset options
    ds = ds.batch(batch_size, drop_remainder=True)  # Batch dataset
    ds = ds.prefetch(AUTO)  # Prefetch next batch
    return ds  # Return the built dataset

```

Now let's create train and valid dataloader using above funciton.


```python
# Build train dataloader
train_texts = train_df.options.tolist()  # Extract training texts
train_labels = train_df.label.tolist()  # Extract training labels
train_ds = build_dataset(
    train_texts,
    train_labels,
    batch_size=CFG.batch_size,
    cache=True,
    shuffle=True,
    repeat=True,
    augment=CFG.augment,
)

# Build valid dataloader
valid_texts = valid_df.options.tolist()  # Extract validation texts
valid_labels = valid_df.label.tolist()  # Extract validation labels
valid_ds = build_dataset(
    valid_texts,
    valid_labels,
    batch_size=CFG.batch_size,
    cache=True,
    shuffle=False,
    repeat=False,
    augment=False,
)

```

---
## LR Schedule

Implementing a learning rate scheduler is crucial for transfer learning. The learning
rate initiates at `lr_start` and gradually tapers down to `lr_min` using **cosine**
curve.

**Importance:** A well-structured learning rate schedule is essential for efficient model
training, ensuring optimal convergence and avoiding issues such as overshooting or
stagnation.


```python
import math


def get_lr_callback(batch_size=8, mode="cos", epochs=10, plot=False):
    lr_start, lr_max, lr_min = 1.0e-6, 0.6e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep = 2, 0

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            decay_total_epochs, decay_epoch_index = (
                epochs - lr_ramp_ep - lr_sus_ep + 3,
                epoch - lr_ramp_ep - lr_sus_ep,
            )
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(epochs),
            [lrfn(epoch) for epoch in np.arange(epochs)],
            marker="o",
        )
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.title("LR Scheduler")
        plt.show()

    return keras.callbacks.LearningRateScheduler(
        lrfn, verbose=False
    )  # Create lr callback


_ = get_lr_callback(CFG.batch_size, plot=True)
```


    
![png](/img/examples/nlp/multiple_choice_task_with_transfer_learning/multiple_choice_task_with_transfer_learning_32_0.png)
    


---
## Callbacks

The function below will gather all the training callbacks, such as `lr_scheduler`,
`model_checkpoint`.


```python

def get_callbacks():
    callbacks = []
    lr_cb = get_lr_callback(CFG.batch_size)  # Get lr callback
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        f"best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
    )  # Get Model checkpoint callback
    callbacks.extend([lr_cb, ckpt_cb])  # Add lr and checkpoint callbacks
    return callbacks  # Return the list of callbacks


callbacks = get_callbacks()
```

---
## MultipleChoice Model





### Pre-trained Models

The `KerasNLP` library provides comprehensive, ready-to-use implementations of popular
NLP model architectures. It features a variety of pre-trained models including `Bert`,
`Roberta`, `DebertaV3`, and more. In this notebook, we'll showcase the usage of
`DistillBert`. However, feel free to explore all available models in the [KerasNLP
documentation](https://keras.io/api/keras_nlp/models/). Also for a deeper understanding
of `KerasNLP`, refer to the informative [getting started
guide](https://keras.io/guides/keras_nlp/getting_started/).

Our approach involves using `keras_nlp.models.XXClassifier` to process each question and
option pari (e.g. (Q+A), (Q+B), etc.), generating logits. These logits are then combined
and passed through a softmax function to produce the final output.

### Classifier for Multiple-Choice Tasks

When dealing with multiple-choice questions, instead of giving the model the question and
all options together `(Q + A + B + C ...)`, we provide the model with one option at a
time along with the question. For instance, `(Q + A)`, `(Q + B)`, and so on. Once we have
the prediction scores (logits) for all options, we combine them using the `Softmax`
function to get the ultimate result. If we had given all options at once to the model,
the text's length would increase, making it harder for the model to handle. The picture
below illustrates this idea:

![Model Diagram](https://pbs.twimg.com/media/F3NUju_a8AAS8Fq?format=png&name=large)

<div align="center"><b> Picture Credict: </b> <a
href="https://twitter.com/johnowhitaker"> @johnowhitaker </a> </div></div><br>

From a coding perspective, remember that we use the same model for all five options, with
shared weights. Despite the figure suggesting five separate models, they are, in fact,
one model with shared weights. Another point to consider is the the input shapes of
Classifier and MultipleChoice.

* Input shape for **Multiple Choice**: $(batch\_size, num\_choices, seq\_length)$
* Input shape for **Classifier**: $(batch\_size, seq\_length)$

Certainly, it's clear that we can't directly give the data for the multiple-choice task
to the model because the input shapes don't match. To handle this, we'll use **slicing**.
This means we'll separate the features of each option, like $feature_{(Q + A)}$ and
$feature_{(Q + B)}$, and give them one by one to the NLP classifier. After we get the
prediction scores $logits_{(Q + A)}$ and $logits_{(Q + B)}$ for all the options, we'll
use the Softmax function, like $\operatorname{Softmax}([logits_{(Q + A)}, logits_{(Q +
B)}])$, to combine them. This final step helps us make the ultimate decision or choice.

> Note that in the classifier, we set `num_classes=1` instead of `5`. This is because the
classifier produces a single output for each option. When dealing with five options,
these individual outputs are joined together and then processed through a softmax
function to generate the final result, which has a dimension of `5`.


```python

# Selects one option from five
class SelectOption(keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        # Selects a specific slice from the inputs tensor
        return inputs[:, self.index, :]

    def get_config(self):
        # For serialize the model
        base_config = super().get_config()
        config = {
            "index": self.index,
        }
        return {**base_config, **config}


def build_model():
    # Define input layers
    inputs = {
        "token_ids": keras.Input(shape=(4, None), dtype="int32", name="token_ids"),
        "padding_mask": keras.Input(
            shape=(4, None), dtype="int32", name="padding_mask"
        ),
    }
    # Create a DebertaV3Classifier model
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        CFG.preset,
        preprocessor=None,
        num_classes=1,  # one output per one option, for five options total 5 outputs
    )
    logits = []
    # Loop through each option (Q+A), (Q+B) etc and compute associted logits
    for option_idx in range(4):
        option = {
            k: SelectOption(option_idx, name=f"{k}_{option_idx}")(v)
            for k, v in inputs.items()
        }
        logit = classifier(option)
        logits.append(logit)

    # Compute final output
    logits = keras.layers.Concatenate(axis=-1)(logits)
    outputs = keras.layers.Softmax(axis=-1)(logits)
    model = keras.Model(inputs, outputs)

    # Compile the model with optimizer, loss, and metrics
    model.compile(
        optimizer=keras.optimizers.AdamW(5e-6),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
        jit_compile=True,
    )
    return model


# Build the Build
model = build_model()
```

<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/deberta_v3/keras/deberta_v3_extra_small_en/2/download/config.json...

```
</div>
    
  0%|                                                                                                                                                         | 0.00/539 [00:00<?, ?B/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 539/539 [00:00<00:00, 1.16MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/deberta_v3/keras/deberta_v3_extra_small_en/2/download/model.weights.h5...

```
</div>
    
  0%|                                                                                                                                                        | 0.00/270M [00:00<?, ?B/s]

    
  0%|▌                                                                                                                                              | 1.00M/270M [00:00<01:38, 2.88MB/s]

    
  1%|██                                                                                                                                             | 4.00M/270M [00:00<00:26, 10.4MB/s]

    
  3%|████▊                                                                                                                                          | 9.00M/270M [00:00<00:12, 22.0MB/s]

    
  5%|██████▉                                                                                                                                        | 13.0M/270M [00:00<00:09, 27.0MB/s]

    
  7%|█████████▌                                                                                                                                     | 18.0M/270M [00:00<00:08, 33.0MB/s]

    
  8%|███████████▋                                                                                                                                   | 22.0M/270M [00:00<00:08, 29.6MB/s]

    
 10%|██████████████▎                                                                                                                                | 27.0M/270M [00:01<00:07, 33.4MB/s]

    
 11%|████████████████▍                                                                                                                              | 31.0M/270M [00:01<00:07, 31.7MB/s]

    
 13%|██████████████████▌                                                                                                                            | 35.0M/270M [00:01<00:09, 24.9MB/s]

    
 14%|████████████████████                                                                                                                           | 38.0M/270M [00:01<00:09, 25.3MB/s]

    
 16%|███████████████████████▎                                                                                                                       | 44.0M/270M [00:01<00:07, 31.2MB/s]

    
 19%|██████████████████████████▍                                                                                                                    | 50.0M/270M [00:01<00:06, 36.8MB/s]

    
 20%|████████████████████████████▌                                                                                                                  | 54.0M/270M [00:01<00:06, 37.0MB/s]

    
 22%|███████████████████████████████▏                                                                                                               | 59.0M/270M [00:02<00:06, 36.8MB/s]

    
 24%|█████████████████████████████████▉                                                                                                             | 64.0M/270M [00:02<00:05, 40.4MB/s]

    
 26%|████████████████████████████████████▌                                                                                                          | 69.0M/270M [00:02<00:07, 29.2MB/s]

    
 28%|███████████████████████████████████████▋                                                                                                       | 75.0M/270M [00:02<00:06, 33.8MB/s]

    
 30%|██████████████████████████████████████████▉                                                                                                    | 81.0M/270M [00:02<00:05, 39.5MB/s]

    
 32%|█████████████████████████████████████████████▌                                                                                                 | 86.0M/270M [00:02<00:04, 39.6MB/s]

    
 34%|████████████████████████████████████████████████▏                                                                                              | 91.0M/270M [00:03<00:05, 34.2MB/s]

    
 36%|███████████████████████████████████████████████████▎                                                                                           | 97.0M/270M [00:03<00:05, 32.7MB/s]

    
 38%|██████████████████████████████████████████████████████▉                                                                                         | 103M/270M [00:03<00:04, 36.7MB/s]

    
 40%|█████████████████████████████████████████████████████████                                                                                       | 107M/270M [00:03<00:05, 30.1MB/s]

    
 41%|███████████████████████████████████████████████████████████▋                                                                                    | 112M/270M [00:03<00:04, 34.3MB/s]

    
 43%|█████████████████████████████████████████████████████████████▊                                                                                  | 116M/270M [00:03<00:05, 30.8MB/s]

    
 45%|████████████████████████████████████████████████████████████████▌                                                                               | 121M/270M [00:04<00:05, 28.5MB/s]

    
 47%|███████████████████████████████████████████████████████████████████▋                                                                            | 127M/270M [00:04<00:04, 34.4MB/s]

    
 49%|█████████████████████████████████████████████████████████████████████▊                                                                          | 131M/270M [00:04<00:04, 31.4MB/s]

    
 51%|█████████████████████████████████████████████████████████████████████████                                                                       | 137M/270M [00:04<00:05, 27.6MB/s]

    
 53%|████████████████████████████████████████████████████████████████████████████▊                                                                   | 144M/270M [00:04<00:03, 35.0MB/s]

    
 55%|██████████████████████████████████████████████████████████████████████████████▉                                                                 | 148M/270M [00:05<00:04, 31.0MB/s]

    
 57%|█████████████████████████████████████████████████████████████████████████████████▌                                                              | 153M/270M [00:05<00:04, 30.6MB/s]

    
 59%|████████████████████████████████████████████████████████████████████████████████████▊                                                           | 159M/270M [00:05<00:03, 36.5MB/s]

    
 60%|██████████████████████████████████████████████████████████████████████████████████████▉                                                         | 163M/270M [00:05<00:03, 32.7MB/s]

    
 63%|██████████████████████████████████████████████████████████████████████████████████████████                                                      | 169M/270M [00:05<00:03, 30.7MB/s]

    
 65%|█████████████████████████████████████████████████████████████████████████████████████████████▎                                                  | 175M/270M [00:05<00:02, 36.5MB/s]

    
 66%|███████████████████████████████████████████████████████████████████████████████████████████████▍                                                | 179M/270M [00:05<00:02, 32.4MB/s]

    
 69%|██████████████████████████████████████████████████████████████████████████████████████████████████▋                                             | 185M/270M [00:06<00:02, 36.7MB/s]

    
 71%|█████████████████████████████████████████████████████████████████████████████████████████████████████▊                                          | 191M/270M [00:06<00:01, 41.7MB/s]

    
 73%|████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                       | 196M/270M [00:06<00:01, 40.4MB/s]

    
 74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                    | 201M/270M [00:06<00:01, 37.4MB/s]

    
 77%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                 | 207M/270M [00:06<00:01, 43.0MB/s]

    
 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████                               | 212M/270M [00:06<00:01, 34.8MB/s]

    
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                            | 217M/270M [00:07<00:01, 30.7MB/s]

    
 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                         | 222M/270M [00:07<00:01, 33.8MB/s]

    
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                       | 226M/270M [00:07<00:01, 33.9MB/s]

    
 86%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                    | 232M/270M [00:07<00:01, 39.8MB/s]

    
 88%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                 | 237M/270M [00:07<00:00, 35.1MB/s]

    
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌               | 241M/270M [00:07<00:00, 30.8MB/s]

    
 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏            | 246M/270M [00:07<00:00, 33.6MB/s]

    
 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎         | 252M/270M [00:08<00:00, 38.0MB/s]

    
 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████       | 257M/270M [00:08<00:00, 32.6MB/s]

    
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋    | 262M/270M [00:08<00:00, 36.2MB/s]

    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍| 269M/270M [00:08<00:00, 42.9MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 270M/270M [00:08<00:00, 33.4MB/s]

    
<div class="k-default-codeblock">
```
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras_nlp/src/models/backbone.py:37: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  return id(getattr(self, attr)) not in self._functional_layer_ids
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras_nlp/src/models/backbone.py:37: UserWarning: `layer.updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  return id(getattr(self, attr)) not in self._functional_layer_ids

```
</div>
Let's checkout the model summary to have a better insight on the model.


```python
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold"> Param # </span>┃<span style="font-weight: bold"> Connected to         </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ padding_mask        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ token_ids           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ padding_mask_0      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ token_ids_0         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ padding_mask_1      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ token_ids_1         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ padding_mask_2      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ token_ids_2         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ padding_mask_3      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ padding_mask[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ token_ids_3         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ token_ids[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SelectOption</span>)      │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ deberta_v3_classif… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │ <span style="color: #00af00; text-decoration-color: #00af00">70,830…</span> │ padding_mask_0[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">DebertaV3Classifi…</span> │                   │         │ token_ids_0[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],   │
│                     │                   │         │ padding_mask_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│                     │                   │         │ token_ids_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],   │
│                     │                   │         │ padding_mask_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│                     │                   │         │ token_ids_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],   │
│                     │                   │         │ padding_mask_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│                     │                   │         │ token_ids_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ concatenate         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)         │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ deberta_v3_classifi… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │                   │         │ deberta_v3_classifi… │
│                     │                   │         │ deberta_v3_classifi… │
│                     │                   │         │ deberta_v3_classifi… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ softmax (<span style="color: #0087ff; text-decoration-color: #0087ff">Softmax</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)         │       <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ concatenate[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">70,830,337</span> (270.20 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">70,830,337</span> (270.20 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Finally, let's check the model structure visually if everything is in place.


```python
keras.utils.plot_model(model, show_shapes=True)
```




    
![png](/img/examples/nlp/multiple_choice_task_with_transfer_learning/multiple_choice_task_with_transfer_learning_42_0.png)
    



---
## Training


```python
# Start training the model
history = model.fit(
    train_ds,
    epochs=CFG.epochs,
    validation_data=valid_ds,
    callbacks=callbacks,
    steps_per_epoch=int(len(train_df) / CFG.batch_size),
    verbose=1,
)
```

<div class="k-default-codeblock">
```
Epoch 1/5

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1704605251.846942   12962 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

```
</div>
    
  1/183 [..............................] - ETA: 16:51:51 - loss: 1.3865 - accuracy: 0.3750

<div class="k-default-codeblock">
```

```
</div>
  2/183 [..............................] - ETA: 44:52 - loss: 1.3652 - accuracy: 0.3750   

<div class="k-default-codeblock">
```

```
</div>
  3/183 [..............................] - ETA: 44:29 - loss: 1.3851 - accuracy: 0.2500

<div class="k-default-codeblock">
```

```
</div>
  4/183 [..............................] - ETA: 44:12 - loss: 1.3875 - accuracy: 0.2188

<div class="k-default-codeblock">
```

```
</div>
  5/183 [..............................] - ETA: 43:54 - loss: 1.3896 - accuracy: 0.2000

<div class="k-default-codeblock">
```

```
</div>
  6/183 [..............................] - ETA: 43:41 - loss: 1.3942 - accuracy: 0.1875

<div class="k-default-codeblock">
```

```
</div>
  7/183 [>.............................] - ETA: 43:25 - loss: 1.3928 - accuracy: 0.2143

<div class="k-default-codeblock">
```

```
</div>
  8/183 [>.............................] - ETA: 43:22 - loss: 1.3933 - accuracy: 0.2031

<div class="k-default-codeblock">
```

```
</div>
  9/183 [>.............................] - ETA: 42:54 - loss: 1.3934 - accuracy: 0.2083

<div class="k-default-codeblock">
```

```
</div>
 10/183 [>.............................] - ETA: 42:27 - loss: 1.3901 - accuracy: 0.2375

<div class="k-default-codeblock">
```

```
</div>
 11/183 [>.............................] - ETA: 42:04 - loss: 1.3898 - accuracy: 0.2500

<div class="k-default-codeblock">
```

```
</div>
 12/183 [>.............................] - ETA: 41:42 - loss: 1.3903 - accuracy: 0.2396

<div class="k-default-codeblock">
```

```
</div>
 13/183 [=>............................] - ETA: 41:21 - loss: 1.3900 - accuracy: 0.2500

<div class="k-default-codeblock">
```

```
</div>
 14/183 [=>............................] - ETA: 41:02 - loss: 1.3908 - accuracy: 0.2321

<div class="k-default-codeblock">
```

```
</div>
 15/183 [=>............................] - ETA: 40:45 - loss: 1.3919 - accuracy: 0.2167

<div class="k-default-codeblock">
```

```
</div>
 16/183 [=>............................] - ETA: 40:26 - loss: 1.3927 - accuracy: 0.2109

<div class="k-default-codeblock">
```

```
</div>
 17/183 [=>............................] - ETA: 40:08 - loss: 1.3935 - accuracy: 0.2132

<div class="k-default-codeblock">
```

```
</div>
 18/183 [=>............................] - ETA: 39:51 - loss: 1.3920 - accuracy: 0.2153

<div class="k-default-codeblock">
```

```
</div>
 19/183 [==>...........................] - ETA: 39:34 - loss: 1.3905 - accuracy: 0.2171

<div class="k-default-codeblock">
```

```
</div>
 20/183 [==>...........................] - ETA: 39:16 - loss: 1.3900 - accuracy: 0.2188

<div class="k-default-codeblock">
```

```
</div>
 21/183 [==>...........................] - ETA: 39:01 - loss: 1.3908 - accuracy: 0.2202

<div class="k-default-codeblock">
```

```
</div>
 22/183 [==>...........................] - ETA: 38:44 - loss: 1.3915 - accuracy: 0.2159

<div class="k-default-codeblock">
```

```
</div>
 23/183 [==>...........................] - ETA: 38:28 - loss: 1.3892 - accuracy: 0.2391

<div class="k-default-codeblock">
```

```
</div>
 24/183 [==>...........................] - ETA: 38:12 - loss: 1.3891 - accuracy: 0.2396

<div class="k-default-codeblock">
```

```
</div>
 25/183 [===>..........................] - ETA: 37:56 - loss: 1.3893 - accuracy: 0.2350

<div class="k-default-codeblock">
```

```
</div>
 26/183 [===>..........................] - ETA: 37:40 - loss: 1.3893 - accuracy: 0.2404

<div class="k-default-codeblock">
```

```
</div>
 27/183 [===>..........................] - ETA: 37:25 - loss: 1.3886 - accuracy: 0.2454

<div class="k-default-codeblock">
```

```
</div>
 28/183 [===>..........................] - ETA: 37:09 - loss: 1.3880 - accuracy: 0.2455

<div class="k-default-codeblock">
```

```
</div>
 29/183 [===>..........................] - ETA: 36:55 - loss: 1.3874 - accuracy: 0.2457

<div class="k-default-codeblock">
```

```
</div>
 30/183 [===>..........................] - ETA: 36:39 - loss: 1.3877 - accuracy: 0.2458

<div class="k-default-codeblock">
```

```
</div>
 31/183 [====>.........................] - ETA: 36:23 - loss: 1.3870 - accuracy: 0.2540

<div class="k-default-codeblock">
```

```
</div>
 32/183 [====>.........................] - ETA: 36:08 - loss: 1.3870 - accuracy: 0.2578

<div class="k-default-codeblock">
```

```
</div>
 33/183 [====>.........................] - ETA: 35:53 - loss: 1.3870 - accuracy: 0.2538

<div class="k-default-codeblock">
```

```
</div>
 34/183 [====>.........................] - ETA: 35:38 - loss: 1.3882 - accuracy: 0.2500

<div class="k-default-codeblock">
```

```
</div>
 35/183 [====>.........................] - ETA: 35:23 - loss: 1.3876 - accuracy: 0.2536

<div class="k-default-codeblock">
```

```
</div>
 36/183 [====>.........................] - ETA: 35:07 - loss: 1.3878 - accuracy: 0.2500

<div class="k-default-codeblock">
```

```
</div>
 37/183 [=====>........................] - ETA: 34:53 - loss: 1.3871 - accuracy: 0.2568

<div class="k-default-codeblock">
```

```
</div>
 38/183 [=====>........................] - ETA: 34:38 - loss: 1.3875 - accuracy: 0.2566

<div class="k-default-codeblock">
```

```
</div>
 39/183 [=====>........................] - ETA: 34:22 - loss: 1.3875 - accuracy: 0.2564

<div class="k-default-codeblock">
```

```
</div>
 40/183 [=====>........................] - ETA: 34:07 - loss: 1.3875 - accuracy: 0.2531

<div class="k-default-codeblock">
```

```
</div>
 41/183 [=====>........................] - ETA: 33:52 - loss: 1.3874 - accuracy: 0.2530

<div class="k-default-codeblock">
```

```
</div>
 42/183 [=====>........................] - ETA: 33:38 - loss: 1.3886 - accuracy: 0.2500

<div class="k-default-codeblock">
```

```
</div>
 43/183 [======>.......................] - ETA: 33:23 - loss: 1.3876 - accuracy: 0.2529

<div class="k-default-codeblock">
```

```
</div>
 44/183 [======>.......................] - ETA: 33:08 - loss: 1.3876 - accuracy: 0.2557

<div class="k-default-codeblock">
```

```
</div>
 45/183 [======>.......................] - ETA: 32:54 - loss: 1.3875 - accuracy: 0.2583

<div class="k-default-codeblock">
```

```
</div>
 46/183 [======>.......................] - ETA: 32:39 - loss: 1.3875 - accuracy: 0.2582

<div class="k-default-codeblock">
```

```
</div>
 47/183 [======>.......................] - ETA: 32:24 - loss: 1.3873 - accuracy: 0.2553

<div class="k-default-codeblock">
```

```
</div>
 48/183 [======>.......................] - ETA: 32:09 - loss: 1.3866 - accuracy: 0.2604

<div class="k-default-codeblock">
```

```
</div>
 49/183 [=======>......................] - ETA: 31:54 - loss: 1.3870 - accuracy: 0.2602

<div class="k-default-codeblock">
```

```
</div>
 50/183 [=======>......................] - ETA: 31:40 - loss: 1.3869 - accuracy: 0.2600

<div class="k-default-codeblock">
```

```
</div>
 51/183 [=======>......................] - ETA: 31:25 - loss: 1.3862 - accuracy: 0.2647

<div class="k-default-codeblock">
```

```
</div>
 52/183 [=======>......................] - ETA: 31:10 - loss: 1.3865 - accuracy: 0.2596

<div class="k-default-codeblock">
```

```
</div>
 53/183 [=======>......................] - ETA: 30:56 - loss: 1.3869 - accuracy: 0.2571

<div class="k-default-codeblock">
```

```
</div>
 54/183 [=======>......................] - ETA: 30:41 - loss: 1.3869 - accuracy: 0.2546

<div class="k-default-codeblock">
```

```
</div>
 55/183 [========>.....................] - ETA: 30:27 - loss: 1.3866 - accuracy: 0.2591

<div class="k-default-codeblock">
```

```
</div>
 56/183 [========>.....................] - ETA: 30:12 - loss: 1.3869 - accuracy: 0.2589

<div class="k-default-codeblock">
```

```
</div>
 57/183 [========>.....................] - ETA: 29:58 - loss: 1.3861 - accuracy: 0.2632

<div class="k-default-codeblock">
```

```
</div>
 58/183 [========>.....................] - ETA: 29:43 - loss: 1.3857 - accuracy: 0.2651

<div class="k-default-codeblock">
```

```
</div>
 59/183 [========>.....................] - ETA: 29:29 - loss: 1.3854 - accuracy: 0.2669

<div class="k-default-codeblock">
```

```
</div>
 60/183 [========>.....................] - ETA: 29:14 - loss: 1.3854 - accuracy: 0.2688

<div class="k-default-codeblock">
```

```
</div>
 61/183 [=========>....................] - ETA: 29:00 - loss: 1.3852 - accuracy: 0.2705

<div class="k-default-codeblock">
```

```
</div>
 62/183 [=========>....................] - ETA: 28:45 - loss: 1.3851 - accuracy: 0.2702

<div class="k-default-codeblock">
```

```
</div>
 63/183 [=========>....................] - ETA: 28:31 - loss: 1.3855 - accuracy: 0.2679

<div class="k-default-codeblock">
```

```
</div>
 64/183 [=========>....................] - ETA: 28:17 - loss: 1.3857 - accuracy: 0.2656

<div class="k-default-codeblock">
```

```
</div>
 65/183 [=========>....................] - ETA: 28:02 - loss: 1.3867 - accuracy: 0.2615

<div class="k-default-codeblock">
```

```
</div>
 66/183 [=========>....................] - ETA: 27:48 - loss: 1.3868 - accuracy: 0.2614

<div class="k-default-codeblock">
```

```
</div>
 67/183 [=========>....................] - ETA: 27:34 - loss: 1.3868 - accuracy: 0.2631

<div class="k-default-codeblock">
```

```
</div>
 68/183 [==========>...................] - ETA: 27:19 - loss: 1.3864 - accuracy: 0.2610

<div class="k-default-codeblock">
```

```
</div>
 69/183 [==========>...................] - ETA: 27:05 - loss: 1.3864 - accuracy: 0.2591

<div class="k-default-codeblock">
```

```
</div>
 70/183 [==========>...................] - ETA: 26:51 - loss: 1.3860 - accuracy: 0.2607

<div class="k-default-codeblock">
```

```
</div>
 71/183 [==========>...................] - ETA: 26:36 - loss: 1.3866 - accuracy: 0.2570

<div class="k-default-codeblock">
```

```
</div>
 72/183 [==========>...................] - ETA: 26:22 - loss: 1.3858 - accuracy: 0.2622

<div class="k-default-codeblock">
```

```
</div>
 73/183 [==========>...................] - ETA: 26:07 - loss: 1.3854 - accuracy: 0.2637

<div class="k-default-codeblock">
```

```
</div>
 74/183 [===========>..................] - ETA: 25:53 - loss: 1.3858 - accuracy: 0.2601

<div class="k-default-codeblock">
```

```
</div>
 75/183 [===========>..................] - ETA: 25:39 - loss: 1.3859 - accuracy: 0.2583

<div class="k-default-codeblock">
```

```
</div>
 76/183 [===========>..................] - ETA: 25:24 - loss: 1.3859 - accuracy: 0.2599

<div class="k-default-codeblock">
```

```
</div>
 77/183 [===========>..................] - ETA: 25:10 - loss: 1.3860 - accuracy: 0.2597

<div class="k-default-codeblock">
```

```
</div>
 78/183 [===========>..................] - ETA: 24:55 - loss: 1.3862 - accuracy: 0.2580

<div class="k-default-codeblock">
```

```
</div>
 79/183 [===========>..................] - ETA: 24:41 - loss: 1.3855 - accuracy: 0.2611

<div class="k-default-codeblock">
```

```
</div>
 80/183 [============>.................] - ETA: 24:27 - loss: 1.3857 - accuracy: 0.2625

<div class="k-default-codeblock">
```

```
</div>
 81/183 [============>.................] - ETA: 24:12 - loss: 1.3853 - accuracy: 0.2639

<div class="k-default-codeblock">
```

```
</div>
 82/183 [============>.................] - ETA: 23:58 - loss: 1.3854 - accuracy: 0.2622

<div class="k-default-codeblock">
```

```
</div>
 83/183 [============>.................] - ETA: 23:43 - loss: 1.3854 - accuracy: 0.2636

<div class="k-default-codeblock">
```

```
</div>
 84/183 [============>.................] - ETA: 23:29 - loss: 1.3857 - accuracy: 0.2634

<div class="k-default-codeblock">
```

```
</div>
 85/183 [============>.................] - ETA: 23:15 - loss: 1.3853 - accuracy: 0.2647

<div class="k-default-codeblock">
```

```
</div>
 86/183 [=============>................] - ETA: 23:00 - loss: 1.3852 - accuracy: 0.2689

<div class="k-default-codeblock">
```

```
</div>
 87/183 [=============>................] - ETA: 22:46 - loss: 1.3852 - accuracy: 0.2687

<div class="k-default-codeblock">
```

```
</div>
 88/183 [=============>................] - ETA: 22:32 - loss: 1.3852 - accuracy: 0.2685

<div class="k-default-codeblock">
```

```
</div>
 89/183 [=============>................] - ETA: 22:17 - loss: 1.3852 - accuracy: 0.2711

<div class="k-default-codeblock">
```

```
</div>
 90/183 [=============>................] - ETA: 22:03 - loss: 1.3852 - accuracy: 0.2694

<div class="k-default-codeblock">
```

```
</div>
 91/183 [=============>................] - ETA: 21:49 - loss: 1.3851 - accuracy: 0.2720

<div class="k-default-codeblock">
```

```
</div>
 92/183 [==============>...............] - ETA: 21:35 - loss: 1.3851 - accuracy: 0.2717

<div class="k-default-codeblock">
```

```
</div>
 93/183 [==============>...............] - ETA: 21:20 - loss: 1.3853 - accuracy: 0.2702

<div class="k-default-codeblock">
```

```
</div>
 94/183 [==============>...............] - ETA: 21:06 - loss: 1.3856 - accuracy: 0.2699

<div class="k-default-codeblock">
```

```
</div>
 95/183 [==============>...............] - ETA: 20:52 - loss: 1.3855 - accuracy: 0.2711

<div class="k-default-codeblock">
```

```
</div>
 96/183 [==============>...............] - ETA: 20:37 - loss: 1.3856 - accuracy: 0.2695

<div class="k-default-codeblock">
```

```
</div>
 97/183 [==============>...............] - ETA: 20:23 - loss: 1.3855 - accuracy: 0.2706

<div class="k-default-codeblock">
```

```
</div>
 98/183 [===============>..............] - ETA: 20:09 - loss: 1.3852 - accuracy: 0.2730

<div class="k-default-codeblock">
```

```
</div>
 99/183 [===============>..............] - ETA: 19:54 - loss: 1.3853 - accuracy: 0.2715

<div class="k-default-codeblock">
```

```
</div>
100/183 [===============>..............] - ETA: 19:40 - loss: 1.3854 - accuracy: 0.2713

<div class="k-default-codeblock">
```

```
</div>
101/183 [===============>..............] - ETA: 19:26 - loss: 1.3856 - accuracy: 0.2698

<div class="k-default-codeblock">
```

```
</div>
102/183 [===============>..............] - ETA: 19:12 - loss: 1.3856 - accuracy: 0.2684

<div class="k-default-codeblock">
```

```
</div>
103/183 [===============>..............] - ETA: 18:57 - loss: 1.3854 - accuracy: 0.2694

<div class="k-default-codeblock">
```

```
</div>
104/183 [================>.............] - ETA: 18:43 - loss: 1.3852 - accuracy: 0.2716

<div class="k-default-codeblock">
```

```
</div>
105/183 [================>.............] - ETA: 18:29 - loss: 1.3854 - accuracy: 0.2714

<div class="k-default-codeblock">
```

```
</div>
106/183 [================>.............] - ETA: 18:15 - loss: 1.3851 - accuracy: 0.2748

<div class="k-default-codeblock">
```

```
</div>
107/183 [================>.............] - ETA: 18:00 - loss: 1.3852 - accuracy: 0.2745

<div class="k-default-codeblock">
```

```
</div>
108/183 [================>.............] - ETA: 17:46 - loss: 1.3853 - accuracy: 0.2720

<div class="k-default-codeblock">
```

```
</div>
109/183 [================>.............] - ETA: 17:32 - loss: 1.3854 - accuracy: 0.2718

<div class="k-default-codeblock">
```

```
</div>
110/183 [=================>............] - ETA: 17:18 - loss: 1.3855 - accuracy: 0.2716

<div class="k-default-codeblock">
```

```
</div>
111/183 [=================>............] - ETA: 17:03 - loss: 1.3856 - accuracy: 0.2703

<div class="k-default-codeblock">
```

```
</div>
112/183 [=================>............] - ETA: 16:49 - loss: 1.3859 - accuracy: 0.2701

<div class="k-default-codeblock">
```

```
</div>
113/183 [=================>............] - ETA: 16:35 - loss: 1.3859 - accuracy: 0.2699

<div class="k-default-codeblock">
```

```
</div>
114/183 [=================>............] - ETA: 16:21 - loss: 1.3859 - accuracy: 0.2697

<div class="k-default-codeblock">
```

```
</div>
115/183 [=================>............] - ETA: 16:06 - loss: 1.3862 - accuracy: 0.2707

<div class="k-default-codeblock">
```

```
</div>
116/183 [==================>...........] - ETA: 15:52 - loss: 1.3858 - accuracy: 0.2737

<div class="k-default-codeblock">
```

```
</div>
117/183 [==================>...........] - ETA: 15:38 - loss: 1.3857 - accuracy: 0.2746

<div class="k-default-codeblock">
```

```
</div>
118/183 [==================>...........] - ETA: 15:24 - loss: 1.3856 - accuracy: 0.2744

<div class="k-default-codeblock">
```

```
</div>
119/183 [==================>...........] - ETA: 15:09 - loss: 1.3858 - accuracy: 0.2731

<div class="k-default-codeblock">
```

```
</div>
120/183 [==================>...........] - ETA: 14:55 - loss: 1.3861 - accuracy: 0.2719

<div class="k-default-codeblock">
```

```
</div>
121/183 [==================>...........] - ETA: 14:41 - loss: 1.3862 - accuracy: 0.2707

<div class="k-default-codeblock">
```

```
</div>
122/183 [===================>..........] - ETA: 14:27 - loss: 1.3862 - accuracy: 0.2705

<div class="k-default-codeblock">
```

```
</div>
123/183 [===================>..........] - ETA: 14:12 - loss: 1.3861 - accuracy: 0.2703

<div class="k-default-codeblock">
```

```
</div>
124/183 [===================>..........] - ETA: 13:58 - loss: 1.3860 - accuracy: 0.2692

<div class="k-default-codeblock">
```

```
</div>
125/183 [===================>..........] - ETA: 13:44 - loss: 1.3861 - accuracy: 0.2670

<div class="k-default-codeblock">
```

```
</div>
126/183 [===================>..........] - ETA: 13:30 - loss: 1.3860 - accuracy: 0.2679

<div class="k-default-codeblock">
```

```
</div>
127/183 [===================>..........] - ETA: 13:15 - loss: 1.3858 - accuracy: 0.2677

<div class="k-default-codeblock">
```

```
</div>
128/183 [===================>..........] - ETA: 13:01 - loss: 1.3856 - accuracy: 0.2715

<div class="k-default-codeblock">
```

```
</div>
129/183 [====================>.........] - ETA: 12:47 - loss: 1.3855 - accuracy: 0.2723

<div class="k-default-codeblock">
```

```
</div>
130/183 [====================>.........] - ETA: 12:33 - loss: 1.3854 - accuracy: 0.2731

<div class="k-default-codeblock">
```

```
</div>
131/183 [====================>.........] - ETA: 12:19 - loss: 1.3854 - accuracy: 0.2719

<div class="k-default-codeblock">
```

```
</div>
132/183 [====================>.........] - ETA: 12:04 - loss: 1.3854 - accuracy: 0.2718

<div class="k-default-codeblock">
```

```
</div>
133/183 [====================>.........] - ETA: 11:50 - loss: 1.3854 - accuracy: 0.2707

<div class="k-default-codeblock">
```

```
</div>
134/183 [====================>.........] - ETA: 11:36 - loss: 1.3854 - accuracy: 0.2705

<div class="k-default-codeblock">
```

```
</div>
135/183 [=====================>........] - ETA: 11:22 - loss: 1.3853 - accuracy: 0.2694

<div class="k-default-codeblock">
```

```
</div>
136/183 [=====================>........] - ETA: 11:07 - loss: 1.3854 - accuracy: 0.2693

<div class="k-default-codeblock">
```

```
</div>
137/183 [=====================>........] - ETA: 10:53 - loss: 1.3854 - accuracy: 0.2710

<div class="k-default-codeblock">
```

```
</div>
138/183 [=====================>........] - ETA: 10:39 - loss: 1.3856 - accuracy: 0.2717

<div class="k-default-codeblock">
```

```
</div>
139/183 [=====================>........] - ETA: 10:25 - loss: 1.3855 - accuracy: 0.2743

<div class="k-default-codeblock">
```

```
</div>
140/183 [=====================>........] - ETA: 10:10 - loss: 1.3853 - accuracy: 0.2741

<div class="k-default-codeblock">
```

```
</div>
141/183 [======================>.......] - ETA: 9:56 - loss: 1.3854 - accuracy: 0.2730 

<div class="k-default-codeblock">
```

```
</div>
142/183 [======================>.......] - ETA: 9:42 - loss: 1.3854 - accuracy: 0.2720

<div class="k-default-codeblock">
```

```
</div>
143/183 [======================>.......] - ETA: 9:28 - loss: 1.3852 - accuracy: 0.2719

<div class="k-default-codeblock">
```

```
</div>
144/183 [======================>.......] - ETA: 9:14 - loss: 1.3852 - accuracy: 0.2717

<div class="k-default-codeblock">
```

```
</div>
145/183 [======================>.......] - ETA: 8:59 - loss: 1.3852 - accuracy: 0.2724

<div class="k-default-codeblock">
```

```
</div>
146/183 [======================>.......] - ETA: 8:45 - loss: 1.3853 - accuracy: 0.2723

<div class="k-default-codeblock">
```

```
</div>
147/183 [=======================>......] - ETA: 8:31 - loss: 1.3852 - accuracy: 0.2730

<div class="k-default-codeblock">
```

```
</div>
148/183 [=======================>......] - ETA: 8:17 - loss: 1.3852 - accuracy: 0.2736

<div class="k-default-codeblock">
```

```
</div>
149/183 [=======================>......] - ETA: 8:03 - loss: 1.3851 - accuracy: 0.2743

<div class="k-default-codeblock">
```

```
</div>
150/183 [=======================>......] - ETA: 7:48 - loss: 1.3848 - accuracy: 0.2758

<div class="k-default-codeblock">
```

```
</div>
151/183 [=======================>......] - ETA: 7:34 - loss: 1.3847 - accuracy: 0.2757

<div class="k-default-codeblock">
```

```
</div>
152/183 [=======================>......] - ETA: 7:20 - loss: 1.3847 - accuracy: 0.2755

<div class="k-default-codeblock">
```

```
</div>
153/183 [========================>.....] - ETA: 7:06 - loss: 1.3847 - accuracy: 0.2753

<div class="k-default-codeblock">
```

```
</div>
154/183 [========================>.....] - ETA: 6:52 - loss: 1.3848 - accuracy: 0.2744

<div class="k-default-codeblock">
```

```
</div>
155/183 [========================>.....] - ETA: 6:37 - loss: 1.3847 - accuracy: 0.2742

<div class="k-default-codeblock">
```

```
</div>
156/183 [========================>.....] - ETA: 6:23 - loss: 1.3848 - accuracy: 0.2740

<div class="k-default-codeblock">
```

```
</div>
157/183 [========================>.....] - ETA: 6:09 - loss: 1.3848 - accuracy: 0.2731

<div class="k-default-codeblock">
```

```
</div>
158/183 [========================>.....] - ETA: 5:55 - loss: 1.3848 - accuracy: 0.2737

<div class="k-default-codeblock">
```

```
</div>
159/183 [=========================>....] - ETA: 5:40 - loss: 1.3848 - accuracy: 0.2752

<div class="k-default-codeblock">
```

```
</div>
160/183 [=========================>....] - ETA: 5:26 - loss: 1.3849 - accuracy: 0.2742

<div class="k-default-codeblock">
```

```
</div>
161/183 [=========================>....] - ETA: 5:12 - loss: 1.3845 - accuracy: 0.2772

<div class="k-default-codeblock">
```

```
</div>
162/183 [=========================>....] - ETA: 4:58 - loss: 1.3847 - accuracy: 0.2770

<div class="k-default-codeblock">
```

```
</div>
163/183 [=========================>....] - ETA: 4:44 - loss: 1.3849 - accuracy: 0.2768

<div class="k-default-codeblock">
```

```
</div>
164/183 [=========================>....] - ETA: 4:29 - loss: 1.3848 - accuracy: 0.2774

<div class="k-default-codeblock">
```

```
</div>
165/183 [==========================>...] - ETA: 4:15 - loss: 1.3848 - accuracy: 0.2780

<div class="k-default-codeblock">
```

```
</div>
166/183 [==========================>...] - ETA: 4:01 - loss: 1.3848 - accuracy: 0.2764

<div class="k-default-codeblock">
```

```
</div>
167/183 [==========================>...] - ETA: 3:47 - loss: 1.3847 - accuracy: 0.2762

<div class="k-default-codeblock">
```

```
</div>
168/183 [==========================>...] - ETA: 3:33 - loss: 1.3848 - accuracy: 0.2753

<div class="k-default-codeblock">
```

```
</div>
169/183 [==========================>...] - ETA: 3:18 - loss: 1.3848 - accuracy: 0.2759

<div class="k-default-codeblock">
```

```
</div>
170/183 [==========================>...] - ETA: 3:04 - loss: 1.3846 - accuracy: 0.2779

<div class="k-default-codeblock">
```

```
</div>
171/183 [===========================>..] - ETA: 2:50 - loss: 1.3849 - accuracy: 0.2770

<div class="k-default-codeblock">
```

```
</div>
172/183 [===========================>..] - ETA: 2:36 - loss: 1.3850 - accuracy: 0.2754

<div class="k-default-codeblock">
```

```
</div>
173/183 [===========================>..] - ETA: 2:22 - loss: 1.3849 - accuracy: 0.2760

<div class="k-default-codeblock">
```

```
</div>
174/183 [===========================>..] - ETA: 2:07 - loss: 1.3849 - accuracy: 0.2766

<div class="k-default-codeblock">
```

```
</div>
175/183 [===========================>..] - ETA: 1:53 - loss: 1.3849 - accuracy: 0.2779

<div class="k-default-codeblock">
```

```
</div>
176/183 [===========================>..] - ETA: 1:39 - loss: 1.3846 - accuracy: 0.2784

<div class="k-default-codeblock">
```

```
</div>
177/183 [============================>.] - ETA: 1:25 - loss: 1.3846 - accuracy: 0.2797

<div class="k-default-codeblock">
```

```
</div>
178/183 [============================>.] - ETA: 1:11 - loss: 1.3844 - accuracy: 0.2809

<div class="k-default-codeblock">
```

```
</div>
179/183 [============================>.] - ETA: 56s - loss: 1.3844 - accuracy: 0.2807 

<div class="k-default-codeblock">
```

```
</div>
180/183 [============================>.] - ETA: 42s - loss: 1.3843 - accuracy: 0.2806

<div class="k-default-codeblock">
```

```
</div>
181/183 [============================>.] - ETA: 28s - loss: 1.3843 - accuracy: 0.2818

<div class="k-default-codeblock">
```

```
</div>
182/183 [============================>.] - ETA: 14s - loss: 1.3843 - accuracy: 0.2823

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - ETA: 0s - loss: 1.3842 - accuracy: 0.2828 

<div class="k-default-codeblock">
```
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras_nlp/src/models/task.py:47: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  return id(getattr(self, attr)) not in self._functional_layer_ids
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras_nlp/src/models/task.py:47: UserWarning: `layer.updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  return id(getattr(self, attr)) not in self._functional_layer_ids
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras_nlp/src/models/backbone.py:37: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  return id(getattr(self, attr)) not in self._functional_layer_ids
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras_nlp/src/models/backbone.py:37: UserWarning: `layer.updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  return id(getattr(self, attr)) not in self._functional_layer_ids


```
</div>
183/183 [==============================] - 3111s 15s/step - loss: 1.3842 - accuracy: 0.2828 - val_loss: 1.3755 - val_accuracy: 0.5225 - lr: 1.0000e-06


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/183 [..............................] - ETA: 46:52 - loss: 1.3583 - accuracy: 0.5000

<div class="k-default-codeblock">
```

```
</div>
  2/183 [..............................] - ETA: 43:21 - loss: 1.3603 - accuracy: 0.3750

<div class="k-default-codeblock">
```

```
</div>
  3/183 [..............................] - ETA: 43:12 - loss: 1.3608 - accuracy: 0.3750

<div class="k-default-codeblock">
```

```
</div>
  4/183 [..............................] - ETA: 42:59 - loss: 1.3601 - accuracy: 0.4062

<div class="k-default-codeblock">
```

```
</div>
  5/183 [..............................] - ETA: 42:33 - loss: 1.3611 - accuracy: 0.3750

<div class="k-default-codeblock">
```

```
</div>
  6/183 [..............................] - ETA: 42:14 - loss: 1.3652 - accuracy: 0.3333

<div class="k-default-codeblock">
```

```
</div>
  7/183 [>.............................] - ETA: 41:54 - loss: 1.3719 - accuracy: 0.2857

<div class="k-default-codeblock">
```

```
</div>
  8/183 [>.............................] - ETA: 41:38 - loss: 1.3726 - accuracy: 0.2969

<div class="k-default-codeblock">
```

```
</div>
  9/183 [>.............................] - ETA: 41:22 - loss: 1.3738 - accuracy: 0.2917

<div class="k-default-codeblock">
```

```
</div>
 10/183 [>.............................] - ETA: 41:05 - loss: 1.3772 - accuracy: 0.2875

<div class="k-default-codeblock">
```

```
</div>
 11/183 [>.............................] - ETA: 40:54 - loss: 1.3764 - accuracy: 0.2955

<div class="k-default-codeblock">
```

```
</div>
 12/183 [>.............................] - ETA: 40:37 - loss: 1.3753 - accuracy: 0.3021

<div class="k-default-codeblock">
```

```
</div>
 13/183 [=>............................] - ETA: 40:22 - loss: 1.3747 - accuracy: 0.3173

<div class="k-default-codeblock">
```

```
</div>
 14/183 [=>............................] - ETA: 40:06 - loss: 1.3742 - accuracy: 0.3214

<div class="k-default-codeblock">
```

```
</div>
 15/183 [=>............................] - ETA: 39:50 - loss: 1.3715 - accuracy: 0.3333

<div class="k-default-codeblock">
```

```
</div>
 16/183 [=>............................] - ETA: 39:34 - loss: 1.3716 - accuracy: 0.3359

<div class="k-default-codeblock">
```

```
</div>
 17/183 [=>............................] - ETA: 39:19 - loss: 1.3721 - accuracy: 0.3382

<div class="k-default-codeblock">
```

```
</div>
 18/183 [=>............................] - ETA: 39:03 - loss: 1.3735 - accuracy: 0.3194

<div class="k-default-codeblock">
```

```
</div>
 19/183 [==>...........................] - ETA: 38:48 - loss: 1.3746 - accuracy: 0.3158

<div class="k-default-codeblock">
```

```
</div>
 20/183 [==>...........................] - ETA: 38:33 - loss: 1.3740 - accuracy: 0.3250

<div class="k-default-codeblock">
```

```
</div>
 21/183 [==>...........................] - ETA: 38:18 - loss: 1.3744 - accuracy: 0.3274

<div class="k-default-codeblock">
```

```
</div>
 22/183 [==>...........................] - ETA: 38:04 - loss: 1.3718 - accuracy: 0.3352

<div class="k-default-codeblock">
```

```
</div>
 23/183 [==>...........................] - ETA: 37:49 - loss: 1.3711 - accuracy: 0.3424

<div class="k-default-codeblock">
```

```
</div>
 24/183 [==>...........................] - ETA: 37:34 - loss: 1.3712 - accuracy: 0.3333

<div class="k-default-codeblock">
```

```
</div>
 25/183 [===>..........................] - ETA: 37:20 - loss: 1.3700 - accuracy: 0.3400

<div class="k-default-codeblock">
```

```
</div>
 26/183 [===>..........................] - ETA: 37:05 - loss: 1.3709 - accuracy: 0.3365

<div class="k-default-codeblock">
```

```
</div>
 27/183 [===>..........................] - ETA: 36:51 - loss: 1.3708 - accuracy: 0.3380

<div class="k-default-codeblock">
```

```
</div>
 28/183 [===>..........................] - ETA: 36:36 - loss: 1.3709 - accuracy: 0.3348

<div class="k-default-codeblock">
```

```
</div>
 29/183 [===>..........................] - ETA: 36:22 - loss: 1.3708 - accuracy: 0.3362

<div class="k-default-codeblock">
```

```
</div>
 30/183 [===>..........................] - ETA: 36:07 - loss: 1.3706 - accuracy: 0.3375

<div class="k-default-codeblock">
```

```
</div>
 31/183 [====>.........................] - ETA: 35:53 - loss: 1.3708 - accuracy: 0.3387

<div class="k-default-codeblock">
```

```
</div>
 32/183 [====>.........................] - ETA: 35:38 - loss: 1.3699 - accuracy: 0.3359

<div class="k-default-codeblock">
```

```
</div>
 33/183 [====>.........................] - ETA: 35:24 - loss: 1.3704 - accuracy: 0.3295

<div class="k-default-codeblock">
```

```
</div>
 34/183 [====>.........................] - ETA: 35:10 - loss: 1.3712 - accuracy: 0.3199

<div class="k-default-codeblock">
```

```
</div>
 35/183 [====>.........................] - ETA: 34:55 - loss: 1.3718 - accuracy: 0.3214

<div class="k-default-codeblock">
```

```
</div>
 36/183 [====>.........................] - ETA: 34:41 - loss: 1.3712 - accuracy: 0.3264

<div class="k-default-codeblock">
```

```
</div>
 37/183 [=====>........................] - ETA: 34:27 - loss: 1.3713 - accuracy: 0.3243

<div class="k-default-codeblock">
```

```
</div>
 38/183 [=====>........................] - ETA: 34:13 - loss: 1.3701 - accuracy: 0.3289

<div class="k-default-codeblock">
```

```
</div>
 39/183 [=====>........................] - ETA: 33:58 - loss: 1.3701 - accuracy: 0.3237

<div class="k-default-codeblock">
```

```
</div>
 40/183 [=====>........................] - ETA: 33:44 - loss: 1.3697 - accuracy: 0.3250

<div class="k-default-codeblock">
```

```
</div>
 41/183 [=====>........................] - ETA: 33:30 - loss: 1.3700 - accuracy: 0.3201

<div class="k-default-codeblock">
```

```
</div>
 42/183 [=====>........................] - ETA: 33:16 - loss: 1.3699 - accuracy: 0.3214

<div class="k-default-codeblock">
```

```
</div>
 43/183 [======>.......................] - ETA: 33:01 - loss: 1.3695 - accuracy: 0.3227

<div class="k-default-codeblock">
```

```
</div>
 44/183 [======>.......................] - ETA: 32:48 - loss: 1.3689 - accuracy: 0.3210

<div class="k-default-codeblock">
```

```
</div>
 45/183 [======>.......................] - ETA: 32:33 - loss: 1.3685 - accuracy: 0.3222

<div class="k-default-codeblock">
```

```
</div>
 46/183 [======>.......................] - ETA: 32:19 - loss: 1.3692 - accuracy: 0.3207

<div class="k-default-codeblock">
```

```
</div>
 47/183 [======>.......................] - ETA: 32:05 - loss: 1.3682 - accuracy: 0.3271

<div class="k-default-codeblock">
```

```
</div>
 48/183 [======>.......................] - ETA: 31:50 - loss: 1.3682 - accuracy: 0.3281

<div class="k-default-codeblock">
```

```
</div>
 49/183 [=======>......................] - ETA: 31:36 - loss: 1.3685 - accuracy: 0.3291

<div class="k-default-codeblock">
```

```
</div>
 50/183 [=======>......................] - ETA: 31:22 - loss: 1.3682 - accuracy: 0.3250

<div class="k-default-codeblock">
```

```
</div>
 51/183 [=======>......................] - ETA: 31:08 - loss: 1.3690 - accuracy: 0.3235

<div class="k-default-codeblock">
```

```
</div>
 52/183 [=======>......................] - ETA: 30:53 - loss: 1.3690 - accuracy: 0.3221

<div class="k-default-codeblock">
```

```
</div>
 53/183 [=======>......................] - ETA: 30:39 - loss: 1.3692 - accuracy: 0.3231

<div class="k-default-codeblock">
```

```
</div>
 54/183 [=======>......................] - ETA: 30:25 - loss: 1.3690 - accuracy: 0.3241

<div class="k-default-codeblock">
```

```
</div>
 55/183 [========>.....................] - ETA: 30:11 - loss: 1.3688 - accuracy: 0.3273

<div class="k-default-codeblock">
```

```
</div>
 56/183 [========>.....................] - ETA: 29:56 - loss: 1.3685 - accuracy: 0.3281

<div class="k-default-codeblock">
```

```
</div>
 57/183 [========>.....................] - ETA: 29:42 - loss: 1.3679 - accuracy: 0.3311

<div class="k-default-codeblock">
```

```
</div>
 58/183 [========>.....................] - ETA: 29:28 - loss: 1.3671 - accuracy: 0.3319

<div class="k-default-codeblock">
```

```
</div>
 59/183 [========>.....................] - ETA: 29:14 - loss: 1.3670 - accuracy: 0.3326

<div class="k-default-codeblock">
```

```
</div>
 60/183 [========>.....................] - ETA: 29:00 - loss: 1.3672 - accuracy: 0.3313

<div class="k-default-codeblock">
```

```
</div>
 61/183 [=========>....................] - ETA: 28:45 - loss: 1.3673 - accuracy: 0.3279

<div class="k-default-codeblock">
```

```
</div>
 62/183 [=========>....................] - ETA: 28:31 - loss: 1.3669 - accuracy: 0.3286

<div class="k-default-codeblock">
```

```
</div>
 63/183 [=========>....................] - ETA: 28:17 - loss: 1.3667 - accuracy: 0.3234

<div class="k-default-codeblock">
```

```
</div>
 64/183 [=========>....................] - ETA: 28:03 - loss: 1.3669 - accuracy: 0.3223

<div class="k-default-codeblock">
```

```
</div>
 65/183 [=========>....................] - ETA: 27:49 - loss: 1.3662 - accuracy: 0.3231

<div class="k-default-codeblock">
```

```
</div>
 66/183 [=========>....................] - ETA: 27:35 - loss: 1.3663 - accuracy: 0.3239

<div class="k-default-codeblock">
```

```
</div>
 67/183 [=========>....................] - ETA: 27:20 - loss: 1.3659 - accuracy: 0.3265

<div class="k-default-codeblock">
```

```
</div>
 68/183 [==========>...................] - ETA: 27:06 - loss: 1.3657 - accuracy: 0.3272

<div class="k-default-codeblock">
```

```
</div>
 69/183 [==========>...................] - ETA: 26:52 - loss: 1.3648 - accuracy: 0.3315

<div class="k-default-codeblock">
```

```
</div>
 70/183 [==========>...................] - ETA: 26:38 - loss: 1.3654 - accuracy: 0.3286

<div class="k-default-codeblock">
```

```
</div>
 71/183 [==========>...................] - ETA: 26:24 - loss: 1.3644 - accuracy: 0.3327

<div class="k-default-codeblock">
```

```
</div>
 72/183 [==========>...................] - ETA: 26:10 - loss: 1.3645 - accuracy: 0.3316

<div class="k-default-codeblock">
```

```
</div>
 73/183 [==========>...................] - ETA: 25:56 - loss: 1.3645 - accuracy: 0.3305

<div class="k-default-codeblock">
```

```
</div>
 74/183 [===========>..................] - ETA: 25:41 - loss: 1.3641 - accuracy: 0.3311

<div class="k-default-codeblock">
```

```
</div>
 75/183 [===========>..................] - ETA: 25:27 - loss: 1.3628 - accuracy: 0.3367

<div class="k-default-codeblock">
```

```
</div>
 76/183 [===========>..................] - ETA: 25:13 - loss: 1.3616 - accuracy: 0.3421

<div class="k-default-codeblock">
```

```
</div>
 77/183 [===========>..................] - ETA: 24:59 - loss: 1.3614 - accuracy: 0.3425

<div class="k-default-codeblock">
```

```
</div>
 78/183 [===========>..................] - ETA: 24:45 - loss: 1.3616 - accuracy: 0.3397

<div class="k-default-codeblock">
```

```
</div>
 79/183 [===========>..................] - ETA: 24:31 - loss: 1.3610 - accuracy: 0.3418

<div class="k-default-codeblock">
```

```
</div>
 80/183 [============>.................] - ETA: 24:17 - loss: 1.3608 - accuracy: 0.3438

<div class="k-default-codeblock">
```

```
</div>
 81/183 [============>.................] - ETA: 24:03 - loss: 1.3599 - accuracy: 0.3441

<div class="k-default-codeblock">
```

```
</div>
 82/183 [============>.................] - ETA: 23:49 - loss: 1.3591 - accuracy: 0.3460

<div class="k-default-codeblock">
```

```
</div>
 83/183 [============>.................] - ETA: 23:35 - loss: 1.3589 - accuracy: 0.3479

<div class="k-default-codeblock">
```

```
</div>
 84/183 [============>.................] - ETA: 23:21 - loss: 1.3583 - accuracy: 0.3482

<div class="k-default-codeblock">
```

```
</div>
 85/183 [============>.................] - ETA: 23:07 - loss: 1.3575 - accuracy: 0.3515

<div class="k-default-codeblock">
```

```
</div>
 86/183 [=============>................] - ETA: 22:53 - loss: 1.3558 - accuracy: 0.3561

<div class="k-default-codeblock">
```

```
</div>
 87/183 [=============>................] - ETA: 22:39 - loss: 1.3560 - accuracy: 0.3549

<div class="k-default-codeblock">
```

```
</div>
 88/183 [=============>................] - ETA: 22:25 - loss: 1.3535 - accuracy: 0.3594

<div class="k-default-codeblock">
```

```
</div>
 89/183 [=============>................] - ETA: 22:10 - loss: 1.3529 - accuracy: 0.3596

<div class="k-default-codeblock">
```

```
</div>
 90/183 [=============>................] - ETA: 21:56 - loss: 1.3529 - accuracy: 0.3597

<div class="k-default-codeblock">
```

```
</div>
 91/183 [=============>................] - ETA: 21:42 - loss: 1.3510 - accuracy: 0.3613

<div class="k-default-codeblock">
```

```
</div>
 92/183 [==============>...............] - ETA: 21:28 - loss: 1.3498 - accuracy: 0.3614

<div class="k-default-codeblock">
```

```
</div>
 93/183 [==============>...............] - ETA: 21:14 - loss: 1.3492 - accuracy: 0.3629

<div class="k-default-codeblock">
```

```
</div>
 94/183 [==============>...............] - ETA: 21:00 - loss: 1.3485 - accuracy: 0.3644

<div class="k-default-codeblock">
```

```
</div>
 95/183 [==============>...............] - ETA: 20:45 - loss: 1.3484 - accuracy: 0.3658

<div class="k-default-codeblock">
```

```
</div>
 96/183 [==============>...............] - ETA: 20:31 - loss: 1.3486 - accuracy: 0.3672

<div class="k-default-codeblock">
```

```
</div>
 97/183 [==============>...............] - ETA: 20:17 - loss: 1.3488 - accuracy: 0.3647

<div class="k-default-codeblock">
```

```
</div>
 98/183 [===============>..............] - ETA: 20:03 - loss: 1.3484 - accuracy: 0.3648

<div class="k-default-codeblock">
```

```
</div>
 99/183 [===============>..............] - ETA: 19:49 - loss: 1.3478 - accuracy: 0.3662

<div class="k-default-codeblock">
```

```
</div>
100/183 [===============>..............] - ETA: 19:34 - loss: 1.3469 - accuracy: 0.3650

<div class="k-default-codeblock">
```

```
</div>
101/183 [===============>..............] - ETA: 19:20 - loss: 1.3474 - accuracy: 0.3651

<div class="k-default-codeblock">
```

```
</div>
102/183 [===============>..............] - ETA: 19:06 - loss: 1.3457 - accuracy: 0.3689

<div class="k-default-codeblock">
```

```
</div>
103/183 [===============>..............] - ETA: 18:52 - loss: 1.3453 - accuracy: 0.3689

<div class="k-default-codeblock">
```

```
</div>
104/183 [================>.............] - ETA: 18:38 - loss: 1.3444 - accuracy: 0.3714

<div class="k-default-codeblock">
```

```
</div>
105/183 [================>.............] - ETA: 18:24 - loss: 1.3441 - accuracy: 0.3726

<div class="k-default-codeblock">
```

```
</div>
106/183 [================>.............] - ETA: 18:10 - loss: 1.3431 - accuracy: 0.3750

<div class="k-default-codeblock">
```

```
</div>
107/183 [================>.............] - ETA: 17:55 - loss: 1.3425 - accuracy: 0.3738

<div class="k-default-codeblock">
```

```
</div>
108/183 [================>.............] - ETA: 17:41 - loss: 1.3421 - accuracy: 0.3738

<div class="k-default-codeblock">
```

```
</div>
109/183 [================>.............] - ETA: 17:27 - loss: 1.3419 - accuracy: 0.3739

<div class="k-default-codeblock">
```

```
</div>
110/183 [=================>............] - ETA: 17:13 - loss: 1.3412 - accuracy: 0.3750

<div class="k-default-codeblock">
```

```
</div>
111/183 [=================>............] - ETA: 16:59 - loss: 1.3400 - accuracy: 0.3773

<div class="k-default-codeblock">
```

```
</div>
112/183 [=================>............] - ETA: 16:45 - loss: 1.3393 - accuracy: 0.3795

<div class="k-default-codeblock">
```

```
</div>
113/183 [=================>............] - ETA: 16:30 - loss: 1.3380 - accuracy: 0.3827

<div class="k-default-codeblock">
```

```
</div>
114/183 [=================>............] - ETA: 16:16 - loss: 1.3366 - accuracy: 0.3860

<div class="k-default-codeblock">
```

```
</div>
115/183 [=================>............] - ETA: 16:02 - loss: 1.3350 - accuracy: 0.3870

<div class="k-default-codeblock">
```

```
</div>
116/183 [==================>...........] - ETA: 15:48 - loss: 1.3334 - accuracy: 0.3901

<div class="k-default-codeblock">
```

```
</div>
117/183 [==================>...........] - ETA: 15:34 - loss: 1.3331 - accuracy: 0.3900

<div class="k-default-codeblock">
```

```
</div>
118/183 [==================>...........] - ETA: 15:20 - loss: 1.3324 - accuracy: 0.3909

<div class="k-default-codeblock">
```

```
</div>
119/183 [==================>...........] - ETA: 15:06 - loss: 1.3308 - accuracy: 0.3929

<div class="k-default-codeblock">
```

```
</div>
120/183 [==================>...........] - ETA: 14:51 - loss: 1.3316 - accuracy: 0.3906

<div class="k-default-codeblock">
```

```
</div>
121/183 [==================>...........] - ETA: 14:37 - loss: 1.3312 - accuracy: 0.3895

<div class="k-default-codeblock">
```

```
</div>
122/183 [===================>..........] - ETA: 14:23 - loss: 1.3294 - accuracy: 0.3924

<div class="k-default-codeblock">
```

```
</div>
123/183 [===================>..........] - ETA: 14:09 - loss: 1.3284 - accuracy: 0.3953

<div class="k-default-codeblock">
```

```
</div>
124/183 [===================>..........] - ETA: 13:55 - loss: 1.3269 - accuracy: 0.3972

<div class="k-default-codeblock">
```

```
</div>
125/183 [===================>..........] - ETA: 13:40 - loss: 1.3249 - accuracy: 0.4000

<div class="k-default-codeblock">
```

```
</div>
126/183 [===================>..........] - ETA: 13:26 - loss: 1.3245 - accuracy: 0.3988

<div class="k-default-codeblock">
```

```
</div>
127/183 [===================>..........] - ETA: 13:12 - loss: 1.3231 - accuracy: 0.4006

<div class="k-default-codeblock">
```

```
</div>
128/183 [===================>..........] - ETA: 12:58 - loss: 1.3233 - accuracy: 0.3994

<div class="k-default-codeblock">
```

```
</div>
129/183 [====================>.........] - ETA: 12:44 - loss: 1.3212 - accuracy: 0.4021

<div class="k-default-codeblock">
```

```
</div>
130/183 [====================>.........] - ETA: 12:30 - loss: 1.3206 - accuracy: 0.4029

<div class="k-default-codeblock">
```

```
</div>
131/183 [====================>.........] - ETA: 12:16 - loss: 1.3187 - accuracy: 0.4065

<div class="k-default-codeblock">
```

```
</div>
132/183 [====================>.........] - ETA: 12:01 - loss: 1.3173 - accuracy: 0.4091

<div class="k-default-codeblock">
```

```
</div>
133/183 [====================>.........] - ETA: 11:47 - loss: 1.3162 - accuracy: 0.4098

<div class="k-default-codeblock">
```

```
</div>
134/183 [====================>.........] - ETA: 11:33 - loss: 1.3148 - accuracy: 0.4123

<div class="k-default-codeblock">
```

```
</div>
135/183 [=====================>........] - ETA: 11:19 - loss: 1.3149 - accuracy: 0.4120

<div class="k-default-codeblock">
```

```
</div>
136/183 [=====================>........] - ETA: 11:05 - loss: 1.3150 - accuracy: 0.4118

<div class="k-default-codeblock">
```

```
</div>
137/183 [=====================>........] - ETA: 10:51 - loss: 1.3143 - accuracy: 0.4106

<div class="k-default-codeblock">
```

```
</div>
138/183 [=====================>........] - ETA: 10:36 - loss: 1.3133 - accuracy: 0.4130

<div class="k-default-codeblock">
```

```
</div>
139/183 [=====================>........] - ETA: 10:22 - loss: 1.3127 - accuracy: 0.4128

<div class="k-default-codeblock">
```

```
</div>
140/183 [=====================>........] - ETA: 10:08 - loss: 1.3136 - accuracy: 0.4125

<div class="k-default-codeblock">
```

```
</div>
141/183 [======================>.......] - ETA: 9:54 - loss: 1.3125 - accuracy: 0.4131 

<div class="k-default-codeblock">
```

```
</div>
142/183 [======================>.......] - ETA: 9:40 - loss: 1.3115 - accuracy: 0.4155

<div class="k-default-codeblock">
```

```
</div>
143/183 [======================>.......] - ETA: 9:26 - loss: 1.3108 - accuracy: 0.4161

<div class="k-default-codeblock">
```

```
</div>
144/183 [======================>.......] - ETA: 9:11 - loss: 1.3098 - accuracy: 0.4167

<div class="k-default-codeblock">
```

```
</div>
145/183 [======================>.......] - ETA: 8:57 - loss: 1.3086 - accuracy: 0.4172

<div class="k-default-codeblock">
```

```
</div>
146/183 [======================>.......] - ETA: 8:43 - loss: 1.3083 - accuracy: 0.4161

<div class="k-default-codeblock">
```

```
</div>
147/183 [=======================>......] - ETA: 8:29 - loss: 1.3067 - accuracy: 0.4184

<div class="k-default-codeblock">
```

```
</div>
148/183 [=======================>......] - ETA: 8:15 - loss: 1.3047 - accuracy: 0.4215

<div class="k-default-codeblock">
```

```
</div>
149/183 [=======================>......] - ETA: 8:01 - loss: 1.3037 - accuracy: 0.4211

<div class="k-default-codeblock">
```

```
</div>
150/183 [=======================>......] - ETA: 7:47 - loss: 1.3029 - accuracy: 0.4217

<div class="k-default-codeblock">
```

```
</div>
151/183 [=======================>......] - ETA: 7:32 - loss: 1.3021 - accuracy: 0.4222

<div class="k-default-codeblock">
```

```
</div>
152/183 [=======================>......] - ETA: 7:18 - loss: 1.3007 - accuracy: 0.4243

<div class="k-default-codeblock">
```

```
</div>
153/183 [========================>.....] - ETA: 7:04 - loss: 1.3017 - accuracy: 0.4240

<div class="k-default-codeblock">
```

```
</div>
154/183 [========================>.....] - ETA: 6:50 - loss: 1.3006 - accuracy: 0.4253

<div class="k-default-codeblock">
```

```
</div>
155/183 [========================>.....] - ETA: 6:36 - loss: 1.2986 - accuracy: 0.4282

<div class="k-default-codeblock">
```

```
</div>
156/183 [========================>.....] - ETA: 6:22 - loss: 1.2965 - accuracy: 0.4303

<div class="k-default-codeblock">
```

```
</div>
157/183 [========================>.....] - ETA: 6:08 - loss: 1.2964 - accuracy: 0.4283

<div class="k-default-codeblock">
```

```
</div>
158/183 [========================>.....] - ETA: 5:53 - loss: 1.2953 - accuracy: 0.4288

<div class="k-default-codeblock">
```

```
</div>
159/183 [=========================>....] - ETA: 5:39 - loss: 1.2931 - accuracy: 0.4316

<div class="k-default-codeblock">
```

```
</div>
160/183 [=========================>....] - ETA: 5:25 - loss: 1.2923 - accuracy: 0.4320

<div class="k-default-codeblock">
```

```
</div>
161/183 [=========================>....] - ETA: 5:11 - loss: 1.2916 - accuracy: 0.4332

<div class="k-default-codeblock">
```

```
</div>
162/183 [=========================>....] - ETA: 4:57 - loss: 1.2905 - accuracy: 0.4344

<div class="k-default-codeblock">
```

```
</div>
163/183 [=========================>....] - ETA: 4:43 - loss: 1.2901 - accuracy: 0.4348

<div class="k-default-codeblock">
```

```
</div>
164/183 [=========================>....] - ETA: 4:28 - loss: 1.2891 - accuracy: 0.4360

<div class="k-default-codeblock">
```

```
</div>
165/183 [==========================>...] - ETA: 4:14 - loss: 1.2881 - accuracy: 0.4356

<div class="k-default-codeblock">
```

```
</div>
166/183 [==========================>...] - ETA: 4:00 - loss: 1.2860 - accuracy: 0.4375

<div class="k-default-codeblock">
```

```
</div>
167/183 [==========================>...] - ETA: 3:46 - loss: 1.2835 - accuracy: 0.4386

<div class="k-default-codeblock">
```

```
</div>
168/183 [==========================>...] - ETA: 3:32 - loss: 1.2838 - accuracy: 0.4375

<div class="k-default-codeblock">
```

```
</div>
169/183 [==========================>...] - ETA: 3:18 - loss: 1.2829 - accuracy: 0.4379

<div class="k-default-codeblock">
```

```
</div>
170/183 [==========================>...] - ETA: 3:03 - loss: 1.2830 - accuracy: 0.4390

<div class="k-default-codeblock">
```

```
</div>
171/183 [===========================>..] - ETA: 2:49 - loss: 1.2800 - accuracy: 0.4415

<div class="k-default-codeblock">
```

```
</div>
172/183 [===========================>..] - ETA: 2:35 - loss: 1.2784 - accuracy: 0.4419

<div class="k-default-codeblock">
```

```
</div>
173/183 [===========================>..] - ETA: 2:21 - loss: 1.2765 - accuracy: 0.4429

<div class="k-default-codeblock">
```

```
</div>
174/183 [===========================>..] - ETA: 2:07 - loss: 1.2746 - accuracy: 0.4447

<div class="k-default-codeblock">
```

```
</div>
175/183 [===========================>..] - ETA: 1:53 - loss: 1.2730 - accuracy: 0.4457

<div class="k-default-codeblock">
```

```
</div>
176/183 [===========================>..] - ETA: 1:39 - loss: 1.2705 - accuracy: 0.4489

<div class="k-default-codeblock">
```

```
</div>
177/183 [============================>.] - ETA: 1:24 - loss: 1.2696 - accuracy: 0.4492

<div class="k-default-codeblock">
```

```
</div>
178/183 [============================>.] - ETA: 1:10 - loss: 1.2686 - accuracy: 0.4501

<div class="k-default-codeblock">
```

```
</div>
179/183 [============================>.] - ETA: 56s - loss: 1.2679 - accuracy: 0.4511 

<div class="k-default-codeblock">
```

```
</div>
180/183 [============================>.] - ETA: 42s - loss: 1.2671 - accuracy: 0.4521

<div class="k-default-codeblock">
```

```
</div>
181/183 [============================>.] - ETA: 28s - loss: 1.2666 - accuracy: 0.4523

<div class="k-default-codeblock">
```

```
</div>
182/183 [============================>.] - ETA: 14s - loss: 1.2642 - accuracy: 0.4547

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - ETA: 0s - loss: 1.2631 - accuracy: 0.4556 

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - 2748s 15s/step - loss: 1.2631 - accuracy: 0.4556 - val_loss: 0.9210 - val_accuracy: 0.7075 - lr: 2.9000e-06


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/183 [..............................] - ETA: 46:11 - loss: 1.1680 - accuracy: 0.3750

<div class="k-default-codeblock">
```

```
</div>
  2/183 [..............................] - ETA: 43:28 - loss: 1.0826 - accuracy: 0.5625

<div class="k-default-codeblock">
```

```
</div>
  3/183 [..............................] - ETA: 43:21 - loss: 1.1881 - accuracy: 0.5417

<div class="k-default-codeblock">
```

```
</div>
  4/183 [..............................] - ETA: 43:06 - loss: 1.1313 - accuracy: 0.5625

<div class="k-default-codeblock">
```

```
</div>
  5/183 [..............................] - ETA: 42:50 - loss: 1.1148 - accuracy: 0.6000

<div class="k-default-codeblock">
```

```
</div>
  6/183 [..............................] - ETA: 42:32 - loss: 1.0660 - accuracy: 0.6458

<div class="k-default-codeblock">
```

```
</div>
  7/183 [>.............................] - ETA: 42:14 - loss: 1.0620 - accuracy: 0.6429

<div class="k-default-codeblock">
```

```
</div>
  8/183 [>.............................] - ETA: 41:56 - loss: 1.0516 - accuracy: 0.6562

<div class="k-default-codeblock">
```

```
</div>
  9/183 [>.............................] - ETA: 41:40 - loss: 1.0623 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 10/183 [>.............................] - ETA: 41:28 - loss: 1.0573 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 11/183 [>.............................] - ETA: 41:12 - loss: 1.0500 - accuracy: 0.6364

<div class="k-default-codeblock">
```

```
</div>
 12/183 [>.............................] - ETA: 40:55 - loss: 1.0126 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
 13/183 [=>............................] - ETA: 40:39 - loss: 1.0203 - accuracy: 0.6538

<div class="k-default-codeblock">
```

```
</div>
 14/183 [=>............................] - ETA: 40:22 - loss: 1.0258 - accuracy: 0.6518

<div class="k-default-codeblock">
```

```
</div>
 15/183 [=>............................] - ETA: 40:06 - loss: 1.0312 - accuracy: 0.6583

<div class="k-default-codeblock">
```

```
</div>
 16/183 [=>............................] - ETA: 39:50 - loss: 1.0269 - accuracy: 0.6562

<div class="k-default-codeblock">
```

```
</div>
 17/183 [=>............................] - ETA: 39:33 - loss: 1.0329 - accuracy: 0.6618

<div class="k-default-codeblock">
```

```
</div>
 18/183 [=>............................] - ETA: 39:18 - loss: 1.0382 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
 19/183 [==>...........................] - ETA: 39:03 - loss: 1.0372 - accuracy: 0.6579

<div class="k-default-codeblock">
```

```
</div>
 20/183 [==>...........................] - ETA: 38:48 - loss: 1.0363 - accuracy: 0.6500

<div class="k-default-codeblock">
```

```
</div>
 21/183 [==>...........................] - ETA: 38:33 - loss: 1.0430 - accuracy: 0.6369

<div class="k-default-codeblock">
```

```
</div>
 22/183 [==>...........................] - ETA: 38:18 - loss: 1.0392 - accuracy: 0.6420

<div class="k-default-codeblock">
```

```
</div>
 23/183 [==>...........................] - ETA: 38:03 - loss: 1.0438 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 24/183 [==>...........................] - ETA: 37:49 - loss: 1.0405 - accuracy: 0.6302

<div class="k-default-codeblock">
```

```
</div>
 25/183 [===>..........................] - ETA: 37:34 - loss: 1.0383 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 26/183 [===>..........................] - ETA: 37:20 - loss: 1.0392 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 27/183 [===>..........................] - ETA: 37:05 - loss: 1.0422 - accuracy: 0.6204

<div class="k-default-codeblock">
```

```
</div>
 28/183 [===>..........................] - ETA: 36:50 - loss: 1.0441 - accuracy: 0.6205

<div class="k-default-codeblock">
```

```
</div>
 29/183 [===>..........................] - ETA: 36:35 - loss: 1.0482 - accuracy: 0.6164

<div class="k-default-codeblock">
```

```
</div>
 30/183 [===>..........................] - ETA: 36:21 - loss: 1.0541 - accuracy: 0.6042

<div class="k-default-codeblock">
```

```
</div>
 31/183 [====>.........................] - ETA: 36:06 - loss: 1.0507 - accuracy: 0.6089

<div class="k-default-codeblock">
```

```
</div>
 32/183 [====>.........................] - ETA: 35:52 - loss: 1.0431 - accuracy: 0.6211

<div class="k-default-codeblock">
```

```
</div>
 33/183 [====>.........................] - ETA: 35:37 - loss: 1.0418 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 34/183 [====>.........................] - ETA: 35:22 - loss: 1.0382 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 35/183 [====>.........................] - ETA: 35:08 - loss: 1.0369 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 36/183 [====>.........................] - ETA: 34:53 - loss: 1.0396 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
 37/183 [=====>........................] - ETA: 34:38 - loss: 1.0396 - accuracy: 0.6216

<div class="k-default-codeblock">
```

```
</div>
 38/183 [=====>........................] - ETA: 34:24 - loss: 1.0478 - accuracy: 0.6118

<div class="k-default-codeblock">
```

```
</div>
 39/183 [=====>........................] - ETA: 34:09 - loss: 1.0412 - accuracy: 0.6154

<div class="k-default-codeblock">
```

```
</div>
 40/183 [=====>........................] - ETA: 33:55 - loss: 1.0397 - accuracy: 0.6156

<div class="k-default-codeblock">
```

```
</div>
 41/183 [=====>........................] - ETA: 33:41 - loss: 1.0378 - accuracy: 0.6159

<div class="k-default-codeblock">
```

```
</div>
 42/183 [=====>........................] - ETA: 33:26 - loss: 1.0331 - accuracy: 0.6190

<div class="k-default-codeblock">
```

```
</div>
 43/183 [======>.......................] - ETA: 33:13 - loss: 1.0287 - accuracy: 0.6192

<div class="k-default-codeblock">
```

```
</div>
 44/183 [======>.......................] - ETA: 32:58 - loss: 1.0366 - accuracy: 0.6136

<div class="k-default-codeblock">
```

```
</div>
 45/183 [======>.......................] - ETA: 32:43 - loss: 1.0337 - accuracy: 0.6167

<div class="k-default-codeblock">
```

```
</div>
 46/183 [======>.......................] - ETA: 32:29 - loss: 1.0309 - accuracy: 0.6196

<div class="k-default-codeblock">
```

```
</div>
 47/183 [======>.......................] - ETA: 32:15 - loss: 1.0355 - accuracy: 0.6144

<div class="k-default-codeblock">
```

```
</div>
 48/183 [======>.......................] - ETA: 32:00 - loss: 1.0365 - accuracy: 0.6146

<div class="k-default-codeblock">
```

```
</div>
 49/183 [=======>......................] - ETA: 31:46 - loss: 1.0344 - accuracy: 0.6173

<div class="k-default-codeblock">
```

```
</div>
 50/183 [=======>......................] - ETA: 31:32 - loss: 1.0364 - accuracy: 0.6175

<div class="k-default-codeblock">
```

```
</div>
 51/183 [=======>......................] - ETA: 31:17 - loss: 1.0385 - accuracy: 0.6152

<div class="k-default-codeblock">
```

```
</div>
 52/183 [=======>......................] - ETA: 31:03 - loss: 1.0415 - accuracy: 0.6106

<div class="k-default-codeblock">
```

```
</div>
 53/183 [=======>......................] - ETA: 30:50 - loss: 1.0360 - accuracy: 0.6156

<div class="k-default-codeblock">
```

```
</div>
 54/183 [=======>......................] - ETA: 30:36 - loss: 1.0329 - accuracy: 0.6181

<div class="k-default-codeblock">
```

```
</div>
 55/183 [========>.....................] - ETA: 30:21 - loss: 1.0340 - accuracy: 0.6136

<div class="k-default-codeblock">
```

```
</div>
 56/183 [========>.....................] - ETA: 30:07 - loss: 1.0357 - accuracy: 0.6094

<div class="k-default-codeblock">
```

```
</div>
 57/183 [========>.....................] - ETA: 29:53 - loss: 1.0389 - accuracy: 0.6096

<div class="k-default-codeblock">
```

```
</div>
 58/183 [========>.....................] - ETA: 29:38 - loss: 1.0403 - accuracy: 0.6099

<div class="k-default-codeblock">
```

```
</div>
 59/183 [========>.....................] - ETA: 29:24 - loss: 1.0437 - accuracy: 0.6038

<div class="k-default-codeblock">
```

```
</div>
 60/183 [========>.....................] - ETA: 29:09 - loss: 1.0459 - accuracy: 0.6042

<div class="k-default-codeblock">
```

```
</div>
 61/183 [=========>....................] - ETA: 28:55 - loss: 1.0509 - accuracy: 0.6025

<div class="k-default-codeblock">
```

```
</div>
 62/183 [=========>....................] - ETA: 28:41 - loss: 1.0511 - accuracy: 0.5988

<div class="k-default-codeblock">
```

```
</div>
 63/183 [=========>....................] - ETA: 28:27 - loss: 1.0479 - accuracy: 0.5992

<div class="k-default-codeblock">
```

```
</div>
 64/183 [=========>....................] - ETA: 28:12 - loss: 1.0548 - accuracy: 0.5977

<div class="k-default-codeblock">
```

```
</div>
 65/183 [=========>....................] - ETA: 27:58 - loss: 1.0570 - accuracy: 0.5962

<div class="k-default-codeblock">
```

```
</div>
 66/183 [=========>....................] - ETA: 27:44 - loss: 1.0555 - accuracy: 0.5966

<div class="k-default-codeblock">
```

```
</div>
 67/183 [=========>....................] - ETA: 27:29 - loss: 1.0566 - accuracy: 0.5951

<div class="k-default-codeblock">
```

```
</div>
 68/183 [==========>...................] - ETA: 27:15 - loss: 1.0571 - accuracy: 0.5938

<div class="k-default-codeblock">
```

```
</div>
 69/183 [==========>...................] - ETA: 27:01 - loss: 1.0523 - accuracy: 0.5978

<div class="k-default-codeblock">
```

```
</div>
 70/183 [==========>...................] - ETA: 26:47 - loss: 1.0483 - accuracy: 0.6000

<div class="k-default-codeblock">
```

```
</div>
 71/183 [==========>...................] - ETA: 26:32 - loss: 1.0489 - accuracy: 0.5968

<div class="k-default-codeblock">
```

```
</div>
 72/183 [==========>...................] - ETA: 26:18 - loss: 1.0491 - accuracy: 0.5955

<div class="k-default-codeblock">
```

```
</div>
 73/183 [==========>...................] - ETA: 26:04 - loss: 1.0486 - accuracy: 0.5959

<div class="k-default-codeblock">
```

```
</div>
 74/183 [===========>..................] - ETA: 25:49 - loss: 1.0532 - accuracy: 0.5912

<div class="k-default-codeblock">
```

```
</div>
 75/183 [===========>..................] - ETA: 25:35 - loss: 1.0535 - accuracy: 0.5917

<div class="k-default-codeblock">
```

```
</div>
 76/183 [===========>..................] - ETA: 25:21 - loss: 1.0568 - accuracy: 0.5888

<div class="k-default-codeblock">
```

```
</div>
 77/183 [===========>..................] - ETA: 25:06 - loss: 1.0600 - accuracy: 0.5844

<div class="k-default-codeblock">
```

```
</div>
 78/183 [===========>..................] - ETA: 24:52 - loss: 1.0557 - accuracy: 0.5865

<div class="k-default-codeblock">
```

```
</div>
 79/183 [===========>..................] - ETA: 24:38 - loss: 1.0538 - accuracy: 0.5870

<div class="k-default-codeblock">
```

```
</div>
 80/183 [============>.................] - ETA: 24:23 - loss: 1.0543 - accuracy: 0.5844

<div class="k-default-codeblock">
```

```
</div>
 81/183 [============>.................] - ETA: 24:09 - loss: 1.0528 - accuracy: 0.5864

<div class="k-default-codeblock">
```

```
</div>
 82/183 [============>.................] - ETA: 23:55 - loss: 1.0547 - accuracy: 0.5838

<div class="k-default-codeblock">
```

```
</div>
 83/183 [============>.................] - ETA: 23:41 - loss: 1.0558 - accuracy: 0.5798

<div class="k-default-codeblock">
```

```
</div>
 84/183 [============>.................] - ETA: 23:26 - loss: 1.0564 - accuracy: 0.5804

<div class="k-default-codeblock">
```

```
</div>
 85/183 [============>.................] - ETA: 23:12 - loss: 1.0548 - accuracy: 0.5794

<div class="k-default-codeblock">
```

```
</div>
 86/183 [=============>................] - ETA: 22:58 - loss: 1.0542 - accuracy: 0.5770

<div class="k-default-codeblock">
```

```
</div>
 87/183 [=============>................] - ETA: 22:44 - loss: 1.0569 - accuracy: 0.5761

<div class="k-default-codeblock">
```

```
</div>
 88/183 [=============>................] - ETA: 22:29 - loss: 1.0551 - accuracy: 0.5781

<div class="k-default-codeblock">
```

```
</div>
 89/183 [=============>................] - ETA: 22:15 - loss: 1.0525 - accuracy: 0.5815

<div class="k-default-codeblock">
```

```
</div>
 90/183 [=============>................] - ETA: 22:01 - loss: 1.0552 - accuracy: 0.5778

<div class="k-default-codeblock">
```

```
</div>
 91/183 [=============>................] - ETA: 21:47 - loss: 1.0529 - accuracy: 0.5797

<div class="k-default-codeblock">
```

```
</div>
 92/183 [==============>...............] - ETA: 21:32 - loss: 1.0501 - accuracy: 0.5802

<div class="k-default-codeblock">
```

```
</div>
 93/183 [==============>...............] - ETA: 21:18 - loss: 1.0496 - accuracy: 0.5793

<div class="k-default-codeblock">
```

```
</div>
 94/183 [==============>...............] - ETA: 21:04 - loss: 1.0512 - accuracy: 0.5798

<div class="k-default-codeblock">
```

```
</div>
 95/183 [==============>...............] - ETA: 20:49 - loss: 1.0518 - accuracy: 0.5803

<div class="k-default-codeblock">
```

```
</div>
 96/183 [==============>...............] - ETA: 20:35 - loss: 1.0520 - accuracy: 0.5807

<div class="k-default-codeblock">
```

```
</div>
 97/183 [==============>...............] - ETA: 20:21 - loss: 1.0550 - accuracy: 0.5812

<div class="k-default-codeblock">
```

```
</div>
 98/183 [===============>..............] - ETA: 20:07 - loss: 1.0576 - accuracy: 0.5791

<div class="k-default-codeblock">
```

```
</div>
 99/183 [===============>..............] - ETA: 19:53 - loss: 1.0571 - accuracy: 0.5770

<div class="k-default-codeblock">
```

```
</div>
100/183 [===============>..............] - ETA: 19:38 - loss: 1.0547 - accuracy: 0.5775

<div class="k-default-codeblock">
```

```
</div>
101/183 [===============>..............] - ETA: 19:24 - loss: 1.0547 - accuracy: 0.5767

<div class="k-default-codeblock">
```

```
</div>
102/183 [===============>..............] - ETA: 19:10 - loss: 1.0536 - accuracy: 0.5748

<div class="k-default-codeblock">
```

```
</div>
103/183 [===============>..............] - ETA: 18:56 - loss: 1.0514 - accuracy: 0.5752

<div class="k-default-codeblock">
```

```
</div>
104/183 [================>.............] - ETA: 18:41 - loss: 1.0495 - accuracy: 0.5769

<div class="k-default-codeblock">
```

```
</div>
105/183 [================>.............] - ETA: 18:27 - loss: 1.0462 - accuracy: 0.5774

<div class="k-default-codeblock">
```

```
</div>
106/183 [================>.............] - ETA: 18:13 - loss: 1.0451 - accuracy: 0.5790

<div class="k-default-codeblock">
```

```
</div>
107/183 [================>.............] - ETA: 17:59 - loss: 1.0434 - accuracy: 0.5818

<div class="k-default-codeblock">
```

```
</div>
108/183 [================>.............] - ETA: 17:44 - loss: 1.0424 - accuracy: 0.5833

<div class="k-default-codeblock">
```

```
</div>
109/183 [================>.............] - ETA: 17:30 - loss: 1.0421 - accuracy: 0.5826

<div class="k-default-codeblock">
```

```
</div>
110/183 [=================>............] - ETA: 17:16 - loss: 1.0409 - accuracy: 0.5830

<div class="k-default-codeblock">
```

```
</div>
111/183 [=================>............] - ETA: 17:02 - loss: 1.0415 - accuracy: 0.5833

<div class="k-default-codeblock">
```

```
</div>
112/183 [=================>............] - ETA: 16:47 - loss: 1.0403 - accuracy: 0.5837

<div class="k-default-codeblock">
```

```
</div>
113/183 [=================>............] - ETA: 16:33 - loss: 1.0377 - accuracy: 0.5852

<div class="k-default-codeblock">
```

```
</div>
114/183 [=================>............] - ETA: 16:19 - loss: 1.0367 - accuracy: 0.5866

<div class="k-default-codeblock">
```

```
</div>
115/183 [=================>............] - ETA: 16:05 - loss: 1.0357 - accuracy: 0.5880

<div class="k-default-codeblock">
```

```
</div>
116/183 [==================>...........] - ETA: 15:50 - loss: 1.0347 - accuracy: 0.5884

<div class="k-default-codeblock">
```

```
</div>
117/183 [==================>...........] - ETA: 15:36 - loss: 1.0329 - accuracy: 0.5897

<div class="k-default-codeblock">
```

```
</div>
118/183 [==================>...........] - ETA: 15:22 - loss: 1.0314 - accuracy: 0.5900

<div class="k-default-codeblock">
```

```
</div>
119/183 [==================>...........] - ETA: 15:08 - loss: 1.0309 - accuracy: 0.5903

<div class="k-default-codeblock">
```

```
</div>
120/183 [==================>...........] - ETA: 14:54 - loss: 1.0273 - accuracy: 0.5927

<div class="k-default-codeblock">
```

```
</div>
121/183 [==================>...........] - ETA: 14:39 - loss: 1.0271 - accuracy: 0.5919

<div class="k-default-codeblock">
```

```
</div>
122/183 [===================>..........] - ETA: 14:25 - loss: 1.0277 - accuracy: 0.5922

<div class="k-default-codeblock">
```

```
</div>
123/183 [===================>..........] - ETA: 14:11 - loss: 1.0252 - accuracy: 0.5925

<div class="k-default-codeblock">
```

```
</div>
124/183 [===================>..........] - ETA: 13:57 - loss: 1.0260 - accuracy: 0.5927

<div class="k-default-codeblock">
```

```
</div>
125/183 [===================>..........] - ETA: 13:42 - loss: 1.0269 - accuracy: 0.5930

<div class="k-default-codeblock">
```

```
</div>
126/183 [===================>..........] - ETA: 13:28 - loss: 1.0241 - accuracy: 0.5942

<div class="k-default-codeblock">
```

```
</div>
127/183 [===================>..........] - ETA: 13:14 - loss: 1.0223 - accuracy: 0.5965

<div class="k-default-codeblock">
```

```
</div>
128/183 [===================>..........] - ETA: 13:00 - loss: 1.0215 - accuracy: 0.5947

<div class="k-default-codeblock">
```

```
</div>
129/183 [====================>.........] - ETA: 12:46 - loss: 1.0225 - accuracy: 0.5940

<div class="k-default-codeblock">
```

```
</div>
130/183 [====================>.........] - ETA: 12:31 - loss: 1.0200 - accuracy: 0.5962

<div class="k-default-codeblock">
```

```
</div>
131/183 [====================>.........] - ETA: 12:17 - loss: 1.0190 - accuracy: 0.5964

<div class="k-default-codeblock">
```

```
</div>
132/183 [====================>.........] - ETA: 12:03 - loss: 1.0178 - accuracy: 0.5975

<div class="k-default-codeblock">
```

```
</div>
133/183 [====================>.........] - ETA: 11:49 - loss: 1.0159 - accuracy: 0.5987

<div class="k-default-codeblock">
```

```
</div>
134/183 [====================>.........] - ETA: 11:35 - loss: 1.0160 - accuracy: 0.5989

<div class="k-default-codeblock">
```

```
</div>
135/183 [=====================>........] - ETA: 11:20 - loss: 1.0147 - accuracy: 0.5991

<div class="k-default-codeblock">
```

```
</div>
136/183 [=====================>........] - ETA: 11:06 - loss: 1.0153 - accuracy: 0.5993

<div class="k-default-codeblock">
```

```
</div>
137/183 [=====================>........] - ETA: 10:52 - loss: 1.0184 - accuracy: 0.5976

<div class="k-default-codeblock">
```

```
</div>
138/183 [=====================>........] - ETA: 10:38 - loss: 1.0174 - accuracy: 0.5978

<div class="k-default-codeblock">
```

```
</div>
139/183 [=====================>........] - ETA: 10:24 - loss: 1.0192 - accuracy: 0.5980

<div class="k-default-codeblock">
```

```
</div>
140/183 [=====================>........] - ETA: 10:09 - loss: 1.0221 - accuracy: 0.5955

<div class="k-default-codeblock">
```

```
</div>
141/183 [======================>.......] - ETA: 9:55 - loss: 1.0244 - accuracy: 0.5931 

<div class="k-default-codeblock">
```

```
</div>
142/183 [======================>.......] - ETA: 9:41 - loss: 1.0228 - accuracy: 0.5951

<div class="k-default-codeblock">
```

```
</div>
143/183 [======================>.......] - ETA: 9:27 - loss: 1.0211 - accuracy: 0.5970

<div class="k-default-codeblock">
```

```
</div>
144/183 [======================>.......] - ETA: 9:13 - loss: 1.0235 - accuracy: 0.5946

<div class="k-default-codeblock">
```

```
</div>
145/183 [======================>.......] - ETA: 8:59 - loss: 1.0241 - accuracy: 0.5940

<div class="k-default-codeblock">
```

```
</div>
146/183 [======================>.......] - ETA: 8:44 - loss: 1.0228 - accuracy: 0.5950

<div class="k-default-codeblock">
```

```
</div>
147/183 [=======================>......] - ETA: 8:30 - loss: 1.0229 - accuracy: 0.5961

<div class="k-default-codeblock">
```

```
</div>
148/183 [=======================>......] - ETA: 8:16 - loss: 1.0204 - accuracy: 0.5980

<div class="k-default-codeblock">
```

```
</div>
149/183 [=======================>......] - ETA: 8:02 - loss: 1.0193 - accuracy: 0.5990

<div class="k-default-codeblock">
```

```
</div>
150/183 [=======================>......] - ETA: 7:48 - loss: 1.0173 - accuracy: 0.6000

<div class="k-default-codeblock">
```

```
</div>
151/183 [=======================>......] - ETA: 7:33 - loss: 1.0190 - accuracy: 0.5993

<div class="k-default-codeblock">
```

```
</div>
152/183 [=======================>......] - ETA: 7:19 - loss: 1.0233 - accuracy: 0.5979

<div class="k-default-codeblock">
```

```
</div>
153/183 [========================>.....] - ETA: 7:05 - loss: 1.0230 - accuracy: 0.5972

<div class="k-default-codeblock">
```

```
</div>
154/183 [========================>.....] - ETA: 6:51 - loss: 1.0230 - accuracy: 0.5974

<div class="k-default-codeblock">
```

```
</div>
155/183 [========================>.....] - ETA: 6:37 - loss: 1.0223 - accuracy: 0.5992

<div class="k-default-codeblock">
```

```
</div>
156/183 [========================>.....] - ETA: 6:22 - loss: 1.0204 - accuracy: 0.5994

<div class="k-default-codeblock">
```

```
</div>
157/183 [========================>.....] - ETA: 6:08 - loss: 1.0204 - accuracy: 0.5987

<div class="k-default-codeblock">
```

```
</div>
158/183 [========================>.....] - ETA: 5:54 - loss: 1.0201 - accuracy: 0.5965

<div class="k-default-codeblock">
```

```
</div>
159/183 [=========================>....] - ETA: 5:40 - loss: 1.0199 - accuracy: 0.5975

<div class="k-default-codeblock">
```

```
</div>
160/183 [=========================>....] - ETA: 5:26 - loss: 1.0168 - accuracy: 0.5992

<div class="k-default-codeblock">
```

```
</div>
161/183 [=========================>....] - ETA: 5:12 - loss: 1.0155 - accuracy: 0.6009

<div class="k-default-codeblock">
```

```
</div>
162/183 [=========================>....] - ETA: 4:57 - loss: 1.0131 - accuracy: 0.6019

<div class="k-default-codeblock">
```

```
</div>
163/183 [=========================>....] - ETA: 4:43 - loss: 1.0144 - accuracy: 0.6020

<div class="k-default-codeblock">
```

```
</div>
164/183 [=========================>....] - ETA: 4:29 - loss: 1.0145 - accuracy: 0.6021

<div class="k-default-codeblock">
```

```
</div>
165/183 [==========================>...] - ETA: 4:15 - loss: 1.0138 - accuracy: 0.6030

<div class="k-default-codeblock">
```

```
</div>
166/183 [==========================>...] - ETA: 4:01 - loss: 1.0127 - accuracy: 0.6032

<div class="k-default-codeblock">
```

```
</div>
167/183 [==========================>...] - ETA: 3:46 - loss: 1.0127 - accuracy: 0.6018

<div class="k-default-codeblock">
```

```
</div>
168/183 [==========================>...] - ETA: 3:32 - loss: 1.0102 - accuracy: 0.6027

<div class="k-default-codeblock">
```

```
</div>
169/183 [==========================>...] - ETA: 3:18 - loss: 1.0119 - accuracy: 0.6013

<div class="k-default-codeblock">
```

```
</div>
170/183 [==========================>...] - ETA: 3:04 - loss: 1.0113 - accuracy: 0.6007

<div class="k-default-codeblock">
```

```
</div>
171/183 [===========================>..] - ETA: 2:50 - loss: 1.0134 - accuracy: 0.6001

<div class="k-default-codeblock">
```

```
</div>
172/183 [===========================>..] - ETA: 2:36 - loss: 1.0117 - accuracy: 0.6010

<div class="k-default-codeblock">
```

```
</div>
173/183 [===========================>..] - ETA: 2:21 - loss: 1.0100 - accuracy: 0.6012

<div class="k-default-codeblock">
```

```
</div>
174/183 [===========================>..] - ETA: 2:07 - loss: 1.0084 - accuracy: 0.6020

<div class="k-default-codeblock">
```

```
</div>
175/183 [===========================>..] - ETA: 1:53 - loss: 1.0060 - accuracy: 0.6043

<div class="k-default-codeblock">
```

```
</div>
176/183 [===========================>..] - ETA: 1:39 - loss: 1.0041 - accuracy: 0.6044

<div class="k-default-codeblock">
```

```
</div>
177/183 [============================>.] - ETA: 1:25 - loss: 1.0025 - accuracy: 0.6059

<div class="k-default-codeblock">
```

```
</div>
178/183 [============================>.] - ETA: 1:10 - loss: 1.0034 - accuracy: 0.6053

<div class="k-default-codeblock">
```

```
</div>
179/183 [============================>.] - ETA: 56s - loss: 1.0040 - accuracy: 0.6054 

<div class="k-default-codeblock">
```

```
</div>
180/183 [============================>.] - ETA: 42s - loss: 1.0036 - accuracy: 0.6049

<div class="k-default-codeblock">
```

```
</div>
181/183 [============================>.] - ETA: 28s - loss: 1.0028 - accuracy: 0.6050

<div class="k-default-codeblock">
```

```
</div>
182/183 [============================>.] - ETA: 14s - loss: 1.0028 - accuracy: 0.6051

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - ETA: 0s - loss: 1.0013 - accuracy: 0.6052 

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - 2755s 15s/step - loss: 1.0013 - accuracy: 0.6052 - val_loss: 0.7730 - val_accuracy: 0.7475 - lr: 4.8000e-06


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/183 [..............................] - ETA: 45:34 - loss: 0.7885 - accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
  2/183 [..............................] - ETA: 43:17 - loss: 0.8742 - accuracy: 0.6875

<div class="k-default-codeblock">
```

```
</div>
  3/183 [..............................] - ETA: 43:02 - loss: 0.7851 - accuracy: 0.7083

<div class="k-default-codeblock">
```

```
</div>
  4/183 [..............................] - ETA: 42:51 - loss: 0.9176 - accuracy: 0.6562

<div class="k-default-codeblock">
```

```
</div>
  5/183 [..............................] - ETA: 42:39 - loss: 0.8677 - accuracy: 0.7000

<div class="k-default-codeblock">
```

```
</div>
  6/183 [..............................] - ETA: 42:25 - loss: 0.8238 - accuracy: 0.7292

<div class="k-default-codeblock">
```

```
</div>
  7/183 [>.............................] - ETA: 42:09 - loss: 0.8209 - accuracy: 0.7143

<div class="k-default-codeblock">
```

```
</div>
  8/183 [>.............................] - ETA: 41:54 - loss: 0.8494 - accuracy: 0.6875

<div class="k-default-codeblock">
```

```
</div>
  9/183 [>.............................] - ETA: 41:40 - loss: 0.8458 - accuracy: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 10/183 [>.............................] - ETA: 41:25 - loss: 0.8841 - accuracy: 0.6625

<div class="k-default-codeblock">
```

```
</div>
 11/183 [>.............................] - ETA: 41:10 - loss: 0.8833 - accuracy: 0.6705

<div class="k-default-codeblock">
```

```
</div>
 12/183 [>.............................] - ETA: 40:54 - loss: 0.8520 - accuracy: 0.6979

<div class="k-default-codeblock">
```

```
</div>
 13/183 [=>............................] - ETA: 40:38 - loss: 0.8262 - accuracy: 0.7115

<div class="k-default-codeblock">
```

```
</div>
 14/183 [=>............................] - ETA: 40:23 - loss: 0.8389 - accuracy: 0.7054

<div class="k-default-codeblock">
```

```
</div>
 15/183 [=>............................] - ETA: 40:08 - loss: 0.8262 - accuracy: 0.7250

<div class="k-default-codeblock">
```

```
</div>
 16/183 [=>............................] - ETA: 39:53 - loss: 0.8368 - accuracy: 0.7188

<div class="k-default-codeblock">
```

```
</div>
 17/183 [=>............................] - ETA: 39:39 - loss: 0.8293 - accuracy: 0.7279

<div class="k-default-codeblock">
```

```
</div>
 18/183 [=>............................] - ETA: 39:25 - loss: 0.8353 - accuracy: 0.7222

<div class="k-default-codeblock">
```

```
</div>
 19/183 [==>...........................] - ETA: 39:09 - loss: 0.8331 - accuracy: 0.7237

<div class="k-default-codeblock">
```

```
</div>
 20/183 [==>...........................] - ETA: 38:54 - loss: 0.8452 - accuracy: 0.7063

<div class="k-default-codeblock">
```

```
</div>
 21/183 [==>...........................] - ETA: 38:38 - loss: 0.8396 - accuracy: 0.7083

<div class="k-default-codeblock">
```

```
</div>
 22/183 [==>...........................] - ETA: 38:23 - loss: 0.8370 - accuracy: 0.7102

<div class="k-default-codeblock">
```

```
</div>
 23/183 [==>...........................] - ETA: 38:08 - loss: 0.8350 - accuracy: 0.7120

<div class="k-default-codeblock">
```

```
</div>
 24/183 [==>...........................] - ETA: 37:53 - loss: 0.8551 - accuracy: 0.7031

<div class="k-default-codeblock">
```

```
</div>
 25/183 [===>..........................] - ETA: 37:38 - loss: 0.8385 - accuracy: 0.7150

<div class="k-default-codeblock">
```

```
</div>
 26/183 [===>..........................] - ETA: 37:23 - loss: 0.8379 - accuracy: 0.7163

<div class="k-default-codeblock">
```

```
</div>
 27/183 [===>..........................] - ETA: 37:09 - loss: 0.8418 - accuracy: 0.7130

<div class="k-default-codeblock">
```

```
</div>
 28/183 [===>..........................] - ETA: 36:54 - loss: 0.8280 - accuracy: 0.7232

<div class="k-default-codeblock">
```

```
</div>
 29/183 [===>..........................] - ETA: 36:39 - loss: 0.8371 - accuracy: 0.7069

<div class="k-default-codeblock">
```

```
</div>
 30/183 [===>..........................] - ETA: 36:24 - loss: 0.8461 - accuracy: 0.7042

<div class="k-default-codeblock">
```

```
</div>
 31/183 [====>.........................] - ETA: 36:10 - loss: 0.8476 - accuracy: 0.7056

<div class="k-default-codeblock">
```

```
</div>
 32/183 [====>.........................] - ETA: 35:55 - loss: 0.8437 - accuracy: 0.7031

<div class="k-default-codeblock">
```

```
</div>
 33/183 [====>.........................] - ETA: 35:40 - loss: 0.8428 - accuracy: 0.7045

<div class="k-default-codeblock">
```

```
</div>
 34/183 [====>.........................] - ETA: 35:25 - loss: 0.8396 - accuracy: 0.7096

<div class="k-default-codeblock">
```

```
</div>
 35/183 [====>.........................] - ETA: 35:11 - loss: 0.8330 - accuracy: 0.7107

<div class="k-default-codeblock">
```

```
</div>
 36/183 [====>.........................] - ETA: 34:56 - loss: 0.8283 - accuracy: 0.7118

<div class="k-default-codeblock">
```

```
</div>
 37/183 [=====>........................] - ETA: 34:41 - loss: 0.8235 - accuracy: 0.7162

<div class="k-default-codeblock">
```

```
</div>
 38/183 [=====>........................] - ETA: 34:27 - loss: 0.8247 - accuracy: 0.7171

<div class="k-default-codeblock">
```

```
</div>
 39/183 [=====>........................] - ETA: 34:12 - loss: 0.8305 - accuracy: 0.7147

<div class="k-default-codeblock">
```

```
</div>
 40/183 [=====>........................] - ETA: 33:58 - loss: 0.8233 - accuracy: 0.7219

<div class="k-default-codeblock">
```

```
</div>
 41/183 [=====>........................] - ETA: 33:43 - loss: 0.8217 - accuracy: 0.7195

<div class="k-default-codeblock">
```

```
</div>
 42/183 [=====>........................] - ETA: 33:28 - loss: 0.8180 - accuracy: 0.7202

<div class="k-default-codeblock">
```

```
</div>
 43/183 [======>.......................] - ETA: 33:15 - loss: 0.8268 - accuracy: 0.7209

<div class="k-default-codeblock">
```

```
</div>
 44/183 [======>.......................] - ETA: 33:00 - loss: 0.8309 - accuracy: 0.7131

<div class="k-default-codeblock">
```

```
</div>
 45/183 [======>.......................] - ETA: 32:46 - loss: 0.8377 - accuracy: 0.7111

<div class="k-default-codeblock">
```

```
</div>
 46/183 [======>.......................] - ETA: 32:31 - loss: 0.8367 - accuracy: 0.7120

<div class="k-default-codeblock">
```

```
</div>
 47/183 [======>.......................] - ETA: 32:17 - loss: 0.8331 - accuracy: 0.7154

<div class="k-default-codeblock">
```

```
</div>
 48/183 [======>.......................] - ETA: 32:02 - loss: 0.8366 - accuracy: 0.7109

<div class="k-default-codeblock">
```

```
</div>
 49/183 [=======>......................] - ETA: 31:48 - loss: 0.8383 - accuracy: 0.7117

<div class="k-default-codeblock">
```

```
</div>
 50/183 [=======>......................] - ETA: 31:34 - loss: 0.8408 - accuracy: 0.7075

<div class="k-default-codeblock">
```

```
</div>
 51/183 [=======>......................] - ETA: 31:19 - loss: 0.8529 - accuracy: 0.7010

<div class="k-default-codeblock">
```

```
</div>
 52/183 [=======>......................] - ETA: 31:05 - loss: 0.8545 - accuracy: 0.7019

<div class="k-default-codeblock">
```

```
</div>
 53/183 [=======>......................] - ETA: 30:50 - loss: 0.8502 - accuracy: 0.7075

<div class="k-default-codeblock">
```

```
</div>
 54/183 [=======>......................] - ETA: 30:36 - loss: 0.8479 - accuracy: 0.7083

<div class="k-default-codeblock">
```

```
</div>
 55/183 [========>.....................] - ETA: 30:21 - loss: 0.8517 - accuracy: 0.7068

<div class="k-default-codeblock">
```

```
</div>
 56/183 [========>.....................] - ETA: 30:07 - loss: 0.8541 - accuracy: 0.7009

<div class="k-default-codeblock">
```

```
</div>
 57/183 [========>.....................] - ETA: 29:53 - loss: 0.8560 - accuracy: 0.6996

<div class="k-default-codeblock">
```

```
</div>
 58/183 [========>.....................] - ETA: 29:38 - loss: 0.8572 - accuracy: 0.6983

<div class="k-default-codeblock">
```

```
</div>
 59/183 [========>.....................] - ETA: 29:24 - loss: 0.8588 - accuracy: 0.6928

<div class="k-default-codeblock">
```

```
</div>
 60/183 [========>.....................] - ETA: 29:10 - loss: 0.8648 - accuracy: 0.6896

<div class="k-default-codeblock">
```

```
</div>
 61/183 [=========>....................] - ETA: 28:55 - loss: 0.8662 - accuracy: 0.6885

<div class="k-default-codeblock">
```

```
</div>
 62/183 [=========>....................] - ETA: 28:41 - loss: 0.8701 - accuracy: 0.6855

<div class="k-default-codeblock">
```

```
</div>
 63/183 [=========>....................] - ETA: 28:27 - loss: 0.8699 - accuracy: 0.6845

<div class="k-default-codeblock">
```

```
</div>
 64/183 [=========>....................] - ETA: 28:13 - loss: 0.8668 - accuracy: 0.6875

<div class="k-default-codeblock">
```

```
</div>
 65/183 [=========>....................] - ETA: 27:58 - loss: 0.8760 - accuracy: 0.6846

<div class="k-default-codeblock">
```

```
</div>
 66/183 [=========>....................] - ETA: 27:44 - loss: 0.8789 - accuracy: 0.6799

<div class="k-default-codeblock">
```

```
</div>
 67/183 [=========>....................] - ETA: 27:30 - loss: 0.8779 - accuracy: 0.6791

<div class="k-default-codeblock">
```

```
</div>
 68/183 [==========>...................] - ETA: 27:15 - loss: 0.8763 - accuracy: 0.6783

<div class="k-default-codeblock">
```

```
</div>
 69/183 [==========>...................] - ETA: 27:01 - loss: 0.8751 - accuracy: 0.6775

<div class="k-default-codeblock">
```

```
</div>
 70/183 [==========>...................] - ETA: 26:47 - loss: 0.8755 - accuracy: 0.6750

<div class="k-default-codeblock">
```

```
</div>
 71/183 [==========>...................] - ETA: 26:32 - loss: 0.8697 - accuracy: 0.6778

<div class="k-default-codeblock">
```

```
</div>
 72/183 [==========>...................] - ETA: 26:18 - loss: 0.8729 - accuracy: 0.6736

<div class="k-default-codeblock">
```

```
</div>
 73/183 [==========>...................] - ETA: 26:04 - loss: 0.8713 - accuracy: 0.6747

<div class="k-default-codeblock">
```

```
</div>
 74/183 [===========>..................] - ETA: 25:50 - loss: 0.8721 - accuracy: 0.6740

<div class="k-default-codeblock">
```

```
</div>
 75/183 [===========>..................] - ETA: 25:35 - loss: 0.8787 - accuracy: 0.6700

<div class="k-default-codeblock">
```

```
</div>
 76/183 [===========>..................] - ETA: 25:21 - loss: 0.8769 - accuracy: 0.6694

<div class="k-default-codeblock">
```

```
</div>
 77/183 [===========>..................] - ETA: 25:07 - loss: 0.8743 - accuracy: 0.6705

<div class="k-default-codeblock">
```

```
</div>
 78/183 [===========>..................] - ETA: 24:53 - loss: 0.8794 - accuracy: 0.6683

<div class="k-default-codeblock">
```

```
</div>
 79/183 [===========>..................] - ETA: 24:39 - loss: 0.8800 - accuracy: 0.6677

<div class="k-default-codeblock">
```

```
</div>
 80/183 [============>.................] - ETA: 24:24 - loss: 0.8773 - accuracy: 0.6687

<div class="k-default-codeblock">
```

```
</div>
 81/183 [============>.................] - ETA: 24:10 - loss: 0.8769 - accuracy: 0.6682

<div class="k-default-codeblock">
```

```
</div>
 82/183 [============>.................] - ETA: 23:56 - loss: 0.8788 - accuracy: 0.6677

<div class="k-default-codeblock">
```

```
</div>
 83/183 [============>.................] - ETA: 23:42 - loss: 0.8773 - accuracy: 0.6672

<div class="k-default-codeblock">
```

```
</div>
 84/183 [============>.................] - ETA: 23:27 - loss: 0.8800 - accuracy: 0.6652

<div class="k-default-codeblock">
```

```
</div>
 85/183 [============>.................] - ETA: 23:13 - loss: 0.8819 - accuracy: 0.6632

<div class="k-default-codeblock">
```

```
</div>
 86/183 [=============>................] - ETA: 22:59 - loss: 0.8789 - accuracy: 0.6657

<div class="k-default-codeblock">
```

```
</div>
 87/183 [=============>................] - ETA: 22:45 - loss: 0.8813 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
 88/183 [=============>................] - ETA: 22:31 - loss: 0.8872 - accuracy: 0.6619

<div class="k-default-codeblock">
```

```
</div>
 89/183 [=============>................] - ETA: 22:16 - loss: 0.8847 - accuracy: 0.6629

<div class="k-default-codeblock">
```

```
</div>
 90/183 [=============>................] - ETA: 22:02 - loss: 0.8823 - accuracy: 0.6639

<div class="k-default-codeblock">
```

```
</div>
 91/183 [=============>................] - ETA: 21:48 - loss: 0.8859 - accuracy: 0.6621

<div class="k-default-codeblock">
```

```
</div>
 92/183 [==============>...............] - ETA: 21:34 - loss: 0.8902 - accuracy: 0.6617

<div class="k-default-codeblock">
```

```
</div>
 93/183 [==============>...............] - ETA: 21:20 - loss: 0.8896 - accuracy: 0.6626

<div class="k-default-codeblock">
```

```
</div>
 94/183 [==============>...............] - ETA: 21:05 - loss: 0.8932 - accuracy: 0.6622

<div class="k-default-codeblock">
```

```
</div>
 95/183 [==============>...............] - ETA: 20:51 - loss: 0.8946 - accuracy: 0.6618

<div class="k-default-codeblock">
```

```
</div>
 96/183 [==============>...............] - ETA: 20:37 - loss: 0.8960 - accuracy: 0.6628

<div class="k-default-codeblock">
```

```
</div>
 97/183 [==============>...............] - ETA: 20:23 - loss: 0.8934 - accuracy: 0.6649

<div class="k-default-codeblock">
```

```
</div>
 98/183 [===============>..............] - ETA: 20:08 - loss: 0.8942 - accuracy: 0.6658

<div class="k-default-codeblock">
```

```
</div>
 99/183 [===============>..............] - ETA: 19:54 - loss: 0.9002 - accuracy: 0.6629

<div class="k-default-codeblock">
```

```
</div>
100/183 [===============>..............] - ETA: 19:40 - loss: 0.8975 - accuracy: 0.6637

<div class="k-default-codeblock">
```

```
</div>
101/183 [===============>..............] - ETA: 19:26 - loss: 0.8965 - accuracy: 0.6646

<div class="k-default-codeblock">
```

```
</div>
102/183 [===============>..............] - ETA: 19:11 - loss: 0.8963 - accuracy: 0.6642

<div class="k-default-codeblock">
```

```
</div>
103/183 [===============>..............] - ETA: 18:57 - loss: 0.8950 - accuracy: 0.6650

<div class="k-default-codeblock">
```

```
</div>
104/183 [================>.............] - ETA: 18:43 - loss: 0.8935 - accuracy: 0.6659

<div class="k-default-codeblock">
```

```
</div>
105/183 [================>.............] - ETA: 18:29 - loss: 0.8912 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
106/183 [================>.............] - ETA: 18:14 - loss: 0.8940 - accuracy: 0.6663

<div class="k-default-codeblock">
```

```
</div>
107/183 [================>.............] - ETA: 18:00 - loss: 0.8913 - accuracy: 0.6671

<div class="k-default-codeblock">
```

```
</div>
108/183 [================>.............] - ETA: 17:46 - loss: 0.8920 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
109/183 [================>.............] - ETA: 17:32 - loss: 0.8915 - accuracy: 0.6674

<div class="k-default-codeblock">
```

```
</div>
110/183 [=================>............] - ETA: 17:18 - loss: 0.8893 - accuracy: 0.6682

<div class="k-default-codeblock">
```

```
</div>
111/183 [=================>............] - ETA: 17:03 - loss: 0.8909 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
112/183 [=================>............] - ETA: 16:49 - loss: 0.8894 - accuracy: 0.6674

<div class="k-default-codeblock">
```

```
</div>
113/183 [=================>............] - ETA: 16:35 - loss: 0.8909 - accuracy: 0.6659

<div class="k-default-codeblock">
```

```
</div>
114/183 [=================>............] - ETA: 16:21 - loss: 0.8908 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
115/183 [=================>............] - ETA: 16:07 - loss: 0.8943 - accuracy: 0.6652

<div class="k-default-codeblock">
```

```
</div>
116/183 [==================>...........] - ETA: 15:53 - loss: 0.8956 - accuracy: 0.6659

<div class="k-default-codeblock">
```

```
</div>
117/183 [==================>...........] - ETA: 15:40 - loss: 0.8949 - accuracy: 0.6656

<div class="k-default-codeblock">
```

```
</div>
118/183 [==================>...........] - ETA: 15:26 - loss: 0.8907 - accuracy: 0.6684

<div class="k-default-codeblock">
```

```
</div>
119/183 [==================>...........] - ETA: 15:11 - loss: 0.8892 - accuracy: 0.6691

<div class="k-default-codeblock">
```

```
</div>
120/183 [==================>...........] - ETA: 14:57 - loss: 0.8889 - accuracy: 0.6677

<div class="k-default-codeblock">
```

```
</div>
121/183 [==================>...........] - ETA: 14:43 - loss: 0.8853 - accuracy: 0.6694

<div class="k-default-codeblock">
```

```
</div>
122/183 [===================>..........] - ETA: 14:29 - loss: 0.8860 - accuracy: 0.6691

<div class="k-default-codeblock">
```

```
</div>
123/183 [===================>..........] - ETA: 14:14 - loss: 0.8832 - accuracy: 0.6707

<div class="k-default-codeblock">
```

```
</div>
124/183 [===================>..........] - ETA: 14:00 - loss: 0.8812 - accuracy: 0.6724

<div class="k-default-codeblock">
```

```
</div>
125/183 [===================>..........] - ETA: 13:46 - loss: 0.8806 - accuracy: 0.6720

<div class="k-default-codeblock">
```

```
</div>
126/183 [===================>..........] - ETA: 13:32 - loss: 0.8807 - accuracy: 0.6696

<div class="k-default-codeblock">
```

```
</div>
127/183 [===================>..........] - ETA: 13:17 - loss: 0.8778 - accuracy: 0.6713

<div class="k-default-codeblock">
```

```
</div>
128/183 [===================>..........] - ETA: 13:03 - loss: 0.8790 - accuracy: 0.6719

<div class="k-default-codeblock">
```

```
</div>
129/183 [====================>.........] - ETA: 12:49 - loss: 0.8775 - accuracy: 0.6725

<div class="k-default-codeblock">
```

```
</div>
130/183 [====================>.........] - ETA: 12:35 - loss: 0.8775 - accuracy: 0.6731

<div class="k-default-codeblock">
```

```
</div>
131/183 [====================>.........] - ETA: 12:21 - loss: 0.8785 - accuracy: 0.6718

<div class="k-default-codeblock">
```

```
</div>
132/183 [====================>.........] - ETA: 12:07 - loss: 0.8759 - accuracy: 0.6733

<div class="k-default-codeblock">
```

```
</div>
133/183 [====================>.........] - ETA: 11:52 - loss: 0.8739 - accuracy: 0.6739

<div class="k-default-codeblock">
```

```
</div>
134/183 [====================>.........] - ETA: 11:38 - loss: 0.8749 - accuracy: 0.6735

<div class="k-default-codeblock">
```

```
</div>
135/183 [=====================>........] - ETA: 11:24 - loss: 0.8742 - accuracy: 0.6741

<div class="k-default-codeblock">
```

```
</div>
136/183 [=====================>........] - ETA: 11:10 - loss: 0.8738 - accuracy: 0.6728

<div class="k-default-codeblock">
```

```
</div>
137/183 [=====================>........] - ETA: 10:56 - loss: 0.8752 - accuracy: 0.6715

<div class="k-default-codeblock">
```

```
</div>
138/183 [=====================>........] - ETA: 10:41 - loss: 0.8792 - accuracy: 0.6694

<div class="k-default-codeblock">
```

```
</div>
139/183 [=====================>........] - ETA: 10:27 - loss: 0.8795 - accuracy: 0.6691

<div class="k-default-codeblock">
```

```
</div>
140/183 [=====================>........] - ETA: 10:13 - loss: 0.8808 - accuracy: 0.6670

<div class="k-default-codeblock">
```

```
</div>
141/183 [======================>.......] - ETA: 9:59 - loss: 0.8836 - accuracy: 0.6667 

<div class="k-default-codeblock">
```

```
</div>
142/183 [======================>.......] - ETA: 9:44 - loss: 0.8848 - accuracy: 0.6664

<div class="k-default-codeblock">
```

```
</div>
143/183 [======================>.......] - ETA: 9:30 - loss: 0.8833 - accuracy: 0.6661

<div class="k-default-codeblock">
```

```
</div>
144/183 [======================>.......] - ETA: 9:16 - loss: 0.8850 - accuracy: 0.6641

<div class="k-default-codeblock">
```

```
</div>
145/183 [======================>.......] - ETA: 9:02 - loss: 0.8841 - accuracy: 0.6647

<div class="k-default-codeblock">
```

```
</div>
146/183 [======================>.......] - ETA: 8:47 - loss: 0.8825 - accuracy: 0.6661

<div class="k-default-codeblock">
```

```
</div>
147/183 [=======================>......] - ETA: 8:33 - loss: 0.8814 - accuracy: 0.6675

<div class="k-default-codeblock">
```

```
</div>
148/183 [=======================>......] - ETA: 8:19 - loss: 0.8801 - accuracy: 0.6698

<div class="k-default-codeblock">
```

```
</div>
149/183 [=======================>......] - ETA: 8:05 - loss: 0.8782 - accuracy: 0.6703

<div class="k-default-codeblock">
```

```
</div>
150/183 [=======================>......] - ETA: 7:51 - loss: 0.8789 - accuracy: 0.6700

<div class="k-default-codeblock">
```

```
</div>
151/183 [=======================>......] - ETA: 7:36 - loss: 0.8776 - accuracy: 0.6705

<div class="k-default-codeblock">
```

```
</div>
152/183 [=======================>......] - ETA: 7:22 - loss: 0.8797 - accuracy: 0.6694

<div class="k-default-codeblock">
```

```
</div>
153/183 [========================>.....] - ETA: 7:08 - loss: 0.8804 - accuracy: 0.6691

<div class="k-default-codeblock">
```

```
</div>
154/183 [========================>.....] - ETA: 6:53 - loss: 0.8793 - accuracy: 0.6688

<div class="k-default-codeblock">
```

```
</div>
155/183 [========================>.....] - ETA: 6:39 - loss: 0.8793 - accuracy: 0.6685

<div class="k-default-codeblock">
```

```
</div>
156/183 [========================>.....] - ETA: 6:25 - loss: 0.8776 - accuracy: 0.6691

<div class="k-default-codeblock">
```

```
</div>
157/183 [========================>.....] - ETA: 6:11 - loss: 0.8755 - accuracy: 0.6704

<div class="k-default-codeblock">
```

```
</div>
158/183 [========================>.....] - ETA: 5:57 - loss: 0.8768 - accuracy: 0.6701

<div class="k-default-codeblock">
```

```
</div>
159/183 [=========================>....] - ETA: 5:42 - loss: 0.8771 - accuracy: 0.6698

<div class="k-default-codeblock">
```

```
</div>
160/183 [=========================>....] - ETA: 5:28 - loss: 0.8761 - accuracy: 0.6703

<div class="k-default-codeblock">
```

```
</div>
161/183 [=========================>....] - ETA: 5:14 - loss: 0.8734 - accuracy: 0.6716

<div class="k-default-codeblock">
```

```
</div>
162/183 [=========================>....] - ETA: 4:59 - loss: 0.8730 - accuracy: 0.6713

<div class="k-default-codeblock">
```

```
</div>
163/183 [=========================>....] - ETA: 4:45 - loss: 0.8722 - accuracy: 0.6710

<div class="k-default-codeblock">
```

```
</div>
164/183 [=========================>....] - ETA: 4:31 - loss: 0.8728 - accuracy: 0.6707

<div class="k-default-codeblock">
```

```
</div>
165/183 [==========================>...] - ETA: 4:17 - loss: 0.8743 - accuracy: 0.6697

<div class="k-default-codeblock">
```

```
</div>
166/183 [==========================>...] - ETA: 4:02 - loss: 0.8745 - accuracy: 0.6702

<div class="k-default-codeblock">
```

```
</div>
167/183 [==========================>...] - ETA: 3:48 - loss: 0.8753 - accuracy: 0.6684

<div class="k-default-codeblock">
```

```
</div>
168/183 [==========================>...] - ETA: 3:34 - loss: 0.8738 - accuracy: 0.6696

<div class="k-default-codeblock">
```

```
</div>
169/183 [==========================>...] - ETA: 3:20 - loss: 0.8724 - accuracy: 0.6694

<div class="k-default-codeblock">
```

```
</div>
170/183 [==========================>...] - ETA: 3:05 - loss: 0.8757 - accuracy: 0.6676

<div class="k-default-codeblock">
```

```
</div>
171/183 [===========================>..] - ETA: 2:51 - loss: 0.8752 - accuracy: 0.6674

<div class="k-default-codeblock">
```

```
</div>
172/183 [===========================>..] - ETA: 2:37 - loss: 0.8768 - accuracy: 0.6672

<div class="k-default-codeblock">
```

```
</div>
173/183 [===========================>..] - ETA: 2:22 - loss: 0.8750 - accuracy: 0.6676

<div class="k-default-codeblock">
```

```
</div>
174/183 [===========================>..] - ETA: 2:08 - loss: 0.8753 - accuracy: 0.6681

<div class="k-default-codeblock">
```

```
</div>
175/183 [===========================>..] - ETA: 1:54 - loss: 0.8749 - accuracy: 0.6679

<div class="k-default-codeblock">
```

```
</div>
176/183 [===========================>..] - ETA: 1:40 - loss: 0.8739 - accuracy: 0.6683

<div class="k-default-codeblock">
```

```
</div>
177/183 [============================>.] - ETA: 1:25 - loss: 0.8720 - accuracy: 0.6688

<div class="k-default-codeblock">
```

```
</div>
178/183 [============================>.] - ETA: 1:11 - loss: 0.8719 - accuracy: 0.6692

<div class="k-default-codeblock">
```

```
</div>
179/183 [============================>.] - ETA: 57s - loss: 0.8739 - accuracy: 0.6669 

<div class="k-default-codeblock">
```

```
</div>
180/183 [============================>.] - ETA: 42s - loss: 0.8729 - accuracy: 0.6674

<div class="k-default-codeblock">
```

```
</div>
181/183 [============================>.] - ETA: 28s - loss: 0.8722 - accuracy: 0.6678

<div class="k-default-codeblock">
```

```
</div>
182/183 [============================>.] - ETA: 14s - loss: 0.8715 - accuracy: 0.6683

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - ETA: 0s - loss: 0.8728 - accuracy: 0.6680 

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - 2778s 15s/step - loss: 0.8728 - accuracy: 0.6680 - val_loss: 0.7296 - val_accuracy: 0.7525 - lr: 4.7230e-06


<div class="k-default-codeblock">
```
Epoch 5/5
183/183 [==============================] - 2764s 15s/step - loss: 0.7714 - accuracy: 0.7158 - val_loss: 0.7098 - val_accuracy: 0.7500 - lr: 4.4984e-06

```
</div>
    
  1/183 [..............................] - ETA: 45:36 - loss: 0.7472 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
  2/183 [..............................] - ETA: 44:02 - loss: 0.7747 - accuracy: 0.6875

<div class="k-default-codeblock">
```

```
</div>
  3/183 [..............................] - ETA: 43:51 - loss: 0.7801 - accuracy: 0.6667

<div class="k-default-codeblock">
```

```
</div>
  4/183 [..............................] - ETA: 43:41 - loss: 0.7945 - accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
  5/183 [..............................] - ETA: 43:26 - loss: 0.8398 - accuracy: 0.6500

<div class="k-default-codeblock">
```

```
</div>
  6/183 [..............................] - ETA: 43:07 - loss: 0.7811 - accuracy: 0.6875

<div class="k-default-codeblock">
```

```
</div>
  7/183 [>.............................] - ETA: 42:47 - loss: 0.8120 - accuracy: 0.6964

<div class="k-default-codeblock">
```

```
</div>
  8/183 [>.............................] - ETA: 42:30 - loss: 0.7819 - accuracy: 0.6875

<div class="k-default-codeblock">
```

```
</div>
  9/183 [>.............................] - ETA: 42:09 - loss: 0.7870 - accuracy: 0.6944

<div class="k-default-codeblock">
```

```
</div>
 10/183 [>.............................] - ETA: 41:50 - loss: 0.7538 - accuracy: 0.7125

<div class="k-default-codeblock">
```

```
</div>
 11/183 [>.............................] - ETA: 41:33 - loss: 0.7518 - accuracy: 0.7159

<div class="k-default-codeblock">
```

```
</div>
 12/183 [>.............................] - ETA: 41:17 - loss: 0.7440 - accuracy: 0.7188

<div class="k-default-codeblock">
```

```
</div>
 13/183 [=>............................] - ETA: 41:00 - loss: 0.7263 - accuracy: 0.7404

<div class="k-default-codeblock">
```

```
</div>
 14/183 [=>............................] - ETA: 40:44 - loss: 0.7001 - accuracy: 0.7589

<div class="k-default-codeblock">
```

```
</div>
 15/183 [=>............................] - ETA: 40:28 - loss: 0.7132 - accuracy: 0.7583

<div class="k-default-codeblock">
```

```
</div>
 16/183 [=>............................] - ETA: 40:11 - loss: 0.7123 - accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
 17/183 [=>............................] - ETA: 39:55 - loss: 0.7370 - accuracy: 0.7426

<div class="k-default-codeblock">
```

```
</div>
 18/183 [=>............................] - ETA: 39:38 - loss: 0.7365 - accuracy: 0.7431

<div class="k-default-codeblock">
```

```
</div>
 19/183 [==>...........................] - ETA: 39:22 - loss: 0.7469 - accuracy: 0.7303

<div class="k-default-codeblock">
```

```
</div>
 20/183 [==>...........................] - ETA: 39:07 - loss: 0.7500 - accuracy: 0.7375

<div class="k-default-codeblock">
```

```
</div>
 21/183 [==>...........................] - ETA: 38:52 - loss: 0.7459 - accuracy: 0.7440

<div class="k-default-codeblock">
```

```
</div>
 22/183 [==>...........................] - ETA: 38:37 - loss: 0.7396 - accuracy: 0.7557

<div class="k-default-codeblock">
```

```
</div>
 23/183 [==>...........................] - ETA: 38:21 - loss: 0.7387 - accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
 24/183 [==>...........................] - ETA: 38:06 - loss: 0.7266 - accuracy: 0.7552

<div class="k-default-codeblock">
```

```
</div>
 25/183 [===>..........................] - ETA: 37:51 - loss: 0.7416 - accuracy: 0.7400

<div class="k-default-codeblock">
```

```
</div>
 26/183 [===>..........................] - ETA: 37:37 - loss: 0.7411 - accuracy: 0.7404

<div class="k-default-codeblock">
```

```
</div>
 27/183 [===>..........................] - ETA: 37:23 - loss: 0.7337 - accuracy: 0.7454

<div class="k-default-codeblock">
```

```
</div>
 28/183 [===>..........................] - ETA: 37:09 - loss: 0.7336 - accuracy: 0.7411

<div class="k-default-codeblock">
```

```
</div>
 29/183 [===>..........................] - ETA: 36:54 - loss: 0.7294 - accuracy: 0.7457

<div class="k-default-codeblock">
```

```
</div>
 30/183 [===>..........................] - ETA: 36:40 - loss: 0.7293 - accuracy: 0.7417

<div class="k-default-codeblock">
```

```
</div>
 31/183 [====>.........................] - ETA: 36:26 - loss: 0.7399 - accuracy: 0.7379

<div class="k-default-codeblock">
```

```
</div>
 32/183 [====>.........................] - ETA: 36:11 - loss: 0.7355 - accuracy: 0.7422

<div class="k-default-codeblock">
```

```
</div>
 33/183 [====>.........................] - ETA: 35:57 - loss: 0.7283 - accuracy: 0.7462

<div class="k-default-codeblock">
```

```
</div>
 34/183 [====>.........................] - ETA: 35:43 - loss: 0.7330 - accuracy: 0.7426

<div class="k-default-codeblock">
```

```
</div>
 35/183 [====>.........................] - ETA: 35:28 - loss: 0.7390 - accuracy: 0.7321

<div class="k-default-codeblock">
```

```
</div>
 36/183 [====>.........................] - ETA: 35:14 - loss: 0.7381 - accuracy: 0.7292

<div class="k-default-codeblock">
```

```
</div>
 37/183 [=====>........................] - ETA: 34:59 - loss: 0.7374 - accuracy: 0.7264

<div class="k-default-codeblock">
```

```
</div>
 38/183 [=====>........................] - ETA: 34:45 - loss: 0.7359 - accuracy: 0.7303

<div class="k-default-codeblock">
```

```
</div>
 39/183 [=====>........................] - ETA: 34:31 - loss: 0.7345 - accuracy: 0.7340

<div class="k-default-codeblock">
```

```
</div>
 40/183 [=====>........................] - ETA: 34:16 - loss: 0.7355 - accuracy: 0.7344

<div class="k-default-codeblock">
```

```
</div>
 41/183 [=====>........................] - ETA: 34:01 - loss: 0.7277 - accuracy: 0.7378

<div class="k-default-codeblock">
```

```
</div>
 42/183 [=====>........................] - ETA: 33:48 - loss: 0.7305 - accuracy: 0.7381

<div class="k-default-codeblock">
```

```
</div>
 43/183 [======>.......................] - ETA: 33:33 - loss: 0.7320 - accuracy: 0.7355

<div class="k-default-codeblock">
```

```
</div>
 44/183 [======>.......................] - ETA: 33:18 - loss: 0.7283 - accuracy: 0.7386

<div class="k-default-codeblock">
```

```
</div>
 45/183 [======>.......................] - ETA: 33:03 - loss: 0.7341 - accuracy: 0.7333

<div class="k-default-codeblock">
```

```
</div>
 46/183 [======>.......................] - ETA: 32:48 - loss: 0.7445 - accuracy: 0.7255

<div class="k-default-codeblock">
```

```
</div>
 47/183 [======>.......................] - ETA: 32:33 - loss: 0.7413 - accuracy: 0.7287

<div class="k-default-codeblock">
```

```
</div>
 48/183 [======>.......................] - ETA: 32:18 - loss: 0.7428 - accuracy: 0.7292

<div class="k-default-codeblock">
```

```
</div>
 49/183 [=======>......................] - ETA: 32:03 - loss: 0.7457 - accuracy: 0.7270

<div class="k-default-codeblock">
```

```
</div>
 50/183 [=======>......................] - ETA: 31:49 - loss: 0.7504 - accuracy: 0.7250

<div class="k-default-codeblock">
```

```
</div>
 51/183 [=======>......................] - ETA: 31:34 - loss: 0.7524 - accuracy: 0.7181

<div class="k-default-codeblock">
```

```
</div>
 52/183 [=======>......................] - ETA: 31:19 - loss: 0.7636 - accuracy: 0.7163

<div class="k-default-codeblock">
```

```
</div>
 53/183 [=======>......................] - ETA: 31:04 - loss: 0.7691 - accuracy: 0.7123

<div class="k-default-codeblock">
```

```
</div>
 54/183 [=======>......................] - ETA: 30:50 - loss: 0.7681 - accuracy: 0.7130

<div class="k-default-codeblock">
```

```
</div>
 55/183 [========>.....................] - ETA: 30:35 - loss: 0.7717 - accuracy: 0.7091

<div class="k-default-codeblock">
```

```
</div>
 56/183 [========>.....................] - ETA: 30:20 - loss: 0.7725 - accuracy: 0.7098

<div class="k-default-codeblock">
```

```
</div>
 57/183 [========>.....................] - ETA: 30:05 - loss: 0.7781 - accuracy: 0.7105

<div class="k-default-codeblock">
```

```
</div>
 58/183 [========>.....................] - ETA: 29:51 - loss: 0.7811 - accuracy: 0.7091

<div class="k-default-codeblock">
```

```
</div>
 59/183 [========>.....................] - ETA: 29:36 - loss: 0.7856 - accuracy: 0.7097

<div class="k-default-codeblock">
```

```
</div>
 60/183 [========>.....................] - ETA: 29:22 - loss: 0.7853 - accuracy: 0.7083

<div class="k-default-codeblock">
```

```
</div>
 61/183 [=========>....................] - ETA: 29:07 - loss: 0.7866 - accuracy: 0.7111

<div class="k-default-codeblock">
```

```
</div>
 62/183 [=========>....................] - ETA: 28:53 - loss: 0.7861 - accuracy: 0.7117

<div class="k-default-codeblock">
```

```
</div>
 63/183 [=========>....................] - ETA: 28:38 - loss: 0.7909 - accuracy: 0.7083

<div class="k-default-codeblock">
```

```
</div>
 64/183 [=========>....................] - ETA: 28:24 - loss: 0.7890 - accuracy: 0.7070

<div class="k-default-codeblock">
```

```
</div>
 65/183 [=========>....................] - ETA: 28:09 - loss: 0.7919 - accuracy: 0.7058

<div class="k-default-codeblock">
```

```
</div>
 66/183 [=========>....................] - ETA: 27:54 - loss: 0.7981 - accuracy: 0.7027

<div class="k-default-codeblock">
```

```
</div>
 67/183 [=========>....................] - ETA: 27:40 - loss: 0.7995 - accuracy: 0.7015

<div class="k-default-codeblock">
```

```
</div>
 68/183 [==========>...................] - ETA: 27:25 - loss: 0.7972 - accuracy: 0.7022

<div class="k-default-codeblock">
```

```
</div>
 69/183 [==========>...................] - ETA: 27:11 - loss: 0.7957 - accuracy: 0.7029

<div class="k-default-codeblock">
```

```
</div>
 70/183 [==========>...................] - ETA: 26:57 - loss: 0.7951 - accuracy: 0.7036

<div class="k-default-codeblock">
```

```
</div>
 71/183 [==========>...................] - ETA: 26:43 - loss: 0.7939 - accuracy: 0.7042

<div class="k-default-codeblock">
```

```
</div>
 72/183 [==========>...................] - ETA: 26:28 - loss: 0.7897 - accuracy: 0.7049

<div class="k-default-codeblock">
```

```
</div>
 73/183 [==========>...................] - ETA: 26:13 - loss: 0.7879 - accuracy: 0.7055

<div class="k-default-codeblock">
```

```
</div>
 74/183 [===========>..................] - ETA: 25:59 - loss: 0.7864 - accuracy: 0.7044

<div class="k-default-codeblock">
```

```
</div>
 75/183 [===========>..................] - ETA: 25:44 - loss: 0.7859 - accuracy: 0.7050

<div class="k-default-codeblock">
```

```
</div>
 76/183 [===========>..................] - ETA: 25:30 - loss: 0.7883 - accuracy: 0.7056

<div class="k-default-codeblock">
```

```
</div>
 77/183 [===========>..................] - ETA: 25:16 - loss: 0.7881 - accuracy: 0.7062

<div class="k-default-codeblock">
```

```
</div>
 78/183 [===========>..................] - ETA: 25:01 - loss: 0.7931 - accuracy: 0.7035

<div class="k-default-codeblock">
```

```
</div>
 79/183 [===========>..................] - ETA: 24:47 - loss: 0.7971 - accuracy: 0.7009

<div class="k-default-codeblock">
```

```
</div>
 80/183 [============>.................] - ETA: 24:32 - loss: 0.7980 - accuracy: 0.7000

<div class="k-default-codeblock">
```

```
</div>
 81/183 [============>.................] - ETA: 24:18 - loss: 0.7948 - accuracy: 0.7022

<div class="k-default-codeblock">
```

```
</div>
 82/183 [============>.................] - ETA: 24:03 - loss: 0.7942 - accuracy: 0.7027

<div class="k-default-codeblock">
```

```
</div>
 83/183 [============>.................] - ETA: 23:49 - loss: 0.7900 - accuracy: 0.7048

<div class="k-default-codeblock">
```

```
</div>
 84/183 [============>.................] - ETA: 23:35 - loss: 0.7926 - accuracy: 0.7024

<div class="k-default-codeblock">
```

```
</div>
 85/183 [============>.................] - ETA: 23:20 - loss: 0.7966 - accuracy: 0.7000

<div class="k-default-codeblock">
```

```
</div>
 86/183 [=============>................] - ETA: 23:06 - loss: 0.7997 - accuracy: 0.6977

<div class="k-default-codeblock">
```

```
</div>
 87/183 [=============>................] - ETA: 22:52 - loss: 0.7973 - accuracy: 0.6997

<div class="k-default-codeblock">
```

```
</div>
 88/183 [=============>................] - ETA: 22:38 - loss: 0.8052 - accuracy: 0.6960

<div class="k-default-codeblock">
```

```
</div>
 89/183 [=============>................] - ETA: 22:23 - loss: 0.8064 - accuracy: 0.6952

<div class="k-default-codeblock">
```

```
</div>
 90/183 [=============>................] - ETA: 22:09 - loss: 0.8023 - accuracy: 0.6972

<div class="k-default-codeblock">
```

```
</div>
 91/183 [=============>................] - ETA: 21:54 - loss: 0.8045 - accuracy: 0.6978

<div class="k-default-codeblock">
```

```
</div>
 92/183 [==============>...............] - ETA: 21:40 - loss: 0.8056 - accuracy: 0.6984

<div class="k-default-codeblock">
```

```
</div>
 93/183 [==============>...............] - ETA: 21:26 - loss: 0.8058 - accuracy: 0.6976

<div class="k-default-codeblock">
```

```
</div>
 94/183 [==============>...............] - ETA: 21:11 - loss: 0.8039 - accuracy: 0.6981

<div class="k-default-codeblock">
```

```
</div>
 95/183 [==============>...............] - ETA: 20:57 - loss: 0.8073 - accuracy: 0.6987

<div class="k-default-codeblock">
```

```
</div>
 96/183 [==============>...............] - ETA: 20:42 - loss: 0.8063 - accuracy: 0.6992

<div class="k-default-codeblock">
```

```
</div>
 97/183 [==============>...............] - ETA: 20:28 - loss: 0.8076 - accuracy: 0.6972

<div class="k-default-codeblock">
```

```
</div>
 98/183 [===============>..............] - ETA: 20:14 - loss: 0.8061 - accuracy: 0.6977

<div class="k-default-codeblock">
```

```
</div>
 99/183 [===============>..............] - ETA: 19:59 - loss: 0.8094 - accuracy: 0.6944

<div class="k-default-codeblock">
```

```
</div>
100/183 [===============>..............] - ETA: 19:45 - loss: 0.8154 - accuracy: 0.6925

<div class="k-default-codeblock">
```

```
</div>
101/183 [===============>..............] - ETA: 19:31 - loss: 0.8130 - accuracy: 0.6943

<div class="k-default-codeblock">
```

```
</div>
102/183 [===============>..............] - ETA: 19:16 - loss: 0.8113 - accuracy: 0.6949

<div class="k-default-codeblock">
```

```
</div>
103/183 [===============>..............] - ETA: 19:02 - loss: 0.8098 - accuracy: 0.6966

<div class="k-default-codeblock">
```

```
</div>
104/183 [================>.............] - ETA: 18:48 - loss: 0.8088 - accuracy: 0.6971

<div class="k-default-codeblock">
```

```
</div>
105/183 [================>.............] - ETA: 18:33 - loss: 0.8074 - accuracy: 0.6976

<div class="k-default-codeblock">
```

```
</div>
106/183 [================>.............] - ETA: 18:19 - loss: 0.8061 - accuracy: 0.6981

<div class="k-default-codeblock">
```

```
</div>
107/183 [================>.............] - ETA: 18:05 - loss: 0.8043 - accuracy: 0.6986

<div class="k-default-codeblock">
```

```
</div>
108/183 [================>.............] - ETA: 17:50 - loss: 0.8021 - accuracy: 0.7002

<div class="k-default-codeblock">
```

```
</div>
109/183 [================>.............] - ETA: 17:36 - loss: 0.8002 - accuracy: 0.7018

<div class="k-default-codeblock">
```

```
</div>
110/183 [=================>............] - ETA: 17:22 - loss: 0.7979 - accuracy: 0.7045

<div class="k-default-codeblock">
```

```
</div>
111/183 [=================>............] - ETA: 17:07 - loss: 0.7951 - accuracy: 0.7050

<div class="k-default-codeblock">
```

```
</div>
112/183 [=================>............] - ETA: 16:53 - loss: 0.7978 - accuracy: 0.7009

<div class="k-default-codeblock">
```

```
</div>
113/183 [=================>............] - ETA: 16:39 - loss: 0.7977 - accuracy: 0.7013

<div class="k-default-codeblock">
```

```
</div>
114/183 [=================>............] - ETA: 16:24 - loss: 0.8027 - accuracy: 0.6974

<div class="k-default-codeblock">
```

```
</div>
115/183 [=================>............] - ETA: 16:10 - loss: 0.7983 - accuracy: 0.7000

<div class="k-default-codeblock">
```

```
</div>
116/183 [==================>...........] - ETA: 15:56 - loss: 0.8009 - accuracy: 0.6983

<div class="k-default-codeblock">
```

```
</div>
117/183 [==================>...........] - ETA: 15:41 - loss: 0.8030 - accuracy: 0.6976

<div class="k-default-codeblock">
```

```
</div>
118/183 [==================>...........] - ETA: 15:27 - loss: 0.8015 - accuracy: 0.6981

<div class="k-default-codeblock">
```

```
</div>
119/183 [==================>...........] - ETA: 15:13 - loss: 0.7986 - accuracy: 0.6996

<div class="k-default-codeblock">
```

```
</div>
120/183 [==================>...........] - ETA: 14:59 - loss: 0.7966 - accuracy: 0.7010

<div class="k-default-codeblock">
```

```
</div>
121/183 [==================>...........] - ETA: 14:44 - loss: 0.7946 - accuracy: 0.7025

<div class="k-default-codeblock">
```

```
</div>
122/183 [===================>..........] - ETA: 14:30 - loss: 0.7923 - accuracy: 0.7049

<div class="k-default-codeblock">
```

```
</div>
123/183 [===================>..........] - ETA: 14:16 - loss: 0.7925 - accuracy: 0.7043

<div class="k-default-codeblock">
```

```
</div>
124/183 [===================>..........] - ETA: 14:01 - loss: 0.7932 - accuracy: 0.7046

<div class="k-default-codeblock">
```

```
</div>
125/183 [===================>..........] - ETA: 13:47 - loss: 0.7914 - accuracy: 0.7070

<div class="k-default-codeblock">
```

```
</div>
126/183 [===================>..........] - ETA: 13:33 - loss: 0.7918 - accuracy: 0.7054

<div class="k-default-codeblock">
```

```
</div>
127/183 [===================>..........] - ETA: 13:19 - loss: 0.7911 - accuracy: 0.7057

<div class="k-default-codeblock">
```

```
</div>
128/183 [===================>..........] - ETA: 13:04 - loss: 0.7901 - accuracy: 0.7061

<div class="k-default-codeblock">
```

```
</div>
129/183 [====================>.........] - ETA: 12:50 - loss: 0.7908 - accuracy: 0.7054

<div class="k-default-codeblock">
```

```
</div>
130/183 [====================>.........] - ETA: 12:36 - loss: 0.7887 - accuracy: 0.7067

<div class="k-default-codeblock">
```

```
</div>
131/183 [====================>.........] - ETA: 12:21 - loss: 0.7877 - accuracy: 0.7080

<div class="k-default-codeblock">
```

```
</div>
132/183 [====================>.........] - ETA: 12:07 - loss: 0.7871 - accuracy: 0.7083

<div class="k-default-codeblock">
```

```
</div>
133/183 [====================>.........] - ETA: 11:53 - loss: 0.7852 - accuracy: 0.7096

<div class="k-default-codeblock">
```

```
</div>
134/183 [====================>.........] - ETA: 11:39 - loss: 0.7828 - accuracy: 0.7118

<div class="k-default-codeblock">
```

```
</div>
135/183 [=====================>........] - ETA: 11:24 - loss: 0.7804 - accuracy: 0.7139

<div class="k-default-codeblock">
```

```
</div>
136/183 [=====================>........] - ETA: 11:10 - loss: 0.7790 - accuracy: 0.7151

<div class="k-default-codeblock">
```

```
</div>
137/183 [=====================>........] - ETA: 10:56 - loss: 0.7804 - accuracy: 0.7144

<div class="k-default-codeblock">
```

```
</div>
138/183 [=====================>........] - ETA: 10:41 - loss: 0.7795 - accuracy: 0.7156

<div class="k-default-codeblock">
```

```
</div>
139/183 [=====================>........] - ETA: 10:27 - loss: 0.7800 - accuracy: 0.7149

<div class="k-default-codeblock">
```

```
</div>
140/183 [=====================>........] - ETA: 10:13 - loss: 0.7802 - accuracy: 0.7143

<div class="k-default-codeblock">
```

```
</div>
141/183 [======================>.......] - ETA: 9:59 - loss: 0.7813 - accuracy: 0.7137 

<div class="k-default-codeblock">
```

```
</div>
142/183 [======================>.......] - ETA: 9:44 - loss: 0.7843 - accuracy: 0.7130

<div class="k-default-codeblock">
```

```
</div>
143/183 [======================>.......] - ETA: 9:30 - loss: 0.7841 - accuracy: 0.7133

<div class="k-default-codeblock">
```

```
</div>
144/183 [======================>.......] - ETA: 9:16 - loss: 0.7848 - accuracy: 0.7135

<div class="k-default-codeblock">
```

```
</div>
145/183 [======================>.......] - ETA: 9:01 - loss: 0.7835 - accuracy: 0.7138

<div class="k-default-codeblock">
```

```
</div>
146/183 [======================>.......] - ETA: 8:47 - loss: 0.7817 - accuracy: 0.7149

<div class="k-default-codeblock">
```

```
</div>
147/183 [=======================>......] - ETA: 8:33 - loss: 0.7827 - accuracy: 0.7143

<div class="k-default-codeblock">
```

```
</div>
148/183 [=======================>......] - ETA: 8:19 - loss: 0.7813 - accuracy: 0.7145

<div class="k-default-codeblock">
```

```
</div>
149/183 [=======================>......] - ETA: 8:04 - loss: 0.7820 - accuracy: 0.7148

<div class="k-default-codeblock">
```

```
</div>
150/183 [=======================>......] - ETA: 7:50 - loss: 0.7804 - accuracy: 0.7158

<div class="k-default-codeblock">
```

```
</div>
151/183 [=======================>......] - ETA: 7:36 - loss: 0.7809 - accuracy: 0.7169

<div class="k-default-codeblock">
```

```
</div>
152/183 [=======================>......] - ETA: 7:22 - loss: 0.7802 - accuracy: 0.7171

<div class="k-default-codeblock">
```

```
</div>
153/183 [========================>.....] - ETA: 7:07 - loss: 0.7851 - accuracy: 0.7141

<div class="k-default-codeblock">
```

```
</div>
154/183 [========================>.....] - ETA: 6:53 - loss: 0.7865 - accuracy: 0.7135

<div class="k-default-codeblock">
```

```
</div>
155/183 [========================>.....] - ETA: 6:39 - loss: 0.7845 - accuracy: 0.7145

<div class="k-default-codeblock">
```

```
</div>
156/183 [========================>.....] - ETA: 6:24 - loss: 0.7830 - accuracy: 0.7155

<div class="k-default-codeblock">
```

```
</div>
157/183 [========================>.....] - ETA: 6:10 - loss: 0.7830 - accuracy: 0.7150

<div class="k-default-codeblock">
```

```
</div>
158/183 [========================>.....] - ETA: 5:56 - loss: 0.7843 - accuracy: 0.7152

<div class="k-default-codeblock">
```

```
</div>
159/183 [=========================>....] - ETA: 5:42 - loss: 0.7841 - accuracy: 0.7146

<div class="k-default-codeblock">
```

```
</div>
160/183 [=========================>....] - ETA: 5:27 - loss: 0.7842 - accuracy: 0.7148

<div class="k-default-codeblock">
```

```
</div>
161/183 [=========================>....] - ETA: 5:13 - loss: 0.7826 - accuracy: 0.7151

<div class="k-default-codeblock">
```

```
</div>
162/183 [=========================>....] - ETA: 4:59 - loss: 0.7822 - accuracy: 0.7160

<div class="k-default-codeblock">
```

```
</div>
163/183 [=========================>....] - ETA: 4:45 - loss: 0.7811 - accuracy: 0.7163

<div class="k-default-codeblock">
```

```
</div>
164/183 [=========================>....] - ETA: 4:30 - loss: 0.7808 - accuracy: 0.7157

<div class="k-default-codeblock">
```

```
</div>
165/183 [==========================>...] - ETA: 4:16 - loss: 0.7798 - accuracy: 0.7167

<div class="k-default-codeblock">
```

```
</div>
166/183 [==========================>...] - ETA: 4:02 - loss: 0.7796 - accuracy: 0.7169

<div class="k-default-codeblock">
```

```
</div>
167/183 [==========================>...] - ETA: 3:48 - loss: 0.7798 - accuracy: 0.7171

<div class="k-default-codeblock">
```

```
</div>
168/183 [==========================>...] - ETA: 3:33 - loss: 0.7794 - accuracy: 0.7173

<div class="k-default-codeblock">
```

```
</div>
169/183 [==========================>...] - ETA: 3:19 - loss: 0.7796 - accuracy: 0.7167

<div class="k-default-codeblock">
```

```
</div>
170/183 [==========================>...] - ETA: 3:05 - loss: 0.7789 - accuracy: 0.7162

<div class="k-default-codeblock">
```

```
</div>
171/183 [===========================>..] - ETA: 2:51 - loss: 0.7807 - accuracy: 0.7142

<div class="k-default-codeblock">
```

```
</div>
172/183 [===========================>..] - ETA: 2:36 - loss: 0.7788 - accuracy: 0.7151

<div class="k-default-codeblock">
```

```
</div>
173/183 [===========================>..] - ETA: 2:22 - loss: 0.7807 - accuracy: 0.7139

<div class="k-default-codeblock">
```

```
</div>
174/183 [===========================>..] - ETA: 2:08 - loss: 0.7793 - accuracy: 0.7148

<div class="k-default-codeblock">
```

```
</div>
175/183 [===========================>..] - ETA: 1:54 - loss: 0.7791 - accuracy: 0.7157

<div class="k-default-codeblock">
```

```
</div>
176/183 [===========================>..] - ETA: 1:39 - loss: 0.7799 - accuracy: 0.7138

<div class="k-default-codeblock">
```

```
</div>
177/183 [============================>.] - ETA: 1:25 - loss: 0.7768 - accuracy: 0.7154

<div class="k-default-codeblock">
```

```
</div>
178/183 [============================>.] - ETA: 1:11 - loss: 0.7748 - accuracy: 0.7156

<div class="k-default-codeblock">
```

```
</div>
179/183 [============================>.] - ETA: 57s - loss: 0.7741 - accuracy: 0.7158 

<div class="k-default-codeblock">
```

```
</div>
180/183 [============================>.] - ETA: 42s - loss: 0.7740 - accuracy: 0.7153

<div class="k-default-codeblock">
```

```
</div>
181/183 [============================>.] - ETA: 28s - loss: 0.7727 - accuracy: 0.7162

<div class="k-default-codeblock">
```

```
</div>
182/183 [============================>.] - ETA: 14s - loss: 0.7719 - accuracy: 0.7163

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - ETA: 0s - loss: 0.7714 - accuracy: 0.7158 

<div class="k-default-codeblock">
```

```
</div>
183/183 [==============================] - 2764s 15s/step - loss: 0.7714 - accuracy: 0.7158 - val_loss: 0.7098 - val_accuracy: 0.7500 - lr: 4.4984e-06


---
## Inference


```python
# Make predictions using the trained model on last validation data
predictions = model.predict(
    valid_ds,
    batch_size=CFG.batch_size,  # max batch size = valid size
    verbose=1,
)

# Format predictions and true answers
pred_answers = np.arange(4)[np.argsort(-predictions)][:, 0]
true_answers = valid_df.label.values

# Check 5 Predictions
print("# Predictions\n")
for i in range(0, 50, 10):
    row = valid_df.iloc[i]
    question = row.startphrase
    pred_answer = f"ending{pred_answers[i]}"
    true_answer = f"ending{true_answers[i]}"
    print(f"❓  Sentence {i+1}:\n{question}\n")
    print(f"✅  True Ending: {true_answer}\n   >> {row[true_answer]}\n")
    print(f"🤖  Predicted Ending: {pred_answer}\n   >> {row[pred_answer]}\n")
    print("-" * 90, "\n")
```

    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  27:32 34s/step

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  2:17 3s/step  

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  2:14 3s/step

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  2:12 3s/step

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  2:10 3s/step

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  2:06 3s/step

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  2:04 3s/step

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  2:01 3s/step

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:58 3s/step

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:55 3s/step

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:52 3s/step

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:49 3s/step

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:46 3s/step

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:43 3s/step

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:41 3s/step

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:38 3s/step

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:35 3s/step

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  1:32 3s/step

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  1:29 3s/step

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  1:26 3s/step

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  1:23 3s/step

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  1:20 3s/step

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  1:17 3s/step

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  1:15 3s/step

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  1:12 3s/step

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  1:09 3s/step

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  1:06 3s/step

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  1:03 3s/step

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  1:00 3s/step

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  57s 3s/step 

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  54s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  51s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  49s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  46s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  43s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  40s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  37s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  34s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  31s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  28s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  25s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  23s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  20s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  17s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  14s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  11s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  8s 3s/step 

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  5s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  2s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 175s 3s/step


<div class="k-default-codeblock">
```
# Predictions
```
</div>
    
<div class="k-default-codeblock">
```
❓  Sentence 1:
The man shows the teens how to move the oars. The teens
```
</div>
    
<div class="k-default-codeblock">
```
✅  True Ending: ending3
   >> follow the instructions of the man and row the oars.
```
</div>
    
<div class="k-default-codeblock">
```
🤖  Predicted Ending: ending3
   >> follow the instructions of the man and row the oars.
```
</div>
    
<div class="k-default-codeblock">
```
------------------------------------------------------------------------------------------ 
```
</div>
    
<div class="k-default-codeblock">
```
❓  Sentence 11:
A lake reflects the mountains and the sky. Someone
```
</div>
    
<div class="k-default-codeblock">
```
✅  True Ending: ending2
   >> runs along a desert highway.
```
</div>
    
<div class="k-default-codeblock">
```
🤖  Predicted Ending: ending1
   >> remains by the door.
```
</div>
    
<div class="k-default-codeblock">
```
------------------------------------------------------------------------------------------ 
```
</div>
    
<div class="k-default-codeblock">
```
❓  Sentence 21:
On screen, she smiles as someone holds up a present. He watches somberly as on screen, his mother
```
</div>
    
<div class="k-default-codeblock">
```
✅  True Ending: ending1
   >> picks him up and plays with him in the garden.
```
</div>
    
<div class="k-default-codeblock">
```
🤖  Predicted Ending: ending0
   >> comes out of her apartment, glowers at her laptop.
```
</div>
    
<div class="k-default-codeblock">
```
------------------------------------------------------------------------------------------ 
```
</div>
    
<div class="k-default-codeblock">
```
❓  Sentence 31:
A woman in a black shirt is sitting on a bench. A man
```
</div>
    
<div class="k-default-codeblock">
```
✅  True Ending: ending2
   >> sits behind a desk.
```
</div>
    
<div class="k-default-codeblock">
```
🤖  Predicted Ending: ending0
   >> is dancing on a stage.
```
</div>
    
<div class="k-default-codeblock">
```
------------------------------------------------------------------------------------------ 
```
</div>
    
<div class="k-default-codeblock">
```
❓  Sentence 41:
People are standing on sand wearing red shirts. They
```
</div>
    
<div class="k-default-codeblock">
```
✅  True Ending: ending3
   >> are playing a game of soccer in the sand.
```
</div>
    
<div class="k-default-codeblock">
```
🤖  Predicted Ending: ending3
   >> are playing a game of soccer in the sand.
```
</div>
    
<div class="k-default-codeblock">
```
------------------------------------------------------------------------------------------ 
```
</div>
    


---
## Reference
* [Multiple Choice with
HF](https://twitter.com/johnowhitaker/status/1689790373454041089?s=20)
* [Keras NLP](https://keras.io/api/keras_nlp/)
* [BirdCLEF23: Pretraining is All you Need
[Train]](https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train)
[Train]](https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train)
* [Triple Stratified KFold with
TFRecords](https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords)
