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

    
swag.zip                [ <=>                ]  84.24K   206KB/s               

    
swag.zip                [  <=>               ]   1.49M  2.39MB/s               

    
swag.zip                [   <=>              ]   2.19M  1.32MB/s               

    
swag.zip                [    <=>             ]   8.59M  2.79MB/s               

    
swag.zip                [     <=>            ]  15.41M  4.38MB/s               

    
swag.zip                [      <=>           ]  17.54M  4.47MB/s               
swag.zip                [       <=>          ]  19.94M  5.06MB/s    in 3.9s    
    
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
I0000 00:00:1704937783.212808    6043 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

```
</div>
    
   1/183 [37m━━━━━━━━━━━━━━━━━━━━  18:42:09 370s/step - accuracy: 0.5000 - loss: 1.3671

<div class="k-default-codeblock">
```

```
</div>
   2/183 [37m━━━━━━━━━━━━━━━━━━━━  46:46 16s/step - accuracy: 0.4688 - loss: 1.3707    

<div class="k-default-codeblock">
```

```
</div>
   3/183 [37m━━━━━━━━━━━━━━━━━━━━  44:54 15s/step - accuracy: 0.4236 - loss: 1.3734

<div class="k-default-codeblock">
```

```
</div>
   4/183 [37m━━━━━━━━━━━━━━━━━━━━  43:36 15s/step - accuracy: 0.3958 - loss: 1.3745

<div class="k-default-codeblock">
```

```
</div>
   5/183 [37m━━━━━━━━━━━━━━━━━━━━  42:47 14s/step - accuracy: 0.3967 - loss: 1.3732

<div class="k-default-codeblock">
```

```
</div>
   6/183 [37m━━━━━━━━━━━━━━━━━━━━  42:16 14s/step - accuracy: 0.3861 - loss: 1.3736

<div class="k-default-codeblock">
```

```
</div>
   7/183 [37m━━━━━━━━━━━━━━━━━━━━  41:50 14s/step - accuracy: 0.3794 - loss: 1.3736

<div class="k-default-codeblock">
```

```
</div>
   8/183 [37m━━━━━━━━━━━━━━━━━━━━  41:26 14s/step - accuracy: 0.3711 - loss: 1.3739

<div class="k-default-codeblock">
```

```
</div>
   9/183 [37m━━━━━━━━━━━━━━━━━━━━  41:05 14s/step - accuracy: 0.3622 - loss: 1.3743

<div class="k-default-codeblock">
```

```
</div>
  10/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:46 14s/step - accuracy: 0.3560 - loss: 1.3746

<div class="k-default-codeblock">
```

```
</div>
  11/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:29 14s/step - accuracy: 0.3505 - loss: 1.3749

<div class="k-default-codeblock">
```

```
</div>
  12/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:11 14s/step - accuracy: 0.3473 - loss: 1.3750

<div class="k-default-codeblock">
```

```
</div>
  13/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:53 14s/step - accuracy: 0.3450 - loss: 1.3752

<div class="k-default-codeblock">
```

```
</div>
  14/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:38 14s/step - accuracy: 0.3414 - loss: 1.3754

<div class="k-default-codeblock">
```

```
</div>
  15/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:23 14s/step - accuracy: 0.3387 - loss: 1.3758

<div class="k-default-codeblock">
```

```
</div>
  16/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:09 14s/step - accuracy: 0.3370 - loss: 1.3761

<div class="k-default-codeblock">
```

```
</div>
  17/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:55 14s/step - accuracy: 0.3354 - loss: 1.3764

<div class="k-default-codeblock">
```

```
</div>
  18/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:39 14s/step - accuracy: 0.3333 - loss: 1.3767

<div class="k-default-codeblock">
```

```
</div>
  19/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:25 14s/step - accuracy: 0.3310 - loss: 1.3771

<div class="k-default-codeblock">
```

```
</div>
  20/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:11 14s/step - accuracy: 0.3295 - loss: 1.3774

<div class="k-default-codeblock">
```

```
</div>
  21/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:56 14s/step - accuracy: 0.3280 - loss: 1.3777

<div class="k-default-codeblock">
```

```
</div>
  22/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:44 14s/step - accuracy: 0.3267 - loss: 1.3780

<div class="k-default-codeblock">
```

```
</div>
  23/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:31 14s/step - accuracy: 0.3253 - loss: 1.3783

<div class="k-default-codeblock">
```

```
</div>
  24/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:17 14s/step - accuracy: 0.3243 - loss: 1.3784

<div class="k-default-codeblock">
```

```
</div>
  25/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:03 14s/step - accuracy: 0.3235 - loss: 1.3786

<div class="k-default-codeblock">
```

```
</div>
  26/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:50 14s/step - accuracy: 0.3228 - loss: 1.3788

<div class="k-default-codeblock">
```

```
</div>
  27/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:37 14s/step - accuracy: 0.3219 - loss: 1.3790

<div class="k-default-codeblock">
```

```
</div>
  28/183 ━━━[37m━━━━━━━━━━━━━━━━━  36:23 14s/step - accuracy: 0.3211 - loss: 1.3791

<div class="k-default-codeblock">
```

```
</div>
  29/183 ━━━[37m━━━━━━━━━━━━━━━━━  36:09 14s/step - accuracy: 0.3203 - loss: 1.3793

<div class="k-default-codeblock">
```

```
</div>
  30/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:54 14s/step - accuracy: 0.3196 - loss: 1.3794

<div class="k-default-codeblock">
```

```
</div>
  31/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:40 14s/step - accuracy: 0.3191 - loss: 1.3795

<div class="k-default-codeblock">
```

```
</div>
  32/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:27 14s/step - accuracy: 0.3185 - loss: 1.3796

<div class="k-default-codeblock">
```

```
</div>
  33/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:13 14s/step - accuracy: 0.3182 - loss: 1.3797

<div class="k-default-codeblock">
```

```
</div>
  34/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:58 14s/step - accuracy: 0.3180 - loss: 1.3797

<div class="k-default-codeblock">
```

```
</div>
  35/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:44 14s/step - accuracy: 0.3177 - loss: 1.3798

<div class="k-default-codeblock">
```

```
</div>
  36/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:30 14s/step - accuracy: 0.3171 - loss: 1.3799

<div class="k-default-codeblock">
```

```
</div>
  37/183 ━━━━[37m━━━━━━━━━━━━━━━━  34:16 14s/step - accuracy: 0.3165 - loss: 1.3800

<div class="k-default-codeblock">
```

```
</div>
  38/183 ━━━━[37m━━━━━━━━━━━━━━━━  34:02 14s/step - accuracy: 0.3159 - loss: 1.3801

<div class="k-default-codeblock">
```

```
</div>
  39/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:48 14s/step - accuracy: 0.3153 - loss: 1.3802

<div class="k-default-codeblock">
```

```
</div>
  40/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:33 14s/step - accuracy: 0.3147 - loss: 1.3803

<div class="k-default-codeblock">
```

```
</div>
  41/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:19 14s/step - accuracy: 0.3140 - loss: 1.3804

<div class="k-default-codeblock">
```

```
</div>
  42/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:05 14s/step - accuracy: 0.3132 - loss: 1.3805

<div class="k-default-codeblock">
```

```
</div>
  43/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:50 14s/step - accuracy: 0.3125 - loss: 1.3806

<div class="k-default-codeblock">
```

```
</div>
  44/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:35 14s/step - accuracy: 0.3117 - loss: 1.3807

<div class="k-default-codeblock">
```

```
</div>
  45/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:20 14s/step - accuracy: 0.3110 - loss: 1.3807

<div class="k-default-codeblock">
```

```
</div>
  46/183 ━━━━━[37m━━━━━━━━━━━━━━━  32:05 14s/step - accuracy: 0.3102 - loss: 1.3808

<div class="k-default-codeblock">
```

```
</div>
  47/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:51 14s/step - accuracy: 0.3095 - loss: 1.3809

<div class="k-default-codeblock">
```

```
</div>
  48/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:37 14s/step - accuracy: 0.3087 - loss: 1.3809

<div class="k-default-codeblock">
```

```
</div>
  49/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:23 14s/step - accuracy: 0.3079 - loss: 1.3810

<div class="k-default-codeblock">
```

```
</div>
  50/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:08 14s/step - accuracy: 0.3071 - loss: 1.3811

<div class="k-default-codeblock">
```

```
</div>
  51/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:53 14s/step - accuracy: 0.3065 - loss: 1.3811

<div class="k-default-codeblock">
```

```
</div>
  52/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:38 14s/step - accuracy: 0.3058 - loss: 1.3812

<div class="k-default-codeblock">
```

```
</div>
  53/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:24 14s/step - accuracy: 0.3052 - loss: 1.3812

<div class="k-default-codeblock">
```

```
</div>
  54/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:09 14s/step - accuracy: 0.3045 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
  55/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:55 14s/step - accuracy: 0.3039 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
  56/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:40 14s/step - accuracy: 0.3033 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
  57/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:26 14s/step - accuracy: 0.3026 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
  58/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:12 14s/step - accuracy: 0.3019 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
  59/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:58 14s/step - accuracy: 0.3014 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
  60/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:44 14s/step - accuracy: 0.3008 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
  61/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:30 14s/step - accuracy: 0.3002 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
  62/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:16 14s/step - accuracy: 0.2997 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
  63/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:02 14s/step - accuracy: 0.2992 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
  64/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:47 14s/step - accuracy: 0.2987 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
  65/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:33 14s/step - accuracy: 0.2982 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
  66/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:19 14s/step - accuracy: 0.2977 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
  67/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:05 14s/step - accuracy: 0.2973 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
  68/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:50 14s/step - accuracy: 0.2968 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
  69/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:36 14s/step - accuracy: 0.2964 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
  70/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:22 14s/step - accuracy: 0.2961 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
  71/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:07 14s/step - accuracy: 0.2957 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
  72/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:53 14s/step - accuracy: 0.2953 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
  73/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:39 14s/step - accuracy: 0.2949 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
  74/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:25 14s/step - accuracy: 0.2946 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
  75/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:10 14s/step - accuracy: 0.2944 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  76/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:56 14s/step - accuracy: 0.2940 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  77/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:42 14s/step - accuracy: 0.2937 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  78/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:28 14s/step - accuracy: 0.2934 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  79/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:14 14s/step - accuracy: 0.2931 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  80/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:59 14s/step - accuracy: 0.2928 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  81/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:45 14s/step - accuracy: 0.2925 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  82/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:33 14s/step - accuracy: 0.2922 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  83/183 ━━━━━━━━━[37m━━━━━━━━━━━  23:19 14s/step - accuracy: 0.2919 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  84/183 ━━━━━━━━━[37m━━━━━━━━━━━  23:05 14s/step - accuracy: 0.2917 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  85/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:52 14s/step - accuracy: 0.2914 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  86/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:38 14s/step - accuracy: 0.2912 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  87/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:24 14s/step - accuracy: 0.2909 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  88/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:11 14s/step - accuracy: 0.2906 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  89/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:57 14s/step - accuracy: 0.2904 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  90/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:43 14s/step - accuracy: 0.2901 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  91/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:29 14s/step - accuracy: 0.2898 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  92/183 ━━━━━━━━━━[37m━━━━━━━━━━  21:14 14s/step - accuracy: 0.2896 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  93/183 ━━━━━━━━━━[37m━━━━━━━━━━  21:01 14s/step - accuracy: 0.2894 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  94/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:46 14s/step - accuracy: 0.2892 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  95/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:32 14s/step - accuracy: 0.2890 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  96/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:19 14s/step - accuracy: 0.2888 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  97/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:05 14s/step - accuracy: 0.2886 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  98/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:51 14s/step - accuracy: 0.2885 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
  99/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:37 14s/step - accuracy: 0.2883 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
 100/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:23 14s/step - accuracy: 0.2881 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
 101/183 ━━━━━━━━━━━[37m━━━━━━━━━  19:09 14s/step - accuracy: 0.2880 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
 102/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:55 14s/step - accuracy: 0.2879 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
 103/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:41 14s/step - accuracy: 0.2877 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
 104/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:27 14s/step - accuracy: 0.2876 - loss: 1.3820

<div class="k-default-codeblock">
```

```
</div>
 105/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:13 14s/step - accuracy: 0.2875 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
 106/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:59 14s/step - accuracy: 0.2875 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
 107/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:45 14s/step - accuracy: 0.2874 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
 108/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:31 14s/step - accuracy: 0.2873 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
 109/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:17 14s/step - accuracy: 0.2872 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
 110/183 ━━━━━━━━━━━━[37m━━━━━━━━  17:03 14s/step - accuracy: 0.2871 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
 111/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:49 14s/step - accuracy: 0.2871 - loss: 1.3819

<div class="k-default-codeblock">
```

```
</div>
 112/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:35 14s/step - accuracy: 0.2870 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 113/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:21 14s/step - accuracy: 0.2869 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 114/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:07 14s/step - accuracy: 0.2869 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 115/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:53 14s/step - accuracy: 0.2868 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 116/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:39 14s/step - accuracy: 0.2867 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 117/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:25 14s/step - accuracy: 0.2867 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 118/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:11 14s/step - accuracy: 0.2866 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 119/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:57 14s/step - accuracy: 0.2865 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 120/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:43 14s/step - accuracy: 0.2864 - loss: 1.3818

<div class="k-default-codeblock">
```

```
</div>
 121/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:29 14s/step - accuracy: 0.2863 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 122/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:15 14s/step - accuracy: 0.2862 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 123/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:01 14s/step - accuracy: 0.2862 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 124/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:47 14s/step - accuracy: 0.2861 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 125/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:33 14s/step - accuracy: 0.2860 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 126/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:19 14s/step - accuracy: 0.2859 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 127/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:05 14s/step - accuracy: 0.2859 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 128/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:51 14s/step - accuracy: 0.2858 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 129/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:37 14s/step - accuracy: 0.2858 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 130/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:23 14s/step - accuracy: 0.2857 - loss: 1.3817

<div class="k-default-codeblock">
```

```
</div>
 131/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:09 14s/step - accuracy: 0.2857 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 132/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:55 14s/step - accuracy: 0.2856 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 133/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:41 14s/step - accuracy: 0.2856 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 134/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:27 14s/step - accuracy: 0.2855 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 135/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:13 14s/step - accuracy: 0.2855 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 136/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:59 14s/step - accuracy: 0.2855 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 137/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:45 14s/step - accuracy: 0.2855 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 138/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:31 14s/step - accuracy: 0.2854 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 139/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:17 14s/step - accuracy: 0.2854 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 140/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:03 14s/step - accuracy: 0.2854 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 141/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:49 14s/step - accuracy: 0.2853 - loss: 1.3816 

<div class="k-default-codeblock">
```

```
</div>
 142/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:35 14s/step - accuracy: 0.2853 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 143/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:21 14s/step - accuracy: 0.2853 - loss: 1.3816

<div class="k-default-codeblock">
```

```
</div>
 144/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:07 14s/step - accuracy: 0.2852 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 145/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:53 14s/step - accuracy: 0.2852 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 146/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:39 14s/step - accuracy: 0.2852 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 147/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:25 14s/step - accuracy: 0.2851 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 148/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:11 14s/step - accuracy: 0.2851 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 149/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:57 14s/step - accuracy: 0.2851 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 150/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:43 14s/step - accuracy: 0.2850 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 151/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:29 14s/step - accuracy: 0.2850 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 152/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:15 14s/step - accuracy: 0.2850 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 153/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:01 14s/step - accuracy: 0.2850 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 154/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:47 14s/step - accuracy: 0.2850 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 155/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:33 14s/step - accuracy: 0.2849 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 156/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:19 14s/step - accuracy: 0.2849 - loss: 1.3815

<div class="k-default-codeblock">
```

```
</div>
 157/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:05 14s/step - accuracy: 0.2849 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 158/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:51 14s/step - accuracy: 0.2849 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 159/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:37 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 160/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:23 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 161/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:09 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 162/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:55 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 163/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:40 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 164/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:26 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 165/183 ━━━━━━━━━━━━━━━━━━[37m━━  4:12 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 166/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:58 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 167/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:44 14s/step - accuracy: 0.2848 - loss: 1.3814

<div class="k-default-codeblock">
```

```
</div>
 168/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:30 14s/step - accuracy: 0.2848 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 169/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:16 14s/step - accuracy: 0.2848 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 170/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:02 14s/step - accuracy: 0.2848 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 171/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:48 14s/step - accuracy: 0.2848 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 172/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:34 14s/step - accuracy: 0.2848 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 173/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:20 14s/step - accuracy: 0.2848 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 174/183 ━━━━━━━━━━━━━━━━━━━[37m━  2:06 14s/step - accuracy: 0.2849 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 175/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:52 14s/step - accuracy: 0.2849 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 176/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:38 14s/step - accuracy: 0.2849 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 177/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:24 14s/step - accuracy: 0.2850 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 178/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:10 14s/step - accuracy: 0.2850 - loss: 1.3813

<div class="k-default-codeblock">
```

```
</div>
 179/183 ━━━━━━━━━━━━━━━━━━━[37m━  56s 14s/step - accuracy: 0.2850 - loss: 1.3813 

<div class="k-default-codeblock">
```

```
</div>
 180/183 ━━━━━━━━━━━━━━━━━━━[37m━  42s 14s/step - accuracy: 0.2851 - loss: 1.3812

<div class="k-default-codeblock">
```

```
</div>
 181/183 ━━━━━━━━━━━━━━━━━━━[37m━  28s 14s/step - accuracy: 0.2851 - loss: 1.3812

<div class="k-default-codeblock">
```

```
</div>
 182/183 ━━━━━━━━━━━━━━━━━━━[37m━  14s 14s/step - accuracy: 0.2851 - loss: 1.3812

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 0s 14s/step - accuracy: 0.2852 - loss: 1.3812 

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 3110s 15s/step - accuracy: 0.2852 - loss: 1.3812 - val_accuracy: 0.5575 - val_loss: 1.3673 - learning_rate: 1.0000e-06


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
   1/183 [37m━━━━━━━━━━━━━━━━━━━━  45:28 15s/step - accuracy: 0.3750 - loss: 1.3267

<div class="k-default-codeblock">
```

```
</div>
   2/183 [37m━━━━━━━━━━━━━━━━━━━━  43:00 14s/step - accuracy: 0.3438 - loss: 1.3379

<div class="k-default-codeblock">
```

```
</div>
   3/183 [37m━━━━━━━━━━━━━━━━━━━━  42:53 14s/step - accuracy: 0.3403 - loss: 1.3452

<div class="k-default-codeblock">
```

```
</div>
   4/183 [37m━━━━━━━━━━━━━━━━━━━━  42:37 14s/step - accuracy: 0.3255 - loss: 1.3510

<div class="k-default-codeblock">
```

```
</div>
   5/183 [37m━━━━━━━━━━━━━━━━━━━━  42:19 14s/step - accuracy: 0.3204 - loss: 1.3548

<div class="k-default-codeblock">
```

```
</div>
   6/183 [37m━━━━━━━━━━━━━━━━━━━━  42:05 14s/step - accuracy: 0.3156 - loss: 1.3585

<div class="k-default-codeblock">
```

```
</div>
   7/183 [37m━━━━━━━━━━━━━━━━━━━━  41:45 14s/step - accuracy: 0.3165 - loss: 1.3608

<div class="k-default-codeblock">
```

```
</div>
   8/183 [37m━━━━━━━━━━━━━━━━━━━━  41:31 14s/step - accuracy: 0.3160 - loss: 1.3626

<div class="k-default-codeblock">
```

```
</div>
   9/183 [37m━━━━━━━━━━━━━━━━━━━━  41:12 14s/step - accuracy: 0.3194 - loss: 1.3634

<div class="k-default-codeblock">
```

```
</div>
  10/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:56 14s/step - accuracy: 0.3225 - loss: 1.3643

<div class="k-default-codeblock">
```

```
</div>
  11/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:42 14s/step - accuracy: 0.3262 - loss: 1.3648

<div class="k-default-codeblock">
```

```
</div>
  12/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:28 14s/step - accuracy: 0.3294 - loss: 1.3651

<div class="k-default-codeblock">
```

```
</div>
  13/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:13 14s/step - accuracy: 0.3322 - loss: 1.3655

<div class="k-default-codeblock">
```

```
</div>
  14/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:59 14s/step - accuracy: 0.3340 - loss: 1.3656

<div class="k-default-codeblock">
```

```
</div>
  15/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:46 14s/step - accuracy: 0.3345 - loss: 1.3659

<div class="k-default-codeblock">
```

```
</div>
  16/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:33 14s/step - accuracy: 0.3351 - loss: 1.3662

<div class="k-default-codeblock">
```

```
</div>
  17/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:18 14s/step - accuracy: 0.3357 - loss: 1.3664

<div class="k-default-codeblock">
```

```
</div>
  18/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:03 14s/step - accuracy: 0.3356 - loss: 1.3667

<div class="k-default-codeblock">
```

```
</div>
  19/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:49 14s/step - accuracy: 0.3352 - loss: 1.3670

<div class="k-default-codeblock">
```

```
</div>
  20/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:34 14s/step - accuracy: 0.3344 - loss: 1.3673

<div class="k-default-codeblock">
```

```
</div>
  21/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:19 14s/step - accuracy: 0.3343 - loss: 1.3674

<div class="k-default-codeblock">
```

```
</div>
  22/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:04 14s/step - accuracy: 0.3346 - loss: 1.3675

<div class="k-default-codeblock">
```

```
</div>
  23/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:48 14s/step - accuracy: 0.3352 - loss: 1.3675

<div class="k-default-codeblock">
```

```
</div>
  24/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:33 14s/step - accuracy: 0.3356 - loss: 1.3676

<div class="k-default-codeblock">
```

```
</div>
  25/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:19 14s/step - accuracy: 0.3357 - loss: 1.3676

<div class="k-default-codeblock">
```

```
</div>
  26/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:03 14s/step - accuracy: 0.3358 - loss: 1.3677

<div class="k-default-codeblock">
```

```
</div>
  27/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:47 14s/step - accuracy: 0.3360 - loss: 1.3676

<div class="k-default-codeblock">
```

```
</div>
  28/183 ━━━[37m━━━━━━━━━━━━━━━━━  36:31 14s/step - accuracy: 0.3361 - loss: 1.3676

<div class="k-default-codeblock">
```

```
</div>
  29/183 ━━━[37m━━━━━━━━━━━━━━━━━  36:15 14s/step - accuracy: 0.3363 - loss: 1.3675

<div class="k-default-codeblock">
```

```
</div>
  30/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:59 14s/step - accuracy: 0.3365 - loss: 1.3674

<div class="k-default-codeblock">
```

```
</div>
  31/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:44 14s/step - accuracy: 0.3365 - loss: 1.3674

<div class="k-default-codeblock">
```

```
</div>
  32/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:28 14s/step - accuracy: 0.3368 - loss: 1.3673

<div class="k-default-codeblock">
```

```
</div>
  33/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:13 14s/step - accuracy: 0.3368 - loss: 1.3672

<div class="k-default-codeblock">
```

```
</div>
  34/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:58 14s/step - accuracy: 0.3371 - loss: 1.3670

<div class="k-default-codeblock">
```

```
</div>
  35/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:42 14s/step - accuracy: 0.3373 - loss: 1.3669

<div class="k-default-codeblock">
```

```
</div>
  36/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:27 14s/step - accuracy: 0.3378 - loss: 1.3667

<div class="k-default-codeblock">
```

```
</div>
  37/183 ━━━━[37m━━━━━━━━━━━━━━━━  34:12 14s/step - accuracy: 0.3383 - loss: 1.3665

<div class="k-default-codeblock">
```

```
</div>
  38/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:58 14s/step - accuracy: 0.3390 - loss: 1.3662

<div class="k-default-codeblock">
```

```
</div>
  39/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:44 14s/step - accuracy: 0.3398 - loss: 1.3660

<div class="k-default-codeblock">
```

```
</div>
  40/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:30 14s/step - accuracy: 0.3405 - loss: 1.3657

<div class="k-default-codeblock">
```

```
</div>
  41/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:16 14s/step - accuracy: 0.3411 - loss: 1.3654

<div class="k-default-codeblock">
```

```
</div>
  42/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:02 14s/step - accuracy: 0.3419 - loss: 1.3651

<div class="k-default-codeblock">
```

```
</div>
  43/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:49 14s/step - accuracy: 0.3428 - loss: 1.3647

<div class="k-default-codeblock">
```

```
</div>
  44/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:35 14s/step - accuracy: 0.3436 - loss: 1.3644

<div class="k-default-codeblock">
```

```
</div>
  45/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:21 14s/step - accuracy: 0.3446 - loss: 1.3640

<div class="k-default-codeblock">
```

```
</div>
  46/183 ━━━━━[37m━━━━━━━━━━━━━━━  32:08 14s/step - accuracy: 0.3454 - loss: 1.3637

<div class="k-default-codeblock">
```

```
</div>
  47/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:54 14s/step - accuracy: 0.3463 - loss: 1.3633

<div class="k-default-codeblock">
```

```
</div>
  48/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:40 14s/step - accuracy: 0.3471 - loss: 1.3629

<div class="k-default-codeblock">
```

```
</div>
  49/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:26 14s/step - accuracy: 0.3479 - loss: 1.3625

<div class="k-default-codeblock">
```

```
</div>
  50/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:12 14s/step - accuracy: 0.3487 - loss: 1.3622

<div class="k-default-codeblock">
```

```
</div>
  51/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:59 14s/step - accuracy: 0.3495 - loss: 1.3618

<div class="k-default-codeblock">
```

```
</div>
  52/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:45 14s/step - accuracy: 0.3503 - loss: 1.3614

<div class="k-default-codeblock">
```

```
</div>
  53/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:31 14s/step - accuracy: 0.3511 - loss: 1.3610

<div class="k-default-codeblock">
```

```
</div>
  54/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:17 14s/step - accuracy: 0.3518 - loss: 1.3606

<div class="k-default-codeblock">
```

```
</div>
  55/183 ━━━━━━[37m━━━━━━━━━━━━━━  30:03 14s/step - accuracy: 0.3524 - loss: 1.3603

<div class="k-default-codeblock">
```

```
</div>
  56/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:49 14s/step - accuracy: 0.3530 - loss: 1.3600

<div class="k-default-codeblock">
```

```
</div>
  57/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:35 14s/step - accuracy: 0.3536 - loss: 1.3597

<div class="k-default-codeblock">
```

```
</div>
  58/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:21 14s/step - accuracy: 0.3541 - loss: 1.3593

<div class="k-default-codeblock">
```

```
</div>
  59/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:07 14s/step - accuracy: 0.3547 - loss: 1.3590

<div class="k-default-codeblock">
```

```
</div>
  60/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:52 14s/step - accuracy: 0.3551 - loss: 1.3588

<div class="k-default-codeblock">
```

```
</div>
  61/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:38 14s/step - accuracy: 0.3555 - loss: 1.3585

<div class="k-default-codeblock">
```

```
</div>
  62/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:24 14s/step - accuracy: 0.3559 - loss: 1.3582

<div class="k-default-codeblock">
```

```
</div>
  63/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:10 14s/step - accuracy: 0.3563 - loss: 1.3579

<div class="k-default-codeblock">
```

```
</div>
  64/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:55 14s/step - accuracy: 0.3567 - loss: 1.3576

<div class="k-default-codeblock">
```

```
</div>
  65/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:41 14s/step - accuracy: 0.3571 - loss: 1.3573

<div class="k-default-codeblock">
```

```
</div>
  66/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:27 14s/step - accuracy: 0.3575 - loss: 1.3570

<div class="k-default-codeblock">
```

```
</div>
  67/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:13 14s/step - accuracy: 0.3579 - loss: 1.3567

<div class="k-default-codeblock">
```

```
</div>
  68/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:59 14s/step - accuracy: 0.3583 - loss: 1.3564

<div class="k-default-codeblock">
```

```
</div>
  69/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:44 14s/step - accuracy: 0.3587 - loss: 1.3560

<div class="k-default-codeblock">
```

```
</div>
  70/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:30 14s/step - accuracy: 0.3591 - loss: 1.3557

<div class="k-default-codeblock">
```

```
</div>
  71/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:16 14s/step - accuracy: 0.3595 - loss: 1.3554

<div class="k-default-codeblock">
```

```
</div>
  72/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:02 14s/step - accuracy: 0.3598 - loss: 1.3550

<div class="k-default-codeblock">
```

```
</div>
  73/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:48 14s/step - accuracy: 0.3603 - loss: 1.3547

<div class="k-default-codeblock">
```

```
</div>
  74/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:34 14s/step - accuracy: 0.3607 - loss: 1.3543

<div class="k-default-codeblock">
```

```
</div>
  75/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:20 14s/step - accuracy: 0.3611 - loss: 1.3540

<div class="k-default-codeblock">
```

```
</div>
  76/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:06 14s/step - accuracy: 0.3615 - loss: 1.3536

<div class="k-default-codeblock">
```

```
</div>
  77/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:52 14s/step - accuracy: 0.3619 - loss: 1.3533

<div class="k-default-codeblock">
```

```
</div>
  78/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:38 14s/step - accuracy: 0.3623 - loss: 1.3529

<div class="k-default-codeblock">
```

```
</div>
  79/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:24 14s/step - accuracy: 0.3627 - loss: 1.3525

<div class="k-default-codeblock">
```

```
</div>
  80/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:09 14s/step - accuracy: 0.3631 - loss: 1.3521

<div class="k-default-codeblock">
```

```
</div>
  81/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:55 14s/step - accuracy: 0.3634 - loss: 1.3517

<div class="k-default-codeblock">
```

```
</div>
  82/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:41 14s/step - accuracy: 0.3639 - loss: 1.3513

<div class="k-default-codeblock">
```

```
</div>
  83/183 ━━━━━━━━━[37m━━━━━━━━━━━  23:26 14s/step - accuracy: 0.3643 - loss: 1.3509

<div class="k-default-codeblock">
```

```
</div>
  84/183 ━━━━━━━━━[37m━━━━━━━━━━━  23:12 14s/step - accuracy: 0.3648 - loss: 1.3504

<div class="k-default-codeblock">
```

```
</div>
  85/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:58 14s/step - accuracy: 0.3652 - loss: 1.3500

<div class="k-default-codeblock">
```

```
</div>
  86/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:44 14s/step - accuracy: 0.3657 - loss: 1.3495

<div class="k-default-codeblock">
```

```
</div>
  87/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:29 14s/step - accuracy: 0.3662 - loss: 1.3491

<div class="k-default-codeblock">
```

```
</div>
  88/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:15 14s/step - accuracy: 0.3667 - loss: 1.3486

<div class="k-default-codeblock">
```

```
</div>
  89/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:01 14s/step - accuracy: 0.3672 - loss: 1.3481

<div class="k-default-codeblock">
```

```
</div>
  90/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:46 14s/step - accuracy: 0.3677 - loss: 1.3477

<div class="k-default-codeblock">
```

```
</div>
  91/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:32 14s/step - accuracy: 0.3682 - loss: 1.3472

<div class="k-default-codeblock">
```

```
</div>
  92/183 ━━━━━━━━━━[37m━━━━━━━━━━  21:19 14s/step - accuracy: 0.3687 - loss: 1.3467

<div class="k-default-codeblock">
```

```
</div>
  93/183 ━━━━━━━━━━[37m━━━━━━━━━━  21:05 14s/step - accuracy: 0.3692 - loss: 1.3463

<div class="k-default-codeblock">
```

```
</div>
  94/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:51 14s/step - accuracy: 0.3696 - loss: 1.3458

<div class="k-default-codeblock">
```

```
</div>
  95/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:37 14s/step - accuracy: 0.3700 - loss: 1.3454

<div class="k-default-codeblock">
```

```
</div>
  96/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:23 14s/step - accuracy: 0.3705 - loss: 1.3449

<div class="k-default-codeblock">
```

```
</div>
  97/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:09 14s/step - accuracy: 0.3709 - loss: 1.3444

<div class="k-default-codeblock">
```

```
</div>
  98/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:55 14s/step - accuracy: 0.3714 - loss: 1.3440

<div class="k-default-codeblock">
```

```
</div>
  99/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:41 14s/step - accuracy: 0.3718 - loss: 1.3435

<div class="k-default-codeblock">
```

```
</div>
 100/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:27 14s/step - accuracy: 0.3723 - loss: 1.3430

<div class="k-default-codeblock">
```

```
</div>
 101/183 ━━━━━━━━━━━[37m━━━━━━━━━  19:13 14s/step - accuracy: 0.3728 - loss: 1.3425

<div class="k-default-codeblock">
```

```
</div>
 102/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:59 14s/step - accuracy: 0.3732 - loss: 1.3420

<div class="k-default-codeblock">
```

```
</div>
 103/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:45 14s/step - accuracy: 0.3738 - loss: 1.3415

<div class="k-default-codeblock">
```

```
</div>
 104/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:31 14s/step - accuracy: 0.3743 - loss: 1.3410

<div class="k-default-codeblock">
```

```
</div>
 105/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:17 14s/step - accuracy: 0.3748 - loss: 1.3405

<div class="k-default-codeblock">
```

```
</div>
 106/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:03 14s/step - accuracy: 0.3753 - loss: 1.3400

<div class="k-default-codeblock">
```

```
</div>
 107/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:49 14s/step - accuracy: 0.3758 - loss: 1.3394

<div class="k-default-codeblock">
```

```
</div>
 108/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:35 14s/step - accuracy: 0.3763 - loss: 1.3389

<div class="k-default-codeblock">
```

```
</div>
 109/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:21 14s/step - accuracy: 0.3769 - loss: 1.3384

<div class="k-default-codeblock">
```

```
</div>
 110/183 ━━━━━━━━━━━━[37m━━━━━━━━  17:07 14s/step - accuracy: 0.3774 - loss: 1.3378

<div class="k-default-codeblock">
```

```
</div>
 111/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:53 14s/step - accuracy: 0.3779 - loss: 1.3373

<div class="k-default-codeblock">
```

```
</div>
 112/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:39 14s/step - accuracy: 0.3785 - loss: 1.3368

<div class="k-default-codeblock">
```

```
</div>
 113/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:25 14s/step - accuracy: 0.3790 - loss: 1.3362

<div class="k-default-codeblock">
```

```
</div>
 114/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:11 14s/step - accuracy: 0.3796 - loss: 1.3357

<div class="k-default-codeblock">
```

```
</div>
 115/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:56 14s/step - accuracy: 0.3801 - loss: 1.3351

<div class="k-default-codeblock">
```

```
</div>
 116/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:42 14s/step - accuracy: 0.3807 - loss: 1.3346

<div class="k-default-codeblock">
```

```
</div>
 117/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:28 14s/step - accuracy: 0.3813 - loss: 1.3340

<div class="k-default-codeblock">
```

```
</div>
 118/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:14 14s/step - accuracy: 0.3818 - loss: 1.3335

<div class="k-default-codeblock">
```

```
</div>
 119/183 ━━━━━━━━━━━━━[37m━━━━━━━  15:00 14s/step - accuracy: 0.3824 - loss: 1.3329

<div class="k-default-codeblock">
```

```
</div>
 120/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:46 14s/step - accuracy: 0.3830 - loss: 1.3323

<div class="k-default-codeblock">
```

```
</div>
 121/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:32 14s/step - accuracy: 0.3835 - loss: 1.3318

<div class="k-default-codeblock">
```

```
</div>
 122/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:18 14s/step - accuracy: 0.3841 - loss: 1.3312

<div class="k-default-codeblock">
```

```
</div>
 123/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:04 14s/step - accuracy: 0.3847 - loss: 1.3306

<div class="k-default-codeblock">
```

```
</div>
 124/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:50 14s/step - accuracy: 0.3853 - loss: 1.3301

<div class="k-default-codeblock">
```

```
</div>
 125/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:36 14s/step - accuracy: 0.3859 - loss: 1.3295

<div class="k-default-codeblock">
```

```
</div>
 126/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:22 14s/step - accuracy: 0.3865 - loss: 1.3289

<div class="k-default-codeblock">
```

```
</div>
 127/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:08 14s/step - accuracy: 0.3871 - loss: 1.3283

<div class="k-default-codeblock">
```

```
</div>
 128/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:54 14s/step - accuracy: 0.3877 - loss: 1.3277

<div class="k-default-codeblock">
```

```
</div>
 129/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:40 14s/step - accuracy: 0.3883 - loss: 1.3271

<div class="k-default-codeblock">
```

```
</div>
 130/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:26 14s/step - accuracy: 0.3889 - loss: 1.3265

<div class="k-default-codeblock">
```

```
</div>
 131/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:12 14s/step - accuracy: 0.3895 - loss: 1.3259

<div class="k-default-codeblock">
```

```
</div>
 132/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:58 14s/step - accuracy: 0.3901 - loss: 1.3253

<div class="k-default-codeblock">
```

```
</div>
 133/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:44 14s/step - accuracy: 0.3907 - loss: 1.3247

<div class="k-default-codeblock">
```

```
</div>
 134/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:30 14s/step - accuracy: 0.3913 - loss: 1.3241

<div class="k-default-codeblock">
```

```
</div>
 135/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:16 14s/step - accuracy: 0.3918 - loss: 1.3235

<div class="k-default-codeblock">
```

```
</div>
 136/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:02 14s/step - accuracy: 0.3924 - loss: 1.3229

<div class="k-default-codeblock">
```

```
</div>
 137/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:48 14s/step - accuracy: 0.3930 - loss: 1.3223

<div class="k-default-codeblock">
```

```
</div>
 138/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:33 14s/step - accuracy: 0.3935 - loss: 1.3217

<div class="k-default-codeblock">
```

```
</div>
 139/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:19 14s/step - accuracy: 0.3941 - loss: 1.3211

<div class="k-default-codeblock">
```

```
</div>
 140/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:05 14s/step - accuracy: 0.3946 - loss: 1.3205

<div class="k-default-codeblock">
```

```
</div>
 141/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:51 14s/step - accuracy: 0.3951 - loss: 1.3200 

<div class="k-default-codeblock">
```

```
</div>
 142/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:37 14s/step - accuracy: 0.3956 - loss: 1.3194

<div class="k-default-codeblock">
```

```
</div>
 143/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:23 14s/step - accuracy: 0.3962 - loss: 1.3188

<div class="k-default-codeblock">
```

```
</div>
 144/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:09 14s/step - accuracy: 0.3967 - loss: 1.3183

<div class="k-default-codeblock">
```

```
</div>
 145/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:55 14s/step - accuracy: 0.3972 - loss: 1.3177

<div class="k-default-codeblock">
```

```
</div>
 146/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:41 14s/step - accuracy: 0.3977 - loss: 1.3172

<div class="k-default-codeblock">
```

```
</div>
 147/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:27 14s/step - accuracy: 0.3982 - loss: 1.3166

<div class="k-default-codeblock">
```

```
</div>
 148/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:13 14s/step - accuracy: 0.3987 - loss: 1.3160

<div class="k-default-codeblock">
```

```
</div>
 149/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:58 14s/step - accuracy: 0.3992 - loss: 1.3154

<div class="k-default-codeblock">
```

```
</div>
 150/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:44 14s/step - accuracy: 0.3998 - loss: 1.3149

<div class="k-default-codeblock">
```

```
</div>
 151/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:30 14s/step - accuracy: 0.4003 - loss: 1.3143

<div class="k-default-codeblock">
```

```
</div>
 152/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:16 14s/step - accuracy: 0.4008 - loss: 1.3137

<div class="k-default-codeblock">
```

```
</div>
 153/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:02 14s/step - accuracy: 0.4012 - loss: 1.3132

<div class="k-default-codeblock">
```

```
</div>
 154/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:48 14s/step - accuracy: 0.4017 - loss: 1.3126

<div class="k-default-codeblock">
```

```
</div>
 155/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:34 14s/step - accuracy: 0.4022 - loss: 1.3121

<div class="k-default-codeblock">
```

```
</div>
 156/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:20 14s/step - accuracy: 0.4027 - loss: 1.3115

<div class="k-default-codeblock">
```

```
</div>
 157/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:06 14s/step - accuracy: 0.4032 - loss: 1.3110

<div class="k-default-codeblock">
```

```
</div>
 158/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:52 14s/step - accuracy: 0.4036 - loss: 1.3104

<div class="k-default-codeblock">
```

```
</div>
 159/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:37 14s/step - accuracy: 0.4041 - loss: 1.3099

<div class="k-default-codeblock">
```

```
</div>
 160/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:23 14s/step - accuracy: 0.4046 - loss: 1.3093

<div class="k-default-codeblock">
```

```
</div>
 161/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:09 14s/step - accuracy: 0.4050 - loss: 1.3088

<div class="k-default-codeblock">
```

```
</div>
 162/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:55 14s/step - accuracy: 0.4055 - loss: 1.3083

<div class="k-default-codeblock">
```

```
</div>
 163/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:41 14s/step - accuracy: 0.4059 - loss: 1.3077

<div class="k-default-codeblock">
```

```
</div>
 164/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:27 14s/step - accuracy: 0.4064 - loss: 1.3072

<div class="k-default-codeblock">
```

```
</div>
 165/183 ━━━━━━━━━━━━━━━━━━[37m━━  4:13 14s/step - accuracy: 0.4068 - loss: 1.3066

<div class="k-default-codeblock">
```

```
</div>
 166/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:59 14s/step - accuracy: 0.4073 - loss: 1.3061

<div class="k-default-codeblock">
```

```
</div>
 167/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:45 14s/step - accuracy: 0.4077 - loss: 1.3055

<div class="k-default-codeblock">
```

```
</div>
 168/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:31 14s/step - accuracy: 0.4082 - loss: 1.3050

<div class="k-default-codeblock">
```

```
</div>
 169/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:16 14s/step - accuracy: 0.4086 - loss: 1.3045

<div class="k-default-codeblock">
```

```
</div>
 170/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:02 14s/step - accuracy: 0.4091 - loss: 1.3039

<div class="k-default-codeblock">
```

```
</div>
 171/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:48 14s/step - accuracy: 0.4095 - loss: 1.3034

<div class="k-default-codeblock">
```

```
</div>
 172/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:34 14s/step - accuracy: 0.4100 - loss: 1.3028

<div class="k-default-codeblock">
```

```
</div>
 173/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:20 14s/step - accuracy: 0.4104 - loss: 1.3023

<div class="k-default-codeblock">
```

```
</div>
 174/183 ━━━━━━━━━━━━━━━━━━━[37m━  2:06 14s/step - accuracy: 0.4109 - loss: 1.3018

<div class="k-default-codeblock">
```

```
</div>
 175/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:52 14s/step - accuracy: 0.4113 - loss: 1.3012

<div class="k-default-codeblock">
```

```
</div>
 176/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:38 14s/step - accuracy: 0.4118 - loss: 1.3007

<div class="k-default-codeblock">
```

```
</div>
 177/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:24 14s/step - accuracy: 0.4122 - loss: 1.3002

<div class="k-default-codeblock">
```

```
</div>
 178/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:10 14s/step - accuracy: 0.4126 - loss: 1.2996

<div class="k-default-codeblock">
```

```
</div>
 179/183 ━━━━━━━━━━━━━━━━━━━[37m━  56s 14s/step - accuracy: 0.4131 - loss: 1.2991 

<div class="k-default-codeblock">
```

```
</div>
 180/183 ━━━━━━━━━━━━━━━━━━━[37m━  42s 14s/step - accuracy: 0.4135 - loss: 1.2986

<div class="k-default-codeblock">
```

```
</div>
 181/183 ━━━━━━━━━━━━━━━━━━━[37m━  28s 14s/step - accuracy: 0.4139 - loss: 1.2981

<div class="k-default-codeblock">
```

```
</div>
 182/183 ━━━━━━━━━━━━━━━━━━━[37m━  14s 14s/step - accuracy: 0.4143 - loss: 1.2975

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 0s 14s/step - accuracy: 0.4147 - loss: 1.2970 

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 2725s 15s/step - accuracy: 0.4151 - loss: 1.2965 - val_accuracy: 0.7225 - val_loss: 0.8847 - learning_rate: 2.9000e-06


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
   1/183 [37m━━━━━━━━━━━━━━━━━━━━  44:47 15s/step - accuracy: 0.7500 - loss: 0.9652

<div class="k-default-codeblock">
```

```
</div>
   2/183 [37m━━━━━━━━━━━━━━━━━━━━  43:26 14s/step - accuracy: 0.7500 - loss: 0.9399

<div class="k-default-codeblock">
```

```
</div>
   3/183 [37m━━━━━━━━━━━━━━━━━━━━  43:07 14s/step - accuracy: 0.7500 - loss: 0.9463

<div class="k-default-codeblock">
```

```
</div>
   4/183 [37m━━━━━━━━━━━━━━━━━━━━  42:38 14s/step - accuracy: 0.7578 - loss: 0.9380

<div class="k-default-codeblock">
```

```
</div>
   5/183 [37m━━━━━━━━━━━━━━━━━━━━  42:16 14s/step - accuracy: 0.7613 - loss: 0.9331

<div class="k-default-codeblock">
```

```
</div>
   6/183 [37m━━━━━━━━━━━━━━━━━━━━  41:57 14s/step - accuracy: 0.7559 - loss: 0.9292

<div class="k-default-codeblock">
```

```
</div>
   7/183 [37m━━━━━━━━━━━━━━━━━━━━  41:35 14s/step - accuracy: 0.7500 - loss: 0.9297

<div class="k-default-codeblock">
```

```
</div>
   8/183 [37m━━━━━━━━━━━━━━━━━━━━  41:17 14s/step - accuracy: 0.7480 - loss: 0.9275

<div class="k-default-codeblock">
```

```
</div>
   9/183 [37m━━━━━━━━━━━━━━━━━━━━  41:01 14s/step - accuracy: 0.7421 - loss: 0.9286

<div class="k-default-codeblock">
```

```
</div>
  10/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:45 14s/step - accuracy: 0.7366 - loss: 0.9294

<div class="k-default-codeblock">
```

```
</div>
  11/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:29 14s/step - accuracy: 0.7316 - loss: 0.9302

<div class="k-default-codeblock">
```

```
</div>
  12/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:11 14s/step - accuracy: 0.7288 - loss: 0.9287

<div class="k-default-codeblock">
```

```
</div>
  13/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:55 14s/step - accuracy: 0.7260 - loss: 0.9278

<div class="k-default-codeblock">
```

```
</div>
  14/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:39 14s/step - accuracy: 0.7239 - loss: 0.9275

<div class="k-default-codeblock">
```

```
</div>
  15/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:22 14s/step - accuracy: 0.7223 - loss: 0.9271

<div class="k-default-codeblock">
```

```
</div>
  16/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:07 14s/step - accuracy: 0.7201 - loss: 0.9277

<div class="k-default-codeblock">
```

```
</div>
  17/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:53 14s/step - accuracy: 0.7184 - loss: 0.9287

<div class="k-default-codeblock">
```

```
</div>
  18/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:37 14s/step - accuracy: 0.7155 - loss: 0.9305

<div class="k-default-codeblock">
```

```
</div>
  19/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:24 14s/step - accuracy: 0.7125 - loss: 0.9324

<div class="k-default-codeblock">
```

```
</div>
  20/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:10 14s/step - accuracy: 0.7097 - loss: 0.9338

<div class="k-default-codeblock">
```

```
</div>
  21/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:56 14s/step - accuracy: 0.7062 - loss: 0.9355

<div class="k-default-codeblock">
```

```
</div>
  22/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:42 14s/step - accuracy: 0.7033 - loss: 0.9369

<div class="k-default-codeblock">
```

```
</div>
  23/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:28 14s/step - accuracy: 0.7009 - loss: 0.9383

<div class="k-default-codeblock">
```

```
</div>
  24/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:13 14s/step - accuracy: 0.6988 - loss: 0.9392

<div class="k-default-codeblock">
```

```
</div>
  25/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:58 14s/step - accuracy: 0.6970 - loss: 0.9399

<div class="k-default-codeblock">
```

```
</div>
  26/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:44 14s/step - accuracy: 0.6954 - loss: 0.9404

<div class="k-default-codeblock">
```

```
</div>
  27/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:29 14s/step - accuracy: 0.6938 - loss: 0.9411

<div class="k-default-codeblock">
```

```
</div>
  28/183 ━━━[37m━━━━━━━━━━━━━━━━━  36:15 14s/step - accuracy: 0.6920 - loss: 0.9420

<div class="k-default-codeblock">
```

```
</div>
  29/183 ━━━[37m━━━━━━━━━━━━━━━━━  36:00 14s/step - accuracy: 0.6901 - loss: 0.9428

<div class="k-default-codeblock">
```

```
</div>
  30/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:46 14s/step - accuracy: 0.6884 - loss: 0.9439

<div class="k-default-codeblock">
```

```
</div>
  31/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:31 14s/step - accuracy: 0.6867 - loss: 0.9449

<div class="k-default-codeblock">
```

```
</div>
  32/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:16 14s/step - accuracy: 0.6851 - loss: 0.9457

<div class="k-default-codeblock">
```

```
</div>
  33/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:02 14s/step - accuracy: 0.6837 - loss: 0.9463

<div class="k-default-codeblock">
```

```
</div>
  34/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:48 14s/step - accuracy: 0.6822 - loss: 0.9470

<div class="k-default-codeblock">
```

```
</div>
  35/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:33 14s/step - accuracy: 0.6806 - loss: 0.9476

<div class="k-default-codeblock">
```

```
</div>
  36/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:20 14s/step - accuracy: 0.6791 - loss: 0.9484

<div class="k-default-codeblock">
```

```
</div>
  37/183 ━━━━[37m━━━━━━━━━━━━━━━━  34:05 14s/step - accuracy: 0.6777 - loss: 0.9491

<div class="k-default-codeblock">
```

```
</div>
  38/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:50 14s/step - accuracy: 0.6764 - loss: 0.9498

<div class="k-default-codeblock">
```

```
</div>
  39/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:37 14s/step - accuracy: 0.6752 - loss: 0.9504

<div class="k-default-codeblock">
```

```
</div>
  40/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:22 14s/step - accuracy: 0.6739 - loss: 0.9509

<div class="k-default-codeblock">
```

```
</div>
  41/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:08 14s/step - accuracy: 0.6726 - loss: 0.9514

<div class="k-default-codeblock">
```

```
</div>
  42/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:54 14s/step - accuracy: 0.6715 - loss: 0.9519

<div class="k-default-codeblock">
```

```
</div>
  43/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:40 14s/step - accuracy: 0.6703 - loss: 0.9525

<div class="k-default-codeblock">
```

```
</div>
  44/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:26 14s/step - accuracy: 0.6689 - loss: 0.9533

<div class="k-default-codeblock">
```

```
</div>
  45/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:12 14s/step - accuracy: 0.6678 - loss: 0.9540

<div class="k-default-codeblock">
```

```
</div>
  46/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:58 14s/step - accuracy: 0.6666 - loss: 0.9547

<div class="k-default-codeblock">
```

```
</div>
  47/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:45 14s/step - accuracy: 0.6654 - loss: 0.9553

<div class="k-default-codeblock">
```

```
</div>
  48/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:32 14s/step - accuracy: 0.6644 - loss: 0.9558

<div class="k-default-codeblock">
```

```
</div>
  49/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:18 14s/step - accuracy: 0.6635 - loss: 0.9563

<div class="k-default-codeblock">
```

```
</div>
  50/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:04 14s/step - accuracy: 0.6625 - loss: 0.9568

<div class="k-default-codeblock">
```

```
</div>
  51/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:50 14s/step - accuracy: 0.6616 - loss: 0.9573

<div class="k-default-codeblock">
```

```
</div>
  52/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:36 14s/step - accuracy: 0.6606 - loss: 0.9580

<div class="k-default-codeblock">
```

```
</div>
  53/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:21 14s/step - accuracy: 0.6596 - loss: 0.9585

<div class="k-default-codeblock">
```

```
</div>
  54/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:07 14s/step - accuracy: 0.6586 - loss: 0.9590

<div class="k-default-codeblock">
```

```
</div>
  55/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:53 14s/step - accuracy: 0.6577 - loss: 0.9596

<div class="k-default-codeblock">
```

```
</div>
  56/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:39 14s/step - accuracy: 0.6568 - loss: 0.9602

<div class="k-default-codeblock">
```

```
</div>
  57/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:25 14s/step - accuracy: 0.6559 - loss: 0.9608

<div class="k-default-codeblock">
```

```
</div>
  58/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:11 14s/step - accuracy: 0.6551 - loss: 0.9613

<div class="k-default-codeblock">
```

```
</div>
  59/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:57 14s/step - accuracy: 0.6541 - loss: 0.9619

<div class="k-default-codeblock">
```

```
</div>
  60/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:43 14s/step - accuracy: 0.6532 - loss: 0.9625

<div class="k-default-codeblock">
```

```
</div>
  61/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:29 14s/step - accuracy: 0.6524 - loss: 0.9632

<div class="k-default-codeblock">
```

```
</div>
  62/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:15 14s/step - accuracy: 0.6516 - loss: 0.9637

<div class="k-default-codeblock">
```

```
</div>
  63/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:01 14s/step - accuracy: 0.6508 - loss: 0.9642

<div class="k-default-codeblock">
```

```
</div>
  64/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:47 14s/step - accuracy: 0.6501 - loss: 0.9648

<div class="k-default-codeblock">
```

```
</div>
  65/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:33 14s/step - accuracy: 0.6493 - loss: 0.9653

<div class="k-default-codeblock">
```

```
</div>
  66/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:18 14s/step - accuracy: 0.6486 - loss: 0.9659

<div class="k-default-codeblock">
```

```
</div>
  67/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:04 14s/step - accuracy: 0.6479 - loss: 0.9664

<div class="k-default-codeblock">
```

```
</div>
  68/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:50 14s/step - accuracy: 0.6472 - loss: 0.9669

<div class="k-default-codeblock">
```

```
</div>
  69/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:37 14s/step - accuracy: 0.6465 - loss: 0.9673

<div class="k-default-codeblock">
```

```
</div>
  70/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:22 14s/step - accuracy: 0.6460 - loss: 0.9676

<div class="k-default-codeblock">
```

```
</div>
  71/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:08 14s/step - accuracy: 0.6454 - loss: 0.9679

<div class="k-default-codeblock">
```

```
</div>
  72/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:54 14s/step - accuracy: 0.6448 - loss: 0.9683

<div class="k-default-codeblock">
```

```
</div>
  73/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:40 14s/step - accuracy: 0.6442 - loss: 0.9686

<div class="k-default-codeblock">
```

```
</div>
  74/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:26 14s/step - accuracy: 0.6437 - loss: 0.9689

<div class="k-default-codeblock">
```

```
</div>
  75/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:12 14s/step - accuracy: 0.6432 - loss: 0.9692

<div class="k-default-codeblock">
```

```
</div>
  76/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:58 14s/step - accuracy: 0.6426 - loss: 0.9695

<div class="k-default-codeblock">
```

```
</div>
  77/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:44 14s/step - accuracy: 0.6420 - loss: 0.9699

<div class="k-default-codeblock">
```

```
</div>
  78/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:30 14s/step - accuracy: 0.6415 - loss: 0.9703

<div class="k-default-codeblock">
```

```
</div>
  79/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:16 14s/step - accuracy: 0.6409 - loss: 0.9706

<div class="k-default-codeblock">
```

```
</div>
  80/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:02 14s/step - accuracy: 0.6403 - loss: 0.9710

<div class="k-default-codeblock">
```

```
</div>
  81/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:48 14s/step - accuracy: 0.6398 - loss: 0.9714

<div class="k-default-codeblock">
```

```
</div>
  82/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:34 14s/step - accuracy: 0.6392 - loss: 0.9718

<div class="k-default-codeblock">
```

```
</div>
  83/183 ━━━━━━━━━[37m━━━━━━━━━━━  23:20 14s/step - accuracy: 0.6386 - loss: 0.9722

<div class="k-default-codeblock">
```

```
</div>
  84/183 ━━━━━━━━━[37m━━━━━━━━━━━  23:06 14s/step - accuracy: 0.6380 - loss: 0.9726

<div class="k-default-codeblock">
```

```
</div>
  85/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:52 14s/step - accuracy: 0.6374 - loss: 0.9729

<div class="k-default-codeblock">
```

```
</div>
  86/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:38 14s/step - accuracy: 0.6368 - loss: 0.9734

<div class="k-default-codeblock">
```

```
</div>
  87/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:25 14s/step - accuracy: 0.6362 - loss: 0.9738

<div class="k-default-codeblock">
```

```
</div>
  88/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:11 14s/step - accuracy: 0.6357 - loss: 0.9741

<div class="k-default-codeblock">
```

```
</div>
  89/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:57 14s/step - accuracy: 0.6352 - loss: 0.9745

<div class="k-default-codeblock">
```

```
</div>
  90/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:43 14s/step - accuracy: 0.6347 - loss: 0.9748

<div class="k-default-codeblock">
```

```
</div>
  91/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:29 14s/step - accuracy: 0.6341 - loss: 0.9752

<div class="k-default-codeblock">
```

```
</div>
  92/183 ━━━━━━━━━━[37m━━━━━━━━━━  21:15 14s/step - accuracy: 0.6336 - loss: 0.9755

<div class="k-default-codeblock">
```

```
</div>
  93/183 ━━━━━━━━━━[37m━━━━━━━━━━  21:02 14s/step - accuracy: 0.6330 - loss: 0.9759

<div class="k-default-codeblock">
```

```
</div>
  94/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:48 14s/step - accuracy: 0.6325 - loss: 0.9762

<div class="k-default-codeblock">
```

```
</div>
  95/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:34 14s/step - accuracy: 0.6319 - loss: 0.9766

<div class="k-default-codeblock">
```

```
</div>
  96/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:20 14s/step - accuracy: 0.6314 - loss: 0.9770

<div class="k-default-codeblock">
```

```
</div>
  97/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:06 14s/step - accuracy: 0.6309 - loss: 0.9773

<div class="k-default-codeblock">
```

```
</div>
  98/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:52 14s/step - accuracy: 0.6303 - loss: 0.9777

<div class="k-default-codeblock">
```

```
</div>
  99/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:38 14s/step - accuracy: 0.6298 - loss: 0.9780

<div class="k-default-codeblock">
```

```
</div>
 100/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:24 14s/step - accuracy: 0.6293 - loss: 0.9784

<div class="k-default-codeblock">
```

```
</div>
 101/183 ━━━━━━━━━━━[37m━━━━━━━━━  19:10 14s/step - accuracy: 0.6288 - loss: 0.9787

<div class="k-default-codeblock">
```

```
</div>
 102/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:56 14s/step - accuracy: 0.6284 - loss: 0.9790

<div class="k-default-codeblock">
```

```
</div>
 103/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:43 14s/step - accuracy: 0.6279 - loss: 0.9793

<div class="k-default-codeblock">
```

```
</div>
 104/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:29 14s/step - accuracy: 0.6275 - loss: 0.9796

<div class="k-default-codeblock">
```

```
</div>
 105/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:15 14s/step - accuracy: 0.6271 - loss: 0.9798

<div class="k-default-codeblock">
```

```
</div>
 106/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:01 14s/step - accuracy: 0.6267 - loss: 0.9801

<div class="k-default-codeblock">
```

```
</div>
 107/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:47 14s/step - accuracy: 0.6263 - loss: 0.9803

<div class="k-default-codeblock">
```

```
</div>
 108/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:34 14s/step - accuracy: 0.6259 - loss: 0.9805

<div class="k-default-codeblock">
```

```
</div>
 109/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:20 14s/step - accuracy: 0.6255 - loss: 0.9807

<div class="k-default-codeblock">
```

```
</div>
 110/183 ━━━━━━━━━━━━[37m━━━━━━━━  17:06 14s/step - accuracy: 0.6251 - loss: 0.9809

<div class="k-default-codeblock">
```

```
</div>
 111/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:52 14s/step - accuracy: 0.6247 - loss: 0.9811

<div class="k-default-codeblock">
```

```
</div>
 112/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:38 14s/step - accuracy: 0.6244 - loss: 0.9813

<div class="k-default-codeblock">
```

```
</div>
 113/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:24 14s/step - accuracy: 0.6240 - loss: 0.9815

<div class="k-default-codeblock">
```

```
</div>
 114/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:10 14s/step - accuracy: 0.6237 - loss: 0.9816

<div class="k-default-codeblock">
```

```
</div>
 115/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:56 14s/step - accuracy: 0.6233 - loss: 0.9818

<div class="k-default-codeblock">
```

```
</div>
 116/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:42 14s/step - accuracy: 0.6230 - loss: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 117/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:28 14s/step - accuracy: 0.6227 - loss: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 118/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:14 14s/step - accuracy: 0.6224 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 119/183 ━━━━━━━━━━━━━[37m━━━━━━━  15:00 14s/step - accuracy: 0.6221 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 120/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:46 14s/step - accuracy: 0.6219 - loss: 0.9823

<div class="k-default-codeblock">
```

```
</div>
 121/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:32 14s/step - accuracy: 0.6216 - loss: 0.9824

<div class="k-default-codeblock">
```

```
</div>
 122/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:18 14s/step - accuracy: 0.6213 - loss: 0.9824

<div class="k-default-codeblock">
```

```
</div>
 123/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:03 14s/step - accuracy: 0.6210 - loss: 0.9825

<div class="k-default-codeblock">
```

```
</div>
 124/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:49 14s/step - accuracy: 0.6208 - loss: 0.9825

<div class="k-default-codeblock">
```

```
</div>
 125/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:35 14s/step - accuracy: 0.6206 - loss: 0.9825

<div class="k-default-codeblock">
```

```
</div>
 126/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:21 14s/step - accuracy: 0.6204 - loss: 0.9825

<div class="k-default-codeblock">
```

```
</div>
 127/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:07 14s/step - accuracy: 0.6201 - loss: 0.9825

<div class="k-default-codeblock">
```

```
</div>
 128/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:53 14s/step - accuracy: 0.6199 - loss: 0.9824

<div class="k-default-codeblock">
```

```
</div>
 129/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:39 14s/step - accuracy: 0.6197 - loss: 0.9824

<div class="k-default-codeblock">
```

```
</div>
 130/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:25 14s/step - accuracy: 0.6195 - loss: 0.9824

<div class="k-default-codeblock">
```

```
</div>
 131/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:11 14s/step - accuracy: 0.6193 - loss: 0.9824

<div class="k-default-codeblock">
```

```
</div>
 132/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:57 14s/step - accuracy: 0.6191 - loss: 0.9823

<div class="k-default-codeblock">
```

```
</div>
 133/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:43 14s/step - accuracy: 0.6189 - loss: 0.9823

<div class="k-default-codeblock">
```

```
</div>
 134/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:29 14s/step - accuracy: 0.6188 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 135/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:14 14s/step - accuracy: 0.6186 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 136/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:00 14s/step - accuracy: 0.6184 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 137/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:46 14s/step - accuracy: 0.6182 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 138/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:32 14s/step - accuracy: 0.6180 - loss: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 139/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:18 14s/step - accuracy: 0.6178 - loss: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 140/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:04 14s/step - accuracy: 0.6176 - loss: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 141/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:50 14s/step - accuracy: 0.6174 - loss: 0.9821 

<div class="k-default-codeblock">
```

```
</div>
 142/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:36 14s/step - accuracy: 0.6172 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 143/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:22 14s/step - accuracy: 0.6170 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 144/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:08 14s/step - accuracy: 0.6168 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 145/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:54 14s/step - accuracy: 0.6166 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 146/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:40 14s/step - accuracy: 0.6164 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 147/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:26 14s/step - accuracy: 0.6162 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 148/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:11 14s/step - accuracy: 0.6161 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 149/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:57 14s/step - accuracy: 0.6159 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 150/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:43 14s/step - accuracy: 0.6158 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 151/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:29 14s/step - accuracy: 0.6156 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 152/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:15 14s/step - accuracy: 0.6155 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 153/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:01 14s/step - accuracy: 0.6153 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 154/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:47 14s/step - accuracy: 0.6152 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 155/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:33 14s/step - accuracy: 0.6151 - loss: 0.9823

<div class="k-default-codeblock">
```

```
</div>
 156/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:19 14s/step - accuracy: 0.6149 - loss: 0.9823

<div class="k-default-codeblock">
```

```
</div>
 157/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:05 14s/step - accuracy: 0.6148 - loss: 0.9823

<div class="k-default-codeblock">
```

```
</div>
 158/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:51 14s/step - accuracy: 0.6147 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 159/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:37 14s/step - accuracy: 0.6146 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 160/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:23 14s/step - accuracy: 0.6145 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 161/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:09 14s/step - accuracy: 0.6144 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 162/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:55 14s/step - accuracy: 0.6143 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 163/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:41 14s/step - accuracy: 0.6142 - loss: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 164/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:27 14s/step - accuracy: 0.6141 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 165/183 ━━━━━━━━━━━━━━━━━━[37m━━  4:12 14s/step - accuracy: 0.6140 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 166/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:58 14s/step - accuracy: 0.6140 - loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 167/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:44 14s/step - accuracy: 0.6139 - loss: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 168/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:30 14s/step - accuracy: 0.6138 - loss: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 169/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:16 14s/step - accuracy: 0.6137 - loss: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 170/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:02 14s/step - accuracy: 0.6136 - loss: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 171/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:48 14s/step - accuracy: 0.6135 - loss: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 172/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:34 14s/step - accuracy: 0.6134 - loss: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 173/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:20 14s/step - accuracy: 0.6134 - loss: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 174/183 ━━━━━━━━━━━━━━━━━━━[37m━  2:06 14s/step - accuracy: 0.6133 - loss: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 175/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:52 14s/step - accuracy: 0.6132 - loss: 0.9818

<div class="k-default-codeblock">
```

```
</div>
 176/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:38 14s/step - accuracy: 0.6132 - loss: 0.9818

<div class="k-default-codeblock">
```

```
</div>
 177/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:24 14s/step - accuracy: 0.6131 - loss: 0.9817

<div class="k-default-codeblock">
```

```
</div>
 178/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:10 14s/step - accuracy: 0.6131 - loss: 0.9817

<div class="k-default-codeblock">
```

```
</div>
 179/183 ━━━━━━━━━━━━━━━━━━━[37m━  56s 14s/step - accuracy: 0.6130 - loss: 0.9816 

<div class="k-default-codeblock">
```

```
</div>
 180/183 ━━━━━━━━━━━━━━━━━━━[37m━  42s 14s/step - accuracy: 0.6129 - loss: 0.9816

<div class="k-default-codeblock">
```

```
</div>
 181/183 ━━━━━━━━━━━━━━━━━━━[37m━  28s 14s/step - accuracy: 0.6129 - loss: 0.9815

<div class="k-default-codeblock">
```

```
</div>
 182/183 ━━━━━━━━━━━━━━━━━━━[37m━  14s 14s/step - accuracy: 0.6128 - loss: 0.9815

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 0s 14s/step - accuracy: 0.6128 - loss: 0.9815 

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 2722s 15s/step - accuracy: 0.6127 - loss: 0.9814 - val_accuracy: 0.7575 - val_loss: 0.7755 - learning_rate: 4.8000e-06


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
   1/183 [37m━━━━━━━━━━━━━━━━━━━━  44:30 15s/step - accuracy: 0.7500 - loss: 0.8347

<div class="k-default-codeblock">
```

```
</div>
   2/183 [37m━━━━━━━━━━━━━━━━━━━━  42:33 14s/step - accuracy: 0.7188 - loss: 0.8343

<div class="k-default-codeblock">
```

```
</div>
   3/183 [37m━━━━━━━━━━━━━━━━━━━━  42:29 14s/step - accuracy: 0.7292 - loss: 0.7993

<div class="k-default-codeblock">
```

```
</div>
   4/183 [37m━━━━━━━━━━━━━━━━━━━━  42:09 14s/step - accuracy: 0.7266 - loss: 0.8006

<div class="k-default-codeblock">
```

```
</div>
   5/183 [37m━━━━━━━━━━━━━━━━━━━━  41:48 14s/step - accuracy: 0.7312 - loss: 0.8018

<div class="k-default-codeblock">
```

```
</div>
   6/183 [37m━━━━━━━━━━━━━━━━━━━━  41:35 14s/step - accuracy: 0.7344 - loss: 0.7997

<div class="k-default-codeblock">
```

```
</div>
   7/183 [37m━━━━━━━━━━━━━━━━━━━━  41:19 14s/step - accuracy: 0.7366 - loss: 0.7965

<div class="k-default-codeblock">
```

```
</div>
   8/183 [37m━━━━━━━━━━━━━━━━━━━━  41:02 14s/step - accuracy: 0.7363 - loss: 0.7959

<div class="k-default-codeblock">
```

```
</div>
   9/183 [37m━━━━━━━━━━━━━━━━━━━━  40:48 14s/step - accuracy: 0.7378 - loss: 0.7934

<div class="k-default-codeblock">
```

```
</div>
  10/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:30 14s/step - accuracy: 0.7378 - loss: 0.7934

<div class="k-default-codeblock">
```

```
</div>
  11/183 ━[37m━━━━━━━━━━━━━━━━━━━  40:12 14s/step - accuracy: 0.7369 - loss: 0.7948

<div class="k-default-codeblock">
```

```
</div>
  12/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:55 14s/step - accuracy: 0.7371 - loss: 0.7945

<div class="k-default-codeblock">
```

```
</div>
  13/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:39 14s/step - accuracy: 0.7381 - loss: 0.7932

<div class="k-default-codeblock">
```

```
</div>
  14/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:23 14s/step - accuracy: 0.7389 - loss: 0.7922

<div class="k-default-codeblock">
```

```
</div>
  15/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:08 14s/step - accuracy: 0.7397 - loss: 0.7905

<div class="k-default-codeblock">
```

```
</div>
  16/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:53 14s/step - accuracy: 0.7408 - loss: 0.7893

<div class="k-default-codeblock">
```

```
</div>
  17/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:37 14s/step - accuracy: 0.7418 - loss: 0.7880

<div class="k-default-codeblock">
```

```
</div>
  18/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:22 14s/step - accuracy: 0.7418 - loss: 0.7879

<div class="k-default-codeblock">
```

```
</div>
  19/183 ━━[37m━━━━━━━━━━━━━━━━━━  38:07 14s/step - accuracy: 0.7419 - loss: 0.7881

<div class="k-default-codeblock">
```

```
</div>
  20/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:52 14s/step - accuracy: 0.7417 - loss: 0.7885

<div class="k-default-codeblock">
```

```
</div>
  21/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:37 14s/step - accuracy: 0.7413 - loss: 0.7890

<div class="k-default-codeblock">
```

```
</div>
  22/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:23 14s/step - accuracy: 0.7409 - loss: 0.7894

<div class="k-default-codeblock">
```

```
</div>
  23/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:08 14s/step - accuracy: 0.7408 - loss: 0.7896

<div class="k-default-codeblock">
```

```
</div>
  24/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:54 14s/step - accuracy: 0.7399 - loss: 0.7910

<div class="k-default-codeblock">
```

```
</div>
  25/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:39 14s/step - accuracy: 0.7391 - loss: 0.7918

<div class="k-default-codeblock">
```

```
</div>
  26/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:25 14s/step - accuracy: 0.7382 - loss: 0.7926

<div class="k-default-codeblock">
```

```
</div>
  27/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:10 14s/step - accuracy: 0.7374 - loss: 0.7935

<div class="k-default-codeblock">
```

```
</div>
  28/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:55 14s/step - accuracy: 0.7368 - loss: 0.7944

<div class="k-default-codeblock">
```

```
</div>
  29/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:41 14s/step - accuracy: 0.7360 - loss: 0.7957

<div class="k-default-codeblock">
```

```
</div>
  30/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:27 14s/step - accuracy: 0.7350 - loss: 0.7970

<div class="k-default-codeblock">
```

```
</div>
  31/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:13 14s/step - accuracy: 0.7340 - loss: 0.7981

<div class="k-default-codeblock">
```

```
</div>
  32/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:59 14s/step - accuracy: 0.7332 - loss: 0.7990

<div class="k-default-codeblock">
```

```
</div>
  33/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:44 14s/step - accuracy: 0.7324 - loss: 0.7997

<div class="k-default-codeblock">
```

```
</div>
  34/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:30 14s/step - accuracy: 0.7319 - loss: 0.8003

<div class="k-default-codeblock">
```

```
</div>
  35/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:16 14s/step - accuracy: 0.7316 - loss: 0.8006

<div class="k-default-codeblock">
```

```
</div>
  36/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:01 14s/step - accuracy: 0.7312 - loss: 0.8009

<div class="k-default-codeblock">
```

```
</div>
  37/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:47 14s/step - accuracy: 0.7309 - loss: 0.8011

<div class="k-default-codeblock">
```

```
</div>
  38/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:33 14s/step - accuracy: 0.7305 - loss: 0.8012

<div class="k-default-codeblock">
```

```
</div>
  39/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:19 14s/step - accuracy: 0.7300 - loss: 0.8014

<div class="k-default-codeblock">
```

```
</div>
  40/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:05 14s/step - accuracy: 0.7295 - loss: 0.8015

<div class="k-default-codeblock">
```

```
</div>
  41/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:50 14s/step - accuracy: 0.7290 - loss: 0.8015

<div class="k-default-codeblock">
```

```
</div>
  42/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:36 14s/step - accuracy: 0.7286 - loss: 0.8016

<div class="k-default-codeblock">
```

```
</div>
  43/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:22 14s/step - accuracy: 0.7281 - loss: 0.8017

<div class="k-default-codeblock">
```

```
</div>
  44/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:08 14s/step - accuracy: 0.7277 - loss: 0.8017

<div class="k-default-codeblock">
```

```
</div>
  45/183 ━━━━[37m━━━━━━━━━━━━━━━━  31:54 14s/step - accuracy: 0.7270 - loss: 0.8020

<div class="k-default-codeblock">
```

```
</div>
  46/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:40 14s/step - accuracy: 0.7264 - loss: 0.8022

<div class="k-default-codeblock">
```

```
</div>
  47/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:26 14s/step - accuracy: 0.7258 - loss: 0.8023

<div class="k-default-codeblock">
```

```
</div>
  48/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:13 14s/step - accuracy: 0.7250 - loss: 0.8027

<div class="k-default-codeblock">
```

```
</div>
  49/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:59 14s/step - accuracy: 0.7243 - loss: 0.8031

<div class="k-default-codeblock">
```

```
</div>
  50/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:45 14s/step - accuracy: 0.7236 - loss: 0.8035

<div class="k-default-codeblock">
```

```
</div>
  51/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:31 14s/step - accuracy: 0.7229 - loss: 0.8040

<div class="k-default-codeblock">
```

```
</div>
  52/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:17 14s/step - accuracy: 0.7222 - loss: 0.8045

<div class="k-default-codeblock">
```

```
</div>
  53/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:03 14s/step - accuracy: 0.7217 - loss: 0.8050

<div class="k-default-codeblock">
```

```
</div>
  54/183 ━━━━━[37m━━━━━━━━━━━━━━━  29:49 14s/step - accuracy: 0.7212 - loss: 0.8055

<div class="k-default-codeblock">
```

```
</div>
  55/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:35 14s/step - accuracy: 0.7206 - loss: 0.8060

<div class="k-default-codeblock">
```

```
</div>
  56/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:21 14s/step - accuracy: 0.7201 - loss: 0.8065

<div class="k-default-codeblock">
```

```
</div>
  57/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:07 14s/step - accuracy: 0.7196 - loss: 0.8070

<div class="k-default-codeblock">
```

```
</div>
  58/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:53 14s/step - accuracy: 0.7190 - loss: 0.8077

<div class="k-default-codeblock">
```

```
</div>
  59/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:39 14s/step - accuracy: 0.7185 - loss: 0.8083

<div class="k-default-codeblock">
```

```
</div>
  60/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:25 14s/step - accuracy: 0.7179 - loss: 0.8090

<div class="k-default-codeblock">
```

```
</div>
  61/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:12 14s/step - accuracy: 0.7173 - loss: 0.8097

<div class="k-default-codeblock">
```

```
</div>
  62/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:59 14s/step - accuracy: 0.7168 - loss: 0.8105

<div class="k-default-codeblock">
```

```
</div>
  63/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:45 14s/step - accuracy: 0.7162 - loss: 0.8114

<div class="k-default-codeblock">
```

```
</div>
  64/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:32 14s/step - accuracy: 0.7156 - loss: 0.8122

<div class="k-default-codeblock">
```

```
</div>
  65/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:18 14s/step - accuracy: 0.7149 - loss: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  66/183 ━━━━━━━[37m━━━━━━━━━━━━━  27:05 14s/step - accuracy: 0.7143 - loss: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  67/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:51 14s/step - accuracy: 0.7136 - loss: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  68/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:37 14s/step - accuracy: 0.7130 - loss: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  69/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:23 14s/step - accuracy: 0.7125 - loss: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  70/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:09 14s/step - accuracy: 0.7119 - loss: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  71/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:55 14s/step - accuracy: 0.7115 - loss: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  72/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:41 14s/step - accuracy: 0.7110 - loss: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  73/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:28 14s/step - accuracy: 0.7105 - loss: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  74/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:14 14s/step - accuracy: 0.7100 - loss: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  75/183 ━━━━━━━━[37m━━━━━━━━━━━━  25:01 14s/step - accuracy: 0.7095 - loss: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  76/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:47 14s/step - accuracy: 0.7090 - loss: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  77/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:33 14s/step - accuracy: 0.7085 - loss: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  78/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:19 14s/step - accuracy: 0.7080 - loss: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  79/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:05 14s/step - accuracy: 0.7076 - loss: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  80/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:52 14s/step - accuracy: 0.7071 - loss: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  81/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:38 14s/step - accuracy: 0.7066 - loss: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  82/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:24 14s/step - accuracy: 0.7062 - loss: 0.8261

<div class="k-default-codeblock">
```

```
</div>
  83/183 ━━━━━━━━━[37m━━━━━━━━━━━  23:10 14s/step - accuracy: 0.7058 - loss: 0.8267

<div class="k-default-codeblock">
```

```
</div>
  84/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:56 14s/step - accuracy: 0.7054 - loss: 0.8273

<div class="k-default-codeblock">
```

```
</div>
  85/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:42 14s/step - accuracy: 0.7050 - loss: 0.8279

<div class="k-default-codeblock">
```

```
</div>
  86/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:28 14s/step - accuracy: 0.7047 - loss: 0.8285

<div class="k-default-codeblock">
```

```
</div>
  87/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:15 14s/step - accuracy: 0.7043 - loss: 0.8292

<div class="k-default-codeblock">
```

```
</div>
  88/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:01 14s/step - accuracy: 0.7039 - loss: 0.8298

<div class="k-default-codeblock">
```

```
</div>
  89/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:47 14s/step - accuracy: 0.7035 - loss: 0.8304

<div class="k-default-codeblock">
```

```
</div>
  90/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:33 14s/step - accuracy: 0.7031 - loss: 0.8310

<div class="k-default-codeblock">
```

```
</div>
  91/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:19 14s/step - accuracy: 0.7027 - loss: 0.8316

<div class="k-default-codeblock">
```

```
</div>
  92/183 ━━━━━━━━━━[37m━━━━━━━━━━  21:05 14s/step - accuracy: 0.7024 - loss: 0.8321

<div class="k-default-codeblock">
```

```
</div>
  93/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:51 14s/step - accuracy: 0.7020 - loss: 0.8326

<div class="k-default-codeblock">
```

```
</div>
  94/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:37 14s/step - accuracy: 0.7017 - loss: 0.8332

<div class="k-default-codeblock">
```

```
</div>
  95/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:23 14s/step - accuracy: 0.7013 - loss: 0.8338

<div class="k-default-codeblock">
```

```
</div>
  96/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:09 14s/step - accuracy: 0.7009 - loss: 0.8344

<div class="k-default-codeblock">
```

```
</div>
  97/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:55 14s/step - accuracy: 0.7005 - loss: 0.8350

<div class="k-default-codeblock">
```

```
</div>
  98/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:41 14s/step - accuracy: 0.7002 - loss: 0.8356

<div class="k-default-codeblock">
```

```
</div>
  99/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:27 14s/step - accuracy: 0.6998 - loss: 0.8362

<div class="k-default-codeblock">
```

```
</div>
 100/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:13 14s/step - accuracy: 0.6994 - loss: 0.8368

<div class="k-default-codeblock">
```

```
</div>
 101/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:59 14s/step - accuracy: 0.6990 - loss: 0.8375

<div class="k-default-codeblock">
```

```
</div>
 102/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:45 14s/step - accuracy: 0.6986 - loss: 0.8381

<div class="k-default-codeblock">
```

```
</div>
 103/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:31 14s/step - accuracy: 0.6983 - loss: 0.8386

<div class="k-default-codeblock">
```

```
</div>
 104/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:17 14s/step - accuracy: 0.6979 - loss: 0.8392

<div class="k-default-codeblock">
```

```
</div>
 105/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:03 14s/step - accuracy: 0.6976 - loss: 0.8397

<div class="k-default-codeblock">
```

```
</div>
 106/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:50 14s/step - accuracy: 0.6973 - loss: 0.8402

<div class="k-default-codeblock">
```

```
</div>
 107/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:36 14s/step - accuracy: 0.6969 - loss: 0.8406

<div class="k-default-codeblock">
```

```
</div>
 108/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:22 14s/step - accuracy: 0.6966 - loss: 0.8411

<div class="k-default-codeblock">
```

```
</div>
 109/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:08 14s/step - accuracy: 0.6963 - loss: 0.8416

<div class="k-default-codeblock">
```

```
</div>
 110/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:54 14s/step - accuracy: 0.6960 - loss: 0.8420

<div class="k-default-codeblock">
```

```
</div>
 111/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:40 14s/step - accuracy: 0.6957 - loss: 0.8424

<div class="k-default-codeblock">
```

```
</div>
 112/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:26 14s/step - accuracy: 0.6954 - loss: 0.8428

<div class="k-default-codeblock">
```

```
</div>
 113/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:12 14s/step - accuracy: 0.6951 - loss: 0.8432

<div class="k-default-codeblock">
```

```
</div>
 114/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:58 14s/step - accuracy: 0.6949 - loss: 0.8436

<div class="k-default-codeblock">
```

```
</div>
 115/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:44 14s/step - accuracy: 0.6946 - loss: 0.8440

<div class="k-default-codeblock">
```

```
</div>
 116/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:30 14s/step - accuracy: 0.6943 - loss: 0.8444

<div class="k-default-codeblock">
```

```
</div>
 117/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:16 14s/step - accuracy: 0.6941 - loss: 0.8448

<div class="k-default-codeblock">
```

```
</div>
 118/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:02 14s/step - accuracy: 0.6938 - loss: 0.8451

<div class="k-default-codeblock">
```

```
</div>
 119/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:48 14s/step - accuracy: 0.6936 - loss: 0.8455

<div class="k-default-codeblock">
```

```
</div>
 120/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:34 14s/step - accuracy: 0.6934 - loss: 0.8458

<div class="k-default-codeblock">
```

```
</div>
 121/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:20 14s/step - accuracy: 0.6932 - loss: 0.8461

<div class="k-default-codeblock">
```

```
</div>
 122/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:06 14s/step - accuracy: 0.6930 - loss: 0.8464

<div class="k-default-codeblock">
```

```
</div>
 123/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:52 14s/step - accuracy: 0.6928 - loss: 0.8467

<div class="k-default-codeblock">
```

```
</div>
 124/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:38 14s/step - accuracy: 0.6926 - loss: 0.8470

<div class="k-default-codeblock">
```

```
</div>
 125/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:24 14s/step - accuracy: 0.6924 - loss: 0.8473

<div class="k-default-codeblock">
```

```
</div>
 126/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:10 14s/step - accuracy: 0.6922 - loss: 0.8476

<div class="k-default-codeblock">
```

```
</div>
 127/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:57 14s/step - accuracy: 0.6920 - loss: 0.8478

<div class="k-default-codeblock">
```

```
</div>
 128/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:43 14s/step - accuracy: 0.6918 - loss: 0.8481

<div class="k-default-codeblock">
```

```
</div>
 129/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:29 14s/step - accuracy: 0.6917 - loss: 0.8483

<div class="k-default-codeblock">
```

```
</div>
 130/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:15 14s/step - accuracy: 0.6915 - loss: 0.8485

<div class="k-default-codeblock">
```

```
</div>
 131/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:01 14s/step - accuracy: 0.6913 - loss: 0.8487

<div class="k-default-codeblock">
```

```
</div>
 132/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:47 14s/step - accuracy: 0.6912 - loss: 0.8489

<div class="k-default-codeblock">
```

```
</div>
 133/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:33 14s/step - accuracy: 0.6910 - loss: 0.8491

<div class="k-default-codeblock">
```

```
</div>
 134/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:19 14s/step - accuracy: 0.6909 - loss: 0.8493

<div class="k-default-codeblock">
```

```
</div>
 135/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:05 14s/step - accuracy: 0.6908 - loss: 0.8494

<div class="k-default-codeblock">
```

```
</div>
 136/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:51 14s/step - accuracy: 0.6906 - loss: 0.8495

<div class="k-default-codeblock">
```

```
</div>
 137/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:37 14s/step - accuracy: 0.6905 - loss: 0.8497

<div class="k-default-codeblock">
```

```
</div>
 138/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:23 14s/step - accuracy: 0.6904 - loss: 0.8499

<div class="k-default-codeblock">
```

```
</div>
 139/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:10 14s/step - accuracy: 0.6902 - loss: 0.8500

<div class="k-default-codeblock">
```

```
</div>
 140/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:56 14s/step - accuracy: 0.6901 - loss: 0.8502 

<div class="k-default-codeblock">
```

```
</div>
 141/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:42 14s/step - accuracy: 0.6900 - loss: 0.8503

<div class="k-default-codeblock">
```

```
</div>
 142/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:28 14s/step - accuracy: 0.6898 - loss: 0.8505

<div class="k-default-codeblock">
```

```
</div>
 143/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:14 14s/step - accuracy: 0.6897 - loss: 0.8507

<div class="k-default-codeblock">
```

```
</div>
 144/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:00 14s/step - accuracy: 0.6895 - loss: 0.8509

<div class="k-default-codeblock">
```

```
</div>
 145/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:46 14s/step - accuracy: 0.6894 - loss: 0.8511

<div class="k-default-codeblock">
```

```
</div>
 146/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:32 14s/step - accuracy: 0.6892 - loss: 0.8514

<div class="k-default-codeblock">
```

```
</div>
 147/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:18 14s/step - accuracy: 0.6891 - loss: 0.8516

<div class="k-default-codeblock">
```

```
</div>
 148/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:05 14s/step - accuracy: 0.6890 - loss: 0.8517

<div class="k-default-codeblock">
```

```
</div>
 149/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:51 14s/step - accuracy: 0.6888 - loss: 0.8519

<div class="k-default-codeblock">
```

```
</div>
 150/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:37 14s/step - accuracy: 0.6887 - loss: 0.8521

<div class="k-default-codeblock">
```

```
</div>
 151/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:23 14s/step - accuracy: 0.6886 - loss: 0.8522

<div class="k-default-codeblock">
```

```
</div>
 152/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:09 14s/step - accuracy: 0.6884 - loss: 0.8524

<div class="k-default-codeblock">
```

```
</div>
 153/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:55 14s/step - accuracy: 0.6883 - loss: 0.8525

<div class="k-default-codeblock">
```

```
</div>
 154/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:41 14s/step - accuracy: 0.6882 - loss: 0.8527

<div class="k-default-codeblock">
```

```
</div>
 155/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:27 14s/step - accuracy: 0.6881 - loss: 0.8528

<div class="k-default-codeblock">
```

```
</div>
 156/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:14 14s/step - accuracy: 0.6880 - loss: 0.8530

<div class="k-default-codeblock">
```

```
</div>
 157/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:00 14s/step - accuracy: 0.6879 - loss: 0.8531

<div class="k-default-codeblock">
```

```
</div>
 158/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:46 14s/step - accuracy: 0.6878 - loss: 0.8532

<div class="k-default-codeblock">
```

```
</div>
 159/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:32 14s/step - accuracy: 0.6877 - loss: 0.8533

<div class="k-default-codeblock">
```

```
</div>
 160/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:18 14s/step - accuracy: 0.6877 - loss: 0.8535

<div class="k-default-codeblock">
```

```
</div>
 161/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:04 14s/step - accuracy: 0.6876 - loss: 0.8536

<div class="k-default-codeblock">
```

```
</div>
 162/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:50 14s/step - accuracy: 0.6875 - loss: 0.8537

<div class="k-default-codeblock">
```

```
</div>
 163/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:36 14s/step - accuracy: 0.6875 - loss: 0.8538

<div class="k-default-codeblock">
```

```
</div>
 164/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:23 14s/step - accuracy: 0.6874 - loss: 0.8539

<div class="k-default-codeblock">
```

```
</div>
 165/183 ━━━━━━━━━━━━━━━━━━[37m━━  4:09 14s/step - accuracy: 0.6873 - loss: 0.8540

<div class="k-default-codeblock">
```

```
</div>
 166/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:55 14s/step - accuracy: 0.6873 - loss: 0.8540

<div class="k-default-codeblock">
```

```
</div>
 167/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:41 14s/step - accuracy: 0.6872 - loss: 0.8541

<div class="k-default-codeblock">
```

```
</div>
 168/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:27 14s/step - accuracy: 0.6872 - loss: 0.8542

<div class="k-default-codeblock">
```

```
</div>
 169/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:13 14s/step - accuracy: 0.6871 - loss: 0.8543

<div class="k-default-codeblock">
```

```
</div>
 170/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:59 14s/step - accuracy: 0.6871 - loss: 0.8544

<div class="k-default-codeblock">
```

```
</div>
 171/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:46 14s/step - accuracy: 0.6870 - loss: 0.8544

<div class="k-default-codeblock">
```

```
</div>
 172/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:32 14s/step - accuracy: 0.6870 - loss: 0.8545

<div class="k-default-codeblock">
```

```
</div>
 173/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:18 14s/step - accuracy: 0.6870 - loss: 0.8546

<div class="k-default-codeblock">
```

```
</div>
 174/183 ━━━━━━━━━━━━━━━━━━━[37m━  2:04 14s/step - accuracy: 0.6869 - loss: 0.8547

<div class="k-default-codeblock">
```

```
</div>
 175/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:50 14s/step - accuracy: 0.6869 - loss: 0.8548

<div class="k-default-codeblock">
```

```
</div>
 176/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:36 14s/step - accuracy: 0.6868 - loss: 0.8548

<div class="k-default-codeblock">
```

```
</div>
 177/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:23 14s/step - accuracy: 0.6868 - loss: 0.8549

<div class="k-default-codeblock">
```

```
</div>
 178/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:09 14s/step - accuracy: 0.6867 - loss: 0.8550

<div class="k-default-codeblock">
```

```
</div>
 179/183 ━━━━━━━━━━━━━━━━━━━[37m━  55s 14s/step - accuracy: 0.6867 - loss: 0.8550 

<div class="k-default-codeblock">
```

```
</div>
 180/183 ━━━━━━━━━━━━━━━━━━━[37m━  41s 14s/step - accuracy: 0.6867 - loss: 0.8551

<div class="k-default-codeblock">
```

```
</div>
 181/183 ━━━━━━━━━━━━━━━━━━━[37m━  27s 14s/step - accuracy: 0.6866 - loss: 0.8552

<div class="k-default-codeblock">
```

```
</div>
 182/183 ━━━━━━━━━━━━━━━━━━━[37m━  13s 14s/step - accuracy: 0.6866 - loss: 0.8552

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 0s 14s/step - accuracy: 0.6865 - loss: 0.8553 

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 2677s 15s/step - accuracy: 0.6865 - loss: 0.8554 - val_accuracy: 0.7400 - val_loss: 0.7678 - learning_rate: 4.7230e-06


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
   1/183 [37m━━━━━━━━━━━━━━━━━━━━  41:51 14s/step - accuracy: 0.6250 - loss: 0.7948

<div class="k-default-codeblock">
```

```
</div>
   2/183 [37m━━━━━━━━━━━━━━━━━━━━  41:14 14s/step - accuracy: 0.6562 - loss: 0.7674

<div class="k-default-codeblock">
```

```
</div>
   3/183 [37m━━━━━━━━━━━━━━━━━━━━  41:10 14s/step - accuracy: 0.6597 - loss: 0.7733

<div class="k-default-codeblock">
```

```
</div>
   4/183 [37m━━━━━━━━━━━━━━━━━━━━  40:52 14s/step - accuracy: 0.6589 - loss: 0.7755

<div class="k-default-codeblock">
```

```
</div>
   5/183 [37m━━━━━━━━━━━━━━━━━━━━  40:38 14s/step - accuracy: 0.6621 - loss: 0.7814

<div class="k-default-codeblock">
```

```
</div>
   6/183 [37m━━━━━━━━━━━━━━━━━━━━  40:26 14s/step - accuracy: 0.6698 - loss: 0.7804

<div class="k-default-codeblock">
```

```
</div>
   7/183 [37m━━━━━━━━━━━━━━━━━━━━  40:09 14s/step - accuracy: 0.6761 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
   8/183 [37m━━━━━━━━━━━━━━━━━━━━  39:59 14s/step - accuracy: 0.6834 - loss: 0.7721

<div class="k-default-codeblock">
```

```
</div>
   9/183 [37m━━━━━━━━━━━━━━━━━━━━  39:46 14s/step - accuracy: 0.6908 - loss: 0.7682

<div class="k-default-codeblock">
```

```
</div>
  10/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:33 14s/step - accuracy: 0.6967 - loss: 0.7655

<div class="k-default-codeblock">
```

```
</div>
  11/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:23 14s/step - accuracy: 0.7005 - loss: 0.7653

<div class="k-default-codeblock">
```

```
</div>
  12/183 ━[37m━━━━━━━━━━━━━━━━━━━  39:08 14s/step - accuracy: 0.7021 - loss: 0.7669

<div class="k-default-codeblock">
```

```
</div>
  13/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:57 14s/step - accuracy: 0.7028 - loss: 0.7679

<div class="k-default-codeblock">
```

```
</div>
  14/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:44 14s/step - accuracy: 0.7043 - loss: 0.7668

<div class="k-default-codeblock">
```

```
</div>
  15/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:29 14s/step - accuracy: 0.7062 - loss: 0.7654

<div class="k-default-codeblock">
```

```
</div>
  16/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:16 14s/step - accuracy: 0.7089 - loss: 0.7633

<div class="k-default-codeblock">
```

```
</div>
  17/183 ━[37m━━━━━━━━━━━━━━━━━━━  38:00 14s/step - accuracy: 0.7113 - loss: 0.7622

<div class="k-default-codeblock">
```

```
</div>
  18/183 ━[37m━━━━━━━━━━━━━━━━━━━  37:49 14s/step - accuracy: 0.7127 - loss: 0.7620

<div class="k-default-codeblock">
```

```
</div>
  19/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:34 14s/step - accuracy: 0.7133 - loss: 0.7624

<div class="k-default-codeblock">
```

```
</div>
  20/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:21 14s/step - accuracy: 0.7142 - loss: 0.7625

<div class="k-default-codeblock">
```

```
</div>
  21/183 ━━[37m━━━━━━━━━━━━━━━━━━  37:08 14s/step - accuracy: 0.7148 - loss: 0.7631

<div class="k-default-codeblock">
```

```
</div>
  22/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:54 14s/step - accuracy: 0.7151 - loss: 0.7641

<div class="k-default-codeblock">
```

```
</div>
  23/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:40 14s/step - accuracy: 0.7152 - loss: 0.7652

<div class="k-default-codeblock">
```

```
</div>
  24/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:26 14s/step - accuracy: 0.7153 - loss: 0.7664

<div class="k-default-codeblock">
```

```
</div>
  25/183 ━━[37m━━━━━━━━━━━━━━━━━━  36:11 14s/step - accuracy: 0.7151 - loss: 0.7678

<div class="k-default-codeblock">
```

```
</div>
  26/183 ━━[37m━━━━━━━━━━━━━━━━━━  35:57 14s/step - accuracy: 0.7152 - loss: 0.7691

<div class="k-default-codeblock">
```

```
</div>
  27/183 ━━[37m━━━━━━━━━━━━━━━━━━  35:43 14s/step - accuracy: 0.7154 - loss: 0.7699

<div class="k-default-codeblock">
```

```
</div>
  28/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:29 14s/step - accuracy: 0.7155 - loss: 0.7706

<div class="k-default-codeblock">
```

```
</div>
  29/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:15 14s/step - accuracy: 0.7160 - loss: 0.7709

<div class="k-default-codeblock">
```

```
</div>
  30/183 ━━━[37m━━━━━━━━━━━━━━━━━  35:02 14s/step - accuracy: 0.7163 - loss: 0.7713

<div class="k-default-codeblock">
```

```
</div>
  31/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:48 14s/step - accuracy: 0.7165 - loss: 0.7717

<div class="k-default-codeblock">
```

```
</div>
  32/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:35 14s/step - accuracy: 0.7169 - loss: 0.7719

<div class="k-default-codeblock">
```

```
</div>
  33/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:21 14s/step - accuracy: 0.7174 - loss: 0.7720

<div class="k-default-codeblock">
```

```
</div>
  34/183 ━━━[37m━━━━━━━━━━━━━━━━━  34:07 14s/step - accuracy: 0.7180 - loss: 0.7720

<div class="k-default-codeblock">
```

```
</div>
  35/183 ━━━[37m━━━━━━━━━━━━━━━━━  33:54 14s/step - accuracy: 0.7184 - loss: 0.7721

<div class="k-default-codeblock">
```

```
</div>
  36/183 ━━━[37m━━━━━━━━━━━━━━━━━  33:40 14s/step - accuracy: 0.7190 - loss: 0.7719

<div class="k-default-codeblock">
```

```
</div>
  37/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:27 14s/step - accuracy: 0.7195 - loss: 0.7718

<div class="k-default-codeblock">
```

```
</div>
  38/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:14 14s/step - accuracy: 0.7201 - loss: 0.7715

<div class="k-default-codeblock">
```

```
</div>
  39/183 ━━━━[37m━━━━━━━━━━━━━━━━  33:00 14s/step - accuracy: 0.7206 - loss: 0.7712

<div class="k-default-codeblock">
```

```
</div>
  40/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:46 14s/step - accuracy: 0.7211 - loss: 0.7708

<div class="k-default-codeblock">
```

```
</div>
  41/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:32 14s/step - accuracy: 0.7217 - loss: 0.7702

<div class="k-default-codeblock">
```

```
</div>
  42/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:17 14s/step - accuracy: 0.7223 - loss: 0.7697

<div class="k-default-codeblock">
```

```
</div>
  43/183 ━━━━[37m━━━━━━━━━━━━━━━━  32:03 14s/step - accuracy: 0.7229 - loss: 0.7691

<div class="k-default-codeblock">
```

```
</div>
  44/183 ━━━━[37m━━━━━━━━━━━━━━━━  31:50 14s/step - accuracy: 0.7233 - loss: 0.7687

<div class="k-default-codeblock">
```

```
</div>
  45/183 ━━━━[37m━━━━━━━━━━━━━━━━  31:35 14s/step - accuracy: 0.7237 - loss: 0.7684

<div class="k-default-codeblock">
```

```
</div>
  46/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:22 14s/step - accuracy: 0.7238 - loss: 0.7683

<div class="k-default-codeblock">
```

```
</div>
  47/183 ━━━━━[37m━━━━━━━━━━━━━━━  31:08 14s/step - accuracy: 0.7240 - loss: 0.7681

<div class="k-default-codeblock">
```

```
</div>
  48/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:54 14s/step - accuracy: 0.7242 - loss: 0.7679

<div class="k-default-codeblock">
```

```
</div>
  49/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:40 14s/step - accuracy: 0.7243 - loss: 0.7679

<div class="k-default-codeblock">
```

```
</div>
  50/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:26 14s/step - accuracy: 0.7244 - loss: 0.7678

<div class="k-default-codeblock">
```

```
</div>
  51/183 ━━━━━[37m━━━━━━━━━━━━━━━  30:12 14s/step - accuracy: 0.7244 - loss: 0.7678

<div class="k-default-codeblock">
```

```
</div>
  52/183 ━━━━━[37m━━━━━━━━━━━━━━━  29:58 14s/step - accuracy: 0.7244 - loss: 0.7678

<div class="k-default-codeblock">
```

```
</div>
  53/183 ━━━━━[37m━━━━━━━━━━━━━━━  29:44 14s/step - accuracy: 0.7244 - loss: 0.7678

<div class="k-default-codeblock">
```

```
</div>
  54/183 ━━━━━[37m━━━━━━━━━━━━━━━  29:31 14s/step - accuracy: 0.7243 - loss: 0.7677

<div class="k-default-codeblock">
```

```
</div>
  55/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:17 14s/step - accuracy: 0.7243 - loss: 0.7677

<div class="k-default-codeblock">
```

```
</div>
  56/183 ━━━━━━[37m━━━━━━━━━━━━━━  29:03 14s/step - accuracy: 0.7242 - loss: 0.7678

<div class="k-default-codeblock">
```

```
</div>
  57/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:49 14s/step - accuracy: 0.7242 - loss: 0.7678

<div class="k-default-codeblock">
```

```
</div>
  58/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:35 14s/step - accuracy: 0.7242 - loss: 0.7679

<div class="k-default-codeblock">
```

```
</div>
  59/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:21 14s/step - accuracy: 0.7241 - loss: 0.7680

<div class="k-default-codeblock">
```

```
</div>
  60/183 ━━━━━━[37m━━━━━━━━━━━━━━  28:07 14s/step - accuracy: 0.7241 - loss: 0.7681

<div class="k-default-codeblock">
```

```
</div>
  61/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:54 14s/step - accuracy: 0.7241 - loss: 0.7682

<div class="k-default-codeblock">
```

```
</div>
  62/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:40 14s/step - accuracy: 0.7240 - loss: 0.7684

<div class="k-default-codeblock">
```

```
</div>
  63/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:26 14s/step - accuracy: 0.7238 - loss: 0.7687

<div class="k-default-codeblock">
```

```
</div>
  64/183 ━━━━━━[37m━━━━━━━━━━━━━━  27:12 14s/step - accuracy: 0.7238 - loss: 0.7689

<div class="k-default-codeblock">
```

```
</div>
  65/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:58 14s/step - accuracy: 0.7238 - loss: 0.7691

<div class="k-default-codeblock">
```

```
</div>
  66/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:44 14s/step - accuracy: 0.7237 - loss: 0.7694

<div class="k-default-codeblock">
```

```
</div>
  67/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:31 14s/step - accuracy: 0.7237 - loss: 0.7697

<div class="k-default-codeblock">
```

```
</div>
  68/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:17 14s/step - accuracy: 0.7237 - loss: 0.7699

<div class="k-default-codeblock">
```

```
</div>
  69/183 ━━━━━━━[37m━━━━━━━━━━━━━  26:03 14s/step - accuracy: 0.7236 - loss: 0.7701

<div class="k-default-codeblock">
```

```
</div>
  70/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:49 14s/step - accuracy: 0.7236 - loss: 0.7704

<div class="k-default-codeblock">
```

```
</div>
  71/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:35 14s/step - accuracy: 0.7235 - loss: 0.7705

<div class="k-default-codeblock">
```

```
</div>
  72/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:21 14s/step - accuracy: 0.7235 - loss: 0.7706

<div class="k-default-codeblock">
```

```
</div>
  73/183 ━━━━━━━[37m━━━━━━━━━━━━━  25:07 14s/step - accuracy: 0.7235 - loss: 0.7708

<div class="k-default-codeblock">
```

```
</div>
  74/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:54 14s/step - accuracy: 0.7235 - loss: 0.7710

<div class="k-default-codeblock">
```

```
</div>
  75/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:40 14s/step - accuracy: 0.7235 - loss: 0.7711

<div class="k-default-codeblock">
```

```
</div>
  76/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:26 14s/step - accuracy: 0.7235 - loss: 0.7713

<div class="k-default-codeblock">
```

```
</div>
  77/183 ━━━━━━━━[37m━━━━━━━━━━━━  24:12 14s/step - accuracy: 0.7234 - loss: 0.7714

<div class="k-default-codeblock">
```

```
</div>
  78/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:59 14s/step - accuracy: 0.7234 - loss: 0.7716

<div class="k-default-codeblock">
```

```
</div>
  79/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:45 14s/step - accuracy: 0.7233 - loss: 0.7717

<div class="k-default-codeblock">
```

```
</div>
  80/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:31 14s/step - accuracy: 0.7233 - loss: 0.7718

<div class="k-default-codeblock">
```

```
</div>
  81/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:17 14s/step - accuracy: 0.7232 - loss: 0.7719

<div class="k-default-codeblock">
```

```
</div>
  82/183 ━━━━━━━━[37m━━━━━━━━━━━━  23:04 14s/step - accuracy: 0.7232 - loss: 0.7721

<div class="k-default-codeblock">
```

```
</div>
  83/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:50 14s/step - accuracy: 0.7231 - loss: 0.7722

<div class="k-default-codeblock">
```

```
</div>
  84/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:36 14s/step - accuracy: 0.7231 - loss: 0.7723

<div class="k-default-codeblock">
```

```
</div>
  85/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:22 14s/step - accuracy: 0.7231 - loss: 0.7724

<div class="k-default-codeblock">
```

```
</div>
  86/183 ━━━━━━━━━[37m━━━━━━━━━━━  22:08 14s/step - accuracy: 0.7230 - loss: 0.7725

<div class="k-default-codeblock">
```

```
</div>
  87/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:55 14s/step - accuracy: 0.7230 - loss: 0.7726

<div class="k-default-codeblock">
```

```
</div>
  88/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:41 14s/step - accuracy: 0.7230 - loss: 0.7727

<div class="k-default-codeblock">
```

```
</div>
  89/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:27 14s/step - accuracy: 0.7229 - loss: 0.7728

<div class="k-default-codeblock">
```

```
</div>
  90/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:13 14s/step - accuracy: 0.7229 - loss: 0.7729

<div class="k-default-codeblock">
```

```
</div>
  91/183 ━━━━━━━━━[37m━━━━━━━━━━━  21:00 14s/step - accuracy: 0.7229 - loss: 0.7731

<div class="k-default-codeblock">
```

```
</div>
  92/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:46 14s/step - accuracy: 0.7228 - loss: 0.7733

<div class="k-default-codeblock">
```

```
</div>
  93/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:32 14s/step - accuracy: 0.7228 - loss: 0.7734

<div class="k-default-codeblock">
```

```
</div>
  94/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:18 14s/step - accuracy: 0.7229 - loss: 0.7735

<div class="k-default-codeblock">
```

```
</div>
  95/183 ━━━━━━━━━━[37m━━━━━━━━━━  20:05 14s/step - accuracy: 0.7229 - loss: 0.7736

<div class="k-default-codeblock">
```

```
</div>
  96/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:51 14s/step - accuracy: 0.7229 - loss: 0.7737

<div class="k-default-codeblock">
```

```
</div>
  97/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:37 14s/step - accuracy: 0.7229 - loss: 0.7739

<div class="k-default-codeblock">
```

```
</div>
  98/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:23 14s/step - accuracy: 0.7230 - loss: 0.7740

<div class="k-default-codeblock">
```

```
</div>
  99/183 ━━━━━━━━━━[37m━━━━━━━━━━  19:10 14s/step - accuracy: 0.7230 - loss: 0.7742

<div class="k-default-codeblock">
```

```
</div>
 100/183 ━━━━━━━━━━[37m━━━━━━━━━━  18:56 14s/step - accuracy: 0.7230 - loss: 0.7744

<div class="k-default-codeblock">
```

```
</div>
 101/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:42 14s/step - accuracy: 0.7230 - loss: 0.7745

<div class="k-default-codeblock">
```

```
</div>
 102/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:28 14s/step - accuracy: 0.7231 - loss: 0.7747

<div class="k-default-codeblock">
```

```
</div>
 103/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:15 14s/step - accuracy: 0.7231 - loss: 0.7748

<div class="k-default-codeblock">
```

```
</div>
 104/183 ━━━━━━━━━━━[37m━━━━━━━━━  18:01 14s/step - accuracy: 0.7231 - loss: 0.7750

<div class="k-default-codeblock">
```

```
</div>
 105/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:47 14s/step - accuracy: 0.7231 - loss: 0.7752

<div class="k-default-codeblock">
```

```
</div>
 106/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:33 14s/step - accuracy: 0.7231 - loss: 0.7753

<div class="k-default-codeblock">
```

```
</div>
 107/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:20 14s/step - accuracy: 0.7231 - loss: 0.7754

<div class="k-default-codeblock">
```

```
</div>
 108/183 ━━━━━━━━━━━[37m━━━━━━━━━  17:06 14s/step - accuracy: 0.7231 - loss: 0.7755

<div class="k-default-codeblock">
```

```
</div>
 109/183 ━━━━━━━━━━━[37m━━━━━━━━━  16:52 14s/step - accuracy: 0.7231 - loss: 0.7757

<div class="k-default-codeblock">
```

```
</div>
 110/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:39 14s/step - accuracy: 0.7231 - loss: 0.7758

<div class="k-default-codeblock">
```

```
</div>
 111/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:25 14s/step - accuracy: 0.7231 - loss: 0.7759

<div class="k-default-codeblock">
```

```
</div>
 112/183 ━━━━━━━━━━━━[37m━━━━━━━━  16:11 14s/step - accuracy: 0.7231 - loss: 0.7760

<div class="k-default-codeblock">
```

```
</div>
 113/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:58 14s/step - accuracy: 0.7231 - loss: 0.7761

<div class="k-default-codeblock">
```

```
</div>
 114/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:44 14s/step - accuracy: 0.7230 - loss: 0.7763

<div class="k-default-codeblock">
```

```
</div>
 115/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:30 14s/step - accuracy: 0.7230 - loss: 0.7764

<div class="k-default-codeblock">
```

```
</div>
 116/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:16 14s/step - accuracy: 0.7230 - loss: 0.7765

<div class="k-default-codeblock">
```

```
</div>
 117/183 ━━━━━━━━━━━━[37m━━━━━━━━  15:03 14s/step - accuracy: 0.7230 - loss: 0.7767

<div class="k-default-codeblock">
```

```
</div>
 118/183 ━━━━━━━━━━━━[37m━━━━━━━━  14:49 14s/step - accuracy: 0.7230 - loss: 0.7768

<div class="k-default-codeblock">
```

```
</div>
 119/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:35 14s/step - accuracy: 0.7230 - loss: 0.7769

<div class="k-default-codeblock">
```

```
</div>
 120/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:22 14s/step - accuracy: 0.7230 - loss: 0.7770

<div class="k-default-codeblock">
```

```
</div>
 121/183 ━━━━━━━━━━━━━[37m━━━━━━━  14:08 14s/step - accuracy: 0.7230 - loss: 0.7771

<div class="k-default-codeblock">
```

```
</div>
 122/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:54 14s/step - accuracy: 0.7231 - loss: 0.7771

<div class="k-default-codeblock">
```

```
</div>
 123/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:40 14s/step - accuracy: 0.7230 - loss: 0.7772

<div class="k-default-codeblock">
```

```
</div>
 124/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:27 14s/step - accuracy: 0.7230 - loss: 0.7773

<div class="k-default-codeblock">
```

```
</div>
 125/183 ━━━━━━━━━━━━━[37m━━━━━━━  13:13 14s/step - accuracy: 0.7230 - loss: 0.7773

<div class="k-default-codeblock">
```

```
</div>
 126/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:59 14s/step - accuracy: 0.7230 - loss: 0.7774

<div class="k-default-codeblock">
```

```
</div>
 127/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:46 14s/step - accuracy: 0.7230 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 128/183 ━━━━━━━━━━━━━[37m━━━━━━━  12:32 14s/step - accuracy: 0.7230 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 129/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:18 14s/step - accuracy: 0.7231 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 130/183 ━━━━━━━━━━━━━━[37m━━━━━━  12:05 14s/step - accuracy: 0.7231 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 131/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:51 14s/step - accuracy: 0.7231 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 132/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:37 14s/step - accuracy: 0.7231 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 133/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:24 14s/step - accuracy: 0.7232 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 134/183 ━━━━━━━━━━━━━━[37m━━━━━━  11:10 14s/step - accuracy: 0.7232 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 135/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:56 14s/step - accuracy: 0.7232 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 136/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:42 14s/step - accuracy: 0.7233 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 137/183 ━━━━━━━━━━━━━━[37m━━━━━━  10:29 14s/step - accuracy: 0.7233 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 138/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:15 14s/step - accuracy: 0.7233 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 139/183 ━━━━━━━━━━━━━━━[37m━━━━━  10:01 14s/step - accuracy: 0.7233 - loss: 0.7775

<div class="k-default-codeblock">
```

```
</div>
 140/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:48 14s/step - accuracy: 0.7233 - loss: 0.7776 

<div class="k-default-codeblock">
```

```
</div>
 141/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:34 14s/step - accuracy: 0.7233 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 142/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:20 14s/step - accuracy: 0.7233 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 143/183 ━━━━━━━━━━━━━━━[37m━━━━━  9:07 14s/step - accuracy: 0.7233 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 144/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:53 14s/step - accuracy: 0.7233 - loss: 0.7776

<div class="k-default-codeblock">
```

```
</div>
 145/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:39 14s/step - accuracy: 0.7232 - loss: 0.7777

<div class="k-default-codeblock">
```

```
</div>
 146/183 ━━━━━━━━━━━━━━━[37m━━━━━  8:26 14s/step - accuracy: 0.7232 - loss: 0.7777

<div class="k-default-codeblock">
```

```
</div>
 147/183 ━━━━━━━━━━━━━━━━[37m━━━━  8:12 14s/step - accuracy: 0.7232 - loss: 0.7778

<div class="k-default-codeblock">
```

```
</div>
 148/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:58 14s/step - accuracy: 0.7231 - loss: 0.7778

<div class="k-default-codeblock">
```

```
</div>
 149/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:44 14s/step - accuracy: 0.7231 - loss: 0.7778

<div class="k-default-codeblock">
```

```
</div>
 150/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:31 14s/step - accuracy: 0.7230 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 151/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:17 14s/step - accuracy: 0.7230 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 152/183 ━━━━━━━━━━━━━━━━[37m━━━━  7:03 14s/step - accuracy: 0.7230 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 153/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:50 14s/step - accuracy: 0.7229 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 154/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:36 14s/step - accuracy: 0.7229 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 155/183 ━━━━━━━━━━━━━━━━[37m━━━━  6:22 14s/step - accuracy: 0.7228 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 156/183 ━━━━━━━━━━━━━━━━━[37m━━━  6:09 14s/step - accuracy: 0.7228 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 157/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:55 14s/step - accuracy: 0.7228 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 158/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:41 14s/step - accuracy: 0.7228 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 159/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:28 14s/step - accuracy: 0.7227 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 160/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:14 14s/step - accuracy: 0.7227 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 161/183 ━━━━━━━━━━━━━━━━━[37m━━━  5:00 14s/step - accuracy: 0.7226 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 162/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:47 14s/step - accuracy: 0.7226 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 163/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:33 14s/step - accuracy: 0.7226 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 164/183 ━━━━━━━━━━━━━━━━━[37m━━━  4:19 14s/step - accuracy: 0.7225 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 165/183 ━━━━━━━━━━━━━━━━━━[37m━━  4:06 14s/step - accuracy: 0.7225 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 166/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:52 14s/step - accuracy: 0.7225 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 167/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:38 14s/step - accuracy: 0.7224 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 168/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:25 14s/step - accuracy: 0.7224 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 169/183 ━━━━━━━━━━━━━━━━━━[37m━━  3:11 14s/step - accuracy: 0.7224 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 170/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:57 14s/step - accuracy: 0.7223 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 171/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:44 14s/step - accuracy: 0.7223 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 172/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:30 14s/step - accuracy: 0.7223 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 173/183 ━━━━━━━━━━━━━━━━━━[37m━━  2:16 14s/step - accuracy: 0.7222 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 174/183 ━━━━━━━━━━━━━━━━━━━[37m━  2:03 14s/step - accuracy: 0.7222 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 175/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:49 14s/step - accuracy: 0.7222 - loss: 0.7780

<div class="k-default-codeblock">
```

```
</div>
 176/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:35 14s/step - accuracy: 0.7221 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 177/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:22 14s/step - accuracy: 0.7221 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 178/183 ━━━━━━━━━━━━━━━━━━━[37m━  1:08 14s/step - accuracy: 0.7221 - loss: 0.7779

<div class="k-default-codeblock">
```

```
</div>
 179/183 ━━━━━━━━━━━━━━━━━━━[37m━  54s 14s/step - accuracy: 0.7221 - loss: 0.7779 

<div class="k-default-codeblock">
```

```
</div>
 180/183 ━━━━━━━━━━━━━━━━━━━[37m━  41s 14s/step - accuracy: 0.7221 - loss: 0.7778

<div class="k-default-codeblock">
```

```
</div>
 181/183 ━━━━━━━━━━━━━━━━━━━[37m━  27s 14s/step - accuracy: 0.7220 - loss: 0.7778

<div class="k-default-codeblock">
```

```
</div>
 182/183 ━━━━━━━━━━━━━━━━━━━[37m━  13s 14s/step - accuracy: 0.7220 - loss: 0.7778

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 0s 14s/step - accuracy: 0.7220 - loss: 0.7778 

<div class="k-default-codeblock">
```

```
</div>
 183/183 ━━━━━━━━━━━━━━━━━━━━ 2649s 14s/step - accuracy: 0.7220 - loss: 0.7777 - val_accuracy: 0.7650 - val_loss: 0.7110 - learning_rate: 4.4984e-06


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
