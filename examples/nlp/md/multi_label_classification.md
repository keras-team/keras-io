# Large-scale multi-label text classification

**Author:** [Sayak Paul](https://twitter.com/RisingSayak), [Soumik Rakshit](https://github.com/soumik12345)<br>
**Date created:** 2020/09/25<br>
**Last modified:** 2020/09/26<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/multi_label_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/multi_label_classification.py)


**Description:** Implementing a large-scale multi-label text classification model.

---
## Introduction

In this example, we will build a multi-label text classifier to predict the subject areas
of arXiv papers from their abstract bodies. This type of classifier can be useful for
conference submission portals like [OpenReview](https://openreview.net/). Given a paper
abstract, the portal could provide suggestions for which areas the paper would
best belong to.

The dataset was collected using the
[`arXiv` Python library](https://github.com/lukasschwab/arxiv.py)
that provides a wrapper around the
[original arXiv API](http://arxiv.org/help/api/index).
To learn more about the data collection process, please refer to
[this notebook](https://github.com/soumik12345/multi-label-text-classification/blob/master/arxiv_scrape.ipynb).
Additionally, you can also find the dataset on
[Kaggle](https://www.kaggle.com/spsayakpaul/arxiv-paper-abstracts).

---
## Imports


```python
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

---
## Perform exploratory data analysis

In this section, we first load the dataset into a `pandas` dataframe and then perform
some basic exploratory data analysis (EDA).


```python
arxiv_data = pd.read_csv(
    "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv"
)
arxiv_data.head()
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
      <th>titles</th>
      <th>summaries</th>
      <th>terms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Survey on Semantic Stereo Matching / Semantic ...</td>
      <td>Stereo matching is one of the widely used tech...</td>
      <td>['cs.CV', 'cs.LG']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FUTURE-AI: Guiding Principles and Consensus Re...</td>
      <td>The recent advancements in artificial intellig...</td>
      <td>['cs.CV', 'cs.AI', 'cs.LG']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Enforcing Mutual Consistency of Hard Regions f...</td>
      <td>In this paper, we proposed a novel mutual cons...</td>
      <td>['cs.CV', 'cs.AI']</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Parameter Decoupling Strategy for Semi-supervi...</td>
      <td>Consistency training has proven to be an advan...</td>
      <td>['cs.CV']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Background-Foreground Segmentation for Interio...</td>
      <td>To ensure safety in automated driving, the cor...</td>
      <td>['cs.CV', 'cs.LG']</td>
    </tr>
  </tbody>
</table>
</div>



Our text features are present in the `summaries` column and their corresponding labels
are in `terms`. As you can notice, there are multiple categories associated with a
particular entry.


```python
print(f"There are {len(arxiv_data)} rows in the dataset.")
```

<div class="k-default-codeblock">
```
There are 51774 rows in the dataset.

```
</div>
Real-world data is noisy. One of the most commonly observed source of noise is data
duplication. Here we notice that our initial dataset has got about 13k duplicate entries.


```python
total_duplicate_titles = sum(arxiv_data["titles"].duplicated())
print(f"There are {total_duplicate_titles} duplicate titles.")
```

<div class="k-default-codeblock">
```
There are 12802 duplicate titles.

```
</div>
Before proceeding further, we drop these entries.


```python
arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]
print(f"There are {len(arxiv_data)} rows in the deduplicated dataset.")

# There are some terms with occurrence as low as 1.
print(sum(arxiv_data["terms"].value_counts() == 1))

# How many unique terms?
print(arxiv_data["terms"].nunique())
```

<div class="k-default-codeblock">
```
There are 38972 rows in the deduplicated dataset.
2321
3157

```
</div>
As observed above, out of 3,157 unique combinations of `terms`, 2,321 entries have the
lowest occurrence. To prepare our train, validation, and test sets with
[stratification](https://en.wikipedia.org/wiki/Stratified_sampling), we need to drop
these terms.


```python
# Filtering the rare terms.
arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
arxiv_data_filtered.shape
```




<div class="k-default-codeblock">
```
(36651, 3)

```
</div>
---
## Convert the string labels to lists of strings

The initial labels are represented as raw strings. Here we make them `List[str]` for a
more compact representation.


```python
arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(
    lambda x: literal_eval(x)
)
arxiv_data_filtered["terms"].values[:5]
```




<div class="k-default-codeblock">
```
array([list(['cs.CV', 'cs.LG']), list(['cs.CV', 'cs.AI', 'cs.LG']),
       list(['cs.CV', 'cs.AI']), list(['cs.CV']),
       list(['cs.CV', 'cs.LG'])], dtype=object)

```
</div>
---
## Use stratified splits because of class imbalance

The dataset has a
[class imbalance problem](https://developers.google.com/machine-learning/glossary/#class-imbalanced-dataset).
So, to have a fair evaluation result, we need to ensure the datasets are sampled with
stratification. To know more about different strategies to deal with the class imbalance
problem, you can follow
[this tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
For an end-to-end demonstration of classification with imbablanced data, refer to
[Imbalanced classification: credit card fraud detection](https://keras.io/examples/structured_data/imbalanced_classification/).


```python
test_split = 0.1

# Initial train and test split.
train_df, test_df = train_test_split(
    arxiv_data_filtered,
    test_size=test_split,
    stratify=arxiv_data_filtered["terms"].values,
)

# Splitting the test set further into validation
# and new test sets.
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)

print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")
```

<div class="k-default-codeblock">
```
Number of rows in training set: 32985
Number of rows in validation set: 1833
Number of rows in test set: 1833

```
</div>
---
## Multi-label binarization

Now we preprocess our labels using the
[`StringLookup`](https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup)
layer.


```python
terms = tf.ragged.constant(train_df["terms"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()


def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)


print("Vocabulary:\n")
print(vocab)

```

<div class="k-default-codeblock">
```
Vocabulary:
```
</div>
    
<div class="k-default-codeblock">
```
['[UNK]', 'cs.CV', 'cs.LG', 'stat.ML', 'cs.AI', 'eess.IV', 'cs.RO', 'cs.CL', 'cs.NE', 'cs.CR', 'math.OC', 'eess.SP', 'cs.GR', 'cs.SI', 'cs.MM', 'cs.SY', 'cs.IR', 'cs.MA', 'eess.SY', 'cs.HC', 'math.IT', 'cs.IT', 'cs.DC', 'cs.CY', 'stat.AP', 'stat.TH', 'math.ST', 'stat.ME', 'eess.AS', 'cs.SD', 'q-bio.QM', 'q-bio.NC', 'cs.DS', 'cs.GT', 'cs.CG', 'cs.SE', 'cs.NI', 'I.2.6', 'stat.CO', 'math.NA', 'cs.NA', 'physics.chem-ph', 'cs.DB', 'q-bio.BM', 'cs.LO', 'cond-mat.dis-nn', '68T45', 'math.PR', 'cs.PL', 'physics.comp-ph', 'cs.CE', 'cs.AR', 'I.2.10', 'q-fin.ST', 'cond-mat.stat-mech', '68T05', 'math.DS', 'cs.CC', 'quant-ph', 'physics.data-an', 'I.4.6', 'physics.soc-ph', 'physics.ao-ph', 'econ.EM', 'cs.DM', 'q-bio.GN', 'physics.med-ph', 'cs.PF', 'astro-ph.IM', 'I.4.8', 'math.AT', 'I.4', 'q-fin.TR', 'cs.FL', 'I.5.4', 'I.2', '68U10', 'hep-ex', 'cond-mat.mtrl-sci', '68T10', 'physics.optics', 'physics.geo-ph', 'physics.flu-dyn', 'math.CO', 'math.AP', 'I.4; I.5', 'I.4.9', 'I.2.6; I.2.8', '68T01', '65D19', 'q-fin.CP', 'nlin.CD', 'cs.MS', 'I.2.6; I.5.1', 'I.2.10; I.4; I.5', 'I.2.0; I.2.6', '68T07', 'cs.SC', 'cs.ET', 'K.3.2', 'I.2; I.5', 'I.2.8', '68U01', '68T30', 'q-fin.GN', 'q-fin.EC', 'q-bio.MN', 'econ.GN', 'I.4.9; I.5.4', 'I.4.5', 'I.2; I.4; I.5', 'I.2.6; I.2.7', 'I.2.10; I.4.8', '68T99', '68Q32', '68', '62H30', 'q-fin.RM', 'q-fin.PM', 'q-bio.TO', 'q-bio.OT', 'physics.bio-ph', 'nlin.AO', 'math.LO', 'math.FA', 'hep-ph', 'cond-mat.soft', 'I.4.6; I.4.8', 'I.4.4', 'I.4.3', 'I.4.0', 'I.2; J.2', 'I.2; I.2.6; I.2.7', 'I.2.7', 'I.2.6; I.5.4', 'I.2.6; I.2.9', 'I.2.6; I.2.7; H.3.1; H.3.3', 'I.2.6; I.2.10', 'I.2.6, I.5.4', 'I.2.1; J.3', 'I.2.10; I.5.1; I.4.8', 'I.2.10; I.4.8; I.5.4', 'I.2.10; I.2.6', 'I.2.1', 'H.3.1; I.2.6; I.2.7', 'H.3.1; H.3.3; I.2.6; I.2.7', 'G.3', 'F.2.2; I.2.7', 'E.5; E.4; E.2; H.1.1; F.1.1; F.1.3', '68Txx', '62H99', '62H35', '14J60 (Primary) 14F05, 14J26 (Secondary)']

```
</div>
Here we are separating the individual unique classes available from the label
pool and then using this information to represent a given label set with 0's and 1's.
Below is an example.


```python
sample_label = train_df["terms"].iloc[0]
print(f"Original label: {sample_label}")

label_binarized = lookup([sample_label])
print(f"Label-binarized representation: {label_binarized}")
```

<div class="k-default-codeblock">
```
Original label: ['cs.LG', 'cs.CY']
Label-binarized representation: [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0.]]

```
</div>
---
## Data preprocessing and `tf.data.Dataset` objects

We first get percentile estimates of the sequence lengths. The purpose will be clear in a
moment.


```python
train_df["summaries"].apply(lambda x: len(x.split(" "))).describe()
```




<div class="k-default-codeblock">
```
count    32985.000000
mean       156.419706
std         41.528906
min          5.000000
25%        128.000000
50%        154.000000
75%        183.000000
max        462.000000
Name: summaries, dtype: float64

```
</div>
Notice that 50% of the abstracts have a length of 154 (you may get a different number
based on the split). So, any number close to that value is a good enough approximate for the
maximum sequence length.

Now, we implement utilities to prepare our datasets that would go straight to the text
classifier model.


```python
max_seqlen = 150
batch_size = 128
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


def unify_text_length(text, label):
    # Split the given abstract and calculate its length.
    word_splits = tf.strings.split(text, sep=" ")
    sequence_length = tf.shape(word_splits)[0]

    # Calculate the padding amount.
    padding_amount = max_seqlen - sequence_length

    # Check if we need to pad or truncate.
    if padding_amount > 0:
        unified_text = tf.pad([text], [[0, padding_amount]], constant_values="<pad>")
        unified_text = tf.strings.reduce_join(unified_text, separator="")
    else:
        unified_text = tf.strings.reduce_join(word_splits[:max_seqlen], separator=" ")

    # The expansion is needed for subsequent vectorization.
    return tf.expand_dims(unified_text, -1), label


def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["terms"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    dataset = dataset.map(unify_text_length, num_parallel_calls=auto).cache()
    return dataset.batch(batch_size)

```

Now we can prepare the `tf.data.Dataset` objects.


```python
train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)
```

---
## Dataset preview


```python
text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text[0]}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    print(" ")
```

<div class="k-default-codeblock">
```
Abstract: b'For the integration of renewable energy sources, power grid operators need\nrealistic information about the effects of energy production and consumption to\nassess grid stability.\n  Recently, research in scenario planning benefits from utilizing generative\nadversarial networks (GANs) as generative models for operational scenario\nplanning.\n  In these scenarios, operators examine temporal as well as spatial influences\nof different energy sources on the grid.\n  The analysis of how renewable energy resources affect the grid enables the\noperators to evaluate the stability and to identify potential weak points such\nas a limiting transformer.\n  However, due to their novelty, there are limited studies on how well GANs\nmodel the underlying power distribution.\n  This analysis is essential because, e.g., especially extreme situations with\nlow or high power generation are required to evaluate grid stability.\n  We conduct a comparative study of the Wasserstein distance,\nbinary-cross-entropy loss, and a Gaussian copula as the baseline applied on two\nwind and two solar datasets'
Label(s): ['cs.LG' 'eess.SP']
 
Abstract: b'We study the optimization problem for decomposing $d$ dimensional\nfourth-order Tensors with $k$ non-orthogonal components. We derive\n\\textit{deterministic} conditions under which such a problem does not have\nspurious local minima. In particular, we show that if $\\kappa =\n\\frac{\\lambda_{max}}{\\lambda_{min}} < \\frac{5}{4}$, and incoherence coefficient\nis of the order $O(\\frac{1}{\\sqrt{d}})$, then all the local minima are globally\noptimal. Using standard techniques, these conditions could be easily\ntransformed into conditions that would hold with high probability in high\ndimensions when the components are generated randomly. Finally, we prove that\nthe tensor power method with deflation and restarts could efficiently extract\nall the components within a tolerance level $O(\\kappa \\sqrt{k\\tau^3})$ that\nseems to be the noise floor of non-orthogonal tensor decomposition.<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
Label(s): ['cs.LG']
 
Abstract: b'Explainable Artificial Intelligence (XAI) is an emerging area of research in\nthe field of Artificial Intelligence (AI). XAI can explain how AI obtained a\nparticular solution (e.g., classification or object detection) and can also\nanswer other "wh" questions. This explainability is not possible in traditional\nAI. Explainability is essential for critical applications, such as defense,\nhealth care, law and order, and autonomous driving vehicles, etc, where the\nknow-how is required for trust and transparency. A number of XAI techniques so\nfar have been purposed for such applications. This paper provides an overview\nof these techniques from a multimedia (i.e., text, image, audio, and video)\npoint of view. The advantages and shortcomings of these techniques have been\ndiscussed, and pointers to some future directions have also been provided.<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
Label(s): ['cs.LG' 'cs.AI']
 
Abstract: b'Some of the most important tasks take place in environments which lack cheap\nand perfect simulators, thus hampering the application of model-free\nreinforcement learning (RL). While model-based RL aims to learn a dynamics\nmodel, in a more general case the learner does not know a priori what the\naction space is. Here we propose a formalism where the learner induces a world\nprogram by learning a dynamics model and the actions in graph-based\ncompositional environments by observing state-state transition examples. Then,\nthe learner can perform RL with the world program as the simulator for complex\nplanning tasks. We highlight a recent application, and propose a challenge for\nthe community to assess world program-based planning.<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
Label(s): ['cs.LG' 'stat.ML']
 
Abstract: b'Deep learning based image compression has recently witnessed exciting\nprogress and in some cases even managed to surpass transform coding based\napproaches that have been established and refined over many decades. However,\nstate-of-the-art solutions for deep image compression typically employ\nautoencoders which map the input to a lower dimensional latent space and thus\nirreversibly discard information already before quantization. Due to that, they\ninherently limit the range of quality levels that can be covered. In contrast,\ntraditional approaches in image compression allow for a larger range of quality\nlevels. Interestingly, they employ an invertible transformation before\nperforming the quantization step which explicitly discards information.\nInspired by this, we propose a deep image compression method that is able to go\nfrom low bit-rates to near lossless quality by leveraging normalizing flows to\nlearn a bijective mapping from the image space to a latent representation. In\naddition to this, we demonstrate further advantages unique to our solution,\nsuch as the ability to maintain constant quality results'
Label(s): ['cs.CV']
 

```
</div>
---
## Vectorization

Before we feed the data to our model, we need to vectorize it (represent it in a numerical form).
For that purpose, we will use the
[`TextVectorization` layer](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization).
It can operate as a part of your main model so that the model is excluded from the core
preprocessing logic. This greatly reduces the chances of training / serving skew during inference.

We first calculate the number of unique words present in the abstracts.


```python
train_df["total_words"] = train_df["summaries"].str.split().str.len()
vocabulary_size = train_df["total_words"].max()
print(f"Vocabulary size: {vocabulary_size}")
```

<div class="k-default-codeblock">
```
Vocabulary size: 498

```
</div>
We now create our vectorization layer and `map()` to the `tf.data.Dataset`s created
earlier.


```python
text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

# `TextVectorization` layer needs to be adapted as per the vocabulary from our
# training set.
with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)

```

A batch of raw text will first go through the `TextVectorization` layer and it will
generate their integer representations. Internally, the `TextVectorization` layer will
first create bi-grams out of the sequences and then represent them using
[TF-IDF](https://wikipedia.org/wiki/Tf%E2%80%93idf). The output representations will then
be passed to the shallow model responsible for text classification.

To learn more about other possible configurations with `TextVectorizer`, please consult
the
[official documentation](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization).

**Note**: Setting the `max_tokens` argument to a pre-calculated vocabulary size is
not a requirement.

---
## Create a text classification model

We will keep our model simple -- it will be a small stack of fully-connected layers with
ReLU as the non-linearity.


```python

def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )
    return shallow_mlp_model

```

---
## Train the model

We will train our model using the binary crossentropy loss. This is because the labels
are not disjoint. For a given abstract, we may have multiple categories. So, we will
divide the prediction task into a series of multiple binary classification problems. This
is also why we kept the activation function of the classification layer in our model to
sigmoid. Researchers have used other combinations of loss function and activation
function as well. For example, in
[Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932),
Mahajan et al. used the softmax activation function and cross-entropy loss to train
their models.


```python
epochs = 20

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)

history = shallow_mlp_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("categorical_accuracy")
```

<div class="k-default-codeblock">
```
Epoch 1/20
258/258 [==============================] - 3s 7ms/step - loss: 0.0607 - categorical_accuracy: 0.8037 - val_loss: 0.0226 - val_categorical_accuracy: 0.8767
Epoch 2/20
258/258 [==============================] - 1s 5ms/step - loss: 0.0225 - categorical_accuracy: 0.8726 - val_loss: 0.0213 - val_categorical_accuracy: 0.8871
Epoch 3/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0215 - categorical_accuracy: 0.8750 - val_loss: 0.0210 - val_categorical_accuracy: 0.8893
Epoch 4/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0207 - categorical_accuracy: 0.8794 - val_loss: 0.0209 - val_categorical_accuracy: 0.8860
Epoch 5/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0201 - categorical_accuracy: 0.8823 - val_loss: 0.0208 - val_categorical_accuracy: 0.8882
Epoch 6/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0196 - categorical_accuracy: 0.8857 - val_loss: 0.0203 - val_categorical_accuracy: 0.8925
Epoch 7/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0191 - categorical_accuracy: 0.8876 - val_loss: 0.0196 - val_categorical_accuracy: 0.8914
Epoch 8/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0187 - categorical_accuracy: 0.8900 - val_loss: 0.0195 - val_categorical_accuracy: 0.8729
Epoch 9/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0183 - categorical_accuracy: 0.8919 - val_loss: 0.0193 - val_categorical_accuracy: 0.8800
Epoch 10/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0179 - categorical_accuracy: 0.8932 - val_loss: 0.0190 - val_categorical_accuracy: 0.8958
Epoch 11/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0176 - categorical_accuracy: 0.8950 - val_loss: 0.0192 - val_categorical_accuracy: 0.8974
Epoch 12/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0172 - categorical_accuracy: 0.8967 - val_loss: 0.0191 - val_categorical_accuracy: 0.8936
Epoch 13/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0169 - categorical_accuracy: 0.8980 - val_loss: 0.0192 - val_categorical_accuracy: 0.8920
Epoch 14/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0166 - categorical_accuracy: 0.8993 - val_loss: 0.0194 - val_categorical_accuracy: 0.8811
Epoch 15/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0162 - categorical_accuracy: 0.9008 - val_loss: 0.0196 - val_categorical_accuracy: 0.8822
Epoch 16/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0159 - categorical_accuracy: 0.9032 - val_loss: 0.0196 - val_categorical_accuracy: 0.8794
Epoch 17/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0156 - categorical_accuracy: 0.9047 - val_loss: 0.0197 - val_categorical_accuracy: 0.8652
Epoch 18/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0153 - categorical_accuracy: 0.9061 - val_loss: 0.0198 - val_categorical_accuracy: 0.8718
Epoch 19/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0150 - categorical_accuracy: 0.9067 - val_loss: 0.0200 - val_categorical_accuracy: 0.8734
Epoch 20/20
258/258 [==============================] - 1s 6ms/step - loss: 0.0146 - categorical_accuracy: 0.9087 - val_loss: 0.0202 - val_categorical_accuracy: 0.8691

```
</div>
![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_1.png)



![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_2.png)


While training, we notice an initial sharp fall in the loss followed by a gradual decay.

### Evaluate the model


```python
_, categorical_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")
```

<div class="k-default-codeblock">
```
15/15 [==============================] - 0s 13ms/step - loss: 0.0208 - categorical_accuracy: 0.8642
Categorical accuracy on the test set: 86.42%.

```
</div>
The trained model gives us an evaluation accuracy of ~87%.

---
## Inference

An important feature of the
[preprocessing layers provided by Keras](https://keras.io/guides/preprocessing_layers/)
is that they can be included inside a `tf.keras.Model`. We will export an inference model
by including the `text_vectorization` layer on top of `shallow_mlp_model`. This will
allow our inference model to directly operate on raw strings.

**Note** that during training it is always preferable to use these preprocessing
layers as a part of the data input pipeline rather than the model to avoid
surfacing bottlenecks for the hardware accelerators. This also allows for
asynchronous data processing.


```python
# Create a model for inference.
model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

# Create a small dataset just for demoing inference.
inference_dataset = make_dataset(test_df.sample(100), is_train=False)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)

# Perform inference.
for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text[0]}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    predicted_proba = [proba for proba in predicted_probabilities[i]]
    top_3_labels = [
        x
        for _, x in sorted(
            zip(predicted_probabilities[i], lookup.get_vocabulary()),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ][:3]
    print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])})")
    print(" ")
```

<div class="k-default-codeblock">
```
Abstract: b'Learning interpretable and interpolatable latent representations has been an\nemerging research direction, allowing researchers to understand and utilize the\nderived latent space for further applications such as visual synthesis or\nrecognition. While most existing approaches derive an interpolatable latent\nspace and induces smooth transition in image appearance, it is still not clear\nhow to observe desirable representations which would contain semantic\ninformation of interest. In this paper, we aim to learn meaningful\nrepresentations and simultaneously perform semantic-oriented and\nvisually-smooth interpolation. To this end, we propose an angular\ntriplet-neighbor loss (ATNL) that enables learning a latent representation\nwhose distribution matches the semantic information of interest. With the\nlatent space guided by ATNL, we further utilize spherical semantic\ninterpolation for generating semantic warping of images, allowing synthesis of\ndesirable visual data. Experiments on MNIST and CMU Multi-PIE datasets\nqualitatively and quantitatively verify the effectiveness of our method.<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
Label(s): ['cs.CV']
Predicted Label(s): (cs.CV, cs.LG, stat.ML)
 
Abstract: b'Emergence of artificial intelligence techniques in biomedical applications\nurges the researchers to pay more attention on the uncertainty quantification\n(UQ) in machine-assisted medical decision making. For classification tasks,\nprior studies on UQ are difficult to compare with each other, due to the lack\nof a unified quantitative evaluation metric. Considering that well-performing\nUQ models ought to know when the classification models act incorrectly, we\ndesign a new evaluation metric, area under Confidence-Classification\nCharacteristic curves (AUCCC), to quantitatively evaluate the performance of\nthe UQ models. AUCCC is threshold-free, robust to perturbation, and insensitive\nto the classification performance. We evaluate several UQ methods (e.g., max\nsoftmax output) with AUCCC to validate its effectiveness. Furthermore, a simple\nscheme, named Uncertainty Distillation (UDist), is developed to boost the UQ\nperformance, where a confidence model is distilling the confidence estimated by\ndeep ensembles. The proposed method is easy to implement; it consistently\noutperforms strong baselines on natural and medical image datasets in our\nexperiments.<pad><pad><pad><pad><pad><pad>'
Label(s): ['cs.LG' 'cs.AI']
Predicted Label(s): (cs.LG, cs.CV, stat.ML)
 
Abstract: b'High-dimensional data are ubiquitous in contemporary science and finding\nmethods to compress them is one of the primary goals of machine learning. Given\na dataset lying in a high-dimensional space (in principle hundreds to several\nthousands of dimensions), it is often useful to project it onto a\nlower-dimensional manifold, without loss of information. Identifying the\nminimal dimension of such manifold is a challenging problem known in the\nliterature as intrinsic dimension estimation (IDE). Traditionally, most IDE\nalgorithms are either based on multiscale principal component analysis (PCA) or\non the notion of correlation dimension (and more in general on\nk-nearest-neighbors distances). These methods are affected, in different ways,\nby a severe curse of dimensionality. In particular, none of the existing\nalgorithms can provide accurate ID estimates in the extreme locally\nundersampled regime, i.e. in the limit where the number of samples in any local\npatch of the manifold is less than (or of the same order of) the ID of the\ndataset. Here we introduce'
Label(s): ['cs.LG' 'stat.ML' 'cond-mat.dis-nn']
Predicted Label(s): (cs.LG, stat.ML, stat.TH)
 
Abstract: b'Gradient boosted decision trees (GBDTs) are widely used in machine learning,\nand the output of current GBDT implementations is a single variable. When there\nare multiple outputs, GBDT constructs multiple trees corresponding to the\noutput variables. The correlations between variables are ignored by such a\nstrategy causing redundancy of the learned tree structures. In this paper, we\npropose a general method to learn GBDT for multiple outputs, called GBDT-MO.\nEach leaf of GBDT-MO constructs predictions of all variables or a subset of\nautomatically selected variables. This is achieved by considering the summation\nof objective gains over all output variables. Moreover, we extend histogram\napproximation into multiple output case to speed up the training process.\nVarious experiments on synthetic and real-world datasets verify that GBDT-MO\nachieves outstanding performance in terms of both accuracy and training speed.\nOur codes are available on-line.<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
Label(s): ['cs.CV' 'cs.LG']
Predicted Label(s): (cs.LG, stat.ML, cs.AI)
 
Abstract: b'Image segmentation is an important step in most visual tasks. While\nconvolutional neural networks have shown to perform well on single image\nsegmentation, to our knowledge, no study has been been done on leveraging\nrecurrent gated architectures for video segmentation. Accordingly, we propose a\nnovel method for online segmentation of video sequences that incorporates\ntemporal data. The network is built from fully convolutional element and\nrecurrent unit that works on a sliding window over the temporal data. We also\nintroduce a novel convolutional gated recurrent unit that preserves the spatial\ninformation and reduces the parameters learned. Our method has the advantage\nthat it can work in an online fashion instead of operating over the whole input\nbatch of video frames. The network is tested on the change detection dataset,\nand proved to have 5.5\\% improvement in F-measure over a plain fully\nconvolutional network for per frame segmentation. It was also shown to have\nimprovement of 1.4\\% for the F-measure compared to our baseline'
Label(s): ['cs.CV']
Predicted Label(s): (cs.CV, eess.IV, cs.LG)
 

```
</div>
The prediction results are not that great but not below the par for a simple model like
ours. We can improve this performance with models that consider word order like LSTM or
even those that use Transformers ([Vaswani et al.](https://arxiv.org/abs/1706.03762)).

---
## Acknowledgements

We would like to thank [Matt Watson](https://github.com/mattdangerw) for helping us
tackle the multi-label binarization part and inverse-transforming the processed labels
to the original form.
