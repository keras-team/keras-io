# Large-scale multi-label text classification

**Author:** [Sayak Paul](https://twitter.com/RisingSayak), [Soumik Rakshit](https://github.com/soumik12345)<br>
**Date created:** 2020/09/25<br>
**Last modified:** 2020/12/23<br>
**Description:** Implementing a large-scale multi-label text classification model.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team\keras-io\blob\master\examples\nlp/ipynb/multi_label_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team\keras-io\blob\master\examples\nlp/multi_label_classification.py)



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
['[UNK]', 'cs.CV', 'cs.LG', 'stat.ML', 'cs.AI', 'eess.IV', 'cs.RO', 'cs.CL', 'cs.NE', 'cs.CR', 'math.OC', 'eess.SP', 'cs.GR', 'cs.SI', 'cs.MM', 'cs.SY', 'cs.IR', 'cs.MA', 'eess.SY', 'cs.HC', 'math.IT', 'cs.IT', 'cs.DC', 'cs.CY', 'stat.AP', 'stat.TH', 'math.ST', 'stat.ME', 'eess.AS', 'cs.SD', 'q-bio.QM', 'q-bio.NC', 'cs.DS', 'cs.GT', 'cs.CG', 'cs.SE', 'cs.NI', 'I.2.6', 'stat.CO', 'math.NA', 'cs.NA', 'physics.chem-ph', 'cs.DB', 'q-bio.BM', 'cs.PL', 'cs.LO', 'cond-mat.dis-nn', '68T45', 'math.PR', 'physics.comp-ph', 'I.2.10', 'cs.CE', 'cs.AR', 'q-fin.ST', 'cond-mat.stat-mech', '68T05', 'quant-ph', 'math.DS', 'physics.data-an', 'cs.CC', 'I.4.6', 'physics.soc-ph', 'physics.ao-ph', 'cs.DM', 'econ.EM', 'q-bio.GN', 'physics.med-ph', 'astro-ph.IM', 'I.4.8', 'math.AT', 'cs.PF', 'cs.FL', 'I.4', 'q-fin.TR', 'I.5.4', 'I.2', '68U10', 'hep-ex', 'cond-mat.mtrl-sci', '68T10', 'physics.optics', 'physics.geo-ph', 'physics.flu-dyn', 'math.CO', 'math.AP', 'I.4; I.5', 'I.4.9', 'I.2.6; I.2.8', '68T01', '65D19', 'q-fin.CP', 'nlin.CD', 'cs.MS', 'I.2.6; I.5.1', 'I.2.10; I.4; I.5', 'I.2.0; I.2.6', '68T07', 'q-fin.GN', 'cs.SC', 'cs.ET', 'K.3.2', 'I.2.8', '68U01', '68T30', 'q-fin.EC', 'q-bio.MN', 'econ.GN', 'I.4.9; I.5.4', 'I.4.5', 'I.2; I.5', 'I.2; I.4; I.5', 'I.2.6; I.2.7', 'I.2.10; I.4.8', '68T99', '68Q32', '68', '62H30', 'q-fin.RM', 'q-fin.PM', 'q-bio.TO', 'q-bio.OT', 'physics.bio-ph', 'nlin.AO', 'math.LO', 'math.FA', 'hep-ph', 'cond-mat.soft', 'I.4.6; I.4.8', 'I.4.4', 'I.4.3', 'I.4.0', 'I.2; J.2', 'I.2; I.2.6; I.2.7', 'I.2.7', 'I.2.6; I.5.4', 'I.2.6; I.2.9', 'I.2.6; I.2.7; H.3.1; H.3.3', 'I.2.6; I.2.10', 'I.2.6, I.5.4', 'I.2.1; J.3', 'I.2.10; I.5.1; I.4.8', 'I.2.10; I.4.8; I.5.4', 'I.2.10; I.2.6', 'I.2.1', 'H.3.1; I.2.6; I.2.7', 'H.3.1; H.3.3; I.2.6; I.2.7', 'G.3', 'F.2.2; I.2.7', 'E.5; E.4; E.2; H.1.1; F.1.1; F.1.3', '68Txx', '62H99', '62H35', '14J60 (Primary) 14F05, 14J26 (Secondary)']
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
Original label: ['cs.LG', 'cs.CV', 'eess.IV']
Label-binarized representation: [[0. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
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
mean       156.497105
std         41.528225
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

Now, we implement utilities to prepare our datasets.


```python
max_seqlen = 150
batch_size = 128
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["terms"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["summaries"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
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
    print(f"Abstract: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    print(" ")
```

<div class="k-default-codeblock">
```
Abstract: b"In this paper we show how using satellite images can improve the accuracy of\nhousing price estimation models. Using Los Angeles County's property assessment\ndataset, by transferring learning from an Inception-v3 model pretrained on\nImageNet, we could achieve an improvement of ~10% in R-squared score compared\nto two baseline models that only use non-image features of the house."
Label(s): ['cs.LG' 'stat.ML']
 
Abstract: b'Learning from data streams is an increasingly important topic in data mining,\nmachine learning, and artificial intelligence in general. A major focus in the\ndata stream literature is on designing methods that can deal with concept\ndrift, a challenge where the generating distribution changes over time. A\ngeneral assumption in most of this literature is that instances are\nindependently distributed in the stream. In this work we show that, in the\ncontext of concept drift, this assumption is contradictory, and that the\npresence of concept drift necessarily implies temporal dependence; and thus\nsome form of time series. This has important implications on model design and\ndeployment. We explore and highlight the these implications, and show that\nHoeffding-tree based ensembles, which are very popular for learning in streams,\nare not naturally suited to learning \\emph{within} drift; and can perform in\nthis scenario only at significant computational cost of destructive adaptation.\nOn the other hand, we develop and parameterize gradient-descent methods and\ndemonstrate how they can perform \\emph{continuous} adaptation with no explicit\ndrift-detection mechanism, offering major advantages in terms of accuracy and\nefficiency. As a consequence of our theoretical discussion and empirical\nobservations, we outline a number of recommendations for deploying methods in\nconcept-drifting streams.'
Label(s): ['cs.LG' 'stat.ML']
 
Abstract: b"As reinforcement learning (RL) achieves more success in solving complex\ntasks, more care is needed to ensure that RL research is reproducible and that\nalgorithms herein can be compared easily and fairly with minimal bias. RL\nresults are, however, notoriously hard to reproduce due to the algorithms'\nintrinsic variance, the environments' stochasticity, and numerous (potentially\nunreported) hyper-parameters. In this work we investigate the many issues\nleading to irreproducible research and how to manage those. We further show how\nto utilise a rigorous and standardised evaluation approach for easing the\nprocess of documentation, evaluation and fair comparison of different\nalgorithms, where we emphasise the importance of choosing the right measurement\nmetrics and conducting proper statistics on the results, for unbiased reporting\nof the results."
Label(s): ['cs.LG' 'stat.ML' 'cs.AI' 'cs.RO']
 
Abstract: b'Estimating dense correspondences between images is a long-standing image\nunder-standing task. Recent works introduce convolutional neural networks\n(CNNs) to extract high-level feature maps and find correspondences through\nfeature matching. However,high-level feature maps are in low spatial resolution\nand therefore insufficient to provide accurate and fine-grained features to\ndistinguish intra-class variations for correspondence matching. To address this\nproblem, we generate robust features by dynamically selecting features at\ndifferent scales. To resolve two critical issues in feature selection,i.e.,how\nmany and which scales of features to be selected, we frame the feature\nselection process as a sequential Markov decision-making process (MDP) and\nintroduce an optimal selection strategy using reinforcement learning (RL). We\ndefine an RL environment for image matching in which each individual action\neither requires new features or terminates the selection episode by referring a\nmatching score. Deep neural networks are incorporated into our method and\ntrained for decision making. Experimental results show that our method achieves\ncomparable/superior performance with state-of-the-art methods on three\nbenchmarks, demonstrating the effectiveness of our feature selection strategy.'
Label(s): ['cs.CV']
 
Abstract: b'Dense reconstructions often contain errors that prior work has so far\nminimised using high quality sensors and regularising the output. Nevertheless,\nerrors still persist. This paper proposes a machine learning technique to\nidentify errors in three dimensional (3D) meshes. Beyond simply identifying\nerrors, our method quantifies both the magnitude and the direction of depth\nestimate errors when viewing the scene. This enables us to improve the\nreconstruction accuracy.\n  We train a suitably deep network architecture with two 3D meshes: a\nhigh-quality laser reconstruction, and a lower quality stereo image\nreconstruction. The network predicts the amount of error in the lower quality\nreconstruction with respect to the high-quality one, having only view the\nformer through its input. We evaluate our approach by correcting\ntwo-dimensional (2D) inverse-depth images extracted from the 3D model, and show\nthat our method improves the quality of these depth reconstructions by up to a\nrelative 10% RMSE.'
Label(s): ['cs.CV' 'cs.RO']
 
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
# Source: https://stackoverflow.com/a/18937309/7636462
vocabulary = set()
train_df["summaries"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

```

<div class="k-default-codeblock">
```
153338
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
function as well. For example, in [Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932),
Mahajan et al. used the softmax activation function and cross-entropy loss to train
their models.

There are several options of metrics that can be used in multi-label classification.
To keep this code example narrow we decided to use the
[binary accuracy metric](https://keras.io/api/metrics/accuracy_metrics/#binaryaccuracy-class).
To see the explanation why this metric is used we refer to this
[pull-request](https://github.com/keras-team/keras-io/pull/1133#issuecomment-1322736860).
There are also other suitable metrics for multi-label classification, like
[F1 Score](https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score) or
[Hamming loss](https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/HammingLoss).


```python
epochs = 20

shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
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
plot_result("binary_accuracy")
```

<div class="k-default-codeblock">
```
Epoch 1/20
258/258 [==============================] - 87s 332ms/step - loss: 0.0326 - binary_accuracy: 0.9893 - val_loss: 0.0189 - val_binary_accuracy: 0.9943
Epoch 2/20
258/258 [==============================] - 100s 387ms/step - loss: 0.0033 - binary_accuracy: 0.9990 - val_loss: 0.0271 - val_binary_accuracy: 0.9940
Epoch 3/20
258/258 [==============================] - 99s 384ms/step - loss: 7.8393e-04 - binary_accuracy: 0.9999 - val_loss: 0.0328 - val_binary_accuracy: 0.9939
Epoch 4/20
258/258 [==============================] - 109s 421ms/step - loss: 3.0132e-04 - binary_accuracy: 1.0000 - val_loss: 0.0366 - val_binary_accuracy: 0.9939
Epoch 5/20
258/258 [==============================] - 105s 405ms/step - loss: 1.6006e-04 - binary_accuracy: 1.0000 - val_loss: 0.0399 - val_binary_accuracy: 0.9939
Epoch 6/20
258/258 [==============================] - 107s 414ms/step - loss: 1.2400e-04 - binary_accuracy: 1.0000 - val_loss: 0.0412 - val_binary_accuracy: 0.9939
Epoch 7/20
258/258 [==============================] - 110s 425ms/step - loss: 7.7131e-05 - binary_accuracy: 1.0000 - val_loss: 0.0439 - val_binary_accuracy: 0.9940
Epoch 8/20
258/258 [==============================] - 105s 405ms/step - loss: 5.5611e-05 - binary_accuracy: 1.0000 - val_loss: 0.0446 - val_binary_accuracy: 0.9940
Epoch 9/20
258/258 [==============================] - 103s 397ms/step - loss: 4.5994e-05 - binary_accuracy: 1.0000 - val_loss: 0.0454 - val_binary_accuracy: 0.9940
Epoch 10/20
258/258 [==============================] - 105s 405ms/step - loss: 3.5126e-05 - binary_accuracy: 1.0000 - val_loss: 0.0472 - val_binary_accuracy: 0.9939
Epoch 11/20
258/258 [==============================] - 109s 422ms/step - loss: 2.9927e-05 - binary_accuracy: 1.0000 - val_loss: 0.0466 - val_binary_accuracy: 0.9940
Epoch 12/20
258/258 [==============================] - 133s 516ms/step - loss: 2.5748e-05 - binary_accuracy: 1.0000 - val_loss: 0.0484 - val_binary_accuracy: 0.9940
Epoch 13/20
258/258 [==============================] - 129s 497ms/step - loss: 4.3529e-05 - binary_accuracy: 1.0000 - val_loss: 0.0500 - val_binary_accuracy: 0.9940
Epoch 14/20
258/258 [==============================] - 158s 611ms/step - loss: 8.1068e-04 - binary_accuracy: 0.9998 - val_loss: 0.0377 - val_binary_accuracy: 0.9936
Epoch 15/20
258/258 [==============================] - 144s 558ms/step - loss: 0.0016 - binary_accuracy: 0.9995 - val_loss: 0.0418 - val_binary_accuracy: 0.9935
Epoch 16/20
258/258 [==============================] - 131s 506ms/step - loss: 0.0018 - binary_accuracy: 0.9995 - val_loss: 0.0479 - val_binary_accuracy: 0.9931
Epoch 17/20
258/258 [==============================] - 127s 491ms/step - loss: 0.0012 - binary_accuracy: 0.9997 - val_loss: 0.0521 - val_binary_accuracy: 0.9931
Epoch 18/20
258/258 [==============================] - 153s 594ms/step - loss: 6.3144e-04 - binary_accuracy: 0.9998 - val_loss: 0.0549 - val_binary_accuracy: 0.9934
Epoch 19/20
258/258 [==============================] - 142s 550ms/step - loss: 3.1753e-04 - binary_accuracy: 0.9999 - val_loss: 0.0589 - val_binary_accuracy: 0.9934
Epoch 20/20
258/258 [==============================] - 153s 594ms/step - loss: 2.0258e-04 - binary_accuracy: 1.0000 - val_loss: 0.0585 - val_binary_accuracy: 0.9933
```
</div>
    


    
![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_1.png)
    



    
![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_2.png)
    


While training, we notice an initial sharp fall in the loss followed by a gradual decay.

### Evaluate the model


```python
_, binary_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")
```

<div class="k-default-codeblock">
```
15/15 [==============================] - 3s 196ms/step - loss: 0.0580 - binary_accuracy: 0.9933
Categorical accuracy on the test set: 99.33%.
```
</div>
    

The trained model gives us an evaluation accuracy of ~99%.

---
## Inference

An important feature of the
[preprocessing layers provided by Keras](https://keras.io/api/layers/preprocessing_layers/)
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
    print(f"Abstract: {text}")
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
4/4 [==============================] - 0s 62ms/step
Abstract: b'We investigate the training of sparse layers that use different parameters\nfor different inputs based on hashing in large Transformer models.\nSpecifically, we modify the feedforward layer to hash to different sets of\nweights depending on the current token, over all tokens in the sequence. We\nshow that this procedure either outperforms or is competitive with\nlearning-to-route mixture-of-expert methods such as Switch Transformers and\nBASE Layers, while requiring no routing parameters or extra terms in the\nobjective function such as a load balancing loss, and no sophisticated\nassignment algorithm. We study the performance of different hashing techniques,\nhash sizes and input features, and show that balanced and random hashes focused\non the most local features work best, compared to either learning clusters or\nusing longer-range context. We show our approach works well both on large\nlanguage modeling and dialogue tasks, and on downstream fine-tuning tasks.'
Label(s): ['cs.LG' 'cs.CL']
Predicted Label(s): (cs.LG, cs.CL, stat.ML)
 
Abstract: b'We present the first method capable of photorealistically reconstructing\ndeformable scenes using photos/videos captured casually from mobile phones. Our\napproach augments neural radiance fields (NeRF) by optimizing an additional\ncontinuous volumetric deformation field that warps each observed point into a\ncanonical 5D NeRF. We observe that these NeRF-like deformation fields are prone\nto local minima, and propose a coarse-to-fine optimization method for\ncoordinate-based models that allows for more robust optimization. By adapting\nprinciples from geometry processing and physical simulation to NeRF-like\nmodels, we propose an elastic regularization of the deformation field that\nfurther improves robustness. We show that our method can turn casually captured\nselfie photos/videos into deformable NeRF models that allow for photorealistic\nrenderings of the subject from arbitrary viewpoints, which we dub "nerfies." We\nevaluate our method by collecting time-synchronized data using a rig with two\nmobile phones, yielding train/validation images of the same pose at different\nviewpoints. We show that our method faithfully reconstructs non-rigidly\ndeforming scenes and reproduces unseen views with high fidelity.'
Label(s): ['cs.CV' 'cs.GR']
Predicted Label(s): (cs.CV, cs.GR, cs.RO)
 
Abstract: b'We propose to jointly learn multi-view geometry and warping between views of\nthe same object instances for robust cross-view object detection. What makes\nmulti-view object instance detection difficult are strong changes in viewpoint,\nlighting conditions, high similarity of neighbouring objects, and strong\nvariability in scale. By turning object detection and instance\nre-identification in different views into a joint learning task, we are able to\nincorporate both image appearance and geometric soft constraints into a single,\nmulti-view detection process that is learnable end-to-end. We validate our\nmethod on a new, large data set of street-level panoramas of urban objects and\nshow superior performance compared to various baselines. Our contribution is\nthreefold: a large-scale, publicly available data set for multi-view instance\ndetection and re-identification; an annotation tool custom-tailored for\nmulti-view instance detection; and a novel, holistic multi-view instance\ndetection and re-identification method that jointly models geometry and\nappearance across views.'
Label(s): ['cs.CV' 'cs.LG' 'stat.ML']
Predicted Label(s): (cs.CV, cs.RO, cs.MM)
 
Abstract: b'Learning graph convolutional networks (GCNs) is an emerging field which aims\nat generalizing deep learning to arbitrary non-regular domains. Most of the\nexisting GCNs follow a neighborhood aggregation scheme, where the\nrepresentation of a node is recursively obtained by aggregating its neighboring\nnode representations using averaging or sorting operations. However, these\noperations are either ill-posed or weak to be discriminant or increase the\nnumber of training parameters and thereby the computational complexity and the\nrisk of overfitting. In this paper, we introduce a novel GCN framework that\nachieves spatial graph convolution in a reproducing kernel Hilbert space\n(RKHS). The latter makes it possible to design, via implicit kernel\nrepresentations, convolutional graph filters in a high dimensional and more\ndiscriminating space without increasing the number of training parameters. The\nparticularity of our GCN model also resides in its ability to achieve\nconvolutions without explicitly realigning nodes in the receptive fields of the\nlearned graph filters with those of the input graphs, thereby making\nconvolutions permutation agnostic and well defined. Experiments conducted on\nthe challenging task of skeleton-based action recognition show the superiority\nof the proposed method against different baselines as well as the related work.'
Label(s): ['cs.CV']
Predicted Label(s): (cs.LG, cs.CV, cs.NE)
 
Abstract: b'Recurrent meta reinforcement learning (meta-RL) agents are agents that employ\na recurrent neural network (RNN) for the purpose of "learning a learning\nalgorithm". After being trained on a pre-specified task distribution, the\nlearned weights of the agent\'s RNN are said to implement an efficient learning\nalgorithm through their activity dynamics, which allows the agent to quickly\nsolve new tasks sampled from the same distribution. However, due to the\nblack-box nature of these agents, the way in which they work is not yet fully\nunderstood. In this study, we shed light on the internal working mechanisms of\nthese agents by reformulating the meta-RL problem using the Partially\nObservable Markov Decision Process (POMDP) framework. We hypothesize that the\nlearned activity dynamics is acting as belief states for such agents. Several\nillustrative experiments suggest that this hypothesis is true, and that\nrecurrent meta-RL agents can be viewed as agents that learn to act optimally in\npartially observable environments consisting of multiple related tasks. This\nview helps in understanding their failure cases and some interesting\nmodel-based results reported in the literature.'
Label(s): ['cs.LG' 'cs.AI']
Predicted Label(s): (stat.ML, cs.LG, cs.AI)
 
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

Thanks [Cingis Kratochvil](https://github.com/cumbalik) for suggesting and extending
this code example by the binary accuracy.
