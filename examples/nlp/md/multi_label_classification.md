# Large-scale multi-label text classification

**Author:** [Sayak Paul](https://twitter.com/RisingSayak), [Soumik Rakshit](https://github.com/soumik12345)<br>
**Date created:** 2020/09/25<br>
**Last modified:** 2020/12/23<br>
**Description:** Implementing a large-scale multi-label text classification model.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/multi_label_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/multi_label_classification.py)



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
2021-12-23 15:25:26.502792: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-12-23 15:25:28.783738: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38442 MB memory:  -> device: 0, name: A100-SXM4-40GB, pci bus id: 0000:00:04.0, compute capability: 8.0

Vocabulary:
```
</div>
    
<div class="k-default-codeblock">
```
['[UNK]', 'cs.CV', 'cs.LG', 'stat.ML', 'cs.AI', 'eess.IV', 'cs.RO', 'cs.CL', 'cs.NE', 'cs.CR', 'math.OC', 'eess.SP', 'cs.GR', 'cs.SI', 'cs.MM', 'cs.SY', 'cs.IR', 'cs.MA', 'eess.SY', 'cs.HC', 'math.IT', 'cs.IT', 'cs.DC', 'cs.CY', 'stat.AP', 'stat.TH', 'math.ST', 'stat.ME', 'eess.AS', 'cs.SD', 'q-bio.QM', 'q-bio.NC', 'cs.DS', 'cs.GT', 'cs.NI', 'cs.CG', 'cs.SE', 'I.2.6', 'stat.CO', 'math.NA', 'cs.NA', 'physics.chem-ph', 'cs.DB', 'q-bio.BM', 'cs.LO', 'cond-mat.dis-nn', '68T45', 'math.PR', 'cs.PL', 'physics.comp-ph', 'cs.CE', 'cs.AR', 'I.2.10', 'q-fin.ST', 'cond-mat.stat-mech', '68T05', 'quant-ph', 'math.DS', 'cs.CC', 'I.4.6', 'physics.soc-ph', 'physics.data-an', 'physics.ao-ph', 'q-bio.GN', 'econ.EM', 'cs.DM', 'physics.med-ph', 'cs.PF', 'astro-ph.IM', 'I.4.8', 'math.AT', 'I.4', 'q-fin.TR', 'cs.FL', 'I.5.4', 'I.2', '68U10', 'hep-ex', 'cond-mat.mtrl-sci', '68T10', 'physics.optics', 'physics.geo-ph', 'physics.flu-dyn', 'math.AP', 'I.4; I.5', 'I.4.9', 'I.2.6; I.2.8', 'I.2.10; I.4; I.5', '68T01', '65D19', 'q-fin.CP', 'nlin.CD', 'math.CO', 'cs.MS', 'I.2.6; I.5.1', 'I.2.0; I.2.6', '68T07', 'cs.SC', 'cs.ET', 'K.3.2', 'I.2; I.5', 'I.2.8', '68U01', '68T30', '68', 'q-fin.GN', 'q-fin.EC', 'q-bio.MN', 'econ.GN', 'I.4.9; I.5.4', 'I.4.5', 'I.2; I.4; I.5', 'I.2.6; I.2.7', 'I.2.10; I.4.8', '68T99', '68Q32', '62H30', 'q-fin.RM', 'q-fin.PM', 'q-bio.TO', 'q-bio.OT', 'physics.bio-ph', 'nlin.AO', 'math.LO', 'math.FA', 'hep-ph', 'cond-mat.soft', 'I.4.6; I.4.8', 'I.4.4', 'I.4.3', 'I.4.0', 'I.2; J.2', 'I.2; I.2.6; I.2.7', 'I.2.7', 'I.2.6; I.5.4', 'I.2.6; I.2.9', 'I.2.6; I.2.7; H.3.1; H.3.3', 'I.2.6; I.2.10', 'I.2.6, I.5.4', 'I.2.1; J.3', 'I.2.10; I.5.1; I.4.8', 'I.2.10; I.4.8; I.5.4', 'I.2.10; I.2.6', 'I.2.1', 'H.3.1; I.2.6; I.2.7', 'H.3.1; H.3.3; I.2.6; I.2.7', 'G.3', 'F.2.2; I.2.7', 'E.5; E.4; E.2; H.1.1; F.1.1; F.1.3', '68Txx', '62H99', '62H35', '14J60 (Primary) 14F05, 14J26 (Secondary)']

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
Original label: ['cs.LG', 'cs.RO', 'cs.SY', 'eess.SY']
Label-binarized representation: [[0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0.
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
mean       156.513355
std         41.514411
min          5.000000
25%        128.000000
50%        154.000000
75%        183.000000
max        297.000000
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
Abstract: b'Graph Neural Networks (GNNs) have recently demonstrated superior capability\nof tackling graph analytical problems in various applications. Nevertheless,\nwith the wide-spreading practice of GNNs in high-stake decision-making\nprocesses, there is an increasing societal concern that GNNs could make\ndiscriminatory decisions that may be illegal towards certain demographic\ngroups. Although some explorations have been made towards developing fair GNNs,\nexisting approaches are tailored for a specific GNN model. However, in\npractical scenarios, myriads of GNN variants have been proposed for different\ntasks, and it is costly to train and fine-tune existing debiasing models for\ndifferent GNNs. Also, bias in a trained model could originate from training\ndata, while how to mitigate bias in the graph data is usually overlooked. In\nthis work, different from existing work, we first propose novel definitions and\nmetrics to measure the bias in an attributed network, which leads to the\noptimization objective to mitigate bias. Based on the optimization objective,\nwe develop a framework named EDITS to mitigate the bias in attributed networks\nwhile preserving useful information. EDITS works in a model-agnostic manner,\nwhich means that it is independent of the specific GNNs applied for downstream\ntasks. Extensive experiments on both synthetic and real-world datasets\ndemonstrate the validity of the proposed bias metrics and the superiority of\nEDITS on both bias mitigation and utility maintenance. Open-source\nimplementation: https://github.com/yushundong/EDITS.'
Label(s): ['cs.LG']
 
Abstract: b'Graph Neural Networks (GNNs) are widely used for analyzing graph-structured\ndata. Most GNN methods are highly sensitive to the quality of graph structures\nand usually require a perfect graph structure for learning informative\nembeddings. However, the pervasiveness of noise in graphs necessitates learning\nrobust representations for real-world problems. To improve the robustness of\nGNN models, many studies have been proposed around the central concept of Graph\nStructure Learning (GSL), which aims to jointly learn an optimized graph\nstructure and corresponding representations. Towards this end, in the presented\nsurvey, we broadly review recent progress of GSL methods for learning robust\nrepresentations. Specifically, we first formulate a general paradigm of GSL,\nand then review state-of-the-art methods classified by how they model graph\nstructures, followed by applications that incorporate the idea of GSL in other\ngraph tasks. Finally, we point out some issues in current studies and discuss\nfuture directions.'
Label(s): ['cs.LG' 'cs.SI']
 
Abstract: b'Conventional saliency maps highlight input features to which neural network\npredictions are highly sensitive. We take a different approach to saliency, in\nwhich we identify and analyze the network parameters, rather than inputs, which\nare responsible for erroneous decisions. We find that samples which cause\nsimilar parameters to malfunction are semantically similar. We also show that\npruning the most salient parameters for a wrongly classified sample often\nimproves model behavior. Furthermore, fine-tuning a small number of the most\nsalient parameters on a single sample results in error correction on other\nsamples that are misclassified for similar reasons. Based on our parameter\nsaliency method, we also introduce an input-space saliency technique that\nreveals how image features cause specific network components to malfunction.\nFurther, we rigorously validate the meaningfulness of our saliency maps on both\nthe dataset and case-study levels.'
Label(s): ['cs.CV' 'cs.LG']
 
Abstract: b'Recent advances in object detection have benefited significantly from rapid\ndevelopments in deep neural networks. However, neural networks suffer from the\nwell-known issue of catastrophic forgetting, which makes continual or lifelong\nlearning problematic. In this paper, we leverage the fact that new training\nclasses arrive in a sequential manner and incrementally refine the model so\nthat it additionally detects new object classes in the absence of previous\ntraining data. Specifically, we consider the representative object detector,\nFaster R-CNN, for both accurate and efficient prediction. To prevent abrupt\nperformance degradation due to catastrophic forgetting, we propose to apply\nknowledge distillation on both the region proposal network and the region\nclassification network, to retain the detection of previously trained classes.\nA pseudo-positive-aware sampling strategy is also introduced for distillation\nsample selection. We evaluate the proposed method on PASCAL VOC 2007 and MS\nCOCO benchmarks and show competitive mAP and 6x inference speed improvement,\nwhich makes the approach more suitable for real-time applications. Our\nimplementation will be publicly available.'
Label(s): ['cs.CV']
 
Abstract: b'In this paper, we present an unsupervised learning approach to identify the\nuser points of interest (POI) by exploiting WiFi measurements from smartphone\napplication data. Due to the lack of GPS positioning accuracy in indoor,\nsheltered, and high rise building environments, we rely on widely available\nWiFi access points (AP) in contemporary urban areas to accurately identify POI\nand mobility patterns, by comparing the similarity in the WiFi measurements. We\npropose a system architecture to scan the surrounding WiFi AP, and perform\nunsupervised learning to demonstrate that it is possible to identify three\nmajor insights, namely the indoor POI within a building, neighbourhood\nactivity, and micro-mobility of the users. Our results show that it is possible\nto identify the aforementioned insights, with the fusion of WiFi and GPS, which\nare not possible to identify by only using GPS.'
Label(s): ['cs.LG']
 

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
153292

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
  1/258 [..............................] - ETA: 12:43 - loss: 0.8533 - categorical_accuracy: 0.0000e+00

2021-12-23 15:25:45.182167: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.

258/258 [==============================] - 10s 26ms/step - loss: 0.0339 - categorical_accuracy: 0.8613 - val_loss: 0.0188 - val_categorical_accuracy: 0.8936
Epoch 2/20
258/258 [==============================] - 7s 25ms/step - loss: 0.0032 - categorical_accuracy: 0.8942 - val_loss: 0.0268 - val_categorical_accuracy: 0.8745
Epoch 3/20
258/258 [==============================] - 7s 25ms/step - loss: 8.3650e-04 - categorical_accuracy: 0.8621 - val_loss: 0.0317 - val_categorical_accuracy: 0.8822
Epoch 4/20
258/258 [==============================] - 7s 25ms/step - loss: 3.2207e-04 - categorical_accuracy: 0.8633 - val_loss: 0.0373 - val_categorical_accuracy: 0.8920
Epoch 5/20
258/258 [==============================] - 7s 25ms/step - loss: 1.8659e-04 - categorical_accuracy: 0.8499 - val_loss: 0.0398 - val_categorical_accuracy: 0.8843
Epoch 6/20
258/258 [==============================] - 7s 25ms/step - loss: 1.1636e-04 - categorical_accuracy: 0.8583 - val_loss: 0.0407 - val_categorical_accuracy: 0.8843
Epoch 7/20
258/258 [==============================] - 7s 25ms/step - loss: 1.0806e-04 - categorical_accuracy: 0.8554 - val_loss: 0.0419 - val_categorical_accuracy: 0.8773
Epoch 8/20
258/258 [==============================] - 7s 25ms/step - loss: 7.9192e-05 - categorical_accuracy: 0.8593 - val_loss: 0.0445 - val_categorical_accuracy: 0.8729
Epoch 9/20
258/258 [==============================] - 7s 25ms/step - loss: 7.5030e-05 - categorical_accuracy: 0.8489 - val_loss: 0.0466 - val_categorical_accuracy: 0.8489
Epoch 10/20
258/258 [==============================] - 7s 25ms/step - loss: 6.2200e-05 - categorical_accuracy: 0.8552 - val_loss: 0.0443 - val_categorical_accuracy: 0.8756
Epoch 11/20
258/258 [==============================] - 7s 25ms/step - loss: 5.5798e-05 - categorical_accuracy: 0.8589 - val_loss: 0.0454 - val_categorical_accuracy: 0.8778
Epoch 12/20
258/258 [==============================] - 7s 25ms/step - loss: 5.0180e-05 - categorical_accuracy: 0.8704 - val_loss: 0.0475 - val_categorical_accuracy: 0.8833
Epoch 13/20
258/258 [==============================] - 7s 25ms/step - loss: 1.0158e-04 - categorical_accuracy: 0.8757 - val_loss: 0.0444 - val_categorical_accuracy: 0.8783
Epoch 14/20
258/258 [==============================] - 7s 25ms/step - loss: 9.6758e-04 - categorical_accuracy: 0.8610 - val_loss: 0.0395 - val_categorical_accuracy: 0.8707
Epoch 15/20
258/258 [==============================] - 7s 25ms/step - loss: 0.0022 - categorical_accuracy: 0.8466 - val_loss: 0.0418 - val_categorical_accuracy: 0.8652
Epoch 16/20
258/258 [==============================] - 7s 25ms/step - loss: 0.0015 - categorical_accuracy: 0.8305 - val_loss: 0.0466 - val_categorical_accuracy: 0.8794
Epoch 17/20
258/258 [==============================] - 7s 25ms/step - loss: 7.3772e-04 - categorical_accuracy: 0.8020 - val_loss: 0.0521 - val_categorical_accuracy: 0.8603
Epoch 18/20
258/258 [==============================] - 7s 25ms/step - loss: 4.3354e-04 - categorical_accuracy: 0.7905 - val_loss: 0.0545 - val_categorical_accuracy: 0.8636
Epoch 19/20
258/258 [==============================] - 7s 25ms/step - loss: 2.7111e-04 - categorical_accuracy: 0.7756 - val_loss: 0.0548 - val_categorical_accuracy: 0.8358
Epoch 20/20
258/258 [==============================] - 7s 25ms/step - loss: 1.2118e-04 - categorical_accuracy: 0.7819 - val_loss: 0.0601 - val_categorical_accuracy: 0.8685

```
</div>
    
![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_3.png)
    



    
![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_4.png)
    


While training, we notice an initial sharp fall in the loss followed by a gradual decay.

### Evaluate the model


```python
_, categorical_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")
```

<div class="k-default-codeblock">
```
15/15 [==============================] - 0s 19ms/step - loss: 0.0609 - categorical_accuracy: 0.8642
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
Abstract: b'In this paper, we propose a method that disentangles the effects of multiple\ninput conditions in Generative Adversarial Networks (GANs). In particular, we\ndemonstrate our method in controlling color, texture, and shape of a generated\ngarment image for computer-aided fashion design. To disentangle the effect of\ninput attributes, we customize conditional GANs with consistency loss\nfunctions. In our experiments, we tune one input at a time and show that we can\nguide our network to generate novel and realistic images of clothing articles.\nIn addition, we present a fashion design process that estimates the input\nattributes of an existing garment and modifies them using our generator.'
Label(s): ['cs.CV' 'stat.ML']
Predicted Label(s): (cs.CV, cs.LG, stat.ML)
 
Abstract: b'Recently there has been an enormous interest in generative models for images\nin deep learning. In pursuit of this, Generative Adversarial Networks (GAN) and\nVariational Auto-Encoder (VAE) have surfaced as two most prominent and popular\nmodels. While VAEs tend to produce excellent reconstructions but blurry\nsamples, GANs generate sharp but slightly distorted images. In this paper we\npropose a new model called Variational InfoGAN (ViGAN). Our aim is two fold:\n(i) To generated new images conditioned on visual descriptions, and (ii) modify\nthe image, by fixing the latent representation of image and varying the visual\ndescription. We evaluate our model on Labeled Faces in the Wild (LFW), celebA\nand a modified version of MNIST datasets and demonstrate the ability of our\nmodel to generate new images as well as to modify a given image by changing\nattributes.'
Label(s): ['cs.CV']
Predicted Label(s): (cs.CV, cs.LG, stat.ML)
 
Abstract: b'Generative adversarial networks (GANs) are a class of generative models,\nknown for producing accurate samples. The key feature of GANs is that there are\ntwo antagonistic neural networks: the generator and the discriminator. The main\nbottleneck for their implementation is that the neural networks are very hard\nto train. One way to improve their performance is to design reliable algorithms\nfor the adversarial process. Since the training can be cast as a stochastic\nNash equilibrium problem, we rewrite it as a variational inequality and\nintroduce an algorithm to compute an approximate solution. Specifically, we\npropose a stochastic relaxed forward-backward algorithm for GANs. We prove that\nwhen the pseudogradient mapping of the game is monotone, we have convergence to\nan exact solution or in a neighbourhood of it.'
Label(s): ['cs.LG' 'stat.ML' 'math.OC' 'cs.GT']
Predicted Label(s): (cs.LG, cs.GT, cs.AI)
 
Abstract: b'Optical flow estimation is an important yet challenging problem in the field\nof video analytics. The features of different semantics levels/layers of a\nconvolutional neural network can provide information of different granularity.\nTo exploit such flexible and comprehensive information, we propose a\nsemi-supervised Feature Pyramidal Correlation and Residual Reconstruction\nNetwork (FPCR-Net) for optical flow estimation from frame pairs. It consists of\ntwo main modules: pyramid correlation mapping and residual reconstruction. The\npyramid correlation mapping module takes advantage of the multi-scale\ncorrelations of global/local patches by aggregating features of different\nscales to form a multi-level cost volume. The residual reconstruction module\naims to reconstruct the sub-band high-frequency residuals of finer optical flow\nin each stage. Based on the pyramid correlation mapping, we further propose a\ncorrelation-warping-normalization (CWN) module to efficiently exploit the\ncorrelation dependency. Experiment results show that the proposed scheme\nachieves the state-of-the-art performance, with improvement by 0.80, 1.15 and\n0.10 in terms of average end-point error (AEE) against competing baseline\nmethods - FlowNet2, LiteFlowNet and PWC-Net on the Final pass of Sintel\ndataset, respectively.'
Label(s): ['cs.CV']
Predicted Label(s): (cs.CV, eess.IV, physics.optics)
 
Abstract: b'Caricature is an artistic drawing created to abstract or exaggerate facial\nfeatures of a person. Rendering visually pleasing caricatures is a difficult\ntask that requires professional skills, and thus it is of great interest to\ndesign a method to automatically generate such drawings. To deal with large\nshape changes, we propose an algorithm based on a semantic shape transform to\nproduce diverse and plausible shape exaggerations. Specifically, we predict\npixel-wise semantic correspondences and perform image warping on the input\nphoto to achieve dense shape transformation. We show that the proposed\nframework is able to render visually pleasing shape exaggerations while\nmaintaining their facial structures. In addition, our model allows users to\nmanipulate the shape via the semantic map. We demonstrate the effectiveness of\nour approach on a large photograph-caricature benchmark dataset with\ncomparisons to the state-of-the-art methods.'
Label(s): ['cs.CV']
Predicted Label(s): (cs.CV, cs.GR, cs.AI)
 

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
