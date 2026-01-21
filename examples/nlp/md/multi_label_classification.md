# Large-scale multi-label text classification

**Author:** [Sayak Paul](https://twitter.com/RisingSayak), [Soumik Rakshit](https://github.com/soumik12345)<br>
**Date created:** 2020/09/25<br>
**Last modified:** 2025/02/27<br>
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
import os

os.environ["KERAS_BACKEND"] = "jax"  # or tensorflow, or torch

import keras
from keras import layers, ops

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
There are 2000 rows in the dataset.

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
There are 9 duplicate titles.

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
There are 1991 rows in the deduplicated dataset.
208
275

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
(1783, 3)

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
Number of rows in training set: 1604
Number of rows in validation set: 90
Number of rows in test set: 89

```
</div>
---
## Multi-label binarization

Now we preprocess our labels using the
[`StringLookup`](https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup)
layer.


```python
# For RaggedTensor
import tensorflow as tf

terms = tf.ragged.constant(train_df["terms"].values)
lookup = layers.StringLookup(output_mode="multi_hot")
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
['[UNK]', 'cs.CV', 'cs.LG', 'cs.AI', 'stat.ML', 'eess.IV', 'cs.NE', 'cs.RO', 'cs.CL', 'cs.SI', 'cs.MM', 'math.NA', 'cs.CG', 'cs.CR', 'I.4.6', 'math.OC', 'cs.GR', 'cs.NA', 'cs.HC', 'cs.DS', '68U10', 'stat.ME', 'q-bio.NC', 'math.AP', 'eess.SP', 'cs.DM', '62H30']

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
Original label: ['cs.CV']

An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.

Label-binarized representation: [[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

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
count    1604.000000
mean      158.151496
std        41.543130
min        25.000000
25%       130.000000
50%       156.000000
75%       184.250000
max       283.000000
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
Abstract: b"For the Domain Generalization (DG) problem where the hypotheses are composed\nof a common representation function followed by a labeling function, we point\nout a shortcoming in existing approaches that fail to explicitly optimize for a\nterm, appearing in a well-known and widely adopted upper bound to the risk on\nthe unseen domain, that is dependent on the representation to be learned. To\nthis end, we first derive a novel upper bound to the prediction risk. We show\nthat imposing a mild assumption on the representation to be learned, namely\nmanifold restricted invertibility, is sufficient to deal with this issue.\nFurther, unlike existing approaches, our novel upper bound doesn't require the\nassumption of Lipschitzness of the loss function. In addition, the\ndistributional discrepancy in the representation space is handled via the\nWasserstein-2 barycenter cost. In this context, we creatively leverage old and\nrecent transport inequalities, which link various optimal transport metrics, in\nparticular the $L^1$ distance (also known as the total variation distance) and\nthe Wasserstein-2 distances, with the Kullback-Liebler divergence. These\nanalyses and insights motivate a new representation learning cost for DG that\nadditively balances three competing objectives: 1) minimizing classification\nerror across seen domains via cross-entropy, 2) enforcing domain-invariance in\nthe representation space via the Wasserstein-2 barycenter cost, and 3)\npromoting non-degenerate, nearly-invertible representation via one of two\nmechanisms, viz., an autoencoder-based reconstruction loss or a mutual\ninformation loss. It is to be noted that the proposed algorithms completely\nbypass the use of any adversarial training mechanism that is typical of many\ncurrent domain generalization approaches. Simulation results on several\nstandard datasets demonstrate superior performance compared to several\nwell-known DG algorithms."
Label(s): ['cs.LG' 'stat.ML']
 
Abstract: b'Image segmentation of touching objects plays a key role in providing accurate\nclassification for computer vision technologies. A new line profile based\nimaging segmentation algorithm has been developed to provide a robust and\naccurate segmentation of a group of touching corns. The performance of the line\nprofile based algorithm has been compared to a watershed based imaging\nsegmentation algorithm. Both algorithms are tested on three different patterns\nof images, which are isolated corns, single-lines, and random distributed\nformations. The experimental results show that the algorithm can segment a\nlarge number of touching corn kernels efficiently and accurately.'
Label(s): ['cs.CV']
 
Abstract: b'Semantic image segmentation is a principal problem in computer vision, where\nthe aim is to correctly classify each individual pixel of an image into a\nsemantic label. Its widespread use in many areas, including medical imaging and\nautonomous driving, has fostered extensive research in recent years. Empirical\nimprovements in tackling this task have primarily been motivated by successful\nexploitation of Convolutional Neural Networks (CNNs) pre-trained for image\nclassification and object recognition. However, the pixel-wise labelling with\nCNNs has its own unique challenges: (1) an accurate deconvolution, or\nupsampling, of low-resolution output into a higher-resolution segmentation mask\nand (2) an inclusion of global information, or context, within locally\nextracted features. To address these issues, we propose a novel architecture to\nconduct the equivalent of the deconvolution operation globally and acquire\ndense predictions. We demonstrate that it leads to improved performance of\nstate-of-the-art semantic segmentation models on the PASCAL VOC 2012 benchmark,\nreaching 74.0% mean IU accuracy on the test set.'
Label(s): ['cs.CV']
 
Abstract: b'Modern deep learning models have revolutionized the field of computer vision.\nBut, a significant drawback of most of these models is that they require a\nlarge number of labelled examples to generalize properly. Recent developments\nin few-shot learning aim to alleviate this requirement. In this paper, we\npropose a novel lightweight CNN architecture for 1-shot image segmentation. The\nproposed model is created by taking inspiration from well-performing\narchitectures for semantic segmentation and adapting it to the 1-shot domain.\nWe train our model using 4 meta-learning algorithms that have worked well for\nimage classification and compare the results. For the chosen dataset, our\nproposed model has a 70% lower parameter count than the benchmark, while having\nbetter or comparable mean IoU scores using all 4 of the meta-learning\nalgorithms.'
Label(s): ['cs.CV' 'cs.LG' 'eess.IV']
 
Abstract: b'In this work, we propose CARLS, a novel framework for augmenting the capacity\nof existing deep learning frameworks by enabling multiple components -- model\ntrainers, knowledge makers and knowledge banks -- to concertedly work together\nin an asynchronous fashion across hardware platforms. The proposed CARLS is\nparticularly suitable for learning paradigms where model training benefits from\nadditional knowledge inferred or discovered during training, such as node\nembeddings for graph neural networks or reliable pseudo labels from model\npredictions. We also describe three learning paradigms -- semi-supervised\nlearning, curriculum learning and multimodal learning -- as examples that can\nbe scaled up efficiently by CARLS. One version of CARLS has been open-sourced\nand available for download at:\nhttps://github.com/tensorflow/neural-structured-learning/tree/master/research/carls'
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
20498

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

    
  1/13 ━[37m━━━━━━━━━━━━━━━━━━━  26s 2s/step - binary_accuracy: 0.4491 - loss: 1.4007

<div class="k-default-codeblock">
```

```
</div>
  2/13 ━━━[37m━━━━━━━━━━━━━━━━━  3s 307ms/step - binary_accuracy: 0.5609 - loss: 1.1359

<div class="k-default-codeblock">
```

```
</div>
  3/13 ━━━━[37m━━━━━━━━━━━━━━━━  2s 290ms/step - binary_accuracy: 0.6315 - loss: 0.9654

<div class="k-default-codeblock">
```

```
</div>
  4/13 ━━━━━━[37m━━━━━━━━━━━━━━  2s 286ms/step - binary_accuracy: 0.6785 - loss: 0.8508

<div class="k-default-codeblock">
```

```
</div>
  5/13 ━━━━━━━[37m━━━━━━━━━━━━━  2s 282ms/step - binary_accuracy: 0.7128 - loss: 0.7661

<div class="k-default-codeblock">
```

```
</div>
  6/13 ━━━━━━━━━[37m━━━━━━━━━━━  1s 283ms/step - binary_accuracy: 0.7391 - loss: 0.7006

<div class="k-default-codeblock">
```

```
</div>
  7/13 ━━━━━━━━━━[37m━━━━━━━━━━  1s 277ms/step - binary_accuracy: 0.7600 - loss: 0.6485

<div class="k-default-codeblock">
```

```
</div>
  8/13 ━━━━━━━━━━━━[37m━━━━━━━━  1s 275ms/step - binary_accuracy: 0.7770 - loss: 0.6054

<div class="k-default-codeblock">
```

```
</div>
  9/13 ━━━━━━━━━━━━━[37m━━━━━━━  1s 272ms/step - binary_accuracy: 0.7913 - loss: 0.5693

<div class="k-default-codeblock">
```

```
</div>
 10/13 ━━━━━━━━━━━━━━━[37m━━━━━  0s 270ms/step - binary_accuracy: 0.8033 - loss: 0.5389

<div class="k-default-codeblock">
```

```
</div>
 11/13 ━━━━━━━━━━━━━━━━[37m━━━━  0s 272ms/step - binary_accuracy: 0.8136 - loss: 0.5127

<div class="k-default-codeblock">
```

```
</div>
 12/13 ━━━━━━━━━━━━━━━━━━[37m━━  0s 273ms/step - binary_accuracy: 0.8225 - loss: 0.4899

<div class="k-default-codeblock">
```

```
</div>
 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 363ms/step - binary_accuracy: 0.8303 - loss: 0.4702

<div class="k-default-codeblock">
```

```
</div>
 13/13 ━━━━━━━━━━━━━━━━━━━━ 7s 402ms/step - binary_accuracy: 0.8369 - loss: 0.4532 - val_binary_accuracy: 0.9782 - val_loss: 0.0867



    
![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_14.png)
    



    
![png](/img/examples/nlp/multi_label_classification/multi_label_classification_38_15.png)
    


While training, we notice an initial sharp fall in the loss followed by a gradual decay.

### Evaluate the model


```python
_, binary_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 483ms/step - binary_accuracy: 0.9734 - loss: 0.0927

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 486ms/step - binary_accuracy: 0.9734 - loss: 0.0927


<div class="k-default-codeblock">
```
Categorical accuracy on the test set: 97.34%.

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

# We create a custom Model to override the predict method so
# that it first vectorizes text data
class ModelEndtoEnd(keras.Model):

    def predict(self, inputs):
        indices = text_vectorizer(inputs)
        return super().predict(indices)


def get_inference_model(model):
    inputs = shallow_mlp_model.inputs
    outputs = shallow_mlp_model.outputs
    end_to_end_model = ModelEndtoEnd(inputs, outputs, name="end_to_end_model")
    end_to_end_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


model_for_inference = get_inference_model(shallow_mlp_model)

# Create a small dataset just for demonstrating inference.
inference_dataset = make_dataset(test_df.sample(2), is_train=False)
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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 141ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 142ms/step


<div class="k-default-codeblock">
```
Abstract: b'High-resolution image segmentation remains challenging and error-prone due to\nthe enormous size of intermediate feature maps. Conventional methods avoid this\nproblem by using patch based approaches where each patch is segmented\nindependently. However, independent patch segmentation induces errors,\nparticularly at the patch boundary due to the lack of contextual information in\nvery high-resolution images where the patch size is much smaller compared to\nthe full image. To overcome these limitations, in this paper, we propose a\nnovel framework to segment a particular patch by incorporating contextual\ninformation from its neighboring patches. This allows the segmentation network\nto see the target patch with a wider field of view without the need of larger\nfeature maps. Comparative analysis from a number of experiments shows that our\nproposed framework is able to segment high resolution images with significantly\nimproved mean Intersection over Union and overall accuracy.'
Label(s): ['cs.CV']
Predicted Label(s): (cs.CV, eess.IV, cs.LG)
 
Abstract: b"Convolutional neural networks for visual recognition require large amounts of\ntraining samples and usually benefit from data augmentation. This paper\nproposes PatchMix, a data augmentation method that creates new samples by\ncomposing patches from pairs of images in a grid-like pattern. These new\nsamples' ground truth labels are set as proportional to the number of patches\nfrom each image. We then add a set of additional losses at the patch-level to\nregularize and to encourage good representations at both the patch and image\nlevels. A ResNet-50 model trained on ImageNet using PatchMix exhibits superior\ntransfer learning capabilities across a wide array of benchmarks. Although\nPatchMix can rely on random pairings and random grid-like patterns for mixing,\nwe explore evolutionary search as a guiding strategy to discover optimal\ngrid-like patterns and image pairing jointly. For this purpose, we conceive a\nfitness function that bypasses the need to re-train a model to evaluate each\nchoice. In this way, PatchMix outperforms a base model on CIFAR-10 (+1.91),\nCIFAR-100 (+5.31), Tiny Imagenet (+3.52), and ImageNet (+1.16) by significant\nmargins, also outperforming previous state-of-the-art pairwise augmentation\nstrategies."
Label(s): ['cs.CV' 'cs.LG' 'cs.NE']
Predicted Label(s): (cs.CV, cs.LG, stat.ML)
 

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/models/functional.py:252: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: ['keras_tensor_2']
Received: inputs=Tensor(shape=(2, 20498))
  warnings.warn(msg)

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

Thanks to [Cingis Kratochvil](https://github.com/cumbalik) for suggesting and extending this code example by introducing binary accuracy as the evaluation metric.

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 14: Text classification](https://deeplearningwithpython.io/chapters/chapter14_text-classification)
