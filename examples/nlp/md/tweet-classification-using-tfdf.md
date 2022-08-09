# Text classification using Decision Forests and pretrained embeddings

**Author:** Gitesh Chawda<br>
**Date created:** 09/05/2022<br>
**Last modified:** 09/05/2022<br>
**Description:** Using Tensorflow Decision Forests for text classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/tweet-classification-using-tfdf.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/tweet-classification-using-tfdf.py)



---
## Introduction

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF)
is a collection of state-of-the-art algorithms for Decision Forest models that are
compatible with Keras APIs. The module includes Random Forests, Gradient Boosted Trees,
and CART, and can be used for regression, classification, and ranking tasks.

In this example we will use Gradient Boosted Trees with pretrained embeddings to
classify disaster-related tweets.

### See also:

- [TF-DF beginner tutorial](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab)
- [TF-DF intermediate tutorial](https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab).

Install Tensorflow Decision Forest using following command :
`pip install tensorflow_decision_forests`

---
## Imports


```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt
```

---
## Get the data

The Dataset is avalaible on [Kaggle](https://www.kaggle.com/c/nlp-getting-started)

Dataset description:

**Files:**

- train.csv: the training set

**Columns:**

- id: a unique identifier for each tweet
- text: the text of the tweet
- location: the location the tweet was sent from (may be blank)
- keyword: a particular keyword from the tweet (may be blank)
- target: in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)


```python
# Turn .csv files into pandas DataFrame's
df = pd.read_csv(
    "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
)
print(df.head())
```

<div class="k-default-codeblock">
```
   id keyword location                                               text  \
0   1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...   
1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada   
2   5     NaN      NaN  All residents asked to 'shelter in place' are ...   
3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...   
4   7     NaN      NaN  Just got sent this photo from Ruby #Alaska as ...   
```
</div>
    
<div class="k-default-codeblock">
```
   target  
0       1  
1       1  
2       1  
3       1  
4       1  

```
</div>
The dataset includes 7613 samples with 5 columns:


```python
print(f"Training dataset shape: {df.shape}")
```

<div class="k-default-codeblock">
```
Training dataset shape: (7613, 5)

```
</div>
Shuffling and dropping unnecessary columns:


```python
df_shuffled = df.sample(frac=1, random_state=42)
# Dropping id, keyword and location columns as these columns consists of mostly nan values
# we will be using only text and target columns
df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
df_shuffled.reset_index(inplace=True, drop=True)
print(df_shuffled.head())
```

<div class="k-default-codeblock">
```
                                                text  target
0  So you have a new weapon that can cause un-ima...       1
1  The f$&amp;@ing things I do for #GISHWHES Just...       0
2  DT @georgegalloway: RT @Galloway4Mayor: ÛÏThe...       1
3  Aftershock back to school kick off was great. ...       0
4  in response to trauma Children of Addicts deve...       0

```
</div>
Printing information about the shuffled dataframe:


```python
print(df_shuffled.info())
```

<div class="k-default-codeblock">
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7613 entries, 0 to 7612
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   text    7613 non-null   object
 1   target  7613 non-null   int64 
dtypes: int64(1), object(1)
memory usage: 119.1+ KB
None

```
</div>
Total number of "disaster" and "non-disaster" tweets:


```python
print(
    "Total Number of disaster and non-disaster tweets: "
    f"{df_shuffled.target.value_counts()}"
)
```

<div class="k-default-codeblock">
```
Total Number of disaster and non-disaster tweets: 0    4342
1    3271
Name: target, dtype: int64

```
</div>
Let's preview a few samples:


```python
for index, example in df_shuffled[:5].iterrows():
    print(f"Example #{index}")
    print(f"\tTarget : {example['target']}")
    print(f"\tText : {example['text']}")
```

<div class="k-default-codeblock">
```
Example #0
	Target : 1
	Text : So you have a new weapon that can cause un-imaginable destruction.
Example #1
	Target : 0
	Text : The f$&amp;@ing things I do for #GISHWHES Just got soaked in a deluge going for pads and tampons. Thx @mishacollins @/@
Example #2
	Target : 1
	Text : DT @georgegalloway: RT @Galloway4Mayor: ÛÏThe CoL police can catch a pickpocket in Liverpool Stree... http://t.co/vXIn1gOq4Q
Example #3
	Target : 0
	Text : Aftershock back to school kick off was great. I want to thank everyone for making it possible. What a great night.
Example #4
	Target : 0
	Text : in response to trauma Children of Addicts develop a defensive self - one that decreases vulnerability. (3

```
</div>
Splitting dataset into training and test sets:


```python
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)
print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")
```

<div class="k-default-codeblock">
```
Using 6852 samples for training and 761 for validation

```
</div>
Total number of "disaster" and "non-disaster" tweets in the training data:


```python
print(train_df["target"].value_counts())
```

<div class="k-default-codeblock">
```
0    3929
1    2923
Name: target, dtype: int64

```
</div>
Total number of "disaster" and "non-disaster" tweets in the test data:


```python
print(test_df["target"].value_counts())
```

<div class="k-default-codeblock">
```
0    413
1    348
Name: target, dtype: int64

```
</div>
---
## Convert data to a `tf.data.Dataset`


```python

def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["text"].to_numpy(), dataframe["target"].to_numpy())
    )
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)
```

---
## Downloading pretrained embeddings

The Universal Sentence Encoder embeddings encode text into high-dimensional vectors that can be
used for text classification, semantic similarity, clustering and other natural language
tasks. They're trained on a variety of data sources and a variety of tasks. Their input is
variable-length English text and their output is a 512 dimensional vector.

To learn more about these pretrained embeddings, see
[Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4).


```python
sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4"
)
```

---
## Creating our models

We create two models. In the first model (model_1) raw text will be first encoded via
pretrained embeddings and then passed to a Gradient Boosted Tree model for
classification. In the second model (model_2) raw text will be directly passed to
the Gradient Boosted Trees model.

Building model_1


```python
inputs = layers.Input(shape=(), dtype=tf.string)
outputs = sentence_encoder_layer(inputs)
preprocessor = keras.Model(inputs=inputs, outputs=outputs)
model_1 = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)
```

<div class="k-default-codeblock">
```
Use /tmp/tmpsp7fmsyk as temporary training directory

```
</div>
Building model_2


```python
model_2 = tfdf.keras.GradientBoostedTreesModel()
```

<div class="k-default-codeblock">
```
Use /tmp/tmpl0zj3vw0 as temporary training directory

```
</div>
---
## Train the models

We compile our model by passing the metrics `Accuracy`, `Recall`, `Precision` and
`AUC`. When it comes to the loss, TF-DF automatically detects the best loss for the task
(Classification or regression). It is printed in the model summary.

Also, because they're batch-training models rather than mini-batch gradient descent models,
TF-DF models do not need a validation dataset to monitor overfitting, or to stop
training early. Some algorithms do not use a validation dataset (e.g. Random Forest)
while some others do (e.g. Gradient Boosted Trees). If a validation dataset is
needed, it will be extracted automatically from the training dataset.


```python
# Compiling model_1
model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_1.fit(train_ds)

# Compiling model_2
model_2.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_2.fit(train_ds)
```

<div class="k-default-codeblock">
```
Reading training dataset...
Training dataset read in 0:00:06.473683. Found 6852 examples.
Training model...
Model trained in 0:00:41.461477
Compiling model...

[INFO kernel.cc:1176] Loading model from path /tmp/tmpsp7fmsyk/model/ with prefix 20297ba36a694abd
[INFO abstract_model.cc:1248] Engine "GradientBoostedTreesQuickScorerExtended" built
[INFO kernel.cc:1022] Use fast generic engine

WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7fe0aaefb5b0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7fe0aaefb5b0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

WARNING: AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7fe0aaefb5b0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
Model compiled.
Reading training dataset...
Training dataset read in 0:00:00.087930. Found 6852 examples.
Training model...
Model trained in 0:00:00.367492
Compiling model...

[INFO kernel.cc:1176] Loading model from path /tmp/tmpl0zj3vw0/model/ with prefix a03b7a91241248af
[INFO kernel.cc:1022] Use fast generic engine

Model compiled.

<keras.callbacks.History at 0x7fe09ded1b40>

```
</div>
Prints training logs of model_1


```python
logs_1 = model_1.make_inspector().training_logs()
print(logs_1)
```

<div class="k-default-codeblock">
```
[TrainLog(num_trees=1, evaluation=Evaluation(num_examples=None, accuracy=0.5656716227531433, loss=1.3077609539031982, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=2, evaluation=Evaluation(num_examples=None, accuracy=0.6865671873092651, loss=1.2614209651947021, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=3, evaluation=Evaluation(num_examples=None, accuracy=0.7417910695075989, loss=1.2177786827087402, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=4, evaluation=Evaluation(num_examples=None, accuracy=0.7701492309570312, loss=1.1783922910690308, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=5, evaluation=Evaluation(num_examples=None, accuracy=0.7776119112968445, loss=1.1459333896636963, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=6, evaluation=Evaluation(num_examples=None, accuracy=0.7835820913314819, loss=1.1136521100997925, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=7, evaluation=Evaluation(num_examples=None, accuracy=0.7970149517059326, loss=1.090031385421753, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=8, evaluation=Evaluation(num_examples=None, accuracy=0.8029850721359253, loss=1.0662610530853271, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=9, evaluation=Evaluation(num_examples=None, accuracy=0.8029850721359253, loss=1.0432935953140259, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=10, evaluation=Evaluation(num_examples=None, accuracy=0.8014925122261047, loss=1.0251210927963257, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=11, evaluation=Evaluation(num_examples=None, accuracy=0.8074626922607422, loss=1.0080381631851196, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=12, evaluation=Evaluation(num_examples=None, accuracy=0.8089552521705627, loss=0.9945542812347412, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=13, evaluation=Evaluation(num_examples=None, accuracy=0.8104477524757385, loss=0.9828057885169983, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=14, evaluation=Evaluation(num_examples=None, accuracy=0.8134328126907349, loss=0.9698775410652161, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=15, evaluation=Evaluation(num_examples=None, accuracy=0.8149253726005554, loss=0.9586601257324219, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=16, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.9511503577232361, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=17, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.9447258710861206, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=18, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.9361305832862854, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=19, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.927909255027771, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=20, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.9187226891517639, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=21, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.9130189418792725, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=22, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.9079279899597168, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=23, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.9030703902244568, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=24, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8996779322624207, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=25, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8957289457321167, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=26, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8929482698440552, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=27, evaluation=Evaluation(num_examples=None, accuracy=0.8223880529403687, loss=0.8866317868232727, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=28, evaluation=Evaluation(num_examples=None, accuracy=0.8179104328155518, loss=0.8825350403785706, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=29, evaluation=Evaluation(num_examples=None, accuracy=0.8223880529403687, loss=0.8786805868148804, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=30, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8753916025161743, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=31, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8716778755187988, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=32, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8713094592094421, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=33, evaluation=Evaluation(num_examples=None, accuracy=0.8223880529403687, loss=0.8663764595985413, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=34, evaluation=Evaluation(num_examples=None, accuracy=0.8179104328155518, loss=0.8634982109069824, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=35, evaluation=Evaluation(num_examples=None, accuracy=0.8179104328155518, loss=0.8620611429214478, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=36, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8590946197509766, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=37, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8560273051261902, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=38, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8528116345405579, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=39, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8532706499099731, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=40, evaluation=Evaluation(num_examples=None, accuracy=0.8223880529403687, loss=0.854380190372467, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=41, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8527907133102417, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=42, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8510931134223938, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=43, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8495559692382812, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=44, evaluation=Evaluation(num_examples=None, accuracy=0.8149253726005554, loss=0.8494127988815308, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=45, evaluation=Evaluation(num_examples=None, accuracy=0.8179104328155518, loss=0.8501511812210083, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=46, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8470982313156128, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=47, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.844197154045105, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=48, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.843916654586792, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=49, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.843466579914093, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=50, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8425527215003967, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=51, evaluation=Evaluation(num_examples=None, accuracy=0.8179104328155518, loss=0.8413452506065369, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=52, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.839697003364563, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=53, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8391197323799133, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=54, evaluation=Evaluation(num_examples=None, accuracy=0.8223880529403687, loss=0.8362838625907898, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=55, evaluation=Evaluation(num_examples=None, accuracy=0.8223880529403687, loss=0.8340254426002502, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=56, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8306224942207336, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=57, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8284339904785156, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=58, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8282331824302673, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=59, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8264115452766418, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=60, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8263145685195923, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=61, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8259852528572083, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=62, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.825827956199646, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=63, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8244388699531555, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=64, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8259344100952148, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=65, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.825884222984314, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=66, evaluation=Evaluation(num_examples=None, accuracy=0.8313432931900024, loss=0.8257853984832764, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=67, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8254727125167847, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=68, evaluation=Evaluation(num_examples=None, accuracy=0.8313432931900024, loss=0.8246610760688782, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=69, evaluation=Evaluation(num_examples=None, accuracy=0.8313432931900024, loss=0.8239343762397766, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=70, evaluation=Evaluation(num_examples=None, accuracy=0.8343283534049988, loss=0.8229695558547974, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=71, evaluation=Evaluation(num_examples=None, accuracy=0.8328357934951782, loss=0.8233013153076172, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=72, evaluation=Evaluation(num_examples=None, accuracy=0.8328357934951782, loss=0.8221727609634399, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=73, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8245336413383484, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=74, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8238235116004944, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=75, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.823878288269043, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=76, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8247055411338806, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=77, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8246185183525085, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=78, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.824006974697113, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=79, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8224233388900757, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=80, evaluation=Evaluation(num_examples=None, accuracy=0.8328357934951782, loss=0.821762204170227, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=81, evaluation=Evaluation(num_examples=None, accuracy=0.8343283534049988, loss=0.8203343749046326, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=82, evaluation=Evaluation(num_examples=None, accuracy=0.8313432931900024, loss=0.818412721157074, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=83, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8189152479171753, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=84, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8178151249885559, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=85, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8165677785873413, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=86, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8167412877082825, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=87, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8149023652076721, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=88, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8154326677322388, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=89, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8148661255836487, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=90, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8154740333557129, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=91, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8146263957023621, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=92, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8146853446960449, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=93, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.814373254776001, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=94, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8130836486816406, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=95, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8107629418373108, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=96, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8106600046157837, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=97, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8108977675437927, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=98, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8095542192459106, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=99, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8095746040344238, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=100, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.810272216796875, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=101, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8091635704040527, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=102, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8085538148880005, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=103, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.808313250541687, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=104, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8085122108459473, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=105, evaluation=Evaluation(num_examples=None, accuracy=0.8313432931900024, loss=0.8088634610176086, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=106, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.807800829410553, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=107, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8077428936958313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=108, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8071678280830383, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=109, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8068002462387085, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=110, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8077936172485352, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=111, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.807645320892334, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=112, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.808992862701416, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=113, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8098238110542297, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=114, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8099804520606995, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=115, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8084514141082764, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=116, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8086163401603699, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=117, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.808234453201294, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=118, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8080092072486877, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=119, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.806970477104187, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=120, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8083237409591675, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=121, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8073570728302002, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=122, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8080623745918274, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=123, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8079558610916138, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=124, evaluation=Evaluation(num_examples=None, accuracy=0.8223880529403687, loss=0.80722576379776, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=125, evaluation=Evaluation(num_examples=None, accuracy=0.8179104328155518, loss=0.8075857162475586, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=126, evaluation=Evaluation(num_examples=None, accuracy=0.8208954930305481, loss=0.8091554045677185, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=127, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8091134428977966, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=128, evaluation=Evaluation(num_examples=None, accuracy=0.8194029927253723, loss=0.8106313943862915, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=129, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8099302649497986, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=130, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8106984496116638, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=131, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.810023307800293, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=132, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8097068071365356, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=133, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8086251020431519, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=134, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8085694313049316, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=135, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8075850605964661, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=136, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8077738881111145, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=137, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8067774772644043, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=138, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8076777458190918, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=139, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8073931932449341, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=140, evaluation=Evaluation(num_examples=None, accuracy=0.8238806128501892, loss=0.8078530430793762, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=141, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8086029887199402, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=142, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8084178566932678, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=143, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8078324198722839, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=144, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8077776432037354, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=145, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8074193000793457, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=146, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8083519339561462, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=147, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8101204633712769, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=148, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8095722198486328, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=149, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8094549179077148, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=150, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8092751502990723, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=151, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8095303773880005, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=152, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.808938205242157, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=153, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8088335394859314, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=154, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8086864948272705, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=155, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.8081754446029663, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=156, evaluation=Evaluation(num_examples=None, accuracy=0.8283582329750061, loss=0.808414876461029, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=157, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8094819188117981, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=158, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8095358610153198, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=159, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8095360994338989, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=160, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8099076747894287, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=161, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8096389174461365, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=162, evaluation=Evaluation(num_examples=None, accuracy=0.825373113155365, loss=0.8101357221603394, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=163, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8095107078552246, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=164, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8108780980110168, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=165, evaluation=Evaluation(num_examples=None, accuracy=0.8268656730651855, loss=0.8121252059936523, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=166, evaluation=Evaluation(num_examples=None, accuracy=0.8298507332801819, loss=0.8118909001350403, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=167, evaluation=Evaluation(num_examples=None, accuracy=0.8313432931900024, loss=0.8114330768585205, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None))]

```
</div>
Prints training logs of model_2


```python
logs_2 = model_2.make_inspector().training_logs()
print(logs_2)
```

<div class="k-default-codeblock">
```
[TrainLog(num_trees=1, evaluation=Evaluation(num_examples=None, accuracy=0.5656716227531433, loss=1.3685380220413208, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=2, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3679413795471191, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=3, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3674567937850952, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=4, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3670554161071777, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=5, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3667176961898804, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=6, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.366430401802063, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=7, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3661839962005615, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=8, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3659707307815552, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=9, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3657851219177246, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=10, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.365623116493225, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=11, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3654807806015015, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=12, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.365355134010315, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=13, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3652442693710327, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=14, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3651456832885742, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=15, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.365058422088623, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=16, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3649805784225464, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=17, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3649111986160278, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=18, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3648490905761719, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=19, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3647936582565308, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=20, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3647440671920776, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=21, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3646996021270752, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=22, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3646595478057861, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=23, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3646236658096313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=24, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364591360092163, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=25, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3645622730255127, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=26, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3645362854003906, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=27, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.36451256275177, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=28, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364491581916809, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=29, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3644723892211914, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=30, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364455223083496, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=31, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3644394874572754, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=32, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3644256591796875, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=33, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3644130229949951, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=34, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3644014596939087, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=35, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643912076950073, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=36, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643817901611328, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=37, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643734455108643, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=38, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364365816116333, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=39, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643587827682495, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=40, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364352822303772, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=41, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643471002578735, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=42, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643420934677124, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=43, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.36433744430542, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=44, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643333911895752, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=45, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643295764923096, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=46, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643261194229126, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=47, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643230199813843, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=48, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643202781677246, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=49, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643178939819336, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=50, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643155097961426, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=51, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643133640289307, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=52, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364311695098877, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=53, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643099069595337, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=54, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643083572387695, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=55, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643070459365845, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=56, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643057346343994, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=57, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643046617507935, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=58, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643035888671875, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=59, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643028736114502, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=60, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643019199371338, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=61, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364301085472107, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=62, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3643003702163696, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=63, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642996549606323, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=64, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642994165420532, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=65, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364298701286316, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=66, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642982244491577, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=67, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642977476119995, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=68, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642975091934204, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=69, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642971515655518, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=70, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642969131469727, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=71, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364296555519104, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=72, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642961978912354, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=73, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642959594726562, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=74, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642958402633667, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=75, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642957210540771, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=76, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642956018447876, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=77, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364295244216919, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=78, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364295244216919, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=79, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642951250076294, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=80, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642950057983398, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=81, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642948865890503, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=82, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642948865890503, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=83, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642946481704712, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=84, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642946481704712, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=85, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642946481704712, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=86, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642945289611816, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=87, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294409751892, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=88, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294409751892, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=89, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294409751892, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=90, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294409751892, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=91, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642942905426025, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=92, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=93, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=94, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=95, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=96, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=97, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=98, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642942905426025, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=99, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=100, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=101, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=102, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.364294171333313, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=103, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=104, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=105, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=106, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=107, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=108, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=109, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=110, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=111, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=112, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=113, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=114, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=115, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=116, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642940521240234, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=117, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=118, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=119, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=120, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=121, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=122, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=123, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=124, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=125, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=126, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=127, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=128, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=129, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=130, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=131, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=132, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=133, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=134, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=135, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=136, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=137, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=138, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=139, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=140, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=141, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=142, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=143, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=144, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=145, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=146, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)), TrainLog(num_trees=147, evaluation=Evaluation(num_examples=None, accuracy=0.5686567425727844, loss=1.3642939329147339, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None))]

```
</div>
The model.summary() method prints a variety of information about your decision tree model, including model type, task, input features, and feature importance.


```python
print("model_1 summary: ")
print(model_1.summary())
print()
print("model_2 summary: ")
print(model_2.summary())
```

<div class="k-default-codeblock">
```
model_1 summary: 
Model: "gradient_boosted_trees_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 model (Functional)          (None, 512)               256797824 
                                                                 
=================================================================
Total params: 256,797,825
Trainable params: 0
Non-trainable params: 256,797,825
_________________________________________________________________
Type: "GRADIENT_BOOSTED_TREES"
Task: CLASSIFICATION
Label: "__LABEL"
```
</div>
    
<div class="k-default-codeblock">
```
Input Features (512):
	model/keras_layer/StatefulPartitionedCall:0.0
	model/keras_layer/StatefulPartitionedCall:0.1
	model/keras_layer/StatefulPartitionedCall:0.10
	model/keras_layer/StatefulPartitionedCall:0.100
	model/keras_layer/StatefulPartitionedCall:0.101
	model/keras_layer/StatefulPartitionedCall:0.102
	model/keras_layer/StatefulPartitionedCall:0.103
	model/keras_layer/StatefulPartitionedCall:0.104
	model/keras_layer/StatefulPartitionedCall:0.105
	model/keras_layer/StatefulPartitionedCall:0.106
	model/keras_layer/StatefulPartitionedCall:0.107
	model/keras_layer/StatefulPartitionedCall:0.108
	model/keras_layer/StatefulPartitionedCall:0.109
	model/keras_layer/StatefulPartitionedCall:0.11
	model/keras_layer/StatefulPartitionedCall:0.110
	model/keras_layer/StatefulPartitionedCall:0.111
	model/keras_layer/StatefulPartitionedCall:0.112
	model/keras_layer/StatefulPartitionedCall:0.113
	model/keras_layer/StatefulPartitionedCall:0.114
	model/keras_layer/StatefulPartitionedCall:0.115
	model/keras_layer/StatefulPartitionedCall:0.116
	model/keras_layer/StatefulPartitionedCall:0.117
	model/keras_layer/StatefulPartitionedCall:0.118
	model/keras_layer/StatefulPartitionedCall:0.119
	model/keras_layer/StatefulPartitionedCall:0.12
	model/keras_layer/StatefulPartitionedCall:0.120
	model/keras_layer/StatefulPartitionedCall:0.121
	model/keras_layer/StatefulPartitionedCall:0.122
	model/keras_layer/StatefulPartitionedCall:0.123
	model/keras_layer/StatefulPartitionedCall:0.124
	model/keras_layer/StatefulPartitionedCall:0.125
	model/keras_layer/StatefulPartitionedCall:0.126
	model/keras_layer/StatefulPartitionedCall:0.127
	model/keras_layer/StatefulPartitionedCall:0.128
	model/keras_layer/StatefulPartitionedCall:0.129
	model/keras_layer/StatefulPartitionedCall:0.13
	model/keras_layer/StatefulPartitionedCall:0.130
	model/keras_layer/StatefulPartitionedCall:0.131
	model/keras_layer/StatefulPartitionedCall:0.132
	model/keras_layer/StatefulPartitionedCall:0.133
	model/keras_layer/StatefulPartitionedCall:0.134
	model/keras_layer/StatefulPartitionedCall:0.135
	model/keras_layer/StatefulPartitionedCall:0.136
	model/keras_layer/StatefulPartitionedCall:0.137
	model/keras_layer/StatefulPartitionedCall:0.138
	model/keras_layer/StatefulPartitionedCall:0.139
	model/keras_layer/StatefulPartitionedCall:0.14
	model/keras_layer/StatefulPartitionedCall:0.140
	model/keras_layer/StatefulPartitionedCall:0.141
	model/keras_layer/StatefulPartitionedCall:0.142
	model/keras_layer/StatefulPartitionedCall:0.143
	model/keras_layer/StatefulPartitionedCall:0.144
	model/keras_layer/StatefulPartitionedCall:0.145
	model/keras_layer/StatefulPartitionedCall:0.146
	model/keras_layer/StatefulPartitionedCall:0.147
	model/keras_layer/StatefulPartitionedCall:0.148
	model/keras_layer/StatefulPartitionedCall:0.149
	model/keras_layer/StatefulPartitionedCall:0.15
	model/keras_layer/StatefulPartitionedCall:0.150
	model/keras_layer/StatefulPartitionedCall:0.151
	model/keras_layer/StatefulPartitionedCall:0.152
	model/keras_layer/StatefulPartitionedCall:0.153
	model/keras_layer/StatefulPartitionedCall:0.154
	model/keras_layer/StatefulPartitionedCall:0.155
	model/keras_layer/StatefulPartitionedCall:0.156
	model/keras_layer/StatefulPartitionedCall:0.157
	model/keras_layer/StatefulPartitionedCall:0.158
	model/keras_layer/StatefulPartitionedCall:0.159
	model/keras_layer/StatefulPartitionedCall:0.16
	model/keras_layer/StatefulPartitionedCall:0.160
	model/keras_layer/StatefulPartitionedCall:0.161
	model/keras_layer/StatefulPartitionedCall:0.162
	model/keras_layer/StatefulPartitionedCall:0.163
	model/keras_layer/StatefulPartitionedCall:0.164
	model/keras_layer/StatefulPartitionedCall:0.165
	model/keras_layer/StatefulPartitionedCall:0.166
	model/keras_layer/StatefulPartitionedCall:0.167
	model/keras_layer/StatefulPartitionedCall:0.168
	model/keras_layer/StatefulPartitionedCall:0.169
	model/keras_layer/StatefulPartitionedCall:0.17
	model/keras_layer/StatefulPartitionedCall:0.170
	model/keras_layer/StatefulPartitionedCall:0.171
	model/keras_layer/StatefulPartitionedCall:0.172
	model/keras_layer/StatefulPartitionedCall:0.173
	model/keras_layer/StatefulPartitionedCall:0.174
	model/keras_layer/StatefulPartitionedCall:0.175
	model/keras_layer/StatefulPartitionedCall:0.176
	model/keras_layer/StatefulPartitionedCall:0.177
	model/keras_layer/StatefulPartitionedCall:0.178
	model/keras_layer/StatefulPartitionedCall:0.179
	model/keras_layer/StatefulPartitionedCall:0.18
	model/keras_layer/StatefulPartitionedCall:0.180
	model/keras_layer/StatefulPartitionedCall:0.181
	model/keras_layer/StatefulPartitionedCall:0.182
	model/keras_layer/StatefulPartitionedCall:0.183
	model/keras_layer/StatefulPartitionedCall:0.184
	model/keras_layer/StatefulPartitionedCall:0.185
	model/keras_layer/StatefulPartitionedCall:0.186
	model/keras_layer/StatefulPartitionedCall:0.187
	model/keras_layer/StatefulPartitionedCall:0.188
	model/keras_layer/StatefulPartitionedCall:0.189
	model/keras_layer/StatefulPartitionedCall:0.19
	model/keras_layer/StatefulPartitionedCall:0.190
	model/keras_layer/StatefulPartitionedCall:0.191
	model/keras_layer/StatefulPartitionedCall:0.192
	model/keras_layer/StatefulPartitionedCall:0.193
	model/keras_layer/StatefulPartitionedCall:0.194
	model/keras_layer/StatefulPartitionedCall:0.195
	model/keras_layer/StatefulPartitionedCall:0.196
	model/keras_layer/StatefulPartitionedCall:0.197
	model/keras_layer/StatefulPartitionedCall:0.198
	model/keras_layer/StatefulPartitionedCall:0.199
	model/keras_layer/StatefulPartitionedCall:0.2
	model/keras_layer/StatefulPartitionedCall:0.20
	model/keras_layer/StatefulPartitionedCall:0.200
	model/keras_layer/StatefulPartitionedCall:0.201
	model/keras_layer/StatefulPartitionedCall:0.202
	model/keras_layer/StatefulPartitionedCall:0.203
	model/keras_layer/StatefulPartitionedCall:0.204
	model/keras_layer/StatefulPartitionedCall:0.205
	model/keras_layer/StatefulPartitionedCall:0.206
	model/keras_layer/StatefulPartitionedCall:0.207
	model/keras_layer/StatefulPartitionedCall:0.208
	model/keras_layer/StatefulPartitionedCall:0.209
	model/keras_layer/StatefulPartitionedCall:0.21
	model/keras_layer/StatefulPartitionedCall:0.210
	model/keras_layer/StatefulPartitionedCall:0.211
	model/keras_layer/StatefulPartitionedCall:0.212
	model/keras_layer/StatefulPartitionedCall:0.213
	model/keras_layer/StatefulPartitionedCall:0.214
	model/keras_layer/StatefulPartitionedCall:0.215
	model/keras_layer/StatefulPartitionedCall:0.216
	model/keras_layer/StatefulPartitionedCall:0.217
	model/keras_layer/StatefulPartitionedCall:0.218
	model/keras_layer/StatefulPartitionedCall:0.219
	model/keras_layer/StatefulPartitionedCall:0.22
	model/keras_layer/StatefulPartitionedCall:0.220
	model/keras_layer/StatefulPartitionedCall:0.221
	model/keras_layer/StatefulPartitionedCall:0.222
	model/keras_layer/StatefulPartitionedCall:0.223
	model/keras_layer/StatefulPartitionedCall:0.224
	model/keras_layer/StatefulPartitionedCall:0.225
	model/keras_layer/StatefulPartitionedCall:0.226
	model/keras_layer/StatefulPartitionedCall:0.227
	model/keras_layer/StatefulPartitionedCall:0.228
	model/keras_layer/StatefulPartitionedCall:0.229
	model/keras_layer/StatefulPartitionedCall:0.23
	model/keras_layer/StatefulPartitionedCall:0.230
	model/keras_layer/StatefulPartitionedCall:0.231
	model/keras_layer/StatefulPartitionedCall:0.232
	model/keras_layer/StatefulPartitionedCall:0.233
	model/keras_layer/StatefulPartitionedCall:0.234
	model/keras_layer/StatefulPartitionedCall:0.235
	model/keras_layer/StatefulPartitionedCall:0.236
	model/keras_layer/StatefulPartitionedCall:0.237
	model/keras_layer/StatefulPartitionedCall:0.238
	model/keras_layer/StatefulPartitionedCall:0.239
	model/keras_layer/StatefulPartitionedCall:0.24
	model/keras_layer/StatefulPartitionedCall:0.240
	model/keras_layer/StatefulPartitionedCall:0.241
	model/keras_layer/StatefulPartitionedCall:0.242
	model/keras_layer/StatefulPartitionedCall:0.243
	model/keras_layer/StatefulPartitionedCall:0.244
	model/keras_layer/StatefulPartitionedCall:0.245
	model/keras_layer/StatefulPartitionedCall:0.246
	model/keras_layer/StatefulPartitionedCall:0.247
	model/keras_layer/StatefulPartitionedCall:0.248
	model/keras_layer/StatefulPartitionedCall:0.249
	model/keras_layer/StatefulPartitionedCall:0.25
	model/keras_layer/StatefulPartitionedCall:0.250
	model/keras_layer/StatefulPartitionedCall:0.251
	model/keras_layer/StatefulPartitionedCall:0.252
	model/keras_layer/StatefulPartitionedCall:0.253
	model/keras_layer/StatefulPartitionedCall:0.254
	model/keras_layer/StatefulPartitionedCall:0.255
	model/keras_layer/StatefulPartitionedCall:0.256
	model/keras_layer/StatefulPartitionedCall:0.257
	model/keras_layer/StatefulPartitionedCall:0.258
	model/keras_layer/StatefulPartitionedCall:0.259
	model/keras_layer/StatefulPartitionedCall:0.26
	model/keras_layer/StatefulPartitionedCall:0.260
	model/keras_layer/StatefulPartitionedCall:0.261
	model/keras_layer/StatefulPartitionedCall:0.262
	model/keras_layer/StatefulPartitionedCall:0.263
	model/keras_layer/StatefulPartitionedCall:0.264
	model/keras_layer/StatefulPartitionedCall:0.265
	model/keras_layer/StatefulPartitionedCall:0.266
	model/keras_layer/StatefulPartitionedCall:0.267
	model/keras_layer/StatefulPartitionedCall:0.268
	model/keras_layer/StatefulPartitionedCall:0.269
	model/keras_layer/StatefulPartitionedCall:0.27
	model/keras_layer/StatefulPartitionedCall:0.270
	model/keras_layer/StatefulPartitionedCall:0.271
	model/keras_layer/StatefulPartitionedCall:0.272
	model/keras_layer/StatefulPartitionedCall:0.273
	model/keras_layer/StatefulPartitionedCall:0.274
	model/keras_layer/StatefulPartitionedCall:0.275
	model/keras_layer/StatefulPartitionedCall:0.276
	model/keras_layer/StatefulPartitionedCall:0.277
	model/keras_layer/StatefulPartitionedCall:0.278
	model/keras_layer/StatefulPartitionedCall:0.279
	model/keras_layer/StatefulPartitionedCall:0.28
	model/keras_layer/StatefulPartitionedCall:0.280
	model/keras_layer/StatefulPartitionedCall:0.281
	model/keras_layer/StatefulPartitionedCall:0.282
	model/keras_layer/StatefulPartitionedCall:0.283
	model/keras_layer/StatefulPartitionedCall:0.284
	model/keras_layer/StatefulPartitionedCall:0.285
	model/keras_layer/StatefulPartitionedCall:0.286
	model/keras_layer/StatefulPartitionedCall:0.287
	model/keras_layer/StatefulPartitionedCall:0.288
	model/keras_layer/StatefulPartitionedCall:0.289
	model/keras_layer/StatefulPartitionedCall:0.29
	model/keras_layer/StatefulPartitionedCall:0.290
	model/keras_layer/StatefulPartitionedCall:0.291
	model/keras_layer/StatefulPartitionedCall:0.292
	model/keras_layer/StatefulPartitionedCall:0.293
	model/keras_layer/StatefulPartitionedCall:0.294
	model/keras_layer/StatefulPartitionedCall:0.295
	model/keras_layer/StatefulPartitionedCall:0.296
	model/keras_layer/StatefulPartitionedCall:0.297
	model/keras_layer/StatefulPartitionedCall:0.298
	model/keras_layer/StatefulPartitionedCall:0.299
	model/keras_layer/StatefulPartitionedCall:0.3
	model/keras_layer/StatefulPartitionedCall:0.30
	model/keras_layer/StatefulPartitionedCall:0.300
	model/keras_layer/StatefulPartitionedCall:0.301
	model/keras_layer/StatefulPartitionedCall:0.302
	model/keras_layer/StatefulPartitionedCall:0.303
	model/keras_layer/StatefulPartitionedCall:0.304
	model/keras_layer/StatefulPartitionedCall:0.305
	model/keras_layer/StatefulPartitionedCall:0.306
	model/keras_layer/StatefulPartitionedCall:0.307
	model/keras_layer/StatefulPartitionedCall:0.308
	model/keras_layer/StatefulPartitionedCall:0.309
	model/keras_layer/StatefulPartitionedCall:0.31
	model/keras_layer/StatefulPartitionedCall:0.310
	model/keras_layer/StatefulPartitionedCall:0.311
	model/keras_layer/StatefulPartitionedCall:0.312
	model/keras_layer/StatefulPartitionedCall:0.313
	model/keras_layer/StatefulPartitionedCall:0.314
	model/keras_layer/StatefulPartitionedCall:0.315
	model/keras_layer/StatefulPartitionedCall:0.316
	model/keras_layer/StatefulPartitionedCall:0.317
	model/keras_layer/StatefulPartitionedCall:0.318
	model/keras_layer/StatefulPartitionedCall:0.319
	model/keras_layer/StatefulPartitionedCall:0.32
	model/keras_layer/StatefulPartitionedCall:0.320
	model/keras_layer/StatefulPartitionedCall:0.321
	model/keras_layer/StatefulPartitionedCall:0.322
	model/keras_layer/StatefulPartitionedCall:0.323
	model/keras_layer/StatefulPartitionedCall:0.324
	model/keras_layer/StatefulPartitionedCall:0.325
	model/keras_layer/StatefulPartitionedCall:0.326
	model/keras_layer/StatefulPartitionedCall:0.327
	model/keras_layer/StatefulPartitionedCall:0.328
	model/keras_layer/StatefulPartitionedCall:0.329
	model/keras_layer/StatefulPartitionedCall:0.33
	model/keras_layer/StatefulPartitionedCall:0.330
	model/keras_layer/StatefulPartitionedCall:0.331
	model/keras_layer/StatefulPartitionedCall:0.332
	model/keras_layer/StatefulPartitionedCall:0.333
	model/keras_layer/StatefulPartitionedCall:0.334
	model/keras_layer/StatefulPartitionedCall:0.335
	model/keras_layer/StatefulPartitionedCall:0.336
	model/keras_layer/StatefulPartitionedCall:0.337
	model/keras_layer/StatefulPartitionedCall:0.338
	model/keras_layer/StatefulPartitionedCall:0.339
	model/keras_layer/StatefulPartitionedCall:0.34
	model/keras_layer/StatefulPartitionedCall:0.340
	model/keras_layer/StatefulPartitionedCall:0.341
	model/keras_layer/StatefulPartitionedCall:0.342
	model/keras_layer/StatefulPartitionedCall:0.343
	model/keras_layer/StatefulPartitionedCall:0.344
	model/keras_layer/StatefulPartitionedCall:0.345
	model/keras_layer/StatefulPartitionedCall:0.346
	model/keras_layer/StatefulPartitionedCall:0.347
	model/keras_layer/StatefulPartitionedCall:0.348
	model/keras_layer/StatefulPartitionedCall:0.349
	model/keras_layer/StatefulPartitionedCall:0.35
	model/keras_layer/StatefulPartitionedCall:0.350
	model/keras_layer/StatefulPartitionedCall:0.351
	model/keras_layer/StatefulPartitionedCall:0.352
	model/keras_layer/StatefulPartitionedCall:0.353
	model/keras_layer/StatefulPartitionedCall:0.354
	model/keras_layer/StatefulPartitionedCall:0.355
	model/keras_layer/StatefulPartitionedCall:0.356
	model/keras_layer/StatefulPartitionedCall:0.357
	model/keras_layer/StatefulPartitionedCall:0.358
	model/keras_layer/StatefulPartitionedCall:0.359
	model/keras_layer/StatefulPartitionedCall:0.36
	model/keras_layer/StatefulPartitionedCall:0.360
	model/keras_layer/StatefulPartitionedCall:0.361
	model/keras_layer/StatefulPartitionedCall:0.362
	model/keras_layer/StatefulPartitionedCall:0.363
	model/keras_layer/StatefulPartitionedCall:0.364
	model/keras_layer/StatefulPartitionedCall:0.365
	model/keras_layer/StatefulPartitionedCall:0.366
	model/keras_layer/StatefulPartitionedCall:0.367
	model/keras_layer/StatefulPartitionedCall:0.368
	model/keras_layer/StatefulPartitionedCall:0.369
	model/keras_layer/StatefulPartitionedCall:0.37
	model/keras_layer/StatefulPartitionedCall:0.370
	model/keras_layer/StatefulPartitionedCall:0.371
	model/keras_layer/StatefulPartitionedCall:0.372
	model/keras_layer/StatefulPartitionedCall:0.373
	model/keras_layer/StatefulPartitionedCall:0.374
	model/keras_layer/StatefulPartitionedCall:0.375
	model/keras_layer/StatefulPartitionedCall:0.376
	model/keras_layer/StatefulPartitionedCall:0.377
	model/keras_layer/StatefulPartitionedCall:0.378
	model/keras_layer/StatefulPartitionedCall:0.379
	model/keras_layer/StatefulPartitionedCall:0.38
	model/keras_layer/StatefulPartitionedCall:0.380
	model/keras_layer/StatefulPartitionedCall:0.381
	model/keras_layer/StatefulPartitionedCall:0.382
	model/keras_layer/StatefulPartitionedCall:0.383
	model/keras_layer/StatefulPartitionedCall:0.384
	model/keras_layer/StatefulPartitionedCall:0.385
	model/keras_layer/StatefulPartitionedCall:0.386
	model/keras_layer/StatefulPartitionedCall:0.387
	model/keras_layer/StatefulPartitionedCall:0.388
	model/keras_layer/StatefulPartitionedCall:0.389
	model/keras_layer/StatefulPartitionedCall:0.39
	model/keras_layer/StatefulPartitionedCall:0.390
	model/keras_layer/StatefulPartitionedCall:0.391
	model/keras_layer/StatefulPartitionedCall:0.392
	model/keras_layer/StatefulPartitionedCall:0.393
	model/keras_layer/StatefulPartitionedCall:0.394
	model/keras_layer/StatefulPartitionedCall:0.395
	model/keras_layer/StatefulPartitionedCall:0.396
	model/keras_layer/StatefulPartitionedCall:0.397
	model/keras_layer/StatefulPartitionedCall:0.398
	model/keras_layer/StatefulPartitionedCall:0.399
	model/keras_layer/StatefulPartitionedCall:0.4
	model/keras_layer/StatefulPartitionedCall:0.40
	model/keras_layer/StatefulPartitionedCall:0.400
	model/keras_layer/StatefulPartitionedCall:0.401
	model/keras_layer/StatefulPartitionedCall:0.402
	model/keras_layer/StatefulPartitionedCall:0.403
	model/keras_layer/StatefulPartitionedCall:0.404
	model/keras_layer/StatefulPartitionedCall:0.405
	model/keras_layer/StatefulPartitionedCall:0.406
	model/keras_layer/StatefulPartitionedCall:0.407
	model/keras_layer/StatefulPartitionedCall:0.408
	model/keras_layer/StatefulPartitionedCall:0.409
	model/keras_layer/StatefulPartitionedCall:0.41
	model/keras_layer/StatefulPartitionedCall:0.410
	model/keras_layer/StatefulPartitionedCall:0.411
	model/keras_layer/StatefulPartitionedCall:0.412
	model/keras_layer/StatefulPartitionedCall:0.413
	model/keras_layer/StatefulPartitionedCall:0.414
	model/keras_layer/StatefulPartitionedCall:0.415
	model/keras_layer/StatefulPartitionedCall:0.416
	model/keras_layer/StatefulPartitionedCall:0.417
	model/keras_layer/StatefulPartitionedCall:0.418
	model/keras_layer/StatefulPartitionedCall:0.419
	model/keras_layer/StatefulPartitionedCall:0.42
	model/keras_layer/StatefulPartitionedCall:0.420
	model/keras_layer/StatefulPartitionedCall:0.421
	model/keras_layer/StatefulPartitionedCall:0.422
	model/keras_layer/StatefulPartitionedCall:0.423
	model/keras_layer/StatefulPartitionedCall:0.424
	model/keras_layer/StatefulPartitionedCall:0.425
	model/keras_layer/StatefulPartitionedCall:0.426
	model/keras_layer/StatefulPartitionedCall:0.427
	model/keras_layer/StatefulPartitionedCall:0.428
	model/keras_layer/StatefulPartitionedCall:0.429
	model/keras_layer/StatefulPartitionedCall:0.43
	model/keras_layer/StatefulPartitionedCall:0.430
	model/keras_layer/StatefulPartitionedCall:0.431
	model/keras_layer/StatefulPartitionedCall:0.432
	model/keras_layer/StatefulPartitionedCall:0.433
	model/keras_layer/StatefulPartitionedCall:0.434
	model/keras_layer/StatefulPartitionedCall:0.435
	model/keras_layer/StatefulPartitionedCall:0.436
	model/keras_layer/StatefulPartitionedCall:0.437
	model/keras_layer/StatefulPartitionedCall:0.438
	model/keras_layer/StatefulPartitionedCall:0.439
	model/keras_layer/StatefulPartitionedCall:0.44
	model/keras_layer/StatefulPartitionedCall:0.440
	model/keras_layer/StatefulPartitionedCall:0.441
	model/keras_layer/StatefulPartitionedCall:0.442
	model/keras_layer/StatefulPartitionedCall:0.443
	model/keras_layer/StatefulPartitionedCall:0.444
	model/keras_layer/StatefulPartitionedCall:0.445
	model/keras_layer/StatefulPartitionedCall:0.446
	model/keras_layer/StatefulPartitionedCall:0.447
	model/keras_layer/StatefulPartitionedCall:0.448
	model/keras_layer/StatefulPartitionedCall:0.449
	model/keras_layer/StatefulPartitionedCall:0.45
	model/keras_layer/StatefulPartitionedCall:0.450
	model/keras_layer/StatefulPartitionedCall:0.451
	model/keras_layer/StatefulPartitionedCall:0.452
	model/keras_layer/StatefulPartitionedCall:0.453
	model/keras_layer/StatefulPartitionedCall:0.454
	model/keras_layer/StatefulPartitionedCall:0.455
	model/keras_layer/StatefulPartitionedCall:0.456
	model/keras_layer/StatefulPartitionedCall:0.457
	model/keras_layer/StatefulPartitionedCall:0.458
	model/keras_layer/StatefulPartitionedCall:0.459
	model/keras_layer/StatefulPartitionedCall:0.46
	model/keras_layer/StatefulPartitionedCall:0.460
	model/keras_layer/StatefulPartitionedCall:0.461
	model/keras_layer/StatefulPartitionedCall:0.462
	model/keras_layer/StatefulPartitionedCall:0.463
	model/keras_layer/StatefulPartitionedCall:0.464
	model/keras_layer/StatefulPartitionedCall:0.465
	model/keras_layer/StatefulPartitionedCall:0.466
	model/keras_layer/StatefulPartitionedCall:0.467
	model/keras_layer/StatefulPartitionedCall:0.468
	model/keras_layer/StatefulPartitionedCall:0.469
	model/keras_layer/StatefulPartitionedCall:0.47
	model/keras_layer/StatefulPartitionedCall:0.470
	model/keras_layer/StatefulPartitionedCall:0.471
	model/keras_layer/StatefulPartitionedCall:0.472
	model/keras_layer/StatefulPartitionedCall:0.473
	model/keras_layer/StatefulPartitionedCall:0.474
	model/keras_layer/StatefulPartitionedCall:0.475
	model/keras_layer/StatefulPartitionedCall:0.476
	model/keras_layer/StatefulPartitionedCall:0.477
	model/keras_layer/StatefulPartitionedCall:0.478
	model/keras_layer/StatefulPartitionedCall:0.479
	model/keras_layer/StatefulPartitionedCall:0.48
	model/keras_layer/StatefulPartitionedCall:0.480
	model/keras_layer/StatefulPartitionedCall:0.481
	model/keras_layer/StatefulPartitionedCall:0.482
	model/keras_layer/StatefulPartitionedCall:0.483
	model/keras_layer/StatefulPartitionedCall:0.484
	model/keras_layer/StatefulPartitionedCall:0.485
	model/keras_layer/StatefulPartitionedCall:0.486
	model/keras_layer/StatefulPartitionedCall:0.487
	model/keras_layer/StatefulPartitionedCall:0.488
	model/keras_layer/StatefulPartitionedCall:0.489
	model/keras_layer/StatefulPartitionedCall:0.49
	model/keras_layer/StatefulPartitionedCall:0.490
	model/keras_layer/StatefulPartitionedCall:0.491
	model/keras_layer/StatefulPartitionedCall:0.492
	model/keras_layer/StatefulPartitionedCall:0.493
	model/keras_layer/StatefulPartitionedCall:0.494
	model/keras_layer/StatefulPartitionedCall:0.495
	model/keras_layer/StatefulPartitionedCall:0.496
	model/keras_layer/StatefulPartitionedCall:0.497
	model/keras_layer/StatefulPartitionedCall:0.498
	model/keras_layer/StatefulPartitionedCall:0.499
	model/keras_layer/StatefulPartitionedCall:0.5
	model/keras_layer/StatefulPartitionedCall:0.50
	model/keras_layer/StatefulPartitionedCall:0.500
	model/keras_layer/StatefulPartitionedCall:0.501
	model/keras_layer/StatefulPartitionedCall:0.502
	model/keras_layer/StatefulPartitionedCall:0.503
	model/keras_layer/StatefulPartitionedCall:0.504
	model/keras_layer/StatefulPartitionedCall:0.505
	model/keras_layer/StatefulPartitionedCall:0.506
	model/keras_layer/StatefulPartitionedCall:0.507
	model/keras_layer/StatefulPartitionedCall:0.508
	model/keras_layer/StatefulPartitionedCall:0.509
	model/keras_layer/StatefulPartitionedCall:0.51
	model/keras_layer/StatefulPartitionedCall:0.510
	model/keras_layer/StatefulPartitionedCall:0.511
	model/keras_layer/StatefulPartitionedCall:0.52
	model/keras_layer/StatefulPartitionedCall:0.53
	model/keras_layer/StatefulPartitionedCall:0.54
	model/keras_layer/StatefulPartitionedCall:0.55
	model/keras_layer/StatefulPartitionedCall:0.56
	model/keras_layer/StatefulPartitionedCall:0.57
	model/keras_layer/StatefulPartitionedCall:0.58
	model/keras_layer/StatefulPartitionedCall:0.59
	model/keras_layer/StatefulPartitionedCall:0.6
	model/keras_layer/StatefulPartitionedCall:0.60
	model/keras_layer/StatefulPartitionedCall:0.61
	model/keras_layer/StatefulPartitionedCall:0.62
	model/keras_layer/StatefulPartitionedCall:0.63
	model/keras_layer/StatefulPartitionedCall:0.64
	model/keras_layer/StatefulPartitionedCall:0.65
	model/keras_layer/StatefulPartitionedCall:0.66
	model/keras_layer/StatefulPartitionedCall:0.67
	model/keras_layer/StatefulPartitionedCall:0.68
	model/keras_layer/StatefulPartitionedCall:0.69
	model/keras_layer/StatefulPartitionedCall:0.7
	model/keras_layer/StatefulPartitionedCall:0.70
	model/keras_layer/StatefulPartitionedCall:0.71
	model/keras_layer/StatefulPartitionedCall:0.72
	model/keras_layer/StatefulPartitionedCall:0.73
	model/keras_layer/StatefulPartitionedCall:0.74
	model/keras_layer/StatefulPartitionedCall:0.75
	model/keras_layer/StatefulPartitionedCall:0.76
	model/keras_layer/StatefulPartitionedCall:0.77
	model/keras_layer/StatefulPartitionedCall:0.78
	model/keras_layer/StatefulPartitionedCall:0.79
	model/keras_layer/StatefulPartitionedCall:0.8
	model/keras_layer/StatefulPartitionedCall:0.80
	model/keras_layer/StatefulPartitionedCall:0.81
	model/keras_layer/StatefulPartitionedCall:0.82
	model/keras_layer/StatefulPartitionedCall:0.83
	model/keras_layer/StatefulPartitionedCall:0.84
	model/keras_layer/StatefulPartitionedCall:0.85
	model/keras_layer/StatefulPartitionedCall:0.86
	model/keras_layer/StatefulPartitionedCall:0.87
	model/keras_layer/StatefulPartitionedCall:0.88
	model/keras_layer/StatefulPartitionedCall:0.89
	model/keras_layer/StatefulPartitionedCall:0.9
	model/keras_layer/StatefulPartitionedCall:0.90
	model/keras_layer/StatefulPartitionedCall:0.91
	model/keras_layer/StatefulPartitionedCall:0.92
	model/keras_layer/StatefulPartitionedCall:0.93
	model/keras_layer/StatefulPartitionedCall:0.94
	model/keras_layer/StatefulPartitionedCall:0.95
	model/keras_layer/StatefulPartitionedCall:0.96
	model/keras_layer/StatefulPartitionedCall:0.97
	model/keras_layer/StatefulPartitionedCall:0.98
	model/keras_layer/StatefulPartitionedCall:0.99
```
</div>
    
<div class="k-default-codeblock">
```
No weights
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: MEAN_MIN_DEPTH:
    1. "model/keras_layer/StatefulPartitionedCall:0.111"  4.778885 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.121"  4.778885 ################
    3. "model/keras_layer/StatefulPartitionedCall:0.191"  4.778885 ################
    4. "model/keras_layer/StatefulPartitionedCall:0.210"  4.778885 ################
    5. "model/keras_layer/StatefulPartitionedCall:0.230"  4.778885 ################
    6. "model/keras_layer/StatefulPartitionedCall:0.318"  4.778885 ################
    7. "model/keras_layer/StatefulPartitionedCall:0.320"  4.778885 ################
    8. "model/keras_layer/StatefulPartitionedCall:0.385"  4.778885 ################
    9. "model/keras_layer/StatefulPartitionedCall:0.445"  4.778885 ################
   10. "model/keras_layer/StatefulPartitionedCall:0.481"  4.778885 ################
   11.  "model/keras_layer/StatefulPartitionedCall:0.59"  4.778885 ################
   12.                                         "__LABEL"  4.778885 ################
   13.  "model/keras_layer/StatefulPartitionedCall:0.12"  4.778429 ###############
   14. "model/keras_layer/StatefulPartitionedCall:0.430"  4.778415 ###############
   15. "model/keras_layer/StatefulPartitionedCall:0.162"  4.778399 ###############
   16. "model/keras_layer/StatefulPartitionedCall:0.240"  4.778364 ###############
   17. "model/keras_layer/StatefulPartitionedCall:0.424"  4.778364 ###############
   18. "model/keras_layer/StatefulPartitionedCall:0.119"  4.778345 ###############
   19. "model/keras_layer/StatefulPartitionedCall:0.257"  4.778345 ###############
   20. "model/keras_layer/StatefulPartitionedCall:0.324"  4.778345 ###############
   21. "model/keras_layer/StatefulPartitionedCall:0.436"  4.778345 ###############
   22.  "model/keras_layer/StatefulPartitionedCall:0.43"  4.778301 ###############
   23. "model/keras_layer/StatefulPartitionedCall:0.296"  4.778222 ###############
   24.  "model/keras_layer/StatefulPartitionedCall:0.36"  4.778190 ###############
   25. "model/keras_layer/StatefulPartitionedCall:0.125"  4.778074 ###############
   26.  "model/keras_layer/StatefulPartitionedCall:0.93"  4.777958 ###############
   27. "model/keras_layer/StatefulPartitionedCall:0.487"  4.777943 ###############
   28. "model/keras_layer/StatefulPartitionedCall:0.213"  4.777912 ###############
   29. "model/keras_layer/StatefulPartitionedCall:0.221"  4.777911 ###############
   30. "model/keras_layer/StatefulPartitionedCall:0.394"  4.777889 ###############
   31. "model/keras_layer/StatefulPartitionedCall:0.482"  4.777874 ###############
   32. "model/keras_layer/StatefulPartitionedCall:0.224"  4.777780 ###############
   33. "model/keras_layer/StatefulPartitionedCall:0.502"  4.777764 ###############
   34. "model/keras_layer/StatefulPartitionedCall:0.206"  4.777761 ###############
   35. "model/keras_layer/StatefulPartitionedCall:0.311"  4.777740 ###############
   36. "model/keras_layer/StatefulPartitionedCall:0.274"  4.777718 ###############
   37. "model/keras_layer/StatefulPartitionedCall:0.211"  4.777669 ###############
   38.  "model/keras_layer/StatefulPartitionedCall:0.35"  4.777667 ###############
   39.  "model/keras_layer/StatefulPartitionedCall:0.82"  4.777616 ###############
   40. "model/keras_layer/StatefulPartitionedCall:0.396"  4.777596 ###############
   41. "model/keras_layer/StatefulPartitionedCall:0.352"  4.777556 ###############
   42. "model/keras_layer/StatefulPartitionedCall:0.400"  4.777457 ###############
   43. "model/keras_layer/StatefulPartitionedCall:0.236"  4.777426 ###############
   44.  "model/keras_layer/StatefulPartitionedCall:0.49"  4.777407 ###############
   45. "model/keras_layer/StatefulPartitionedCall:0.408"  4.777404 ###############
   46. "model/keras_layer/StatefulPartitionedCall:0.167"  4.777309 ###############
   47. "model/keras_layer/StatefulPartitionedCall:0.431"  4.777304 ###############
   48. "model/keras_layer/StatefulPartitionedCall:0.501"  4.777276 ###############
   49. "model/keras_layer/StatefulPartitionedCall:0.407"  4.777264 ###############
   50.  "model/keras_layer/StatefulPartitionedCall:0.85"  4.777252 ###############
   51. "model/keras_layer/StatefulPartitionedCall:0.204"  4.777237 ###############
   52. "model/keras_layer/StatefulPartitionedCall:0.107"  4.777227 ###############
   53.  "model/keras_layer/StatefulPartitionedCall:0.74"  4.777226 ###############
   54. "model/keras_layer/StatefulPartitionedCall:0.242"  4.777187 ###############
   55. "model/keras_layer/StatefulPartitionedCall:0.151"  4.777182 ###############
   56. "model/keras_layer/StatefulPartitionedCall:0.359"  4.777174 ###############
   57. "model/keras_layer/StatefulPartitionedCall:0.404"  4.777100 ###############
   58. "model/keras_layer/StatefulPartitionedCall:0.390"  4.777073 ###############
   59. "model/keras_layer/StatefulPartitionedCall:0.177"  4.777061 ###############
   60. "model/keras_layer/StatefulPartitionedCall:0.146"  4.777055 ###############
   61. "model/keras_layer/StatefulPartitionedCall:0.182"  4.777047 ###############
   62.  "model/keras_layer/StatefulPartitionedCall:0.76"  4.776941 ###############
   63. "model/keras_layer/StatefulPartitionedCall:0.212"  4.776939 ###############
   64. "model/keras_layer/StatefulPartitionedCall:0.158"  4.776936 ###############
   65.  "model/keras_layer/StatefulPartitionedCall:0.25"  4.776858 ###############
   66. "model/keras_layer/StatefulPartitionedCall:0.198"  4.776824 ###############
   67. "model/keras_layer/StatefulPartitionedCall:0.267"  4.776824 ###############
   68.  "model/keras_layer/StatefulPartitionedCall:0.77"  4.776772 ###############
   69. "model/keras_layer/StatefulPartitionedCall:0.277"  4.776640 ###############
   70. "model/keras_layer/StatefulPartitionedCall:0.395"  4.776614 ###############
   71. "model/keras_layer/StatefulPartitionedCall:0.361"  4.776576 ###############
   72.  "model/keras_layer/StatefulPartitionedCall:0.40"  4.776574 ###############
   73. "model/keras_layer/StatefulPartitionedCall:0.195"  4.776562 ###############
   74. "model/keras_layer/StatefulPartitionedCall:0.376"  4.776550 ###############
   75. "model/keras_layer/StatefulPartitionedCall:0.181"  4.776544 ###############
   76. "model/keras_layer/StatefulPartitionedCall:0.335"  4.776541 ###############
   77. "model/keras_layer/StatefulPartitionedCall:0.496"  4.776520 ###############
   78. "model/keras_layer/StatefulPartitionedCall:0.442"  4.776510 ###############
   79. "model/keras_layer/StatefulPartitionedCall:0.391"  4.776427 ###############
   80. "model/keras_layer/StatefulPartitionedCall:0.319"  4.776389 ###############
   81. "model/keras_layer/StatefulPartitionedCall:0.379"  4.776348 ###############
   82. "model/keras_layer/StatefulPartitionedCall:0.138"  4.776294 ###############
   83. "model/keras_layer/StatefulPartitionedCall:0.499"  4.776238 ###############
   84. "model/keras_layer/StatefulPartitionedCall:0.264"  4.776219 ###############
   85. "model/keras_layer/StatefulPartitionedCall:0.293"  4.776211 ###############
   86. "model/keras_layer/StatefulPartitionedCall:0.413"  4.776193 ###############
   87. "model/keras_layer/StatefulPartitionedCall:0.492"  4.776177 ###############
   88. "model/keras_layer/StatefulPartitionedCall:0.124"  4.776161 ###############
   89.  "model/keras_layer/StatefulPartitionedCall:0.53"  4.776113 ###############
   90. "model/keras_layer/StatefulPartitionedCall:0.425"  4.776038 ###############
   91. "model/keras_layer/StatefulPartitionedCall:0.347"  4.775851 ###############
   92.  "model/keras_layer/StatefulPartitionedCall:0.52"  4.775817 ###############
   93. "model/keras_layer/StatefulPartitionedCall:0.465"  4.775798 ###############
   94. "model/keras_layer/StatefulPartitionedCall:0.115"  4.775779 ###############
   95. "model/keras_layer/StatefulPartitionedCall:0.351"  4.775771 ###############
   96.   "model/keras_layer/StatefulPartitionedCall:0.7"  4.775730 ###############
   97. "model/keras_layer/StatefulPartitionedCall:0.248"  4.775631 ###############
   98. "model/keras_layer/StatefulPartitionedCall:0.342"  4.775626 ###############
   99.  "model/keras_layer/StatefulPartitionedCall:0.19"  4.775591 ###############
  100.  "model/keras_layer/StatefulPartitionedCall:0.81"  4.775587 ###############
  101.  "model/keras_layer/StatefulPartitionedCall:0.64"  4.775549 ###############
  102. "model/keras_layer/StatefulPartitionedCall:0.406"  4.775515 ###############
  103. "model/keras_layer/StatefulPartitionedCall:0.251"  4.775470 ###############
  104. "model/keras_layer/StatefulPartitionedCall:0.491"  4.775470 ###############
  105.  "model/keras_layer/StatefulPartitionedCall:0.55"  4.775444 ###############
  106. "model/keras_layer/StatefulPartitionedCall:0.416"  4.775443 ###############
  107. "model/keras_layer/StatefulPartitionedCall:0.360"  4.775372 ###############
  108. "model/keras_layer/StatefulPartitionedCall:0.131"  4.775361 ###############
  109. "model/keras_layer/StatefulPartitionedCall:0.471"  4.775309 ###############
  110. "model/keras_layer/StatefulPartitionedCall:0.443"  4.775278 ###############
  111. "model/keras_layer/StatefulPartitionedCall:0.409"  4.775271 ###############
  112. "model/keras_layer/StatefulPartitionedCall:0.339"  4.775201 ###############
  113.  "model/keras_layer/StatefulPartitionedCall:0.30"  4.775031 ###############
  114. "model/keras_layer/StatefulPartitionedCall:0.307"  4.775029 ###############
  115. "model/keras_layer/StatefulPartitionedCall:0.105"  4.775027 ###############
  116. "model/keras_layer/StatefulPartitionedCall:0.348"  4.775021 ###############
  117. "model/keras_layer/StatefulPartitionedCall:0.136"  4.774960 ###############
  118. "model/keras_layer/StatefulPartitionedCall:0.456"  4.774887 ###############
  119. "model/keras_layer/StatefulPartitionedCall:0.137"  4.774876 ###############
  120.   "model/keras_layer/StatefulPartitionedCall:0.2"  4.774875 ###############
  121.  "model/keras_layer/StatefulPartitionedCall:0.67"  4.774852 ###############
  122. "model/keras_layer/StatefulPartitionedCall:0.186"  4.774847 ###############
  123. "model/keras_layer/StatefulPartitionedCall:0.506"  4.774843 ###############
  124.  "model/keras_layer/StatefulPartitionedCall:0.94"  4.774814 ###############
  125. "model/keras_layer/StatefulPartitionedCall:0.232"  4.774800 ###############
  126. "model/keras_layer/StatefulPartitionedCall:0.280"  4.774790 ###############
  127.  "model/keras_layer/StatefulPartitionedCall:0.75"  4.774740 ###############
  128. "model/keras_layer/StatefulPartitionedCall:0.200"  4.774695 ###############
  129. "model/keras_layer/StatefulPartitionedCall:0.301"  4.774672 ###############
  130. "model/keras_layer/StatefulPartitionedCall:0.370"  4.774647 ###############
  131. "model/keras_layer/StatefulPartitionedCall:0.220"  4.774600 ###############
  132. "model/keras_layer/StatefulPartitionedCall:0.134"  4.774502 ###############
  133. "model/keras_layer/StatefulPartitionedCall:0.334"  4.774409 ###############
  134. "model/keras_layer/StatefulPartitionedCall:0.355"  4.774406 ###############
  135. "model/keras_layer/StatefulPartitionedCall:0.426"  4.774401 ###############
  136. "model/keras_layer/StatefulPartitionedCall:0.316"  4.774344 ###############
  137. "model/keras_layer/StatefulPartitionedCall:0.507"  4.774241 ###############
  138.  "model/keras_layer/StatefulPartitionedCall:0.32"  4.774234 ###############
  139. "model/keras_layer/StatefulPartitionedCall:0.403"  4.774175 ###############
  140.  "model/keras_layer/StatefulPartitionedCall:0.72"  4.774173 ###############
  141.  "model/keras_layer/StatefulPartitionedCall:0.62"  4.774157 ###############
  142. "model/keras_layer/StatefulPartitionedCall:0.288"  4.774110 ###############
  143. "model/keras_layer/StatefulPartitionedCall:0.330"  4.774029 ###############
  144. "model/keras_layer/StatefulPartitionedCall:0.381"  4.774019 ###############
  145. "model/keras_layer/StatefulPartitionedCall:0.154"  4.774019 ###############
  146.  "model/keras_layer/StatefulPartitionedCall:0.38"  4.774006 ###############
  147. "model/keras_layer/StatefulPartitionedCall:0.498"  4.773984 ###############
  148. "model/keras_layer/StatefulPartitionedCall:0.452"  4.773983 ###############
  149. "model/keras_layer/StatefulPartitionedCall:0.312"  4.773943 ###############
  150. "model/keras_layer/StatefulPartitionedCall:0.283"  4.773936 ###############
  151. "model/keras_layer/StatefulPartitionedCall:0.375"  4.773740 ###############
  152. "model/keras_layer/StatefulPartitionedCall:0.122"  4.773713 ###############
  153. "model/keras_layer/StatefulPartitionedCall:0.384"  4.773701 ###############
  154. "model/keras_layer/StatefulPartitionedCall:0.308"  4.773692 ###############
  155. "model/keras_layer/StatefulPartitionedCall:0.175"  4.773686 ###############
  156. "model/keras_layer/StatefulPartitionedCall:0.246"  4.773658 ###############
  157.  "model/keras_layer/StatefulPartitionedCall:0.14"  4.773612 ###############
  158. "model/keras_layer/StatefulPartitionedCall:0.434"  4.773548 ###############
  159. "model/keras_layer/StatefulPartitionedCall:0.143"  4.773544 ###############
  160. "model/keras_layer/StatefulPartitionedCall:0.128"  4.773529 ###############
  161. "model/keras_layer/StatefulPartitionedCall:0.475"  4.773517 ###############
  162.  "model/keras_layer/StatefulPartitionedCall:0.16"  4.773493 ###############
  163. "model/keras_layer/StatefulPartitionedCall:0.106"  4.773461 ###############
  164. "model/keras_layer/StatefulPartitionedCall:0.474"  4.773406 ###############
  165. "model/keras_layer/StatefulPartitionedCall:0.123"  4.773404 ###############
  166. "model/keras_layer/StatefulPartitionedCall:0.331"  4.773369 ###############
  167.  "model/keras_layer/StatefulPartitionedCall:0.78"  4.773345 ###############
  168. "model/keras_layer/StatefulPartitionedCall:0.258"  4.773339 ###############
  169. "model/keras_layer/StatefulPartitionedCall:0.227"  4.773298 ###############
  170. "model/keras_layer/StatefulPartitionedCall:0.346"  4.773257 ###############
  171. "model/keras_layer/StatefulPartitionedCall:0.108"  4.773242 ###############
  172. "model/keras_layer/StatefulPartitionedCall:0.509"  4.773234 ###############
  173. "model/keras_layer/StatefulPartitionedCall:0.389"  4.773219 ###############
  174.  "model/keras_layer/StatefulPartitionedCall:0.90"  4.773209 ###############
  175.  "model/keras_layer/StatefulPartitionedCall:0.31"  4.773179 ###############
  176. "model/keras_layer/StatefulPartitionedCall:0.433"  4.773150 ###############
  177. "model/keras_layer/StatefulPartitionedCall:0.349"  4.773137 ###############
  178.  "model/keras_layer/StatefulPartitionedCall:0.58"  4.773120 ###############
  179. "model/keras_layer/StatefulPartitionedCall:0.461"  4.773046 ###############
  180. "model/keras_layer/StatefulPartitionedCall:0.421"  4.773006 ###############
  181. "model/keras_layer/StatefulPartitionedCall:0.215"  4.772864 ###############
  182. "model/keras_layer/StatefulPartitionedCall:0.435"  4.772803 ###############
  183. "model/keras_layer/StatefulPartitionedCall:0.383"  4.772768 ###############
  184. "model/keras_layer/StatefulPartitionedCall:0.269"  4.772673 ###############
  185. "model/keras_layer/StatefulPartitionedCall:0.172"  4.772657 ###############
  186.  "model/keras_layer/StatefulPartitionedCall:0.20"  4.772632 ###############
  187.  "model/keras_layer/StatefulPartitionedCall:0.60"  4.772603 ###############
  188. "model/keras_layer/StatefulPartitionedCall:0.290"  4.772534 ###############
  189.  "model/keras_layer/StatefulPartitionedCall:0.97"  4.772508 ###############
  190.  "model/keras_layer/StatefulPartitionedCall:0.45"  4.772442 ###############
  191. "model/keras_layer/StatefulPartitionedCall:0.165"  4.772354 ###############
  192. "model/keras_layer/StatefulPartitionedCall:0.275"  4.772322 ###############
  193. "model/keras_layer/StatefulPartitionedCall:0.303"  4.772312 ###############
  194. "model/keras_layer/StatefulPartitionedCall:0.364"  4.772229 ###############
  195.  "model/keras_layer/StatefulPartitionedCall:0.48"  4.772216 ###############
  196. "model/keras_layer/StatefulPartitionedCall:0.490"  4.772197 ###############
  197. "model/keras_layer/StatefulPartitionedCall:0.201"  4.772096 ###############
  198. "model/keras_layer/StatefulPartitionedCall:0.203"  4.772095 ###############
  199. "model/keras_layer/StatefulPartitionedCall:0.503"  4.772073 ###############
  200. "model/keras_layer/StatefulPartitionedCall:0.194"  4.772053 ###############
  201.  "model/keras_layer/StatefulPartitionedCall:0.83"  4.772034 ###############
  202. "model/keras_layer/StatefulPartitionedCall:0.494"  4.772016 ###############
  203. "model/keras_layer/StatefulPartitionedCall:0.402"  4.771983 ###############
  204. "model/keras_layer/StatefulPartitionedCall:0.341"  4.771937 ###############
  205. "model/keras_layer/StatefulPartitionedCall:0.245"  4.771926 ###############
  206.  "model/keras_layer/StatefulPartitionedCall:0.66"  4.771828 ###############
  207. "model/keras_layer/StatefulPartitionedCall:0.270"  4.771819 ###############
  208.  "model/keras_layer/StatefulPartitionedCall:0.15"  4.771761 ###############
  209.  "model/keras_layer/StatefulPartitionedCall:0.91"  4.771666 ###############
  210. "model/keras_layer/StatefulPartitionedCall:0.401"  4.771641 ###############
  211.   "model/keras_layer/StatefulPartitionedCall:0.4"  4.771631 ###############
  212. "model/keras_layer/StatefulPartitionedCall:0.412"  4.771500 ###############
  213. "model/keras_layer/StatefulPartitionedCall:0.205"  4.771495 ###############
  214. "model/keras_layer/StatefulPartitionedCall:0.238"  4.771491 ###############
  215. "model/keras_layer/StatefulPartitionedCall:0.305"  4.771419 ###############
  216. "model/keras_layer/StatefulPartitionedCall:0.155"  4.771381 ###############
  217.  "model/keras_layer/StatefulPartitionedCall:0.21"  4.771373 ###############
  218. "model/keras_layer/StatefulPartitionedCall:0.437"  4.771221 ###############
  219.  "model/keras_layer/StatefulPartitionedCall:0.57"  4.771095 ###############
  220. "model/keras_layer/StatefulPartitionedCall:0.157"  4.771078 ###############
  221.  "model/keras_layer/StatefulPartitionedCall:0.26"  4.771075 ###############
  222.  "model/keras_layer/StatefulPartitionedCall:0.54"  4.771029 ###############
  223. "model/keras_layer/StatefulPartitionedCall:0.484"  4.770926 ###############
  224. "model/keras_layer/StatefulPartitionedCall:0.116"  4.770919 ###############
  225. "model/keras_layer/StatefulPartitionedCall:0.173"  4.770898 ###############
  226.  "model/keras_layer/StatefulPartitionedCall:0.28"  4.770715 ###############
  227. "model/keras_layer/StatefulPartitionedCall:0.336"  4.770454 ###############
  228.  "model/keras_layer/StatefulPartitionedCall:0.63"  4.770032 ###############
  229. "model/keras_layer/StatefulPartitionedCall:0.217"  4.769971 ###############
  230. "model/keras_layer/StatefulPartitionedCall:0.423"  4.769947 ###############
  231. "model/keras_layer/StatefulPartitionedCall:0.508"  4.769761 ###############
  232. "model/keras_layer/StatefulPartitionedCall:0.222"  4.769716 ###############
  233. "model/keras_layer/StatefulPartitionedCall:0.285"  4.769690 ###############
  234. "model/keras_layer/StatefulPartitionedCall:0.225"  4.769686 ###############
  235. "model/keras_layer/StatefulPartitionedCall:0.428"  4.769635 ###############
  236. "model/keras_layer/StatefulPartitionedCall:0.190"  4.769619 ###############
  237. "model/keras_layer/StatefulPartitionedCall:0.478"  4.769617 ###############
  238.  "model/keras_layer/StatefulPartitionedCall:0.42"  4.769417 ###############
  239. "model/keras_layer/StatefulPartitionedCall:0.417"  4.769269 ###############
  240. "model/keras_layer/StatefulPartitionedCall:0.477"  4.769183 ###############
  241. "model/keras_layer/StatefulPartitionedCall:0.112"  4.769176 ###############
  242. "model/keras_layer/StatefulPartitionedCall:0.103"  4.769150 ###############
  243. "model/keras_layer/StatefulPartitionedCall:0.504"  4.769132 ###############
  244. "model/keras_layer/StatefulPartitionedCall:0.196"  4.768978 ###############
  245. "model/keras_layer/StatefulPartitionedCall:0.453"  4.768750 ###############
  246.  "model/keras_layer/StatefulPartitionedCall:0.34"  4.768739 ###############
  247. "model/keras_layer/StatefulPartitionedCall:0.148"  4.768709 ###############
  248. "model/keras_layer/StatefulPartitionedCall:0.244"  4.768458 ###############
  249. "model/keras_layer/StatefulPartitionedCall:0.234"  4.768445 ###############
  250.  "model/keras_layer/StatefulPartitionedCall:0.47"  4.768411 ###############
  251. "model/keras_layer/StatefulPartitionedCall:0.276"  4.768259 ###############
  252. "model/keras_layer/StatefulPartitionedCall:0.495"  4.768099 ###############
  253. "model/keras_layer/StatefulPartitionedCall:0.228"  4.767937 ###############
  254.  "model/keras_layer/StatefulPartitionedCall:0.79"  4.767917 ###############
  255. "model/keras_layer/StatefulPartitionedCall:0.295"  4.767781 ###############
  256. "model/keras_layer/StatefulPartitionedCall:0.440"  4.767713 ###############
  257.  "model/keras_layer/StatefulPartitionedCall:0.88"  4.767683 ###############
  258.  "model/keras_layer/StatefulPartitionedCall:0.23"  4.767670 ###############
  259. "model/keras_layer/StatefulPartitionedCall:0.378"  4.767512 ###############
  260.  "model/keras_layer/StatefulPartitionedCall:0.70"  4.767455 ###############
  261. "model/keras_layer/StatefulPartitionedCall:0.263"  4.767449 ###############
  262. "model/keras_layer/StatefulPartitionedCall:0.382"  4.767386 ###############
  263. "model/keras_layer/StatefulPartitionedCall:0.185"  4.767355 ###############
  264. "model/keras_layer/StatefulPartitionedCall:0.422"  4.767284 ###############
  265.  "model/keras_layer/StatefulPartitionedCall:0.84"  4.767269 ###############
  266. "model/keras_layer/StatefulPartitionedCall:0.279"  4.767056 ###############
  267. "model/keras_layer/StatefulPartitionedCall:0.174"  4.766998 ###############
  268. "model/keras_layer/StatefulPartitionedCall:0.164"  4.766980 ###############
  269. "model/keras_layer/StatefulPartitionedCall:0.266"  4.766722 ###############
  270. "model/keras_layer/StatefulPartitionedCall:0.208"  4.766706 ###############
  271. "model/keras_layer/StatefulPartitionedCall:0.202"  4.766628 ###############
  272. "model/keras_layer/StatefulPartitionedCall:0.284"  4.766225 ###############
  273. "model/keras_layer/StatefulPartitionedCall:0.446"  4.766106 ###############
  274.  "model/keras_layer/StatefulPartitionedCall:0.61"  4.766061 ###############
  275.  "model/keras_layer/StatefulPartitionedCall:0.68"  4.766058 ###############
  276. "model/keras_layer/StatefulPartitionedCall:0.271"  4.766039 ###############
  277. "model/keras_layer/StatefulPartitionedCall:0.493"  4.765990 ###############
  278. "model/keras_layer/StatefulPartitionedCall:0.304"  4.765955 ###############
  279. "model/keras_layer/StatefulPartitionedCall:0.147"  4.765938 ###############
  280. "model/keras_layer/StatefulPartitionedCall:0.344"  4.765902 ###############
  281. "model/keras_layer/StatefulPartitionedCall:0.209"  4.765895 ###############
  282. "model/keras_layer/StatefulPartitionedCall:0.314"  4.765850 ###############
  283. "model/keras_layer/StatefulPartitionedCall:0.468"  4.765762 ###############
  284. "model/keras_layer/StatefulPartitionedCall:0.243"  4.765724 ###############
  285. "model/keras_layer/StatefulPartitionedCall:0.410"  4.765722 ###############
  286.  "model/keras_layer/StatefulPartitionedCall:0.33"  4.765708 ###############
  287. "model/keras_layer/StatefulPartitionedCall:0.447"  4.765553 ###############
  288. "model/keras_layer/StatefulPartitionedCall:0.163"  4.765454 ###############
  289. "model/keras_layer/StatefulPartitionedCall:0.306"  4.765448 ###############
  290. "model/keras_layer/StatefulPartitionedCall:0.239"  4.765388 ###############
  291. "model/keras_layer/StatefulPartitionedCall:0.216"  4.765369 ###############
  292. "model/keras_layer/StatefulPartitionedCall:0.350"  4.765330 ###############
  293. "model/keras_layer/StatefulPartitionedCall:0.340"  4.765257 ###############
  294. "model/keras_layer/StatefulPartitionedCall:0.388"  4.765209 ###############
  295. "model/keras_layer/StatefulPartitionedCall:0.398"  4.765199 ###############
  296. "model/keras_layer/StatefulPartitionedCall:0.170"  4.765104 ###############
  297. "model/keras_layer/StatefulPartitionedCall:0.309"  4.765091 ###############
  298. "model/keras_layer/StatefulPartitionedCall:0.414"  4.765023 ###############
  299. "model/keras_layer/StatefulPartitionedCall:0.476"  4.764803 ###############
  300. "model/keras_layer/StatefulPartitionedCall:0.419"  4.764779 ###############
  301. "model/keras_layer/StatefulPartitionedCall:0.329"  4.764715 ###############
  302. "model/keras_layer/StatefulPartitionedCall:0.282"  4.764668 ###############
  303.  "model/keras_layer/StatefulPartitionedCall:0.69"  4.764641 ###############
  304. "model/keras_layer/StatefulPartitionedCall:0.415"  4.764637 ###############
  305. "model/keras_layer/StatefulPartitionedCall:0.429"  4.764553 ###############
  306. "model/keras_layer/StatefulPartitionedCall:0.441"  4.764350 ###############
  307. "model/keras_layer/StatefulPartitionedCall:0.479"  4.764100 ###############
  308. "model/keras_layer/StatefulPartitionedCall:0.488"  4.763995 ###############
  309. "model/keras_layer/StatefulPartitionedCall:0.183"  4.763878 ###############
  310. "model/keras_layer/StatefulPartitionedCall:0.189"  4.763876 ###############
  311. "model/keras_layer/StatefulPartitionedCall:0.254"  4.763873 ###############
  312. "model/keras_layer/StatefulPartitionedCall:0.145"  4.763818 ###############
  313. "model/keras_layer/StatefulPartitionedCall:0.371"  4.763804 ###############
  314. "model/keras_layer/StatefulPartitionedCall:0.197"  4.763635 ###############
  315. "model/keras_layer/StatefulPartitionedCall:0.353"  4.763609 ###############
  316. "model/keras_layer/StatefulPartitionedCall:0.510"  4.763557 ###############
  317. "model/keras_layer/StatefulPartitionedCall:0.420"  4.763513 ###############
  318. "model/keras_layer/StatefulPartitionedCall:0.321"  4.763493 ###############
  319. "model/keras_layer/StatefulPartitionedCall:0.223"  4.763455 ###############
  320. "model/keras_layer/StatefulPartitionedCall:0.467"  4.763425 ###############
  321.  "model/keras_layer/StatefulPartitionedCall:0.56"  4.763410 ###############
  322. "model/keras_layer/StatefulPartitionedCall:0.357"  4.763398 ###############
  323. "model/keras_layer/StatefulPartitionedCall:0.302"  4.763394 ###############
  324. "model/keras_layer/StatefulPartitionedCall:0.110"  4.763378 ###############
  325.  "model/keras_layer/StatefulPartitionedCall:0.27"  4.763344 ###############
  326. "model/keras_layer/StatefulPartitionedCall:0.229"  4.763335 ###############
  327. "model/keras_layer/StatefulPartitionedCall:0.268"  4.763119 ###############
  328. "model/keras_layer/StatefulPartitionedCall:0.218"  4.763064 ###############
  329. "model/keras_layer/StatefulPartitionedCall:0.273"  4.763054 ###############
  330.  "model/keras_layer/StatefulPartitionedCall:0.92"  4.762850 ###############
  331.  "model/keras_layer/StatefulPartitionedCall:0.86"  4.762711 ###############
  332. "model/keras_layer/StatefulPartitionedCall:0.132"  4.762462 ###############
  333. "model/keras_layer/StatefulPartitionedCall:0.497"  4.762430 ###############
  334. "model/keras_layer/StatefulPartitionedCall:0.141"  4.762289 ###############
  335.   "model/keras_layer/StatefulPartitionedCall:0.9"  4.762281 ###############
  336.  "model/keras_layer/StatefulPartitionedCall:0.22"  4.762091 ###############
  337. "model/keras_layer/StatefulPartitionedCall:0.259"  4.761995 ###############
  338. "model/keras_layer/StatefulPartitionedCall:0.345"  4.761919 ###############
  339. "model/keras_layer/StatefulPartitionedCall:0.262"  4.761840 ###############
  340. "model/keras_layer/StatefulPartitionedCall:0.265"  4.761827 ###############
  341. "model/keras_layer/StatefulPartitionedCall:0.297"  4.761502 ###############
  342. "model/keras_layer/StatefulPartitionedCall:0.120"  4.761344 ###############
  343. "model/keras_layer/StatefulPartitionedCall:0.176"  4.761300 ###############
  344. "model/keras_layer/StatefulPartitionedCall:0.113"  4.761043 ###############
  345.  "model/keras_layer/StatefulPartitionedCall:0.10"  4.760594 ###############
  346. "model/keras_layer/StatefulPartitionedCall:0.109"  4.760564 ###############
  347. "model/keras_layer/StatefulPartitionedCall:0.374"  4.759551 ###############
  348.  "model/keras_layer/StatefulPartitionedCall:0.98"  4.759375 ###############
  349. "model/keras_layer/StatefulPartitionedCall:0.358"  4.759352 ###############
  350. "model/keras_layer/StatefulPartitionedCall:0.231"  4.759197 ###############
  351. "model/keras_layer/StatefulPartitionedCall:0.101"  4.759035 ###############
  352. "model/keras_layer/StatefulPartitionedCall:0.299"  4.758870 ###############
  353. "model/keras_layer/StatefulPartitionedCall:0.313"  4.758795 ###############
  354. "model/keras_layer/StatefulPartitionedCall:0.486"  4.758787 ###############
  355. "model/keras_layer/StatefulPartitionedCall:0.139"  4.758654 ###############
  356. "model/keras_layer/StatefulPartitionedCall:0.326"  4.758463 ###############
  357. "model/keras_layer/StatefulPartitionedCall:0.179"  4.758414 ###############
  358. "model/keras_layer/StatefulPartitionedCall:0.448"  4.758180 ###############
  359. "model/keras_layer/StatefulPartitionedCall:0.368"  4.758030 ###############
  360. "model/keras_layer/StatefulPartitionedCall:0.432"  4.758025 ###############
  361.  "model/keras_layer/StatefulPartitionedCall:0.71"  4.757920 ###############
  362. "model/keras_layer/StatefulPartitionedCall:0.287"  4.757540 ###############
  363. "model/keras_layer/StatefulPartitionedCall:0.149"  4.757448 ###############
  364. "model/keras_layer/StatefulPartitionedCall:0.241"  4.757333 ###############
  365. "model/keras_layer/StatefulPartitionedCall:0.369"  4.757269 ###############
  366. "model/keras_layer/StatefulPartitionedCall:0.377"  4.756940 ###############
  367.  "model/keras_layer/StatefulPartitionedCall:0.39"  4.756934 ###############
  368.   "model/keras_layer/StatefulPartitionedCall:0.3"  4.756282 ###############
  369. "model/keras_layer/StatefulPartitionedCall:0.483"  4.756226 ###############
  370. "model/keras_layer/StatefulPartitionedCall:0.300"  4.756138 ###############
  371. "model/keras_layer/StatefulPartitionedCall:0.256"  4.755537 ###############
  372. "model/keras_layer/StatefulPartitionedCall:0.237"  4.755177 ###############
  373. "model/keras_layer/StatefulPartitionedCall:0.327"  4.754911 ###############
  374. "model/keras_layer/StatefulPartitionedCall:0.104"  4.754898 ###############
  375.   "model/keras_layer/StatefulPartitionedCall:0.5"  4.754805 ###############
  376. "model/keras_layer/StatefulPartitionedCall:0.462"  4.754518 ###############
  377.   "model/keras_layer/StatefulPartitionedCall:0.8"  4.753995 ###############
  378. "model/keras_layer/StatefulPartitionedCall:0.485"  4.753970 ###############
  379. "model/keras_layer/StatefulPartitionedCall:0.380"  4.753925 ###############
  380.  "model/keras_layer/StatefulPartitionedCall:0.80"  4.753901 ###############
  381. "model/keras_layer/StatefulPartitionedCall:0.160"  4.753148 ###############
  382. "model/keras_layer/StatefulPartitionedCall:0.363"  4.753111 ###############
  383. "model/keras_layer/StatefulPartitionedCall:0.102"  4.753095 ###############
  384. "model/keras_layer/StatefulPartitionedCall:0.392"  4.752777 ###############
  385. "model/keras_layer/StatefulPartitionedCall:0.272"  4.751716 ###############
  386.  "model/keras_layer/StatefulPartitionedCall:0.24"  4.751655 ###############
  387.  "model/keras_layer/StatefulPartitionedCall:0.11"  4.751581 ###############
  388. "model/keras_layer/StatefulPartitionedCall:0.451"  4.751193 ###############
  389. "model/keras_layer/StatefulPartitionedCall:0.226"  4.750957 ###############
  390. "model/keras_layer/StatefulPartitionedCall:0.328"  4.750797 ###############
  391. "model/keras_layer/StatefulPartitionedCall:0.480"  4.750656 ###############
  392. "model/keras_layer/StatefulPartitionedCall:0.129"  4.750606 ###############
  393. "model/keras_layer/StatefulPartitionedCall:0.366"  4.750467 ###############
  394. "model/keras_layer/StatefulPartitionedCall:0.130"  4.750427 ###############
  395. "model/keras_layer/StatefulPartitionedCall:0.372"  4.750098 ###############
  396. "model/keras_layer/StatefulPartitionedCall:0.418"  4.750078 ###############
  397. "model/keras_layer/StatefulPartitionedCall:0.260"  4.749678 ###############
  398. "model/keras_layer/StatefulPartitionedCall:0.337"  4.749650 ###############
  399.  "model/keras_layer/StatefulPartitionedCall:0.46"  4.749462 ###############
  400. "model/keras_layer/StatefulPartitionedCall:0.252"  4.748726 ###############
  401. "model/keras_layer/StatefulPartitionedCall:0.235"  4.748133 ###############
  402. "model/keras_layer/StatefulPartitionedCall:0.292"  4.748102 ###############
  403. "model/keras_layer/StatefulPartitionedCall:0.184"  4.747882 ###############
  404.   "model/keras_layer/StatefulPartitionedCall:0.6"  4.747433 ##############
  405.  "model/keras_layer/StatefulPartitionedCall:0.89"  4.746856 ##############
  406. "model/keras_layer/StatefulPartitionedCall:0.207"  4.746087 ##############
  407. "model/keras_layer/StatefulPartitionedCall:0.193"  4.745772 ##############
  408. "model/keras_layer/StatefulPartitionedCall:0.427"  4.744871 ##############
  409. "model/keras_layer/StatefulPartitionedCall:0.387"  4.744315 ##############
  410.  "model/keras_layer/StatefulPartitionedCall:0.96"  4.743747 ##############
  411. "model/keras_layer/StatefulPartitionedCall:0.156"  4.743743 ##############
  412. "model/keras_layer/StatefulPartitionedCall:0.365"  4.743654 ##############
  413. "model/keras_layer/StatefulPartitionedCall:0.100"  4.743286 ##############
  414. "model/keras_layer/StatefulPartitionedCall:0.405"  4.742528 ##############
  415. "model/keras_layer/StatefulPartitionedCall:0.466"  4.742190 ##############
  416. "model/keras_layer/StatefulPartitionedCall:0.460"  4.742065 ##############
  417. "model/keras_layer/StatefulPartitionedCall:0.411"  4.741944 ##############
  418. "model/keras_layer/StatefulPartitionedCall:0.298"  4.741862 ##############
  419. "model/keras_layer/StatefulPartitionedCall:0.455"  4.741692 ##############
  420. "model/keras_layer/StatefulPartitionedCall:0.459"  4.741480 ##############
  421.  "model/keras_layer/StatefulPartitionedCall:0.29"  4.741453 ##############
  422. "model/keras_layer/StatefulPartitionedCall:0.161"  4.740886 ##############
  423. "model/keras_layer/StatefulPartitionedCall:0.505"  4.739873 ##############
  424. "model/keras_layer/StatefulPartitionedCall:0.169"  4.739166 ##############
  425. "model/keras_layer/StatefulPartitionedCall:0.473"  4.738603 ##############
  426. "model/keras_layer/StatefulPartitionedCall:0.472"  4.738585 ##############
  427. "model/keras_layer/StatefulPartitionedCall:0.281"  4.737970 ##############
  428.  "model/keras_layer/StatefulPartitionedCall:0.99"  4.736463 ##############
  429.  "model/keras_layer/StatefulPartitionedCall:0.17"  4.735636 ##############
  430. "model/keras_layer/StatefulPartitionedCall:0.367"  4.735470 ##############
  431. "model/keras_layer/StatefulPartitionedCall:0.449"  4.735173 ##############
  432. "model/keras_layer/StatefulPartitionedCall:0.439"  4.732335 ##############
  433. "model/keras_layer/StatefulPartitionedCall:0.393"  4.732229 ##############
  434.  "model/keras_layer/StatefulPartitionedCall:0.13"  4.731879 ##############
  435. "model/keras_layer/StatefulPartitionedCall:0.450"  4.731607 ##############
  436. "model/keras_layer/StatefulPartitionedCall:0.397"  4.731203 ##############
  437. "model/keras_layer/StatefulPartitionedCall:0.333"  4.730847 ##############
  438. "model/keras_layer/StatefulPartitionedCall:0.457"  4.730395 ##############
  439. "model/keras_layer/StatefulPartitionedCall:0.261"  4.730061 ##############
  440. "model/keras_layer/StatefulPartitionedCall:0.438"  4.729569 ##############
  441.  "model/keras_layer/StatefulPartitionedCall:0.37"  4.729108 ##############
  442. "model/keras_layer/StatefulPartitionedCall:0.135"  4.727419 ##############
  443. "model/keras_layer/StatefulPartitionedCall:0.199"  4.726401 ##############
  444. "model/keras_layer/StatefulPartitionedCall:0.454"  4.724409 ##############
  445. "model/keras_layer/StatefulPartitionedCall:0.470"  4.722369 ##############
  446. "model/keras_layer/StatefulPartitionedCall:0.255"  4.721200 ##############
  447. "model/keras_layer/StatefulPartitionedCall:0.362"  4.720077 ##############
  448. "model/keras_layer/StatefulPartitionedCall:0.178"  4.718396 ##############
  449.  "model/keras_layer/StatefulPartitionedCall:0.18"  4.718127 ##############
  450. "model/keras_layer/StatefulPartitionedCall:0.192"  4.717856 ##############
  451. "model/keras_layer/StatefulPartitionedCall:0.386"  4.717247 ##############
  452. "model/keras_layer/StatefulPartitionedCall:0.469"  4.717178 ##############
  453.  "model/keras_layer/StatefulPartitionedCall:0.41"  4.716544 ##############
  454. "model/keras_layer/StatefulPartitionedCall:0.114"  4.716535 ##############
  455. "model/keras_layer/StatefulPartitionedCall:0.373"  4.715111 #############
  456. "model/keras_layer/StatefulPartitionedCall:0.286"  4.712118 #############
  457. "model/keras_layer/StatefulPartitionedCall:0.187"  4.710842 #############
  458. "model/keras_layer/StatefulPartitionedCall:0.278"  4.710705 #############
  459. "model/keras_layer/StatefulPartitionedCall:0.117"  4.710478 #############
  460. "model/keras_layer/StatefulPartitionedCall:0.133"  4.707640 #############
  461. "model/keras_layer/StatefulPartitionedCall:0.444"  4.707070 #############
  462. "model/keras_layer/StatefulPartitionedCall:0.152"  4.706112 #############
  463. "model/keras_layer/StatefulPartitionedCall:0.511"  4.704641 #############
  464. "model/keras_layer/StatefulPartitionedCall:0.142"  4.703771 #############
  465. "model/keras_layer/StatefulPartitionedCall:0.289"  4.703735 #############
  466. "model/keras_layer/StatefulPartitionedCall:0.338"  4.703547 #############
  467.  "model/keras_layer/StatefulPartitionedCall:0.51"  4.700078 #############
  468.   "model/keras_layer/StatefulPartitionedCall:0.1"  4.698796 #############
  469. "model/keras_layer/StatefulPartitionedCall:0.291"  4.698767 #############
  470. "model/keras_layer/StatefulPartitionedCall:0.219"  4.696392 #############
  471. "model/keras_layer/StatefulPartitionedCall:0.249"  4.695566 #############
  472. "model/keras_layer/StatefulPartitionedCall:0.250"  4.693820 #############
  473. "model/keras_layer/StatefulPartitionedCall:0.463"  4.693692 #############
  474. "model/keras_layer/StatefulPartitionedCall:0.171"  4.693328 #############
  475. "model/keras_layer/StatefulPartitionedCall:0.168"  4.692340 #############
  476. "model/keras_layer/StatefulPartitionedCall:0.315"  4.691698 #############
  477. "model/keras_layer/StatefulPartitionedCall:0.144"  4.691448 #############
  478. "model/keras_layer/StatefulPartitionedCall:0.325"  4.691429 #############
  479. "model/keras_layer/StatefulPartitionedCall:0.150"  4.689540 #############
  480.  "model/keras_layer/StatefulPartitionedCall:0.44"  4.684939 #############
  481. "model/keras_layer/StatefulPartitionedCall:0.247"  4.684100 ############
  482.  "model/keras_layer/StatefulPartitionedCall:0.87"  4.682369 ############
  483. "model/keras_layer/StatefulPartitionedCall:0.294"  4.681529 ############
  484.   "model/keras_layer/StatefulPartitionedCall:0.0"  4.680142 ############
  485. "model/keras_layer/StatefulPartitionedCall:0.500"  4.678263 ############
  486. "model/keras_layer/StatefulPartitionedCall:0.323"  4.676850 ############
  487. "model/keras_layer/StatefulPartitionedCall:0.140"  4.674632 ############
  488. "model/keras_layer/StatefulPartitionedCall:0.489"  4.672511 ############
  489. "model/keras_layer/StatefulPartitionedCall:0.159"  4.672099 ############
  490. "model/keras_layer/StatefulPartitionedCall:0.253"  4.671934 ############
  491. "model/keras_layer/StatefulPartitionedCall:0.118"  4.666036 ############
  492.  "model/keras_layer/StatefulPartitionedCall:0.73"  4.654070 ############
  493.  "model/keras_layer/StatefulPartitionedCall:0.65"  4.651423 ###########
  494. "model/keras_layer/StatefulPartitionedCall:0.399"  4.651111 ###########
  495. "model/keras_layer/StatefulPartitionedCall:0.233"  4.639194 ###########
  496. "model/keras_layer/StatefulPartitionedCall:0.317"  4.638624 ###########
  497.  "model/keras_layer/StatefulPartitionedCall:0.95"  4.637432 ###########
  498. "model/keras_layer/StatefulPartitionedCall:0.214"  4.619871 ##########
  499. "model/keras_layer/StatefulPartitionedCall:0.322"  4.619229 ##########
  500. "model/keras_layer/StatefulPartitionedCall:0.354"  4.617549 ##########
  501. "model/keras_layer/StatefulPartitionedCall:0.343"  4.616924 ##########
  502. "model/keras_layer/StatefulPartitionedCall:0.332"  4.607581 ##########
  503. "model/keras_layer/StatefulPartitionedCall:0.356"  4.597226 ##########
  504. "model/keras_layer/StatefulPartitionedCall:0.127"  4.592482 ##########
  505. "model/keras_layer/StatefulPartitionedCall:0.464"  4.587562 #########
  506. "model/keras_layer/StatefulPartitionedCall:0.166"  4.543928 ########
  507. "model/keras_layer/StatefulPartitionedCall:0.310"  4.541366 ########
  508. "model/keras_layer/StatefulPartitionedCall:0.153"  4.532315 ########
  509. "model/keras_layer/StatefulPartitionedCall:0.126"  4.516429 #######
  510. "model/keras_layer/StatefulPartitionedCall:0.188"  4.507356 #######
  511. "model/keras_layer/StatefulPartitionedCall:0.180"  4.459171 #####
  512. "model/keras_layer/StatefulPartitionedCall:0.458"  4.418949 ####
  513.  "model/keras_layer/StatefulPartitionedCall:0.50"  4.277410 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_AS_ROOT:
    1.  "model/keras_layer/StatefulPartitionedCall:0.50" 14.000000 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.180"  7.000000 #######
    3. "model/keras_layer/StatefulPartitionedCall:0.188"  6.000000 ######
    4. "model/keras_layer/StatefulPartitionedCall:0.126"  5.000000 ####
    5. "model/keras_layer/StatefulPartitionedCall:0.153"  5.000000 ####
    6. "model/keras_layer/StatefulPartitionedCall:0.310"  5.000000 ####
    7. "model/keras_layer/StatefulPartitionedCall:0.214"  4.000000 ###
    8. "model/keras_layer/StatefulPartitionedCall:0.322"  4.000000 ###
    9. "model/keras_layer/StatefulPartitionedCall:0.332"  4.000000 ###
   10. "model/keras_layer/StatefulPartitionedCall:0.458"  4.000000 ###
   11. "model/keras_layer/StatefulPartitionedCall:0.127"  3.000000 ##
   12. "model/keras_layer/StatefulPartitionedCall:0.140"  3.000000 ##
   13. "model/keras_layer/StatefulPartitionedCall:0.233"  3.000000 ##
   14. "model/keras_layer/StatefulPartitionedCall:0.253"  3.000000 ##
   15. "model/keras_layer/StatefulPartitionedCall:0.354"  3.000000 ##
   16. "model/keras_layer/StatefulPartitionedCall:0.356"  3.000000 ##
   17.  "model/keras_layer/StatefulPartitionedCall:0.65"  3.000000 ##
   18.  "model/keras_layer/StatefulPartitionedCall:0.73"  3.000000 ##
   19.  "model/keras_layer/StatefulPartitionedCall:0.95"  3.000000 ##
   20. "model/keras_layer/StatefulPartitionedCall:0.118"  2.000000 #
   21. "model/keras_layer/StatefulPartitionedCall:0.144"  2.000000 #
   22. "model/keras_layer/StatefulPartitionedCall:0.150"  2.000000 #
   23. "model/keras_layer/StatefulPartitionedCall:0.247"  2.000000 #
   24. "model/keras_layer/StatefulPartitionedCall:0.291"  2.000000 #
   25. "model/keras_layer/StatefulPartitionedCall:0.317"  2.000000 #
   26. "model/keras_layer/StatefulPartitionedCall:0.343"  2.000000 #
   27. "model/keras_layer/StatefulPartitionedCall:0.399"  2.000000 #
   28. "model/keras_layer/StatefulPartitionedCall:0.500"  2.000000 #
   29.   "model/keras_layer/StatefulPartitionedCall:0.0"  1.000000 
   30. "model/keras_layer/StatefulPartitionedCall:0.100"  1.000000 
   31. "model/keras_layer/StatefulPartitionedCall:0.114"  1.000000 
   32. "model/keras_layer/StatefulPartitionedCall:0.117"  1.000000 
   33.  "model/keras_layer/StatefulPartitionedCall:0.13"  1.000000 
   34. "model/keras_layer/StatefulPartitionedCall:0.166"  1.000000 
   35. "model/keras_layer/StatefulPartitionedCall:0.168"  1.000000 
   36. "model/keras_layer/StatefulPartitionedCall:0.169"  1.000000 
   37. "model/keras_layer/StatefulPartitionedCall:0.171"  1.000000 
   38. "model/keras_layer/StatefulPartitionedCall:0.178"  1.000000 
   39.  "model/keras_layer/StatefulPartitionedCall:0.18"  1.000000 
   40. "model/keras_layer/StatefulPartitionedCall:0.192"  1.000000 
   41. "model/keras_layer/StatefulPartitionedCall:0.219"  1.000000 
   42. "model/keras_layer/StatefulPartitionedCall:0.249"  1.000000 
   43. "model/keras_layer/StatefulPartitionedCall:0.250"  1.000000 
   44. "model/keras_layer/StatefulPartitionedCall:0.289"  1.000000 
   45. "model/keras_layer/StatefulPartitionedCall:0.323"  1.000000 
   46. "model/keras_layer/StatefulPartitionedCall:0.338"  1.000000 
   47. "model/keras_layer/StatefulPartitionedCall:0.365"  1.000000 
   48. "model/keras_layer/StatefulPartitionedCall:0.367"  1.000000 
   49.  "model/keras_layer/StatefulPartitionedCall:0.37"  1.000000 
   50. "model/keras_layer/StatefulPartitionedCall:0.386"  1.000000 
   51. "model/keras_layer/StatefulPartitionedCall:0.405"  1.000000 
   52.  "model/keras_layer/StatefulPartitionedCall:0.41"  1.000000 
   53. "model/keras_layer/StatefulPartitionedCall:0.450"  1.000000 
   54. "model/keras_layer/StatefulPartitionedCall:0.454"  1.000000 
   55. "model/keras_layer/StatefulPartitionedCall:0.457"  1.000000 
   56. "model/keras_layer/StatefulPartitionedCall:0.464"  1.000000 
   57. "model/keras_layer/StatefulPartitionedCall:0.466"  1.000000 
   58. "model/keras_layer/StatefulPartitionedCall:0.469"  1.000000 
   59. "model/keras_layer/StatefulPartitionedCall:0.489"  1.000000 
   60. "model/keras_layer/StatefulPartitionedCall:0.511"  1.000000 
   61.  "model/keras_layer/StatefulPartitionedCall:0.87"  1.000000 
   62.  "model/keras_layer/StatefulPartitionedCall:0.99"  1.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_NODES:
    1. "model/keras_layer/StatefulPartitionedCall:0.458" 41.000000 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.464" 32.000000 ############
    3. "model/keras_layer/StatefulPartitionedCall:0.166" 31.000000 ############
    4.  "model/keras_layer/StatefulPartitionedCall:0.50" 28.000000 ##########
    5. "model/keras_layer/StatefulPartitionedCall:0.127" 27.000000 ##########
    6. "model/keras_layer/StatefulPartitionedCall:0.188" 25.000000 #########
    7. "model/keras_layer/StatefulPartitionedCall:0.343" 24.000000 #########
    8. "model/keras_layer/StatefulPartitionedCall:0.159" 23.000000 ########
    9. "model/keras_layer/StatefulPartitionedCall:0.126" 22.000000 ########
   10. "model/keras_layer/StatefulPartitionedCall:0.133" 22.000000 ########
   11.  "model/keras_layer/StatefulPartitionedCall:0.44" 22.000000 ########
   12. "model/keras_layer/StatefulPartitionedCall:0.180" 21.000000 ########
   13. "model/keras_layer/StatefulPartitionedCall:0.281" 21.000000 ########
   14. "model/keras_layer/StatefulPartitionedCall:0.444" 21.000000 ########
   15. "model/keras_layer/StatefulPartitionedCall:0.142" 20.000000 #######
   16. "model/keras_layer/StatefulPartitionedCall:0.153" 20.000000 #######
   17. "model/keras_layer/StatefulPartitionedCall:0.286" 20.000000 #######
   18. "model/keras_layer/StatefulPartitionedCall:0.310" 20.000000 #######
   19. "model/keras_layer/StatefulPartitionedCall:0.323" 20.000000 #######
   20. "model/keras_layer/StatefulPartitionedCall:0.325" 20.000000 #######
   21. "model/keras_layer/StatefulPartitionedCall:0.249" 19.000000 #######
   22. "model/keras_layer/StatefulPartitionedCall:0.294" 18.000000 ######
   23.   "model/keras_layer/StatefulPartitionedCall:0.0" 17.000000 ######
   24.   "model/keras_layer/StatefulPartitionedCall:0.1" 17.000000 ######
   25. "model/keras_layer/StatefulPartitionedCall:0.250" 17.000000 ######
   26. "model/keras_layer/StatefulPartitionedCall:0.354" 17.000000 ######
   27. "model/keras_layer/StatefulPartitionedCall:0.399" 17.000000 ######
   28. "model/keras_layer/StatefulPartitionedCall:0.152" 16.000000 ######
   29. "model/keras_layer/StatefulPartitionedCall:0.386" 16.000000 ######
   30. "model/keras_layer/StatefulPartitionedCall:0.451" 16.000000 ######
   31.  "model/keras_layer/StatefulPartitionedCall:0.17" 15.000000 #####
   32. "model/keras_layer/StatefulPartitionedCall:0.187" 15.000000 #####
   33. "model/keras_layer/StatefulPartitionedCall:0.315" 15.000000 #####
   34. "model/keras_layer/StatefulPartitionedCall:0.317" 15.000000 #####
   35. "model/keras_layer/StatefulPartitionedCall:0.322" 15.000000 #####
   36. "model/keras_layer/StatefulPartitionedCall:0.356" 15.000000 #####
   37. "model/keras_layer/StatefulPartitionedCall:0.469" 15.000000 #####
   38. "model/keras_layer/StatefulPartitionedCall:0.135" 14.000000 #####
   39. "model/keras_layer/StatefulPartitionedCall:0.219" 14.000000 #####
   40. "model/keras_layer/StatefulPartitionedCall:0.233" 14.000000 #####
   41. "model/keras_layer/StatefulPartitionedCall:0.261" 14.000000 #####
   42. "model/keras_layer/StatefulPartitionedCall:0.332" 14.000000 #####
   43. "model/keras_layer/StatefulPartitionedCall:0.463" 14.000000 #####
   44.  "model/keras_layer/StatefulPartitionedCall:0.87" 14.000000 #####
   45. "model/keras_layer/StatefulPartitionedCall:0.118" 13.000000 ####
   46. "model/keras_layer/StatefulPartitionedCall:0.161" 13.000000 ####
   47. "model/keras_layer/StatefulPartitionedCall:0.214" 13.000000 ####
   48. "model/keras_layer/StatefulPartitionedCall:0.327" 13.000000 ####
   49. "model/keras_layer/StatefulPartitionedCall:0.397" 13.000000 ####
   50. "model/keras_layer/StatefulPartitionedCall:0.472" 13.000000 ####
   51. "model/keras_layer/StatefulPartitionedCall:0.489" 13.000000 ####
   52. "model/keras_layer/StatefulPartitionedCall:0.171" 12.000000 ####
   53. "model/keras_layer/StatefulPartitionedCall:0.178" 12.000000 ####
   54. "model/keras_layer/StatefulPartitionedCall:0.192" 12.000000 ####
   55. "model/keras_layer/StatefulPartitionedCall:0.199" 12.000000 ####
   56. "model/keras_layer/StatefulPartitionedCall:0.247" 12.000000 ####
   57. "model/keras_layer/StatefulPartitionedCall:0.387" 12.000000 ####
   58.  "model/keras_layer/StatefulPartitionedCall:0.41" 12.000000 ####
   59. "model/keras_layer/StatefulPartitionedCall:0.455" 12.000000 ####
   60. "model/keras_layer/StatefulPartitionedCall:0.480" 12.000000 ####
   61. "model/keras_layer/StatefulPartitionedCall:0.505" 12.000000 ####
   62. "model/keras_layer/StatefulPartitionedCall:0.511" 12.000000 ####
   63. "model/keras_layer/StatefulPartitionedCall:0.101" 11.000000 ####
   64. "model/keras_layer/StatefulPartitionedCall:0.150" 11.000000 ####
   65. "model/keras_layer/StatefulPartitionedCall:0.168" 11.000000 ####
   66. "model/keras_layer/StatefulPartitionedCall:0.226" 11.000000 ####
   67. "model/keras_layer/StatefulPartitionedCall:0.291" 11.000000 ####
   68. "model/keras_layer/StatefulPartitionedCall:0.362" 11.000000 ####
   69. "model/keras_layer/StatefulPartitionedCall:0.414" 11.000000 ####
   70. "model/keras_layer/StatefulPartitionedCall:0.432" 11.000000 ####
   71. "model/keras_layer/StatefulPartitionedCall:0.449" 11.000000 ####
   72. "model/keras_layer/StatefulPartitionedCall:0.473" 11.000000 ####
   73.  "model/keras_layer/StatefulPartitionedCall:0.98" 11.000000 ####
   74.  "model/keras_layer/StatefulPartitionedCall:0.10" 10.000000 ###
   75. "model/keras_layer/StatefulPartitionedCall:0.110" 10.000000 ###
   76. "model/keras_layer/StatefulPartitionedCall:0.117" 10.000000 ###
   77. "model/keras_layer/StatefulPartitionedCall:0.130" 10.000000 ###
   78. "model/keras_layer/StatefulPartitionedCall:0.260" 10.000000 ###
   79. "model/keras_layer/StatefulPartitionedCall:0.272" 10.000000 ###
   80. "model/keras_layer/StatefulPartitionedCall:0.278" 10.000000 ###
   81. "model/keras_layer/StatefulPartitionedCall:0.289" 10.000000 ###
   82. "model/keras_layer/StatefulPartitionedCall:0.358" 10.000000 ###
   83. "model/keras_layer/StatefulPartitionedCall:0.363" 10.000000 ###
   84. "model/keras_layer/StatefulPartitionedCall:0.373" 10.000000 ###
   85. "model/keras_layer/StatefulPartitionedCall:0.411" 10.000000 ###
   86. "model/keras_layer/StatefulPartitionedCall:0.460" 10.000000 ###
   87.  "model/keras_layer/StatefulPartitionedCall:0.65" 10.000000 ###
   88.  "model/keras_layer/StatefulPartitionedCall:0.73" 10.000000 ###
   89. "model/keras_layer/StatefulPartitionedCall:0.120"  9.000000 ###
   90.  "model/keras_layer/StatefulPartitionedCall:0.13"  9.000000 ###
   91. "model/keras_layer/StatefulPartitionedCall:0.144"  9.000000 ###
   92. "model/keras_layer/StatefulPartitionedCall:0.201"  9.000000 ###
   93. "model/keras_layer/StatefulPartitionedCall:0.235"  9.000000 ###
   94. "model/keras_layer/StatefulPartitionedCall:0.276"  9.000000 ###
   95.  "model/keras_layer/StatefulPartitionedCall:0.29"  9.000000 ###
   96.   "model/keras_layer/StatefulPartitionedCall:0.3"  9.000000 ###
   97. "model/keras_layer/StatefulPartitionedCall:0.337"  9.000000 ###
   98. "model/keras_layer/StatefulPartitionedCall:0.344"  9.000000 ###
   99. "model/keras_layer/StatefulPartitionedCall:0.368"  9.000000 ###
  100.  "model/keras_layer/StatefulPartitionedCall:0.37"  9.000000 ###
  101. "model/keras_layer/StatefulPartitionedCall:0.372"  9.000000 ###
  102. "model/keras_layer/StatefulPartitionedCall:0.392"  9.000000 ###
  103. "model/keras_layer/StatefulPartitionedCall:0.415"  9.000000 ###
  104. "model/keras_layer/StatefulPartitionedCall:0.418"  9.000000 ###
  105. "model/keras_layer/StatefulPartitionedCall:0.438"  9.000000 ###
  106. "model/keras_layer/StatefulPartitionedCall:0.446"  9.000000 ###
  107. "model/keras_layer/StatefulPartitionedCall:0.467"  9.000000 ###
  108.  "model/keras_layer/StatefulPartitionedCall:0.47"  9.000000 ###
  109.  "model/keras_layer/StatefulPartitionedCall:0.61"  9.000000 ###
  110.  "model/keras_layer/StatefulPartitionedCall:0.89"  9.000000 ###
  111. "model/keras_layer/StatefulPartitionedCall:0.103"  8.000000 ##
  112. "model/keras_layer/StatefulPartitionedCall:0.104"  8.000000 ##
  113. "model/keras_layer/StatefulPartitionedCall:0.147"  8.000000 ##
  114.  "model/keras_layer/StatefulPartitionedCall:0.20"  8.000000 ##
  115. "model/keras_layer/StatefulPartitionedCall:0.202"  8.000000 ##
  116. "model/keras_layer/StatefulPartitionedCall:0.207"  8.000000 ##
  117. "model/keras_layer/StatefulPartitionedCall:0.223"  8.000000 ##
  118. "model/keras_layer/StatefulPartitionedCall:0.228"  8.000000 ##
  119. "model/keras_layer/StatefulPartitionedCall:0.255"  8.000000 ##
  120. "model/keras_layer/StatefulPartitionedCall:0.265"  8.000000 ##
  121. "model/keras_layer/StatefulPartitionedCall:0.285"  8.000000 ##
  122. "model/keras_layer/StatefulPartitionedCall:0.304"  8.000000 ##
  123. "model/keras_layer/StatefulPartitionedCall:0.329"  8.000000 ##
  124.  "model/keras_layer/StatefulPartitionedCall:0.33"  8.000000 ##
  125. "model/keras_layer/StatefulPartitionedCall:0.338"  8.000000 ##
  126. "model/keras_layer/StatefulPartitionedCall:0.340"  8.000000 ##
  127. "model/keras_layer/StatefulPartitionedCall:0.388"  8.000000 ##
  128. "model/keras_layer/StatefulPartitionedCall:0.393"  8.000000 ##
  129. "model/keras_layer/StatefulPartitionedCall:0.462"  8.000000 ##
  130. "model/keras_layer/StatefulPartitionedCall:0.486"  8.000000 ##
  131.  "model/keras_layer/StatefulPartitionedCall:0.51"  8.000000 ##
  132.   "model/keras_layer/StatefulPartitionedCall:0.8"  8.000000 ##
  133.  "model/keras_layer/StatefulPartitionedCall:0.86"  8.000000 ##
  134.   "model/keras_layer/StatefulPartitionedCall:0.9"  8.000000 ##
  135. "model/keras_layer/StatefulPartitionedCall:0.109"  7.000000 ##
  136.  "model/keras_layer/StatefulPartitionedCall:0.11"  7.000000 ##
  137. "model/keras_layer/StatefulPartitionedCall:0.114"  7.000000 ##
  138. "model/keras_layer/StatefulPartitionedCall:0.140"  7.000000 ##
  139. "model/keras_layer/StatefulPartitionedCall:0.141"  7.000000 ##
  140. "model/keras_layer/StatefulPartitionedCall:0.143"  7.000000 ##
  141. "model/keras_layer/StatefulPartitionedCall:0.148"  7.000000 ##
  142.  "model/keras_layer/StatefulPartitionedCall:0.15"  7.000000 ##
  143. "model/keras_layer/StatefulPartitionedCall:0.160"  7.000000 ##
  144. "model/keras_layer/StatefulPartitionedCall:0.163"  7.000000 ##
  145. "model/keras_layer/StatefulPartitionedCall:0.164"  7.000000 ##
  146.  "model/keras_layer/StatefulPartitionedCall:0.18"  7.000000 ##
  147. "model/keras_layer/StatefulPartitionedCall:0.185"  7.000000 ##
  148. "model/keras_layer/StatefulPartitionedCall:0.193"  7.000000 ##
  149. "model/keras_layer/StatefulPartitionedCall:0.197"  7.000000 ##
  150. "model/keras_layer/StatefulPartitionedCall:0.218"  7.000000 ##
  151. "model/keras_layer/StatefulPartitionedCall:0.239"  7.000000 ##
  152. "model/keras_layer/StatefulPartitionedCall:0.252"  7.000000 ##
  153. "model/keras_layer/StatefulPartitionedCall:0.253"  7.000000 ##
  154. "model/keras_layer/StatefulPartitionedCall:0.256"  7.000000 ##
  155. "model/keras_layer/StatefulPartitionedCall:0.266"  7.000000 ##
  156.  "model/keras_layer/StatefulPartitionedCall:0.28"  7.000000 ##
  157. "model/keras_layer/StatefulPartitionedCall:0.298"  7.000000 ##
  158. "model/keras_layer/StatefulPartitionedCall:0.300"  7.000000 ##
  159. "model/keras_layer/StatefulPartitionedCall:0.333"  7.000000 ##
  160. "model/keras_layer/StatefulPartitionedCall:0.336"  7.000000 ##
  161. "model/keras_layer/StatefulPartitionedCall:0.353"  7.000000 ##
  162. "model/keras_layer/StatefulPartitionedCall:0.357"  7.000000 ##
  163. "model/keras_layer/StatefulPartitionedCall:0.410"  7.000000 ##
  164. "model/keras_layer/StatefulPartitionedCall:0.427"  7.000000 ##
  165. "model/keras_layer/StatefulPartitionedCall:0.439"  7.000000 ##
  166. "model/keras_layer/StatefulPartitionedCall:0.441"  7.000000 ##
  167. "model/keras_layer/StatefulPartitionedCall:0.447"  7.000000 ##
  168. "model/keras_layer/StatefulPartitionedCall:0.454"  7.000000 ##
  169. "model/keras_layer/StatefulPartitionedCall:0.457"  7.000000 ##
  170. "model/keras_layer/StatefulPartitionedCall:0.495"  7.000000 ##
  171.   "model/keras_layer/StatefulPartitionedCall:0.5"  7.000000 ##
  172.   "model/keras_layer/StatefulPartitionedCall:0.6"  7.000000 ##
  173.  "model/keras_layer/StatefulPartitionedCall:0.80"  7.000000 ##
  174.  "model/keras_layer/StatefulPartitionedCall:0.88"  7.000000 ##
  175.  "model/keras_layer/StatefulPartitionedCall:0.95"  7.000000 ##
  176.  "model/keras_layer/StatefulPartitionedCall:0.96"  7.000000 ##
  177. "model/keras_layer/StatefulPartitionedCall:0.122"  6.000000 ##
  178. "model/keras_layer/StatefulPartitionedCall:0.136"  6.000000 ##
  179.  "model/keras_layer/StatefulPartitionedCall:0.14"  6.000000 ##
  180. "model/keras_layer/StatefulPartitionedCall:0.149"  6.000000 ##
  181. "model/keras_layer/StatefulPartitionedCall:0.169"  6.000000 ##
  182. "model/keras_layer/StatefulPartitionedCall:0.176"  6.000000 ##
  183. "model/keras_layer/StatefulPartitionedCall:0.184"  6.000000 ##
  184. "model/keras_layer/StatefulPartitionedCall:0.186"  6.000000 ##
  185. "model/keras_layer/StatefulPartitionedCall:0.196"  6.000000 ##
  186. "model/keras_layer/StatefulPartitionedCall:0.205"  6.000000 ##
  187.  "model/keras_layer/StatefulPartitionedCall:0.21"  6.000000 ##
  188. "model/keras_layer/StatefulPartitionedCall:0.217"  6.000000 ##
  189. "model/keras_layer/StatefulPartitionedCall:0.227"  6.000000 ##
  190. "model/keras_layer/StatefulPartitionedCall:0.237"  6.000000 ##
  191. "model/keras_layer/StatefulPartitionedCall:0.243"  6.000000 ##
  192. "model/keras_layer/StatefulPartitionedCall:0.254"  6.000000 ##
  193. "model/keras_layer/StatefulPartitionedCall:0.258"  6.000000 ##
  194. "model/keras_layer/StatefulPartitionedCall:0.262"  6.000000 ##
  195.  "model/keras_layer/StatefulPartitionedCall:0.27"  6.000000 ##
  196. "model/keras_layer/StatefulPartitionedCall:0.270"  6.000000 ##
  197. "model/keras_layer/StatefulPartitionedCall:0.275"  6.000000 ##
  198. "model/keras_layer/StatefulPartitionedCall:0.297"  6.000000 ##
  199. "model/keras_layer/StatefulPartitionedCall:0.299"  6.000000 ##
  200.  "model/keras_layer/StatefulPartitionedCall:0.30"  6.000000 ##
  201. "model/keras_layer/StatefulPartitionedCall:0.303"  6.000000 ##
  202. "model/keras_layer/StatefulPartitionedCall:0.321"  6.000000 ##
  203. "model/keras_layer/StatefulPartitionedCall:0.326"  6.000000 ##
  204. "model/keras_layer/StatefulPartitionedCall:0.328"  6.000000 ##
  205. "model/keras_layer/StatefulPartitionedCall:0.355"  6.000000 ##
  206. "model/keras_layer/StatefulPartitionedCall:0.370"  6.000000 ##
  207. "model/keras_layer/StatefulPartitionedCall:0.375"  6.000000 ##
  208. "model/keras_layer/StatefulPartitionedCall:0.384"  6.000000 ##
  209. "model/keras_layer/StatefulPartitionedCall:0.412"  6.000000 ##
  210. "model/keras_layer/StatefulPartitionedCall:0.419"  6.000000 ##
  211. "model/keras_layer/StatefulPartitionedCall:0.420"  6.000000 ##
  212. "model/keras_layer/StatefulPartitionedCall:0.422"  6.000000 ##
  213. "model/keras_layer/StatefulPartitionedCall:0.426"  6.000000 ##
  214. "model/keras_layer/StatefulPartitionedCall:0.428"  6.000000 ##
  215. "model/keras_layer/StatefulPartitionedCall:0.433"  6.000000 ##
  216. "model/keras_layer/StatefulPartitionedCall:0.440"  6.000000 ##
  217. "model/keras_layer/StatefulPartitionedCall:0.448"  6.000000 ##
  218. "model/keras_layer/StatefulPartitionedCall:0.453"  6.000000 ##
  219.  "model/keras_layer/StatefulPartitionedCall:0.46"  6.000000 ##
  220. "model/keras_layer/StatefulPartitionedCall:0.475"  6.000000 ##
  221. "model/keras_layer/StatefulPartitionedCall:0.476"  6.000000 ##
  222. "model/keras_layer/StatefulPartitionedCall:0.483"  6.000000 ##
  223. "model/keras_layer/StatefulPartitionedCall:0.484"  6.000000 ##
  224. "model/keras_layer/StatefulPartitionedCall:0.488"  6.000000 ##
  225. "model/keras_layer/StatefulPartitionedCall:0.497"  6.000000 ##
  226. "model/keras_layer/StatefulPartitionedCall:0.500"  6.000000 ##
  227. "model/keras_layer/StatefulPartitionedCall:0.510"  6.000000 ##
  228.  "model/keras_layer/StatefulPartitionedCall:0.56"  6.000000 ##
  229.  "model/keras_layer/StatefulPartitionedCall:0.57"  6.000000 ##
  230.  "model/keras_layer/StatefulPartitionedCall:0.81"  6.000000 ##
  231.  "model/keras_layer/StatefulPartitionedCall:0.92"  6.000000 ##
  232.  "model/keras_layer/StatefulPartitionedCall:0.99"  6.000000 ##
  233. "model/keras_layer/StatefulPartitionedCall:0.108"  5.000000 #
  234. "model/keras_layer/StatefulPartitionedCall:0.112"  5.000000 #
  235. "model/keras_layer/StatefulPartitionedCall:0.129"  5.000000 #
  236. "model/keras_layer/StatefulPartitionedCall:0.145"  5.000000 #
  237. "model/keras_layer/StatefulPartitionedCall:0.154"  5.000000 #
  238. "model/keras_layer/StatefulPartitionedCall:0.155"  5.000000 #
  239. "model/keras_layer/StatefulPartitionedCall:0.175"  5.000000 #
  240. "model/keras_layer/StatefulPartitionedCall:0.189"  5.000000 #
  241. "model/keras_layer/StatefulPartitionedCall:0.194"  5.000000 #
  242. "model/keras_layer/StatefulPartitionedCall:0.208"  5.000000 #
  243. "model/keras_layer/StatefulPartitionedCall:0.209"  5.000000 #
  244.  "model/keras_layer/StatefulPartitionedCall:0.22"  5.000000 #
  245. "model/keras_layer/StatefulPartitionedCall:0.220"  5.000000 #
  246. "model/keras_layer/StatefulPartitionedCall:0.222"  5.000000 #
  247.  "model/keras_layer/StatefulPartitionedCall:0.24"  5.000000 #
  248. "model/keras_layer/StatefulPartitionedCall:0.241"  5.000000 #
  249. "model/keras_layer/StatefulPartitionedCall:0.246"  5.000000 #
  250. "model/keras_layer/StatefulPartitionedCall:0.259"  5.000000 #
  251. "model/keras_layer/StatefulPartitionedCall:0.263"  5.000000 #
  252. "model/keras_layer/StatefulPartitionedCall:0.271"  5.000000 #
  253. "model/keras_layer/StatefulPartitionedCall:0.273"  5.000000 #
  254. "model/keras_layer/StatefulPartitionedCall:0.279"  5.000000 #
  255. "model/keras_layer/StatefulPartitionedCall:0.288"  5.000000 #
  256. "model/keras_layer/StatefulPartitionedCall:0.290"  5.000000 #
  257. "model/keras_layer/StatefulPartitionedCall:0.295"  5.000000 #
  258. "model/keras_layer/StatefulPartitionedCall:0.306"  5.000000 #
  259. "model/keras_layer/StatefulPartitionedCall:0.309"  5.000000 #
  260. "model/keras_layer/StatefulPartitionedCall:0.312"  5.000000 #
  261. "model/keras_layer/StatefulPartitionedCall:0.313"  5.000000 #
  262. "model/keras_layer/StatefulPartitionedCall:0.314"  5.000000 #
  263.  "model/keras_layer/StatefulPartitionedCall:0.32"  5.000000 #
  264. "model/keras_layer/StatefulPartitionedCall:0.346"  5.000000 #
  265. "model/keras_layer/StatefulPartitionedCall:0.350"  5.000000 #
  266. "model/keras_layer/StatefulPartitionedCall:0.351"  5.000000 #
  267. "model/keras_layer/StatefulPartitionedCall:0.360"  5.000000 #
  268. "model/keras_layer/StatefulPartitionedCall:0.367"  5.000000 #
  269. "model/keras_layer/StatefulPartitionedCall:0.377"  5.000000 #
  270. "model/keras_layer/StatefulPartitionedCall:0.378"  5.000000 #
  271. "model/keras_layer/StatefulPartitionedCall:0.380"  5.000000 #
  272. "model/keras_layer/StatefulPartitionedCall:0.383"  5.000000 #
  273. "model/keras_layer/StatefulPartitionedCall:0.389"  5.000000 #
  274.  "model/keras_layer/StatefulPartitionedCall:0.39"  5.000000 #
  275. "model/keras_layer/StatefulPartitionedCall:0.401"  5.000000 #
  276. "model/keras_layer/StatefulPartitionedCall:0.403"  5.000000 #
  277. "model/keras_layer/StatefulPartitionedCall:0.450"  5.000000 #
  278. "model/keras_layer/StatefulPartitionedCall:0.459"  5.000000 #
  279. "model/keras_layer/StatefulPartitionedCall:0.465"  5.000000 #
  280. "model/keras_layer/StatefulPartitionedCall:0.470"  5.000000 #
  281. "model/keras_layer/StatefulPartitionedCall:0.477"  5.000000 #
  282. "model/keras_layer/StatefulPartitionedCall:0.478"  5.000000 #
  283. "model/keras_layer/StatefulPartitionedCall:0.479"  5.000000 #
  284.  "model/keras_layer/StatefulPartitionedCall:0.48"  5.000000 #
  285. "model/keras_layer/StatefulPartitionedCall:0.485"  5.000000 #
  286. "model/keras_layer/StatefulPartitionedCall:0.490"  5.000000 #
  287. "model/keras_layer/StatefulPartitionedCall:0.504"  5.000000 #
  288.  "model/keras_layer/StatefulPartitionedCall:0.66"  5.000000 #
  289.  "model/keras_layer/StatefulPartitionedCall:0.67"  5.000000 #
  290.  "model/keras_layer/StatefulPartitionedCall:0.69"  5.000000 #
  291.  "model/keras_layer/StatefulPartitionedCall:0.71"  5.000000 #
  292.  "model/keras_layer/StatefulPartitionedCall:0.75"  5.000000 #
  293.  "model/keras_layer/StatefulPartitionedCall:0.94"  5.000000 #
  294.  "model/keras_layer/StatefulPartitionedCall:0.97"  5.000000 #
  295. "model/keras_layer/StatefulPartitionedCall:0.102"  4.000000 #
  296. "model/keras_layer/StatefulPartitionedCall:0.105"  4.000000 #
  297. "model/keras_layer/StatefulPartitionedCall:0.113"  4.000000 #
  298. "model/keras_layer/StatefulPartitionedCall:0.123"  4.000000 #
  299. "model/keras_layer/StatefulPartitionedCall:0.131"  4.000000 #
  300. "model/keras_layer/StatefulPartitionedCall:0.134"  4.000000 #
  301. "model/keras_layer/StatefulPartitionedCall:0.138"  4.000000 #
  302. "model/keras_layer/StatefulPartitionedCall:0.139"  4.000000 #
  303. "model/keras_layer/StatefulPartitionedCall:0.157"  4.000000 #
  304. "model/keras_layer/StatefulPartitionedCall:0.165"  4.000000 #
  305. "model/keras_layer/StatefulPartitionedCall:0.170"  4.000000 #
  306. "model/keras_layer/StatefulPartitionedCall:0.172"  4.000000 #
  307. "model/keras_layer/StatefulPartitionedCall:0.174"  4.000000 #
  308. "model/keras_layer/StatefulPartitionedCall:0.179"  4.000000 #
  309. "model/keras_layer/StatefulPartitionedCall:0.183"  4.000000 #
  310.  "model/keras_layer/StatefulPartitionedCall:0.19"  4.000000 #
  311. "model/keras_layer/StatefulPartitionedCall:0.195"  4.000000 #
  312.   "model/keras_layer/StatefulPartitionedCall:0.2"  4.000000 #
  313. "model/keras_layer/StatefulPartitionedCall:0.200"  4.000000 #
  314. "model/keras_layer/StatefulPartitionedCall:0.203"  4.000000 #
  315. "model/keras_layer/StatefulPartitionedCall:0.215"  4.000000 #
  316. "model/keras_layer/StatefulPartitionedCall:0.216"  4.000000 #
  317.  "model/keras_layer/StatefulPartitionedCall:0.23"  4.000000 #
  318. "model/keras_layer/StatefulPartitionedCall:0.231"  4.000000 #
  319. "model/keras_layer/StatefulPartitionedCall:0.232"  4.000000 #
  320. "model/keras_layer/StatefulPartitionedCall:0.234"  4.000000 #
  321. "model/keras_layer/StatefulPartitionedCall:0.245"  4.000000 #
  322. "model/keras_layer/StatefulPartitionedCall:0.248"  4.000000 #
  323. "model/keras_layer/StatefulPartitionedCall:0.267"  4.000000 #
  324. "model/keras_layer/StatefulPartitionedCall:0.292"  4.000000 #
  325. "model/keras_layer/StatefulPartitionedCall:0.293"  4.000000 #
  326. "model/keras_layer/StatefulPartitionedCall:0.301"  4.000000 #
  327. "model/keras_layer/StatefulPartitionedCall:0.305"  4.000000 #
  328. "model/keras_layer/StatefulPartitionedCall:0.308"  4.000000 #
  329.  "model/keras_layer/StatefulPartitionedCall:0.31"  4.000000 #
  330. "model/keras_layer/StatefulPartitionedCall:0.319"  4.000000 #
  331. "model/keras_layer/StatefulPartitionedCall:0.335"  4.000000 #
  332. "model/keras_layer/StatefulPartitionedCall:0.339"  4.000000 #
  333. "model/keras_layer/StatefulPartitionedCall:0.341"  4.000000 #
  334. "model/keras_layer/StatefulPartitionedCall:0.342"  4.000000 #
  335. "model/keras_layer/StatefulPartitionedCall:0.345"  4.000000 #
  336. "model/keras_layer/StatefulPartitionedCall:0.347"  4.000000 #
  337. "model/keras_layer/StatefulPartitionedCall:0.348"  4.000000 #
  338. "model/keras_layer/StatefulPartitionedCall:0.349"  4.000000 #
  339. "model/keras_layer/StatefulPartitionedCall:0.371"  4.000000 #
  340. "model/keras_layer/StatefulPartitionedCall:0.374"  4.000000 #
  341. "model/keras_layer/StatefulPartitionedCall:0.379"  4.000000 #
  342. "model/keras_layer/StatefulPartitionedCall:0.395"  4.000000 #
  343. "model/keras_layer/StatefulPartitionedCall:0.402"  4.000000 #
  344.  "model/keras_layer/StatefulPartitionedCall:0.42"  4.000000 #
  345. "model/keras_layer/StatefulPartitionedCall:0.423"  4.000000 #
  346. "model/keras_layer/StatefulPartitionedCall:0.425"  4.000000 #
  347. "model/keras_layer/StatefulPartitionedCall:0.429"  4.000000 #
  348. "model/keras_layer/StatefulPartitionedCall:0.468"  4.000000 #
  349. "model/keras_layer/StatefulPartitionedCall:0.471"  4.000000 #
  350. "model/keras_layer/StatefulPartitionedCall:0.474"  4.000000 #
  351. "model/keras_layer/StatefulPartitionedCall:0.493"  4.000000 #
  352. "model/keras_layer/StatefulPartitionedCall:0.498"  4.000000 #
  353. "model/keras_layer/StatefulPartitionedCall:0.506"  4.000000 #
  354. "model/keras_layer/StatefulPartitionedCall:0.507"  4.000000 #
  355.  "model/keras_layer/StatefulPartitionedCall:0.52"  4.000000 #
  356.  "model/keras_layer/StatefulPartitionedCall:0.60"  4.000000 #
  357.  "model/keras_layer/StatefulPartitionedCall:0.70"  4.000000 #
  358.  "model/keras_layer/StatefulPartitionedCall:0.77"  4.000000 #
  359.  "model/keras_layer/StatefulPartitionedCall:0.78"  4.000000 #
  360.  "model/keras_layer/StatefulPartitionedCall:0.90"  4.000000 #
  361. "model/keras_layer/StatefulPartitionedCall:0.100"  3.000000 
  362. "model/keras_layer/StatefulPartitionedCall:0.106"  3.000000 
  363. "model/keras_layer/StatefulPartitionedCall:0.115"  3.000000 
  364. "model/keras_layer/StatefulPartitionedCall:0.128"  3.000000 
  365. "model/keras_layer/StatefulPartitionedCall:0.137"  3.000000 
  366. "model/keras_layer/StatefulPartitionedCall:0.146"  3.000000 
  367. "model/keras_layer/StatefulPartitionedCall:0.151"  3.000000 
  368. "model/keras_layer/StatefulPartitionedCall:0.156"  3.000000 
  369. "model/keras_layer/StatefulPartitionedCall:0.158"  3.000000 
  370.  "model/keras_layer/StatefulPartitionedCall:0.16"  3.000000 
  371. "model/keras_layer/StatefulPartitionedCall:0.167"  3.000000 
  372. "model/keras_layer/StatefulPartitionedCall:0.181"  3.000000 
  373. "model/keras_layer/StatefulPartitionedCall:0.190"  3.000000 
  374. "model/keras_layer/StatefulPartitionedCall:0.204"  3.000000 
  375. "model/keras_layer/StatefulPartitionedCall:0.225"  3.000000 
  376. "model/keras_layer/StatefulPartitionedCall:0.238"  3.000000 
  377. "model/keras_layer/StatefulPartitionedCall:0.251"  3.000000 
  378. "model/keras_layer/StatefulPartitionedCall:0.280"  3.000000 
  379. "model/keras_layer/StatefulPartitionedCall:0.282"  3.000000 
  380. "model/keras_layer/StatefulPartitionedCall:0.283"  3.000000 
  381. "model/keras_layer/StatefulPartitionedCall:0.284"  3.000000 
  382. "model/keras_layer/StatefulPartitionedCall:0.287"  3.000000 
  383. "model/keras_layer/StatefulPartitionedCall:0.302"  3.000000 
  384. "model/keras_layer/StatefulPartitionedCall:0.307"  3.000000 
  385. "model/keras_layer/StatefulPartitionedCall:0.316"  3.000000 
  386. "model/keras_layer/StatefulPartitionedCall:0.330"  3.000000 
  387. "model/keras_layer/StatefulPartitionedCall:0.331"  3.000000 
  388. "model/keras_layer/StatefulPartitionedCall:0.334"  3.000000 
  389.  "model/keras_layer/StatefulPartitionedCall:0.34"  3.000000 
  390. "model/keras_layer/StatefulPartitionedCall:0.359"  3.000000 
  391. "model/keras_layer/StatefulPartitionedCall:0.361"  3.000000 
  392. "model/keras_layer/StatefulPartitionedCall:0.364"  3.000000 
  393. "model/keras_layer/StatefulPartitionedCall:0.366"  3.000000 
  394. "model/keras_layer/StatefulPartitionedCall:0.369"  3.000000 
  395.  "model/keras_layer/StatefulPartitionedCall:0.38"  3.000000 
  396. "model/keras_layer/StatefulPartitionedCall:0.382"  3.000000 
  397. "model/keras_layer/StatefulPartitionedCall:0.390"  3.000000 
  398. "model/keras_layer/StatefulPartitionedCall:0.391"  3.000000 
  399. "model/keras_layer/StatefulPartitionedCall:0.398"  3.000000 
  400. "model/keras_layer/StatefulPartitionedCall:0.400"  3.000000 
  401. "model/keras_layer/StatefulPartitionedCall:0.404"  3.000000 
  402. "model/keras_layer/StatefulPartitionedCall:0.406"  3.000000 
  403. "model/keras_layer/StatefulPartitionedCall:0.407"  3.000000 
  404. "model/keras_layer/StatefulPartitionedCall:0.408"  3.000000 
  405. "model/keras_layer/StatefulPartitionedCall:0.409"  3.000000 
  406. "model/keras_layer/StatefulPartitionedCall:0.413"  3.000000 
  407. "model/keras_layer/StatefulPartitionedCall:0.416"  3.000000 
  408. "model/keras_layer/StatefulPartitionedCall:0.417"  3.000000 
  409. "model/keras_layer/StatefulPartitionedCall:0.421"  3.000000 
  410. "model/keras_layer/StatefulPartitionedCall:0.434"  3.000000 
  411. "model/keras_layer/StatefulPartitionedCall:0.437"  3.000000 
  412. "model/keras_layer/StatefulPartitionedCall:0.442"  3.000000 
  413. "model/keras_layer/StatefulPartitionedCall:0.443"  3.000000 
  414.  "model/keras_layer/StatefulPartitionedCall:0.45"  3.000000 
  415. "model/keras_layer/StatefulPartitionedCall:0.452"  3.000000 
  416.  "model/keras_layer/StatefulPartitionedCall:0.49"  3.000000 
  417. "model/keras_layer/StatefulPartitionedCall:0.491"  3.000000 
  418. "model/keras_layer/StatefulPartitionedCall:0.501"  3.000000 
  419. "model/keras_layer/StatefulPartitionedCall:0.503"  3.000000 
  420.  "model/keras_layer/StatefulPartitionedCall:0.53"  3.000000 
  421.  "model/keras_layer/StatefulPartitionedCall:0.54"  3.000000 
  422.  "model/keras_layer/StatefulPartitionedCall:0.55"  3.000000 
  423.  "model/keras_layer/StatefulPartitionedCall:0.58"  3.000000 
  424.  "model/keras_layer/StatefulPartitionedCall:0.62"  3.000000 
  425.  "model/keras_layer/StatefulPartitionedCall:0.63"  3.000000 
  426.  "model/keras_layer/StatefulPartitionedCall:0.64"  3.000000 
  427.  "model/keras_layer/StatefulPartitionedCall:0.68"  3.000000 
  428.   "model/keras_layer/StatefulPartitionedCall:0.7"  3.000000 
  429.  "model/keras_layer/StatefulPartitionedCall:0.72"  3.000000 
  430.  "model/keras_layer/StatefulPartitionedCall:0.74"  3.000000 
  431.  "model/keras_layer/StatefulPartitionedCall:0.79"  3.000000 
  432.  "model/keras_layer/StatefulPartitionedCall:0.83"  3.000000 
  433.  "model/keras_layer/StatefulPartitionedCall:0.84"  3.000000 
  434.  "model/keras_layer/StatefulPartitionedCall:0.91"  3.000000 
  435. "model/keras_layer/StatefulPartitionedCall:0.116"  2.000000 
  436. "model/keras_layer/StatefulPartitionedCall:0.124"  2.000000 
  437. "model/keras_layer/StatefulPartitionedCall:0.173"  2.000000 
  438. "model/keras_layer/StatefulPartitionedCall:0.182"  2.000000 
  439. "model/keras_layer/StatefulPartitionedCall:0.198"  2.000000 
  440. "model/keras_layer/StatefulPartitionedCall:0.206"  2.000000 
  441. "model/keras_layer/StatefulPartitionedCall:0.211"  2.000000 
  442. "model/keras_layer/StatefulPartitionedCall:0.221"  2.000000 
  443. "model/keras_layer/StatefulPartitionedCall:0.224"  2.000000 
  444. "model/keras_layer/StatefulPartitionedCall:0.242"  2.000000 
  445. "model/keras_layer/StatefulPartitionedCall:0.244"  2.000000 
  446.  "model/keras_layer/StatefulPartitionedCall:0.26"  2.000000 
  447. "model/keras_layer/StatefulPartitionedCall:0.264"  2.000000 
  448. "model/keras_layer/StatefulPartitionedCall:0.269"  2.000000 
  449. "model/keras_layer/StatefulPartitionedCall:0.274"  2.000000 
  450. "model/keras_layer/StatefulPartitionedCall:0.311"  2.000000 
  451.  "model/keras_layer/StatefulPartitionedCall:0.35"  2.000000 
  452. "model/keras_layer/StatefulPartitionedCall:0.352"  2.000000 
  453. "model/keras_layer/StatefulPartitionedCall:0.365"  2.000000 
  454. "model/keras_layer/StatefulPartitionedCall:0.394"  2.000000 
  455. "model/keras_layer/StatefulPartitionedCall:0.396"  2.000000 
  456.   "model/keras_layer/StatefulPartitionedCall:0.4"  2.000000 
  457.  "model/keras_layer/StatefulPartitionedCall:0.40"  2.000000 
  458. "model/keras_layer/StatefulPartitionedCall:0.405"  2.000000 
  459. "model/keras_layer/StatefulPartitionedCall:0.431"  2.000000 
  460. "model/keras_layer/StatefulPartitionedCall:0.456"  2.000000 
  461. "model/keras_layer/StatefulPartitionedCall:0.466"  2.000000 
  462. "model/keras_layer/StatefulPartitionedCall:0.482"  2.000000 
  463. "model/keras_layer/StatefulPartitionedCall:0.487"  2.000000 
  464. "model/keras_layer/StatefulPartitionedCall:0.492"  2.000000 
  465. "model/keras_layer/StatefulPartitionedCall:0.496"  2.000000 
  466. "model/keras_layer/StatefulPartitionedCall:0.499"  2.000000 
  467. "model/keras_layer/StatefulPartitionedCall:0.502"  2.000000 
  468.  "model/keras_layer/StatefulPartitionedCall:0.76"  2.000000 
  469.  "model/keras_layer/StatefulPartitionedCall:0.82"  2.000000 
  470.  "model/keras_layer/StatefulPartitionedCall:0.85"  2.000000 
  471.  "model/keras_layer/StatefulPartitionedCall:0.93"  2.000000 
  472. "model/keras_layer/StatefulPartitionedCall:0.107"  1.000000 
  473. "model/keras_layer/StatefulPartitionedCall:0.119"  1.000000 
  474.  "model/keras_layer/StatefulPartitionedCall:0.12"  1.000000 
  475. "model/keras_layer/StatefulPartitionedCall:0.125"  1.000000 
  476. "model/keras_layer/StatefulPartitionedCall:0.132"  1.000000 
  477. "model/keras_layer/StatefulPartitionedCall:0.162"  1.000000 
  478. "model/keras_layer/StatefulPartitionedCall:0.177"  1.000000 
  479. "model/keras_layer/StatefulPartitionedCall:0.212"  1.000000 
  480. "model/keras_layer/StatefulPartitionedCall:0.213"  1.000000 
  481. "model/keras_layer/StatefulPartitionedCall:0.229"  1.000000 
  482. "model/keras_layer/StatefulPartitionedCall:0.236"  1.000000 
  483. "model/keras_layer/StatefulPartitionedCall:0.240"  1.000000 
  484.  "model/keras_layer/StatefulPartitionedCall:0.25"  1.000000 
  485. "model/keras_layer/StatefulPartitionedCall:0.257"  1.000000 
  486. "model/keras_layer/StatefulPartitionedCall:0.268"  1.000000 
  487. "model/keras_layer/StatefulPartitionedCall:0.277"  1.000000 
  488. "model/keras_layer/StatefulPartitionedCall:0.296"  1.000000 
  489. "model/keras_layer/StatefulPartitionedCall:0.324"  1.000000 
  490.  "model/keras_layer/StatefulPartitionedCall:0.36"  1.000000 
  491. "model/keras_layer/StatefulPartitionedCall:0.376"  1.000000 
  492. "model/keras_layer/StatefulPartitionedCall:0.381"  1.000000 
  493. "model/keras_layer/StatefulPartitionedCall:0.424"  1.000000 
  494.  "model/keras_layer/StatefulPartitionedCall:0.43"  1.000000 
  495. "model/keras_layer/StatefulPartitionedCall:0.430"  1.000000 
  496. "model/keras_layer/StatefulPartitionedCall:0.435"  1.000000 
  497. "model/keras_layer/StatefulPartitionedCall:0.436"  1.000000 
  498. "model/keras_layer/StatefulPartitionedCall:0.461"  1.000000 
  499. "model/keras_layer/StatefulPartitionedCall:0.494"  1.000000 
  500. "model/keras_layer/StatefulPartitionedCall:0.508"  1.000000 
  501. "model/keras_layer/StatefulPartitionedCall:0.509"  1.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: SUM_SCORE:
    1.  "model/keras_layer/StatefulPartitionedCall:0.50" 838.379362 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.464" 330.585542 ######
    3. "model/keras_layer/StatefulPartitionedCall:0.166" 289.710751 #####
    4. "model/keras_layer/StatefulPartitionedCall:0.458" 258.786829 ####
    5. "model/keras_layer/StatefulPartitionedCall:0.126" 212.260061 ####
    6. "model/keras_layer/StatefulPartitionedCall:0.356" 175.866835 ###
    7. "model/keras_layer/StatefulPartitionedCall:0.127" 163.573925 ###
    8. "model/keras_layer/StatefulPartitionedCall:0.463" 158.028789 ###
    9. "model/keras_layer/StatefulPartitionedCall:0.188" 140.634815 ##
   10.  "model/keras_layer/StatefulPartitionedCall:0.44" 114.152228 ##
   11. "model/keras_layer/StatefulPartitionedCall:0.159" 97.610935 #
   12. "model/keras_layer/StatefulPartitionedCall:0.152" 97.210397 #
   13. "model/keras_layer/StatefulPartitionedCall:0.294" 95.163534 #
   14. "model/keras_layer/StatefulPartitionedCall:0.142" 85.034309 #
   15. "model/keras_layer/StatefulPartitionedCall:0.249" 65.924972 #
   16. "model/keras_layer/StatefulPartitionedCall:0.343" 63.638596 #
   17. "model/keras_layer/StatefulPartitionedCall:0.180" 59.145095 #
   18. "model/keras_layer/StatefulPartitionedCall:0.323" 58.541181 #
   19. "model/keras_layer/StatefulPartitionedCall:0.469" 57.129003 #
   20. "model/keras_layer/StatefulPartitionedCall:0.250" 55.272036 #
   21. "model/keras_layer/StatefulPartitionedCall:0.281" 54.195124 #
   22. "model/keras_layer/StatefulPartitionedCall:0.399" 54.165640 #
   23. "model/keras_layer/StatefulPartitionedCall:0.219" 51.505787 
   24. "model/keras_layer/StatefulPartitionedCall:0.325" 50.337055 
   25. "model/keras_layer/StatefulPartitionedCall:0.354" 46.359333 
   26. "model/keras_layer/StatefulPartitionedCall:0.286" 35.064276 
   27. "model/keras_layer/StatefulPartitionedCall:0.261" 34.881848 
   28. "model/keras_layer/StatefulPartitionedCall:0.153" 33.694494 
   29.   "model/keras_layer/StatefulPartitionedCall:0.1" 33.406319 
   30.  "model/keras_layer/StatefulPartitionedCall:0.10" 32.658641 
   31. "model/keras_layer/StatefulPartitionedCall:0.291" 31.783786 
   32. "model/keras_layer/StatefulPartitionedCall:0.133" 28.579423 
   33. "model/keras_layer/StatefulPartitionedCall:0.135" 27.819231 
   34.  "model/keras_layer/StatefulPartitionedCall:0.56" 27.738759 
   35. "model/keras_layer/StatefulPartitionedCall:0.472" 27.712910 
   36. "model/keras_layer/StatefulPartitionedCall:0.327" 27.210564 
   37. "model/keras_layer/StatefulPartitionedCall:0.362" 25.950382 
   38. "model/keras_layer/StatefulPartitionedCall:0.193" 25.675064 
   39.  "model/keras_layer/StatefulPartitionedCall:0.80" 24.668023 
   40.  "model/keras_layer/StatefulPartitionedCall:0.88" 23.862810 
   41.   "model/keras_layer/StatefulPartitionedCall:0.0" 23.661121 
   42. "model/keras_layer/StatefulPartitionedCall:0.363" 23.438736 
   43. "model/keras_layer/StatefulPartitionedCall:0.310" 23.189145 
   44. "model/keras_layer/StatefulPartitionedCall:0.317" 22.907657 
   45. "model/keras_layer/StatefulPartitionedCall:0.386" 22.428897 
   46.  "model/keras_layer/StatefulPartitionedCall:0.46" 21.825639 
   47. "model/keras_layer/StatefulPartitionedCall:0.289" 21.650669 
   48. "model/keras_layer/StatefulPartitionedCall:0.397" 21.560924 
   49. "model/keras_layer/StatefulPartitionedCall:0.168" 21.542155 
   50. "model/keras_layer/StatefulPartitionedCall:0.457" 21.280251 
   51. "model/keras_layer/StatefulPartitionedCall:0.337" 21.186361 
   52. "model/keras_layer/StatefulPartitionedCall:0.178" 20.704643 
   53. "model/keras_layer/StatefulPartitionedCall:0.272" 20.074757 
   54. "model/keras_layer/StatefulPartitionedCall:0.444" 19.271157 
   55. "model/keras_layer/StatefulPartitionedCall:0.171" 19.155420 
   56. "model/keras_layer/StatefulPartitionedCall:0.144" 19.150801 
   57. "model/keras_layer/StatefulPartitionedCall:0.480" 18.201900 
   58. "model/keras_layer/StatefulPartitionedCall:0.332" 18.179473 
   59. "model/keras_layer/StatefulPartitionedCall:0.285" 18.163323 
   60. "model/keras_layer/StatefulPartitionedCall:0.118" 18.070012 
   61. "model/keras_layer/StatefulPartitionedCall:0.150" 17.865256 
   62.  "model/keras_layer/StatefulPartitionedCall:0.41" 17.832549 
   63. "model/keras_layer/StatefulPartitionedCall:0.511" 17.794719 
   64.  "model/keras_layer/StatefulPartitionedCall:0.98" 17.732325 
   65.  "model/keras_layer/StatefulPartitionedCall:0.61" 17.638717 
   66. "model/keras_layer/StatefulPartitionedCall:0.110" 17.324494 
   67. "model/keras_layer/StatefulPartitionedCall:0.278" 17.108181 
   68. "model/keras_layer/StatefulPartitionedCall:0.226" 16.803146 
   69. "model/keras_layer/StatefulPartitionedCall:0.315" 16.670160 
   70. "model/keras_layer/StatefulPartitionedCall:0.207" 16.607113 
   71.  "model/keras_layer/StatefulPartitionedCall:0.51" 16.134801 
   72.  "model/keras_layer/StatefulPartitionedCall:0.87" 16.044115 
   73.  "model/keras_layer/StatefulPartitionedCall:0.15" 15.892792 
   74. "model/keras_layer/StatefulPartitionedCall:0.432" 15.862979 
   75. "model/keras_layer/StatefulPartitionedCall:0.378" 15.792202 
   76. "model/keras_layer/StatefulPartitionedCall:0.214" 15.630141 
   77. "model/keras_layer/StatefulPartitionedCall:0.414" 15.527817 
   78. "model/keras_layer/StatefulPartitionedCall:0.489" 15.436102 
   79. "model/keras_layer/StatefulPartitionedCall:0.341" 15.239140 
   80. "model/keras_layer/StatefulPartitionedCall:0.373" 15.238609 
   81. "model/keras_layer/StatefulPartitionedCall:0.247" 15.179812 
   82. "model/keras_layer/StatefulPartitionedCall:0.235" 15.094328 
   83. "model/keras_layer/StatefulPartitionedCall:0.449" 14.845643 
   84. "model/keras_layer/StatefulPartitionedCall:0.340" 14.634422 
   85. "model/keras_layer/StatefulPartitionedCall:0.109" 14.537108 
   86. "model/keras_layer/StatefulPartitionedCall:0.233" 14.413046 
   87. "model/keras_layer/StatefulPartitionedCall:0.120" 14.249481 
   88. "model/keras_layer/StatefulPartitionedCall:0.473" 14.064432 
   89. "model/keras_layer/StatefulPartitionedCall:0.104" 14.039795 
   90. "model/keras_layer/StatefulPartitionedCall:0.199" 13.070340 
   91. "model/keras_layer/StatefulPartitionedCall:0.192" 13.042509 
   92. "model/keras_layer/StatefulPartitionedCall:0.415" 13.014721 
   93. "model/keras_layer/StatefulPartitionedCall:0.451" 12.869406 
   94.   "model/keras_layer/StatefulPartitionedCall:0.5" 12.774739 
   95. "model/keras_layer/StatefulPartitionedCall:0.454" 12.640059 
   96. "model/keras_layer/StatefulPartitionedCall:0.300" 12.609441 
   97. "model/keras_layer/StatefulPartitionedCall:0.306" 12.606017 
   98. "model/keras_layer/StatefulPartitionedCall:0.145" 12.400937 
   99. "model/keras_layer/StatefulPartitionedCall:0.322" 12.283406 
  100.  "model/keras_layer/StatefulPartitionedCall:0.37" 12.097364 
  101.  "model/keras_layer/StatefulPartitionedCall:0.17" 11.965903 
  102. "model/keras_layer/StatefulPartitionedCall:0.187" 11.759120 
  103.  "model/keras_layer/StatefulPartitionedCall:0.89" 11.674169 
  104. "model/keras_layer/StatefulPartitionedCall:0.141" 11.597447 
  105. "model/keras_layer/StatefulPartitionedCall:0.218" 11.418217 
  106. "model/keras_layer/StatefulPartitionedCall:0.455" 11.414024 
  107. "model/keras_layer/StatefulPartitionedCall:0.329" 11.221322 
  108. "model/keras_layer/StatefulPartitionedCall:0.147" 11.082280 
  109. "model/keras_layer/StatefulPartitionedCall:0.321" 11.067425 
  110.  "model/keras_layer/StatefulPartitionedCall:0.14" 11.062514 
  111.  "model/keras_layer/StatefulPartitionedCall:0.65" 10.956432 
  112. "model/keras_layer/StatefulPartitionedCall:0.202" 10.823786 
  113. "model/keras_layer/StatefulPartitionedCall:0.438" 10.720394 
  114.  "model/keras_layer/StatefulPartitionedCall:0.66" 10.711686 
  115. "model/keras_layer/StatefulPartitionedCall:0.505" 10.710361 
  116.  "model/keras_layer/StatefulPartitionedCall:0.73" 10.705946 
  117. "model/keras_layer/StatefulPartitionedCall:0.388" 10.656579 
  118. "model/keras_layer/StatefulPartitionedCall:0.275" 10.605852 
  119. "model/keras_layer/StatefulPartitionedCall:0.460" 10.552856 
  120. "model/keras_layer/StatefulPartitionedCall:0.148" 10.468588 
  121.  "model/keras_layer/StatefulPartitionedCall:0.28" 10.457200 
  122. "model/keras_layer/StatefulPartitionedCall:0.255" 10.449962 
  123. "model/keras_layer/StatefulPartitionedCall:0.161" 10.283998 
  124.  "model/keras_layer/StatefulPartitionedCall:0.96" 10.237003 
  125.   "model/keras_layer/StatefulPartitionedCall:0.9" 10.180678 
  126. "model/keras_layer/StatefulPartitionedCall:0.488" 10.039594 
  127. "model/keras_layer/StatefulPartitionedCall:0.500" 10.029043 
  128.  "model/keras_layer/StatefulPartitionedCall:0.67" 10.007469 
  129.  "model/keras_layer/StatefulPartitionedCall:0.29"  9.966911 
  130. "model/keras_layer/StatefulPartitionedCall:0.475"  9.902671 
  131.  "model/keras_layer/StatefulPartitionedCall:0.86"  9.866690 
  132. "model/keras_layer/StatefulPartitionedCall:0.113"  9.787091 
  133. "model/keras_layer/StatefulPartitionedCall:0.446"  9.783535 
  134. "model/keras_layer/StatefulPartitionedCall:0.101"  9.491681 
  135. "model/keras_layer/StatefulPartitionedCall:0.387"  9.444876 
  136. "model/keras_layer/StatefulPartitionedCall:0.453"  9.437917 
  137. "model/keras_layer/StatefulPartitionedCall:0.260"  9.395804 
  138. "model/keras_layer/StatefulPartitionedCall:0.243"  9.394567 
  139. "model/keras_layer/StatefulPartitionedCall:0.328"  9.339020 
  140. "model/keras_layer/StatefulPartitionedCall:0.223"  9.333140 
  141. "model/keras_layer/StatefulPartitionedCall:0.297"  9.251887 
  142. "model/keras_layer/StatefulPartitionedCall:0.498"  9.121491 
  143. "model/keras_layer/StatefulPartitionedCall:0.357"  9.019604 
  144.  "model/keras_layer/StatefulPartitionedCall:0.21"  9.003671 
  145. "model/keras_layer/StatefulPartitionedCall:0.304"  8.763306 
  146.  "model/keras_layer/StatefulPartitionedCall:0.94"  8.750402 
  147. "model/keras_layer/StatefulPartitionedCall:0.185"  8.736767 
  148. "model/keras_layer/StatefulPartitionedCall:0.350"  8.671147 
  149. "model/keras_layer/StatefulPartitionedCall:0.477"  8.638697 
  150. "model/keras_layer/StatefulPartitionedCall:0.485"  8.560580 
  151. "model/keras_layer/StatefulPartitionedCall:0.440"  8.513065 
  152. "model/keras_layer/StatefulPartitionedCall:0.298"  8.478455 
  153. "model/keras_layer/StatefulPartitionedCall:0.163"  8.437502 
  154. "model/keras_layer/StatefulPartitionedCall:0.428"  8.422228 
  155. "model/keras_layer/StatefulPartitionedCall:0.117"  8.385912 
  156. "model/keras_layer/StatefulPartitionedCall:0.495"  8.378498 
  157. "model/keras_layer/StatefulPartitionedCall:0.138"  8.371942 
  158. "model/keras_layer/StatefulPartitionedCall:0.486"  8.330224 
  159. "model/keras_layer/StatefulPartitionedCall:0.154"  8.276672 
  160. "model/keras_layer/StatefulPartitionedCall:0.160"  8.115374 
  161. "model/keras_layer/StatefulPartitionedCall:0.334"  8.112398 
  162. "model/keras_layer/StatefulPartitionedCall:0.462"  8.069798 
  163. "model/keras_layer/StatefulPartitionedCall:0.419"  8.061528 
  164. "model/keras_layer/StatefulPartitionedCall:0.426"  7.897863 
  165.  "model/keras_layer/StatefulPartitionedCall:0.47"  7.884702 
  166. "model/keras_layer/StatefulPartitionedCall:0.130"  7.826164 
  167. "model/keras_layer/StatefulPartitionedCall:0.471"  7.697936 
  168. "model/keras_layer/StatefulPartitionedCall:0.375"  7.582082 
  169. "model/keras_layer/StatefulPartitionedCall:0.392"  7.534869 
  170. "model/keras_layer/StatefulPartitionedCall:0.467"  7.492224 
  171.  "model/keras_layer/StatefulPartitionedCall:0.20"  7.469165 
  172. "model/keras_layer/StatefulPartitionedCall:0.353"  7.454562 
  173. "model/keras_layer/StatefulPartitionedCall:0.136"  7.449279 
  174. "model/keras_layer/StatefulPartitionedCall:0.368"  7.422351 
  175. "model/keras_layer/StatefulPartitionedCall:0.389"  7.346667 
  176. "model/keras_layer/StatefulPartitionedCall:0.252"  7.335040 
  177. "model/keras_layer/StatefulPartitionedCall:0.497"  7.305868 
  178. "model/keras_layer/StatefulPartitionedCall:0.418"  7.281423 
  179.  "model/keras_layer/StatefulPartitionedCall:0.93"  7.195282 
  180. "model/keras_layer/StatefulPartitionedCall:0.265"  7.083741 
  181. "model/keras_layer/StatefulPartitionedCall:0.336"  7.079395 
  182. "model/keras_layer/StatefulPartitionedCall:0.372"  7.070156 
  183. "model/keras_layer/StatefulPartitionedCall:0.411"  7.032047 
  184. "model/keras_layer/StatefulPartitionedCall:0.441"  7.017258 
  185. "model/keras_layer/StatefulPartitionedCall:0.326"  7.010852 
  186. "model/keras_layer/StatefulPartitionedCall:0.452"  6.956202 
  187.  "model/keras_layer/StatefulPartitionedCall:0.57"  6.945561 
  188. "model/keras_layer/StatefulPartitionedCall:0.448"  6.907252 
  189. "model/keras_layer/StatefulPartitionedCall:0.384"  6.842926 
  190. "model/keras_layer/StatefulPartitionedCall:0.358"  6.823216 
  191. "model/keras_layer/StatefulPartitionedCall:0.395"  6.785542 
  192. "model/keras_layer/StatefulPartitionedCall:0.439"  6.742225 
  193. "model/keras_layer/StatefulPartitionedCall:0.437"  6.716798 
  194. "model/keras_layer/StatefulPartitionedCall:0.490"  6.578961 
  195. "model/keras_layer/StatefulPartitionedCall:0.200"  6.575920 
  196. "model/keras_layer/StatefulPartitionedCall:0.155"  6.515163 
  197. "model/keras_layer/StatefulPartitionedCall:0.164"  6.512211 
  198.  "model/keras_layer/StatefulPartitionedCall:0.33"  6.455061 
  199. "model/keras_layer/StatefulPartitionedCall:0.256"  6.433432 
  200. "model/keras_layer/StatefulPartitionedCall:0.195"  6.394581 
  201.  "model/keras_layer/StatefulPartitionedCall:0.30"  6.393787 
  202. "model/keras_layer/StatefulPartitionedCall:0.176"  6.298694 
  203.  "model/keras_layer/StatefulPartitionedCall:0.18"  6.244403 
  204. "model/keras_layer/StatefulPartitionedCall:0.201"  6.242590 
  205. "model/keras_layer/StatefulPartitionedCall:0.338"  6.215177 
  206. "model/keras_layer/StatefulPartitionedCall:0.484"  6.135061 
  207. "model/keras_layer/StatefulPartitionedCall:0.470"  6.053713 
  208.  "model/keras_layer/StatefulPartitionedCall:0.11"  6.023713 
  209. "model/keras_layer/StatefulPartitionedCall:0.355"  6.015947 
  210. "model/keras_layer/StatefulPartitionedCall:0.422"  5.975865 
  211. "model/keras_layer/StatefulPartitionedCall:0.360"  5.975242 
  212.   "model/keras_layer/StatefulPartitionedCall:0.2"  5.968374 
  213. "model/keras_layer/StatefulPartitionedCall:0.290"  5.967661 
  214. "model/keras_layer/StatefulPartitionedCall:0.412"  5.934552 
  215.   "model/keras_layer/StatefulPartitionedCall:0.8"  5.884568 
  216. "model/keras_layer/StatefulPartitionedCall:0.346"  5.868656 
  217. "model/keras_layer/StatefulPartitionedCall:0.476"  5.859954 
  218. "model/keras_layer/StatefulPartitionedCall:0.427"  5.844647 
  219. "model/keras_layer/StatefulPartitionedCall:0.504"  5.768807 
  220. "model/keras_layer/StatefulPartitionedCall:0.433"  5.765950 
  221. "model/keras_layer/StatefulPartitionedCall:0.344"  5.736910 
  222. "model/keras_layer/StatefulPartitionedCall:0.254"  5.730472 
  223.  "model/keras_layer/StatefulPartitionedCall:0.71"  5.700085 
  224. "model/keras_layer/StatefulPartitionedCall:0.510"  5.692781 
  225. "model/keras_layer/StatefulPartitionedCall:0.408"  5.692421 
  226. "model/keras_layer/StatefulPartitionedCall:0.231"  5.692175 
  227.  "model/keras_layer/StatefulPartitionedCall:0.92"  5.634529 
  228. "model/keras_layer/StatefulPartitionedCall:0.280"  5.622100 
  229. "model/keras_layer/StatefulPartitionedCall:0.206"  5.519398 
  230. "model/keras_layer/StatefulPartitionedCall:0.205"  5.505467 
  231. "model/keras_layer/StatefulPartitionedCall:0.359"  5.493744 
  232. "model/keras_layer/StatefulPartitionedCall:0.175"  5.451168 
  233.  "model/keras_layer/StatefulPartitionedCall:0.13"  5.412627 
  234. "model/keras_layer/StatefulPartitionedCall:0.194"  5.405903 
  235. "model/keras_layer/StatefulPartitionedCall:0.351"  5.393027 
  236. "model/keras_layer/StatefulPartitionedCall:0.410"  5.384902 
  237.  "model/keras_layer/StatefulPartitionedCall:0.60"  5.365257 
  238. "model/keras_layer/StatefulPartitionedCall:0.246"  5.331075 
  239. "model/keras_layer/StatefulPartitionedCall:0.228"  5.284374 
  240. "model/keras_layer/StatefulPartitionedCall:0.253"  5.275314 
  241. "model/keras_layer/StatefulPartitionedCall:0.103"  5.223350 
  242. "model/keras_layer/StatefulPartitionedCall:0.447"  5.202484 
  243.  "model/keras_layer/StatefulPartitionedCall:0.48"  5.201715 
  244. "model/keras_layer/StatefulPartitionedCall:0.393"  5.195196 
  245. "model/keras_layer/StatefulPartitionedCall:0.112"  5.174736 
  246. "model/keras_layer/StatefulPartitionedCall:0.468"  5.154760 
  247. "model/keras_layer/StatefulPartitionedCall:0.279"  5.152297 
  248.  "model/keras_layer/StatefulPartitionedCall:0.27"  5.150008 
  249. "model/keras_layer/StatefulPartitionedCall:0.391"  5.145888 
  250. "model/keras_layer/StatefulPartitionedCall:0.301"  5.111114 
  251. "model/keras_layer/StatefulPartitionedCall:0.222"  5.102308 
  252.  "model/keras_layer/StatefulPartitionedCall:0.32"  5.082763 
  253. "model/keras_layer/StatefulPartitionedCall:0.259"  5.075625 
  254. "model/keras_layer/StatefulPartitionedCall:0.401"  5.058767 
  255. "model/keras_layer/StatefulPartitionedCall:0.347"  4.903329 
  256. "model/keras_layer/StatefulPartitionedCall:0.370"  4.897512 
  257. "model/keras_layer/StatefulPartitionedCall:0.335"  4.850670 
  258. "model/keras_layer/StatefulPartitionedCall:0.266"  4.844850 
  259. "model/keras_layer/StatefulPartitionedCall:0.308"  4.809818 
  260. "model/keras_layer/StatefulPartitionedCall:0.295"  4.791184 
  261. "model/keras_layer/StatefulPartitionedCall:0.114"  4.789980 
  262. "model/keras_layer/StatefulPartitionedCall:0.400"  4.748586 
  263.  "model/keras_layer/StatefulPartitionedCall:0.77"  4.734667 
  264. "model/keras_layer/StatefulPartitionedCall:0.122"  4.717070 
  265. "model/keras_layer/StatefulPartitionedCall:0.333"  4.715997 
  266. "model/keras_layer/StatefulPartitionedCall:0.303"  4.697519 
  267. "model/keras_layer/StatefulPartitionedCall:0.134"  4.681885 
  268. "model/keras_layer/StatefulPartitionedCall:0.129"  4.657790 
  269. "model/keras_layer/StatefulPartitionedCall:0.409"  4.631557 
  270. "model/keras_layer/StatefulPartitionedCall:0.276"  4.583160 
  271.  "model/keras_layer/StatefulPartitionedCall:0.95"  4.582428 
  272. "model/keras_layer/StatefulPartitionedCall:0.115"  4.549172 
  273. "model/keras_layer/StatefulPartitionedCall:0.267"  4.544612 
  274. "model/keras_layer/StatefulPartitionedCall:0.140"  4.533141 
  275. "model/keras_layer/StatefulPartitionedCall:0.196"  4.499229 
  276. "model/keras_layer/StatefulPartitionedCall:0.423"  4.489519 
  277. "model/keras_layer/StatefulPartitionedCall:0.361"  4.485020 
  278.  "model/keras_layer/StatefulPartitionedCall:0.39"  4.471374 
  279. "model/keras_layer/StatefulPartitionedCall:0.262"  4.430145 
  280. "model/keras_layer/StatefulPartitionedCall:0.407"  4.402494 
  281. "model/keras_layer/StatefulPartitionedCall:0.158"  4.369274 
  282. "model/keras_layer/StatefulPartitionedCall:0.106"  4.337908 
  283. "model/keras_layer/StatefulPartitionedCall:0.139"  4.322775 
  284. "model/keras_layer/StatefulPartitionedCall:0.183"  4.297304 
  285. "model/keras_layer/StatefulPartitionedCall:0.420"  4.296876 
  286. "model/keras_layer/StatefulPartitionedCall:0.217"  4.295840 
  287.  "model/keras_layer/StatefulPartitionedCall:0.40"  4.287280 
  288. "model/keras_layer/StatefulPartitionedCall:0.312"  4.278814 
  289.  "model/keras_layer/StatefulPartitionedCall:0.22"  4.271490 
  290. "model/keras_layer/StatefulPartitionedCall:0.299"  4.271226 
  291. "model/keras_layer/StatefulPartitionedCall:0.221"  4.260014 
  292. "model/keras_layer/StatefulPartitionedCall:0.143"  4.251409 
  293. "model/keras_layer/StatefulPartitionedCall:0.479"  4.243402 
  294. "model/keras_layer/StatefulPartitionedCall:0.383"  4.236765 
  295. "model/keras_layer/StatefulPartitionedCall:0.131"  4.175798 
  296. "model/keras_layer/StatefulPartitionedCall:0.313"  4.148036 
  297. "model/keras_layer/StatefulPartitionedCall:0.403"  4.142175 
  298. "model/keras_layer/StatefulPartitionedCall:0.232"  4.119152 
  299.  "model/keras_layer/StatefulPartitionedCall:0.64"  4.105359 
  300. "model/keras_layer/StatefulPartitionedCall:0.258"  4.100108 
  301.  "model/keras_layer/StatefulPartitionedCall:0.49"  4.070590 
  302.  "model/keras_layer/StatefulPartitionedCall:0.24"  4.066158 
  303. "model/keras_layer/StatefulPartitionedCall:0.169"  4.055565 
  304. "model/keras_layer/StatefulPartitionedCall:0.506"  4.049839 
  305. "model/keras_layer/StatefulPartitionedCall:0.434"  4.045234 
  306. "model/keras_layer/StatefulPartitionedCall:0.305"  4.042371 
  307.  "model/keras_layer/StatefulPartitionedCall:0.31"  3.986282 
  308. "model/keras_layer/StatefulPartitionedCall:0.425"  3.982928 
  309.  "model/keras_layer/StatefulPartitionedCall:0.81"  3.979033 
  310.  "model/keras_layer/StatefulPartitionedCall:0.85"  3.964330 
  311. "model/keras_layer/StatefulPartitionedCall:0.339"  3.952287 
  312. "model/keras_layer/StatefulPartitionedCall:0.288"  3.931458 
  313.   "model/keras_layer/StatefulPartitionedCall:0.3"  3.924181 
  314. "model/keras_layer/StatefulPartitionedCall:0.314"  3.899695 
  315. "model/keras_layer/StatefulPartitionedCall:0.421"  3.782109 
  316. "model/keras_layer/StatefulPartitionedCall:0.105"  3.780756 
  317.  "model/keras_layer/StatefulPartitionedCall:0.69"  3.773343 
  318. "model/keras_layer/StatefulPartitionedCall:0.239"  3.767121 
  319. "model/keras_layer/StatefulPartitionedCall:0.483"  3.761714 
  320. "model/keras_layer/StatefulPartitionedCall:0.186"  3.742255 
  321.  "model/keras_layer/StatefulPartitionedCall:0.54"  3.713073 
  322. "model/keras_layer/StatefulPartitionedCall:0.216"  3.702136 
  323. "model/keras_layer/StatefulPartitionedCall:0.374"  3.696787 
  324. "model/keras_layer/StatefulPartitionedCall:0.478"  3.679575 
  325. "model/keras_layer/StatefulPartitionedCall:0.102"  3.657214 
  326. "model/keras_layer/StatefulPartitionedCall:0.465"  3.648336 
  327. "model/keras_layer/StatefulPartitionedCall:0.165"  3.610599 
  328. "model/keras_layer/StatefulPartitionedCall:0.204"  3.597861 
  329. "model/keras_layer/StatefulPartitionedCall:0.108"  3.582952 
  330. "model/keras_layer/StatefulPartitionedCall:0.197"  3.538006 
  331.  "model/keras_layer/StatefulPartitionedCall:0.97"  3.535416 
  332. "model/keras_layer/StatefulPartitionedCall:0.316"  3.534409 
  333. "model/keras_layer/StatefulPartitionedCall:0.442"  3.488580 
  334. "model/keras_layer/StatefulPartitionedCall:0.184"  3.483041 
  335. "model/keras_layer/StatefulPartitionedCall:0.149"  3.481915 
  336. "model/keras_layer/StatefulPartitionedCall:0.367"  3.473952 
  337. "model/keras_layer/StatefulPartitionedCall:0.382"  3.462887 
  338. "model/keras_layer/StatefulPartitionedCall:0.377"  3.460358 
  339.  "model/keras_layer/StatefulPartitionedCall:0.26"  3.457099 
  340. "model/keras_layer/StatefulPartitionedCall:0.345"  3.411288 
  341.  "model/keras_layer/StatefulPartitionedCall:0.16"  3.398802 
  342.  "model/keras_layer/StatefulPartitionedCall:0.45"  3.368484 
  343. "model/keras_layer/StatefulPartitionedCall:0.146"  3.360452 
  344.   "model/keras_layer/StatefulPartitionedCall:0.6"  3.346616 
  345. "model/keras_layer/StatefulPartitionedCall:0.241"  3.332303 
  346. "model/keras_layer/StatefulPartitionedCall:0.242"  3.274217 
  347. "model/keras_layer/StatefulPartitionedCall:0.271"  3.259943 
  348. "model/keras_layer/StatefulPartitionedCall:0.450"  3.246402 
  349.  "model/keras_layer/StatefulPartitionedCall:0.34"  3.244836 
  350. "model/keras_layer/StatefulPartitionedCall:0.220"  3.234319 
  351. "model/keras_layer/StatefulPartitionedCall:0.273"  3.166152 
  352. "model/keras_layer/StatefulPartitionedCall:0.482"  3.162813 
  353. "model/keras_layer/StatefulPartitionedCall:0.302"  3.162668 
  354.  "model/keras_layer/StatefulPartitionedCall:0.74"  3.141260 
  355. "model/keras_layer/StatefulPartitionedCall:0.380"  3.124364 
  356. "model/keras_layer/StatefulPartitionedCall:0.208"  3.088909 
  357. "model/keras_layer/StatefulPartitionedCall:0.507"  3.042860 
  358.  "model/keras_layer/StatefulPartitionedCall:0.75"  3.040051 
  359. "model/keras_layer/StatefulPartitionedCall:0.237"  3.035536 
  360. "model/keras_layer/StatefulPartitionedCall:0.177"  3.028791 
  361. "model/keras_layer/StatefulPartitionedCall:0.189"  3.014158 
  362.  "model/keras_layer/StatefulPartitionedCall:0.53"  3.009051 
  363. "model/keras_layer/StatefulPartitionedCall:0.330"  3.006781 
  364. "model/keras_layer/StatefulPartitionedCall:0.348"  2.996314 
  365. "model/keras_layer/StatefulPartitionedCall:0.369"  2.978435 
  366. "model/keras_layer/StatefulPartitionedCall:0.227"  2.976948 
  367. "model/keras_layer/StatefulPartitionedCall:0.379"  2.970680 
  368. "model/keras_layer/StatefulPartitionedCall:0.459"  2.959983 
  369. "model/keras_layer/StatefulPartitionedCall:0.174"  2.951204 
  370. "model/keras_layer/StatefulPartitionedCall:0.270"  2.942866 
  371.  "model/keras_layer/StatefulPartitionedCall:0.55"  2.929567 
  372.  "model/keras_layer/StatefulPartitionedCall:0.99"  2.914323 
  373.  "model/keras_layer/StatefulPartitionedCall:0.63"  2.914018 
  374. "model/keras_layer/StatefulPartitionedCall:0.238"  2.860308 
  375. "model/keras_layer/StatefulPartitionedCall:0.190"  2.848553 
  376. "model/keras_layer/StatefulPartitionedCall:0.234"  2.829395 
  377. "model/keras_layer/StatefulPartitionedCall:0.248"  2.811553 
  378. "model/keras_layer/StatefulPartitionedCall:0.466"  2.752703 
  379.   "model/keras_layer/StatefulPartitionedCall:0.7"  2.745605 
  380. "model/keras_layer/StatefulPartitionedCall:0.100"  2.743648 
  381.  "model/keras_layer/StatefulPartitionedCall:0.19"  2.732798 
  382.  "model/keras_layer/StatefulPartitionedCall:0.42"  2.725202 
  383. "model/keras_layer/StatefulPartitionedCall:0.170"  2.720832 
  384. "model/keras_layer/StatefulPartitionedCall:0.209"  2.677810 
  385.  "model/keras_layer/StatefulPartitionedCall:0.84"  2.668112 
  386.  "model/keras_layer/StatefulPartitionedCall:0.52"  2.641139 
  387. "model/keras_layer/StatefulPartitionedCall:0.502"  2.600776 
  388.  "model/keras_layer/StatefulPartitionedCall:0.62"  2.583134 
  389. "model/keras_layer/StatefulPartitionedCall:0.404"  2.582792 
  390. "model/keras_layer/StatefulPartitionedCall:0.292"  2.579436 
  391. "model/keras_layer/StatefulPartitionedCall:0.203"  2.578253 
  392. "model/keras_layer/StatefulPartitionedCall:0.179"  2.573130 
  393. "model/keras_layer/StatefulPartitionedCall:0.402"  2.559845 
  394.  "model/keras_layer/StatefulPartitionedCall:0.70"  2.553216 
  395.  "model/keras_layer/StatefulPartitionedCall:0.79"  2.547396 
  396. "model/keras_layer/StatefulPartitionedCall:0.349"  2.546764 
  397. "model/keras_layer/StatefulPartitionedCall:0.456"  2.534803 
  398.  "model/keras_layer/StatefulPartitionedCall:0.35"  2.524507 
  399. "model/keras_layer/StatefulPartitionedCall:0.172"  2.500489 
  400.  "model/keras_layer/StatefulPartitionedCall:0.91"  2.484547 
  401.  "model/keras_layer/StatefulPartitionedCall:0.90"  2.469070 
  402. "model/keras_layer/StatefulPartitionedCall:0.245"  2.439867 
  403. "model/keras_layer/StatefulPartitionedCall:0.406"  2.402767 
  404.  "model/keras_layer/StatefulPartitionedCall:0.23"  2.371818 
  405. "model/keras_layer/StatefulPartitionedCall:0.493"  2.360209 
  406. "model/keras_layer/StatefulPartitionedCall:0.487"  2.346846 
  407. "model/keras_layer/StatefulPartitionedCall:0.501"  2.346362 
  408. "model/keras_layer/StatefulPartitionedCall:0.394"  2.305750 
  409.  "model/keras_layer/StatefulPartitionedCall:0.78"  2.300405 
  410.  "model/keras_layer/StatefulPartitionedCall:0.72"  2.286669 
  411. "model/keras_layer/StatefulPartitionedCall:0.283"  2.286359 
  412. "model/keras_layer/StatefulPartitionedCall:0.215"  2.283919 
  413. "model/keras_layer/StatefulPartitionedCall:0.319"  2.224205 
  414. "model/keras_layer/StatefulPartitionedCall:0.509"  2.194776 
  415. "model/keras_layer/StatefulPartitionedCall:0.309"  2.189398 
  416.  "model/keras_layer/StatefulPartitionedCall:0.83"  2.187086 
  417.  "model/keras_layer/StatefulPartitionedCall:0.38"  2.184138 
  418. "model/keras_layer/StatefulPartitionedCall:0.429"  2.162904 
  419. "model/keras_layer/StatefulPartitionedCall:0.496"  2.151104 
  420. "model/keras_layer/StatefulPartitionedCall:0.236"  2.125714 
  421. "model/keras_layer/StatefulPartitionedCall:0.181"  2.124045 
  422. "model/keras_layer/StatefulPartitionedCall:0.263"  2.060562 
  423. "model/keras_layer/StatefulPartitionedCall:0.157"  2.058507 
  424. "model/keras_layer/StatefulPartitionedCall:0.225"  2.037724 
  425. "model/keras_layer/StatefulPartitionedCall:0.364"  2.031506 
  426. "model/keras_layer/StatefulPartitionedCall:0.371"  2.016917 
  427. "model/keras_layer/StatefulPartitionedCall:0.413"  2.010814 
  428. "model/keras_layer/StatefulPartitionedCall:0.417"  2.001799 
  429. "model/keras_layer/StatefulPartitionedCall:0.212"  1.989237 
  430.  "model/keras_layer/StatefulPartitionedCall:0.12"  1.976753 
  431. "model/keras_layer/StatefulPartitionedCall:0.264"  1.975281 
  432. "model/keras_layer/StatefulPartitionedCall:0.167"  1.971795 
  433. "model/keras_layer/StatefulPartitionedCall:0.342"  1.952435 
  434. "model/keras_layer/StatefulPartitionedCall:0.474"  1.941928 
  435.  "model/keras_layer/StatefulPartitionedCall:0.76"  1.929990 
  436. "model/keras_layer/StatefulPartitionedCall:0.365"  1.928158 
  437. "model/keras_layer/StatefulPartitionedCall:0.491"  1.876092 
  438. "model/keras_layer/StatefulPartitionedCall:0.123"  1.802867 
  439. "model/keras_layer/StatefulPartitionedCall:0.398"  1.759222 
  440. "model/keras_layer/StatefulPartitionedCall:0.492"  1.731385 
  441. "model/keras_layer/StatefulPartitionedCall:0.293"  1.671152 
  442. "model/keras_layer/StatefulPartitionedCall:0.287"  1.662094 
  443. "model/keras_layer/StatefulPartitionedCall:0.151"  1.629292 
  444. "model/keras_layer/StatefulPartitionedCall:0.331"  1.572921 
  445. "model/keras_layer/StatefulPartitionedCall:0.390"  1.571510 
  446. "model/keras_layer/StatefulPartitionedCall:0.244"  1.568583 
  447. "model/keras_layer/StatefulPartitionedCall:0.307"  1.534373 
  448. "model/keras_layer/StatefulPartitionedCall:0.282"  1.499824 
  449. "model/keras_layer/StatefulPartitionedCall:0.416"  1.475428 
  450. "model/keras_layer/StatefulPartitionedCall:0.269"  1.454821 
  451. "model/keras_layer/StatefulPartitionedCall:0.251"  1.452584 
  452. "model/keras_layer/StatefulPartitionedCall:0.284"  1.433233 
  453.  "model/keras_layer/StatefulPartitionedCall:0.68"  1.431660 
  454. "model/keras_layer/StatefulPartitionedCall:0.366"  1.400220 
  455. "model/keras_layer/StatefulPartitionedCall:0.224"  1.396701 
  456. "model/keras_layer/StatefulPartitionedCall:0.257"  1.364005 
  457. "model/keras_layer/StatefulPartitionedCall:0.503"  1.352172 
  458. "model/keras_layer/StatefulPartitionedCall:0.128"  1.344210 
  459.  "model/keras_layer/StatefulPartitionedCall:0.58"  1.239213 
  460. "model/keras_layer/StatefulPartitionedCall:0.499"  1.222429 
  461. "model/keras_layer/StatefulPartitionedCall:0.116"  1.195167 
  462. "model/keras_layer/StatefulPartitionedCall:0.431"  1.166997 
  463. "model/keras_layer/StatefulPartitionedCall:0.137"  1.121556 
  464. "model/keras_layer/StatefulPartitionedCall:0.211"  1.097893 
  465. "model/keras_layer/StatefulPartitionedCall:0.198"  1.080846 
  466. "model/keras_layer/StatefulPartitionedCall:0.156"  1.078826 
  467. "model/keras_layer/StatefulPartitionedCall:0.277"  1.049874 
  468. "model/keras_layer/StatefulPartitionedCall:0.274"  1.048964 
  469. "model/keras_layer/StatefulPartitionedCall:0.443"  1.044820 
  470. "model/keras_layer/StatefulPartitionedCall:0.430"  1.040101 
  471. "model/keras_layer/StatefulPartitionedCall:0.376"  1.039601 
  472. "model/keras_layer/StatefulPartitionedCall:0.119"  1.002154 
  473. "model/keras_layer/StatefulPartitionedCall:0.173"  0.975196 
  474. "model/keras_layer/StatefulPartitionedCall:0.125"  0.967407 
  475. "model/keras_layer/StatefulPartitionedCall:0.107"  0.892719 
  476.  "model/keras_layer/StatefulPartitionedCall:0.43"  0.831053 
  477. "model/keras_layer/StatefulPartitionedCall:0.124"  0.810664 
  478. "model/keras_layer/StatefulPartitionedCall:0.396"  0.791375 
  479. "model/keras_layer/StatefulPartitionedCall:0.324"  0.769868 
  480. "model/keras_layer/StatefulPartitionedCall:0.182"  0.747309 
  481. "model/keras_layer/StatefulPartitionedCall:0.132"  0.741131 
  482.  "model/keras_layer/StatefulPartitionedCall:0.82"  0.702190 
  483.  "model/keras_layer/StatefulPartitionedCall:0.36"  0.700087 
  484. "model/keras_layer/StatefulPartitionedCall:0.405"  0.675658 
  485. "model/keras_layer/StatefulPartitionedCall:0.213"  0.645575 
  486. "model/keras_layer/StatefulPartitionedCall:0.268"  0.548454 
  487. "model/keras_layer/StatefulPartitionedCall:0.311"  0.483642 
  488.   "model/keras_layer/StatefulPartitionedCall:0.4"  0.461858 
  489.  "model/keras_layer/StatefulPartitionedCall:0.25"  0.459785 
  490. "model/keras_layer/StatefulPartitionedCall:0.435"  0.449380 
  491. "model/keras_layer/StatefulPartitionedCall:0.494"  0.444434 
  492. "model/keras_layer/StatefulPartitionedCall:0.508"  0.423675 
  493. "model/keras_layer/StatefulPartitionedCall:0.229"  0.398815 
  494. "model/keras_layer/StatefulPartitionedCall:0.461"  0.375449 
  495. "model/keras_layer/StatefulPartitionedCall:0.240"  0.373172 
  496. "model/keras_layer/StatefulPartitionedCall:0.424"  0.371246 
  497. "model/keras_layer/StatefulPartitionedCall:0.352"  0.346150 
  498. "model/keras_layer/StatefulPartitionedCall:0.381"  0.227352 
  499. "model/keras_layer/StatefulPartitionedCall:0.162"  0.195585 
  500. "model/keras_layer/StatefulPartitionedCall:0.436"  0.166405 
  501. "model/keras_layer/StatefulPartitionedCall:0.296"  0.050727 
```
</div>
    
    
    
<div class="k-default-codeblock">
```
Loss: BINOMIAL_LOG_LIKELIHOOD
Validation loss value: 0.806777
Number of trees per iteration: 1
Node format: NOT_SET
Number of trees: 137
Total number of nodes: 6671
```
</div>
    
<div class="k-default-codeblock">
```
Number of nodes by tree:
Count: 137 Average: 48.6934 StdDev: 9.91023
Min: 21 Max: 63 Ignored: 0
----------------------------------------------
[ 21, 23)  1   0.73%   0.73%
[ 23, 25)  1   0.73%   1.46%
[ 25, 27)  0   0.00%   1.46%
[ 27, 29)  1   0.73%   2.19%
[ 29, 31)  3   2.19%   4.38% #
[ 31, 33)  3   2.19%   6.57% #
[ 33, 36)  9   6.57%  13.14% ####
[ 36, 38)  4   2.92%  16.06% ##
[ 38, 40)  4   2.92%  18.98% ##
[ 40, 42)  8   5.84%  24.82% ####
[ 42, 44)  8   5.84%  30.66% ####
[ 44, 46)  9   6.57%  37.23% ####
[ 46, 48)  7   5.11%  42.34% ###
[ 48, 51) 10   7.30%  49.64% #####
[ 51, 53) 13   9.49%  59.12% ######
[ 53, 55) 10   7.30%  66.42% #####
[ 55, 57) 10   7.30%  73.72% #####
[ 57, 59)  6   4.38%  78.10% ###
[ 59, 61)  8   5.84%  83.94% ####
[ 61, 63] 22  16.06% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Depth by leafs:
Count: 3404 Average: 4.81052 StdDev: 0.557183
Min: 1 Max: 5 Ignored: 0
----------------------------------------------
[ 1, 2)    6   0.18%   0.18%
[ 2, 3)   38   1.12%   1.29%
[ 3, 4)  117   3.44%   4.73%
[ 4, 5)  273   8.02%  12.75% #
[ 5, 5] 2970  87.25% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Number of training obs by leaf:
Count: 3404 Average: 248.806 StdDev: 517.403
Min: 5 Max: 4709 Ignored: 0
----------------------------------------------
[    5,  240) 2615  76.82%  76.82% ##########
[  240,  475)  243   7.14%  83.96% #
[  475,  710)  162   4.76%  88.72% #
[  710,  946)  104   3.06%  91.77%
[  946, 1181)   80   2.35%  94.12%
[ 1181, 1416)   48   1.41%  95.53%
[ 1416, 1651)   44   1.29%  96.83%
[ 1651, 1887)   27   0.79%  97.62%
[ 1887, 2122)   18   0.53%  98.15%
[ 2122, 2357)   19   0.56%  98.71%
[ 2357, 2592)   10   0.29%  99.00%
[ 2592, 2828)    6   0.18%  99.18%
[ 2828, 3063)    8   0.24%  99.41%
[ 3063, 3298)    7   0.21%  99.62%
[ 3298, 3533)    3   0.09%  99.71%
[ 3533, 3769)    5   0.15%  99.85%
[ 3769, 4004)    2   0.06%  99.91%
[ 4004, 4239)    1   0.03%  99.94%
[ 4239, 4474)    1   0.03%  99.97%
[ 4474, 4709]    1   0.03% 100.00%
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes:
	41 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	32 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	31 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	28 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	27 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	25 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	23 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	22 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	22 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	22 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.451 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.414 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.61 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.47 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.438 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.3 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.201 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.9 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.86 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.340 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.33 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.228 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.223 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.202 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.20 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.5 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.495 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.447 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.441 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.410 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.353 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.336 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.28 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.239 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.218 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.163 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.15 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.143 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.81 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.57 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.488 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.475 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.433 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.428 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.422 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.384 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.370 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.355 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.326 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.303 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.30 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.270 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.258 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.227 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.217 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.21 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.205 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.196 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.186 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.14 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.136 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.122 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.97 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.94 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.75 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.67 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.66 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.504 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.490 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.478 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.477 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.465 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.403 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.389 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.378 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.377 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.346 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.32 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.313 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.312 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.309 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.295 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.290 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.288 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.279 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.271 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.263 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.259 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.220 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.194 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.175 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.154 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.145 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.112 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.108 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.90 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.77 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.70 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.52 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.507 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.506 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.493 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.474 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.471 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.425 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.395 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.371 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.349 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.347 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.345 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.342 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.339 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.335 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.308 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.305 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.301 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.293 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.267 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.248 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.245 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.234 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.232 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.23 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.216 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.215 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.203 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.200 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.2 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.195 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.19 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.172 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.170 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.165 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.157 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.138 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.134 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.131 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.123 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.105 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.91 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.84 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.83 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.74 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.72 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.7 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.64 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.55 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.53 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.503 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.501 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.491 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.49 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.452 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.434 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.421 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.413 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.409 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.408 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.407 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.406 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.404 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.400 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.398 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.391 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.390 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.382 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.38 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.361 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.359 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.334 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.331 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.330 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.316 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.307 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.283 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.282 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.280 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.251 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.238 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.204 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.181 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.167 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.158 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.151 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.146 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.137 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.115 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.93 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.82 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.502 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.499 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.496 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.492 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.487 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.482 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.456 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.431 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.40 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.4 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.396 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.394 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.352 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.35 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.311 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.274 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.269 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.264 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.26 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.244 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.242 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.224 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.221 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.211 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.206 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.198 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.182 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.124 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.508 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.494 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.461 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.436 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.430 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.43 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.424 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.376 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.36 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.324 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.296 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.277 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.257 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.240 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.236 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.213 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.212 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.177 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.162 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.125 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.12 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.119 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.107 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 0:
	14 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 1:
	14 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.438 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.488 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.398 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.377 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.313 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.309 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.263 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.170 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 2:
	25 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.438 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.259 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.70 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.504 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.5 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.478 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.410 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.4 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.371 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.353 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.345 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.326 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.309 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.3 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.295 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.282 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.279 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.271 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.244 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.216 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.163 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.9 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.86 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.84 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.83 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.61 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.508 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.495 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.494 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.493 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.490 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.488 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.477 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.465 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.461 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.451 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.447 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.441 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.428 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.422 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.421 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.398 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.382 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.378 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.377 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.340 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.313 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.305 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.30 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.270 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.269 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.263 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.26 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.239 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.238 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.23 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.228 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.227 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.223 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.218 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.215 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.203 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.202 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.196 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.194 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.186 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.172 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.170 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.157 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.145 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.134 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.122 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.112 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 3:
	33 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	22 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.3 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.451 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.86 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.438 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.5 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.414 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.353 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.336 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.33 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.217 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.145 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.9 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.61 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.493 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.47 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.410 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.371 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.295 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.271 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.259 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.239 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.234 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.228 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.223 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.218 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.163 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.91 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.70 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.66 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.57 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.503 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.495 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.488 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.478 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.447 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.398 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.382 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.378 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.345 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.340 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.326 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.309 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.303 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.282 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.28 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.279 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.263 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.227 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.215 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.21 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.202 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.15 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.112 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.97 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.90 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.84 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.83 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.72 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.507 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.504 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.490 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.477 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.474 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.456 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.452 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.441 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.434 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.433 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.422 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.409 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.4 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.389 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.38 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.377 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.349 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.346 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.334 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.330 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.316 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.312 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.305 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.290 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.288 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.283 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.280 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.270 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.26 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.251 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.245 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.244 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.238 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.23 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.216 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.205 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.203 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.196 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.186 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.165 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.157 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.154 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.14 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.137 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.134 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.122 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.94 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.75 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.74 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.7 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.67 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.64 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.55 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.53 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.52 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.508 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.506 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.499 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.496 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.494 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.492 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.491 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.475 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.471 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.465 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.461 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.428 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.425 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.421 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.413 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.406 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.403 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.40 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.391 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.384 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.376 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.370 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.361 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.355 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.347 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.342 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.339 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.331 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.32 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.313 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.308 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.307 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.301 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.30 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.277 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.269 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.264 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.258 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.248 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.242 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.236 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.232 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.220 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.212 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.201 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.200 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.20 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.2 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.198 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.194 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.19 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.182 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.181 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.177 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.175 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.172 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.170 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.167 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.151 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.143 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.138 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.136 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.131 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.124 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.123 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.115 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.108 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.107 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.105 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 5:
	41 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	32 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	31 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	28 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	27 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	25 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	23 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	22 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	22 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	22 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.451 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	16 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.414 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.61 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.47 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.438 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.3 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.201 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.9 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.86 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.340 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.33 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.228 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.223 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.202 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.20 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.5 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.495 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.447 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.441 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.410 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.353 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.336 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.28 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.239 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.218 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.163 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.15 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.143 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.81 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.57 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.488 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.475 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.433 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.428 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.422 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.384 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.370 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.355 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.326 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.303 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.30 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.270 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.258 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.227 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.217 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.21 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.205 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.196 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.186 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.14 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.136 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.122 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.97 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.94 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.75 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.67 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.66 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.504 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.490 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.478 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.477 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.465 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.403 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.389 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.378 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.377 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.346 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.32 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.313 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.312 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.309 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.295 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.290 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.288 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.279 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.271 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.263 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.259 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.220 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.194 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.175 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.154 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.145 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.112 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.108 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.90 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.77 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.70 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.52 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.507 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.506 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.493 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.474 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.471 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.425 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.395 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.371 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.349 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.347 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.345 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.342 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.339 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.335 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.308 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.305 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.301 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.293 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.267 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.248 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.245 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.234 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.232 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.23 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.216 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.215 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.203 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.200 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.2 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.195 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.19 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.172 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.170 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.165 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.157 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.138 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.134 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.131 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.123 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.105 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.91 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.84 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.83 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.74 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.72 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.7 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.64 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.55 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.53 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.503 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.501 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.491 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.49 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.452 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.434 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.421 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.413 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.409 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.408 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.407 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.406 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.404 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.400 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.398 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.391 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.390 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.382 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.38 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.361 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.359 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.334 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.331 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.330 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.316 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.307 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.283 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.282 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.280 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.251 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.238 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.204 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.181 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.167 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.158 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.151 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.146 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.137 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.115 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.93 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.82 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.502 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.499 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.496 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.492 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.487 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.482 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.456 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.431 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.40 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.4 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.396 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.394 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.352 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.35 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.311 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.274 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.269 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.264 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.26 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.244 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.242 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.224 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.221 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.211 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.206 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.198 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.182 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.124 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.508 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.494 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.461 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.436 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.430 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.43 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.424 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.376 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.36 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.324 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.296 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.277 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.257 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.240 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.236 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.213 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.212 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.177 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.162 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.125 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.12 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.119 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.107 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Condition type in nodes:
	3267 : HigherCondition
Condition type in nodes with depth <= 0:
	137 : HigherCondition
Condition type in nodes with depth <= 1:
	405 : HigherCondition
Condition type in nodes with depth <= 2:
	903 : HigherCondition
Condition type in nodes with depth <= 3:
	1782 : HigherCondition
Condition type in nodes with depth <= 5:
	3267 : HigherCondition
```
</div>
    
<div class="k-default-codeblock">
```
None
```
</div>
    
<div class="k-default-codeblock">
```
model_2 summary: 
Model: "gradient_boosted_trees_model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
=================================================================
Total params: 1
Trainable params: 0
Non-trainable params: 1
_________________________________________________________________
Type: "GRADIENT_BOOSTED_TREES"
Task: CLASSIFICATION
Label: "__LABEL"
```
</div>
    
<div class="k-default-codeblock">
```
Input Features (1):
	data:0
```
</div>
    
<div class="k-default-codeblock">
```
No weights
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: MEAN_MIN_DEPTH:
    1. "__LABEL"  2.250000 ################
    2.  "data:0"  0.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_AS_ROOT:
    1. "data:0" 117.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_NODES:
    1. "data:0" 351.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: SUM_SCORE:
    1. "data:0" 32.035971 
```
</div>
    
    
    
<div class="k-default-codeblock">
```
Loss: BINOMIAL_LOG_LIKELIHOOD
Validation loss value: 1.36429
Number of trees per iteration: 1
Node format: NOT_SET
Number of trees: 117
Total number of nodes: 819
```
</div>
    
<div class="k-default-codeblock">
```
Number of nodes by tree:
Count: 117 Average: 7 StdDev: 0
Min: 7 Max: 7 Ignored: 0
----------------------------------------------
[ 7, 7] 117 100.00% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Depth by leafs:
Count: 468 Average: 2.25 StdDev: 0.829156
Min: 1 Max: 3 Ignored: 0
----------------------------------------------
[ 1, 2) 117  25.00%  25.00% #####
[ 2, 3) 117  25.00%  50.00% #####
[ 3, 3] 234  50.00% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Number of training obs by leaf:
Count: 468 Average: 1545.5 StdDev: 2660.15
Min: 5 Max: 6153 Ignored: 0
----------------------------------------------
[    5,  312) 351  75.00%  75.00% ##########
[  312,  619)   0   0.00%  75.00%
[  619,  927)   0   0.00%  75.00%
[  927, 1234)   0   0.00%  75.00%
[ 1234, 1542)   0   0.00%  75.00%
[ 1542, 1849)   0   0.00%  75.00%
[ 1849, 2157)   0   0.00%  75.00%
[ 2157, 2464)   0   0.00%  75.00%
[ 2464, 2772)   0   0.00%  75.00%
[ 2772, 3079)   0   0.00%  75.00%
[ 3079, 3386)   0   0.00%  75.00%
[ 3386, 3694)   0   0.00%  75.00%
[ 3694, 4001)   0   0.00%  75.00%
[ 4001, 4309)   0   0.00%  75.00%
[ 4309, 4616)   0   0.00%  75.00%
[ 4616, 4924)   0   0.00%  75.00%
[ 4924, 5231)   0   0.00%  75.00%
[ 5231, 5539)   0   0.00%  75.00%
[ 5539, 5846)   0   0.00%  75.00%
[ 5846, 6153] 117  25.00% 100.00% ###
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes:
	351 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 0:
	117 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 1:
	234 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 2:
	351 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 3:
	351 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 5:
	351 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Condition type in nodes:
	351 : ContainsBitmapCondition
Condition type in nodes with depth <= 0:
	117 : ContainsBitmapCondition
Condition type in nodes with depth <= 1:
	234 : ContainsBitmapCondition
Condition type in nodes with depth <= 2:
	351 : ContainsBitmapCondition
Condition type in nodes with depth <= 3:
	351 : ContainsBitmapCondition
Condition type in nodes with depth <= 5:
	351 : ContainsBitmapCondition
```
</div>
    
<div class="k-default-codeblock">
```
None

```
</div>
---
## Plotting training metrics


```python

def plot_curve(logs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Loss")

    plt.show()


plot_curve(logs_1)
plot_curve(logs_2)
```


    
![png](/img/examples/nlp/tweet-classification-using-tfdf/tweet-classification-using-tfdf_41_0.png)
    



    
![png](/img/examples/nlp/tweet-classification-using-tfdf/tweet-classification-using-tfdf_41_1.png)
    


---
## Evaluating on test data


```python
results = model_1.evaluate(test_ds, return_dict=True, verbose=0)
print("model_1 Evaluation: \n")
for name, value in results.items():
    print(f"{name}: {value:.4f}")

results = model_2.evaluate(test_ds, return_dict=True, verbose=0)
print("model_2 Evaluation: \n")
for name, value in results.items():
    print(f"{name}: {value:.4f}")
```

<div class="k-default-codeblock">
```
model_1 Evaluation: 
```
</div>
    
<div class="k-default-codeblock">
```
loss: 0.0000
Accuracy: 0.8160
recall: 0.7241
precision: 0.8514
auc: 0.8700
model_2 Evaluation: 
```
</div>
    
<div class="k-default-codeblock">
```
loss: 0.0000
Accuracy: 0.5440
recall: 0.0029
precision: 1.0000
auc: 0.5026

```
</div>
---
## Predicting on validation data


```python
test_df.reset_index(inplace=True, drop=True)
for index, row in test_df.iterrows():
    text = tf.expand_dims(row["text"], axis=0)
    preds = model_1.predict_step(text)
    preds = tf.squeeze(tf.round(preds))
    print(f"Text: {row['text']}")
    print(f"Prediction: {int(preds)}")
    print(f"Ground Truth : {row['target']}")
    if index == 10:
        break
```

<div class="k-default-codeblock">
```
Text: DFR EP016 Monthly Meltdown - On Dnbheaven 2015.08.06 http://t.co/EjKRf8N8A8 #Drum and Bass #heavy #nasty http://t.co/SPHWE6wFI5
Prediction: 0
Ground Truth : 0
Text: FedEx no longer to transport bioterror germs in wake of anthrax lab mishaps http://t.co/qZQc8WWwcN via @usatoday
Prediction: 1
Ground Truth : 0
Text: Gunmen kill four in El Salvador bus attack: Suspected Salvadoran gang members killed four people and wounded s... http://t.co/CNtwB6ScZj
Prediction: 1
Ground Truth : 1
Text: @camilacabello97 Internally and externally screaming
Prediction: 0
Ground Truth : 1
Text: Radiation emergency #preparedness starts with knowing to: get inside stay inside and stay tuned http://t.co/RFFPqBAz2F via @CDCgov
Prediction: 1
Ground Truth : 1
Text: Investigators rule catastrophic structural failure resulted in 2014 Virg.. Related Articles: http://t.co/Cy1LFeNyV8
Prediction: 1
Ground Truth : 1
Text: How the West was burned: Thousands of wildfires ablaze in #California alone http://t.co/iCSjGZ9tE1 #climate #energy http://t.co/9FxmN0l0Bd
Prediction: 1
Ground Truth : 1
Text: Map: Typhoon Soudelor's predicted path as it approaches Taiwan; expected to make landfall over southern China by SÛ_ http://t.co/JDVSGVhlIs
Prediction: 1
Ground Truth : 1
Text: Ûª93 blasts accused Yeda Yakub dies in Karachi of heart attack http://t.co/mfKqyxd8XG #Mumbai
Prediction: 1
Ground Truth : 1
Text: My ears are bleeding  https://t.co/k5KnNwugwT
Prediction: 0
Ground Truth : 0
Text: @RedCoatJackpot *As it was typical for them their bullets collided and none managed to reach their targets; such was the ''curse'' of a --
Prediction: 0
Ground Truth : 0

```
</div>
---
## Concluding remarks

The TensorFlow Decision Forests package provides powerful models
that work especially well with structured data. In our experiments,
the Gradient Boosted Tree model with pretrained embeddings achieved 81.6%
test accuracy while the plain Gradient Boosted Tree model had 54.4% accuracy.
