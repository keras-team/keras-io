# Text classification using Decision Forests and pretrained embeddings

**Author:** Gitesh Chawda<br>
**Date created:** 09/05/2022<br>
**Last modified:** 09/05/2022<br>
**Description:** Using Tensorflow Decision Forests for text classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/Tweet-classification-using-TFDF.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/Tweet-classification-using-TFDF.py)



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

<div class="k-default-codeblock">
```
WARNING:root:TF Parameter Server distributed training not available (this is expected for the pre-build release).

```
</div>
---
## Get the data

The Dataset is avalaible on [Kaggle](https://www.kaggle.com/c/nlp-getting-started)

Dataset description:

1. Files

- train.csv: the training set

2. Columns

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
        (df["text"].to_numpy(), df["target"].to_numpy())
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
Use /tmp/tmpkpl10aj9 as temporary training directory

```
</div>
Building model_2


```python
model_2 = tfdf.keras.GradientBoostedTreesModel()
```

<div class="k-default-codeblock">
```
Use /tmp/tmpysfsq6o0 as temporary training directory

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
Starting reading the dataset
77/77 [==============================] - ETA: 0s
Dataset read in 0:00:15.844516
Training model
Model trained in 0:02:30.922245
Compiling model
77/77 [==============================] - 167s 2s/step
WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7f45bd2ada70> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

WARNING:tensorflow:AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7f45bd2ada70> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

WARNING: AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7f45bd2ada70> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
Starting reading the dataset
55/77 [====================>.........] - ETA: 0s
Dataset read in 0:00:00.219258
Training model
Model trained in 0:00:00.289591
Compiling model
77/77 [==============================] - 1s 6ms/step

<keras.callbacks.History at 0x7f453f9349d0>

```
</div>
Prints training logs of model_1


```python
logs_1 = model_1.make_inspector().training_logs()
print(logs_1)
```

<div class="k-default-codeblock">
```
[TrainLog(num_trees=1, evaluation=Evaluation(num_examples=None, accuracy=0.5467914342880249, loss=1.3187708854675293, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=2, evaluation=Evaluation(num_examples=None, accuracy=0.6898396015167236, loss=1.2692136764526367, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=3, evaluation=Evaluation(num_examples=None, accuracy=0.7286096215248108, loss=1.228997826576233, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=4, evaluation=Evaluation(num_examples=None, accuracy=0.7566844820976257, loss=1.1951442956924438, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=5, evaluation=Evaluation(num_examples=None, accuracy=0.7526738047599792, loss=1.164238691329956, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=6, evaluation=Evaluation(num_examples=None, accuracy=0.7540106773376465, loss=1.1361148357391357, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=7, evaluation=Evaluation(num_examples=None, accuracy=0.7633689641952515, loss=1.108812689781189, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=8, evaluation=Evaluation(num_examples=None, accuracy=0.7647058963775635, loss=1.0895808935165405, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=9, evaluation=Evaluation(num_examples=None, accuracy=0.7633689641952515, loss=1.070176601409912, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=10, evaluation=Evaluation(num_examples=None, accuracy=0.7740641832351685, loss=1.0516939163208008, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=11, evaluation=Evaluation(num_examples=None, accuracy=0.7700534462928772, loss=1.0360163450241089, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=12, evaluation=Evaluation(num_examples=None, accuracy=0.7673797011375427, loss=1.0204159021377563, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=13, evaluation=Evaluation(num_examples=None, accuracy=0.7780748605728149, loss=1.0069549083709717, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=14, evaluation=Evaluation(num_examples=None, accuracy=0.779411792755127, loss=0.9998055696487427, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=15, evaluation=Evaluation(num_examples=None, accuracy=0.779411792755127, loss=0.9909116625785828, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=16, evaluation=Evaluation(num_examples=None, accuracy=0.7780748605728149, loss=0.984338104724884, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=17, evaluation=Evaluation(num_examples=None, accuracy=0.7807486653327942, loss=0.9740555286407471, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=18, evaluation=Evaluation(num_examples=None, accuracy=0.7847593426704407, loss=0.9652271270751953, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=19, evaluation=Evaluation(num_examples=None, accuracy=0.7874331474304199, loss=0.9603098630905151, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=20, evaluation=Evaluation(num_examples=None, accuracy=0.779411792755127, loss=0.9573495388031006, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=21, evaluation=Evaluation(num_examples=None, accuracy=0.7887700796127319, loss=0.9562695026397705, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=22, evaluation=Evaluation(num_examples=None, accuracy=0.7914438247680664, loss=0.9525935649871826, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=23, evaluation=Evaluation(num_examples=None, accuracy=0.7914438247680664, loss=0.9479075074195862, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=24, evaluation=Evaluation(num_examples=None, accuracy=0.7901069521903992, loss=0.9444673657417297, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=25, evaluation=Evaluation(num_examples=None, accuracy=0.7927807569503784, loss=0.9409474730491638, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=26, evaluation=Evaluation(num_examples=None, accuracy=0.7901069521903992, loss=0.9377444982528687, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=27, evaluation=Evaluation(num_examples=None, accuracy=0.7914438247680664, loss=0.9360072612762451, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=28, evaluation=Evaluation(num_examples=None, accuracy=0.7927807569503784, loss=0.9330317378044128, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=29, evaluation=Evaluation(num_examples=None, accuracy=0.7914438247680664, loss=0.9277840256690979, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=30, evaluation=Evaluation(num_examples=None, accuracy=0.7927807569503784, loss=0.9260849952697754, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=31, evaluation=Evaluation(num_examples=None, accuracy=0.7967914342880249, loss=0.9228695034980774, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=32, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.9213683009147644, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=33, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.9199848771095276, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=34, evaluation=Evaluation(num_examples=None, accuracy=0.7914438247680664, loss=0.9179971814155579, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=35, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.915500283241272, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=36, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.9146019816398621, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=37, evaluation=Evaluation(num_examples=None, accuracy=0.7967914342880249, loss=0.911679744720459, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=38, evaluation=Evaluation(num_examples=None, accuracy=0.7981283664703369, loss=0.9122370481491089, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=39, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.9120528101921082, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=40, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.9095850586891174, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=41, evaluation=Evaluation(num_examples=None, accuracy=0.7967914342880249, loss=0.9075614809989929, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=42, evaluation=Evaluation(num_examples=None, accuracy=0.7981283664703369, loss=0.9063277244567871, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=43, evaluation=Evaluation(num_examples=None, accuracy=0.7967914342880249, loss=0.9072968363761902, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=44, evaluation=Evaluation(num_examples=None, accuracy=0.7927807569503784, loss=0.907780110836029, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=45, evaluation=Evaluation(num_examples=None, accuracy=0.7901069521903992, loss=0.9056413769721985, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=46, evaluation=Evaluation(num_examples=None, accuracy=0.7941176295280457, loss=0.9053149223327637, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=47, evaluation=Evaluation(num_examples=None, accuracy=0.7914438247680664, loss=0.9039382934570312, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=48, evaluation=Evaluation(num_examples=None, accuracy=0.7927807569503784, loss=0.9047112464904785, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=49, evaluation=Evaluation(num_examples=None, accuracy=0.7941176295280457, loss=0.9048668146133423, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=50, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.905793309211731, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=51, evaluation=Evaluation(num_examples=None, accuracy=0.7927807569503784, loss=0.903205394744873, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=52, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.9017136693000793, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=53, evaluation=Evaluation(num_examples=None, accuracy=0.7941176295280457, loss=0.903002142906189, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=54, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.902319610118866, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=55, evaluation=Evaluation(num_examples=None, accuracy=0.7941176295280457, loss=0.9025460481643677, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=56, evaluation=Evaluation(num_examples=None, accuracy=0.7914438247680664, loss=0.9016172885894775, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=57, evaluation=Evaluation(num_examples=None, accuracy=0.7954545617103577, loss=0.9021195769309998, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=58, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.9016126990318298, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=59, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.9004343748092651, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=60, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.9001455307006836, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=61, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8995491862297058, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=62, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.8982800841331482, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=63, evaluation=Evaluation(num_examples=None, accuracy=0.8008021116256714, loss=0.8976819515228271, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=64, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8969248533248901, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=65, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8960849642753601, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=66, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.8977264761924744, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=67, evaluation=Evaluation(num_examples=None, accuracy=0.8008021116256714, loss=0.8977041244506836, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=68, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8984429240226746, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=69, evaluation=Evaluation(num_examples=None, accuracy=0.8008021116256714, loss=0.8994362354278564, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=70, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8986279964447021, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=71, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.8968974351882935, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=72, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.8962607979774475, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=73, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.8944525122642517, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=74, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8941737413406372, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=75, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8943476676940918, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=76, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.8942290544509888, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=77, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.8952121138572693, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=78, evaluation=Evaluation(num_examples=None, accuracy=0.7994652390480042, loss=0.8949428796768188, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=79, evaluation=Evaluation(num_examples=None, accuracy=0.8008021116256714, loss=0.8941347599029541, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=80, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8917156457901001, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=81, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8903207182884216, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=82, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8897010087966919, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=83, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8891079425811768, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=84, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8890292048454285, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=85, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8888816833496094, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=86, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8897340297698975, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=87, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8904791474342346, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=88, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8898884654045105, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=89, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.890166699886322, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=90, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.888908326625824, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=91, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.8875435590744019, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=92, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8855466842651367, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=93, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8862167000770569, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=94, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.886319100856781, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=95, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8846173286437988, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=96, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8846728801727295, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=97, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8838931918144226, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=98, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8847978711128235, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=99, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8851138353347778, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=100, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8843891620635986, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=101, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.8843651413917542, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=102, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8828133940696716, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=103, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8822276592254639, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=104, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8826838731765747, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=105, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8817507028579712, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=106, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.883398175239563, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=107, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8830643892288208, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=108, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.882763683795929, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=109, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8816952109336853, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=110, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8812451362609863, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=111, evaluation=Evaluation(num_examples=None, accuracy=0.8114973306655884, loss=0.8809582591056824, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=112, evaluation=Evaluation(num_examples=None, accuracy=0.8101603984832764, loss=0.8810463547706604, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=113, evaluation=Evaluation(num_examples=None, accuracy=0.8101603984832764, loss=0.8804474472999573, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=114, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8801485896110535, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=115, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8801633715629578, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=116, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8807496428489685, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=117, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8812755346298218, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=118, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.881771445274353, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=119, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8818649649620056, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=120, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8817685842514038, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=121, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8813406229019165, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=122, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8804141283035278, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=123, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8793104290962219, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=124, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8790532946586609, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=125, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8786349296569824, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=126, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8788155913352966, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=127, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8795849084854126, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=128, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8792861104011536, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=129, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8786776661872864, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=130, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.8783702254295349, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=131, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8798111081123352, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=132, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8798418641090393, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=133, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8807860612869263, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=134, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8804180026054382, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=135, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8817436099052429, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=136, evaluation=Evaluation(num_examples=None, accuracy=0.8101603984832764, loss=0.8804024457931519, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=137, evaluation=Evaluation(num_examples=None, accuracy=0.8101603984832764, loss=0.8815509676933289, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=138, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8822447061538696, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=139, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8818819522857666, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=140, evaluation=Evaluation(num_examples=None, accuracy=0.8101603984832764, loss=0.8830416202545166, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=141, evaluation=Evaluation(num_examples=None, accuracy=0.8101603984832764, loss=0.8831774592399597, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=142, evaluation=Evaluation(num_examples=None, accuracy=0.8101603984832764, loss=0.8839144110679626, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=143, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8835448026657104, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=144, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8833995461463928, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=145, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8841006755828857, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=146, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8846890926361084, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=147, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8853747248649597, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=148, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8856629729270935, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=149, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.8865883946418762, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=150, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.8867426514625549, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=151, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.8860962390899658, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=152, evaluation=Evaluation(num_examples=None, accuracy=0.8034759163856506, loss=0.8880785703659058, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=153, evaluation=Evaluation(num_examples=None, accuracy=0.8021390438079834, loss=0.8879771828651428, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=154, evaluation=Evaluation(num_examples=None, accuracy=0.8074866533279419, loss=0.8906283974647522, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=155, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8912577629089355, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=156, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8894665837287903, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=157, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8886289596557617, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=158, evaluation=Evaluation(num_examples=None, accuracy=0.8061497211456299, loss=0.8878296613693237, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=159, evaluation=Evaluation(num_examples=None, accuracy=0.8088235259056091, loss=0.8881493210792542, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=160, evaluation=Evaluation(num_examples=None, accuracy=0.8048128485679626, loss=0.887750506401062, rmse=None, ndcg=None, aucs=None))]

```
</div>
Prints training logs of model_2


```python
logs_2 = model_2.make_inspector().training_logs()
print(logs_2)
```

<div class="k-default-codeblock">
```
[TrainLog(num_trees=1, evaluation=Evaluation(num_examples=None, accuracy=0.5467914342880249, loss=1.379634976387024, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=2, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3791122436523438, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=3, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3786882162094116, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=4, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3783376216888428, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=5, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3780431747436523, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=6, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3777928352355957, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=7, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3775783777236938, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=8, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3773930072784424, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=9, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3772317171096802, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=10, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3770910501480103, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=11, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3769675493240356, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=12, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3768584728240967, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=13, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3767625093460083, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=14, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3766772747039795, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=15, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3766014575958252, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=16, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376534342765808, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=17, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376474142074585, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=18, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3764206171035767, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=19, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3763726949691772, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=20, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3763296604156494, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=21, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376291275024414, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=22, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3762567043304443, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=23, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3762257099151611, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=24, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3761978149414062, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=25, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376172661781311, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=26, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3761500120162964, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=27, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3761297464370728, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=28, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3761115074157715, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=29, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376094937324524, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=30, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3760799169540405, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=31, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3760665655136108, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=32, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3760545253753662, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=33, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376043438911438, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=34, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3760336637496948, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=35, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376024603843689, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=36, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376016616821289, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=37, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.376009225845337, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=38, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3760027885437012, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=39, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375996708869934, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=40, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759914636611938, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=41, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759865760803223, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=42, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759822845458984, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=43, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759781122207642, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=44, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759746551513672, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=45, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759711980819702, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=46, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759685754776, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=47, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759658336639404, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=48, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759633302688599, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=49, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759609460830688, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=50, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759591579437256, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=51, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759574890136719, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=52, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759559392929077, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=53, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375954270362854, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=54, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759530782699585, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=55, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759517669677734, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=56, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375950813293457, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=57, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375949740409851, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=58, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759489059448242, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=59, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375948190689087, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=60, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.37594735622406, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=61, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759466409683228, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=62, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375946044921875, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=63, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759455680847168, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=64, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759450912475586, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=65, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759446144104004, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=66, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759442567825317, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=67, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759437799453735, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=68, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375943660736084, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=69, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759431838989258, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=70, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759430646896362, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=71, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759427070617676, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=72, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759424686431885, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=73, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375942349433899, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=74, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759422302246094, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=75, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759419918060303, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=76, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759417533874512, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=77, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759417533874512, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=78, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759416341781616, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=79, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375941514968872, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=80, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375941276550293, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=81, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.375941276550293, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=82, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759410381317139, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=83, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759411573410034, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=84, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759411573410034, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=85, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759410381317139, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=86, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759409189224243, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=87, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759409189224243, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=88, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759409189224243, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=89, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759409189224243, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=90, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759407997131348, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=91, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759407997131348, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=92, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759407997131348, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=93, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759406805038452, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=94, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759406805038452, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=95, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759406805038452, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=96, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759406805038452, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=97, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759406805038452, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=98, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759406805038452, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=99, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759406805038452, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=100, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759407997131348, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=101, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=102, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=103, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=104, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=105, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=106, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=107, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=108, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=109, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=110, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=111, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759405612945557, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=112, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=113, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=114, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=115, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=116, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=117, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=118, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=119, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=120, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=121, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=122, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=123, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=124, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=125, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=126, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=127, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=128, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=129, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=130, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=131, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=132, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=133, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=134, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=135, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=136, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=137, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=138, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=139, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=140, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=141, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None)), TrainLog(num_trees=142, evaluation=Evaluation(num_examples=None, accuracy=0.5494652390480042, loss=1.3759404420852661, rmse=None, ndcg=None, aucs=None))]

```
</div>
The `model.summary()` method prints a variety of information about your decision tree
model, including model type, task, input features, and feature importance.


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
    1. "model/keras_layer/StatefulPartitionedCall:0.119"  4.791797 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.170"  4.791797 ################
    3.  "model/keras_layer/StatefulPartitionedCall:0.21"  4.791797 ################
    4. "model/keras_layer/StatefulPartitionedCall:0.212"  4.791797 ################
    5. "model/keras_layer/StatefulPartitionedCall:0.213"  4.791797 ################
    6. "model/keras_layer/StatefulPartitionedCall:0.277"  4.791797 ################
    7. "model/keras_layer/StatefulPartitionedCall:0.398"  4.791797 ################
    8.   "model/keras_layer/StatefulPartitionedCall:0.4"  4.791797 ################
    9. "model/keras_layer/StatefulPartitionedCall:0.445"  4.791797 ################
   10. "model/keras_layer/StatefulPartitionedCall:0.508"  4.791797 ################
   11.  "model/keras_layer/StatefulPartitionedCall:0.55"  4.791797 ################
   12.  "model/keras_layer/StatefulPartitionedCall:0.83"  4.791797 ################
   13.                                         "__LABEL"  4.791797 ################
   14. "model/keras_layer/StatefulPartitionedCall:0.121"  4.791316 ###############
   15. "model/keras_layer/StatefulPartitionedCall:0.240"  4.791316 ###############
   16. "model/keras_layer/StatefulPartitionedCall:0.312"  4.791316 ###############
   17. "model/keras_layer/StatefulPartitionedCall:0.499"  4.791316 ###############
   18. "model/keras_layer/StatefulPartitionedCall:0.195"  4.791284 ###############
   19. "model/keras_layer/StatefulPartitionedCall:0.474"  4.791266 ###############
   20. "model/keras_layer/StatefulPartitionedCall:0.182"  4.791247 ###############
   21. "model/keras_layer/StatefulPartitionedCall:0.112"  4.791205 ###############
   22.  "model/keras_layer/StatefulPartitionedCall:0.23"  4.791181 ###############
   23. "model/keras_layer/StatefulPartitionedCall:0.236"  4.791181 ###############
   24. "model/keras_layer/StatefulPartitionedCall:0.181"  4.791156 ###############
   25. "model/keras_layer/StatefulPartitionedCall:0.163"  4.791128 ###############
   26. "model/keras_layer/StatefulPartitionedCall:0.157"  4.791097 ###############
   27. "model/keras_layer/StatefulPartitionedCall:0.438"  4.791064 ###############
   28.  "model/keras_layer/StatefulPartitionedCall:0.93"  4.790835 ###############
   29. "model/keras_layer/StatefulPartitionedCall:0.491"  4.790803 ###############
   30. "model/keras_layer/StatefulPartitionedCall:0.264"  4.790788 ###############
   31. "model/keras_layer/StatefulPartitionedCall:0.177"  4.790785 ###############
   32. "model/keras_layer/StatefulPartitionedCall:0.122"  4.790771 ###############
   33. "model/keras_layer/StatefulPartitionedCall:0.279"  4.790766 ###############
   34. "model/keras_layer/StatefulPartitionedCall:0.134"  4.790685 ###############
   35. "model/keras_layer/StatefulPartitionedCall:0.216"  4.790677 ###############
   36.  "model/keras_layer/StatefulPartitionedCall:0.52"  4.790590 ###############
   37. "model/keras_layer/StatefulPartitionedCall:0.430"  4.790578 ###############
   38. "model/keras_layer/StatefulPartitionedCall:0.452"  4.790528 ###############
   39.  "model/keras_layer/StatefulPartitionedCall:0.19"  4.790528 ###############
   40. "model/keras_layer/StatefulPartitionedCall:0.244"  4.790528 ###############
   41.  "model/keras_layer/StatefulPartitionedCall:0.61"  4.790515 ###############
   42.  "model/keras_layer/StatefulPartitionedCall:0.30"  4.790512 ###############
   43. "model/keras_layer/StatefulPartitionedCall:0.143"  4.790393 ###############
   44. "model/keras_layer/StatefulPartitionedCall:0.465"  4.790342 ###############
   45.  "model/keras_layer/StatefulPartitionedCall:0.13"  4.790317 ###############
   46. "model/keras_layer/StatefulPartitionedCall:0.248"  4.790291 ###############
   47. "model/keras_layer/StatefulPartitionedCall:0.313"  4.790274 ###############
   48. "model/keras_layer/StatefulPartitionedCall:0.309"  4.790265 ###############
   49. "model/keras_layer/StatefulPartitionedCall:0.490"  4.790258 ###############
   50. "model/keras_layer/StatefulPartitionedCall:0.396"  4.790223 ###############
   51. "model/keras_layer/StatefulPartitionedCall:0.263"  4.790200 ###############
   52. "model/keras_layer/StatefulPartitionedCall:0.296"  4.790194 ###############
   53. "model/keras_layer/StatefulPartitionedCall:0.137"  4.790159 ###############
   54. "model/keras_layer/StatefulPartitionedCall:0.395"  4.790156 ###############
   55. "model/keras_layer/StatefulPartitionedCall:0.342"  4.790104 ###############
   56. "model/keras_layer/StatefulPartitionedCall:0.493"  4.790093 ###############
   57.   "model/keras_layer/StatefulPartitionedCall:0.7"  4.790090 ###############
   58.  "model/keras_layer/StatefulPartitionedCall:0.81"  4.790088 ###############
   59. "model/keras_layer/StatefulPartitionedCall:0.205"  4.790062 ###############
   60. "model/keras_layer/StatefulPartitionedCall:0.403"  4.790062 ###############
   61.   "model/keras_layer/StatefulPartitionedCall:0.5"  4.790034 ###############
   62. "model/keras_layer/StatefulPartitionedCall:0.162"  4.790016 ###############
   63.  "model/keras_layer/StatefulPartitionedCall:0.72"  4.789965 ###############
   64. "model/keras_layer/StatefulPartitionedCall:0.431"  4.789965 ###############
   65. "model/keras_layer/StatefulPartitionedCall:0.224"  4.789963 ###############
   66.  "model/keras_layer/StatefulPartitionedCall:0.49"  4.789899 ###############
   67. "model/keras_layer/StatefulPartitionedCall:0.318"  4.789865 ###############
   68. "model/keras_layer/StatefulPartitionedCall:0.410"  4.789856 ###############
   69. "model/keras_layer/StatefulPartitionedCall:0.488"  4.789773 ###############
   70. "model/keras_layer/StatefulPartitionedCall:0.139"  4.789745 ###############
   71. "model/keras_layer/StatefulPartitionedCall:0.408"  4.789745 ###############
   72. "model/keras_layer/StatefulPartitionedCall:0.349"  4.789709 ###############
   73. "model/keras_layer/StatefulPartitionedCall:0.409"  4.789680 ###############
   74. "model/keras_layer/StatefulPartitionedCall:0.228"  4.789675 ###############
   75. "model/keras_layer/StatefulPartitionedCall:0.125"  4.789586 ###############
   76. "model/keras_layer/StatefulPartitionedCall:0.326"  4.789566 ###############
   77. "model/keras_layer/StatefulPartitionedCall:0.422"  4.789533 ###############
   78.  "model/keras_layer/StatefulPartitionedCall:0.53"  4.789526 ###############
   79. "model/keras_layer/StatefulPartitionedCall:0.156"  4.789518 ###############
   80. "model/keras_layer/StatefulPartitionedCall:0.466"  4.789518 ###############
   81.  "model/keras_layer/StatefulPartitionedCall:0.91"  4.789393 ###############
   82. "model/keras_layer/StatefulPartitionedCall:0.217"  4.789335 ###############
   83. "model/keras_layer/StatefulPartitionedCall:0.316"  4.789335 ###############
   84. "model/keras_layer/StatefulPartitionedCall:0.404"  4.789331 ###############
   85.  "model/keras_layer/StatefulPartitionedCall:0.40"  4.789299 ###############
   86. "model/keras_layer/StatefulPartitionedCall:0.239"  4.789247 ###############
   87.   "model/keras_layer/StatefulPartitionedCall:0.6"  4.789233 ###############
   88. "model/keras_layer/StatefulPartitionedCall:0.441"  4.789178 ###############
   89. "model/keras_layer/StatefulPartitionedCall:0.215"  4.789173 ###############
   90. "model/keras_layer/StatefulPartitionedCall:0.424"  4.789121 ###############
   91. "model/keras_layer/StatefulPartitionedCall:0.107"  4.789103 ###############
   92.  "model/keras_layer/StatefulPartitionedCall:0.26"  4.789086 ###############
   93. "model/keras_layer/StatefulPartitionedCall:0.478"  4.789046 ###############
   94. "model/keras_layer/StatefulPartitionedCall:0.377"  4.789003 ###############
   95. "model/keras_layer/StatefulPartitionedCall:0.191"  4.788973 ###############
   96. "model/keras_layer/StatefulPartitionedCall:0.136"  4.788949 ###############
   97. "model/keras_layer/StatefulPartitionedCall:0.447"  4.788905 ###############
   98.  "model/keras_layer/StatefulPartitionedCall:0.43"  4.788882 ###############
   99.  "model/keras_layer/StatefulPartitionedCall:0.38"  4.788880 ###############
  100. "model/keras_layer/StatefulPartitionedCall:0.385"  4.788811 ###############
  101. "model/keras_layer/StatefulPartitionedCall:0.305"  4.788782 ###############
  102. "model/keras_layer/StatefulPartitionedCall:0.200"  4.788749 ###############
  103. "model/keras_layer/StatefulPartitionedCall:0.353"  4.788624 ###############
  104.   "model/keras_layer/StatefulPartitionedCall:0.2"  4.788608 ###############
  105. "model/keras_layer/StatefulPartitionedCall:0.167"  4.788581 ###############
  106. "model/keras_layer/StatefulPartitionedCall:0.295"  4.788532 ###############
  107. "model/keras_layer/StatefulPartitionedCall:0.271"  4.788511 ###############
  108. "model/keras_layer/StatefulPartitionedCall:0.242"  4.788498 ###############
  109. "model/keras_layer/StatefulPartitionedCall:0.350"  4.788474 ###############
  110. "model/keras_layer/StatefulPartitionedCall:0.355"  4.788450 ###############
  111. "model/keras_layer/StatefulPartitionedCall:0.406"  4.788449 ###############
  112. "model/keras_layer/StatefulPartitionedCall:0.238"  4.788431 ###############
  113.  "model/keras_layer/StatefulPartitionedCall:0.36"  4.788427 ###############
  114. "model/keras_layer/StatefulPartitionedCall:0.151"  4.788413 ###############
  115. "model/keras_layer/StatefulPartitionedCall:0.382"  4.788400 ###############
  116. "model/keras_layer/StatefulPartitionedCall:0.370"  4.788389 ###############
  117. "model/keras_layer/StatefulPartitionedCall:0.123"  4.788311 ###############
  118. "model/keras_layer/StatefulPartitionedCall:0.179"  4.788256 ###############
  119.   "model/keras_layer/StatefulPartitionedCall:0.3"  4.788195 ###############
  120. "model/keras_layer/StatefulPartitionedCall:0.346"  4.788105 ###############
  121. "model/keras_layer/StatefulPartitionedCall:0.270"  4.788093 ###############
  122. "model/keras_layer/StatefulPartitionedCall:0.172"  4.788084 ###############
  123. "model/keras_layer/StatefulPartitionedCall:0.311"  4.788042 ###############
  124. "model/keras_layer/StatefulPartitionedCall:0.428"  4.788040 ###############
  125. "model/keras_layer/StatefulPartitionedCall:0.487"  4.788038 ###############
  126.  "model/keras_layer/StatefulPartitionedCall:0.57"  4.787961 ###############
  127. "model/keras_layer/StatefulPartitionedCall:0.391"  4.787922 ###############
  128. "model/keras_layer/StatefulPartitionedCall:0.394"  4.787908 ###############
  129. "model/keras_layer/StatefulPartitionedCall:0.109"  4.787908 ###############
  130. "model/keras_layer/StatefulPartitionedCall:0.283"  4.787852 ###############
  131. "model/keras_layer/StatefulPartitionedCall:0.258"  4.787843 ###############
  132. "model/keras_layer/StatefulPartitionedCall:0.301"  4.787825 ###############
  133. "model/keras_layer/StatefulPartitionedCall:0.345"  4.787800 ###############
  134. "model/keras_layer/StatefulPartitionedCall:0.470"  4.787769 ###############
  135.  "model/keras_layer/StatefulPartitionedCall:0.97"  4.787743 ###############
  136. "model/keras_layer/StatefulPartitionedCall:0.495"  4.787739 ###############
  137. "model/keras_layer/StatefulPartitionedCall:0.389"  4.787730 ###############
  138. "model/keras_layer/StatefulPartitionedCall:0.435"  4.787724 ###############
  139. "model/keras_layer/StatefulPartitionedCall:0.340"  4.787724 ###############
  140.  "model/keras_layer/StatefulPartitionedCall:0.64"  4.787719 ###############
  141. "model/keras_layer/StatefulPartitionedCall:0.425"  4.787639 ###############
  142.  "model/keras_layer/StatefulPartitionedCall:0.15"  4.787568 ###############
  143. "model/keras_layer/StatefulPartitionedCall:0.269"  4.787560 ###############
  144.  "model/keras_layer/StatefulPartitionedCall:0.70"  4.787557 ###############
  145.  "model/keras_layer/StatefulPartitionedCall:0.20"  4.787553 ###############
  146.  "model/keras_layer/StatefulPartitionedCall:0.77"  4.787548 ###############
  147.  "model/keras_layer/StatefulPartitionedCall:0.86"  4.787533 ###############
  148. "model/keras_layer/StatefulPartitionedCall:0.108"  4.787523 ###############
  149. "model/keras_layer/StatefulPartitionedCall:0.366"  4.787401 ###############
  150. "model/keras_layer/StatefulPartitionedCall:0.475"  4.787401 ###############
  151.  "model/keras_layer/StatefulPartitionedCall:0.94"  4.787342 ###############
  152. "model/keras_layer/StatefulPartitionedCall:0.503"  4.787322 ###############
  153. "model/keras_layer/StatefulPartitionedCall:0.220"  4.787320 ###############
  154. "model/keras_layer/StatefulPartitionedCall:0.267"  4.787287 ###############
  155. "model/keras_layer/StatefulPartitionedCall:0.496"  4.787282 ###############
  156. "model/keras_layer/StatefulPartitionedCall:0.436"  4.787272 ###############
  157. "model/keras_layer/StatefulPartitionedCall:0.494"  4.787258 ###############
  158. "model/keras_layer/StatefulPartitionedCall:0.336"  4.787226 ###############
  159. "model/keras_layer/StatefulPartitionedCall:0.223"  4.787198 ###############
  160. "model/keras_layer/StatefulPartitionedCall:0.237"  4.787195 ###############
  161. "model/keras_layer/StatefulPartitionedCall:0.459"  4.787192 ###############
  162.  "model/keras_layer/StatefulPartitionedCall:0.33"  4.787138 ###############
  163. "model/keras_layer/StatefulPartitionedCall:0.347"  4.786996 ###############
  164. "model/keras_layer/StatefulPartitionedCall:0.210"  4.786974 ###############
  165. "model/keras_layer/StatefulPartitionedCall:0.334"  4.786959 ###############
  166. "model/keras_layer/StatefulPartitionedCall:0.198"  4.786896 ###############
  167. "model/keras_layer/StatefulPartitionedCall:0.331"  4.786896 ###############
  168. "model/keras_layer/StatefulPartitionedCall:0.481"  4.786872 ###############
  169. "model/keras_layer/StatefulPartitionedCall:0.477"  4.786838 ###############
  170. "model/keras_layer/StatefulPartitionedCall:0.363"  4.786809 ###############
  171.  "model/keras_layer/StatefulPartitionedCall:0.47"  4.786682 ###############
  172. "model/keras_layer/StatefulPartitionedCall:0.324"  4.786665 ###############
  173. "model/keras_layer/StatefulPartitionedCall:0.421"  4.786655 ###############
  174. "model/keras_layer/StatefulPartitionedCall:0.292"  4.786654 ###############
  175. "model/keras_layer/StatefulPartitionedCall:0.371"  4.786641 ###############
  176. "model/keras_layer/StatefulPartitionedCall:0.361"  4.786627 ###############
  177. "model/keras_layer/StatefulPartitionedCall:0.259"  4.786613 ###############
  178. "model/keras_layer/StatefulPartitionedCall:0.504"  4.786611 ###############
  179. "model/keras_layer/StatefulPartitionedCall:0.141"  4.786580 ###############
  180.  "model/keras_layer/StatefulPartitionedCall:0.90"  4.786566 ###############
  181. "model/keras_layer/StatefulPartitionedCall:0.506"  4.786554 ###############
  182. "model/keras_layer/StatefulPartitionedCall:0.492"  4.786539 ###############
  183. "model/keras_layer/StatefulPartitionedCall:0.429"  4.786482 ###############
  184. "model/keras_layer/StatefulPartitionedCall:0.186"  4.786405 ###############
  185.  "model/keras_layer/StatefulPartitionedCall:0.12"  4.786393 ###############
  186.  "model/keras_layer/StatefulPartitionedCall:0.84"  4.786151 ###############
  187. "model/keras_layer/StatefulPartitionedCall:0.288"  4.786149 ###############
  188.  "model/keras_layer/StatefulPartitionedCall:0.39"  4.786085 ###############
  189. "model/keras_layer/StatefulPartitionedCall:0.339"  4.786063 ###############
  190.  "model/keras_layer/StatefulPartitionedCall:0.34"  4.786048 ###############
  191. "model/keras_layer/StatefulPartitionedCall:0.480"  4.785980 ###############
  192. "model/keras_layer/StatefulPartitionedCall:0.407"  4.785853 ###############
  193. "model/keras_layer/StatefulPartitionedCall:0.467"  4.785841 ###############
  194.  "model/keras_layer/StatefulPartitionedCall:0.99"  4.785809 ###############
  195. "model/keras_layer/StatefulPartitionedCall:0.420"  4.785798 ###############
  196. "model/keras_layer/StatefulPartitionedCall:0.505"  4.785792 ###############
  197.  "model/keras_layer/StatefulPartitionedCall:0.63"  4.785771 ###############
  198.  "model/keras_layer/StatefulPartitionedCall:0.75"  4.785584 ###############
  199. "model/keras_layer/StatefulPartitionedCall:0.359"  4.785556 ###############
  200. "model/keras_layer/StatefulPartitionedCall:0.365"  4.785552 ###############
  201. "model/keras_layer/StatefulPartitionedCall:0.115"  4.785517 ###############
  202.  "model/keras_layer/StatefulPartitionedCall:0.22"  4.785502 ###############
  203. "model/keras_layer/StatefulPartitionedCall:0.234"  4.785500 ###############
  204. "model/keras_layer/StatefulPartitionedCall:0.333"  4.785478 ###############
  205. "model/keras_layer/StatefulPartitionedCall:0.413"  4.785468 ###############
  206.  "model/keras_layer/StatefulPartitionedCall:0.67"  4.785277 ###############
  207. "model/keras_layer/StatefulPartitionedCall:0.233"  4.785250 ###############
  208. "model/keras_layer/StatefulPartitionedCall:0.218"  4.785154 ###############
  209. "model/keras_layer/StatefulPartitionedCall:0.378"  4.785132 ###############
  210.  "model/keras_layer/StatefulPartitionedCall:0.29"  4.785098 ###############
  211. "model/keras_layer/StatefulPartitionedCall:0.282"  4.785089 ###############
  212.  "model/keras_layer/StatefulPartitionedCall:0.60"  4.785020 ###############
  213. "model/keras_layer/StatefulPartitionedCall:0.303"  4.784974 ###############
  214. "model/keras_layer/StatefulPartitionedCall:0.211"  4.784924 ###############
  215. "model/keras_layer/StatefulPartitionedCall:0.497"  4.784905 ###############
  216. "model/keras_layer/StatefulPartitionedCall:0.498"  4.784779 ###############
  217.  "model/keras_layer/StatefulPartitionedCall:0.28"  4.784707 ###############
  218. "model/keras_layer/StatefulPartitionedCall:0.482"  4.784654 ###############
  219. "model/keras_layer/StatefulPartitionedCall:0.379"  4.784541 ###############
  220. "model/keras_layer/StatefulPartitionedCall:0.390"  4.784454 ###############
  221. "model/keras_layer/StatefulPartitionedCall:0.201"  4.784399 ###############
  222. "model/keras_layer/StatefulPartitionedCall:0.227"  4.784391 ###############
  223. "model/keras_layer/StatefulPartitionedCall:0.105"  4.784355 ###############
  224. "model/keras_layer/StatefulPartitionedCall:0.214"  4.784346 ###############
  225. "model/keras_layer/StatefulPartitionedCall:0.246"  4.784317 ###############
  226. "model/keras_layer/StatefulPartitionedCall:0.158"  4.783917 ###############
  227. "model/keras_layer/StatefulPartitionedCall:0.131"  4.783867 ###############
  228. "model/keras_layer/StatefulPartitionedCall:0.257"  4.783861 ###############
  229. "model/keras_layer/StatefulPartitionedCall:0.448"  4.783647 ###############
  230. "model/keras_layer/StatefulPartitionedCall:0.101"  4.783562 ###############
  231. "model/keras_layer/StatefulPartitionedCall:0.254"  4.783447 ###############
  232. "model/keras_layer/StatefulPartitionedCall:0.329"  4.783436 ###############
  233. "model/keras_layer/StatefulPartitionedCall:0.414"  4.783421 ###############
  234.  "model/keras_layer/StatefulPartitionedCall:0.59"  4.783289 ###############
  235. "model/keras_layer/StatefulPartitionedCall:0.307"  4.783259 ###############
  236.  "model/keras_layer/StatefulPartitionedCall:0.68"  4.783223 ###############
  237. "model/keras_layer/StatefulPartitionedCall:0.507"  4.783132 ###############
  238. "model/keras_layer/StatefulPartitionedCall:0.423"  4.783107 ###############
  239. "model/keras_layer/StatefulPartitionedCall:0.230"  4.783088 ###############
  240. "model/keras_layer/StatefulPartitionedCall:0.388"  4.783082 ###############
  241.  "model/keras_layer/StatefulPartitionedCall:0.56"  4.782973 ###############
  242. "model/keras_layer/StatefulPartitionedCall:0.124"  4.782935 ###############
  243. "model/keras_layer/StatefulPartitionedCall:0.175"  4.782934 ###############
  244. "model/keras_layer/StatefulPartitionedCall:0.196"  4.782887 ###############
  245. "model/keras_layer/StatefulPartitionedCall:0.457"  4.782757 ###############
  246. "model/keras_layer/StatefulPartitionedCall:0.352"  4.782689 ###############
  247. "model/keras_layer/StatefulPartitionedCall:0.203"  4.782674 ###############
  248. "model/keras_layer/StatefulPartitionedCall:0.202"  4.782642 ###############
  249. "model/keras_layer/StatefulPartitionedCall:0.111"  4.782636 ###############
  250.  "model/keras_layer/StatefulPartitionedCall:0.82"  4.782625 ###############
  251. "model/keras_layer/StatefulPartitionedCall:0.190"  4.782606 ###############
  252. "model/keras_layer/StatefulPartitionedCall:0.128"  4.782568 ###############
  253. "model/keras_layer/StatefulPartitionedCall:0.266"  4.782524 ###############
  254. "model/keras_layer/StatefulPartitionedCall:0.320"  4.782437 ###############
  255. "model/keras_layer/StatefulPartitionedCall:0.154"  4.782366 ###############
  256. "model/keras_layer/StatefulPartitionedCall:0.129"  4.782081 ###############
  257.  "model/keras_layer/StatefulPartitionedCall:0.79"  4.781861 ###############
  258. "model/keras_layer/StatefulPartitionedCall:0.145"  4.781858 ###############
  259. "model/keras_layer/StatefulPartitionedCall:0.437"  4.781847 ###############
  260. "model/keras_layer/StatefulPartitionedCall:0.321"  4.781836 ###############
  261. "model/keras_layer/StatefulPartitionedCall:0.138"  4.781812 ###############
  262. "model/keras_layer/StatefulPartitionedCall:0.471"  4.781721 ###############
  263. "model/keras_layer/StatefulPartitionedCall:0.461"  4.781717 ###############
  264. "model/keras_layer/StatefulPartitionedCall:0.386"  4.781679 ###############
  265. "model/keras_layer/StatefulPartitionedCall:0.165"  4.781611 ###############
  266. "model/keras_layer/StatefulPartitionedCall:0.434"  4.781507 ###############
  267. "model/keras_layer/StatefulPartitionedCall:0.462"  4.781441 ###############
  268. "model/keras_layer/StatefulPartitionedCall:0.197"  4.781425 ###############
  269.   "model/keras_layer/StatefulPartitionedCall:0.9"  4.781382 ###############
  270. "model/keras_layer/StatefulPartitionedCall:0.433"  4.781330 ###############
  271. "model/keras_layer/StatefulPartitionedCall:0.400"  4.781238 ###############
  272. "model/keras_layer/StatefulPartitionedCall:0.222"  4.781176 ###############
  273. "model/keras_layer/StatefulPartitionedCall:0.164"  4.781136 ###############
  274. "model/keras_layer/StatefulPartitionedCall:0.479"  4.781038 ###############
  275. "model/keras_layer/StatefulPartitionedCall:0.146"  4.780970 ###############
  276. "model/keras_layer/StatefulPartitionedCall:0.245"  4.780911 ###############
  277. "model/keras_layer/StatefulPartitionedCall:0.251"  4.780898 ###############
  278. "model/keras_layer/StatefulPartitionedCall:0.384"  4.780850 ###############
  279. "model/keras_layer/StatefulPartitionedCall:0.401"  4.780791 ###############
  280. "model/keras_layer/StatefulPartitionedCall:0.232"  4.780781 ###############
  281. "model/keras_layer/StatefulPartitionedCall:0.308"  4.780764 ###############
  282.  "model/keras_layer/StatefulPartitionedCall:0.66"  4.780636 ###############
  283. "model/keras_layer/StatefulPartitionedCall:0.330"  4.780620 ###############
  284.  "model/keras_layer/StatefulPartitionedCall:0.24"  4.780511 ###############
  285. "model/keras_layer/StatefulPartitionedCall:0.174"  4.780496 ###############
  286. "model/keras_layer/StatefulPartitionedCall:0.502"  4.780244 ###############
  287. "model/keras_layer/StatefulPartitionedCall:0.299"  4.779967 ###############
  288. "model/keras_layer/StatefulPartitionedCall:0.243"  4.779800 ###############
  289. "model/keras_layer/StatefulPartitionedCall:0.194"  4.779479 ###############
  290. "model/keras_layer/StatefulPartitionedCall:0.432"  4.779445 ###############
  291. "model/keras_layer/StatefulPartitionedCall:0.106"  4.779271 ###############
  292. "model/keras_layer/StatefulPartitionedCall:0.274"  4.779252 ###############
  293. "model/keras_layer/StatefulPartitionedCall:0.302"  4.778608 ###############
  294. "model/keras_layer/StatefulPartitionedCall:0.369"  4.778527 ###############
  295. "model/keras_layer/StatefulPartitionedCall:0.235"  4.778101 ###############
  296. "model/keras_layer/StatefulPartitionedCall:0.206"  4.778054 ###############
  297. "model/keras_layer/StatefulPartitionedCall:0.451"  4.777826 ###############
  298. "model/keras_layer/StatefulPartitionedCall:0.272"  4.777674 ###############
  299. "model/keras_layer/StatefulPartitionedCall:0.415"  4.777599 ###############
  300.  "model/keras_layer/StatefulPartitionedCall:0.96"  4.777530 ###############
  301. "model/keras_layer/StatefulPartitionedCall:0.376"  4.777388 ###############
  302. "model/keras_layer/StatefulPartitionedCall:0.456"  4.776773 ###############
  303. "model/keras_layer/StatefulPartitionedCall:0.483"  4.776688 ###############
  304.  "model/keras_layer/StatefulPartitionedCall:0.31"  4.776656 ###############
  305. "model/keras_layer/StatefulPartitionedCall:0.509"  4.776567 ###############
  306. "model/keras_layer/StatefulPartitionedCall:0.472"  4.776521 ###############
  307. "model/keras_layer/StatefulPartitionedCall:0.104"  4.776201 ###############
  308. "model/keras_layer/StatefulPartitionedCall:0.155"  4.776042 ###############
  309. "model/keras_layer/StatefulPartitionedCall:0.468"  4.776003 ###############
  310.  "model/keras_layer/StatefulPartitionedCall:0.25"  4.775908 ###############
  311. "model/keras_layer/StatefulPartitionedCall:0.183"  4.775870 ###############
  312. "model/keras_layer/StatefulPartitionedCall:0.185"  4.775836 ###############
  313. "model/keras_layer/StatefulPartitionedCall:0.160"  4.775718 ###############
  314.  "model/keras_layer/StatefulPartitionedCall:0.80"  4.775669 ###############
  315. "model/keras_layer/StatefulPartitionedCall:0.265"  4.775536 ###############
  316.   "model/keras_layer/StatefulPartitionedCall:0.8"  4.775347 ###############
  317.  "model/keras_layer/StatefulPartitionedCall:0.27"  4.775324 ###############
  318.  "model/keras_layer/StatefulPartitionedCall:0.58"  4.775125 ###############
  319. "model/keras_layer/StatefulPartitionedCall:0.387"  4.775078 ###############
  320. "model/keras_layer/StatefulPartitionedCall:0.444"  4.775011 ###############
  321. "model/keras_layer/StatefulPartitionedCall:0.473"  4.774921 ###############
  322. "model/keras_layer/StatefulPartitionedCall:0.226"  4.774907 ###############
  323. "model/keras_layer/StatefulPartitionedCall:0.184"  4.774875 ###############
  324.  "model/keras_layer/StatefulPartitionedCall:0.32"  4.774689 ###############
  325. "model/keras_layer/StatefulPartitionedCall:0.405"  4.774685 ###############
  326. "model/keras_layer/StatefulPartitionedCall:0.132"  4.774628 ###############
  327. "model/keras_layer/StatefulPartitionedCall:0.113"  4.774566 ###############
  328. "model/keras_layer/StatefulPartitionedCall:0.501"  4.774377 ###############
  329. "model/keras_layer/StatefulPartitionedCall:0.161"  4.774028 ###############
  330.  "model/keras_layer/StatefulPartitionedCall:0.87"  4.773828 ###############
  331.  "model/keras_layer/StatefulPartitionedCall:0.14"  4.773612 ###############
  332. "model/keras_layer/StatefulPartitionedCall:0.231"  4.773444 ###############
  333. "model/keras_layer/StatefulPartitionedCall:0.314"  4.773373 ###############
  334. "model/keras_layer/StatefulPartitionedCall:0.412"  4.773217 ###############
  335. "model/keras_layer/StatefulPartitionedCall:0.306"  4.773211 ###############
  336. "model/keras_layer/StatefulPartitionedCall:0.130"  4.772546 ###############
  337. "model/keras_layer/StatefulPartitionedCall:0.280"  4.772329 ###############
  338.  "model/keras_layer/StatefulPartitionedCall:0.74"  4.772204 ###############
  339. "model/keras_layer/StatefulPartitionedCall:0.284"  4.772141 ###############
  340. "model/keras_layer/StatefulPartitionedCall:0.256"  4.772130 ###############
  341. "model/keras_layer/StatefulPartitionedCall:0.148"  4.772107 ###############
  342. "model/keras_layer/StatefulPartitionedCall:0.298"  4.771996 ###############
  343. "model/keras_layer/StatefulPartitionedCall:0.135"  4.771717 ###############
  344. "model/keras_layer/StatefulPartitionedCall:0.476"  4.771620 ###############
  345. "model/keras_layer/StatefulPartitionedCall:0.392"  4.771404 ###############
  346. "model/keras_layer/StatefulPartitionedCall:0.375"  4.771202 ###############
  347. "model/keras_layer/StatefulPartitionedCall:0.102"  4.771159 ###############
  348. "model/keras_layer/StatefulPartitionedCall:0.360"  4.770975 ###############
  349.  "model/keras_layer/StatefulPartitionedCall:0.69"  4.770768 ###############
  350. "model/keras_layer/StatefulPartitionedCall:0.147"  4.770582 ###############
  351. "model/keras_layer/StatefulPartitionedCall:0.325"  4.770576 ###############
  352.  "model/keras_layer/StatefulPartitionedCall:0.98"  4.770401 ###############
  353. "model/keras_layer/StatefulPartitionedCall:0.440"  4.770258 ###############
  354. "model/keras_layer/StatefulPartitionedCall:0.380"  4.770252 ###############
  355. "model/keras_layer/StatefulPartitionedCall:0.341"  4.770007 ###############
  356.  "model/keras_layer/StatefulPartitionedCall:0.35"  4.769625 ###############
  357. "model/keras_layer/StatefulPartitionedCall:0.446"  4.769560 ###############
  358. "model/keras_layer/StatefulPartitionedCall:0.229"  4.769457 ###############
  359. "model/keras_layer/StatefulPartitionedCall:0.402"  4.769197 ###############
  360. "model/keras_layer/StatefulPartitionedCall:0.287"  4.768515 ###############
  361.  "model/keras_layer/StatefulPartitionedCall:0.71"  4.768411 ###############
  362. "model/keras_layer/StatefulPartitionedCall:0.208"  4.767568 ###############
  363.  "model/keras_layer/StatefulPartitionedCall:0.45"  4.767253 ###############
  364. "model/keras_layer/StatefulPartitionedCall:0.357"  4.766753 ###############
  365. "model/keras_layer/StatefulPartitionedCall:0.250"  4.766620 ###############
  366. "model/keras_layer/StatefulPartitionedCall:0.397"  4.766411 ###############
  367. "model/keras_layer/StatefulPartitionedCall:0.133"  4.766367 ###############
  368. "model/keras_layer/StatefulPartitionedCall:0.373"  4.766178 ###############
  369. "model/keras_layer/StatefulPartitionedCall:0.262"  4.765391 ###############
  370. "model/keras_layer/StatefulPartitionedCall:0.374"  4.765345 ###############
  371. "model/keras_layer/StatefulPartitionedCall:0.443"  4.764812 ###############
  372. "model/keras_layer/StatefulPartitionedCall:0.304"  4.764479 ##############
  373. "model/keras_layer/StatefulPartitionedCall:0.221"  4.764447 ##############
  374. "model/keras_layer/StatefulPartitionedCall:0.383"  4.764423 ##############
  375. "model/keras_layer/StatefulPartitionedCall:0.500"  4.764301 ##############
  376. "model/keras_layer/StatefulPartitionedCall:0.416"  4.764296 ##############
  377. "model/keras_layer/StatefulPartitionedCall:0.485"  4.763649 ##############
  378. "model/keras_layer/StatefulPartitionedCall:0.393"  4.763562 ##############
  379. "model/keras_layer/StatefulPartitionedCall:0.411"  4.763456 ##############
  380. "model/keras_layer/StatefulPartitionedCall:0.327"  4.763453 ##############
  381. "model/keras_layer/StatefulPartitionedCall:0.364"  4.763217 ##############
  382.  "model/keras_layer/StatefulPartitionedCall:0.48"  4.763089 ##############
  383. "model/keras_layer/StatefulPartitionedCall:0.419"  4.762776 ##############
  384. "model/keras_layer/StatefulPartitionedCall:0.358"  4.762472 ##############
  385.  "model/keras_layer/StatefulPartitionedCall:0.16"  4.762430 ##############
  386. "model/keras_layer/StatefulPartitionedCall:0.460"  4.762384 ##############
  387. "model/keras_layer/StatefulPartitionedCall:0.189"  4.762125 ##############
  388. "model/keras_layer/StatefulPartitionedCall:0.291"  4.761875 ##############
  389. "model/keras_layer/StatefulPartitionedCall:0.335"  4.761792 ##############
  390.  "model/keras_layer/StatefulPartitionedCall:0.78"  4.761614 ##############
  391. "model/keras_layer/StatefulPartitionedCall:0.187"  4.761347 ##############
  392. "model/keras_layer/StatefulPartitionedCall:0.297"  4.761315 ##############
  393. "model/keras_layer/StatefulPartitionedCall:0.114"  4.760673 ##############
  394.  "model/keras_layer/StatefulPartitionedCall:0.54"  4.760538 ##############
  395. "model/keras_layer/StatefulPartitionedCall:0.372"  4.760477 ##############
  396. "model/keras_layer/StatefulPartitionedCall:0.510"  4.760309 ##############
  397. "model/keras_layer/StatefulPartitionedCall:0.293"  4.760166 ##############
  398. "model/keras_layer/StatefulPartitionedCall:0.209"  4.757214 ##############
  399.  "model/keras_layer/StatefulPartitionedCall:0.85"  4.756668 ##############
  400.  "model/keras_layer/StatefulPartitionedCall:0.51"  4.756567 ##############
  401. "model/keras_layer/StatefulPartitionedCall:0.290"  4.755849 ##############
  402. "model/keras_layer/StatefulPartitionedCall:0.260"  4.755524 ##############
  403.  "model/keras_layer/StatefulPartitionedCall:0.65"  4.755361 ##############
  404.  "model/keras_layer/StatefulPartitionedCall:0.42"  4.755255 ##############
  405. "model/keras_layer/StatefulPartitionedCall:0.268"  4.755203 ##############
  406.  "model/keras_layer/StatefulPartitionedCall:0.17"  4.755183 ##############
  407. "model/keras_layer/StatefulPartitionedCall:0.120"  4.755042 ##############
  408. "model/keras_layer/StatefulPartitionedCall:0.275"  4.754991 ##############
  409. "model/keras_layer/StatefulPartitionedCall:0.337"  4.753927 ##############
  410. "model/keras_layer/StatefulPartitionedCall:0.116"  4.753578 ##############
  411.  "model/keras_layer/StatefulPartitionedCall:0.18"  4.753455 ##############
  412. "model/keras_layer/StatefulPartitionedCall:0.442"  4.753428 ##############
  413. "model/keras_layer/StatefulPartitionedCall:0.417"  4.753358 ##############
  414. "model/keras_layer/StatefulPartitionedCall:0.110"  4.752974 ##############
  415. "model/keras_layer/StatefulPartitionedCall:0.286"  4.752872 ##############
  416. "model/keras_layer/StatefulPartitionedCall:0.149"  4.752845 ##############
  417. "model/keras_layer/StatefulPartitionedCall:0.193"  4.751394 ##############
  418. "model/keras_layer/StatefulPartitionedCall:0.319"  4.750936 ##############
  419. "model/keras_layer/StatefulPartitionedCall:0.173"  4.750250 ##############
  420. "model/keras_layer/StatefulPartitionedCall:0.367"  4.749730 ##############
  421. "model/keras_layer/StatefulPartitionedCall:0.328"  4.749677 ##############
  422. "model/keras_layer/StatefulPartitionedCall:0.348"  4.749582 ##############
  423. "model/keras_layer/StatefulPartitionedCall:0.344"  4.749558 ##############
  424.  "model/keras_layer/StatefulPartitionedCall:0.11"  4.749109 ##############
  425.  "model/keras_layer/StatefulPartitionedCall:0.37"  4.748526 ##############
  426. "model/keras_layer/StatefulPartitionedCall:0.252"  4.748041 ##############
  427. "model/keras_layer/StatefulPartitionedCall:0.261"  4.747185 ##############
  428. "model/keras_layer/StatefulPartitionedCall:0.276"  4.747183 ##############
  429. "model/keras_layer/StatefulPartitionedCall:0.449"  4.746890 ##############
  430. "model/keras_layer/StatefulPartitionedCall:0.241"  4.746889 ##############
  431. "model/keras_layer/StatefulPartitionedCall:0.199"  4.746053 ##############
  432. "model/keras_layer/StatefulPartitionedCall:0.300"  4.745888 ##############
  433. "model/keras_layer/StatefulPartitionedCall:0.351"  4.745590 ##############
  434. "model/keras_layer/StatefulPartitionedCall:0.204"  4.745588 ##############
  435. "model/keras_layer/StatefulPartitionedCall:0.368"  4.745192 ##############
  436. "model/keras_layer/StatefulPartitionedCall:0.171"  4.744926 ##############
  437. "model/keras_layer/StatefulPartitionedCall:0.317"  4.744826 ##############
  438. "model/keras_layer/StatefulPartitionedCall:0.484"  4.744421 ##############
  439.  "model/keras_layer/StatefulPartitionedCall:0.92"  4.743447 ##############
  440. "model/keras_layer/StatefulPartitionedCall:0.178"  4.743032 ##############
  441. "model/keras_layer/StatefulPartitionedCall:0.255"  4.741322 ##############
  442.  "model/keras_layer/StatefulPartitionedCall:0.62"  4.740898 ##############
  443. "model/keras_layer/StatefulPartitionedCall:0.426"  4.739398 ##############
  444.  "model/keras_layer/StatefulPartitionedCall:0.41"  4.739029 ##############
  445. "model/keras_layer/StatefulPartitionedCall:0.142"  4.739014 ##############
  446. "model/keras_layer/StatefulPartitionedCall:0.117"  4.738909 ##############
  447. "model/keras_layer/StatefulPartitionedCall:0.381"  4.738347 ##############
  448.  "model/keras_layer/StatefulPartitionedCall:0.89"  4.737696 ##############
  449.  "model/keras_layer/StatefulPartitionedCall:0.76"  4.737439 #############
  450. "model/keras_layer/StatefulPartitionedCall:0.278"  4.737354 #############
  451. "model/keras_layer/StatefulPartitionedCall:0.140"  4.735784 #############
  452. "model/keras_layer/StatefulPartitionedCall:0.469"  4.735758 #############
  453. "model/keras_layer/StatefulPartitionedCall:0.418"  4.731954 #############
  454. "model/keras_layer/StatefulPartitionedCall:0.455"  4.731811 #############
  455. "model/keras_layer/StatefulPartitionedCall:0.118"  4.731623 #############
  456. "model/keras_layer/StatefulPartitionedCall:0.253"  4.730163 #############
  457. "model/keras_layer/StatefulPartitionedCall:0.454"  4.729726 #############
  458. "model/keras_layer/StatefulPartitionedCall:0.207"  4.728658 #############
  459. "model/keras_layer/StatefulPartitionedCall:0.192"  4.727481 #############
  460. "model/keras_layer/StatefulPartitionedCall:0.486"  4.726338 #############
  461. "model/keras_layer/StatefulPartitionedCall:0.176"  4.726193 #############
  462. "model/keras_layer/StatefulPartitionedCall:0.289"  4.723309 #############
  463.  "model/keras_layer/StatefulPartitionedCall:0.95"  4.722600 #############
  464. "model/keras_layer/StatefulPartitionedCall:0.362"  4.721751 #############
  465. "model/keras_layer/StatefulPartitionedCall:0.168"  4.721548 #############
  466. "model/keras_layer/StatefulPartitionedCall:0.453"  4.721497 #############
  467.  "model/keras_layer/StatefulPartitionedCall:0.88"  4.720156 #############
  468.  "model/keras_layer/StatefulPartitionedCall:0.46"  4.719808 #############
  469. "model/keras_layer/StatefulPartitionedCall:0.439"  4.719009 #############
  470. "model/keras_layer/StatefulPartitionedCall:0.285"  4.713931 #############
  471. "model/keras_layer/StatefulPartitionedCall:0.511"  4.712969 #############
  472. "model/keras_layer/StatefulPartitionedCall:0.427"  4.711664 #############
  473. "model/keras_layer/StatefulPartitionedCall:0.273"  4.711022 #############
  474. "model/keras_layer/StatefulPartitionedCall:0.152"  4.710258 ############
  475. "model/keras_layer/StatefulPartitionedCall:0.450"  4.709386 ############
  476.  "model/keras_layer/StatefulPartitionedCall:0.73"  4.706532 ############
  477. "model/keras_layer/StatefulPartitionedCall:0.315"  4.705647 ############
  478. "model/keras_layer/StatefulPartitionedCall:0.249"  4.704767 ############
  479. "model/keras_layer/StatefulPartitionedCall:0.338"  4.704629 ############
  480.  "model/keras_layer/StatefulPartitionedCall:0.44"  4.704616 ############
  481.   "model/keras_layer/StatefulPartitionedCall:0.0"  4.704243 ############
  482.   "model/keras_layer/StatefulPartitionedCall:0.1"  4.703584 ############
  483. "model/keras_layer/StatefulPartitionedCall:0.100"  4.696695 ############
  484. "model/keras_layer/StatefulPartitionedCall:0.322"  4.695234 ############
  485. "model/keras_layer/StatefulPartitionedCall:0.159"  4.694935 ############
  486. "model/keras_layer/StatefulPartitionedCall:0.219"  4.691967 ############
  487. "model/keras_layer/StatefulPartitionedCall:0.103"  4.685531 ############
  488. "model/keras_layer/StatefulPartitionedCall:0.225"  4.679884 ###########
  489. "model/keras_layer/StatefulPartitionedCall:0.169"  4.673999 ###########
  490. "model/keras_layer/StatefulPartitionedCall:0.144"  4.672773 ###########
  491. "model/keras_layer/StatefulPartitionedCall:0.281"  4.672398 ###########
  492. "model/keras_layer/StatefulPartitionedCall:0.356"  4.671043 ###########
  493. "model/keras_layer/StatefulPartitionedCall:0.150"  4.662033 ###########
  494. "model/keras_layer/StatefulPartitionedCall:0.489"  4.656941 ###########
  495. "model/keras_layer/StatefulPartitionedCall:0.294"  4.655826 ##########
  496. "model/keras_layer/StatefulPartitionedCall:0.310"  4.647856 ##########
  497. "model/keras_layer/StatefulPartitionedCall:0.332"  4.640438 ##########
  498. "model/keras_layer/StatefulPartitionedCall:0.323"  4.629164 #########
  499.  "model/keras_layer/StatefulPartitionedCall:0.10"  4.620969 #########
  500. "model/keras_layer/StatefulPartitionedCall:0.354"  4.619797 #########
  501. "model/keras_layer/StatefulPartitionedCall:0.463"  4.614919 #########
  502. "model/keras_layer/StatefulPartitionedCall:0.247"  4.612530 #########
  503. "model/keras_layer/StatefulPartitionedCall:0.464"  4.603565 #########
  504. "model/keras_layer/StatefulPartitionedCall:0.343"  4.601945 ########
  505. "model/keras_layer/StatefulPartitionedCall:0.399"  4.589854 ########
  506. "model/keras_layer/StatefulPartitionedCall:0.188"  4.546768 ######
  507. "model/keras_layer/StatefulPartitionedCall:0.127"  4.543450 ######
  508. "model/keras_layer/StatefulPartitionedCall:0.180"  4.539494 ######
  509. "model/keras_layer/StatefulPartitionedCall:0.458"  4.506813 #####
  510. "model/keras_layer/StatefulPartitionedCall:0.166"  4.486642 ####
  511. "model/keras_layer/StatefulPartitionedCall:0.153"  4.481521 ####
  512. "model/keras_layer/StatefulPartitionedCall:0.126"  4.432638 ##
  513.  "model/keras_layer/StatefulPartitionedCall:0.50"  4.358862 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_AS_ROOT:
    1.  "model/keras_layer/StatefulPartitionedCall:0.50"  9.000000 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.126"  7.000000 ############
    3. "model/keras_layer/StatefulPartitionedCall:0.153"  6.000000 ##########
    4. "model/keras_layer/StatefulPartitionedCall:0.127"  5.000000 ########
    5. "model/keras_layer/StatefulPartitionedCall:0.180"  5.000000 ########
    6. "model/keras_layer/StatefulPartitionedCall:0.188"  4.000000 ######
    7. "model/keras_layer/StatefulPartitionedCall:0.247"  4.000000 ######
    8. "model/keras_layer/StatefulPartitionedCall:0.399"  4.000000 ######
    9. "model/keras_layer/StatefulPartitionedCall:0.103"  3.000000 ####
   10. "model/keras_layer/StatefulPartitionedCall:0.150"  3.000000 ####
   11. "model/keras_layer/StatefulPartitionedCall:0.169"  3.000000 ####
   12. "model/keras_layer/StatefulPartitionedCall:0.225"  3.000000 ####
   13. "model/keras_layer/StatefulPartitionedCall:0.310"  3.000000 ####
   14. "model/keras_layer/StatefulPartitionedCall:0.323"  3.000000 ####
   15. "model/keras_layer/StatefulPartitionedCall:0.332"  3.000000 ####
   16. "model/keras_layer/StatefulPartitionedCall:0.343"  3.000000 ####
   17. "model/keras_layer/StatefulPartitionedCall:0.354"  3.000000 ####
   18. "model/keras_layer/StatefulPartitionedCall:0.458"  3.000000 ####
   19.   "model/keras_layer/StatefulPartitionedCall:0.1"  2.000000 ##
   20. "model/keras_layer/StatefulPartitionedCall:0.100"  2.000000 ##
   21. "model/keras_layer/StatefulPartitionedCall:0.144"  2.000000 ##
   22. "model/keras_layer/StatefulPartitionedCall:0.166"  2.000000 ##
   23. "model/keras_layer/StatefulPartitionedCall:0.281"  2.000000 ##
   24. "model/keras_layer/StatefulPartitionedCall:0.338"  2.000000 ##
   25. "model/keras_layer/StatefulPartitionedCall:0.356"  2.000000 ##
   26. "model/keras_layer/StatefulPartitionedCall:0.439"  2.000000 ##
   27. "model/keras_layer/StatefulPartitionedCall:0.450"  2.000000 ##
   28. "model/keras_layer/StatefulPartitionedCall:0.463"  2.000000 ##
   29. "model/keras_layer/StatefulPartitionedCall:0.489"  2.000000 ##
   30.   "model/keras_layer/StatefulPartitionedCall:0.0"  1.000000 
   31. "model/keras_layer/StatefulPartitionedCall:0.140"  1.000000 
   32. "model/keras_layer/StatefulPartitionedCall:0.152"  1.000000 
   33. "model/keras_layer/StatefulPartitionedCall:0.173"  1.000000 
   34. "model/keras_layer/StatefulPartitionedCall:0.176"  1.000000 
   35. "model/keras_layer/StatefulPartitionedCall:0.192"  1.000000 
   36. "model/keras_layer/StatefulPartitionedCall:0.199"  1.000000 
   37. "model/keras_layer/StatefulPartitionedCall:0.207"  1.000000 
   38. "model/keras_layer/StatefulPartitionedCall:0.219"  1.000000 
   39. "model/keras_layer/StatefulPartitionedCall:0.249"  1.000000 
   40. "model/keras_layer/StatefulPartitionedCall:0.253"  1.000000 
   41. "model/keras_layer/StatefulPartitionedCall:0.273"  1.000000 
   42. "model/keras_layer/StatefulPartitionedCall:0.294"  1.000000 
   43. "model/keras_layer/StatefulPartitionedCall:0.300"  1.000000 
   44. "model/keras_layer/StatefulPartitionedCall:0.319"  1.000000 
   45. "model/keras_layer/StatefulPartitionedCall:0.322"  1.000000 
   46. "model/keras_layer/StatefulPartitionedCall:0.344"  1.000000 
   47. "model/keras_layer/StatefulPartitionedCall:0.348"  1.000000 
   48. "model/keras_layer/StatefulPartitionedCall:0.351"  1.000000 
   49. "model/keras_layer/StatefulPartitionedCall:0.362"  1.000000 
   50. "model/keras_layer/StatefulPartitionedCall:0.367"  1.000000 
   51.  "model/keras_layer/StatefulPartitionedCall:0.37"  1.000000 
   52. "model/keras_layer/StatefulPartitionedCall:0.381"  1.000000 
   53. "model/keras_layer/StatefulPartitionedCall:0.426"  1.000000 
   54. "model/keras_layer/StatefulPartitionedCall:0.442"  1.000000 
   55. "model/keras_layer/StatefulPartitionedCall:0.454"  1.000000 
   56. "model/keras_layer/StatefulPartitionedCall:0.455"  1.000000 
   57. "model/keras_layer/StatefulPartitionedCall:0.464"  1.000000 
   58. "model/keras_layer/StatefulPartitionedCall:0.486"  1.000000 
   59.  "model/keras_layer/StatefulPartitionedCall:0.62"  1.000000 
   60.  "model/keras_layer/StatefulPartitionedCall:0.73"  1.000000 
   61.  "model/keras_layer/StatefulPartitionedCall:0.76"  1.000000 
   62.  "model/keras_layer/StatefulPartitionedCall:0.92"  1.000000 
   63.  "model/keras_layer/StatefulPartitionedCall:0.95"  1.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_NODES:
    1. "model/keras_layer/StatefulPartitionedCall:0.458" 37.000000 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.166" 32.000000 #############
    3. "model/keras_layer/StatefulPartitionedCall:0.464" 30.000000 ############
    4.  "model/keras_layer/StatefulPartitionedCall:0.50" 30.000000 ############
    5. "model/keras_layer/StatefulPartitionedCall:0.188" 28.000000 ############
    6. "model/keras_layer/StatefulPartitionedCall:0.126" 25.000000 ##########
    7. "model/keras_layer/StatefulPartitionedCall:0.153" 24.000000 ##########
    8. "model/keras_layer/StatefulPartitionedCall:0.354" 24.000000 ##########
    9. "model/keras_layer/StatefulPartitionedCall:0.159" 23.000000 #########
   10. "model/keras_layer/StatefulPartitionedCall:0.127" 21.000000 ########
   11.  "model/keras_layer/StatefulPartitionedCall:0.10" 20.000000 ########
   12. "model/keras_layer/StatefulPartitionedCall:0.247" 20.000000 ########
   13. "model/keras_layer/StatefulPartitionedCall:0.356" 20.000000 ########
   14. "model/keras_layer/StatefulPartitionedCall:0.294" 19.000000 ########
   15. "model/keras_layer/StatefulPartitionedCall:0.343" 19.000000 ########
   16.  "model/keras_layer/StatefulPartitionedCall:0.44" 19.000000 ########
   17. "model/keras_layer/StatefulPartitionedCall:0.180" 18.000000 #######
   18.  "model/keras_layer/StatefulPartitionedCall:0.46" 18.000000 #######
   19. "model/keras_layer/StatefulPartitionedCall:0.133" 17.000000 #######
   20. "model/keras_layer/StatefulPartitionedCall:0.178" 17.000000 #######
   21. "model/keras_layer/StatefulPartitionedCall:0.281" 17.000000 #######
   22. "model/keras_layer/StatefulPartitionedCall:0.427" 17.000000 #######
   23. "model/keras_layer/StatefulPartitionedCall:0.152" 15.000000 ######
   24. "model/keras_layer/StatefulPartitionedCall:0.219" 15.000000 ######
   25. "model/keras_layer/StatefulPartitionedCall:0.289" 15.000000 ######
   26. "model/keras_layer/StatefulPartitionedCall:0.310" 15.000000 ######
   27. "model/keras_layer/StatefulPartitionedCall:0.323" 15.000000 ######
   28. "model/keras_layer/StatefulPartitionedCall:0.337" 15.000000 ######
   29. "model/keras_layer/StatefulPartitionedCall:0.463" 15.000000 ######
   30. "model/keras_layer/StatefulPartitionedCall:0.511" 15.000000 ######
   31.   "model/keras_layer/StatefulPartitionedCall:0.0" 14.000000 #####
   32. "model/keras_layer/StatefulPartitionedCall:0.142" 14.000000 #####
   33. "model/keras_layer/StatefulPartitionedCall:0.171" 14.000000 #####
   34. "model/keras_layer/StatefulPartitionedCall:0.322" 14.000000 #####
   35. "model/keras_layer/StatefulPartitionedCall:0.399" 14.000000 #####
   36. "model/keras_layer/StatefulPartitionedCall:0.144" 13.000000 #####
   37. "model/keras_layer/StatefulPartitionedCall:0.209" 13.000000 #####
   38. "model/keras_layer/StatefulPartitionedCall:0.260" 13.000000 #####
   39. "model/keras_layer/StatefulPartitionedCall:0.286" 13.000000 #####
   40. "model/keras_layer/StatefulPartitionedCall:0.290" 13.000000 #####
   41. "model/keras_layer/StatefulPartitionedCall:0.315" 13.000000 #####
   42. "model/keras_layer/StatefulPartitionedCall:0.362" 13.000000 #####
   43. "model/keras_layer/StatefulPartitionedCall:0.489" 13.000000 #####
   44.  "model/keras_layer/StatefulPartitionedCall:0.88" 13.000000 #####
   45.   "model/keras_layer/StatefulPartitionedCall:0.1" 12.000000 ####
   46. "model/keras_layer/StatefulPartitionedCall:0.120" 12.000000 ####
   47. "model/keras_layer/StatefulPartitionedCall:0.140" 12.000000 ####
   48. "model/keras_layer/StatefulPartitionedCall:0.168" 12.000000 ####
   49.  "model/keras_layer/StatefulPartitionedCall:0.18" 12.000000 ####
   50. "model/keras_layer/StatefulPartitionedCall:0.192" 12.000000 ####
   51. "model/keras_layer/StatefulPartitionedCall:0.193" 12.000000 ####
   52. "model/keras_layer/StatefulPartitionedCall:0.250" 12.000000 ####
   53. "model/keras_layer/StatefulPartitionedCall:0.278" 12.000000 ####
   54. "model/keras_layer/StatefulPartitionedCall:0.419" 12.000000 ####
   55. "model/keras_layer/StatefulPartitionedCall:0.469" 12.000000 ####
   56.  "model/keras_layer/StatefulPartitionedCall:0.73" 12.000000 ####
   57.  "model/keras_layer/StatefulPartitionedCall:0.87" 12.000000 ####
   58.  "model/keras_layer/StatefulPartitionedCall:0.11" 11.000000 ####
   59. "model/keras_layer/StatefulPartitionedCall:0.273" 11.000000 ####
   60. "model/keras_layer/StatefulPartitionedCall:0.285" 11.000000 ####
   61. "model/keras_layer/StatefulPartitionedCall:0.332" 11.000000 ####
   62. "model/keras_layer/StatefulPartitionedCall:0.368" 11.000000 ####
   63.  "model/keras_layer/StatefulPartitionedCall:0.41" 11.000000 ####
   64. "model/keras_layer/StatefulPartitionedCall:0.460" 11.000000 ####
   65. "model/keras_layer/StatefulPartitionedCall:0.485" 11.000000 ####
   66.  "model/keras_layer/StatefulPartitionedCall:0.51" 11.000000 ####
   67.  "model/keras_layer/StatefulPartitionedCall:0.14" 10.000000 ####
   68. "model/keras_layer/StatefulPartitionedCall:0.149" 10.000000 ####
   69. "model/keras_layer/StatefulPartitionedCall:0.204" 10.000000 ####
   70. "model/keras_layer/StatefulPartitionedCall:0.207" 10.000000 ####
   71. "model/keras_layer/StatefulPartitionedCall:0.252" 10.000000 ####
   72. "model/keras_layer/StatefulPartitionedCall:0.291" 10.000000 ####
   73. "model/keras_layer/StatefulPartitionedCall:0.297" 10.000000 ####
   74. "model/keras_layer/StatefulPartitionedCall:0.338" 10.000000 ####
   75. "model/keras_layer/StatefulPartitionedCall:0.358" 10.000000 ####
   76. "model/keras_layer/StatefulPartitionedCall:0.360" 10.000000 ####
   77. "model/keras_layer/StatefulPartitionedCall:0.449" 10.000000 ####
   78. "model/keras_layer/StatefulPartitionedCall:0.451" 10.000000 ####
   79. "model/keras_layer/StatefulPartitionedCall:0.454" 10.000000 ####
   80. "model/keras_layer/StatefulPartitionedCall:0.473" 10.000000 ####
   81.  "model/keras_layer/StatefulPartitionedCall:0.76" 10.000000 ####
   82.  "model/keras_layer/StatefulPartitionedCall:0.89" 10.000000 ####
   83. "model/keras_layer/StatefulPartitionedCall:0.135"  9.000000 ###
   84. "model/keras_layer/StatefulPartitionedCall:0.145"  9.000000 ###
   85. "model/keras_layer/StatefulPartitionedCall:0.150"  9.000000 ###
   86. "model/keras_layer/StatefulPartitionedCall:0.175"  9.000000 ###
   87. "model/keras_layer/StatefulPartitionedCall:0.229"  9.000000 ###
   88. "model/keras_layer/StatefulPartitionedCall:0.241"  9.000000 ###
   89. "model/keras_layer/StatefulPartitionedCall:0.255"  9.000000 ###
   90. "model/keras_layer/StatefulPartitionedCall:0.261"  9.000000 ###
   91. "model/keras_layer/StatefulPartitionedCall:0.275"  9.000000 ###
   92. "model/keras_layer/StatefulPartitionedCall:0.300"  9.000000 ###
   93. "model/keras_layer/StatefulPartitionedCall:0.317"  9.000000 ###
   94.  "model/keras_layer/StatefulPartitionedCall:0.32"  9.000000 ###
   95. "model/keras_layer/StatefulPartitionedCall:0.327"  9.000000 ###
   96. "model/keras_layer/StatefulPartitionedCall:0.418"  9.000000 ###
   97. "model/keras_layer/StatefulPartitionedCall:0.471"  9.000000 ###
   98. "model/keras_layer/StatefulPartitionedCall:0.472"  9.000000 ###
   99. "model/keras_layer/StatefulPartitionedCall:0.500"  9.000000 ###
  100.  "model/keras_layer/StatefulPartitionedCall:0.65"  9.000000 ###
  101.  "model/keras_layer/StatefulPartitionedCall:0.80"  9.000000 ###
  102. "model/keras_layer/StatefulPartitionedCall:0.117"  8.000000 ###
  103. "model/keras_layer/StatefulPartitionedCall:0.118"  8.000000 ###
  104. "model/keras_layer/StatefulPartitionedCall:0.165"  8.000000 ###
  105. "model/keras_layer/StatefulPartitionedCall:0.169"  8.000000 ###
  106. "model/keras_layer/StatefulPartitionedCall:0.187"  8.000000 ###
  107. "model/keras_layer/StatefulPartitionedCall:0.194"  8.000000 ###
  108. "model/keras_layer/StatefulPartitionedCall:0.196"  8.000000 ###
  109. "model/keras_layer/StatefulPartitionedCall:0.203"  8.000000 ###
  110. "model/keras_layer/StatefulPartitionedCall:0.221"  8.000000 ###
  111. "model/keras_layer/StatefulPartitionedCall:0.225"  8.000000 ###
  112. "model/keras_layer/StatefulPartitionedCall:0.249"  8.000000 ###
  113.  "model/keras_layer/StatefulPartitionedCall:0.28"  8.000000 ###
  114.  "model/keras_layer/StatefulPartitionedCall:0.29"  8.000000 ###
  115. "model/keras_layer/StatefulPartitionedCall:0.325"  8.000000 ###
  116. "model/keras_layer/StatefulPartitionedCall:0.328"  8.000000 ###
  117. "model/keras_layer/StatefulPartitionedCall:0.330"  8.000000 ###
  118. "model/keras_layer/StatefulPartitionedCall:0.351"  8.000000 ###
  119. "model/keras_layer/StatefulPartitionedCall:0.357"  8.000000 ###
  120.  "model/keras_layer/StatefulPartitionedCall:0.37"  8.000000 ###
  121. "model/keras_layer/StatefulPartitionedCall:0.387"  8.000000 ###
  122. "model/keras_layer/StatefulPartitionedCall:0.392"  8.000000 ###
  123. "model/keras_layer/StatefulPartitionedCall:0.411"  8.000000 ###
  124. "model/keras_layer/StatefulPartitionedCall:0.453"  8.000000 ###
  125. "model/keras_layer/StatefulPartitionedCall:0.457"  8.000000 ###
  126.  "model/keras_layer/StatefulPartitionedCall:0.47"  8.000000 ###
  127. "model/keras_layer/StatefulPartitionedCall:0.502"  8.000000 ###
  128.  "model/keras_layer/StatefulPartitionedCall:0.58"  8.000000 ###
  129.  "model/keras_layer/StatefulPartitionedCall:0.69"  8.000000 ###
  130.  "model/keras_layer/StatefulPartitionedCall:0.78"  8.000000 ###
  131.  "model/keras_layer/StatefulPartitionedCall:0.95"  8.000000 ###
  132. "model/keras_layer/StatefulPartitionedCall:0.102"  7.000000 ##
  133. "model/keras_layer/StatefulPartitionedCall:0.104"  7.000000 ##
  134. "model/keras_layer/StatefulPartitionedCall:0.113"  7.000000 ##
  135. "model/keras_layer/StatefulPartitionedCall:0.114"  7.000000 ##
  136. "model/keras_layer/StatefulPartitionedCall:0.115"  7.000000 ##
  137. "model/keras_layer/StatefulPartitionedCall:0.148"  7.000000 ##
  138. "model/keras_layer/StatefulPartitionedCall:0.154"  7.000000 ##
  139. "model/keras_layer/StatefulPartitionedCall:0.161"  7.000000 ##
  140.  "model/keras_layer/StatefulPartitionedCall:0.17"  7.000000 ##
  141. "model/keras_layer/StatefulPartitionedCall:0.183"  7.000000 ##
  142. "model/keras_layer/StatefulPartitionedCall:0.184"  7.000000 ##
  143. "model/keras_layer/StatefulPartitionedCall:0.199"  7.000000 ##
  144. "model/keras_layer/StatefulPartitionedCall:0.208"  7.000000 ##
  145. "model/keras_layer/StatefulPartitionedCall:0.214"  7.000000 ##
  146. "model/keras_layer/StatefulPartitionedCall:0.231"  7.000000 ##
  147. "model/keras_layer/StatefulPartitionedCall:0.232"  7.000000 ##
  148. "model/keras_layer/StatefulPartitionedCall:0.235"  7.000000 ##
  149. "model/keras_layer/StatefulPartitionedCall:0.243"  7.000000 ##
  150. "model/keras_layer/StatefulPartitionedCall:0.266"  7.000000 ##
  151. "model/keras_layer/StatefulPartitionedCall:0.280"  7.000000 ##
  152. "model/keras_layer/StatefulPartitionedCall:0.284"  7.000000 ##
  153. "model/keras_layer/StatefulPartitionedCall:0.287"  7.000000 ##
  154. "model/keras_layer/StatefulPartitionedCall:0.298"  7.000000 ##
  155.  "model/keras_layer/StatefulPartitionedCall:0.35"  7.000000 ##
  156. "model/keras_layer/StatefulPartitionedCall:0.364"  7.000000 ##
  157. "model/keras_layer/StatefulPartitionedCall:0.375"  7.000000 ##
  158. "model/keras_layer/StatefulPartitionedCall:0.381"  7.000000 ##
  159. "model/keras_layer/StatefulPartitionedCall:0.393"  7.000000 ##
  160. "model/keras_layer/StatefulPartitionedCall:0.397"  7.000000 ##
  161. "model/keras_layer/StatefulPartitionedCall:0.446"  7.000000 ##
  162. "model/keras_layer/StatefulPartitionedCall:0.476"  7.000000 ##
  163. "model/keras_layer/StatefulPartitionedCall:0.479"  7.000000 ##
  164. "model/keras_layer/StatefulPartitionedCall:0.484"  7.000000 ##
  165. "model/keras_layer/StatefulPartitionedCall:0.486"  7.000000 ##
  166.  "model/keras_layer/StatefulPartitionedCall:0.74"  7.000000 ##
  167.  "model/keras_layer/StatefulPartitionedCall:0.90"  7.000000 ##
  168.  "model/keras_layer/StatefulPartitionedCall:0.96"  7.000000 ##
  169. "model/keras_layer/StatefulPartitionedCall:0.106"  6.000000 ##
  170. "model/keras_layer/StatefulPartitionedCall:0.110"  6.000000 ##
  171. "model/keras_layer/StatefulPartitionedCall:0.124"  6.000000 ##
  172. "model/keras_layer/StatefulPartitionedCall:0.129"  6.000000 ##
  173. "model/keras_layer/StatefulPartitionedCall:0.138"  6.000000 ##
  174. "model/keras_layer/StatefulPartitionedCall:0.146"  6.000000 ##
  175.  "model/keras_layer/StatefulPartitionedCall:0.16"  6.000000 ##
  176. "model/keras_layer/StatefulPartitionedCall:0.160"  6.000000 ##
  177. "model/keras_layer/StatefulPartitionedCall:0.218"  6.000000 ##
  178. "model/keras_layer/StatefulPartitionedCall:0.226"  6.000000 ##
  179. "model/keras_layer/StatefulPartitionedCall:0.265"  6.000000 ##
  180. "model/keras_layer/StatefulPartitionedCall:0.272"  6.000000 ##
  181. "model/keras_layer/StatefulPartitionedCall:0.274"  6.000000 ##
  182. "model/keras_layer/StatefulPartitionedCall:0.276"  6.000000 ##
  183. "model/keras_layer/StatefulPartitionedCall:0.282"  6.000000 ##
  184. "model/keras_layer/StatefulPartitionedCall:0.292"  6.000000 ##
  185. "model/keras_layer/StatefulPartitionedCall:0.293"  6.000000 ##
  186. "model/keras_layer/StatefulPartitionedCall:0.335"  6.000000 ##
  187. "model/keras_layer/StatefulPartitionedCall:0.341"  6.000000 ##
  188. "model/keras_layer/StatefulPartitionedCall:0.344"  6.000000 ##
  189. "model/keras_layer/StatefulPartitionedCall:0.352"  6.000000 ##
  190. "model/keras_layer/StatefulPartitionedCall:0.402"  6.000000 ##
  191. "model/keras_layer/StatefulPartitionedCall:0.405"  6.000000 ##
  192. "model/keras_layer/StatefulPartitionedCall:0.413"  6.000000 ##
  193. "model/keras_layer/StatefulPartitionedCall:0.416"  6.000000 ##
  194. "model/keras_layer/StatefulPartitionedCall:0.420"  6.000000 ##
  195. "model/keras_layer/StatefulPartitionedCall:0.432"  6.000000 ##
  196. "model/keras_layer/StatefulPartitionedCall:0.433"  6.000000 ##
  197. "model/keras_layer/StatefulPartitionedCall:0.434"  6.000000 ##
  198. "model/keras_layer/StatefulPartitionedCall:0.444"  6.000000 ##
  199. "model/keras_layer/StatefulPartitionedCall:0.455"  6.000000 ##
  200. "model/keras_layer/StatefulPartitionedCall:0.456"  6.000000 ##
  201. "model/keras_layer/StatefulPartitionedCall:0.468"  6.000000 ##
  202.  "model/keras_layer/StatefulPartitionedCall:0.48"  6.000000 ##
  203. "model/keras_layer/StatefulPartitionedCall:0.480"  6.000000 ##
  204. "model/keras_layer/StatefulPartitionedCall:0.483"  6.000000 ##
  205. "model/keras_layer/StatefulPartitionedCall:0.497"  6.000000 ##
  206. "model/keras_layer/StatefulPartitionedCall:0.504"  6.000000 ##
  207.  "model/keras_layer/StatefulPartitionedCall:0.56"  6.000000 ##
  208.  "model/keras_layer/StatefulPartitionedCall:0.66"  6.000000 ##
  209.  "model/keras_layer/StatefulPartitionedCall:0.85"  6.000000 ##
  210.   "model/keras_layer/StatefulPartitionedCall:0.9"  6.000000 ##
  211.  "model/keras_layer/StatefulPartitionedCall:0.92"  6.000000 ##
  212.  "model/keras_layer/StatefulPartitionedCall:0.98"  6.000000 ##
  213. "model/keras_layer/StatefulPartitionedCall:0.101"  5.000000 #
  214. "model/keras_layer/StatefulPartitionedCall:0.103"  5.000000 #
  215. "model/keras_layer/StatefulPartitionedCall:0.108"  5.000000 #
  216. "model/keras_layer/StatefulPartitionedCall:0.123"  5.000000 #
  217. "model/keras_layer/StatefulPartitionedCall:0.130"  5.000000 #
  218. "model/keras_layer/StatefulPartitionedCall:0.136"  5.000000 #
  219. "model/keras_layer/StatefulPartitionedCall:0.147"  5.000000 #
  220. "model/keras_layer/StatefulPartitionedCall:0.158"  5.000000 #
  221. "model/keras_layer/StatefulPartitionedCall:0.164"  5.000000 #
  222. "model/keras_layer/StatefulPartitionedCall:0.172"  5.000000 #
  223. "model/keras_layer/StatefulPartitionedCall:0.174"  5.000000 #
  224. "model/keras_layer/StatefulPartitionedCall:0.176"  5.000000 #
  225. "model/keras_layer/StatefulPartitionedCall:0.185"  5.000000 #
  226. "model/keras_layer/StatefulPartitionedCall:0.189"  5.000000 #
  227. "model/keras_layer/StatefulPartitionedCall:0.190"  5.000000 #
  228. "model/keras_layer/StatefulPartitionedCall:0.191"  5.000000 #
  229. "model/keras_layer/StatefulPartitionedCall:0.197"  5.000000 #
  230.   "model/keras_layer/StatefulPartitionedCall:0.2"  5.000000 #
  231. "model/keras_layer/StatefulPartitionedCall:0.202"  5.000000 #
  232. "model/keras_layer/StatefulPartitionedCall:0.222"  5.000000 #
  233. "model/keras_layer/StatefulPartitionedCall:0.223"  5.000000 #
  234. "model/keras_layer/StatefulPartitionedCall:0.233"  5.000000 #
  235. "model/keras_layer/StatefulPartitionedCall:0.237"  5.000000 #
  236. "model/keras_layer/StatefulPartitionedCall:0.245"  5.000000 #
  237.  "model/keras_layer/StatefulPartitionedCall:0.25"  5.000000 #
  238. "model/keras_layer/StatefulPartitionedCall:0.251"  5.000000 #
  239. "model/keras_layer/StatefulPartitionedCall:0.254"  5.000000 #
  240. "model/keras_layer/StatefulPartitionedCall:0.257"  5.000000 #
  241. "model/keras_layer/StatefulPartitionedCall:0.258"  5.000000 #
  242. "model/keras_layer/StatefulPartitionedCall:0.259"  5.000000 #
  243. "model/keras_layer/StatefulPartitionedCall:0.262"  5.000000 #
  244. "model/keras_layer/StatefulPartitionedCall:0.267"  5.000000 #
  245. "model/keras_layer/StatefulPartitionedCall:0.269"  5.000000 #
  246. "model/keras_layer/StatefulPartitionedCall:0.299"  5.000000 #
  247. "model/keras_layer/StatefulPartitionedCall:0.302"  5.000000 #
  248. "model/keras_layer/StatefulPartitionedCall:0.307"  5.000000 #
  249.  "model/keras_layer/StatefulPartitionedCall:0.31"  5.000000 #
  250. "model/keras_layer/StatefulPartitionedCall:0.314"  5.000000 #
  251. "model/keras_layer/StatefulPartitionedCall:0.320"  5.000000 #
  252. "model/keras_layer/StatefulPartitionedCall:0.324"  5.000000 #
  253.  "model/keras_layer/StatefulPartitionedCall:0.33"  5.000000 #
  254. "model/keras_layer/StatefulPartitionedCall:0.331"  5.000000 #
  255. "model/keras_layer/StatefulPartitionedCall:0.334"  5.000000 #
  256. "model/keras_layer/StatefulPartitionedCall:0.336"  5.000000 #
  257. "model/keras_layer/StatefulPartitionedCall:0.345"  5.000000 #
  258. "model/keras_layer/StatefulPartitionedCall:0.348"  5.000000 #
  259. "model/keras_layer/StatefulPartitionedCall:0.367"  5.000000 #
  260. "model/keras_layer/StatefulPartitionedCall:0.372"  5.000000 #
  261. "model/keras_layer/StatefulPartitionedCall:0.373"  5.000000 #
  262. "model/keras_layer/StatefulPartitionedCall:0.377"  5.000000 #
  263. "model/keras_layer/StatefulPartitionedCall:0.380"  5.000000 #
  264. "model/keras_layer/StatefulPartitionedCall:0.383"  5.000000 #
  265. "model/keras_layer/StatefulPartitionedCall:0.384"  5.000000 #
  266. "model/keras_layer/StatefulPartitionedCall:0.386"  5.000000 #
  267. "model/keras_layer/StatefulPartitionedCall:0.412"  5.000000 #
  268. "model/keras_layer/StatefulPartitionedCall:0.414"  5.000000 #
  269.  "model/keras_layer/StatefulPartitionedCall:0.42"  5.000000 #
  270. "model/keras_layer/StatefulPartitionedCall:0.423"  5.000000 #
  271. "model/keras_layer/StatefulPartitionedCall:0.426"  5.000000 #
  272. "model/keras_layer/StatefulPartitionedCall:0.428"  5.000000 #
  273.  "model/keras_layer/StatefulPartitionedCall:0.43"  5.000000 #
  274. "model/keras_layer/StatefulPartitionedCall:0.437"  5.000000 #
  275. "model/keras_layer/StatefulPartitionedCall:0.442"  5.000000 #
  276. "model/keras_layer/StatefulPartitionedCall:0.443"  5.000000 #
  277.  "model/keras_layer/StatefulPartitionedCall:0.45"  5.000000 #
  278. "model/keras_layer/StatefulPartitionedCall:0.450"  5.000000 #
  279. "model/keras_layer/StatefulPartitionedCall:0.461"  5.000000 #
  280. "model/keras_layer/StatefulPartitionedCall:0.494"  5.000000 #
  281. "model/keras_layer/StatefulPartitionedCall:0.510"  5.000000 #
  282.  "model/keras_layer/StatefulPartitionedCall:0.54"  5.000000 #
  283.   "model/keras_layer/StatefulPartitionedCall:0.8"  5.000000 #
  284.  "model/keras_layer/StatefulPartitionedCall:0.84"  5.000000 #
  285.  "model/keras_layer/StatefulPartitionedCall:0.99"  5.000000 #
  286. "model/keras_layer/StatefulPartitionedCall:0.100"  4.000000 #
  287. "model/keras_layer/StatefulPartitionedCall:0.105"  4.000000 #
  288. "model/keras_layer/StatefulPartitionedCall:0.111"  4.000000 #
  289. "model/keras_layer/StatefulPartitionedCall:0.116"  4.000000 #
  290. "model/keras_layer/StatefulPartitionedCall:0.128"  4.000000 #
  291. "model/keras_layer/StatefulPartitionedCall:0.131"  4.000000 #
  292. "model/keras_layer/StatefulPartitionedCall:0.132"  4.000000 #
  293. "model/keras_layer/StatefulPartitionedCall:0.141"  4.000000 #
  294.  "model/keras_layer/StatefulPartitionedCall:0.15"  4.000000 #
  295. "model/keras_layer/StatefulPartitionedCall:0.173"  4.000000 #
  296. "model/keras_layer/StatefulPartitionedCall:0.186"  4.000000 #
  297. "model/keras_layer/StatefulPartitionedCall:0.201"  4.000000 #
  298. "model/keras_layer/StatefulPartitionedCall:0.206"  4.000000 #
  299. "model/keras_layer/StatefulPartitionedCall:0.210"  4.000000 #
  300. "model/keras_layer/StatefulPartitionedCall:0.215"  4.000000 #
  301.  "model/keras_layer/StatefulPartitionedCall:0.22"  4.000000 #
  302. "model/keras_layer/StatefulPartitionedCall:0.227"  4.000000 #
  303. "model/keras_layer/StatefulPartitionedCall:0.234"  4.000000 #
  304. "model/keras_layer/StatefulPartitionedCall:0.253"  4.000000 #
  305. "model/keras_layer/StatefulPartitionedCall:0.256"  4.000000 #
  306. "model/keras_layer/StatefulPartitionedCall:0.268"  4.000000 #
  307. "model/keras_layer/StatefulPartitionedCall:0.270"  4.000000 #
  308. "model/keras_layer/StatefulPartitionedCall:0.288"  4.000000 #
  309. "model/keras_layer/StatefulPartitionedCall:0.295"  4.000000 #
  310. "model/keras_layer/StatefulPartitionedCall:0.304"  4.000000 #
  311. "model/keras_layer/StatefulPartitionedCall:0.306"  4.000000 #
  312. "model/keras_layer/StatefulPartitionedCall:0.308"  4.000000 #
  313. "model/keras_layer/StatefulPartitionedCall:0.311"  4.000000 #
  314. "model/keras_layer/StatefulPartitionedCall:0.321"  4.000000 #
  315. "model/keras_layer/StatefulPartitionedCall:0.333"  4.000000 #
  316. "model/keras_layer/StatefulPartitionedCall:0.339"  4.000000 #
  317. "model/keras_layer/StatefulPartitionedCall:0.346"  4.000000 #
  318. "model/keras_layer/StatefulPartitionedCall:0.347"  4.000000 #
  319. "model/keras_layer/StatefulPartitionedCall:0.359"  4.000000 #
  320. "model/keras_layer/StatefulPartitionedCall:0.361"  4.000000 #
  321. "model/keras_layer/StatefulPartitionedCall:0.365"  4.000000 #
  322. "model/keras_layer/StatefulPartitionedCall:0.369"  4.000000 #
  323. "model/keras_layer/StatefulPartitionedCall:0.371"  4.000000 #
  324. "model/keras_layer/StatefulPartitionedCall:0.378"  4.000000 #
  325.  "model/keras_layer/StatefulPartitionedCall:0.39"  4.000000 #
  326. "model/keras_layer/StatefulPartitionedCall:0.401"  4.000000 #
  327. "model/keras_layer/StatefulPartitionedCall:0.407"  4.000000 #
  328. "model/keras_layer/StatefulPartitionedCall:0.415"  4.000000 #
  329. "model/keras_layer/StatefulPartitionedCall:0.417"  4.000000 #
  330. "model/keras_layer/StatefulPartitionedCall:0.422"  4.000000 #
  331. "model/keras_layer/StatefulPartitionedCall:0.425"  4.000000 #
  332. "model/keras_layer/StatefulPartitionedCall:0.429"  4.000000 #
  333. "model/keras_layer/StatefulPartitionedCall:0.447"  4.000000 #
  334. "model/keras_layer/StatefulPartitionedCall:0.448"  4.000000 #
  335. "model/keras_layer/StatefulPartitionedCall:0.459"  4.000000 #
  336. "model/keras_layer/StatefulPartitionedCall:0.462"  4.000000 #
  337. "model/keras_layer/StatefulPartitionedCall:0.470"  4.000000 #
  338. "model/keras_layer/StatefulPartitionedCall:0.477"  4.000000 #
  339. "model/keras_layer/StatefulPartitionedCall:0.481"  4.000000 #
  340. "model/keras_layer/StatefulPartitionedCall:0.501"  4.000000 #
  341. "model/keras_layer/StatefulPartitionedCall:0.503"  4.000000 #
  342. "model/keras_layer/StatefulPartitionedCall:0.506"  4.000000 #
  343. "model/keras_layer/StatefulPartitionedCall:0.507"  4.000000 #
  344.  "model/keras_layer/StatefulPartitionedCall:0.57"  4.000000 #
  345.  "model/keras_layer/StatefulPartitionedCall:0.62"  4.000000 #
  346.  "model/keras_layer/StatefulPartitionedCall:0.67"  4.000000 #
  347.  "model/keras_layer/StatefulPartitionedCall:0.68"  4.000000 #
  348.  "model/keras_layer/StatefulPartitionedCall:0.71"  4.000000 #
  349.  "model/keras_layer/StatefulPartitionedCall:0.82"  4.000000 #
  350.  "model/keras_layer/StatefulPartitionedCall:0.86"  4.000000 #
  351. "model/keras_layer/StatefulPartitionedCall:0.107"  3.000000 
  352. "model/keras_layer/StatefulPartitionedCall:0.109"  3.000000 
  353. "model/keras_layer/StatefulPartitionedCall:0.125"  3.000000 
  354. "model/keras_layer/StatefulPartitionedCall:0.151"  3.000000 
  355. "model/keras_layer/StatefulPartitionedCall:0.155"  3.000000 
  356. "model/keras_layer/StatefulPartitionedCall:0.162"  3.000000 
  357. "model/keras_layer/StatefulPartitionedCall:0.167"  3.000000 
  358. "model/keras_layer/StatefulPartitionedCall:0.179"  3.000000 
  359. "model/keras_layer/StatefulPartitionedCall:0.198"  3.000000 
  360. "model/keras_layer/StatefulPartitionedCall:0.200"  3.000000 
  361. "model/keras_layer/StatefulPartitionedCall:0.205"  3.000000 
  362. "model/keras_layer/StatefulPartitionedCall:0.211"  3.000000 
  363. "model/keras_layer/StatefulPartitionedCall:0.220"  3.000000 
  364. "model/keras_layer/StatefulPartitionedCall:0.224"  3.000000 
  365. "model/keras_layer/StatefulPartitionedCall:0.230"  3.000000 
  366. "model/keras_layer/StatefulPartitionedCall:0.238"  3.000000 
  367. "model/keras_layer/StatefulPartitionedCall:0.239"  3.000000 
  368.  "model/keras_layer/StatefulPartitionedCall:0.24"  3.000000 
  369. "model/keras_layer/StatefulPartitionedCall:0.242"  3.000000 
  370. "model/keras_layer/StatefulPartitionedCall:0.248"  3.000000 
  371. "model/keras_layer/StatefulPartitionedCall:0.263"  3.000000 
  372. "model/keras_layer/StatefulPartitionedCall:0.271"  3.000000 
  373. "model/keras_layer/StatefulPartitionedCall:0.283"  3.000000 
  374.   "model/keras_layer/StatefulPartitionedCall:0.3"  3.000000 
  375. "model/keras_layer/StatefulPartitionedCall:0.301"  3.000000 
  376. "model/keras_layer/StatefulPartitionedCall:0.313"  3.000000 
  377. "model/keras_layer/StatefulPartitionedCall:0.318"  3.000000 
  378. "model/keras_layer/StatefulPartitionedCall:0.319"  3.000000 
  379. "model/keras_layer/StatefulPartitionedCall:0.326"  3.000000 
  380.  "model/keras_layer/StatefulPartitionedCall:0.34"  3.000000 
  381. "model/keras_layer/StatefulPartitionedCall:0.340"  3.000000 
  382. "model/keras_layer/StatefulPartitionedCall:0.342"  3.000000 
  383. "model/keras_layer/StatefulPartitionedCall:0.350"  3.000000 
  384. "model/keras_layer/StatefulPartitionedCall:0.353"  3.000000 
  385.  "model/keras_layer/StatefulPartitionedCall:0.36"  3.000000 
  386. "model/keras_layer/StatefulPartitionedCall:0.370"  3.000000 
  387. "model/keras_layer/StatefulPartitionedCall:0.376"  3.000000 
  388. "model/keras_layer/StatefulPartitionedCall:0.382"  3.000000 
  389. "model/keras_layer/StatefulPartitionedCall:0.385"  3.000000 
  390. "model/keras_layer/StatefulPartitionedCall:0.389"  3.000000 
  391. "model/keras_layer/StatefulPartitionedCall:0.390"  3.000000 
  392. "model/keras_layer/StatefulPartitionedCall:0.391"  3.000000 
  393. "model/keras_layer/StatefulPartitionedCall:0.394"  3.000000 
  394. "model/keras_layer/StatefulPartitionedCall:0.400"  3.000000 
  395. "model/keras_layer/StatefulPartitionedCall:0.403"  3.000000 
  396. "model/keras_layer/StatefulPartitionedCall:0.409"  3.000000 
  397. "model/keras_layer/StatefulPartitionedCall:0.410"  3.000000 
  398. "model/keras_layer/StatefulPartitionedCall:0.421"  3.000000 
  399. "model/keras_layer/StatefulPartitionedCall:0.431"  3.000000 
  400. "model/keras_layer/StatefulPartitionedCall:0.439"  3.000000 
  401. "model/keras_layer/StatefulPartitionedCall:0.475"  3.000000 
  402. "model/keras_layer/StatefulPartitionedCall:0.487"  3.000000 
  403. "model/keras_layer/StatefulPartitionedCall:0.488"  3.000000 
  404. "model/keras_layer/StatefulPartitionedCall:0.492"  3.000000 
  405. "model/keras_layer/StatefulPartitionedCall:0.493"  3.000000 
  406. "model/keras_layer/StatefulPartitionedCall:0.495"  3.000000 
  407. "model/keras_layer/StatefulPartitionedCall:0.498"  3.000000 
  408.   "model/keras_layer/StatefulPartitionedCall:0.5"  3.000000 
  409. "model/keras_layer/StatefulPartitionedCall:0.509"  3.000000 
  410.  "model/keras_layer/StatefulPartitionedCall:0.60"  3.000000 
  411.  "model/keras_layer/StatefulPartitionedCall:0.63"  3.000000 
  412.  "model/keras_layer/StatefulPartitionedCall:0.64"  3.000000 
  413.   "model/keras_layer/StatefulPartitionedCall:0.7"  3.000000 
  414.  "model/keras_layer/StatefulPartitionedCall:0.70"  3.000000 
  415.  "model/keras_layer/StatefulPartitionedCall:0.81"  3.000000 
  416.  "model/keras_layer/StatefulPartitionedCall:0.94"  3.000000 
  417.  "model/keras_layer/StatefulPartitionedCall:0.97"  3.000000 
  418.  "model/keras_layer/StatefulPartitionedCall:0.12"  2.000000 
  419. "model/keras_layer/StatefulPartitionedCall:0.122"  2.000000 
  420. "model/keras_layer/StatefulPartitionedCall:0.134"  2.000000 
  421. "model/keras_layer/StatefulPartitionedCall:0.137"  2.000000 
  422. "model/keras_layer/StatefulPartitionedCall:0.143"  2.000000 
  423. "model/keras_layer/StatefulPartitionedCall:0.156"  2.000000 
  424. "model/keras_layer/StatefulPartitionedCall:0.177"  2.000000 
  425.  "model/keras_layer/StatefulPartitionedCall:0.19"  2.000000 
  426. "model/keras_layer/StatefulPartitionedCall:0.216"  2.000000 
  427. "model/keras_layer/StatefulPartitionedCall:0.244"  2.000000 
  428. "model/keras_layer/StatefulPartitionedCall:0.246"  2.000000 
  429.  "model/keras_layer/StatefulPartitionedCall:0.26"  2.000000 
  430. "model/keras_layer/StatefulPartitionedCall:0.264"  2.000000 
  431.  "model/keras_layer/StatefulPartitionedCall:0.27"  2.000000 
  432. "model/keras_layer/StatefulPartitionedCall:0.279"  2.000000 
  433.  "model/keras_layer/StatefulPartitionedCall:0.30"  2.000000 
  434. "model/keras_layer/StatefulPartitionedCall:0.303"  2.000000 
  435. "model/keras_layer/StatefulPartitionedCall:0.305"  2.000000 
  436. "model/keras_layer/StatefulPartitionedCall:0.309"  2.000000 
  437. "model/keras_layer/StatefulPartitionedCall:0.329"  2.000000 
  438. "model/keras_layer/StatefulPartitionedCall:0.349"  2.000000 
  439. "model/keras_layer/StatefulPartitionedCall:0.355"  2.000000 
  440. "model/keras_layer/StatefulPartitionedCall:0.363"  2.000000 
  441. "model/keras_layer/StatefulPartitionedCall:0.374"  2.000000 
  442. "model/keras_layer/StatefulPartitionedCall:0.379"  2.000000 
  443.  "model/keras_layer/StatefulPartitionedCall:0.38"  2.000000 
  444. "model/keras_layer/StatefulPartitionedCall:0.388"  2.000000 
  445. "model/keras_layer/StatefulPartitionedCall:0.395"  2.000000 
  446. "model/keras_layer/StatefulPartitionedCall:0.396"  2.000000 
  447.  "model/keras_layer/StatefulPartitionedCall:0.40"  2.000000 
  448. "model/keras_layer/StatefulPartitionedCall:0.404"  2.000000 
  449. "model/keras_layer/StatefulPartitionedCall:0.406"  2.000000 
  450. "model/keras_layer/StatefulPartitionedCall:0.430"  2.000000 
  451. "model/keras_layer/StatefulPartitionedCall:0.436"  2.000000 
  452. "model/keras_layer/StatefulPartitionedCall:0.440"  2.000000 
  453. "model/keras_layer/StatefulPartitionedCall:0.441"  2.000000 
  454. "model/keras_layer/StatefulPartitionedCall:0.452"  2.000000 
  455. "model/keras_layer/StatefulPartitionedCall:0.465"  2.000000 
  456. "model/keras_layer/StatefulPartitionedCall:0.478"  2.000000 
  457.  "model/keras_layer/StatefulPartitionedCall:0.49"  2.000000 
  458. "model/keras_layer/StatefulPartitionedCall:0.491"  2.000000 
  459. "model/keras_layer/StatefulPartitionedCall:0.496"  2.000000 
  460. "model/keras_layer/StatefulPartitionedCall:0.505"  2.000000 
  461.  "model/keras_layer/StatefulPartitionedCall:0.52"  2.000000 
  462.  "model/keras_layer/StatefulPartitionedCall:0.53"  2.000000 
  463.  "model/keras_layer/StatefulPartitionedCall:0.59"  2.000000 
  464.  "model/keras_layer/StatefulPartitionedCall:0.61"  2.000000 
  465.  "model/keras_layer/StatefulPartitionedCall:0.75"  2.000000 
  466.  "model/keras_layer/StatefulPartitionedCall:0.77"  2.000000 
  467.  "model/keras_layer/StatefulPartitionedCall:0.79"  2.000000 
  468.  "model/keras_layer/StatefulPartitionedCall:0.91"  2.000000 
  469. "model/keras_layer/StatefulPartitionedCall:0.112"  1.000000 
  470. "model/keras_layer/StatefulPartitionedCall:0.121"  1.000000 
  471.  "model/keras_layer/StatefulPartitionedCall:0.13"  1.000000 
  472. "model/keras_layer/StatefulPartitionedCall:0.139"  1.000000 
  473. "model/keras_layer/StatefulPartitionedCall:0.157"  1.000000 
  474. "model/keras_layer/StatefulPartitionedCall:0.163"  1.000000 
  475. "model/keras_layer/StatefulPartitionedCall:0.181"  1.000000 
  476. "model/keras_layer/StatefulPartitionedCall:0.182"  1.000000 
  477. "model/keras_layer/StatefulPartitionedCall:0.195"  1.000000 
  478.  "model/keras_layer/StatefulPartitionedCall:0.20"  1.000000 
  479. "model/keras_layer/StatefulPartitionedCall:0.217"  1.000000 
  480. "model/keras_layer/StatefulPartitionedCall:0.228"  1.000000 
  481.  "model/keras_layer/StatefulPartitionedCall:0.23"  1.000000 
  482. "model/keras_layer/StatefulPartitionedCall:0.236"  1.000000 
  483. "model/keras_layer/StatefulPartitionedCall:0.240"  1.000000 
  484. "model/keras_layer/StatefulPartitionedCall:0.296"  1.000000 
  485. "model/keras_layer/StatefulPartitionedCall:0.312"  1.000000 
  486. "model/keras_layer/StatefulPartitionedCall:0.316"  1.000000 
  487. "model/keras_layer/StatefulPartitionedCall:0.366"  1.000000 
  488. "model/keras_layer/StatefulPartitionedCall:0.408"  1.000000 
  489. "model/keras_layer/StatefulPartitionedCall:0.424"  1.000000 
  490. "model/keras_layer/StatefulPartitionedCall:0.435"  1.000000 
  491. "model/keras_layer/StatefulPartitionedCall:0.438"  1.000000 
  492. "model/keras_layer/StatefulPartitionedCall:0.466"  1.000000 
  493. "model/keras_layer/StatefulPartitionedCall:0.467"  1.000000 
  494. "model/keras_layer/StatefulPartitionedCall:0.474"  1.000000 
  495. "model/keras_layer/StatefulPartitionedCall:0.482"  1.000000 
  496. "model/keras_layer/StatefulPartitionedCall:0.490"  1.000000 
  497. "model/keras_layer/StatefulPartitionedCall:0.499"  1.000000 
  498.   "model/keras_layer/StatefulPartitionedCall:0.6"  1.000000 
  499.  "model/keras_layer/StatefulPartitionedCall:0.72"  1.000000 
  500.  "model/keras_layer/StatefulPartitionedCall:0.93"  1.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: SUM_SCORE:
    1.  "model/keras_layer/StatefulPartitionedCall:0.50" 959.389904 ################
    2. "model/keras_layer/StatefulPartitionedCall:0.464" 393.296619 ######
    3. "model/keras_layer/StatefulPartitionedCall:0.166" 335.727634 #####
    4. "model/keras_layer/StatefulPartitionedCall:0.458" 297.187343 ####
    5. "model/keras_layer/StatefulPartitionedCall:0.126" 215.472971 ###
    6. "model/keras_layer/StatefulPartitionedCall:0.463" 167.443032 ##
    7. "model/keras_layer/StatefulPartitionedCall:0.188" 162.475388 ##
    8. "model/keras_layer/StatefulPartitionedCall:0.127" 152.567734 ##
    9. "model/keras_layer/StatefulPartitionedCall:0.356" 141.886091 ##
   10. "model/keras_layer/StatefulPartitionedCall:0.159" 132.817127 ##
   11. "model/keras_layer/StatefulPartitionedCall:0.294" 116.038681 #
   12.  "model/keras_layer/StatefulPartitionedCall:0.44" 110.361116 #
   13. "model/keras_layer/StatefulPartitionedCall:0.142" 107.375283 #
   14. "model/keras_layer/StatefulPartitionedCall:0.323" 81.288890 #
   15.  "model/keras_layer/StatefulPartitionedCall:0.46" 76.387143 #
   16. "model/keras_layer/StatefulPartitionedCall:0.281" 66.603516 #
   17. "model/keras_layer/StatefulPartitionedCall:0.152" 66.142325 #
   18. "model/keras_layer/StatefulPartitionedCall:0.354" 63.202183 #
   19. "model/keras_layer/StatefulPartitionedCall:0.362" 62.305059 #
   20. "model/keras_layer/StatefulPartitionedCall:0.219" 59.790261 
   21. "model/keras_layer/StatefulPartitionedCall:0.180" 58.247166 
   22. "model/keras_layer/StatefulPartitionedCall:0.250" 56.606922 
   23. "model/keras_layer/StatefulPartitionedCall:0.343" 48.859654 
   24. "model/keras_layer/StatefulPartitionedCall:0.171" 45.566105 
   25. "model/keras_layer/StatefulPartitionedCall:0.249" 44.760371 
   26. "model/keras_layer/StatefulPartitionedCall:0.399" 44.523333 
   27. "model/keras_layer/StatefulPartitionedCall:0.144" 42.974708 
   28. "model/keras_layer/StatefulPartitionedCall:0.315" 42.700591 
   29. "model/keras_layer/StatefulPartitionedCall:0.178" 42.217015 
   30. "model/keras_layer/StatefulPartitionedCall:0.153" 41.948426 
   31.  "model/keras_layer/StatefulPartitionedCall:0.10" 39.234599 
   32.  "model/keras_layer/StatefulPartitionedCall:0.88" 38.970713 
   33. "model/keras_layer/StatefulPartitionedCall:0.289" 38.678238 
   34.  "model/keras_layer/StatefulPartitionedCall:0.56" 36.595683 
   35. "model/keras_layer/StatefulPartitionedCall:0.133" 35.663547 
   36.  "model/keras_layer/StatefulPartitionedCall:0.80" 35.623764 
   37. "model/keras_layer/StatefulPartitionedCall:0.511" 31.527654 
   38. "model/keras_layer/StatefulPartitionedCall:0.168" 31.033030 
   39. "model/keras_layer/StatefulPartitionedCall:0.247" 30.508206 
   40. "model/keras_layer/StatefulPartitionedCall:0.427" 29.847008 
   41.   "model/keras_layer/StatefulPartitionedCall:0.1" 28.124667 
   42. "model/keras_layer/StatefulPartitionedCall:0.118" 25.724113 
   43. "model/keras_layer/StatefulPartitionedCall:0.286" 24.821231 
   44. "model/keras_layer/StatefulPartitionedCall:0.327" 24.746541 
   45. "model/keras_layer/StatefulPartitionedCall:0.310" 24.441551 
   46. "model/keras_layer/StatefulPartitionedCall:0.135" 24.147893 
   47. "model/keras_layer/StatefulPartitionedCall:0.341" 23.578999 
   48. "model/keras_layer/StatefulPartitionedCall:0.290" 23.028560 
   49. "model/keras_layer/StatefulPartitionedCall:0.275" 22.297273 
   50. "model/keras_layer/StatefulPartitionedCall:0.325" 22.247234 
   51. "model/keras_layer/StatefulPartitionedCall:0.337" 21.944532 
   52. "model/keras_layer/StatefulPartitionedCall:0.261" 21.583323 
   53. "model/keras_layer/StatefulPartitionedCall:0.291" 21.286374 
   54. "model/keras_layer/StatefulPartitionedCall:0.285" 21.139952 
   55.  "model/keras_layer/StatefulPartitionedCall:0.32" 21.006744 
   56. "model/keras_layer/StatefulPartitionedCall:0.457" 20.341704 
   57. "model/keras_layer/StatefulPartitionedCall:0.469" 20.135403 
   58.  "model/keras_layer/StatefulPartitionedCall:0.37" 20.053329 
   59. "model/keras_layer/StatefulPartitionedCall:0.120" 19.991364 
   60. "model/keras_layer/StatefulPartitionedCall:0.175" 19.905914 
   61.  "model/keras_layer/StatefulPartitionedCall:0.18" 19.430191 
   62.  "model/keras_layer/StatefulPartitionedCall:0.87" 19.030491 
   63.  "model/keras_layer/StatefulPartitionedCall:0.58" 18.954950 
   64. "model/keras_layer/StatefulPartitionedCall:0.485" 18.725738 
   65.   "model/keras_layer/StatefulPartitionedCall:0.0" 18.376702 
   66. "model/keras_layer/StatefulPartitionedCall:0.419" 18.022019 
   67. "model/keras_layer/StatefulPartitionedCall:0.360" 18.007008 
   68. "model/keras_layer/StatefulPartitionedCall:0.252" 17.929390 
   69.  "model/keras_layer/StatefulPartitionedCall:0.51" 17.327463 
   70. "model/keras_layer/StatefulPartitionedCall:0.449" 17.312243 
   71. "model/keras_layer/StatefulPartitionedCall:0.193" 16.987797 
   72. "model/keras_layer/StatefulPartitionedCall:0.500" 16.704093 
   73. "model/keras_layer/StatefulPartitionedCall:0.288" 16.636890 
   74. "model/keras_layer/StatefulPartitionedCall:0.368" 16.527698 
   75. "model/keras_layer/StatefulPartitionedCall:0.235" 16.448028 
   76. "model/keras_layer/StatefulPartitionedCall:0.300" 16.408046 
   77.  "model/keras_layer/StatefulPartitionedCall:0.73" 16.382455 
   78. "model/keras_layer/StatefulPartitionedCall:0.150" 16.138034 
   79. "model/keras_layer/StatefulPartitionedCall:0.332" 16.129575 
   80. "model/keras_layer/StatefulPartitionedCall:0.110" 15.846866 
   81. "model/keras_layer/StatefulPartitionedCall:0.451" 15.824734 
   82.  "model/keras_layer/StatefulPartitionedCall:0.89" 15.770294 
   83. "model/keras_layer/StatefulPartitionedCall:0.272" 15.682898 
   84. "model/keras_layer/StatefulPartitionedCall:0.104" 15.526674 
   85. "model/keras_layer/StatefulPartitionedCall:0.149" 15.465243 
   86.  "model/keras_layer/StatefulPartitionedCall:0.11" 15.064285 
   87. "model/keras_layer/StatefulPartitionedCall:0.140" 14.993470 
   88. "model/keras_layer/StatefulPartitionedCall:0.473" 14.618271 
   89. "model/keras_layer/StatefulPartitionedCall:0.113" 14.507725 
   90. "model/keras_layer/StatefulPartitionedCall:0.460" 14.331183 
   91. "model/keras_layer/StatefulPartitionedCall:0.454" 14.283899 
   92.  "model/keras_layer/StatefulPartitionedCall:0.41" 14.193149 
   93. "model/keras_layer/StatefulPartitionedCall:0.278" 14.138157 
   94. "model/keras_layer/StatefulPartitionedCall:0.322" 14.117497 
   95. "model/keras_layer/StatefulPartitionedCall:0.489" 13.816848 
   96. "model/keras_layer/StatefulPartitionedCall:0.226" 13.724339 
   97. "model/keras_layer/StatefulPartitionedCall:0.317" 13.308935 
   98. "model/keras_layer/StatefulPartitionedCall:0.192" 13.159820 
   99. "model/keras_layer/StatefulPartitionedCall:0.472" 13.018894 
  100. "model/keras_layer/StatefulPartitionedCall:0.351" 12.883126 
  101. "model/keras_layer/StatefulPartitionedCall:0.184" 12.741747 
  102. "model/keras_layer/StatefulPartitionedCall:0.297" 12.611756 
  103.  "model/keras_layer/StatefulPartitionedCall:0.96" 12.548598 
  104. "model/keras_layer/StatefulPartitionedCall:0.214" 12.181298 
  105. "model/keras_layer/StatefulPartitionedCall:0.169" 11.965796 
  106. "model/keras_layer/StatefulPartitionedCall:0.260" 11.950862 
  107. "model/keras_layer/StatefulPartitionedCall:0.207" 11.941023 
  108. "model/keras_layer/StatefulPartitionedCall:0.411" 11.917942 
  109. "model/keras_layer/StatefulPartitionedCall:0.338" 11.714042 
  110. "model/keras_layer/StatefulPartitionedCall:0.476" 11.585360 
  111.  "model/keras_layer/StatefulPartitionedCall:0.66" 11.582600 
  112.  "model/keras_layer/StatefulPartitionedCall:0.29" 11.578613 
  113. "model/keras_layer/StatefulPartitionedCall:0.437" 11.429894 
  114. "model/keras_layer/StatefulPartitionedCall:0.418" 11.422134 
  115.  "model/keras_layer/StatefulPartitionedCall:0.65" 11.373112 
  116. "model/keras_layer/StatefulPartitionedCall:0.194" 11.359884 
  117. "model/keras_layer/StatefulPartitionedCall:0.209" 11.034177 
  118. "model/keras_layer/StatefulPartitionedCall:0.203" 11.013233 
  119.  "model/keras_layer/StatefulPartitionedCall:0.99" 10.869757 
  120. "model/keras_layer/StatefulPartitionedCall:0.375" 10.666772 
  121. "model/keras_layer/StatefulPartitionedCall:0.161" 10.655896 
  122. "model/keras_layer/StatefulPartitionedCall:0.115" 10.608551 
  123. "model/keras_layer/StatefulPartitionedCall:0.373" 10.519848 
  124. "model/keras_layer/StatefulPartitionedCall:0.330" 10.433566 
  125. "model/keras_layer/StatefulPartitionedCall:0.486" 10.237372 
  126. "model/keras_layer/StatefulPartitionedCall:0.357" 10.196323 
  127. "model/keras_layer/StatefulPartitionedCall:0.287"  9.982982 
  128. "model/keras_layer/StatefulPartitionedCall:0.154"  9.779863 
  129. "model/keras_layer/StatefulPartitionedCall:0.320"  9.777763 
  130. "model/keras_layer/StatefulPartitionedCall:0.255"  9.690103 
  131. "model/keras_layer/StatefulPartitionedCall:0.397"  9.657859 
  132. "model/keras_layer/StatefulPartitionedCall:0.204"  9.648140 
  133. "model/keras_layer/StatefulPartitionedCall:0.114"  9.642827 
  134. "model/keras_layer/StatefulPartitionedCall:0.218"  9.493243 
  135. "model/keras_layer/StatefulPartitionedCall:0.432"  9.375304 
  136. "model/keras_layer/StatefulPartitionedCall:0.334"  9.310171 
  137. "model/keras_layer/StatefulPartitionedCall:0.229"  9.200314 
  138. "model/keras_layer/StatefulPartitionedCall:0.187"  9.195401 
  139. "model/keras_layer/StatefulPartitionedCall:0.392"  9.180533 
  140. "model/keras_layer/StatefulPartitionedCall:0.479"  9.160187 
  141.  "model/keras_layer/StatefulPartitionedCall:0.98"  9.153514 
  142. "model/keras_layer/StatefulPartitionedCall:0.471"  9.142840 
  143. "model/keras_layer/StatefulPartitionedCall:0.328"  9.054396 
  144. "model/keras_layer/StatefulPartitionedCall:0.146"  8.959918 
  145.  "model/keras_layer/StatefulPartitionedCall:0.35"  8.891348 
  146. "model/keras_layer/StatefulPartitionedCall:0.352"  8.680381 
  147. "model/keras_layer/StatefulPartitionedCall:0.221"  8.640680 
  148. "model/keras_layer/StatefulPartitionedCall:0.358"  8.604610 
  149.  "model/keras_layer/StatefulPartitionedCall:0.74"  8.596334 
  150. "model/keras_layer/StatefulPartitionedCall:0.308"  8.573031 
  151.  "model/keras_layer/StatefulPartitionedCall:0.28"  8.560739 
  152. "model/keras_layer/StatefulPartitionedCall:0.243"  8.510179 
  153. "model/keras_layer/StatefulPartitionedCall:0.117"  8.486306 
  154.  "model/keras_layer/StatefulPartitionedCall:0.14"  8.453653 
  155. "model/keras_layer/StatefulPartitionedCall:0.434"  8.408244 
  156.  "model/keras_layer/StatefulPartitionedCall:0.69"  8.290076 
  157. "model/keras_layer/StatefulPartitionedCall:0.453"  8.251279 
  158.  "model/keras_layer/StatefulPartitionedCall:0.47"  8.192982 
  159. "model/keras_layer/StatefulPartitionedCall:0.191"  8.132127 
  160. "model/keras_layer/StatefulPartitionedCall:0.462"  8.093112 
  161. "model/keras_layer/StatefulPartitionedCall:0.433"  7.998883 
  162. "model/keras_layer/StatefulPartitionedCall:0.160"  7.991137 
  163.  "model/keras_layer/StatefulPartitionedCall:0.78"  7.945626 
  164. "model/keras_layer/StatefulPartitionedCall:0.377"  7.843937 
  165. "model/keras_layer/StatefulPartitionedCall:0.497"  7.813645 
  166. "model/keras_layer/StatefulPartitionedCall:0.284"  7.792035 
  167. "model/keras_layer/StatefulPartitionedCall:0.353"  7.785796 
  168. "model/keras_layer/StatefulPartitionedCall:0.298"  7.755369 
  169. "model/keras_layer/StatefulPartitionedCall:0.131"  7.752735 
  170. "model/keras_layer/StatefulPartitionedCall:0.270"  7.655761 
  171. "model/keras_layer/StatefulPartitionedCall:0.165"  7.654142 
  172. "model/keras_layer/StatefulPartitionedCall:0.108"  7.601935 
  173. "model/keras_layer/StatefulPartitionedCall:0.347"  7.544927 
  174. "model/keras_layer/StatefulPartitionedCall:0.393"  7.475519 
  175. "model/keras_layer/StatefulPartitionedCall:0.302"  7.256439 
  176. "model/keras_layer/StatefulPartitionedCall:0.383"  7.240998 
  177. "model/keras_layer/StatefulPartitionedCall:0.237"  7.215865 
  178.  "model/keras_layer/StatefulPartitionedCall:0.85"  7.171661 
  179. "model/keras_layer/StatefulPartitionedCall:0.387"  7.163346 
  180.  "model/keras_layer/StatefulPartitionedCall:0.76"  7.161638 
  181. "model/keras_layer/StatefulPartitionedCall:0.124"  7.125549 
  182. "model/keras_layer/StatefulPartitionedCall:0.441"  7.101787 
  183. "model/keras_layer/StatefulPartitionedCall:0.241"  7.101360 
  184.  "model/keras_layer/StatefulPartitionedCall:0.92"  7.090490 
  185. "model/keras_layer/StatefulPartitionedCall:0.443"  7.074228 
  186. "model/keras_layer/StatefulPartitionedCall:0.428"  7.048878 
  187. "model/keras_layer/StatefulPartitionedCall:0.269"  7.029913 
  188. "model/keras_layer/StatefulPartitionedCall:0.414"  6.976240 
  189. "model/keras_layer/StatefulPartitionedCall:0.444"  6.906858 
  190. "model/keras_layer/StatefulPartitionedCall:0.116"  6.893878 
  191. "model/keras_layer/StatefulPartitionedCall:0.179"  6.881754 
  192. "model/keras_layer/StatefulPartitionedCall:0.172"  6.854306 
  193. "model/keras_layer/StatefulPartitionedCall:0.164"  6.825862 
  194. "model/keras_layer/StatefulPartitionedCall:0.196"  6.812775 
  195. "model/keras_layer/StatefulPartitionedCall:0.199"  6.773179 
  196. "model/keras_layer/StatefulPartitionedCall:0.350"  6.763141 
  197. "model/keras_layer/StatefulPartitionedCall:0.148"  6.760933 
  198. "model/keras_layer/StatefulPartitionedCall:0.293"  6.715807 
  199.  "model/keras_layer/StatefulPartitionedCall:0.94"  6.703833 
  200. "model/keras_layer/StatefulPartitionedCall:0.129"  6.703430 
  201. "model/keras_layer/StatefulPartitionedCall:0.502"  6.621521 
  202.   "model/keras_layer/StatefulPartitionedCall:0.9"  6.617845 
  203. "model/keras_layer/StatefulPartitionedCall:0.145"  6.561358 
  204. "model/keras_layer/StatefulPartitionedCall:0.494"  6.466436 
  205. "model/keras_layer/StatefulPartitionedCall:0.101"  6.442233 
  206. "model/keras_layer/StatefulPartitionedCall:0.345"  6.435505 
  207. "model/keras_layer/StatefulPartitionedCall:0.256"  6.386036 
  208. "model/keras_layer/StatefulPartitionedCall:0.422"  6.368373 
  209. "model/keras_layer/StatefulPartitionedCall:0.231"  6.329492 
  210. "model/keras_layer/StatefulPartitionedCall:0.248"  6.299469 
  211. "model/keras_layer/StatefulPartitionedCall:0.416"  6.281936 
  212. "model/keras_layer/StatefulPartitionedCall:0.339"  6.253476 
  213.  "model/keras_layer/StatefulPartitionedCall:0.15"  6.246779 
  214. "model/keras_layer/StatefulPartitionedCall:0.266"  6.245821 
  215.  "model/keras_layer/StatefulPartitionedCall:0.17"  6.219548 
  216.  "model/keras_layer/StatefulPartitionedCall:0.33"  6.211613 
  217. "model/keras_layer/StatefulPartitionedCall:0.299"  6.188691 
  218. "model/keras_layer/StatefulPartitionedCall:0.223"  6.184935 
  219. "model/keras_layer/StatefulPartitionedCall:0.385"  6.166133 
  220. "model/keras_layer/StatefulPartitionedCall:0.335"  6.158915 
  221. "model/keras_layer/StatefulPartitionedCall:0.484"  6.141772 
  222.  "model/keras_layer/StatefulPartitionedCall:0.90"  6.094871 
  223. "model/keras_layer/StatefulPartitionedCall:0.251"  6.092598 
  224. "model/keras_layer/StatefulPartitionedCall:0.483"  6.068509 
  225. "model/keras_layer/StatefulPartitionedCall:0.459"  6.060700 
  226. "model/keras_layer/StatefulPartitionedCall:0.232"  6.030481 
  227. "model/keras_layer/StatefulPartitionedCall:0.344"  6.011938 
  228. "model/keras_layer/StatefulPartitionedCall:0.183"  5.970037 
  229. "model/keras_layer/StatefulPartitionedCall:0.359"  5.920704 
  230. "model/keras_layer/StatefulPartitionedCall:0.369"  5.919326 
  231. "model/keras_layer/StatefulPartitionedCall:0.372"  5.918357 
  232.  "model/keras_layer/StatefulPartitionedCall:0.84"  5.860459 
  233. "model/keras_layer/StatefulPartitionedCall:0.413"  5.839362 
  234. "model/keras_layer/StatefulPartitionedCall:0.415"  5.838154 
  235. "model/keras_layer/StatefulPartitionedCall:0.176"  5.777905 
  236. "model/keras_layer/StatefulPartitionedCall:0.208"  5.770804 
  237. "model/keras_layer/StatefulPartitionedCall:0.102"  5.758823 
  238. "model/keras_layer/StatefulPartitionedCall:0.480"  5.748162 
  239. "model/keras_layer/StatefulPartitionedCall:0.381"  5.734125 
  240. "model/keras_layer/StatefulPartitionedCall:0.504"  5.727180 
  241. "model/keras_layer/StatefulPartitionedCall:0.406"  5.723869 
  242. "model/keras_layer/StatefulPartitionedCall:0.222"  5.708333 
  243. "model/keras_layer/StatefulPartitionedCall:0.254"  5.623589 
  244.  "model/keras_layer/StatefulPartitionedCall:0.71"  5.567681 
  245. "model/keras_layer/StatefulPartitionedCall:0.268"  5.546424 
  246.   "model/keras_layer/StatefulPartitionedCall:0.2"  5.541273 
  247. "model/keras_layer/StatefulPartitionedCall:0.158"  5.519200 
  248. "model/keras_layer/StatefulPartitionedCall:0.130"  5.510826 
  249. "model/keras_layer/StatefulPartitionedCall:0.174"  5.509232 
  250. "model/keras_layer/StatefulPartitionedCall:0.336"  5.485028 
  251. "model/keras_layer/StatefulPartitionedCall:0.225"  5.448486 
  252.  "model/keras_layer/StatefulPartitionedCall:0.91"  5.409002 
  253. "model/keras_layer/StatefulPartitionedCall:0.401"  5.356538 
  254. "model/keras_layer/StatefulPartitionedCall:0.227"  5.333519 
  255. "model/keras_layer/StatefulPartitionedCall:0.138"  5.325525 
  256. "model/keras_layer/StatefulPartitionedCall:0.186"  5.290089 
  257. "model/keras_layer/StatefulPartitionedCall:0.105"  5.269920 
  258. "model/keras_layer/StatefulPartitionedCall:0.446"  5.195781 
  259. "model/keras_layer/StatefulPartitionedCall:0.470"  5.166897 
  260. "model/keras_layer/StatefulPartitionedCall:0.259"  5.158494 
  261.  "model/keras_layer/StatefulPartitionedCall:0.31"  5.141092 
  262. "model/keras_layer/StatefulPartitionedCall:0.173"  5.136466 
  263. "model/keras_layer/StatefulPartitionedCall:0.364"  5.133627 
  264. "model/keras_layer/StatefulPartitionedCall:0.273"  5.105121 
  265. "model/keras_layer/StatefulPartitionedCall:0.265"  5.077867 
  266. "model/keras_layer/StatefulPartitionedCall:0.371"  5.069744 
  267. "model/keras_layer/StatefulPartitionedCall:0.378"  5.041357 
  268. "model/keras_layer/StatefulPartitionedCall:0.267"  5.040658 
  269. "model/keras_layer/StatefulPartitionedCall:0.215"  5.025911 
  270. "model/keras_layer/StatefulPartitionedCall:0.197"  5.013517 
  271. "model/keras_layer/StatefulPartitionedCall:0.141"  4.981632 
  272. "model/keras_layer/StatefulPartitionedCall:0.234"  4.966883 
  273. "model/keras_layer/StatefulPartitionedCall:0.477"  4.889969 
  274. "model/keras_layer/StatefulPartitionedCall:0.510"  4.872476 
  275. "model/keras_layer/StatefulPartitionedCall:0.106"  4.871483 
  276.  "model/keras_layer/StatefulPartitionedCall:0.82"  4.855385 
  277.  "model/keras_layer/StatefulPartitionedCall:0.48"  4.810244 
  278. "model/keras_layer/StatefulPartitionedCall:0.420"  4.809900 
  279. "model/keras_layer/StatefulPartitionedCall:0.280"  4.741931 
  280.  "model/keras_layer/StatefulPartitionedCall:0.67"  4.740705 
  281. "model/keras_layer/StatefulPartitionedCall:0.292"  4.732461 
  282. "model/keras_layer/StatefulPartitionedCall:0.306"  4.705740 
  283. "model/keras_layer/StatefulPartitionedCall:0.455"  4.705180 
  284. "model/keras_layer/StatefulPartitionedCall:0.365"  4.683675 
  285. "model/keras_layer/StatefulPartitionedCall:0.239"  4.669809 
  286.  "model/keras_layer/StatefulPartitionedCall:0.25"  4.638117 
  287. "model/keras_layer/StatefulPartitionedCall:0.386"  4.636329 
  288.  "model/keras_layer/StatefulPartitionedCall:0.68"  4.611972 
  289.  "model/keras_layer/StatefulPartitionedCall:0.40"  4.586663 
  290. "model/keras_layer/StatefulPartitionedCall:0.425"  4.583293 
  291. "model/keras_layer/StatefulPartitionedCall:0.177"  4.567895 
  292.  "model/keras_layer/StatefulPartitionedCall:0.57"  4.565003 
  293. "model/keras_layer/StatefulPartitionedCall:0.342"  4.554643 
  294. "model/keras_layer/StatefulPartitionedCall:0.128"  4.533961 
  295. "model/keras_layer/StatefulPartitionedCall:0.487"  4.528112 
  296. "model/keras_layer/StatefulPartitionedCall:0.501"  4.494283 
  297. "model/keras_layer/StatefulPartitionedCall:0.198"  4.468368 
  298. "model/keras_layer/StatefulPartitionedCall:0.307"  4.464016 
  299.  "model/keras_layer/StatefulPartitionedCall:0.95"  4.456392 
  300. "model/keras_layer/StatefulPartitionedCall:0.456"  4.453833 
  301.  "model/keras_layer/StatefulPartitionedCall:0.54"  4.342939 
  302. "model/keras_layer/StatefulPartitionedCall:0.461"  4.319604 
  303.   "model/keras_layer/StatefulPartitionedCall:0.5"  4.277122 
  304. "model/keras_layer/StatefulPartitionedCall:0.311"  4.273914 
  305.  "model/keras_layer/StatefulPartitionedCall:0.39"  4.255758 
  306. "model/keras_layer/StatefulPartitionedCall:0.276"  4.240512 
  307.  "model/keras_layer/StatefulPartitionedCall:0.34"  4.126914 
  308. "model/keras_layer/StatefulPartitionedCall:0.263"  4.126399 
  309. "model/keras_layer/StatefulPartitionedCall:0.346"  4.099002 
  310. "model/keras_layer/StatefulPartitionedCall:0.282"  4.027431 
  311. "model/keras_layer/StatefulPartitionedCall:0.100"  4.018021 
  312. "model/keras_layer/StatefulPartitionedCall:0.314"  4.006797 
  313. "model/keras_layer/StatefulPartitionedCall:0.380"  3.974540 
  314. "model/keras_layer/StatefulPartitionedCall:0.506"  3.952839 
  315. "model/keras_layer/StatefulPartitionedCall:0.475"  3.928019 
  316. "model/keras_layer/StatefulPartitionedCall:0.423"  3.878334 
  317. "model/keras_layer/StatefulPartitionedCall:0.245"  3.806617 
  318. "model/keras_layer/StatefulPartitionedCall:0.190"  3.783337 
  319. "model/keras_layer/StatefulPartitionedCall:0.405"  3.762053 
  320.  "model/keras_layer/StatefulPartitionedCall:0.70"  3.752657 
  321.  "model/keras_layer/StatefulPartitionedCall:0.42"  3.737024 
  322. "model/keras_layer/StatefulPartitionedCall:0.257"  3.734198 
  323. "model/keras_layer/StatefulPartitionedCall:0.468"  3.726559 
  324. "model/keras_layer/StatefulPartitionedCall:0.495"  3.709509 
  325. "model/keras_layer/StatefulPartitionedCall:0.448"  3.702890 
  326. "model/keras_layer/StatefulPartitionedCall:0.301"  3.625670 
  327. "model/keras_layer/StatefulPartitionedCall:0.394"  3.618161 
  328. "model/keras_layer/StatefulPartitionedCall:0.274"  3.603334 
  329. "model/keras_layer/StatefulPartitionedCall:0.103"  3.581393 
  330. "model/keras_layer/StatefulPartitionedCall:0.134"  3.580213 
  331. "model/keras_layer/StatefulPartitionedCall:0.321"  3.565317 
  332. "model/keras_layer/StatefulPartitionedCall:0.107"  3.550612 
  333. "model/keras_layer/StatefulPartitionedCall:0.402"  3.527571 
  334. "model/keras_layer/StatefulPartitionedCall:0.412"  3.499199 
  335. "model/keras_layer/StatefulPartitionedCall:0.429"  3.496518 
  336. "model/keras_layer/StatefulPartitionedCall:0.407"  3.488807 
  337. "model/keras_layer/StatefulPartitionedCall:0.348"  3.474226 
  338. "model/keras_layer/StatefulPartitionedCall:0.331"  3.449404 
  339. "model/keras_layer/StatefulPartitionedCall:0.238"  3.444785 
  340. "model/keras_layer/StatefulPartitionedCall:0.507"  3.443128 
  341. "model/keras_layer/StatefulPartitionedCall:0.111"  3.434171 
  342. "model/keras_layer/StatefulPartitionedCall:0.324"  3.398306 
  343. "model/keras_layer/StatefulPartitionedCall:0.421"  3.394655 
  344. "model/keras_layer/StatefulPartitionedCall:0.233"  3.389123 
  345.  "model/keras_layer/StatefulPartitionedCall:0.45"  3.385508 
  346. "model/keras_layer/StatefulPartitionedCall:0.367"  3.365305 
  347. "model/keras_layer/StatefulPartitionedCall:0.147"  3.352014 
  348. "model/keras_layer/StatefulPartitionedCall:0.318"  3.344063 
  349. "model/keras_layer/StatefulPartitionedCall:0.155"  3.319917 
  350. "model/keras_layer/StatefulPartitionedCall:0.426"  3.278855 
  351. "model/keras_layer/StatefulPartitionedCall:0.326"  3.260668 
  352. "model/keras_layer/StatefulPartitionedCall:0.384"  3.253344 
  353. "model/keras_layer/StatefulPartitionedCall:0.442"  3.248678 
  354. "model/keras_layer/StatefulPartitionedCall:0.481"  3.234564 
  355. "model/keras_layer/StatefulPartitionedCall:0.202"  3.225811 
  356. "model/keras_layer/StatefulPartitionedCall:0.189"  3.224019 
  357.  "model/keras_layer/StatefulPartitionedCall:0.16"  3.223055 
  358. "model/keras_layer/StatefulPartitionedCall:0.246"  3.187610 
  359. "model/keras_layer/StatefulPartitionedCall:0.230"  3.137264 
  360. "model/keras_layer/StatefulPartitionedCall:0.258"  3.131663 
  361.  "model/keras_layer/StatefulPartitionedCall:0.81"  3.104973 
  362.  "model/keras_layer/StatefulPartitionedCall:0.36"  3.091691 
  363. "model/keras_layer/StatefulPartitionedCall:0.447"  3.088783 
  364. "model/keras_layer/StatefulPartitionedCall:0.340"  3.045935 
  365. "model/keras_layer/StatefulPartitionedCall:0.123"  3.004205 
  366. "model/keras_layer/StatefulPartitionedCall:0.450"  3.003949 
  367. "model/keras_layer/StatefulPartitionedCall:0.109"  2.979840 
  368. "model/keras_layer/StatefulPartitionedCall:0.132"  2.944965 
  369. "model/keras_layer/StatefulPartitionedCall:0.417"  2.892208 
  370. "model/keras_layer/StatefulPartitionedCall:0.185"  2.887377 
  371. "model/keras_layer/StatefulPartitionedCall:0.125"  2.885752 
  372. "model/keras_layer/StatefulPartitionedCall:0.210"  2.880575 
  373. "model/keras_layer/StatefulPartitionedCall:0.262"  2.844423 
  374. "model/keras_layer/StatefulPartitionedCall:0.408"  2.829988 
  375. "model/keras_layer/StatefulPartitionedCall:0.478"  2.826282 
  376.  "model/keras_layer/StatefulPartitionedCall:0.64"  2.815428 
  377. "model/keras_layer/StatefulPartitionedCall:0.509"  2.814878 
  378. "model/keras_layer/StatefulPartitionedCall:0.488"  2.813426 
  379. "model/keras_layer/StatefulPartitionedCall:0.409"  2.804216 
  380. "model/keras_layer/StatefulPartitionedCall:0.382"  2.797588 
  381.  "model/keras_layer/StatefulPartitionedCall:0.75"  2.755797 
  382.  "model/keras_layer/StatefulPartitionedCall:0.27"  2.753730 
  383. "model/keras_layer/StatefulPartitionedCall:0.201"  2.715082 
  384. "model/keras_layer/StatefulPartitionedCall:0.313"  2.689072 
  385.  "model/keras_layer/StatefulPartitionedCall:0.63"  2.668630 
  386. "model/keras_layer/StatefulPartitionedCall:0.136"  2.657904 
  387. "model/keras_layer/StatefulPartitionedCall:0.200"  2.648429 
  388.  "model/keras_layer/StatefulPartitionedCall:0.86"  2.608584 
  389. "model/keras_layer/StatefulPartitionedCall:0.370"  2.608259 
  390.   "model/keras_layer/StatefulPartitionedCall:0.7"  2.589564 
  391. "model/keras_layer/StatefulPartitionedCall:0.361"  2.543632 
  392. "model/keras_layer/StatefulPartitionedCall:0.205"  2.525291 
  393. "model/keras_layer/StatefulPartitionedCall:0.403"  2.516556 
  394. "model/keras_layer/StatefulPartitionedCall:0.400"  2.513631 
  395.  "model/keras_layer/StatefulPartitionedCall:0.60"  2.508916 
  396. "model/keras_layer/StatefulPartitionedCall:0.211"  2.495102 
  397. "model/keras_layer/StatefulPartitionedCall:0.498"  2.493361 
  398.  "model/keras_layer/StatefulPartitionedCall:0.62"  2.491742 
  399. "model/keras_layer/StatefulPartitionedCall:0.283"  2.484420 
  400.  "model/keras_layer/StatefulPartitionedCall:0.26"  2.482942 
  401. "model/keras_layer/StatefulPartitionedCall:0.410"  2.479620 
  402. "model/keras_layer/StatefulPartitionedCall:0.319"  2.479137 
  403.   "model/keras_layer/StatefulPartitionedCall:0.8"  2.451880 
  404. "model/keras_layer/StatefulPartitionedCall:0.253"  2.443417 
  405. "model/keras_layer/StatefulPartitionedCall:0.264"  2.437445 
  406. "model/keras_layer/StatefulPartitionedCall:0.312"  2.419576 
  407. "model/keras_layer/StatefulPartitionedCall:0.224"  2.412407 
  408. "model/keras_layer/StatefulPartitionedCall:0.216"  2.408795 
  409. "model/keras_layer/StatefulPartitionedCall:0.492"  2.408456 
  410.  "model/keras_layer/StatefulPartitionedCall:0.22"  2.383164 
  411.  "model/keras_layer/StatefulPartitionedCall:0.43"  2.380971 
  412. "model/keras_layer/StatefulPartitionedCall:0.389"  2.379928 
  413. "model/keras_layer/StatefulPartitionedCall:0.279"  2.359800 
  414. "model/keras_layer/StatefulPartitionedCall:0.379"  2.356843 
  415. "model/keras_layer/StatefulPartitionedCall:0.363"  2.356482 
  416. "model/keras_layer/StatefulPartitionedCall:0.431"  2.339803 
  417. "model/keras_layer/StatefulPartitionedCall:0.503"  2.327017 
  418.  "model/keras_layer/StatefulPartitionedCall:0.59"  2.312303 
  419. "model/keras_layer/StatefulPartitionedCall:0.388"  2.299962 
  420. "model/keras_layer/StatefulPartitionedCall:0.491"  2.294748 
  421. "model/keras_layer/StatefulPartitionedCall:0.505"  2.262696 
  422. "model/keras_layer/StatefulPartitionedCall:0.376"  2.255529 
  423.   "model/keras_layer/StatefulPartitionedCall:0.3"  2.220372 
  424. "model/keras_layer/StatefulPartitionedCall:0.404"  2.202129 
  425. "model/keras_layer/StatefulPartitionedCall:0.333"  2.159506 
  426. "model/keras_layer/StatefulPartitionedCall:0.271"  2.141655 
  427. "model/keras_layer/StatefulPartitionedCall:0.304"  2.102964 
  428. "model/keras_layer/StatefulPartitionedCall:0.295"  2.065895 
  429.  "model/keras_layer/StatefulPartitionedCall:0.24"  2.038466 
  430. "model/keras_layer/StatefulPartitionedCall:0.139"  2.022834 
  431. "model/keras_layer/StatefulPartitionedCall:0.206"  1.948106 
  432. "model/keras_layer/StatefulPartitionedCall:0.499"  1.921655 
  433. "model/keras_layer/StatefulPartitionedCall:0.220"  1.916158 
  434. "model/keras_layer/StatefulPartitionedCall:0.349"  1.900472 
  435.  "model/keras_layer/StatefulPartitionedCall:0.97"  1.883489 
  436. "model/keras_layer/StatefulPartitionedCall:0.366"  1.883261 
  437. "model/keras_layer/StatefulPartitionedCall:0.496"  1.873397 
  438.  "model/keras_layer/StatefulPartitionedCall:0.53"  1.873188 
  439. "model/keras_layer/StatefulPartitionedCall:0.162"  1.866710 
  440. "model/keras_layer/StatefulPartitionedCall:0.121"  1.829404 
  441.  "model/keras_layer/StatefulPartitionedCall:0.49"  1.813523 
  442. "model/keras_layer/StatefulPartitionedCall:0.151"  1.785530 
  443. "model/keras_layer/StatefulPartitionedCall:0.355"  1.779231 
  444.  "model/keras_layer/StatefulPartitionedCall:0.52"  1.754021 
  445.  "model/keras_layer/StatefulPartitionedCall:0.77"  1.724590 
  446. "model/keras_layer/StatefulPartitionedCall:0.391"  1.699200 
  447. "model/keras_layer/StatefulPartitionedCall:0.466"  1.691939 
  448.  "model/keras_layer/StatefulPartitionedCall:0.38"  1.654453 
  449. "model/keras_layer/StatefulPartitionedCall:0.493"  1.650817 
  450. "model/keras_layer/StatefulPartitionedCall:0.305"  1.507206 
  451. "model/keras_layer/StatefulPartitionedCall:0.452"  1.501890 
  452. "model/keras_layer/StatefulPartitionedCall:0.430"  1.501603 
  453. "model/keras_layer/StatefulPartitionedCall:0.329"  1.500705 
  454. "model/keras_layer/StatefulPartitionedCall:0.167"  1.474582 
  455. "model/keras_layer/StatefulPartitionedCall:0.137"  1.473506 
  456. "model/keras_layer/StatefulPartitionedCall:0.439"  1.469060 
  457. "model/keras_layer/StatefulPartitionedCall:0.440"  1.449919 
  458. "model/keras_layer/StatefulPartitionedCall:0.374"  1.446098 
  459. "model/keras_layer/StatefulPartitionedCall:0.390"  1.445947 
  460. "model/keras_layer/StatefulPartitionedCall:0.309"  1.339351 
  461. "model/keras_layer/StatefulPartitionedCall:0.396"  1.264196 
  462. "model/keras_layer/StatefulPartitionedCall:0.474"  1.249280 
  463. "model/keras_layer/StatefulPartitionedCall:0.143"  1.218771 
  464. "model/keras_layer/StatefulPartitionedCall:0.467"  1.184494 
  465. "model/keras_layer/StatefulPartitionedCall:0.122"  1.167522 
  466. "model/keras_layer/StatefulPartitionedCall:0.436"  1.159209 
  467. "model/keras_layer/StatefulPartitionedCall:0.303"  1.142481 
  468.  "model/keras_layer/StatefulPartitionedCall:0.19"  1.114048 
  469.  "model/keras_layer/StatefulPartitionedCall:0.20"  1.022762 
  470.  "model/keras_layer/StatefulPartitionedCall:0.12"  1.005345 
  471. "model/keras_layer/StatefulPartitionedCall:0.163"  0.940448 
  472. "model/keras_layer/StatefulPartitionedCall:0.242"  0.886268 
  473. "model/keras_layer/StatefulPartitionedCall:0.438"  0.864613 
  474.  "model/keras_layer/StatefulPartitionedCall:0.23"  0.862492 
  475.  "model/keras_layer/StatefulPartitionedCall:0.13"  0.849717 
  476. "model/keras_layer/StatefulPartitionedCall:0.395"  0.843081 
  477. "model/keras_layer/StatefulPartitionedCall:0.156"  0.785958 
  478. "model/keras_layer/StatefulPartitionedCall:0.316"  0.753866 
  479. "model/keras_layer/StatefulPartitionedCall:0.244"  0.727165 
  480. "model/keras_layer/StatefulPartitionedCall:0.195"  0.713669 
  481. "model/keras_layer/StatefulPartitionedCall:0.240"  0.650237 
  482.  "model/keras_layer/StatefulPartitionedCall:0.79"  0.620307 
  483. "model/keras_layer/StatefulPartitionedCall:0.236"  0.596349 
  484. "model/keras_layer/StatefulPartitionedCall:0.424"  0.584488 
  485. "model/keras_layer/StatefulPartitionedCall:0.157"  0.580960 
  486. "model/keras_layer/StatefulPartitionedCall:0.228"  0.560028 
  487.  "model/keras_layer/StatefulPartitionedCall:0.93"  0.557761 
  488. "model/keras_layer/StatefulPartitionedCall:0.217"  0.528437 
  489.  "model/keras_layer/StatefulPartitionedCall:0.30"  0.522167 
  490.   "model/keras_layer/StatefulPartitionedCall:0.6"  0.507292 
  491. "model/keras_layer/StatefulPartitionedCall:0.482"  0.465616 
  492. "model/keras_layer/StatefulPartitionedCall:0.296"  0.457078 
  493.  "model/keras_layer/StatefulPartitionedCall:0.61"  0.445589 
  494. "model/keras_layer/StatefulPartitionedCall:0.465"  0.400996 
  495.  "model/keras_layer/StatefulPartitionedCall:0.72"  0.396369 
  496. "model/keras_layer/StatefulPartitionedCall:0.181"  0.329822 
  497. "model/keras_layer/StatefulPartitionedCall:0.490"  0.313447 
  498. "model/keras_layer/StatefulPartitionedCall:0.435"  0.235094 
  499. "model/keras_layer/StatefulPartitionedCall:0.112"  0.191988 
  500. "model/keras_layer/StatefulPartitionedCall:0.182"  0.089356 
```
</div>
    
    
    
<div class="k-default-codeblock">
```
Loss: BINOMIAL_LOG_LIKELIHOOD
Validation loss value: 0.87837
Number of trees per iteration: 1
Node format: NOT_SET
Number of trees: 130
Total number of nodes: 6352
```
</div>
    
<div class="k-default-codeblock">
```
Number of nodes by tree:
Count: 130 Average: 48.8615 StdDev: 11.4574
Min: 23 Max: 63 Ignored: 0
----------------------------------------------
[ 23, 25)  1   0.77%   0.77%
[ 25, 27)  2   1.54%   2.31% #
[ 27, 29)  4   3.08%   5.38% #
[ 29, 31)  2   1.54%   6.92% #
[ 31, 33)  7   5.38%  12.31% ##
[ 33, 35)  7   5.38%  17.69% ##
[ 35, 37)  3   2.31%  20.00% #
[ 37, 39)  0   0.00%  20.00%
[ 39, 41)  1   0.77%  20.77%
[ 41, 43)  6   4.62%  25.38% ##
[ 43, 45)  9   6.92%  32.31% ###
[ 45, 47)  9   6.92%  39.23% ###
[ 47, 49)  4   3.08%  42.31% #
[ 49, 51)  8   6.15%  48.46% ###
[ 51, 53)  6   4.62%  53.08% ##
[ 53, 55)  8   6.15%  59.23% ###
[ 55, 57)  9   6.92%  66.15% ###
[ 57, 59)  7   5.38%  71.54% ##
[ 59, 61)  8   6.15%  77.69% ###
[ 61, 63] 29  22.31% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Depth by leafs:
Count: 3241 Average: 4.82783 StdDev: 0.547618
Min: 1 Max: 5 Ignored: 0
----------------------------------------------
[ 1, 2)   16   0.49%   0.49%
[ 2, 3)   26   0.80%   1.30%
[ 3, 4)   81   2.50%   3.80%
[ 4, 5)  254   7.84%  11.63% #
[ 5, 5] 2864  88.37% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Number of training obs by leaf:
Count: 3241 Average: 275.363 StdDev: 593.208
Min: 5 Max: 6520 Ignored: 0
----------------------------------------------
[    5,  330) 2583  79.70%  79.70% ##########
[  330,  656)  249   7.68%  87.38% #
[  656,  982)  135   4.17%  91.55% #
[  982, 1308)   97   2.99%  94.54%
[ 1308, 1634)   44   1.36%  95.90%
[ 1634, 1959)   40   1.23%  97.13%
[ 1959, 2285)   25   0.77%  97.90%
[ 2285, 2611)   13   0.40%  98.30%
[ 2611, 2937)   18   0.56%  98.86%
[ 2937, 3263)   11   0.34%  99.20%
[ 3263, 3588)    6   0.19%  99.38%
[ 3588, 3914)    8   0.25%  99.63%
[ 3914, 4240)    7   0.22%  99.85%
[ 4240, 4566)    3   0.09%  99.94%
[ 4566, 4892)    0   0.00%  99.94%
[ 4892, 5217)    0   0.00%  99.94%
[ 5217, 5543)    0   0.00%  99.94%
[ 5543, 5869)    1   0.03%  99.97%
[ 5869, 6195)    0   0.00%  99.97%
[ 6195, 6520]    1   0.03% 100.00%
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes:
	37 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	32 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	30 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	30 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	28 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	25 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	23 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.290 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.451 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.204 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.14 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.471 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.32 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.175 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.145 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.502 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.47 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.330 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.28 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.221 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.203 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.196 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.194 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.165 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.90 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.74 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.35 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.280 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.232 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.154 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.115 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.9 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.66 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.504 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.456 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.434 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.433 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.413 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.352 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.335 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.293 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.282 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.274 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.218 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.146 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.138 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.124 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.84 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.494 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.461 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.43 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.428 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.414 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.384 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.377 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.345 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.336 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.334 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.331 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.33 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.324 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.320 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.307 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.269 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.267 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.259 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.258 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.257 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.251 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.245 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.223 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.202 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.2 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.191 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.172 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.158 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.136 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.123 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.108 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.86 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.82 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.67 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.57 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.507 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.506 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.503 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.501 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.481 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.477 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.447 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.425 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.422 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.407 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.378 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.371 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.361 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.359 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.347 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.346 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.339 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.311 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.308 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.295 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.288 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.270 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.234 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.227 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.215 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.210 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.206 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.201 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.186 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.15 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.131 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.111 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.105 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.97 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.94 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.81 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.70 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.7 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.64 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.5 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.495 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.493 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.492 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.488 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.487 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.475 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.431 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.421 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.410 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.409 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.403 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.400 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.394 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.391 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.390 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.389 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.385 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.382 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.376 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.370 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.36 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.353 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.342 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.340 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.326 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.318 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.313 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.301 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.3 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.283 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.271 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.263 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.248 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.242 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.239 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.238 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.230 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.224 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.220 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.211 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.205 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.200 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.198 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.167 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.162 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.151 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.125 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.107 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.91 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.77 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.75 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.61 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.59 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.53 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.52 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.496 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.491 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.49 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.478 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.465 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.452 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.441 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.436 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.430 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.406 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.404 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.40 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.396 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.395 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.38 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.355 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.349 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.309 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.305 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.303 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.30 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.279 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.264 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.26 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.244 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.216 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.19 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.177 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.143 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.137 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.134 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.122 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.12 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.93 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.72 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.499 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.490 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.482 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.474 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.438 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.424 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.408 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.316 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.312 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.296 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.240 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.236 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.23 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.228 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.217 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.20 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.195 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.182 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.181 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.163 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.157 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.121 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.112 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 0:
	9 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 1:
	12 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.501 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.335 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.293 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.221 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.204 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 2:
	21 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.376 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.35 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.335 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.290 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.74 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.456 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.390 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.280 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.274 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.204 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.146 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.82 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.67 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.66 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.59 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.57 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.507 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.502 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.501 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.482 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.471 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.47 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.400 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.384 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.361 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.352 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.320 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.32 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.311 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.308 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.307 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.303 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.293 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.257 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.251 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.245 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.230 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.221 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.206 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.201 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.20 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.181 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.158 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.154 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.143 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.14 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.138 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.131 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.124 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.111 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.105 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 3:
	29 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.290 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.456 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.433 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.35 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.293 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.280 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.221 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.14 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.9 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.74 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.471 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.461 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.434 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.232 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.202 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.194 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.165 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.82 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.502 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.451 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.414 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.384 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.376 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.371 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.335 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.330 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.308 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.257 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.251 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.234 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.227 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.204 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.203 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.196 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.175 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.158 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.154 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.145 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.111 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.94 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.84 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.77 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.75 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.66 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.507 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.506 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.501 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.495 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.494 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.492 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.475 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.421 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.413 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.407 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.406 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.400 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.390 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.378 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.361 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.359 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.352 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.347 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.340 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.324 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.320 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.32 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.307 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.301 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.288 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.282 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.274 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.245 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.230 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.220 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.218 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.211 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.210 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.201 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.200 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.198 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.186 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.146 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.138 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.12 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.115 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.97 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.91 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.90 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.86 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.72 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.70 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.67 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.64 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.59 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.57 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.53 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.504 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.503 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.496 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.490 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.49 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.487 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.482 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.481 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.478 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.477 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.47 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.441 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.436 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.428 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.425 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.424 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.409 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.408 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.404 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.40 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.394 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.391 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.389 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.385 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.382 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.38 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.370 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.36 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.355 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.353 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.349 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.346 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.345 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.339 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.336 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.334 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.331 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.33 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.318 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.316 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.311 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.305 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.303 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.3 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.296 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.295 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.283 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.28 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.271 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.270 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.269 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.267 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.26 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.259 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.258 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.242 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.239 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.238 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.228 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.223 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.217 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.206 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.20 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.181 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.172 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.167 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.151 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.15 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.143 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.136 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.131 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.124 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.112 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.108 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.107 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.105 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 5:
	37 : model/keras_layer/StatefulPartitionedCall:0.458 [NUMERICAL]
	32 : model/keras_layer/StatefulPartitionedCall:0.166 [NUMERICAL]
	30 : model/keras_layer/StatefulPartitionedCall:0.50 [NUMERICAL]
	30 : model/keras_layer/StatefulPartitionedCall:0.464 [NUMERICAL]
	28 : model/keras_layer/StatefulPartitionedCall:0.188 [NUMERICAL]
	25 : model/keras_layer/StatefulPartitionedCall:0.126 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.354 [NUMERICAL]
	24 : model/keras_layer/StatefulPartitionedCall:0.153 [NUMERICAL]
	23 : model/keras_layer/StatefulPartitionedCall:0.159 [NUMERICAL]
	21 : model/keras_layer/StatefulPartitionedCall:0.127 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.356 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.247 [NUMERICAL]
	20 : model/keras_layer/StatefulPartitionedCall:0.10 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.44 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.343 [NUMERICAL]
	19 : model/keras_layer/StatefulPartitionedCall:0.294 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.46 [NUMERICAL]
	18 : model/keras_layer/StatefulPartitionedCall:0.180 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.427 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.281 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.178 [NUMERICAL]
	17 : model/keras_layer/StatefulPartitionedCall:0.133 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.511 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.463 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.337 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.323 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.310 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.289 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.219 [NUMERICAL]
	15 : model/keras_layer/StatefulPartitionedCall:0.152 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.399 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.322 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.171 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.142 [NUMERICAL]
	14 : model/keras_layer/StatefulPartitionedCall:0.0 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.88 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.489 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.362 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.315 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.290 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.286 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.260 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.209 [NUMERICAL]
	13 : model/keras_layer/StatefulPartitionedCall:0.144 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.87 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.73 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.469 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.419 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.278 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.250 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.193 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.192 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.18 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.168 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.140 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.120 [NUMERICAL]
	12 : model/keras_layer/StatefulPartitionedCall:0.1 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.51 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.485 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.460 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.41 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.368 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.332 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.285 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.273 [NUMERICAL]
	11 : model/keras_layer/StatefulPartitionedCall:0.11 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.89 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.76 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.473 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.454 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.451 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.449 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.360 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.358 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.338 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.297 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.291 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.252 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.207 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.204 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.149 [NUMERICAL]
	10 : model/keras_layer/StatefulPartitionedCall:0.14 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.80 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.65 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.500 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.472 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.471 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.418 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.327 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.32 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.317 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.300 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.275 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.261 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.255 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.241 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.229 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.175 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.150 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.145 [NUMERICAL]
	9 : model/keras_layer/StatefulPartitionedCall:0.135 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.95 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.78 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.69 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.58 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.502 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.47 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.457 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.453 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.411 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.392 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.387 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.37 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.357 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.351 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.330 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.328 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.325 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.29 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.28 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.249 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.225 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.221 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.203 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.196 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.194 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.187 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.169 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.165 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.118 [NUMERICAL]
	8 : model/keras_layer/StatefulPartitionedCall:0.117 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.96 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.90 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.74 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.486 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.484 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.479 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.476 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.446 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.397 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.393 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.381 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.375 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.364 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.35 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.298 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.287 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.284 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.280 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.266 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.243 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.235 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.232 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.231 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.214 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.208 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.199 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.184 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.183 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.17 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.161 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.154 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.148 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.115 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.114 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.113 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.104 [NUMERICAL]
	7 : model/keras_layer/StatefulPartitionedCall:0.102 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.98 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.92 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.9 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.85 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.66 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.56 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.504 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.497 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.483 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.480 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.48 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.468 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.456 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.455 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.444 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.434 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.433 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.432 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.420 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.416 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.413 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.405 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.402 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.352 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.344 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.341 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.335 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.293 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.292 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.282 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.276 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.274 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.272 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.265 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.226 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.218 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.160 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.16 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.146 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.138 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.129 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.124 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.110 [NUMERICAL]
	6 : model/keras_layer/StatefulPartitionedCall:0.106 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.99 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.84 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.8 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.54 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.510 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.494 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.461 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.450 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.45 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.443 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.442 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.437 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.43 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.428 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.426 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.423 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.42 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.414 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.412 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.386 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.384 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.383 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.380 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.377 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.373 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.372 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.367 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.348 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.345 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.336 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.334 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.331 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.33 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.324 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.320 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.314 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.31 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.307 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.302 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.299 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.269 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.267 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.262 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.259 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.258 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.257 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.254 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.251 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.25 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.245 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.237 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.233 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.223 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.222 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.202 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.2 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.197 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.191 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.190 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.189 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.185 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.176 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.174 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.172 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.164 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.158 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.147 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.136 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.130 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.123 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.108 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.103 [NUMERICAL]
	5 : model/keras_layer/StatefulPartitionedCall:0.101 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.86 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.82 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.71 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.68 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.67 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.62 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.57 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.507 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.506 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.503 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.501 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.481 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.477 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.470 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.462 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.459 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.448 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.447 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.429 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.425 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.422 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.417 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.415 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.407 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.401 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.39 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.378 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.371 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.369 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.365 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.361 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.359 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.347 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.346 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.339 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.333 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.321 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.311 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.308 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.306 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.304 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.295 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.288 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.270 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.268 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.256 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.253 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.234 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.227 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.22 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.215 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.210 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.206 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.201 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.186 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.173 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.15 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.141 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.132 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.131 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.128 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.116 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.111 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.105 [NUMERICAL]
	4 : model/keras_layer/StatefulPartitionedCall:0.100 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.97 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.94 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.81 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.70 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.7 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.64 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.63 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.60 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.509 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.5 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.498 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.495 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.493 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.492 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.488 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.487 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.475 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.439 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.431 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.421 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.410 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.409 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.403 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.400 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.394 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.391 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.390 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.389 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.385 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.382 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.376 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.370 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.36 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.353 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.350 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.342 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.340 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.34 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.326 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.319 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.318 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.313 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.301 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.3 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.283 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.271 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.263 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.248 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.242 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.24 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.239 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.238 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.230 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.224 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.220 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.211 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.205 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.200 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.198 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.179 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.167 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.162 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.155 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.151 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.125 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.109 [NUMERICAL]
	3 : model/keras_layer/StatefulPartitionedCall:0.107 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.91 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.79 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.77 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.75 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.61 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.59 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.53 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.52 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.505 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.496 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.491 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.49 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.478 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.465 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.452 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.441 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.440 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.436 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.430 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.406 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.404 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.40 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.396 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.395 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.388 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.38 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.379 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.374 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.363 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.355 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.349 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.329 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.309 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.305 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.303 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.30 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.279 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.27 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.264 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.26 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.246 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.244 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.216 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.19 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.177 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.156 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.143 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.137 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.134 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.122 [NUMERICAL]
	2 : model/keras_layer/StatefulPartitionedCall:0.12 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.93 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.72 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.6 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.499 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.490 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.482 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.474 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.467 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.466 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.438 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.435 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.424 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.408 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.366 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.316 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.312 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.296 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.240 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.236 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.23 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.228 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.217 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.20 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.195 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.182 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.181 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.163 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.157 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.139 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.13 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.121 [NUMERICAL]
	1 : model/keras_layer/StatefulPartitionedCall:0.112 [NUMERICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Condition type in nodes:
	3111 : HigherCondition
Condition type in nodes with depth <= 0:
	130 : HigherCondition
Condition type in nodes with depth <= 1:
	374 : HigherCondition
Condition type in nodes with depth <= 2:
	836 : HigherCondition
Condition type in nodes with depth <= 3:
	1679 : HigherCondition
Condition type in nodes with depth <= 5:
	3111 : HigherCondition
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
    1. "data:0" 112.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_NODES:
    1. "data:0" 336.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: SUM_SCORE:
    1. "data:0" 33.654399 
```
</div>
    
    
    
<div class="k-default-codeblock">
```
Loss: BINOMIAL_LOG_LIKELIHOOD
Validation loss value: 1.37594
Number of trees per iteration: 1
Node format: NOT_SET
Number of trees: 112
Total number of nodes: 784
```
</div>
    
<div class="k-default-codeblock">
```
Number of nodes by tree:
Count: 112 Average: 7 StdDev: 0
Min: 7 Max: 7 Ignored: 0
----------------------------------------------
[ 7, 7] 112 100.00% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Depth by leafs:
Count: 448 Average: 2.25 StdDev: 0.829156
Min: 1 Max: 3 Ignored: 0
----------------------------------------------
[ 1, 2) 112  25.00%  25.00% #####
[ 2, 3) 112  25.00%  50.00% #####
[ 3, 3] 224  50.00% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Number of training obs by leaf:
Count: 448 Average: 1716.25 StdDev: 2955.32
Min: 5 Max: 6835 Ignored: 0
----------------------------------------------
[    5,  346) 336  75.00%  75.00% ##########
[  346,  688)   0   0.00%  75.00%
[  688, 1029)   0   0.00%  75.00%
[ 1029, 1371)   0   0.00%  75.00%
[ 1371, 1712)   0   0.00%  75.00%
[ 1712, 2054)   0   0.00%  75.00%
[ 2054, 2395)   0   0.00%  75.00%
[ 2395, 2737)   0   0.00%  75.00%
[ 2737, 3078)   0   0.00%  75.00%
[ 3078, 3420)   0   0.00%  75.00%
[ 3420, 3762)   0   0.00%  75.00%
[ 3762, 4103)   0   0.00%  75.00%
[ 4103, 4445)   0   0.00%  75.00%
[ 4445, 4786)   0   0.00%  75.00%
[ 4786, 5128)   0   0.00%  75.00%
[ 5128, 5469)   0   0.00%  75.00%
[ 5469, 5811)   0   0.00%  75.00%
[ 5811, 6152)   0   0.00%  75.00%
[ 6152, 6494)   0   0.00%  75.00%
[ 6494, 6835] 112  25.00% 100.00% ###
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes:
	336 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 0:
	112 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 1:
	224 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 2:
	336 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 3:
	336 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 5:
	336 : data:0 [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Condition type in nodes:
	336 : ContainsBitmapCondition
Condition type in nodes with depth <= 0:
	112 : ContainsBitmapCondition
Condition type in nodes with depth <= 1:
	224 : ContainsBitmapCondition
Condition type in nodes with depth <= 2:
	336 : ContainsBitmapCondition
Condition type in nodes with depth <= 3:
	336 : ContainsBitmapCondition
Condition type in nodes with depth <= 5:
	336 : ContainsBitmapCondition
```
</div>
    
<div class="k-default-codeblock">
```
None

```
</div>
Plotting training an logs


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


![png](/img/examples/nlp/Tweet-classification-using-TFDF/Tweet-classification-using-TFDF_41_0.png)



![png](/img/examples/nlp/Tweet-classification-using-TFDF/Tweet-classification-using-TFDF_41_1.png)


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
Accuracy: 0.9631
recall: 0.9425
precision: 0.9707
auc: 0.9890
model_2 Evaluation: 
```
</div>
    
<div class="k-default-codeblock">
```
loss: 0.0000
Accuracy: 0.5731
recall: 0.0064
precision: 1.0000
auc: 0.5035

```
</div>
# Predicting on validation data


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
Prediction: 1
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
the Gradient Boosted Tree model with pretrained embeddings achieved 96.31%
test accuracy while the plain Gradient Boosted Tree model had 57.31% accuracy.
