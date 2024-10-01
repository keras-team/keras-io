"""
Title: Abstractive Summarization with Hugging Face Transformers
Author: Sreyan Ghosh
Date created: 2022/07/04
Last modified: 2022/08/28
Description: Training T5 using Hugging Face Transformers for Abstractive Summarization.
Accelerator: GPU
"""

"""
## Introduction

Automatic summarization is one of the central problems in
Natural Language Processing (NLP). It poses several challenges relating to language
understanding (e.g. identifying important content)
and generation (e.g. aggregating and rewording the identified content into a summary).

In this tutorial, we tackle the single-document summarization task
with an abstractive modeling approach. The primary idea here is to generate a short,
single-sentence news summary answering the question “What is the news article about?”.
This approach to summarization is also known as *Abstractive Summarization* and has
seen growing interest among researchers in various disciplines.

Following prior work, we aim to tackle this problem using a
sequence-to-sequence model. [Text-to-Text Transfer Transformer (`T5`)](https://arxiv.org/abs/1910.10683)
is a [Transformer-based](https://arxiv.org/abs/1706.03762) model built on the encoder-decoder
architecture, pretrained on a multi-task mixture of unsupervised and supervised tasks where each task
is converted into a text-to-text format. T5 shows impressive results in a variety of sequence-to-sequence
(sequence in this notebook refers to text) like summarization, translation, etc.

In this notebook, we will fine-tune the pretrained T5 on the Abstractive Summarization
task using Hugging Face Transformers on the `XSum` dataset loaded from Hugging Face Datasets.
"""

"""
## Setup
"""

"""
### Installing the requirements
"""

"""shell
!pip install transformers==4.20.0
!pip install keras_hub==0.3.0
!pip install datasets
!pip install huggingface-hub
!pip install nltk
!pip install rouge-score
"""

"""
### Importing the necessary libraries
"""

import os
import logging

import nltk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Only log error messages
tf.get_logger().setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
### Define certain variables
"""
# The percentage of the dataset you want to split as train and test
TRAIN_TEST_SPLIT = 0.1

MAX_INPUT_LENGTH = 1024  # Maximum length of the input to the model
MIN_TARGET_LENGTH = 5  # Minimum length of the output by the model
MAX_TARGET_LENGTH = 128  # Maximum length of the output by the model
BATCH_SIZE = 8  # Batch-size for training our model
LEARNING_RATE = 2e-5  # Learning-rate for training our model
MAX_EPOCHS = 1  # Maximum number of epochs we will train the model for

# This notebook is built on the t5-small checkpoint from the Hugging Face Model Hub
MODEL_CHECKPOINT = "t5-small"

"""
## Load the dataset

We will now download the [Extreme Summarization (XSum)](https://arxiv.org/abs/1808.08745).
The dataset consists of BBC articles and accompanying single sentence summaries.
Specifically, each article is prefaced with an introductory sentence (aka summary) which is
professionally written, typically by the author of the article. That dataset has 226,711 articles
divided into training (90%, 204,045), validation (5%, 11,332), and test (5%, 11,334) sets.

Following much of literature, we use the Recall-Oriented Understudy for Gisting Evaluation
(ROUGE) metric to evaluate our sequence-to-sequence abstrative summarization approach.

We will use the [Hugging Face Datasets](https://github.com/huggingface/datasets) library to download
the data we need to use for training and evaluation. This can be easily done with the
`load_dataset` function.
"""

from datasets import load_dataset

raw_datasets = load_dataset("xsum", split="train")

"""
The dataset has the following fields:

- **document**: the original BBC article to be summarized
- **summary**: the single sentence summary of the BBC article
- **id**: ID of the document-summary pair
"""

print(raw_datasets)

"""
We will now see how the data looks like:
"""

print(raw_datasets[0])

"""
For the sake of demonstrating the workflow, in this notebook we will only take
small stratified balanced splits (10%) of the train as our training and test sets.
We can easily split the dataset using the `train_test_split` method which expects
the split size and the name of the column relative to which you want to stratify.
"""

raw_datasets = raw_datasets.train_test_split(
    train_size=TRAIN_TEST_SPLIT, test_size=TRAIN_TEST_SPLIT
)

"""
## Data Pre-processing

Before we can feed those texts to our model, we need to pre-process them and get them
ready for the task. This is done by a Hugging Face Transformers `Tokenizer` which will tokenize
the inputs (including converting the tokens to their corresponding IDs in the pretrained
vocabulary) and put it in a format the model expects, as well as generate the other inputs
that model requires.

The `from_pretrained()` method expects the name of a model from the Hugging Face Model Hub. This is
exactly similar to MODEL_CHECKPOINT declared earlier and we will just pass that.
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

"""
If you are using one of the five T5 checkpoints we have to prefix the inputs with
"summarize:" (the model can also translate and it needs the prefix to know which task it
has to perform).
"""

if MODEL_CHECKPOINT in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

"""
We will write a simple function that helps us in the pre-processing that is compatible
with Hugging Face Datasets. To summarize, our pre-processing function should:

- Tokenize the text dataset (input and targets) into it's corresponding token ids that
will be used for embedding look-up in BERT
- Add the prefix to the tokens
- Create additional inputs for the model like `token_type_ids`, `attention_mask`, etc.
"""


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


"""
To apply this function on all the pairs of sentences in our dataset, we just use the
`map` method of our `dataset` object we created earlier. This will apply the function on
all the elements of all the splits in `dataset`, so our training and testing
data will be preprocessed in one single command.
"""

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

"""
## Defining the model

Now we can download the pretrained model and fine-tune it. Since our task is
sequence-to-sequence (both the input and output are text sequences), we use the
`TFAutoModelForSeq2SeqLM` class from the Hugging Face Transformers library. Like with the
tokenizer, the `from_pretrained` method will download and cache the model for us.

The `from_pretrained()` method expects the name of a model from the Hugging Face Model Hub. As
mentioned earlier, we will use the `t5-small` model checkpoint.
"""

from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

"""
For training Sequence to Sequence models, we need a special kind of data collator,
which will not only pad the inputs to the maximum length in the batch, but also the
labels. Thus, we use the `DataCollatorForSeq2Seq` provided by the Hugging Face Transformers
library on our dataset. The `return_tensors='tf'` ensures that we get `tf.Tensor`
objects back.
"""

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

"""
Next we define our training and testing sets with which we will train our model. Again, Hugging Face
Datasets provides us with the `to_tf_dataset` method which will help us integrate our
dataset with the `collator` defined above. The method expects certain parameters:

- **columns**: the columns which will serve as our independent variables
- **batch_size**: our batch size for training
- **shuffle**: whether we want to shuffle our dataset
- **collate_fn**: our collator function

Additionally, we also define a relatively smaller `generation_dataset` to calculate
`ROUGE` scores on the fly while training.
"""

train_dataset = tokenized_datasets["train"].to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    collate_fn=data_collator,
)
test_dataset = tokenized_datasets["test"].to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=False,
    collate_fn=data_collator,
)
generation_dataset = (
    tokenized_datasets["test"]
    .shuffle()
    .select(list(range(200)))
    .to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
)

"""
## Building and Compiling the the model

Now we will define our optimizer and compile the model. The loss calculation is handled
internally and so we need not worry about that!
"""

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer)

"""
## Training and Evaluating the model

To evaluate our model on-the-fly while training, we will define `metric_fn` which will
calculate the `ROUGE` score between the groud-truth and predictions.
"""

import keras_hub

rouge_l = keras_hub.metrics.RougeL()


def metric_fn(eval_predictions):
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_l(decoded_labels, decoded_predictions)
    # We will print only the F1 score, you can use other aggregation metrics as well
    result = {"RougeL": result["f1_score"]}

    return result


"""
Now we can finally start training our model!
"""

from transformers.keras_callbacks import KerasMetricCallback

metric_callback = KerasMetricCallback(
    metric_fn, eval_dataset=generation_dataset, predict_with_generate=True
)

callbacks = [metric_callback]

# For now we will use our test set as our validation_data
model.fit(
    train_dataset, validation_data=test_dataset, epochs=MAX_EPOCHS, callbacks=callbacks
)

"""
For best results, we recommend training the model for atleast 5 epochs on the entire
training dataset!
"""

"""
## Inference

Now we will try to infer the model we trained on an arbitrary article. To do so,
we will use the `pipeline` method from Hugging Face Transformers. Hugging Face Transformers provides
us with a variety of pipelines to choose from. For our task, we use the `summarization`
pipeline.

The `pipeline` method takes in the trained model and tokenizer as arguments. The
`framework="tf"` argument ensures that you are passing a model that was trained with TF.
"""

from transformers import pipeline

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")

summarizer(
    raw_datasets["test"][0]["document"],
    min_length=MIN_TARGET_LENGTH,
    max_length=MAX_TARGET_LENGTH,
)

"""
Now you can push this model to Hugging Face Model Hub and also share it with with all your friends,
family, favorite pets: they can all load it with the identifier
`"your-username/the-name-you-picked"` so for instance:

```python
model.push_to_hub("transformers-qa", organization="keras-io")
tokenizer.push_to_hub("transformers-qa", organization="keras-io")
```
And after you push your model this is how you can load it in the future!

```python
from transformers import TFAutoModelForSeq2SeqLM

model = TFAutoModelForSeq2SeqLM.from_pretrained("your-username/my-awesome-model")
```
"""
