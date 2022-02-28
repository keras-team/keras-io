"""
Title: Question Answering with Hugging Face Transformers
Author: Matthew Carrigan and Merve Noyan
Date created: 13/01/2022
Last modified: 13/01/2022
Description: Question answering implementation using Keras and Hugging Face Transformers.
"""
"""
## Introduction to Question Answering

Question answering is a common NLP task with several variants. In some variants, the task
is multiple-choice:
A list of possible answers are supplied with each question, and the model simply needs to
return a probability distribution over the options. A more challenging variant of
question answering, which is more applicable to real-life tasks, is when the options are
not provided. Instead, the model is given an input document -- called context -- and a
question about the document, and it must extract the span of text in the document that
contains the answer. In this case, the model is not computing a probability distribution
over answers, but two probability distributions over the tokens in the document text,
representing the start and end of the span containing the answer. This variant is called
"extractive question answering".

Extractive question answering is a very challenging NLP task, and the dataset size
required to train such a model from scratch when the questions and answers are natural
language is prohibitively huge. As a result, question answering (like almost all NLP
tasks) benefits enormously from starting from a strong pretrained foundation model -
starting from a strong pretrained language model can reduce the dataset size required to
reach a given accuracy by multiple orders of magnitude, enabling you to reach very strong
performance with surprisingly reasonable datasets.

Starting with a pretrained model adds difficulties, though - where do you get the model
from? How do you ensure that your input data is preprocessed and tokenized the same way
as the original model? How do you modify the model to add an output head that matches
your task of interest?

In this example, we'll show you how to load a model from the Hugging Face
[🤗Transformers](https://github.com/huggingface/transformers) library to tackle this
challenge. We'll also load a benchmark question answering dataset from the
[🤗Datasets](https://github.com/huggingface/datasets) library - this is another open-source
repository containing a wide range of datasets across many modalities, from NLP to vision
and beyond. Note, though, that there is no requirement that these libraries must be used
with each other. If you want to train a model from
[🤗Transformers](https://github.com/huggingface/transformers) on your own data, or you want
to load data from [🤗 Datasets](https://github.com/huggingface/datasets) and train your
own entirely unrelated models with it, that is of course possible (and highly
encouraged!)
"""

"""
## Installing the requirements
"""

"""shell
pip install git+https://github.com/huggingface/transformers.git
pip install datasets
pip install huggingface-hub
"""
"""
## Loading the dataset
"""

"""
We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to download
the SQUAD question answering dataset using `load_dataset()`.
"""

from datasets import load_dataset

datasets = load_dataset("squad")

"""
The `datasets` object itself is a
`DatasetDict`, which contains one key for the training, validation and test set. We can see
the training, validation and test sets all have a column for the context, the question
and the answers to those questions. To access an actual element, you need to select a
split first, then give an index. We can see the answers are indicated by their start
position in the text and their full text, which is a substring of the context as we
mentioned above. Let's take a look at what a single training example looks like.
"""

print(datasets["train"][0])

"""
## Preprocessing the training data
"""

"""
Before we can feed those texts to our model, we need to preprocess them. This is done by
a 🤗 Transformers `Tokenizer` which will (as the name indicates) tokenize the inputs
(including converting the tokens to their corresponding IDs in the pretrained vocabulary)
and put it in a format the model expects, as well as generate the other inputs that model
requires.

To do all of this, we instantiate our tokenizer with the `AutoTokenizer.from_pretrained`
method, which will ensure:

- We get a tokenizer that corresponds to the model architecture we want to use.
- We download the vocabulary used when pretraining this specific checkpoint.

That vocabulary will be cached, so it's not downloaded again the next time we run the
cell.

The `from_pretrained()` method expects the name of a model. If you're unsure which model to
pick, don't panic! The list of models to choose from can be bewildering, but in general
there is a simple tradeoff: Larger models are slower and consume more memory, but usually
yield slightly better final accuracies after fine-tuning. For this example, we have
chosen the (relatively) lightweight `"distilbert"`, a smaller, distilled version of the
famous BERT language model. If you absolutely must have the highest possible accuracy for
an important task, though, and you have the GPU memory (and free time) to handle it, you
may prefer to use a larger model, such as `"roberta-large"`. Newer and even larger models
than `"roberta"` exist in [🤗 Transformers](https://github.com/huggingface/transformers),
but we leave the task of finding and training them as an exercise to readers who are
either particularly masochistic or have 40GB of VRAM to throw around.
"""

from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

"""
Depending on the model you selected, you will see different keys in the dictionary
returned by the cell above. They don't matter much for what we're doing here (just know
they are required by the model we will instantiate later), but you can learn more about
them in [this tutorial](https://huggingface.co/transformers/preprocessing.html) if you're
interested.

One specific issue for the preprocessing in question answering is how to deal with very
long documents. We usually truncate them in other tasks, when they are longer than the
model maximum sentence length, but here, removing part of the the context might result in
losing the answer we are looking for. To deal with this, we will allow one (long) example
in our dataset to give several input features, each of length shorter than the maximum
length of the model (or the one we set as a hyper-parameter). Also, just in case the
answer lies at the point we split a long context, we allow some overlap between the
features we generate controlled by the hyper-parameter `doc_stride`.

If we simply truncate with a fixed size (`max_length`), we will lose information. We want to
avoid truncating the question, and instead only truncate the context to ensure the task
remains solvable. To do that, we'll set `truncation` to `"only_second"`, so that only the
second sequence (the context) in each pair is truncated. To get the list of features
capped by the maximum length, we need to set `return_overflowing_tokens` to True and pass
the `doc_stride` to `stride`. To see which feature of the original context contain the
answer, we can return `"offset_mapping"`.
"""

max_length = 384  # The maximum length of a feature (question and context)
doc_stride = (
    128  # The authorized overlap between two part of the context when splitting
)
# it is needed.

"""
In the case of impossible answers (the answer is in another feature given by an example
with a long context), we set the cls index for both the start and end position. We could
also simply discard those examples from the training set if the flag
`allow_impossible_answers` is `False`. Since the preprocessing is already complex enough
as it is, we've kept is simple for this part.
"""


def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a
    # stride. This results in one example possible giving several features when a context is long,
    # each of those features having a context that overlaps a bit the context of the previous
    # feature.
    examples["question"] = [q.lstrip() for q in examples["question"]]
    examples["context"] = [c.lstrip() for c in examples["context"]]
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a
    # map from a feature to its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original
    # context. This will help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what
        # is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this
        # span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the
            # CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the
                # answer.
                # Note: we could go after the last offset if the answer is the last word (edge
                # case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


"""
To apply this function on all the sentences (or pairs of sentences) in our dataset, we
just use the `map()` method of our `Dataset` object, which will apply the function on all
the elements of.

We'll use `batched=True` to encode the texts in batches together. This is to leverage the
full benefit of the fast tokenizer we loaded earlier, which will use multi-threading to
treat the texts in a batch concurrently. We also use the `remove_columns` argument to
remove the columns that existed before tokenization was applied - this ensures that the
only features remaining are the ones we actually want to pass to our model.
"""

tokenized_datasets = datasets.map(
    prepare_train_features,
    batched=True,
    remove_columns=datasets["train"].column_names,
    num_proc=3,
)

"""
Even better, the results are automatically cached by the 🤗 Datasets library to avoid
spending time on this step the next time you run your notebook. The 🤗 Datasets library is
normally smart enough to detect when the function you pass to map has changed (and thus
requires to not use the cache data). For instance, it will properly detect if you change
the task in the first cell and rerun the notebook. 🤗 Datasets warns you when it uses
cached files, you can pass `load_from_cache_file=False` in the call to `map()` to not use
the cached files and force the preprocessing to be applied again.

Because all our data has been padded or truncated to the same length, and it is not too
large, we can now simply convert it to a dict of numpy arrays, ready for training.

Although we will not use it here, 🤗 Datasets have a `to_tf_dataset()` helper method
designed to assist you when the data cannot be easily converted to arrays, such as when
it has variable sequence lengths, or is too large to fit in memory. This method wraps a
`tf.data.Dataset` around the underlying 🤗 Dataset, streaming samples from the underlying
dataset and batching them on the fly, thus minimizing wasted memory and computation from
unnecessary padding. If your use-case requires it, please see the
[docs](https://huggingface.co/docs/transformers/custom_datasets#finetune-with-tensorflow)
on to_tf_dataset and data collator for an example. If not, feel free to follow this example
and simply convert to dicts!
"""

train_set = tokenized_datasets["train"].with_format("numpy")[
    :
]  # Load the whole dataset as a dict of numpy arrays
validation_set = tokenized_datasets["validation"].with_format("numpy")[:]

"""
## Fine-tuning the model
"""

"""
That was a lot of work! But now that our data is ready, everything is going to run very
smoothly. First, we download the pretrained model and fine-tune it. Since our task is
question answering, we use the `TFAutoModelForQuestionAnswering` class. Like with the
tokenizer, the `from_pretrained()` method will download and cache the model for us:
"""

from transformers import TFAutoModelForQuestionAnswering

model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

"""
The warning is telling us we are throwing away some weights and newly initializing some
others. Don't panic! This is absolutely normal. Recall that models like BERT and
Distilbert are trained on a **language modeling** task, but we're loading the model as
a `TFAutoModelForQuestionAnswering`, which means we want the model to perform a
**question answering** task. This change requires the final output layer or "head" to be
removed and replaced with a new head suited for the new task. The `from_pretrained`
method will handle all of this for us, and the warning is there simply to remind us that
some model surgery has been performed, and that the model will not generate useful
predictions until the newly-initialized layers have been fine-tuned on some data.

Next, we can create an optimizer and specify a loss function. You can usually get
slightly better performance by using learning rate decay and decoupled weight decay, but
for the purposes of this example the standard `Adam` optimizer will work fine. Note,
however, that when fine-tuning a pretrained transformer model you will generally want to
use a low learning rate! We find the best results are obtained with values in the range
1e-5 to 1e-4, and training may completely diverge at the default Adam learning rate of 1e-3.
"""

import tensorflow as tf
from tensorflow import keras

optimizer = keras.optimizers.Adam(learning_rate=5e-5)

"""
And now we just compile and fit the model. As a convenience, all 🤗 Transformers models
come with a default loss which matches their output head, although you're of course free
to use your own. Because the built-in loss is computed internally during the forward
pass, when using it you may find that some Keras metrics misbehave or give unexpected
outputs. This is an area of very active development in 🤗 Transformers, though, so
hopefully we'll have a good solution to that issue soon!

For now, though, let's use the built-in loss without any metrics. To get the built-in
loss, simply leave out the `loss` argument to `compile`.
"""

# Optionally uncomment the next line for float16 training
keras.mixed_precision.set_global_policy("mixed_float16")

model.compile(optimizer=optimizer)

"""
And now we can train our model. Note that we're not passing separate labels - the labels
are keys in the input dict, to make them visible to the model during the forward pass so
it can compute the built-in loss.
"""

model.fit(train_set, validation_data=validation_set, epochs=1)

"""
And we're done! Let's give it a try, using some text from the keras.io frontpage:
"""

context = """Keras is an API designed for human beings, not machines. Keras follows best
practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes
the number of user actions required for common use cases, and it provides clear &
actionable error messages. It also has extensive documentation and developer guides. """
question = "What is Keras?"

inputs = tokenizer([context], [question], return_tensors="np")
outputs = model(inputs)
start_position = tf.argmax(outputs.start_logits, axis=1)
end_position = tf.argmax(outputs.end_logits, axis=1)
print(int(start_position), int(end_position[0]))

"""
Looks like our model thinks the answer is the span from tokens 1 to 12 (inclusive). No
prizes for guessing which tokens those are!
"""

answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
print(answer)

"""
And now we can use the `tokenizer.decode()` method to turn those token IDs back into text:
"""

print(tokenizer.decode(answer))

"""
And that's it! Remember that this example was designed to be quick to run rather than
state-of-the-art, and the model trained here will certainly make mistakes. If you use a
larger model to base your training on, and you take time to tune the hyperparameters
appropriately, you'll find that you can achieve much better losses (and correspondingly
more accurate answers).

Finally, you can push the model to the HuggingFace Hub. By pushing this model you will
have:

- A nice model card generated for you containing hyperparameters and metrics of the model
training,
- A web API for inference calls,
- A widget in the model page that enables others to test your model.
This model is currently hosted [here](https://huggingface.co/keras-io/transformers-qa)
and we have prepared a separate neat UI for you
[here](https://huggingface.co/spaces/keras-io/keras-qa).

```python
model.push_to_hub("transformers-qa", organization="keras-io")
tokenizer.push_to_hub("transformers-qa", organization="keras-io")
```

If you have non-Transformers based Keras models, you can also push them with
`push_to_hub_keras`. You can use `from_pretrained_keras` to load easily.

```python
from huggingface_hub.keras_mixin import push_to_hub_keras

push_to_hub_keras(
    model=model, repo_url="https://huggingface.co/your-username/your-awesome-model"
)
from_pretrained_keras("your-username/your-awesome-model") # load your model
```
"""
