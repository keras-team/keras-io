
# Question Answering with Hugging Face Transformers

**Author:** Matthew Carrigan and Merve Noyan<br>
**Date created:** 13/01/2022<br>
**Last modified:** 13/01/2022<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/question_answering.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/question_answering.py)


**Description:** Question answering implementation using Keras and Hugging Face Transformers.

---
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

## Installing the requirements


```python
!pip install git+https://github.com/huggingface/transformers.git
!pip install datasets
!pip install huggingface-hub
```

<div class="k-default-codeblock">
```
Collecting git+https://github.com/huggingface/transformers.git
  Cloning https://github.com/huggingface/transformers.git to /tmp/pip-req-build-iqf_och1
  Running command git clone -q https://github.com/huggingface/transformers.git /tmp/pip-req-build-iqf_och1
  Installing build dependencies ... [?25l[?25hdone
  Getting requirements to build wheel ... [?25l[?25hdone
    Preparing wheel metadata ... [?25l[?25hdone
Collecting pyyaml>=5.1
  Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
[K     |████████████████████████████████| 596 kB 15.1 MB/s 
[?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers==4.16.0.dev0) (2019.12.20)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers==4.16.0.dev0) (4.62.3)
Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers==4.16.0.dev0) (4.10.0)
Collecting huggingface-hub<1.0,>=0.1.0
  Downloading huggingface_hub-0.4.0-py3-none-any.whl (67 kB)
[K     |████████████████████████████████| 67 kB 6.0 MB/s 
[?25hRequirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers==4.16.0.dev0) (2.23.0)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers==4.16.0.dev0) (1.19.5)
Collecting tokenizers!=0.11.3,>=0.10.1
  Downloading tokenizers-0.11.4-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.8 MB)
[K     |████████████████████████████████| 6.8 MB 86.0 MB/s 
[?25hRequirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers==4.16.0.dev0) (21.3)
Collecting sacremoses
  Downloading sacremoses-0.0.47-py2.py3-none-any.whl (895 kB)
[K     |████████████████████████████████| 895 kB 76.9 MB/s 
[?25hRequirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers==4.16.0.dev0) (3.4.2)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0,>=0.1.0->transformers==4.16.0.dev0) (3.10.0.2)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->transformers==4.16.0.dev0) (3.0.6)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers==4.16.0.dev0) (3.7.0)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==4.16.0.dev0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==4.16.0.dev0) (2021.10.8)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==4.16.0.dev0) (1.24.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==4.16.0.dev0) (3.0.4)
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers==4.16.0.dev0) (1.15.0)
Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers==4.16.0.dev0) (1.1.0)
Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers==4.16.0.dev0) (7.1.2)
Building wheels for collected packages: transformers
  Building wheel for transformers (PEP 517) ... [?25l[?25hdone
  Created wheel for transformers: filename=transformers-4.16.0.dev0-py3-none-any.whl size=3524819 sha256=07e8305bdb8d6ddb00a597848c404f8775c2f0e7ff59d6705135cddac07539d4
  Stored in directory: /tmp/pip-ephem-wheel-cache-wwd7ynd3/wheels/90/a5/44/6bcd83827c8a60628c5ad602f429cd5076bcce5f2a90054947
Successfully built transformers
Installing collected packages: pyyaml, tokenizers, sacremoses, huggingface-hub, transformers
  Attempting uninstall: pyyaml
    Found existing installation: PyYAML 3.13
    Uninstalling PyYAML-3.13:
      Successfully uninstalled PyYAML-3.13
Successfully installed huggingface-hub-0.4.0 pyyaml-6.0 sacremoses-0.0.47 tokenizers-0.11.4 transformers-4.16.0.dev0
Collecting datasets
  Downloading datasets-1.18.0-py3-none-any.whl (311 kB)
[K     |████████████████████████████████| 311 kB 12.8 MB/s 
[?25hRequirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from datasets) (1.1.5)
Requirement already satisfied: multiprocess in /usr/local/lib/python3.7/dist-packages (from datasets) (0.70.12.2)
Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (4.62.3)
Requirement already satisfied: dill in /usr/local/lib/python3.7/dist-packages (from datasets) (0.3.4)
Requirement already satisfied: huggingface-hub<1.0.0,>=0.1.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.4.0)
Requirement already satisfied: pyarrow!=4.0.0,>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (3.0.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from datasets) (21.3)
Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from datasets) (4.10.0)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from datasets) (1.19.5)
Collecting fsspec[http]>=2021.05.0
  Downloading fsspec-2022.1.0-py3-none-any.whl (133 kB)
[K     |████████████████████████████████| 133 kB 72.4 MB/s 
[?25hCollecting aiohttp
  Downloading aiohttp-3.8.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)
[K     |████████████████████████████████| 1.1 MB 83.6 MB/s 
[?25hRequirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (2.23.0)
Collecting xxhash
  Downloading xxhash-2.0.2-cp37-cp37m-manylinux2010_x86_64.whl (243 kB)
[K     |████████████████████████████████| 243 kB 89.9 MB/s 
[?25hRequirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.4.2)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.10.0.2)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (6.0)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->datasets) (3.0.6)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2021.10.8)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (1.24.3)
Collecting multidict<7.0,>=4.5
  Downloading multidict-6.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (94 kB)
[K     |████████████████████████████████| 94 kB 4.3 MB/s 
[?25hRequirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (21.4.0)
Collecting yarl<2.0,>=1.0
  Downloading yarl-1.7.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (271 kB)
[K     |████████████████████████████████| 271 kB 56.2 MB/s 
[?25hCollecting aiosignal>=1.1.2
  Downloading aiosignal-1.2.0-py3-none-any.whl (8.2 kB)
Collecting asynctest==0.13.0
  Downloading asynctest-0.13.0-py3-none-any.whl (26 kB)
Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (2.0.10)
Collecting frozenlist>=1.1.1
  Downloading frozenlist-1.3.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (144 kB)
[K     |████████████████████████████████| 144 kB 87.4 MB/s 
[?25hCollecting async-timeout<5.0,>=4.0.0a3
  Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->datasets) (3.7.0)
Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2.8.2)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2018.9)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.15.0)
Installing collected packages: multidict, frozenlist, yarl, asynctest, async-timeout, aiosignal, fsspec, aiohttp, xxhash, datasets
Successfully installed aiohttp-3.8.1 aiosignal-1.2.0 async-timeout-4.0.2 asynctest-0.13.0 datasets-1.18.0 frozenlist-1.3.0 fsspec-2022.1.0 multidict-6.0.2 xxhash-2.0.2 yarl-1.7.2
Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.7/dist-packages (0.4.0)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from huggingface-hub) (6.0)
Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub) (21.3)
Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from huggingface-hub) (2.23.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from huggingface-hub) (3.4.2)
Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from huggingface-hub) (4.62.3)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub) (3.10.0.2)
Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from huggingface-hub) (4.10.0)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.9->huggingface-hub) (3.0.6)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->huggingface-hub) (3.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->huggingface-hub) (1.24.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->huggingface-hub) (2021.10.8)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->huggingface-hub) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->huggingface-hub) (3.0.4)

```
</div>
---
## Loading the dataset

We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to download
the SQUAD question answering dataset using `load_dataset()`.


```python
from datasets import load_dataset

datasets = load_dataset("squad")
```


<div class="k-default-codeblock">
```
Downloading:   0%|          | 0.00/1.97k [00:00<?, ?B/s]

Downloading:   0%|          | 0.00/1.02k [00:00<?, ?B/s]

Downloading and preparing dataset squad/plain_text (download: 33.51 MiB, generated: 85.63 MiB, post-processed: Unknown size, total: 119.14 MiB) to /root/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453...

  0%|          | 0/2 [00:00<?, ?it/s]

Downloading:   0%|          | 0.00/8.12M [00:00<?, ?B/s]

Downloading:   0%|          | 0.00/1.05M [00:00<?, ?B/s]

  0%|          | 0/2 [00:00<?, ?it/s]

0 examples [00:00, ? examples/s]

0 examples [00:00, ? examples/s]

Dataset squad downloaded and prepared to /root/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453. Subsequent calls will reuse this data.

  0%|          | 0/2 [00:00<?, ?it/s]

```
</div>
The `datasets` object itself is a
`DatasetDict`, which contains one key for the training, validation and test set. We can see
the training, validation and test sets all have a column for the context, the question
and the answers to those questions. To access an actual element, you need to select a
split first, then give an index. We can see the answers are indicated by their start
position in the text and their full text, which is a substring of the context as we
mentioned above. Let's take a look at what a single training example looks like.


```python
print(datasets["train"][0])
```

<div class="k-default-codeblock">
```
{'id': '5733be284776f41900661182', 'title': 'University_of_Notre_Dame', 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}

```
</div>
---
## Preprocessing the training data

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


```python
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```


<div class="k-default-codeblock">
```
Downloading:   0%|          | 0.00/29.0 [00:00<?, ?B/s]

Downloading:   0%|          | 0.00/411 [00:00<?, ?B/s]

Downloading:   0%|          | 0.00/208k [00:00<?, ?B/s]

Downloading:   0%|          | 0.00/426k [00:00<?, ?B/s]

```
</div>
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


```python
max_length = 384  # The maximum length of a feature (question and context)
doc_stride = (
    128  # The authorized overlap between two part of the context when splitting
)
# it is needed.
```

In the case of impossible answers (the answer is in another feature given by an example
with a long context), we set the cls index for both the start and end position. We could
also simply discard those examples from the training set if the flag
`allow_impossible_answers` is `False`. Since the preprocessing is already complex enough
as it is, we've kept is simple for this part.


```python

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

```

To apply this function on all the sentences (or pairs of sentences) in our dataset, we
just use the `map()` method of our `Dataset` object, which will apply the function on all
the elements of.

We'll use `batched=True` to encode the texts in batches together. This is to leverage the
full benefit of the fast tokenizer we loaded earlier, which will use multi-threading to
treat the texts in a batch concurrently. We also use the `remove_columns` argument to
remove the columns that existed before tokenization was applied - this ensures that the
only features remaining are the ones we actually want to pass to our model.


```python
tokenized_datasets = datasets.map(
    prepare_train_features,
    batched=True,
    remove_columns=datasets["train"].column_names,
    num_proc=3,
)
```

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


```python
train_set = tokenized_datasets["train"].with_format("numpy")[
    :
]  # Load the whole dataset as a dict of numpy arrays
validation_set = tokenized_datasets["validation"].with_format("numpy")[:]
```

---
## Fine-tuning the model

That was a lot of work! But now that our data is ready, everything is going to run very
smoothly. First, we download the pretrained model and fine-tune it. Since our task is
question answering, we use the `TFAutoModelForQuestionAnswering` class. Like with the
tokenizer, the `from_pretrained()` method will download and cache the model for us:


```python
from transformers import TFAutoModelForQuestionAnswering

model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
```


<div class="k-default-codeblock">
```
Downloading:   0%|          | 0.00/338M [00:00<?, ?B/s]

Some layers from the model checkpoint at distilbert-base-cased were not used when initializing TFDistilBertForQuestionAnswering: ['vocab_transform', 'activation_13', 'vocab_projector', 'vocab_layer_norm']
- This IS expected if you are initializing TFDistilBertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFDistilBertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some layers of TFDistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-cased and are newly initialized: ['dropout_19', 'qa_outputs']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

```
</div>
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


```python
import tensorflow as tf
from tensorflow import keras

optimizer = keras.optimizers.Adam(learning_rate=5e-5)
```

And now we just compile and fit the model. As a convenience, all 🤗 Transformers models
come with a default loss which matches their output head, although you're of course free
to use your own. Because the built-in loss is computed internally during the forward
pass, when using it you may find that some Keras metrics misbehave or give unexpected
outputs. This is an area of very active development in 🤗 Transformers, though, so
hopefully we'll have a good solution to that issue soon!

For now, though, let's use the built-in loss without any metrics. To get the built-in
loss, simply leave out the `loss` argument to `compile`.


```python
# Optionally uncomment the next line for float16 training
keras.mixed_precision.set_global_policy("mixed_float16")

model.compile(optimizer=optimizer)
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Mixed precision compatibility check (mixed_float16): OK
Your GPU will likely run quickly with dtype policy mixed_float16 as it has compute capability of at least 7.0. Your GPU: Tesla V100-SXM2-16GB, compute capability 7.0

No loss specified in compile() - the model's internal loss computation will be used as the loss. Don't panic - this is a common way to train TensorFlow models in Transformers! Please ensure your labels are passed as keys in the input dict so that they are accessible to the model during the forward pass. To disable this behaviour, please pass a loss argument, or explicitly pass loss=None if you do not want your model to compute a loss.

```
</div>
And now we can train our model. Note that we're not passing separate labels - the labels
are keys in the input dict, to make them visible to the model during the forward pass so
it can compute the built-in loss.


```python
model.fit(train_set, validation_data=validation_set, epochs=1)
```

<div class="k-default-codeblock">
```
2773/2773 [==============================] - 1205s 431ms/step - loss: 1.5360 - val_loss: 1.1816

<keras.callbacks.History at 0x7f0b104fab90>

```
</div>
And we're done! Let's give it a try, using some text from the keras.io frontpage:


```python
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
```

<div class="k-default-codeblock">
```
26 30

```
</div>
Looks like our model thinks the answer is the span from tokens 1 to 12 (inclusive). No
prizes for guessing which tokens those are!


```python
answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
print(answer)
```

<div class="k-default-codeblock">
```
[ 8080   111  3014 20480  1116]

```
</div>
And now we can use the `tokenizer.decode()` method to turn those token IDs back into text:


```python
print(tokenizer.decode(answer))
```

<div class="k-default-codeblock">
```
consistent & simple APIs

```
</div>
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
