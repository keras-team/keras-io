"""
Title: Text Extraction with BERT
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/23
Last modified: 2026/03/12
Description: Fine tune pretrained BERT from HuggingFace Transformers on SQuAD.
Accelerator: TPU
Converted to Keras 3 by: [LakshmiKalaKadali](https://github.com/LakshmiKalaKadali)
"""

"""
## Introduction

This demonstration uses SQuAD (Stanford Question-Answering Dataset).
In SQuAD, an input consists of a question, and a paragraph for context.
The goal is to find the span of text in the paragraph that answers the question.
We evaluate our performance on this data with the "Exact Match" metric,
which measures the percentage of predictions that exactly match any one of the
ground-truth answers.

We fine-tune a BERT model to perform this task as follows:

1. Feed the context and the question as inputs to BERT.
2. Take two vectors S and T with dimensions equal to that of
   hidden states in BERT.
3. Compute the probability of each token being the start and end of
   the answer span. The probability of a token being the start of
   the answer is given by a dot product between S and the representation
   of the token in the last layer of BERT, followed by a softmax over all tokens.
   The probability of a token being the end of the answer is computed
   similarly with the vector T.
4. Fine-tune BERT and learn S and T along the way.

**References:**

- [BERT](https://arxiv.org/abs/1810.04805)
- [SQuAD](https://arxiv.org/abs/1606.05250)
"""
"""
## Setup
"""
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import re
import json
import string
import numpy as np
import keras
from keras import layers
from keras import ops
import keras_hub
from tokenizers import BertWordPieceTokenizer

# from transformers import BertTokenizer, TFBertModel, BertConfig

# --- CONFIGURATION ---
max_len = 384
BATCH_SIZE = 32  # Optimized for TPU memory
EPOCHS = 1
PRESET = "bert_base_en"
LEARNING_RATE = 3e-5

"""
## Set-up BERT tokenizer

We fetch the vocabulary directly from the keras_hub preset to ensure
that the Token IDs match the model weights exactly.
"""

tokenizer = keras_hub.models.BertTokenizer.from_preset(PRESET)
vocab = tokenizer.get_vocabulary()
vocab_dict = {word: i for i, word in enumerate(vocab)}
tokenizer = BertWordPieceTokenizer(vocab=vocab_dict, lowercase=True)


"""
## Load the data
"""
train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path = keras.utils.get_file("eval.json", eval_data_url)

"""
## Preprocess the data

1. Iterate through the JSON file and use the `SquadExample` helper class to process each record.
2. Implement `keras.utils.PyDataset` to stream the processed data in batches, ensuring high memory
   efficiency across JAX, PyTorch, and TensorFlow backends.
"""


class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False

    def preprocess(self):
        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())
        answer = " ".join(str(self.answer_text).split())
        end_char_idx = self.start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return
        is_char_in_ans = [0] * len(context)
        for idx in range(self.start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        tokenized_context = tokenizer.encode(context)
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)
        if len(ans_token_idx) == 0:
            self.skip = True
            return
        self.start_token_idx = ans_token_idx[0]
        self.end_token_idx = ans_token_idx[-1]
        tokenized_question = tokenizer.encode(question)
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)
        padding_length = max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        self.input_ids, self.token_type_ids, self.attention_mask = (
            input_ids,
            token_type_ids,
            attention_mask,
        )
        self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(raw_data):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                squad_eg = SquadExample(
                    qa["question"],
                    context,
                    qa["answers"][0]["answer_start"],
                    qa["answers"][0]["text"],
                    [_["text"] for _ in qa["answers"]],
                )
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples


with open(train_path) as f:
    raw_train_data = json.load(f)
with open(eval_path) as f:
    raw_eval_data = json.load(f)
train_squad_examples = create_squad_examples(raw_train_data)
eval_squad_examples = create_squad_examples(raw_eval_data)

"""
## PyDataset Implementation
Memory-safe streaming for JAX/Torch/TPU.
"""


class SQuADDataset(keras.utils.PyDataset):
    def __init__(self, squad_examples, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.examples = [e for e in squad_examples if not e.skip]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.examples) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.examples))
        batch_examples = self.examples[start:end]

        x = {
            "input_ids": np.array([e.input_ids for e in batch_examples], dtype="int32"),
            "token_type_ids": np.array(
                [e.token_type_ids for e in batch_examples], dtype="int32"
            ),
            "attention_mask": np.array(
                [e.attention_mask for e in batch_examples], dtype="int32"
            ),
        }

        y_start = np.array([e.start_token_idx for e in batch_examples], dtype="int32")
        y_end = np.array([e.end_token_idx for e in batch_examples], dtype="int32")

        return x, (y_start, y_end)


train_dataset = SQuADDataset(train_squad_examples, BATCH_SIZE)
valid_data_no_skip = [e for e in eval_squad_examples if not e.skip]
x_eval = {
    "input_ids": np.array([e.input_ids for e in valid_data_no_skip], dtype="int32"),
    "token_type_ids": np.array(
        [e.token_type_ids for e in valid_data_no_skip], dtype="int32"
    ),
    "attention_mask": np.array(
        [e.attention_mask for e in valid_data_no_skip], dtype="int32"
    ),
}
y_eval = (
    np.array([e.start_token_idx for e in valid_data_no_skip], dtype="int32"),
    np.array([e.end_token_idx for e in valid_data_no_skip], dtype="int32"),
)

print(f"{len(train_squad_examples)} training points created.")
print(f"{len(eval_squad_examples)} evaluation points created.")

"""
Create the Question-Answering Model using BERT and Functional API
"""


def create_model():
    backbone = keras_hub.models.BertBackbone.from_preset(PRESET)

    input_ids = layers.Input(shape=(max_len,), dtype="int32", name="input_ids")
    token_type_ids = layers.Input(
        shape=(max_len,), dtype="int32", name="token_type_ids"
    )
    attention_mask = layers.Input(
        shape=(max_len,), dtype="int32", name="attention_mask"
    )

    outputs = backbone(
        {
            "token_ids": input_ids,
            "segment_ids": token_type_ids,
            "padding_mask": attention_mask,
        }
    )
    embedding = outputs["sequence_output"]

    start_logits = layers.Dense(1, use_bias=False)(embedding)
    start_logits = layers.Reshape((max_len,), name="start_logit")(start_logits)
    end_logits = layers.Dense(1, use_bias=False)(embedding)
    end_logits = layers.Reshape((max_len,), name="end_logit")(end_logits)

    model = keras.Model(
        inputs={
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        },
        outputs=[start_logits, end_logits],
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE, global_clipnorm=1.0
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


"""
This code should preferably be run on Google Colab TPU runtime.
With Colab TPUs, each epoch will take 5-6 minutes.
"""
devices = keras.distribution.list_devices()
tpu_devices = [d for d in devices if "tpu" in d.lower()]
if tpu_devices:
    mesh = keras.distribution.DeviceMesh(
        shape=(len(tpu_devices),), axis_names=["batch"], devices=tpu_devices
    )
    distributor = keras.distribution.DataParallel(device_mesh=mesh)
    with distributor.scope():
        model = create_model()
else:
    model = create_model()
model.summary()

"""
## Create evaluation Callback

This callback will compute the exact match score using the validation data
after every epoch.
"""


def normalize_text(text):
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return " ".join(re.sub(regex, " ", text).split())


class ExactMatch(keras.callbacks.Callback):
    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start_logits, pred_end_logits = self.model.predict(
            self.x_eval, batch_size=32
        )
        pred_start = ops.convert_to_numpy(pred_start_logits)
        pred_end = ops.convert_to_numpy(pred_end_logits)

        count = 0
        for idx, (start_logits, end_logits) in enumerate(zip(pred_start, pred_end)):
            squad_eg = valid_data_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start, end = np.argmax(start_logits), np.argmax(end_logits)
            if start >= len(offsets):
                continue
            p_start = offsets[start][0]
            p_end = offsets[end][1] if end < len(offsets) else offsets[-1][1]
            if normalize_text(squad_eg.context[p_start:p_end]) in [
                normalize_text(a) for a in squad_eg.all_answers
            ]:
                count += 1
        print(f"\nepoch={epoch+1}, exact match score={count / len(self.y_eval[0]):.2f}")


"""
## Train and Evaluate
"""
exact_match_callback = ExactMatch(x_eval, y_eval)
model.fit(
    train_dataset, epochs=EPOCHS, verbose=1, callbacks=[ExactMatch(x_eval, y_eval)]
)


"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)
"""
