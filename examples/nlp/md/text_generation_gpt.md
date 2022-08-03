# simple GPT text generation with KerasNLP

**Author:** [Jesse Chan](https://github.com/jessechancy)<br>
**Date created:** 2022/07/25<br>
**Last modified:** 2022/07/25<br>
**Description:** Using KerasNLP to train a mini-GPT model for text generation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_generation_gpt.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_generation_gpt.py)



---
## Introduction

In this example, we will use KerasNLP to build a scaled down Generative
Pre-Trained (GPT) model. GPT is a Transformer-based model that allows you to generate
sophisticated text from a prompt.

We will train the model on the [simplebooks-92](https://arxiv.org/abs/1911.12391) corpus,
which is a dataset made from several novels. It is a good dataset for this example since
it has a small vocabulary and high word frequency, which is beneficial when training a
model with few parameters.

This example combines concepts from
[Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)
with KerasNLP abstractions. We will demonstrate how KerasNLP tokenization, layers and
metrics simplify the training
process, and then show how to generate output text using the KerasNLP sampling utilities.

Note: If you are running this example on a Colab,
make sure to enable GPU runtime for faster training.

This example requires KerasNLP. You can install it via the following command:
`pip install keras-nlp`

---
## Imports


```python
import os
import keras_nlp
import tensorflow as tf
from tensorflow import keras
```

---
## Settings & hyperparameters


```python
# Data
BATCH_SIZE = 64
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 100

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # Limits parameters in model.

# Training
LEARNING_RATE = 5e-4
EPOCHS = 6

# Inference
NUM_TOKENS_TO_GENERATE = 80
```

---
## Load the data

Now, let's download the dataset! The SimpleBooks dataset consists of 1,573 Gutenberg books, and has
one of the smallest vocabulary size to word-level tokens ratio. It has a vocabulary size of ~98k,
a third of WikiText-103's, with around the same number of tokens (~100M). This makes it easy to fit a small model.


```python
keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

# Load simplebooks-92 train set and filter out short lines.
raw_train_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# Load simplebooks-92 validation set and filter out short lines.
raw_val_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
)
```

<div class="k-default-codeblock">
```
2022-08-02 23:27:07.182636: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.193250: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.193979: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.195812: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-08-02 23:27:07.196715: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.197370: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.197995: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.814282: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.815055: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.815694: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-08-02 23:27:07.816286: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13795 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5

```
</div>
---
## Train the tokenizer

We train the tokenizer from the training dataset for a vocabulary size of `VOCAB_SIZE`,
which is a tuned hyperparameter. We want to limit the vocabulary as much as possible, as
we will see later on
that it has a large affect on the number of model parameters. We also don't want to include
*too few* vocabulary terms, or there would be too many out-of-vocabulary (OOV) sub-words. In
addition, three tokens are reserved in the vocabulary:

- `"[PAD]"` for padding sequences to `SEQ_LEN`. This token has index 0 in both
`reserved_tokens` and `vocab`, since `WordPieceTokenizer` (and other layers) consider
`0`/`vocab[0]` as the default padding.
- `"[UNK]"` for OOV sub-words, which should match the default `oov_token="[UNK]"` in
`WordPieceTokenizer`.
- `"[BOS]"` stands for beginning of sentence, but here technically it is a token
representing the beginning of each line of training data.


```python
# Train tokenizer vocabulary
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)
```

---
## Load tokenizer

We use the vocabulary data to initialize
`keras_nlp.tokenizers.WordPieceTokenizer`. WordPieceTokenizer is an efficient
implementation of the WordPiece algorithm used by BERT and other models. It will strip,
lower-case and do other irreversible preprocessing operations.


```python
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab, sequence_length=SEQ_LEN
)
```

---
## Tokenize data

We preprocess the dataset by tokenizing and splitting it into `features` and `labels`.


```python
# packer adds a start token
start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


# Tokenize and split into train and label sequences.
train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
```

---
## Build the model

We create our scaled down GPT model with the following layers:

- One `keras_nlp.layers.TokenAndPositionEmbedding` layer, which combines the embedding
for the token and its position.
- Multiple `keras_nlp.layers.TransformerDecoder` layers, with the default causal masking.
The layer has no cross-attention when run with decoder sequence only.
- One final dense linear layer


```python
inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
# Embedding.
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)
# Transformer decoders.
for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)  # Giving one argument only skips cross-attention.
# Output.
outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])
```

Let's take a look at our model summary - a large majority of the
parameters are in the `token_and_position_embedding` and the output `dense` layer!
This means that the vocabulary size (`VOCAB_SIZE`) has a large affect on the size of the model,
while the number of Transformer decoder layers (`NUM_LAYERS`) doesn't affect it as much.


```python
model.summary()
```

<div class="k-default-codeblock">
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None)]            0         
                                                                 
 token_and_position_embeddin  (None, None, 256)        1312768   
 g (TokenAndPositionEmbeddin                                     
 g)                                                              
                                                                 
 transformer_decoder (Transf  (None, None, 256)        394749    
 ormerDecoder)                                                   
                                                                 
 transformer_decoder_1 (Tran  (None, None, 256)        394749    
 sformerDecoder)                                                 
                                                                 
 dense (Dense)               (None, None, 5000)        1285000   
                                                                 
=================================================================
Total params: 3,387,266
Trainable params: 3,387,266
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Training

Now that we have our model, let's train it with the `fit()` method.


```python
model.fit(train_ds, validation_data=val_ds, verbose=2, epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/6
18802/18802 - 1232s - loss: 2.1755 - perplexity: 52.7088 - val_loss: 1.5022 - val_perplexity: 40.0212 - 1232s/epoch - 66ms/step
Epoch 2/6
18802/18802 - 1230s - loss: 2.0320 - perplexity: 40.5656 - val_loss: 1.4593 - val_perplexity: 36.1564 - 1230s/epoch - 65ms/step
Epoch 3/6
18802/18802 - 1230s - loss: 2.0009 - perplexity: 38.3450 - val_loss: 1.4493 - val_perplexity: 35.1694 - 1230s/epoch - 65ms/step
Epoch 4/6
18802/18802 - 1230s - loss: 1.9853 - perplexity: 37.2722 - val_loss: 1.4428 - val_perplexity: 34.6659 - 1230s/epoch - 65ms/step
Epoch 5/6
18802/18802 - 1230s - loss: 1.9758 - perplexity: 36.6308 - val_loss: 1.4373 - val_perplexity: 34.2467 - 1230s/epoch - 65ms/step
Epoch 6/6
18802/18802 - 1230s - loss: 1.9689 - perplexity: 36.1800 - val_loss: 1.4237 - val_perplexity: 33.1127 - 1230s/epoch - 65ms/step

<keras.callbacks.History at 0x7f75c04c6b90>

```
</div>
---
## Inference

With our trained model, we can test it out to gauge it's performance. Since this model is
built with a `"[BOS]"` token, we can have an empty starting prompt for text generation.


```python
# Unpadded bos token.
prompt_tokens = tf.convert_to_tensor([tokenizer.token_to_id("[BOS]")])
```

We will use the `keras_nlp.utils` module for inference. Every text generation
utility requires a `token_logits_fn()` wrapper around the model. This wrapper takes
in an unpadded token sequence, and requires the logits of the next token as the output.


```python

def token_logits_fn(inputs):
    cur_len = inputs.shape[1]
    output = model(inputs)
    return output[:, cur_len - 1, :]  # return next token logits

```

Creating the wrapper function is the most complex part of using these functions. Now that
it's done, let's test out the different utilties, starting with greedy search.

### Greedy search

We greedily pick the most probable token at each timestep. In other words, we get the
argmax of the model output.


```python
output_tokens = keras_nlp.utils.greedy_search(
    token_logits_fn,
    prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Greedy search generated text: 
b'[BOS] " i \' m going to be a little girl , " said the doctor , " and i \' m going to be a nice little girl . i \' m going to be a nice little girl , and i \' m going to be a nice little girl . " [PAD] said she to herself , " and i \' m going to be a nice little girl . " [PAD] , " she said ,'
```
</div>
    


As you can see, greedy search starts out making some sense, but quickly starts repeating
itself. This is a common problem with text generation that can be fixed by some of the
probabilistic text generation utilities shown later on!

### Beam search

At a high-level, beam search keeps track of the `num_beams` most probable sequences at
each timestep, and predicts the best next token from all sequences. It is an improvement
over greedy search since it stores more possibilities. However, it is less efficient than
greedy search since it has to compute and store multiple potential sequences.

**Note:** beam search with `num_beams=1` is identical to greedy search.


```python
output_tokens = keras_nlp.utils.beam_search(
    token_logits_fn,
    prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    num_beams=10,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Beam search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Beam search generated text: 
b'[BOS] " i don \' t know , " he said , " but i don \' t know what i \' m going to do . i don \' t know what i \' m going to do . i don \' t know what i \' m going to do , but i \' m going to do it , and i \' m going to do it . " [PAD] went on , " he went on'
```
</div>
    


Similar to greedy search, beam search quickly starts repeating itself, since it is still
a deterministic method.

### Random search

Random search is our first probabilistic method. At each time step, it samples the next
token using the softmax probabilities provided by the model.


```python
output_tokens = keras_nlp.utils.random_search(
    token_logits_fn,
    prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Random search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Random search generated text: 
b'[BOS] they brought down their own anchorage , and nipped out a strong knock overboard . harry went a rough de pillow , but there was not any noise ; and george mumbled over the bags . [PAD] was it warmer gas till they were sure to lock the door , when the alarm was ploughin that afforded shelter . a pile of torch opened the door , just then a'
```
</div>
    


Voila, no repetitions! However, with random search, we may see some nonsensical words
appearing since any word in the vocabulary has a chance of appearing with this sampling
method. This is fixed by our next search utility, top-k search.

### Top-K search

Similar to random search, we sample the next token from the probability distribution
provided by the model. The only difference is that here, we select out the top `k` most
probable tokens, and distribute the probabiltiy mass over them before sampling. This way,
we won't be sampling from low probability tokens, and hence we would have less
nonsensical words!


```python
output_tokens = keras_nlp.utils.top_k_search(
    token_logits_fn,
    prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    k=10,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Top-K search generated text: 
b'[BOS] it is a trifle late that you will find to have a profuse communicated , so you \' ve had a very continent in a telegraphic , and i have a very small note to miss thayer . but there , in one of you , i know you have seen it . " there will be two other elect'
```
</div>
    


### Top-P search

Even with the top-k search, there is something to improve upon. With top-k search, the
number `k` is fixed, which means it selects the same number of tokens for any probability
distribution. Consider two scenarios, one where the probability mass is concentrated over
2 words and another where the probability mass is evenly concentrated across 10. Should
we choose `k=2` or `k=10`? There is not a one size fits all `k` here.

This is where top-p search comes in! Instead of choosing a `k`, we choose a probability
`p` that we want the probabilities of the top tokens to sum up to. This way, we can
dynamically adjust the `k` based on the probability distribution. By setting `p=0.9`, if
90% of the probability mass is concentrated on the top 2 tokens, we can filter out the
top 2 tokens to sample from. If instead the 90% is distributed over 10 tokens, it will
similarly filter out the top 10 tokens to sample from.


```python
output_tokens = keras_nlp.utils.top_p_search(
    token_logits_fn,
    prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    p=0.5,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Top-P search generated text: 
b'[BOS] " you have , mr . rosalind , " said the colonel , with a smile , " i \' m sorry you don \' t know anything about it . " [PAD] is that a commissioner , the man of the town . [PAD] had a letter , too , that he had no children to pay for the family , and that he would have given up to the family of all'
```
</div>
    


### Using callbacks for text generation

We can also wrap the utilities in a callback, which allows you to print out a prediction
sequence for every epoch of the model! Here is an example of a callback for top-k search:


```python

class TopKTextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model using top-k."""

    def __init__(self, k):
        self.k = k

    def on_epoch_end(self, epoch, logs=None):
        output_tokens = keras_nlp.utils.top_k_search(
            token_logits_fn,
            prompt_tokens,
            max_length=NUM_TOKENS_TO_GENERATE,
            k=self.k,
            from_logits=True,
        )
        txt = tokenizer.detokenize(output_tokens)
        print(f"Top-K search generated text: \n{txt}\n")


text_generation_callback = TopKTextGenerator(k=10)
# Dummy training loop to demonstrate callback.
model.fit(train_ds.take(1), verbose=2, epochs=2, callbacks=[text_generation_callback])
```

<div class="k-default-codeblock">
```
Epoch 1/2
Top-K search generated text: 
b'[BOS] " we have a good deal in the house , " said anne , and the others looked in the direction of the corridor . [PAD] , and it seemed as if a little more , that would have been to her , but , though she was not so sure whether to have been on her mind at all , or not , i \' m afraid . " but the young man said ,'
```
</div>
    
<div class="k-default-codeblock">
```
1/1 - 4s - loss: 1.5802 - perplexity: 32.5390 - 4s/epoch - 4s/step
Epoch 2/2
Top-K search generated text: 
b'[BOS] the commissioner was not a sensation of his own , for he would not have been in such a hurry to start for the city ; but the commissions had not been sent to the hospital for many days , and the men who had come , with their wives , and their wives and daughters in the evenings , and all the women , had given a special re'
```
</div>
    
<div class="k-default-codeblock">
```
1/1 - 4s - loss: 1.3344 - perplexity: 40.1874 - 4s/epoch - 4s/step

<keras.callbacks.History at 0x7f75657321d0>

```
</div>
---
## Conclusion

To recap, in this example, we use KerasNLP layers to train a sub-word vocabulary,
tokenize training data, create a miniature GPT model, and perform inference with the
text generation library.

If you would like to understand how Transformers work, or learn more about training the
full GPT model, here are some further readings:

- Attention Is All You Need [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- GPT-3 Paper [Brown et al., 2020](https://arxiv.org/abs/2005.14165)
