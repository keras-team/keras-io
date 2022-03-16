# Character-level recurrent sequence-to-sequence model

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2017/09/29<br>
**Last modified:** 2020/04/26<br>
**Description:** Character-level recurrent sequence-to-sequence model.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/lstm_seq2seq.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/lstm_seq2seq.py)



---
## Introduction

This example demonstrates how to implement a basic character-level
recurrent sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.


---
## Setup



```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

```

---
## Download the data



```python
!!curl -O http://www.manythings.org/anki/fra-eng.zip
!!unzip fra-eng.zip

```




<div class="k-default-codeblock">
```
['Archive:  fra-eng.zip',
 '  inflating: _about.txt              ',
 '  inflating: fra.txt                 ']

```
</div>
---
## Configuration



```python
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "fra.txt"

```

---
## Prepare the data



```python
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

```

<div class="k-default-codeblock">
```
Number of samples: 10000
Number of unique input tokens: 71
Number of unique output tokens: 93
Max sequence length for inputs: 16
Max sequence length for outputs: 59

```
</div>
---
## Build the model



```python
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

```

---
## Train the model



```python
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
model.save("s2s")

```

<div class="k-default-codeblock">
```
Epoch 1/100
125/125 [==============================] - 2s 16ms/step - loss: 1.1806 - accuracy: 0.7246 - val_loss: 1.0825 - val_accuracy: 0.6995
Epoch 2/100
125/125 [==============================] - 1s 11ms/step - loss: 0.8599 - accuracy: 0.7671 - val_loss: 0.8524 - val_accuracy: 0.7646
Epoch 3/100
125/125 [==============================] - 1s 11ms/step - loss: 0.6867 - accuracy: 0.8069 - val_loss: 0.7129 - val_accuracy: 0.7928
Epoch 4/100
125/125 [==============================] - 1s 11ms/step - loss: 0.5982 - accuracy: 0.8262 - val_loss: 0.6547 - val_accuracy: 0.8111
Epoch 5/100
125/125 [==============================] - 1s 11ms/step - loss: 0.5490 - accuracy: 0.8398 - val_loss: 0.6407 - val_accuracy: 0.8114
Epoch 6/100
125/125 [==============================] - 1s 11ms/step - loss: 0.5140 - accuracy: 0.8489 - val_loss: 0.5834 - val_accuracy: 0.8288
Epoch 7/100
125/125 [==============================] - 1s 11ms/step - loss: 0.4854 - accuracy: 0.8569 - val_loss: 0.5577 - val_accuracy: 0.8357
Epoch 8/100
125/125 [==============================] - 1s 11ms/step - loss: 0.4613 - accuracy: 0.8632 - val_loss: 0.5384 - val_accuracy: 0.8407
Epoch 9/100
125/125 [==============================] - 1s 11ms/step - loss: 0.4405 - accuracy: 0.8691 - val_loss: 0.5255 - val_accuracy: 0.8435
Epoch 10/100
125/125 [==============================] - 1s 11ms/step - loss: 0.4219 - accuracy: 0.8743 - val_loss: 0.5049 - val_accuracy: 0.8497
Epoch 11/100
125/125 [==============================] - 1s 11ms/step - loss: 0.4042 - accuracy: 0.8791 - val_loss: 0.4986 - val_accuracy: 0.8522
Epoch 12/100
125/125 [==============================] - 1s 11ms/step - loss: 0.3888 - accuracy: 0.8836 - val_loss: 0.4854 - val_accuracy: 0.8552
Epoch 13/100
125/125 [==============================] - 1s 11ms/step - loss: 0.3735 - accuracy: 0.8883 - val_loss: 0.4754 - val_accuracy: 0.8586
Epoch 14/100
125/125 [==============================] - 1s 11ms/step - loss: 0.3595 - accuracy: 0.8915 - val_loss: 0.4753 - val_accuracy: 0.8589
Epoch 15/100
125/125 [==============================] - 1s 11ms/step - loss: 0.3467 - accuracy: 0.8956 - val_loss: 0.4611 - val_accuracy: 0.8634
Epoch 16/100
125/125 [==============================] - 1s 11ms/step - loss: 0.3346 - accuracy: 0.8991 - val_loss: 0.4535 - val_accuracy: 0.8658
Epoch 17/100
125/125 [==============================] - 1s 11ms/step - loss: 0.3231 - accuracy: 0.9025 - val_loss: 0.4504 - val_accuracy: 0.8665
Epoch 18/100
125/125 [==============================] - 1s 11ms/step - loss: 0.3120 - accuracy: 0.9059 - val_loss: 0.4442 - val_accuracy: 0.8699
Epoch 19/100
125/125 [==============================] - 1s 10ms/step - loss: 0.3015 - accuracy: 0.9088 - val_loss: 0.4439 - val_accuracy: 0.8692
Epoch 20/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2917 - accuracy: 0.9118 - val_loss: 0.4415 - val_accuracy: 0.8712
Epoch 21/100
125/125 [==============================] - 1s 10ms/step - loss: 0.2821 - accuracy: 0.9147 - val_loss: 0.4372 - val_accuracy: 0.8722
Epoch 22/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2731 - accuracy: 0.9174 - val_loss: 0.4424 - val_accuracy: 0.8713
Epoch 23/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2642 - accuracy: 0.9201 - val_loss: 0.4371 - val_accuracy: 0.8725
Epoch 24/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2561 - accuracy: 0.9226 - val_loss: 0.4400 - val_accuracy: 0.8728
Epoch 25/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2481 - accuracy: 0.9245 - val_loss: 0.4358 - val_accuracy: 0.8757
Epoch 26/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2404 - accuracy: 0.9270 - val_loss: 0.4407 - val_accuracy: 0.8746
Epoch 27/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2332 - accuracy: 0.9294 - val_loss: 0.4462 - val_accuracy: 0.8736
Epoch 28/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2263 - accuracy: 0.9310 - val_loss: 0.4436 - val_accuracy: 0.8736
Epoch 29/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2194 - accuracy: 0.9328 - val_loss: 0.4411 - val_accuracy: 0.8755
Epoch 30/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2126 - accuracy: 0.9351 - val_loss: 0.4457 - val_accuracy: 0.8755
Epoch 31/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2069 - accuracy: 0.9370 - val_loss: 0.4498 - val_accuracy: 0.8752
Epoch 32/100
125/125 [==============================] - 1s 11ms/step - loss: 0.2010 - accuracy: 0.9388 - val_loss: 0.4518 - val_accuracy: 0.8755
Epoch 33/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1953 - accuracy: 0.9404 - val_loss: 0.4545 - val_accuracy: 0.8758
Epoch 34/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1897 - accuracy: 0.9423 - val_loss: 0.4547 - val_accuracy: 0.8769
Epoch 35/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1846 - accuracy: 0.9435 - val_loss: 0.4582 - val_accuracy: 0.8763
Epoch 36/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1794 - accuracy: 0.9451 - val_loss: 0.4653 - val_accuracy: 0.8755
Epoch 37/100
125/125 [==============================] - 1s 10ms/step - loss: 0.1747 - accuracy: 0.9464 - val_loss: 0.4633 - val_accuracy: 0.8768
Epoch 38/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1700 - accuracy: 0.9479 - val_loss: 0.4665 - val_accuracy: 0.8772
Epoch 39/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1657 - accuracy: 0.9493 - val_loss: 0.4725 - val_accuracy: 0.8755
Epoch 40/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1612 - accuracy: 0.9504 - val_loss: 0.4799 - val_accuracy: 0.8752
Epoch 41/100
125/125 [==============================] - 1s 10ms/step - loss: 0.1576 - accuracy: 0.9516 - val_loss: 0.4777 - val_accuracy: 0.8760
Epoch 42/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1531 - accuracy: 0.9530 - val_loss: 0.4842 - val_accuracy: 0.8761
Epoch 43/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1495 - accuracy: 0.9542 - val_loss: 0.4879 - val_accuracy: 0.8761
Epoch 44/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1456 - accuracy: 0.9552 - val_loss: 0.4933 - val_accuracy: 0.8757
Epoch 45/100
125/125 [==============================] - 1s 10ms/step - loss: 0.1419 - accuracy: 0.9562 - val_loss: 0.4988 - val_accuracy: 0.8753
Epoch 46/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1385 - accuracy: 0.9574 - val_loss: 0.5012 - val_accuracy: 0.8758
Epoch 47/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1356 - accuracy: 0.9581 - val_loss: 0.5040 - val_accuracy: 0.8763
Epoch 48/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1325 - accuracy: 0.9591 - val_loss: 0.5114 - val_accuracy: 0.8761
Epoch 49/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1291 - accuracy: 0.9601 - val_loss: 0.5151 - val_accuracy: 0.8764
Epoch 50/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1263 - accuracy: 0.9607 - val_loss: 0.5214 - val_accuracy: 0.8761
Epoch 51/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1232 - accuracy: 0.9621 - val_loss: 0.5210 - val_accuracy: 0.8759
Epoch 52/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1205 - accuracy: 0.9626 - val_loss: 0.5232 - val_accuracy: 0.8761
Epoch 53/100
125/125 [==============================] - 1s 10ms/step - loss: 0.1177 - accuracy: 0.9633 - val_loss: 0.5329 - val_accuracy: 0.8754
Epoch 54/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1152 - accuracy: 0.9644 - val_loss: 0.5317 - val_accuracy: 0.8753
Epoch 55/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1132 - accuracy: 0.9648 - val_loss: 0.5418 - val_accuracy: 0.8748
Epoch 56/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1102 - accuracy: 0.9658 - val_loss: 0.5456 - val_accuracy: 0.8745
Epoch 57/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1083 - accuracy: 0.9663 - val_loss: 0.5438 - val_accuracy: 0.8753
Epoch 58/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1058 - accuracy: 0.9669 - val_loss: 0.5519 - val_accuracy: 0.8753
Epoch 59/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1035 - accuracy: 0.9675 - val_loss: 0.5543 - val_accuracy: 0.8753
Epoch 60/100
125/125 [==============================] - 1s 11ms/step - loss: 0.1017 - accuracy: 0.9679 - val_loss: 0.5619 - val_accuracy: 0.8756
Epoch 61/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0993 - accuracy: 0.9686 - val_loss: 0.5680 - val_accuracy: 0.8751
Epoch 62/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0975 - accuracy: 0.9690 - val_loss: 0.5768 - val_accuracy: 0.8737
Epoch 63/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0954 - accuracy: 0.9697 - val_loss: 0.5800 - val_accuracy: 0.8733
Epoch 64/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0936 - accuracy: 0.9700 - val_loss: 0.5782 - val_accuracy: 0.8744
Epoch 65/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0918 - accuracy: 0.9709 - val_loss: 0.5832 - val_accuracy: 0.8743
Epoch 66/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0897 - accuracy: 0.9714 - val_loss: 0.5863 - val_accuracy: 0.8744
Epoch 67/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0880 - accuracy: 0.9718 - val_loss: 0.5912 - val_accuracy: 0.8742
Epoch 68/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0863 - accuracy: 0.9722 - val_loss: 0.5972 - val_accuracy: 0.8741
Epoch 69/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0850 - accuracy: 0.9727 - val_loss: 0.5969 - val_accuracy: 0.8743
Epoch 70/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0832 - accuracy: 0.9732 - val_loss: 0.6046 - val_accuracy: 0.8736
Epoch 71/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0815 - accuracy: 0.9738 - val_loss: 0.6037 - val_accuracy: 0.8746
Epoch 72/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0799 - accuracy: 0.9741 - val_loss: 0.6092 - val_accuracy: 0.8744
Epoch 73/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0785 - accuracy: 0.9746 - val_loss: 0.6118 - val_accuracy: 0.8750
Epoch 74/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0769 - accuracy: 0.9751 - val_loss: 0.6150 - val_accuracy: 0.8737
Epoch 75/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0753 - accuracy: 0.9754 - val_loss: 0.6196 - val_accuracy: 0.8736
Epoch 76/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0742 - accuracy: 0.9759 - val_loss: 0.6237 - val_accuracy: 0.8738
Epoch 77/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0731 - accuracy: 0.9760 - val_loss: 0.6310 - val_accuracy: 0.8731
Epoch 78/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0719 - accuracy: 0.9765 - val_loss: 0.6335 - val_accuracy: 0.8746
Epoch 79/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0702 - accuracy: 0.9770 - val_loss: 0.6366 - val_accuracy: 0.8744
Epoch 80/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0692 - accuracy: 0.9773 - val_loss: 0.6368 - val_accuracy: 0.8745
Epoch 81/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0678 - accuracy: 0.9777 - val_loss: 0.6472 - val_accuracy: 0.8735
Epoch 82/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0669 - accuracy: 0.9778 - val_loss: 0.6474 - val_accuracy: 0.8735
Epoch 83/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0653 - accuracy: 0.9783 - val_loss: 0.6466 - val_accuracy: 0.8745
Epoch 84/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0645 - accuracy: 0.9787 - val_loss: 0.6576 - val_accuracy: 0.8733
Epoch 85/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0633 - accuracy: 0.9790 - val_loss: 0.6539 - val_accuracy: 0.8742
Epoch 86/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0626 - accuracy: 0.9792 - val_loss: 0.6609 - val_accuracy: 0.8738
Epoch 87/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0614 - accuracy: 0.9794 - val_loss: 0.6641 - val_accuracy: 0.8739
Epoch 88/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0602 - accuracy: 0.9799 - val_loss: 0.6677 - val_accuracy: 0.8739
Epoch 89/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0594 - accuracy: 0.9801 - val_loss: 0.6659 - val_accuracy: 0.8731
Epoch 90/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0581 - accuracy: 0.9803 - val_loss: 0.6744 - val_accuracy: 0.8740
Epoch 91/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0575 - accuracy: 0.9806 - val_loss: 0.6722 - val_accuracy: 0.8737
Epoch 92/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0568 - accuracy: 0.9808 - val_loss: 0.6778 - val_accuracy: 0.8737
Epoch 93/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0557 - accuracy: 0.9814 - val_loss: 0.6837 - val_accuracy: 0.8733
Epoch 94/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0548 - accuracy: 0.9814 - val_loss: 0.6906 - val_accuracy: 0.8732
Epoch 95/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0543 - accuracy: 0.9816 - val_loss: 0.6913 - val_accuracy: 0.8733
Epoch 96/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0536 - accuracy: 0.9816 - val_loss: 0.6955 - val_accuracy: 0.8723
Epoch 97/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0531 - accuracy: 0.9817 - val_loss: 0.7001 - val_accuracy: 0.8724
Epoch 98/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0521 - accuracy: 0.9821 - val_loss: 0.7017 - val_accuracy: 0.8738
Epoch 99/100
125/125 [==============================] - 1s 10ms/step - loss: 0.0512 - accuracy: 0.9822 - val_loss: 0.7069 - val_accuracy: 0.8731
Epoch 100/100
125/125 [==============================] - 1s 11ms/step - loss: 0.0506 - accuracy: 0.9826 - val_loss: 0.7050 - val_accuracy: 0.8726
WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow/python/training/tracking/tracking.py:105: Network.state_updates (from tensorflow.python.keras.engine.network) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
INFO:tensorflow:Assets written to: s2s/assets

```
</div>
---
## Run inference (sampling)

1. encode input and retrieve initial decoder state
2. run one step of decoder with this initial state
and a "start of sequence" token as target.
Output will be the next target token.
3. Repeat with the current target token and current states



```python
# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("s2s")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


```

You can now generate decoded sentences as such:



```python
for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)

```

<div class="k-default-codeblock">
```
-
Input sentence: Go.
Decoded sentence: Va !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Hi.
Decoded sentence: Salut !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Hi.
Decoded sentence: Salut !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Run!
Decoded sentence: Cours !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Run!
Decoded sentence: Cours !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Who?
Decoded sentence: Qui ?
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Wow!
Decoded sentence: Ça alors !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Fire!
Decoded sentence: Au feu !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Help!
Decoded sentence: À l'aide !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Jump.
Decoded sentence: Saute.
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Stop!
Decoded sentence: Stop !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Stop!
Decoded sentence: Stop !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Stop!
Decoded sentence: Stop !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Wait!
Decoded sentence: Attendez !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Wait!
Decoded sentence: Attendez !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Go on.
Decoded sentence: Poursuis.
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Go on.
Decoded sentence: Poursuis.
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Go on.
Decoded sentence: Poursuis.
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Hello!
Decoded sentence: Salut !
```
</div>
    
<div class="k-default-codeblock">
```
-
Input sentence: Hello!
Decoded sentence: Salut !
```
</div>


