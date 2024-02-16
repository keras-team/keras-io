# Sequence to sequence learning for performing number addition

**Author:** [Smerity](https://twitter.com/Smerity) and others<br>
**Date created:** 2015/08/17<br>
**Last modified:** 2024/02/13<br>
**Description:** A model that learns to add strings of numbers, e.g. "535+61" -> "596".


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/addition_rnn.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/addition_rnn.py)



---
## Introduction

In this example, we train a model to learn to add two numbers, provided as strings.

**Example:**

- Input: "535+61"
- Output: "596"

Input may optionally be reversed, which was shown to increase performance in many tasks
 in: [Learning to Execute](http://arxiv.org/abs/1410.4615) and
[Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).

Theoretically, sequence order inversion introduces shorter term dependencies between
 source and target for this problem.

**Results:**

For two digits (reversed):

+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits (reversed):

+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits (reversed):

+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits (reversed):

+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

---
## Setup


```python
import keras
from keras import layers
import numpy as np

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS
```

---
## Generate the data


```python

class CharacterTable:
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)


# All the numbers, plus sign and space for padding.
chars = "0123456789+ "
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print("Generating data...")
while len(questions) < TRAINING_SIZE:
    f = lambda: int(
        "".join(
            np.random.choice(list("0123456789"))
            for i in range(np.random.randint(1, DIGITS + 1))
        )
    )
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    q = "{}+{}".format(a, b)
    query = q + " " * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1.
    ans += " " * (DIGITS + 1 - len(ans))
    if REVERSE:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print("Total questions:", len(questions))
```

<div class="k-default-codeblock">
```
Generating data...
Total questions: 50000

```
</div>
---
## Vectorize the data


```python
print("Vectorization...")
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print("Training Data:")
print(x_train.shape)
print(y_train.shape)

print("Validation Data:")
print(x_val.shape)
print(y_val.shape)
```

<div class="k-default-codeblock">
```
Vectorization...
Training Data:
(45000, 7, 12)
(45000, 4, 12)
Validation Data:
(5000, 7, 12)
(5000, 4, 12)

```
</div>
---
## Build the model


```python
print("Build model...")
num_layers = 1  # Try to add more LSTM layers!

model = keras.Sequential()
# "Encode" the input sequence using a LSTM, producing an output of size 128.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(layers.Input((MAXLEN, len(chars))))
model.add(layers.LSTM(128))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(num_layers):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(layers.LSTM(128, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.Dense(len(chars), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
```

<div class="k-default-codeblock">
```
Build model...

```
</div>
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)               │     <span style="color: #00af00; text-decoration-color: #00af00">72,192</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ repeat_vector (<span style="color: #0087ff; text-decoration-color: #0087ff">RepeatVector</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ lstm_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │    <span style="color: #00af00; text-decoration-color: #00af00">131,584</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>)             │      <span style="color: #00af00; text-decoration-color: #00af00">1,548</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">205,324</span> (802.05 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">205,324</span> (802.05 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



---
## Train the model


```python
# Training parameters.
epochs = 30
batch_size = 32

# Formatting characters for results display.
green_color = "\033[92m"
red_color = "\033[91m"
end_char = "\033[0m"

# Train the model each generation and show predictions against the validation
# dataset.
for epoch in range(1, epochs):
    print()
    print("Iteration", epoch)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1,
        validation_data=(x_val, y_val),
    )
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = np.argmax(model.predict(rowx, verbose=0), axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print("Q", q[::-1] if REVERSE else q, end=" ")
        print("T", correct, end=" ")
        if correct == guess:
            print(f"{green_color}☑ {guess}{end_char}")
        else:
            print(f"{red_color}☒ {guess}{end_char}")
```

    
<div class="k-default-codeblock">
```
Iteration 1
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 6ms/step - accuracy: 0.3258 - loss: 1.8801 - val_accuracy: 0.4268 - val_loss: 1.5506
Q 499+58  T 557  ☒ 511 
Q 51+638  T 689  ☒ 662 
Q 87+12   T 99   ☒ 11  
Q 259+55  T 314  ☒ 561 
Q 704+87  T 791  ☒ 811 
Q 988+67  T 1055 ☒ 101 
Q 94+116  T 210  ☒ 111 
Q 724+4   T 728  ☒ 777 
Q 8+673   T 681  ☒ 772 
Q 8+991   T 999  ☒ 900 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 2
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.4688 - loss: 1.4235 - val_accuracy: 0.5846 - val_loss: 1.1293
Q 379+6   T 385  ☒ 387 
Q 15+504  T 519  ☒ 525 
Q 552+299 T 851  ☒ 727 
Q 664+0   T 664  ☒ 667 
Q 500+257 T 757  ☒ 797 
Q 50+818  T 868  ☒ 861 
Q 310+691 T 1001 ☒ 900 
Q 378+548 T 926  ☒ 827 
Q 46+59   T 105  ☒ 122 
Q 49+817  T 866  ☒ 871 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 3
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.6053 - loss: 1.0648 - val_accuracy: 0.6665 - val_loss: 0.9070
Q 1+266   T 267  ☒ 260 
Q 73+257  T 330  ☒ 324 
Q 421+628 T 1049 ☒ 1022
Q 85+590  T 675  ☒ 660 
Q 66+34   T 100  ☒ 90  
Q 256+639 T 895  ☒ 890 
Q 6+677   T 683  ☑ 683 
Q 162+637 T 799  ☒ 792 
Q 5+324   T 329  ☒ 337 
Q 848+34  T 882  ☒ 889 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 4
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.6781 - loss: 0.8751 - val_accuracy: 0.7037 - val_loss: 0.8092
Q 677+1   T 678  ☒ 676 
Q 1+531   T 532  ☒ 535 
Q 699+60  T 759  ☒ 756 
Q 475+139 T 614  ☒ 616 
Q 327+592 T 919  ☒ 915 
Q 48+912  T 960  ☒ 956 
Q 520+78  T 598  ☒ 505 
Q 318+8   T 326  ☒ 327 
Q 914+53  T 967  ☒ 966 
Q 734+0   T 734  ☒ 733 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 5
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.7142 - loss: 0.7807 - val_accuracy: 0.7164 - val_loss: 0.7622
Q 150+337 T 487  ☒ 489 
Q 72+934  T 1006 ☒ 1005
Q 171+62  T 233  ☒ 231 
Q 108+21  T 129  ☒ 135 
Q 755+896 T 1651 ☒ 1754
Q 117+1   T 118  ☒ 119 
Q 148+95  T 243  ☒ 241 
Q 719+956 T 1675 ☒ 1684
Q 656+43  T 699  ☒ 695 
Q 368+8   T 376  ☒ 372 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 6
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.7377 - loss: 0.7157 - val_accuracy: 0.7541 - val_loss: 0.6684
Q 945+364 T 1309 ☒ 1305
Q 762+96  T 858  ☒ 855 
Q 5+650   T 655  ☑ 655 
Q 52+680  T 732  ☒ 735 
Q 77+724  T 801  ☒ 800 
Q 46+739  T 785  ☑ 785 
Q 843+43  T 886  ☒ 885 
Q 158+3   T 161  ☒ 160 
Q 426+711 T 1137 ☒ 1138
Q 157+41  T 198  ☒ 190 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 7
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.7642 - loss: 0.6462 - val_accuracy: 0.7955 - val_loss: 0.5433
Q 822+27  T 849  ☑ 849 
Q 82+495  T 577  ☒ 563 
Q 9+366   T 375  ☒ 373 
Q 9+598   T 607  ☒ 696 
Q 186+41  T 227  ☒ 226 
Q 920+920 T 1840 ☒ 1846
Q 445+345 T 790  ☒ 797 
Q 783+588 T 1371 ☒ 1360
Q 36+473  T 509  ☒ 502 
Q 354+61  T 415  ☒ 416 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 8
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.8326 - loss: 0.4626 - val_accuracy: 0.9069 - val_loss: 0.2744
Q 458+154 T 612  ☑ 612 
Q 309+19  T 328  ☑ 328 
Q 808+97  T 905  ☑ 905 
Q 28+736  T 764  ☑ 764 
Q 28+79   T 107  ☑ 107 
Q 44+84   T 128  ☒ 129 
Q 744+13  T 757  ☑ 757 
Q 24+996  T 1020 ☒ 1011
Q 8+193   T 201  ☒ 101 
Q 483+9   T 492  ☒ 491 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 9
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9365 - loss: 0.2275 - val_accuracy: 0.9657 - val_loss: 0.1393
Q 330+61  T 391  ☑ 391 
Q 207+82  T 289  ☒ 299 
Q 23+234  T 257  ☑ 257 
Q 690+567 T 1257 ☑ 1257
Q 293+97  T 390  ☒ 380 
Q 312+868 T 1180 ☑ 1180
Q 956+40  T 996  ☑ 996 
Q 97+105  T 202  ☒ 203 
Q 365+44  T 409  ☑ 409 
Q 76+639  T 715  ☑ 715 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 10
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - accuracy: 0.9717 - loss: 0.1223 - val_accuracy: 0.9744 - val_loss: 0.0965
Q 123+143 T 266  ☑ 266 
Q 599+1   T 600  ☑ 600 
Q 729+237 T 966  ☑ 966 
Q 51+120  T 171  ☑ 171 
Q 97+672  T 769  ☑ 769 
Q 840+5   T 845  ☑ 845 
Q 86+494  T 580  ☒ 570 
Q 278+51  T 329  ☑ 329 
Q 8+832   T 840  ☑ 840 
Q 383+9   T 392  ☑ 392 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 11
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - accuracy: 0.9842 - loss: 0.0729 - val_accuracy: 0.9808 - val_loss: 0.0690
Q 181+923 T 1104 ☑ 1104
Q 747+24  T 771  ☑ 771 
Q 6+65    T 71   ☑ 71  
Q 75+994  T 1069 ☑ 1069
Q 712+587 T 1299 ☑ 1299
Q 977+10  T 987  ☑ 987 
Q 742+24  T 766  ☑ 766 
Q 215+44  T 259  ☑ 259 
Q 817+683 T 1500 ☑ 1500
Q 102+48  T 150  ☒ 140 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 12
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9820 - loss: 0.0695 - val_accuracy: 0.9823 - val_loss: 0.0596
Q 819+885 T 1704 ☒ 1604
Q 34+20   T 54   ☑ 54  
Q 9+996   T 1005 ☑ 1005
Q 915+811 T 1726 ☑ 1726
Q 166+640 T 806  ☑ 806 
Q 229+82  T 311  ☑ 311 
Q 1+418   T 419  ☑ 419 
Q 552+28  T 580  ☑ 580 
Q 279+733 T 1012 ☑ 1012
Q 756+734 T 1490 ☑ 1490
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 13
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9836 - loss: 0.0587 - val_accuracy: 0.9941 - val_loss: 0.0296
Q 793+0   T 793  ☑ 793 
Q 79+48   T 127  ☑ 127 
Q 484+92  T 576  ☑ 576 
Q 39+655  T 694  ☑ 694 
Q 64+708  T 772  ☑ 772 
Q 568+341 T 909  ☑ 909 
Q 9+918   T 927  ☑ 927 
Q 48+912  T 960  ☑ 960 
Q 31+289  T 320  ☑ 320 
Q 378+548 T 926  ☑ 926 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 14
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9915 - loss: 0.0353 - val_accuracy: 0.9901 - val_loss: 0.0358
Q 318+8   T 326  ☒ 325 
Q 886+63  T 949  ☒ 959 
Q 77+8    T 85   ☑ 85  
Q 418+40  T 458  ☑ 458 
Q 30+32   T 62   ☑ 62  
Q 541+93  T 634  ☑ 634 
Q 6+7     T 13   ☒ 14  
Q 670+74  T 744  ☑ 744 
Q 97+57   T 154  ☑ 154 
Q 60+13   T 73   ☑ 73  
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 15
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9911 - loss: 0.0335 - val_accuracy: 0.9934 - val_loss: 0.0262
Q 24+533  T 557  ☑ 557 
Q 324+44  T 368  ☑ 368 
Q 63+505  T 568  ☑ 568 
Q 670+74  T 744  ☑ 744 
Q 58+359  T 417  ☑ 417 
Q 16+428  T 444  ☑ 444 
Q 17+99   T 116  ☑ 116 
Q 779+903 T 1682 ☑ 1682
Q 40+576  T 616  ☑ 616 
Q 947+773 T 1720 ☑ 1720
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 16
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9968 - loss: 0.0175 - val_accuracy: 0.9901 - val_loss: 0.0360
Q 315+155 T 470  ☑ 470 
Q 594+950 T 1544 ☑ 1544
Q 372+37  T 409  ☑ 409 
Q 537+47  T 584  ☑ 584 
Q 8+263   T 271  ☑ 271 
Q 81+500  T 581  ☑ 581 
Q 75+270  T 345  ☑ 345 
Q 0+796   T 796  ☑ 796 
Q 655+965 T 1620 ☑ 1620
Q 384+1   T 385  ☑ 385 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 17
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9972 - loss: 0.0148 - val_accuracy: 0.9924 - val_loss: 0.0278
Q 168+83  T 251  ☑ 251 
Q 951+53  T 1004 ☑ 1004
Q 400+37  T 437  ☑ 437 
Q 996+473 T 1469 ☒ 1569
Q 996+847 T 1843 ☑ 1843
Q 842+550 T 1392 ☑ 1392
Q 479+72  T 551  ☑ 551 
Q 753+782 T 1535 ☑ 1535
Q 99+188  T 287  ☑ 287 
Q 2+974   T 976  ☑ 976 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 18
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - accuracy: 0.9929 - loss: 0.0258 - val_accuracy: 0.9973 - val_loss: 0.0135
Q 380+62  T 442  ☑ 442 
Q 774+305 T 1079 ☑ 1079
Q 248+272 T 520  ☑ 520 
Q 479+736 T 1215 ☑ 1215
Q 859+743 T 1602 ☑ 1602
Q 667+20  T 687  ☑ 687 
Q 932+56  T 988  ☑ 988 
Q 740+31  T 771  ☑ 771 
Q 588+88  T 676  ☑ 676 
Q 109+57  T 166  ☑ 166 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 19
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9977 - loss: 0.0116 - val_accuracy: 0.9571 - val_loss: 0.1416
Q 635+89  T 724  ☑ 724 
Q 50+818  T 868  ☑ 868 
Q 37+622  T 659  ☑ 659 
Q 913+49  T 962  ☑ 962 
Q 641+962 T 1603 ☒ 1503
Q 11+626  T 637  ☑ 637 
Q 20+405  T 425  ☑ 425 
Q 667+208 T 875  ☑ 875 
Q 89+794  T 883  ☑ 883 
Q 234+55  T 289  ☑ 289 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 20
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9947 - loss: 0.0194 - val_accuracy: 0.9967 - val_loss: 0.0136
Q 5+777   T 782  ☑ 782 
Q 1+266   T 267  ☑ 267 
Q 579+1   T 580  ☑ 580 
Q 665+6   T 671  ☑ 671 
Q 210+546 T 756  ☑ 756 
Q 660+86  T 746  ☑ 746 
Q 75+349  T 424  ☑ 424 
Q 984+36  T 1020 ☑ 1020
Q 4+367   T 371  ☑ 371 
Q 249+213 T 462  ☑ 462 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 21
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - accuracy: 0.9987 - loss: 0.0081 - val_accuracy: 0.9840 - val_loss: 0.0481
Q 228+95  T 323  ☑ 323 
Q 72+18   T 90   ☑ 90  
Q 34+687  T 721  ☑ 721 
Q 932+0   T 932  ☑ 932 
Q 933+54  T 987  ☑ 987 
Q 735+455 T 1190 ☑ 1190
Q 790+70  T 860  ☑ 860 
Q 416+36  T 452  ☒ 462 
Q 194+110 T 304  ☑ 304 
Q 349+70  T 419  ☑ 419 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 22
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 40s 28ms/step - accuracy: 0.9902 - loss: 0.0326 - val_accuracy: 0.9947 - val_loss: 0.0190
Q 95+237  T 332  ☑ 332 
Q 5+188   T 193  ☑ 193 
Q 19+931  T 950  ☑ 950 
Q 38+499  T 537  ☑ 537 
Q 25+21   T 46   ☑ 46  
Q 55+85   T 140  ☑ 140 
Q 555+7   T 562  ☑ 562 
Q 83+873  T 956  ☑ 956 
Q 95+527  T 622  ☑ 622 
Q 556+558 T 1114 ☑ 1114
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 23
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9835 - loss: 0.0572 - val_accuracy: 0.9962 - val_loss: 0.0141
Q 48+413  T 461  ☑ 461 
Q 71+431  T 502  ☑ 502 
Q 892+534 T 1426 ☑ 1426
Q 934+201 T 1135 ☑ 1135
Q 898+967 T 1865 ☒ 1855
Q 958+0   T 958  ☑ 958 
Q 23+179  T 202  ☑ 202 
Q 138+60  T 198  ☑ 198 
Q 718+5   T 723  ☑ 723 
Q 816+514 T 1330 ☑ 1330
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 24
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 20s 14ms/step - accuracy: 0.9932 - loss: 0.0255 - val_accuracy: 0.9932 - val_loss: 0.0243
Q 4+583   T 587  ☑ 587 
Q 49+466  T 515  ☑ 515 
Q 920+26  T 946  ☑ 946 
Q 624+813 T 1437 ☑ 1437
Q 87+315  T 402  ☑ 402 
Q 368+73  T 441  ☑ 441 
Q 86+833  T 919  ☑ 919 
Q 528+423 T 951  ☑ 951 
Q 0+705   T 705  ☑ 705 
Q 581+928 T 1509 ☑ 1509
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 25
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9908 - loss: 0.0303 - val_accuracy: 0.9944 - val_loss: 0.0169
Q 107+34  T 141  ☑ 141 
Q 998+90  T 1088 ☑ 1088
Q 71+520  T 591  ☑ 591 
Q 91+996  T 1087 ☑ 1087
Q 94+69   T 163  ☑ 163 
Q 108+21  T 129  ☑ 129 
Q 785+60  T 845  ☑ 845 
Q 71+628  T 699  ☑ 699 
Q 294+9   T 303  ☑ 303 
Q 399+34  T 433  ☑ 433 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 26
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.9965 - loss: 0.0139 - val_accuracy: 0.9979 - val_loss: 0.0094
Q 19+133  T 152  ☑ 152 
Q 841+3   T 844  ☑ 844 
Q 698+6   T 704  ☑ 704 
Q 942+28  T 970  ☑ 970 
Q 81+735  T 816  ☑ 816 
Q 325+14  T 339  ☑ 339 
Q 790+64  T 854  ☑ 854 
Q 4+839   T 843  ☑ 843 
Q 505+96  T 601  ☑ 601 
Q 917+42  T 959  ☑ 959 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 27
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 72s 51ms/step - accuracy: 0.9952 - loss: 0.0173 - val_accuracy: 0.9992 - val_loss: 0.0036
Q 71+628  T 699  ☑ 699 
Q 791+9   T 800  ☑ 800 
Q 19+148  T 167  ☑ 167 
Q 7+602   T 609  ☑ 609 
Q 6+566   T 572  ☑ 572 
Q 437+340 T 777  ☑ 777 
Q 614+533 T 1147 ☑ 1147
Q 948+332 T 1280 ☑ 1280
Q 56+619  T 675  ☑ 675 
Q 86+251  T 337  ☑ 337 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 28
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9964 - loss: 0.0124 - val_accuracy: 0.9990 - val_loss: 0.0047
Q 2+572   T 574  ☑ 574 
Q 437+96  T 533  ☑ 533 
Q 15+224  T 239  ☑ 239 
Q 16+655  T 671  ☑ 671 
Q 714+5   T 719  ☑ 719 
Q 645+417 T 1062 ☑ 1062
Q 25+919  T 944  ☑ 944 
Q 89+329  T 418  ☑ 418 
Q 22+513  T 535  ☑ 535 
Q 497+983 T 1480 ☑ 1480
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 29
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - accuracy: 0.9970 - loss: 0.0106 - val_accuracy: 0.9990 - val_loss: 0.0048
Q 2+962   T 964  ☑ 964 
Q 6+76    T 82   ☑ 82  
Q 986+20  T 1006 ☑ 1006
Q 727+49  T 776  ☑ 776 
Q 948+332 T 1280 ☑ 1280
Q 921+463 T 1384 ☑ 1384
Q 77+556  T 633  ☑ 633 
Q 133+849 T 982  ☑ 982 
Q 301+478 T 779  ☑ 779 
Q 3+243   T 246  ☑ 246 

```
</div>
You'll get to 99+% validation accuracy after ~30 epochs.
