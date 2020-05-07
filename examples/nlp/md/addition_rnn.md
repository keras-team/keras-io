# Sequence to sequence learning for performing number addition

**Author:** [Smerity](https://twitter.com/Smerity) and others<br>
**Date created:** 2015/08/17<br>
**Last modified:** 2020/04/17<br>
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
[Sequence to Sequence Learning with Neural Networks](

 http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

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
from tensorflow import keras
from tensorflow.keras import layers
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
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
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
model.add(layers.LSTM(128, input_shape=(MAXLEN, len(chars))))
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
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 128)               72192     
_________________________________________________________________
repeat_vector (RepeatVector) (None, 4, 128)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 4, 128)            131584    
_________________________________________________________________
dense (Dense)                (None, 4, 12)             1548      
=================================================================
Total params: 205,324
Trainable params: 205,324
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Train the model



```python
epochs = 30
batch_size = 32

# For colorful logging.
class colors:
    ok = "\033[92m"
    fail = "\033[91m"
    close = "\033[0m"


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
        preds = np.argmax(model.predict(rowx), axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print("Q", q[::-1] if REVERSE else q, end=" ")
        print("T", correct, end=" ")
        if correct == guess:
            print(colors.ok + "☑" + colors.close, end=" ")
        else:
            print(colors.fail + "☒" + colors.close, end=" ")
        print(guess)

```

    
<div class="k-default-codeblock">
```
Iteration 1
1407/1407 [==============================] - 9s 6ms/step - loss: 1.7611 - accuracy: 0.3561 - val_loss: 1.5781 - val_accuracy: 0.4063
Q 73+205  T 278  [91m☒[0m 386 
Q 829+890 T 1719 [91m☒[0m 1688
Q 507+855 T 1362 [91m☒[0m 1368
Q 34+774  T 808  [91m☒[0m 488 
Q 1+351   T 352  [91m☒[0m 111 
Q 938+1   T 939  [91m☒[0m 901 
Q 33+551  T 584  [91m☒[0m 468 
Q 50+361  T 411  [91m☒[0m 261 
Q 715+352 T 1067 [91m☒[0m 1366
Q 85+473  T 558  [91m☒[0m 581 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 2
1407/1407 [==============================] - 8s 6ms/step - loss: 1.3420 - accuracy: 0.4974 - val_loss: 1.1634 - val_accuracy: 0.5644
Q 299+761 T 1060 [91m☒[0m 105 
Q 829+890 T 1719 [91m☒[0m 1685
Q 203+796 T 999  [91m☒[0m 105 
Q 63+573  T 636  [91m☒[0m 645 
Q 194+37  T 231  [91m☒[0m 204 
Q 813+444 T 1257 [91m☒[0m 1157
Q 49+974  T 1023 [91m☒[0m 104 
Q 16+82   T 98   [91m☒[0m 11  
Q 730+267 T 997  [91m☒[0m 905 
Q 9+87    T 96   [91m☒[0m 80  
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 3
1407/1407 [==============================] - 8s 6ms/step - loss: 1.0113 - accuracy: 0.6249 - val_loss: 0.9255 - val_accuracy: 0.6507
Q 282+96  T 378  [91m☒[0m 381 
Q 59+262  T 321  [91m☒[0m 313 
Q 99+248  T 347  [91m☒[0m 332 
Q 1+606   T 607  [91m☒[0m 601 
Q 955+43  T 998  [91m☒[0m 990 
Q 272+491 T 763  [91m☒[0m 751 
Q 717+721 T 1438 [91m☒[0m 1463
Q 348+635 T 983  [91m☒[0m 988 
Q 49+651  T 700  [91m☒[0m 693 
Q 458+334 T 792  [91m☒[0m 791 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 4
1407/1407 [==============================] - 8s 6ms/step - loss: 0.8382 - accuracy: 0.6931 - val_loss: 0.7930 - val_accuracy: 0.7067
Q 123+50  T 173  [91m☒[0m 177 
Q 834+1   T 835  [92m☑[0m 835 
Q 337+8   T 345  [91m☒[0m 343 
Q 383+4   T 387  [91m☒[0m 389 
Q 60+544  T 604  [91m☒[0m 507 
Q 842+793 T 1635 [91m☒[0m 1643
Q 43+22   T 65   [91m☒[0m 69  
Q 40+784  T 824  [91m☒[0m 826 
Q 9+168   T 177  [92m☑[0m 177 
Q 715+112 T 827  [91m☒[0m 830 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 5
1407/1407 [==============================] - 8s 6ms/step - loss: 0.7372 - accuracy: 0.7319 - val_loss: 0.6791 - val_accuracy: 0.7559
Q 582+1   T 583  [91m☒[0m 581 
Q 83+921  T 1004 [91m☒[0m 1003
Q 277+71  T 348  [91m☒[0m 349 
Q 42+382  T 424  [91m☒[0m 428 
Q 675+236 T 911  [91m☒[0m 912 
Q 370+29  T 399  [91m☒[0m 398 
Q 381+518 T 899  [91m☒[0m 808 
Q 874+9   T 883  [91m☒[0m 882 
Q 259+72  T 331  [91m☒[0m 338 
Q 89+464  T 553  [91m☒[0m 558 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 6
1407/1407 [==============================] - 8s 6ms/step - loss: 0.5745 - accuracy: 0.7899 - val_loss: 0.4565 - val_accuracy: 0.8423
Q 512+58  T 570  [92m☑[0m 570 
Q 4+497   T 501  [92m☑[0m 501 
Q 3+724   T 727  [92m☑[0m 727 
Q 22+52   T 74   [91m☒[0m 73  
Q 44+93   T 137  [91m☒[0m 136 
Q 32+240  T 272  [92m☑[0m 272 
Q 599+2   T 601  [92m☑[0m 601 
Q 842+602 T 1444 [91m☒[0m 1334
Q 45+981  T 1026 [92m☑[0m 1026
Q 141+218 T 359  [92m☑[0m 359 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 7
1407/1407 [==============================] - 8s 6ms/step - loss: 0.2913 - accuracy: 0.9099 - val_loss: 0.2197 - val_accuracy: 0.9323
Q 980+41  T 1021 [91m☒[0m 1012
Q 24+284  T 308  [92m☑[0m 308 
Q 918+862 T 1780 [92m☑[0m 1780
Q 35+46   T 81   [92m☑[0m 81  
Q 617+72  T 689  [92m☑[0m 689 
Q 73+532  T 605  [92m☑[0m 605 
Q 35+581  T 616  [92m☑[0m 616 
Q 97+672  T 769  [91m☒[0m 779 
Q 49+47   T 96   [91m☒[0m 95  
Q 257+33  T 290  [92m☑[0m 290 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 8
1407/1407 [==============================] - 8s 6ms/step - loss: 0.1562 - accuracy: 0.9592 - val_loss: 0.1451 - val_accuracy: 0.9559
Q 542+803 T 1345 [91m☒[0m 1344
Q 80+147  T 227  [92m☑[0m 227 
Q 79+25   T 104  [92m☑[0m 104 
Q 953+72  T 1025 [92m☑[0m 1025
Q 769+62  T 831  [92m☑[0m 831 
Q 974+800 T 1774 [92m☑[0m 1774
Q 3+429   T 432  [92m☑[0m 432 
Q 13+650  T 663  [92m☑[0m 663 
Q 798+73  T 871  [92m☑[0m 871 
Q 344+79  T 423  [92m☑[0m 423 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 9
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0950 - accuracy: 0.9771 - val_loss: 0.0845 - val_accuracy: 0.9772
Q 51+989  T 1040 [91m☒[0m 1030
Q 220+885 T 1105 [92m☑[0m 1105
Q 222+61  T 283  [92m☑[0m 283 
Q 76+972  T 1048 [92m☑[0m 1048
Q 247+53  T 300  [92m☑[0m 300 
Q 715+256 T 971  [92m☑[0m 971 
Q 30+513  T 543  [92m☑[0m 543 
Q 87+60   T 147  [92m☑[0m 147 
Q 873+331 T 1204 [92m☑[0m 1204
Q 921+121 T 1042 [92m☑[0m 1042
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 10
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0730 - accuracy: 0.9818 - val_loss: 0.0516 - val_accuracy: 0.9878
Q 373+118 T 491  [92m☑[0m 491 
Q 73+852  T 925  [92m☑[0m 925 
Q 47+92   T 139  [92m☑[0m 139 
Q 670+51  T 721  [92m☑[0m 721 
Q 397+59  T 456  [92m☑[0m 456 
Q 847+33  T 880  [92m☑[0m 880 
Q 68+140  T 208  [92m☑[0m 208 
Q 67+65   T 132  [92m☑[0m 132 
Q 12+765  T 777  [92m☑[0m 777 
Q 472+891 T 1363 [92m☑[0m 1363
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 11
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0569 - accuracy: 0.9855 - val_loss: 0.0495 - val_accuracy: 0.9872
Q 541+0   T 541  [92m☑[0m 541 
Q 10+226  T 236  [92m☑[0m 236 
Q 682+462 T 1144 [92m☑[0m 1144
Q 765+273 T 1038 [92m☑[0m 1038
Q 3+688   T 691  [92m☑[0m 691 
Q 6+648   T 654  [92m☑[0m 654 
Q 4+180   T 184  [92m☑[0m 184 
Q 83+3    T 86   [92m☑[0m 86  
Q 786+58  T 844  [92m☑[0m 844 
Q 36+233  T 269  [92m☑[0m 269 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 12
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0511 - accuracy: 0.9871 - val_loss: 0.0687 - val_accuracy: 0.9780
Q 24+782  T 806  [92m☑[0m 806 
Q 2+448   T 450  [92m☑[0m 450 
Q 99+26   T 125  [92m☑[0m 125 
Q 1+66    T 67   [92m☑[0m 67  
Q 8+87    T 95   [92m☑[0m 95  
Q 871+36  T 907  [92m☑[0m 907 
Q 205+137 T 342  [92m☑[0m 342 
Q 725+454 T 1179 [92m☑[0m 1179
Q 172+283 T 455  [92m☑[0m 455 
Q 922+694 T 1616 [92m☑[0m 1616
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 13
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0383 - accuracy: 0.9897 - val_loss: 0.0562 - val_accuracy: 0.9823
Q 370+254 T 624  [92m☑[0m 624 
Q 27+196  T 223  [92m☑[0m 223 
Q 370+29  T 399  [92m☑[0m 399 
Q 983+95  T 1078 [91m☒[0m 1068
Q 990+52  T 1042 [92m☑[0m 1042
Q 620+78  T 698  [92m☑[0m 698 
Q 700+29  T 729  [92m☑[0m 729 
Q 19+274  T 293  [92m☑[0m 293 
Q 49+872  T 921  [92m☑[0m 921 
Q 297+78  T 375  [92m☑[0m 375 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 14
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0332 - accuracy: 0.9915 - val_loss: 0.0673 - val_accuracy: 0.9756
Q 631+78  T 709  [92m☑[0m 709 
Q 71+32   T 103  [91m☒[0m 104 
Q 218+838 T 1056 [92m☑[0m 1056
Q 804+662 T 1466 [92m☑[0m 1466
Q 160+651 T 811  [92m☑[0m 811 
Q 1+898   T 899  [91m☒[0m 999 
Q 750+79  T 829  [92m☑[0m 829 
Q 894+2   T 896  [91m☒[0m 897 
Q 104+473 T 577  [92m☑[0m 577 
Q 400+703 T 1103 [91m☒[0m 1113
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 15
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0367 - accuracy: 0.9896 - val_loss: 0.0597 - val_accuracy: 0.9811
Q 933+72  T 1005 [92m☑[0m 1005
Q 35+46   T 81   [92m☑[0m 81  
Q 964+289 T 1253 [91m☒[0m 1153
Q 25+291  T 316  [92m☑[0m 316 
Q 31+4    T 35   [92m☑[0m 35  
Q 9+337   T 346  [92m☑[0m 346 
Q 346+0   T 346  [92m☑[0m 346 
Q 190+56  T 246  [92m☑[0m 246 
Q 802+82  T 884  [92m☑[0m 884 
Q 23+150  T 173  [92m☑[0m 173 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 16
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0260 - accuracy: 0.9931 - val_loss: 0.1539 - val_accuracy: 0.9498
Q 822+81  T 903  [92m☑[0m 903 
Q 589+68  T 657  [92m☑[0m 657 
Q 17+375  T 392  [92m☑[0m 392 
Q 783+94  T 877  [91m☒[0m 867 
Q 431+92  T 523  [92m☑[0m 523 
Q 251+869 T 1120 [92m☑[0m 1120
Q 19+274  T 293  [92m☑[0m 293 
Q 249+785 T 1034 [92m☑[0m 1034
Q 331+82  T 413  [92m☑[0m 413 
Q 33+33   T 66   [92m☑[0m 66  
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 17
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0397 - accuracy: 0.9892 - val_loss: 0.0281 - val_accuracy: 0.9924
Q 674+68  T 742  [92m☑[0m 742 
Q 856+18  T 874  [92m☑[0m 874 
Q 79+29   T 108  [92m☑[0m 108 
Q 73+905  T 978  [92m☑[0m 978 
Q 42+33   T 75   [92m☑[0m 75  
Q 102+241 T 343  [92m☑[0m 343 
Q 340+792 T 1132 [92m☑[0m 1132
Q 476+826 T 1302 [92m☑[0m 1302
Q 500+537 T 1037 [92m☑[0m 1037
Q 147+278 T 425  [92m☑[0m 425 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 18
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0216 - accuracy: 0.9941 - val_loss: 0.0508 - val_accuracy: 0.9833
Q 290+586 T 876  [92m☑[0m 876 
Q 42+789  T 831  [92m☑[0m 831 
Q 983+40  T 1023 [92m☑[0m 1023
Q 302+525 T 827  [92m☑[0m 827 
Q 722+808 T 1530 [92m☑[0m 1530
Q 566+84  T 650  [92m☑[0m 650 
Q 89+373  T 462  [92m☑[0m 462 
Q 318+247 T 565  [92m☑[0m 565 
Q 799+7   T 806  [92m☑[0m 806 
Q 292+34  T 326  [92m☑[0m 326 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 19
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0264 - accuracy: 0.9932 - val_loss: 0.0214 - val_accuracy: 0.9940
Q 10+82   T 92   [92m☑[0m 92  
Q 20+89   T 109  [92m☑[0m 109 
Q 87+917  T 1004 [92m☑[0m 1004
Q 65+43   T 108  [92m☑[0m 108 
Q 17+279  T 296  [92m☑[0m 296 
Q 98+273  T 371  [92m☑[0m 371 
Q 849+837 T 1686 [92m☑[0m 1686
Q 64+296  T 360  [92m☑[0m 360 
Q 91+605  T 696  [92m☑[0m 696 
Q 3+374   T 377  [92m☑[0m 377 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 20
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0149 - accuracy: 0.9963 - val_loss: 0.0690 - val_accuracy: 0.9790
Q 229+90  T 319  [92m☑[0m 319 
Q 82+382  T 464  [92m☑[0m 464 
Q 12+681  T 693  [92m☑[0m 693 
Q 616+147 T 763  [92m☑[0m 763 
Q 73+80   T 153  [92m☑[0m 153 
Q 6+434   T 440  [92m☑[0m 440 
Q 322+4   T 326  [92m☑[0m 326 
Q 468+9   T 477  [92m☑[0m 477 
Q 34+709  T 743  [92m☑[0m 743 
Q 31+645  T 676  [92m☑[0m 676 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 21
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0206 - accuracy: 0.9944 - val_loss: 0.0062 - val_accuracy: 0.9989
Q 591+2   T 593  [92m☑[0m 593 
Q 339+17  T 356  [92m☑[0m 356 
Q 98+694  T 792  [92m☑[0m 792 
Q 80+165  T 245  [92m☑[0m 245 
Q 5+410   T 415  [92m☑[0m 415 
Q 5+163   T 168  [92m☑[0m 168 
Q 902+21  T 923  [92m☑[0m 923 
Q 818+2   T 820  [92m☑[0m 820 
Q 954+254 T 1208 [92m☑[0m 1208
Q 428+92  T 520  [92m☑[0m 520 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 22
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0343 - accuracy: 0.9901 - val_loss: 0.0215 - val_accuracy: 0.9934
Q 672+79  T 751  [92m☑[0m 751 
Q 82+3    T 85   [92m☑[0m 85  
Q 0+698   T 698  [92m☑[0m 698 
Q 564+9   T 573  [92m☑[0m 573 
Q 79+665  T 744  [92m☑[0m 744 
Q 212+19  T 231  [92m☑[0m 231 
Q 81+50   T 131  [92m☑[0m 131 
Q 80+147  T 227  [92m☑[0m 227 
Q 495+0   T 495  [92m☑[0m 495 
Q 268+92  T 360  [92m☑[0m 360 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 23
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0152 - accuracy: 0.9961 - val_loss: 0.0506 - val_accuracy: 0.9829
Q 61+19   T 80   [92m☑[0m 80  
Q 7+584   T 591  [92m☑[0m 591 
Q 3+724   T 727  [92m☑[0m 727 
Q 44+311  T 355  [92m☑[0m 355 
Q 497+27  T 524  [92m☑[0m 524 
Q 672+79  T 751  [92m☑[0m 751 
Q 42+65   T 107  [92m☑[0m 107 
Q 74+195  T 269  [92m☑[0m 269 
Q 1+101   T 102  [92m☑[0m 102 
Q 6+431   T 437  [91m☒[0m 438 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 24
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0234 - accuracy: 0.9937 - val_loss: 0.0195 - val_accuracy: 0.9951
Q 740+40  T 780  [92m☑[0m 780 
Q 525+874 T 1399 [92m☑[0m 1399
Q 699+7   T 706  [92m☑[0m 706 
Q 459+862 T 1321 [92m☑[0m 1321
Q 50+590  T 640  [92m☑[0m 640 
Q 522+885 T 1407 [92m☑[0m 1407
Q 9+861   T 870  [92m☑[0m 870 
Q 0+304   T 304  [92m☑[0m 304 
Q 116+76  T 192  [92m☑[0m 192 
Q 82+382  T 464  [92m☑[0m 464 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 25
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0130 - accuracy: 0.9967 - val_loss: 0.0349 - val_accuracy: 0.9889
Q 468+9   T 477  [92m☑[0m 477 
Q 701+72  T 773  [92m☑[0m 773 
Q 10+36   T 46   [92m☑[0m 46  
Q 849+9   T 858  [92m☑[0m 858 
Q 469+683 T 1152 [92m☑[0m 1152
Q 18+885  T 903  [92m☑[0m 903 
Q 560+112 T 672  [92m☑[0m 672 
Q 681+31  T 712  [92m☑[0m 712 
Q 2+323   T 325  [92m☑[0m 325 
Q 27+414  T 441  [92m☑[0m 441 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 26
1407/1407 [==============================] - 9s 6ms/step - loss: 0.0187 - accuracy: 0.9946 - val_loss: 0.0313 - val_accuracy: 0.9900
Q 849+9   T 858  [92m☑[0m 858 
Q 27+944  T 971  [92m☑[0m 971 
Q 955+43  T 998  [92m☑[0m 998 
Q 502+181 T 683  [92m☑[0m 683 
Q 37+49   T 86   [92m☑[0m 86  
Q 1+294   T 295  [92m☑[0m 295 
Q 211+915 T 1126 [92m☑[0m 1126
Q 928+390 T 1318 [92m☑[0m 1318
Q 936+251 T 1187 [92m☑[0m 1187
Q 325+76  T 401  [92m☑[0m 401 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 27
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0147 - accuracy: 0.9962 - val_loss: 0.0256 - val_accuracy: 0.9919
Q 6+788   T 794  [92m☑[0m 794 
Q 229+459 T 688  [92m☑[0m 688 
Q 27+13   T 40   [92m☑[0m 40  
Q 200+55  T 255  [92m☑[0m 255 
Q 22+23   T 45   [92m☑[0m 45  
Q 761+864 T 1625 [92m☑[0m 1625
Q 99+886  T 985  [92m☑[0m 985 
Q 917+4   T 921  [92m☑[0m 921 
Q 770+838 T 1608 [92m☑[0m 1608
Q 769+62  T 831  [92m☑[0m 831 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 28
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0150 - accuracy: 0.9958 - val_loss: 0.0061 - val_accuracy: 0.9987
Q 87+754  T 841  [92m☑[0m 841 
Q 962+16  T 978  [92m☑[0m 978 
Q 142+39  T 181  [92m☑[0m 181 
Q 44+331  T 375  [92m☑[0m 375 
Q 1+882   T 883  [92m☑[0m 883 
Q 5+563   T 568  [92m☑[0m 568 
Q 612+98  T 710  [92m☑[0m 710 
Q 63+713  T 776  [92m☑[0m 776 
Q 964+667 T 1631 [92m☑[0m 1631
Q 51+691  T 742  [92m☑[0m 742 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 29
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0187 - accuracy: 0.9946 - val_loss: 0.0138 - val_accuracy: 0.9958
Q 70+651  T 721  [92m☑[0m 721 
Q 984+0   T 984  [92m☑[0m 984 
Q 50+64   T 114  [92m☑[0m 114 
Q 79+58   T 137  [92m☑[0m 137 
Q 936+251 T 1187 [92m☑[0m 1187
Q 42+477  T 519  [92m☑[0m 519 
Q 51+136  T 187  [92m☑[0m 187 
Q 444+95  T 539  [92m☑[0m 539 
Q 36+323  T 359  [92m☑[0m 359 
Q 504+545 T 1049 [92m☑[0m 1049

```
</div>
You'll get to 99+% validation accuracy after ~30 epochs.

