# Imbalanced classification: credit card fraud detection

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2019/05/28<br>
**Last modified:** 2020/04/17<br>
**Description:** Demonstration of how to handle highly imbalanced classification problems.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/imbalanced_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/imbalanced_classification.py)



---
## Introduction

This example looks at the
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud/)
dataset to demonstrate how
to train a classification model on data with highly imbalanced classes.

---
## First, vectorize the CSV data


```python
import csv
import numpy as np

# Get the real data from https://www.kaggle.com/mlg-ulb/creditcardfraud/
fname = "/Users/fchollet/Downloads/creditcard.csv"

all_features = []
all_targets = []
with open(fname) as f:
    for i, line in enumerate(f):
        if i == 0:
            print("HEADER:", line.strip())
            continue  # Skip header
        fields = line.strip().split(",")
        all_features.append([float(v.replace('"', "")) for v in fields[:-1]])
        all_targets.append([int(fields[-1].replace('"', ""))])
        if i == 1:
            print("EXAMPLE FEATURES:", all_features[-1])

features = np.array(all_features, dtype="float32")
targets = np.array(all_targets, dtype="uint8")
print("features.shape:", features.shape)
print("targets.shape:", targets.shape)

```

<div class="k-default-codeblock">
```
HEADER: "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"
EXAMPLE FEATURES: [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]
features.shape: (284807, 30)
targets.shape: (284807, 1)

```
</div>
---
## Prepare a validation set


```python
num_val_samples = int(len(features) * 0.2)
train_features = features[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_features = features[-num_val_samples:]
val_targets = targets[-num_val_samples:]

print("Number of training samples:", len(train_features))
print("Number of validation samples:", len(val_features))

```

<div class="k-default-codeblock">
```
Number of training samples: 227846
Number of validation samples: 56961

```
</div>
---
## Analyze class imbalance in the targets


```python
counts = np.bincount(train_targets[:, 0])
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(train_targets)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]

```

<div class="k-default-codeblock">
```
Number of positive samples in training data: 417 (0.18% of total)

```
</div>
---
## Normalize the data using training set statistics


```python
mean = np.mean(train_features, axis=0)
train_features -= mean
val_features -= mean
std = np.std(train_features, axis=0)
train_features /= std
val_features /= std

```

---
## Build a binary classification model


```python
from tensorflow import keras

model = keras.Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=(train_features.shape[-1],)
        ),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

```

<div class="k-default-codeblock">
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               7936      
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 257       
=================================================================
Total params: 139,777
Trainable params: 139,777
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Train the model with `class_weight` argument


```python
metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
)

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]
class_weight = {0: weight_for_0, 1: weight_for_1}

model.fit(
    train_features,
    train_targets,
    batch_size=2048,
    epochs=30,
    verbose=2,
    callbacks=callbacks,
    validation_data=(val_features, val_targets),
    class_weight=class_weight,
)

```

<div class="k-default-codeblock">
```
Epoch 1/30
112/112 - 2s - loss: 2.4210e-06 - fn: 51.0000 - fp: 29417.0000 - tn: 198012.0000 - tp: 366.0000 - precision: 0.0123 - recall: 0.8777 - val_loss: 0.0759 - val_fn: 9.0000 - val_fp: 611.0000 - val_tn: 56275.0000 - val_tp: 66.0000 - val_precision: 0.0975 - val_recall: 0.8800
Epoch 2/30
112/112 - 2s - loss: 1.4337e-06 - fn: 35.0000 - fp: 7058.0000 - tn: 220371.0000 - tp: 382.0000 - precision: 0.0513 - recall: 0.9161 - val_loss: 0.1632 - val_fn: 6.0000 - val_fp: 2343.0000 - val_tn: 54543.0000 - val_tp: 69.0000 - val_precision: 0.0286 - val_recall: 0.9200
Epoch 3/30
112/112 - 2s - loss: 1.2100e-06 - fn: 27.0000 - fp: 7382.0000 - tn: 220047.0000 - tp: 390.0000 - precision: 0.0502 - recall: 0.9353 - val_loss: 0.1882 - val_fn: 5.0000 - val_fp: 3690.0000 - val_tn: 53196.0000 - val_tp: 70.0000 - val_precision: 0.0186 - val_recall: 0.9333
Epoch 4/30
112/112 - 2s - loss: 1.0770e-06 - fn: 24.0000 - fp: 7306.0000 - tn: 220123.0000 - tp: 393.0000 - precision: 0.0510 - recall: 0.9424 - val_loss: 0.0444 - val_fn: 9.0000 - val_fp: 674.0000 - val_tn: 56212.0000 - val_tp: 66.0000 - val_precision: 0.0892 - val_recall: 0.8800
Epoch 5/30
112/112 - 2s - loss: 9.3284e-07 - fn: 18.0000 - fp: 5607.0000 - tn: 221822.0000 - tp: 399.0000 - precision: 0.0664 - recall: 0.9568 - val_loss: 0.0455 - val_fn: 8.0000 - val_fp: 604.0000 - val_tn: 56282.0000 - val_tp: 67.0000 - val_precision: 0.0999 - val_recall: 0.8933
Epoch 6/30
112/112 - 2s - loss: 8.9186e-07 - fn: 21.0000 - fp: 6917.0000 - tn: 220512.0000 - tp: 396.0000 - precision: 0.0542 - recall: 0.9496 - val_loss: 0.0385 - val_fn: 9.0000 - val_fp: 462.0000 - val_tn: 56424.0000 - val_tp: 66.0000 - val_precision: 0.1250 - val_recall: 0.8800
Epoch 7/30
112/112 - 2s - loss: 6.4562e-07 - fn: 13.0000 - fp: 5878.0000 - tn: 221551.0000 - tp: 404.0000 - precision: 0.0643 - recall: 0.9688 - val_loss: 0.0205 - val_fn: 9.0000 - val_fp: 372.0000 - val_tn: 56514.0000 - val_tp: 66.0000 - val_precision: 0.1507 - val_recall: 0.8800
Epoch 8/30
112/112 - 2s - loss: 7.3378e-07 - fn: 15.0000 - fp: 6825.0000 - tn: 220604.0000 - tp: 402.0000 - precision: 0.0556 - recall: 0.9640 - val_loss: 0.0188 - val_fn: 10.0000 - val_fp: 246.0000 - val_tn: 56640.0000 - val_tp: 65.0000 - val_precision: 0.2090 - val_recall: 0.8667
Epoch 9/30
112/112 - 2s - loss: 5.1385e-07 - fn: 9.0000 - fp: 5265.0000 - tn: 222164.0000 - tp: 408.0000 - precision: 0.0719 - recall: 0.9784 - val_loss: 0.0244 - val_fn: 11.0000 - val_fp: 495.0000 - val_tn: 56391.0000 - val_tp: 64.0000 - val_precision: 0.1145 - val_recall: 0.8533
Epoch 10/30
112/112 - 2s - loss: 8.6498e-07 - fn: 13.0000 - fp: 8506.0000 - tn: 218923.0000 - tp: 404.0000 - precision: 0.0453 - recall: 0.9688 - val_loss: 0.0177 - val_fn: 11.0000 - val_fp: 367.0000 - val_tn: 56519.0000 - val_tp: 64.0000 - val_precision: 0.1485 - val_recall: 0.8533
Epoch 11/30
112/112 - 2s - loss: 6.0585e-07 - fn: 12.0000 - fp: 6676.0000 - tn: 220753.0000 - tp: 405.0000 - precision: 0.0572 - recall: 0.9712 - val_loss: 0.0356 - val_fn: 9.0000 - val_fp: 751.0000 - val_tn: 56135.0000 - val_tp: 66.0000 - val_precision: 0.0808 - val_recall: 0.8800
Epoch 12/30
112/112 - 2s - loss: 6.0788e-07 - fn: 9.0000 - fp: 6219.0000 - tn: 221210.0000 - tp: 408.0000 - precision: 0.0616 - recall: 0.9784 - val_loss: 0.0249 - val_fn: 10.0000 - val_fp: 487.0000 - val_tn: 56399.0000 - val_tp: 65.0000 - val_precision: 0.1178 - val_recall: 0.8667
Epoch 13/30
112/112 - 3s - loss: 8.3899e-07 - fn: 12.0000 - fp: 6612.0000 - tn: 220817.0000 - tp: 405.0000 - precision: 0.0577 - recall: 0.9712 - val_loss: 0.0905 - val_fn: 5.0000 - val_fp: 2159.0000 - val_tn: 54727.0000 - val_tp: 70.0000 - val_precision: 0.0314 - val_recall: 0.9333
Epoch 14/30
112/112 - 3s - loss: 6.0584e-07 - fn: 8.0000 - fp: 6823.0000 - tn: 220606.0000 - tp: 409.0000 - precision: 0.0566 - recall: 0.9808 - val_loss: 0.0205 - val_fn: 10.0000 - val_fp: 446.0000 - val_tn: 56440.0000 - val_tp: 65.0000 - val_precision: 0.1272 - val_recall: 0.8667
Epoch 15/30
112/112 - 2s - loss: 3.9569e-07 - fn: 6.0000 - fp: 3820.0000 - tn: 223609.0000 - tp: 411.0000 - precision: 0.0971 - recall: 0.9856 - val_loss: 0.0212 - val_fn: 10.0000 - val_fp: 413.0000 - val_tn: 56473.0000 - val_tp: 65.0000 - val_precision: 0.1360 - val_recall: 0.8667
Epoch 16/30
112/112 - 2s - loss: 5.4548e-07 - fn: 5.0000 - fp: 3910.0000 - tn: 223519.0000 - tp: 412.0000 - precision: 0.0953 - recall: 0.9880 - val_loss: 0.0906 - val_fn: 8.0000 - val_fp: 1905.0000 - val_tn: 54981.0000 - val_tp: 67.0000 - val_precision: 0.0340 - val_recall: 0.8933
Epoch 17/30
112/112 - 3s - loss: 6.2734e-07 - fn: 8.0000 - fp: 6005.0000 - tn: 221424.0000 - tp: 409.0000 - precision: 0.0638 - recall: 0.9808 - val_loss: 0.0161 - val_fn: 10.0000 - val_fp: 340.0000 - val_tn: 56546.0000 - val_tp: 65.0000 - val_precision: 0.1605 - val_recall: 0.8667
Epoch 18/30
112/112 - 3s - loss: 4.9752e-07 - fn: 5.0000 - fp: 4302.0000 - tn: 223127.0000 - tp: 412.0000 - precision: 0.0874 - recall: 0.9880 - val_loss: 0.0186 - val_fn: 10.0000 - val_fp: 408.0000 - val_tn: 56478.0000 - val_tp: 65.0000 - val_precision: 0.1374 - val_recall: 0.8667
Epoch 19/30
112/112 - 3s - loss: 6.7296e-07 - fn: 5.0000 - fp: 5986.0000 - tn: 221443.0000 - tp: 412.0000 - precision: 0.0644 - recall: 0.9880 - val_loss: 0.0165 - val_fn: 10.0000 - val_fp: 276.0000 - val_tn: 56610.0000 - val_tp: 65.0000 - val_precision: 0.1906 - val_recall: 0.8667
Epoch 20/30
112/112 - 3s - loss: 5.0178e-07 - fn: 7.0000 - fp: 5161.0000 - tn: 222268.0000 - tp: 410.0000 - precision: 0.0736 - recall: 0.9832 - val_loss: 0.2156 - val_fn: 7.0000 - val_fp: 1041.0000 - val_tn: 55845.0000 - val_tp: 68.0000 - val_precision: 0.0613 - val_recall: 0.9067
Epoch 21/30
112/112 - 3s - loss: 7.1907e-07 - fn: 7.0000 - fp: 5825.0000 - tn: 221604.0000 - tp: 410.0000 - precision: 0.0658 - recall: 0.9832 - val_loss: 0.0283 - val_fn: 8.0000 - val_fp: 511.0000 - val_tn: 56375.0000 - val_tp: 67.0000 - val_precision: 0.1159 - val_recall: 0.8933
Epoch 22/30
112/112 - 3s - loss: 3.6405e-07 - fn: 6.0000 - fp: 4149.0000 - tn: 223280.0000 - tp: 411.0000 - precision: 0.0901 - recall: 0.9856 - val_loss: 0.0269 - val_fn: 8.0000 - val_fp: 554.0000 - val_tn: 56332.0000 - val_tp: 67.0000 - val_precision: 0.1079 - val_recall: 0.8933
Epoch 23/30
112/112 - 3s - loss: 2.8464e-07 - fn: 1.0000 - fp: 4131.0000 - tn: 223298.0000 - tp: 416.0000 - precision: 0.0915 - recall: 0.9976 - val_loss: 0.0097 - val_fn: 10.0000 - val_fp: 191.0000 - val_tn: 56695.0000 - val_tp: 65.0000 - val_precision: 0.2539 - val_recall: 0.8667
Epoch 24/30
112/112 - 3s - loss: 3.2445e-07 - fn: 3.0000 - fp: 4040.0000 - tn: 223389.0000 - tp: 414.0000 - precision: 0.0930 - recall: 0.9928 - val_loss: 0.0129 - val_fn: 9.0000 - val_fp: 278.0000 - val_tn: 56608.0000 - val_tp: 66.0000 - val_precision: 0.1919 - val_recall: 0.8800
Epoch 25/30
112/112 - 3s - loss: 5.4032e-07 - fn: 4.0000 - fp: 4834.0000 - tn: 222595.0000 - tp: 413.0000 - precision: 0.0787 - recall: 0.9904 - val_loss: 0.1334 - val_fn: 7.0000 - val_fp: 885.0000 - val_tn: 56001.0000 - val_tp: 68.0000 - val_precision: 0.0714 - val_recall: 0.9067
Epoch 26/30
112/112 - 3s - loss: 1.2099e-06 - fn: 9.0000 - fp: 5767.0000 - tn: 221662.0000 - tp: 408.0000 - precision: 0.0661 - recall: 0.9784 - val_loss: 0.0426 - val_fn: 11.0000 - val_fp: 211.0000 - val_tn: 56675.0000 - val_tp: 64.0000 - val_precision: 0.2327 - val_recall: 0.8533
Epoch 27/30
112/112 - 2s - loss: 5.0924e-07 - fn: 7.0000 - fp: 4185.0000 - tn: 223244.0000 - tp: 410.0000 - precision: 0.0892 - recall: 0.9832 - val_loss: 0.0345 - val_fn: 6.0000 - val_fp: 710.0000 - val_tn: 56176.0000 - val_tp: 69.0000 - val_precision: 0.0886 - val_recall: 0.9200
Epoch 28/30
112/112 - 3s - loss: 4.9177e-07 - fn: 7.0000 - fp: 3871.0000 - tn: 223558.0000 - tp: 410.0000 - precision: 0.0958 - recall: 0.9832 - val_loss: 0.0631 - val_fn: 7.0000 - val_fp: 912.0000 - val_tn: 55974.0000 - val_tp: 68.0000 - val_precision: 0.0694 - val_recall: 0.9067
Epoch 29/30
112/112 - 3s - loss: 1.8390e-06 - fn: 9.0000 - fp: 7199.0000 - tn: 220230.0000 - tp: 408.0000 - precision: 0.0536 - recall: 0.9784 - val_loss: 0.0661 - val_fn: 10.0000 - val_fp: 292.0000 - val_tn: 56594.0000 - val_tp: 65.0000 - val_precision: 0.1821 - val_recall: 0.8667
Epoch 30/30
112/112 - 3s - loss: 3.5976e-06 - fn: 14.0000 - fp: 5541.0000 - tn: 221888.0000 - tp: 403.0000 - precision: 0.0678 - recall: 0.9664 - val_loss: 0.1205 - val_fn: 10.0000 - val_fp: 206.0000 - val_tn: 56680.0000 - val_tp: 65.0000 - val_precision: 0.2399 - val_recall: 0.8667

<tensorflow.python.keras.callbacks.History at 0x16ab3d310>

```
</div>
---
## Conclusions

At the end of training, out of 56,961 validation transactions, we are:

- Correctly identifying 66 of them as fraudulent
- Missing 9 fraudulent transactions
- At the cost of incorrectly flagging 441 legitimate transactions

In the real world, one would put an even higher weight on class 1,
so as to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets  declined in an online purchase -- this is why.

Example available on HuggingFace.

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/🤗%20Model-Imbalanced%20Classification-black.svg)](https://huggingface.co/keras-io/imbalanced_classification) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-Imbalanced%20Classification-black.svg)](https://huggingface.co/spaces/keras-io/Credit_Card_Fraud_Detection) |
