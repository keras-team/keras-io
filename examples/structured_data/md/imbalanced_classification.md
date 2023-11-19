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
import keras

model = keras.Sequential(
    [
        keras.Input(shape=train_features.shape[1:]),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)               │      <span style="color: #00af00; text-decoration-color: #00af00">7,936</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)               │     <span style="color: #00af00; text-decoration-color: #00af00">65,792</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)               │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)               │     <span style="color: #00af00; text-decoration-color: #00af00">65,792</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)               │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                 │        <span style="color: #00af00; text-decoration-color: #00af00">257</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">139,777</span> (546.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">139,777</span> (546.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



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

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.keras")]
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
112/112 - 3s - 24ms/step - fn: 39.0000 - fp: 25593.0000 - loss: 2.2586e-06 - precision: 0.0146 - recall: 0.9065 - tn: 201836.0000 - tp: 378.0000 - val_fn: 5.0000 - val_fp: 3430.0000 - val_loss: 0.1872 - val_precision: 0.0200 - val_recall: 0.9333 - val_tn: 53456.0000 - val_tp: 70.0000
Epoch 2/30
112/112 - 0s - 991us/step - fn: 32.0000 - fp: 7936.0000 - loss: 1.5505e-06 - precision: 0.0463 - recall: 0.9233 - tn: 219493.0000 - tp: 385.0000 - val_fn: 7.0000 - val_fp: 2351.0000 - val_loss: 0.1930 - val_precision: 0.0281 - val_recall: 0.9067 - val_tn: 54535.0000 - val_tp: 68.0000
Epoch 3/30
112/112 - 0s - 1ms/step - fn: 31.0000 - fp: 6716.0000 - loss: 1.2987e-06 - precision: 0.0544 - recall: 0.9257 - tn: 220713.0000 - tp: 386.0000 - val_fn: 4.0000 - val_fp: 3374.0000 - val_loss: 0.1781 - val_precision: 0.0206 - val_recall: 0.9467 - val_tn: 53512.0000 - val_tp: 71.0000
Epoch 4/30
112/112 - 0s - 1ms/step - fn: 25.0000 - fp: 7348.0000 - loss: 1.1292e-06 - precision: 0.0506 - recall: 0.9400 - tn: 220081.0000 - tp: 392.0000 - val_fn: 6.0000 - val_fp: 1405.0000 - val_loss: 0.0796 - val_precision: 0.0468 - val_recall: 0.9200 - val_tn: 55481.0000 - val_tp: 69.0000
Epoch 5/30
112/112 - 0s - 926us/step - fn: 19.0000 - fp: 6720.0000 - loss: 8.0334e-07 - precision: 0.0559 - recall: 0.9544 - tn: 220709.0000 - tp: 398.0000 - val_fn: 11.0000 - val_fp: 315.0000 - val_loss: 0.0212 - val_precision: 0.1689 - val_recall: 0.8533 - val_tn: 56571.0000 - val_tp: 64.0000
Epoch 6/30
112/112 - 0s - 1ms/step - fn: 19.0000 - fp: 6706.0000 - loss: 8.6899e-07 - precision: 0.0560 - recall: 0.9544 - tn: 220723.0000 - tp: 398.0000 - val_fn: 8.0000 - val_fp: 1262.0000 - val_loss: 0.0801 - val_precision: 0.0504 - val_recall: 0.8933 - val_tn: 55624.0000 - val_tp: 67.0000
Epoch 7/30
112/112 - 0s - 1ms/step - fn: 15.0000 - fp: 5161.0000 - loss: 6.5298e-07 - precision: 0.0723 - recall: 0.9640 - tn: 222268.0000 - tp: 402.0000 - val_fn: 7.0000 - val_fp: 1157.0000 - val_loss: 0.0623 - val_precision: 0.0555 - val_recall: 0.9067 - val_tn: 55729.0000 - val_tp: 68.0000
Epoch 8/30
112/112 - 0s - 1ms/step - fn: 11.0000 - fp: 6381.0000 - loss: 6.7164e-07 - precision: 0.0598 - recall: 0.9736 - tn: 221048.0000 - tp: 406.0000 - val_fn: 10.0000 - val_fp: 346.0000 - val_loss: 0.0270 - val_precision: 0.1582 - val_recall: 0.8667 - val_tn: 56540.0000 - val_tp: 65.0000
Epoch 9/30
112/112 - 0s - 1ms/step - fn: 16.0000 - fp: 7259.0000 - loss: 8.9098e-07 - precision: 0.0523 - recall: 0.9616 - tn: 220170.0000 - tp: 401.0000 - val_fn: 7.0000 - val_fp: 1998.0000 - val_loss: 0.1073 - val_precision: 0.0329 - val_recall: 0.9067 - val_tn: 54888.0000 - val_tp: 68.0000
Epoch 10/30
112/112 - 0s - 999us/step - fn: 19.0000 - fp: 7792.0000 - loss: 9.2179e-07 - precision: 0.0486 - recall: 0.9544 - tn: 219637.0000 - tp: 398.0000 - val_fn: 7.0000 - val_fp: 1515.0000 - val_loss: 0.0800 - val_precision: 0.0430 - val_recall: 0.9067 - val_tn: 55371.0000 - val_tp: 68.0000
Epoch 11/30
112/112 - 0s - 1ms/step - fn: 13.0000 - fp: 5828.0000 - loss: 6.4193e-07 - precision: 0.0648 - recall: 0.9688 - tn: 221601.0000 - tp: 404.0000 - val_fn: 9.0000 - val_fp: 794.0000 - val_loss: 0.0410 - val_precision: 0.0767 - val_recall: 0.8800 - val_tn: 56092.0000 - val_tp: 66.0000
Epoch 12/30
112/112 - 0s - 959us/step - fn: 10.0000 - fp: 6400.0000 - loss: 7.4358e-07 - precision: 0.0598 - recall: 0.9760 - tn: 221029.0000 - tp: 407.0000 - val_fn: 8.0000 - val_fp: 593.0000 - val_loss: 0.0466 - val_precision: 0.1015 - val_recall: 0.8933 - val_tn: 56293.0000 - val_tp: 67.0000
Epoch 13/30
112/112 - 0s - 913us/step - fn: 9.0000 - fp: 5756.0000 - loss: 6.8158e-07 - precision: 0.0662 - recall: 0.9784 - tn: 221673.0000 - tp: 408.0000 - val_fn: 11.0000 - val_fp: 280.0000 - val_loss: 0.0336 - val_precision: 0.1860 - val_recall: 0.8533 - val_tn: 56606.0000 - val_tp: 64.0000
Epoch 14/30
112/112 - 0s - 960us/step - fn: 13.0000 - fp: 6699.0000 - loss: 1.0667e-06 - precision: 0.0569 - recall: 0.9688 - tn: 220730.0000 - tp: 404.0000 - val_fn: 9.0000 - val_fp: 1165.0000 - val_loss: 0.0885 - val_precision: 0.0536 - val_recall: 0.8800 - val_tn: 55721.0000 - val_tp: 66.0000
Epoch 15/30
112/112 - 0s - 1ms/step - fn: 15.0000 - fp: 6705.0000 - loss: 6.8100e-07 - precision: 0.0566 - recall: 0.9640 - tn: 220724.0000 - tp: 402.0000 - val_fn: 10.0000 - val_fp: 750.0000 - val_loss: 0.0367 - val_precision: 0.0798 - val_recall: 0.8667 - val_tn: 56136.0000 - val_tp: 65.0000
Epoch 16/30
112/112 - 0s - 1ms/step - fn: 8.0000 - fp: 4288.0000 - loss: 4.1541e-07 - precision: 0.0871 - recall: 0.9808 - tn: 223141.0000 - tp: 409.0000 - val_fn: 11.0000 - val_fp: 351.0000 - val_loss: 0.0199 - val_precision: 0.1542 - val_recall: 0.8533 - val_tn: 56535.0000 - val_tp: 64.0000
Epoch 17/30
112/112 - 0s - 949us/step - fn: 8.0000 - fp: 4598.0000 - loss: 4.3510e-07 - precision: 0.0817 - recall: 0.9808 - tn: 222831.0000 - tp: 409.0000 - val_fn: 10.0000 - val_fp: 688.0000 - val_loss: 0.0296 - val_precision: 0.0863 - val_recall: 0.8667 - val_tn: 56198.0000 - val_tp: 65.0000
Epoch 18/30
112/112 - 0s - 946us/step - fn: 7.0000 - fp: 5544.0000 - loss: 4.6239e-07 - precision: 0.0689 - recall: 0.9832 - tn: 221885.0000 - tp: 410.0000 - val_fn: 8.0000 - val_fp: 444.0000 - val_loss: 0.0260 - val_precision: 0.1311 - val_recall: 0.8933 - val_tn: 56442.0000 - val_tp: 67.0000
Epoch 19/30
112/112 - 0s - 972us/step - fn: 3.0000 - fp: 2920.0000 - loss: 2.7543e-07 - precision: 0.1242 - recall: 0.9928 - tn: 224509.0000 - tp: 414.0000 - val_fn: 9.0000 - val_fp: 510.0000 - val_loss: 0.0245 - val_precision: 0.1146 - val_recall: 0.8800 - val_tn: 56376.0000 - val_tp: 66.0000
Epoch 20/30
112/112 - 0s - 1ms/step - fn: 6.0000 - fp: 5351.0000 - loss: 5.7495e-07 - precision: 0.0713 - recall: 0.9856 - tn: 222078.0000 - tp: 411.0000 - val_fn: 9.0000 - val_fp: 547.0000 - val_loss: 0.0255 - val_precision: 0.1077 - val_recall: 0.8800 - val_tn: 56339.0000 - val_tp: 66.0000
Epoch 21/30
112/112 - 0s - 1ms/step - fn: 6.0000 - fp: 3808.0000 - loss: 5.1475e-07 - precision: 0.0974 - recall: 0.9856 - tn: 223621.0000 - tp: 411.0000 - val_fn: 10.0000 - val_fp: 624.0000 - val_loss: 0.0320 - val_precision: 0.0943 - val_recall: 0.8667 - val_tn: 56262.0000 - val_tp: 65.0000
Epoch 22/30
112/112 - 0s - 1ms/step - fn: 6.0000 - fp: 5117.0000 - loss: 5.5465e-07 - precision: 0.0743 - recall: 0.9856 - tn: 222312.0000 - tp: 411.0000 - val_fn: 10.0000 - val_fp: 836.0000 - val_loss: 0.0556 - val_precision: 0.0721 - val_recall: 0.8667 - val_tn: 56050.0000 - val_tp: 65.0000
Epoch 23/30
112/112 - 0s - 939us/step - fn: 8.0000 - fp: 5583.0000 - loss: 5.5407e-07 - precision: 0.0683 - recall: 0.9808 - tn: 221846.0000 - tp: 409.0000 - val_fn: 12.0000 - val_fp: 501.0000 - val_loss: 0.0300 - val_precision: 0.1117 - val_recall: 0.8400 - val_tn: 56385.0000 - val_tp: 63.0000
Epoch 24/30
112/112 - 0s - 958us/step - fn: 5.0000 - fp: 3933.0000 - loss: 4.7133e-07 - precision: 0.0948 - recall: 0.9880 - tn: 223496.0000 - tp: 412.0000 - val_fn: 12.0000 - val_fp: 211.0000 - val_loss: 0.0326 - val_precision: 0.2299 - val_recall: 0.8400 - val_tn: 56675.0000 - val_tp: 63.0000
Epoch 25/30
112/112 - 0s - 1ms/step - fn: 7.0000 - fp: 5695.0000 - loss: 7.1277e-07 - precision: 0.0672 - recall: 0.9832 - tn: 221734.0000 - tp: 410.0000 - val_fn: 9.0000 - val_fp: 802.0000 - val_loss: 0.0598 - val_precision: 0.0760 - val_recall: 0.8800 - val_tn: 56084.0000 - val_tp: 66.0000
Epoch 26/30
112/112 - 0s - 949us/step - fn: 5.0000 - fp: 3853.0000 - loss: 4.1797e-07 - precision: 0.0966 - recall: 0.9880 - tn: 223576.0000 - tp: 412.0000 - val_fn: 8.0000 - val_fp: 771.0000 - val_loss: 0.0409 - val_precision: 0.0800 - val_recall: 0.8933 - val_tn: 56115.0000 - val_tp: 67.0000
Epoch 27/30
112/112 - 0s - 947us/step - fn: 4.0000 - fp: 3873.0000 - loss: 3.7369e-07 - precision: 0.0964 - recall: 0.9904 - tn: 223556.0000 - tp: 413.0000 - val_fn: 6.0000 - val_fp: 2208.0000 - val_loss: 0.1370 - val_precision: 0.0303 - val_recall: 0.9200 - val_tn: 54678.0000 - val_tp: 69.0000
Epoch 28/30
112/112 - 0s - 892us/step - fn: 5.0000 - fp: 4619.0000 - loss: 4.1290e-07 - precision: 0.0819 - recall: 0.9880 - tn: 222810.0000 - tp: 412.0000 - val_fn: 8.0000 - val_fp: 551.0000 - val_loss: 0.0273 - val_precision: 0.1084 - val_recall: 0.8933 - val_tn: 56335.0000 - val_tp: 67.0000
Epoch 29/30
112/112 - 0s - 931us/step - fn: 1.0000 - fp: 3336.0000 - loss: 2.5478e-07 - precision: 0.1109 - recall: 0.9976 - tn: 224093.0000 - tp: 416.0000 - val_fn: 9.0000 - val_fp: 487.0000 - val_loss: 0.0238 - val_precision: 0.1193 - val_recall: 0.8800 - val_tn: 56399.0000 - val_tp: 66.0000
Epoch 30/30
112/112 - 0s - 1ms/step - fn: 2.0000 - fp: 3521.0000 - loss: 4.1991e-07 - precision: 0.1054 - recall: 0.9952 - tn: 223908.0000 - tp: 415.0000 - val_fn: 10.0000 - val_fp: 462.0000 - val_loss: 0.0331 - val_precision: 0.1233 - val_recall: 0.8667 - val_tn: 56424.0000 - val_tp: 65.0000

<keras.src.callbacks.history.History at 0x7f22b41f3430>

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
