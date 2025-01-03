# Classification with Gated Residual and Variable Selection Networks

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/02/10<br>
**Last modified:** 2025/01/03<br>
**Description:** Using Gated Residual and Variable Selection Networks for income level prediction.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/classification_with_grn_and_vsn.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/classification_with_grn_and_vsn.py)



---
## Introduction

This example demonstrates the use of Gated
Residual Networks (GRN) and Variable Selection Networks (VSN), proposed by
Bryan Lim et al. in
[Temporal Fusion Transformers (TFT) for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363),
for structured data classification. GRNs give the flexibility to the model to apply
non-linear processing only where needed. VSNs allow the model to softly remove any
unnecessary noisy inputs which could negatively impact performance.
Together, those techniques help improving the learning capacity of deep neural
network models.

Note that this example implements only the GRN and VSN components described in
in the paper, rather than the whole TFT model, as GRN and VSN can be useful on
their own for structured data learning tasks.


To run the code you need to use TensorFlow 2.3 or higher.

---
## The dataset

This example uses the
[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29)
provided by the
[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The task is binary classification to determine whether a person makes over 50K a year.

The dataset includes ~300K instances with 41 input features: 7 numerical features
and 34 categorical features.

---
## Setup


```python
import os
import subprocess
import tarfile

# Only the TensorFlow backend supports string inputs.
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
```

---
## Prepare the data

First we load the data from the UCI Machine Learning Repository into a Pandas DataFrame.


```python
# Column names.
CSV_HEADER = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_stat",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "instance_weight",
    "migration_code-change_in_msa",
    "migration_code-change_in_reg",
    "migration_code-move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "income_level",
]

data_url = "https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip"
keras.utils.get_file(origin=data_url, extract=True)
```




<div class="k-default-codeblock">
```
'/home/humbulani/.keras/datasets/census+income+kdd.zip'

```
</div>
determine the downloaded .tar.gz file path and
extract the files from the downloaded .tar.gz file


```python
extracted_path = os.path.join(
    os.path.expanduser("~"), ".keras", "datasets", "census+income+kdd.zip"
)
for root, dirs, files in os.walk(extracted_path):
    for file in files:
        if file.endswith(".tar.gz"):
            tar_gz_path = os.path.join(root, file)
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=root)

train_data_path = os.path.join(
    os.path.expanduser("~"),
    ".keras",
    "datasets",
    "census+income+kdd.zip",
    "census-income.data",
)
test_data_path = os.path.join(
    os.path.expanduser("~"),
    ".keras",
    "datasets",
    "census+income+kdd.zip",
    "census-income.test",
)

data = pd.read_csv(train_data_path, header=None, names=CSV_HEADER)
test_data = pd.read_csv(test_data_path, header=None, names=CSV_HEADER)

print(f"Data shape: {data.shape}")
print(f"Test data shape: {test_data.shape}")

```

<div class="k-default-codeblock">
```
Data shape: (199523, 42)
Test data shape: (99762, 42)

```
</div>
We convert the target column from string to integer.


```python
data["income_level"] = data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)
test_data["income_level"] = test_data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1
)

```

Then, We split the dataset into train and validation sets.


```python
random_selection = np.random.rand(len(data.index)) <= 0.85
train_data = data[random_selection]
valid_data = data[~random_selection]

```

Finally we store the train and test data splits locally to CSV files.


```python
train_data_file = "train_data.csv"
valid_data_file = "valid_data.csv"
test_data_file = "test_data.csv"

train_data.to_csv(train_data_file, index=False, header=False)
valid_data.to_csv(valid_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)

```

clean the directory for the downloaded files except the .tar.gz file and
also remove the empty directories


```python
subprocess.run(
    f'find {extracted_path} -type f ! -name "*.tar.gz" -exec rm -f {{}} +',
    shell=True,
    check=True,
)
subprocess.run(
    f"find {extracted_path} -type d -empty -exec rmdir {{}} +", shell=True, check=True
)
```




<div class="k-default-codeblock">
```
CompletedProcess(args='find /home/humbulani/.keras/datasets/census+income+kdd.zip -type d -empty -exec rmdir {} +', returncode=0)

```
</div>
---
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for reading and
parsing the data into input features, and encoding the input features with respect
to their types.


```python
# Target feature name.
TARGET_FEATURE_NAME = "income_level"
# Weight column name.
WEIGHT_COLUMN_NAME = "instance_weight"
# Numeric feature names.
NUMERIC_FEATURE_NAMES = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]
# Categorical features and their vocabulary lists.
# Note that we add 'v=' as a prefix to all categorical feature values to make
# sure that they are treated as strings.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    feature_name: sorted([str(value) for value in list(data[feature_name].unique())])
    for feature_name in CSV_HEADER
    if feature_name
    not in list(NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME, TARGET_FEATURE_NAME])
}
# All features names.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(
    CATEGORICAL_FEATURES_WITH_VOCABULARY.keys()
)
# Feature default values.
COLUMN_DEFAULTS = [
    (
        [0.0]
        if feature_name
        in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME, WEIGHT_COLUMN_NAME]
        else ["NA"]
    )
    for feature_name in CSV_HEADER
]
```

---
## Create a `tf.data.Dataset` for training and evaluation

We create an input function to read and parse the file, and convert features and
labels into a [`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets) for
training and evaluation.


```python

def process(features, target):
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # Cast categorical feature values to string.
            features[feature_name] = keras.ops.cast(features[feature_name], "string")
    # Get the instance weight.
    weight = features.pop(WEIGHT_COLUMN_NAME)
    return dict(features), target, weight


def get_dataset_from_csv(csv_file_path, batch_size, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        shuffle=shuffle,
    ).map(process)

    return dataset

```

---
## Create model inputs


```python

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="float32"
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype="string"
            )
    return inputs

```

---
## Encode input features

For categorical features, we encode them using `layers.Embedding` using the
`encoding_size` as the embedding dimensions. For the numerical features,
we apply linear transformation using `layers.Dense` to project each feature into
`encoding_size`-dimensional vector. Thus, all the encoded features will have the
same dimensionality.


```python

def encode_inputs(inputs, encoding_size):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            index = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=1
            )
            # Convert the string input values into integer indices.
            value_index = index(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            embedding_ecoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=encoding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = embedding_ecoder(value_index)
        else:
            # Project the numeric feature to encoding_size using linear transformation.
            encoded_feature = keras.ops.expand_dims(inputs[feature_name], -1)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features

```

---
## Implement the Gated Linear Unit

[Gated Linear Units (GLUs)](https://arxiv.org/abs/1612.08083) provide the
flexibility to suppress input that are not relevant for a given task.


```python

class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

    # to remove the build warnings
    def build(self):
        self.built = True

```

---
## Implement the Gated Residual Network

The Gated Residual Network (GRN) works as follows:

1. Applies the nonlinear ELU transformation to the inputs.
2. Applies linear transformation followed by dropout.
4. Applies GLU and adds the original inputs to the output of the GLU to perform skip
(residual) connection.
6. Applies layer normalization and produces the output.


```python

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

    # to remove the build warnings
    def build(self):
        self.build = True

```

---
## Implement the Variable Selection Network

The Variable Selection Network (VSN) works as follows:

1. Applies a GRN to each feature individually.
2. Applies a GRN on the concatenation of all the features, followed by a softmax to
produce feature weights.
3. Produces a weighted sum of the output of the individual GRN.

Note that the output of the VSN is [batch_size, encoding_size], regardless of the
number of the input features.


```python

class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super().__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = keras.ops.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = keras.ops.stack(x, axis=1)

        outputs = keras.ops.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs

    # to remove the build warnings
    def build(self):
        self.built = True

```

---
## Create Gated Residual and Variable Selection Networks model


```python

def create_model(encoding_size):
    inputs = create_model_inputs()
    feature_list = encode_inputs(inputs, encoding_size)
    num_features = len(feature_list)

    features = VariableSelection(num_features, encoding_size, dropout_rate)(
        feature_list
    )

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

---
## Compile, train, and evaluate the model


```python
learning_rate = 0.001
dropout_rate = 0.15
batch_size = 265
num_epochs = 1
encoding_size = 16

model = create_model(encoding_size)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)


# Create an early stopping callback.
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("Start training the model...")
train_dataset = get_dataset_from_csv(
    train_data_file,
    batch_size=batch_size,
    shuffle=True,
)
valid_dataset = get_dataset_from_csv(valid_data_file, batch_size=batch_size)
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)
print("Model training finished.")

print("Evaluating model performance...")
test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)
_, accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
```

<div class="k-default-codeblock">
```
Start training the model...

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  10s 10s/step - accuracy: 0.3660 - loss: 1207.7571


  2/Unknown  14s 4s/step - accuracy: 0.5075 - loss: 1102.4080 


  3/Unknown  18s 4s/step - accuracy: 0.5870 - loss: 1019.7374


  4/Unknown  22s 4s/step - accuracy: 0.6410 - loss: 949.8593 


  5/Unknown  26s 4s/step - accuracy: 0.6782 - loss: 895.4934


  6/Unknown  30s 4s/step - accuracy: 0.7066 - loss: 849.8503


  7/Unknown  34s 4s/step - accuracy: 0.7285 - loss: 813.4320


  8/Unknown  38s 4s/step - accuracy: 0.7457 - loss: 785.3990


  9/Unknown  43s 4s/step - accuracy: 0.7602 - loss: 760.9548


 10/Unknown  47s 4s/step - accuracy: 0.7725 - loss: 738.8033


 11/Unknown  51s 4s/step - accuracy: 0.7829 - loss: 719.7887


 12/Unknown  56s 4s/step - accuracy: 0.7918 - loss: 702.8824


 13/Unknown  60s 4s/step - accuracy: 0.7996 - loss: 688.2739


 14/Unknown  64s 4s/step - accuracy: 0.8065 - loss: 675.0721


 15/Unknown  69s 4s/step - accuracy: 0.8126 - loss: 662.8633


 16/Unknown  73s 4s/step - accuracy: 0.8181 - loss: 651.6475


 17/Unknown  77s 4s/step - accuracy: 0.8231 - loss: 640.9540


 18/Unknown  82s 4s/step - accuracy: 0.8277 - loss: 631.2631


 19/Unknown  86s 4s/step - accuracy: 0.8320 - loss: 622.0033


 20/Unknown  91s 4s/step - accuracy: 0.8358 - loss: 613.5731


 21/Unknown  95s 4s/step - accuracy: 0.8394 - loss: 605.6331


 22/Unknown  99s 4s/step - accuracy: 0.8427 - loss: 598.0710


 23/Unknown  104s 4s/step - accuracy: 0.8458 - loss: 590.9688


 24/Unknown  109s 4s/step - accuracy: 0.8487 - loss: 584.1125


 25/Unknown  114s 4s/step - accuracy: 0.8514 - loss: 577.9108


 26/Unknown  118s 4s/step - accuracy: 0.8539 - loss: 572.1182


 27/Unknown  123s 4s/step - accuracy: 0.8563 - loss: 566.6541


 28/Unknown  128s 4s/step - accuracy: 0.8584 - loss: 561.5195


 29/Unknown  132s 4s/step - accuracy: 0.8605 - loss: 556.6824


 30/Unknown  137s 4s/step - accuracy: 0.8624 - loss: 551.9829


 31/Unknown  141s 4s/step - accuracy: 0.8642 - loss: 547.4396


 32/Unknown  146s 4s/step - accuracy: 0.8660 - loss: 543.0052


 33/Unknown  151s 4s/step - accuracy: 0.8677 - loss: 538.6678


 34/Unknown  155s 4s/step - accuracy: 0.8692 - loss: 534.5511


 35/Unknown  159s 4s/step - accuracy: 0.8707 - loss: 530.6713


 36/Unknown  164s 4s/step - accuracy: 0.8721 - loss: 526.8998


 37/Unknown  169s 4s/step - accuracy: 0.8734 - loss: 523.2978


 38/Unknown  173s 4s/step - accuracy: 0.8747 - loss: 519.7983


 39/Unknown  178s 4s/step - accuracy: 0.8760 - loss: 516.3888


 40/Unknown  183s 4s/step - accuracy: 0.8772 - loss: 513.1334


 41/Unknown  188s 4s/step - accuracy: 0.8783 - loss: 509.9753


 42/Unknown  193s 4s/step - accuracy: 0.8794 - loss: 506.8998


 43/Unknown  197s 4s/step - accuracy: 0.8805 - loss: 503.8833


 44/Unknown  202s 4s/step - accuracy: 0.8815 - loss: 500.9615


 45/Unknown  208s 4s/step - accuracy: 0.8824 - loss: 498.1530


 46/Unknown  213s 4s/step - accuracy: 0.8834 - loss: 495.3979


 47/Unknown  217s 4s/step - accuracy: 0.8843 - loss: 492.7300


 48/Unknown  222s 5s/step - accuracy: 0.8852 - loss: 490.1727


 49/Unknown  227s 5s/step - accuracy: 0.8860 - loss: 487.6551


 50/Unknown  231s 5s/step - accuracy: 0.8868 - loss: 485.1848


 51/Unknown  236s 5s/step - accuracy: 0.8876 - loss: 482.7541


 52/Unknown  241s 5s/step - accuracy: 0.8884 - loss: 480.3756


 53/Unknown  246s 5s/step - accuracy: 0.8892 - loss: 478.0660


 54/Unknown  251s 5s/step - accuracy: 0.8899 - loss: 475.8299


 55/Unknown  256s 5s/step - accuracy: 0.8906 - loss: 473.6529


 56/Unknown  260s 5s/step - accuracy: 0.8913 - loss: 471.5191


 57/Unknown  265s 5s/step - accuracy: 0.8919 - loss: 469.4434


 58/Unknown  270s 5s/step - accuracy: 0.8926 - loss: 467.4065


 59/Unknown  276s 5s/step - accuracy: 0.8932 - loss: 465.4138


 60/Unknown  281s 5s/step - accuracy: 0.8938 - loss: 463.4726


 61/Unknown  286s 5s/step - accuracy: 0.8944 - loss: 461.6012


 62/Unknown  291s 5s/step - accuracy: 0.8949 - loss: 459.7856


 63/Unknown  296s 5s/step - accuracy: 0.8955 - loss: 457.9802


 64/Unknown  301s 5s/step - accuracy: 0.8960 - loss: 456.1978


 65/Unknown  305s 5s/step - accuracy: 0.8966 - loss: 454.4304


 66/Unknown  310s 5s/step - accuracy: 0.8971 - loss: 452.6868


 67/Unknown  314s 5s/step - accuracy: 0.8976 - loss: 450.9761


 68/Unknown  319s 5s/step - accuracy: 0.8981 - loss: 449.3022


 69/Unknown  324s 5s/step - accuracy: 0.8986 - loss: 447.6501


 70/Unknown  328s 5s/step - accuracy: 0.8991 - loss: 446.0159


 71/Unknown  333s 5s/step - accuracy: 0.8996 - loss: 444.4040


 72/Unknown  337s 5s/step - accuracy: 0.9000 - loss: 442.8098


 73/Unknown  342s 5s/step - accuracy: 0.9005 - loss: 441.2607


 74/Unknown  347s 5s/step - accuracy: 0.9009 - loss: 439.7383


 75/Unknown  351s 5s/step - accuracy: 0.9014 - loss: 438.2570


 76/Unknown  356s 5s/step - accuracy: 0.9018 - loss: 436.8094


 77/Unknown  360s 5s/step - accuracy: 0.9022 - loss: 435.3762


 78/Unknown  365s 5s/step - accuracy: 0.9026 - loss: 433.9661


 79/Unknown  369s 5s/step - accuracy: 0.9031 - loss: 432.5754


 80/Unknown  374s 5s/step - accuracy: 0.9035 - loss: 431.2206


 81/Unknown  379s 5s/step - accuracy: 0.9038 - loss: 429.8713


 82/Unknown  383s 5s/step - accuracy: 0.9042 - loss: 428.5655


 83/Unknown  388s 5s/step - accuracy: 0.9046 - loss: 427.2886


 84/Unknown  392s 5s/step - accuracy: 0.9050 - loss: 426.0367


 85/Unknown  397s 5s/step - accuracy: 0.9053 - loss: 424.7995


 86/Unknown  402s 5s/step - accuracy: 0.9057 - loss: 423.5760


 87/Unknown  406s 5s/step - accuracy: 0.9060 - loss: 422.3595


 88/Unknown  411s 5s/step - accuracy: 0.9064 - loss: 421.1734


 89/Unknown  416s 5s/step - accuracy: 0.9067 - loss: 420.0010


 90/Unknown  420s 5s/step - accuracy: 0.9070 - loss: 418.8433


 91/Unknown  425s 5s/step - accuracy: 0.9073 - loss: 417.7053


 92/Unknown  430s 5s/step - accuracy: 0.9077 - loss: 416.5843


 93/Unknown  434s 5s/step - accuracy: 0.9080 - loss: 415.4904


 94/Unknown  439s 5s/step - accuracy: 0.9083 - loss: 414.4215


 95/Unknown  443s 5s/step - accuracy: 0.9086 - loss: 413.3767


 96/Unknown  448s 5s/step - accuracy: 0.9089 - loss: 412.3375


 97/Unknown  453s 5s/step - accuracy: 0.9092 - loss: 411.3148


 98/Unknown  457s 5s/step - accuracy: 0.9094 - loss: 410.3085


 99/Unknown  462s 5s/step - accuracy: 0.9097 - loss: 409.3198


100/Unknown  466s 5s/step - accuracy: 0.9100 - loss: 408.3477


101/Unknown  471s 5s/step - accuracy: 0.9103 - loss: 407.3878


102/Unknown  476s 5s/step - accuracy: 0.9105 - loss: 406.4406


103/Unknown  481s 5s/step - accuracy: 0.9108 - loss: 405.5037


104/Unknown  486s 5s/step - accuracy: 0.9110 - loss: 404.5758


105/Unknown  490s 5s/step - accuracy: 0.9113 - loss: 403.6591


106/Unknown  495s 5s/step - accuracy: 0.9115 - loss: 402.7492


107/Unknown  500s 5s/step - accuracy: 0.9118 - loss: 401.8447


108/Unknown  504s 5s/step - accuracy: 0.9120 - loss: 400.9514


109/Unknown  509s 5s/step - accuracy: 0.9123 - loss: 400.0719


110/Unknown  514s 5s/step - accuracy: 0.9125 - loss: 399.2086


111/Unknown  518s 5s/step - accuracy: 0.9127 - loss: 398.3539


112/Unknown  523s 5s/step - accuracy: 0.9130 - loss: 397.5079


113/Unknown  528s 5s/step - accuracy: 0.9132 - loss: 396.6739


114/Unknown  533s 5s/step - accuracy: 0.9134 - loss: 395.8583


115/Unknown  538s 5s/step - accuracy: 0.9136 - loss: 395.0499


116/Unknown  542s 5s/step - accuracy: 0.9138 - loss: 394.2525


117/Unknown  547s 5s/step - accuracy: 0.9141 - loss: 393.4650


118/Unknown  552s 5s/step - accuracy: 0.9143 - loss: 392.6905


119/Unknown  556s 5s/step - accuracy: 0.9145 - loss: 391.9308


120/Unknown  561s 5s/step - accuracy: 0.9147 - loss: 391.1761


121/Unknown  566s 5s/step - accuracy: 0.9149 - loss: 390.4323


122/Unknown  570s 5s/step - accuracy: 0.9151 - loss: 389.6927


123/Unknown  575s 5s/step - accuracy: 0.9153 - loss: 388.9673


124/Unknown  580s 5s/step - accuracy: 0.9154 - loss: 388.2496


125/Unknown  585s 5s/step - accuracy: 0.9156 - loss: 387.5363


126/Unknown  590s 5s/step - accuracy: 0.9158 - loss: 386.8299


127/Unknown  595s 5s/step - accuracy: 0.9160 - loss: 386.1295


128/Unknown  599s 5s/step - accuracy: 0.9162 - loss: 385.4340


129/Unknown  604s 5s/step - accuracy: 0.9164 - loss: 384.7438


130/Unknown  609s 5s/step - accuracy: 0.9165 - loss: 384.0630


131/Unknown  613s 5s/step - accuracy: 0.9167 - loss: 383.3891


132/Unknown  618s 5s/step - accuracy: 0.9169 - loss: 382.7205


133/Unknown  622s 5s/step - accuracy: 0.9171 - loss: 382.0653


134/Unknown  627s 5s/step - accuracy: 0.9172 - loss: 381.4217


135/Unknown  632s 5s/step - accuracy: 0.9174 - loss: 380.7866


136/Unknown  637s 5s/step - accuracy: 0.9176 - loss: 380.1596


137/Unknown  641s 5s/step - accuracy: 0.9177 - loss: 379.5403


138/Unknown  646s 5s/step - accuracy: 0.9179 - loss: 378.9232


139/Unknown  650s 5s/step - accuracy: 0.9181 - loss: 378.3159


140/Unknown  655s 5s/step - accuracy: 0.9182 - loss: 377.7125


141/Unknown  660s 5s/step - accuracy: 0.9184 - loss: 377.1123


142/Unknown  665s 5s/step - accuracy: 0.9185 - loss: 376.5167


143/Unknown  669s 5s/step - accuracy: 0.9187 - loss: 375.9280


144/Unknown  674s 5s/step - accuracy: 0.9188 - loss: 375.3439


145/Unknown  678s 5s/step - accuracy: 0.9190 - loss: 374.7696


146/Unknown  683s 5s/step - accuracy: 0.9191 - loss: 374.2022


147/Unknown  687s 5s/step - accuracy: 0.9192 - loss: 373.6388


148/Unknown  692s 5s/step - accuracy: 0.9194 - loss: 373.0820


149/Unknown  697s 5s/step - accuracy: 0.9195 - loss: 372.5319


150/Unknown  701s 5s/step - accuracy: 0.9197 - loss: 371.9951


151/Unknown  706s 5s/step - accuracy: 0.9198 - loss: 371.4630


152/Unknown  710s 5s/step - accuracy: 0.9199 - loss: 370.9363


153/Unknown  715s 5s/step - accuracy: 0.9201 - loss: 370.4174


154/Unknown  720s 5s/step - accuracy: 0.9202 - loss: 369.9002


155/Unknown  725s 5s/step - accuracy: 0.9203 - loss: 369.3880


156/Unknown  729s 5s/step - accuracy: 0.9204 - loss: 368.8798


157/Unknown  734s 5s/step - accuracy: 0.9206 - loss: 368.3802


158/Unknown  739s 5s/step - accuracy: 0.9207 - loss: 367.8825


159/Unknown  743s 5s/step - accuracy: 0.9208 - loss: 367.3949


160/Unknown  748s 5s/step - accuracy: 0.9209 - loss: 366.9076


161/Unknown  752s 5s/step - accuracy: 0.9211 - loss: 366.4265


162/Unknown  757s 5s/step - accuracy: 0.9212 - loss: 365.9467


163/Unknown  762s 5s/step - accuracy: 0.9213 - loss: 365.4701


164/Unknown  766s 5s/step - accuracy: 0.9214 - loss: 364.9950


165/Unknown  770s 5s/step - accuracy: 0.9215 - loss: 364.5236


166/Unknown  775s 5s/step - accuracy: 0.9217 - loss: 364.0565


167/Unknown  780s 5s/step - accuracy: 0.9218 - loss: 363.5902


168/Unknown  784s 5s/step - accuracy: 0.9219 - loss: 363.1264


169/Unknown  789s 5s/step - accuracy: 0.9220 - loss: 362.6704


170/Unknown  794s 5s/step - accuracy: 0.9221 - loss: 362.2173


171/Unknown  799s 5s/step - accuracy: 0.9222 - loss: 361.7654


172/Unknown  803s 5s/step - accuracy: 0.9223 - loss: 361.3171


173/Unknown  808s 5s/step - accuracy: 0.9225 - loss: 360.8713


174/Unknown  813s 5s/step - accuracy: 0.9226 - loss: 360.4279


175/Unknown  818s 5s/step - accuracy: 0.9227 - loss: 359.9890


176/Unknown  823s 5s/step - accuracy: 0.9228 - loss: 359.5543


177/Unknown  827s 5s/step - accuracy: 0.9229 - loss: 359.1236


178/Unknown  832s 5s/step - accuracy: 0.9230 - loss: 358.6957


179/Unknown  837s 5s/step - accuracy: 0.9231 - loss: 358.2695


180/Unknown  842s 5s/step - accuracy: 0.9232 - loss: 357.8491


181/Unknown  847s 5s/step - accuracy: 0.9233 - loss: 357.4330


182/Unknown  851s 5s/step - accuracy: 0.9234 - loss: 357.0235


183/Unknown  856s 5s/step - accuracy: 0.9235 - loss: 356.6167


184/Unknown  861s 5s/step - accuracy: 0.9236 - loss: 356.2136


185/Unknown  865s 5s/step - accuracy: 0.9237 - loss: 355.8134


186/Unknown  870s 5s/step - accuracy: 0.9238 - loss: 355.4167


187/Unknown  875s 5s/step - accuracy: 0.9239 - loss: 355.0250


188/Unknown  879s 5s/step - accuracy: 0.9240 - loss: 354.6354


189/Unknown  884s 5s/step - accuracy: 0.9241 - loss: 354.2495


190/Unknown  889s 5s/step - accuracy: 0.9242 - loss: 353.8654


191/Unknown  893s 5s/step - accuracy: 0.9243 - loss: 353.4833


192/Unknown  898s 5s/step - accuracy: 0.9244 - loss: 353.1022


193/Unknown  902s 5s/step - accuracy: 0.9245 - loss: 352.7214


194/Unknown  907s 5s/step - accuracy: 0.9246 - loss: 352.3439


195/Unknown  911s 5s/step - accuracy: 0.9246 - loss: 351.9707


196/Unknown  916s 5s/step - accuracy: 0.9247 - loss: 351.5989


197/Unknown  920s 5s/step - accuracy: 0.9248 - loss: 351.2298


198/Unknown  925s 5s/step - accuracy: 0.9249 - loss: 350.8626


199/Unknown  930s 5s/step - accuracy: 0.9250 - loss: 350.4981


200/Unknown  935s 5s/step - accuracy: 0.9251 - loss: 350.1355


201/Unknown  939s 5s/step - accuracy: 0.9252 - loss: 349.7766


202/Unknown  944s 5s/step - accuracy: 0.9253 - loss: 349.4206


203/Unknown  949s 5s/step - accuracy: 0.9253 - loss: 349.0690


204/Unknown  954s 5s/step - accuracy: 0.9254 - loss: 348.7199


205/Unknown  958s 5s/step - accuracy: 0.9255 - loss: 348.3728


206/Unknown  964s 5s/step - accuracy: 0.9256 - loss: 348.0292


207/Unknown  968s 5s/step - accuracy: 0.9257 - loss: 347.6898


208/Unknown  973s 5s/step - accuracy: 0.9258 - loss: 347.3521


209/Unknown  978s 5s/step - accuracy: 0.9258 - loss: 347.0195


210/Unknown  983s 5s/step - accuracy: 0.9259 - loss: 346.6897


211/Unknown  987s 5s/step - accuracy: 0.9260 - loss: 346.3630


212/Unknown  992s 5s/step - accuracy: 0.9261 - loss: 346.0395


213/Unknown  997s 5s/step - accuracy: 0.9261 - loss: 345.7180


214/Unknown  1002s 5s/step - accuracy: 0.9262 - loss: 345.3971


215/Unknown  1007s 5s/step - accuracy: 0.9263 - loss: 345.0785


216/Unknown  1012s 5s/step - accuracy: 0.9264 - loss: 344.7623


217/Unknown  1017s 5s/step - accuracy: 0.9264 - loss: 344.4494


218/Unknown  1022s 5s/step - accuracy: 0.9265 - loss: 344.1385


219/Unknown  1027s 5s/step - accuracy: 0.9266 - loss: 343.8289


220/Unknown  1032s 5s/step - accuracy: 0.9267 - loss: 343.5211


221/Unknown  1037s 5s/step - accuracy: 0.9267 - loss: 343.2156


222/Unknown  1042s 5s/step - accuracy: 0.9268 - loss: 342.9126


223/Unknown  1047s 5s/step - accuracy: 0.9269 - loss: 342.6116


224/Unknown  1052s 5s/step - accuracy: 0.9270 - loss: 342.3128


225/Unknown  1057s 5s/step - accuracy: 0.9270 - loss: 342.0152


226/Unknown  1062s 5s/step - accuracy: 0.9271 - loss: 341.7184


227/Unknown  1067s 5s/step - accuracy: 0.9272 - loss: 341.4229


228/Unknown  1072s 5s/step - accuracy: 0.9272 - loss: 341.1292


229/Unknown  1076s 5s/step - accuracy: 0.9273 - loss: 340.8392


230/Unknown  1081s 5s/step - accuracy: 0.9274 - loss: 340.5511


231/Unknown  1086s 5s/step - accuracy: 0.9274 - loss: 340.2639


232/Unknown  1092s 5s/step - accuracy: 0.9275 - loss: 339.9771


233/Unknown  1096s 5s/step - accuracy: 0.9276 - loss: 339.6896


234/Unknown  1101s 5s/step - accuracy: 0.9276 - loss: 339.4058


235/Unknown  1106s 5s/step - accuracy: 0.9277 - loss: 339.1237


236/Unknown  1111s 5s/step - accuracy: 0.9278 - loss: 338.8431


237/Unknown  1116s 5s/step - accuracy: 0.9278 - loss: 338.5645


238/Unknown  1121s 5s/step - accuracy: 0.9279 - loss: 338.2875


239/Unknown  1126s 5s/step - accuracy: 0.9280 - loss: 338.0133


240/Unknown  1131s 5s/step - accuracy: 0.9280 - loss: 337.7398


241/Unknown  1136s 5s/step - accuracy: 0.9281 - loss: 337.4683


242/Unknown  1141s 5s/step - accuracy: 0.9282 - loss: 337.1974


243/Unknown  1146s 5s/step - accuracy: 0.9282 - loss: 336.9279


244/Unknown  1151s 5s/step - accuracy: 0.9283 - loss: 336.6605


245/Unknown  1157s 5s/step - accuracy: 0.9283 - loss: 336.3942


246/Unknown  1162s 5s/step - accuracy: 0.9284 - loss: 336.1295


247/Unknown  1167s 5s/step - accuracy: 0.9285 - loss: 335.8667


248/Unknown  1172s 5s/step - accuracy: 0.9285 - loss: 335.6055


249/Unknown  1177s 5s/step - accuracy: 0.9286 - loss: 335.3467


250/Unknown  1182s 5s/step - accuracy: 0.9286 - loss: 335.0888


251/Unknown  1187s 5s/step - accuracy: 0.9287 - loss: 334.8347


252/Unknown  1192s 5s/step - accuracy: 0.9288 - loss: 334.5807


253/Unknown  1197s 5s/step - accuracy: 0.9288 - loss: 334.3291


254/Unknown  1202s 5s/step - accuracy: 0.9289 - loss: 334.0790


255/Unknown  1208s 5s/step - accuracy: 0.9289 - loss: 333.8303


256/Unknown  1213s 5s/step - accuracy: 0.9290 - loss: 333.5836


257/Unknown  1219s 5s/step - accuracy: 0.9290 - loss: 333.3395


258/Unknown  1224s 5s/step - accuracy: 0.9291 - loss: 333.0960


259/Unknown  1229s 5s/step - accuracy: 0.9292 - loss: 332.8548


260/Unknown  1234s 5s/step - accuracy: 0.9292 - loss: 332.6145


261/Unknown  1239s 5s/step - accuracy: 0.9293 - loss: 332.3764


262/Unknown  1244s 5s/step - accuracy: 0.9293 - loss: 332.1390


263/Unknown  1249s 5s/step - accuracy: 0.9294 - loss: 331.9027


264/Unknown  1254s 5s/step - accuracy: 0.9294 - loss: 331.6683


265/Unknown  1259s 5s/step - accuracy: 0.9295 - loss: 331.4364


266/Unknown  1264s 5s/step - accuracy: 0.9295 - loss: 331.2059


267/Unknown  1269s 5s/step - accuracy: 0.9296 - loss: 330.9770


268/Unknown  1274s 5s/step - accuracy: 0.9296 - loss: 330.7490


269/Unknown  1279s 5s/step - accuracy: 0.9297 - loss: 330.5221


270/Unknown  1284s 5s/step - accuracy: 0.9297 - loss: 330.2965


271/Unknown  1289s 5s/step - accuracy: 0.9298 - loss: 330.0719


272/Unknown  1294s 5s/step - accuracy: 0.9298 - loss: 329.8488


273/Unknown  1299s 5s/step - accuracy: 0.9299 - loss: 329.6284


274/Unknown  1304s 5s/step - accuracy: 0.9300 - loss: 329.4088


275/Unknown  1310s 5s/step - accuracy: 0.9300 - loss: 329.1902


276/Unknown  1315s 5s/step - accuracy: 0.9301 - loss: 328.9727


277/Unknown  1320s 5s/step - accuracy: 0.9301 - loss: 328.7570


278/Unknown  1326s 5s/step - accuracy: 0.9301 - loss: 328.5421


279/Unknown  1331s 5s/step - accuracy: 0.9302 - loss: 328.3284


280/Unknown  1336s 5s/step - accuracy: 0.9302 - loss: 328.1152


281/Unknown  1341s 5s/step - accuracy: 0.9303 - loss: 327.9040


282/Unknown  1347s 5s/step - accuracy: 0.9303 - loss: 327.6935


283/Unknown  1353s 5s/step - accuracy: 0.9304 - loss: 327.4832


284/Unknown  1359s 5s/step - accuracy: 0.9304 - loss: 327.2734


285/Unknown  1364s 5s/step - accuracy: 0.9305 - loss: 327.0658


286/Unknown  1370s 5s/step - accuracy: 0.9305 - loss: 326.8585


287/Unknown  1375s 5s/step - accuracy: 0.9306 - loss: 326.6521


288/Unknown  1381s 5s/step - accuracy: 0.9306 - loss: 326.4481


289/Unknown  1387s 5s/step - accuracy: 0.9307 - loss: 326.2443


290/Unknown  1393s 5s/step - accuracy: 0.9307 - loss: 326.0433


291/Unknown  1399s 5s/step - accuracy: 0.9308 - loss: 325.8439


292/Unknown  1405s 5s/step - accuracy: 0.9308 - loss: 325.6450


293/Unknown  1411s 5s/step - accuracy: 0.9309 - loss: 325.4478


294/Unknown  1418s 5s/step - accuracy: 0.9309 - loss: 325.2528


295/Unknown  1426s 5s/step - accuracy: 0.9309 - loss: 325.0590


296/Unknown  1431s 5s/step - accuracy: 0.9310 - loss: 324.8657


297/Unknown  1436s 5s/step - accuracy: 0.9310 - loss: 324.6730


298/Unknown  1441s 5s/step - accuracy: 0.9311 - loss: 324.4815


299/Unknown  1446s 5s/step - accuracy: 0.9311 - loss: 324.2902


300/Unknown  1451s 5s/step - accuracy: 0.9312 - loss: 324.1008


301/Unknown  1458s 5s/step - accuracy: 0.9312 - loss: 323.9124


302/Unknown  1463s 5s/step - accuracy: 0.9312 - loss: 323.7238


303/Unknown  1467s 5s/step - accuracy: 0.9313 - loss: 323.5364


304/Unknown  1472s 5s/step - accuracy: 0.9313 - loss: 323.3500


305/Unknown  1477s 5s/step - accuracy: 0.9314 - loss: 323.1646


306/Unknown  1482s 5s/step - accuracy: 0.9314 - loss: 322.9795


307/Unknown  1487s 5s/step - accuracy: 0.9314 - loss: 322.7953


308/Unknown  1491s 5s/step - accuracy: 0.9315 - loss: 322.6122


309/Unknown  1495s 5s/step - accuracy: 0.9315 - loss: 322.4298


310/Unknown  1500s 5s/step - accuracy: 0.9316 - loss: 322.2483


311/Unknown  1504s 5s/step - accuracy: 0.9316 - loss: 322.0682


312/Unknown  1508s 5s/step - accuracy: 0.9316 - loss: 321.8883


313/Unknown  1513s 5s/step - accuracy: 0.9317 - loss: 321.7096


314/Unknown  1518s 5s/step - accuracy: 0.9317 - loss: 321.5312


315/Unknown  1523s 5s/step - accuracy: 0.9318 - loss: 321.3531


316/Unknown  1527s 5s/step - accuracy: 0.9318 - loss: 321.1761


317/Unknown  1532s 5s/step - accuracy: 0.9318 - loss: 321.0012


318/Unknown  1536s 5s/step - accuracy: 0.9319 - loss: 320.8268


319/Unknown  1542s 5s/step - accuracy: 0.9319 - loss: 320.6526


320/Unknown  1547s 5s/step - accuracy: 0.9320 - loss: 320.4793


321/Unknown  1552s 5s/step - accuracy: 0.9320 - loss: 320.3062


322/Unknown  1556s 5s/step - accuracy: 0.9320 - loss: 320.1337


323/Unknown  1561s 5s/step - accuracy: 0.9321 - loss: 319.9617


324/Unknown  1566s 5s/step - accuracy: 0.9321 - loss: 319.7900


325/Unknown  1571s 5s/step - accuracy: 0.9321 - loss: 319.6192


326/Unknown  1575s 5s/step - accuracy: 0.9322 - loss: 319.4486


327/Unknown  1580s 5s/step - accuracy: 0.9322 - loss: 319.2791


328/Unknown  1584s 5s/step - accuracy: 0.9323 - loss: 319.1109


329/Unknown  1588s 5s/step - accuracy: 0.9323 - loss: 318.9430


330/Unknown  1593s 5s/step - accuracy: 0.9323 - loss: 318.7765


331/Unknown  1597s 5s/step - accuracy: 0.9324 - loss: 318.6100


332/Unknown  1602s 5s/step - accuracy: 0.9324 - loss: 318.4451


333/Unknown  1606s 5s/step - accuracy: 0.9324 - loss: 318.2813


334/Unknown  1610s 5s/step - accuracy: 0.9325 - loss: 318.1185


335/Unknown  1615s 5s/step - accuracy: 0.9325 - loss: 317.9565


336/Unknown  1620s 5s/step - accuracy: 0.9325 - loss: 317.7953


337/Unknown  1625s 5s/step - accuracy: 0.9326 - loss: 317.6343


338/Unknown  1629s 5s/step - accuracy: 0.9326 - loss: 317.4747


339/Unknown  1633s 5s/step - accuracy: 0.9326 - loss: 317.3152


340/Unknown  1637s 5s/step - accuracy: 0.9327 - loss: 317.1561


341/Unknown  1642s 5s/step - accuracy: 0.9327 - loss: 316.9976


342/Unknown  1646s 5s/step - accuracy: 0.9328 - loss: 316.8401


343/Unknown  1651s 5s/step - accuracy: 0.9328 - loss: 316.6830


344/Unknown  1655s 5s/step - accuracy: 0.9328 - loss: 316.5267


345/Unknown  1659s 5s/step - accuracy: 0.9329 - loss: 316.3714


346/Unknown  1664s 5s/step - accuracy: 0.9329 - loss: 316.2170


347/Unknown  1668s 5s/step - accuracy: 0.9329 - loss: 316.0630


348/Unknown  1672s 5s/step - accuracy: 0.9330 - loss: 315.9096


349/Unknown  1677s 5s/step - accuracy: 0.9330 - loss: 315.7571


350/Unknown  1681s 5s/step - accuracy: 0.9330 - loss: 315.6042


351/Unknown  1686s 5s/step - accuracy: 0.9331 - loss: 315.4518


352/Unknown  1690s 5s/step - accuracy: 0.9331 - loss: 315.3003


353/Unknown  1695s 5s/step - accuracy: 0.9331 - loss: 315.1490


354/Unknown  1700s 5s/step - accuracy: 0.9331 - loss: 314.9982


355/Unknown  1705s 5s/step - accuracy: 0.9332 - loss: 314.8481


356/Unknown  1710s 5s/step - accuracy: 0.9332 - loss: 314.6982


357/Unknown  1715s 5s/step - accuracy: 0.9332 - loss: 314.5501


358/Unknown  1720s 5s/step - accuracy: 0.9333 - loss: 314.4027


359/Unknown  1725s 5s/step - accuracy: 0.9333 - loss: 314.2560


360/Unknown  1729s 5s/step - accuracy: 0.9333 - loss: 314.1105


361/Unknown  1734s 5s/step - accuracy: 0.9334 - loss: 313.9658


362/Unknown  1739s 5s/step - accuracy: 0.9334 - loss: 313.8213


363/Unknown  1744s 5s/step - accuracy: 0.9334 - loss: 313.6775


364/Unknown  1749s 5s/step - accuracy: 0.9335 - loss: 313.5339


365/Unknown  1754s 5s/step - accuracy: 0.9335 - loss: 313.3907


366/Unknown  1760s 5s/step - accuracy: 0.9335 - loss: 313.2493


367/Unknown  1765s 5s/step - accuracy: 0.9336 - loss: 313.1085


368/Unknown  1770s 5s/step - accuracy: 0.9336 - loss: 312.9681


369/Unknown  1774s 5s/step - accuracy: 0.9336 - loss: 312.8284


370/Unknown  1779s 5s/step - accuracy: 0.9336 - loss: 312.6895


371/Unknown  1784s 5s/step - accuracy: 0.9337 - loss: 312.5517


372/Unknown  1790s 5s/step - accuracy: 0.9337 - loss: 312.4143


373/Unknown  1795s 5s/step - accuracy: 0.9337 - loss: 312.2771


374/Unknown  1800s 5s/step - accuracy: 0.9338 - loss: 312.1399


375/Unknown  1805s 5s/step - accuracy: 0.9338 - loss: 312.0028


376/Unknown  1810s 5s/step - accuracy: 0.9338 - loss: 311.8658


377/Unknown  1816s 5s/step - accuracy: 0.9338 - loss: 311.7296


378/Unknown  1821s 5s/step - accuracy: 0.9339 - loss: 311.5934


379/Unknown  1827s 5s/step - accuracy: 0.9339 - loss: 311.4581


380/Unknown  1832s 5s/step - accuracy: 0.9339 - loss: 311.3236


381/Unknown  1837s 5s/step - accuracy: 0.9340 - loss: 311.1892


382/Unknown  1842s 5s/step - accuracy: 0.9340 - loss: 311.0562


383/Unknown  1847s 5s/step - accuracy: 0.9340 - loss: 310.9238


384/Unknown  1852s 5s/step - accuracy: 0.9340 - loss: 310.7918


385/Unknown  1857s 5s/step - accuracy: 0.9341 - loss: 310.6603


386/Unknown  1863s 5s/step - accuracy: 0.9341 - loss: 310.5298


387/Unknown  1868s 5s/step - accuracy: 0.9341 - loss: 310.3997


388/Unknown  1873s 5s/step - accuracy: 0.9342 - loss: 310.2701


389/Unknown  1879s 5s/step - accuracy: 0.9342 - loss: 310.1407


390/Unknown  1884s 5s/step - accuracy: 0.9342 - loss: 310.0117


391/Unknown  1889s 5s/step - accuracy: 0.9342 - loss: 309.8839


392/Unknown  1894s 5s/step - accuracy: 0.9343 - loss: 309.7567


393/Unknown  1900s 5s/step - accuracy: 0.9343 - loss: 309.6298


394/Unknown  1905s 5s/step - accuracy: 0.9343 - loss: 309.5036


395/Unknown  1910s 5s/step - accuracy: 0.9343 - loss: 309.3778


396/Unknown  1915s 5s/step - accuracy: 0.9344 - loss: 309.2533


397/Unknown  1920s 5s/step - accuracy: 0.9344 - loss: 309.1294


398/Unknown  1925s 5s/step - accuracy: 0.9344 - loss: 309.0062


399/Unknown  1931s 5s/step - accuracy: 0.9345 - loss: 308.8835


400/Unknown  1936s 5s/step - accuracy: 0.9345 - loss: 308.7615


401/Unknown  1941s 5s/step - accuracy: 0.9345 - loss: 308.6405


402/Unknown  1946s 5s/step - accuracy: 0.9345 - loss: 308.5200


403/Unknown  1951s 5s/step - accuracy: 0.9346 - loss: 308.4001


404/Unknown  1957s 5s/step - accuracy: 0.9346 - loss: 308.2805


405/Unknown  1962s 5s/step - accuracy: 0.9346 - loss: 308.1610


406/Unknown  1968s 5s/step - accuracy: 0.9346 - loss: 308.0417


407/Unknown  1973s 5s/step - accuracy: 0.9347 - loss: 307.9226


408/Unknown  1978s 5s/step - accuracy: 0.9347 - loss: 307.8040


409/Unknown  1983s 5s/step - accuracy: 0.9347 - loss: 307.6858


410/Unknown  1989s 5s/step - accuracy: 0.9347 - loss: 307.5679


411/Unknown  1994s 5s/step - accuracy: 0.9348 - loss: 307.4503


412/Unknown  2000s 5s/step - accuracy: 0.9348 - loss: 307.3333


413/Unknown  2005s 5s/step - accuracy: 0.9348 - loss: 307.2177


414/Unknown  2010s 5s/step - accuracy: 0.9348 - loss: 307.1020


415/Unknown  2015s 5s/step - accuracy: 0.9349 - loss: 306.9868


416/Unknown  2021s 5s/step - accuracy: 0.9349 - loss: 306.8718


417/Unknown  2025s 5s/step - accuracy: 0.9349 - loss: 306.7578


418/Unknown  2031s 5s/step - accuracy: 0.9349 - loss: 306.6442


419/Unknown  2036s 5s/step - accuracy: 0.9350 - loss: 306.5310


420/Unknown  2041s 5s/step - accuracy: 0.9350 - loss: 306.4184


421/Unknown  2046s 5s/step - accuracy: 0.9350 - loss: 306.3060


422/Unknown  2052s 5s/step - accuracy: 0.9350 - loss: 306.1940


423/Unknown  2057s 5s/step - accuracy: 0.9350 - loss: 306.0820


424/Unknown  2062s 5s/step - accuracy: 0.9351 - loss: 305.9709


425/Unknown  2068s 5s/step - accuracy: 0.9351 - loss: 305.8597


426/Unknown  2073s 5s/step - accuracy: 0.9351 - loss: 305.7486


427/Unknown  2079s 5s/step - accuracy: 0.9351 - loss: 305.6375


428/Unknown  2083s 5s/step - accuracy: 0.9352 - loss: 305.5270


429/Unknown  2089s 5s/step - accuracy: 0.9352 - loss: 305.4171


430/Unknown  2094s 5s/step - accuracy: 0.9352 - loss: 305.3074


431/Unknown  2100s 5s/step - accuracy: 0.9352 - loss: 305.1978


432/Unknown  2105s 5s/step - accuracy: 0.9353 - loss: 305.0883


433/Unknown  2110s 5s/step - accuracy: 0.9353 - loss: 304.9792


434/Unknown  2116s 5s/step - accuracy: 0.9353 - loss: 304.8710


435/Unknown  2121s 5s/step - accuracy: 0.9353 - loss: 304.7630


436/Unknown  2127s 5s/step - accuracy: 0.9353 - loss: 304.6555


437/Unknown  2132s 5s/step - accuracy: 0.9354 - loss: 304.5482


438/Unknown  2137s 5s/step - accuracy: 0.9354 - loss: 304.4412


439/Unknown  2143s 5s/step - accuracy: 0.9354 - loss: 304.3348


440/Unknown  2148s 5s/step - accuracy: 0.9354 - loss: 304.2284


441/Unknown  2154s 5s/step - accuracy: 0.9355 - loss: 304.1221


442/Unknown  2159s 5s/step - accuracy: 0.9355 - loss: 304.0159


443/Unknown  2164s 5s/step - accuracy: 0.9355 - loss: 303.9098


444/Unknown  2170s 5s/step - accuracy: 0.9355 - loss: 303.8045


445/Unknown  2175s 5s/step - accuracy: 0.9355 - loss: 303.6997


446/Unknown  2181s 5s/step - accuracy: 0.9356 - loss: 303.5954


447/Unknown  2186s 5s/step - accuracy: 0.9356 - loss: 303.4915


448/Unknown  2191s 5s/step - accuracy: 0.9356 - loss: 303.3879


449/Unknown  2197s 5s/step - accuracy: 0.9356 - loss: 303.2853


450/Unknown  2202s 5s/step - accuracy: 0.9357 - loss: 303.1830


451/Unknown  2208s 5s/step - accuracy: 0.9357 - loss: 303.0814


452/Unknown  2213s 5s/step - accuracy: 0.9357 - loss: 302.9807


453/Unknown  2219s 5s/step - accuracy: 0.9357 - loss: 302.8800


454/Unknown  2224s 5s/step - accuracy: 0.9357 - loss: 302.7801


455/Unknown  2229s 5s/step - accuracy: 0.9358 - loss: 302.6810


456/Unknown  2235s 5s/step - accuracy: 0.9358 - loss: 302.5824


457/Unknown  2241s 5s/step - accuracy: 0.9358 - loss: 302.4843


458/Unknown  2246s 5s/step - accuracy: 0.9358 - loss: 302.3867


459/Unknown  2252s 5s/step - accuracy: 0.9358 - loss: 302.2893


460/Unknown  2257s 5s/step - accuracy: 0.9359 - loss: 302.1924


461/Unknown  2262s 5s/step - accuracy: 0.9359 - loss: 302.0958


462/Unknown  2268s 5s/step - accuracy: 0.9359 - loss: 301.9993


463/Unknown  2273s 5s/step - accuracy: 0.9359 - loss: 301.9033


464/Unknown  2279s 5s/step - accuracy: 0.9359 - loss: 301.8077


465/Unknown  2284s 5s/step - accuracy: 0.9360 - loss: 301.7121


466/Unknown  2289s 5s/step - accuracy: 0.9360 - loss: 301.6171


467/Unknown  2295s 5s/step - accuracy: 0.9360 - loss: 301.5226


468/Unknown  2300s 5s/step - accuracy: 0.9360 - loss: 301.4286


469/Unknown  2306s 5s/step - accuracy: 0.9360 - loss: 301.3351


470/Unknown  2311s 5s/step - accuracy: 0.9361 - loss: 301.2416


471/Unknown  2317s 5s/step - accuracy: 0.9361 - loss: 301.1487


472/Unknown  2322s 5s/step - accuracy: 0.9361 - loss: 301.0562


473/Unknown  2328s 5s/step - accuracy: 0.9361 - loss: 300.9637


474/Unknown  2334s 5s/step - accuracy: 0.9361 - loss: 300.8715


475/Unknown  2339s 5s/step - accuracy: 0.9361 - loss: 300.7798


476/Unknown  2344s 5s/step - accuracy: 0.9362 - loss: 300.6884


477/Unknown  2350s 5s/step - accuracy: 0.9362 - loss: 300.5971


478/Unknown  2355s 5s/step - accuracy: 0.9362 - loss: 300.5059


479/Unknown  2361s 5s/step - accuracy: 0.9362 - loss: 300.4148


480/Unknown  2367s 5s/step - accuracy: 0.9362 - loss: 300.3242


481/Unknown  2372s 5s/step - accuracy: 0.9363 - loss: 300.2337


482/Unknown  2378s 5s/step - accuracy: 0.9363 - loss: 300.1434


483/Unknown  2383s 5s/step - accuracy: 0.9363 - loss: 300.0537


484/Unknown  2389s 5s/step - accuracy: 0.9363 - loss: 299.9649


485/Unknown  2395s 5s/step - accuracy: 0.9363 - loss: 299.8761


486/Unknown  2400s 5s/step - accuracy: 0.9364 - loss: 299.7877


487/Unknown  2406s 5s/step - accuracy: 0.9364 - loss: 299.6995


488/Unknown  2412s 5s/step - accuracy: 0.9364 - loss: 299.6112


489/Unknown  2417s 5s/step - accuracy: 0.9364 - loss: 299.5231


490/Unknown  2423s 5s/step - accuracy: 0.9364 - loss: 299.4352


491/Unknown  2428s 5s/step - accuracy: 0.9364 - loss: 299.3475


492/Unknown  2434s 5s/step - accuracy: 0.9365 - loss: 299.2601


493/Unknown  2440s 5s/step - accuracy: 0.9365 - loss: 299.1726


494/Unknown  2445s 5s/step - accuracy: 0.9365 - loss: 299.0855


495/Unknown  2451s 5s/step - accuracy: 0.9365 - loss: 298.9986


496/Unknown  2456s 5s/step - accuracy: 0.9365 - loss: 298.9120


497/Unknown  2462s 5s/step - accuracy: 0.9366 - loss: 298.8255


498/Unknown  2467s 5s/step - accuracy: 0.9366 - loss: 298.7390


499/Unknown  2473s 5s/step - accuracy: 0.9366 - loss: 298.6525


500/Unknown  2479s 5s/step - accuracy: 0.9366 - loss: 298.5663


501/Unknown  2484s 5s/step - accuracy: 0.9366 - loss: 298.4802


502/Unknown  2489s 5s/step - accuracy: 0.9366 - loss: 298.3942


503/Unknown  2495s 5s/step - accuracy: 0.9367 - loss: 298.3082


504/Unknown  2500s 5s/step - accuracy: 0.9367 - loss: 298.2220


505/Unknown  2506s 5s/step - accuracy: 0.9367 - loss: 298.1357


506/Unknown  2512s 5s/step - accuracy: 0.9367 - loss: 298.0498


507/Unknown  2518s 5s/step - accuracy: 0.9367 - loss: 297.9643


508/Unknown  2524s 5s/step - accuracy: 0.9367 - loss: 297.8791


509/Unknown  2529s 5s/step - accuracy: 0.9368 - loss: 297.7941


510/Unknown  2535s 5s/step - accuracy: 0.9368 - loss: 297.7096


511/Unknown  2540s 5s/step - accuracy: 0.9368 - loss: 297.6250


512/Unknown  2546s 5s/step - accuracy: 0.9368 - loss: 297.5409


513/Unknown  2551s 5s/step - accuracy: 0.9368 - loss: 297.4570


514/Unknown  2558s 5s/step - accuracy: 0.9369 - loss: 297.3736


515/Unknown  2563s 5s/step - accuracy: 0.9369 - loss: 297.2903


516/Unknown  2569s 5s/step - accuracy: 0.9369 - loss: 297.2072


517/Unknown  2574s 5s/step - accuracy: 0.9369 - loss: 297.1243


518/Unknown  2580s 5s/step - accuracy: 0.9369 - loss: 297.0417


519/Unknown  2586s 5s/step - accuracy: 0.9369 - loss: 296.9596


520/Unknown  2592s 5s/step - accuracy: 0.9370 - loss: 296.8777


521/Unknown  2597s 5s/step - accuracy: 0.9370 - loss: 296.7960


522/Unknown  2603s 5s/step - accuracy: 0.9370 - loss: 296.7152


523/Unknown  2608s 5s/step - accuracy: 0.9370 - loss: 296.6345


524/Unknown  2614s 5s/step - accuracy: 0.9370 - loss: 296.5539


525/Unknown  2619s 5s/step - accuracy: 0.9370 - loss: 296.4738


526/Unknown  2625s 5s/step - accuracy: 0.9371 - loss: 296.3938


527/Unknown  2630s 5s/step - accuracy: 0.9371 - loss: 296.3137


528/Unknown  2636s 5s/step - accuracy: 0.9371 - loss: 296.2339


529/Unknown  2641s 5s/step - accuracy: 0.9371 - loss: 296.1544


530/Unknown  2647s 5s/step - accuracy: 0.9371 - loss: 296.0752


531/Unknown  2653s 5s/step - accuracy: 0.9371 - loss: 295.9962


532/Unknown  2658s 5s/step - accuracy: 0.9371 - loss: 295.9171


533/Unknown  2664s 5s/step - accuracy: 0.9372 - loss: 295.8383


534/Unknown  2670s 5s/step - accuracy: 0.9372 - loss: 295.7601


535/Unknown  2676s 5s/step - accuracy: 0.9372 - loss: 295.6820


536/Unknown  2682s 5s/step - accuracy: 0.9372 - loss: 295.6042


537/Unknown  2687s 5s/step - accuracy: 0.9372 - loss: 295.5266


538/Unknown  2693s 5s/step - accuracy: 0.9372 - loss: 295.4493


539/Unknown  2698s 5s/step - accuracy: 0.9373 - loss: 295.3722


540/Unknown  2704s 5s/step - accuracy: 0.9373 - loss: 295.2952


541/Unknown  2709s 5s/step - accuracy: 0.9373 - loss: 295.2184


542/Unknown  2715s 5s/step - accuracy: 0.9373 - loss: 295.1418


543/Unknown  2720s 5s/step - accuracy: 0.9373 - loss: 295.0652


544/Unknown  2726s 5s/step - accuracy: 0.9373 - loss: 294.9886


545/Unknown  2732s 5s/step - accuracy: 0.9374 - loss: 294.9122


546/Unknown  2738s 5s/step - accuracy: 0.9374 - loss: 294.8366


547/Unknown  2743s 5s/step - accuracy: 0.9374 - loss: 294.7612


548/Unknown  2749s 5s/step - accuracy: 0.9374 - loss: 294.6865


549/Unknown  2754s 5s/step - accuracy: 0.9374 - loss: 294.6119


550/Unknown  2760s 5s/step - accuracy: 0.9374 - loss: 294.5374


551/Unknown  2765s 5s/step - accuracy: 0.9374 - loss: 294.4630


552/Unknown  2771s 5s/step - accuracy: 0.9375 - loss: 294.3886


553/Unknown  2777s 5s/step - accuracy: 0.9375 - loss: 294.3147


554/Unknown  2783s 5s/step - accuracy: 0.9375 - loss: 294.2409


555/Unknown  2789s 5s/step - accuracy: 0.9375 - loss: 294.1676


556/Unknown  2794s 5s/step - accuracy: 0.9375 - loss: 294.0944


557/Unknown  2800s 5s/step - accuracy: 0.9375 - loss: 294.0215


558/Unknown  2806s 5s/step - accuracy: 0.9375 - loss: 293.9488


559/Unknown  2812s 5s/step - accuracy: 0.9376 - loss: 293.8765


560/Unknown  2817s 5s/step - accuracy: 0.9376 - loss: 293.8043


561/Unknown  2823s 5s/step - accuracy: 0.9376 - loss: 293.7322


562/Unknown  2829s 5s/step - accuracy: 0.9376 - loss: 293.6603


563/Unknown  2835s 5s/step - accuracy: 0.9376 - loss: 293.5886


564/Unknown  2841s 5s/step - accuracy: 0.9376 - loss: 293.5170


565/Unknown  2846s 5s/step - accuracy: 0.9376 - loss: 293.4457


566/Unknown  2852s 5s/step - accuracy: 0.9377 - loss: 293.3749


567/Unknown  2857s 5s/step - accuracy: 0.9377 - loss: 293.3044


568/Unknown  2863s 5s/step - accuracy: 0.9377 - loss: 293.2340


569/Unknown  2869s 5s/step - accuracy: 0.9377 - loss: 293.1638


570/Unknown  2875s 5s/step - accuracy: 0.9377 - loss: 293.0937


571/Unknown  2881s 5s/step - accuracy: 0.9377 - loss: 293.0238


572/Unknown  2886s 5s/step - accuracy: 0.9377 - loss: 292.9538


573/Unknown  2891s 5s/step - accuracy: 0.9378 - loss: 292.8841


574/Unknown  2897s 5s/step - accuracy: 0.9378 - loss: 292.8147


575/Unknown  2903s 5s/step - accuracy: 0.9378 - loss: 292.7452


576/Unknown  2909s 5s/step - accuracy: 0.9378 - loss: 292.6761


577/Unknown  2915s 5s/step - accuracy: 0.9378 - loss: 292.6073


578/Unknown  2921s 5s/step - accuracy: 0.9378 - loss: 292.5388


579/Unknown  2926s 5s/step - accuracy: 0.9378 - loss: 292.4704


580/Unknown  2932s 5s/step - accuracy: 0.9379 - loss: 292.4021


581/Unknown  2938s 5s/step - accuracy: 0.9379 - loss: 292.3340


582/Unknown  2944s 5s/step - accuracy: 0.9379 - loss: 292.2664


583/Unknown  2950s 5s/step - accuracy: 0.9379 - loss: 292.1990


584/Unknown  2955s 5s/step - accuracy: 0.9379 - loss: 292.1319


585/Unknown  2961s 5s/step - accuracy: 0.9379 - loss: 292.0648


586/Unknown  2967s 5s/step - accuracy: 0.9379 - loss: 291.9982


587/Unknown  2973s 5s/step - accuracy: 0.9380 - loss: 291.9317


588/Unknown  2979s 5s/step - accuracy: 0.9380 - loss: 291.8656


589/Unknown  2985s 5s/step - accuracy: 0.9380 - loss: 291.7996


590/Unknown  2990s 5s/step - accuracy: 0.9380 - loss: 291.7336


591/Unknown  2996s 5s/step - accuracy: 0.9380 - loss: 291.6675


592/Unknown  3002s 5s/step - accuracy: 0.9380 - loss: 291.6015


593/Unknown  3008s 5s/step - accuracy: 0.9380 - loss: 291.5358


594/Unknown  3014s 5s/step - accuracy: 0.9380 - loss: 291.4701


595/Unknown  3019s 5s/step - accuracy: 0.9381 - loss: 291.4042


596/Unknown  3025s 5s/step - accuracy: 0.9381 - loss: 291.3387


597/Unknown  3030s 5s/step - accuracy: 0.9381 - loss: 291.2737


598/Unknown  3036s 5s/step - accuracy: 0.9381 - loss: 291.2088


599/Unknown  3041s 5s/step - accuracy: 0.9381 - loss: 291.1442


600/Unknown  3047s 5s/step - accuracy: 0.9381 - loss: 291.0795


601/Unknown  3053s 5s/step - accuracy: 0.9381 - loss: 291.0151


602/Unknown  3058s 5s/step - accuracy: 0.9381 - loss: 290.9508


603/Unknown  3064s 5s/step - accuracy: 0.9382 - loss: 290.8867


604/Unknown  3071s 5s/step - accuracy: 0.9382 - loss: 290.8229


605/Unknown  3077s 5s/step - accuracy: 0.9382 - loss: 290.7592


606/Unknown  3083s 5s/step - accuracy: 0.9382 - loss: 290.6953


607/Unknown  3089s 5s/step - accuracy: 0.9382 - loss: 290.6319


608/Unknown  3095s 5s/step - accuracy: 0.9382 - loss: 290.5686


609/Unknown  3101s 5s/step - accuracy: 0.9382 - loss: 290.5056


610/Unknown  3107s 5s/step - accuracy: 0.9382 - loss: 290.4426


611/Unknown  3113s 5s/step - accuracy: 0.9383 - loss: 290.3796


612/Unknown  3119s 5s/step - accuracy: 0.9383 - loss: 290.3168


613/Unknown  3124s 5s/step - accuracy: 0.9383 - loss: 290.2538


614/Unknown  3130s 5s/step - accuracy: 0.9383 - loss: 290.1913


615/Unknown  3136s 5s/step - accuracy: 0.9383 - loss: 290.1292


616/Unknown  3142s 5s/step - accuracy: 0.9383 - loss: 290.0670


617/Unknown  3148s 5s/step - accuracy: 0.9383 - loss: 290.0049


618/Unknown  3153s 5s/step - accuracy: 0.9384 - loss: 289.9432


619/Unknown  3160s 5s/step - accuracy: 0.9384 - loss: 289.8817


620/Unknown  3166s 5s/step - accuracy: 0.9384 - loss: 289.8200


621/Unknown  3172s 5s/step - accuracy: 0.9384 - loss: 289.7584


622/Unknown  3177s 5s/step - accuracy: 0.9384 - loss: 289.6972


623/Unknown  3183s 5s/step - accuracy: 0.9384 - loss: 289.6363


624/Unknown  3190s 5s/step - accuracy: 0.9384 - loss: 289.5756


625/Unknown  3196s 5s/step - accuracy: 0.9384 - loss: 289.5153


626/Unknown  3201s 5s/step - accuracy: 0.9385 - loss: 289.4552


627/Unknown  3207s 5s/step - accuracy: 0.9385 - loss: 289.3954


628/Unknown  3212s 5s/step - accuracy: 0.9385 - loss: 289.3359


629/Unknown  3219s 5s/step - accuracy: 0.9385 - loss: 289.2766


630/Unknown  3224s 5s/step - accuracy: 0.9385 - loss: 289.2176


631/Unknown  3230s 5s/step - accuracy: 0.9385 - loss: 289.1588


632/Unknown  3237s 5s/step - accuracy: 0.9385 - loss: 289.1000


633/Unknown  3243s 5s/step - accuracy: 0.9385 - loss: 289.0413


634/Unknown  3249s 5s/step - accuracy: 0.9385 - loss: 288.9830


635/Unknown  3255s 5s/step - accuracy: 0.9386 - loss: 288.9246


636/Unknown  3261s 5s/step - accuracy: 0.9386 - loss: 288.8665


637/Unknown  3267s 5s/step - accuracy: 0.9386 - loss: 288.8085


638/Unknown  3274s 5s/step - accuracy: 0.9386 - loss: 288.7508


639/Unknown  3280s 5s/step - accuracy: 0.9386 - loss: 288.6931


640/Unknown  3286s 5s/step - accuracy: 0.9386 - loss: 288.6355


641/Unknown  3292s 5s/step - accuracy: 0.9386 - loss: 288.5780

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()


```
</div>
 641/641 ━━━━━━━━━━━━━━━━━━━━ 3383s 5s/step - accuracy: 0.9386 - loss: 288.5208 - val_accuracy: 0.9500 - val_loss: 230.6449


<div class="k-default-codeblock">
```
Model training finished.
Evaluating model performance...

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  1s 969ms/step - accuracy: 0.9585 - loss: 159.8705


  2/Unknown  2s 784ms/step - accuracy: 0.9538 - loss: 181.6473


  3/Unknown  3s 796ms/step - accuracy: 0.9503 - loss: 201.4849


  4/Unknown  3s 797ms/step - accuracy: 0.9479 - loss: 213.0796


  5/Unknown  4s 820ms/step - accuracy: 0.9473 - loss: 219.6181


  6/Unknown  5s 852ms/step - accuracy: 0.9476 - loss: 223.2487


  7/Unknown  6s 852ms/step - accuracy: 0.9482 - loss: 223.5582


  8/Unknown  7s 845ms/step - accuracy: 0.9488 - loss: 222.6751


  9/Unknown  8s 831ms/step - accuracy: 0.9493 - loss: 221.3878


 10/Unknown  8s 824ms/step - accuracy: 0.9497 - loss: 219.9191


 11/Unknown  9s 823ms/step - accuracy: 0.9501 - loss: 218.7335


 12/Unknown  10s 823ms/step - accuracy: 0.9503 - loss: 217.6873


 13/Unknown  11s 822ms/step - accuracy: 0.9504 - loss: 217.1383


 14/Unknown  12s 820ms/step - accuracy: 0.9505 - loss: 216.5114


 15/Unknown  12s 820ms/step - accuracy: 0.9506 - loss: 216.1290


 16/Unknown  13s 825ms/step - accuracy: 0.9507 - loss: 215.6186


 17/Unknown  14s 837ms/step - accuracy: 0.9509 - loss: 215.1525


 18/Unknown  15s 845ms/step - accuracy: 0.9510 - loss: 215.0510


 19/Unknown  16s 847ms/step - accuracy: 0.9510 - loss: 215.1579


 20/Unknown  17s 845ms/step - accuracy: 0.9511 - loss: 215.3697


 21/Unknown  18s 848ms/step - accuracy: 0.9511 - loss: 215.7154


 22/Unknown  19s 847ms/step - accuracy: 0.9511 - loss: 216.0274


 23/Unknown  20s 843ms/step - accuracy: 0.9510 - loss: 216.3101


 24/Unknown  20s 839ms/step - accuracy: 0.9510 - loss: 216.5270


 25/Unknown  21s 839ms/step - accuracy: 0.9510 - loss: 216.8894


 26/Unknown  22s 839ms/step - accuracy: 0.9510 - loss: 217.1713


 27/Unknown  23s 837ms/step - accuracy: 0.9510 - loss: 217.3974


 28/Unknown  24s 837ms/step - accuracy: 0.9510 - loss: 217.6237


 29/Unknown  24s 836ms/step - accuracy: 0.9510 - loss: 217.8101


 30/Unknown  25s 836ms/step - accuracy: 0.9509 - loss: 217.9722


 31/Unknown  26s 843ms/step - accuracy: 0.9509 - loss: 218.1803


 32/Unknown  27s 846ms/step - accuracy: 0.9509 - loss: 218.3734


 33/Unknown  28s 846ms/step - accuracy: 0.9508 - loss: 218.6991


 34/Unknown  29s 848ms/step - accuracy: 0.9507 - loss: 219.0575


 35/Unknown  30s 851ms/step - accuracy: 0.9506 - loss: 219.3678


 36/Unknown  31s 852ms/step - accuracy: 0.9506 - loss: 219.6365


 37/Unknown  32s 854ms/step - accuracy: 0.9505 - loss: 219.8749


 38/Unknown  33s 855ms/step - accuracy: 0.9505 - loss: 220.0916


 39/Unknown  33s 854ms/step - accuracy: 0.9504 - loss: 220.2763


 40/Unknown  34s 851ms/step - accuracy: 0.9504 - loss: 220.4740


 41/Unknown  35s 849ms/step - accuracy: 0.9504 - loss: 220.6481


 42/Unknown  36s 849ms/step - accuracy: 0.9503 - loss: 220.7809


 43/Unknown  37s 848ms/step - accuracy: 0.9503 - loss: 220.9724


 44/Unknown  37s 847ms/step - accuracy: 0.9503 - loss: 221.1543


 45/Unknown  38s 850ms/step - accuracy: 0.9502 - loss: 221.3142


 46/Unknown  39s 852ms/step - accuracy: 0.9502 - loss: 221.4476


 47/Unknown  40s 853ms/step - accuracy: 0.9502 - loss: 221.5518


 48/Unknown  41s 851ms/step - accuracy: 0.9501 - loss: 221.6600


 49/Unknown  42s 849ms/step - accuracy: 0.9501 - loss: 221.7569


 50/Unknown  43s 848ms/step - accuracy: 0.9501 - loss: 221.8995


 51/Unknown  43s 848ms/step - accuracy: 0.9500 - loss: 222.0217


 52/Unknown  44s 847ms/step - accuracy: 0.9500 - loss: 222.1107


 53/Unknown  45s 848ms/step - accuracy: 0.9500 - loss: 222.1740


 54/Unknown  46s 851ms/step - accuracy: 0.9500 - loss: 222.2534


 55/Unknown  47s 852ms/step - accuracy: 0.9500 - loss: 222.3450


 56/Unknown  48s 851ms/step - accuracy: 0.9499 - loss: 222.4026


 57/Unknown  49s 850ms/step - accuracy: 0.9499 - loss: 222.4373


 58/Unknown  49s 849ms/step - accuracy: 0.9499 - loss: 222.5173


 59/Unknown  50s 848ms/step - accuracy: 0.9499 - loss: 222.5910


 60/Unknown  51s 847ms/step - accuracy: 0.9499 - loss: 222.6672


 61/Unknown  52s 846ms/step - accuracy: 0.9499 - loss: 222.7544


 62/Unknown  53s 849ms/step - accuracy: 0.9499 - loss: 222.8256


 63/Unknown  54s 850ms/step - accuracy: 0.9498 - loss: 222.9229


 64/Unknown  54s 849ms/step - accuracy: 0.9498 - loss: 223.0382


 65/Unknown  55s 848ms/step - accuracy: 0.9498 - loss: 223.1463


 66/Unknown  56s 846ms/step - accuracy: 0.9498 - loss: 223.2352


 67/Unknown  57s 845ms/step - accuracy: 0.9498 - loss: 223.3278


 68/Unknown  58s 845ms/step - accuracy: 0.9497 - loss: 223.4134


 69/Unknown  58s 844ms/step - accuracy: 0.9497 - loss: 223.4989


 70/Unknown  59s 845ms/step - accuracy: 0.9497 - loss: 223.5741


 71/Unknown  60s 847ms/step - accuracy: 0.9497 - loss: 223.6425


 72/Unknown  61s 849ms/step - accuracy: 0.9497 - loss: 223.7317


 73/Unknown  62s 849ms/step - accuracy: 0.9497 - loss: 223.8276


 74/Unknown  63s 849ms/step - accuracy: 0.9497 - loss: 223.9164


 75/Unknown  64s 850ms/step - accuracy: 0.9496 - loss: 223.9929


 76/Unknown  65s 849ms/step - accuracy: 0.9496 - loss: 224.0643


 77/Unknown  65s 848ms/step - accuracy: 0.9496 - loss: 224.1311


 78/Unknown  66s 847ms/step - accuracy: 0.9496 - loss: 224.2050


 79/Unknown  67s 846ms/step - accuracy: 0.9496 - loss: 224.2719


 80/Unknown  68s 846ms/step - accuracy: 0.9496 - loss: 224.3298


 81/Unknown  69s 846ms/step - accuracy: 0.9496 - loss: 224.3810


 82/Unknown  69s 845ms/step - accuracy: 0.9495 - loss: 224.4418


 83/Unknown  70s 845ms/step - accuracy: 0.9495 - loss: 224.5023


 84/Unknown  71s 847ms/step - accuracy: 0.9495 - loss: 224.5610


 85/Unknown  72s 848ms/step - accuracy: 0.9495 - loss: 224.6218


 86/Unknown  73s 848ms/step - accuracy: 0.9495 - loss: 224.6898


 87/Unknown  74s 848ms/step - accuracy: 0.9494 - loss: 224.7545


 88/Unknown  75s 848ms/step - accuracy: 0.9494 - loss: 224.8130


 89/Unknown  76s 847ms/step - accuracy: 0.9494 - loss: 224.8751


 90/Unknown  76s 846ms/step - accuracy: 0.9494 - loss: 224.9411


 91/Unknown  77s 846ms/step - accuracy: 0.9494 - loss: 225.0047


 92/Unknown  78s 845ms/step - accuracy: 0.9493 - loss: 225.0705


 93/Unknown  79s 845ms/step - accuracy: 0.9493 - loss: 225.1329


 94/Unknown  80s 845ms/step - accuracy: 0.9493 - loss: 225.2086


 95/Unknown  81s 847ms/step - accuracy: 0.9493 - loss: 225.2813


 96/Unknown  82s 849ms/step - accuracy: 0.9493 - loss: 225.3567


 97/Unknown  82s 849ms/step - accuracy: 0.9492 - loss: 225.4247


 98/Unknown  83s 848ms/step - accuracy: 0.9492 - loss: 225.4865


 99/Unknown  84s 847ms/step - accuracy: 0.9492 - loss: 225.5487


100/Unknown  85s 846ms/step - accuracy: 0.9492 - loss: 225.6207


101/Unknown  85s 844ms/step - accuracy: 0.9492 - loss: 225.6911


102/Unknown  86s 843ms/step - accuracy: 0.9492 - loss: 225.7601


103/Unknown  87s 842ms/step - accuracy: 0.9491 - loss: 225.8305


104/Unknown  88s 841ms/step - accuracy: 0.9491 - loss: 225.8976


105/Unknown  88s 840ms/step - accuracy: 0.9491 - loss: 225.9651


106/Unknown  89s 840ms/step - accuracy: 0.9491 - loss: 226.0321


107/Unknown  90s 841ms/step - accuracy: 0.9491 - loss: 226.1055


108/Unknown  91s 841ms/step - accuracy: 0.9490 - loss: 226.1780


109/Unknown  92s 841ms/step - accuracy: 0.9490 - loss: 226.2571


110/Unknown  92s 840ms/step - accuracy: 0.9490 - loss: 226.3349


111/Unknown  93s 839ms/step - accuracy: 0.9490 - loss: 226.4156


112/Unknown  94s 839ms/step - accuracy: 0.9490 - loss: 226.4971


113/Unknown  95s 839ms/step - accuracy: 0.9490 - loss: 226.5796


114/Unknown  96s 839ms/step - accuracy: 0.9489 - loss: 226.6544


115/Unknown  97s 839ms/step - accuracy: 0.9489 - loss: 226.7242


116/Unknown  98s 840ms/step - accuracy: 0.9489 - loss: 226.7973


117/Unknown  98s 840ms/step - accuracy: 0.9489 - loss: 226.8732


118/Unknown  99s 840ms/step - accuracy: 0.9489 - loss: 226.9461


119/Unknown  100s 840ms/step - accuracy: 0.9489 - loss: 227.0159


120/Unknown  101s 841ms/step - accuracy: 0.9489 - loss: 227.0847


121/Unknown  102s 841ms/step - accuracy: 0.9489 - loss: 227.1547


122/Unknown  103s 842ms/step - accuracy: 0.9488 - loss: 227.2229


123/Unknown  104s 841ms/step - accuracy: 0.9488 - loss: 227.2818


124/Unknown  104s 840ms/step - accuracy: 0.9488 - loss: 227.3406


125/Unknown  105s 840ms/step - accuracy: 0.9488 - loss: 227.4021


126/Unknown  106s 839ms/step - accuracy: 0.9488 - loss: 227.4603


127/Unknown  107s 839ms/step - accuracy: 0.9488 - loss: 227.5192


128/Unknown  107s 839ms/step - accuracy: 0.9488 - loss: 227.5745


129/Unknown  108s 838ms/step - accuracy: 0.9488 - loss: 227.6251


130/Unknown  109s 838ms/step - accuracy: 0.9487 - loss: 227.6736


131/Unknown  110s 838ms/step - accuracy: 0.9487 - loss: 227.7185


132/Unknown  111s 839ms/step - accuracy: 0.9487 - loss: 227.7645


133/Unknown  112s 840ms/step - accuracy: 0.9487 - loss: 227.8065


134/Unknown  113s 839ms/step - accuracy: 0.9487 - loss: 227.8499


135/Unknown  113s 839ms/step - accuracy: 0.9487 - loss: 227.8890


136/Unknown  114s 838ms/step - accuracy: 0.9487 - loss: 227.9345


137/Unknown  115s 838ms/step - accuracy: 0.9487 - loss: 227.9822


138/Unknown  116s 838ms/step - accuracy: 0.9487 - loss: 228.0268


139/Unknown  117s 838ms/step - accuracy: 0.9487 - loss: 228.0708


140/Unknown  117s 838ms/step - accuracy: 0.9487 - loss: 228.1134


141/Unknown  118s 839ms/step - accuracy: 0.9487 - loss: 228.1554


142/Unknown  119s 839ms/step - accuracy: 0.9487 - loss: 228.1968


143/Unknown  120s 839ms/step - accuracy: 0.9487 - loss: 228.2378


144/Unknown  121s 838ms/step - accuracy: 0.9486 - loss: 228.2773


145/Unknown  122s 837ms/step - accuracy: 0.9486 - loss: 228.3152


146/Unknown  122s 837ms/step - accuracy: 0.9486 - loss: 228.3522


147/Unknown  123s 837ms/step - accuracy: 0.9486 - loss: 228.3878


148/Unknown  124s 836ms/step - accuracy: 0.9486 - loss: 228.4215


149/Unknown  125s 836ms/step - accuracy: 0.9486 - loss: 228.4591


150/Unknown  126s 837ms/step - accuracy: 0.9486 - loss: 228.4945


151/Unknown  127s 837ms/step - accuracy: 0.9486 - loss: 228.5322


152/Unknown  127s 837ms/step - accuracy: 0.9486 - loss: 228.5719


153/Unknown  128s 837ms/step - accuracy: 0.9486 - loss: 228.6125


154/Unknown  129s 836ms/step - accuracy: 0.9486 - loss: 228.6527


155/Unknown  130s 835ms/step - accuracy: 0.9486 - loss: 228.6951


156/Unknown  130s 834ms/step - accuracy: 0.9486 - loss: 228.7370


157/Unknown  131s 833ms/step - accuracy: 0.9486 - loss: 228.7771


158/Unknown  132s 833ms/step - accuracy: 0.9486 - loss: 228.8156


159/Unknown  132s 832ms/step - accuracy: 0.9486 - loss: 228.8530


160/Unknown  133s 831ms/step - accuracy: 0.9486 - loss: 228.8878


161/Unknown  134s 831ms/step - accuracy: 0.9486 - loss: 228.9214


162/Unknown  135s 831ms/step - accuracy: 0.9486 - loss: 228.9536


163/Unknown  136s 831ms/step - accuracy: 0.9485 - loss: 228.9850


164/Unknown  136s 830ms/step - accuracy: 0.9485 - loss: 229.0151


165/Unknown  137s 830ms/step - accuracy: 0.9485 - loss: 229.0431


166/Unknown  138s 830ms/step - accuracy: 0.9485 - loss: 229.0700


167/Unknown  139s 829ms/step - accuracy: 0.9485 - loss: 229.0960


168/Unknown  139s 829ms/step - accuracy: 0.9485 - loss: 229.1205


169/Unknown  140s 829ms/step - accuracy: 0.9485 - loss: 229.1428


170/Unknown  141s 830ms/step - accuracy: 0.9485 - loss: 229.1650


171/Unknown  142s 830ms/step - accuracy: 0.9485 - loss: 229.1879


172/Unknown  143s 830ms/step - accuracy: 0.9485 - loss: 229.2117


173/Unknown  144s 830ms/step - accuracy: 0.9485 - loss: 229.2364


174/Unknown  144s 829ms/step - accuracy: 0.9485 - loss: 229.2587


175/Unknown  145s 829ms/step - accuracy: 0.9485 - loss: 229.2807


176/Unknown  146s 829ms/step - accuracy: 0.9485 - loss: 229.3035


177/Unknown  147s 829ms/step - accuracy: 0.9485 - loss: 229.3256


178/Unknown  148s 829ms/step - accuracy: 0.9485 - loss: 229.3470


179/Unknown  148s 829ms/step - accuracy: 0.9485 - loss: 229.3654


180/Unknown  149s 828ms/step - accuracy: 0.9485 - loss: 229.3816


181/Unknown  150s 828ms/step - accuracy: 0.9485 - loss: 229.3967


182/Unknown  151s 829ms/step - accuracy: 0.9485 - loss: 229.4137


183/Unknown  152s 829ms/step - accuracy: 0.9485 - loss: 229.4307


184/Unknown  153s 829ms/step - accuracy: 0.9485 - loss: 229.4493


185/Unknown  154s 829ms/step - accuracy: 0.9485 - loss: 229.4674


186/Unknown  154s 828ms/step - accuracy: 0.9485 - loss: 229.4838


187/Unknown  155s 828ms/step - accuracy: 0.9485 - loss: 229.4995


188/Unknown  156s 828ms/step - accuracy: 0.9485 - loss: 229.5143


189/Unknown  157s 828ms/step - accuracy: 0.9485 - loss: 229.5294


190/Unknown  157s 828ms/step - accuracy: 0.9485 - loss: 229.5453


191/Unknown  158s 828ms/step - accuracy: 0.9485 - loss: 229.5600


192/Unknown  159s 828ms/step - accuracy: 0.9485 - loss: 229.5728


193/Unknown  160s 828ms/step - accuracy: 0.9485 - loss: 229.5844


194/Unknown  161s 828ms/step - accuracy: 0.9485 - loss: 229.5949


195/Unknown  162s 828ms/step - accuracy: 0.9485 - loss: 229.6060


196/Unknown  163s 829ms/step - accuracy: 0.9485 - loss: 229.6188


197/Unknown  163s 829ms/step - accuracy: 0.9485 - loss: 229.6285


198/Unknown  164s 828ms/step - accuracy: 0.9485 - loss: 229.6418


199/Unknown  165s 828ms/step - accuracy: 0.9485 - loss: 229.6544


200/Unknown  166s 828ms/step - accuracy: 0.9485 - loss: 229.6676


201/Unknown  166s 827ms/step - accuracy: 0.9485 - loss: 229.6789


202/Unknown  167s 827ms/step - accuracy: 0.9485 - loss: 229.6895


203/Unknown  168s 827ms/step - accuracy: 0.9485 - loss: 229.7000


204/Unknown  169s 826ms/step - accuracy: 0.9485 - loss: 229.7093


205/Unknown  169s 826ms/step - accuracy: 0.9485 - loss: 229.7209


206/Unknown  170s 826ms/step - accuracy: 0.9485 - loss: 229.7325


207/Unknown  171s 826ms/step - accuracy: 0.9485 - loss: 229.7428


208/Unknown  172s 826ms/step - accuracy: 0.9485 - loss: 229.7533


209/Unknown  173s 826ms/step - accuracy: 0.9485 - loss: 229.7653


210/Unknown  174s 826ms/step - accuracy: 0.9485 - loss: 229.7759


211/Unknown  175s 827ms/step - accuracy: 0.9485 - loss: 229.7850


212/Unknown  175s 827ms/step - accuracy: 0.9485 - loss: 229.7936


213/Unknown  176s 827ms/step - accuracy: 0.9485 - loss: 229.8021


214/Unknown  177s 826ms/step - accuracy: 0.9485 - loss: 229.8111


215/Unknown  178s 826ms/step - accuracy: 0.9485 - loss: 229.8193


216/Unknown  179s 826ms/step - accuracy: 0.9485 - loss: 229.8291


217/Unknown  179s 826ms/step - accuracy: 0.9485 - loss: 229.8381


218/Unknown  180s 826ms/step - accuracy: 0.9485 - loss: 229.8456


219/Unknown  181s 826ms/step - accuracy: 0.9485 - loss: 229.8517


220/Unknown  182s 826ms/step - accuracy: 0.9485 - loss: 229.8562


221/Unknown  183s 827ms/step - accuracy: 0.9485 - loss: 229.8600


222/Unknown  184s 827ms/step - accuracy: 0.9485 - loss: 229.8624


223/Unknown  184s 826ms/step - accuracy: 0.9485 - loss: 229.8643


224/Unknown  185s 826ms/step - accuracy: 0.9486 - loss: 229.8661


225/Unknown  186s 826ms/step - accuracy: 0.9486 - loss: 229.8689


226/Unknown  187s 826ms/step - accuracy: 0.9486 - loss: 229.8717


227/Unknown  188s 826ms/step - accuracy: 0.9486 - loss: 229.8735


228/Unknown  188s 826ms/step - accuracy: 0.9486 - loss: 229.8744


229/Unknown  189s 827ms/step - accuracy: 0.9486 - loss: 229.8756


230/Unknown  190s 827ms/step - accuracy: 0.9486 - loss: 229.8762


231/Unknown  191s 827ms/step - accuracy: 0.9486 - loss: 229.8766


232/Unknown  192s 827ms/step - accuracy: 0.9486 - loss: 229.8752


233/Unknown  193s 827ms/step - accuracy: 0.9486 - loss: 229.8736


234/Unknown  194s 828ms/step - accuracy: 0.9486 - loss: 229.8711


235/Unknown  195s 828ms/step - accuracy: 0.9486 - loss: 229.8693


236/Unknown  196s 828ms/step - accuracy: 0.9486 - loss: 229.8688


237/Unknown  196s 828ms/step - accuracy: 0.9486 - loss: 229.8675


238/Unknown  197s 828ms/step - accuracy: 0.9486 - loss: 229.8669


239/Unknown  198s 828ms/step - accuracy: 0.9486 - loss: 229.8669


240/Unknown  199s 828ms/step - accuracy: 0.9486 - loss: 229.8657


241/Unknown  200s 827ms/step - accuracy: 0.9486 - loss: 229.8640


242/Unknown  200s 827ms/step - accuracy: 0.9486 - loss: 229.8625


243/Unknown  201s 827ms/step - accuracy: 0.9486 - loss: 229.8612


244/Unknown  202s 828ms/step - accuracy: 0.9486 - loss: 229.8591


245/Unknown  203s 829ms/step - accuracy: 0.9486 - loss: 229.8574


246/Unknown  204s 829ms/step - accuracy: 0.9486 - loss: 229.8549


247/Unknown  205s 829ms/step - accuracy: 0.9486 - loss: 229.8528


248/Unknown  206s 829ms/step - accuracy: 0.9486 - loss: 229.8504


249/Unknown  206s 829ms/step - accuracy: 0.9486 - loss: 229.8476


250/Unknown  207s 829ms/step - accuracy: 0.9486 - loss: 229.8455


251/Unknown  208s 829ms/step - accuracy: 0.9486 - loss: 229.8447


252/Unknown  209s 829ms/step - accuracy: 0.9486 - loss: 229.8435


253/Unknown  210s 829ms/step - accuracy: 0.9487 - loss: 229.8426


254/Unknown  211s 829ms/step - accuracy: 0.9487 - loss: 229.8415


255/Unknown  212s 829ms/step - accuracy: 0.9487 - loss: 229.8404


256/Unknown  212s 829ms/step - accuracy: 0.9487 - loss: 229.8398


257/Unknown  213s 830ms/step - accuracy: 0.9487 - loss: 229.8387


258/Unknown  214s 830ms/step - accuracy: 0.9487 - loss: 229.8384


259/Unknown  215s 830ms/step - accuracy: 0.9487 - loss: 229.8384


260/Unknown  216s 830ms/step - accuracy: 0.9487 - loss: 229.8374


261/Unknown  217s 830ms/step - accuracy: 0.9487 - loss: 229.8360


262/Unknown  218s 830ms/step - accuracy: 0.9487 - loss: 229.8331


263/Unknown  218s 830ms/step - accuracy: 0.9487 - loss: 229.8302


264/Unknown  219s 830ms/step - accuracy: 0.9487 - loss: 229.8270


265/Unknown  220s 830ms/step - accuracy: 0.9487 - loss: 229.8239


266/Unknown  221s 829ms/step - accuracy: 0.9487 - loss: 229.8224


267/Unknown  222s 829ms/step - accuracy: 0.9487 - loss: 229.8216


268/Unknown  222s 829ms/step - accuracy: 0.9487 - loss: 229.8201


269/Unknown  223s 830ms/step - accuracy: 0.9487 - loss: 229.8193


270/Unknown  224s 830ms/step - accuracy: 0.9487 - loss: 229.8182


271/Unknown  225s 831ms/step - accuracy: 0.9487 - loss: 229.8172


272/Unknown  226s 831ms/step - accuracy: 0.9487 - loss: 229.8150


273/Unknown  227s 831ms/step - accuracy: 0.9487 - loss: 229.8130


274/Unknown  228s 831ms/step - accuracy: 0.9487 - loss: 229.8102


275/Unknown  228s 830ms/step - accuracy: 0.9487 - loss: 229.8071


276/Unknown  229s 830ms/step - accuracy: 0.9487 - loss: 229.8048


277/Unknown  230s 830ms/step - accuracy: 0.9487 - loss: 229.8029


278/Unknown  231s 830ms/step - accuracy: 0.9487 - loss: 229.8008


279/Unknown  232s 831ms/step - accuracy: 0.9487 - loss: 229.7984


280/Unknown  233s 831ms/step - accuracy: 0.9487 - loss: 229.7951


281/Unknown  234s 831ms/step - accuracy: 0.9488 - loss: 229.7920


282/Unknown  234s 831ms/step - accuracy: 0.9488 - loss: 229.7879


283/Unknown  235s 831ms/step - accuracy: 0.9488 - loss: 229.7829


284/Unknown  236s 831ms/step - accuracy: 0.9488 - loss: 229.7788


285/Unknown  237s 831ms/step - accuracy: 0.9488 - loss: 229.7740


286/Unknown  238s 831ms/step - accuracy: 0.9488 - loss: 229.7697


287/Unknown  239s 831ms/step - accuracy: 0.9488 - loss: 229.7654


288/Unknown  239s 830ms/step - accuracy: 0.9488 - loss: 229.7608


289/Unknown  240s 830ms/step - accuracy: 0.9488 - loss: 229.7555


290/Unknown  241s 831ms/step - accuracy: 0.9488 - loss: 229.7503


291/Unknown  242s 831ms/step - accuracy: 0.9488 - loss: 229.7450


292/Unknown  243s 831ms/step - accuracy: 0.9488 - loss: 229.7401


293/Unknown  244s 831ms/step - accuracy: 0.9488 - loss: 229.7349


294/Unknown  244s 831ms/step - accuracy: 0.9488 - loss: 229.7300


295/Unknown  245s 831ms/step - accuracy: 0.9488 - loss: 229.7247


296/Unknown  246s 831ms/step - accuracy: 0.9488 - loss: 229.7213


297/Unknown  247s 830ms/step - accuracy: 0.9488 - loss: 229.7185


298/Unknown  248s 831ms/step - accuracy: 0.9488 - loss: 229.7160


299/Unknown  249s 831ms/step - accuracy: 0.9488 - loss: 229.7128


300/Unknown  250s 832ms/step - accuracy: 0.9488 - loss: 229.7092


301/Unknown  250s 832ms/step - accuracy: 0.9488 - loss: 229.7068


302/Unknown  251s 832ms/step - accuracy: 0.9488 - loss: 229.7043


303/Unknown  252s 831ms/step - accuracy: 0.9488 - loss: 229.7027


304/Unknown  253s 831ms/step - accuracy: 0.9488 - loss: 229.7015


305/Unknown  254s 831ms/step - accuracy: 0.9488 - loss: 229.7003


306/Unknown  254s 831ms/step - accuracy: 0.9489 - loss: 229.6985


307/Unknown  255s 831ms/step - accuracy: 0.9489 - loss: 229.6976


308/Unknown  256s 831ms/step - accuracy: 0.9489 - loss: 229.6960


309/Unknown  257s 831ms/step - accuracy: 0.9489 - loss: 229.6949


310/Unknown  258s 831ms/step - accuracy: 0.9489 - loss: 229.6949


311/Unknown  259s 831ms/step - accuracy: 0.9489 - loss: 229.6954


312/Unknown  260s 832ms/step - accuracy: 0.9489 - loss: 229.6953


313/Unknown  260s 832ms/step - accuracy: 0.9489 - loss: 229.6953


314/Unknown  261s 832ms/step - accuracy: 0.9489 - loss: 229.6953


315/Unknown  262s 832ms/step - accuracy: 0.9489 - loss: 229.6954


316/Unknown  263s 833ms/step - accuracy: 0.9489 - loss: 229.6950


317/Unknown  264s 833ms/step - accuracy: 0.9489 - loss: 229.6952


318/Unknown  265s 832ms/step - accuracy: 0.9489 - loss: 229.6959


319/Unknown  266s 832ms/step - accuracy: 0.9489 - loss: 229.6970


320/Unknown  266s 832ms/step - accuracy: 0.9489 - loss: 229.6978


321/Unknown  267s 832ms/step - accuracy: 0.9489 - loss: 229.6986


322/Unknown  268s 832ms/step - accuracy: 0.9489 - loss: 229.6996


323/Unknown  269s 832ms/step - accuracy: 0.9489 - loss: 229.7004


324/Unknown  270s 832ms/step - accuracy: 0.9489 - loss: 229.7011


325/Unknown  270s 831ms/step - accuracy: 0.9489 - loss: 229.7024


326/Unknown  271s 831ms/step - accuracy: 0.9489 - loss: 229.7045


327/Unknown  272s 831ms/step - accuracy: 0.9489 - loss: 229.7063


328/Unknown  273s 832ms/step - accuracy: 0.9489 - loss: 229.7081


329/Unknown  274s 832ms/step - accuracy: 0.9489 - loss: 229.7094


330/Unknown  275s 833ms/step - accuracy: 0.9489 - loss: 229.7113


331/Unknown  276s 832ms/step - accuracy: 0.9489 - loss: 229.7128


332/Unknown  276s 832ms/step - accuracy: 0.9489 - loss: 229.7154


333/Unknown  277s 832ms/step - accuracy: 0.9489 - loss: 229.7180


334/Unknown  278s 832ms/step - accuracy: 0.9489 - loss: 229.7203


335/Unknown  279s 832ms/step - accuracy: 0.9489 - loss: 229.7221


336/Unknown  280s 832ms/step - accuracy: 0.9489 - loss: 229.7232


337/Unknown  280s 832ms/step - accuracy: 0.9489 - loss: 229.7246


338/Unknown  281s 832ms/step - accuracy: 0.9489 - loss: 229.7261


339/Unknown  282s 832ms/step - accuracy: 0.9489 - loss: 229.7272


340/Unknown  283s 832ms/step - accuracy: 0.9489 - loss: 229.7288


341/Unknown  284s 832ms/step - accuracy: 0.9489 - loss: 229.7313


342/Unknown  285s 832ms/step - accuracy: 0.9489 - loss: 229.7336


343/Unknown  286s 832ms/step - accuracy: 0.9489 - loss: 229.7360


344/Unknown  286s 832ms/step - accuracy: 0.9490 - loss: 229.7383


345/Unknown  287s 832ms/step - accuracy: 0.9490 - loss: 229.7407


346/Unknown  288s 832ms/step - accuracy: 0.9490 - loss: 229.7439


347/Unknown  289s 832ms/step - accuracy: 0.9490 - loss: 229.7469


348/Unknown  290s 832ms/step - accuracy: 0.9490 - loss: 229.7512


349/Unknown  291s 832ms/step - accuracy: 0.9490 - loss: 229.7557


350/Unknown  291s 832ms/step - accuracy: 0.9490 - loss: 229.7600


351/Unknown  292s 832ms/step - accuracy: 0.9490 - loss: 229.7649


352/Unknown  293s 833ms/step - accuracy: 0.9490 - loss: 229.7693


353/Unknown  294s 832ms/step - accuracy: 0.9490 - loss: 229.7743


354/Unknown  295s 832ms/step - accuracy: 0.9490 - loss: 229.7806


355/Unknown  295s 832ms/step - accuracy: 0.9490 - loss: 229.7871


356/Unknown  296s 832ms/step - accuracy: 0.9490 - loss: 229.7933


357/Unknown  297s 832ms/step - accuracy: 0.9490 - loss: 229.7989


358/Unknown  298s 832ms/step - accuracy: 0.9490 - loss: 229.8050


359/Unknown  299s 832ms/step - accuracy: 0.9490 - loss: 229.8113


360/Unknown  299s 831ms/step - accuracy: 0.9490 - loss: 229.8174


361/Unknown  300s 831ms/step - accuracy: 0.9490 - loss: 229.8240


362/Unknown  301s 832ms/step - accuracy: 0.9490 - loss: 229.8305


363/Unknown  302s 832ms/step - accuracy: 0.9490 - loss: 229.8366


364/Unknown  303s 832ms/step - accuracy: 0.9490 - loss: 229.8427


365/Unknown  304s 832ms/step - accuracy: 0.9490 - loss: 229.8484


366/Unknown  304s 831ms/step - accuracy: 0.9490 - loss: 229.8530


367/Unknown  305s 831ms/step - accuracy: 0.9490 - loss: 229.8573


368/Unknown  306s 831ms/step - accuracy: 0.9490 - loss: 229.8619


369/Unknown  307s 831ms/step - accuracy: 0.9490 - loss: 229.8664


370/Unknown  308s 831ms/step - accuracy: 0.9490 - loss: 229.8705


371/Unknown  308s 831ms/step - accuracy: 0.9490 - loss: 229.8742


372/Unknown  309s 831ms/step - accuracy: 0.9490 - loss: 229.8776


373/Unknown  310s 832ms/step - accuracy: 0.9490 - loss: 229.8809


374/Unknown  311s 832ms/step - accuracy: 0.9490 - loss: 229.8844


375/Unknown  312s 832ms/step - accuracy: 0.9490 - loss: 229.8880


376/Unknown  313s 832ms/step - accuracy: 0.9490 - loss: 229.8915


377/Unknown  314s 832ms/step - accuracy: 0.9490 - loss: 229.8950


```
</div>
 377/377 ━━━━━━━━━━━━━━━━━━━━ 314s 832ms/step - accuracy: 0.9490 - loss: 229.8986


<div class="k-default-codeblock">
```
Test accuracy: 94.95%

```
</div>
You should achieve more than 95% accuracy on the test set.

To increase the learning capacity of the model, you can try increasing the
`encoding_size` value, or stacking multiple GRN layers on top of the VSN layer.
This may require to also increase the `dropout_rate` value to avoid overfitting.

**Example available on HuggingFace**

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Classification%20With%20GRN%20%26%20VSN-red)](https://huggingface.co/keras-io/structured-data-classification-grn-vsn) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Classification%20With%20GRN%20%26%20VSN-red)](https://huggingface.co/spaces/keras-io/structured-data-classification-grn-vsn) |
