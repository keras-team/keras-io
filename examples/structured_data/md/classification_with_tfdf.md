# Classification with TensorFlow Decision Forests

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2022/01/15<br>
**Last modified:** 2022/01/15<br>
**Description:** Using TensorFlow Decision Forests for structured data classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/classification_with_tfdf.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/classification_with_tfdf.py)



---
## Introduction

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests)
is a collection of state-of-the-art algorithms of Decision Forest models
that are compatible with Keras APIs.
The models include [Random Forests](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel),
[Gradient Boosted Trees](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel),
and [CART](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/CartModel),
and can be used for regression, classification, and ranking task.

This example uses Gradient Boosted Trees model in binary classification of
structured data, and covers the following scenarios:

1. Build a decision forests model by specifying the input feature usage.
2. Implement a custom *Binary Target encoder* as a [Keras Preprocessing layer](https://keras.io/api/layers/preprocessing_layers/)
to encode the categorical features with respect to their target value co-occurrences,
and then use the encoded features to build a decision forests model.
3. Encode the categorical features as [embeddings](https://keras.io/api/layers/core_layers/embedding),
train these embeddings in a simple linear model, and then use the
trained embeddings as inputs to build decision forests model.

This example uses TensorFlow 2.7 or higher,
as well as [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests),
which you can install using the following command:

```python
pip install -U tensorflow_decision_forests
```

---
## Setup


```python
import math
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
```

---
## Prepare the data

This example uses the
[United States Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29)
provided by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The task is binary classification to determine whether a person makes over 50K a year.

The dataset includes ~300K instances with 41 input features: 7 numerical features
and 34 categorical features.

First we load the data from the UCI Machine Learning Repository into a Pandas DataFrame.


```python
BASE_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income"
CSV_HEADER = [
    l.decode("utf-8").split(":")[0].replace(" ", "_")
    for l in urllib.request.urlopen(f"{BASE_PATH}.names")
    if not l.startswith(b"|")
][2:]
CSV_HEADER.append("income_level")

train_data = pd.read_csv(f"{BASE_PATH}.data.gz", header=None, names=CSV_HEADER,)
test_data = pd.read_csv(f"{BASE_PATH}.test.gz", header=None, names=CSV_HEADER,)
```

We convert the target column from string to integer.


```python
target_labels = [" - 50000.", " 50000+."]
train_data["income_level"] = train_data["income_level"].map(target_labels.index)
test_data["income_level"] = test_data["income_level"].map(target_labels.index)
```

Now let's show the shapes of the training and test dataframes, and display some instances.


```python
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(train_data.head().T)
```

<div class="k-default-codeblock">
```
Train data shape: (199523, 42)
Test data shape: (99762, 42)
                                                                                    0  \
age                                                                                73   
class_of_worker                                                       Not in universe   
detailed_industry_recode                                                            0   
detailed_occupation_recode                                                          0   
education                                                        High school graduate   
wage_per_hour                                                                       0   
enroll_in_edu_inst_last_wk                                            Not in universe   
marital_stat                                                                  Widowed   
major_industry_code                                       Not in universe or children   
major_occupation_code                                                 Not in universe   
race                                                                            White   
hispanic_origin                                                             All other   
sex                                                                            Female   
member_of_a_labor_union                                               Not in universe   
reason_for_unemployment                                               Not in universe   
full_or_part_time_employment_stat                                  Not in labor force   
capital_gains                                                                       0   
capital_losses                                                                      0   
dividends_from_stocks                                                               0   
tax_filer_stat                                                               Nonfiler   
region_of_previous_residence                                          Not in universe   
state_of_previous_residence                                           Not in universe   
detailed_household_and_family_stat           Other Rel 18+ ever marr not in subfamily   
detailed_household_summary_in_household                 Other relative of householder   
instance_weight                                                               1700.09   
migration_code-change_in_msa                                                        ?   
migration_code-change_in_reg                                                        ?   
migration_code-move_within_reg                                                      ?   
live_in_this_house_1_year_ago                        Not in universe under 1 year old   
migration_prev_res_in_sunbelt                                                       ?   
num_persons_worked_for_employer                                                     0   
family_members_under_18                                               Not in universe   
country_of_birth_father                                                 United-States   
country_of_birth_mother                                                 United-States   
country_of_birth_self                                                   United-States   
citizenship                                         Native- Born in the United States   
own_business_or_self_employed                                                       0   
fill_inc_questionnaire_for_veteran's_admin                            Not in universe   
veterans_benefits                                                                   2   
weeks_worked_in_year                                                                0   
year                                                                               95   
income_level                                                                        0   
```
</div>
    
<div class="k-default-codeblock">
```
                                                                               1  \
age                                                                           58   
class_of_worker                                   Self-employed-not incorporated   
detailed_industry_recode                                                       4   
detailed_occupation_recode                                                    34   
education                                             Some college but no degree   
wage_per_hour                                                                  0   
enroll_in_edu_inst_last_wk                                       Not in universe   
marital_stat                                                            Divorced   
major_industry_code                                                 Construction   
major_occupation_code                        Precision production craft & repair   
race                                                                       White   
hispanic_origin                                                        All other   
sex                                                                         Male   
member_of_a_labor_union                                          Not in universe   
reason_for_unemployment                                          Not in universe   
full_or_part_time_employment_stat                       Children or Armed Forces   
capital_gains                                                                  0   
capital_losses                                                                 0   
dividends_from_stocks                                                          0   
tax_filer_stat                                                 Head of household   
region_of_previous_residence                                               South   
state_of_previous_residence                                             Arkansas   
detailed_household_and_family_stat                                   Householder   
detailed_household_summary_in_household                              Householder   
instance_weight                                                          1053.55   
migration_code-change_in_msa                                          MSA to MSA   
migration_code-change_in_reg                                         Same county   
migration_code-move_within_reg                                       Same county   
live_in_this_house_1_year_ago                                                 No   
migration_prev_res_in_sunbelt                                                Yes   
num_persons_worked_for_employer                                                1   
family_members_under_18                                          Not in universe   
country_of_birth_father                                            United-States   
country_of_birth_mother                                            United-States   
country_of_birth_self                                              United-States   
citizenship                                    Native- Born in the United States   
own_business_or_self_employed                                                  0   
fill_inc_questionnaire_for_veteran's_admin                       Not in universe   
veterans_benefits                                                              2   
weeks_worked_in_year                                                          52   
year                                                                          94   
income_level                                                                   0   
```
</div>
    
<div class="k-default-codeblock">
```
                                                                                   2  \
age                                                                               18   
class_of_worker                                                      Not in universe   
detailed_industry_recode                                                           0   
detailed_occupation_recode                                                         0   
education                                                                 10th grade   
wage_per_hour                                                                      0   
enroll_in_edu_inst_last_wk                                               High school   
marital_stat                                                           Never married   
major_industry_code                                      Not in universe or children   
major_occupation_code                                                Not in universe   
race                                                       Asian or Pacific Islander   
hispanic_origin                                                            All other   
sex                                                                           Female   
member_of_a_labor_union                                              Not in universe   
reason_for_unemployment                                              Not in universe   
full_or_part_time_employment_stat                                 Not in labor force   
capital_gains                                                                      0   
capital_losses                                                                     0   
dividends_from_stocks                                                              0   
tax_filer_stat                                                              Nonfiler   
region_of_previous_residence                                         Not in universe   
state_of_previous_residence                                          Not in universe   
detailed_household_and_family_stat           Child 18+ never marr Not in a subfamily   
detailed_household_summary_in_household                            Child 18 or older   
instance_weight                                                               991.95   
migration_code-change_in_msa                                                       ?   
migration_code-change_in_reg                                                       ?   
migration_code-move_within_reg                                                     ?   
live_in_this_house_1_year_ago                       Not in universe under 1 year old   
migration_prev_res_in_sunbelt                                                      ?   
num_persons_worked_for_employer                                                    0   
family_members_under_18                                              Not in universe   
country_of_birth_father                                                      Vietnam   
country_of_birth_mother                                                      Vietnam   
country_of_birth_self                                                        Vietnam   
citizenship                                      Foreign born- Not a citizen of U S    
own_business_or_self_employed                                                      0   
fill_inc_questionnaire_for_veteran's_admin                           Not in universe   
veterans_benefits                                                                  2   
weeks_worked_in_year                                                               0   
year                                                                              95   
income_level                                                                       0   
```
</div>
    
<div class="k-default-codeblock">
```
                                                                                 3  \
age                                                                              9   
class_of_worker                                                    Not in universe   
detailed_industry_recode                                                         0   
detailed_occupation_recode                                                       0   
education                                                                 Children   
wage_per_hour                                                                    0   
enroll_in_edu_inst_last_wk                                         Not in universe   
marital_stat                                                         Never married   
major_industry_code                                    Not in universe or children   
major_occupation_code                                              Not in universe   
race                                                                         White   
hispanic_origin                                                          All other   
sex                                                                         Female   
member_of_a_labor_union                                            Not in universe   
reason_for_unemployment                                            Not in universe   
full_or_part_time_employment_stat                         Children or Armed Forces   
capital_gains                                                                    0   
capital_losses                                                                   0   
dividends_from_stocks                                                            0   
tax_filer_stat                                                            Nonfiler   
region_of_previous_residence                                       Not in universe   
state_of_previous_residence                                        Not in universe   
detailed_household_and_family_stat           Child <18 never marr not in subfamily   
detailed_household_summary_in_household               Child under 18 never married   
instance_weight                                                            1758.14   
migration_code-change_in_msa                                              Nonmover   
migration_code-change_in_reg                                              Nonmover   
migration_code-move_within_reg                                            Nonmover   
live_in_this_house_1_year_ago                                                  Yes   
migration_prev_res_in_sunbelt                                      Not in universe   
num_persons_worked_for_employer                                                  0   
family_members_under_18                                       Both parents present   
country_of_birth_father                                              United-States   
country_of_birth_mother                                              United-States   
country_of_birth_self                                                United-States   
citizenship                                      Native- Born in the United States   
own_business_or_self_employed                                                    0   
fill_inc_questionnaire_for_veteran's_admin                         Not in universe   
veterans_benefits                                                                0   
weeks_worked_in_year                                                             0   
year                                                                            94   
income_level                                                                     0   
```
</div>
    
<div class="k-default-codeblock">
```
                                                                                 4  
age                                                                             10  
class_of_worker                                                    Not in universe  
detailed_industry_recode                                                         0  
detailed_occupation_recode                                                       0  
education                                                                 Children  
wage_per_hour                                                                    0  
enroll_in_edu_inst_last_wk                                         Not in universe  
marital_stat                                                         Never married  
major_industry_code                                    Not in universe or children  
major_occupation_code                                              Not in universe  
race                                                                         White  
hispanic_origin                                                          All other  
sex                                                                         Female  
member_of_a_labor_union                                            Not in universe  
reason_for_unemployment                                            Not in universe  
full_or_part_time_employment_stat                         Children or Armed Forces  
capital_gains                                                                    0  
capital_losses                                                                   0  
dividends_from_stocks                                                            0  
tax_filer_stat                                                            Nonfiler  
region_of_previous_residence                                       Not in universe  
state_of_previous_residence                                        Not in universe  
detailed_household_and_family_stat           Child <18 never marr not in subfamily  
detailed_household_summary_in_household               Child under 18 never married  
instance_weight                                                            1069.16  
migration_code-change_in_msa                                              Nonmover  
migration_code-change_in_reg                                              Nonmover  
migration_code-move_within_reg                                            Nonmover  
live_in_this_house_1_year_ago                                                  Yes  
migration_prev_res_in_sunbelt                                      Not in universe  
num_persons_worked_for_employer                                                  0  
family_members_under_18                                       Both parents present  
country_of_birth_father                                              United-States  
country_of_birth_mother                                              United-States  
country_of_birth_self                                                United-States  
citizenship                                      Native- Born in the United States  
own_business_or_self_employed                                                    0  
fill_inc_questionnaire_for_veteran's_admin                         Not in universe  
veterans_benefits                                                                0  
weeks_worked_in_year                                                             0  
year                                                                            94  
income_level                                                                     0  

```
</div>
---
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for encoding
the input features with respect to their types.


```python
# Target column name.
TARGET_COLUMN_NAME = "income_level"
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
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    feature_name: sorted(
        [str(value) for value in list(train_data[feature_name].unique())]
    )
    for feature_name in CSV_HEADER
    if feature_name
    not in list(NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME, TARGET_COLUMN_NAME])
}
# All features names.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(
    CATEGORICAL_FEATURES_WITH_VOCABULARY.keys()
)
```

---
## Configure hyperparameters

You can find all the parameters of the Gradient Boosted Tree model in the
[documentation](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel)


```python
GROWING_STRATEGY = "BEST_FIRST_GLOBAL"
NUM_TREES = 250
MIN_EXAMPLES = 6
MAX_DEPTH = 5
SUBSAMPLE = 0.65
SAMPLING_METHOD = "RANDOM"
VALIDATION_RATIO = 0.1
```

---
## Implement a training and evaluation procedure


```python

def prepare_sample(features, target, weight):
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            if features[feature_name].dtype != tf.dtypes.string:
                # Convert categorical feature values to string.
                features[feature_name] = tf.strings.as_string(features[feature_name])
    return features, target, weight


def run_experiment(model, train_data, test_data, num_epochs=1, batch_size=None):

    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        train_data, label=TARGET_COLUMN_NAME, weight=WEIGHT_COLUMN_NAME
    ).map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data, label=TARGET_COLUMN_NAME, weight=WEIGHT_COLUMN_NAME
    ).map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)

    model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

```

---
## Create model inputs


```python

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs

```

---
## Experiment 1: Decision Forests with raw features

### Specify model input feature usages

You can attach semantics to each feature to control how it is used by the model.
If not specified, the semantics are inferred from the representation type.
It is recommended to specify the [feature usages](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/FeatureUsage)
explicitly to avoid incorrect inferred semantics is incorrect.
For example, a categorical value identifier (integer) will be be inferred as numerical,
while it is semantically categorical.

For numerical features, you can set the `discretized` parameters to the number
of buckets by which the numerical feature should be discretized.
This makes the training faster but may lead to worse models.


```python

def specify_feature_usages(inputs):
    feature_usages = []

    for feature_name in inputs:
        if inputs[feature_name].dtype == tf.dtypes.float32:
            feature_usage = tfdf.keras.FeatureUsage(
                name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL
            )
        else:
            feature_usage = tfdf.keras.FeatureUsage(
                name=feature_name, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL
            )

        feature_usages.append(feature_usage)
    return feature_usages

```

### Create a Gradient Boosted Trees model

When compiling a decision forests model, you may only provide extra evaluation metrics.
The loss is specified in the model construction,
and the optimizer is irrelevant to decision forests models.


```python

def create_gbt_model():
    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        features=specify_feature_usages(create_model_inputs()),
        exclude_non_specified_features=True,
        growing_strategy=GROWING_STRATEGY,
        num_trees=NUM_TREES,
        max_depth=MAX_DEPTH,
        min_examples=MIN_EXAMPLES,
        subsample=SUBSAMPLE,
        validation_ratio=VALIDATION_RATIO,
        task=tfdf.keras.Task.CLASSIFICATION,
        loss="DEFAULT",
    )

    gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    return gbt_model

```

### Train and evaluate the model

Note that when training a Decision Forests model, only one epoch is needed to
read the full dataset. Any extra steps will result in unnecessary slower training.
Therefore, the default `num_epochs=1` is used in the `run_experiment` method.


```python
gbt_model = create_gbt_model()
run_experiment(gbt_model, train_data, test_data)
```

<div class="k-default-codeblock">
```
Starting reading the dataset
198/200 [============================>.] - ETA: 0s
Dataset read in 0:00:10.359662
Training model
Model trained in 0:01:11.608668
Compiling model
200/200 [==============================] - 82s 386ms/step
Test accuracy: 95.79%

```
</div>
### Inspect the model

The `model.summary()` method will display several types of information about
your decision trees model, model type, task, input features, and feature importance.


```python
print(gbt_model.summary())
```

<div class="k-default-codeblock">
```
Model: "gradient_boosted_trees_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
=================================================================
Total params: 1
Trainable params: 0
Non-trainable params: 1
_________________________________________________________________
Type: "GRADIENT_BOOSTED_TREES"
Task: CLASSIFICATION
Label: "__LABEL"
```
</div>
    
<div class="k-default-codeblock">
```
Input Features (40):
	age
	capital_gains
	capital_losses
	citizenship
	class_of_worker
	country_of_birth_father
	country_of_birth_mother
	country_of_birth_self
	detailed_household_and_family_stat
	detailed_household_summary_in_household
	detailed_industry_recode
	detailed_occupation_recode
	dividends_from_stocks
	education
	enroll_in_edu_inst_last_wk
	family_members_under_18
	fill_inc_questionnaire_for_veteran's_admin
	full_or_part_time_employment_stat
	hispanic_origin
	live_in_this_house_1_year_ago
	major_industry_code
	major_occupation_code
	marital_stat
	member_of_a_labor_union
	migration_code-change_in_msa
	migration_code-change_in_reg
	migration_code-move_within_reg
	migration_prev_res_in_sunbelt
	num_persons_worked_for_employer
	own_business_or_self_employed
	race
	reason_for_unemployment
	region_of_previous_residence
	sex
	state_of_previous_residence
	tax_filer_stat
	veterans_benefits
	wage_per_hour
	weeks_worked_in_year
	year
```
</div>
    
<div class="k-default-codeblock">
```
Trained with weights
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: MEAN_MIN_DEPTH:
    1.                    "family_members_under_18"  4.866616 ################
    2.              "live_in_this_house_1_year_ago"  4.866616 ################
    3.               "region_of_previous_residence"  4.866616 ################
    4.                                    "__LABEL"  4.866616 ################
    5.                                  "__WEIGHTS"  4.866616 ################
    6.                                       "year"  4.865916 ###############
    7.                 "enroll_in_edu_inst_last_wk"  4.865883 ###############
    8.               "migration_code-change_in_msa"  4.865727 ###############
    9.              "migration_prev_res_in_sunbelt"  4.865583 ###############
   10.    "detailed_household_summary_in_household"  4.865518 ###############
   11.                          "veterans_benefits"  4.865518 ###############
   12.                                "citizenship"  4.865033 ###############
   13.               "migration_code-change_in_reg"  4.864541 ###############
   14.             "migration_code-move_within_reg"  4.860145 ###############
   15.                      "major_occupation_code"  4.860127 ###############
   16.                        "major_industry_code"  4.847848 ###############
   17.                    "reason_for_unemployment"  4.840486 ###############
   18.                    "member_of_a_labor_union"  4.813595 ###############
   19.                            "hispanic_origin"  4.745123 ###############
   20.                                       "race"  4.744956 ###############
   21.            "num_persons_worked_for_employer"  4.715408 ##############
   22.                               "marital_stat"  4.702930 ##############
   23.              "own_business_or_self_employed"  4.699103 ##############
   24.          "full_or_part_time_employment_stat"  4.692144 ##############
   25.                             "tax_filer_stat"  4.689370 ##############
   26. "fill_inc_questionnaire_for_veteran's_admin"  4.623981 ##############
   27.                    "country_of_birth_father"  4.582914 ##############
   28.                    "country_of_birth_mother"  4.580564 ##############
   29.                            "class_of_worker"  4.568425 #############
   30.                       "weeks_worked_in_year"  4.565960 #############
   31.                                        "sex"  4.513133 #############
   32.                      "country_of_birth_self"  4.458652 #############
   33.                "state_of_previous_residence"  4.458643 #############
   34.                                        "age"  4.447823 #############
   35.         "detailed_household_and_family_stat"  4.386165 ############
   36.                              "wage_per_hour"  4.271582 ###########
   37.                             "capital_losses"  4.224308 ###########
   38.                      "dividends_from_stocks"  4.109909 ##########
   39.                                  "education"  3.874190 #########
   40.                              "capital_gains"  3.833155 ########
   41.                   "detailed_industry_recode"  3.275964 #####
   42.                 "detailed_occupation_recode"  2.505073 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_AS_ROOT:
    1.                              "capital_gains" 38.000000 ################
    2.                                  "education" 23.000000 #########
    3.                              "wage_per_hour" 23.000000 #########
    4.                             "capital_losses" 21.000000 ########
    5.                      "dividends_from_stocks" 17.000000 ######
    6.                 "detailed_occupation_recode" 14.000000 #####
    7.         "detailed_household_and_family_stat" 13.000000 ####
    8. "fill_inc_questionnaire_for_veteran's_admin" 11.000000 ###
    9.                "state_of_previous_residence" 11.000000 ###
   10.                      "country_of_birth_self" 10.000000 ###
   11.                                        "age"  8.000000 ##
   12.                            "class_of_worker"  6.000000 #
   13.          "full_or_part_time_employment_stat"  6.000000 #
   14.                                        "sex"  6.000000 #
   15.                       "weeks_worked_in_year"  6.000000 #
   16.                                       "race"  5.000000 
   17.                    "country_of_birth_mother"  4.000000 
   18.                            "hispanic_origin"  4.000000 
   19.                    "country_of_birth_father"  3.000000 
   20.                               "marital_stat"  3.000000 
   21.              "own_business_or_self_employed"  3.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_NODES:
    1.                 "detailed_occupation_recode" 1373.000000 ################
    2.                   "detailed_industry_recode" 1263.000000 ##############
    3.                                  "education" 417.000000 ####
    4.                      "dividends_from_stocks" 381.000000 ####
    5.                              "capital_gains" 337.000000 ###
    6.                             "capital_losses" 276.000000 ###
    7.                                        "age" 269.000000 ###
    8.                    "country_of_birth_father" 255.000000 ##
    9.                    "country_of_birth_mother" 235.000000 ##
   10.                "state_of_previous_residence" 187.000000 ##
   11.                      "country_of_birth_self" 155.000000 #
   12.                              "wage_per_hour" 148.000000 #
   13.         "detailed_household_and_family_stat" 135.000000 #
   14.                            "class_of_worker" 128.000000 #
   15.            "num_persons_worked_for_employer" 112.000000 #
   16.                             "tax_filer_stat" 104.000000 #
   17.                       "weeks_worked_in_year" 99.000000 #
   18.                                        "sex" 89.000000 #
   19.                               "marital_stat" 61.000000 
   20.          "full_or_part_time_employment_stat" 44.000000 
   21.              "own_business_or_self_employed" 44.000000 
   22.                        "major_industry_code" 39.000000 
   23.                    "member_of_a_labor_union" 29.000000 
   24.                            "hispanic_origin" 26.000000 
   25. "fill_inc_questionnaire_for_veteran's_admin" 23.000000 
   26.                                       "race" 20.000000 
   27.                      "major_occupation_code" 15.000000 
   28.             "migration_code-move_within_reg"  9.000000 
   29.               "migration_code-change_in_reg"  7.000000 
   30.                                "citizenship"  4.000000 
   31.    "detailed_household_summary_in_household"  4.000000 
   32.                    "reason_for_unemployment"  4.000000 
   33.               "migration_code-change_in_msa"  3.000000 
   34.              "migration_prev_res_in_sunbelt"  3.000000 
   35.                                       "year"  2.000000 
   36.                 "enroll_in_edu_inst_last_wk"  1.000000 
   37.                          "veterans_benefits"  1.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: SUM_SCORE:
    1.                 "detailed_occupation_recode" 16396598.123479 ################
    2.                              "capital_gains" 5360688.465133 #####
    3.                                  "education" 5111382.550547 ####
    4.                   "detailed_industry_recode" 3751681.298295 ###
    5.                      "dividends_from_stocks" 3556889.615833 ###
    6.                                        "age" 2431834.947651 ##
    7.                                        "sex" 2275814.552990 ##
    8.                             "capital_losses" 1739087.548472 #
    9.                       "weeks_worked_in_year" 1257814.870561 #
   10.            "num_persons_worked_for_employer" 817246.190043 
   11.                             "tax_filer_stat" 739524.449400 
   12.         "detailed_household_and_family_stat" 496338.413802 
   13.                    "country_of_birth_mother" 446776.448403 
   14.                    "country_of_birth_father" 431989.557821 
   15.                            "class_of_worker" 399011.337558 
   16.                "state_of_previous_residence" 359157.377648 
   17.                              "wage_per_hour" 257399.821752 
   18.                      "country_of_birth_self" 241391.751734 
   19.                               "marital_stat" 182216.006652 
   20.              "own_business_or_self_employed" 96722.488980 
   21.          "full_or_part_time_employment_stat" 62637.493922 
   22.                      "major_occupation_code" 38796.587402 
   23.                        "major_industry_code" 37891.746821 
   24.                    "member_of_a_labor_union" 29161.354467 
   25. "fill_inc_questionnaire_for_veteran's_admin" 27952.466597 
   26.                            "hispanic_origin" 23902.628941 
   27.                                       "race" 20478.807236 
   28.             "migration_code-move_within_reg" 7910.678611 
   29.    "detailed_household_summary_in_household" 7870.980130 
   30.               "migration_code-change_in_reg" 7416.953436 
   31.                                "citizenship" 3782.359596 
   32.               "migration_code-change_in_msa" 3187.757106 
   33.                    "reason_for_unemployment" 3031.521377 
   34.                 "enroll_in_edu_inst_last_wk" 2422.099254 
   35.              "migration_prev_res_in_sunbelt" 2212.001457 
   36.                          "veterans_benefits" 840.453664 
   37.                                       "year" 696.065219 
```
</div>
    
    
    
<div class="k-default-codeblock">
```
Loss: BINOMIAL_LOG_LIKELIHOOD
Validation loss value: 0.227394
Number of trees per iteration: 1
Node format: NOT_SET
Number of trees: 235
Total number of nodes: 12839
```
</div>
    
<div class="k-default-codeblock">
```
Number of nodes by tree:
Count: 235 Average: 54.634 StdDev: 8.02457
Min: 27 Max: 61 Ignored: 0
----------------------------------------------
[ 27, 28)  1   0.43%   0.43%
[ 28, 30)  3   1.28%   1.70%
[ 30, 32)  1   0.43%   2.13%
[ 32, 34)  1   0.43%   2.55%
[ 34, 35)  0   0.00%   2.55%
[ 35, 37)  5   2.13%   4.68% #
[ 37, 39)  1   0.43%   5.11%
[ 39, 41)  8   3.40%   8.51% #
[ 41, 42)  5   2.13%  10.64% #
[ 42, 44)  2   0.85%  11.49%
[ 44, 46)  5   2.13%  13.62% #
[ 46, 48) 10   4.26%  17.87% #
[ 48, 49)  0   0.00%  17.87%
[ 49, 51) 11   4.68%  22.55% #
[ 51, 53) 13   5.53%  28.09% #
[ 53, 55) 17   7.23%  35.32% ##
[ 55, 56) 19   8.09%  43.40% ##
[ 56, 58) 19   8.09%  51.49% ##
[ 58, 60) 17   7.23%  58.72% ##
[ 60, 61] 97  41.28% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Depth by leafs:
Count: 6537 Average: 4.88542 StdDev: 0.405557
Min: 2 Max: 5 Ignored: 0
----------------------------------------------
[ 2, 3)   28   0.43%   0.43%
[ 3, 4)  122   1.87%   2.29%
[ 4, 5)  421   6.44%   8.73% #
[ 5, 5] 5966  91.27% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Number of training obs by leaf:
Count: 6537 Average: 0 StdDev: 0
Min: 0 Max: 0 Ignored: 0
----------------------------------------------
[ 0, 0] 6537 100.00% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes:
	1373 : detailed_occupation_recode [CATEGORICAL]
	1263 : detailed_industry_recode [CATEGORICAL]
	417 : education [CATEGORICAL]
	381 : dividends_from_stocks [NUMERICAL]
	337 : capital_gains [NUMERICAL]
	276 : capital_losses [NUMERICAL]
	269 : age [NUMERICAL]
	255 : country_of_birth_father [CATEGORICAL]
	235 : country_of_birth_mother [CATEGORICAL]
	187 : state_of_previous_residence [CATEGORICAL]
	155 : country_of_birth_self [CATEGORICAL]
	148 : wage_per_hour [NUMERICAL]
	135 : detailed_household_and_family_stat [CATEGORICAL]
	128 : class_of_worker [CATEGORICAL]
	112 : num_persons_worked_for_employer [NUMERICAL]
	104 : tax_filer_stat [CATEGORICAL]
	99 : weeks_worked_in_year [NUMERICAL]
	89 : sex [CATEGORICAL]
	61 : marital_stat [CATEGORICAL]
	44 : own_business_or_self_employed [CATEGORICAL]
	44 : full_or_part_time_employment_stat [CATEGORICAL]
	39 : major_industry_code [CATEGORICAL]
	29 : member_of_a_labor_union [CATEGORICAL]
	26 : hispanic_origin [CATEGORICAL]
	23 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	20 : race [CATEGORICAL]
	15 : major_occupation_code [CATEGORICAL]
	9 : migration_code-move_within_reg [CATEGORICAL]
	7 : migration_code-change_in_reg [CATEGORICAL]
	4 : reason_for_unemployment [CATEGORICAL]
	4 : detailed_household_summary_in_household [CATEGORICAL]
	4 : citizenship [CATEGORICAL]
	3 : migration_prev_res_in_sunbelt [CATEGORICAL]
	3 : migration_code-change_in_msa [CATEGORICAL]
	2 : year [CATEGORICAL]
	1 : veterans_benefits [CATEGORICAL]
	1 : enroll_in_edu_inst_last_wk [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 0:
	38 : capital_gains [NUMERICAL]
	23 : wage_per_hour [NUMERICAL]
	23 : education [CATEGORICAL]
	21 : capital_losses [NUMERICAL]
	17 : dividends_from_stocks [NUMERICAL]
	14 : detailed_occupation_recode [CATEGORICAL]
	13 : detailed_household_and_family_stat [CATEGORICAL]
	11 : state_of_previous_residence [CATEGORICAL]
	11 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	10 : country_of_birth_self [CATEGORICAL]
	8 : age [NUMERICAL]
	6 : weeks_worked_in_year [NUMERICAL]
	6 : sex [CATEGORICAL]
	6 : full_or_part_time_employment_stat [CATEGORICAL]
	6 : class_of_worker [CATEGORICAL]
	5 : race [CATEGORICAL]
	4 : hispanic_origin [CATEGORICAL]
	4 : country_of_birth_mother [CATEGORICAL]
	3 : own_business_or_self_employed [CATEGORICAL]
	3 : marital_stat [CATEGORICAL]
	3 : country_of_birth_father [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 1:
	139 : detailed_occupation_recode [CATEGORICAL]
	88 : capital_gains [NUMERICAL]
	59 : capital_losses [NUMERICAL]
	57 : detailed_industry_recode [CATEGORICAL]
	53 : education [CATEGORICAL]
	46 : wage_per_hour [NUMERICAL]
	46 : dividends_from_stocks [NUMERICAL]
	26 : detailed_household_and_family_stat [CATEGORICAL]
	20 : country_of_birth_self [CATEGORICAL]
	19 : state_of_previous_residence [CATEGORICAL]
	19 : sex [CATEGORICAL]
	17 : weeks_worked_in_year [NUMERICAL]
	17 : age [NUMERICAL]
	14 : class_of_worker [CATEGORICAL]
	11 : own_business_or_self_employed [CATEGORICAL]
	11 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	9 : marital_stat [CATEGORICAL]
	9 : full_or_part_time_employment_stat [CATEGORICAL]
	8 : tax_filer_stat [CATEGORICAL]
	8 : country_of_birth_father [CATEGORICAL]
	7 : country_of_birth_mother [CATEGORICAL]
	6 : race [CATEGORICAL]
	6 : num_persons_worked_for_employer [NUMERICAL]
	6 : hispanic_origin [CATEGORICAL]
	2 : reason_for_unemployment [CATEGORICAL]
	2 : member_of_a_labor_union [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 2:
	376 : detailed_occupation_recode [CATEGORICAL]
	254 : detailed_industry_recode [CATEGORICAL]
	160 : capital_gains [NUMERICAL]
	110 : capital_losses [NUMERICAL]
	97 : dividends_from_stocks [NUMERICAL]
	88 : education [CATEGORICAL]
	72 : wage_per_hour [NUMERICAL]
	42 : sex [CATEGORICAL]
	42 : detailed_household_and_family_stat [CATEGORICAL]
	38 : state_of_previous_residence [CATEGORICAL]
	37 : age [NUMERICAL]
	36 : country_of_birth_self [CATEGORICAL]
	32 : country_of_birth_father [CATEGORICAL]
	29 : weeks_worked_in_year [NUMERICAL]
	29 : tax_filer_stat [CATEGORICAL]
	29 : class_of_worker [CATEGORICAL]
	28 : country_of_birth_mother [CATEGORICAL]
	21 : marital_stat [CATEGORICAL]
	20 : num_persons_worked_for_employer [NUMERICAL]
	17 : own_business_or_self_employed [CATEGORICAL]
	14 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	13 : full_or_part_time_employment_stat [CATEGORICAL]
	12 : hispanic_origin [CATEGORICAL]
	8 : member_of_a_labor_union [CATEGORICAL]
	6 : race [CATEGORICAL]
	3 : major_industry_code [CATEGORICAL]
	2 : reason_for_unemployment [CATEGORICAL]
	1 : migration_prev_res_in_sunbelt [CATEGORICAL]
	1 : major_occupation_code [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 3:
	768 : detailed_occupation_recode [CATEGORICAL]
	656 : detailed_industry_recode [CATEGORICAL]
	231 : capital_gains [NUMERICAL]
	198 : education [CATEGORICAL]
	197 : dividends_from_stocks [NUMERICAL]
	177 : capital_losses [NUMERICAL]
	104 : country_of_birth_mother [CATEGORICAL]
	101 : wage_per_hour [NUMERICAL]
	101 : age [NUMERICAL]
	95 : country_of_birth_father [CATEGORICAL]
	88 : state_of_previous_residence [CATEGORICAL]
	82 : country_of_birth_self [CATEGORICAL]
	70 : detailed_household_and_family_stat [CATEGORICAL]
	68 : class_of_worker [CATEGORICAL]
	63 : sex [CATEGORICAL]
	57 : weeks_worked_in_year [NUMERICAL]
	53 : tax_filer_stat [CATEGORICAL]
	48 : num_persons_worked_for_employer [NUMERICAL]
	34 : marital_stat [CATEGORICAL]
	27 : own_business_or_self_employed [CATEGORICAL]
	16 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	15 : member_of_a_labor_union [CATEGORICAL]
	15 : major_industry_code [CATEGORICAL]
	15 : hispanic_origin [CATEGORICAL]
	15 : full_or_part_time_employment_stat [CATEGORICAL]
	10 : race [CATEGORICAL]
	5 : migration_code-move_within_reg [CATEGORICAL]
	2 : reason_for_unemployment [CATEGORICAL]
	2 : major_occupation_code [CATEGORICAL]
	1 : year [CATEGORICAL]
	1 : veterans_benefits [CATEGORICAL]
	1 : migration_prev_res_in_sunbelt [CATEGORICAL]
	1 : migration_code-change_in_reg [CATEGORICAL]
	1 : enroll_in_edu_inst_last_wk [CATEGORICAL]
	1 : citizenship [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 5:
	1373 : detailed_occupation_recode [CATEGORICAL]
	1263 : detailed_industry_recode [CATEGORICAL]
	417 : education [CATEGORICAL]
	381 : dividends_from_stocks [NUMERICAL]
	337 : capital_gains [NUMERICAL]
	276 : capital_losses [NUMERICAL]
	269 : age [NUMERICAL]
	255 : country_of_birth_father [CATEGORICAL]
	235 : country_of_birth_mother [CATEGORICAL]
	187 : state_of_previous_residence [CATEGORICAL]
	155 : country_of_birth_self [CATEGORICAL]
	148 : wage_per_hour [NUMERICAL]
	135 : detailed_household_and_family_stat [CATEGORICAL]
	128 : class_of_worker [CATEGORICAL]
	112 : num_persons_worked_for_employer [NUMERICAL]
	104 : tax_filer_stat [CATEGORICAL]
	99 : weeks_worked_in_year [NUMERICAL]
	89 : sex [CATEGORICAL]
	61 : marital_stat [CATEGORICAL]
	44 : own_business_or_self_employed [CATEGORICAL]
	44 : full_or_part_time_employment_stat [CATEGORICAL]
	39 : major_industry_code [CATEGORICAL]
	29 : member_of_a_labor_union [CATEGORICAL]
	26 : hispanic_origin [CATEGORICAL]
	23 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	20 : race [CATEGORICAL]
	15 : major_occupation_code [CATEGORICAL]
	9 : migration_code-move_within_reg [CATEGORICAL]
	7 : migration_code-change_in_reg [CATEGORICAL]
	4 : reason_for_unemployment [CATEGORICAL]
	4 : detailed_household_summary_in_household [CATEGORICAL]
	4 : citizenship [CATEGORICAL]
	3 : migration_prev_res_in_sunbelt [CATEGORICAL]
	3 : migration_code-change_in_msa [CATEGORICAL]
	2 : year [CATEGORICAL]
	1 : veterans_benefits [CATEGORICAL]
	1 : enroll_in_edu_inst_last_wk [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Condition type in nodes:
	4638 : ContainsBitmapCondition
	1622 : HigherCondition
	42 : ContainsCondition
Condition type in nodes with depth <= 0:
	118 : ContainsBitmapCondition
	113 : HigherCondition
	4 : ContainsCondition
Condition type in nodes with depth <= 1:
	420 : ContainsBitmapCondition
	279 : HigherCondition
	6 : ContainsCondition
Condition type in nodes with depth <= 2:
	1077 : ContainsBitmapCondition
	525 : HigherCondition
	15 : ContainsCondition
Condition type in nodes with depth <= 3:
	2380 : ContainsBitmapCondition
	912 : HigherCondition
	27 : ContainsCondition
Condition type in nodes with depth <= 5:
	4638 : ContainsBitmapCondition
	1622 : HigherCondition
	42 : ContainsCondition
```
</div>
    
<div class="k-default-codeblock">
```
None

```
</div>
---
## Experiment 2: Decision Forests with target encoding

[Target encoding](https://dl.acm.org/doi/10.1145/507533.507538) is a common preprocessing
technique for categorical features that convert them into numerical features.
Using categorical features with high cardinality as-is may lead to overfitting.
Target encoding aims to replace each categorical feature value with one or more
numerical values that represent its co-occurrence with the target labels.

More precisely, given a categorical feature, the binary target encoder in this example
will produce three new numerical features:

1. `positive_frequency`: How many times each feature value occurred with a positive target label.
2. `negative_frequency`: How many times each feature value occurred with a negative target label.
3. `positive_probability`: The probability that the target label is positive,
given the feature value, which is computed as
`positive_frequency / (positive_frequency + negative_frequency)`.


Note that target encoding is effective with models that cannot automatically
learn dense representations to categorical features, such as decision forests
or kernel methods. If neural network models are used, its recommended to
encode categorical features as embeddings.

### Implement Binary Target Encoder

For simplicity, we assume that the inputs for the `adapt` and `call` methods
are in the expected data types and shapes, so no validation logic is added.


```python

class BinaryTargetEncoding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adapt(self, data):
        # data is expected to be an integer numpy array to a Tensor shape [num_exmples, 2].
        # This contains feature values for a given feature in the dataset, and target values.

        # Convert the data to a tensor.
        data = tf.convert_to_tensor(data)
        # Separate the feature values and target values
        feature_values = tf.cast(data[:, 0], tf.dtypes.int64)
        target_values = tf.cast(data[:, 1], tf.dtypes.bool)

        print("Target encoding: Computing unique feature values...")
        # Get feature vocabulary.
        unique_feature_values = tf.sort(tf.unique(feature_values).y)

        print(
            "Target encoding: Computing frequencies for feature values with positive targets..."
        )
        # Filter the data where the target label is positive.
        positive_indices = tf.where(condition=target_values)
        postive_feature_values = tf.gather_nd(
            params=feature_values, indices=positive_indices
        )
        # Compute how many times each feature value occurred with a positive target label.
        positive_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(postive_feature_values.shape[0], 1), dtype=tf.dtypes.int32
            ),
            segment_ids=postive_feature_values,
            num_segments=unique_feature_values.shape[0],
        )

        print(
            "Target encoding: Computing frequencies for feature values with negative targets..."
        )
        # Filter the data where the target label is negative.
        negative_indices = tf.where(condition=tf.math.logical_not(target_values))
        negative_feature_values = tf.gather_nd(
            params=feature_values, indices=negative_indices
        )
        # Compute how many times each feature value occurred with a negative target label.
        negative_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(negative_feature_values.shape[0], 1), dtype=tf.dtypes.int32
            ),
            segment_ids=negative_feature_values,
            num_segments=unique_feature_values.shape[0],
        )

        print("Target encoding: Storing target encoding statistics...")
        self.positive_frequency_lookup = tf.constant(positive_frequency)
        self.negative_frequency_lookup = tf.constant(negative_frequency)

    def reset_state(self):
        self.positive_frequency_lookup = None
        self.negative_frequency_lookup = None

    def call(self, inputs):
        # inputs is expected to be an integer numpy array to a Tensor shape [num_exmples, 1].
        # This includes the feature values for a given feature in the dataset.

        # Raise an error if the target encoding statistics are not computed.
        if (
            self.positive_frequency_lookup == None
            or self.negative_frequency_lookup == None
        ):
            raise ValueError(
                f"You need to call the adapt method to compute target encoding statistics."
            )

        # Convert the inputs to a tensor.
        inputs = tf.convert_to_tensor(inputs)
        # Cast the inputs int64 a tensor.
        inputs = tf.cast(inputs, tf.dtypes.int64)
        # Lookup positive frequencies for the input feature values.
        positive_fequency = tf.cast(
            tf.gather_nd(self.positive_frequency_lookup, inputs),
            dtype=tf.dtypes.float32,
        )
        # Lookup negative frequencies for the input feature values.
        negative_fequency = tf.cast(
            tf.gather_nd(self.negative_frequency_lookup, inputs),
            dtype=tf.dtypes.float32,
        )
        # Compute positive probability for the input feature values.
        positive_probability = positive_fequency / (
            positive_fequency + negative_fequency
        )
        # Concatenate and return the looked-up statistics.
        return tf.concat(
            [positive_fequency, negative_fequency, positive_probability], axis=1
        )

```

Let's test the binary target encoder


```python
data = tf.constant(
    [
        [0, 1],
        [2, 0],
        [0, 1],
        [1, 1],
        [1, 1],
        [2, 0],
        [1, 0],
        [0, 1],
        [2, 1],
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 1],
        [1, 1],
        [1, 1],
        [2, 0],
        [1, 0],
        [0, 1],
        [2, 0],
    ]
)

binary_target_encoder = BinaryTargetEncoding()
binary_target_encoder.adapt(data)
print(binary_target_encoder([[0], [1], [2]]))
```

<div class="k-default-codeblock">
```
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
tf.Tensor(
[[6.         0.         1.        ]
 [4.         3.         0.5714286 ]
 [1.         5.         0.16666667]], shape=(3, 3), dtype=float32)

```
</div>
### Implement a feature encoding with target encoding


```python

def create_target_encoder():
    inputs = create_model_inputs()
    target_values = train_data[[TARGET_COLUMN_NAME]].to_numpy()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_indices = lookup(inputs[feature_name])
            # Prepare the data to adapt the target encoding.
            print("### Adapting target encoding for:", feature_name)
            feature_values = train_data[[feature_name]].to_numpy().astype(str)
            feature_value_indices = lookup(feature_values)
            data = tf.concat([feature_value_indices, target_values], axis=1)
            feature_encoder = BinaryTargetEncoding()
            feature_encoder.adapt(data)
            # Convert the feature value indices to target encoding representations.
            encoded_feature = feature_encoder(tf.expand_dims(value_indices, -1))
        else:
            # Expand the dimensions of the numerical input feature and use it as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # Add the encoded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features = tf.concat(encoded_features, axis=1)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)

```

### Create a Gradient Boosted Trees model with a preprocessor

In this scenario, we use the target encoding as a preprocessor for the Gradient Boosted Tree model,
and let the model infer semantics of the input features.


```python

def create_gbt_with_preprocessor(preprocessor):

    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        preprocessing=preprocessor,
        growing_strategy=GROWING_STRATEGY,
        num_trees=NUM_TREES,
        max_depth=MAX_DEPTH,
        min_examples=MIN_EXAMPLES,
        subsample=SUBSAMPLE,
        validation_ratio=VALIDATION_RATIO,
        task=tfdf.keras.Task.CLASSIFICATION,
    )

    gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])

    return gbt_model

```

### Train and evaluate the model


```python
gbt_model = create_gbt_with_preprocessor(create_target_encoder())
run_experiment(gbt_model, train_data, test_data)
```

<div class="k-default-codeblock">
```
### Adapting target encoding for: class_of_worker
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: detailed_industry_recode
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: detailed_occupation_recode
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: education
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: enroll_in_edu_inst_last_wk
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: marital_stat
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: major_industry_code
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: major_occupation_code
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: race
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: hispanic_origin
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: sex
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: member_of_a_labor_union
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: reason_for_unemployment
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: full_or_part_time_employment_stat
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: tax_filer_stat
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: region_of_previous_residence
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: state_of_previous_residence
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: detailed_household_and_family_stat
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: detailed_household_summary_in_household
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: migration_code-change_in_msa
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: migration_code-change_in_reg
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: migration_code-move_within_reg
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: live_in_this_house_1_year_ago
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: migration_prev_res_in_sunbelt
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: family_members_under_18
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: country_of_birth_father
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: country_of_birth_mother
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: country_of_birth_self
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: citizenship
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: own_business_or_self_employed
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: fill_inc_questionnaire_for_veteran's_admin
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: veterans_benefits
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
### Adapting target encoding for: year
Target encoding: Computing unique feature values...
Target encoding: Computing frequencies for feature values with positive targets...
Target encoding: Computing frequencies for feature values with negative targets...
Target encoding: Storing target encoding statistics...
Use /tmp/tmpxf8sriy5 as temporary training directory
Starting reading the dataset
199/200 [============================>.] - ETA: 0s
Dataset read in 0:00:07.793913
Training model
Model trained in 0:05:34.905892
Compiling model
200/200 [==============================] - 343s 2s/step
Test accuracy: 95.81%

```
</div>
---
## Experiment 3: Decision Forests with trained embeddings

In this scenario, we build an encoder model that codes the categorical
features to embeddings, where the size of the embedding for a given categorical
feature is the square root to the size of its vocabulary.

We train these embeddings in a simple linear model through backpropagation.
After the embedding encoder is trained, we used it as a preprocessor to the
input features of a Gradient Boosted Tree model.

Note that the embeddings and a decision forest model cannot be trained
synergically in one phase, since decision forest models do not train with backpropagation.
Rather, embeddings has to be trained in an initial phase,
and then used as static inputs to the decision forest model.

### Implement feature encoding with embeddings


```python

def create_embedding_encoder():
    inputs = create_model_inputs()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_index = lookup(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            vocabulary_size = len(vocabulary)
            embedding_size = int(math.sqrt(vocabulary_size))
            feature_encoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = feature_encoder(value_index)
        else:
            # Expand the dimensions of the numerical input feature and use it as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # Add the encoded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features = layers.concatenate(encoded_features, axis=1)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)

```

### Build a linear model to train the embeddings


```python

def create_linear_model(encoder):
    inputs = create_model_inputs()
    embeddings = encoder(inputs)
    linear_output = layers.Dense(units=1, activation="sigmoid")(embeddings)

    linear_model = keras.Model(inputs=inputs, outputs=linear_output)
    linear_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy("accuracy")],
    )
    return linear_model


embedding_encoder = create_embedding_encoder()
run_experiment(
    create_linear_model(embedding_encoder),
    train_data,
    test_data,
    num_epochs=3,
    batch_size=256,
)
```

<div class="k-default-codeblock">
```
Epoch 1/3
200/200 [==============================] - 9s 26ms/step - loss: 67492.5078 - accuracy: 0.9233
Epoch 2/3
200/200 [==============================] - 5s 26ms/step - loss: 2280.0027 - accuracy: 0.9386
Epoch 3/3
200/200 [==============================] - 5s 26ms/step - loss: 251.2815 - accuracy: 0.9480
Test accuracy: 94.84%

```
</div>
### Train and evaluate a Gradient Boosted Tree model with embeddings


```python
gbt_model = create_gbt_with_preprocessor(embedding_encoder)
run_experiment(gbt_model, train_data, test_data)
```

<div class="k-default-codeblock">
```
Use /tmp/tmpjlc3nb8k as temporary training directory
Starting reading the dataset
198/200 [============================>.] - ETA: 0s
Dataset read in 0:00:06.955685
Training model
Model trained in 0:05:43.275620
Compiling model
200/200 [==============================] - 350s 2s/step
Test accuracy: 95.82%

```
</div>
---
## Concluding remarks

TensorFlow Decision Forests provide powerful models, especially with structured data.
In our experiments, the Gradient Boosted Tree model achieved 95.79% test accuracy.
When using the target encoding with categorical feature, the same model achieved 95.81% test accuracy.
When pretraining embeddings to be used as inputs to the Gradient Boosted Tree model,
we achieved 95.82% test accuracy.


Decision Forests can be used with Neural Networks, either by
1) using Neural Networks to learn useful representation of the input data,
and then using Decision Forests for the supervised learning task, or by
2) creating an ensemble of both Decision Forests and Neural Network models.

Note that TensorFlow Decision Forests does not (yet) support hardware accelerators.
All training and inference is done on the CPU.
Besides, Decision Forests require a finite dataset that fits in memory
for their training procedures. However, there are diminishing returns
for increasing the size of the dataset, and Decision Forests algorithms
arguably need fewer examples for convergence than large Neural Network models.
