# Classification with TensorFlow Decision Forests

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2022/01/25<br>
**Last modified:** 2022/01/25<br>
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
For a beginner's guide to TensorFlow Decision Forests,
please refer to this [tutorial](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab).


This example uses Gradient Boosted Trees model in binary classification of
structured data, and covers the following scenarios:

1. Build a decision forests model by specifying the input feature usage.
2. Implement a custom *Binary Target encoder* as a [Keras Preprocessing layer](https://keras.io/api/layers/preprocessing_layers/)
to encode the categorical features with respect to their target value co-occurrences,
and then use the encoded features to build a decision forests model.
3. Encode the categorical features as [embeddings](https://keras.io/api/layers/core_layers/embedding),
train these embeddings in a simple NN model, and then use the
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

---
## Define dataset metadata

Here, we define the metadata of the dataset that will be useful for encoding
the input features with respect to their types.


```python
# Target column name.
TARGET_COLUMN_NAME = "income_level"
# The labels of the target columns.
TARGET_LABELS = [" - 50000.", " 50000+."]
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
CATEGORICAL_FEATURE_NAMES = [
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
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
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "migration_code-change_in_msa",
    "migration_code-change_in_reg",
    "migration_code-move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veteran's_admin",
    "veterans_benefits",
    "year",
]

```

Now we perform basic data preparation.


```python

def prepare_dataframe(dataframe):
    # Convert the target labels from string to integer.
    dataframe[TARGET_COLUMN_NAME] = dataframe[TARGET_COLUMN_NAME].map(
        TARGET_LABELS.index
    )
    # Cast the categorical features to string.
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        dataframe[feature_name] = dataframe[feature_name].astype(str)


prepare_dataframe(train_data)
prepare_dataframe(test_data)
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
## Configure hyperparameters

You can find all the parameters of the Gradient Boosted Tree model in the
[documentation](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel)


```python
# Maximum number of decision trees. The effective number of trained trees can be smaller if early stopping is enabled.
NUM_TREES = 250
# Minimum number of examples in a node.
MIN_EXAMPLES = 6
# Maximum depth of the tree. max_depth=1 means that all trees will be roots.
MAX_DEPTH = 5
# Ratio of the dataset (sampling without replacement) used to train individual trees for the random sampling method.
SUBSAMPLE = 0.65
# Control the sampling of the datasets used to train individual trees.
SAMPLING_METHOD = "RANDOM"
# Ratio of the training dataset used to monitor the training. Require to be >0 if early stopping is enabled.
VALIDATION_RATIO = 0.1
```

---
## Implement a training and evaluation procedure

The `run_experiment()` method is responsible loading the train and test datasets,
training a given model, and evaluating the trained model.

Note that when training a Decision Forests model, only one epoch is needed to
read the full dataset. Any extra steps will result in unnecessary slower training.
Therefore, the default `num_epochs=1` is used in the `run_experiment()` method.


```python

def run_experiment(model, train_data, test_data, num_epochs=1, batch_size=None):

    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        train_data, label=TARGET_COLUMN_NAME, weight=WEIGHT_COLUMN_NAME
    )
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data, label=TARGET_COLUMN_NAME, weight=WEIGHT_COLUMN_NAME
    )

    model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

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

def specify_feature_usages():
    feature_usages = []

    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(
            name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL
        )
        feature_usages.append(feature_usage)

    for feature_name in CATEGORICAL_FEATURE_NAMES:
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
    # See all the model parameters in https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel
    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        features=specify_feature_usages(),
        exclude_non_specified_features=True,
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
gbt_model = create_gbt_model()
run_experiment(gbt_model, train_data, test_data)
```

<div class="k-default-codeblock">
```
Starting reading the dataset
200/200 [==============================] - ETA: 0s
Dataset read in 0:00:08.829036
Training model
Model trained in 0:00:48.639771
Compiling model
200/200 [==============================] - 58s 268ms/step
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
    1.                 "enroll_in_edu_inst_last_wk"  3.942647 ################
    2.                    "family_members_under_18"  3.942647 ################
    3.              "live_in_this_house_1_year_ago"  3.942647 ################
    4.               "migration_code-change_in_msa"  3.942647 ################
    5.             "migration_code-move_within_reg"  3.942647 ################
    6.                                       "year"  3.942647 ################
    7.                                    "__LABEL"  3.942647 ################
    8.                                  "__WEIGHTS"  3.942647 ################
    9.                                "citizenship"  3.942137 ###############
   10.    "detailed_household_summary_in_household"  3.942137 ###############
   11.               "region_of_previous_residence"  3.942137 ###############
   12.                          "veterans_benefits"  3.942137 ###############
   13.              "migration_prev_res_in_sunbelt"  3.940135 ###############
   14.               "migration_code-change_in_reg"  3.939926 ###############
   15.                      "major_occupation_code"  3.937681 ###############
   16.                        "major_industry_code"  3.933687 ###############
   17.                    "reason_for_unemployment"  3.926320 ###############
   18.                            "hispanic_origin"  3.900776 ###############
   19.                    "member_of_a_labor_union"  3.894843 ###############
   20.                                       "race"  3.878617 ###############
   21.            "num_persons_worked_for_employer"  3.818566 ##############
   22.                               "marital_stat"  3.795667 ##############
   23.          "full_or_part_time_employment_stat"  3.795431 ##############
   24.                    "country_of_birth_mother"  3.787967 ##############
   25.                             "tax_filer_stat"  3.784505 ##############
   26. "fill_inc_questionnaire_for_veteran's_admin"  3.783607 ##############
   27.              "own_business_or_self_employed"  3.776398 ##############
   28.                    "country_of_birth_father"  3.715252 #############
   29.                                        "sex"  3.708745 #############
   30.                            "class_of_worker"  3.688424 #############
   31.                       "weeks_worked_in_year"  3.665290 #############
   32.                "state_of_previous_residence"  3.657234 #############
   33.                      "country_of_birth_self"  3.654377 #############
   34.                                        "age"  3.634295 ############
   35.                              "wage_per_hour"  3.617817 ############
   36.         "detailed_household_and_family_stat"  3.594743 ############
   37.                             "capital_losses"  3.439298 ##########
   38.                      "dividends_from_stocks"  3.423652 ##########
   39.                              "capital_gains"  3.222753 ########
   40.                                  "education"  3.158698 ########
   41.                   "detailed_industry_recode"  2.981471 ######
   42.                 "detailed_occupation_recode"  2.364817 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_AS_ROOT:
    1.                                  "education" 33.000000 ################
    2.                              "capital_gains" 29.000000 ##############
    3.                             "capital_losses" 24.000000 ###########
    4.         "detailed_household_and_family_stat" 14.000000 ######
    5.                      "dividends_from_stocks" 14.000000 ######
    6.                              "wage_per_hour" 12.000000 #####
    7.                      "country_of_birth_self" 11.000000 #####
    8.                 "detailed_occupation_recode" 11.000000 #####
    9.                       "weeks_worked_in_year" 11.000000 #####
   10.                                        "age" 10.000000 ####
   11.                "state_of_previous_residence" 10.000000 ####
   12. "fill_inc_questionnaire_for_veteran's_admin"  9.000000 ####
   13.                            "class_of_worker"  8.000000 ###
   14.          "full_or_part_time_employment_stat"  8.000000 ###
   15.                               "marital_stat"  8.000000 ###
   16.              "own_business_or_self_employed"  8.000000 ###
   17.                                        "sex"  6.000000 ##
   18.                             "tax_filer_stat"  5.000000 ##
   19.                    "country_of_birth_father"  4.000000 #
   20.                                       "race"  3.000000 #
   21.                   "detailed_industry_recode"  2.000000 
   22.                            "hispanic_origin"  2.000000 
   23.                    "country_of_birth_mother"  1.000000 
   24.            "num_persons_worked_for_employer"  1.000000 
   25.                    "reason_for_unemployment"  1.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: NUM_NODES:
    1.                 "detailed_occupation_recode" 785.000000 ################
    2.                   "detailed_industry_recode" 668.000000 #############
    3.                              "capital_gains" 275.000000 #####
    4.                      "dividends_from_stocks" 220.000000 ####
    5.                             "capital_losses" 197.000000 ####
    6.                                  "education" 178.000000 ###
    7.                    "country_of_birth_mother" 128.000000 ##
    8.                    "country_of_birth_father" 116.000000 ##
    9.                                        "age" 114.000000 ##
   10.                              "wage_per_hour" 98.000000 #
   11.                "state_of_previous_residence" 95.000000 #
   12.         "detailed_household_and_family_stat" 78.000000 #
   13.                            "class_of_worker" 67.000000 #
   14.                      "country_of_birth_self" 65.000000 #
   15.                                        "sex" 65.000000 #
   16.                       "weeks_worked_in_year" 60.000000 #
   17.                             "tax_filer_stat" 57.000000 #
   18.            "num_persons_worked_for_employer" 54.000000 #
   19.              "own_business_or_self_employed" 30.000000 
   20.                               "marital_stat" 26.000000 
   21.                    "member_of_a_labor_union" 16.000000 
   22. "fill_inc_questionnaire_for_veteran's_admin" 15.000000 
   23.          "full_or_part_time_employment_stat" 15.000000 
   24.                        "major_industry_code" 15.000000 
   25.                            "hispanic_origin"  9.000000 
   26.                      "major_occupation_code"  7.000000 
   27.                                       "race"  7.000000 
   28.                                "citizenship"  1.000000 
   29.    "detailed_household_summary_in_household"  1.000000 
   30.               "migration_code-change_in_reg"  1.000000 
   31.              "migration_prev_res_in_sunbelt"  1.000000 
   32.                    "reason_for_unemployment"  1.000000 
   33.               "region_of_previous_residence"  1.000000 
   34.                          "veterans_benefits"  1.000000 
```
</div>
    
<div class="k-default-codeblock">
```
Variable Importance: SUM_SCORE:
    1.                 "detailed_occupation_recode" 15392441.075369 ################
    2.                              "capital_gains" 5277826.822514 #####
    3.                                  "education" 4751749.289550 ####
    4.                      "dividends_from_stocks" 3792002.951255 ###
    5.                   "detailed_industry_recode" 2882200.882109 ##
    6.                                        "sex" 2559417.877325 ##
    7.                                        "age" 2042990.944829 ##
    8.                             "capital_losses" 1735728.772551 #
    9.                       "weeks_worked_in_year" 1272820.203971 #
   10.                             "tax_filer_stat" 697890.160846 
   11.            "num_persons_worked_for_employer" 671351.905595 
   12.         "detailed_household_and_family_stat" 444620.829557 
   13.                            "class_of_worker" 362250.565331 
   14.                    "country_of_birth_mother" 296311.574426 
   15.                    "country_of_birth_father" 258198.889206 
   16.                              "wage_per_hour" 239764.219048 
   17.                "state_of_previous_residence" 237687.602572 
   18.                      "country_of_birth_self" 103002.168158 
   19.                               "marital_stat" 102449.735314 
   20.              "own_business_or_self_employed" 82938.893541 
   21. "fill_inc_questionnaire_for_veteran's_admin" 22692.700206 
   22.          "full_or_part_time_employment_stat" 19078.398837 
   23.                        "major_industry_code" 18450.345505 
   24.                    "member_of_a_labor_union" 14905.360879 
   25.                            "hispanic_origin" 12602.867902 
   26.                      "major_occupation_code" 8709.665989 
   27.                                       "race" 6116.282065 
   28.                                "citizenship" 3291.490393 
   29.    "detailed_household_summary_in_household" 2733.439375 
   30.                          "veterans_benefits" 1230.940488 
   31.               "region_of_previous_residence" 1139.240981 
   32.                    "reason_for_unemployment" 219.245124 
   33.               "migration_code-change_in_reg" 55.806436 
   34.              "migration_prev_res_in_sunbelt" 37.780635 
```
</div>
    
    
    
<div class="k-default-codeblock">
```
Loss: BINOMIAL_LOG_LIKELIHOOD
Validation loss value: 0.228983
Number of trees per iteration: 1
Node format: NOT_SET
Number of trees: 245
Total number of nodes: 7179
```
</div>
    
<div class="k-default-codeblock">
```
Number of nodes by tree:
Count: 245 Average: 29.302 StdDev: 2.96211
Min: 17 Max: 31 Ignored: 0
----------------------------------------------
[ 17, 18)   2   0.82%   0.82%
[ 18, 19)   0   0.00%   0.82%
[ 19, 20)   3   1.22%   2.04%
[ 20, 21)   0   0.00%   2.04%
[ 21, 22)   4   1.63%   3.67%
[ 22, 23)   0   0.00%   3.67%
[ 23, 24)  15   6.12%   9.80% #
[ 24, 25)   0   0.00%   9.80%
[ 25, 26)   5   2.04%  11.84%
[ 26, 27)   0   0.00%  11.84%
[ 27, 28)  21   8.57%  20.41% #
[ 28, 29)   0   0.00%  20.41%
[ 29, 30)  39  15.92%  36.33% ###
[ 30, 31)   0   0.00%  36.33%
[ 31, 31] 156  63.67% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Depth by leafs:
Count: 3712 Average: 3.95259 StdDev: 0.249814
Min: 2 Max: 4 Ignored: 0
----------------------------------------------
[ 2, 3)   32   0.86%   0.86%
[ 3, 4)  112   3.02%   3.88%
[ 4, 4] 3568  96.12% 100.00% ##########
```
</div>
    
<div class="k-default-codeblock">
```
Number of training obs by leaf:
Count: 3712 Average: 11849.3 StdDev: 33719.3
Min: 6 Max: 179360 Ignored: 0
----------------------------------------------
[      6,   8973) 3100  83.51%  83.51% ##########
[   8973,  17941)  148   3.99%  87.50%
[  17941,  26909)   79   2.13%  89.63%
[  26909,  35877)   36   0.97%  90.60%
[  35877,  44844)   44   1.19%  91.78%
[  44844,  53812)   17   0.46%  92.24%
[  53812,  62780)   20   0.54%  92.78%
[  62780,  71748)   39   1.05%  93.83%
[  71748,  80715)   24   0.65%  94.48%
[  80715,  89683)   12   0.32%  94.80%
[  89683,  98651)   22   0.59%  95.39%
[  98651, 107619)   21   0.57%  95.96%
[ 107619, 116586)   17   0.46%  96.42%
[ 116586, 125554)   17   0.46%  96.88%
[ 125554, 134522)   13   0.35%  97.23%
[ 134522, 143490)    8   0.22%  97.44%
[ 143490, 152457)    5   0.13%  97.58%
[ 152457, 161425)    6   0.16%  97.74%
[ 161425, 170393)   15   0.40%  98.14%
[ 170393, 179360]   69   1.86% 100.00%
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes:
	785 : detailed_occupation_recode [CATEGORICAL]
	668 : detailed_industry_recode [CATEGORICAL]
	275 : capital_gains [NUMERICAL]
	220 : dividends_from_stocks [NUMERICAL]
	197 : capital_losses [NUMERICAL]
	178 : education [CATEGORICAL]
	128 : country_of_birth_mother [CATEGORICAL]
	116 : country_of_birth_father [CATEGORICAL]
	114 : age [NUMERICAL]
	98 : wage_per_hour [NUMERICAL]
	95 : state_of_previous_residence [CATEGORICAL]
	78 : detailed_household_and_family_stat [CATEGORICAL]
	67 : class_of_worker [CATEGORICAL]
	65 : sex [CATEGORICAL]
	65 : country_of_birth_self [CATEGORICAL]
	60 : weeks_worked_in_year [NUMERICAL]
	57 : tax_filer_stat [CATEGORICAL]
	54 : num_persons_worked_for_employer [NUMERICAL]
	30 : own_business_or_self_employed [CATEGORICAL]
	26 : marital_stat [CATEGORICAL]
	16 : member_of_a_labor_union [CATEGORICAL]
	15 : major_industry_code [CATEGORICAL]
	15 : full_or_part_time_employment_stat [CATEGORICAL]
	15 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	9 : hispanic_origin [CATEGORICAL]
	7 : race [CATEGORICAL]
	7 : major_occupation_code [CATEGORICAL]
	1 : veterans_benefits [CATEGORICAL]
	1 : region_of_previous_residence [CATEGORICAL]
	1 : reason_for_unemployment [CATEGORICAL]
	1 : migration_prev_res_in_sunbelt [CATEGORICAL]
	1 : migration_code-change_in_reg [CATEGORICAL]
	1 : detailed_household_summary_in_household [CATEGORICAL]
	1 : citizenship [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 0:
	33 : education [CATEGORICAL]
	29 : capital_gains [NUMERICAL]
	24 : capital_losses [NUMERICAL]
	14 : dividends_from_stocks [NUMERICAL]
	14 : detailed_household_and_family_stat [CATEGORICAL]
	12 : wage_per_hour [NUMERICAL]
	11 : weeks_worked_in_year [NUMERICAL]
	11 : detailed_occupation_recode [CATEGORICAL]
	11 : country_of_birth_self [CATEGORICAL]
	10 : state_of_previous_residence [CATEGORICAL]
	10 : age [NUMERICAL]
	9 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	8 : own_business_or_self_employed [CATEGORICAL]
	8 : marital_stat [CATEGORICAL]
	8 : full_or_part_time_employment_stat [CATEGORICAL]
	8 : class_of_worker [CATEGORICAL]
	6 : sex [CATEGORICAL]
	5 : tax_filer_stat [CATEGORICAL]
	4 : country_of_birth_father [CATEGORICAL]
	3 : race [CATEGORICAL]
	2 : hispanic_origin [CATEGORICAL]
	2 : detailed_industry_recode [CATEGORICAL]
	1 : reason_for_unemployment [CATEGORICAL]
	1 : num_persons_worked_for_employer [NUMERICAL]
	1 : country_of_birth_mother [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 1:
	140 : detailed_occupation_recode [CATEGORICAL]
	82 : capital_gains [NUMERICAL]
	65 : capital_losses [NUMERICAL]
	62 : education [CATEGORICAL]
	59 : detailed_industry_recode [CATEGORICAL]
	47 : dividends_from_stocks [NUMERICAL]
	31 : wage_per_hour [NUMERICAL]
	26 : detailed_household_and_family_stat [CATEGORICAL]
	23 : age [NUMERICAL]
	22 : state_of_previous_residence [CATEGORICAL]
	21 : country_of_birth_self [CATEGORICAL]
	21 : class_of_worker [CATEGORICAL]
	20 : weeks_worked_in_year [NUMERICAL]
	20 : sex [CATEGORICAL]
	15 : country_of_birth_father [CATEGORICAL]
	12 : own_business_or_self_employed [CATEGORICAL]
	11 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	10 : num_persons_worked_for_employer [NUMERICAL]
	9 : tax_filer_stat [CATEGORICAL]
	9 : full_or_part_time_employment_stat [CATEGORICAL]
	8 : marital_stat [CATEGORICAL]
	8 : country_of_birth_mother [CATEGORICAL]
	6 : member_of_a_labor_union [CATEGORICAL]
	5 : race [CATEGORICAL]
	2 : hispanic_origin [CATEGORICAL]
	1 : reason_for_unemployment [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 2:
	399 : detailed_occupation_recode [CATEGORICAL]
	249 : detailed_industry_recode [CATEGORICAL]
	170 : capital_gains [NUMERICAL]
	117 : dividends_from_stocks [NUMERICAL]
	116 : capital_losses [NUMERICAL]
	87 : education [CATEGORICAL]
	59 : wage_per_hour [NUMERICAL]
	45 : detailed_household_and_family_stat [CATEGORICAL]
	43 : country_of_birth_father [CATEGORICAL]
	43 : age [NUMERICAL]
	40 : country_of_birth_self [CATEGORICAL]
	38 : state_of_previous_residence [CATEGORICAL]
	38 : class_of_worker [CATEGORICAL]
	37 : sex [CATEGORICAL]
	36 : weeks_worked_in_year [NUMERICAL]
	33 : country_of_birth_mother [CATEGORICAL]
	28 : num_persons_worked_for_employer [NUMERICAL]
	26 : tax_filer_stat [CATEGORICAL]
	14 : own_business_or_self_employed [CATEGORICAL]
	14 : marital_stat [CATEGORICAL]
	12 : full_or_part_time_employment_stat [CATEGORICAL]
	12 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	8 : member_of_a_labor_union [CATEGORICAL]
	6 : race [CATEGORICAL]
	6 : hispanic_origin [CATEGORICAL]
	2 : major_occupation_code [CATEGORICAL]
	2 : major_industry_code [CATEGORICAL]
	1 : reason_for_unemployment [CATEGORICAL]
	1 : migration_prev_res_in_sunbelt [CATEGORICAL]
	1 : migration_code-change_in_reg [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 3:
	785 : detailed_occupation_recode [CATEGORICAL]
	668 : detailed_industry_recode [CATEGORICAL]
	275 : capital_gains [NUMERICAL]
	220 : dividends_from_stocks [NUMERICAL]
	197 : capital_losses [NUMERICAL]
	178 : education [CATEGORICAL]
	128 : country_of_birth_mother [CATEGORICAL]
	116 : country_of_birth_father [CATEGORICAL]
	114 : age [NUMERICAL]
	98 : wage_per_hour [NUMERICAL]
	95 : state_of_previous_residence [CATEGORICAL]
	78 : detailed_household_and_family_stat [CATEGORICAL]
	67 : class_of_worker [CATEGORICAL]
	65 : sex [CATEGORICAL]
	65 : country_of_birth_self [CATEGORICAL]
	60 : weeks_worked_in_year [NUMERICAL]
	57 : tax_filer_stat [CATEGORICAL]
	54 : num_persons_worked_for_employer [NUMERICAL]
	30 : own_business_or_self_employed [CATEGORICAL]
	26 : marital_stat [CATEGORICAL]
	16 : member_of_a_labor_union [CATEGORICAL]
	15 : major_industry_code [CATEGORICAL]
	15 : full_or_part_time_employment_stat [CATEGORICAL]
	15 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	9 : hispanic_origin [CATEGORICAL]
	7 : race [CATEGORICAL]
	7 : major_occupation_code [CATEGORICAL]
	1 : veterans_benefits [CATEGORICAL]
	1 : region_of_previous_residence [CATEGORICAL]
	1 : reason_for_unemployment [CATEGORICAL]
	1 : migration_prev_res_in_sunbelt [CATEGORICAL]
	1 : migration_code-change_in_reg [CATEGORICAL]
	1 : detailed_household_summary_in_household [CATEGORICAL]
	1 : citizenship [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Attribute in nodes with depth <= 5:
	785 : detailed_occupation_recode [CATEGORICAL]
	668 : detailed_industry_recode [CATEGORICAL]
	275 : capital_gains [NUMERICAL]
	220 : dividends_from_stocks [NUMERICAL]
	197 : capital_losses [NUMERICAL]
	178 : education [CATEGORICAL]
	128 : country_of_birth_mother [CATEGORICAL]
	116 : country_of_birth_father [CATEGORICAL]
	114 : age [NUMERICAL]
	98 : wage_per_hour [NUMERICAL]
	95 : state_of_previous_residence [CATEGORICAL]
	78 : detailed_household_and_family_stat [CATEGORICAL]
	67 : class_of_worker [CATEGORICAL]
	65 : sex [CATEGORICAL]
	65 : country_of_birth_self [CATEGORICAL]
	60 : weeks_worked_in_year [NUMERICAL]
	57 : tax_filer_stat [CATEGORICAL]
	54 : num_persons_worked_for_employer [NUMERICAL]
	30 : own_business_or_self_employed [CATEGORICAL]
	26 : marital_stat [CATEGORICAL]
	16 : member_of_a_labor_union [CATEGORICAL]
	15 : major_industry_code [CATEGORICAL]
	15 : full_or_part_time_employment_stat [CATEGORICAL]
	15 : fill_inc_questionnaire_for_veteran's_admin [CATEGORICAL]
	9 : hispanic_origin [CATEGORICAL]
	7 : race [CATEGORICAL]
	7 : major_occupation_code [CATEGORICAL]
	1 : veterans_benefits [CATEGORICAL]
	1 : region_of_previous_residence [CATEGORICAL]
	1 : reason_for_unemployment [CATEGORICAL]
	1 : migration_prev_res_in_sunbelt [CATEGORICAL]
	1 : migration_code-change_in_reg [CATEGORICAL]
	1 : detailed_household_summary_in_household [CATEGORICAL]
	1 : citizenship [CATEGORICAL]
```
</div>
    
<div class="k-default-codeblock">
```
Condition type in nodes:
	2418 : ContainsBitmapCondition
	1018 : HigherCondition
	31 : ContainsCondition
Condition type in nodes with depth <= 0:
	137 : ContainsBitmapCondition
	101 : HigherCondition
	7 : ContainsCondition
Condition type in nodes with depth <= 1:
	448 : ContainsBitmapCondition
	278 : HigherCondition
	9 : ContainsCondition
Condition type in nodes with depth <= 2:
	1097 : ContainsBitmapCondition
	569 : HigherCondition
	17 : ContainsCondition
Condition type in nodes with depth <= 3:
	2418 : ContainsBitmapCondition
	1018 : HigherCondition
	31 : ContainsCondition
Condition type in nodes with depth <= 5:
	2418 : ContainsBitmapCondition
	1018 : HigherCondition
	31 : ContainsCondition
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
`positive_frequency / (positive_frequency + negative_frequency + correction)`.
The `correction` term is added in to make the division more stable for rare categorical values.
The default value for `correction` is 1.0.



Note that target encoding is effective with models that cannot automatically
learn dense representations to categorical features, such as decision forests
or kernel methods. If neural network models are used, its recommended to
encode categorical features as embeddings.

### Implement Binary Target Encoder

For simplicity, we assume that the inputs for the `adapt` and `call` methods
are in the expected data types and shapes, so no validation logic is added.

It is recommended to pass the `vocabulary_size` of the categorical feature to the
`BinaryTargetEncoding` constructor. If not specified, it will be computed during
the `adapt()` method execution.


```python

class BinaryTargetEncoding(layers.Layer):
    def __init__(self, vocabulary_size=None, correction=1.0, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.correction = correction

    def adapt(self, data):
        # data is expected to be an integer numpy array to a Tensor shape [num_exmples, 2].
        # This contains feature values for a given feature in the dataset, and target values.

        # Convert the data to a tensor.
        data = tf.convert_to_tensor(data)
        # Separate the feature values and target values
        feature_values = tf.cast(data[:, 0], tf.dtypes.int32)
        target_values = tf.cast(data[:, 1], tf.dtypes.bool)

        # Compute the vocabulary_size of not specified.
        if self.vocabulary_size is None:
            self.vocabulary_size = tf.unique(feature_values).y.shape[0]

        # Filter the data where the target label is positive.
        positive_indices = tf.where(condition=target_values)
        postive_feature_values = tf.gather_nd(
            params=feature_values, indices=positive_indices
        )
        # Compute how many times each feature value occurred with a positive target label.
        positive_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(postive_feature_values.shape[0], 1), dtype=tf.dtypes.float64
            ),
            segment_ids=postive_feature_values,
            num_segments=self.vocabulary_size,
        )

        # Filter the data where the target label is negative.
        negative_indices = tf.where(condition=tf.math.logical_not(target_values))
        negative_feature_values = tf.gather_nd(
            params=feature_values, indices=negative_indices
        )
        # Compute how many times each feature value occurred with a negative target label.
        negative_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(negative_feature_values.shape[0], 1), dtype=tf.dtypes.float64
            ),
            segment_ids=negative_feature_values,
            num_segments=self.vocabulary_size,
        )
        # Compute positive probability for the input feature values.
        positive_probability = positive_frequency / (
            positive_frequency + negative_frequency + self.correction
        )
        # Concatenate the computed statistics for traget_encoding.
        target_encoding_statistics = tf.cast(
            tf.concat(
                [positive_frequency, negative_frequency, positive_probability], axis=1
            ),
            dtype=tf.dtypes.float32,
        )
        self.target_encoding_statistics = tf.constant(target_encoding_statistics)

    def call(self, inputs):
        # inputs is expected to be an integer numpy array to a Tensor shape [num_exmples, 1].
        # This includes the feature values for a given feature in the dataset.

        # Raise an error if the target encoding statistics are not computed.
        if self.target_encoding_statistics == None:
            raise ValueError(
                f"You need to call the adapt method to compute target encoding statistics."
            )

        # Convert the inputs to a tensor.
        inputs = tf.convert_to_tensor(inputs)
        # Cast the inputs int64 a tensor.
        inputs = tf.cast(inputs, tf.dtypes.int64)
        # Lookup target encoding statistics for the input feature values.
        target_encoding_statistics = tf.cast(
            tf.gather_nd(self.target_encoding_statistics, inputs),
            dtype=tf.dtypes.float32,
        )
        return target_encoding_statistics

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
tf.Tensor(
[[6.         0.         0.85714287]
 [4.         3.         0.5       ]
 [1.         5.         0.14285715]], shape=(3, 3), dtype=float32)

```
</div>
### Create model inputs


```python

def create_model_inputs():
    inputs = {}

    for feature_name in NUMERIC_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.string
        )

    return inputs

```

### Implement a feature encoding with target encoding


```python

def create_target_encoder():
    inputs = create_model_inputs()
    target_values = train_data[[TARGET_COLUMN_NAME]].to_numpy()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Get the vocabulary of the categorical feature.
            vocabulary = sorted(
                [str(value) for value in list(train_data[feature_name].unique())]
            )
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
### Adapting target encoding for: detailed_industry_recode
### Adapting target encoding for: detailed_occupation_recode
### Adapting target encoding for: education
### Adapting target encoding for: enroll_in_edu_inst_last_wk
### Adapting target encoding for: marital_stat
### Adapting target encoding for: major_industry_code
### Adapting target encoding for: major_occupation_code
### Adapting target encoding for: race
### Adapting target encoding for: hispanic_origin
### Adapting target encoding for: sex
### Adapting target encoding for: member_of_a_labor_union
### Adapting target encoding for: reason_for_unemployment
### Adapting target encoding for: full_or_part_time_employment_stat
### Adapting target encoding for: tax_filer_stat
### Adapting target encoding for: region_of_previous_residence
### Adapting target encoding for: state_of_previous_residence
### Adapting target encoding for: detailed_household_and_family_stat
### Adapting target encoding for: detailed_household_summary_in_household
### Adapting target encoding for: migration_code-change_in_msa
### Adapting target encoding for: migration_code-change_in_reg
### Adapting target encoding for: migration_code-move_within_reg
### Adapting target encoding for: live_in_this_house_1_year_ago
### Adapting target encoding for: migration_prev_res_in_sunbelt
### Adapting target encoding for: family_members_under_18
### Adapting target encoding for: country_of_birth_father
### Adapting target encoding for: country_of_birth_mother
### Adapting target encoding for: country_of_birth_self
### Adapting target encoding for: citizenship
### Adapting target encoding for: own_business_or_self_employed
### Adapting target encoding for: fill_inc_questionnaire_for_veteran's_admin
### Adapting target encoding for: veterans_benefits
### Adapting target encoding for: year
Use /tmp/tmpj_0h78ld as temporary training directory
Starting reading the dataset
198/200 [============================>.] - ETA: 0s
Dataset read in 0:00:06.793717
Training model
Model trained in 0:04:32.752691
Compiling model
200/200 [==============================] - 280s 1s/step
Test accuracy: 95.81%

```
</div>
---
## Experiment 3: Decision Forests with trained embeddings

In this scenario, we build an encoder model that codes the categorical
features to embeddings, where the size of the embedding for a given categorical
feature is the square root to the size of its vocabulary.

We train these embeddings in a simple NN model through backpropagation.
After the embedding encoder is trained, we used it as a preprocessor to the
input features of a Gradient Boosted Tree model.

Note that the embeddings and a decision forest model cannot be trained
synergically in one phase, since decision forest models do not train with backpropagation.
Rather, embeddings has to be trained in an initial phase,
and then used as static inputs to the decision forest model.

### Implement feature encoding with embeddings


```python

def create_embedding_encoder(size=None):
    inputs = create_model_inputs()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Get the vocabulary of the categorical feature.
            vocabulary = sorted(
                [str(value) for value in list(train_data[feature_name].unique())]
            )
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
    # Apply dropout.
    encoded_features = layers.Dropout(rate=0.25)(encoded_features)
    # Perform non-linearity projection.
    encoded_features = layers.Dense(
        units=size if size else encoded_features.shape[-1], activation="gelu"
    )(encoded_features)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)

```

### Build an NN model to train the embeddings


```python

def create_nn_model(encoder):
    inputs = create_model_inputs()
    embeddings = encoder(inputs)
    output = layers.Dense(units=1, activation="sigmoid")(embeddings)

    nn_model = keras.Model(inputs=inputs, outputs=output)
    nn_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy("accuracy")],
    )
    return nn_model


embedding_encoder = create_embedding_encoder(size=64)
run_experiment(
    create_nn_model(embedding_encoder),
    train_data,
    test_data,
    num_epochs=5,
    batch_size=256,
)
```

<div class="k-default-codeblock">
```
Epoch 1/5
200/200 [==============================] - 10s 27ms/step - loss: 8303.1455 - accuracy: 0.9193
Epoch 2/5
200/200 [==============================] - 5s 27ms/step - loss: 1019.4900 - accuracy: 0.9371
Epoch 3/5
200/200 [==============================] - 5s 27ms/step - loss: 612.2844 - accuracy: 0.9416
Epoch 4/5
200/200 [==============================] - 5s 27ms/step - loss: 858.9774 - accuracy: 0.9397
Epoch 5/5
200/200 [==============================] - 5s 26ms/step - loss: 842.3922 - accuracy: 0.9421
Test accuracy: 95.0%

```
</div>
### Train and evaluate a Gradient Boosted Tree model with embeddings


```python
gbt_model = create_gbt_with_preprocessor(embedding_encoder)
run_experiment(gbt_model, train_data, test_data)
```

<div class="k-default-codeblock">
```
Use /tmp/tmpao5o88p6 as temporary training directory
Starting reading the dataset
199/200 [============================>.] - ETA: 0s
Dataset read in 0:00:06.722677
Training model
Model trained in 0:05:18.350298
Compiling model
200/200 [==============================] - 325s 2s/step
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
