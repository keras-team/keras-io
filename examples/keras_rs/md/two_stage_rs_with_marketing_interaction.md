# Two Stage Recommender System with Marketing Interaction

**Author:** Mansi Mehta<br>
**Date created:** 26/11/2025<br>
**Last modified:** 26/11/2025<br>
**Description:** Recommender System with Ranking and Retrival model for Marketing interaction.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/two_stage_rs_with_marketing_interaction.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/two_stage_rs_with_marketing_interaction.py)



# **Introduction**

This tutorial demonstrates a critical business scenario: a user lands on a website, and a
marketing engine must decide which specific ad to display from an inventory of thousands.
The goal is to maximize the Click-Through Rate (CTR). Showing irrelevant ads wastes
marketing budget and annoys the user. Therefore, we need a system that predicts the
probability of a specific user clicking on a specific ad based on their demographics and
browsing habits.

**Architecture**
1. **The Retrieval Stage:** Efficiently select an initial set of roughly 10-100
candidates from millions of possibilities. It weeds out items the user is definitely not
interested in.
User Tower: Embeds user features (ID, demographics, behavior) into a vector.
Item Tower: Embeds ad features (Ad ID, Topic) into a vector.
Interaction: The dot product of these two vectors represents similarity.
2. **The Ranking Stage:** It takes the output of the retrieval model and fine-tune the
order to select the single best ad to show.
A Deep Neural Network (MLP).
Interaction: It takes the User Embedding, Ad Embedding, and their similarity score to
predict a precise probability (0% to 100%) that the user will click.

![jpg](/img/examples/keras_rs/two_stage_rs_with_marketing_interaction/architecture.jpg)

# **Dataset**
We will use the [Ad Click
Prediction](https://www.kaggle.com/datasets/mafrojaakter/ad-click-data) Dataset from
Kaggle

**Feature Distribution of dataset:**
User Tower describes who is looking and features contains i.e Gender, City, Country, Age,
Daily Internet Usage, Daily Time Spent on Site, and Area Income.
Item Tower describes what is being shown and features contains Ad Topic Line, Ad ID.

In this tutorial, we are going to build and train a Two-Tower (User Tower and Ad Tower)
model using the Ad Click Prediction dataset from Kaggle.
We're going to:
1. **Data Pipeline:** Get our data and preprocess it for both Retrieval (implicit
feedback) and Ranking (explicit labels).
2. **Retrieval:** Implement and train a Two-Tower model to generate candidates.
3. **Ranking:** Implement and train a Neural Ranking model to predict click probabilities.
4. **Inference:** Run an end-to-end test (Retrieval --> Ranking) to generate
recommendations for a specific user.


```python
!!pip install -q keras-rs
```



```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import keras_rs
import tensorflow_datasets as tfds
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import layers
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

```
<div class="k-default-codeblock">
```
['',
 '\x1b[1m[\x1b[0m\x1b[34;49mnotice\x1b[0m\x1b[1;39;49m]\x1b[0m\x1b[39;49m A new release of pip is available: \x1b[0m\x1b[31;49m23.2.1\x1b[0m\x1b[39;49m -> \x1b[0m\x1b[32;49m25.3\x1b[0m',
 '\x1b[1m[\x1b[0m\x1b[34;49mnotice\x1b[0m\x1b[1;39;49m]\x1b[0m\x1b[39;49m To update, run: \x1b[0m\x1b[32;49mpip install --upgrade pip\x1b[0m']
```
</div>

# **Preparing Dataset**


```python
!pip install -q kaggle
!# Download the dataset (requires Kaggle API key in ~/.kaggle/kaggle.json)
!kaggle datasets download -d mafrojaakter/ad-click-data --unzip -p ./ad_click_dataset
```

    
<div class="k-default-codeblock">
```
[[34;49mnotice[1;39;49m][39;49m To update, run: [32;49mpip install --upgrade pip

Dataset URL: https://www.kaggle.com/datasets/mafrojaakter/ad-click-data
License(s): unknown

Downloading ad-click-data.zip to ./ad_click_dataset
```
</div>

  0%|                                               | 0.00/37.6k [00:00<?, ?B/s]

    
100%|███████████████████████████████████████| 37.6k/37.6k [00:00<00:00, 207kB/s]
    
100%|███████████████████████████████████████| 37.6k/37.6k [00:00<00:00, 206kB/s]



```python
data_path = "./ad_click_dataset/Ad_click_data.csv"
if not os.path.exists(data_path):
    # Fallback for filenames with spaces or different casing
    data_path = "./ad_click_dataset/Ad Click Data.csv"

ads_df = pd.read_csv(data_path)
# Clean column names
ads_df.columns = ads_df.columns.str.strip()
# Rename the column name
ads_df = ads_df.rename(
    columns={
        "Male": "gender",
        "Ad Topic Line": "ad_topic",
        "City": "city",
        "Country": "country",
        "Daily Time Spent on Site": "time_on_site",
        "Daily Internet Usage": "internet_usage",
        "Area Income": "area_income",
    }
)
# Add user_id and add_id column
ads_df["user_id"] = "user_" + ads_df.index.astype(str)
ads_df["ad_id"] = "ad_" + ads_df["ad_topic"].astype("category").cat.codes.astype(str)
# Remove nulls and normalize
ads_df = ads_df.dropna()
# normalize
numeric_cols = ["time_on_site", "internet_usage", "area_income", "Age"]
scaler = MinMaxScaler()
ads_df[numeric_cols] = scaler.fit_transform(ads_df[numeric_cols])

# Split the train and test datasets
x_train, x_test = train_test_split(ads_df, test_size=0.2, random_state=42)


def dict_to_tensor_features(df_features, continuous_features):
    tensor_dict = {}
    for k, v in df_features.items():
        if k in continuous_features:
            tensor_dict[k] = tf.expand_dims(tf.constant(v, dtype="float32"), axis=-1)
        else:
            v_str = np.array(v).astype(str).tolist()
            tensor_dict[k] = tf.expand_dims(tf.constant(v_str, dtype="string"), axis=-1)
    return tensor_dict


def create_retrieval_dataset(
    data_df,
    all_ads_features,
    all_ad_ids,
    user_features_list,
    ad_features_list,
    continuous_features_list,
):

    # Filter for Positive Interactions (Cicks)
    positive_interactions = data_df[data_df["Clicked on Ad"] == 1].copy()

    if positive_interactions.empty:
        return None

    def sample_negative(positive_ad_id):
        neg_ad_id = positive_ad_id
        while neg_ad_id == positive_ad_id:
            neg_ad_id = np.random.choice(all_ad_ids)
        return neg_ad_id

    def create_triplets_row(pos_row):
        pos_ad_id = pos_row.ad_id
        neg_ad_id = sample_negative(pos_ad_id)

        neg_ad_row = all_ads_features[all_ads_features["ad_id"] == neg_ad_id].iloc[0]
        user_features_dict = {
            name: getattr(pos_row, name) for name in user_features_list
        }
        pos_ad_features_dict = {
            name: getattr(pos_row, name) for name in ad_features_list
        }
        neg_ad_features_dict = {name: neg_ad_row[name] for name in ad_features_list}

        return {
            "user": user_features_dict,
            "positive_ad": pos_ad_features_dict,
            "negative_ad": neg_ad_features_dict,
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        triplets = list(
            executor.map(
                create_triplets_row, positive_interactions.itertuples(index=False)
            )
        )

    triplets_df = pd.DataFrame(triplets)
    user_df = triplets_df["user"].apply(pd.Series)
    pos_ad_df = triplets_df["positive_ad"].apply(pd.Series)
    neg_ad_df = triplets_df["negative_ad"].apply(pd.Series)

    user_features_tensor = dict_to_tensor_features(
        user_df.to_dict("list"), continuous_features_list
    )
    pos_ad_features_tensor = dict_to_tensor_features(
        pos_ad_df.to_dict("list"), continuous_features_list
    )
    neg_ad_features_tensor = dict_to_tensor_features(
        neg_ad_df.to_dict("list"), continuous_features_list
    )

    features = {
        "user": user_features_tensor,
        "positive_ad": pos_ad_features_tensor,
        "negative_ad": neg_ad_features_tensor,
    }
    y_true = tf.ones((triplets_df.shape[0], 1), dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((features, y_true))
    buffer_size = len(triplets_df)
    dataset = (
        dataset.shuffle(buffer_size=buffer_size)
        .batch(64)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


user_clicked_ads = (
    x_train[x_train["Clicked on Ad"] == 1]
    .groupby("user_id")["ad_id"]
    .apply(set)
    .to_dict()
)

for u in x_train["user_id"].unique():
    if u not in user_clicked_ads:
        user_clicked_ads[u] = set()

AD_FEATURES = ["ad_id", "ad_topic"]
USER_FEATURES = [
    "user_id",
    "gender",
    "city",
    "country",
    "time_on_site",
    "internet_usage",
    "area_income",
    "Age",
]
continuous_features = ["time_on_site", "internet_usage", "area_income", "Age"]

all_ads_features = x_train[AD_FEATURES].drop_duplicates().reset_index(drop=True)
all_ad_ids = all_ads_features["ad_id"].tolist()

retrieval_train_dataset = create_retrieval_dataset(
    data_df=x_train,
    all_ads_features=all_ads_features,
    all_ad_ids=all_ad_ids,
    user_features_list=USER_FEATURES,
    ad_features_list=AD_FEATURES,
    continuous_features_list=continuous_features,
)

retrieval_test_dataset = create_retrieval_dataset(
    data_df=x_test,
    all_ads_features=all_ads_features,
    all_ad_ids=all_ad_ids,
    user_features_list=USER_FEATURES,
    ad_features_list=AD_FEATURES,
    continuous_features_list=continuous_features,
)
```

# **Implement the Retrival Model**
For the Retrieval stage, we will build a Two-Tower Model.

**The Architecture Components:**

1. User Tower:User features (User ID, demographics, behavior metrics like time_on_site).
It encodes these mixed features into a fixed-size vector representation called the User
Embedding.
2. Item (Ad) Tower:Ad features (Ad ID, Ad Topic Line).It encodes these features into a
fixed-size vector representation called the Item Embedding.
3. Interaction (Similarity):We calculate the Dot Product between the User Embedding and
the Item Embedding.


```python
keras.utils.set_random_seed(42)

vocab_map = {
    "user_id": x_train["user_id"].unique(),
    "gender": x_train["gender"].astype(str).unique(),
    "city": x_train["city"].unique(),
    "country": x_train["country"].unique(),
    "ad_id": x_train["ad_id"].unique(),
    "ad_topic": x_train["ad_topic"].unique(),
}
cont_feats = ["time_on_site", "internet_usage", "area_income", "Age"]

normalizers = {}
for f in cont_feats:
    norm = layers.Normalization(axis=None)
    norm.adapt(x_train[f].values.astype("float32"))
    normalizers[f] = norm


def build_tower(feature_names, continuous_names=None, embed_dim=64, name="tower"):
    inputs, embeddings = {}, []

    for feat in feature_names:
        if feat in vocab_map:
            inp = keras.Input(shape=(1,), dtype=tf.string, name=feat)
            inputs[feat] = inp
            vocab = list(vocab_map[feat])
            x = layers.StringLookup(vocabulary=vocab)(inp)
            x = layers.Embedding(
                len(vocab) + 1, embed_dim, embeddings_regularizer="l2"
            )(x)
            embeddings.append(layers.Flatten()(x))

    if continuous_names:
        for feat in continuous_names:
            inp = keras.Input(shape=(1,), dtype=tf.float32, name=feat)
            inputs[feat] = inp
            embeddings.append(normalizers[feat](inp))

    x = layers.Concatenate()(embeddings)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(embed_dim)(layers.Dropout(0.2)(x))

    return keras.Model(inputs=inputs, outputs=output, name=name)


user_tower = build_tower(
    ["user_id", "gender", "city", "country"], cont_feats, name="user_tower"
)
ad_tower = build_tower(["ad_id", "ad_topic"], name="ad_tower")


def bpr_hinge_loss(y_true, y_pred):
    margin = 1.0
    return -tf.math.log(tf.nn.sigmoid(y_pred) + 1e-10)


class RetrievalModel(keras.Model):
    def __init__(self, user_tower_instance, ad_tower_instance, **kwargs):
        super().__init__(**kwargs)
        self.user_tower = user_tower
        self.ad_tower = ad_tower
        self.ln_user = layers.LayerNormalization()
        self.ln_ad = layers.LayerNormalization()

    def call(self, inputs):
        u_emb = self.ln_user(self.user_tower(inputs["user"]))
        pos_emb = self.ln_ad(self.ad_tower(inputs["positive_ad"]))
        neg_emb = self.ln_ad(self.ad_tower(inputs["negative_ad"]))
        pos_score = keras.ops.sum(u_emb * pos_emb, axis=1, keepdims=True)
        neg_score = keras.ops.sum(u_emb * neg_emb, axis=1, keepdims=True)
        return pos_score - neg_score

    def get_embeddings(self, inputs):
        u_emb = self.ln_user(self.user_tower(inputs["user"]))
        ad_emb = self.ln_ad(self.ad_tower(inputs["positive_ad"]))
        dot_interaction = keras.ops.sum(u_emb * ad_emb, axis=1, keepdims=True)
        return u_emb, ad_emb, dot_interaction


retrieval_model = RetrievalModel(user_tower, ad_tower)
retrieval_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=bpr_hinge_loss
)
history = retrieval_model.fit(retrieval_train_dataset, epochs=30)

pd.DataFrame(history.history).plot(
    subplots=True, layout=(1, 3), figsize=(12, 4), title="Retrival Model Metrics"
)
plt.show()
```

<div class="k-default-codeblock">
```
Epoch 1/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - loss: 2.8117

Epoch 2/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.3631

Epoch 3/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.0918

Epoch 4/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.9143

Epoch 5/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.7872

Epoch 6/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.6925

Epoch 7/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.6203

Epoch 8/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.5641

Epoch 9/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.5190

Epoch 10/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4817

Epoch 11/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4499

Epoch 12/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4220

Epoch 13/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.3970

Epoch 14/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.3743

Epoch 15/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3537

Epoch 16/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3346

Epoch 17/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3171

Epoch 18/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3009

Epoch 19/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2858

Epoch 20/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2718

Epoch 21/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2587

Epoch 22/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2465

Epoch 23/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2350

Epoch 24/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2243

Epoch 25/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2142

Epoch 26/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.2046

Epoch 27/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.1956

Epoch 28/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.1871

Epoch 29/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.1791

Epoch 30/30

6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.1715
```
</div>

![png](/img/examples/keras_rs/two_stage_rs_with_marketing_interaction/two_stage_rs_with_marketing_interaction_9_90.png)
    


# **Predictions of Retrival Model**
Two-Tower model is trained, we need to use it to generate candidates.

We can implement inference pipeline using three steps:
1. Indexing: We can run the Item Tower once for all available ads to generate their
embeddings.
2. Query Encoding: When a user arrives, we pass their features through the User Tower to
generate a User Embedding.
3. Nearest Neighbor Search: We search the index to find the Ad Embeddings closest to the
User Embedding (highest dot product).

Keras-RS [BruteForceRetrieval
layer](https://keras.io/keras_rs/api/retrieval_layers/brute_force_retrieval/) calculates
dot product between the user and every single item in the index to find exact top-K
matches


```python
USER_CATEGORICAL = ["user_id", "gender", "city", "country"]
CONTINUOUS_FEATURES = ["time_on_site", "internet_usage", "area_income", "Age"]
USER_FEATURES = USER_CATEGORICAL + CONTINUOUS_FEATURES


class BruteForceRetrievalWrapper:
    def __init__(self, model, ads_df, ad_features, user_features, k=10):
        self.model, self.k = model, k
        self.user_features = user_features
        unique_ads = ads_df[ad_features].drop_duplicates("ad_id").reset_index(drop=True)
        self.ids = unique_ads["ad_id"].values
        self.topic_map = dict(zip(unique_ads["ad_id"], unique_ads["ad_topic"]))
        ad_inputs = {
            "ad_id": tf.constant(self.ids.astype(str)),
            "ad_topic": tf.constant(unique_ads["ad_topic"].astype(str).values),
        }
        self.candidate_embs = model.ln_ad(model.ad_tower(ad_inputs))

    def query_batch(self, user_df):
        inputs = {
            k: tf.constant(
                user_df[k].values.astype(float if k in CONTINUOUS_FEATURES else str)
            )
            for k in self.user_features
            if k in user_df.columns
        }
        u_emb = self.model.ln_user(self.model.user_tower(inputs))
        scores = tf.linalg.matmul(u_emb, self.candidate_embs, transpose_b=True)
        top_scores, top_indices = tf.math.top_k(scores, k=self.k)
        return top_scores.numpy(), top_indices.numpy()

    def decode_results(self, scores, indices):
        results = []
        for row_scores, row_indices in zip(scores, indices):
            retrieved_ids = self.ids[row_indices]
            results.append(
                [
                    {"ad_id": aid, "ad_topic": self.topic_map[aid], "score": float(s)}
                    for aid, s in zip(retrieved_ids, row_scores)
                ]
            )
        return results


retrieval_engine = BruteForceRetrievalWrapper(
    model=retrieval_model,
    ads_df=ads_df,
    ad_features=["ad_id", "ad_topic"],
    user_features=USER_FEATURES,
    k=10,
)
sample_user = pd.DataFrame([x_test.iloc[0]])
scores, indices = retrieval_engine.query_batch(sample_user)
top_ads = retrieval_engine.decode_results(scores, indices)[0]
```

# **Implementation of Ranking Model**
Retrieval model  only calculates a simple similarity score (Dot Product). It doesn't
account for complex feature interactions.
So we need to build ranking model after words retrival model.

**Architecture**
1. **Feature Extraction:** We reuse the trained User Tower and Ad Tower from the
Retrieval stage. We freeze these towers (trainable = False) so their weights don't
change.
2. **Interaction:** Instead of just a dot product, we concatenate three inputs- The User
EmbeddingThe Ad EmbeddingThe Dot Product (Similarity)
3. **Scorer(MLP):** These concatenated inputs are fed into a Multi-Layer Perceptron—a
stack of Dense layers. This network learns the non-linear relationships between the user
and the ad.
4. **Output:** The final layer uses a Sigmoid activation to output a single probability
between 0.0 and 1.0 (Likelihood of a Click).


```python
retrieval_model.trainable = False


def create_ranking_ds(df):
    inputs = {
        "user": dict_to_tensor_features(df[USER_FEATURES], continuous_features),
        "positive_ad": dict_to_tensor_features(df[AD_FEATURES], continuous_features),
    }
    return (
        tf.data.Dataset.from_tensor_slices(
            (inputs, df["Clicked on Ad"].values.astype("float32"))
        )
        .shuffle(10000)
        .batch(256)
        .prefetch(tf.data.AUTOTUNE)
    )


ranking_train_dataset = create_ranking_ds(x_train)
ranking_test_dataset = create_ranking_ds(x_test)


class RankingModel(keras.Model):
    def __init__(self, retrieval_model, **kwargs):
        super().__init__(**kwargs)
        self.retrieval = retrieval_model
        self.mlp = keras.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def call(self, inputs):
        u_emb, ad_emb, dot = self.retrieval.get_embeddings(inputs)
        return self.mlp(keras.ops.concatenate([u_emb, ad_emb, dot], axis=-1))


ranking_model = RankingModel(retrieval_model)
ranking_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["AUC", "accuracy"],
)
history1 = ranking_model.fit(ranking_train_dataset, epochs=20)

pd.DataFrame(history1.history).plot(
    subplots=True, layout=(1, 3), figsize=(12, 4), title="Ranking Model Metrics"
)
plt.show()

ranking_model.evaluate(ranking_test_dataset)
```

<div class="k-default-codeblock">
```
Epoch 1/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - AUC: 0.6079 - accuracy: 0.4961 - loss: 0.6890

Epoch 2/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.8329 - accuracy: 0.5748 - loss: 0.6423

Epoch 3/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - AUC: 0.9284 - accuracy: 0.7467 - loss: 0.5995

Epoch 4/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9636 - accuracy: 0.8766 - loss: 0.5599

Epoch 5/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - AUC: 0.9763 - accuracy: 0.9213 - loss: 0.5229

Epoch 6/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9824 - accuracy: 0.9304 - loss: 0.4876

Epoch 7/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9862 - accuracy: 0.9331 - loss: 0.4540

Epoch 8/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - AUC: 0.9880 - accuracy: 0.9357 - loss: 0.4224

Epoch 9/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9898 - accuracy: 0.9436 - loss: 0.3920

Epoch 10/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9911 - accuracy: 0.9475 - loss: 0.3633

Epoch 11/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - AUC: 0.9914 - accuracy: 0.9528 - loss: 0.3361

Epoch 12/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - AUC: 0.9923 - accuracy: 0.9580 - loss: 0.3103

Epoch 13/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9925 - accuracy: 0.9619 - loss: 0.2866

Epoch 14/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9931 - accuracy: 0.9633 - loss: 0.2643

Epoch 15/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9935 - accuracy: 0.9633 - loss: 0.2436

Epoch 16/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9938 - accuracy: 0.9659 - loss: 0.2247

Epoch 17/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9942 - accuracy: 0.9646 - loss: 0.2076

Epoch 18/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - AUC: 0.9945 - accuracy: 0.9659 - loss: 0.1918

Epoch 19/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9947 - accuracy: 0.9672 - loss: 0.1777

Epoch 20/20

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - AUC: 0.9953 - accuracy: 0.9685 - loss: 0.1645
```
</div>

![png](/img/examples/keras_rs/two_stage_rs_with_marketing_interaction/two_stage_rs_with_marketing_interaction_13_60.png)
    


    
<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 230ms/step - AUC: 0.9904 - accuracy: 0.9476 - loss: 0.2319

[0.2318607121706009, 0.9903508424758911, 0.9476439952850342]
```
</div>

# **Predictions of Ranking Model**
The retrieval model gave us a list of ads that are generally relevant (high dot product
similarity). The ranking model will now calculate the specific probability (0% to 100%)
that the user will click each of those ads.

The Ranking model expects pairs of (User, Ad). Since we are scoring 10 ads for 1 user, we
cannot just pass the user features once.We effectively take user's features 10 times to
create a batch.


```python

def rerank_ads_for_user(user_row, retrieved_ads, ranking_model):
    ads_df = pd.DataFrame(retrieved_ads)
    num_ads = len(ads_df)
    user_inputs = {
        k: tf.fill(
            (num_ads, 1),
            str(user_row[k]) if k not in continuous_features else float(user_row[k]),
        )
        for k in USER_FEATURES
    }
    ad_inputs = {
        k: tf.reshape(tf.constant(ads_df[k].astype(str).values), (-1, 1))
        for k in AD_FEATURES
    }
    scores = (
        ranking_model({"user": user_inputs, "positive_ad": ad_inputs}).numpy().flatten()
    )
    ads_df["ranking_score"] = scores
    return ads_df.sort_values("ranking_score", ascending=False).to_dict("records")


sample_user = x_test.iloc[0]
scores, indices = retrieval_engine.query_batch(pd.DataFrame([sample_user]))
top_ads = retrieval_engine.decode_results(scores, indices)[0]
final_ranked_ads = rerank_ads_for_user(sample_user, top_ads, ranking_model)
print(f"User: {sample_user['user_id']}")
print(f"{'Ad ID':<10} | {'Topic':<30} | {'Retrival Score':<11} | {'Rank Probability'}")
for item in final_ranked_ads:
    print(
        f"{item['ad_id']:<10} | {item['ad_topic'][:28]:<30} | {item['score']:.4f}      |{item['ranking_score']*100:.2f}%"
    )
```

<div class="k-default-codeblock">
```
User: user_216
Ad ID      | Topic                          | Retrival Score | Rank Probability
ad_305     | Front-line fault-tolerant in   | 8.2131      |99.27%
ad_318     | Front-line upward-trending g   | 7.6231      |99.17%
ad_758     | Right-sized multi-tasking so   | 7.1814      |99.06%
ad_767     | Robust object-oriented Graph   | 7.2068      |99.02%
ad_620     | Polarized modular function     | 7.2857      |98.92%
ad_522     | Open-architected full-range    | 7.0892      |98.82%
ad_771     | Robust web-enabled attitude    | 7.3828      |98.81%
ad_810     | Sharable optimal capacity      | 6.7046      |98.69%
ad_31      | Ameliorated well-modulated c   | 6.9498      |98.40%
ad_104     | Configurable 24/7 hub          | 6.7244      |98.39%
```
</div>
