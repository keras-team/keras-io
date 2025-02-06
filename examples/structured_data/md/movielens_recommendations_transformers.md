# A Transformer-based recommendation system

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2020/12/30<br>
**Last modified:** 2025/01/27<br>
**Description:** Rating rate prediction using the Behavior Sequence Transformer (BST) model on the Movielens.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/movielens_recommendations_transformers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/movielens_recommendations_transformers.py)



---
## Introduction

This example demonstrates the [Behavior Sequence Transformer (BST)](https://arxiv.org/abs/1905.06874)
model, by Qiwei Chen et al., using the [Movielens dataset](https://grouplens.org/datasets/movielens/).
The BST model leverages the sequential behaviour of the users in watching and rating movies,
as well as user profile and movie features, to predict the rating of the user to a target movie.

More precisely, the BST model aims to predict the rating of a target movie by accepting
the following inputs:

1. A fixed-length *sequence* of `movie_ids` watched by a user.
2. A fixed-length *sequence* of the `ratings` for the movies watched by a user.
3. A *set* of user features, including `user_id`, `sex`, `occupation`, and `age_group`.
4. A *set* of `genres` for each movie in the input sequence and the target movie.
5. A `target_movie_id` for which to predict the rating.

This example modifies the original BST model in the following ways:

1. We incorporate the movie features (genres) into the processing of the embedding of each
movie of the input sequence and the target movie, rather than treating them as "other features"
outside the transformer layer.
2. We utilize the ratings of movies in the input sequence, along with the their positions
in the sequence, to update them before feeding them into the self-attention layer.


Note that this example should be run with TensorFlow 2.4 or higher.

---
## The dataset

We use the [1M version of the Movielens dataset](https://grouplens.org/datasets/movielens/1m/).
The dataset includes around 1 million ratings from 6000 users on 4000 movies,
along with some user features, movie genres. In addition, the timestamp of each user-movie
rating is provided, which allows creating sequences of movie ratings for each user,
as expected by the BST model.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # or torch, or tensorflow

import math
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd

import keras
from keras import layers, ops
from keras.layers import StringLookup
```

---
## Prepare the data

### Download and prepare the DataFrames

First, let's download the movielens data.

The downloaded folder will contain three data files: `users.dat`, `movies.dat`,
and `ratings.dat`.


```python
urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
ZipFile("movielens.zip", "r").extractall()
```

Then, we load the data into pandas DataFrames with their proper column names.


```python
users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    encoding="ISO-8859-1",
    engine="python",
)

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
    encoding="ISO-8859-1",
    engine="python",
)

movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    names=["movie_id", "title", "genres"],
    encoding="ISO-8859-1",
    engine="python",
)
```

Here, we do some simple data processing to fix the data types of the columns.


```python
users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
users["age_group"] = users["age_group"].apply(lambda x: f"group_{x}")
users["occupation"] = users["occupation"].apply(lambda x: f"occupation_{x}")

movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")

ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))
```

Each movie has multiple genres. We split them into separate columns in the `movies`
DataFrame.


```python
genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime"]
genres += ["Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical"]
genres += ["Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

for genre in genres:
    movies[genre] = movies["genres"].apply(
        lambda values: int(genre in values.split("|"))
    )

```

### Transform the movie ratings data into sequences

First, let's sort the the ratings data using the `unix_timestamp`, and then group the
`movie_id` values and the `rating` values by `user_id`.

The output DataFrame will have a record for each `user_id`, with two ordered lists
(sorted by rating datetime): the movies they have rated, and their ratings of these movies.


```python
ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

ratings_data = pd.DataFrame(
    data={
        "user_id": list(ratings_group.groups.keys()),
        "movie_ids": list(ratings_group.movie_id.apply(list)),
        "ratings": list(ratings_group.rating.apply(list)),
        "timestamps": list(ratings_group.unix_timestamp.apply(list)),
    }
)

```

Now, let's split the `movie_ids` list into a set of sequences of a fixed length.
We do the same for the `ratings`. Set the `sequence_length` variable to change the length
of the input sequence to the model. You can also change the `step_size` to control the
number of sequences to generate for each user.


```python
sequence_length = 4
step_size = 2


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

ratings_data.ratings = ratings_data.ratings.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

del ratings_data["timestamps"]
```

After that, we process the output to have each sequence in a separate records in
the DataFrame. In addition, we join the user features with the ratings data.


```python
ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
    "movie_ids", ignore_index=True
)
ratings_data_rating = ratings_data[["ratings"]].explode("ratings", ignore_index=True)
ratings_data_transformed = pd.concat([ratings_data_movies, ratings_data_rating], axis=1)
ratings_data_transformed = ratings_data_transformed.join(
    users.set_index("user_id"), on="user_id"
)
ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
    lambda x: ",".join(x)
)
ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
    lambda x: ",".join([str(v) for v in x])
)

del ratings_data_transformed["zip_code"]

ratings_data_transformed.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
)
```

With `sequence_length` of 4 and `step_size` of 2, we end up with 498,623 sequences.

Finally, we split the data into training and testing splits, with 85% and 15% of
the instances, respectively, and store them to CSV files.


```python
random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
train_data = ratings_data_transformed[random_selection]
test_data = ratings_data_transformed[~random_selection]

train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
test_data.to_csv("test_data.csv", index=False, sep="|", header=False)
```

---
## Define metadata


```python
CSV_HEADER = list(ratings_data_transformed.columns)

CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "user_id": list(users.user_id.unique()),
    "movie_id": list(movies.movie_id.unique()),
    "sex": list(users.sex.unique()),
    "age_group": list(users.age_group.unique()),
    "occupation": list(users.occupation.unique()),
}

USER_FEATURES = ["sex", "age_group", "occupation"]

MOVIE_FEATURES = ["genres"]

```

---
## Encode input features

The `encode_input_features` function works as follows:

1. Each categorical user feature is encoded using `layers.Embedding`, with embedding
dimension equals to the square root of the vocabulary size of the feature.
The embeddings of these features are concatenated to form a single input tensor.

2. Each movie in the movie sequence and the target movie is encoded `layers.Embedding`,
where the dimension size is the square root of the number of movies.

3. A multi-hot genres vector for each movie is concatenated with its embedding vector,
and processed using a non-linear `layers.Dense` to output a vector of the same movie
embedding dimensions.

4. A positional embedding is added to each movie embedding in the sequence, and then
multiplied by its rating from the ratings sequence.

5. The target movie embedding is concatenated to the sequence movie embeddings, producing
a tensor with the shape of `[batch size, sequence length, embedding size]`, as expected
by the attention layer for the transformer architecture.

6. The method returns a tuple of two elements:  `encoded_transformer_features` and
`encoded_other_features`.


```python
# Required for tf.data.Dataset
import tensorflow as tf


def get_dataset_from_csv(csv_file_path, batch_size, shuffle=True):

    def process(features):
        movie_ids_string = features["sequence_movie_ids"]
        sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()
        # The last movie id in the sequence is the target movie.
        features["target_movie_id"] = sequence_movie_ids[:, -1]
        features["sequence_movie_ids"] = sequence_movie_ids[:, :-1]
        # Sequence ratings
        ratings_string = features["sequence_ratings"]
        sequence_ratings = tf.strings.to_number(
            tf.strings.split(ratings_string, ","), tf.dtypes.float32
        ).to_tensor()
        # The last rating in the sequence is the target for the model to predict.
        target = sequence_ratings[:, -1]
        features["sequence_ratings"] = sequence_ratings[:, :-1]

        def encoding_helper(feature_name):

            # This are target_movie_id and sequence_movie_ids and they have the same
            # vocabulary as movie_id.
            if feature_name not in CATEGORICAL_FEATURES_WITH_VOCABULARY:
                vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY["movie_id"]
                index_lookup = StringLookup(
                    vocabulary=vocabulary, mask_token=None, num_oov_indices=0
                )
                # Convert the string input values into integer indices.
                value_index = index_lookup(features[feature_name])
                features[feature_name] = value_index
            else:
                # movie_id is not part of the features, hence not processed. It was mainly required
                # for its vocabulary above.
                if feature_name == "movie_id":
                    pass
                else:
                    vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
                    index_lookup = StringLookup(
                        vocabulary=vocabulary, mask_token=None, num_oov_indices=0
                    )
                    # Convert the string input values into integer indices.
                    value_index = index_lookup(features[feature_name])
                    features[feature_name] = value_index

        # Encode the user features
        for feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            encoding_helper(feature_name)
        # Encoding target_movie_id and returning it as the target variable
        encoding_helper("target_movie_id")
        # Encoding sequence movie_ids.
        encoding_helper("sequence_movie_ids")
        return dict(features), target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        field_delim="|",
        shuffle=shuffle,
    ).map(process)
    return dataset


def encode_input_features(
    inputs,
    include_user_id,
    include_user_features,
    include_movie_features,
):
    encoded_transformer_features = []
    encoded_other_features = []

    other_feature_names = []
    if include_user_id:
        other_feature_names.append("user_id")
    if include_user_features:
        other_feature_names.extend(USER_FEATURES)

    ## Encode user features
    for feature_name in other_feature_names:
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f"{feature_name}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder(inputs[feature_name]))

    ## Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    ## Create a movie embedding encoder
    movie_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY["movie_id"]
    movie_embedding_dims = int(math.sqrt(len(movie_vocabulary)))
    # Create an embedding layer with the specified dimensions.
    movie_embedding_encoder = layers.Embedding(
        input_dim=len(movie_vocabulary),
        output_dim=movie_embedding_dims,
        name=f"movie_embedding",
    )
    # Create a vector lookup for movie genres.
    genre_vectors = movies[genres].to_numpy()
    movie_genres_lookup = layers.Embedding(
        input_dim=genre_vectors.shape[0],
        output_dim=genre_vectors.shape[1],
        embeddings_initializer=keras.initializers.Constant(genre_vectors),
        trainable=False,
        name="genres_vector",
    )
    # Create a processing layer for genres.
    movie_embedding_processor = layers.Dense(
        units=movie_embedding_dims,
        activation="relu",
        name="process_movie_embedding_with_genres",
    )

    ## Define a function to encode a given movie id.
    def encode_movie(movie_id):
        # Convert the string input values into integer indices.
        movie_embedding = movie_embedding_encoder(movie_id)
        encoded_movie = movie_embedding
        if include_movie_features:
            movie_genres_vector = movie_genres_lookup(movie_id)
            encoded_movie = movie_embedding_processor(
                layers.concatenate([movie_embedding, movie_genres_vector])
            )
        return encoded_movie

    ## Encoding target_movie_id
    target_movie_id = inputs["target_movie_id"]
    encoded_target_movie = encode_movie(target_movie_id)

    ## Encoding sequence movie_ids.
    sequence_movies_ids = inputs["sequence_movie_ids"]
    encoded_sequence_movies = encode_movie(sequence_movies_ids)
    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=sequence_length,
        output_dim=movie_embedding_dims,
        name="position_embedding",
    )
    positions = ops.arange(start=0, stop=sequence_length - 1, step=1)
    encodded_positions = position_embedding_encoder(positions)
    # Retrieve sequence ratings to incorporate them into the encoding of the movie.
    sequence_ratings = inputs["sequence_ratings"]
    sequence_ratings = ops.expand_dims(sequence_ratings, -1)
    # Add the positional encoding to the movie encodings and multiply them by rating.
    encoded_sequence_movies_with_poistion_and_rating = layers.Multiply()(
        [(encoded_sequence_movies + encodded_positions), sequence_ratings]
    )

    # Construct the transformer inputs.
    for i in range(sequence_length - 1):
        feature = encoded_sequence_movies_with_poistion_and_rating[:, i, ...]
        feature = ops.expand_dims(feature, 1)
        encoded_transformer_features.append(feature)
    encoded_transformer_features.append(encoded_target_movie)
    encoded_transformer_features = layers.concatenate(
        encoded_transformer_features, axis=1
    )
    return encoded_transformer_features, encoded_other_features

```

---
## Create model inputs


```python

def create_model_inputs():
    return {
        "user_id": keras.Input(name="user_id", shape=(1,), dtype="int32"),
        "sequence_movie_ids": keras.Input(
            name="sequence_movie_ids", shape=(sequence_length - 1,), dtype="int32"
        ),
        "target_movie_id": keras.Input(
            name="target_movie_id", shape=(1,), dtype="int32"
        ),
        "sequence_ratings": keras.Input(
            name="sequence_ratings", shape=(sequence_length - 1,), dtype="float32"
        ),
        "sex": keras.Input(name="sex", shape=(1,), dtype="int32"),
        "age_group": keras.Input(name="age_group", shape=(1,), dtype="int32"),
        "occupation": keras.Input(name="occupation", shape=(1,), dtype="int32"),
    }

```

---
## Create a BST model


```python
include_user_id = False
include_user_features = False
include_movie_features = False

hidden_units = [256, 128]
dropout_rate = 0.1
num_heads = 3


def create_model():

    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(
        inputs, include_user_id, include_user_features, include_movie_features
    )
    # Create a multi-headed attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other_features.
    if other_features is not None:
        features = layers.concatenate(
            [features, layers.Reshape([other_features.shape[-1]])(other_features)]
        )

    # Fully-connected layers.
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()
```

<div class="k-default-codeblock">
```
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.

```
</div>
---
## Run training and evaluation experiment


```python
# Compile the model.
model.compile(
    optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

# Read the training data.

train_dataset = get_dataset_from_csv("train_data.csv", batch_size=265, shuffle=True)

# Fit the model with the training data.
model.fit(train_dataset, epochs=2)

# Read the test data.
test_dataset = get_dataset_from_csv("test_data.csv", batch_size=265)

# Evaluate the model on the test data.
_, rmse = model.evaluate(test_dataset, verbose=0)
print(f"Test MAE: {round(rmse, 3)}")
```

<div class="k-default-codeblock">
```
Epoch 1/2

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  7s 7s/step - loss: 12.6790 - mean_absolute_error: 3.2470


  2/Unknown  7s 175ms/step - loss: 11.8442 - mean_absolute_error: 3.1216


  3/Unknown  7s 169ms/step - loss: 11.2701 - mean_absolute_error: 3.0387


  4/Unknown  7s 162ms/step - loss: 10.7890 - mean_absolute_error: 2.9640


  5/Unknown  8s 160ms/step - loss: 10.3522 - mean_absolute_error: 2.8934


  6/Unknown  8s 173ms/step - loss: 9.9734 - mean_absolute_error: 2.8301 


  7/Unknown  8s 181ms/step - loss: 9.6343 - mean_absolute_error: 2.7711


  8/Unknown  8s 176ms/step - loss: 9.3284 - mean_absolute_error: 2.7167


  9/Unknown  8s 174ms/step - loss: 9.0492 - mean_absolute_error: 2.6662


 10/Unknown  8s 172ms/step - loss: 8.7926 - mean_absolute_error: 2.6188


 11/Unknown  9s 171ms/step - loss: 8.5587 - mean_absolute_error: 2.5750


 12/Unknown  9s 169ms/step - loss: 8.3417 - mean_absolute_error: 2.5337


 13/Unknown  9s 169ms/step - loss: 8.1379 - mean_absolute_error: 2.4944


 14/Unknown  9s 169ms/step - loss: 7.9466 - mean_absolute_error: 2.4568


 15/Unknown  9s 168ms/step - loss: 7.7681 - mean_absolute_error: 2.4216


 16/Unknown  9s 168ms/step - loss: 7.6015 - mean_absolute_error: 2.3885


 17/Unknown  10s 168ms/step - loss: 7.4451 - mean_absolute_error: 2.3571


 18/Unknown  10s 168ms/step - loss: 7.2978 - mean_absolute_error: 2.3273


 19/Unknown  10s 167ms/step - loss: 7.1582 - mean_absolute_error: 2.2988


 20/Unknown  10s 167ms/step - loss: 7.0262 - mean_absolute_error: 2.2716


 21/Unknown  10s 167ms/step - loss: 6.9011 - mean_absolute_error: 2.2458


 22/Unknown  10s 166ms/step - loss: 6.7823 - mean_absolute_error: 2.2211


 23/Unknown  11s 166ms/step - loss: 6.6689 - mean_absolute_error: 2.1972


 24/Unknown  11s 167ms/step - loss: 6.5609 - mean_absolute_error: 2.1744


 25/Unknown  11s 166ms/step - loss: 6.4579 - mean_absolute_error: 2.1524


 26/Unknown  11s 166ms/step - loss: 6.3597 - mean_absolute_error: 2.1315


 27/Unknown  11s 166ms/step - loss: 6.2656 - mean_absolute_error: 2.1112


 28/Unknown  11s 166ms/step - loss: 6.1754 - mean_absolute_error: 2.0917


 29/Unknown  12s 166ms/step - loss: 6.0891 - mean_absolute_error: 2.0730


 30/Unknown  12s 166ms/step - loss: 6.0059 - mean_absolute_error: 2.0548


 31/Unknown  12s 166ms/step - loss: 5.9260 - mean_absolute_error: 2.0372


 32/Unknown  12s 166ms/step - loss: 5.8491 - mean_absolute_error: 2.0203


 33/Unknown  12s 167ms/step - loss: 5.7751 - mean_absolute_error: 2.0039


 34/Unknown  12s 167ms/step - loss: 5.7038 - mean_absolute_error: 1.9880


 35/Unknown  13s 167ms/step - loss: 5.6349 - mean_absolute_error: 1.9727


 36/Unknown  13s 168ms/step - loss: 5.5687 - mean_absolute_error: 1.9579


 37/Unknown  13s 168ms/step - loss: 5.5050 - mean_absolute_error: 1.9435


 38/Unknown  13s 169ms/step - loss: 5.4435 - mean_absolute_error: 1.9297


 39/Unknown  13s 170ms/step - loss: 5.3839 - mean_absolute_error: 1.9163


 40/Unknown  13s 170ms/step - loss: 5.3267 - mean_absolute_error: 1.9033


 41/Unknown  14s 170ms/step - loss: 5.2712 - mean_absolute_error: 1.8908


 42/Unknown  14s 171ms/step - loss: 5.2175 - mean_absolute_error: 1.8786


 43/Unknown  14s 171ms/step - loss: 5.1653 - mean_absolute_error: 1.8667


 44/Unknown  14s 171ms/step - loss: 5.1147 - mean_absolute_error: 1.8551


 45/Unknown  14s 170ms/step - loss: 5.0658 - mean_absolute_error: 1.8439


 46/Unknown  15s 170ms/step - loss: 5.0183 - mean_absolute_error: 1.8330


 47/Unknown  15s 171ms/step - loss: 4.9722 - mean_absolute_error: 1.8224


 48/Unknown  15s 171ms/step - loss: 4.9275 - mean_absolute_error: 1.8121


 49/Unknown  15s 171ms/step - loss: 4.8840 - mean_absolute_error: 1.8021


 50/Unknown  15s 170ms/step - loss: 4.8417 - mean_absolute_error: 1.7923


 51/Unknown  15s 170ms/step - loss: 4.8005 - mean_absolute_error: 1.7828


 52/Unknown  16s 170ms/step - loss: 4.7605 - mean_absolute_error: 1.7735


 53/Unknown  16s 170ms/step - loss: 4.7216 - mean_absolute_error: 1.7645


 54/Unknown  16s 169ms/step - loss: 4.6837 - mean_absolute_error: 1.7557


 55/Unknown  16s 169ms/step - loss: 4.6468 - mean_absolute_error: 1.7471


 56/Unknown  16s 169ms/step - loss: 4.6108 - mean_absolute_error: 1.7387


 57/Unknown  16s 169ms/step - loss: 4.5756 - mean_absolute_error: 1.7305


 58/Unknown  16s 169ms/step - loss: 4.5413 - mean_absolute_error: 1.7224


 59/Unknown  17s 169ms/step - loss: 4.5078 - mean_absolute_error: 1.7146


 60/Unknown  17s 169ms/step - loss: 4.4751 - mean_absolute_error: 1.7069


 61/Unknown  17s 169ms/step - loss: 4.4432 - mean_absolute_error: 1.6994


 62/Unknown  17s 169ms/step - loss: 4.4120 - mean_absolute_error: 1.6921


 63/Unknown  17s 169ms/step - loss: 4.3815 - mean_absolute_error: 1.6849


 64/Unknown  18s 169ms/step - loss: 4.3517 - mean_absolute_error: 1.6778


 65/Unknown  18s 169ms/step - loss: 4.3225 - mean_absolute_error: 1.6709


 66/Unknown  18s 169ms/step - loss: 4.2940 - mean_absolute_error: 1.6642


 67/Unknown  18s 169ms/step - loss: 4.2660 - mean_absolute_error: 1.6576


 68/Unknown  18s 169ms/step - loss: 4.2387 - mean_absolute_error: 1.6511


 69/Unknown  18s 169ms/step - loss: 4.2120 - mean_absolute_error: 1.6447


 70/Unknown  19s 169ms/step - loss: 4.1859 - mean_absolute_error: 1.6385


 71/Unknown  19s 169ms/step - loss: 4.1602 - mean_absolute_error: 1.6324


 72/Unknown  19s 169ms/step - loss: 4.1352 - mean_absolute_error: 1.6265


 73/Unknown  19s 169ms/step - loss: 4.1106 - mean_absolute_error: 1.6206


 74/Unknown  19s 169ms/step - loss: 4.0865 - mean_absolute_error: 1.6149


 75/Unknown  19s 169ms/step - loss: 4.0629 - mean_absolute_error: 1.6093


 76/Unknown  20s 169ms/step - loss: 4.0398 - mean_absolute_error: 1.6038


 77/Unknown  20s 169ms/step - loss: 4.0172 - mean_absolute_error: 1.5984


 78/Unknown  20s 169ms/step - loss: 3.9950 - mean_absolute_error: 1.5931


 79/Unknown  20s 169ms/step - loss: 3.9733 - mean_absolute_error: 1.5880


 80/Unknown  20s 169ms/step - loss: 3.9519 - mean_absolute_error: 1.5829


 81/Unknown  20s 170ms/step - loss: 3.9309 - mean_absolute_error: 1.5779


 82/Unknown  21s 170ms/step - loss: 3.9104 - mean_absolute_error: 1.5730


 83/Unknown  21s 170ms/step - loss: 3.8903 - mean_absolute_error: 1.5682


 84/Unknown  21s 170ms/step - loss: 3.8705 - mean_absolute_error: 1.5634


 85/Unknown  21s 170ms/step - loss: 3.8511 - mean_absolute_error: 1.5588


 86/Unknown  21s 170ms/step - loss: 3.8320 - mean_absolute_error: 1.5543


 87/Unknown  22s 170ms/step - loss: 3.8133 - mean_absolute_error: 1.5498


 88/Unknown  22s 170ms/step - loss: 3.7949 - mean_absolute_error: 1.5454


 89/Unknown  22s 170ms/step - loss: 3.7769 - mean_absolute_error: 1.5411


 90/Unknown  22s 171ms/step - loss: 3.7592 - mean_absolute_error: 1.5369


 91/Unknown  22s 171ms/step - loss: 3.7419 - mean_absolute_error: 1.5327


 92/Unknown  22s 171ms/step - loss: 3.7248 - mean_absolute_error: 1.5286


 93/Unknown  23s 172ms/step - loss: 3.7080 - mean_absolute_error: 1.5246


 94/Unknown  23s 172ms/step - loss: 3.6914 - mean_absolute_error: 1.5206


 95/Unknown  23s 172ms/step - loss: 3.6751 - mean_absolute_error: 1.5167


 96/Unknown  23s 173ms/step - loss: 3.6591 - mean_absolute_error: 1.5129


 97/Unknown  24s 174ms/step - loss: 3.6433 - mean_absolute_error: 1.5091


 98/Unknown  24s 175ms/step - loss: 3.6278 - mean_absolute_error: 1.5053


 99/Unknown  24s 175ms/step - loss: 3.6125 - mean_absolute_error: 1.5017


100/Unknown  24s 175ms/step - loss: 3.5975 - mean_absolute_error: 1.4980


101/Unknown  24s 175ms/step - loss: 3.5827 - mean_absolute_error: 1.4945


102/Unknown  25s 176ms/step - loss: 3.5682 - mean_absolute_error: 1.4910


103/Unknown  25s 176ms/step - loss: 3.5539 - mean_absolute_error: 1.4875


104/Unknown  25s 176ms/step - loss: 3.5397 - mean_absolute_error: 1.4841


105/Unknown  25s 176ms/step - loss: 3.5259 - mean_absolute_error: 1.4808


106/Unknown  26s 179ms/step - loss: 3.5122 - mean_absolute_error: 1.4775


107/Unknown  26s 179ms/step - loss: 3.4987 - mean_absolute_error: 1.4742


108/Unknown  26s 179ms/step - loss: 3.4854 - mean_absolute_error: 1.4710


109/Unknown  26s 179ms/step - loss: 3.4723 - mean_absolute_error: 1.4679


110/Unknown  26s 179ms/step - loss: 3.4594 - mean_absolute_error: 1.4648


111/Unknown  27s 179ms/step - loss: 3.4466 - mean_absolute_error: 1.4617


112/Unknown  27s 179ms/step - loss: 3.4341 - mean_absolute_error: 1.4587


113/Unknown  27s 179ms/step - loss: 3.4217 - mean_absolute_error: 1.4557


114/Unknown  27s 179ms/step - loss: 3.4095 - mean_absolute_error: 1.4527


115/Unknown  27s 178ms/step - loss: 3.3974 - mean_absolute_error: 1.4498


116/Unknown  27s 178ms/step - loss: 3.3856 - mean_absolute_error: 1.4470


117/Unknown  28s 178ms/step - loss: 3.3738 - mean_absolute_error: 1.4441


118/Unknown  28s 178ms/step - loss: 3.3623 - mean_absolute_error: 1.4413


119/Unknown  28s 178ms/step - loss: 3.3509 - mean_absolute_error: 1.4386


120/Unknown  28s 178ms/step - loss: 3.3396 - mean_absolute_error: 1.4358


121/Unknown  28s 178ms/step - loss: 3.3285 - mean_absolute_error: 1.4331


122/Unknown  28s 178ms/step - loss: 3.3175 - mean_absolute_error: 1.4305


123/Unknown  29s 178ms/step - loss: 3.3067 - mean_absolute_error: 1.4279


124/Unknown  29s 177ms/step - loss: 3.2960 - mean_absolute_error: 1.4253


125/Unknown  29s 177ms/step - loss: 3.2855 - mean_absolute_error: 1.4227


126/Unknown  29s 177ms/step - loss: 3.2751 - mean_absolute_error: 1.4202


127/Unknown  29s 177ms/step - loss: 3.2648 - mean_absolute_error: 1.4177


128/Unknown  29s 177ms/step - loss: 3.2546 - mean_absolute_error: 1.4152


129/Unknown  30s 177ms/step - loss: 3.2446 - mean_absolute_error: 1.4128


130/Unknown  30s 177ms/step - loss: 3.2347 - mean_absolute_error: 1.4104


131/Unknown  30s 177ms/step - loss: 3.2249 - mean_absolute_error: 1.4080


132/Unknown  30s 177ms/step - loss: 3.2152 - mean_absolute_error: 1.4056


133/Unknown  30s 177ms/step - loss: 3.2057 - mean_absolute_error: 1.4033


134/Unknown  30s 177ms/step - loss: 3.1962 - mean_absolute_error: 1.4010


135/Unknown  31s 177ms/step - loss: 3.1869 - mean_absolute_error: 1.3988


136/Unknown  31s 177ms/step - loss: 3.1777 - mean_absolute_error: 1.3965


137/Unknown  31s 177ms/step - loss: 3.1686 - mean_absolute_error: 1.3943


138/Unknown  31s 177ms/step - loss: 3.1596 - mean_absolute_error: 1.3921


139/Unknown  31s 177ms/step - loss: 3.1507 - mean_absolute_error: 1.3899


140/Unknown  32s 177ms/step - loss: 3.1419 - mean_absolute_error: 1.3878


141/Unknown  32s 177ms/step - loss: 3.1331 - mean_absolute_error: 1.3856


142/Unknown  32s 178ms/step - loss: 3.1245 - mean_absolute_error: 1.3835


143/Unknown  32s 178ms/step - loss: 3.1160 - mean_absolute_error: 1.3815


144/Unknown  32s 178ms/step - loss: 3.1076 - mean_absolute_error: 1.3794


145/Unknown  32s 178ms/step - loss: 3.0993 - mean_absolute_error: 1.3774


146/Unknown  33s 178ms/step - loss: 3.0910 - mean_absolute_error: 1.3754


147/Unknown  33s 178ms/step - loss: 3.0829 - mean_absolute_error: 1.3734


148/Unknown  33s 178ms/step - loss: 3.0748 - mean_absolute_error: 1.3714


149/Unknown  33s 178ms/step - loss: 3.0668 - mean_absolute_error: 1.3694


150/Unknown  33s 178ms/step - loss: 3.0589 - mean_absolute_error: 1.3675


151/Unknown  34s 178ms/step - loss: 3.0510 - mean_absolute_error: 1.3656


152/Unknown  34s 178ms/step - loss: 3.0433 - mean_absolute_error: 1.3637


153/Unknown  34s 178ms/step - loss: 3.0356 - mean_absolute_error: 1.3618


154/Unknown  34s 178ms/step - loss: 3.0280 - mean_absolute_error: 1.3599


155/Unknown  34s 178ms/step - loss: 3.0204 - mean_absolute_error: 1.3581


156/Unknown  34s 178ms/step - loss: 3.0130 - mean_absolute_error: 1.3562


157/Unknown  35s 178ms/step - loss: 3.0056 - mean_absolute_error: 1.3544


158/Unknown  35s 178ms/step - loss: 2.9982 - mean_absolute_error: 1.3526


159/Unknown  35s 178ms/step - loss: 2.9910 - mean_absolute_error: 1.3508


160/Unknown  35s 178ms/step - loss: 2.9838 - mean_absolute_error: 1.3491


161/Unknown  35s 178ms/step - loss: 2.9767 - mean_absolute_error: 1.3473


162/Unknown  35s 178ms/step - loss: 2.9696 - mean_absolute_error: 1.3456


163/Unknown  36s 178ms/step - loss: 2.9626 - mean_absolute_error: 1.3439


164/Unknown  36s 178ms/step - loss: 2.9557 - mean_absolute_error: 1.3422


165/Unknown  36s 178ms/step - loss: 2.9489 - mean_absolute_error: 1.3405


166/Unknown  36s 178ms/step - loss: 2.9421 - mean_absolute_error: 1.3388


167/Unknown  36s 178ms/step - loss: 2.9353 - mean_absolute_error: 1.3371


168/Unknown  37s 178ms/step - loss: 2.9287 - mean_absolute_error: 1.3355


169/Unknown  37s 178ms/step - loss: 2.9221 - mean_absolute_error: 1.3339


170/Unknown  37s 178ms/step - loss: 2.9155 - mean_absolute_error: 1.3323


171/Unknown  37s 178ms/step - loss: 2.9091 - mean_absolute_error: 1.3307


172/Unknown  37s 178ms/step - loss: 2.9026 - mean_absolute_error: 1.3291


173/Unknown  37s 178ms/step - loss: 2.8963 - mean_absolute_error: 1.3275


174/Unknown  38s 178ms/step - loss: 2.8900 - mean_absolute_error: 1.3259


175/Unknown  38s 178ms/step - loss: 2.8837 - mean_absolute_error: 1.3244


176/Unknown  38s 178ms/step - loss: 2.8775 - mean_absolute_error: 1.3228


177/Unknown  38s 178ms/step - loss: 2.8714 - mean_absolute_error: 1.3213


178/Unknown  38s 178ms/step - loss: 2.8653 - mean_absolute_error: 1.3198


179/Unknown  39s 178ms/step - loss: 2.8592 - mean_absolute_error: 1.3183


180/Unknown  39s 178ms/step - loss: 2.8533 - mean_absolute_error: 1.3168


181/Unknown  39s 178ms/step - loss: 2.8473 - mean_absolute_error: 1.3154


182/Unknown  39s 178ms/step - loss: 2.8414 - mean_absolute_error: 1.3139


183/Unknown  39s 178ms/step - loss: 2.8356 - mean_absolute_error: 1.3125


184/Unknown  39s 178ms/step - loss: 2.8298 - mean_absolute_error: 1.3110


185/Unknown  40s 178ms/step - loss: 2.8241 - mean_absolute_error: 1.3096


186/Unknown  40s 178ms/step - loss: 2.8184 - mean_absolute_error: 1.3082


187/Unknown  40s 178ms/step - loss: 2.8128 - mean_absolute_error: 1.3068


188/Unknown  40s 178ms/step - loss: 2.8072 - mean_absolute_error: 1.3054


189/Unknown  40s 178ms/step - loss: 2.8017 - mean_absolute_error: 1.3040


190/Unknown  40s 178ms/step - loss: 2.7962 - mean_absolute_error: 1.3027


191/Unknown  41s 178ms/step - loss: 2.7907 - mean_absolute_error: 1.3013


192/Unknown  41s 178ms/step - loss: 2.7853 - mean_absolute_error: 1.3000


193/Unknown  41s 178ms/step - loss: 2.7800 - mean_absolute_error: 1.2986


194/Unknown  41s 178ms/step - loss: 2.7746 - mean_absolute_error: 1.2973


195/Unknown  41s 178ms/step - loss: 2.7694 - mean_absolute_error: 1.2960


196/Unknown  42s 178ms/step - loss: 2.7641 - mean_absolute_error: 1.2947


197/Unknown  42s 178ms/step - loss: 2.7590 - mean_absolute_error: 1.2934


198/Unknown  42s 178ms/step - loss: 2.7538 - mean_absolute_error: 1.2921


199/Unknown  42s 178ms/step - loss: 2.7487 - mean_absolute_error: 1.2909


200/Unknown  42s 178ms/step - loss: 2.7436 - mean_absolute_error: 1.2896


201/Unknown  42s 178ms/step - loss: 2.7386 - mean_absolute_error: 1.2884


202/Unknown  43s 178ms/step - loss: 2.7336 - mean_absolute_error: 1.2871


203/Unknown  43s 178ms/step - loss: 2.7287 - mean_absolute_error: 1.2859


204/Unknown  43s 178ms/step - loss: 2.7238 - mean_absolute_error: 1.2846


205/Unknown  43s 178ms/step - loss: 2.7189 - mean_absolute_error: 1.2834


206/Unknown  43s 178ms/step - loss: 2.7140 - mean_absolute_error: 1.2822


207/Unknown  44s 178ms/step - loss: 2.7092 - mean_absolute_error: 1.2810


208/Unknown  44s 178ms/step - loss: 2.7045 - mean_absolute_error: 1.2798


209/Unknown  44s 178ms/step - loss: 2.6997 - mean_absolute_error: 1.2786


210/Unknown  44s 178ms/step - loss: 2.6950 - mean_absolute_error: 1.2775


211/Unknown  44s 178ms/step - loss: 2.6904 - mean_absolute_error: 1.2763


212/Unknown  44s 178ms/step - loss: 2.6857 - mean_absolute_error: 1.2751


213/Unknown  45s 178ms/step - loss: 2.6812 - mean_absolute_error: 1.2740


214/Unknown  45s 178ms/step - loss: 2.6766 - mean_absolute_error: 1.2729


215/Unknown  45s 178ms/step - loss: 2.6721 - mean_absolute_error: 1.2717


216/Unknown  45s 178ms/step - loss: 2.6676 - mean_absolute_error: 1.2706


217/Unknown  45s 178ms/step - loss: 2.6631 - mean_absolute_error: 1.2695


218/Unknown  45s 178ms/step - loss: 2.6587 - mean_absolute_error: 1.2684


219/Unknown  46s 178ms/step - loss: 2.6543 - mean_absolute_error: 1.2673


220/Unknown  46s 177ms/step - loss: 2.6499 - mean_absolute_error: 1.2662


221/Unknown  46s 177ms/step - loss: 2.6456 - mean_absolute_error: 1.2651


222/Unknown  46s 177ms/step - loss: 2.6413 - mean_absolute_error: 1.2640


223/Unknown  46s 177ms/step - loss: 2.6370 - mean_absolute_error: 1.2629


224/Unknown  46s 177ms/step - loss: 2.6328 - mean_absolute_error: 1.2619


225/Unknown  47s 177ms/step - loss: 2.6286 - mean_absolute_error: 1.2608


226/Unknown  47s 177ms/step - loss: 2.6244 - mean_absolute_error: 1.2598


227/Unknown  47s 177ms/step - loss: 2.6202 - mean_absolute_error: 1.2587


228/Unknown  47s 177ms/step - loss: 2.6161 - mean_absolute_error: 1.2577


229/Unknown  47s 177ms/step - loss: 2.6120 - mean_absolute_error: 1.2567


230/Unknown  47s 177ms/step - loss: 2.6079 - mean_absolute_error: 1.2556


231/Unknown  48s 177ms/step - loss: 2.6039 - mean_absolute_error: 1.2546


232/Unknown  48s 177ms/step - loss: 2.5999 - mean_absolute_error: 1.2536


233/Unknown  48s 177ms/step - loss: 2.5959 - mean_absolute_error: 1.2526


234/Unknown  48s 177ms/step - loss: 2.5920 - mean_absolute_error: 1.2516


235/Unknown  48s 177ms/step - loss: 2.5880 - mean_absolute_error: 1.2506


236/Unknown  48s 177ms/step - loss: 2.5841 - mean_absolute_error: 1.2496


237/Unknown  49s 177ms/step - loss: 2.5803 - mean_absolute_error: 1.2487


238/Unknown  49s 177ms/step - loss: 2.5764 - mean_absolute_error: 1.2477


239/Unknown  49s 177ms/step - loss: 2.5726 - mean_absolute_error: 1.2467


240/Unknown  49s 177ms/step - loss: 2.5688 - mean_absolute_error: 1.2458


241/Unknown  49s 177ms/step - loss: 2.5650 - mean_absolute_error: 1.2448


242/Unknown  49s 177ms/step - loss: 2.5613 - mean_absolute_error: 1.2439


243/Unknown  50s 177ms/step - loss: 2.5576 - mean_absolute_error: 1.2429


244/Unknown  50s 177ms/step - loss: 2.5539 - mean_absolute_error: 1.2420


245/Unknown  50s 177ms/step - loss: 2.5502 - mean_absolute_error: 1.2411


246/Unknown  50s 177ms/step - loss: 2.5465 - mean_absolute_error: 1.2402


247/Unknown  50s 177ms/step - loss: 2.5429 - mean_absolute_error: 1.2393


248/Unknown  51s 177ms/step - loss: 2.5393 - mean_absolute_error: 1.2383


249/Unknown  51s 177ms/step - loss: 2.5357 - mean_absolute_error: 1.2374


250/Unknown  51s 177ms/step - loss: 2.5322 - mean_absolute_error: 1.2365


251/Unknown  51s 177ms/step - loss: 2.5286 - mean_absolute_error: 1.2356


252/Unknown  51s 177ms/step - loss: 2.5251 - mean_absolute_error: 1.2347


253/Unknown  52s 178ms/step - loss: 2.5216 - mean_absolute_error: 1.2339


254/Unknown  52s 178ms/step - loss: 2.5181 - mean_absolute_error: 1.2330


255/Unknown  52s 178ms/step - loss: 2.5146 - mean_absolute_error: 1.2321


256/Unknown  52s 178ms/step - loss: 2.5112 - mean_absolute_error: 1.2312


257/Unknown  52s 178ms/step - loss: 2.5078 - mean_absolute_error: 1.2304


258/Unknown  53s 178ms/step - loss: 2.5044 - mean_absolute_error: 1.2295


259/Unknown  53s 178ms/step - loss: 2.5010 - mean_absolute_error: 1.2286


260/Unknown  53s 178ms/step - loss: 2.4976 - mean_absolute_error: 1.2278


261/Unknown  53s 179ms/step - loss: 2.4943 - mean_absolute_error: 1.2269


262/Unknown  53s 179ms/step - loss: 2.4910 - mean_absolute_error: 1.2261


263/Unknown  54s 179ms/step - loss: 2.4877 - mean_absolute_error: 1.2253


264/Unknown  54s 179ms/step - loss: 2.4844 - mean_absolute_error: 1.2244


265/Unknown  54s 179ms/step - loss: 2.4812 - mean_absolute_error: 1.2236


266/Unknown  54s 179ms/step - loss: 2.4779 - mean_absolute_error: 1.2228


267/Unknown  54s 179ms/step - loss: 2.4747 - mean_absolute_error: 1.2220


268/Unknown  55s 179ms/step - loss: 2.4715 - mean_absolute_error: 1.2211


269/Unknown  55s 179ms/step - loss: 2.4683 - mean_absolute_error: 1.2203


270/Unknown  55s 179ms/step - loss: 2.4652 - mean_absolute_error: 1.2195


271/Unknown  55s 179ms/step - loss: 2.4620 - mean_absolute_error: 1.2187


272/Unknown  55s 179ms/step - loss: 2.4589 - mean_absolute_error: 1.2179


273/Unknown  56s 179ms/step - loss: 2.4558 - mean_absolute_error: 1.2172


274/Unknown  56s 179ms/step - loss: 2.4527 - mean_absolute_error: 1.2164


275/Unknown  56s 179ms/step - loss: 2.4496 - mean_absolute_error: 1.2156


276/Unknown  56s 179ms/step - loss: 2.4465 - mean_absolute_error: 1.2148


277/Unknown  56s 179ms/step - loss: 2.4435 - mean_absolute_error: 1.2140


278/Unknown  57s 179ms/step - loss: 2.4405 - mean_absolute_error: 1.2133


279/Unknown  57s 179ms/step - loss: 2.4375 - mean_absolute_error: 1.2125


280/Unknown  57s 179ms/step - loss: 2.4345 - mean_absolute_error: 1.2117


281/Unknown  57s 179ms/step - loss: 2.4315 - mean_absolute_error: 1.2110


282/Unknown  57s 179ms/step - loss: 2.4286 - mean_absolute_error: 1.2102


283/Unknown  57s 179ms/step - loss: 2.4256 - mean_absolute_error: 1.2095


284/Unknown  58s 179ms/step - loss: 2.4227 - mean_absolute_error: 1.2087


285/Unknown  58s 179ms/step - loss: 2.4198 - mean_absolute_error: 1.2080


286/Unknown  58s 179ms/step - loss: 2.4169 - mean_absolute_error: 1.2072


287/Unknown  58s 179ms/step - loss: 2.4140 - mean_absolute_error: 1.2065


288/Unknown  58s 179ms/step - loss: 2.4111 - mean_absolute_error: 1.2058


289/Unknown  59s 179ms/step - loss: 2.4083 - mean_absolute_error: 1.2050


290/Unknown  59s 179ms/step - loss: 2.4054 - mean_absolute_error: 1.2043


291/Unknown  59s 179ms/step - loss: 2.4026 - mean_absolute_error: 1.2036


292/Unknown  59s 179ms/step - loss: 2.3998 - mean_absolute_error: 1.2029


293/Unknown  59s 179ms/step - loss: 2.3970 - mean_absolute_error: 1.2021


294/Unknown  59s 179ms/step - loss: 2.3943 - mean_absolute_error: 1.2014


295/Unknown  60s 179ms/step - loss: 2.3915 - mean_absolute_error: 1.2007


296/Unknown  60s 179ms/step - loss: 2.3887 - mean_absolute_error: 1.2000


297/Unknown  60s 179ms/step - loss: 2.3860 - mean_absolute_error: 1.1993


298/Unknown  60s 179ms/step - loss: 2.3833 - mean_absolute_error: 1.1986


299/Unknown  60s 179ms/step - loss: 2.3806 - mean_absolute_error: 1.1979


300/Unknown  61s 179ms/step - loss: 2.3779 - mean_absolute_error: 1.1972


301/Unknown  61s 179ms/step - loss: 2.3752 - mean_absolute_error: 1.1966


302/Unknown  61s 179ms/step - loss: 2.3726 - mean_absolute_error: 1.1959


303/Unknown  61s 179ms/step - loss: 2.3699 - mean_absolute_error: 1.1952


304/Unknown  61s 180ms/step - loss: 2.3673 - mean_absolute_error: 1.1945


305/Unknown  61s 180ms/step - loss: 2.3647 - mean_absolute_error: 1.1938


306/Unknown  62s 180ms/step - loss: 2.3621 - mean_absolute_error: 1.1932


307/Unknown  62s 180ms/step - loss: 2.3595 - mean_absolute_error: 1.1925


308/Unknown  62s 180ms/step - loss: 2.3569 - mean_absolute_error: 1.1918


309/Unknown  62s 180ms/step - loss: 2.3543 - mean_absolute_error: 1.1912


310/Unknown  62s 180ms/step - loss: 2.3518 - mean_absolute_error: 1.1905


311/Unknown  63s 180ms/step - loss: 2.3492 - mean_absolute_error: 1.1899


312/Unknown  63s 180ms/step - loss: 2.3467 - mean_absolute_error: 1.1892


313/Unknown  63s 180ms/step - loss: 2.3442 - mean_absolute_error: 1.1886


314/Unknown  63s 180ms/step - loss: 2.3417 - mean_absolute_error: 1.1879


315/Unknown  63s 180ms/step - loss: 2.3392 - mean_absolute_error: 1.1873


316/Unknown  64s 180ms/step - loss: 2.3367 - mean_absolute_error: 1.1867


317/Unknown  64s 180ms/step - loss: 2.3343 - mean_absolute_error: 1.1860


318/Unknown  64s 180ms/step - loss: 2.3318 - mean_absolute_error: 1.1854


319/Unknown  64s 180ms/step - loss: 2.3294 - mean_absolute_error: 1.1848


320/Unknown  64s 180ms/step - loss: 2.3270 - mean_absolute_error: 1.1841


321/Unknown  65s 180ms/step - loss: 2.3245 - mean_absolute_error: 1.1835


322/Unknown  65s 180ms/step - loss: 2.3221 - mean_absolute_error: 1.1829


323/Unknown  65s 180ms/step - loss: 2.3197 - mean_absolute_error: 1.1823


324/Unknown  65s 180ms/step - loss: 2.3174 - mean_absolute_error: 1.1817


325/Unknown  65s 180ms/step - loss: 2.3150 - mean_absolute_error: 1.1811


326/Unknown  66s 180ms/step - loss: 2.3127 - mean_absolute_error: 1.1805


327/Unknown  66s 180ms/step - loss: 2.3103 - mean_absolute_error: 1.1799


328/Unknown  66s 180ms/step - loss: 2.3080 - mean_absolute_error: 1.1793


329/Unknown  66s 181ms/step - loss: 2.3057 - mean_absolute_error: 1.1787


330/Unknown  66s 181ms/step - loss: 2.3034 - mean_absolute_error: 1.1781


331/Unknown  66s 181ms/step - loss: 2.3011 - mean_absolute_error: 1.1775


332/Unknown  67s 181ms/step - loss: 2.2988 - mean_absolute_error: 1.1769


333/Unknown  67s 181ms/step - loss: 2.2965 - mean_absolute_error: 1.1763


334/Unknown  67s 181ms/step - loss: 2.2943 - mean_absolute_error: 1.1757


335/Unknown  67s 181ms/step - loss: 2.2920 - mean_absolute_error: 1.1751


336/Unknown  67s 181ms/step - loss: 2.2898 - mean_absolute_error: 1.1745


337/Unknown  68s 181ms/step - loss: 2.2875 - mean_absolute_error: 1.1740


338/Unknown  68s 181ms/step - loss: 2.2853 - mean_absolute_error: 1.1734


339/Unknown  68s 181ms/step - loss: 2.2831 - mean_absolute_error: 1.1728


340/Unknown  68s 181ms/step - loss: 2.2809 - mean_absolute_error: 1.1723


341/Unknown  68s 181ms/step - loss: 2.2787 - mean_absolute_error: 1.1717


342/Unknown  69s 181ms/step - loss: 2.2765 - mean_absolute_error: 1.1711


343/Unknown  69s 181ms/step - loss: 2.2744 - mean_absolute_error: 1.1706


344/Unknown  69s 181ms/step - loss: 2.2722 - mean_absolute_error: 1.1700


345/Unknown  69s 181ms/step - loss: 2.2701 - mean_absolute_error: 1.1695


346/Unknown  69s 181ms/step - loss: 2.2679 - mean_absolute_error: 1.1689


347/Unknown  70s 181ms/step - loss: 2.2658 - mean_absolute_error: 1.1684


348/Unknown  70s 181ms/step - loss: 2.2637 - mean_absolute_error: 1.1678


349/Unknown  70s 181ms/step - loss: 2.2616 - mean_absolute_error: 1.1673


350/Unknown  70s 181ms/step - loss: 2.2595 - mean_absolute_error: 1.1667


351/Unknown  70s 181ms/step - loss: 2.2574 - mean_absolute_error: 1.1662


352/Unknown  71s 181ms/step - loss: 2.2553 - mean_absolute_error: 1.1657


353/Unknown  71s 181ms/step - loss: 2.2533 - mean_absolute_error: 1.1651


354/Unknown  71s 181ms/step - loss: 2.2512 - mean_absolute_error: 1.1646


355/Unknown  71s 181ms/step - loss: 2.2492 - mean_absolute_error: 1.1641


356/Unknown  71s 181ms/step - loss: 2.2471 - mean_absolute_error: 1.1635


357/Unknown  71s 181ms/step - loss: 2.2451 - mean_absolute_error: 1.1630


358/Unknown  72s 182ms/step - loss: 2.2431 - mean_absolute_error: 1.1625


359/Unknown  72s 182ms/step - loss: 2.2411 - mean_absolute_error: 1.1620


360/Unknown  72s 182ms/step - loss: 2.2391 - mean_absolute_error: 1.1614


361/Unknown  72s 182ms/step - loss: 2.2371 - mean_absolute_error: 1.1609


362/Unknown  72s 182ms/step - loss: 2.2351 - mean_absolute_error: 1.1604


363/Unknown  73s 182ms/step - loss: 2.2331 - mean_absolute_error: 1.1599


364/Unknown  73s 182ms/step - loss: 2.2311 - mean_absolute_error: 1.1594


365/Unknown  73s 182ms/step - loss: 2.2292 - mean_absolute_error: 1.1589


366/Unknown  73s 182ms/step - loss: 2.2272 - mean_absolute_error: 1.1584


367/Unknown  74s 182ms/step - loss: 2.2253 - mean_absolute_error: 1.1579


368/Unknown  74s 182ms/step - loss: 2.2234 - mean_absolute_error: 1.1574


369/Unknown  74s 182ms/step - loss: 2.2214 - mean_absolute_error: 1.1569


370/Unknown  74s 183ms/step - loss: 2.2195 - mean_absolute_error: 1.1564


371/Unknown  74s 183ms/step - loss: 2.2176 - mean_absolute_error: 1.1559


372/Unknown  75s 183ms/step - loss: 2.2157 - mean_absolute_error: 1.1554


373/Unknown  75s 183ms/step - loss: 2.2138 - mean_absolute_error: 1.1549


374/Unknown  75s 183ms/step - loss: 2.2119 - mean_absolute_error: 1.1544


375/Unknown  75s 183ms/step - loss: 2.2101 - mean_absolute_error: 1.1539


376/Unknown  75s 183ms/step - loss: 2.2082 - mean_absolute_error: 1.1535


377/Unknown  76s 183ms/step - loss: 2.2064 - mean_absolute_error: 1.1530


378/Unknown  76s 183ms/step - loss: 2.2045 - mean_absolute_error: 1.1525


379/Unknown  76s 183ms/step - loss: 2.2027 - mean_absolute_error: 1.1520


380/Unknown  76s 183ms/step - loss: 2.2008 - mean_absolute_error: 1.1515


381/Unknown  76s 183ms/step - loss: 2.1990 - mean_absolute_error: 1.1511


382/Unknown  77s 183ms/step - loss: 2.1972 - mean_absolute_error: 1.1506


383/Unknown  77s 183ms/step - loss: 2.1954 - mean_absolute_error: 1.1501


384/Unknown  77s 183ms/step - loss: 2.1936 - mean_absolute_error: 1.1497


385/Unknown  77s 183ms/step - loss: 2.1918 - mean_absolute_error: 1.1492


386/Unknown  77s 183ms/step - loss: 2.1900 - mean_absolute_error: 1.1487


387/Unknown  78s 183ms/step - loss: 2.1882 - mean_absolute_error: 1.1483


388/Unknown  78s 183ms/step - loss: 2.1865 - mean_absolute_error: 1.1478


389/Unknown  78s 183ms/step - loss: 2.1847 - mean_absolute_error: 1.1474


390/Unknown  78s 183ms/step - loss: 2.1829 - mean_absolute_error: 1.1469


391/Unknown  78s 183ms/step - loss: 2.1812 - mean_absolute_error: 1.1464


392/Unknown  79s 183ms/step - loss: 2.1794 - mean_absolute_error: 1.1460


393/Unknown  79s 183ms/step - loss: 2.1777 - mean_absolute_error: 1.1455


394/Unknown  79s 183ms/step - loss: 2.1760 - mean_absolute_error: 1.1451


395/Unknown  79s 183ms/step - loss: 2.1743 - mean_absolute_error: 1.1446


396/Unknown  79s 183ms/step - loss: 2.1725 - mean_absolute_error: 1.1442


397/Unknown  79s 183ms/step - loss: 2.1708 - mean_absolute_error: 1.1438


398/Unknown  80s 183ms/step - loss: 2.1691 - mean_absolute_error: 1.1433


399/Unknown  80s 183ms/step - loss: 2.1674 - mean_absolute_error: 1.1429


400/Unknown  80s 183ms/step - loss: 2.1657 - mean_absolute_error: 1.1424


401/Unknown  80s 184ms/step - loss: 2.1641 - mean_absolute_error: 1.1420


402/Unknown  80s 184ms/step - loss: 2.1624 - mean_absolute_error: 1.1416


403/Unknown  81s 184ms/step - loss: 2.1607 - mean_absolute_error: 1.1411


404/Unknown  81s 184ms/step - loss: 2.1590 - mean_absolute_error: 1.1407


405/Unknown  81s 184ms/step - loss: 2.1574 - mean_absolute_error: 1.1403


406/Unknown  81s 184ms/step - loss: 2.1557 - mean_absolute_error: 1.1398


407/Unknown  81s 184ms/step - loss: 2.1541 - mean_absolute_error: 1.1394


408/Unknown  82s 184ms/step - loss: 2.1525 - mean_absolute_error: 1.1390


409/Unknown  82s 184ms/step - loss: 2.1508 - mean_absolute_error: 1.1385


410/Unknown  82s 184ms/step - loss: 2.1492 - mean_absolute_error: 1.1381


411/Unknown  82s 184ms/step - loss: 2.1476 - mean_absolute_error: 1.1377


412/Unknown  82s 184ms/step - loss: 2.1460 - mean_absolute_error: 1.1373


413/Unknown  83s 184ms/step - loss: 2.1444 - mean_absolute_error: 1.1369


414/Unknown  83s 184ms/step - loss: 2.1428 - mean_absolute_error: 1.1364


415/Unknown  83s 184ms/step - loss: 2.1412 - mean_absolute_error: 1.1360


416/Unknown  83s 184ms/step - loss: 2.1396 - mean_absolute_error: 1.1356


417/Unknown  83s 184ms/step - loss: 2.1380 - mean_absolute_error: 1.1352


418/Unknown  84s 184ms/step - loss: 2.1364 - mean_absolute_error: 1.1348


419/Unknown  84s 184ms/step - loss: 2.1349 - mean_absolute_error: 1.1344


420/Unknown  84s 184ms/step - loss: 2.1333 - mean_absolute_error: 1.1340


421/Unknown  84s 184ms/step - loss: 2.1317 - mean_absolute_error: 1.1336


422/Unknown  84s 184ms/step - loss: 2.1302 - mean_absolute_error: 1.1332


423/Unknown  85s 184ms/step - loss: 2.1286 - mean_absolute_error: 1.1328


424/Unknown  85s 184ms/step - loss: 2.1271 - mean_absolute_error: 1.1324


425/Unknown  85s 184ms/step - loss: 2.1256 - mean_absolute_error: 1.1320


426/Unknown  85s 184ms/step - loss: 2.1240 - mean_absolute_error: 1.1316


427/Unknown  85s 184ms/step - loss: 2.1225 - mean_absolute_error: 1.1312


428/Unknown  86s 184ms/step - loss: 2.1210 - mean_absolute_error: 1.1308


429/Unknown  86s 184ms/step - loss: 2.1195 - mean_absolute_error: 1.1304


430/Unknown  86s 184ms/step - loss: 2.1180 - mean_absolute_error: 1.1300


431/Unknown  86s 184ms/step - loss: 2.1165 - mean_absolute_error: 1.1296


432/Unknown  86s 185ms/step - loss: 2.1150 - mean_absolute_error: 1.1292


433/Unknown  87s 185ms/step - loss: 2.1135 - mean_absolute_error: 1.1288


434/Unknown  87s 185ms/step - loss: 2.1120 - mean_absolute_error: 1.1284


435/Unknown  87s 185ms/step - loss: 2.1105 - mean_absolute_error: 1.1280


436/Unknown  87s 185ms/step - loss: 2.1090 - mean_absolute_error: 1.1276


437/Unknown  87s 185ms/step - loss: 2.1076 - mean_absolute_error: 1.1272


438/Unknown  88s 185ms/step - loss: 2.1061 - mean_absolute_error: 1.1269


439/Unknown  88s 185ms/step - loss: 2.1046 - mean_absolute_error: 1.1265


440/Unknown  88s 185ms/step - loss: 2.1032 - mean_absolute_error: 1.1261


441/Unknown  89s 186ms/step - loss: 2.1017 - mean_absolute_error: 1.1257


442/Unknown  89s 186ms/step - loss: 2.1003 - mean_absolute_error: 1.1253


443/Unknown  89s 186ms/step - loss: 2.0988 - mean_absolute_error: 1.1250


444/Unknown  89s 186ms/step - loss: 2.0974 - mean_absolute_error: 1.1246


445/Unknown  89s 186ms/step - loss: 2.0960 - mean_absolute_error: 1.1242


446/Unknown  90s 186ms/step - loss: 2.0945 - mean_absolute_error: 1.1238


447/Unknown  90s 186ms/step - loss: 2.0931 - mean_absolute_error: 1.1235


448/Unknown  90s 186ms/step - loss: 2.0917 - mean_absolute_error: 1.1231


449/Unknown  90s 186ms/step - loss: 2.0903 - mean_absolute_error: 1.1227


450/Unknown  90s 186ms/step - loss: 2.0889 - mean_absolute_error: 1.1223


451/Unknown  91s 186ms/step - loss: 2.0875 - mean_absolute_error: 1.1220


452/Unknown  91s 186ms/step - loss: 2.0861 - mean_absolute_error: 1.1216


453/Unknown  91s 186ms/step - loss: 2.0847 - mean_absolute_error: 1.1212


454/Unknown  91s 186ms/step - loss: 2.0833 - mean_absolute_error: 1.1209


455/Unknown  91s 186ms/step - loss: 2.0820 - mean_absolute_error: 1.1205


456/Unknown  92s 186ms/step - loss: 2.0806 - mean_absolute_error: 1.1202


457/Unknown  92s 186ms/step - loss: 2.0792 - mean_absolute_error: 1.1198


458/Unknown  92s 186ms/step - loss: 2.0778 - mean_absolute_error: 1.1194


459/Unknown  92s 186ms/step - loss: 2.0765 - mean_absolute_error: 1.1191


460/Unknown  92s 186ms/step - loss: 2.0751 - mean_absolute_error: 1.1187


461/Unknown  93s 186ms/step - loss: 2.0738 - mean_absolute_error: 1.1184


462/Unknown  93s 186ms/step - loss: 2.0724 - mean_absolute_error: 1.1180


463/Unknown  93s 187ms/step - loss: 2.0711 - mean_absolute_error: 1.1177


464/Unknown  93s 187ms/step - loss: 2.0697 - mean_absolute_error: 1.1173


465/Unknown  94s 187ms/step - loss: 2.0684 - mean_absolute_error: 1.1169


466/Unknown  94s 187ms/step - loss: 2.0671 - mean_absolute_error: 1.1166


467/Unknown  94s 187ms/step - loss: 2.0658 - mean_absolute_error: 1.1163


468/Unknown  94s 187ms/step - loss: 2.0644 - mean_absolute_error: 1.1159


469/Unknown  94s 187ms/step - loss: 2.0631 - mean_absolute_error: 1.1156


470/Unknown  95s 187ms/step - loss: 2.0618 - mean_absolute_error: 1.1152


471/Unknown  95s 187ms/step - loss: 2.0605 - mean_absolute_error: 1.1149


472/Unknown  95s 187ms/step - loss: 2.0592 - mean_absolute_error: 1.1145


473/Unknown  95s 187ms/step - loss: 2.0579 - mean_absolute_error: 1.1142


474/Unknown  96s 188ms/step - loss: 2.0566 - mean_absolute_error: 1.1138


475/Unknown  96s 188ms/step - loss: 2.0554 - mean_absolute_error: 1.1135


476/Unknown  96s 188ms/step - loss: 2.0541 - mean_absolute_error: 1.1132


477/Unknown  96s 188ms/step - loss: 2.0528 - mean_absolute_error: 1.1128


478/Unknown  96s 188ms/step - loss: 2.0515 - mean_absolute_error: 1.1125


479/Unknown  97s 188ms/step - loss: 2.0502 - mean_absolute_error: 1.1122


480/Unknown  97s 188ms/step - loss: 2.0490 - mean_absolute_error: 1.1118


481/Unknown  97s 188ms/step - loss: 2.0477 - mean_absolute_error: 1.1115


482/Unknown  97s 188ms/step - loss: 2.0465 - mean_absolute_error: 1.1112


483/Unknown  97s 188ms/step - loss: 2.0452 - mean_absolute_error: 1.1108


484/Unknown  98s 188ms/step - loss: 2.0440 - mean_absolute_error: 1.1105


485/Unknown  98s 188ms/step - loss: 2.0427 - mean_absolute_error: 1.1102


486/Unknown  98s 188ms/step - loss: 2.0415 - mean_absolute_error: 1.1098


487/Unknown  98s 188ms/step - loss: 2.0402 - mean_absolute_error: 1.1095


488/Unknown  99s 188ms/step - loss: 2.0390 - mean_absolute_error: 1.1092


489/Unknown  99s 188ms/step - loss: 2.0378 - mean_absolute_error: 1.1089


490/Unknown  99s 188ms/step - loss: 2.0365 - mean_absolute_error: 1.1085


491/Unknown  99s 188ms/step - loss: 2.0353 - mean_absolute_error: 1.1082


492/Unknown  99s 188ms/step - loss: 2.0341 - mean_absolute_error: 1.1079


493/Unknown  100s 188ms/step - loss: 2.0329 - mean_absolute_error: 1.1076


494/Unknown  100s 188ms/step - loss: 2.0317 - mean_absolute_error: 1.1073


495/Unknown  100s 188ms/step - loss: 2.0305 - mean_absolute_error: 1.1069


496/Unknown  100s 188ms/step - loss: 2.0293 - mean_absolute_error: 1.1066


497/Unknown  100s 188ms/step - loss: 2.0281 - mean_absolute_error: 1.1063


498/Unknown  101s 189ms/step - loss: 2.0269 - mean_absolute_error: 1.1060


499/Unknown  101s 189ms/step - loss: 2.0257 - mean_absolute_error: 1.1057


500/Unknown  101s 189ms/step - loss: 2.0245 - mean_absolute_error: 1.1054


501/Unknown  101s 189ms/step - loss: 2.0233 - mean_absolute_error: 1.1051


502/Unknown  101s 189ms/step - loss: 2.0222 - mean_absolute_error: 1.1047


503/Unknown  102s 189ms/step - loss: 2.0210 - mean_absolute_error: 1.1044


504/Unknown  102s 189ms/step - loss: 2.0198 - mean_absolute_error: 1.1041


505/Unknown  102s 189ms/step - loss: 2.0187 - mean_absolute_error: 1.1038


506/Unknown  102s 189ms/step - loss: 2.0175 - mean_absolute_error: 1.1035


507/Unknown  102s 189ms/step - loss: 2.0163 - mean_absolute_error: 1.1032


508/Unknown  103s 189ms/step - loss: 2.0152 - mean_absolute_error: 1.1029


509/Unknown  103s 189ms/step - loss: 2.0140 - mean_absolute_error: 1.1026


510/Unknown  103s 189ms/step - loss: 2.0129 - mean_absolute_error: 1.1023


511/Unknown  103s 189ms/step - loss: 2.0117 - mean_absolute_error: 1.1020


512/Unknown  104s 189ms/step - loss: 2.0106 - mean_absolute_error: 1.1017


513/Unknown  104s 189ms/step - loss: 2.0095 - mean_absolute_error: 1.1014


514/Unknown  104s 189ms/step - loss: 2.0083 - mean_absolute_error: 1.1011


515/Unknown  104s 189ms/step - loss: 2.0072 - mean_absolute_error: 1.1008


516/Unknown  104s 189ms/step - loss: 2.0061 - mean_absolute_error: 1.1005


517/Unknown  105s 189ms/step - loss: 2.0050 - mean_absolute_error: 1.1002


518/Unknown  105s 189ms/step - loss: 2.0038 - mean_absolute_error: 1.0999


519/Unknown  105s 189ms/step - loss: 2.0027 - mean_absolute_error: 1.0996


520/Unknown  105s 189ms/step - loss: 2.0016 - mean_absolute_error: 1.0993


521/Unknown  105s 190ms/step - loss: 2.0005 - mean_absolute_error: 1.0990


522/Unknown  106s 190ms/step - loss: 1.9994 - mean_absolute_error: 1.0987


523/Unknown  106s 190ms/step - loss: 1.9983 - mean_absolute_error: 1.0984


524/Unknown  106s 190ms/step - loss: 1.9972 - mean_absolute_error: 1.0981


525/Unknown  106s 190ms/step - loss: 1.9961 - mean_absolute_error: 1.0978


526/Unknown  106s 190ms/step - loss: 1.9950 - mean_absolute_error: 1.0975


527/Unknown  107s 190ms/step - loss: 1.9939 - mean_absolute_error: 1.0973


528/Unknown  107s 190ms/step - loss: 1.9928 - mean_absolute_error: 1.0970


529/Unknown  107s 190ms/step - loss: 1.9918 - mean_absolute_error: 1.0967


530/Unknown  107s 190ms/step - loss: 1.9907 - mean_absolute_error: 1.0964


531/Unknown  108s 190ms/step - loss: 1.9896 - mean_absolute_error: 1.0961


532/Unknown  108s 190ms/step - loss: 1.9885 - mean_absolute_error: 1.0958


533/Unknown  108s 190ms/step - loss: 1.9875 - mean_absolute_error: 1.0955


534/Unknown  108s 190ms/step - loss: 1.9864 - mean_absolute_error: 1.0953


535/Unknown  109s 190ms/step - loss: 1.9853 - mean_absolute_error: 1.0950


536/Unknown  109s 191ms/step - loss: 1.9843 - mean_absolute_error: 1.0947


537/Unknown  109s 191ms/step - loss: 1.9832 - mean_absolute_error: 1.0944


538/Unknown  109s 191ms/step - loss: 1.9822 - mean_absolute_error: 1.0941


539/Unknown  110s 191ms/step - loss: 1.9811 - mean_absolute_error: 1.0939


540/Unknown  110s 191ms/step - loss: 1.9801 - mean_absolute_error: 1.0936


541/Unknown  110s 191ms/step - loss: 1.9791 - mean_absolute_error: 1.0933


542/Unknown  110s 191ms/step - loss: 1.9780 - mean_absolute_error: 1.0930


543/Unknown  110s 191ms/step - loss: 1.9770 - mean_absolute_error: 1.0928


544/Unknown  111s 191ms/step - loss: 1.9759 - mean_absolute_error: 1.0925


545/Unknown  111s 191ms/step - loss: 1.9749 - mean_absolute_error: 1.0922


546/Unknown  111s 191ms/step - loss: 1.9739 - mean_absolute_error: 1.0919


547/Unknown  111s 191ms/step - loss: 1.9729 - mean_absolute_error: 1.0917


548/Unknown  112s 191ms/step - loss: 1.9718 - mean_absolute_error: 1.0914


549/Unknown  112s 191ms/step - loss: 1.9708 - mean_absolute_error: 1.0911


550/Unknown  112s 191ms/step - loss: 1.9698 - mean_absolute_error: 1.0909


551/Unknown  112s 191ms/step - loss: 1.9688 - mean_absolute_error: 1.0906


552/Unknown  112s 191ms/step - loss: 1.9678 - mean_absolute_error: 1.0903


553/Unknown  113s 192ms/step - loss: 1.9668 - mean_absolute_error: 1.0900


554/Unknown  113s 192ms/step - loss: 1.9658 - mean_absolute_error: 1.0898


555/Unknown  113s 192ms/step - loss: 1.9648 - mean_absolute_error: 1.0895


556/Unknown  113s 192ms/step - loss: 1.9638 - mean_absolute_error: 1.0892


557/Unknown  114s 192ms/step - loss: 1.9628 - mean_absolute_error: 1.0890


558/Unknown  114s 192ms/step - loss: 1.9618 - mean_absolute_error: 1.0887


559/Unknown  114s 192ms/step - loss: 1.9608 - mean_absolute_error: 1.0885


560/Unknown  114s 192ms/step - loss: 1.9598 - mean_absolute_error: 1.0882


561/Unknown  114s 192ms/step - loss: 1.9588 - mean_absolute_error: 1.0879


562/Unknown  115s 192ms/step - loss: 1.9579 - mean_absolute_error: 1.0877


563/Unknown  115s 192ms/step - loss: 1.9569 - mean_absolute_error: 1.0874


564/Unknown  115s 192ms/step - loss: 1.9559 - mean_absolute_error: 1.0872


565/Unknown  115s 192ms/step - loss: 1.9549 - mean_absolute_error: 1.0869


566/Unknown  116s 192ms/step - loss: 1.9540 - mean_absolute_error: 1.0866


567/Unknown  116s 192ms/step - loss: 1.9530 - mean_absolute_error: 1.0864


568/Unknown  116s 193ms/step - loss: 1.9520 - mean_absolute_error: 1.0861


569/Unknown  116s 193ms/step - loss: 1.9511 - mean_absolute_error: 1.0859


570/Unknown  117s 193ms/step - loss: 1.9501 - mean_absolute_error: 1.0856


571/Unknown  117s 193ms/step - loss: 1.9492 - mean_absolute_error: 1.0854


572/Unknown  117s 193ms/step - loss: 1.9482 - mean_absolute_error: 1.0851


573/Unknown  117s 193ms/step - loss: 1.9473 - mean_absolute_error: 1.0848


574/Unknown  118s 193ms/step - loss: 1.9463 - mean_absolute_error: 1.0846


575/Unknown  118s 193ms/step - loss: 1.9454 - mean_absolute_error: 1.0843


576/Unknown  118s 193ms/step - loss: 1.9444 - mean_absolute_error: 1.0841


577/Unknown  118s 193ms/step - loss: 1.9435 - mean_absolute_error: 1.0838


578/Unknown  118s 193ms/step - loss: 1.9426 - mean_absolute_error: 1.0836


579/Unknown  119s 193ms/step - loss: 1.9416 - mean_absolute_error: 1.0833


580/Unknown  119s 193ms/step - loss: 1.9407 - mean_absolute_error: 1.0831


581/Unknown  119s 193ms/step - loss: 1.9398 - mean_absolute_error: 1.0828


582/Unknown  119s 193ms/step - loss: 1.9388 - mean_absolute_error: 1.0826


583/Unknown  119s 193ms/step - loss: 1.9379 - mean_absolute_error: 1.0824


584/Unknown  120s 193ms/step - loss: 1.9370 - mean_absolute_error: 1.0821


585/Unknown  120s 193ms/step - loss: 1.9361 - mean_absolute_error: 1.0819


586/Unknown  120s 193ms/step - loss: 1.9352 - mean_absolute_error: 1.0816


587/Unknown  120s 193ms/step - loss: 1.9342 - mean_absolute_error: 1.0814


588/Unknown  120s 193ms/step - loss: 1.9333 - mean_absolute_error: 1.0811


589/Unknown  121s 193ms/step - loss: 1.9324 - mean_absolute_error: 1.0809


590/Unknown  121s 193ms/step - loss: 1.9315 - mean_absolute_error: 1.0806


591/Unknown  121s 193ms/step - loss: 1.9306 - mean_absolute_error: 1.0804


592/Unknown  121s 193ms/step - loss: 1.9297 - mean_absolute_error: 1.0802


593/Unknown  121s 194ms/step - loss: 1.9288 - mean_absolute_error: 1.0799


594/Unknown  122s 194ms/step - loss: 1.9279 - mean_absolute_error: 1.0797


595/Unknown  122s 194ms/step - loss: 1.9270 - mean_absolute_error: 1.0794


596/Unknown  122s 194ms/step - loss: 1.9261 - mean_absolute_error: 1.0792


597/Unknown  122s 194ms/step - loss: 1.9252 - mean_absolute_error: 1.0790


598/Unknown  122s 194ms/step - loss: 1.9243 - mean_absolute_error: 1.0787


599/Unknown  123s 194ms/step - loss: 1.9234 - mean_absolute_error: 1.0785


600/Unknown  123s 194ms/step - loss: 1.9226 - mean_absolute_error: 1.0782


601/Unknown  123s 194ms/step - loss: 1.9217 - mean_absolute_error: 1.0780


602/Unknown  124s 194ms/step - loss: 1.9208 - mean_absolute_error: 1.0778


603/Unknown  124s 194ms/step - loss: 1.9199 - mean_absolute_error: 1.0775


604/Unknown  124s 194ms/step - loss: 1.9190 - mean_absolute_error: 1.0773


605/Unknown  124s 194ms/step - loss: 1.9182 - mean_absolute_error: 1.0771


606/Unknown  125s 195ms/step - loss: 1.9173 - mean_absolute_error: 1.0768


607/Unknown  125s 195ms/step - loss: 1.9164 - mean_absolute_error: 1.0766


608/Unknown  125s 195ms/step - loss: 1.9156 - mean_absolute_error: 1.0764


609/Unknown  125s 195ms/step - loss: 1.9147 - mean_absolute_error: 1.0761


610/Unknown  125s 195ms/step - loss: 1.9139 - mean_absolute_error: 1.0759


611/Unknown  126s 195ms/step - loss: 1.9130 - mean_absolute_error: 1.0757


612/Unknown  126s 195ms/step - loss: 1.9121 - mean_absolute_error: 1.0754


613/Unknown  126s 195ms/step - loss: 1.9113 - mean_absolute_error: 1.0752


614/Unknown  126s 195ms/step - loss: 1.9104 - mean_absolute_error: 1.0750


615/Unknown  126s 195ms/step - loss: 1.9096 - mean_absolute_error: 1.0748


616/Unknown  127s 195ms/step - loss: 1.9087 - mean_absolute_error: 1.0745


617/Unknown  127s 195ms/step - loss: 1.9079 - mean_absolute_error: 1.0743


618/Unknown  127s 195ms/step - loss: 1.9070 - mean_absolute_error: 1.0741


619/Unknown  127s 195ms/step - loss: 1.9062 - mean_absolute_error: 1.0739


620/Unknown  128s 195ms/step - loss: 1.9054 - mean_absolute_error: 1.0736


621/Unknown  128s 195ms/step - loss: 1.9045 - mean_absolute_error: 1.0734


622/Unknown  128s 195ms/step - loss: 1.9037 - mean_absolute_error: 1.0732


623/Unknown  128s 195ms/step - loss: 1.9029 - mean_absolute_error: 1.0730


624/Unknown  128s 195ms/step - loss: 1.9020 - mean_absolute_error: 1.0727


625/Unknown  129s 195ms/step - loss: 1.9012 - mean_absolute_error: 1.0725


626/Unknown  129s 195ms/step - loss: 1.9004 - mean_absolute_error: 1.0723


627/Unknown  129s 195ms/step - loss: 1.8996 - mean_absolute_error: 1.0721


628/Unknown  129s 195ms/step - loss: 1.8987 - mean_absolute_error: 1.0718


629/Unknown  129s 195ms/step - loss: 1.8979 - mean_absolute_error: 1.0716


630/Unknown  130s 195ms/step - loss: 1.8971 - mean_absolute_error: 1.0714


631/Unknown  130s 195ms/step - loss: 1.8963 - mean_absolute_error: 1.0712


632/Unknown  130s 195ms/step - loss: 1.8955 - mean_absolute_error: 1.0710


633/Unknown  130s 195ms/step - loss: 1.8947 - mean_absolute_error: 1.0708


634/Unknown  131s 195ms/step - loss: 1.8939 - mean_absolute_error: 1.0705


635/Unknown  131s 195ms/step - loss: 1.8931 - mean_absolute_error: 1.0703


636/Unknown  131s 195ms/step - loss: 1.8923 - mean_absolute_error: 1.0701


637/Unknown  131s 195ms/step - loss: 1.8915 - mean_absolute_error: 1.0699


638/Unknown  131s 196ms/step - loss: 1.8907 - mean_absolute_error: 1.0697


639/Unknown  132s 196ms/step - loss: 1.8899 - mean_absolute_error: 1.0695


640/Unknown  132s 196ms/step - loss: 1.8891 - mean_absolute_error: 1.0692


641/Unknown  132s 196ms/step - loss: 1.8883 - mean_absolute_error: 1.0690


642/Unknown  132s 196ms/step - loss: 1.8875 - mean_absolute_error: 1.0688


643/Unknown  133s 196ms/step - loss: 1.8867 - mean_absolute_error: 1.0686


644/Unknown  133s 196ms/step - loss: 1.8859 - mean_absolute_error: 1.0684


645/Unknown  133s 196ms/step - loss: 1.8851 - mean_absolute_error: 1.0682


646/Unknown  133s 196ms/step - loss: 1.8843 - mean_absolute_error: 1.0680


647/Unknown  134s 196ms/step - loss: 1.8835 - mean_absolute_error: 1.0678


648/Unknown  134s 196ms/step - loss: 1.8828 - mean_absolute_error: 1.0675


649/Unknown  134s 196ms/step - loss: 1.8820 - mean_absolute_error: 1.0673


650/Unknown  134s 196ms/step - loss: 1.8812 - mean_absolute_error: 1.0671


651/Unknown  134s 196ms/step - loss: 1.8804 - mean_absolute_error: 1.0669


652/Unknown  135s 196ms/step - loss: 1.8796 - mean_absolute_error: 1.0667


653/Unknown  135s 196ms/step - loss: 1.8789 - mean_absolute_error: 1.0665


654/Unknown  135s 196ms/step - loss: 1.8781 - mean_absolute_error: 1.0663


655/Unknown  135s 196ms/step - loss: 1.8773 - mean_absolute_error: 1.0661


656/Unknown  135s 196ms/step - loss: 1.8766 - mean_absolute_error: 1.0659


657/Unknown  136s 196ms/step - loss: 1.8758 - mean_absolute_error: 1.0657


658/Unknown  136s 196ms/step - loss: 1.8750 - mean_absolute_error: 1.0655


659/Unknown  136s 196ms/step - loss: 1.8743 - mean_absolute_error: 1.0652


660/Unknown  136s 196ms/step - loss: 1.8735 - mean_absolute_error: 1.0650


661/Unknown  137s 197ms/step - loss: 1.8728 - mean_absolute_error: 1.0648


662/Unknown  137s 197ms/step - loss: 1.8720 - mean_absolute_error: 1.0646


663/Unknown  137s 197ms/step - loss: 1.8713 - mean_absolute_error: 1.0644


664/Unknown  137s 197ms/step - loss: 1.8705 - mean_absolute_error: 1.0642


665/Unknown  137s 197ms/step - loss: 1.8698 - mean_absolute_error: 1.0640


666/Unknown  138s 197ms/step - loss: 1.8690 - mean_absolute_error: 1.0638


667/Unknown  138s 197ms/step - loss: 1.8683 - mean_absolute_error: 1.0636


668/Unknown  138s 197ms/step - loss: 1.8675 - mean_absolute_error: 1.0634


669/Unknown  138s 197ms/step - loss: 1.8668 - mean_absolute_error: 1.0632


670/Unknown  139s 197ms/step - loss: 1.8660 - mean_absolute_error: 1.0630


671/Unknown  139s 197ms/step - loss: 1.8653 - mean_absolute_error: 1.0628


672/Unknown  139s 197ms/step - loss: 1.8646 - mean_absolute_error: 1.0626


673/Unknown  139s 197ms/step - loss: 1.8638 - mean_absolute_error: 1.0624


674/Unknown  140s 197ms/step - loss: 1.8631 - mean_absolute_error: 1.0622


675/Unknown  140s 198ms/step - loss: 1.8624 - mean_absolute_error: 1.0620


676/Unknown  140s 198ms/step - loss: 1.8616 - mean_absolute_error: 1.0618


677/Unknown  141s 198ms/step - loss: 1.8609 - mean_absolute_error: 1.0616


678/Unknown  141s 198ms/step - loss: 1.8602 - mean_absolute_error: 1.0614


679/Unknown  141s 198ms/step - loss: 1.8595 - mean_absolute_error: 1.0612


680/Unknown  141s 198ms/step - loss: 1.8587 - mean_absolute_error: 1.0610


681/Unknown  142s 198ms/step - loss: 1.8580 - mean_absolute_error: 1.0609


682/Unknown  142s 198ms/step - loss: 1.8573 - mean_absolute_error: 1.0607


683/Unknown  142s 198ms/step - loss: 1.8566 - mean_absolute_error: 1.0605


684/Unknown  143s 199ms/step - loss: 1.8559 - mean_absolute_error: 1.0603


685/Unknown  143s 199ms/step - loss: 1.8551 - mean_absolute_error: 1.0601


686/Unknown  143s 199ms/step - loss: 1.8544 - mean_absolute_error: 1.0599


687/Unknown  143s 199ms/step - loss: 1.8537 - mean_absolute_error: 1.0597


688/Unknown  144s 199ms/step - loss: 1.8530 - mean_absolute_error: 1.0595


689/Unknown  144s 199ms/step - loss: 1.8523 - mean_absolute_error: 1.0593


690/Unknown  144s 199ms/step - loss: 1.8516 - mean_absolute_error: 1.0591


691/Unknown  144s 199ms/step - loss: 1.8509 - mean_absolute_error: 1.0589


692/Unknown  145s 199ms/step - loss: 1.8502 - mean_absolute_error: 1.0587


693/Unknown  145s 199ms/step - loss: 1.8495 - mean_absolute_error: 1.0585


694/Unknown  145s 199ms/step - loss: 1.8488 - mean_absolute_error: 1.0584


695/Unknown  145s 200ms/step - loss: 1.8481 - mean_absolute_error: 1.0582


696/Unknown  146s 200ms/step - loss: 1.8474 - mean_absolute_error: 1.0580


697/Unknown  146s 200ms/step - loss: 1.8467 - mean_absolute_error: 1.0578


698/Unknown  146s 200ms/step - loss: 1.8460 - mean_absolute_error: 1.0576


699/Unknown  146s 200ms/step - loss: 1.8453 - mean_absolute_error: 1.0574


700/Unknown  147s 200ms/step - loss: 1.8446 - mean_absolute_error: 1.0572


701/Unknown  147s 200ms/step - loss: 1.8439 - mean_absolute_error: 1.0570


702/Unknown  147s 200ms/step - loss: 1.8432 - mean_absolute_error: 1.0569


703/Unknown  147s 200ms/step - loss: 1.8425 - mean_absolute_error: 1.0567


704/Unknown  147s 200ms/step - loss: 1.8419 - mean_absolute_error: 1.0565


705/Unknown  148s 200ms/step - loss: 1.8412 - mean_absolute_error: 1.0563


706/Unknown  148s 200ms/step - loss: 1.8405 - mean_absolute_error: 1.0561


707/Unknown  148s 200ms/step - loss: 1.8398 - mean_absolute_error: 1.0559


708/Unknown  149s 200ms/step - loss: 1.8391 - mean_absolute_error: 1.0557


709/Unknown  149s 200ms/step - loss: 1.8385 - mean_absolute_error: 1.0556


710/Unknown  149s 200ms/step - loss: 1.8378 - mean_absolute_error: 1.0554


711/Unknown  149s 201ms/step - loss: 1.8371 - mean_absolute_error: 1.0552


712/Unknown  150s 201ms/step - loss: 1.8364 - mean_absolute_error: 1.0550


713/Unknown  150s 201ms/step - loss: 1.8358 - mean_absolute_error: 1.0548


714/Unknown  150s 201ms/step - loss: 1.8351 - mean_absolute_error: 1.0547


715/Unknown  150s 201ms/step - loss: 1.8344 - mean_absolute_error: 1.0545


716/Unknown  151s 201ms/step - loss: 1.8338 - mean_absolute_error: 1.0543


717/Unknown  151s 201ms/step - loss: 1.8331 - mean_absolute_error: 1.0541


718/Unknown  151s 201ms/step - loss: 1.8324 - mean_absolute_error: 1.0539


719/Unknown  151s 201ms/step - loss: 1.8318 - mean_absolute_error: 1.0537


720/Unknown  152s 201ms/step - loss: 1.8311 - mean_absolute_error: 1.0536


721/Unknown  152s 201ms/step - loss: 1.8305 - mean_absolute_error: 1.0534


722/Unknown  152s 201ms/step - loss: 1.8298 - mean_absolute_error: 1.0532


723/Unknown  152s 201ms/step - loss: 1.8291 - mean_absolute_error: 1.0530


724/Unknown  152s 201ms/step - loss: 1.8285 - mean_absolute_error: 1.0529


725/Unknown  153s 201ms/step - loss: 1.8278 - mean_absolute_error: 1.0527


726/Unknown  153s 201ms/step - loss: 1.8272 - mean_absolute_error: 1.0525


727/Unknown  153s 201ms/step - loss: 1.8265 - mean_absolute_error: 1.0523


728/Unknown  153s 201ms/step - loss: 1.8259 - mean_absolute_error: 1.0522


729/Unknown  153s 201ms/step - loss: 1.8253 - mean_absolute_error: 1.0520


730/Unknown  154s 201ms/step - loss: 1.8246 - mean_absolute_error: 1.0518


731/Unknown  154s 201ms/step - loss: 1.8240 - mean_absolute_error: 1.0516


732/Unknown  154s 201ms/step - loss: 1.8233 - mean_absolute_error: 1.0515


733/Unknown  154s 201ms/step - loss: 1.8227 - mean_absolute_error: 1.0513


734/Unknown  154s 201ms/step - loss: 1.8220 - mean_absolute_error: 1.0511


735/Unknown  155s 201ms/step - loss: 1.8214 - mean_absolute_error: 1.0509


736/Unknown  155s 201ms/step - loss: 1.8208 - mean_absolute_error: 1.0508


737/Unknown  155s 201ms/step - loss: 1.8201 - mean_absolute_error: 1.0506


738/Unknown  155s 201ms/step - loss: 1.8195 - mean_absolute_error: 1.0504


739/Unknown  155s 201ms/step - loss: 1.8189 - mean_absolute_error: 1.0502


740/Unknown  156s 201ms/step - loss: 1.8182 - mean_absolute_error: 1.0501


741/Unknown  156s 201ms/step - loss: 1.8176 - mean_absolute_error: 1.0499


742/Unknown  156s 201ms/step - loss: 1.8170 - mean_absolute_error: 1.0497


743/Unknown  156s 201ms/step - loss: 1.8164 - mean_absolute_error: 1.0496


744/Unknown  157s 201ms/step - loss: 1.8157 - mean_absolute_error: 1.0494


745/Unknown  157s 201ms/step - loss: 1.8151 - mean_absolute_error: 1.0492


746/Unknown  157s 201ms/step - loss: 1.8145 - mean_absolute_error: 1.0491


747/Unknown  157s 201ms/step - loss: 1.8139 - mean_absolute_error: 1.0489


748/Unknown  157s 202ms/step - loss: 1.8133 - mean_absolute_error: 1.0487


749/Unknown  158s 202ms/step - loss: 1.8126 - mean_absolute_error: 1.0485


750/Unknown  158s 202ms/step - loss: 1.8120 - mean_absolute_error: 1.0484


751/Unknown  158s 202ms/step - loss: 1.8114 - mean_absolute_error: 1.0482


752/Unknown  159s 202ms/step - loss: 1.8108 - mean_absolute_error: 1.0480


753/Unknown  159s 202ms/step - loss: 1.8102 - mean_absolute_error: 1.0479


754/Unknown  159s 202ms/step - loss: 1.8096 - mean_absolute_error: 1.0477


755/Unknown  159s 202ms/step - loss: 1.8090 - mean_absolute_error: 1.0475


756/Unknown  159s 202ms/step - loss: 1.8083 - mean_absolute_error: 1.0474


757/Unknown  160s 202ms/step - loss: 1.8077 - mean_absolute_error: 1.0472


758/Unknown  160s 202ms/step - loss: 1.8071 - mean_absolute_error: 1.0470


759/Unknown  160s 202ms/step - loss: 1.8065 - mean_absolute_error: 1.0469


760/Unknown  160s 202ms/step - loss: 1.8059 - mean_absolute_error: 1.0467


761/Unknown  161s 202ms/step - loss: 1.8053 - mean_absolute_error: 1.0466


762/Unknown  161s 202ms/step - loss: 1.8047 - mean_absolute_error: 1.0464


763/Unknown  161s 202ms/step - loss: 1.8041 - mean_absolute_error: 1.0462


764/Unknown  161s 202ms/step - loss: 1.8035 - mean_absolute_error: 1.0461


765/Unknown  162s 202ms/step - loss: 1.8029 - mean_absolute_error: 1.0459


766/Unknown  162s 203ms/step - loss: 1.8023 - mean_absolute_error: 1.0457


767/Unknown  162s 203ms/step - loss: 1.8017 - mean_absolute_error: 1.0456


768/Unknown  162s 203ms/step - loss: 1.8011 - mean_absolute_error: 1.0454


769/Unknown  163s 203ms/step - loss: 1.8005 - mean_absolute_error: 1.0453


770/Unknown  163s 203ms/step - loss: 1.7999 - mean_absolute_error: 1.0451


771/Unknown  163s 203ms/step - loss: 1.7994 - mean_absolute_error: 1.0449


772/Unknown  163s 203ms/step - loss: 1.7988 - mean_absolute_error: 1.0448


773/Unknown  164s 203ms/step - loss: 1.7982 - mean_absolute_error: 1.0446


774/Unknown  164s 203ms/step - loss: 1.7976 - mean_absolute_error: 1.0444


775/Unknown  164s 203ms/step - loss: 1.7970 - mean_absolute_error: 1.0443


776/Unknown  164s 203ms/step - loss: 1.7964 - mean_absolute_error: 1.0441


777/Unknown  164s 203ms/step - loss: 1.7958 - mean_absolute_error: 1.0440


778/Unknown  165s 203ms/step - loss: 1.7953 - mean_absolute_error: 1.0438


779/Unknown  165s 203ms/step - loss: 1.7947 - mean_absolute_error: 1.0437


780/Unknown  165s 203ms/step - loss: 1.7941 - mean_absolute_error: 1.0435


781/Unknown  166s 203ms/step - loss: 1.7935 - mean_absolute_error: 1.0433


782/Unknown  166s 204ms/step - loss: 1.7929 - mean_absolute_error: 1.0432


783/Unknown  166s 204ms/step - loss: 1.7924 - mean_absolute_error: 1.0430


784/Unknown  166s 204ms/step - loss: 1.7918 - mean_absolute_error: 1.0429


785/Unknown  167s 204ms/step - loss: 1.7912 - mean_absolute_error: 1.0427


786/Unknown  167s 204ms/step - loss: 1.7907 - mean_absolute_error: 1.0426


787/Unknown  167s 204ms/step - loss: 1.7901 - mean_absolute_error: 1.0424


788/Unknown  167s 204ms/step - loss: 1.7895 - mean_absolute_error: 1.0422


789/Unknown  168s 204ms/step - loss: 1.7889 - mean_absolute_error: 1.0421


790/Unknown  168s 204ms/step - loss: 1.7884 - mean_absolute_error: 1.0419


791/Unknown  168s 204ms/step - loss: 1.7878 - mean_absolute_error: 1.0418


792/Unknown  168s 204ms/step - loss: 1.7872 - mean_absolute_error: 1.0416


793/Unknown  168s 204ms/step - loss: 1.7867 - mean_absolute_error: 1.0415


794/Unknown  169s 204ms/step - loss: 1.7861 - mean_absolute_error: 1.0413


795/Unknown  169s 204ms/step - loss: 1.7856 - mean_absolute_error: 1.0412


796/Unknown  169s 204ms/step - loss: 1.7850 - mean_absolute_error: 1.0410


797/Unknown  169s 204ms/step - loss: 1.7844 - mean_absolute_error: 1.0409


798/Unknown  170s 204ms/step - loss: 1.7839 - mean_absolute_error: 1.0407


799/Unknown  170s 204ms/step - loss: 1.7833 - mean_absolute_error: 1.0406


800/Unknown  170s 204ms/step - loss: 1.7828 - mean_absolute_error: 1.0404


801/Unknown  170s 204ms/step - loss: 1.7822 - mean_absolute_error: 1.0402


802/Unknown  170s 204ms/step - loss: 1.7817 - mean_absolute_error: 1.0401


803/Unknown  171s 204ms/step - loss: 1.7811 - mean_absolute_error: 1.0399


804/Unknown  171s 204ms/step - loss: 1.7806 - mean_absolute_error: 1.0398


805/Unknown  171s 204ms/step - loss: 1.7800 - mean_absolute_error: 1.0396


806/Unknown  171s 204ms/step - loss: 1.7794 - mean_absolute_error: 1.0395


807/Unknown  172s 205ms/step - loss: 1.7789 - mean_absolute_error: 1.0393


808/Unknown  172s 205ms/step - loss: 1.7784 - mean_absolute_error: 1.0392


809/Unknown  172s 205ms/step - loss: 1.7778 - mean_absolute_error: 1.0390


810/Unknown  172s 205ms/step - loss: 1.7773 - mean_absolute_error: 1.0389


811/Unknown  173s 205ms/step - loss: 1.7767 - mean_absolute_error: 1.0387


812/Unknown  173s 205ms/step - loss: 1.7762 - mean_absolute_error: 1.0386


813/Unknown  173s 205ms/step - loss: 1.7756 - mean_absolute_error: 1.0384


814/Unknown  173s 205ms/step - loss: 1.7751 - mean_absolute_error: 1.0383


815/Unknown  174s 205ms/step - loss: 1.7745 - mean_absolute_error: 1.0381


816/Unknown  174s 205ms/step - loss: 1.7740 - mean_absolute_error: 1.0380


817/Unknown  174s 205ms/step - loss: 1.7735 - mean_absolute_error: 1.0379


818/Unknown  174s 205ms/step - loss: 1.7729 - mean_absolute_error: 1.0377


819/Unknown  175s 205ms/step - loss: 1.7724 - mean_absolute_error: 1.0376


820/Unknown  175s 205ms/step - loss: 1.7719 - mean_absolute_error: 1.0374


821/Unknown  175s 205ms/step - loss: 1.7713 - mean_absolute_error: 1.0373


822/Unknown  175s 205ms/step - loss: 1.7708 - mean_absolute_error: 1.0371


823/Unknown  176s 205ms/step - loss: 1.7703 - mean_absolute_error: 1.0370


824/Unknown  176s 205ms/step - loss: 1.7697 - mean_absolute_error: 1.0368


825/Unknown  176s 205ms/step - loss: 1.7692 - mean_absolute_error: 1.0367


826/Unknown  176s 206ms/step - loss: 1.7687 - mean_absolute_error: 1.0365


827/Unknown  177s 206ms/step - loss: 1.7681 - mean_absolute_error: 1.0364


828/Unknown  177s 206ms/step - loss: 1.7676 - mean_absolute_error: 1.0362


829/Unknown  177s 206ms/step - loss: 1.7671 - mean_absolute_error: 1.0361


830/Unknown  178s 206ms/step - loss: 1.7666 - mean_absolute_error: 1.0360


831/Unknown  178s 206ms/step - loss: 1.7660 - mean_absolute_error: 1.0358


832/Unknown  178s 206ms/step - loss: 1.7655 - mean_absolute_error: 1.0357


833/Unknown  178s 206ms/step - loss: 1.7650 - mean_absolute_error: 1.0355


834/Unknown  179s 206ms/step - loss: 1.7645 - mean_absolute_error: 1.0354


835/Unknown  179s 206ms/step - loss: 1.7639 - mean_absolute_error: 1.0352


836/Unknown  179s 206ms/step - loss: 1.7634 - mean_absolute_error: 1.0351


837/Unknown  179s 206ms/step - loss: 1.7629 - mean_absolute_error: 1.0350


838/Unknown  179s 206ms/step - loss: 1.7624 - mean_absolute_error: 1.0348


839/Unknown  180s 206ms/step - loss: 1.7619 - mean_absolute_error: 1.0347


840/Unknown  180s 206ms/step - loss: 1.7614 - mean_absolute_error: 1.0345


841/Unknown  180s 206ms/step - loss: 1.7608 - mean_absolute_error: 1.0344


842/Unknown  180s 206ms/step - loss: 1.7603 - mean_absolute_error: 1.0343


843/Unknown  181s 206ms/step - loss: 1.7598 - mean_absolute_error: 1.0341


844/Unknown  181s 206ms/step - loss: 1.7593 - mean_absolute_error: 1.0340


845/Unknown  181s 206ms/step - loss: 1.7588 - mean_absolute_error: 1.0338


846/Unknown  181s 206ms/step - loss: 1.7583 - mean_absolute_error: 1.0337


847/Unknown  181s 206ms/step - loss: 1.7578 - mean_absolute_error: 1.0335


848/Unknown  182s 206ms/step - loss: 1.7573 - mean_absolute_error: 1.0334


849/Unknown  182s 206ms/step - loss: 1.7568 - mean_absolute_error: 1.0333


850/Unknown  182s 206ms/step - loss: 1.7563 - mean_absolute_error: 1.0331


851/Unknown  182s 206ms/step - loss: 1.7558 - mean_absolute_error: 1.0330


852/Unknown  183s 206ms/step - loss: 1.7553 - mean_absolute_error: 1.0329


853/Unknown  183s 207ms/step - loss: 1.7547 - mean_absolute_error: 1.0327


854/Unknown  183s 207ms/step - loss: 1.7542 - mean_absolute_error: 1.0326


855/Unknown  183s 207ms/step - loss: 1.7537 - mean_absolute_error: 1.0324


856/Unknown  183s 207ms/step - loss: 1.7532 - mean_absolute_error: 1.0323


857/Unknown  184s 207ms/step - loss: 1.7527 - mean_absolute_error: 1.0322


858/Unknown  184s 207ms/step - loss: 1.7522 - mean_absolute_error: 1.0320


859/Unknown  184s 207ms/step - loss: 1.7518 - mean_absolute_error: 1.0319


860/Unknown  184s 207ms/step - loss: 1.7513 - mean_absolute_error: 1.0318


861/Unknown  185s 207ms/step - loss: 1.7508 - mean_absolute_error: 1.0316


862/Unknown  185s 207ms/step - loss: 1.7503 - mean_absolute_error: 1.0315


863/Unknown  185s 207ms/step - loss: 1.7498 - mean_absolute_error: 1.0313


864/Unknown  185s 207ms/step - loss: 1.7493 - mean_absolute_error: 1.0312


865/Unknown  185s 207ms/step - loss: 1.7488 - mean_absolute_error: 1.0311


866/Unknown  186s 207ms/step - loss: 1.7483 - mean_absolute_error: 1.0309


867/Unknown  186s 207ms/step - loss: 1.7478 - mean_absolute_error: 1.0308


868/Unknown  186s 207ms/step - loss: 1.7473 - mean_absolute_error: 1.0307


869/Unknown  186s 207ms/step - loss: 1.7468 - mean_absolute_error: 1.0305


870/Unknown  187s 207ms/step - loss: 1.7463 - mean_absolute_error: 1.0304


871/Unknown  187s 207ms/step - loss: 1.7459 - mean_absolute_error: 1.0303


872/Unknown  187s 207ms/step - loss: 1.7454 - mean_absolute_error: 1.0301


873/Unknown  187s 207ms/step - loss: 1.7449 - mean_absolute_error: 1.0300


874/Unknown  188s 207ms/step - loss: 1.7444 - mean_absolute_error: 1.0299


875/Unknown  188s 207ms/step - loss: 1.7439 - mean_absolute_error: 1.0297


876/Unknown  188s 207ms/step - loss: 1.7434 - mean_absolute_error: 1.0296


877/Unknown  188s 207ms/step - loss: 1.7430 - mean_absolute_error: 1.0295


878/Unknown  189s 207ms/step - loss: 1.7425 - mean_absolute_error: 1.0293


879/Unknown  189s 207ms/step - loss: 1.7420 - mean_absolute_error: 1.0292


880/Unknown  189s 207ms/step - loss: 1.7415 - mean_absolute_error: 1.0291


881/Unknown  189s 207ms/step - loss: 1.7410 - mean_absolute_error: 1.0289


882/Unknown  190s 207ms/step - loss: 1.7406 - mean_absolute_error: 1.0288


883/Unknown  190s 207ms/step - loss: 1.7401 - mean_absolute_error: 1.0287


884/Unknown  190s 207ms/step - loss: 1.7396 - mean_absolute_error: 1.0286


885/Unknown  190s 207ms/step - loss: 1.7391 - mean_absolute_error: 1.0284


886/Unknown  191s 208ms/step - loss: 1.7387 - mean_absolute_error: 1.0283


887/Unknown  191s 208ms/step - loss: 1.7382 - mean_absolute_error: 1.0282


888/Unknown  191s 208ms/step - loss: 1.7377 - mean_absolute_error: 1.0280


889/Unknown  191s 208ms/step - loss: 1.7373 - mean_absolute_error: 1.0279


890/Unknown  192s 208ms/step - loss: 1.7368 - mean_absolute_error: 1.0278


891/Unknown  192s 208ms/step - loss: 1.7363 - mean_absolute_error: 1.0276


892/Unknown  192s 208ms/step - loss: 1.7359 - mean_absolute_error: 1.0275


893/Unknown  192s 208ms/step - loss: 1.7354 - mean_absolute_error: 1.0274


894/Unknown  193s 208ms/step - loss: 1.7349 - mean_absolute_error: 1.0273


895/Unknown  193s 208ms/step - loss: 1.7345 - mean_absolute_error: 1.0271


896/Unknown  193s 208ms/step - loss: 1.7340 - mean_absolute_error: 1.0270


897/Unknown  193s 208ms/step - loss: 1.7335 - mean_absolute_error: 1.0269


898/Unknown  194s 208ms/step - loss: 1.7331 - mean_absolute_error: 1.0267


899/Unknown  194s 208ms/step - loss: 1.7326 - mean_absolute_error: 1.0266


900/Unknown  194s 208ms/step - loss: 1.7322 - mean_absolute_error: 1.0265


901/Unknown  194s 208ms/step - loss: 1.7317 - mean_absolute_error: 1.0264


902/Unknown  195s 208ms/step - loss: 1.7312 - mean_absolute_error: 1.0262


903/Unknown  195s 209ms/step - loss: 1.7308 - mean_absolute_error: 1.0261


904/Unknown  195s 209ms/step - loss: 1.7303 - mean_absolute_error: 1.0260


905/Unknown  196s 209ms/step - loss: 1.7299 - mean_absolute_error: 1.0259


906/Unknown  196s 209ms/step - loss: 1.7294 - mean_absolute_error: 1.0257


907/Unknown  196s 209ms/step - loss: 1.7290 - mean_absolute_error: 1.0256


908/Unknown  196s 209ms/step - loss: 1.7285 - mean_absolute_error: 1.0255


909/Unknown  197s 209ms/step - loss: 1.7280 - mean_absolute_error: 1.0254


910/Unknown  197s 209ms/step - loss: 1.7276 - mean_absolute_error: 1.0252


911/Unknown  197s 209ms/step - loss: 1.7271 - mean_absolute_error: 1.0251


912/Unknown  197s 209ms/step - loss: 1.7267 - mean_absolute_error: 1.0250


913/Unknown  198s 209ms/step - loss: 1.7262 - mean_absolute_error: 1.0249


914/Unknown  198s 209ms/step - loss: 1.7258 - mean_absolute_error: 1.0247


915/Unknown  198s 209ms/step - loss: 1.7253 - mean_absolute_error: 1.0246


916/Unknown  198s 209ms/step - loss: 1.7249 - mean_absolute_error: 1.0245


917/Unknown  199s 209ms/step - loss: 1.7244 - mean_absolute_error: 1.0244


918/Unknown  199s 209ms/step - loss: 1.7240 - mean_absolute_error: 1.0242


919/Unknown  199s 209ms/step - loss: 1.7235 - mean_absolute_error: 1.0241


920/Unknown  199s 209ms/step - loss: 1.7231 - mean_absolute_error: 1.0240


921/Unknown  200s 209ms/step - loss: 1.7226 - mean_absolute_error: 1.0239


922/Unknown  200s 209ms/step - loss: 1.7222 - mean_absolute_error: 1.0237


923/Unknown  200s 209ms/step - loss: 1.7218 - mean_absolute_error: 1.0236


924/Unknown  200s 209ms/step - loss: 1.7213 - mean_absolute_error: 1.0235


925/Unknown  200s 209ms/step - loss: 1.7209 - mean_absolute_error: 1.0234


926/Unknown  201s 209ms/step - loss: 1.7204 - mean_absolute_error: 1.0233


927/Unknown  201s 209ms/step - loss: 1.7200 - mean_absolute_error: 1.0231


928/Unknown  201s 209ms/step - loss: 1.7196 - mean_absolute_error: 1.0230


929/Unknown  201s 209ms/step - loss: 1.7191 - mean_absolute_error: 1.0229


930/Unknown  201s 209ms/step - loss: 1.7187 - mean_absolute_error: 1.0228


931/Unknown  202s 209ms/step - loss: 1.7182 - mean_absolute_error: 1.0226


932/Unknown  202s 209ms/step - loss: 1.7178 - mean_absolute_error: 1.0225


933/Unknown  202s 210ms/step - loss: 1.7174 - mean_absolute_error: 1.0224


934/Unknown  202s 210ms/step - loss: 1.7169 - mean_absolute_error: 1.0223


935/Unknown  203s 210ms/step - loss: 1.7165 - mean_absolute_error: 1.0222


936/Unknown  203s 210ms/step - loss: 1.7160 - mean_absolute_error: 1.0220


937/Unknown  203s 210ms/step - loss: 1.7156 - mean_absolute_error: 1.0219


938/Unknown  203s 210ms/step - loss: 1.7152 - mean_absolute_error: 1.0218


939/Unknown  204s 210ms/step - loss: 1.7147 - mean_absolute_error: 1.0217


940/Unknown  204s 210ms/step - loss: 1.7143 - mean_absolute_error: 1.0216


941/Unknown  204s 210ms/step - loss: 1.7139 - mean_absolute_error: 1.0214


942/Unknown  204s 210ms/step - loss: 1.7134 - mean_absolute_error: 1.0213


943/Unknown  205s 210ms/step - loss: 1.7130 - mean_absolute_error: 1.0212


944/Unknown  205s 210ms/step - loss: 1.7126 - mean_absolute_error: 1.0211


945/Unknown  205s 210ms/step - loss: 1.7122 - mean_absolute_error: 1.0210


946/Unknown  205s 210ms/step - loss: 1.7117 - mean_absolute_error: 1.0208


947/Unknown  206s 210ms/step - loss: 1.7113 - mean_absolute_error: 1.0207


948/Unknown  206s 210ms/step - loss: 1.7109 - mean_absolute_error: 1.0206


949/Unknown  206s 210ms/step - loss: 1.7104 - mean_absolute_error: 1.0205


950/Unknown  206s 210ms/step - loss: 1.7100 - mean_absolute_error: 1.0204


951/Unknown  206s 210ms/step - loss: 1.7096 - mean_absolute_error: 1.0202


952/Unknown  207s 210ms/step - loss: 1.7092 - mean_absolute_error: 1.0201


953/Unknown  207s 210ms/step - loss: 1.7087 - mean_absolute_error: 1.0200


954/Unknown  207s 210ms/step - loss: 1.7083 - mean_absolute_error: 1.0199


955/Unknown  207s 210ms/step - loss: 1.7079 - mean_absolute_error: 1.0198


956/Unknown  207s 210ms/step - loss: 1.7075 - mean_absolute_error: 1.0197


957/Unknown  208s 210ms/step - loss: 1.7070 - mean_absolute_error: 1.0195


958/Unknown  208s 210ms/step - loss: 1.7066 - mean_absolute_error: 1.0194


959/Unknown  208s 210ms/step - loss: 1.7062 - mean_absolute_error: 1.0193


960/Unknown  208s 210ms/step - loss: 1.7058 - mean_absolute_error: 1.0192


961/Unknown  209s 210ms/step - loss: 1.7054 - mean_absolute_error: 1.0191


962/Unknown  209s 210ms/step - loss: 1.7049 - mean_absolute_error: 1.0190


963/Unknown  209s 210ms/step - loss: 1.7045 - mean_absolute_error: 1.0188


964/Unknown  209s 210ms/step - loss: 1.7041 - mean_absolute_error: 1.0187


965/Unknown  210s 210ms/step - loss: 1.7037 - mean_absolute_error: 1.0186


966/Unknown  210s 210ms/step - loss: 1.7033 - mean_absolute_error: 1.0185


967/Unknown  210s 210ms/step - loss: 1.7029 - mean_absolute_error: 1.0184


968/Unknown  210s 210ms/step - loss: 1.7025 - mean_absolute_error: 1.0183


969/Unknown  210s 210ms/step - loss: 1.7020 - mean_absolute_error: 1.0181


970/Unknown  211s 210ms/step - loss: 1.7016 - mean_absolute_error: 1.0180


971/Unknown  211s 210ms/step - loss: 1.7012 - mean_absolute_error: 1.0179


972/Unknown  211s 210ms/step - loss: 1.7008 - mean_absolute_error: 1.0178


973/Unknown  211s 210ms/step - loss: 1.7004 - mean_absolute_error: 1.0177


974/Unknown  212s 210ms/step - loss: 1.7000 - mean_absolute_error: 1.0176


975/Unknown  212s 210ms/step - loss: 1.6996 - mean_absolute_error: 1.0175


976/Unknown  212s 210ms/step - loss: 1.6992 - mean_absolute_error: 1.0173


977/Unknown  212s 210ms/step - loss: 1.6987 - mean_absolute_error: 1.0172


978/Unknown  212s 210ms/step - loss: 1.6983 - mean_absolute_error: 1.0171


979/Unknown  213s 210ms/step - loss: 1.6979 - mean_absolute_error: 1.0170


980/Unknown  213s 210ms/step - loss: 1.6975 - mean_absolute_error: 1.0169


981/Unknown  213s 210ms/step - loss: 1.6971 - mean_absolute_error: 1.0168


982/Unknown  213s 210ms/step - loss: 1.6967 - mean_absolute_error: 1.0167


983/Unknown  214s 211ms/step - loss: 1.6963 - mean_absolute_error: 1.0166


984/Unknown  214s 211ms/step - loss: 1.6959 - mean_absolute_error: 1.0164


985/Unknown  214s 211ms/step - loss: 1.6955 - mean_absolute_error: 1.0163


986/Unknown  214s 211ms/step - loss: 1.6951 - mean_absolute_error: 1.0162


987/Unknown  214s 211ms/step - loss: 1.6947 - mean_absolute_error: 1.0161


988/Unknown  215s 211ms/step - loss: 1.6943 - mean_absolute_error: 1.0160


989/Unknown  215s 211ms/step - loss: 1.6939 - mean_absolute_error: 1.0159


990/Unknown  215s 211ms/step - loss: 1.6935 - mean_absolute_error: 1.0158


991/Unknown  215s 211ms/step - loss: 1.6931 - mean_absolute_error: 1.0157


992/Unknown  216s 211ms/step - loss: 1.6927 - mean_absolute_error: 1.0155


993/Unknown  216s 211ms/step - loss: 1.6923 - mean_absolute_error: 1.0154


994/Unknown  216s 211ms/step - loss: 1.6919 - mean_absolute_error: 1.0153


995/Unknown  217s 211ms/step - loss: 1.6915 - mean_absolute_error: 1.0152


996/Unknown  217s 211ms/step - loss: 1.6911 - mean_absolute_error: 1.0151


997/Unknown  217s 211ms/step - loss: 1.6907 - mean_absolute_error: 1.0150


998/Unknown  217s 211ms/step - loss: 1.6903 - mean_absolute_error: 1.0149


999/Unknown  217s 211ms/step - loss: 1.6899 - mean_absolute_error: 1.0148


```
</div>
   1000/Unknown  218s 211ms/step - loss: 1.6895 - mean_absolute_error: 1.0147

<div class="k-default-codeblock">
```

```
</div>
   1001/Unknown  218s 211ms/step - loss: 1.6891 - mean_absolute_error: 1.0145

<div class="k-default-codeblock">
```

```
</div>
   1002/Unknown  218s 211ms/step - loss: 1.6887 - mean_absolute_error: 1.0144

<div class="k-default-codeblock">
```

```
</div>
   1003/Unknown  218s 211ms/step - loss: 1.6883 - mean_absolute_error: 1.0143

<div class="k-default-codeblock">
```

```
</div>
   1004/Unknown  219s 211ms/step - loss: 1.6880 - mean_absolute_error: 1.0142

<div class="k-default-codeblock">
```

```
</div>
   1005/Unknown  219s 211ms/step - loss: 1.6876 - mean_absolute_error: 1.0141

<div class="k-default-codeblock">
```

```
</div>
   1006/Unknown  219s 211ms/step - loss: 1.6872 - mean_absolute_error: 1.0140

<div class="k-default-codeblock">
```

```
</div>
   1007/Unknown  220s 211ms/step - loss: 1.6868 - mean_absolute_error: 1.0139

<div class="k-default-codeblock">
```

```
</div>
   1008/Unknown  220s 211ms/step - loss: 1.6864 - mean_absolute_error: 1.0138

<div class="k-default-codeblock">
```

```
</div>
   1009/Unknown  220s 212ms/step - loss: 1.6860 - mean_absolute_error: 1.0137

<div class="k-default-codeblock">
```

```
</div>
   1010/Unknown  220s 212ms/step - loss: 1.6856 - mean_absolute_error: 1.0136

<div class="k-default-codeblock">
```

```
</div>
   1011/Unknown  221s 212ms/step - loss: 1.6852 - mean_absolute_error: 1.0135

<div class="k-default-codeblock">
```

```
</div>
   1012/Unknown  221s 212ms/step - loss: 1.6848 - mean_absolute_error: 1.0134

<div class="k-default-codeblock">
```

```
</div>
   1013/Unknown  221s 212ms/step - loss: 1.6845 - mean_absolute_error: 1.0132

<div class="k-default-codeblock">
```

```
</div>
   1014/Unknown  221s 212ms/step - loss: 1.6841 - mean_absolute_error: 1.0131

<div class="k-default-codeblock">
```

```
</div>
   1015/Unknown  222s 212ms/step - loss: 1.6837 - mean_absolute_error: 1.0130

<div class="k-default-codeblock">
```

```
</div>
   1016/Unknown  222s 212ms/step - loss: 1.6833 - mean_absolute_error: 1.0129

<div class="k-default-codeblock">
```

```
</div>
   1017/Unknown  222s 212ms/step - loss: 1.6829 - mean_absolute_error: 1.0128

<div class="k-default-codeblock">
```

```
</div>
   1018/Unknown  222s 212ms/step - loss: 1.6825 - mean_absolute_error: 1.0127

<div class="k-default-codeblock">
```

```
</div>
   1019/Unknown  223s 212ms/step - loss: 1.6821 - mean_absolute_error: 1.0126

<div class="k-default-codeblock">
```

```
</div>
   1020/Unknown  223s 212ms/step - loss: 1.6818 - mean_absolute_error: 1.0125

<div class="k-default-codeblock">
```

```
</div>
   1021/Unknown  223s 212ms/step - loss: 1.6814 - mean_absolute_error: 1.0124

<div class="k-default-codeblock">
```

```
</div>
   1022/Unknown  223s 212ms/step - loss: 1.6810 - mean_absolute_error: 1.0123

<div class="k-default-codeblock">
```

```
</div>
   1023/Unknown  224s 212ms/step - loss: 1.6806 - mean_absolute_error: 1.0122

<div class="k-default-codeblock">
```

```
</div>
   1024/Unknown  224s 212ms/step - loss: 1.6802 - mean_absolute_error: 1.0121

<div class="k-default-codeblock">
```

```
</div>
   1025/Unknown  224s 212ms/step - loss: 1.6799 - mean_absolute_error: 1.0120

<div class="k-default-codeblock">
```

```
</div>
   1026/Unknown  224s 212ms/step - loss: 1.6795 - mean_absolute_error: 1.0118

<div class="k-default-codeblock">
```

```
</div>
   1027/Unknown  225s 212ms/step - loss: 1.6791 - mean_absolute_error: 1.0117

<div class="k-default-codeblock">
```

```
</div>
   1028/Unknown  225s 212ms/step - loss: 1.6787 - mean_absolute_error: 1.0116

<div class="k-default-codeblock">
```

```
</div>
   1029/Unknown  225s 212ms/step - loss: 1.6783 - mean_absolute_error: 1.0115

<div class="k-default-codeblock">
```

```
</div>
   1030/Unknown  225s 212ms/step - loss: 1.6780 - mean_absolute_error: 1.0114

<div class="k-default-codeblock">
```

```
</div>
   1031/Unknown  226s 212ms/step - loss: 1.6776 - mean_absolute_error: 1.0113

<div class="k-default-codeblock">
```

```
</div>
   1032/Unknown  226s 212ms/step - loss: 1.6772 - mean_absolute_error: 1.0112

<div class="k-default-codeblock">
```

```
</div>
   1033/Unknown  226s 212ms/step - loss: 1.6768 - mean_absolute_error: 1.0111

<div class="k-default-codeblock">
```

```
</div>
   1034/Unknown  226s 212ms/step - loss: 1.6765 - mean_absolute_error: 1.0110

<div class="k-default-codeblock">
```

```
</div>
   1035/Unknown  227s 212ms/step - loss: 1.6761 - mean_absolute_error: 1.0109

<div class="k-default-codeblock">
```

```
</div>
   1036/Unknown  227s 212ms/step - loss: 1.6757 - mean_absolute_error: 1.0108

<div class="k-default-codeblock">
```

```
</div>
   1037/Unknown  227s 213ms/step - loss: 1.6753 - mean_absolute_error: 1.0107

<div class="k-default-codeblock">
```

```
</div>
   1038/Unknown  227s 213ms/step - loss: 1.6750 - mean_absolute_error: 1.0106

<div class="k-default-codeblock">
```

```
</div>
   1039/Unknown  227s 213ms/step - loss: 1.6746 - mean_absolute_error: 1.0105

<div class="k-default-codeblock">
```

```
</div>
   1040/Unknown  228s 213ms/step - loss: 1.6742 - mean_absolute_error: 1.0104

<div class="k-default-codeblock">
```

```
</div>
   1041/Unknown  228s 213ms/step - loss: 1.6738 - mean_absolute_error: 1.0103

<div class="k-default-codeblock">
```

```
</div>
   1042/Unknown  228s 213ms/step - loss: 1.6735 - mean_absolute_error: 1.0102

<div class="k-default-codeblock">
```

```
</div>
   1043/Unknown  228s 213ms/step - loss: 1.6731 - mean_absolute_error: 1.0101

<div class="k-default-codeblock">
```

```
</div>
   1044/Unknown  229s 213ms/step - loss: 1.6727 - mean_absolute_error: 1.0100

<div class="k-default-codeblock">
```

```
</div>
   1045/Unknown  229s 213ms/step - loss: 1.6724 - mean_absolute_error: 1.0099

<div class="k-default-codeblock">
```

```
</div>
   1046/Unknown  229s 213ms/step - loss: 1.6720 - mean_absolute_error: 1.0097

<div class="k-default-codeblock">
```

```
</div>
   1047/Unknown  229s 213ms/step - loss: 1.6716 - mean_absolute_error: 1.0096

<div class="k-default-codeblock">
```

```
</div>
   1048/Unknown  229s 213ms/step - loss: 1.6713 - mean_absolute_error: 1.0095

<div class="k-default-codeblock">
```

```
</div>
   1049/Unknown  230s 213ms/step - loss: 1.6709 - mean_absolute_error: 1.0094

<div class="k-default-codeblock">
```

```
</div>
   1050/Unknown  230s 213ms/step - loss: 1.6705 - mean_absolute_error: 1.0093

<div class="k-default-codeblock">
```

```
</div>
   1051/Unknown  230s 213ms/step - loss: 1.6702 - mean_absolute_error: 1.0092

<div class="k-default-codeblock">
```

```
</div>
   1052/Unknown  230s 213ms/step - loss: 1.6698 - mean_absolute_error: 1.0091

<div class="k-default-codeblock">
```

```
</div>
   1053/Unknown  231s 213ms/step - loss: 1.6694 - mean_absolute_error: 1.0090

<div class="k-default-codeblock">
```

```
</div>
   1054/Unknown  231s 213ms/step - loss: 1.6691 - mean_absolute_error: 1.0089

<div class="k-default-codeblock">
```

```
</div>
   1055/Unknown  231s 213ms/step - loss: 1.6687 - mean_absolute_error: 1.0088

<div class="k-default-codeblock">
```

```
</div>
   1056/Unknown  231s 213ms/step - loss: 1.6683 - mean_absolute_error: 1.0087

<div class="k-default-codeblock">
```

```
</div>
   1057/Unknown  231s 213ms/step - loss: 1.6680 - mean_absolute_error: 1.0086

<div class="k-default-codeblock">
```

```
</div>
   1058/Unknown  232s 213ms/step - loss: 1.6676 - mean_absolute_error: 1.0085

<div class="k-default-codeblock">
```

```
</div>
   1059/Unknown  232s 213ms/step - loss: 1.6672 - mean_absolute_error: 1.0084

<div class="k-default-codeblock">
```

```
</div>
   1060/Unknown  232s 213ms/step - loss: 1.6669 - mean_absolute_error: 1.0083

<div class="k-default-codeblock">
```

```
</div>
   1061/Unknown  232s 213ms/step - loss: 1.6665 - mean_absolute_error: 1.0082

<div class="k-default-codeblock">
```

```
</div>
   1062/Unknown  233s 213ms/step - loss: 1.6662 - mean_absolute_error: 1.0081

<div class="k-default-codeblock">
```

```
</div>
   1063/Unknown  233s 213ms/step - loss: 1.6658 - mean_absolute_error: 1.0080

<div class="k-default-codeblock">
```

```
</div>
   1064/Unknown  233s 213ms/step - loss: 1.6654 - mean_absolute_error: 1.0079

<div class="k-default-codeblock">
```

```
</div>
   1065/Unknown  233s 213ms/step - loss: 1.6651 - mean_absolute_error: 1.0078

<div class="k-default-codeblock">
```

```
</div>
   1066/Unknown  234s 213ms/step - loss: 1.6647 - mean_absolute_error: 1.0077

<div class="k-default-codeblock">
```

```
</div>
   1067/Unknown  234s 213ms/step - loss: 1.6644 - mean_absolute_error: 1.0076

<div class="k-default-codeblock">
```

```
</div>
   1068/Unknown  234s 213ms/step - loss: 1.6640 - mean_absolute_error: 1.0075

<div class="k-default-codeblock">
```

```
</div>
   1069/Unknown  234s 213ms/step - loss: 1.6637 - mean_absolute_error: 1.0074

<div class="k-default-codeblock">
```

```
</div>
   1070/Unknown  234s 213ms/step - loss: 1.6633 - mean_absolute_error: 1.0073

<div class="k-default-codeblock">
```

```
</div>
   1071/Unknown  235s 213ms/step - loss: 1.6629 - mean_absolute_error: 1.0072

<div class="k-default-codeblock">
```

```
</div>
   1072/Unknown  235s 213ms/step - loss: 1.6626 - mean_absolute_error: 1.0071

<div class="k-default-codeblock">
```

```
</div>
   1073/Unknown  235s 213ms/step - loss: 1.6622 - mean_absolute_error: 1.0070

<div class="k-default-codeblock">
```

```
</div>
   1074/Unknown  236s 213ms/step - loss: 1.6619 - mean_absolute_error: 1.0069

<div class="k-default-codeblock">
```

```
</div>
   1075/Unknown  236s 213ms/step - loss: 1.6615 - mean_absolute_error: 1.0068

<div class="k-default-codeblock">
```

```
</div>
   1076/Unknown  236s 213ms/step - loss: 1.6612 - mean_absolute_error: 1.0067

<div class="k-default-codeblock">
```

```
</div>
   1077/Unknown  236s 213ms/step - loss: 1.6608 - mean_absolute_error: 1.0066

<div class="k-default-codeblock">
```

```
</div>
   1078/Unknown  237s 213ms/step - loss: 1.6605 - mean_absolute_error: 1.0065

<div class="k-default-codeblock">
```

```
</div>
   1079/Unknown  237s 213ms/step - loss: 1.6601 - mean_absolute_error: 1.0064

<div class="k-default-codeblock">
```

```
</div>
   1080/Unknown  237s 213ms/step - loss: 1.6598 - mean_absolute_error: 1.0063

<div class="k-default-codeblock">
```

```
</div>
   1081/Unknown  237s 214ms/step - loss: 1.6594 - mean_absolute_error: 1.0062

<div class="k-default-codeblock">
```

```
</div>
   1082/Unknown  238s 214ms/step - loss: 1.6591 - mean_absolute_error: 1.0061

<div class="k-default-codeblock">
```

```
</div>
   1083/Unknown  238s 214ms/step - loss: 1.6587 - mean_absolute_error: 1.0060

<div class="k-default-codeblock">
```

```
</div>
   1084/Unknown  238s 214ms/step - loss: 1.6584 - mean_absolute_error: 1.0059

<div class="k-default-codeblock">
```

```
</div>
   1085/Unknown  238s 214ms/step - loss: 1.6580 - mean_absolute_error: 1.0058

<div class="k-default-codeblock">
```

```
</div>
   1086/Unknown  239s 214ms/step - loss: 1.6577 - mean_absolute_error: 1.0057

<div class="k-default-codeblock">
```

```
</div>
   1087/Unknown  239s 214ms/step - loss: 1.6573 - mean_absolute_error: 1.0056

<div class="k-default-codeblock">
```

```
</div>
   1088/Unknown  239s 214ms/step - loss: 1.6570 - mean_absolute_error: 1.0055

<div class="k-default-codeblock">
```

```
</div>
   1089/Unknown  239s 214ms/step - loss: 1.6566 - mean_absolute_error: 1.0054

<div class="k-default-codeblock">
```

```
</div>
   1090/Unknown  240s 214ms/step - loss: 1.6563 - mean_absolute_error: 1.0053

<div class="k-default-codeblock">
```

```
</div>
   1091/Unknown  240s 214ms/step - loss: 1.6559 - mean_absolute_error: 1.0052

<div class="k-default-codeblock">
```

```
</div>
   1092/Unknown  240s 214ms/step - loss: 1.6556 - mean_absolute_error: 1.0051

<div class="k-default-codeblock">
```

```
</div>
   1093/Unknown  240s 214ms/step - loss: 1.6552 - mean_absolute_error: 1.0050

<div class="k-default-codeblock">
```

```
</div>
   1094/Unknown  241s 214ms/step - loss: 1.6549 - mean_absolute_error: 1.0049

<div class="k-default-codeblock">
```

```
</div>
   1095/Unknown  241s 214ms/step - loss: 1.6546 - mean_absolute_error: 1.0048

<div class="k-default-codeblock">
```

```
</div>
   1096/Unknown  241s 214ms/step - loss: 1.6542 - mean_absolute_error: 1.0047

<div class="k-default-codeblock">
```

```
</div>
   1097/Unknown  241s 214ms/step - loss: 1.6539 - mean_absolute_error: 1.0046

<div class="k-default-codeblock">
```

```
</div>
   1098/Unknown  241s 214ms/step - loss: 1.6535 - mean_absolute_error: 1.0045

<div class="k-default-codeblock">
```

```
</div>
   1099/Unknown  242s 214ms/step - loss: 1.6532 - mean_absolute_error: 1.0044

<div class="k-default-codeblock">
```

```
</div>
   1100/Unknown  242s 214ms/step - loss: 1.6528 - mean_absolute_error: 1.0043

<div class="k-default-codeblock">
```

```
</div>
   1101/Unknown  242s 214ms/step - loss: 1.6525 - mean_absolute_error: 1.0043

<div class="k-default-codeblock">
```

```
</div>
   1102/Unknown  242s 214ms/step - loss: 1.6522 - mean_absolute_error: 1.0042

<div class="k-default-codeblock">
```

```
</div>
   1103/Unknown  243s 214ms/step - loss: 1.6518 - mean_absolute_error: 1.0041

<div class="k-default-codeblock">
```

```
</div>
   1104/Unknown  243s 214ms/step - loss: 1.6515 - mean_absolute_error: 1.0040

<div class="k-default-codeblock">
```

```
</div>
   1105/Unknown  243s 214ms/step - loss: 1.6511 - mean_absolute_error: 1.0039

<div class="k-default-codeblock">
```

```
</div>
   1106/Unknown  243s 214ms/step - loss: 1.6508 - mean_absolute_error: 1.0038

<div class="k-default-codeblock">
```

```
</div>
   1107/Unknown  244s 214ms/step - loss: 1.6505 - mean_absolute_error: 1.0037

<div class="k-default-codeblock">
```

```
</div>
   1108/Unknown  244s 214ms/step - loss: 1.6501 - mean_absolute_error: 1.0036

<div class="k-default-codeblock">
```

```
</div>
   1109/Unknown  244s 214ms/step - loss: 1.6498 - mean_absolute_error: 1.0035

<div class="k-default-codeblock">
```

```
</div>
   1110/Unknown  244s 214ms/step - loss: 1.6494 - mean_absolute_error: 1.0034

<div class="k-default-codeblock">
```

```
</div>
   1111/Unknown  245s 214ms/step - loss: 1.6491 - mean_absolute_error: 1.0033

<div class="k-default-codeblock">
```

```
</div>
   1112/Unknown  245s 214ms/step - loss: 1.6488 - mean_absolute_error: 1.0032

<div class="k-default-codeblock">
```

```
</div>
   1113/Unknown  245s 214ms/step - loss: 1.6484 - mean_absolute_error: 1.0031

<div class="k-default-codeblock">
```

```
</div>
   1114/Unknown  245s 214ms/step - loss: 1.6481 - mean_absolute_error: 1.0030

<div class="k-default-codeblock">
```

```
</div>
   1115/Unknown  246s 214ms/step - loss: 1.6478 - mean_absolute_error: 1.0029

<div class="k-default-codeblock">
```

```
</div>
   1116/Unknown  246s 214ms/step - loss: 1.6474 - mean_absolute_error: 1.0028

<div class="k-default-codeblock">
```

```
</div>
   1117/Unknown  246s 215ms/step - loss: 1.6471 - mean_absolute_error: 1.0027

<div class="k-default-codeblock">
```

```
</div>
   1118/Unknown  247s 215ms/step - loss: 1.6468 - mean_absolute_error: 1.0026

<div class="k-default-codeblock">
```

```
</div>
   1119/Unknown  247s 215ms/step - loss: 1.6464 - mean_absolute_error: 1.0025

<div class="k-default-codeblock">
```

```
</div>
   1120/Unknown  247s 215ms/step - loss: 1.6461 - mean_absolute_error: 1.0024

<div class="k-default-codeblock">
```

```
</div>
   1121/Unknown  247s 215ms/step - loss: 1.6458 - mean_absolute_error: 1.0023

<div class="k-default-codeblock">
```

```
</div>
   1122/Unknown  248s 215ms/step - loss: 1.6454 - mean_absolute_error: 1.0022

<div class="k-default-codeblock">
```

```
</div>
   1123/Unknown  248s 215ms/step - loss: 1.6451 - mean_absolute_error: 1.0022

<div class="k-default-codeblock">
```

```
</div>
   1124/Unknown  248s 215ms/step - loss: 1.6448 - mean_absolute_error: 1.0021

<div class="k-default-codeblock">
```

```
</div>
   1125/Unknown  248s 215ms/step - loss: 1.6444 - mean_absolute_error: 1.0020

<div class="k-default-codeblock">
```

```
</div>
   1126/Unknown  249s 215ms/step - loss: 1.6441 - mean_absolute_error: 1.0019

<div class="k-default-codeblock">
```

```
</div>
   1127/Unknown  249s 215ms/step - loss: 1.6438 - mean_absolute_error: 1.0018

<div class="k-default-codeblock">
```

```
</div>
   1128/Unknown  249s 215ms/step - loss: 1.6434 - mean_absolute_error: 1.0017

<div class="k-default-codeblock">
```

```
</div>
   1129/Unknown  249s 215ms/step - loss: 1.6431 - mean_absolute_error: 1.0016

<div class="k-default-codeblock">
```

```
</div>
   1130/Unknown  249s 215ms/step - loss: 1.6428 - mean_absolute_error: 1.0015

<div class="k-default-codeblock">
```

```
</div>
   1131/Unknown  250s 215ms/step - loss: 1.6425 - mean_absolute_error: 1.0014

<div class="k-default-codeblock">
```

```
</div>
   1132/Unknown  250s 215ms/step - loss: 1.6421 - mean_absolute_error: 1.0013

<div class="k-default-codeblock">
```

```
</div>
   1133/Unknown  250s 215ms/step - loss: 1.6418 - mean_absolute_error: 1.0012

<div class="k-default-codeblock">
```

```
</div>
   1134/Unknown  250s 215ms/step - loss: 1.6415 - mean_absolute_error: 1.0011

<div class="k-default-codeblock">
```

```
</div>
   1135/Unknown  251s 215ms/step - loss: 1.6411 - mean_absolute_error: 1.0010

<div class="k-default-codeblock">
```

```
</div>
   1136/Unknown  251s 215ms/step - loss: 1.6408 - mean_absolute_error: 1.0009

<div class="k-default-codeblock">
```

```
</div>
   1137/Unknown  251s 215ms/step - loss: 1.6405 - mean_absolute_error: 1.0008

<div class="k-default-codeblock">
```

```
</div>
   1138/Unknown  251s 215ms/step - loss: 1.6402 - mean_absolute_error: 1.0008

<div class="k-default-codeblock">
```

```
</div>
   1139/Unknown  251s 215ms/step - loss: 1.6398 - mean_absolute_error: 1.0007

<div class="k-default-codeblock">
```

```
</div>
   1140/Unknown  252s 215ms/step - loss: 1.6395 - mean_absolute_error: 1.0006

<div class="k-default-codeblock">
```

```
</div>
   1141/Unknown  252s 215ms/step - loss: 1.6392 - mean_absolute_error: 1.0005

<div class="k-default-codeblock">
```

```
</div>
   1142/Unknown  252s 215ms/step - loss: 1.6389 - mean_absolute_error: 1.0004

<div class="k-default-codeblock">
```

```
</div>
   1143/Unknown  252s 215ms/step - loss: 1.6385 - mean_absolute_error: 1.0003

<div class="k-default-codeblock">
```

```
</div>
   1144/Unknown  253s 215ms/step - loss: 1.6382 - mean_absolute_error: 1.0002

<div class="k-default-codeblock">
```

```
</div>
   1145/Unknown  253s 215ms/step - loss: 1.6379 - mean_absolute_error: 1.0001

<div class="k-default-codeblock">
```

```
</div>
   1146/Unknown  253s 215ms/step - loss: 1.6376 - mean_absolute_error: 1.0000

<div class="k-default-codeblock">
```

```
</div>
   1147/Unknown  253s 215ms/step - loss: 1.6373 - mean_absolute_error: 0.9999

<div class="k-default-codeblock">
```

```
</div>
   1148/Unknown  254s 215ms/step - loss: 1.6369 - mean_absolute_error: 0.9998

<div class="k-default-codeblock">
```

```
</div>
   1149/Unknown  254s 215ms/step - loss: 1.6366 - mean_absolute_error: 0.9997

<div class="k-default-codeblock">
```

```
</div>
   1150/Unknown  254s 215ms/step - loss: 1.6363 - mean_absolute_error: 0.9997

<div class="k-default-codeblock">
```

```
</div>
   1151/Unknown  254s 215ms/step - loss: 1.6360 - mean_absolute_error: 0.9996

<div class="k-default-codeblock">
```

```
</div>
   1152/Unknown  255s 215ms/step - loss: 1.6356 - mean_absolute_error: 0.9995

<div class="k-default-codeblock">
```

```
</div>
   1153/Unknown  255s 215ms/step - loss: 1.6353 - mean_absolute_error: 0.9994

<div class="k-default-codeblock">
```

```
</div>
   1154/Unknown  255s 215ms/step - loss: 1.6350 - mean_absolute_error: 0.9993

<div class="k-default-codeblock">
```

```
</div>
   1155/Unknown  255s 215ms/step - loss: 1.6347 - mean_absolute_error: 0.9992

<div class="k-default-codeblock">
```

```
</div>
   1156/Unknown  256s 215ms/step - loss: 1.6344 - mean_absolute_error: 0.9991

<div class="k-default-codeblock">
```

```
</div>
   1157/Unknown  256s 215ms/step - loss: 1.6341 - mean_absolute_error: 0.9990

<div class="k-default-codeblock">
```

```
</div>
   1158/Unknown  256s 215ms/step - loss: 1.6337 - mean_absolute_error: 0.9989

<div class="k-default-codeblock">
```

```
</div>
   1159/Unknown  256s 215ms/step - loss: 1.6334 - mean_absolute_error: 0.9988

<div class="k-default-codeblock">
```

```
</div>
   1160/Unknown  257s 215ms/step - loss: 1.6331 - mean_absolute_error: 0.9987

<div class="k-default-codeblock">
```

```
</div>
   1161/Unknown  257s 215ms/step - loss: 1.6328 - mean_absolute_error: 0.9987

<div class="k-default-codeblock">
```

```
</div>
   1162/Unknown  257s 215ms/step - loss: 1.6325 - mean_absolute_error: 0.9986

<div class="k-default-codeblock">
```

```
</div>
   1163/Unknown  257s 216ms/step - loss: 1.6322 - mean_absolute_error: 0.9985

<div class="k-default-codeblock">
```

```
</div>
   1164/Unknown  257s 216ms/step - loss: 1.6318 - mean_absolute_error: 0.9984

<div class="k-default-codeblock">
```

```
</div>
   1165/Unknown  258s 215ms/step - loss: 1.6315 - mean_absolute_error: 0.9983

<div class="k-default-codeblock">
```

```
</div>
   1166/Unknown  258s 215ms/step - loss: 1.6312 - mean_absolute_error: 0.9982

<div class="k-default-codeblock">
```

```
</div>
   1167/Unknown  258s 215ms/step - loss: 1.6309 - mean_absolute_error: 0.9981

<div class="k-default-codeblock">
```

```
</div>
   1168/Unknown  258s 215ms/step - loss: 1.6306 - mean_absolute_error: 0.9980

<div class="k-default-codeblock">
```

```
</div>
   1169/Unknown  259s 215ms/step - loss: 1.6303 - mean_absolute_error: 0.9979

<div class="k-default-codeblock">
```

```
</div>
   1170/Unknown  259s 215ms/step - loss: 1.6300 - mean_absolute_error: 0.9979

<div class="k-default-codeblock">
```

```
</div>
   1171/Unknown  259s 216ms/step - loss: 1.6297 - mean_absolute_error: 0.9978

<div class="k-default-codeblock">
```

```
</div>
   1172/Unknown  259s 216ms/step - loss: 1.6293 - mean_absolute_error: 0.9977

<div class="k-default-codeblock">
```

```
</div>
   1173/Unknown  259s 216ms/step - loss: 1.6290 - mean_absolute_error: 0.9976

<div class="k-default-codeblock">
```

```
</div>
   1174/Unknown  260s 216ms/step - loss: 1.6287 - mean_absolute_error: 0.9975

<div class="k-default-codeblock">
```

```
</div>
   1175/Unknown  260s 216ms/step - loss: 1.6284 - mean_absolute_error: 0.9974

<div class="k-default-codeblock">
```

```
</div>
   1176/Unknown  260s 216ms/step - loss: 1.6281 - mean_absolute_error: 0.9973

<div class="k-default-codeblock">
```

```
</div>
   1177/Unknown  260s 216ms/step - loss: 1.6278 - mean_absolute_error: 0.9972

<div class="k-default-codeblock">
```

```
</div>
   1178/Unknown  261s 216ms/step - loss: 1.6275 - mean_absolute_error: 0.9971

<div class="k-default-codeblock">
```

```
</div>
   1179/Unknown  261s 216ms/step - loss: 1.6272 - mean_absolute_error: 0.9971

<div class="k-default-codeblock">
```

```
</div>
   1180/Unknown  261s 216ms/step - loss: 1.6269 - mean_absolute_error: 0.9970

<div class="k-default-codeblock">
```

```
</div>
   1181/Unknown  261s 216ms/step - loss: 1.6266 - mean_absolute_error: 0.9969

<div class="k-default-codeblock">
```

```
</div>
   1182/Unknown  262s 216ms/step - loss: 1.6263 - mean_absolute_error: 0.9968

<div class="k-default-codeblock">
```

```
</div>
   1183/Unknown  262s 216ms/step - loss: 1.6259 - mean_absolute_error: 0.9967

<div class="k-default-codeblock">
```

```
</div>
   1184/Unknown  262s 216ms/step - loss: 1.6256 - mean_absolute_error: 0.9966

<div class="k-default-codeblock">
```

```
</div>
   1185/Unknown  262s 216ms/step - loss: 1.6253 - mean_absolute_error: 0.9965

<div class="k-default-codeblock">
```

```
</div>
   1186/Unknown  263s 216ms/step - loss: 1.6250 - mean_absolute_error: 0.9964

<div class="k-default-codeblock">
```

```
</div>
   1187/Unknown  263s 216ms/step - loss: 1.6247 - mean_absolute_error: 0.9964

<div class="k-default-codeblock">
```

```
</div>
   1188/Unknown  263s 216ms/step - loss: 1.6244 - mean_absolute_error: 0.9963

<div class="k-default-codeblock">
```

```
</div>
   1189/Unknown  263s 216ms/step - loss: 1.6241 - mean_absolute_error: 0.9962

<div class="k-default-codeblock">
```

```
</div>
   1190/Unknown  264s 216ms/step - loss: 1.6238 - mean_absolute_error: 0.9961

<div class="k-default-codeblock">
```

```
</div>
   1191/Unknown  264s 216ms/step - loss: 1.6235 - mean_absolute_error: 0.9960

<div class="k-default-codeblock">
```

```
</div>
   1192/Unknown  264s 216ms/step - loss: 1.6232 - mean_absolute_error: 0.9959

<div class="k-default-codeblock">
```

```
</div>
   1193/Unknown  264s 216ms/step - loss: 1.6229 - mean_absolute_error: 0.9958

<div class="k-default-codeblock">
```

```
</div>
   1194/Unknown  264s 216ms/step - loss: 1.6226 - mean_absolute_error: 0.9957

<div class="k-default-codeblock">
```

```
</div>
   1195/Unknown  265s 216ms/step - loss: 1.6223 - mean_absolute_error: 0.9957

<div class="k-default-codeblock">
```

```
</div>
   1196/Unknown  265s 216ms/step - loss: 1.6220 - mean_absolute_error: 0.9956

<div class="k-default-codeblock">
```

```
</div>
   1197/Unknown  265s 216ms/step - loss: 1.6217 - mean_absolute_error: 0.9955

<div class="k-default-codeblock">
```

```
</div>
   1198/Unknown  265s 216ms/step - loss: 1.6214 - mean_absolute_error: 0.9954

<div class="k-default-codeblock">
```

```
</div>
   1199/Unknown  266s 216ms/step - loss: 1.6211 - mean_absolute_error: 0.9953

<div class="k-default-codeblock">
```

```
</div>
   1200/Unknown  266s 216ms/step - loss: 1.6208 - mean_absolute_error: 0.9952

<div class="k-default-codeblock">
```

```
</div>
   1201/Unknown  266s 216ms/step - loss: 1.6205 - mean_absolute_error: 0.9951

<div class="k-default-codeblock">
```

```
</div>
   1202/Unknown  267s 216ms/step - loss: 1.6202 - mean_absolute_error: 0.9951

<div class="k-default-codeblock">
```

```
</div>
   1203/Unknown  267s 216ms/step - loss: 1.6199 - mean_absolute_error: 0.9950

<div class="k-default-codeblock">
```

```
</div>
   1204/Unknown  267s 216ms/step - loss: 1.6196 - mean_absolute_error: 0.9949

<div class="k-default-codeblock">
```

```
</div>
   1205/Unknown  267s 216ms/step - loss: 1.6193 - mean_absolute_error: 0.9948

<div class="k-default-codeblock">
```

```
</div>
   1206/Unknown  267s 216ms/step - loss: 1.6190 - mean_absolute_error: 0.9947

<div class="k-default-codeblock">
```

```
</div>
   1207/Unknown  268s 216ms/step - loss: 1.6187 - mean_absolute_error: 0.9946

<div class="k-default-codeblock">
```

```
</div>
   1208/Unknown  268s 216ms/step - loss: 1.6184 - mean_absolute_error: 0.9945

<div class="k-default-codeblock">
```

```
</div>
   1209/Unknown  268s 216ms/step - loss: 1.6181 - mean_absolute_error: 0.9945

<div class="k-default-codeblock">
```

```
</div>
   1210/Unknown  268s 216ms/step - loss: 1.6178 - mean_absolute_error: 0.9944

<div class="k-default-codeblock">
```

```
</div>
   1211/Unknown  269s 216ms/step - loss: 1.6175 - mean_absolute_error: 0.9943

<div class="k-default-codeblock">
```

```
</div>
   1212/Unknown  269s 216ms/step - loss: 1.6172 - mean_absolute_error: 0.9942

<div class="k-default-codeblock">
```

```
</div>
   1213/Unknown  269s 216ms/step - loss: 1.6169 - mean_absolute_error: 0.9941

<div class="k-default-codeblock">
```

```
</div>
   1214/Unknown  269s 216ms/step - loss: 1.6166 - mean_absolute_error: 0.9940

<div class="k-default-codeblock">
```

```
</div>
   1215/Unknown  270s 216ms/step - loss: 1.6163 - mean_absolute_error: 0.9940

<div class="k-default-codeblock">
```

```
</div>
   1216/Unknown  270s 216ms/step - loss: 1.6160 - mean_absolute_error: 0.9939

<div class="k-default-codeblock">
```

```
</div>
   1217/Unknown  270s 216ms/step - loss: 1.6157 - mean_absolute_error: 0.9938

<div class="k-default-codeblock">
```

```
</div>
   1218/Unknown  270s 216ms/step - loss: 1.6155 - mean_absolute_error: 0.9937

<div class="k-default-codeblock">
```

```
</div>
   1219/Unknown  270s 216ms/step - loss: 1.6152 - mean_absolute_error: 0.9936

<div class="k-default-codeblock">
```

```
</div>
   1220/Unknown  271s 216ms/step - loss: 1.6149 - mean_absolute_error: 0.9935

<div class="k-default-codeblock">
```

```
</div>
   1221/Unknown  271s 216ms/step - loss: 1.6146 - mean_absolute_error: 0.9935

<div class="k-default-codeblock">
```

```
</div>
   1222/Unknown  271s 217ms/step - loss: 1.6143 - mean_absolute_error: 0.9934

<div class="k-default-codeblock">
```

```
</div>
   1223/Unknown  272s 217ms/step - loss: 1.6140 - mean_absolute_error: 0.9933

<div class="k-default-codeblock">
```

```
</div>
   1224/Unknown  272s 217ms/step - loss: 1.6137 - mean_absolute_error: 0.9932

<div class="k-default-codeblock">
```

```
</div>
   1225/Unknown  272s 217ms/step - loss: 1.6134 - mean_absolute_error: 0.9931

<div class="k-default-codeblock">
```

```
</div>
   1226/Unknown  272s 217ms/step - loss: 1.6131 - mean_absolute_error: 0.9930

<div class="k-default-codeblock">
```

```
</div>
   1227/Unknown  273s 217ms/step - loss: 1.6128 - mean_absolute_error: 0.9930

<div class="k-default-codeblock">
```

```
</div>
   1228/Unknown  273s 217ms/step - loss: 1.6125 - mean_absolute_error: 0.9929

<div class="k-default-codeblock">
```

```
</div>
   1229/Unknown  273s 217ms/step - loss: 1.6122 - mean_absolute_error: 0.9928

<div class="k-default-codeblock">
```

```
</div>
   1230/Unknown  273s 217ms/step - loss: 1.6120 - mean_absolute_error: 0.9927

<div class="k-default-codeblock">
```

```
</div>
   1231/Unknown  273s 217ms/step - loss: 1.6117 - mean_absolute_error: 0.9926

<div class="k-default-codeblock">
```

```
</div>
   1232/Unknown  274s 217ms/step - loss: 1.6114 - mean_absolute_error: 0.9925

<div class="k-default-codeblock">
```

```
</div>
   1233/Unknown  274s 217ms/step - loss: 1.6111 - mean_absolute_error: 0.9925

<div class="k-default-codeblock">
```

```
</div>
   1234/Unknown  274s 217ms/step - loss: 1.6108 - mean_absolute_error: 0.9924

<div class="k-default-codeblock">
```

```
</div>
   1235/Unknown  274s 217ms/step - loss: 1.6105 - mean_absolute_error: 0.9923

<div class="k-default-codeblock">
```

```
</div>
   1236/Unknown  274s 217ms/step - loss: 1.6102 - mean_absolute_error: 0.9922

<div class="k-default-codeblock">
```

```
</div>
   1237/Unknown  275s 217ms/step - loss: 1.6099 - mean_absolute_error: 0.9921

<div class="k-default-codeblock">
```

```
</div>
   1238/Unknown  275s 217ms/step - loss: 1.6097 - mean_absolute_error: 0.9920

<div class="k-default-codeblock">
```

```
</div>
   1239/Unknown  275s 217ms/step - loss: 1.6094 - mean_absolute_error: 0.9920

<div class="k-default-codeblock">
```

```
</div>
   1240/Unknown  275s 217ms/step - loss: 1.6091 - mean_absolute_error: 0.9919

<div class="k-default-codeblock">
```

```
</div>
   1241/Unknown  276s 217ms/step - loss: 1.6088 - mean_absolute_error: 0.9918

<div class="k-default-codeblock">
```

```
</div>
   1242/Unknown  276s 217ms/step - loss: 1.6085 - mean_absolute_error: 0.9917

<div class="k-default-codeblock">
```

```
</div>
   1243/Unknown  276s 217ms/step - loss: 1.6082 - mean_absolute_error: 0.9916

<div class="k-default-codeblock">
```

```
</div>
   1244/Unknown  276s 217ms/step - loss: 1.6080 - mean_absolute_error: 0.9916

<div class="k-default-codeblock">
```

```
</div>
   1245/Unknown  276s 217ms/step - loss: 1.6077 - mean_absolute_error: 0.9915

<div class="k-default-codeblock">
```

```
</div>
   1246/Unknown  277s 217ms/step - loss: 1.6074 - mean_absolute_error: 0.9914

<div class="k-default-codeblock">
```

```
</div>
   1247/Unknown  277s 217ms/step - loss: 1.6071 - mean_absolute_error: 0.9913

<div class="k-default-codeblock">
```

```
</div>
   1248/Unknown  277s 217ms/step - loss: 1.6068 - mean_absolute_error: 0.9912

<div class="k-default-codeblock">
```

```
</div>
   1249/Unknown  277s 217ms/step - loss: 1.6065 - mean_absolute_error: 0.9912

<div class="k-default-codeblock">
```

```
</div>
   1250/Unknown  278s 217ms/step - loss: 1.6063 - mean_absolute_error: 0.9911

<div class="k-default-codeblock">
```

```
</div>
   1251/Unknown  278s 217ms/step - loss: 1.6060 - mean_absolute_error: 0.9910

<div class="k-default-codeblock">
```

```
</div>
   1252/Unknown  278s 217ms/step - loss: 1.6057 - mean_absolute_error: 0.9909

<div class="k-default-codeblock">
```

```
</div>
   1253/Unknown  278s 217ms/step - loss: 1.6054 - mean_absolute_error: 0.9908

<div class="k-default-codeblock">
```

```
</div>
   1254/Unknown  279s 217ms/step - loss: 1.6051 - mean_absolute_error: 0.9908

<div class="k-default-codeblock">
```

```
</div>
   1255/Unknown  279s 217ms/step - loss: 1.6049 - mean_absolute_error: 0.9907

<div class="k-default-codeblock">
```

```
</div>
   1256/Unknown  279s 217ms/step - loss: 1.6046 - mean_absolute_error: 0.9906

<div class="k-default-codeblock">
```

```
</div>
   1257/Unknown  279s 217ms/step - loss: 1.6043 - mean_absolute_error: 0.9905

<div class="k-default-codeblock">
```

```
</div>
   1258/Unknown  279s 217ms/step - loss: 1.6040 - mean_absolute_error: 0.9904

<div class="k-default-codeblock">
```

```
</div>
   1259/Unknown  280s 217ms/step - loss: 1.6037 - mean_absolute_error: 0.9904

<div class="k-default-codeblock">
```

```
</div>
   1260/Unknown  280s 217ms/step - loss: 1.6035 - mean_absolute_error: 0.9903

<div class="k-default-codeblock">
```

```
</div>
   1261/Unknown  280s 217ms/step - loss: 1.6032 - mean_absolute_error: 0.9902

<div class="k-default-codeblock">
```

```
</div>
   1262/Unknown  280s 217ms/step - loss: 1.6029 - mean_absolute_error: 0.9901

<div class="k-default-codeblock">
```

```
</div>
   1263/Unknown  281s 217ms/step - loss: 1.6026 - mean_absolute_error: 0.9900

<div class="k-default-codeblock">
```

```
</div>
   1264/Unknown  281s 217ms/step - loss: 1.6024 - mean_absolute_error: 0.9900

<div class="k-default-codeblock">
```

```
</div>
   1265/Unknown  281s 217ms/step - loss: 1.6021 - mean_absolute_error: 0.9899

<div class="k-default-codeblock">
```

```
</div>
   1266/Unknown  281s 217ms/step - loss: 1.6018 - mean_absolute_error: 0.9898

<div class="k-default-codeblock">
```

```
</div>
   1267/Unknown  281s 217ms/step - loss: 1.6015 - mean_absolute_error: 0.9897

<div class="k-default-codeblock">
```

```
</div>
   1268/Unknown  282s 217ms/step - loss: 1.6013 - mean_absolute_error: 0.9896

<div class="k-default-codeblock">
```

```
</div>
   1269/Unknown  282s 217ms/step - loss: 1.6010 - mean_absolute_error: 0.9896

<div class="k-default-codeblock">
```

```
</div>
   1270/Unknown  282s 217ms/step - loss: 1.6007 - mean_absolute_error: 0.9895

<div class="k-default-codeblock">
```

```
</div>
   1271/Unknown  282s 217ms/step - loss: 1.6004 - mean_absolute_error: 0.9894

<div class="k-default-codeblock">
```

```
</div>
   1272/Unknown  283s 217ms/step - loss: 1.6002 - mean_absolute_error: 0.9893

<div class="k-default-codeblock">
```

```
</div>
   1273/Unknown  283s 217ms/step - loss: 1.5999 - mean_absolute_error: 0.9892

<div class="k-default-codeblock">
```

```
</div>
   1274/Unknown  283s 217ms/step - loss: 1.5996 - mean_absolute_error: 0.9892

<div class="k-default-codeblock">
```

```
</div>
   1275/Unknown  283s 217ms/step - loss: 1.5993 - mean_absolute_error: 0.9891

<div class="k-default-codeblock">
```

```
</div>
   1276/Unknown  283s 217ms/step - loss: 1.5991 - mean_absolute_error: 0.9890

<div class="k-default-codeblock">
```

```
</div>
   1277/Unknown  284s 217ms/step - loss: 1.5988 - mean_absolute_error: 0.9889

<div class="k-default-codeblock">
```

```
</div>
   1278/Unknown  284s 217ms/step - loss: 1.5985 - mean_absolute_error: 0.9889

<div class="k-default-codeblock">
```

```
</div>
   1279/Unknown  284s 217ms/step - loss: 1.5983 - mean_absolute_error: 0.9888

<div class="k-default-codeblock">
```

```
</div>
   1280/Unknown  284s 217ms/step - loss: 1.5980 - mean_absolute_error: 0.9887

<div class="k-default-codeblock">
```

```
</div>
   1281/Unknown  285s 217ms/step - loss: 1.5977 - mean_absolute_error: 0.9886

<div class="k-default-codeblock">
```

```
</div>
   1282/Unknown  285s 217ms/step - loss: 1.5974 - mean_absolute_error: 0.9885

<div class="k-default-codeblock">
```

```
</div>
   1283/Unknown  285s 217ms/step - loss: 1.5972 - mean_absolute_error: 0.9885

<div class="k-default-codeblock">
```

```
</div>
   1284/Unknown  285s 217ms/step - loss: 1.5969 - mean_absolute_error: 0.9884

<div class="k-default-codeblock">
```

```
</div>
   1285/Unknown  285s 217ms/step - loss: 1.5966 - mean_absolute_error: 0.9883

<div class="k-default-codeblock">
```

```
</div>
   1286/Unknown  286s 217ms/step - loss: 1.5964 - mean_absolute_error: 0.9882

<div class="k-default-codeblock">
```

```
</div>
   1287/Unknown  286s 217ms/step - loss: 1.5961 - mean_absolute_error: 0.9882

<div class="k-default-codeblock">
```

```
</div>
   1288/Unknown  286s 217ms/step - loss: 1.5958 - mean_absolute_error: 0.9881

<div class="k-default-codeblock">
```

```
</div>
   1289/Unknown  286s 217ms/step - loss: 1.5956 - mean_absolute_error: 0.9880

<div class="k-default-codeblock">
```

```
</div>
   1290/Unknown  286s 217ms/step - loss: 1.5953 - mean_absolute_error: 0.9879

<div class="k-default-codeblock">
```

```
</div>
   1291/Unknown  287s 217ms/step - loss: 1.5950 - mean_absolute_error: 0.9879

<div class="k-default-codeblock">
```

```
</div>
   1292/Unknown  287s 217ms/step - loss: 1.5948 - mean_absolute_error: 0.9878

<div class="k-default-codeblock">
```

```
</div>
   1293/Unknown  287s 217ms/step - loss: 1.5945 - mean_absolute_error: 0.9877

<div class="k-default-codeblock">
```

```
</div>
   1294/Unknown  287s 217ms/step - loss: 1.5942 - mean_absolute_error: 0.9876

<div class="k-default-codeblock">
```

```
</div>
   1295/Unknown  288s 217ms/step - loss: 1.5940 - mean_absolute_error: 0.9875

<div class="k-default-codeblock">
```

```
</div>
   1296/Unknown  288s 217ms/step - loss: 1.5937 - mean_absolute_error: 0.9875

<div class="k-default-codeblock">
```

```
</div>
   1297/Unknown  288s 217ms/step - loss: 1.5934 - mean_absolute_error: 0.9874

<div class="k-default-codeblock">
```

```
</div>
   1298/Unknown  288s 217ms/step - loss: 1.5932 - mean_absolute_error: 0.9873

<div class="k-default-codeblock">
```

```
</div>
   1299/Unknown  288s 217ms/step - loss: 1.5929 - mean_absolute_error: 0.9872

<div class="k-default-codeblock">
```

```
</div>
   1300/Unknown  289s 217ms/step - loss: 1.5926 - mean_absolute_error: 0.9872

<div class="k-default-codeblock">
```

```
</div>
   1301/Unknown  289s 217ms/step - loss: 1.5924 - mean_absolute_error: 0.9871

<div class="k-default-codeblock">
```

```
</div>
   1302/Unknown  289s 217ms/step - loss: 1.5921 - mean_absolute_error: 0.9870

<div class="k-default-codeblock">
```

```
</div>
   1303/Unknown  289s 217ms/step - loss: 1.5918 - mean_absolute_error: 0.9869

<div class="k-default-codeblock">
```

```
</div>
   1304/Unknown  289s 217ms/step - loss: 1.5916 - mean_absolute_error: 0.9869

<div class="k-default-codeblock">
```

```
</div>
   1305/Unknown  290s 217ms/step - loss: 1.5913 - mean_absolute_error: 0.9868

<div class="k-default-codeblock">
```

```
</div>
   1306/Unknown  290s 217ms/step - loss: 1.5910 - mean_absolute_error: 0.9867

<div class="k-default-codeblock">
```

```
</div>
   1307/Unknown  290s 217ms/step - loss: 1.5908 - mean_absolute_error: 0.9866

<div class="k-default-codeblock">
```

```
</div>
   1308/Unknown  290s 217ms/step - loss: 1.5905 - mean_absolute_error: 0.9866

<div class="k-default-codeblock">
```

```
</div>
   1309/Unknown  290s 217ms/step - loss: 1.5903 - mean_absolute_error: 0.9865

<div class="k-default-codeblock">
```

```
</div>
   1310/Unknown  291s 217ms/step - loss: 1.5900 - mean_absolute_error: 0.9864

<div class="k-default-codeblock">
```

```
</div>
   1311/Unknown  291s 217ms/step - loss: 1.5897 - mean_absolute_error: 0.9863

<div class="k-default-codeblock">
```

```
</div>
   1312/Unknown  291s 217ms/step - loss: 1.5895 - mean_absolute_error: 0.9863

<div class="k-default-codeblock">
```

```
</div>
   1313/Unknown  291s 217ms/step - loss: 1.5892 - mean_absolute_error: 0.9862

<div class="k-default-codeblock">
```

```
</div>
   1314/Unknown  291s 217ms/step - loss: 1.5890 - mean_absolute_error: 0.9861

<div class="k-default-codeblock">
```

```
</div>
   1315/Unknown  292s 217ms/step - loss: 1.5887 - mean_absolute_error: 0.9860

<div class="k-default-codeblock">
```

```
</div>
   1316/Unknown  292s 217ms/step - loss: 1.5884 - mean_absolute_error: 0.9860

<div class="k-default-codeblock">
```

```
</div>
   1317/Unknown  292s 217ms/step - loss: 1.5882 - mean_absolute_error: 0.9859

<div class="k-default-codeblock">
```

```
</div>
   1318/Unknown  292s 217ms/step - loss: 1.5879 - mean_absolute_error: 0.9858

<div class="k-default-codeblock">
```

```
</div>
   1319/Unknown  293s 217ms/step - loss: 1.5877 - mean_absolute_error: 0.9857

<div class="k-default-codeblock">
```

```
</div>
   1320/Unknown  293s 217ms/step - loss: 1.5874 - mean_absolute_error: 0.9857

<div class="k-default-codeblock">
```

```
</div>
   1321/Unknown  293s 217ms/step - loss: 1.5871 - mean_absolute_error: 0.9856

<div class="k-default-codeblock">
```

```
</div>
   1322/Unknown  293s 217ms/step - loss: 1.5869 - mean_absolute_error: 0.9855

<div class="k-default-codeblock">
```

```
</div>
   1323/Unknown  293s 217ms/step - loss: 1.5866 - mean_absolute_error: 0.9854

<div class="k-default-codeblock">
```

```
</div>
   1324/Unknown  294s 217ms/step - loss: 1.5864 - mean_absolute_error: 0.9854

<div class="k-default-codeblock">
```

```
</div>
   1325/Unknown  294s 217ms/step - loss: 1.5861 - mean_absolute_error: 0.9853

<div class="k-default-codeblock">
```

```
</div>
   1326/Unknown  294s 217ms/step - loss: 1.5858 - mean_absolute_error: 0.9852

<div class="k-default-codeblock">
```

```
</div>
   1327/Unknown  294s 217ms/step - loss: 1.5856 - mean_absolute_error: 0.9851

<div class="k-default-codeblock">
```

```
</div>
   1328/Unknown  295s 217ms/step - loss: 1.5853 - mean_absolute_error: 0.9851

<div class="k-default-codeblock">
```

```
</div>
   1329/Unknown  295s 217ms/step - loss: 1.5851 - mean_absolute_error: 0.9850

<div class="k-default-codeblock">
```

```
</div>
   1330/Unknown  295s 217ms/step - loss: 1.5848 - mean_absolute_error: 0.9849

<div class="k-default-codeblock">
```

```
</div>
   1331/Unknown  295s 217ms/step - loss: 1.5846 - mean_absolute_error: 0.9848

<div class="k-default-codeblock">
```

```
</div>
   1332/Unknown  295s 217ms/step - loss: 1.5843 - mean_absolute_error: 0.9848

<div class="k-default-codeblock">
```

```
</div>
   1333/Unknown  296s 217ms/step - loss: 1.5841 - mean_absolute_error: 0.9847

<div class="k-default-codeblock">
```

```
</div>
   1334/Unknown  296s 217ms/step - loss: 1.5838 - mean_absolute_error: 0.9846

<div class="k-default-codeblock">
```

```
</div>
   1335/Unknown  296s 217ms/step - loss: 1.5836 - mean_absolute_error: 0.9846

<div class="k-default-codeblock">
```

```
</div>
   1336/Unknown  296s 217ms/step - loss: 1.5833 - mean_absolute_error: 0.9845

<div class="k-default-codeblock">
```

```
</div>
   1337/Unknown  297s 217ms/step - loss: 1.5830 - mean_absolute_error: 0.9844

<div class="k-default-codeblock">
```

```
</div>
   1338/Unknown  297s 217ms/step - loss: 1.5828 - mean_absolute_error: 0.9843

<div class="k-default-codeblock">
```

```
</div>
   1339/Unknown  297s 217ms/step - loss: 1.5825 - mean_absolute_error: 0.9843

<div class="k-default-codeblock">
```

```
</div>
   1340/Unknown  297s 217ms/step - loss: 1.5823 - mean_absolute_error: 0.9842

<div class="k-default-codeblock">
```

```
</div>
   1341/Unknown  297s 217ms/step - loss: 1.5820 - mean_absolute_error: 0.9841

<div class="k-default-codeblock">
```

```
</div>
   1342/Unknown  297s 217ms/step - loss: 1.5818 - mean_absolute_error: 0.9840

<div class="k-default-codeblock">
```

```
</div>
   1343/Unknown  298s 217ms/step - loss: 1.5815 - mean_absolute_error: 0.9840

<div class="k-default-codeblock">
```

```
</div>
   1344/Unknown  298s 217ms/step - loss: 1.5813 - mean_absolute_error: 0.9839

<div class="k-default-codeblock">
```

```
</div>
   1345/Unknown  298s 217ms/step - loss: 1.5810 - mean_absolute_error: 0.9838

<div class="k-default-codeblock">
```

```
</div>
   1346/Unknown  298s 217ms/step - loss: 1.5808 - mean_absolute_error: 0.9838

<div class="k-default-codeblock">
```

```
</div>
   1347/Unknown  298s 217ms/step - loss: 1.5805 - mean_absolute_error: 0.9837

<div class="k-default-codeblock">
```

```
</div>
   1348/Unknown  299s 217ms/step - loss: 1.5803 - mean_absolute_error: 0.9836

<div class="k-default-codeblock">
```

```
</div>
   1349/Unknown  299s 217ms/step - loss: 1.5800 - mean_absolute_error: 0.9835

<div class="k-default-codeblock">
```

```
</div>
   1350/Unknown  299s 217ms/step - loss: 1.5798 - mean_absolute_error: 0.9835

<div class="k-default-codeblock">
```

```
</div>
   1351/Unknown  299s 217ms/step - loss: 1.5795 - mean_absolute_error: 0.9834

<div class="k-default-codeblock">
```

```
</div>
   1352/Unknown  300s 217ms/step - loss: 1.5793 - mean_absolute_error: 0.9833

<div class="k-default-codeblock">
```

```
</div>
   1353/Unknown  300s 217ms/step - loss: 1.5790 - mean_absolute_error: 0.9833

<div class="k-default-codeblock">
```

```
</div>
   1354/Unknown  300s 217ms/step - loss: 1.5788 - mean_absolute_error: 0.9832

<div class="k-default-codeblock">
```

```
</div>
   1355/Unknown  300s 217ms/step - loss: 1.5785 - mean_absolute_error: 0.9831

<div class="k-default-codeblock">
```

```
</div>
   1356/Unknown  300s 217ms/step - loss: 1.5783 - mean_absolute_error: 0.9830

<div class="k-default-codeblock">
```

```
</div>
   1357/Unknown  301s 217ms/step - loss: 1.5781 - mean_absolute_error: 0.9830

<div class="k-default-codeblock">
```

```
</div>
   1358/Unknown  301s 217ms/step - loss: 1.5778 - mean_absolute_error: 0.9829

<div class="k-default-codeblock">
```

```
</div>
   1359/Unknown  301s 217ms/step - loss: 1.5776 - mean_absolute_error: 0.9828

<div class="k-default-codeblock">
```

```
</div>
   1360/Unknown  301s 217ms/step - loss: 1.5773 - mean_absolute_error: 0.9828

<div class="k-default-codeblock">
```

```
</div>
   1361/Unknown  301s 217ms/step - loss: 1.5771 - mean_absolute_error: 0.9827

<div class="k-default-codeblock">
```

```
</div>
   1362/Unknown  302s 217ms/step - loss: 1.5768 - mean_absolute_error: 0.9826

<div class="k-default-codeblock">
```

```
</div>
   1363/Unknown  302s 217ms/step - loss: 1.5766 - mean_absolute_error: 0.9825

<div class="k-default-codeblock">
```

```
</div>
   1364/Unknown  302s 217ms/step - loss: 1.5763 - mean_absolute_error: 0.9825

<div class="k-default-codeblock">
```

```
</div>
   1365/Unknown  302s 217ms/step - loss: 1.5761 - mean_absolute_error: 0.9824

<div class="k-default-codeblock">
```

```
</div>
   1366/Unknown  303s 217ms/step - loss: 1.5759 - mean_absolute_error: 0.9823

<div class="k-default-codeblock">
```

```
</div>
   1367/Unknown  303s 217ms/step - loss: 1.5756 - mean_absolute_error: 0.9823

<div class="k-default-codeblock">
```

```
</div>
   1368/Unknown  303s 217ms/step - loss: 1.5754 - mean_absolute_error: 0.9822

<div class="k-default-codeblock">
```

```
</div>
   1369/Unknown  303s 217ms/step - loss: 1.5751 - mean_absolute_error: 0.9821

<div class="k-default-codeblock">
```

```
</div>
   1370/Unknown  304s 217ms/step - loss: 1.5749 - mean_absolute_error: 0.9821

<div class="k-default-codeblock">
```

```
</div>
   1371/Unknown  304s 217ms/step - loss: 1.5746 - mean_absolute_error: 0.9820

<div class="k-default-codeblock">
```

```
</div>
   1372/Unknown  304s 217ms/step - loss: 1.5744 - mean_absolute_error: 0.9819

<div class="k-default-codeblock">
```

```
</div>
   1373/Unknown  304s 217ms/step - loss: 1.5742 - mean_absolute_error: 0.9818

<div class="k-default-codeblock">
```

```
</div>
   1374/Unknown  304s 217ms/step - loss: 1.5739 - mean_absolute_error: 0.9818

<div class="k-default-codeblock">
```

```
</div>
   1375/Unknown  305s 217ms/step - loss: 1.5737 - mean_absolute_error: 0.9817

<div class="k-default-codeblock">
```

```
</div>
   1376/Unknown  305s 217ms/step - loss: 1.5734 - mean_absolute_error: 0.9816

<div class="k-default-codeblock">
```

```
</div>
   1377/Unknown  305s 217ms/step - loss: 1.5732 - mean_absolute_error: 0.9816

<div class="k-default-codeblock">
```

```
</div>
   1378/Unknown  305s 217ms/step - loss: 1.5729 - mean_absolute_error: 0.9815

<div class="k-default-codeblock">
```

```
</div>
   1379/Unknown  305s 217ms/step - loss: 1.5727 - mean_absolute_error: 0.9814

<div class="k-default-codeblock">
```

```
</div>
   1380/Unknown  306s 217ms/step - loss: 1.5725 - mean_absolute_error: 0.9814

<div class="k-default-codeblock">
```

```
</div>
   1381/Unknown  306s 217ms/step - loss: 1.5722 - mean_absolute_error: 0.9813

<div class="k-default-codeblock">
```

```
</div>
   1382/Unknown  306s 217ms/step - loss: 1.5720 - mean_absolute_error: 0.9812

<div class="k-default-codeblock">
```

```
</div>
   1383/Unknown  306s 217ms/step - loss: 1.5718 - mean_absolute_error: 0.9812

<div class="k-default-codeblock">
```

```
</div>
   1384/Unknown  306s 217ms/step - loss: 1.5715 - mean_absolute_error: 0.9811

<div class="k-default-codeblock">
```

```
</div>
   1385/Unknown  307s 217ms/step - loss: 1.5713 - mean_absolute_error: 0.9810

<div class="k-default-codeblock">
```

```
</div>
   1386/Unknown  307s 217ms/step - loss: 1.5710 - mean_absolute_error: 0.9809

<div class="k-default-codeblock">
```

```
</div>
   1387/Unknown  307s 217ms/step - loss: 1.5708 - mean_absolute_error: 0.9809

<div class="k-default-codeblock">
```

```
</div>
   1388/Unknown  307s 217ms/step - loss: 1.5706 - mean_absolute_error: 0.9808

<div class="k-default-codeblock">
```

```
</div>
   1389/Unknown  307s 217ms/step - loss: 1.5703 - mean_absolute_error: 0.9807

<div class="k-default-codeblock">
```

```
</div>
   1390/Unknown  308s 217ms/step - loss: 1.5701 - mean_absolute_error: 0.9807

<div class="k-default-codeblock">
```

```
</div>
   1391/Unknown  308s 217ms/step - loss: 1.5698 - mean_absolute_error: 0.9806

<div class="k-default-codeblock">
```

```
</div>
   1392/Unknown  308s 217ms/step - loss: 1.5696 - mean_absolute_error: 0.9805

<div class="k-default-codeblock">
```

```
</div>
   1393/Unknown  308s 217ms/step - loss: 1.5694 - mean_absolute_error: 0.9805

<div class="k-default-codeblock">
```

```
</div>
   1394/Unknown  309s 217ms/step - loss: 1.5691 - mean_absolute_error: 0.9804

<div class="k-default-codeblock">
```

```
</div>
   1395/Unknown  309s 217ms/step - loss: 1.5689 - mean_absolute_error: 0.9803

<div class="k-default-codeblock">
```

```
</div>
   1396/Unknown  309s 217ms/step - loss: 1.5687 - mean_absolute_error: 0.9803

<div class="k-default-codeblock">
```

```
</div>
   1397/Unknown  309s 217ms/step - loss: 1.5684 - mean_absolute_error: 0.9802

<div class="k-default-codeblock">
```

```
</div>
   1398/Unknown  309s 217ms/step - loss: 1.5682 - mean_absolute_error: 0.9801

<div class="k-default-codeblock">
```

```
</div>
   1399/Unknown  310s 217ms/step - loss: 1.5680 - mean_absolute_error: 0.9801

<div class="k-default-codeblock">
```

```
</div>
   1400/Unknown  310s 217ms/step - loss: 1.5677 - mean_absolute_error: 0.9800

<div class="k-default-codeblock">
```

```
</div>
   1401/Unknown  310s 217ms/step - loss: 1.5675 - mean_absolute_error: 0.9799

<div class="k-default-codeblock">
```

```
</div>
   1402/Unknown  310s 217ms/step - loss: 1.5673 - mean_absolute_error: 0.9799

<div class="k-default-codeblock">
```

```
</div>
   1403/Unknown  311s 217ms/step - loss: 1.5670 - mean_absolute_error: 0.9798

<div class="k-default-codeblock">
```

```
</div>
   1404/Unknown  311s 217ms/step - loss: 1.5668 - mean_absolute_error: 0.9797

<div class="k-default-codeblock">
```

```
</div>
   1405/Unknown  311s 217ms/step - loss: 1.5666 - mean_absolute_error: 0.9797

<div class="k-default-codeblock">
```

```
</div>
   1406/Unknown  311s 217ms/step - loss: 1.5663 - mean_absolute_error: 0.9796

<div class="k-default-codeblock">
```

```
</div>
   1407/Unknown  311s 217ms/step - loss: 1.5661 - mean_absolute_error: 0.9795

<div class="k-default-codeblock">
```

```
</div>
   1408/Unknown  312s 217ms/step - loss: 1.5659 - mean_absolute_error: 0.9795

<div class="k-default-codeblock">
```

```
</div>
   1409/Unknown  312s 217ms/step - loss: 1.5656 - mean_absolute_error: 0.9794

<div class="k-default-codeblock">
```

```
</div>
   1410/Unknown  312s 217ms/step - loss: 1.5654 - mean_absolute_error: 0.9793

<div class="k-default-codeblock">
```

```
</div>
   1411/Unknown  312s 217ms/step - loss: 1.5652 - mean_absolute_error: 0.9793

<div class="k-default-codeblock">
```

```
</div>
   1412/Unknown  313s 217ms/step - loss: 1.5649 - mean_absolute_error: 0.9792

<div class="k-default-codeblock">
```

```
</div>
   1413/Unknown  313s 217ms/step - loss: 1.5647 - mean_absolute_error: 0.9791

<div class="k-default-codeblock">
```

```
</div>
   1414/Unknown  313s 217ms/step - loss: 1.5645 - mean_absolute_error: 0.9791

<div class="k-default-codeblock">
```

```
</div>
   1415/Unknown  313s 217ms/step - loss: 1.5642 - mean_absolute_error: 0.9790

<div class="k-default-codeblock">
```

```
</div>
   1416/Unknown  313s 217ms/step - loss: 1.5640 - mean_absolute_error: 0.9789

<div class="k-default-codeblock">
```

```
</div>
   1417/Unknown  314s 217ms/step - loss: 1.5638 - mean_absolute_error: 0.9789

<div class="k-default-codeblock">
```

```
</div>
   1418/Unknown  314s 217ms/step - loss: 1.5635 - mean_absolute_error: 0.9788

<div class="k-default-codeblock">
```

```
</div>
   1419/Unknown  314s 217ms/step - loss: 1.5633 - mean_absolute_error: 0.9787

<div class="k-default-codeblock">
```

```
</div>
   1420/Unknown  314s 217ms/step - loss: 1.5631 - mean_absolute_error: 0.9787

<div class="k-default-codeblock">
```

```
</div>
   1421/Unknown  315s 217ms/step - loss: 1.5629 - mean_absolute_error: 0.9786

<div class="k-default-codeblock">
```

```
</div>
   1422/Unknown  315s 217ms/step - loss: 1.5626 - mean_absolute_error: 0.9785

<div class="k-default-codeblock">
```

```
</div>
   1423/Unknown  315s 217ms/step - loss: 1.5624 - mean_absolute_error: 0.9785

<div class="k-default-codeblock">
```

```
</div>
   1424/Unknown  316s 217ms/step - loss: 1.5622 - mean_absolute_error: 0.9784

<div class="k-default-codeblock">
```

```
</div>
   1425/Unknown  316s 217ms/step - loss: 1.5619 - mean_absolute_error: 0.9783

<div class="k-default-codeblock">
```

```
</div>
   1426/Unknown  316s 217ms/step - loss: 1.5617 - mean_absolute_error: 0.9783

<div class="k-default-codeblock">
```

```
</div>
   1427/Unknown  316s 217ms/step - loss: 1.5615 - mean_absolute_error: 0.9782

<div class="k-default-codeblock">
```

```
</div>
   1428/Unknown  317s 217ms/step - loss: 1.5613 - mean_absolute_error: 0.9781

<div class="k-default-codeblock">
```

```
</div>
   1429/Unknown  317s 217ms/step - loss: 1.5610 - mean_absolute_error: 0.9781

<div class="k-default-codeblock">
```

```
</div>
   1430/Unknown  317s 217ms/step - loss: 1.5608 - mean_absolute_error: 0.9780

<div class="k-default-codeblock">
```

```
</div>
   1431/Unknown  317s 217ms/step - loss: 1.5606 - mean_absolute_error: 0.9779

<div class="k-default-codeblock">
```

```
</div>
   1432/Unknown  317s 217ms/step - loss: 1.5604 - mean_absolute_error: 0.9779

<div class="k-default-codeblock">
```

```
</div>
   1433/Unknown  318s 217ms/step - loss: 1.5601 - mean_absolute_error: 0.9778

<div class="k-default-codeblock">
```

```
</div>
   1434/Unknown  318s 217ms/step - loss: 1.5599 - mean_absolute_error: 0.9777

<div class="k-default-codeblock">
```

```
</div>
   1435/Unknown  318s 217ms/step - loss: 1.5597 - mean_absolute_error: 0.9777

<div class="k-default-codeblock">
```

```
</div>
   1436/Unknown  318s 217ms/step - loss: 1.5595 - mean_absolute_error: 0.9776

<div class="k-default-codeblock">
```

```
</div>
   1437/Unknown  318s 217ms/step - loss: 1.5592 - mean_absolute_error: 0.9775

<div class="k-default-codeblock">
```

```
</div>
   1438/Unknown  319s 217ms/step - loss: 1.5590 - mean_absolute_error: 0.9775

<div class="k-default-codeblock">
```

```
</div>
   1439/Unknown  319s 217ms/step - loss: 1.5588 - mean_absolute_error: 0.9774

<div class="k-default-codeblock">
```

```
</div>
   1440/Unknown  319s 217ms/step - loss: 1.5586 - mean_absolute_error: 0.9773

<div class="k-default-codeblock">
```

```
</div>
   1441/Unknown  319s 217ms/step - loss: 1.5583 - mean_absolute_error: 0.9773

<div class="k-default-codeblock">
```

```
</div>
   1442/Unknown  320s 217ms/step - loss: 1.5581 - mean_absolute_error: 0.9772

<div class="k-default-codeblock">
```

```
</div>
   1443/Unknown  320s 217ms/step - loss: 1.5579 - mean_absolute_error: 0.9771

<div class="k-default-codeblock">
```

```
</div>
   1444/Unknown  320s 217ms/step - loss: 1.5577 - mean_absolute_error: 0.9771

<div class="k-default-codeblock">
```

```
</div>
   1445/Unknown  320s 217ms/step - loss: 1.5574 - mean_absolute_error: 0.9770

<div class="k-default-codeblock">
```

```
</div>
   1446/Unknown  320s 217ms/step - loss: 1.5572 - mean_absolute_error: 0.9770

<div class="k-default-codeblock">
```

```
</div>
   1447/Unknown  321s 217ms/step - loss: 1.5570 - mean_absolute_error: 0.9769

<div class="k-default-codeblock">
```

```
</div>
   1448/Unknown  321s 217ms/step - loss: 1.5568 - mean_absolute_error: 0.9768

<div class="k-default-codeblock">
```

```
</div>
   1449/Unknown  321s 217ms/step - loss: 1.5565 - mean_absolute_error: 0.9768

<div class="k-default-codeblock">
```

```
</div>
   1450/Unknown  321s 217ms/step - loss: 1.5563 - mean_absolute_error: 0.9767

<div class="k-default-codeblock">
```

```
</div>
   1451/Unknown  322s 217ms/step - loss: 1.5561 - mean_absolute_error: 0.9766

<div class="k-default-codeblock">
```

```
</div>
   1452/Unknown  322s 217ms/step - loss: 1.5559 - mean_absolute_error: 0.9766

<div class="k-default-codeblock">
```

```
</div>
   1453/Unknown  322s 217ms/step - loss: 1.5557 - mean_absolute_error: 0.9765

<div class="k-default-codeblock">
```

```
</div>
   1454/Unknown  322s 217ms/step - loss: 1.5554 - mean_absolute_error: 0.9764

<div class="k-default-codeblock">
```

```
</div>
   1455/Unknown  323s 217ms/step - loss: 1.5552 - mean_absolute_error: 0.9764

<div class="k-default-codeblock">
```

```
</div>
   1456/Unknown  323s 217ms/step - loss: 1.5550 - mean_absolute_error: 0.9763

<div class="k-default-codeblock">
```

```
</div>
   1457/Unknown  323s 217ms/step - loss: 1.5548 - mean_absolute_error: 0.9763

<div class="k-default-codeblock">
```

```
</div>
   1458/Unknown  323s 217ms/step - loss: 1.5546 - mean_absolute_error: 0.9762

<div class="k-default-codeblock">
```

```
</div>
   1459/Unknown  324s 217ms/step - loss: 1.5543 - mean_absolute_error: 0.9761

<div class="k-default-codeblock">
```

```
</div>
   1460/Unknown  324s 217ms/step - loss: 1.5541 - mean_absolute_error: 0.9761

<div class="k-default-codeblock">
```

```
</div>
   1461/Unknown  324s 217ms/step - loss: 1.5539 - mean_absolute_error: 0.9760

<div class="k-default-codeblock">
```

```
</div>
   1462/Unknown  324s 217ms/step - loss: 1.5537 - mean_absolute_error: 0.9759

<div class="k-default-codeblock">
```

```
</div>
   1463/Unknown  324s 217ms/step - loss: 1.5535 - mean_absolute_error: 0.9759

<div class="k-default-codeblock">
```

```
</div>
   1464/Unknown  325s 217ms/step - loss: 1.5532 - mean_absolute_error: 0.9758

<div class="k-default-codeblock">
```

```
</div>
   1465/Unknown  325s 217ms/step - loss: 1.5530 - mean_absolute_error: 0.9757

<div class="k-default-codeblock">
```

```
</div>
   1466/Unknown  325s 217ms/step - loss: 1.5528 - mean_absolute_error: 0.9757

<div class="k-default-codeblock">
```

```
</div>
   1467/Unknown  325s 217ms/step - loss: 1.5526 - mean_absolute_error: 0.9756

<div class="k-default-codeblock">
```

```
</div>
   1468/Unknown  326s 217ms/step - loss: 1.5524 - mean_absolute_error: 0.9756

<div class="k-default-codeblock">
```

```
</div>
   1469/Unknown  326s 217ms/step - loss: 1.5522 - mean_absolute_error: 0.9755

<div class="k-default-codeblock">
```

```
</div>
   1470/Unknown  326s 217ms/step - loss: 1.5519 - mean_absolute_error: 0.9754

<div class="k-default-codeblock">
```

```
</div>
   1471/Unknown  326s 217ms/step - loss: 1.5517 - mean_absolute_error: 0.9754

<div class="k-default-codeblock">
```

```
</div>
   1472/Unknown  327s 217ms/step - loss: 1.5515 - mean_absolute_error: 0.9753

<div class="k-default-codeblock">
```

```
</div>
   1473/Unknown  327s 217ms/step - loss: 1.5513 - mean_absolute_error: 0.9752

<div class="k-default-codeblock">
```

```
</div>
   1474/Unknown  327s 217ms/step - loss: 1.5511 - mean_absolute_error: 0.9752

<div class="k-default-codeblock">
```

```
</div>
   1475/Unknown  327s 217ms/step - loss: 1.5509 - mean_absolute_error: 0.9751

<div class="k-default-codeblock">
```

```
</div>
   1476/Unknown  328s 218ms/step - loss: 1.5506 - mean_absolute_error: 0.9751

<div class="k-default-codeblock">
```

```
</div>
   1477/Unknown  328s 218ms/step - loss: 1.5504 - mean_absolute_error: 0.9750

<div class="k-default-codeblock">
```

```
</div>
   1478/Unknown  328s 218ms/step - loss: 1.5502 - mean_absolute_error: 0.9749

<div class="k-default-codeblock">
```

```
</div>
   1479/Unknown  329s 218ms/step - loss: 1.5500 - mean_absolute_error: 0.9749

<div class="k-default-codeblock">
```

```
</div>
   1480/Unknown  329s 218ms/step - loss: 1.5498 - mean_absolute_error: 0.9748

<div class="k-default-codeblock">
```

```
</div>
   1481/Unknown  329s 218ms/step - loss: 1.5496 - mean_absolute_error: 0.9747

<div class="k-default-codeblock">
```

```
</div>
   1482/Unknown  329s 218ms/step - loss: 1.5493 - mean_absolute_error: 0.9747

<div class="k-default-codeblock">
```

```
</div>
   1483/Unknown  330s 218ms/step - loss: 1.5491 - mean_absolute_error: 0.9746

<div class="k-default-codeblock">
```

```
</div>
   1484/Unknown  330s 218ms/step - loss: 1.5489 - mean_absolute_error: 0.9746

<div class="k-default-codeblock">
```

```
</div>
   1485/Unknown  330s 218ms/step - loss: 1.5487 - mean_absolute_error: 0.9745

<div class="k-default-codeblock">
```

```
</div>
   1486/Unknown  330s 218ms/step - loss: 1.5485 - mean_absolute_error: 0.9744

<div class="k-default-codeblock">
```

```
</div>
   1487/Unknown  330s 218ms/step - loss: 1.5483 - mean_absolute_error: 0.9744

<div class="k-default-codeblock">
```

```
</div>
   1488/Unknown  331s 218ms/step - loss: 1.5481 - mean_absolute_error: 0.9743

<div class="k-default-codeblock">
```

```
</div>
   1489/Unknown  331s 218ms/step - loss: 1.5478 - mean_absolute_error: 0.9742

<div class="k-default-codeblock">
```

```
</div>
   1490/Unknown  331s 218ms/step - loss: 1.5476 - mean_absolute_error: 0.9742

<div class="k-default-codeblock">
```

```
</div>
   1491/Unknown  331s 218ms/step - loss: 1.5474 - mean_absolute_error: 0.9741

<div class="k-default-codeblock">
```

```
</div>
   1492/Unknown  332s 218ms/step - loss: 1.5472 - mean_absolute_error: 0.9741

<div class="k-default-codeblock">
```

```
</div>
   1493/Unknown  332s 218ms/step - loss: 1.5470 - mean_absolute_error: 0.9740

<div class="k-default-codeblock">
```

```
</div>
   1494/Unknown  332s 218ms/step - loss: 1.5468 - mean_absolute_error: 0.9739

<div class="k-default-codeblock">
```

```
</div>
   1495/Unknown  332s 218ms/step - loss: 1.5466 - mean_absolute_error: 0.9739

<div class="k-default-codeblock">
```

```
</div>
   1496/Unknown  333s 218ms/step - loss: 1.5464 - mean_absolute_error: 0.9738

<div class="k-default-codeblock">
```

```
</div>
   1497/Unknown  333s 218ms/step - loss: 1.5461 - mean_absolute_error: 0.9737

<div class="k-default-codeblock">
```

```
</div>
   1498/Unknown  333s 218ms/step - loss: 1.5459 - mean_absolute_error: 0.9737

<div class="k-default-codeblock">
```

```
</div>
   1499/Unknown  333s 218ms/step - loss: 1.5457 - mean_absolute_error: 0.9736

<div class="k-default-codeblock">
```

```
</div>
   1500/Unknown  334s 218ms/step - loss: 1.5455 - mean_absolute_error: 0.9736

<div class="k-default-codeblock">
```

```
</div>
   1501/Unknown  334s 218ms/step - loss: 1.5453 - mean_absolute_error: 0.9735

<div class="k-default-codeblock">
```

```
</div>
   1502/Unknown  334s 218ms/step - loss: 1.5451 - mean_absolute_error: 0.9734

<div class="k-default-codeblock">
```

```
</div>
   1503/Unknown  334s 218ms/step - loss: 1.5449 - mean_absolute_error: 0.9734

<div class="k-default-codeblock">
```

```
</div>
   1504/Unknown  334s 218ms/step - loss: 1.5447 - mean_absolute_error: 0.9733

<div class="k-default-codeblock">
```

```
</div>
   1505/Unknown  335s 218ms/step - loss: 1.5445 - mean_absolute_error: 0.9733

<div class="k-default-codeblock">
```

```
</div>
   1506/Unknown  335s 218ms/step - loss: 1.5443 - mean_absolute_error: 0.9732

<div class="k-default-codeblock">
```

```
</div>
   1507/Unknown  335s 218ms/step - loss: 1.5440 - mean_absolute_error: 0.9731

<div class="k-default-codeblock">
```

```
</div>
   1508/Unknown  335s 218ms/step - loss: 1.5438 - mean_absolute_error: 0.9731

<div class="k-default-codeblock">
```

```
</div>
   1509/Unknown  335s 218ms/step - loss: 1.5436 - mean_absolute_error: 0.9730

<div class="k-default-codeblock">
```

```
</div>
   1510/Unknown  336s 218ms/step - loss: 1.5434 - mean_absolute_error: 0.9730

<div class="k-default-codeblock">
```

```
</div>
   1511/Unknown  336s 218ms/step - loss: 1.5432 - mean_absolute_error: 0.9729

<div class="k-default-codeblock">
```

```
</div>
   1512/Unknown  336s 218ms/step - loss: 1.5430 - mean_absolute_error: 0.9728

<div class="k-default-codeblock">
```

```
</div>
   1513/Unknown  336s 218ms/step - loss: 1.5428 - mean_absolute_error: 0.9728

<div class="k-default-codeblock">
```

```
</div>
   1514/Unknown  337s 218ms/step - loss: 1.5426 - mean_absolute_error: 0.9727

<div class="k-default-codeblock">
```

```
</div>
   1515/Unknown  337s 218ms/step - loss: 1.5424 - mean_absolute_error: 0.9727

<div class="k-default-codeblock">
```

```
</div>
   1516/Unknown  337s 218ms/step - loss: 1.5422 - mean_absolute_error: 0.9726

<div class="k-default-codeblock">
```

```
</div>
   1517/Unknown  337s 218ms/step - loss: 1.5420 - mean_absolute_error: 0.9725

<div class="k-default-codeblock">
```

```
</div>
   1518/Unknown  338s 218ms/step - loss: 1.5418 - mean_absolute_error: 0.9725

<div class="k-default-codeblock">
```

```
</div>
   1519/Unknown  338s 218ms/step - loss: 1.5415 - mean_absolute_error: 0.9724

<div class="k-default-codeblock">
```

```
</div>
   1520/Unknown  338s 218ms/step - loss: 1.5413 - mean_absolute_error: 0.9724

<div class="k-default-codeblock">
```

```
</div>
   1521/Unknown  338s 218ms/step - loss: 1.5411 - mean_absolute_error: 0.9723

<div class="k-default-codeblock">
```

```
</div>
   1522/Unknown  338s 218ms/step - loss: 1.5409 - mean_absolute_error: 0.9722

<div class="k-default-codeblock">
```

```
</div>
   1523/Unknown  339s 218ms/step - loss: 1.5407 - mean_absolute_error: 0.9722

<div class="k-default-codeblock">
```

```
</div>
   1524/Unknown  339s 218ms/step - loss: 1.5405 - mean_absolute_error: 0.9721

<div class="k-default-codeblock">
```

```
</div>
   1525/Unknown  339s 218ms/step - loss: 1.5403 - mean_absolute_error: 0.9721

<div class="k-default-codeblock">
```

```
</div>
   1526/Unknown  339s 218ms/step - loss: 1.5401 - mean_absolute_error: 0.9720

<div class="k-default-codeblock">
```

```
</div>
   1527/Unknown  340s 218ms/step - loss: 1.5399 - mean_absolute_error: 0.9719

<div class="k-default-codeblock">
```

```
</div>
   1528/Unknown  340s 218ms/step - loss: 1.5397 - mean_absolute_error: 0.9719

<div class="k-default-codeblock">
```

```
</div>
   1529/Unknown  340s 218ms/step - loss: 1.5395 - mean_absolute_error: 0.9718

<div class="k-default-codeblock">
```

```
</div>
   1530/Unknown  340s 218ms/step - loss: 1.5393 - mean_absolute_error: 0.9718

<div class="k-default-codeblock">
```

```
</div>
   1531/Unknown  341s 218ms/step - loss: 1.5391 - mean_absolute_error: 0.9717

<div class="k-default-codeblock">
```

```
</div>
   1532/Unknown  341s 218ms/step - loss: 1.5389 - mean_absolute_error: 0.9716

<div class="k-default-codeblock">
```

```
</div>
   1533/Unknown  341s 218ms/step - loss: 1.5387 - mean_absolute_error: 0.9716

<div class="k-default-codeblock">
```

```
</div>
   1534/Unknown  341s 218ms/step - loss: 1.5385 - mean_absolute_error: 0.9715

<div class="k-default-codeblock">
```

```
</div>
   1535/Unknown  342s 218ms/step - loss: 1.5383 - mean_absolute_error: 0.9715

<div class="k-default-codeblock">
```

```
</div>
   1536/Unknown  342s 218ms/step - loss: 1.5381 - mean_absolute_error: 0.9714

<div class="k-default-codeblock">
```

```
</div>
   1537/Unknown  342s 218ms/step - loss: 1.5379 - mean_absolute_error: 0.9713

<div class="k-default-codeblock">
```

```
</div>
   1538/Unknown  342s 218ms/step - loss: 1.5377 - mean_absolute_error: 0.9713

<div class="k-default-codeblock">
```

```
</div>
   1539/Unknown  343s 218ms/step - loss: 1.5375 - mean_absolute_error: 0.9712

<div class="k-default-codeblock">
```

```
</div>
   1540/Unknown  343s 218ms/step - loss: 1.5373 - mean_absolute_error: 0.9712

<div class="k-default-codeblock">
```

```
</div>
   1541/Unknown  343s 218ms/step - loss: 1.5371 - mean_absolute_error: 0.9711

<div class="k-default-codeblock">
```

```
</div>
   1542/Unknown  344s 218ms/step - loss: 1.5369 - mean_absolute_error: 0.9710

<div class="k-default-codeblock">
```

```
</div>
   1543/Unknown  344s 218ms/step - loss: 1.5366 - mean_absolute_error: 0.9710

<div class="k-default-codeblock">
```

```
</div>
   1544/Unknown  344s 219ms/step - loss: 1.5364 - mean_absolute_error: 0.9709

<div class="k-default-codeblock">
```

```
</div>
   1545/Unknown  344s 219ms/step - loss: 1.5362 - mean_absolute_error: 0.9709

<div class="k-default-codeblock">
```

```
</div>
   1546/Unknown  345s 219ms/step - loss: 1.5360 - mean_absolute_error: 0.9708

<div class="k-default-codeblock">
```

```
</div>
   1547/Unknown  345s 219ms/step - loss: 1.5358 - mean_absolute_error: 0.9708

<div class="k-default-codeblock">
```

```
</div>
   1548/Unknown  345s 219ms/step - loss: 1.5356 - mean_absolute_error: 0.9707

<div class="k-default-codeblock">
```

```
</div>
   1549/Unknown  345s 219ms/step - loss: 1.5354 - mean_absolute_error: 0.9706

<div class="k-default-codeblock">
```

```
</div>
   1550/Unknown  346s 219ms/step - loss: 1.5352 - mean_absolute_error: 0.9706

<div class="k-default-codeblock">
```

```
</div>
   1551/Unknown  346s 219ms/step - loss: 1.5350 - mean_absolute_error: 0.9705

<div class="k-default-codeblock">
```

```
</div>
   1552/Unknown  346s 219ms/step - loss: 1.5348 - mean_absolute_error: 0.9705

<div class="k-default-codeblock">
```

```
</div>
   1553/Unknown  346s 219ms/step - loss: 1.5346 - mean_absolute_error: 0.9704

<div class="k-default-codeblock">
```

```
</div>
   1554/Unknown  347s 219ms/step - loss: 1.5344 - mean_absolute_error: 0.9703

<div class="k-default-codeblock">
```

```
</div>
   1555/Unknown  347s 219ms/step - loss: 1.5342 - mean_absolute_error: 0.9703

<div class="k-default-codeblock">
```

```
</div>
   1556/Unknown  347s 219ms/step - loss: 1.5340 - mean_absolute_error: 0.9702

<div class="k-default-codeblock">
```

```
</div>
   1557/Unknown  348s 219ms/step - loss: 1.5338 - mean_absolute_error: 0.9702

<div class="k-default-codeblock">
```

```
</div>
   1558/Unknown  348s 219ms/step - loss: 1.5336 - mean_absolute_error: 0.9701

<div class="k-default-codeblock">
```

```
</div>
   1559/Unknown  348s 219ms/step - loss: 1.5334 - mean_absolute_error: 0.9701

<div class="k-default-codeblock">
```

```
</div>
   1560/Unknown  349s 219ms/step - loss: 1.5332 - mean_absolute_error: 0.9700

<div class="k-default-codeblock">
```

```
</div>
   1561/Unknown  349s 219ms/step - loss: 1.5330 - mean_absolute_error: 0.9699

<div class="k-default-codeblock">
```

```
</div>
   1562/Unknown  349s 219ms/step - loss: 1.5328 - mean_absolute_error: 0.9699

<div class="k-default-codeblock">
```

```
</div>
   1563/Unknown  349s 219ms/step - loss: 1.5326 - mean_absolute_error: 0.9698

<div class="k-default-codeblock">
```

```
</div>
   1564/Unknown  350s 219ms/step - loss: 1.5325 - mean_absolute_error: 0.9698

<div class="k-default-codeblock">
```

```
</div>
   1565/Unknown  350s 219ms/step - loss: 1.5323 - mean_absolute_error: 0.9697

<div class="k-default-codeblock">
```

```
</div>
   1566/Unknown  350s 219ms/step - loss: 1.5321 - mean_absolute_error: 0.9696

<div class="k-default-codeblock">
```

```
</div>
   1567/Unknown  351s 219ms/step - loss: 1.5319 - mean_absolute_error: 0.9696

<div class="k-default-codeblock">
```

```
</div>
   1568/Unknown  351s 219ms/step - loss: 1.5317 - mean_absolute_error: 0.9695

<div class="k-default-codeblock">
```

```
</div>
   1569/Unknown  351s 219ms/step - loss: 1.5315 - mean_absolute_error: 0.9695

<div class="k-default-codeblock">
```

```
</div>
   1570/Unknown  351s 219ms/step - loss: 1.5313 - mean_absolute_error: 0.9694

<div class="k-default-codeblock">
```

```
</div>
   1571/Unknown  351s 219ms/step - loss: 1.5311 - mean_absolute_error: 0.9694

<div class="k-default-codeblock">
```

```
</div>
   1572/Unknown  352s 219ms/step - loss: 1.5309 - mean_absolute_error: 0.9693

<div class="k-default-codeblock">
```

```
</div>
   1573/Unknown  352s 219ms/step - loss: 1.5307 - mean_absolute_error: 0.9692

<div class="k-default-codeblock">
```

```
</div>
   1574/Unknown  352s 220ms/step - loss: 1.5305 - mean_absolute_error: 0.9692

<div class="k-default-codeblock">
```

```
</div>
   1575/Unknown  352s 220ms/step - loss: 1.5303 - mean_absolute_error: 0.9691

<div class="k-default-codeblock">
```

```
</div>
   1576/Unknown  353s 220ms/step - loss: 1.5301 - mean_absolute_error: 0.9691

<div class="k-default-codeblock">
```

```
</div>
   1577/Unknown  353s 220ms/step - loss: 1.5299 - mean_absolute_error: 0.9690

<div class="k-default-codeblock">
```

```
</div>
   1578/Unknown  353s 220ms/step - loss: 1.5297 - mean_absolute_error: 0.9690

<div class="k-default-codeblock">
```

```
</div>
   1579/Unknown  354s 220ms/step - loss: 1.5295 - mean_absolute_error: 0.9689

<div class="k-default-codeblock">
```

```
</div>
   1580/Unknown  354s 220ms/step - loss: 1.5293 - mean_absolute_error: 0.9688

<div class="k-default-codeblock">
```

```
</div>
   1581/Unknown  354s 220ms/step - loss: 1.5291 - mean_absolute_error: 0.9688

<div class="k-default-codeblock">
```

```
</div>
   1582/Unknown  354s 220ms/step - loss: 1.5289 - mean_absolute_error: 0.9687

<div class="k-default-codeblock">
```

```
</div>
   1583/Unknown  355s 220ms/step - loss: 1.5287 - mean_absolute_error: 0.9687

<div class="k-default-codeblock">
```

```
</div>
   1584/Unknown  355s 220ms/step - loss: 1.5285 - mean_absolute_error: 0.9686

<div class="k-default-codeblock">
```

```
</div>
   1585/Unknown  355s 220ms/step - loss: 1.5283 - mean_absolute_error: 0.9686

<div class="k-default-codeblock">
```

```
</div>
   1586/Unknown  356s 220ms/step - loss: 1.5281 - mean_absolute_error: 0.9685

<div class="k-default-codeblock">
```

```
</div>
   1587/Unknown  356s 220ms/step - loss: 1.5279 - mean_absolute_error: 0.9684

<div class="k-default-codeblock">
```

```
</div>
   1588/Unknown  356s 220ms/step - loss: 1.5277 - mean_absolute_error: 0.9684

<div class="k-default-codeblock">
```

```
</div>
   1589/Unknown  356s 220ms/step - loss: 1.5276 - mean_absolute_error: 0.9683

<div class="k-default-codeblock">
```

```
</div>
   1590/Unknown  357s 220ms/step - loss: 1.5274 - mean_absolute_error: 0.9683

<div class="k-default-codeblock">
```

```
</div>
   1591/Unknown  357s 220ms/step - loss: 1.5272 - mean_absolute_error: 0.9682

<div class="k-default-codeblock">
```

```
</div>
   1592/Unknown  357s 220ms/step - loss: 1.5270 - mean_absolute_error: 0.9682

<div class="k-default-codeblock">
```

```
</div>
   1593/Unknown  357s 220ms/step - loss: 1.5268 - mean_absolute_error: 0.9681

<div class="k-default-codeblock">
```

```
</div>
   1594/Unknown  358s 220ms/step - loss: 1.5266 - mean_absolute_error: 0.9680

<div class="k-default-codeblock">
```

```
</div>
   1595/Unknown  358s 220ms/step - loss: 1.5264 - mean_absolute_error: 0.9680

<div class="k-default-codeblock">
```

```
</div>
   1596/Unknown  358s 220ms/step - loss: 1.5262 - mean_absolute_error: 0.9679

<div class="k-default-codeblock">
```

```
</div>
   1597/Unknown  358s 220ms/step - loss: 1.5260 - mean_absolute_error: 0.9679

<div class="k-default-codeblock">
```

```
</div>
   1598/Unknown  358s 220ms/step - loss: 1.5258 - mean_absolute_error: 0.9678

<div class="k-default-codeblock">
```

```
</div>
   1599/Unknown  364s 224ms/step - loss: 1.5256 - mean_absolute_error: 0.9678

<div class="k-default-codeblock">
```

```
</div>
 1599/1599 ━━━━━━━━━━━━━━━━━━━━ 364s 224ms/step - loss: 1.5254 - mean_absolute_error: 0.9677


<div class="k-default-codeblock">
```
Epoch 2/2

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:159: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()

```
</div>
    
    1/1599 [37m━━━━━━━━━━━━━━━━━━━━  13:19 500ms/step - loss: 1.2442 - mean_absolute_error: 0.8669

<div class="k-default-codeblock">
```

```
</div>
    2/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:39 250ms/step - loss: 1.1797 - mean_absolute_error: 0.8505 

<div class="k-default-codeblock">
```

```
</div>
    3/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:44 253ms/step - loss: 1.1448 - mean_absolute_error: 0.8407

<div class="k-default-codeblock">
```

```
</div>
    4/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:38 250ms/step - loss: 1.1370 - mean_absolute_error: 0.8397

<div class="k-default-codeblock">
```

```
</div>
    5/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:42 253ms/step - loss: 1.1311 - mean_absolute_error: 0.8382

<div class="k-default-codeblock">
```

```
</div>
    6/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:40 252ms/step - loss: 1.1236 - mean_absolute_error: 0.8364

<div class="k-default-codeblock">
```

```
</div>
    7/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:38 251ms/step - loss: 1.1174 - mean_absolute_error: 0.8353

<div class="k-default-codeblock">
```

```
</div>
    8/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:37 250ms/step - loss: 1.1116 - mean_absolute_error: 0.8343

<div class="k-default-codeblock">
```

```
</div>
    9/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:37 250ms/step - loss: 1.1087 - mean_absolute_error: 0.8343

<div class="k-default-codeblock">
```

```
</div>
   10/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:36 249ms/step - loss: 1.1065 - mean_absolute_error: 0.8343

<div class="k-default-codeblock">
```

```
</div>
   11/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:36 250ms/step - loss: 1.1047 - mean_absolute_error: 0.8344

<div class="k-default-codeblock">
```

```
</div>
   12/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:35 249ms/step - loss: 1.1020 - mean_absolute_error: 0.8339

<div class="k-default-codeblock">
```

```
</div>
   13/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:35 250ms/step - loss: 1.0994 - mean_absolute_error: 0.8331

<div class="k-default-codeblock">
```

```
</div>
   14/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:34 249ms/step - loss: 1.0971 - mean_absolute_error: 0.8324

<div class="k-default-codeblock">
```

```
</div>
   15/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:34 249ms/step - loss: 1.0957 - mean_absolute_error: 0.8320

<div class="k-default-codeblock">
```

```
</div>
   16/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:33 249ms/step - loss: 1.0943 - mean_absolute_error: 0.8315

<div class="k-default-codeblock">
```

```
</div>
   17/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:33 248ms/step - loss: 1.0928 - mean_absolute_error: 0.8310

<div class="k-default-codeblock">
```

```
</div>
   18/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:33 249ms/step - loss: 1.0913 - mean_absolute_error: 0.8306

<div class="k-default-codeblock">
```

```
</div>
   19/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:32 248ms/step - loss: 1.0899 - mean_absolute_error: 0.8301

<div class="k-default-codeblock">
```

```
</div>
   20/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:32 249ms/step - loss: 1.0889 - mean_absolute_error: 0.8299

<div class="k-default-codeblock">
```

```
</div>
   21/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:31 248ms/step - loss: 1.0878 - mean_absolute_error: 0.8296

<div class="k-default-codeblock">
```

```
</div>
   22/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:29 247ms/step - loss: 1.0871 - mean_absolute_error: 0.8294

<div class="k-default-codeblock">
```

```
</div>
   23/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:26 246ms/step - loss: 1.0865 - mean_absolute_error: 0.8293

<div class="k-default-codeblock">
```

```
</div>
   24/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:24 244ms/step - loss: 1.0858 - mean_absolute_error: 0.8291

<div class="k-default-codeblock">
```

```
</div>
   25/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:23 244ms/step - loss: 1.0849 - mean_absolute_error: 0.8290

<div class="k-default-codeblock">
```

```
</div>
   26/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:22 243ms/step - loss: 1.0841 - mean_absolute_error: 0.8288

<div class="k-default-codeblock">
```

```
</div>
   27/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:19 242ms/step - loss: 1.0833 - mean_absolute_error: 0.8286

<div class="k-default-codeblock">
```

```
</div>
   28/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:17 241ms/step - loss: 1.0828 - mean_absolute_error: 0.8286

<div class="k-default-codeblock">
```

```
</div>
   29/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:15 239ms/step - loss: 1.0822 - mean_absolute_error: 0.8284

<div class="k-default-codeblock">
```

```
</div>
   30/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:14 239ms/step - loss: 1.0817 - mean_absolute_error: 0.8284

<div class="k-default-codeblock">
```

```
</div>
   31/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:12 238ms/step - loss: 1.0811 - mean_absolute_error: 0.8283

<div class="k-default-codeblock">
```

```
</div>
   32/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:11 237ms/step - loss: 1.0805 - mean_absolute_error: 0.8281

<div class="k-default-codeblock">
```

```
</div>
   33/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:11 237ms/step - loss: 1.0799 - mean_absolute_error: 0.8280

<div class="k-default-codeblock">
```

```
</div>
   34/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:10 237ms/step - loss: 1.0792 - mean_absolute_error: 0.8279

<div class="k-default-codeblock">
```

```
</div>
   35/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:09 236ms/step - loss: 1.0787 - mean_absolute_error: 0.8278

<div class="k-default-codeblock">
```

```
</div>
   36/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:08 236ms/step - loss: 1.0781 - mean_absolute_error: 0.8277

<div class="k-default-codeblock">
```

```
</div>
   37/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:07 236ms/step - loss: 1.0776 - mean_absolute_error: 0.8276

<div class="k-default-codeblock">
```

```
</div>
   38/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:07 235ms/step - loss: 1.0771 - mean_absolute_error: 0.8275

<div class="k-default-codeblock">
```

```
</div>
   39/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:06 235ms/step - loss: 1.0766 - mean_absolute_error: 0.8274

<div class="k-default-codeblock">
```

```
</div>
   40/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:06 235ms/step - loss: 1.0761 - mean_absolute_error: 0.8273

<div class="k-default-codeblock">
```

```
</div>
   41/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:05 234ms/step - loss: 1.0757 - mean_absolute_error: 0.8272

<div class="k-default-codeblock">
```

```
</div>
   42/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:04 234ms/step - loss: 1.0753 - mean_absolute_error: 0.8272

<div class="k-default-codeblock">
```

```
</div>
   43/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:04 234ms/step - loss: 1.0748 - mean_absolute_error: 0.8270

<div class="k-default-codeblock">
```

```
</div>
   44/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:03 234ms/step - loss: 1.0743 - mean_absolute_error: 0.8269

<div class="k-default-codeblock">
```

```
</div>
   45/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:04 234ms/step - loss: 1.0738 - mean_absolute_error: 0.8268

<div class="k-default-codeblock">
```

```
</div>
   46/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:05 235ms/step - loss: 1.0734 - mean_absolute_error: 0.8267

<div class="k-default-codeblock">
```

```
</div>
   47/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:06 236ms/step - loss: 1.0730 - mean_absolute_error: 0.8266

<div class="k-default-codeblock">
```

```
</div>
   48/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:07 237ms/step - loss: 1.0726 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   49/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:06 236ms/step - loss: 1.0724 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   50/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:06 237ms/step - loss: 1.0722 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   51/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:21 247ms/step - loss: 1.0721 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   52/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:21 246ms/step - loss: 1.0720 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   53/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:19 246ms/step - loss: 1.0719 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   54/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:19 245ms/step - loss: 1.0717 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   55/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:17 245ms/step - loss: 1.0716 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   56/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:16 244ms/step - loss: 1.0714 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   57/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:15 243ms/step - loss: 1.0712 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   58/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:13 243ms/step - loss: 1.0709 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
   59/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:12 242ms/step - loss: 1.0707 - mean_absolute_error: 0.8264

<div class="k-default-codeblock">
```

```
</div>
   60/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:11 242ms/step - loss: 1.0704 - mean_absolute_error: 0.8264

<div class="k-default-codeblock">
```

```
</div>
   61/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:11 241ms/step - loss: 1.0701 - mean_absolute_error: 0.8263

<div class="k-default-codeblock">
```

```
</div>
   62/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:10 241ms/step - loss: 1.0699 - mean_absolute_error: 0.8263

<div class="k-default-codeblock">
```

```
</div>
   63/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:09 241ms/step - loss: 1.0696 - mean_absolute_error: 0.8262

<div class="k-default-codeblock">
```

```
</div>
   64/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:09 240ms/step - loss: 1.0694 - mean_absolute_error: 0.8262

<div class="k-default-codeblock">
```

```
</div>
   65/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:08 240ms/step - loss: 1.0691 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
   66/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:07 240ms/step - loss: 1.0690 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
   67/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:07 240ms/step - loss: 1.0688 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
   68/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:06 239ms/step - loss: 1.0686 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
   69/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:05 239ms/step - loss: 1.0684 - mean_absolute_error: 0.8260

<div class="k-default-codeblock">
```

```
</div>
   70/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:05 239ms/step - loss: 1.0682 - mean_absolute_error: 0.8260

<div class="k-default-codeblock">
```

```
</div>
   71/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:04 239ms/step - loss: 1.0680 - mean_absolute_error: 0.8260

<div class="k-default-codeblock">
```

```
</div>
   72/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:04 239ms/step - loss: 1.0678 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
   73/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:03 238ms/step - loss: 1.0676 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
   74/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:03 239ms/step - loss: 1.0674 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
   75/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:04 239ms/step - loss: 1.0672 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
   76/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:04 240ms/step - loss: 1.0670 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
   77/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:05 240ms/step - loss: 1.0668 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
   78/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:05 240ms/step - loss: 1.0666 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
   79/1599 [37m━━━━━━━━━━━━━━━━━━━━  6:05 241ms/step - loss: 1.0664 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
   80/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:06 241ms/step - loss: 1.0662 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
   81/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:06 241ms/step - loss: 1.0661 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
   82/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:05 241ms/step - loss: 1.0659 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   83/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:05 241ms/step - loss: 1.0658 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   84/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:05 241ms/step - loss: 1.0657 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   85/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:05 241ms/step - loss: 1.0656 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   86/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:04 241ms/step - loss: 1.0655 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   87/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:04 241ms/step - loss: 1.0653 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   88/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 241ms/step - loss: 1.0652 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   89/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 241ms/step - loss: 1.0651 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   90/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 241ms/step - loss: 1.0651 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   91/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 241ms/step - loss: 1.0650 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   92/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 241ms/step - loss: 1.0649 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   93/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 241ms/step - loss: 1.0649 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   94/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 241ms/step - loss: 1.0648 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   95/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:01 241ms/step - loss: 1.0648 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   96/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 241ms/step - loss: 1.0647 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   97/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 241ms/step - loss: 1.0647 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   98/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 242ms/step - loss: 1.0646 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
   99/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 242ms/step - loss: 1.0646 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  100/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 243ms/step - loss: 1.0645 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  101/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 243ms/step - loss: 1.0645 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  102/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 243ms/step - loss: 1.0645 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  103/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:03 243ms/step - loss: 1.0644 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  104/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 243ms/step - loss: 1.0644 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  105/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 243ms/step - loss: 1.0643 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  106/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 243ms/step - loss: 1.0643 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  107/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 243ms/step - loss: 1.0642 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  108/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:02 243ms/step - loss: 1.0642 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  109/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:01 243ms/step - loss: 1.0642 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  110/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:01 243ms/step - loss: 1.0642 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  111/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:01 243ms/step - loss: 1.0641 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  112/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:00 243ms/step - loss: 1.0641 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  113/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:00 243ms/step - loss: 1.0641 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  114/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:00 243ms/step - loss: 1.0640 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  115/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:00 243ms/step - loss: 1.0640 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  116/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:00 243ms/step - loss: 1.0640 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  117/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:00 243ms/step - loss: 1.0640 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  118/1599 ━[37m━━━━━━━━━━━━━━━━━━━  6:00 243ms/step - loss: 1.0639 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  119/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:59 243ms/step - loss: 1.0639 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  120/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:59 243ms/step - loss: 1.0639 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  121/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:59 243ms/step - loss: 1.0639 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  122/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:59 243ms/step - loss: 1.0639 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  123/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:59 243ms/step - loss: 1.0639 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  124/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:58 243ms/step - loss: 1.0638 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  125/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:58 243ms/step - loss: 1.0638 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  126/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:57 243ms/step - loss: 1.0638 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  127/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:57 243ms/step - loss: 1.0638 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  128/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:56 242ms/step - loss: 1.0638 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  129/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:56 242ms/step - loss: 1.0638 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  130/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:55 242ms/step - loss: 1.0637 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  131/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:54 242ms/step - loss: 1.0637 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  132/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:54 242ms/step - loss: 1.0637 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  133/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:54 242ms/step - loss: 1.0637 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  134/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:53 241ms/step - loss: 1.0636 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  135/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:53 241ms/step - loss: 1.0636 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  136/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:52 241ms/step - loss: 1.0636 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  137/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:52 241ms/step - loss: 1.0636 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  138/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:52 241ms/step - loss: 1.0635 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  139/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:51 241ms/step - loss: 1.0635 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  140/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:51 241ms/step - loss: 1.0635 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  141/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:51 241ms/step - loss: 1.0635 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  142/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:50 241ms/step - loss: 1.0634 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  143/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:50 241ms/step - loss: 1.0634 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  144/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 240ms/step - loss: 1.0634 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  145/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 240ms/step - loss: 1.0633 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  146/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 240ms/step - loss: 1.0633 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  147/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 241ms/step - loss: 1.0633 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  148/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 241ms/step - loss: 1.0632 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  149/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 241ms/step - loss: 1.0632 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  150/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 241ms/step - loss: 1.0632 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  151/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 241ms/step - loss: 1.0631 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  152/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 242ms/step - loss: 1.0631 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  153/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 242ms/step - loss: 1.0630 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  154/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 242ms/step - loss: 1.0630 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  155/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:49 242ms/step - loss: 1.0629 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  156/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:48 242ms/step - loss: 1.0629 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  157/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:48 242ms/step - loss: 1.0628 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  158/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:48 242ms/step - loss: 1.0627 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  159/1599 ━[37m━━━━━━━━━━━━━━━━━━━  5:47 242ms/step - loss: 1.0627 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  160/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:47 241ms/step - loss: 1.0626 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  161/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:46 241ms/step - loss: 1.0625 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  162/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:46 241ms/step - loss: 1.0625 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  163/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:46 241ms/step - loss: 1.0624 - mean_absolute_error: 0.8251

<div class="k-default-codeblock">
```

```
</div>
  164/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:45 241ms/step - loss: 1.0623 - mean_absolute_error: 0.8251

<div class="k-default-codeblock">
```

```
</div>
  165/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:45 241ms/step - loss: 1.0623 - mean_absolute_error: 0.8251

<div class="k-default-codeblock">
```

```
</div>
  166/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:44 240ms/step - loss: 1.0622 - mean_absolute_error: 0.8251

<div class="k-default-codeblock">
```

```
</div>
  167/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:43 240ms/step - loss: 1.0621 - mean_absolute_error: 0.8250

<div class="k-default-codeblock">
```

```
</div>
  168/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:43 240ms/step - loss: 1.0621 - mean_absolute_error: 0.8250

<div class="k-default-codeblock">
```

```
</div>
  169/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:43 240ms/step - loss: 1.0620 - mean_absolute_error: 0.8250

<div class="k-default-codeblock">
```

```
</div>
  170/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:43 240ms/step - loss: 1.0619 - mean_absolute_error: 0.8250

<div class="k-default-codeblock">
```

```
</div>
  171/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:42 240ms/step - loss: 1.0619 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  172/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:42 240ms/step - loss: 1.0618 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  173/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:42 240ms/step - loss: 1.0617 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  174/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:41 240ms/step - loss: 1.0617 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  175/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:41 240ms/step - loss: 1.0616 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  176/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:41 240ms/step - loss: 1.0616 - mean_absolute_error: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  177/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:40 240ms/step - loss: 1.0615 - mean_absolute_error: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  178/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:40 240ms/step - loss: 1.0614 - mean_absolute_error: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  179/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:40 239ms/step - loss: 1.0614 - mean_absolute_error: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  180/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 239ms/step - loss: 1.0613 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  181/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0612 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  182/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0612 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  183/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0611 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  184/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0611 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  185/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0611 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  186/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0610 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  187/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0610 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  188/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 240ms/step - loss: 1.0609 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  189/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:38 240ms/step - loss: 1.0609 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  190/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:38 240ms/step - loss: 1.0608 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  191/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:38 240ms/step - loss: 1.0608 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  192/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:37 240ms/step - loss: 1.0608 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  193/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:37 240ms/step - loss: 1.0607 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  194/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:36 240ms/step - loss: 1.0607 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  195/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:36 240ms/step - loss: 1.0606 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  196/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:35 239ms/step - loss: 1.0606 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  197/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:35 239ms/step - loss: 1.0605 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  198/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:35 239ms/step - loss: 1.0605 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  199/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:34 239ms/step - loss: 1.0605 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  200/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:34 239ms/step - loss: 1.0604 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  201/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:34 239ms/step - loss: 1.0604 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  202/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:33 239ms/step - loss: 1.0603 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  203/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:33 239ms/step - loss: 1.0603 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  204/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:33 239ms/step - loss: 1.0602 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  205/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:33 239ms/step - loss: 1.0602 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  206/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:33 239ms/step - loss: 1.0601 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  207/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 239ms/step - loss: 1.0601 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  208/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 239ms/step - loss: 1.0600 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  209/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 239ms/step - loss: 1.0600 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  210/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 239ms/step - loss: 1.0599 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  211/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 239ms/step - loss: 1.0599 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  212/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 239ms/step - loss: 1.0598 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  213/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 240ms/step - loss: 1.0598 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  214/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 240ms/step - loss: 1.0597 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  215/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 240ms/step - loss: 1.0597 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  216/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 241ms/step - loss: 1.0596 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  217/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 241ms/step - loss: 1.0596 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  218/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 241ms/step - loss: 1.0595 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  219/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 241ms/step - loss: 1.0595 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  220/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 241ms/step - loss: 1.0594 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  221/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:33 242ms/step - loss: 1.0594 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  222/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:33 242ms/step - loss: 1.0593 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  223/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 242ms/step - loss: 1.0593 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  224/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 242ms/step - loss: 1.0592 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  225/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 242ms/step - loss: 1.0592 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  226/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:32 242ms/step - loss: 1.0591 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  227/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:31 242ms/step - loss: 1.0590 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  228/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:31 242ms/step - loss: 1.0590 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  229/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:31 242ms/step - loss: 1.0589 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  230/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:30 242ms/step - loss: 1.0589 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  231/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:30 242ms/step - loss: 1.0589 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  232/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:30 241ms/step - loss: 1.0588 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  233/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:29 241ms/step - loss: 1.0588 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  234/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:29 241ms/step - loss: 1.0587 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  235/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:28 241ms/step - loss: 1.0587 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  236/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:28 241ms/step - loss: 1.0586 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  237/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:28 241ms/step - loss: 1.0586 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  238/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:27 241ms/step - loss: 1.0585 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  239/1599 ━━[37m━━━━━━━━━━━━━━━━━━  5:27 241ms/step - loss: 1.0585 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  240/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:27 241ms/step - loss: 1.0584 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  241/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:27 241ms/step - loss: 1.0584 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  242/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:26 241ms/step - loss: 1.0583 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  243/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:26 241ms/step - loss: 1.0583 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  244/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:26 241ms/step - loss: 1.0583 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  245/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:26 241ms/step - loss: 1.0582 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  246/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 241ms/step - loss: 1.0582 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  247/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 241ms/step - loss: 1.0581 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  248/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 241ms/step - loss: 1.0581 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  249/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 241ms/step - loss: 1.0580 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  250/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 241ms/step - loss: 1.0580 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  251/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 241ms/step - loss: 1.0579 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  252/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 241ms/step - loss: 1.0578 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  253/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 241ms/step - loss: 1.0578 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  254/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 242ms/step - loss: 1.0577 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  255/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 242ms/step - loss: 1.0577 - mean_absolute_error: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  256/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 242ms/step - loss: 1.0576 - mean_absolute_error: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  257/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 242ms/step - loss: 1.0575 - mean_absolute_error: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  258/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 242ms/step - loss: 1.0575 - mean_absolute_error: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  259/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:23 242ms/step - loss: 1.0574 - mean_absolute_error: 0.8233

<div class="k-default-codeblock">
```

```
</div>
  260/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:23 242ms/step - loss: 1.0573 - mean_absolute_error: 0.8233

<div class="k-default-codeblock">
```

```
</div>
  261/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:23 242ms/step - loss: 1.0573 - mean_absolute_error: 0.8233

<div class="k-default-codeblock">
```

```
</div>
  262/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:22 242ms/step - loss: 1.0572 - mean_absolute_error: 0.8233

<div class="k-default-codeblock">
```

```
</div>
  263/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:22 242ms/step - loss: 1.0572 - mean_absolute_error: 0.8232

<div class="k-default-codeblock">
```

```
</div>
  264/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:22 241ms/step - loss: 1.0571 - mean_absolute_error: 0.8232

<div class="k-default-codeblock">
```

```
</div>
  265/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:22 241ms/step - loss: 1.0570 - mean_absolute_error: 0.8232

<div class="k-default-codeblock">
```

```
</div>
  266/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:21 241ms/step - loss: 1.0569 - mean_absolute_error: 0.8232

<div class="k-default-codeblock">
```

```
</div>
  267/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:21 241ms/step - loss: 1.0569 - mean_absolute_error: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  268/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:21 241ms/step - loss: 1.0568 - mean_absolute_error: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  269/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:21 242ms/step - loss: 1.0567 - mean_absolute_error: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  270/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:20 242ms/step - loss: 1.0567 - mean_absolute_error: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  271/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:20 242ms/step - loss: 1.0566 - mean_absolute_error: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  272/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:20 242ms/step - loss: 1.0565 - mean_absolute_error: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  273/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:20 241ms/step - loss: 1.0565 - mean_absolute_error: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  274/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:19 241ms/step - loss: 1.0564 - mean_absolute_error: 0.8229

<div class="k-default-codeblock">
```

```
</div>
  275/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:19 241ms/step - loss: 1.0563 - mean_absolute_error: 0.8229

<div class="k-default-codeblock">
```

```
</div>
  276/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:19 241ms/step - loss: 1.0563 - mean_absolute_error: 0.8229

<div class="k-default-codeblock">
```

```
</div>
  277/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:18 241ms/step - loss: 1.0562 - mean_absolute_error: 0.8228

<div class="k-default-codeblock">
```

```
</div>
  278/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:18 241ms/step - loss: 1.0561 - mean_absolute_error: 0.8228

<div class="k-default-codeblock">
```

```
</div>
  279/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:18 241ms/step - loss: 1.0560 - mean_absolute_error: 0.8228

<div class="k-default-codeblock">
```

```
</div>
  280/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:18 241ms/step - loss: 1.0560 - mean_absolute_error: 0.8228

<div class="k-default-codeblock">
```

```
</div>
  281/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:17 241ms/step - loss: 1.0559 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  282/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:17 241ms/step - loss: 1.0558 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  283/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:17 241ms/step - loss: 1.0558 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  284/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:17 241ms/step - loss: 1.0557 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  285/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:17 241ms/step - loss: 1.0556 - mean_absolute_error: 0.8226

<div class="k-default-codeblock">
```

```
</div>
  286/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:16 241ms/step - loss: 1.0556 - mean_absolute_error: 0.8226

<div class="k-default-codeblock">
```

```
</div>
  287/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:16 241ms/step - loss: 1.0555 - mean_absolute_error: 0.8226

<div class="k-default-codeblock">
```

```
</div>
  288/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:16 241ms/step - loss: 1.0554 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  289/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:16 241ms/step - loss: 1.0554 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  290/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:16 241ms/step - loss: 1.0553 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  291/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:15 241ms/step - loss: 1.0552 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  292/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:15 241ms/step - loss: 1.0552 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  293/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:15 241ms/step - loss: 1.0551 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  294/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:14 241ms/step - loss: 1.0550 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  295/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:14 241ms/step - loss: 1.0550 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  296/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:13 241ms/step - loss: 1.0549 - mean_absolute_error: 0.8223

<div class="k-default-codeblock">
```

```
</div>
  297/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:13 241ms/step - loss: 1.0548 - mean_absolute_error: 0.8223

<div class="k-default-codeblock">
```

```
</div>
  298/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:13 241ms/step - loss: 1.0548 - mean_absolute_error: 0.8223

<div class="k-default-codeblock">
```

```
</div>
  299/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:12 241ms/step - loss: 1.0547 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  300/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:12 241ms/step - loss: 1.0546 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  301/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:12 240ms/step - loss: 1.0545 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  302/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:11 240ms/step - loss: 1.0545 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  303/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:11 240ms/step - loss: 1.0544 - mean_absolute_error: 0.8221

<div class="k-default-codeblock">
```

```
</div>
  304/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:11 240ms/step - loss: 1.0543 - mean_absolute_error: 0.8221

<div class="k-default-codeblock">
```

```
</div>
  305/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:10 240ms/step - loss: 1.0543 - mean_absolute_error: 0.8221

<div class="k-default-codeblock">
```

```
</div>
  306/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:10 240ms/step - loss: 1.0542 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  307/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:10 240ms/step - loss: 1.0541 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  308/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:09 240ms/step - loss: 1.0541 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  309/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:09 240ms/step - loss: 1.0540 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  310/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:09 240ms/step - loss: 1.0539 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  311/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:09 240ms/step - loss: 1.0539 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  312/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:08 240ms/step - loss: 1.0538 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  313/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:08 240ms/step - loss: 1.0537 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  314/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:08 240ms/step - loss: 1.0536 - mean_absolute_error: 0.8218

<div class="k-default-codeblock">
```

```
</div>
  315/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:08 240ms/step - loss: 1.0536 - mean_absolute_error: 0.8218

<div class="k-default-codeblock">
```

```
</div>
  316/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:08 240ms/step - loss: 1.0535 - mean_absolute_error: 0.8218

<div class="k-default-codeblock">
```

```
</div>
  317/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:08 240ms/step - loss: 1.0534 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  318/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:08 241ms/step - loss: 1.0534 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  319/1599 ━━━[37m━━━━━━━━━━━━━━━━━  5:07 241ms/step - loss: 1.0533 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  320/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 241ms/step - loss: 1.0532 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  321/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 241ms/step - loss: 1.0532 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  322/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 241ms/step - loss: 1.0531 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  323/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 241ms/step - loss: 1.0531 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  324/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 242ms/step - loss: 1.0530 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  325/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 242ms/step - loss: 1.0529 - mean_absolute_error: 0.8215

<div class="k-default-codeblock">
```

```
</div>
  326/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 242ms/step - loss: 1.0529 - mean_absolute_error: 0.8215

<div class="k-default-codeblock">
```

```
</div>
  327/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 242ms/step - loss: 1.0528 - mean_absolute_error: 0.8215

<div class="k-default-codeblock">
```

```
</div>
  328/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 242ms/step - loss: 1.0527 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  329/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 242ms/step - loss: 1.0527 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  330/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:06 242ms/step - loss: 1.0526 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  331/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:06 242ms/step - loss: 1.0525 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  332/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:05 241ms/step - loss: 1.0525 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  333/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:05 241ms/step - loss: 1.0524 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  334/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:05 241ms/step - loss: 1.0523 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  335/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:05 241ms/step - loss: 1.0523 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  336/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:04 241ms/step - loss: 1.0522 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  337/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:04 241ms/step - loss: 1.0522 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  338/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:04 241ms/step - loss: 1.0521 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  339/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:03 241ms/step - loss: 1.0520 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  340/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:03 241ms/step - loss: 1.0520 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  341/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:03 241ms/step - loss: 1.0519 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  342/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:02 241ms/step - loss: 1.0518 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  343/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:02 241ms/step - loss: 1.0518 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  344/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:02 241ms/step - loss: 1.0517 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  345/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:02 241ms/step - loss: 1.0517 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  346/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 241ms/step - loss: 1.0516 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  347/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 241ms/step - loss: 1.0515 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  348/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 241ms/step - loss: 1.0515 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  349/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 241ms/step - loss: 1.0514 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  350/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 241ms/step - loss: 1.0514 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  351/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 241ms/step - loss: 1.0513 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  352/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:00 241ms/step - loss: 1.0513 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  353/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:00 241ms/step - loss: 1.0512 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  354/1599 ━━━━[37m━━━━━━━━━━━━━━━━  5:00 241ms/step - loss: 1.0511 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  355/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:59 241ms/step - loss: 1.0511 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  356/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:59 241ms/step - loss: 1.0510 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  357/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:59 241ms/step - loss: 1.0510 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  358/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:58 241ms/step - loss: 1.0509 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  359/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:58 241ms/step - loss: 1.0509 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  360/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:58 241ms/step - loss: 1.0508 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  361/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:57 241ms/step - loss: 1.0508 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  362/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:57 241ms/step - loss: 1.0507 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  363/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:57 240ms/step - loss: 1.0506 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  364/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:56 240ms/step - loss: 1.0506 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  365/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:56 240ms/step - loss: 1.0505 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  366/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:56 240ms/step - loss: 1.0505 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  367/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:56 240ms/step - loss: 1.0504 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  368/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:55 240ms/step - loss: 1.0504 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  369/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:55 240ms/step - loss: 1.0503 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  370/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:55 240ms/step - loss: 1.0503 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  371/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:54 240ms/step - loss: 1.0502 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  372/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:54 240ms/step - loss: 1.0502 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  373/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:54 240ms/step - loss: 1.0501 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  374/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:54 240ms/step - loss: 1.0501 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  375/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:53 240ms/step - loss: 1.0500 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  376/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:53 240ms/step - loss: 1.0500 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  377/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:53 240ms/step - loss: 1.0499 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  378/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:53 240ms/step - loss: 1.0499 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  379/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:53 240ms/step - loss: 1.0498 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  380/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:53 241ms/step - loss: 1.0498 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  381/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:53 241ms/step - loss: 1.0497 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  382/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:52 241ms/step - loss: 1.0497 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  383/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:52 241ms/step - loss: 1.0497 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  384/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:52 241ms/step - loss: 1.0496 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  385/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:52 241ms/step - loss: 1.0496 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  386/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:51 241ms/step - loss: 1.0495 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  387/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:51 241ms/step - loss: 1.0495 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  388/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:51 240ms/step - loss: 1.0494 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  389/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:50 240ms/step - loss: 1.0494 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  390/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:50 240ms/step - loss: 1.0493 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  391/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:50 240ms/step - loss: 1.0493 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  392/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:49 240ms/step - loss: 1.0492 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  393/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:49 240ms/step - loss: 1.0492 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  394/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:49 240ms/step - loss: 1.0491 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  395/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:48 240ms/step - loss: 1.0491 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  396/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:48 240ms/step - loss: 1.0491 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  397/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:48 240ms/step - loss: 1.0490 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  398/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:48 240ms/step - loss: 1.0490 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  399/1599 ━━━━[37m━━━━━━━━━━━━━━━━  4:47 240ms/step - loss: 1.0489 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  400/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:47 240ms/step - loss: 1.0489 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  401/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:47 240ms/step - loss: 1.0488 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  402/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:47 240ms/step - loss: 1.0488 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  403/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:46 240ms/step - loss: 1.0487 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  404/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:46 240ms/step - loss: 1.0487 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  405/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:46 240ms/step - loss: 1.0486 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  406/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 240ms/step - loss: 1.0486 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  407/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 240ms/step - loss: 1.0485 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  408/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 240ms/step - loss: 1.0485 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  409/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 240ms/step - loss: 1.0485 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  410/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 240ms/step - loss: 1.0484 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  411/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 240ms/step - loss: 1.0484 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  412/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 240ms/step - loss: 1.0483 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  413/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 240ms/step - loss: 1.0483 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  414/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 240ms/step - loss: 1.0482 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  415/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 240ms/step - loss: 1.0482 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  416/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 240ms/step - loss: 1.0481 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  417/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 240ms/step - loss: 1.0481 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  418/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 240ms/step - loss: 1.0480 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  419/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 241ms/step - loss: 1.0480 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  420/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 241ms/step - loss: 1.0480 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  421/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 241ms/step - loss: 1.0479 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  422/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 241ms/step - loss: 1.0479 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  423/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 241ms/step - loss: 1.0478 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  424/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:42 241ms/step - loss: 1.0478 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  425/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:42 241ms/step - loss: 1.0477 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  426/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:42 241ms/step - loss: 1.0477 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  427/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:42 241ms/step - loss: 1.0476 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  428/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 241ms/step - loss: 1.0476 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  429/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 241ms/step - loss: 1.0476 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  430/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 241ms/step - loss: 1.0475 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  431/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 241ms/step - loss: 1.0475 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  432/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 241ms/step - loss: 1.0474 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  433/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:40 241ms/step - loss: 1.0474 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  434/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:40 241ms/step - loss: 1.0473 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  435/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:40 241ms/step - loss: 1.0473 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  436/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:40 241ms/step - loss: 1.0472 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  437/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:40 241ms/step - loss: 1.0472 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  438/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:39 241ms/step - loss: 1.0471 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  439/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:39 241ms/step - loss: 1.0471 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  440/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:39 241ms/step - loss: 1.0471 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  441/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:39 241ms/step - loss: 1.0470 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  442/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:39 241ms/step - loss: 1.0470 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  443/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 241ms/step - loss: 1.0469 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  444/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 241ms/step - loss: 1.0469 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  445/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 241ms/step - loss: 1.0468 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  446/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 241ms/step - loss: 1.0468 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  447/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 241ms/step - loss: 1.0467 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  448/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 241ms/step - loss: 1.0467 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  449/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 241ms/step - loss: 1.0466 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  450/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 242ms/step - loss: 1.0466 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  451/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 242ms/step - loss: 1.0465 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  452/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 242ms/step - loss: 1.0465 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  453/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 242ms/step - loss: 1.0464 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  454/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 242ms/step - loss: 1.0464 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  455/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 242ms/step - loss: 1.0464 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  456/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 242ms/step - loss: 1.0463 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  457/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:36 242ms/step - loss: 1.0463 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  458/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:36 243ms/step - loss: 1.0462 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  459/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:36 243ms/step - loss: 1.0462 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  460/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:36 243ms/step - loss: 1.0461 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  461/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:35 243ms/step - loss: 1.0461 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  462/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:35 243ms/step - loss: 1.0460 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  463/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:35 242ms/step - loss: 1.0460 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  464/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:35 242ms/step - loss: 1.0459 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  465/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:34 242ms/step - loss: 1.0459 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  466/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:34 242ms/step - loss: 1.0458 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  467/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:34 242ms/step - loss: 1.0458 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  468/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:33 242ms/step - loss: 1.0457 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  469/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:33 242ms/step - loss: 1.0457 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  470/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:33 242ms/step - loss: 1.0456 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  471/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:33 242ms/step - loss: 1.0456 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  472/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:32 242ms/step - loss: 1.0455 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  473/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:32 242ms/step - loss: 1.0455 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  474/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:32 242ms/step - loss: 1.0454 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  475/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:32 242ms/step - loss: 1.0454 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  476/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:31 242ms/step - loss: 1.0453 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  477/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:31 242ms/step - loss: 1.0453 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  478/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:31 242ms/step - loss: 1.0452 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  479/1599 ━━━━━[37m━━━━━━━━━━━━━━━  4:31 242ms/step - loss: 1.0452 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  480/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:30 242ms/step - loss: 1.0451 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  481/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:30 242ms/step - loss: 1.0451 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  482/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:30 242ms/step - loss: 1.0450 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  483/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:30 242ms/step - loss: 1.0450 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  484/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 242ms/step - loss: 1.0449 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  485/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 242ms/step - loss: 1.0449 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  486/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 242ms/step - loss: 1.0448 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  487/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 242ms/step - loss: 1.0448 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  488/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 242ms/step - loss: 1.0447 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  489/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 242ms/step - loss: 1.0447 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  490/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:28 242ms/step - loss: 1.0447 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  491/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:28 242ms/step - loss: 1.0446 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  492/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:28 242ms/step - loss: 1.0446 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  493/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:28 243ms/step - loss: 1.0445 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  494/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:28 243ms/step - loss: 1.0445 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  495/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 243ms/step - loss: 1.0444 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  496/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 243ms/step - loss: 1.0444 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  497/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 243ms/step - loss: 1.0443 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  498/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 243ms/step - loss: 1.0443 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  499/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 243ms/step - loss: 1.0443 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  500/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 243ms/step - loss: 1.0442 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  501/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 243ms/step - loss: 1.0442 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  502/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 243ms/step - loss: 1.0441 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  503/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 243ms/step - loss: 1.0441 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  504/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 243ms/step - loss: 1.0440 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  505/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 243ms/step - loss: 1.0440 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  506/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 244ms/step - loss: 1.0440 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  507/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 244ms/step - loss: 1.0439 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  508/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 244ms/step - loss: 1.0439 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  509/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 244ms/step - loss: 1.0438 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  510/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 244ms/step - loss: 1.0438 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  511/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 244ms/step - loss: 1.0437 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  512/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 244ms/step - loss: 1.0437 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  513/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:24 244ms/step - loss: 1.0437 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  514/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:24 244ms/step - loss: 1.0436 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  515/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:24 244ms/step - loss: 1.0436 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  516/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:24 244ms/step - loss: 1.0435 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  517/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:23 244ms/step - loss: 1.0435 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  518/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:23 244ms/step - loss: 1.0435 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  519/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:23 244ms/step - loss: 1.0434 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  520/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:23 244ms/step - loss: 1.0434 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  521/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:22 244ms/step - loss: 1.0433 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  522/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:22 244ms/step - loss: 1.0433 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  523/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:22 244ms/step - loss: 1.0433 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  524/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:22 244ms/step - loss: 1.0432 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  525/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:21 244ms/step - loss: 1.0432 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  526/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:21 244ms/step - loss: 1.0431 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  527/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:21 244ms/step - loss: 1.0431 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  528/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:21 244ms/step - loss: 1.0430 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  529/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:20 244ms/step - loss: 1.0430 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  530/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:20 244ms/step - loss: 1.0430 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  531/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:20 244ms/step - loss: 1.0429 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  532/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:20 244ms/step - loss: 1.0429 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  533/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:20 244ms/step - loss: 1.0428 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  534/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 244ms/step - loss: 1.0428 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  535/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 244ms/step - loss: 1.0428 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  536/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 244ms/step - loss: 1.0427 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  537/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 244ms/step - loss: 1.0427 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  538/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:18 244ms/step - loss: 1.0426 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  539/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:18 244ms/step - loss: 1.0426 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  540/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:18 244ms/step - loss: 1.0426 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  541/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:18 244ms/step - loss: 1.0425 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  542/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 244ms/step - loss: 1.0425 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  543/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 244ms/step - loss: 1.0425 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  544/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 244ms/step - loss: 1.0424 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  545/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 244ms/step - loss: 1.0424 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  546/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 244ms/step - loss: 1.0423 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  547/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:16 244ms/step - loss: 1.0423 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  548/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:16 244ms/step - loss: 1.0423 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  549/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:16 244ms/step - loss: 1.0422 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  550/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:16 244ms/step - loss: 1.0422 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  551/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 244ms/step - loss: 1.0422 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  552/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 244ms/step - loss: 1.0421 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  553/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 244ms/step - loss: 1.0421 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  554/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 244ms/step - loss: 1.0420 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  555/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 244ms/step - loss: 1.0420 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  556/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:14 244ms/step - loss: 1.0420 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  557/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:14 244ms/step - loss: 1.0419 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  558/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:14 244ms/step - loss: 1.0419 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  559/1599 ━━━━━━[37m━━━━━━━━━━━━━━  4:14 244ms/step - loss: 1.0419 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  560/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:14 245ms/step - loss: 1.0418 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  561/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0418 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  562/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0417 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  563/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0417 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  564/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0417 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  565/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0416 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  566/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0416 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  567/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0416 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  568/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 245ms/step - loss: 1.0415 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  569/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:12 245ms/step - loss: 1.0415 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  570/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:12 246ms/step - loss: 1.0415 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  571/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:12 246ms/step - loss: 1.0414 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  572/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:12 246ms/step - loss: 1.0414 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  573/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 246ms/step - loss: 1.0414 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  574/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 246ms/step - loss: 1.0413 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  575/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 246ms/step - loss: 1.0413 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  576/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 246ms/step - loss: 1.0413 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  577/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 246ms/step - loss: 1.0412 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  578/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 246ms/step - loss: 1.0412 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  579/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 246ms/step - loss: 1.0411 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  580/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 246ms/step - loss: 1.0411 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  581/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 246ms/step - loss: 1.0411 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  582/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 246ms/step - loss: 1.0410 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  583/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 246ms/step - loss: 1.0410 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  584/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 246ms/step - loss: 1.0410 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  585/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 247ms/step - loss: 1.0409 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  586/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 247ms/step - loss: 1.0409 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  587/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 247ms/step - loss: 1.0409 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  588/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 247ms/step - loss: 1.0408 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  589/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 247ms/step - loss: 1.0408 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  590/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:08 247ms/step - loss: 1.0408 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  591/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:08 247ms/step - loss: 1.0407 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  592/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:08 247ms/step - loss: 1.0407 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  593/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:08 247ms/step - loss: 1.0406 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  594/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:08 247ms/step - loss: 1.0406 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  595/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 247ms/step - loss: 1.0406 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  596/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 247ms/step - loss: 1.0405 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  597/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 247ms/step - loss: 1.0405 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  598/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 247ms/step - loss: 1.0405 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  599/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 247ms/step - loss: 1.0404 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  600/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 247ms/step - loss: 1.0404 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  601/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 247ms/step - loss: 1.0404 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  602/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 247ms/step - loss: 1.0403 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  603/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 247ms/step - loss: 1.0403 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  604/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 247ms/step - loss: 1.0403 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  605/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 247ms/step - loss: 1.0402 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  606/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 247ms/step - loss: 1.0402 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  607/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 247ms/step - loss: 1.0402 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  608/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 247ms/step - loss: 1.0401 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  609/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 247ms/step - loss: 1.0401 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  610/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 247ms/step - loss: 1.0400 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  611/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 247ms/step - loss: 1.0400 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  612/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 247ms/step - loss: 1.0400 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  613/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 247ms/step - loss: 1.0399 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  614/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 247ms/step - loss: 1.0399 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  615/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 247ms/step - loss: 1.0399 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  616/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:02 247ms/step - loss: 1.0398 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  617/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:02 247ms/step - loss: 1.0398 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  618/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:02 247ms/step - loss: 1.0398 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  619/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:02 247ms/step - loss: 1.0397 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  620/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 247ms/step - loss: 1.0397 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  621/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 247ms/step - loss: 1.0397 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  622/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 247ms/step - loss: 1.0396 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  623/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 247ms/step - loss: 1.0396 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  624/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 247ms/step - loss: 1.0396 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  625/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 248ms/step - loss: 1.0395 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  626/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:00 248ms/step - loss: 1.0395 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  627/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:00 248ms/step - loss: 1.0395 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  628/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:00 248ms/step - loss: 1.0394 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  629/1599 ━━━━━━━[37m━━━━━━━━━━━━━  4:00 248ms/step - loss: 1.0394 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  630/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:59 248ms/step - loss: 1.0394 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  631/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:59 248ms/step - loss: 1.0394 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  632/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:59 247ms/step - loss: 1.0393 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  633/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:59 247ms/step - loss: 1.0393 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  634/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:58 247ms/step - loss: 1.0393 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  635/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:58 247ms/step - loss: 1.0392 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  636/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:58 247ms/step - loss: 1.0392 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  637/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:58 247ms/step - loss: 1.0392 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  638/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:57 247ms/step - loss: 1.0391 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  639/1599 ━━━━━━━[37m━━━━━━━━━━━━━  3:57 247ms/step - loss: 1.0391 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  640/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:57 247ms/step - loss: 1.0391 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  641/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:57 247ms/step - loss: 1.0390 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  642/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:56 247ms/step - loss: 1.0390 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  643/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:56 247ms/step - loss: 1.0390 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  644/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:56 247ms/step - loss: 1.0389 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  645/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:56 247ms/step - loss: 1.0389 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  646/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 247ms/step - loss: 1.0389 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  647/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 248ms/step - loss: 1.0389 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  648/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 248ms/step - loss: 1.0388 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  649/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 248ms/step - loss: 1.0388 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  650/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 248ms/step - loss: 1.0388 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  651/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 248ms/step - loss: 1.0387 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  652/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 248ms/step - loss: 1.0387 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  653/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 248ms/step - loss: 1.0387 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  654/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 248ms/step - loss: 1.0386 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  655/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 248ms/step - loss: 1.0386 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  656/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 248ms/step - loss: 1.0386 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  657/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:53 248ms/step - loss: 1.0386 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  658/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:53 248ms/step - loss: 1.0385 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  659/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:53 248ms/step - loss: 1.0385 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  660/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:53 248ms/step - loss: 1.0385 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  661/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:52 248ms/step - loss: 1.0384 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  662/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:52 248ms/step - loss: 1.0384 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  663/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:52 248ms/step - loss: 1.0384 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  664/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:52 248ms/step - loss: 1.0384 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  665/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:51 248ms/step - loss: 1.0383 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  666/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:51 248ms/step - loss: 1.0383 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  667/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:51 248ms/step - loss: 1.0383 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  668/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:51 248ms/step - loss: 1.0382 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  669/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:50 248ms/step - loss: 1.0382 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  670/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:50 248ms/step - loss: 1.0382 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  671/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:50 248ms/step - loss: 1.0382 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  672/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:50 248ms/step - loss: 1.0381 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  673/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:49 248ms/step - loss: 1.0381 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  674/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:49 248ms/step - loss: 1.0381 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  675/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:49 248ms/step - loss: 1.0380 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  676/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:49 248ms/step - loss: 1.0380 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  677/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 248ms/step - loss: 1.0380 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  678/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 248ms/step - loss: 1.0380 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  679/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 248ms/step - loss: 1.0379 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  680/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 248ms/step - loss: 1.0379 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  681/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 248ms/step - loss: 1.0379 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  682/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:47 249ms/step - loss: 1.0379 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  683/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:47 249ms/step - loss: 1.0378 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  684/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:47 249ms/step - loss: 1.0378 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  685/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:47 249ms/step - loss: 1.0378 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  686/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:46 249ms/step - loss: 1.0378 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  687/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:46 249ms/step - loss: 1.0377 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  688/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:46 249ms/step - loss: 1.0377 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  689/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:46 249ms/step - loss: 1.0377 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  690/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:45 249ms/step - loss: 1.0376 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  691/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:45 249ms/step - loss: 1.0376 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  692/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:45 249ms/step - loss: 1.0376 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  693/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:45 249ms/step - loss: 1.0376 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  694/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:44 249ms/step - loss: 1.0375 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  695/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:44 249ms/step - loss: 1.0375 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  696/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:44 249ms/step - loss: 1.0375 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  697/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:44 249ms/step - loss: 1.0375 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  698/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:43 249ms/step - loss: 1.0374 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  699/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:43 249ms/step - loss: 1.0374 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  700/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:43 249ms/step - loss: 1.0374 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  701/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:43 249ms/step - loss: 1.0374 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  702/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:42 249ms/step - loss: 1.0373 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  703/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:42 249ms/step - loss: 1.0373 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  704/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:42 249ms/step - loss: 1.0373 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  705/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:42 249ms/step - loss: 1.0373 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  706/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 249ms/step - loss: 1.0372 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  707/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 249ms/step - loss: 1.0372 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  708/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 249ms/step - loss: 1.0372 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  709/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 249ms/step - loss: 1.0372 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  710/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 249ms/step - loss: 1.0371 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  711/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 249ms/step - loss: 1.0371 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  712/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 249ms/step - loss: 1.0371 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  713/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 249ms/step - loss: 1.0370 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  714/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 249ms/step - loss: 1.0370 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  715/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 249ms/step - loss: 1.0370 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  716/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:39 249ms/step - loss: 1.0370 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  717/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:39 249ms/step - loss: 1.0369 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  718/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:39 249ms/step - loss: 1.0369 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  719/1599 ━━━━━━━━[37m━━━━━━━━━━━━  3:38 249ms/step - loss: 1.0369 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  720/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:38 249ms/step - loss: 1.0369 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  721/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:38 249ms/step - loss: 1.0368 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  722/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:38 249ms/step - loss: 1.0368 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  723/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:37 249ms/step - loss: 1.0368 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  724/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:37 249ms/step - loss: 1.0368 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  725/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:37 249ms/step - loss: 1.0368 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  726/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:37 249ms/step - loss: 1.0367 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  727/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:37 249ms/step - loss: 1.0367 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  728/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:37 249ms/step - loss: 1.0367 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  729/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:37 249ms/step - loss: 1.0367 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  730/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:36 249ms/step - loss: 1.0366 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  731/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:36 249ms/step - loss: 1.0366 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  732/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:36 249ms/step - loss: 1.0366 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  733/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:35 249ms/step - loss: 1.0366 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  734/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:35 249ms/step - loss: 1.0365 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  735/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:35 249ms/step - loss: 1.0365 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  736/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:35 249ms/step - loss: 1.0365 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  737/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:34 249ms/step - loss: 1.0365 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  738/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:34 249ms/step - loss: 1.0364 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  739/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:34 249ms/step - loss: 1.0364 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  740/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:34 249ms/step - loss: 1.0364 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  741/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:33 249ms/step - loss: 1.0364 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  742/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:33 249ms/step - loss: 1.0363 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  743/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:33 249ms/step - loss: 1.0363 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  744/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:33 249ms/step - loss: 1.0363 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  745/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 249ms/step - loss: 1.0363 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  746/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 249ms/step - loss: 1.0363 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  747/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 249ms/step - loss: 1.0362 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  748/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 249ms/step - loss: 1.0362 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  749/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 249ms/step - loss: 1.0362 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  750/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 250ms/step - loss: 1.0362 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  751/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 250ms/step - loss: 1.0361 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  752/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 250ms/step - loss: 1.0361 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  753/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 250ms/step - loss: 1.0361 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  754/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 250ms/step - loss: 1.0361 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  755/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:30 250ms/step - loss: 1.0360 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  756/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:30 250ms/step - loss: 1.0360 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  757/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:30 250ms/step - loss: 1.0360 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  758/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 250ms/step - loss: 1.0360 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  759/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 250ms/step - loss: 1.0360 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  760/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 250ms/step - loss: 1.0359 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  761/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 250ms/step - loss: 1.0359 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  762/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 250ms/step - loss: 1.0359 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  763/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:28 250ms/step - loss: 1.0359 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  764/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:28 250ms/step - loss: 1.0358 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  765/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:28 250ms/step - loss: 1.0358 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  766/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:28 250ms/step - loss: 1.0358 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  767/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:27 250ms/step - loss: 1.0358 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  768/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:27 250ms/step - loss: 1.0358 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  769/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:27 250ms/step - loss: 1.0357 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  770/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:27 250ms/step - loss: 1.0357 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  771/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:26 250ms/step - loss: 1.0357 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  772/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:26 250ms/step - loss: 1.0357 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  773/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:26 250ms/step - loss: 1.0357 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  774/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:26 250ms/step - loss: 1.0356 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  775/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:25 250ms/step - loss: 1.0356 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  776/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:25 250ms/step - loss: 1.0356 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  777/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:25 250ms/step - loss: 1.0356 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  778/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:25 250ms/step - loss: 1.0356 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  779/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 250ms/step - loss: 1.0355 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  780/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 250ms/step - loss: 1.0355 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  781/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 250ms/step - loss: 1.0355 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  782/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 250ms/step - loss: 1.0355 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  783/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 250ms/step - loss: 1.0355 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  784/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 250ms/step - loss: 1.0354 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  785/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 250ms/step - loss: 1.0354 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  786/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 251ms/step - loss: 1.0354 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  787/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 251ms/step - loss: 1.0354 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  788/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 251ms/step - loss: 1.0354 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  789/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 251ms/step - loss: 1.0353 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  790/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 251ms/step - loss: 1.0353 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  791/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:22 251ms/step - loss: 1.0353 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  792/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:22 251ms/step - loss: 1.0353 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  793/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:22 251ms/step - loss: 1.0352 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  794/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:22 251ms/step - loss: 1.0352 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  795/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:22 251ms/step - loss: 1.0352 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  796/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:21 251ms/step - loss: 1.0352 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  797/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:21 251ms/step - loss: 1.0352 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  798/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:21 251ms/step - loss: 1.0351 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  799/1599 ━━━━━━━━━[37m━━━━━━━━━━━  3:21 251ms/step - loss: 1.0351 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  800/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:20 251ms/step - loss: 1.0351 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  801/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:20 251ms/step - loss: 1.0351 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  802/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:20 251ms/step - loss: 1.0351 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  803/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:20 251ms/step - loss: 1.0351 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  804/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:19 251ms/step - loss: 1.0350 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  805/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:19 252ms/step - loss: 1.0350 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  806/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:19 252ms/step - loss: 1.0350 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  807/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:19 252ms/step - loss: 1.0350 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  808/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:19 252ms/step - loss: 1.0350 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  809/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:18 252ms/step - loss: 1.0349 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  810/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:18 252ms/step - loss: 1.0349 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  811/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:18 252ms/step - loss: 1.0349 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  812/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:18 252ms/step - loss: 1.0349 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  813/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:17 252ms/step - loss: 1.0349 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  814/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:17 252ms/step - loss: 1.0348 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  815/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:17 252ms/step - loss: 1.0348 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  816/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:17 252ms/step - loss: 1.0348 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  817/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:17 252ms/step - loss: 1.0348 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  818/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:17 252ms/step - loss: 1.0348 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  819/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:16 252ms/step - loss: 1.0347 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  820/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:16 253ms/step - loss: 1.0347 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  821/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:16 253ms/step - loss: 1.0347 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  822/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:16 253ms/step - loss: 1.0347 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  823/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:16 253ms/step - loss: 1.0347 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  824/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 253ms/step - loss: 1.0346 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  825/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 253ms/step - loss: 1.0346 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  826/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 253ms/step - loss: 1.0346 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  827/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 253ms/step - loss: 1.0346 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  828/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 253ms/step - loss: 1.0346 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  829/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 253ms/step - loss: 1.0346 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  830/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 253ms/step - loss: 1.0345 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  831/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 253ms/step - loss: 1.0345 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  832/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 253ms/step - loss: 1.0345 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  833/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:13 253ms/step - loss: 1.0345 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  834/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:13 253ms/step - loss: 1.0345 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  835/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:13 253ms/step - loss: 1.0344 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  836/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:13 253ms/step - loss: 1.0344 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  837/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:12 253ms/step - loss: 1.0344 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  838/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:12 253ms/step - loss: 1.0344 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  839/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:12 253ms/step - loss: 1.0344 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  840/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:12 253ms/step - loss: 1.0344 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  841/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:11 253ms/step - loss: 1.0343 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  842/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:11 253ms/step - loss: 1.0343 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  843/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:11 253ms/step - loss: 1.0343 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  844/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:11 253ms/step - loss: 1.0343 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  845/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 253ms/step - loss: 1.0343 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  846/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 253ms/step - loss: 1.0343 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  847/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 253ms/step - loss: 1.0342 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  848/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 253ms/step - loss: 1.0342 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  849/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:09 253ms/step - loss: 1.0342 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  850/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:09 253ms/step - loss: 1.0342 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  851/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:09 253ms/step - loss: 1.0342 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  852/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:09 253ms/step - loss: 1.0341 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  853/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:08 253ms/step - loss: 1.0341 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  854/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:08 253ms/step - loss: 1.0341 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  855/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:08 253ms/step - loss: 1.0341 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  856/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:08 253ms/step - loss: 1.0341 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  857/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:07 253ms/step - loss: 1.0341 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  858/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:07 253ms/step - loss: 1.0340 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  859/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:07 253ms/step - loss: 1.0340 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  860/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:07 253ms/step - loss: 1.0340 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  861/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 253ms/step - loss: 1.0340 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  862/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 253ms/step - loss: 1.0340 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  863/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 253ms/step - loss: 1.0340 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  864/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 253ms/step - loss: 1.0339 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  865/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 253ms/step - loss: 1.0339 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  866/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 253ms/step - loss: 1.0339 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  867/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 253ms/step - loss: 1.0339 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  868/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 253ms/step - loss: 1.0339 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  869/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:04 253ms/step - loss: 1.0338 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  870/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:04 253ms/step - loss: 1.0338 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  871/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:04 253ms/step - loss: 1.0338 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  872/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:04 253ms/step - loss: 1.0338 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  873/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:03 253ms/step - loss: 1.0338 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  874/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:03 253ms/step - loss: 1.0338 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  875/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:03 253ms/step - loss: 1.0337 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  876/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:03 253ms/step - loss: 1.0337 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  877/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:02 253ms/step - loss: 1.0337 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  878/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:02 253ms/step - loss: 1.0337 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  879/1599 ━━━━━━━━━━[37m━━━━━━━━━━  3:02 253ms/step - loss: 1.0337 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  880/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:02 253ms/step - loss: 1.0337 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  881/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:01 253ms/step - loss: 1.0336 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  882/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:01 253ms/step - loss: 1.0336 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  883/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:01 253ms/step - loss: 1.0336 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  884/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:01 253ms/step - loss: 1.0336 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  885/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:00 253ms/step - loss: 1.0336 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  886/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:00 253ms/step - loss: 1.0336 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  887/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:00 253ms/step - loss: 1.0335 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  888/1599 ━━━━━━━━━━━[37m━━━━━━━━━  3:00 253ms/step - loss: 1.0335 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  889/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:59 253ms/step - loss: 1.0335 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  890/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:59 253ms/step - loss: 1.0335 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  891/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:59 253ms/step - loss: 1.0335 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  892/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:59 253ms/step - loss: 1.0335 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  893/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:58 253ms/step - loss: 1.0334 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  894/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:58 253ms/step - loss: 1.0334 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  895/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:58 253ms/step - loss: 1.0334 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  896/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:58 253ms/step - loss: 1.0334 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  897/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:57 253ms/step - loss: 1.0334 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  898/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:57 253ms/step - loss: 1.0334 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  899/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:57 253ms/step - loss: 1.0334 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  900/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:57 253ms/step - loss: 1.0333 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  901/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:56 253ms/step - loss: 1.0333 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  902/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:56 253ms/step - loss: 1.0333 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  903/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:56 253ms/step - loss: 1.0333 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  904/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:56 253ms/step - loss: 1.0333 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  905/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:55 253ms/step - loss: 1.0333 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  906/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:55 253ms/step - loss: 1.0332 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  907/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:55 253ms/step - loss: 1.0332 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  908/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:55 253ms/step - loss: 1.0332 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  909/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:54 254ms/step - loss: 1.0332 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  910/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:54 254ms/step - loss: 1.0332 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  911/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:54 254ms/step - loss: 1.0332 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  912/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:54 253ms/step - loss: 1.0332 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  913/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:53 253ms/step - loss: 1.0331 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  914/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:53 254ms/step - loss: 1.0331 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  915/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:53 254ms/step - loss: 1.0331 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  916/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:53 254ms/step - loss: 1.0331 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  917/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:52 254ms/step - loss: 1.0331 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  918/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:52 254ms/step - loss: 1.0331 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  919/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:52 254ms/step - loss: 1.0330 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  920/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:52 254ms/step - loss: 1.0330 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  921/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:51 254ms/step - loss: 1.0330 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  922/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:51 254ms/step - loss: 1.0330 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  923/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:51 254ms/step - loss: 1.0330 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  924/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:51 254ms/step - loss: 1.0330 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  925/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 254ms/step - loss: 1.0329 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  926/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 254ms/step - loss: 1.0329 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  927/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 254ms/step - loss: 1.0329 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  928/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 254ms/step - loss: 1.0329 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  929/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 254ms/step - loss: 1.0329 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  930/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:49 254ms/step - loss: 1.0329 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  931/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:49 254ms/step - loss: 1.0329 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  932/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:49 254ms/step - loss: 1.0328 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  933/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:49 254ms/step - loss: 1.0328 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  934/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:48 254ms/step - loss: 1.0328 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  935/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:48 254ms/step - loss: 1.0328 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  936/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:48 254ms/step - loss: 1.0328 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  937/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:48 254ms/step - loss: 1.0327 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  938/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:47 254ms/step - loss: 1.0327 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  939/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:47 254ms/step - loss: 1.0327 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  940/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:47 254ms/step - loss: 1.0327 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  941/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:46 254ms/step - loss: 1.0327 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  942/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:46 254ms/step - loss: 1.0327 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  943/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:46 254ms/step - loss: 1.0326 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  944/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:46 254ms/step - loss: 1.0326 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  945/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:45 254ms/step - loss: 1.0326 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  946/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:45 254ms/step - loss: 1.0326 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  947/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:45 254ms/step - loss: 1.0326 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  948/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:45 254ms/step - loss: 1.0326 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  949/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:44 254ms/step - loss: 1.0325 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  950/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:44 254ms/step - loss: 1.0325 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  951/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:44 254ms/step - loss: 1.0325 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  952/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:44 254ms/step - loss: 1.0325 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  953/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:43 254ms/step - loss: 1.0325 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  954/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:43 254ms/step - loss: 1.0325 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  955/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:43 254ms/step - loss: 1.0324 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  956/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:43 254ms/step - loss: 1.0324 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  957/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:42 254ms/step - loss: 1.0324 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  958/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:42 254ms/step - loss: 1.0324 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  959/1599 ━━━━━━━━━━━[37m━━━━━━━━━  2:42 254ms/step - loss: 1.0324 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  960/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:42 254ms/step - loss: 1.0323 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  961/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:41 254ms/step - loss: 1.0323 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  962/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:41 254ms/step - loss: 1.0323 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  963/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:41 254ms/step - loss: 1.0323 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  964/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:41 254ms/step - loss: 1.0323 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  965/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:40 254ms/step - loss: 1.0322 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  966/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:40 254ms/step - loss: 1.0322 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  967/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:40 254ms/step - loss: 1.0322 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  968/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:40 254ms/step - loss: 1.0322 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  969/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:39 254ms/step - loss: 1.0322 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  970/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:39 254ms/step - loss: 1.0322 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  971/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:39 254ms/step - loss: 1.0321 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  972/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:39 254ms/step - loss: 1.0321 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  973/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:38 254ms/step - loss: 1.0321 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  974/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:38 254ms/step - loss: 1.0321 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  975/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:38 254ms/step - loss: 1.0321 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  976/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:38 254ms/step - loss: 1.0320 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  977/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:37 254ms/step - loss: 1.0320 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  978/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:37 254ms/step - loss: 1.0320 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  979/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:37 254ms/step - loss: 1.0320 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  980/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:37 254ms/step - loss: 1.0320 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  981/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:36 254ms/step - loss: 1.0319 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  982/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:36 254ms/step - loss: 1.0319 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  983/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:36 254ms/step - loss: 1.0319 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  984/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:36 254ms/step - loss: 1.0319 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  985/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 254ms/step - loss: 1.0319 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  986/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 254ms/step - loss: 1.0319 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  987/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 254ms/step - loss: 1.0318 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  988/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 254ms/step - loss: 1.0318 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  989/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 254ms/step - loss: 1.0318 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  990/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:34 254ms/step - loss: 1.0318 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  991/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:34 254ms/step - loss: 1.0318 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  992/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:34 254ms/step - loss: 1.0317 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  993/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:34 254ms/step - loss: 1.0317 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  994/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:33 254ms/step - loss: 1.0317 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  995/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:33 254ms/step - loss: 1.0317 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  996/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:33 254ms/step - loss: 1.0317 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  997/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:33 254ms/step - loss: 1.0316 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  998/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:32 254ms/step - loss: 1.0316 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  999/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:32 254ms/step - loss: 1.0316 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1000/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:32 254ms/step - loss: 1.0316 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1001/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:32 255ms/step - loss: 1.0316 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1002/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:31 255ms/step - loss: 1.0316 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1003/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:31 255ms/step - loss: 1.0315 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1004/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:31 255ms/step - loss: 1.0315 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1005/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:31 255ms/step - loss: 1.0315 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1006/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 255ms/step - loss: 1.0315 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1007/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 255ms/step - loss: 1.0315 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1008/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 255ms/step - loss: 1.0315 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
 1009/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 255ms/step - loss: 1.0314 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1010/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 255ms/step - loss: 1.0314 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1011/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:29 255ms/step - loss: 1.0314 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1012/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:29 255ms/step - loss: 1.0314 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1013/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:29 255ms/step - loss: 1.0314 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1014/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:29 255ms/step - loss: 1.0313 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1015/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:28 255ms/step - loss: 1.0313 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1016/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:28 255ms/step - loss: 1.0313 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1017/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:28 255ms/step - loss: 1.0313 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1018/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:28 255ms/step - loss: 1.0313 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1019/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 255ms/step - loss: 1.0313 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1020/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 255ms/step - loss: 1.0312 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1021/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 255ms/step - loss: 1.0312 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1022/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 255ms/step - loss: 1.0312 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1023/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 255ms/step - loss: 1.0312 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1024/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:26 255ms/step - loss: 1.0312 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1025/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:26 255ms/step - loss: 1.0311 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1026/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:26 255ms/step - loss: 1.0311 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1027/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:26 256ms/step - loss: 1.0311 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1028/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 256ms/step - loss: 1.0311 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1029/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 256ms/step - loss: 1.0311 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1030/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 256ms/step - loss: 1.0310 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1031/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 256ms/step - loss: 1.0310 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1032/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 256ms/step - loss: 1.0310 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1033/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:24 256ms/step - loss: 1.0310 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1034/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:24 256ms/step - loss: 1.0310 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1035/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:24 256ms/step - loss: 1.0310 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1036/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:24 256ms/step - loss: 1.0309 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1037/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:23 256ms/step - loss: 1.0309 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1038/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:23 256ms/step - loss: 1.0309 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1039/1599 ━━━━━━━━━━━━[37m━━━━━━━━  2:23 256ms/step - loss: 1.0309 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1040/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:23 256ms/step - loss: 1.0309 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1041/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:22 256ms/step - loss: 1.0308 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1042/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:22 256ms/step - loss: 1.0308 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1043/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:22 256ms/step - loss: 1.0308 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1044/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:22 256ms/step - loss: 1.0308 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1045/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:21 256ms/step - loss: 1.0308 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1046/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:21 256ms/step - loss: 1.0308 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1047/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:21 256ms/step - loss: 1.0307 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1048/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:21 256ms/step - loss: 1.0307 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1049/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:20 256ms/step - loss: 1.0307 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1050/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:20 256ms/step - loss: 1.0307 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1051/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:20 256ms/step - loss: 1.0307 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1052/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:20 256ms/step - loss: 1.0306 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1053/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:20 257ms/step - loss: 1.0306 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1054/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:19 257ms/step - loss: 1.0306 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1055/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:19 257ms/step - loss: 1.0306 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1056/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:19 257ms/step - loss: 1.0306 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1057/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:19 257ms/step - loss: 1.0306 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1058/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:19 257ms/step - loss: 1.0305 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1059/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:18 257ms/step - loss: 1.0305 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1060/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:18 257ms/step - loss: 1.0305 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1061/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:18 257ms/step - loss: 1.0305 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1062/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:18 257ms/step - loss: 1.0305 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1063/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:17 257ms/step - loss: 1.0304 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1064/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:17 257ms/step - loss: 1.0304 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1065/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:17 257ms/step - loss: 1.0304 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1066/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:17 258ms/step - loss: 1.0304 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1067/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:17 258ms/step - loss: 1.0304 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1068/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:16 258ms/step - loss: 1.0304 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1069/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:16 258ms/step - loss: 1.0303 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1070/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:16 258ms/step - loss: 1.0303 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1071/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:16 258ms/step - loss: 1.0303 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1072/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:15 258ms/step - loss: 1.0303 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1073/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:15 258ms/step - loss: 1.0303 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1074/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:15 258ms/step - loss: 1.0302 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1075/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:15 258ms/step - loss: 1.0302 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1076/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:14 258ms/step - loss: 1.0302 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1077/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:14 258ms/step - loss: 1.0302 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1078/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:14 258ms/step - loss: 1.0302 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1079/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:14 258ms/step - loss: 1.0302 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1080/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:13 258ms/step - loss: 1.0301 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1081/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:13 258ms/step - loss: 1.0301 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1082/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:13 258ms/step - loss: 1.0301 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1083/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:13 258ms/step - loss: 1.0301 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1084/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:13 258ms/step - loss: 1.0301 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1085/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:12 258ms/step - loss: 1.0301 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1086/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:12 258ms/step - loss: 1.0300 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1087/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:12 258ms/step - loss: 1.0300 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1088/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:12 258ms/step - loss: 1.0300 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1089/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:11 258ms/step - loss: 1.0300 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1090/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:11 259ms/step - loss: 1.0300 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1091/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:11 259ms/step - loss: 1.0299 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1092/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:11 259ms/step - loss: 1.0299 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1093/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:10 259ms/step - loss: 1.0299 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1094/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:10 259ms/step - loss: 1.0299 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1095/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:10 259ms/step - loss: 1.0299 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1096/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:10 259ms/step - loss: 1.0299 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1097/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:09 259ms/step - loss: 1.0298 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1098/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:09 259ms/step - loss: 1.0298 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1099/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:09 259ms/step - loss: 1.0298 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1100/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:09 259ms/step - loss: 1.0298 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1101/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:08 259ms/step - loss: 1.0298 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1102/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:08 259ms/step - loss: 1.0298 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1103/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:08 259ms/step - loss: 1.0297 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1104/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:08 259ms/step - loss: 1.0297 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1105/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:07 259ms/step - loss: 1.0297 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1106/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:07 259ms/step - loss: 1.0297 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1107/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:07 259ms/step - loss: 1.0297 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1108/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:07 259ms/step - loss: 1.0297 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1109/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:06 259ms/step - loss: 1.0296 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1110/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:06 259ms/step - loss: 1.0296 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1111/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:06 259ms/step - loss: 1.0296 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1112/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:06 259ms/step - loss: 1.0296 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1113/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:05 259ms/step - loss: 1.0296 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1114/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:05 259ms/step - loss: 1.0295 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1115/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:05 259ms/step - loss: 1.0295 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1116/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:05 259ms/step - loss: 1.0295 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1117/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:04 259ms/step - loss: 1.0295 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1118/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:04 259ms/step - loss: 1.0295 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1119/1599 ━━━━━━━━━━━━━[37m━━━━━━━  2:04 259ms/step - loss: 1.0295 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1120/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:04 259ms/step - loss: 1.0294 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1121/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:04 259ms/step - loss: 1.0294 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1122/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:03 259ms/step - loss: 1.0294 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1123/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:03 260ms/step - loss: 1.0294 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1124/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:03 260ms/step - loss: 1.0294 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1125/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:03 260ms/step - loss: 1.0293 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1126/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:02 260ms/step - loss: 1.0293 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1127/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:02 260ms/step - loss: 1.0293 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1128/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:02 260ms/step - loss: 1.0293 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1129/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:02 260ms/step - loss: 1.0293 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1130/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:01 260ms/step - loss: 1.0292 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1131/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:01 260ms/step - loss: 1.0292 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1132/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:01 260ms/step - loss: 1.0292 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1133/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:00 260ms/step - loss: 1.0292 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1134/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:00 260ms/step - loss: 1.0292 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1135/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:00 260ms/step - loss: 1.0292 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1136/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:00 260ms/step - loss: 1.0291 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1137/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:59 260ms/step - loss: 1.0291 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1138/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:59 260ms/step - loss: 1.0291 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1139/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:59 260ms/step - loss: 1.0291 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1140/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:59 260ms/step - loss: 1.0291 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1141/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:58 260ms/step - loss: 1.0291 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1142/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:58 260ms/step - loss: 1.0290 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1143/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:58 260ms/step - loss: 1.0290 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1144/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:58 260ms/step - loss: 1.0290 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1145/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:57 260ms/step - loss: 1.0290 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1146/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:57 260ms/step - loss: 1.0290 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1147/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:57 260ms/step - loss: 1.0289 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1148/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:57 260ms/step - loss: 1.0289 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1149/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:56 260ms/step - loss: 1.0289 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1150/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:56 260ms/step - loss: 1.0289 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1151/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:56 260ms/step - loss: 1.0289 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1152/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:56 260ms/step - loss: 1.0288 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1153/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:55 260ms/step - loss: 1.0288 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1154/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:55 260ms/step - loss: 1.0288 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1155/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:55 260ms/step - loss: 1.0288 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1156/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:55 260ms/step - loss: 1.0288 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1157/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:54 260ms/step - loss: 1.0288 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1158/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:54 260ms/step - loss: 1.0287 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1159/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:54 260ms/step - loss: 1.0287 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1160/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:54 260ms/step - loss: 1.0287 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1161/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:53 260ms/step - loss: 1.0287 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1162/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:53 260ms/step - loss: 1.0287 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1163/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:53 260ms/step - loss: 1.0286 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1164/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:53 260ms/step - loss: 1.0286 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1165/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:52 260ms/step - loss: 1.0286 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1166/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:52 260ms/step - loss: 1.0286 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1167/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:52 260ms/step - loss: 1.0286 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1168/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:52 260ms/step - loss: 1.0286 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1169/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:51 260ms/step - loss: 1.0285 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1170/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:51 260ms/step - loss: 1.0285 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1171/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:51 260ms/step - loss: 1.0285 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1172/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:51 260ms/step - loss: 1.0285 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1173/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:50 260ms/step - loss: 1.0285 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1174/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:50 260ms/step - loss: 1.0285 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1175/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:50 260ms/step - loss: 1.0284 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1176/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:50 260ms/step - loss: 1.0284 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1177/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:49 260ms/step - loss: 1.0284 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1178/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:49 260ms/step - loss: 1.0284 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1179/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:49 260ms/step - loss: 1.0284 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1180/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:49 260ms/step - loss: 1.0283 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1181/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:48 260ms/step - loss: 1.0283 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1182/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:48 260ms/step - loss: 1.0283 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1183/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:48 260ms/step - loss: 1.0283 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1184/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:48 260ms/step - loss: 1.0283 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1185/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:47 260ms/step - loss: 1.0283 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1186/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:47 260ms/step - loss: 1.0282 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1187/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:47 260ms/step - loss: 1.0282 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1188/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:46 260ms/step - loss: 1.0282 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1189/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:46 260ms/step - loss: 1.0282 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1190/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:46 260ms/step - loss: 1.0282 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1191/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:46 260ms/step - loss: 1.0281 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1192/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:45 260ms/step - loss: 1.0281 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1193/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:45 260ms/step - loss: 1.0281 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1194/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:45 260ms/step - loss: 1.0281 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1195/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:45 260ms/step - loss: 1.0281 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1196/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:44 260ms/step - loss: 1.0281 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1197/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:44 260ms/step - loss: 1.0280 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1198/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:44 260ms/step - loss: 1.0280 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1199/1599 ━━━━━━━━━━━━━━[37m━━━━━━  1:44 260ms/step - loss: 1.0280 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1200/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:43 260ms/step - loss: 1.0280 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1201/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:43 260ms/step - loss: 1.0280 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1202/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:43 261ms/step - loss: 1.0280 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1203/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:43 261ms/step - loss: 1.0279 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1204/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:42 261ms/step - loss: 1.0279 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1205/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:42 261ms/step - loss: 1.0279 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1206/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:42 261ms/step - loss: 1.0279 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1207/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:42 261ms/step - loss: 1.0279 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1208/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:41 261ms/step - loss: 1.0279 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1209/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:41 261ms/step - loss: 1.0278 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1210/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:41 261ms/step - loss: 1.0278 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1211/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:41 261ms/step - loss: 1.0278 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1212/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:40 261ms/step - loss: 1.0278 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1213/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:40 261ms/step - loss: 1.0278 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1214/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:40 261ms/step - loss: 1.0278 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1215/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:40 261ms/step - loss: 1.0277 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1216/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:39 261ms/step - loss: 1.0277 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1217/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:39 261ms/step - loss: 1.0277 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1218/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:39 261ms/step - loss: 1.0277 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1219/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:39 261ms/step - loss: 1.0277 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1220/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:38 261ms/step - loss: 1.0276 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1221/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:38 261ms/step - loss: 1.0276 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1222/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:38 261ms/step - loss: 1.0276 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1223/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:38 261ms/step - loss: 1.0276 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1224/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:37 261ms/step - loss: 1.0276 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1225/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:37 261ms/step - loss: 1.0276 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1226/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:37 261ms/step - loss: 1.0275 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1227/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:36 261ms/step - loss: 1.0275 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1228/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:36 261ms/step - loss: 1.0275 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1229/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:36 261ms/step - loss: 1.0275 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1230/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:36 261ms/step - loss: 1.0275 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1231/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:35 261ms/step - loss: 1.0275 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1232/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:35 261ms/step - loss: 1.0274 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1233/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:35 261ms/step - loss: 1.0274 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1234/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:35 261ms/step - loss: 1.0274 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1235/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:34 261ms/step - loss: 1.0274 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1236/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:34 261ms/step - loss: 1.0274 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1237/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:34 261ms/step - loss: 1.0274 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1238/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:34 261ms/step - loss: 1.0273 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1239/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:33 261ms/step - loss: 1.0273 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1240/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:33 261ms/step - loss: 1.0273 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1241/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:33 261ms/step - loss: 1.0273 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1242/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:33 261ms/step - loss: 1.0273 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1243/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:33 261ms/step - loss: 1.0273 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1244/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:32 261ms/step - loss: 1.0272 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1245/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:32 261ms/step - loss: 1.0272 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1246/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:32 261ms/step - loss: 1.0272 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1247/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:31 261ms/step - loss: 1.0272 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1248/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:31 261ms/step - loss: 1.0272 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1249/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:31 261ms/step - loss: 1.0272 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1250/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:31 261ms/step - loss: 1.0271 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1251/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:30 261ms/step - loss: 1.0271 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1252/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:30 261ms/step - loss: 1.0271 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1253/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:30 261ms/step - loss: 1.0271 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1254/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:30 261ms/step - loss: 1.0271 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1255/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:29 261ms/step - loss: 1.0271 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1256/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:29 261ms/step - loss: 1.0270 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1257/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:29 261ms/step - loss: 1.0270 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1258/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:29 261ms/step - loss: 1.0270 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1259/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:28 261ms/step - loss: 1.0270 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1260/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:28 261ms/step - loss: 1.0270 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1261/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:28 261ms/step - loss: 1.0270 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1262/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:28 261ms/step - loss: 1.0269 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1263/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:27 261ms/step - loss: 1.0269 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1264/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:27 261ms/step - loss: 1.0269 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1265/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:27 261ms/step - loss: 1.0269 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1266/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:27 261ms/step - loss: 1.0269 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1267/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:26 261ms/step - loss: 1.0269 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1268/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:26 261ms/step - loss: 1.0268 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1269/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:26 261ms/step - loss: 1.0268 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1270/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:26 261ms/step - loss: 1.0268 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1271/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:25 262ms/step - loss: 1.0268 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1272/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:25 262ms/step - loss: 1.0268 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1273/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:25 262ms/step - loss: 1.0268 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1274/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:25 262ms/step - loss: 1.0268 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1275/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:24 262ms/step - loss: 1.0267 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1276/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:24 262ms/step - loss: 1.0267 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1277/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:24 262ms/step - loss: 1.0267 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1278/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:24 262ms/step - loss: 1.0267 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1279/1599 ━━━━━━━━━━━━━━━[37m━━━━━  1:23 262ms/step - loss: 1.0267 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1280/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:23 262ms/step - loss: 1.0267 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1281/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:23 262ms/step - loss: 1.0266 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1282/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:23 262ms/step - loss: 1.0266 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1283/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:22 262ms/step - loss: 1.0266 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1284/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:22 262ms/step - loss: 1.0266 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1285/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:22 262ms/step - loss: 1.0266 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1286/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:22 262ms/step - loss: 1.0266 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1287/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:21 262ms/step - loss: 1.0265 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1288/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:21 262ms/step - loss: 1.0265 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1289/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:21 262ms/step - loss: 1.0265 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1290/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:20 262ms/step - loss: 1.0265 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1291/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:20 262ms/step - loss: 1.0265 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1292/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:20 262ms/step - loss: 1.0265 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1293/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:20 262ms/step - loss: 1.0265 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1294/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:19 262ms/step - loss: 1.0264 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1295/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:19 262ms/step - loss: 1.0264 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1296/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:19 262ms/step - loss: 1.0264 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1297/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:19 262ms/step - loss: 1.0264 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1298/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:18 262ms/step - loss: 1.0264 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1299/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:18 262ms/step - loss: 1.0264 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1300/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:18 262ms/step - loss: 1.0263 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1301/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:18 262ms/step - loss: 1.0263 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1302/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:17 262ms/step - loss: 1.0263 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1303/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:17 262ms/step - loss: 1.0263 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1304/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:17 262ms/step - loss: 1.0263 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1305/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:17 262ms/step - loss: 1.0263 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1306/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:16 262ms/step - loss: 1.0263 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1307/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:16 262ms/step - loss: 1.0262 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1308/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:16 262ms/step - loss: 1.0262 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1309/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:16 262ms/step - loss: 1.0262 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1310/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:15 262ms/step - loss: 1.0262 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1311/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:15 262ms/step - loss: 1.0262 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1312/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:15 262ms/step - loss: 1.0262 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1313/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 262ms/step - loss: 1.0261 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1314/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 262ms/step - loss: 1.0261 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1315/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 262ms/step - loss: 1.0261 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1316/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 262ms/step - loss: 1.0261 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1317/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 262ms/step - loss: 1.0261 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1318/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:13 262ms/step - loss: 1.0261 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1319/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:13 262ms/step - loss: 1.0261 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1320/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:13 263ms/step - loss: 1.0260 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1321/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:12 263ms/step - loss: 1.0260 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1322/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:12 263ms/step - loss: 1.0260 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1323/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:12 263ms/step - loss: 1.0260 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1324/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:12 263ms/step - loss: 1.0260 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1325/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:11 262ms/step - loss: 1.0260 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1326/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:11 262ms/step - loss: 1.0259 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1327/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:11 262ms/step - loss: 1.0259 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1328/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:11 263ms/step - loss: 1.0259 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1329/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:10 263ms/step - loss: 1.0259 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1330/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:10 263ms/step - loss: 1.0259 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1331/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:10 263ms/step - loss: 1.0259 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1332/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:10 263ms/step - loss: 1.0259 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1333/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:09 263ms/step - loss: 1.0258 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1334/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:09 263ms/step - loss: 1.0258 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1335/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:09 263ms/step - loss: 1.0258 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1336/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:09 263ms/step - loss: 1.0258 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1337/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:08 263ms/step - loss: 1.0258 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1338/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:08 263ms/step - loss: 1.0258 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1339/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:08 263ms/step - loss: 1.0257 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1340/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:08 263ms/step - loss: 1.0257 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1341/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:07 263ms/step - loss: 1.0257 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1342/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:07 263ms/step - loss: 1.0257 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1343/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:07 263ms/step - loss: 1.0257 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1344/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:06 263ms/step - loss: 1.0257 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1345/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:06 263ms/step - loss: 1.0257 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1346/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:06 263ms/step - loss: 1.0256 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1347/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:06 263ms/step - loss: 1.0256 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1348/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:05 263ms/step - loss: 1.0256 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1349/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:05 263ms/step - loss: 1.0256 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1350/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:05 263ms/step - loss: 1.0256 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1351/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:05 263ms/step - loss: 1.0256 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1352/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:04 263ms/step - loss: 1.0256 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1353/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:04 263ms/step - loss: 1.0255 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1354/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:04 263ms/step - loss: 1.0255 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1355/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:04 263ms/step - loss: 1.0255 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1356/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:03 263ms/step - loss: 1.0255 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1357/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:03 263ms/step - loss: 1.0255 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1358/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:03 263ms/step - loss: 1.0255 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1359/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:03 263ms/step - loss: 1.0255 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1360/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:02 263ms/step - loss: 1.0254 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1361/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:02 263ms/step - loss: 1.0254 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1362/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:02 263ms/step - loss: 1.0254 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1363/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:02 263ms/step - loss: 1.0254 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1364/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:01 263ms/step - loss: 1.0254 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1365/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:01 263ms/step - loss: 1.0254 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1366/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:01 263ms/step - loss: 1.0254 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1367/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:01 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1368/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:00 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1369/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:00 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1370/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:00 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1371/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:00 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1372/1599 ━━━━━━━━━━━━━━━━━[37m━━━  59s 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097 

<div class="k-default-codeblock">
```

```
</div>
 1373/1599 ━━━━━━━━━━━━━━━━━[37m━━━  59s 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1374/1599 ━━━━━━━━━━━━━━━━━[37m━━━  59s 263ms/step - loss: 1.0253 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1375/1599 ━━━━━━━━━━━━━━━━━[37m━━━  58s 263ms/step - loss: 1.0252 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1376/1599 ━━━━━━━━━━━━━━━━━[37m━━━  58s 263ms/step - loss: 1.0252 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1377/1599 ━━━━━━━━━━━━━━━━━[37m━━━  58s 263ms/step - loss: 1.0252 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1378/1599 ━━━━━━━━━━━━━━━━━[37m━━━  58s 263ms/step - loss: 1.0252 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1379/1599 ━━━━━━━━━━━━━━━━━[37m━━━  57s 263ms/step - loss: 1.0252 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1380/1599 ━━━━━━━━━━━━━━━━━[37m━━━  57s 263ms/step - loss: 1.0252 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1381/1599 ━━━━━━━━━━━━━━━━━[37m━━━  57s 263ms/step - loss: 1.0252 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1382/1599 ━━━━━━━━━━━━━━━━━[37m━━━  57s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1383/1599 ━━━━━━━━━━━━━━━━━[37m━━━  56s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1384/1599 ━━━━━━━━━━━━━━━━━[37m━━━  56s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1385/1599 ━━━━━━━━━━━━━━━━━[37m━━━  56s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1386/1599 ━━━━━━━━━━━━━━━━━[37m━━━  56s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1387/1599 ━━━━━━━━━━━━━━━━━[37m━━━  55s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1388/1599 ━━━━━━━━━━━━━━━━━[37m━━━  55s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1389/1599 ━━━━━━━━━━━━━━━━━[37m━━━  55s 263ms/step - loss: 1.0251 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1390/1599 ━━━━━━━━━━━━━━━━━[37m━━━  55s 263ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1391/1599 ━━━━━━━━━━━━━━━━━[37m━━━  54s 263ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1392/1599 ━━━━━━━━━━━━━━━━━[37m━━━  54s 263ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1393/1599 ━━━━━━━━━━━━━━━━━[37m━━━  54s 264ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1394/1599 ━━━━━━━━━━━━━━━━━[37m━━━  54s 264ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1395/1599 ━━━━━━━━━━━━━━━━━[37m━━━  53s 264ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1396/1599 ━━━━━━━━━━━━━━━━━[37m━━━  53s 264ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1397/1599 ━━━━━━━━━━━━━━━━━[37m━━━  53s 264ms/step - loss: 1.0250 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1398/1599 ━━━━━━━━━━━━━━━━━[37m━━━  52s 264ms/step - loss: 1.0249 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1399/1599 ━━━━━━━━━━━━━━━━━[37m━━━  52s 264ms/step - loss: 1.0249 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1400/1599 ━━━━━━━━━━━━━━━━━[37m━━━  52s 264ms/step - loss: 1.0249 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1401/1599 ━━━━━━━━━━━━━━━━━[37m━━━  52s 264ms/step - loss: 1.0249 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1402/1599 ━━━━━━━━━━━━━━━━━[37m━━━  51s 264ms/step - loss: 1.0249 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1403/1599 ━━━━━━━━━━━━━━━━━[37m━━━  51s 264ms/step - loss: 1.0249 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1404/1599 ━━━━━━━━━━━━━━━━━[37m━━━  51s 264ms/step - loss: 1.0249 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1405/1599 ━━━━━━━━━━━━━━━━━[37m━━━  51s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1406/1599 ━━━━━━━━━━━━━━━━━[37m━━━  50s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1407/1599 ━━━━━━━━━━━━━━━━━[37m━━━  50s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1408/1599 ━━━━━━━━━━━━━━━━━[37m━━━  50s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1409/1599 ━━━━━━━━━━━━━━━━━[37m━━━  50s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1410/1599 ━━━━━━━━━━━━━━━━━[37m━━━  49s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1411/1599 ━━━━━━━━━━━━━━━━━[37m━━━  49s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1412/1599 ━━━━━━━━━━━━━━━━━[37m━━━  49s 264ms/step - loss: 1.0248 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1413/1599 ━━━━━━━━━━━━━━━━━[37m━━━  49s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1414/1599 ━━━━━━━━━━━━━━━━━[37m━━━  48s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1415/1599 ━━━━━━━━━━━━━━━━━[37m━━━  48s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1416/1599 ━━━━━━━━━━━━━━━━━[37m━━━  48s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1417/1599 ━━━━━━━━━━━━━━━━━[37m━━━  47s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1418/1599 ━━━━━━━━━━━━━━━━━[37m━━━  47s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1419/1599 ━━━━━━━━━━━━━━━━━[37m━━━  47s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1420/1599 ━━━━━━━━━━━━━━━━━[37m━━━  47s 264ms/step - loss: 1.0247 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1421/1599 ━━━━━━━━━━━━━━━━━[37m━━━  46s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1422/1599 ━━━━━━━━━━━━━━━━━[37m━━━  46s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1423/1599 ━━━━━━━━━━━━━━━━━[37m━━━  46s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1424/1599 ━━━━━━━━━━━━━━━━━[37m━━━  46s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1425/1599 ━━━━━━━━━━━━━━━━━[37m━━━  45s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1426/1599 ━━━━━━━━━━━━━━━━━[37m━━━  45s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1427/1599 ━━━━━━━━━━━━━━━━━[37m━━━  45s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1428/1599 ━━━━━━━━━━━━━━━━━[37m━━━  45s 264ms/step - loss: 1.0246 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1429/1599 ━━━━━━━━━━━━━━━━━[37m━━━  44s 264ms/step - loss: 1.0245 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1430/1599 ━━━━━━━━━━━━━━━━━[37m━━━  44s 263ms/step - loss: 1.0245 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1431/1599 ━━━━━━━━━━━━━━━━━[37m━━━  44s 263ms/step - loss: 1.0245 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1432/1599 ━━━━━━━━━━━━━━━━━[37m━━━  43s 263ms/step - loss: 1.0245 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1433/1599 ━━━━━━━━━━━━━━━━━[37m━━━  43s 263ms/step - loss: 1.0245 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1434/1599 ━━━━━━━━━━━━━━━━━[37m━━━  43s 263ms/step - loss: 1.0245 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1435/1599 ━━━━━━━━━━━━━━━━━[37m━━━  43s 263ms/step - loss: 1.0245 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1436/1599 ━━━━━━━━━━━━━━━━━[37m━━━  42s 263ms/step - loss: 1.0245 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1437/1599 ━━━━━━━━━━━━━━━━━[37m━━━  42s 263ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1438/1599 ━━━━━━━━━━━━━━━━━[37m━━━  42s 263ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1439/1599 ━━━━━━━━━━━━━━━━━[37m━━━  42s 263ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1440/1599 ━━━━━━━━━━━━━━━━━━[37m━━  41s 263ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1441/1599 ━━━━━━━━━━━━━━━━━━[37m━━  41s 264ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1442/1599 ━━━━━━━━━━━━━━━━━━[37m━━  41s 264ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1443/1599 ━━━━━━━━━━━━━━━━━━[37m━━  41s 264ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1444/1599 ━━━━━━━━━━━━━━━━━━[37m━━  40s 264ms/step - loss: 1.0244 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1445/1599 ━━━━━━━━━━━━━━━━━━[37m━━  40s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1446/1599 ━━━━━━━━━━━━━━━━━━[37m━━  40s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1447/1599 ━━━━━━━━━━━━━━━━━━[37m━━  40s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1448/1599 ━━━━━━━━━━━━━━━━━━[37m━━  39s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1449/1599 ━━━━━━━━━━━━━━━━━━[37m━━  39s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1450/1599 ━━━━━━━━━━━━━━━━━━[37m━━  39s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1451/1599 ━━━━━━━━━━━━━━━━━━[37m━━  39s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1452/1599 ━━━━━━━━━━━━━━━━━━[37m━━  38s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1453/1599 ━━━━━━━━━━━━━━━━━━[37m━━  38s 264ms/step - loss: 1.0243 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1454/1599 ━━━━━━━━━━━━━━━━━━[37m━━  38s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1455/1599 ━━━━━━━━━━━━━━━━━━[37m━━  37s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1456/1599 ━━━━━━━━━━━━━━━━━━[37m━━  37s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1457/1599 ━━━━━━━━━━━━━━━━━━[37m━━  37s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1458/1599 ━━━━━━━━━━━━━━━━━━[37m━━  37s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1459/1599 ━━━━━━━━━━━━━━━━━━[37m━━  36s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1460/1599 ━━━━━━━━━━━━━━━━━━[37m━━  36s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1461/1599 ━━━━━━━━━━━━━━━━━━[37m━━  36s 264ms/step - loss: 1.0242 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1462/1599 ━━━━━━━━━━━━━━━━━━[37m━━  36s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1463/1599 ━━━━━━━━━━━━━━━━━━[37m━━  35s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1464/1599 ━━━━━━━━━━━━━━━━━━[37m━━  35s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1465/1599 ━━━━━━━━━━━━━━━━━━[37m━━  35s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1466/1599 ━━━━━━━━━━━━━━━━━━[37m━━  35s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1467/1599 ━━━━━━━━━━━━━━━━━━[37m━━  34s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1468/1599 ━━━━━━━━━━━━━━━━━━[37m━━  34s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1469/1599 ━━━━━━━━━━━━━━━━━━[37m━━  34s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1470/1599 ━━━━━━━━━━━━━━━━━━[37m━━  34s 264ms/step - loss: 1.0241 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1471/1599 ━━━━━━━━━━━━━━━━━━[37m━━  33s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1472/1599 ━━━━━━━━━━━━━━━━━━[37m━━  33s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1473/1599 ━━━━━━━━━━━━━━━━━━[37m━━  33s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1474/1599 ━━━━━━━━━━━━━━━━━━[37m━━  32s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1475/1599 ━━━━━━━━━━━━━━━━━━[37m━━  32s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1476/1599 ━━━━━━━━━━━━━━━━━━[37m━━  32s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1477/1599 ━━━━━━━━━━━━━━━━━━[37m━━  32s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1478/1599 ━━━━━━━━━━━━━━━━━━[37m━━  31s 264ms/step - loss: 1.0240 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1479/1599 ━━━━━━━━━━━━━━━━━━[37m━━  31s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1480/1599 ━━━━━━━━━━━━━━━━━━[37m━━  31s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1481/1599 ━━━━━━━━━━━━━━━━━━[37m━━  31s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1482/1599 ━━━━━━━━━━━━━━━━━━[37m━━  30s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1483/1599 ━━━━━━━━━━━━━━━━━━[37m━━  30s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1484/1599 ━━━━━━━━━━━━━━━━━━[37m━━  30s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1485/1599 ━━━━━━━━━━━━━━━━━━[37m━━  30s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1486/1599 ━━━━━━━━━━━━━━━━━━[37m━━  29s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1487/1599 ━━━━━━━━━━━━━━━━━━[37m━━  29s 264ms/step - loss: 1.0239 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1488/1599 ━━━━━━━━━━━━━━━━━━[37m━━  29s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1489/1599 ━━━━━━━━━━━━━━━━━━[37m━━  29s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1490/1599 ━━━━━━━━━━━━━━━━━━[37m━━  28s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1491/1599 ━━━━━━━━━━━━━━━━━━[37m━━  28s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1492/1599 ━━━━━━━━━━━━━━━━━━[37m━━  28s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1493/1599 ━━━━━━━━━━━━━━━━━━[37m━━  27s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1494/1599 ━━━━━━━━━━━━━━━━━━[37m━━  27s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1495/1599 ━━━━━━━━━━━━━━━━━━[37m━━  27s 264ms/step - loss: 1.0238 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1496/1599 ━━━━━━━━━━━━━━━━━━[37m━━  27s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1497/1599 ━━━━━━━━━━━━━━━━━━[37m━━  26s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1498/1599 ━━━━━━━━━━━━━━━━━━[37m━━  26s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1499/1599 ━━━━━━━━━━━━━━━━━━[37m━━  26s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1500/1599 ━━━━━━━━━━━━━━━━━━[37m━━  26s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1501/1599 ━━━━━━━━━━━━━━━━━━[37m━━  25s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1502/1599 ━━━━━━━━━━━━━━━━━━[37m━━  25s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1503/1599 ━━━━━━━━━━━━━━━━━━[37m━━  25s 264ms/step - loss: 1.0237 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1504/1599 ━━━━━━━━━━━━━━━━━━[37m━━  25s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1505/1599 ━━━━━━━━━━━━━━━━━━[37m━━  24s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1506/1599 ━━━━━━━━━━━━━━━━━━[37m━━  24s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1507/1599 ━━━━━━━━━━━━━━━━━━[37m━━  24s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1508/1599 ━━━━━━━━━━━━━━━━━━[37m━━  24s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1509/1599 ━━━━━━━━━━━━━━━━━━[37m━━  23s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1510/1599 ━━━━━━━━━━━━━━━━━━[37m━━  23s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1511/1599 ━━━━━━━━━━━━━━━━━━[37m━━  23s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1512/1599 ━━━━━━━━━━━━━━━━━━[37m━━  22s 264ms/step - loss: 1.0236 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1513/1599 ━━━━━━━━━━━━━━━━━━[37m━━  22s 264ms/step - loss: 1.0235 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1514/1599 ━━━━━━━━━━━━━━━━━━[37m━━  22s 264ms/step - loss: 1.0235 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1515/1599 ━━━━━━━━━━━━━━━━━━[37m━━  22s 264ms/step - loss: 1.0235 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1516/1599 ━━━━━━━━━━━━━━━━━━[37m━━  21s 264ms/step - loss: 1.0235 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1517/1599 ━━━━━━━━━━━━━━━━━━[37m━━  21s 264ms/step - loss: 1.0235 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1518/1599 ━━━━━━━━━━━━━━━━━━[37m━━  21s 264ms/step - loss: 1.0235 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1519/1599 ━━━━━━━━━━━━━━━━━━[37m━━  21s 265ms/step - loss: 1.0235 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1520/1599 ━━━━━━━━━━━━━━━━━━━[37m━  20s 265ms/step - loss: 1.0235 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1521/1599 ━━━━━━━━━━━━━━━━━━━[37m━  20s 265ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1522/1599 ━━━━━━━━━━━━━━━━━━━[37m━  20s 265ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1523/1599 ━━━━━━━━━━━━━━━━━━━[37m━  20s 265ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1524/1599 ━━━━━━━━━━━━━━━━━━━[37m━  19s 264ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1525/1599 ━━━━━━━━━━━━━━━━━━━[37m━  19s 264ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1526/1599 ━━━━━━━━━━━━━━━━━━━[37m━  19s 264ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1527/1599 ━━━━━━━━━━━━━━━━━━━[37m━  19s 264ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1528/1599 ━━━━━━━━━━━━━━━━━━━[37m━  18s 264ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1529/1599 ━━━━━━━━━━━━━━━━━━━[37m━  18s 264ms/step - loss: 1.0234 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1530/1599 ━━━━━━━━━━━━━━━━━━━[37m━  18s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1531/1599 ━━━━━━━━━━━━━━━━━━━[37m━  17s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1532/1599 ━━━━━━━━━━━━━━━━━━━[37m━  17s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1533/1599 ━━━━━━━━━━━━━━━━━━━[37m━  17s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1534/1599 ━━━━━━━━━━━━━━━━━━━[37m━  17s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1535/1599 ━━━━━━━━━━━━━━━━━━━[37m━  16s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1536/1599 ━━━━━━━━━━━━━━━━━━━[37m━  16s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1537/1599 ━━━━━━━━━━━━━━━━━━━[37m━  16s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1538/1599 ━━━━━━━━━━━━━━━━━━━[37m━  16s 264ms/step - loss: 1.0233 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1539/1599 ━━━━━━━━━━━━━━━━━━━[37m━  15s 264ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1540/1599 ━━━━━━━━━━━━━━━━━━━[37m━  15s 264ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1541/1599 ━━━━━━━━━━━━━━━━━━━[37m━  15s 264ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1542/1599 ━━━━━━━━━━━━━━━━━━━[37m━  15s 264ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1543/1599 ━━━━━━━━━━━━━━━━━━━[37m━  14s 265ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1544/1599 ━━━━━━━━━━━━━━━━━━━[37m━  14s 265ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1545/1599 ━━━━━━━━━━━━━━━━━━━[37m━  14s 265ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1546/1599 ━━━━━━━━━━━━━━━━━━━[37m━  14s 265ms/step - loss: 1.0232 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1547/1599 ━━━━━━━━━━━━━━━━━━━[37m━  13s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1548/1599 ━━━━━━━━━━━━━━━━━━━[37m━  13s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1549/1599 ━━━━━━━━━━━━━━━━━━━[37m━  13s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1550/1599 ━━━━━━━━━━━━━━━━━━━[37m━  12s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1551/1599 ━━━━━━━━━━━━━━━━━━━[37m━  12s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1552/1599 ━━━━━━━━━━━━━━━━━━━[37m━  12s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1553/1599 ━━━━━━━━━━━━━━━━━━━[37m━  12s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1554/1599 ━━━━━━━━━━━━━━━━━━━[37m━  11s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1555/1599 ━━━━━━━━━━━━━━━━━━━[37m━  11s 265ms/step - loss: 1.0231 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1556/1599 ━━━━━━━━━━━━━━━━━━━[37m━  11s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1557/1599 ━━━━━━━━━━━━━━━━━━━[37m━  11s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1558/1599 ━━━━━━━━━━━━━━━━━━━[37m━  10s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1559/1599 ━━━━━━━━━━━━━━━━━━━[37m━  10s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1560/1599 ━━━━━━━━━━━━━━━━━━━[37m━  10s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1561/1599 ━━━━━━━━━━━━━━━━━━━[37m━  10s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1562/1599 ━━━━━━━━━━━━━━━━━━━[37m━  9s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086 

<div class="k-default-codeblock">
```

```
</div>
 1563/1599 ━━━━━━━━━━━━━━━━━━━[37m━  9s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1564/1599 ━━━━━━━━━━━━━━━━━━━[37m━  9s 265ms/step - loss: 1.0230 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1565/1599 ━━━━━━━━━━━━━━━━━━━[37m━  8s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1566/1599 ━━━━━━━━━━━━━━━━━━━[37m━  8s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1567/1599 ━━━━━━━━━━━━━━━━━━━[37m━  8s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1568/1599 ━━━━━━━━━━━━━━━━━━━[37m━  8s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1569/1599 ━━━━━━━━━━━━━━━━━━━[37m━  7s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1570/1599 ━━━━━━━━━━━━━━━━━━━[37m━  7s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1571/1599 ━━━━━━━━━━━━━━━━━━━[37m━  7s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1572/1599 ━━━━━━━━━━━━━━━━━━━[37m━  7s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1573/1599 ━━━━━━━━━━━━━━━━━━━[37m━  6s 265ms/step - loss: 1.0229 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1574/1599 ━━━━━━━━━━━━━━━━━━━[37m━  6s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1575/1599 ━━━━━━━━━━━━━━━━━━━[37m━  6s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1576/1599 ━━━━━━━━━━━━━━━━━━━[37m━  6s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1577/1599 ━━━━━━━━━━━━━━━━━━━[37m━  5s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1578/1599 ━━━━━━━━━━━━━━━━━━━[37m━  5s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1579/1599 ━━━━━━━━━━━━━━━━━━━[37m━  5s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1580/1599 ━━━━━━━━━━━━━━━━━━━[37m━  5s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1581/1599 ━━━━━━━━━━━━━━━━━━━[37m━  4s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1582/1599 ━━━━━━━━━━━━━━━━━━━[37m━  4s 265ms/step - loss: 1.0228 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1583/1599 ━━━━━━━━━━━━━━━━━━━[37m━  4s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1584/1599 ━━━━━━━━━━━━━━━━━━━[37m━  3s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1585/1599 ━━━━━━━━━━━━━━━━━━━[37m━  3s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1586/1599 ━━━━━━━━━━━━━━━━━━━[37m━  3s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1587/1599 ━━━━━━━━━━━━━━━━━━━[37m━  3s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1588/1599 ━━━━━━━━━━━━━━━━━━━[37m━  2s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1589/1599 ━━━━━━━━━━━━━━━━━━━[37m━  2s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1590/1599 ━━━━━━━━━━━━━━━━━━━[37m━  2s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1591/1599 ━━━━━━━━━━━━━━━━━━━[37m━  2s 265ms/step - loss: 1.0227 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1592/1599 ━━━━━━━━━━━━━━━━━━━[37m━  1s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1593/1599 ━━━━━━━━━━━━━━━━━━━[37m━  1s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1594/1599 ━━━━━━━━━━━━━━━━━━━[37m━  1s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1595/1599 ━━━━━━━━━━━━━━━━━━━[37m━  1s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1596/1599 ━━━━━━━━━━━━━━━━━━━[37m━  0s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1597/1599 ━━━━━━━━━━━━━━━━━━━[37m━  0s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1598/1599 ━━━━━━━━━━━━━━━━━━━[37m━  0s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1599/1599 ━━━━━━━━━━━━━━━━━━━━ 0s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1599/1599 ━━━━━━━━━━━━━━━━━━━━ 424s 265ms/step - loss: 1.0226 - mean_absolute_error: 0.8084


<div class="k-default-codeblock">
```
Test MAE: 0.779

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:159: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()

```
</div>
You should achieve a Mean Absolute Error (MAE) at or around 0.7 on the test data.

---
## Conclusion

The BST model uses the Transformer layer in its architecture to capture the sequential signals underlying
users’ behavior sequences for recommendation.

You can try training this model with different configurations, for example, by increasing
the input sequence length and training the model for a larger number of epochs. In addition,
you can try including other features like movie release year and customer
zipcode, and including cross features like sex X genre.
