# Graph representation learning with node2vec

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/05/15<br>
**Last modified:** 2026/02/04<br>
**Description:** Implementing the node2vec model to generate embeddings for movies from the MovieLens dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/graph/ipynb/node2vec_movielens.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/graph/node2vec_movielens.py)



---
## Introduction

Learning useful representations from objects structured as graphs is useful for
a variety of machine learning (ML) applications—such as social and communication networks analysis,
biomedicine studies, and recommendation systems.
[Graph representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) aims to
learn embeddings for the graph nodes, which can be used for a variety of ML tasks
such as node label prediction (e.g. categorizing an article based on its citations)
and link prediction (e.g. recommending an interest group to a user in a social network).

[node2vec](https://arxiv.org/abs/1607.00653) is a simple, yet scalable and effective
technique for learning low-dimensional embeddings for nodes in a graph by optimizing
a neighborhood-preserving objective. The aim is to learn similar embeddings for
neighboring nodes, with respect to the graph structure.

Given your data items structured as a graph (where the items are represented as
nodes and the relationship between items are represented as edges),
node2vec works as follows:

1. Generate item sequences using (biased) random walk.
2. Create positive and negative training examples from these sequences.
3. Train a [word2vec](https://www.tensorflow.org/tutorials/text/word2vec) model
(skip-gram) to learn embeddings for the items.

In this example, we demonstrate the node2vec technique on the
[small version of the Movielens dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
to learn movie embeddings. Such a dataset can be represented as a graph by treating
the movies as nodes, and creating edges between movies that have similar ratings
by the users. The learnt movie embeddings can be used for tasks such as movie recommendation,
or movie genres prediction.

This example requires `networkx` package, which can be installed using the following command:

```shell
pip install networkx
```

---
## Setup


```python
import os
from collections import defaultdict
import math
import networkx as nx
import random
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import keras
from keras import ops
from keras import layers
import matplotlib.pyplot as plt

# Set seed for reproducibility
keras.utils.set_random_seed(42)
os.environ["KERAS_BACKEND"] = "jax"  # "jax", "torch", "tensorflow"
```

---
## Download the MovieLens dataset and prepare the data

The small version of the MovieLens dataset includes around 100k ratings
from 610 users on 9,742 movies.

First, let's download the dataset. The downloaded folder will contain
three data files: `users.csv`, `movies.csv`, and `ratings.csv`. In this example,
we will only need the `movies.dat`, and `ratings.dat` data files.


```python
urlretrieve(
    "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "movielens.zip"
)
ZipFile("movielens.zip", "r").extractall()
```

Then, we load the data into a Pandas DataFrame and perform some basic preprocessing.


```python
# Load movies to a DataFrame.
movies = pd.read_csv("ml-latest-small/movies.csv")
# Create a `movieId` string.
movies["movieId"] = movies["movieId"].apply(lambda x: f"movie_{x}")

# Load ratings to a DataFrame.
ratings = pd.read_csv("ml-latest-small/ratings.csv")
# Convert the `ratings` to floating point
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))
# Create the `movie_id` string.
ratings["movieId"] = ratings["movieId"].apply(lambda x: f"movie_{x}")

print("Movies data shape:", movies.shape)
print("Ratings data shape:", ratings.shape)
```

<div class="k-default-codeblock">
```
Movies data shape: (9742, 3)
Ratings data shape: (100836, 4)
```
</div>

Let's inspect a sample instance of the `ratings` DataFrame.


```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<div class="k-default-codeblock">
```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>movie_1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>movie_3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>movie_6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>movie_47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>movie_50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>



Next, let's check a sample instance of the `movies` DataFrame.


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<div class="k-default-codeblock">
```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>movie_1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>movie_2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>movie_3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>movie_4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>movie_5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



Implement two utility functions for the `movies` DataFrame.


```python

def get_movie_title_by_id(movieId):
    return list(movies[movies.movieId == movieId].title)[0]


def get_movie_id_by_title(title):
    return list(movies[movies.title == title].movieId)[0]

```

---
## Construct the Movies graph

We create an edge between two movie nodes in the graph if both movies are rated
by the same user >= `min_rating`. The weight of the edge will be based on the
[pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
between the two movies, which is computed as: `log(xy) - log(x) - log(y) + log(D)`, where:

* `xy` is how many users rated both movie `x` and movie `y` with >= `min_rating`.
* `x` is how many users rated movie `x` >= `min_rating`.
* `y` is how many users rated movie `y` >= `min_rating`.
* `D` total number of movie ratings >= `min_rating`.

### Step 1: create the weighted edges between movies.


```python
min_rating = 5
pair_frequency = defaultdict(int)
item_frequency = defaultdict(int)

# Filter instances where rating is greater than or equal to min_rating.
rated_movies = ratings[ratings.rating >= min_rating]
# Group instances by user.
movies_grouped_by_users = list(rated_movies.groupby("userId"))
for group in tqdm(
    movies_grouped_by_users,
    position=0,
    leave=True,
    desc="Compute movie rating frequencies",
):
    # Get a list of movies rated by the user.
    current_movies = list(group[1]["movieId"])

    for i in range(len(current_movies)):
        item_frequency[current_movies[i]] += 1
        for j in range(i + 1, len(current_movies)):
            x = min(current_movies[i], current_movies[j])
            y = max(current_movies[i], current_movies[j])
            pair_frequency[(x, y)] += 1
```

    
Compute movie rating frequencies:   0%|                                                                               | 0/573 [00:00<?, ?it/s]

    
Compute movie rating frequencies:  49%|█████████████████████████████████                                  | 283/573 [00:00<00:00, 2654.37it/s]

    
Compute movie rating frequencies:  96%|████████████████████████████████████████████████████████████████▏  | 549/573 [00:00<00:00, 2520.45it/s]

    
Compute movie rating frequencies: 100%|███████████████████████████████████████████████████████████████████| 573/573 [00:00<00:00, 2322.37it/s]

    


### Step 2: create the graph with the nodes and the edges

To reduce the number of edges between nodes, we only add an edge between movies
if the weight of the edge is greater than `min_weight`.


```python
min_weight = 10
D = math.log(sum(item_frequency.values()))

# Create the movies undirected graph.
movies_graph = nx.Graph()
# Add weighted edges between movies.
# This automatically adds the movie nodes to the graph.
for pair in tqdm(
    pair_frequency, position=0, leave=True, desc="Creating the movie graph"
):
    x, y = pair
    xy_frequency = pair_frequency[pair]
    x_frequency = item_frequency[x]
    y_frequency = item_frequency[y]
    pmi = math.log(xy_frequency) - math.log(x_frequency) - math.log(y_frequency) + D
    weight = pmi * xy_frequency
    # Only include edges with weight >= min_weight.
    if weight >= min_weight:
        movies_graph.add_edge(x, y, weight=weight)
```

    
Creating the movie graph:   0%|                                                                                    | 0/298586 [00:00<?, ?it/s]

    
Creating the movie graph:  42%|███████████████████████████▌                                      | 124515/298586 [00:00<00:00, 1245097.75it/s]

    
Creating the movie graph:  89%|██████████████████████████████████████████████████████████▉       | 266765/298586 [00:00<00:00, 1349428.26it/s]

    
Creating the movie graph: 100%|██████████████████████████████████████████████████████████████████| 298586/298586 [00:00<00:00, 1355053.16it/s]

    


Let's display the total number of nodes and edges in the graph.
Note that the number of nodes is less than the total number of movies,
since only the movies that have edges to other movies are added.


```python
print("Total number of graph nodes:", movies_graph.number_of_nodes())
print("Total number of graph edges:", movies_graph.number_of_edges())
```

<div class="k-default-codeblock">
```
Total number of graph nodes: 1405
Total number of graph edges: 40043
```
</div>

Let's display the average node degree (number of neighbours) in the graph.


```python
degrees = []
for node in movies_graph.nodes:
    degrees.append(movies_graph.degree[node])

print("Average node degree:", round(sum(degrees) / len(degrees), 2))
```

<div class="k-default-codeblock">
```
Average node degree: 57.0
```
</div>

### Step 3: Create vocabulary and a mapping from tokens to integer indices

The vocabulary is the nodes (movie IDs) in the graph.


```python
vocabulary = ["NA"] + list(movies_graph.nodes)
vocabulary_lookup = {token: idx for idx, token in enumerate(vocabulary)}
```

---
## Implement the biased random walk

A random walk starts from a given node, and randomly picks a neighbour node to move to.
If the edges are weighted, the neighbour is selected *probabilistically* with
respect to weights of the edges between the current node and its neighbours.
This procedure is repeated for `num_steps` to generate a sequence of *related* nodes.

The [*biased* random walk](https://en.wikipedia.org/wiki/Biased_random_walk_on_a_graph) balances between **breadth-first sampling**
(where only local neighbours are visited) and **depth-first sampling**
(where  distant neighbours are visited) by introducing the following two parameters:

1. **Return parameter** (`p`): Controls the likelihood of immediately revisiting
a node in the walk. Setting it to a high value encourages moderate exploration,
while setting it to a low value would keep the walk local.
2. **In-out parameter** (`q`): Allows the search to differentiate
between *inward* and *outward* nodes. Setting it to a high value biases the
random walk towards local nodes, while setting it to a low value biases the walk
to visit nodes which are further away.


```python

def next_step(graph, previous, current, p, q):
    neighbors = list(graph.neighbors(current))

    weights = []
    for neighbor in neighbors:
        if neighbor == previous:
            weights.append(graph[current][neighbor]["weight"] / p)
        elif graph.has_edge(neighbor, previous):
            weights.append(graph[current][neighbor]["weight"])
        else:
            weights.append(graph[current][neighbor]["weight"] / q)

    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]

    next_node = np.random.choice(neighbors, size=1, p=probabilities)[0]
    return next_node


def random_walk(graph, num_walks, num_steps, p, q):
    walks = []
    nodes = list(graph.nodes())
    for walk_iteration in range(num_walks):
        random.shuffle(nodes)
        for node in tqdm(nodes, desc=f"Random walks iteration {walk_iteration + 1}"):
            walk = [node]
            while len(walk) < num_steps:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                walk.append(next_step(graph, previous, current, p, q))
            walks.append([vocabulary_lookup[token] for token in walk])
    return walks

```

---
## Generate training data using the biased random walk

You can explore different configurations of `p` and `q` to different results of
related movies.


```python
# Random walk return parameter.
p = 1
# Random walk in-out parameter.
q = 1
# Number of iterations of random walks.
num_walks = 5
# Number of steps of each random walk.
num_steps = 10
walks = random_walk(movies_graph, num_walks, num_steps, p, q)

print("Number of walks generated:", len(walks))
```

    
Random walks iteration 1:   0%|                                                                                      | 0/1405 [00:00<?, ?it/s]

    
Random walks iteration 1:   6%|████▏                                                                       | 78/1405 [00:00<00:01, 770.26it/s]

    
Random walks iteration 1:  11%|████████▍                                                                  | 159/1405 [00:00<00:01, 791.24it/s]

    
Random walks iteration 1:  17%|████████████▊                                                              | 240/1405 [00:00<00:01, 796.93it/s]

    
Random walks iteration 1:  24%|██████████████████                                                         | 339/1405 [00:00<00:01, 869.95it/s]

    
Random walks iteration 1:  31%|███████████████████████▍                                                   | 438/1405 [00:00<00:01, 912.00it/s]

    
Random walks iteration 1:  39%|████████████████████████████▉                                              | 542/1405 [00:00<00:00, 952.69it/s]

    
Random walks iteration 1:  45%|██████████████████████████████████                                         | 639/1405 [00:00<00:00, 955.77it/s]

    
Random walks iteration 1:  52%|███████████████████████████████████████▏                                   | 735/1405 [00:00<00:00, 951.24it/s]

    
Random walks iteration 1:  59%|████████████████████████████████████████████▌                              | 835/1405 [00:00<00:00, 966.05it/s]

    
Random walks iteration 1:  66%|█████████████████████████████████████████████████▊                         | 932/1405 [00:01<00:00, 948.85it/s]

    
Random walks iteration 1:  73%|██████████████████████████████████████████████████████▏                   | 1029/1405 [00:01<00:00, 954.36it/s]

    
Random walks iteration 1:  80%|███████████████████████████████████████████████████████████▎              | 1127/1405 [00:01<00:00, 960.53it/s]

    
Random walks iteration 1:  87%|████████████████████████████████████████████████████████████████▌         | 1225/1405 [00:01<00:00, 966.30it/s]

    
Random walks iteration 1:  95%|█████████████████████████████████████████████████████████████████████▎   | 1334/1405 [00:01<00:00, 1000.41it/s]

    
Random walks iteration 1: 100%|██████████████████████████████████████████████████████████████████████████| 1405/1405 [00:01<00:00, 943.49it/s]

    


    
Random walks iteration 2:   0%|                                                                                      | 0/1405 [00:00<?, ?it/s]

    
Random walks iteration 2:   7%|█████▏                                                                      | 96/1405 [00:00<00:01, 959.82it/s]

    
Random walks iteration 2:  14%|██████████▍                                                                | 195/1405 [00:00<00:01, 974.04it/s]

    
Random walks iteration 2:  21%|███████████████▋                                                           | 293/1405 [00:00<00:01, 974.90it/s]

    
Random walks iteration 2:  28%|████████████████████▉                                                      | 392/1405 [00:00<00:01, 980.83it/s]

    
Random walks iteration 2:  35%|██████████████████████████▏                                                | 491/1405 [00:00<00:00, 982.21it/s]

    
Random walks iteration 2:  42%|███████████████████████████████▍                                           | 590/1405 [00:00<00:00, 984.36it/s]

    
Random walks iteration 2:  49%|████████████████████████████████████▌                                     | 695/1405 [00:00<00:00, 1005.56it/s]

    
Random walks iteration 2:  57%|██████████████████████████████████████████▍                                | 796/1405 [00:00<00:00, 992.63it/s]

    
Random walks iteration 2:  64%|███████████████████████████████████████████████▊                           | 896/1405 [00:00<00:00, 966.18it/s]

    
Random walks iteration 2:  71%|█████████████████████████████████████████████████████                      | 993/1405 [00:01<00:00, 963.92it/s]

    
Random walks iteration 2:  78%|█████████████████████████████████████████████████████████▍                | 1091/1405 [00:01<00:00, 966.43it/s]

    
Random walks iteration 2:  85%|██████████████████████████████████████████████████████████████▌           | 1188/1405 [00:01<00:00, 966.60it/s]

    
Random walks iteration 2:  92%|███████████████████████████████████████████████████████████████████▊      | 1288/1405 [00:01<00:00, 973.54it/s]

    
Random walks iteration 2:  99%|████████████████████████████████████████████████████████████████████████▉ | 1386/1405 [00:01<00:00, 959.81it/s]

    
Random walks iteration 2: 100%|██████████████████████████████████████████████████████████████████████████| 1405/1405 [00:01<00:00, 971.91it/s]

    


    
Random walks iteration 3:   0%|                                                                                      | 0/1405 [00:00<?, ?it/s]

    
Random walks iteration 3:   7%|█████▏                                                                      | 95/1405 [00:00<00:01, 946.17it/s]

    
Random walks iteration 3:  14%|██████████▌                                                                | 198/1405 [00:00<00:01, 993.42it/s]

    
Random walks iteration 3:  21%|███████████████▉                                                           | 298/1405 [00:00<00:01, 988.51it/s]

    
Random walks iteration 3:  28%|█████████████████████▎                                                     | 399/1405 [00:00<00:01, 992.78it/s]

    
Random walks iteration 3:  36%|██████████████████████████▋                                                | 499/1405 [00:00<00:00, 954.35it/s]

    
Random walks iteration 3:  43%|███████████████████████████████▉                                           | 599/1405 [00:00<00:00, 969.08it/s]

    
Random walks iteration 3:  50%|█████████████████████████████████████▏                                     | 697/1405 [00:00<00:00, 947.02it/s]

    
Random walks iteration 3:  56%|██████████████████████████████████████████▎                                | 792/1405 [00:00<00:00, 927.55it/s]

    
Random walks iteration 3:  64%|███████████████████████████████████████████████▋                           | 893/1405 [00:00<00:00, 949.68it/s]

    
Random walks iteration 3:  70%|████████████████████████████████████████████████████▊                      | 989/1405 [00:01<00:00, 941.90it/s]

    
Random walks iteration 3:  77%|█████████████████████████████████████████████████████████                 | 1084/1405 [00:01<00:00, 910.75it/s]

    
Random walks iteration 3:  84%|█████████████████████████████████████████████████████████████▉            | 1176/1405 [00:01<00:00, 897.82it/s]

    
Random walks iteration 3:  90%|██████████████████████████████████████████████████████████████████▉       | 1271/1405 [00:01<00:00, 912.81it/s]

    
Random walks iteration 3:  97%|███████████████████████████████████████████████████████████████████████▉  | 1366/1405 [00:01<00:00, 922.46it/s]

    
Random walks iteration 3: 100%|██████████████████████████████████████████████████████████████████████████| 1405/1405 [00:01<00:00, 936.88it/s]

    


    
Random walks iteration 4:   0%|                                                                                      | 0/1405 [00:00<?, ?it/s]

    
Random walks iteration 4:   7%|█████▏                                                                      | 97/1405 [00:00<00:01, 969.40it/s]

    
Random walks iteration 4:  14%|██████████▎                                                                | 194/1405 [00:00<00:01, 967.55it/s]

    
Random walks iteration 4:  21%|███████████████▌                                                           | 291/1405 [00:00<00:01, 934.53it/s]

    
Random walks iteration 4:  28%|█████████████████████▏                                                     | 396/1405 [00:00<00:01, 975.60it/s]

    
Random walks iteration 4:  35%|██████████████████████████▎                                                | 494/1405 [00:00<00:00, 963.33it/s]

    
Random walks iteration 4:  42%|███████████████████████████████▋                                           | 594/1405 [00:00<00:00, 975.50it/s]

    
Random walks iteration 4:  49%|████████████████████████████████████▉                                      | 692/1405 [00:00<00:00, 966.69it/s]

    
Random walks iteration 4:  56%|██████████████████████████████████████████▎                                | 792/1405 [00:00<00:00, 973.46it/s]

    
Random walks iteration 4:  63%|███████████████████████████████████████████████▌                           | 890/1405 [00:00<00:00, 959.86it/s]

    
Random walks iteration 4:  70%|████████████████████████████████████████████████████▊                      | 989/1405 [00:01<00:00, 967.23it/s]

    
Random walks iteration 4:  77%|█████████████████████████████████████████████████████████▏                | 1086/1405 [00:01<00:00, 955.65it/s]

    
Random walks iteration 4:  84%|██████████████████████████████████████████████████████████████▎           | 1182/1405 [00:01<00:00, 910.11it/s]

    
Random walks iteration 4:  91%|███████████████████████████████████████████████████████████████████▎      | 1278/1405 [00:01<00:00, 924.06it/s]

    
Random walks iteration 4:  98%|████████████████████████████████████████████████████████████████████████▏ | 1371/1405 [00:01<00:00, 832.62it/s]

    
Random walks iteration 4: 100%|██████████████████████████████████████████████████████████████████████████| 1405/1405 [00:01<00:00, 922.70it/s]

    


    
Random walks iteration 5:   0%|                                                                                      | 0/1405 [00:00<?, ?it/s]

    
Random walks iteration 5:   7%|█████▏                                                                      | 96/1405 [00:00<00:01, 956.28it/s]

    
Random walks iteration 5:  14%|██████████▏                                                                | 192/1405 [00:00<00:01, 915.95it/s]

    
Random walks iteration 5:  21%|███████████████▍                                                           | 289/1405 [00:00<00:01, 939.39it/s]

    
Random walks iteration 5:  28%|████████████████████▊                                                      | 389/1405 [00:00<00:01, 960.28it/s]

    
Random walks iteration 5:  35%|██████████████████████████                                                 | 489/1405 [00:00<00:00, 974.25it/s]

    
Random walks iteration 5:  42%|███████████████████████████████▎                                           | 587/1405 [00:00<00:00, 971.52it/s]

    
Random walks iteration 5:  49%|████████████████████████████████████▌                                      | 685/1405 [00:00<00:00, 957.60it/s]

    
Random walks iteration 5:  56%|█████████████████████████████████████████▋                                 | 781/1405 [00:00<00:00, 956.02it/s]

    
Random walks iteration 5:  62%|██████████████████████████████████████████████▊                            | 878/1405 [00:00<00:00, 959.40it/s]

    
Random walks iteration 5:  70%|████████████████████████████████████████████████████▏                      | 978/1405 [00:01<00:00, 971.70it/s]

    
Random walks iteration 5:  77%|████████████████████████████████████████████████████████▋                 | 1076/1405 [00:01<00:00, 961.12it/s]

    
Random walks iteration 5:  84%|█████████████████████████████████████████████████████████████▉            | 1177/1405 [00:01<00:00, 974.51it/s]

    
Random walks iteration 5:  91%|███████████████████████████████████████████████████████████████████▏      | 1275/1405 [00:01<00:00, 965.41it/s]

    
Random walks iteration 5:  98%|████████████████████████████████████████████████████████████████████████▎ | 1372/1405 [00:01<00:00, 958.60it/s]

    
Random walks iteration 5: 100%|██████████████████████████████████████████████████████████████████████████| 1405/1405 [00:01<00:00, 959.98it/s]

<div class="k-default-codeblock">
```
Number of walks generated: 7025
```
</div>

---
## Generate positive and negative examples

To train a skip-gram model, we use the generated walks to create positive and
negative training examples. In Keras 3, the legacy preprocessing module
has been removed. We now implement a manual skip-gram sampling function
using NumPy to generate positive and negative training examples from our
random walks. Each example includes the following features:

1. `target`: A movie in a walk sequence.
2. `context`: Another movie in a walk sequence.
3. `weight`: How many times these two movies occurred  in walk sequences.
4. `label`: The label is 1 if these two movies are samples from the walk sequences,
otherwise (i.e., if randomly sampled) the label is 0.

### Generate examples


```python

def manual_skipgrams(sequence, vocabulary_size, window_size=5, negative_samples=4):
    """
    A NumPy-based replacement for the legacy keras.preprocessing.sequence.skipgrams.
    Generates (target, context) pairs with positive and negative labels,
    ensuring negative samples are not in the positive context window.
    """
    pairs = []
    labels = []

    for i, target in enumerate(sequence):
        start = max(0, i - window_size)
        end = min(len(sequence), i + window_size + 1)
        positive_contexts = {sequence[j] for j in range(start, end) if i != j}

        for j in range(start, end):
            if i == j:
                continue
            context = sequence[j]

            pairs.append([target, context])
            labels.append(1)

            for _ in range(negative_samples):
                negative_context = np.random.randint(0, vocabulary_size)

                while (
                    negative_context == target or negative_context in positive_contexts
                ):
                    negative_context = np.random.randint(0, vocabulary_size)

                pairs.append([target, negative_context])
                labels.append(0)

    return pairs, labels


def generate_examples(sequences, window_size, num_negative_samples, vocabulary_size):
    example_weights = defaultdict(int)

    # Iterate over all walks
    for sequence in tqdm(sequences, desc="Generating positive and negative examples"):
        # Use our manual skipgrams function
        pairs, labels = manual_skipgrams(
            sequence,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            negative_samples=num_negative_samples,
        )

        for idx in range(len(pairs)):
            pair = pairs[idx]
            label = labels[idx]
            target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
            if target == context:
                continue
            entry = (target, context, label)
            example_weights[entry] += 1

    targets, contexts, labels, weights = [], [], [], []
    for entry, weight in example_weights.items():
        target, context, label = entry
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        weights.append(weight)

    return (
        np.array(targets, dtype="int32"),
        np.array(contexts, dtype="int32"),
        np.array(labels, dtype="float32"),
        np.array(weights, dtype="float32"),
    )


# Execute the generation
num_negative_samples = 4
targets, contexts, labels, weights = generate_examples(
    sequences=walks,
    window_size=num_steps,
    num_negative_samples=num_negative_samples,
    vocabulary_size=len(vocabulary),
)
```

    
Generating positive and negative examples:   0%|                                                                     | 0/7025 [00:00<?, ?it/s]

    
Generating positive and negative examples:   2%|█▏                                                       | 149/7025 [00:00<00:04, 1480.47it/s]

    
Generating positive and negative examples:   4%|██▍                                                      | 298/7025 [00:00<00:04, 1477.76it/s]

    
Generating positive and negative examples:   6%|███▌                                                     | 446/7025 [00:00<00:04, 1471.28it/s]

    
Generating positive and negative examples:   8%|████▊                                                    | 594/7025 [00:00<00:04, 1444.59it/s]

    
Generating positive and negative examples:  11%|█████▉                                                   | 739/7025 [00:00<00:04, 1310.46it/s]

    
Generating positive and negative examples:  12%|███████                                                  | 872/7025 [00:00<00:04, 1300.17it/s]

    
Generating positive and negative examples:  14%|████████                                                | 1004/7025 [00:00<00:05, 1060.46it/s]

    
Generating positive and negative examples:  16%|█████████                                               | 1143/7025 [00:00<00:05, 1145.10it/s]

    
Generating positive and negative examples:  18%|██████████▏                                             | 1276/7025 [00:01<00:04, 1194.00it/s]

    
Generating positive and negative examples:  20%|███████████▏                                            | 1411/7025 [00:01<00:04, 1236.20it/s]

    
Generating positive and negative examples:  22%|████████████▎                                           | 1546/7025 [00:01<00:04, 1267.97it/s]

    
Generating positive and negative examples:  24%|█████████████▍                                          | 1681/7025 [00:01<00:04, 1290.49it/s]

    
Generating positive and negative examples:  26%|██████████████▍                                         | 1816/7025 [00:01<00:03, 1305.88it/s]

    
Generating positive and negative examples:  28%|███████████████▌                                        | 1949/7025 [00:01<00:03, 1310.88it/s]

    
Generating positive and negative examples:  30%|████████████████▌                                       | 2083/7025 [00:01<00:03, 1316.91it/s]

    
Generating positive and negative examples:  32%|█████████████████▋                                      | 2217/7025 [00:01<00:03, 1321.22it/s]

    
Generating positive and negative examples:  33%|██████████████████▋                                     | 2350/7025 [00:01<00:03, 1322.42it/s]

    
Generating positive and negative examples:  35%|███████████████████▊                                    | 2484/7025 [00:01<00:03, 1325.62it/s]

    
Generating positive and negative examples:  37%|████████████████████▊                                   | 2617/7025 [00:02<00:03, 1322.78it/s]

    
Generating positive and negative examples:  39%|█████████████████████▉                                  | 2750/7025 [00:02<00:03, 1318.66it/s]

    
Generating positive and negative examples:  41%|██████████████████████▉                                 | 2883/7025 [00:02<00:03, 1313.22it/s]

    
Generating positive and negative examples:  43%|████████████████████████                                | 3015/7025 [00:02<00:03, 1314.43it/s]

    
Generating positive and negative examples:  45%|█████████████████████████                               | 3147/7025 [00:02<00:02, 1308.94it/s]

    
Generating positive and negative examples:  47%|██████████████████████████▏                             | 3278/7025 [00:02<00:02, 1304.27it/s]

    
Generating positive and negative examples:  49%|███████████████████████████▏                            | 3410/7025 [00:02<00:02, 1307.06it/s]

    
Generating positive and negative examples:  50%|████████████████████████████▏                           | 3541/7025 [00:02<00:02, 1306.07it/s]

    
Generating positive and negative examples:  52%|█████████████████████████████▎                          | 3672/7025 [00:02<00:02, 1305.56it/s]

    
Generating positive and negative examples:  54%|██████████████████████████████▎                         | 3803/7025 [00:02<00:02, 1305.49it/s]

    
Generating positive and negative examples:  56%|███████████████████████████████▎                        | 3934/7025 [00:03<00:02, 1305.95it/s]

    
Generating positive and negative examples:  58%|████████████████████████████████▍                       | 4065/7025 [00:03<00:02, 1240.41it/s]

    
Generating positive and negative examples:  60%|█████████████████████████████████▍                      | 4192/7025 [00:03<00:02, 1247.10it/s]

    
Generating positive and negative examples:  61%|██████████████████████████████████▍                     | 4319/7025 [00:03<00:02, 1252.92it/s]

    
Generating positive and negative examples:  63%|███████████████████████████████████▍                    | 4446/7025 [00:03<00:02, 1257.71it/s]

    
Generating positive and negative examples:  65%|████████████████████████████████████▍                   | 4573/7025 [00:03<00:01, 1257.47it/s]

    
Generating positive and negative examples:  67%|█████████████████████████████████████▍                  | 4699/7025 [00:03<00:01, 1249.81it/s]

    
Generating positive and negative examples:  69%|██████████████████████████████████████▍                 | 4825/7025 [00:03<00:01, 1247.82it/s]

    
Generating positive and negative examples:  70%|███████████████████████████████████████▍                | 4952/7025 [00:03<00:01, 1252.94it/s]

    
Generating positive and negative examples:  72%|████████████████████████████████████████▍               | 5080/7025 [00:03<00:01, 1258.48it/s]

    
Generating positive and negative examples:  74%|█████████████████████████████████████████▍              | 5206/7025 [00:04<00:01, 1256.32it/s]

    
Generating positive and negative examples:  76%|██████████████████████████████████████████▌             | 5333/7025 [00:04<00:01, 1258.27it/s]

    
Generating positive and negative examples:  78%|███████████████████████████████████████████▌            | 5461/7025 [00:04<00:01, 1263.60it/s]

    
Generating positive and negative examples:  80%|████████████████████████████████████████████▌           | 5589/7025 [00:04<00:01, 1268.09it/s]

    
Generating positive and negative examples:  81%|█████████████████████████████████████████████▌          | 5716/7025 [00:04<00:01, 1266.32it/s]

    
Generating positive and negative examples:  83%|██████████████████████████████████████████████▌         | 5843/7025 [00:04<00:00, 1264.77it/s]

    
Generating positive and negative examples:  85%|███████████████████████████████████████████████▌        | 5970/7025 [00:04<00:00, 1263.65it/s]

    
Generating positive and negative examples:  87%|████████████████████████████████████████████████▌       | 6097/7025 [00:04<00:00, 1262.36it/s]

    
Generating positive and negative examples:  89%|█████████████████████████████████████████████████▌      | 6224/7025 [00:04<00:00, 1031.00it/s]

    
Generating positive and negative examples:  90%|██████████████████████████████████████████████████▋     | 6351/7025 [00:05<00:00, 1091.13it/s]

    
Generating positive and negative examples:  92%|███████████████████████████████████████████████████▋    | 6479/7025 [00:05<00:00, 1140.10it/s]

    
Generating positive and negative examples:  94%|████████████████████████████████████████████████████▋   | 6603/7025 [00:05<00:00, 1166.60it/s]

    
Generating positive and negative examples:  96%|█████████████████████████████████████████████████████▋  | 6729/7025 [00:05<00:00, 1192.49it/s]

    
Generating positive and negative examples:  98%|██████████████████████████████████████████████████████▋ | 6853/7025 [00:05<00:00, 1204.79it/s]

    
Generating positive and negative examples:  99%|███████████████████████████████████████████████████████▌| 6976/7025 [00:05<00:00, 1206.19it/s]

    
Generating positive and negative examples: 100%|████████████████████████████████████████████████████████| 7025/7025 [00:05<00:00, 1257.73it/s]

    


Let's display the shapes of the outputs


```python
print(f"Targets shape: {targets.shape}")
print(f"Contexts shape: {contexts.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Weights shape: {weights.shape}")
```

<div class="k-default-codeblock">
```
Targets shape: (883654,)
Contexts shape: (883654,)
Labels shape: (883654,)
Weights shape: (883654,)
```
</div>

### Data Loading with PyDataset

We replace the tf.data pipeline with keras.utils.PyDataset.
This ensures our data pipeline is fully backend-agnostic and
avoids symbolic tensor errors when running on JAX or PyTorch.


```python
batch_size = 1024


class MovieLensDataset(keras.utils.PyDataset):
    def __init__(self, targets, contexts, labels, weights, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.targets = targets
        self.contexts = contexts
        self.labels = labels
        self.weights = weights
        self.batch_size = batch_size

    def __len__(self):
        return len(self.targets) // self.batch_size

    def __getitem__(self, index):
        low = index * self.batch_size
        high = (index + 1) * self.batch_size

        target = self.targets[low:high]
        context = self.contexts[low:high]
        label = self.labels[low:high]
        weight = self.weights[low:high]

        return {"target": target, "context": context}, label, weight


batch_size = 1024
dataset = MovieLensDataset(targets, contexts, labels, weights, batch_size)
```

---
## Train the skip-gram model

Our skip-gram is a simple binary classification model that works as follows:

1. An embedding is looked up for the `target` movie.
2. An embedding is looked up for the `context` movie.
3. The dot product is computed between these two embeddings.
4. The result (after a sigmoid activation) is compared to the label.
5. A binary crossentropy loss is used.


```python
learning_rate = 0.001
embedding_dim = 50
num_epochs = 10
```

### Implement the model


```python

def create_model(vocabulary_size, embedding_dim):
    target_in = layers.Input(name="target", shape=(), dtype="int32")
    context_in = layers.Input(name="context", shape=(), dtype="int32")

    embed_item = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="item_embeddings",
    )
    target_embed = embed_item(target_in)
    context_embed = embed_item(context_in)

    dot_similarity = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
        [target_embed, context_embed]
    )

    output = layers.Reshape((1,))(dot_similarity)

    return keras.Model(inputs=[target_in, context_in], outputs=output)

```

### Train the model

We instantiate the model and compile it.


```python
model = create_model(len(vocabulary), embedding_dim)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)
```

Let's plot the model.


```python
keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
)
```

![png](/img/examples/graph/node2vec_movielens/node2vec_movielens_44_0.png)

Now we train the model on the `dataset`.


```python
history = model.fit(dataset, epochs=num_epochs)
```

<div class="k-default-codeblock">
```
Epoch 1/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.4523

Epoch 2/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3459

Epoch 3/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3349

Epoch 4/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3312

Epoch 5/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3273

Epoch 6/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3236

Epoch 7/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3201

Epoch 8/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3177

Epoch 9/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3149

Epoch 10/10

862/862 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 2.3127
```
</div>

Finally we plot the learning history.


```python
plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

```


    
![png](/img/examples/graph/node2vec_movielens/node2vec_movielens_48_0.png)
    


---
## Analyze the learnt embeddings.


```python
movie_embeddings = model.get_layer("item_embeddings").get_weights()[0]
print("Embeddings shape:", movie_embeddings.shape)
```

<div class="k-default-codeblock">
```
Embeddings shape: (1406, 50)
```
</div>

### Find related movies

Define a list with some movies called `query_movies`.


```python
query_movies = [
    "Matrix, The (1999)",
    "Star Wars: Episode IV - A New Hope (1977)",
    "Lion King, The (1994)",
    "Terminator 2: Judgment Day (1991)",
    "Godfather, The (1972)",
]
```

Get the embeddings of the movies in `query_movies`.


```python
query_tokens = []
for title in query_movies:
    movieId = get_movie_id_by_title(title)
    query_tokens.append(vocabulary_lookup[movieId])

query_tokens = np.array(query_tokens, dtype="int32")
```

Compute the [consine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between the embeddings of `query_movies`
and all the other movies, then pick the top k for each.


```python

def compute_similarities(query_indices, all_embeddings):
    # Lookup embeddings
    query_embeds = ops.take(all_embeddings, query_indices, axis=0)

    # L2 Normalize using Keras Ops
    def l2_norm(x):
        # Ensure x is a Keras Tensor before operations with keras.ops
        x_tensor = ops.convert_to_tensor(x)
        return x_tensor / ops.sqrt(
            ops.maximum(ops.sum(ops.square(x_tensor), axis=-1, keepdims=True), 1e-12)
        )

    query_embeds = l2_norm(query_embeds)
    all_embeddings = l2_norm(all_embeddings)

    # Cosine Similarity
    similarities = ops.matmul(query_embeds, ops.transpose(all_embeddings))

    # Get Top K
    vals, inds = ops.top_k(similarities, k=5)
    return inds


# Convert movie_embeddings to a Keras Tensor before calling compute_similarities
movie_embeddings_tensor = ops.convert_to_tensor(movie_embeddings)
indices = compute_similarities(query_tokens, movie_embeddings_tensor)
indices = keras.ops.convert_to_numpy(indices).tolist()
```

Display the top related movies in `query_movies`.


```python
for idx, title in enumerate(query_movies):
    print(f"{title}\n{'-' * len(title)}")
    for token in indices[idx]:
        print(f"- {get_movie_title_by_id(vocabulary[token])}")
    print()
```

<div class="k-default-codeblock">
```
Matrix, The (1999)
------------------
- Matrix, The (1999)
- Pulp Fiction (1994)
- Saving Private Ryan (1998)
- Star Wars: Episode V - The Empire Strikes Back (1980)
- Full Metal Jacket (1987)

Star Wars: Episode IV - A New Hope (1977)
-----------------------------------------
- Star Wars: Episode IV - A New Hope (1977)
- Princess Bride, The (1987)
- Star Wars: Episode V - The Empire Strikes Back (1980)
- Monty Python and the Holy Grail (1975)
- Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)

Lion King, The (1994)
---------------------
- Lion King, The (1994)
- Beauty and the Beast (1991)
- Speed (1994)
- Die Hard: With a Vengeance (1995)
- Mrs. Doubtfire (1993)

Terminator 2: Judgment Day (1991)
---------------------------------
- Terminator 2: Judgment Day (1991)
- Braveheart (1995)
- Forrest Gump (1994)
- Star Wars: Episode V - The Empire Strikes Back (1980)
- Shawshank Redemption, The (1994)

Godfather, The (1972)
---------------------
- Godfather, The (1972)
- Godfather: Part II, The (1974)
- American Beauty (1999)
- Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
- Monty Python and the Holy Grail (1975)
```
</div>

### Visualize the embeddings using the Embedding Projector


```python
import io

# Ensure embeddings are converted to a standard format regardless of the backend
# This is the "Keras 3 way" to access trained weights for post-processing
embeddings_np = ops.convert_to_numpy(movie_embeddings)

out_v = io.open("embeddings.tsv", "w", encoding="utf-8")
out_m = io.open("metadata.tsv", "w", encoding="utf-8")

for idx, movie_id in enumerate(vocabulary[1:]):
    # The movie_id at vocabulary[1:] corresponds to weights at index 1 and onwards
    vector = embeddings_np[idx + 1]

    # Standard Pandas/Python logic for metadata
    movie_title = movies[movies.movieId == movie_id]["title"].values[0]

    # Write tab-separated values for the projector
    out_v.write("\t".join([str(x) for x in vector]) + "\n")
    out_m.write(movie_title + "\n")

out_v.close()
out_m.close()

print("Embeddings and metadata saved for projector.")
```

<div class="k-default-codeblock">
```
Embeddings and metadata saved for projector.
```
</div>

Download the `embeddings.tsv` and `metadata.tsv` to analyze the obtained embeddings
in the [Embedding Projector](https://projector.tensorflow.org/).

**Example available on HuggingFace**

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model%3A%20-Node2Vec%20Movielens-black.svg)](https://huggingface.co/keras-io/Node2Vec_MovieLens) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces%3A-Node2Vec%20Movielens-black.svg)](https://huggingface.co/spaces/keras-io/Node2Vec_MovieLens) |

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 14: Text classification](https://deeplearningwithpython.io/chapters/chapter14_text-classification)
