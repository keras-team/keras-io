# Keras Tuner

Keras Tuner is an easy-to-use, scalable hyperparameter optimization framework
that solves the pain points of hyperparameter search. Easily configure your
search space with a define-by-run syntax, then leverage one of the available
search algorithms to find the best hyperparameter values for your models. Keras
Tuner comes with Bayesian Optimization, Hyperband, and Random Search algorithms
built-in, and is also designed to be easy for researchers to extend in order to
experiment with new search algorithms.

## Installation

Keras Tuner requires **Python 3.6+** and **TensorFlow 2.0+**.

Install latest release:

```
pip install -U keras-tuner
```

You can also check out other versions in our
[GitHub repository](https://github.com/keras-team/keras-tuner).

## Usage

Here is a simple example of how to use Keras Tuner.
Import Keras Tuner and TensorFlow.
```python
import kerastuner as kt
from tensorflow import keras
```

Write a function that creates and returns a Keras model.
Use the `hp` argument to define the hyperparameters during model creation.

```python
def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu'))
  model.add(keras.layers.Dense(1, activation='relu'))
  model.compile(loss='mse')
  return model
```

Initialize the tuning algorithm, `RandomSearch`,
We use `objective` to specify the objective to select the best models,
and use `max_trials` to specify the number of different models to try. 
```python
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)
```

Start the search and get the best model.
```python
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

## Further resources
* [Getting started notebook](/keras-tuner/getting_started).
A Python notebook to quickly learn the basics of Keras Tuner.

* [Developer guides](/guides/keras-tuner/).
Deep-dives into specific topics of Keras Tuner such as customize the tuner
or distributed tuning.

* [API reference](/api/keras-tuner/).
The documentation of all the public APIs of Keras Tuner.

## Citing Keras Tuner

We appreciate your citations if it helps your research.
Here is the BibTeX entry:

```bibtex
@misc{omalley2019kerastuner,
	title        = {Keras {Tuner}},
	author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin, Haifeng and Invernizzi, Luca and others},
	year         = 2019,
	howpublished = {\url{https://github.com/keras-team/keras-tuner}}
}
```
