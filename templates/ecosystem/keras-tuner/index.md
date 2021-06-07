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

You can also checkout other versions in our
[GitHub repository](https://github.com/keras-team/keras-tuner)

## Getting started

To quickly learn the basics of Keras Tuner,
please refer to the
[getting started](/ecosystem/keras-tuner/getting_started) notebook.

## Developer guides
For deep-dives into specific topics of Keras Tuner such as customize the tuner
or distributed tuning, please refer to our [developer
guides](/guides/keras-tuner/).

## API reference

The documentation of all the public APIs of Keras Tuner is in the [API
reference](/api/keras-tuner/) page.

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
