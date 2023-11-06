# KerasTuner HyperModels

The `HyperModel` base class makes the search space better encapsulated for
sharing and reuse.  A `HyperModel` subclass only needs to implement a
`build(self, hp)` method, which creates a `keras.Model` using the `hp` argument
to define the hyperparameters and returns the model instance.
A simple code example is shown as follows.

```python
class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse')
    return model
```

You can pass a `HyperModel` instance to the `Tuner` as the search space.

```python
tuner = kt.RandomSearch(
    MyHyperModel(),
    objective='val_loss',
    max_trials=5)
```

There are also some built-in `HyperModel` subclasses (e.g. `HyperResNet`,
`HyperXception`) for the users to directly use so that the users don't need to
write their own search spaces.

```python
tuner = kt.RandomSearch(
    HyperResNet(input_shape=(28, 28, 1), classes=10),
    objective='val_loss',
    max_trials=5)
```


{{toc}}
