# Callbacks API

A callback is an object that can perform actions at various stages of training
(e.g. at the start or end of an epoch, before or after a single batch, etc).

You can use callbacks to:

- Write TensorBoard logs after every batch of training to monitor your metrics
- Periodically save your model to disk
- Do early stopping
- Get a view on internal states and statistics of a model during training
- ...and more

---

## Usage of callbacks via the built-in `fit()` loop

You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of a model:

```python
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
```

The relevant methods of the callbacks will then be called at each stage of the training.

---

## Using custom callbacks

Creating new callbacks is a simple and powerful way to customize a training loop.
Learn more about creating new callbacks in the guide
[Writing your own Callbacks](/guides/writing_your_own_callbacks), and refer to
the documentation for [the base `Callback` class](base_callback).

---

## Available callbacks

{{toc}}
