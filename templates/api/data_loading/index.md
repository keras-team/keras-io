# Data loading

Keras data loading utilities, located in `tf.keras.utils`,
help you go from raw data on disk to a `tf.data.Dataset` object that can be
used to efficiently train a model.

These loading utilites can be combined with
[preprocessing layers](https://keras.io/guides/preprocessing_layers/) to
futher transform your input dataset before training.

Here's a quick example: let's say you have 10 folders, each containing
10,000 images from a different category, and you want to train a
classifier that maps an image to its category.

Your training data folder would look like this:

```
training_data/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
etc.
```

You may also have a validation data folder `validation_data/` structured in the
same way.

You could simply do:

```python
from tensorflow import keras

train_ds = keras.utils.image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
validation_ds = keras.utils.image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))

model = keras.applications.Xception(
    weights=None, input_shape=(256, 256, 3), classes=10)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(train_ds, epochs=10, validation_data=validation_ds)
```


## Available dataset loading utilities

{{toc}}

