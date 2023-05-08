
"""
Next, lets assemble a `keras_cv` augmentation pipeline.
In this guide, we use the standard pipeline
[CutMix, MixUp, and RandAugment](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
augmentation pipeline.  More information on the behavior of these augmentations
may be found in their
[corresponding Keras.io guide](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/).
"""

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
        keras_cv.layers.MixUp(),
    ]
)

train_dataset = train_dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

images = next(iter(train_dataset.take(1)))["images"]
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))

model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(2, activation="softmax"),
    ]
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

"""
All that is left to do is construct a standard Keras `model.fit()` loop!
"""


def unpackage_data(inputs):
    return inputs["images"], inputs["labels"]


train_dataset.map(unpackage_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

model.fit(train_dataset.map(unpackage_data, num_parallel_calls=tf.data.AUTOTUNE))

"""
Let's look at how our model performs after the fine tuning!
"""

predictions = model.predict(np.expand_dims(image, axis=0))

classes = {0: "cat", 1: "dog"}
print("Top class is:", classes[predictions[0].argmax()])

"""
Awesome!  Looks like the model correctly classified the image.
"""

"""
# Train a Classifier from Scratch

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_advanced.png)

Now that we've gotten our hands dirty with classification, let's take on one last task:
training a classification model from scratch!
A standard benchmark for image classification is the ImageNet dataset, however
due to licensing constraints we will use the CalTech 101 image classification
dataset in this tutorial.
While we use the simpler CalTech 101 dataset in this guide, the same training
template may be used on ImageNet to achieve state of the art scores.

Finally, let's train ...

- https://www.tensorflow.org/datasets/catalog/places365_small
- https://www.tensorflow.org/datasets/catalog/caltech101
"""

"""
## Conclusions

KerasCV makes image classification easy.
Making use of the KerasCV `ImageClassifier` API, pretrained weights, and the
KerasCV data augmentations allows you to train a powerful classifier in `<50`
lines of code.

As a follow up exercise, give the following a try:

- Fine tune a KerasCV classifier on your own dataset
- Learn more about [KerasCV's data augmentations](https://keras.io/guides/keras_cv/cut_mix_mix_up_and_rand_augment/)
- Check out how we train our models on [ImageNet](https://github.com/keras-team/keras-cv/blob/master/examples/training/classification/imagenet/basic_training.py)
"""
