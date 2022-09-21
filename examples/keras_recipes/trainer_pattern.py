"""
Title: Trainer pattern
Author: [nkovela1](https://nkovela1.github.io/)
Date created: 2022/09/19
Last modified: 2022/09/19
Description: Demonstration of sharing train_step across multiple Keras Models.
"""

"""
## Setup
"""

import tensorflow as tf

# Load MNIST dataset and normalize
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


"""
## Sharing custom training step across multiple models

A custom training step can be created by overriding the train_step method of a Model subclass.
""" 
class MyTrainer(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, None)

"""
## Define multiple models to share the custom training step

Let's define two different models that can share our Trainer class and its custom train_step
"""

# Defined using Sequential API
MyModel = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Defined using Functional API
func_input = tf.keras.Input(shape=(28,28,1))
x = tf.keras.layers.Flatten(input_shape=(28,28))(func_input)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
func_output = tf.keras.layers.Dense(10, activation='softmax')(x)

YourModel = tf.keras.Model(func_input, func_output)

"""
## Create Trainer class objects from the models
"""
trainer_1 = MyTrainer(MyModel)
trainer_2 = MyTrainer(YourModel)

"""
## Compile and fit the models to the MNIST dataset
"""
trainer_1.compile(tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
trainer_1.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

trainer_2.compile(tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
trainer_2.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))


