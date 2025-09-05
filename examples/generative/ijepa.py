"""
Title: I-JEPA: One latent encoder, multiple downstream tasks.
Author: [Damoon Shahhosseini](https://www.linkedin.com/in/damoonsh/)
Date created: 2025/07/08
Last modified: 2025/07/08
Description: JEPA is getting lots of traction in the self-supervised learning community. This tutorial implements I-JEPA, a variant of JEPA that uses a context encoder to predict masked patches, and applies it to ImageNet for pretraining and Clevr for object counting.
Accelerator: GPU

## Introduction
Joint-Embedding Predictive Architectures (JEPA) is a self-supervised learning framework that learns robust representations by predicting masked patches of data. 
The trained encoder within JEPA can be used for various downstream tasks, such as image classification and object counting.
This tutorial implements I-JEPA, a variant of JEPA that uses a context encoder to predict masked patches, and applies it to ImageNet for pretraining and Clevr for object counting.

Here, we will:
1. Load and preprocess ImageNet and Clevr datasets.
2. Implement the I-JEPA model with a context encoder.
3. Train the context encoder using InfoNCE loss.
4. Create separate models for linear classification on ImageNet and object counting on Clevr.
5. Train the models on their respective tasks using the frozen pretrained context encoder

References:
- [I-JEPA: Self-Supervised Learning with Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2301.08243)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)

## Imports
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import tensorflow_datasets as tfds

"""
## Constants
"""
BATCH_SIZE = 128
PRETRAIN_EPOCHS = 1
FINETUNE_EPOCHS = 1
LEARNING_RATE = 0.001
MASK_RATIO = 0.75

"""
## Data Preparation
Load ImageNet for pretraining and linear classification, Clevr for object counting.
"""
input_shape = (224, 224, 3)
patch_size = 16
num_patches = (224 // patch_size) ** 2
embed_dim = 384
num_heads = 6
mlp_dim = 1152
num_blocks = 12
num_classes = 1000  # ImageNet classes
max_objects = 10    # Clevr max object count

# ImageNet
(ds_train, ds_test), ds_info = tfds.load(
    'imagenet2012',
    split=['train', 'validation'],
    as_supervised=True,
    with_info=True
)
train_loader = DataLoader(ds_train, batch_size=128, patch_size=patch_size, input_shape=input_shape)
test_loader = DataLoader(ds_test, batch_size=128, patch_size=patch_size, input_shape=input_shape)
ds_train = train_loader.get_dataset()
ds_test = test_loader.get_dataset()

# Clevr
clevr_ds = tfds.load('clevr', split='train', as_supervised=False)
clevr_loader = DataLoader(clevr_ds, batch_size=128, patch_size=patch_size, input_shape=input_shape, is_clevr=True)
clevr_ds = clevr_loader.get_dataset()

"""
## DataLoader Class

The I-JEPA DataLoader class, divides the input images into 16x16 patches and prepares them for training. These patches fit into the Vision Transformer architecture used in I-JEPA.
"""
class DataLoader:
    def __init__(self, dataset, batch_size, patch_size, input_shape, is_clevr=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_shape = input_shape
        self.num_patches = (input_shape[0] // patch_size) ** 2
        self.is_clevr = is_clevr
        self.patch_embed = layers.Conv2D(384, patch_size, strides=patch_size, padding="valid")

    def preprocess_image(self, image, label):
        image = tf.image.resize(image, [self.input_shape[0], self.input_shape[1]])
        image = tf.cast(image, tf.float32) / 255.0
        if self.is_clevr:
            label = tf.cast(label['objects/count'], tf.float32)
        return image, label

    def create_patches(self, image):
        patches = self.patch_embed(image)
        patches = tf.reshape(patches, [-1, self.num_patches, 384])
        pos_embed = tf.zeros([1, self.num_patches, 384])
        patches += pos_embed
        return patches

    def get_dataset(self):
        dataset = self.dataset.map(self.preprocess_image).map(
            lambda x, y: (self.create_patches(x), y)
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

"""
## I-JEPA Components
### Vision Transformer Block
Standard ViT block with multi-head self-attention and MLP.
"""
def vit_block(x, embed_dim, num_heads, mlp_dim):
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
    x = layers.Add()([shortcut, x])
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(mlp_dim, activation="gelu")(x)
    x = layers.Dense(embed_dim)(x)
    x = layers.Add()([shortcut, x])
    return x

"""
### I-JEPA Model
Each iteration, masks a portion of the input patches, encodes the context patches, and predicts the masked target patches. The encoder aim to learn abstract representations of the input data.
"""
class IJEPA(keras.Model):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_blocks):
        super(IJEPA, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Context and Target Encoders
        self.context_encoder = keras.Sequential([vit_block(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim) for _ in range(num_blocks)])
        self.target_encoder = keras.Sequential([vit_block(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim) for _ in range(num_blocks)])
        
        # Predictor
        self.predictor = keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])

    def mask_patches(self, patches, mask_ratio):
        batch_size = tf.shape(patches)[0]
        num_masked = int(self.num_patches * mask_ratio)
        indices = tf.argsort(tf.random.uniform([batch_size, self.num_patches]), axis=-1)
        mask_indices = indices[:, :num_masked]
        context_indices = indices[:, num_masked:]
        return mask_indices, context_indices

    def call(self, x, training=False):
        mask_indices, context_indices = self.mask_patches(x, MASK_RATIO)
        
        # Gather context and target patches
        batch_size = tf.shape(x)[0]
        context_patches = tf.gather(x, context_indices, batch_dims=1)
        target_patches = tf.gather(x, mask_indices, batch_dims=1)
        
        # Encode
        context_repr = self.context_encoder(context_patches)
        target_repr = self.target_encoder(target_patches)
        
        # Predict target representations
        predicted_repr = self.predictor(context_repr)
        
        return predicted_repr, target_repr

    def compute_loss(self, predicted_repr, target_repr):
        predicted_repr = tf.nn.l2_normalize(predicted_repr, axis=-1)
        target_repr = tf.nn.l2_normalize(target_repr, axis=-1)
        loss = -tf.reduce_mean(tf.reduce_sum(predicted_repr * target_repr, axis=-1))
        return loss

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            predicted_repr, target_repr = self(x, training=True)
            loss = self.compute_loss(predicted_repr, target_repr)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, _ = data
        predicted_repr, target_repr = self(x, training=False)
        loss = self.compute_loss(predicted_repr, target_repr)
        return {"loss": loss}

"""
## Downstream Task Models
### Linear Classification Model (ImageNet)
Uses pretrained context encoder for linear classification.
"""
class LinearClassifier(keras.Model):
    def __init__(self, context_encoder, num_classes):
        super(LinearClassifier, self).__init__()
        self.context_encoder = context_encoder
        self.context_encoder.trainable = False 
        self.classifier = keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(num_classes, activation="softmax")
        ])

    def call(self, x, training=False):
        features = self.context_encoder(x)
        return self.classifier(features)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

"""
## Pretraining I-JEPA
Train the context encoder on ImageNet with InfoNCE loss.
"""
ijepa_model = IJEPA(embed_dim, num_heads, mlp_dim, num_blocks)
ijepa_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
ijepa_model.fit(
    ds_train,
    epochs=PRETRAIN_EPOCHS,
    validation_data=ds_test,
    verbose=0
)

"""
### Object Counting Model (Clevr)
Uses pretrained context encoder for object counting.
"""
class ObjectCounter(keras.Model):
    def __init__(self, context_encoder, max_objects):
        super(ObjectCounter, self).__init__()
        self.context_encoder = context_encoder
        self.context_encoder.trainable = False 
        self.counter = keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(max_objects + 1, activation="softmax")
        ])

    def call(self, x, training=False):
        features = self.context_encoder(x)
        return self.counter(features)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}



"""
## Downstream Task: Linear Classification
Train a linear classifier on top of the pretrained context encoder for ImageNet classification.
"""
classifier = LinearClassifier(ijepa_model.context_encoder, num_classes)
classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
classifier.fit(
    ds_train,
    epochs=FINETUNE_EPOCHS,
    validation_data=ds_test,
    verbose=0
)
score = classifier.evaluate(ds_test, verbose=0)
print("Linear Classification Test loss:", score[0])
print("Linear Classification Test accuracy:", score[1])

"""
## Downstream Task: Object Counting
Train an object counter on top of the pretrained context encoder for Clevr.
"""
counter = ObjectCounter(ijepa_model.context_encoder, max_objects)
counter.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
counter.fit(
    clevr_ds,
    epochs=FINETUNE_EPOCHS,
    verbose=0
)
score = counter.evaluate(clevr_ds, verbose=0)
print("Object Counting Test loss:", score[0])
print("Object Counting Test accuracy:", score[1])

"""
## Conclusion
This implementation pretrains I-JEPA on ImageNet and reuses the context encoder for downstream tasks (linear classification on ImageNet, object counting on Clevr), as in [Assran et al., 2023](https://arxiv.org/abs/2301.08243). Key aspects:
1. **Pretraining**: InfoNCE loss on ImageNet for robust representations.
2. **Downstream Tasks**: Linear classification and object counting using the pretrained encoder.
3. **DataLoader**: Efficiently handles 16x16 patch creation.

The pretrained encoder enables effective transfer learning for both tasks.
"""