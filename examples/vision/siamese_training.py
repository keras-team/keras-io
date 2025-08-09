"""
Title: Image similarity estimation using a Siamese Network with triplet loss
Author: Amin Nasiri
Date created: 2025/08/25
Last modified: 2025/09/08
Description: Implementation of Siamese Networks with triplet loss for image similarity learning,
featuring distributed training and comprehensive evaluation.
Accelerator: GPU
"""

"""
## Introduction

This example demonstrates how to train a Siamese Network using triplet loss for image similarity learning.
Siamese networks are particularly useful for tasks where you need to learn similarity between images,
such as face recognition, person re-identification, or product matching.

### Key Features:
- Custom triplet loss implementation
- Efficient tf.data pipeline for triplet generation
- Distributed training across multiple GPUs
- Progressive training (frozen backbone → fine-tuning)
- Comprehensive evaluation with confusion matrices

### What you'll learn:
- How to implement triplet loss for similarity learning
- Creating efficient data pipelines for triplet training
- Using transfer learning with ResNet50 for embeddings
- Distributed training strategies
- Evaluating similarity models

### References:
- Image similarity (Keras example):
  https://keras.io/examples/vision/siamese_network/

- Distributed training with Keras (TensorFlow tutorial)
  https://www.tensorflow.org/tutorials/distribute/keras

- Distributed training guide (TensorFlow guide):
  https://www.tensorflow.org/guide/distributed_training

"""

"""
## Setup and Imports
"""

import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.saving import register_keras_serializable

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""
## Configuration

We define all hyperparameters and settings in a configuration class.
"""

class Config:
    BATCH_SIZE = 32
    TARGET_SHAPE = (224, 224)
    IMG_SHAPE = TARGET_SHAPE + (3,)
    INITIAL_EPOCHS = 1
    FINE_TUNE_EPOCHS = 2
    TRIPLETS_PER_EPOCH = 10000
    MARGIN = 0.5  # Triplet loss margin
    LEARNING_RATE = 0.001
    MIN_IMAGES_PER_CLASS = 20
    SPLIT_RATIO = (0.7, 0.2, 0.1)  # train, val, test
    STEPS_PER_EPOCH = TRIPLETS_PER_EPOCH // BATCH_SIZE

"""
## Sample Dataset Creation

For demonstration, we'll create a simple dataset. In practice, replace this with your actual dataset.
Your dataset should be organized as: `path/class1/*.jpg, path/class2/*.jpg, etc.`
"""

def create_sample_dataset(base_path="./sample_data"):
    """Loads the oxford_flowers102 dataset into a tf.data.Dataset.
    Iterate over the dataset and save the images of samples as: `base_path/class_001/*.jpg, base_path/class_002/*.jpg, etc.`
    """
    print("Creating sample dataset...")
    os.makedirs(base_path, exist_ok=True)

    ds = tfds.load('oxford_flowers102', split='test', shuffle_files=True)

    for sample in ds.as_numpy_iterator():
        class_id = sample['label']
        class_path = os.path.join(base_path, f"class_{class_id:03d}")
        os.makedirs(class_path, exist_ok=True)

        img = sample['image']
        imgName = sample['file_name'].decode('utf-8')
        imgPath = os.path.join(class_path, imgName)
        cv2.imwrite(imgPath, img)

    print(f"Sample dataset created at {base_path}")
    return base_path

"""
## Dataset Management

The `DatasetManager` class handles dataset directory structure and creates train/validation/test splits.
"""

class DatasetManager:
    """Manages dataset directory structure and train/val/test splitting."""

    def __init__(self, dataset_path, min_images=20, split_ratio=(0.7, 0.2, 0.1)):
        self.dataset_path = dataset_path
        self.min_images = min_images
        self.split_ratio = split_ratio
        self.class_folders = self._collect_valid_folders()

    def _collect_valid_folders(self):
        """Collect folders with sufficient images."""
        valid_folders = []

        if not os.path.exists(self.dataset_path):
            print("Dataset path not found. Creating sample dataset...")
            self.dataset_path = create_sample_dataset()

        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if os.path.isdir(folder_path):
                num_images = len([f for f in os.listdir(folder_path)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if num_images >= self.min_images:
                    valid_folders.append(folder_path)

        random.shuffle(valid_folders)
        print(f"Found {len(valid_folders)} valid classes")
        return valid_folders

    def get_splits(self):
        """Split folders into train/val/test sets."""
        n_total = len(self.class_folders)
        n_train = int(self.split_ratio[0] * n_total)
        n_val = int(self.split_ratio[1] * n_total)

        train_folders = self.class_folders[:n_train]
        val_folders = self.class_folders[n_train:n_train + n_val]
        test_folders = self.class_folders[n_train + n_val:]

        print(f"Dataset split - Train: {len(train_folders)}, "
              f"Val: {len(val_folders)}, Test: {len(test_folders)}")

        return train_folders, val_folders, test_folders

"""
## Triplet Generation

The `TripletGenerator` creates triplet samples (anchor, positive, negative) for training.
For each triplet:
- **Anchor** and **Positive** come from the same class
- **Negative** comes from a different class

The goal is to minimize the distance between anchor-positive pairs while maximizing
the distance between anchor-negative pairs.
"""

class TripletGenerator:
    """Generates triplet samples for training."""

    def __init__(self, class_folders, images_per_class=20, mode='Train'):
        self.class_folders = class_folders
        self.images_per_class = images_per_class
        self.mode = mode
        self.class_images = self._load_class_images()

        print(f"[{mode}] TripletGenerator: {len(self.class_images)} classes")

    def _load_class_images(self):
        """Load and sample images for each class."""
        class_images = {}

        for class_folder in self.class_folders:
            image_files = glob.glob(os.path.join(class_folder, "*.jpg")) + \
                         glob.glob(os.path.join(class_folder, "*.jpeg")) + \
                         glob.glob(os.path.join(class_folder, "*.png"))

            if len(image_files) >= self.images_per_class:
                # Sample evenly distributed images
                indices = np.linspace(0, len(image_files) - 1,
                                    self.images_per_class, dtype=int)
                selected_images = [image_files[i] for i in indices]
                class_images[class_folder] = selected_images

        return class_images

    def generate_triplets(self):
        """Generator yielding triplet paths."""
        class_names = list(self.class_images.keys())

        while True:
            # Select anchor class and different negative class
            anchor_class = random.choice(class_names)
            negative_class = random.choice([c for c in class_names if c != anchor_class])

            # Select images
            anchor_img, positive_img = np.random.choice(
                self.class_images[anchor_class], 2, replace=False
            )
            negative_img = random.choice(self.class_images[negative_class])

            yield anchor_img, positive_img, negative_img

"""
## Image Processing

The `ImageProcessor` handles loading, decoding, and preprocessing of images.
"""

class ImageProcessor:
    """Handles image loading and preprocessing."""

    def __init__(self, target_shape):
        self.target_shape = target_shape

    def decode_and_resize(self, image_path):
        """Load and preprocess a single image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, self.target_shape)
        image.set_shape(self.target_shape + (3,))
        return image

    def process_triplet(self, anchor_path, positive_path, negative_path):
        """Process a triplet of image paths."""
        anchor = self.decode_and_resize(anchor_path)
        positive = self.decode_and_resize(positive_path)
        negative = self.decode_and_resize(negative_path)

        return (anchor, positive, negative), 0.0  # Dummy label

"""
## tf.data Pipeline

Create an optimized tf.data pipeline for efficient triplet loading and preprocessing.
"""

def create_tf_dataset(class_folders, config, mode='train'):
    """Create optimized tf.data.Dataset for triplet training."""

    generator = TripletGenerator(class_folders, config.MIN_IMAGES_PER_CLASS, mode.title())
    processor = ImageProcessor(config.TARGET_SHAPE)

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator.generate_triplets,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )

    # Apply processing pipeline
    dataset = dataset.map(processor.process_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.take(config.TRIPLETS_PER_EPOCH)
    dataset = dataset.cache()

    if mode == 'train':
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

"""
## Triplet Loss Implementation

Custom triplet margin loss encourages the network to place anchor closer to positive
than to negative by at least a margin.

**Loss = max(distance(anchor, positive) - distance(anchor, negative) + margin, 0)**
"""

@register_keras_serializable()
class TripletMarginLoss(tf.keras.losses.Loss):
    """Custom triplet margin loss implementation."""

    def __init__(self, margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Unused (dummy labels)
            y_pred: Tensor of shape (batch_size, 2) containing [ap_distance, an_distance]
        """
        ap_distance = y_pred[:, 0]  # Anchor-positive distance
        an_distance = y_pred[:, 1]  # Anchor-negative distance

        # Triplet loss: max(ap_distance - an_distance + margin, 0)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config

"""
## Distance Layer

Custom layer to compute pairwise distances for triplet loss.
"""

class DistanceLayer(tf.keras.layers.Layer):
    """Computes L2 distances between anchor-positive and anchor-negative pairs."""

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return tf.stack([ap_distance, an_distance], axis=1)


"""
## L2Normalization Layer

Custom layer to perform L2 normalization on inputs along a specified axis.
"""

class L2Normalization(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

"""
## Model Architecture

We create a Siamese network using ResNet50 as the backbone encoder.
The network produces normalized embeddings that are compared using L2 distance.

### Why normalize embeddings?
- Projects embeddings onto unit hypersphere
- Makes Euclidean distance equivalent to cosine similarity
- Improves training stability and convergence
"""

def create_siamese_model(config, fine_tuning=False):
    """Create Siamese network with ResNet50 backbone."""

    # ResNet50 backbone
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        input_shape=config.IMG_SHAPE,
        include_top=False
    )

    # Configure trainable layers
    if fine_tuning:
        base_model.trainable = True
        # Freeze early layers, unfreeze last layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    else:
        base_model.trainable = False

    # Build embedding network
    inputs = tf.keras.Input(shape=config.IMG_SHAPE)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128)(x)
    embeddings = L2Normalization(axis=1)(x)

    embedding_model = tf.keras.Model(inputs, embeddings, name='embedding')

    # Build Siamese network
    anchor_input = tf.keras.Input(shape=config.IMG_SHAPE, name='anchor')
    positive_input = tf.keras.Input(shape=config.IMG_SHAPE, name='positive')
    negative_input = tf.keras.Input(shape=config.IMG_SHAPE, name='negative')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    distances = DistanceLayer()(anchor_embedding, positive_embedding, negative_embedding)

    siamese_model = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name='siamese_network'
    )

    return siamese_model, embedding_model

"""
## Visualization Functions

Let's create some helper functions to visualize our data and results.
"""

def visualize_triplets(dataset, num_examples=3):
    """Visualize sample triplets from the dataset."""
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))

    for batch in dataset.take(1):
        inputs, _ = batch
        anchor_batch, positive_batch, negative_batch = inputs

        for i in range(min(num_examples, len(anchor_batch))):
            titles = ['Anchor', 'Positive', 'Negative']
            images = [anchor_batch[i], positive_batch[i], negative_batch[i]]

            for j, (img, title) in enumerate(zip(images, titles)):
                ax = axes[i, j] if num_examples > 1 else axes[j]
                ax.imshow(img.numpy())
                ax.set_title(f'{title}' if i == 0 else '')
                ax.axis('off')
        break

    plt.tight_layout()
    plt.show()


def plot_training_history(history1, history2=None, config=None):
    """Plot training and validation loss."""
    plt.figure(figsize=(12, 4))

    loss = history1.history['loss']
    val_loss = history1.history['val_loss']

    if history2:
        loss += history2.history['loss']
        val_loss += history2.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')

    if config and history2:
        plt.axvline(x=config.INITIAL_EPOCHS, color='orange',
                   linestyle='--', label='Start Fine-tuning')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

"""
## Training Pipeline

We use a two-phase training approach:
1. **Phase 1**: Train with frozen ResNet50 backbone
2. **Phase 2**: Fine-tune by unfreezing some layers

This progressive approach helps stabilize training and often leads to better results.
"""

def train_model():
    """Main training pipeline."""

    # Initialize configuration
    config = Config()

    # Setup distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Training on {strategy.num_replicas_in_sync} devices")

    # Prepare dataset
    dataset_manager = DatasetManager("./sample_data", config.MIN_IMAGES_PER_CLASS)
    train_folders, val_folders, test_folders = dataset_manager.get_splits()

    # Create datasets
    train_dataset = create_tf_dataset(train_folders, config, 'train')
    val_dataset = create_tf_dataset(val_folders, config, 'val')

    # Visualize sample triplets
    print("Sample triplets from training data:")
    visualize_triplets(train_dataset)

    # Build and compile model
    with strategy.scope():
        siamese_model, embedding_model = create_siamese_model(config)

        siamese_model.compile(
            optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
            loss=TripletMarginLoss(margin=config.MARGIN)
        )

    print("\nSiamese Network Architecture:")
    siamese_model.summary()

    # Setup callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath="./siamese_model.keras",
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Phase 1: Train with frozen backbone
    print(f"\n=== Phase 1: Frozen backbone training ({config.INITIAL_EPOCHS} epochs) ===")
    history1 = siamese_model.fit(
        train_dataset,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_data=val_dataset,
        epochs=config.INITIAL_EPOCHS,
        callbacks=[checkpoint_cb]
    )

    # Phase 2: Fine-tuning
    print(f"\n=== Phase 2: Fine-tuning ({config.FINE_TUNE_EPOCHS} epochs) ===")

    with strategy.scope():
        siamese_model_ft, _ = create_siamese_model(config, fine_tuning=True)
        siamese_model_ft.load_weights("./siamese_model.keras")

        # Use lower learning rate for fine-tuning
        siamese_model_ft.compile(
            optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE * 0.1),
            loss=TripletMarginLoss(margin=config.MARGIN)
        )

    history2 = siamese_model_ft.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.INITIAL_EPOCHS + config.FINE_TUNE_EPOCHS,
        initial_epoch=config.INITIAL_EPOCHS,
        callbacks=[checkpoint_cb]
    )

    # Plot training history
    plot_training_history(history1, history2, config)

    return siamese_model_ft

"""
## Model Evaluation

Evaluate the trained model using a confusion matrix approach.
"""

def evaluate_model(model, test_folders, config):
    """Evaluate the trained model on test set."""

    def load_images(folder_path, num_images=21):
        """Load and preprocess images from a folder."""
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                     glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                     glob.glob(os.path.join(folder_path, "*.png"))

        if len(image_files) < num_images:
            return None

        indices = np.linspace(0, len(image_files) - 1, num_images, dtype=int)
        selected_files = [image_files[i] for i in indices]

        images = []
        for img_path in selected_files:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.resize(img, config.TARGET_SHAPE)
            images.append(img)

        images = tf.stack(images)
        return tf.keras.applications.resnet.preprocess_input(images)

    # Extract embedding model
    embedding_model = None
    for layer in model.layers:
        if hasattr(layer, 'name') and layer.name == 'embedding':
            embedding_model = layer
            embedding_model.save("./embedding_model.keras") # Save the trained model to a file named "embedding_model.keras"
            break

    if not embedding_model:
        print("Embedding model not found!")
        return

    # Load test data and extract embeddings
    indicators = []
    test_samples = []

    print("Extracting embeddings from test set...")
    for folder in test_folders:
        images = load_images(folder)
        if images is not None:
            embeddings = embedding_model(images)
            indicators.append(embeddings[0])  # First as indicator
            test_samples.append(embeddings[1:])  # Rest as test samples

    if not indicators:
        print("No valid test data found!")
        return

    indicators = tf.stack(indicators)
    test_samples = tf.stack(test_samples)

    print(f"Test data shape - Indicators: {indicators.shape}, Test samples: {test_samples.shape}")

    # Compute confusion matrix based on distance comparisons
    confusion_matrix = []
    num_classes = len(indicators)

    for i in range(num_classes):
        indicator = indicators[i]
        pos_samples = test_samples[i]

        # Compute anchor-positive distances
        ap_distances = tf.reduce_sum(tf.square(indicator - pos_samples), axis=-1)

        confusion_row = [0] * num_classes

        for j in range(num_classes):
            if i == j:
                continue

            neg_samples = test_samples[j]
            # Compute anchor-negative distances
            an_distances = tf.reduce_sum(tf.square(indicator - neg_samples), axis=-1)

            # Compare distances - we want ap_distance < an_distance
            comparisons = tf.stack([ap_distances, an_distances])
            min_indices = tf.argmin(comparisons, axis=0)

            # Count correct predictions
            correct_count = tf.reduce_sum(1 - min_indices).numpy()
            incorrect_count = tf.reduce_sum(min_indices).numpy()

            confusion_row[i] += correct_count
            confusion_row[j] += incorrect_count

        confusion_matrix.append(confusion_row)

    confusion_matrix = np.array(confusion_matrix)

    # Visualize results
    plt.figure(figsize=(10, 8))
    class_names = [f'Class_{i}' for i in range(num_classes)]
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Compute metrics
    metrics = []
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - tp - fp - fn

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics.append([accuracy, precision, recall])

    metrics = np.array(metrics)
    avg_metrics = metrics.mean(axis=0)

    print(f"\nAverage Metrics:")
    print(f"Accuracy: {avg_metrics[0]:.3f}")
    print(f"Precision: {avg_metrics[1]:.3f}")
    print(f"Recall: {avg_metrics[2]:.3f}")


"""
## Load embedding network from disk.
Because the model includes a custom layer (L2Normalization),
we need to provide it in the `custom_objects` dictionary
so Keras knows how to reconstruct that layer when loading.
"""
def load_embedding_network(model_path="./embedding_model.keras"):
    trainedEmbedding = tf.keras.models.load_model(
        model_path,
        custom_objects={"L2Normalization": L2Normalization}
    )

    return trainedEmbedding

"""
## Complete Example

Let's put it all together and run the complete training and evaluation pipeline.
"""

# Train the model
trained_model = train_model()

# For evaluation, we need test data
dataset_manager = DatasetManager("./sample_data", Config.MIN_IMAGES_PER_CLASS)
_, _, test_folders = dataset_manager.get_splits()

if test_folders:
    print("\n=== Model Evaluation ===")
    evaluate_model(trained_model, test_folders, Config())
else:
    print("No test data available for evaluation")

print("\nTraining completed successfully!")
