# Python script for Knowledge Distillation with Keras
# Source: model_distillation_keras.md
# This script combines the runnable code with explanations from the Markdown guide.

# # Knowledge Distillation
#
# **Author:** Kenneth Borup
# **Date created:** 2020/09/01
# **Last modified:** 2020/09/01
# **Description:** Implementation of classical Knowledge Distillation.
#
# ⓘ This example uses Keras 3
#
# [View in Colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/knowledge_distillation.ipynb) • [GitHub source](https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py)

# ## Introduction to Knowledge Distillation
#
# Knowledge Distillation is a model compression technique where a smaller, simpler model
# (the "student") is trained to mimic the behavior of a larger, more complex, pre-trained model
# (the "teacher"). The goal is to transfer the "knowledge" from the teacher model to the
# student model, enabling the student to achieve comparable performance with significantly
# fewer parameters and computational resources.

# ### Teacher-Student Model Architecture
#
# The core idea revolves around a two-stage process:
#
# 1.  **Teacher Model:** A large, high-performing model is first trained on a dataset.
#     This model, once trained, serves as the "teacher." It has learned complex patterns
#     and relationships within the data.
# 2.  **Student Model:** A smaller, more lightweight model is then defined. The student model
#     is trained to learn from both the original dataset (using true labels) and the
#     teacher model's outputs.
#
# The student's training objective is a combination of learning to predict the correct labels
# and learning to replicate the teacher's output distribution.

# ### Softened Logits and Temperature
#
# Instead of just using the final hard predictions (e.g., the class with the highest probability)
# from the teacher, knowledge distillation often uses the teacher's *logits* (the raw,
# unnormalized outputs before the final activation function like softmax). These logits
# provide richer information about how the teacher model "thinks."
#
# To further enhance this knowledge transfer, the logits are often "softened." This is
# achieved by applying a **temperature (T)** scaling factor within the softmax function:
#
# `Softmax(logit_i / T) = exp(logit_i / T) / Σ_j exp(logit_j / T)`
#
# *   **High Temperature (T > 1):** A higher temperature produces a softer probability
#     distribution over classes. This means the probabilities for incorrect classes become
#     larger, revealing more information about which classes the teacher considers similar
#     or confusable for a given input. This "dark knowledge" (a term popularized by
#     Hinton et al.) refers to the rich information in the softened probabilities,
#     revealing how the teacher model generalizes and perceives similarities between
#     classes, which can be very valuable for the student.
# *   **Low Temperature (T = 1):** This is the standard softmax function.
# *   **Very Low Temperature (T -> 0):** The output approaches a one-hot encoding, similar
#     to hard labels.
#
# By training the student on these softened teacher probabilities (soft targets), the student
# learns a more nuanced representation of the data as seen by the teacher.

# ### Loss Functions in Knowledge Distillation
#
# The student model is typically trained using a composite loss function:
#
# 1.  **Distillation Loss (Soft Loss):** This loss component encourages the student
#     model's softened predictions to match the teacher model's softened predictions.
#     A common choice for this is the **Kullback-Leibler (KL) Divergence** between the
#     student's soft probability distribution and the teacher's soft probability
#     distribution. The KL divergence measures how one probability distribution
#     diverges from a second, expected probability distribution.
#     *   The Keras example uses `keras.losses.KLDivergence()` for this purpose.
#     *   This loss is calculated using the outputs from both student and teacher after
#         applying the temperature scaling to their logits.
#     *   The loss is often scaled by `T^2`. This scaling is used because when using KL
#         divergence with softmax outputs softened by temperature T, the magnitude of the
#         gradients is scaled by `1/T^2`. Multiplying by `T^2` counteracts this effect,
#         ensuring that the relative contribution of the distillation loss to the total
#         gradient remains somewhat consistent as temperature T is varied.
#
# 2.  **Student Loss (Hard Loss):** This is the standard loss function that measures the
#     difference between the student's predictions (on the original, "hard" labels) and
#     the ground-truth labels. For classification tasks, this is typically
#     **Sparse Categorical Cross-Entropy** or **Categorical Cross-Entropy**.
#     *   The Keras example uses `keras.losses.SparseCategoricalCrossentropy(from_logits=True)`.
#     *   This loss is calculated using the student's standard output logits (or
#         probabilities, with T=1) and the true labels from the dataset.
#
# The total loss is a weighted sum of these two components, controlled by a
# hyperparameter `alpha`:
#
# `Total Loss = alpha * Student Loss + (1 - alpha) * Distillation Loss`
#
# *   `alpha` balances the importance of matching the ground-truth labels versus
#     mimicking the teacher's behavior.
# *   A common practice is to use a smaller `alpha` (e.g., 0.1 as in the example)
#     to give more weight to the distillation loss initially, encouraging the student
#     to learn the teacher's generalizations.

# ## Advantages of Knowledge Distillation
#
# Knowledge distillation offers several significant benefits:
#
# *   **Model Compression:** It allows the creation of smaller, faster models that
#     retain much of the performance of larger, more cumbersome teacher models. This
#     is crucial for deploying models on devices with limited computational resources.
# *   **Performance Improvement on Edge Devices:** Smaller student models consume less
#     memory, use less power, and have lower latency, making them ideal for
#     deployment on mobile phones, IoT devices, and other edge computing platforms.
# *   **Transfer of Learning from Complex Models:** Teacher models can be very large
#     ensembles or complex architectures that are difficult to deploy directly.
#     Distillation allows their learned knowledge to be transferred to a more
#     manageable student model.
# *   **Improved Generalization:** By learning from the teacher's soft targets, the
#     student can sometimes learn better generalizations and even outperform a student
#     model of the same architecture trained solely on hard labels from scratch. The
#     teacher provides a richer training signal that can guide the student to a
#     better optimum.
# *   **Reduced Need for Extensive Labeled Data (in some contexts):** While the
#     initial teacher model requires labeled data, the student can learn effectively
#     from the teacher's outputs, potentially reducing the reliance on vast amounts
#     of labeled data for training the student directly, especially if the teacher
#     has generalized well.

# **Reference:**
# * [Hinton et al. (2015)](https://arxiv.org/abs/1503.02531).

# ## Practical Implementation with Keras
#
# This section provides a step-by-step guide to implementing knowledge distillation
# using Keras, based on the official Keras example.

# ### Step 1: Setup and Imports
#
# First, we import the necessary libraries. This includes Keras for building and
# training models, and NumPy for numerical operations.
import os

import keras
from keras import layers
from keras import ops # Keras 3 introduces ops for backend-agnostic operations
import numpy as np

# ### Step 2: Implementing the Distiller Class
#
# The core of the knowledge distillation process is encapsulated in a custom
# `Distiller` class. This class inherits from `keras.Model` and orchestrates the
# training of the student model.
#
# **Purpose of the `Distiller` Class:**
# The `Distiller` class is designed to manage the unique training loop required for
# knowledge distillation. It takes both the student and teacher models as input.
# Its primary responsibilities are:
# *   Applying the temperature scaling to the logits of both teacher and student.
# *   Calculating the two separate loss components:
#     1.  `student_loss`: Based on the student's performance against the true labels
#         (hard loss).
#     2.  `distillation_loss`: Based on the student's performance against the
#         teacher's softened labels (soft loss).
# *   Combining these losses using the `alpha` factor.
# *   Ensuring that only the student model's weights are updated during training.
#
# **Overridden Methods:**
#
# *   **`__init__(self, student, teacher)`:** The constructor simply stores the
#     student and teacher models.
# *   **`compile(...)`:** This method is overridden to configure the distiller. It
#     takes standard Keras arguments like `optimizer` and `metrics`. Additionally,
#     it accepts:
#     *   `student_loss_fn`: The loss function for the hard labels
#         (e.g., `SparseCategoricalCrossentropy`).
#     *   `distillation_loss_fn`: The loss function for the soft labels
#         (e.g., `KLDivergence`).
#     *   `alpha`: The weight for the `student_loss_fn`. The `distillation_loss_fn`
#         will be weighted by `(1 - alpha)`.
#     *   `temperature`: The temperature value used for softening the logits.
# *   **`compute_loss(...)`:** This is where the custom loss calculation happens.
#     1.  It first gets predictions from the teacher model (`teacher_pred`) for the
#         given input `x`. The teacher is in `training=False` mode as its weights
#         should not be updated.
#     2.  It calculates the `student_loss` using the provided `student_loss_fn`,
#         comparing the student's predictions (`y_pred`, which are the direct
#         output of the student model for input `x`) against the true labels `y`.
#     3.  It calculates the `distillation_loss`. This involves:
#         *   Softening the teacher's predictions (`teacher_pred`) and the student's
#             predictions (`y_pred`) using the `temperature`. This is done by
#             dividing the logits by `temperature` before applying `ops.softmax`.
#         *   Applying the `distillation_loss_fn` (e.g., KL Divergence) to these
#             softened probability distributions.
#         *   Multiplying the result by `self.temperature**2`. This scaling factor
#             is used because KL divergence with softened targets tends to result
#             in gradients that are `1/T^2` smaller. Multiplying by `T^2` ensures
#             that the relative contribution of the distillation loss to the total
#             gradient magnitude remains somewhat consistent across different
#             temperatures.
#     4.  The total loss is then computed as `loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss`.
# *   **`call(self, x)`:** This method defines the forward pass of the `Distiller`
#     model. It simply returns the student's predictions for the input `x`. This
#     means that when `Distiller.fit()` is called, `y_pred` in `compute_loss`
#     will be the student's direct output.
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student # The student model to train
        self.teacher = teacher # The teacher model, already trained

    def compile(
        self,
        optimizer,  # Optimizer for the student's weights
        metrics,  # Metrics to evaluate the student
        student_loss_fn,  # Loss function for student vs. true labels
        distillation_loss_fn,  # Loss function for student vs. teacher soft labels
        alpha=0.1,  # Weight for the student_loss
        temperature=3,  # Temperature for softening logits
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights.
            metrics: Keras metrics for evaluation.
            student_loss_fn: Loss function for the difference between student
                predictions and ground-truth labels (hard loss).
            distillation_loss_fn: Loss function for the difference between soft
                student predictions and soft teacher predictions (soft loss).
            alpha: Weight for student_loss_fn (0 to 1). The distillation_loss_fn
                   is weighted by (1 - alpha).
            temperature: Temperature for softening probability distributions.
                         Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        # Get teacher's predictions (logits) for the input x.
        # Teacher model is set to inference mode (training=False)
        # as its weights are frozen during distillation.
        teacher_pred = self.teacher(x, training=False)

        # Calculate the student loss (hard loss).
        # y_pred are the student's logits for input x (output of self.student(x)).
        # y are the ground-truth labels.
        student_loss = self.student_loss_fn(y, y_pred)

        # Calculate the distillation loss (soft loss).
        # Soften teacher and student predictions using temperature.
        # ops.softmax is applied to logits.
        distillation_loss = self.distillation_loss_fn(
            ops.softmax(teacher_pred / self.temperature, axis=1),
            ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2) # Scale loss by T^2, as per Hinton et al. (2015).

        # Combine student loss and distillation loss using alpha.
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        # The forward pass of the Distiller returns the student's predictions.
        # This is what `y_pred` in `compute_loss` will be.
        return self.student(x)

# ### Step 3: Defining the Teacher and Student Models
#
# For knowledge distillation, we need two models:
#
# *   **Teacher Model:** This is typically a larger, more complex model that has
#     already been trained or will be trained to a high performance level. Its
#     "knowledge" will be distilled into the student.
# *   **Student Model:** This is a smaller, more lightweight model that we want to
#     train. It should be less computationally expensive than the teacher.
#
# In this example, the architectures below precisely follow the Keras.io example's
# source code, defining simple Convolutional Neural Networks (CNNs) for image
# classification on MNIST. The teacher has more filters and thus more parameters
# than the student.
# Create the teacher model (larger and more complex)
teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"), # Second Conv2D layer
        # Note: The Keras.io example source code does not have a LeakyReLU or MaxPooling after the second Conv2D
        layers.Flatten(),
        layers.Dense(10), # Output layer for 10 classes (MNIST), logits output
    ],
    name="teacher",
)

# Create the student model (smaller and simpler)
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"), # Second Conv2D layer
        # Note: The Keras.io example source code does not have a LeakyReLU or MaxPooling after the second Conv2D
        layers.Flatten(),
        layers.Dense(10), # Output layer for 10 classes (MNIST), logits output
    ],
    name="student",
)

# For a fair comparison later, we also create a copy of the student model.
# This copy will be trained from scratch without distillation.
student_scratch = keras.models.clone_model(student)

# ### Step 4: Preparing the Dataset
#
# This example uses the MNIST dataset. The data preparation involves loading the
# dataset, normalizing pixel values to the range [0, 1], and reshaping the images
# to include a channel dimension, as expected by convolutional layers.
#
# While MNIST is used here, the knowledge distillation procedure is general and
# can be applied to any dataset with appropriate model architectures.
# Define batch size for training and evaluation
batch_size = 64

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
# Add a channel dimension (MNIST images are grayscale, shape becomes (samples, 28, 28, 1))
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# ### Step 5: Training the Teacher Model
#
# Before we can distill knowledge, the teacher model must be trained. This is
# standard model training using the training dataset and the true labels. The goal
# is to get a well-performing teacher whose knowledge can then be transferred.
# Compile the teacher model
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Use from_logits=True as the last layer is linear (no softmax activation)
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the teacher model
print("Training the teacher model...")
teacher.fit(x_train, y_train, epochs=5, batch_size=batch_size)

# Evaluate the trained teacher model on the test set
print("\nEvaluating the teacher model...")
teacher_eval_results = teacher.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Teacher Model - Test Loss: {teacher_eval_results[0]:.4f}, Test Accuracy: {teacher_eval_results[1]:.4f}")

# _(Note: The example output from the Keras.io tutorial is shown below for reference. Actual output may vary slightly based on environment and library versions.)_
# ```
# Epoch 1/5
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - loss: 0.2408 - sparse_categorical_accuracy: 0.9259
# Epoch 2/5
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.0912 - sparse_categorical_accuracy: 0.9726
# Epoch 3/5
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.0758 - sparse_categorical_accuracy: 0.9777
# Epoch 4/5
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.0690 - sparse_categorical_accuracy: 0.9797
# Epoch 5/5
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9825
#  313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0931 - sparse_categorical_accuracy: 0.9760
#
# [0.09044107794761658, 0.978100061416626]
# ```

# ### Step 6: Distilling Knowledge to the Student Model
#
# Now, we use the `Distiller` class to train the student model.
# The `Distiller` is initialized with the `student` and the already trained
# `teacher` model.
# When compiling the `Distiller`, we provide:
# *   An optimizer (this will only update the student's weights).
# *   Metrics to evaluate the student.
# *   `student_loss_fn`: The loss for hard labels (e.g., `SparseCategoricalCrossentropy`).
#     This encourages the student to learn the true labels.
# *   `distillation_loss_fn`: The loss for soft labels (e.g., `KLDivergence`).
#     This encourages the student to mimic the teacher's output distribution.
# *   `alpha`: A hyperparameter (0 to 1) that balances the `student_loss_fn` and
#     `distillation_loss_fn`. A typical value is 0.1, meaning `student_loss` has a
#     weight of 0.1 and `distillation_loss` has a weight of 0.9.
# *   `temperature`: The temperature for softening logits. A higher temperature
#     creates softer probability distributions.
#
# The `fit` method of the `Distiller` is then called. Internally, the `Distiller`
# uses its custom `compute_loss` method, which combines the student loss and the
# distillation loss to guide the student's training.
# Initialize and compile the Distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,       # Weight for student_loss; (1-alpha) is for distillation_loss
    temperature=10,  # Temperature for softening logits
)

# Distill knowledge from teacher to student
print("\nDistilling knowledge to student model...")
distiller.fit(x_train, y_train, epochs=3, batch_size=batch_size)

# Evaluate the distilled student model
print("\nEvaluating the distilled student model...")
distilled_student_eval_results = distiller.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Distilled Student Model - Test Loss: {distilled_student_eval_results[0]:.4f}, Test Accuracy: {distilled_student_eval_results[1]:.4f}")

# _(Note: The example output from the Keras.io tutorial is shown below for reference. Actual output may vary slightly based on environment and library versions.)_
# ```
# Epoch 1/3
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - loss: 1.8752 - sparse_categorical_accuracy: 0.7357
# Epoch 2/3
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - loss: 0.0333 - sparse_categorical_accuracy: 0.9475
# Epoch 3/3
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - loss: 0.0223 - sparse_categorical_accuracy: 0.9621
#  313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - loss: 0.0189 - sparse_categorical_accuracy: 0.9629
#
# [0.017046602442860603, 0.969200074672699]
# ```

# ### Step 7: Training a Student Model from Scratch (for Comparison)
#
# To assess the effectiveness of knowledge distillation, it's important to compare
# the distilled student model's performance against a student model of the exact
# same architecture trained conventionally (from scratch) using only the true labels.
# We use the `student_scratch` model (a clone of the original student architecture
# created in Step 3) for this purpose. This model is trained like any standard Keras model.
# Train the student model from scratch (without distillation)
print("\nTraining student model from scratch...")
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=3, batch_size=batch_size)

# Evaluate the student model trained from scratch
print("\nEvaluating the student model trained from scratch...")
scratch_student_eval_results = student_scratch.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Student Model (trained from scratch) - Test Loss: {scratch_student_eval_results[0]:.4f}, Test Accuracy: {scratch_student_eval_results[1]:.4f}")

# _(Note: The example output from the Keras.io tutorial is shown below for reference. Actual output may vary slightly based on environment and library versions.)_
# ```
# Epoch 1/3
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5111 - sparse_categorical_accuracy: 0.8460
# Epoch 2/3
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - loss: 0.1039 - sparse_categorical_accuracy: 0.9687
# Epoch 3/3
#  1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - loss: 0.0748 - sparse_categorical_accuracy: 0.9780
#  313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.0744 - sparse_categorical_accuracy: 0.9737
#
# [0.0629437193274498, 0.9778000712394714]
# ```

# ### Step 8: Evaluating and Comparing Performance
#
# After training all three models (teacher, distilled student, and student trained
# from scratch), we compare their performance on the test set.
#
# The key comparisons are:
# *   **Teacher vs. Distilled Student:** How close does the distilled student get to
#     the teacher's performance? The aim is for the student to be significantly
#     smaller yet achieve comparable (or even better) accuracy.
# *   **Distilled Student vs. Student Trained from Scratch:** Does distillation provide
#     a performance boost over training the same student architecture conventionally?
#     Often, the distilled student will outperform the one trained from scratch,
#     demonstrating the benefit of the "dark knowledge" transferred from the teacher.
#
# The example output from the Keras.io tutorial suggests the following typical
# accuracies (though these can vary with library versions, epochs, and random seeds):
# *   Teacher: ~97.6%
# *   Student trained from scratch: ~97.6% (or ~97.78% in the provided output logs)
# *   Distilled student: ~98.1% (or ~96.92% in the provided output logs)
#
# **Note:** The actual performance can vary based on random initializations, minor
# differences in library versions, specific model architectures, and hyperparameters.
# The important aspect is the relative performance trend. In many well-tuned
# scenarios, the distilled student outperforms the student trained from scratch and
# can approach or even exceed the teacher's performance on the given task, despite
# being a smaller model.
# Summary of performance
print("\nPerformance Summary:")
print(f"Teacher Model - Test Accuracy: {teacher_eval_results[1]:.4f}")
print(f"Distilled Student Model - Test Accuracy: {distilled_student_eval_results[1]:.4f}")
print(f"Student Model (trained from scratch) - Test Accuracy: {scratch_student_eval_results[1]:.4f}")

# Expected outcome:
# The distilled student model often achieves better performance than
# the student model trained from scratch and can approach the teacher's
# performance, despite being a smaller model.
# For example, if the teacher achieved 97.8%, the student from scratch 97.7%,
# the distilled student might achieve 97.9% or higher with fewer parameters. This also depends
# on the complexity of the task and the capacity of the student model.

# The final paragraph from the original Keras.io example is insightful:
# "If the teacher is trained for 5 full epochs and the student is distilled on this
# teacher for 3 full epochs, you should in this example experience a performance boost
# compared to training the same student model from scratch, and even compared to the
# teacher itself. You should expect the teacher to have accuracy around 97.6%, the
# student trained from scratch should be around 97.6%, and the distilled student
# should be around 98.1%. Remove or try out different seeds to use different weight
# initializations."
# This highlights that the distilled student can sometimes even surpass the teacher,
# possibly due to the regularizing effect of the distillation process (acting as a
# form of label smoothing and guiding the optimization) or the student model, being
# simpler, is less prone to overfitting on the specific dataset.
print("\nNote: The accuracies achieved can vary based on initialization seeds, library versions, and specific hardware.")
print("The Keras.io tutorial, for instance, reported the following accuracies with their setup:")
print("- Teacher: ~97.6%")
print("- Student trained from scratch: ~97.6%")
print("- Distilled student: ~98.1%")


# ## Conclusion
#
# Knowledge distillation is a powerful and versatile technique for transferring
# knowledge from a large, complex teacher model to a smaller, more efficient student
# model. By training the student to mimic the softened outputs of the teacher, as
# well as learning from the true labels, it's possible to create compact models
# that retain a significant portion of the teacher's performance.
#
# The key benefits include effective model compression, which is crucial for
# deployment on resource-constrained devices, and the potential for improved
# generalization and performance in the student model. This transfer of "dark
# knowledge" allows the student to learn nuances that might not be apparent from
# hard labels alone.
#
# As demonstrated in the Keras example, the core components of implementing
# knowledge distillation involve:
# *   Defining a capable **teacher model** and a smaller **student model**.
# *   Utilizing a custom **`Distiller` class** that manages the specialized
#     training loop, overriding methods like `compile`, `compute_loss`, and `call`.
# *   Employing a composite **loss function** that combines the standard student
#     loss (e.g., cross-entropy with true labels) and a distillation loss
#     (e.g., KL divergence between student's and teacher's soft predictions),
#     balanced by `alpha` and influenced by `temperature`.
#
# Knowledge distillation opens up possibilities for deploying sophisticated machine
# learning models in a wider range of applications, particularly where computational
# efficiency is paramount. It represents an intelligent approach to leveraging the
# capabilities of large models to create more practical and accessible AI solutions.

# End of script
