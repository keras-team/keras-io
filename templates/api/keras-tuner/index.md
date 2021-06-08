# Keras Tuner API

The **Hyperparameters** class is used by the user to define all the hyperparameters
in the model building function.

The **Tuner** subclasses corresponding to different tuning algorithms are the APIs
directly called by the user to start the search or to get the best models.

The **Oracle** subclasses are the core tuning algorithms receiving model evaluation
results from the Tuner and provide new hyperparameter values to the Tuner.

The **HyperModel** subclasses are predefined search spaces for certain model
families like ResNet and XceptionNet.

{{toc}}

