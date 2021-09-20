# KerasTuner API

The **Hyperparameters** class is used to specify a set of hyperparameters
and their values, to be used in the model building function.

The **Tuner** subclasses corresponding to different tuning algorithms are
called directly by the user to start the search or to get the best models.

The **Oracle** subclasses are the core search algorithms, receiving model evaluation
results from the Tuner and providing new hyperparameter values.

The **HyperModel** subclasses are predefined search spaces for certain model
families like ResNet and XceptionNet.

{{toc}}

