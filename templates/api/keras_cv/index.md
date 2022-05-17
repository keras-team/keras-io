# KerasCV API

KerasCV is a toolbox of modular building blocks (layers, metrics, losses, data augmentation) that computer vision engineers can leverage to quickly assemble production-grade, state-of-the-art training and inference pipelines for common use cases such as image classification, object detection, image segmentation, image data augmentation, etc.

KerasCV **Layers** can be used independently, or with the `keras.Model` class. Layers
implement specific self contained logic such as image data augmentation, regularization
during training, and more!

KerasCV **Metrics** are used for model evaluation. They can be used for both
train time and post training evaluation.  They can be used independently, or as part of the
standard `Model.fit()`, `Model.evaluate()`, `Model.predict()` flow.

{{toc}}
