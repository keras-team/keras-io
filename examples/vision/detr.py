"""
Title: DETR : End-to-End Object Detection with Transformers
Author: [Ayyuce Demirbas](https://twitter.com/demirbasayyuce)
Date created: 2022/03/11
Last modified: 2022/03/11
Description: TensorFlow implementation of [End-to-End Object Detection with
Transformers paper](https://arxiv.org/pdf/2005.12872.pdf)
"""
"""
"""

"""
## Introduction 

Unlike traditional computer vision techniques, DETR approaches object detection as a
direct set prediction problem. It consists of a set-based global loss, which forces
unique predictions via bipartite matching, and a Transformer encoder-decoder
architecture. Given a fixed small set of learned object queries, DETR reasons about the
relations of the objects and the global image context to directly output the final set of
predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.
[1]

![](https://raw.githubusercontent.com/facebookresearch/detr/main/.github/DETR.png)
Figure 1 [2]
"""

"""
## Cloning the repository
"""

"""shell
!git clone https://github.com/Visual-Behavior/detr-tensorflow.git
"""

"""
To change directory to detr-tensorflow:
"""

"""shell
%cd detr-tensorflow
"""

"""
## Installing the requirements
"""

"""
Without imgaug, you can't import from detr_tf
"""

"""shell
!pip install imgaug==0.4.0
"""

"""shell
!pip install -r requirements.txt
"""

"""
## Model inference on your webcam
"""

"""
Make sure that you are in the detr-tensorflow folder.
"""

"""shell
!pwd
"""

import tensorflow as tf
import numpy as np
import cv2

from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf.networks.detr import get_detr_model
from detr_tf.data import processing
from detr_tf.data.coco import COCO_CLASS_NAME
from detr_tf.inference import get_model_inference, numpy_bbox_to_image


@tf.function
def run_inference(model, images, config):
    m_outputs = model(images, training=False)
    predicted_bbox, predicted_labels, predicted_scores = get_model_inference(
        m_outputs, config.background_class, bbox_format="xy_center"
    )
    return predicted_bbox, predicted_labels, predicted_scores


def run_webcam_inference(detr):

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert to RGB and process the input image
        model_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_input = processing.normalized_images(model_input, config)

        # Run inference
        predicted_bbox, predicted_labels, predicted_scores = run_inference(
            detr, np.expand_dims(model_input, axis=0), config
        )

        frame = frame.astype(np.float32)
        frame = frame / 255
        frame = numpy_bbox_to_image(
            frame,
            predicted_bbox,
            labels=predicted_labels,
            scores=predicted_scores,
            class_name=COCO_CLASS_NAME,
        )

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) == 1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = TrainingConfig()

# Load the model with the new layers to finetune
detr = get_detr_model(config, include_top=True, weights="detr")
config.background_class = 91

# Run webcam inference
run_webcam_inference(detr)

"""
![](https://raw.githubusercontent.com/Visual-Behavior/detr-tensorflow/main/images/webcam_detr.png)

Figure 2 [3]
"""

"""
## References

[1] https://github.com/facebookresearch/detr

[2] https://arxiv.org/pdf/2005.12872.pdf

[3] https://github.com/Visual-Behavior/detr-tensorflow
"""
