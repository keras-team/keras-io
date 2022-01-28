"""
Title: Text Image Denoiser.
Author: [Anish B](https://twitter.com/anishhacko)
Date created: 2021/01/18
Last modified: 2022/01/27
Description: Example of Text Image Denoiser coupled with Tesseract OCR engine to improve text readability.
"""
"""
## Introduction

This Example explains a simple **Text Image Denoiser for OCR** using U-Net based
Architectutre. **Autoencoders** are mostly employed for **Image Restoration** problems
but in recent times **U-Net** based architectures with **skip-connections** have gained
popularity for several Image-to-Image tasks. We aim to solve the problem with simple
**Pretrained U-Net as Encoder** and **Efficient Sub-Pixel CNN as Decoder**. The problem
we focus here is enhancing the Document Images which are deteriorated through external
degradations like blur, corrupt text blocks etc. We have followed up on few seminal
papers which are presented below for reference. At the end of this tutorial user will
gain clear understanding of building Custom Image Pre-Processors using Deep-Learning that
helps us to mitigate real world OCR issues.

**References:**
- [Enhancing OCR Accuracy with Super
Resolution](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/ocr_Ankit_Lat_ICPR_2018.pdf)
- [Improving the Perceptual Quality of Document
Images Using Deep Neural
Network](http://mile.ee.iisc.ac.in/publications/softCopy/DocumentAnalysis/ISNN_11page_65.pdf)
"""

"""
## Additional Set-up
"""

# !pip install pybind11
# !pip install fastwer
# !pip install pytesseract
# !sudo apt install tesseract-ocr

import fastwer
import pytesseract
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input

"""
## Dataset
We have created a synthetic document dataset that depicts several real world document
noises; we have used Blurs and Morphological filters to reconstruct real world noises.
"""

"""shell
!gdown https://drive.google.com/uc?id=1jQfHJUUQcpktQcjIzitvb4SldKwqcAMr
!unzip -q data.zip
"""

"""
## Dataset Pipeline
"""

train_source = sorted(glob("data/source/*.jpg"))[:-80]
train_targets = sorted(glob("data/target/*.jpg"))[:-80]

val_source = sorted(glob("data/source/*.jpg"))[-80:]
val_targets = sorted(glob("data/target/*.jpg"))[-80:]

HEIGHT, WIDTH = 352, 608
BATCH_SIZE = 4
BUFFER_SIZE = int(len(train_source) * 0.2)


def resize_with_pad(image, height, width):
    return tf.image.resize_with_pad(
        image=image, target_height=height, target_width=width
    )


def preprocess(image):
    image = tf.io.decode_png(tf.io.read_file(image), channels=3)
    image = resize_with_pad(image, height=HEIGHT, width=WIDTH)
    image = preprocess_input(image)
    return image


def data_preprocess(source, target):
    source_image = preprocess(source)
    target_image = preprocess(target)
    return source_image, target_image


def denormalize(array):
    array += 1
    array *= 127.5
    return array


train_pipe = tf.data.Dataset.from_tensor_slices((train_source, train_targets))
train_pipe = train_pipe.map(data_preprocess, tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)
train_set = train_pipe.batch(BATCH_SIZE).repeat()

val_pipe = tf.data.Dataset.from_tensor_slices((val_source, val_targets))
valid_set = val_pipe.map(data_preprocess, tf.data.AUTOTUNE).batch(BATCH_SIZE)

"""
## Dataset Visualiztion
"""

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax = ax.flatten()
for i, f in enumerate(valid_set.take(1)):
    noise, clean = f
    noise = denormalize(noise[i].numpy())
    clean = denormalize(clean[i].numpy())
    ax[0].imshow(noise.astype(np.int))
    ax[0].set_xlabel("source")
    ax[1].imshow(clean.astype(np.int))
    ax[1].set_xlabel("target")
fig.text(x=0.45, y=0.78, s="Dataset Visualization", fontweight="bold")
fig.set_tight_layout(None)

"""
## Model

We have used Pre-trained ```ResNet50V2``` U-Net as Encoder, The Feature Maps are
extracted at different levels forming Downscaling path, Then ```Sub-Pixel Layers``` on
the Upscaling path with skip-connections.
"""


def conv_unit(x, filters=64, padding="same"):
    x = keras.layers.Conv2D(filters=filters, kernel_size=3, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def subpixel_upsample(x, target_channels, upscale_ratio):
    estimated_filters = target_channels * (upscale_ratio ** 2)
    x = keras.layers.Conv2D(estimated_filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = tf.nn.depth_to_space(x, block_size=upscale_ratio)
    return x


def upscale_unit(previous_input, current_input, target_channels=3, upscale_ratio=2):
    previous_low_scale = subpixel_upsample(
        previous_input, target_channels=target_channels, upscale_ratio=upscale_ratio
    )
    current_high_scale = conv_unit(current_input, filters=target_channels)
    concat_scales = tf.concat((previous_low_scale, current_high_scale), axis=-1)
    feature_manifold = conv_unit(concat_scales, filters=target_channels)
    return feature_manifold


def denoiser():
    base_model = ResNet50V2(
        include_top=False, weights="imagenet", input_shape=(352, 608, 3)
    )
    encode0 = base_model.get_layer("input_1").output  # (None,352,608,3)
    encode1 = base_model.get_layer("conv1_conv").output  # (None,176,304,64)
    encode2 = base_model.get_layer("conv2_block3_1_relu").output  # (None,88,152,64)
    encode3 = base_model.get_layer("conv3_block4_1_relu").output  # (None,44,76,128)

    decode1 = upscale_unit(
        previous_input=encode3, current_input=encode2, target_channels=128
    )
    decode2 = upscale_unit(
        previous_input=decode1, current_input=encode1, target_channels=64
    )
    decode3 = upscale_unit(
        previous_input=decode2, current_input=encode0, target_channels=32
    )

    out_conv = keras.layers.Conv2D(3, 5, padding="same", activation="sigmoid")(decode3)
    denoiser_model = keras.Model(base_model.input, out_conv, name="denoiser")
    denoiser_model.summary()
    return denoiser_model


"""
## Training

The Model is Trained with an Objective of minimizing ```Mean Squared Error```.
"""

EPOCHS = 40
LEARNING_RATE = 1e-4

denoiser_net = denoiser()

early_stop = callbacks.EarlyStopping(patience=8)
ckpt = callbacks.ModelCheckpoint(
    "best_ckpt.h5", save_best_only=True, save_weights_only=True, verbose=1
)
optimizer = keras.optimizers.Adam(LEARNING_RATE)
denoiser_net.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

denoiser_net.fit(
    train_set,
    validation_data=valid_set,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=len(train_source) // BATCH_SIZE,
    validation_steps=len(val_source) // BATCH_SIZE,
    workers=-1,
    callbacks=[early_stop, ckpt],
)

"""
## Evaluation Utilities

Image Super-Resolution or Denoiser models can be evaluated with PSNR, SSIM metrics for
perceptual quality assessments but our task of **Text Restoration** requires a bit more,
our objective is to Enhance simple OCR accuracy engine where our model can act as a
**Pre-Processor**, to serve the purpose we will use <a
href="https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-ra
te-cer-and-word-error-rate-wer-853175297510">WER(Word Error Rate) </a> as a metric to
evaluate the real-world performance. The following codes will help plot the visual
comparisons of outputs given the **Raw Noisy Input** and **Model Restored Input** to
tesseract OCR engine.
"""


def word_error_rate(noise_image, reference_image, prediction_image):
    reference = pytesseract.image_to_string(reference_image)
    noiseimage_ocr_out = pytesseract.image_to_string(noise_image)
    predimage_ocr_out = pytesseract.image_to_string(prediction_image)
    noise_WER_score = fastwer.score_sent(noiseimage_ocr_out, reference)
    pred_WER_score = fastwer.score_sent(predimage_ocr_out, reference)
    return noise_WER_score, pred_WER_score


def generate_results(file):
    """
    The following function generates visual and WER (Word Error Rates) comparison
    plots between raw input image when passed through OCR engine and model reconstructed
    image when passed through OCR, signifying the major improvement the model can offer
    in a workflow.
    """

    source_image = tf.io.decode_png(tf.io.read_file(file), channels=3)
    source_image = resize_with_pad(source_image, height=HEIGHT, width=WIDTH)
    source_image = preprocess_input(source_image)
    noise_ground_truth = denormalize(source_image)
    noise_ground_truth = keras.utils.array_to_img(noise_ground_truth)
    clean_ground_truth = tf.io.decode_png(
        tf.io.read_file(file.replace("source", "target")), channels=3
    )
    clean_ground_truth = resize_with_pad(clean_ground_truth, height=HEIGHT, width=WIDTH)
    clean_ground_truth = keras.utils.array_to_img(clean_ground_truth)

    source_image = tf.expand_dims(source_image, axis=0)
    prediction = denoiser_net.predict(source_image)
    prediction = denormalize(prediction[0])
    prediction = keras.utils.array_to_img(prediction)

    noise_error_rate, reconstructed_error_rate = word_error_rate(
        noise_image=noise_ground_truth,
        reference_image=clean_ground_truth,
        prediction_image=prediction,
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(noise_ground_truth)
    axes[0].set_xlabel(f"Input-WER score : {round(noise_error_rate,2)}")
    axes[1].imshow(prediction)
    axes[1].set_xlabel(f"Reconstructed-WER score : {round(reconstructed_error_rate,2)}")
    #     axes[2].imshow(clean_ground_truth); axes[2].set_xlabel("clean_ground-truth")
    fig.suptitle(
        "Visual comparison and WER(Word Error Rate) Evaluation charts",
        weight="bold",
        fontsize=12,
    )
    return noise_ground_truth, prediction, clean_ground_truth


"""
## Visualizing Results 
"""

test_images = [np.random.choice(val_source) for i in range(6)]
for f in test_images:
    generate_results(f)

"""
## Conclusion

We are able to observe using the model as a Pre-Processor enables significant amount of
improvement in Word Error Rate and perceptual quality of the image, the experiment can be
extended to different Image-to-Image applications, also Training with huge Dataset and
longer Epochs can elevate the performance of the model further.
"""
