"""
Title: Video Classification with a CNN-RNN Architecture
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/05/28
Last modified: 2021/05/28
Description: Training a video classifier with transfer learning and a recurrent model with the UCF101 dataset.
"""
"""
This example demonstrates video classification. It is an important use-case with
applications in surveillance, security, and so on. We will be using the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)
to build our video classifier. The dataset consists of videos categorized into different
actions like cricket shot, punching, biking, etc. This is why the dataset is known to
build action recognizers which is just an extension of video classification.

A video is made of an ordered sequence of frames. While the frames constitue
**spatiality** the sequence of those frames constitute the **temporality** of a video. To
systematically model both of these aspects we generally use a hybrid architecture that
consists of convolutions (for spatiality) as well as recurrent layers (for temporality).
In this example, we will be using such a hybrid architecture consisting of a
Convolutional Neural Network (CNN) and a Recurrent Neural Network (RNN) consisting of
[GRU layers](https://keras.io/api/layers/recurrent_layers/gru/). These kinds of hybrid
architectures are popularly known as **CNN-RNN**.

This example requires TensorFlow 2.4 or higher, as well as TensorFlow Docs, which can be
installed using the following command:
"""

"""shell
!pip install -q git+https://github.com/tensorflow/docs
"""

"""
## Data collection

In order to keep the runtime of this example relatively short, we will be using a
subsampled version of the original UCF101 dataset. You can refer to [this notebook](https://colab.research.google.com/github/sayakpaul/Action-Recognition-in-TensorFlow/blob/main/Data_Preparation_UCF101.ipynb)
to know how the subsampling was done. 
"""

"""shell
!wget -q https://storage.googleapis.com/demo-experiments/ucf101_top5.tar.gz
!tar xf ucf101_top5.tar.gz
"""

"""
## Setup
"""

from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

"""
## Define hyperparameters
"""

EPOCHS = 100
BATCH_SIZE = 64
IMG_SIZE = 224

SEQ_LENGTH = 5

"""
## Data preparation
"""

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

train_df.sample(10)

"""
One of the many challenges of training video classifiers is figuring out a way to feed
the videos to a network. [This blogpost](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5)
discusses five such methods. As a video is an ordered sequence of frames, we can extract
the frames, organize them, and then feed them to our network. But the number of frames
may differ which would not allow mini-batch learning. To account for all these factors,
we can do the following:

1. Capture the frames of a video. 
2. Save the frames within a predefined serialization interval until a maximum frame count
is reached.

Videos of the UCF101 dataset is [known](https://www.crcv.ucf.edu/papers/UCF101_CRCV-TR-12-01.pdf)
to not contain extreme variations in objects and actions across frames. Because of this,
it is okay to only consider a few frames for the learning task. We will be using
[OpenCV's `VideoCapture()` method](https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html) to read
frames from videos.
"""


def separate_frames(
    video_names, root_dir, output_dir, frame_count=SEQ_LENGTH, save_interval=10
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    broken_video_names = []

    for video_name in tqdm(video_names):
        count = 0
        video_path = os.path.join(root_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        try:
            while True:
                (grabbed, frame) = cap.read()
                # Check if OpenCV was able to start reading the frames and
                # if not, then break.
                if not grabbed:
                    broken_video_names.append(video_name)
                    break
                if count == frame_count:
                    break
                if nb_frames % save_interval == 0:
                    frame_name = video_name.split(".")[0] + "_frame%d.jpg" % count
                    filename = os.path.join(output_dir, frame_name)
                    count += 1
                    cv2.imwrite(filename, frame)
        finally:
            cap.release()

    return broken_video_names


train_video_names = train_df["video_name"].values.tolist()
test_video_names = test_df["video_name"].values.tolist()

broken_video_names_train = separate_frames(train_video_names, "train", "train_frames")
broken_video_names_test = separate_frames(test_video_names, "test", "test_frames")

"""
Now that we have extracted the frames and serialized them we can organize them in a
[Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) for easy information retrieval.
"""


def create_df(dir):
    images = sorted(list(paths.list_images(dir)))
    images_paths = []
    classes = []
    for i in tqdm(range(len(images))):
        images_paths.append(dir + "/" + images[i].split("/")[1])
        classes.append(images[i].split("/")[1].split("_")[1])

    new_df = pd.DataFrame()
    new_df["image_path"] = images_paths
    new_df["class"] = classes

    return new_df


new_train_df = create_df("train_frames")
new_test_df = create_df("test_frames")

print()
print(f"Total number of training frames: {len(new_train_df)}")
print(f"Total number of testing frames: {len(new_test_df)}")

new_train_df.head(10)

"""
A video is now represented as an ordered sequence of five frames. This almost concludes
our initial data preparation steps. To ensure our model is invariant to order of the
videos, we need to shuffle the entries of the training DataFrame. But we cannot just
randomly shuffle them -- this would hurt the order in which consecutive frames of an
individual video is present. We need to shuffle respecting that interval.
"""


def shuffle_df(df, interval=SEQ_LENGTH):
    df_copy = df.copy()
    random_indices = list(range(0, len(df), interval))
    np.random.shuffle(random_indices)
    for idx in range(0, int(len(df) / interval)):
        df_copy.loc[idx * interval : idx * interval + interval - 1] = df.loc[
            random_indices[idx] : random_indices[idx] + interval - 1
        ].values

    return df_copy


"""
Thanks to [Ranjan Debnath](https://in.linkedin.com/in/desmond00) for helping with this
method.
"""

new_train_df = shuffle_df(new_train_df)
new_train_df.head(10)

"""
The training frames are now shuffled as expected.
"""

"""
## Feature extraction from frames

We can use a pre-trained network to extract meaningful features from the extracted
frames. The [`Applications`](https://keras.io/api/applications/) class of Keras provides
a number of state-of-the-art models pre-trained on the [ImageNet-1k dataset](http://image-net.org/).
We will be using the [InceptionV3 model](https://arxiv.org/abs/1512.00567) for this purpose.
"""


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    feature_extractor.trainable = False

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = keras.applications.inception_v3.preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


"""
We now create TensorFlow dataset objects from the video frame paths to efficiently
extract their features.
"""


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image


train_images_list = new_train_df["image_path"].values.tolist()
train_ds = tf.data.Dataset.from_tensor_slices(train_images_list)
train_ds = train_ds.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
).batch(BATCH_SIZE)

test_images_list = new_test_df["image_path"].values.tolist()
test_ds = tf.data.Dataset.from_tensor_slices(test_images_list)
test_ds = test_ds.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
).batch(BATCH_SIZE)

feature_extractor = build_feature_extractor()
train_features = feature_extractor.predict(train_ds)
test_features = feature_extractor.predict(test_ds)

"""
## Reshape features and prepare labels

Now that we have extracted the frame features, we need to feed them to our recurrent
network considering the temporality present inside of them. Data fed to the recurrent
layers must have a time-step dimension. So, how do we incorporate that in the extracted
features?

The features have a shape of `len(train_frames), 5, 5, 2048`). Recall that each video is
now represented as an ordered sequence of **five** frames. So, we can reshape our features
like so: `len(train_frames)/5, 5, 5*5*2048`). The second dimension (5) is now the
time-step dimension. It turns out that this heuristic works quite well in practice. It
just needs as adjustment in the number of examples such that it is divisible by the
number of frames representing a video. Fortunately, for this example, we did not have to
do that explicitly.
"""

train_features = train_features.reshape(
    len(train_features) // SEQ_LENGTH, SEQ_LENGTH, 5 * 5 * 2048
)
test_features = test_features.reshape(
    len(test_features) // SEQ_LENGTH, SEQ_LENGTH, 5 * 5 * 2048
)

"""
We now need to prepare the labels i.e. the action classes which are present as strings.
Neural networks do not understand string values, they must be converted to some numerical
form before they are fed to neural networks. Here we will use the [`StringLookup`](https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup)
layer to binarize the string class labels. It will first give us an integer encoding of
the string values and then it will [one-hot encode](https://developers.google.com/machine-learning/glossary#one-hot-encoding)
those values. The point of making the values one-hot encoded will be made clear in a moment. 
"""

label_processor = keras.layers.experimental.preprocessing.StringLookup(
    max_tokens=6, num_oov_indices=0, output_mode="binary", sparse=False
)
label_processor.adapt(new_train_df["class"].values)

train_labels = label_processor(new_train_df["class"].values[..., None])
test_labels = label_processor(new_test_df["class"].values[..., None])


# We can verify the vocabulary like so.
class_vocab = label_processor.get_vocabulary()
print(class_vocab)

# We can also verify if the string values were mapped as expected.
idx = np.random.choice(len(train_labels))
assert (
    class_vocab[train_labels[idx].numpy().argmax(-1)]
    == new_train_df["class"].values[idx]
)

"""
We then need to skip some labels from both `train_labels` and `test_labels` since the
number of samples has changed now. 
"""


def prepare_labels(tags):
    labels = []
    for (i, label) in enumerate(tags):
        if i % SEQ_LENGTH == 0:
            labels.append(label)
    return np.stack(labels)


train_labels = prepare_labels(train_labels)
test_labels = prepare_labels(test_labels)

"""
## Train the sequence model

Empirically, [label-smoothing](https://arxiv.org/abs/1906.02629) was found to provide an
improved performance. This is why we represented the integer labels as one-hot encoded
vectors in the first place.
"""


def get_sequence_model(optimizer, label_smoothing=0.1):
    rnn_model = keras.Sequential(
        [
            keras.layers.GRU(
                64, input_shape=(SEQ_LENGTH, 5 * 5 * 2048), return_sequences=True
            ),
            keras.layers.GRU(32),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(len(class_vocab), activation="softmax"),
        ]
    )

    rnn_model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return rnn_model


def run_experiment():
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model("adam")
    history = seq_model.fit(
        train_features,
        train_labels,
        validation_data=(test_features, test_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    return history, seq_model


history, sequence_model = run_experiment()

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="train_loss")
plt.legend()
plt.grid()
plt.show()

"""
The plot above suggests that our model has fit the data well. 
"""

"""
## Inference
"""

# For inference, we first extract frames from a single video and then extract features
# from those using the pre-trained InceptionV3 network.
def extract_features_test(video_name):
    output_dir = video_name.split(".")[0]
    separate_frames(
        [video_name],
        root_dir="test",
        output_dir=output_dir,
        frame_count=SEQ_LENGTH,
        save_interval=10,
    )

    frame_paths = sorted(list(paths.list_images(output_dir)))
    features = []
    frames = []

    for frame_path in frame_paths:
        frame = load_image(frame_path)
        feature = feature_extractor.predict(frame[None, ...])
        frames.append(frame)
        features.append(feature.squeeze())

    return np.stack(frames), np.stack(features)


# After feature extraction, we are good to pass the
# features to our recurrent model and obtain predictions.
def sequence_prediction(video_name):
    frames, features = extract_features_test(video_name)
    features = features.reshape(1, SEQ_LENGTH, 5 * 5 * 2048)
    probabilities = sequence_model.predict(features)[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("./animation.gif", converted_images, fps=10)
    return embed.embed_file("./animation.gif")


# We first make sure we are predicting for a video that
# is not broken.
correct_video_names_test = [
    x for x in test_video_names if x not in broken_video_names_test
]

test_video = np.random.choice(correct_video_names_test, 1)[0]
print()
print(f"Testing video: {test_video}")
frames = sequence_prediction(test_video)
to_gif(frames)

"""
## Next steps

* In this example, we made use of transfer learning for extracting meaningful features
from video frames. You could also fine-tune the pre-trained network to notice how that
affects the end results. You may even have better results with a smaller,
higher-accuracy model like [EfficientNetB0](https://arxiv.org/abs/1905.11946).
* Try different combinations of `max_frame_counts` and `save_interval` to observe how
that affects the performance.
* Train on a higher number of classes and see if you are able to get good performance.
* Following [this tutorial](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub), try a
[pre-trained action recognition model](https://arxiv.org/abs/1705.07750) from DeepMind.
* Rolling-averaging can be useful technique for video classification and it can be
combined with a standard image classification model to infer on videos. [This tutorial](https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/)
will help understand how to use rolling-averaging with an image classifier.
"""
