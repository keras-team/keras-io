# Keras Applications

Keras Applications are deep learning models that are made available alongside pre-trained weights.
These models can be used for prediction, feature extraction, and fine-tuning.

Weights are downloaded automatically when instantiating a model. They are stored at `~/.keras/models/`.

Upon instantiation, the models will be built according to the image data format set in your Keras configuration file at `~/.keras/keras.json`.
For instance, if you have set `image_data_format=channels_last`,
then any model loaded from this repository will get built according to the TensorFlow data format convention, "Height-Width-Depth".


## Available models

| Model | Size (MB)| Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth | Time (ms) per inference step (CPU) | Time (ms) per inference step (GPU) |
| ----- | -------: | -------------: | -------------: |  --------: | ----: | ---------------------------------: | ---------------------------------: |
| [Xception](xception) | 88 | 79.0% | 94.5% | 22.9M | 81 | 109.4 | 8.1 |
| [VGG16](vgg/#vgg16-function) | 528 | 71.3% | 90.1% | 138.4M | 16 | 69.5 | 4.2 |
| [VGG19](vgg/#vgg19-function) | 549 | 71.3% | 90.0% | 143.7M | 19 | 84.8 | 4.4 |
| [ResNet50](resnet/#resnet50-function) | 98 | 74.9% | 92.1% | 25.6M | 107 | 58.2 | 4.6 |
| [ResNet50V2](resnet/#resnet50v2-function) | 98 | 76.0% | 93.0% | 25.6M | 103 | 45.6 | 4.4 |
| [ResNet101](resnet/#resnet101-function) | 171 | 76.4% | 92.8% | 44.7M | 209 | 89.6 | 5.2 |
| [ResNet101V2](resnet/#resnet101v2-function) | 171 | 77.2% | 93.8% | 44.7M | 205 | 72.7 | 5.4 |
| [ResNet152](resnet/#resnet152-function) | 232 | 76.6% | 93.1% | 60.4M | 311 | 127.4 | 6.5 |
| [ResNet152V2](resnet/#resnet152v2-function) | 232 | 78.0% | 94.2% | 60.4M | 307 | 107.5 | 6.6 |
| [InceptionV3](inceptionv3) | 92 | 77.9% | 93.7% | 23.9M | 189 | 42.2 | 6.9 |
| [InceptionResNetV2](inceptionresnetv2) | 215 | 80.3% | 95.3% | 55.9M | 449 | 130.2 | 10.0 |
| [MobileNet](mobilenet) | 16 | 70.4% | 89.5% | 4.3M | 55 | 22.6 | 3.4 |
| [MobileNetV2](mobilenet/#mobilenetv2-function) | 14 | 71.3% | 90.1% | 3.5M | 105 | 25.9 | 3.8 |
| [DenseNet121](densenet/#densenet121-function) | 33 | 75.0% | 92.3% | 8.1M | 242 | 77.1 | 5.4 |
| [DenseNet169](densenet/#densenet169-function) | 57 | 76.2% | 93.2% | 14.3M | 338 | 96.4 | 6.3 |
| [DenseNet201](densenet/#densenet201-function) | 80 | 77.3% | 93.6% | 20.2M | 402 | 127.2 | 6.7 |
| [NASNetMobile](nasnet/#nasnetmobile-function) | 23 | 74.4% | 91.9% | 5.3M | 389 | 27.0 | 6.7 |
| [NASNetLarge](nasnet/#nasnetlarge-function) | 343 | 82.5% | 96.0% | 88.9M | 533 | 344.5 | 20.0 |
| [EfficientNetB0](efficientnet/#efficientnetb0-function) | 29 | 77.1% | 93.3% | 5.3M | 132 | 46.0 | 4.9 |
| [EfficientNetB1](efficientnet/#efficientnetb1-function) | 31 | 79.1% | 94.4% | 7.9M | 186 | 60.2 | 5.6 |
| [EfficientNetB2](efficientnet/#efficientnetb2-function) | 36 | 80.1% | 94.9% | 9.2M | 186 | 80.8 | 6.5 |
| [EfficientNetB3](efficientnet/#efficientnetb3-function) | 48 | 81.6% | 95.7% | 12.3M | 210 | 140.0 | 8.8 |
| [EfficientNetB4](efficientnet/#efficientnetb4-function) | 75 | 82.9% | 96.4% | 19.5M | 258 | 308.3 | 15.1 |
| [EfficientNetB5](efficientnet/#efficientnetb5-function) | 118 | 83.6% | 96.7% | 30.6M | 312 | 579.2 | 25.3 |
| [EfficientNetB6](efficientnet/#efficientnetb6-function) | 166 | 84.0% | 96.8% | 43.3M | 360 | 958.1 | 40.4 |
| [EfficientNetB7](efficientnet/#efficientnetb7-function) | 256 | 84.3% | 97.0% | 66.7M | 438 | 1578.9 | 61.6 | 
| [EfficientNetV2B0](efficientnet_v2/#efficientnetv2b0-function) | 29 | 78.7% | 94.3% | 7.2M | - | - | - |
| [EfficientNetV2B1](efficientnet_v2/#efficientnetv2b1-function) | 34 | 79.8% | 95.0% | 8.2M | - | - | - |
| [EfficientNetV2B2](efficientnet_v2/#efficientnetv2b2-function) | 42 | 80.5% | 95.1% | 10.2M | - | - | - |
| [EfficientNetV2B3](efficientnet_v2/#efficientnetv2b3-function) | 59 | 82.0% | 95.8% | 14.5M | - | - | - |
| [EfficientNetV2S](efficientnet_v2/#efficientnetv2s-function) | 88 | 83.9% | 96.7% | 21.6M | - | - | - |
| [EfficientNetV2M](efficientnet_v2/#efficientnetv2m-function) | 220 | 85.3% | 97.4% | 54.4M | - | - | - |
| [EfficientNetV2L](efficientnet_v2/#efficientnetv2l-function) | 479 | 85.7% | 97.5% | 119.0M | - | - | - |

The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.

Time per inference step is the average of 30 batches and 10 repetitions.

- CPU: AMD EPYC Processor (with IBPB) (92 core)
- RAM: 1.7T
- GPU: Tesla A100
- Batch size: 32

Depth counts the number of layers with parameters.

-----

## Usage examples for image classification models

### Classify ImageNet classes with ResNet50

```python
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
```

### Extract features with VGG16

```python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### Extract features from an arbitrary intermediate layer with VGG19

```python
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

### Fine-tune InceptionV3 on a new set of classes

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(...)
```


### Build InceptionV3 over a custom input tensor

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```



