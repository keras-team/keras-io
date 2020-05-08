# Keras Applications

Keras Applications are deep learning models that are made available alongside pre-trained weights.
These models can be used for prediction, feature extraction, and fine-tuning.

Weights are downloaded automatically when instantiating a model. They are stored at `~/.keras/models/`.

Upon instantiation, the models will be built according to the image data format set in your Keras configuration file at `~/.keras/keras.json`.
For instance, if you have set `image_data_format=channels_last`,
then any model loaded from this repository will get built according to the TensorFlow data format convention, "Height-Width-Depth".


## Available models

| Model | Size | Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth |
| ----- | ----: | --------------: | --------------: | ----------: | -----: |
| [Xception](xception) | 88 MB | 0.790 | 0.945 | 22,910,480 | 126 |
| [VGG16](vgg/#vgg16-function) | 528 MB | 0.713 | 0.901 | 138,357,544 | 23 |
| [VGG19](vgg/#vgg19-function) | 549 MB | 0.713 | 0.900 | 143,667,240 | 26 |
| [ResNet50](resnet/#resnet50-function) | 98 MB | 0.749 | 0.921 | 25,636,712 | - |
| [ResNet101](resnet/#resnet101-function) | 171 MB | 0.764 | 0.928 | 44,707,176 | - |
| [ResNet152](resnet/#resnet152-function) | 232 MB | 0.766 | 0.931 | 60,419,944 | - |
| [ResNet50V2](resnet/#resnet50v2-function) | 98 MB | 0.760 | 0.930 | 25,613,800 | - |
| [ResNet101V2](resnet/#resnet101v2-function) | 171 MB | 0.772 | 0.938 | 44,675,560 | - |
| [ResNet152V2](resnet/#resnet152v2-function) | 232 MB | 0.780 | 0.942 | 60,380,648 | - |
| [InceptionV3](inceptionv3) | 92 MB | 0.779 | 0.937 | 23,851,784 | 159 |
| [InceptionResNetV2](inceptionresnetv2) | 215 MB | 0.803 | 0.953 | 55,873,736 | 572 |
| [MobileNet](mobilenet) | 16 MB | 0.704 | 0.895 | 4,253,864 | 88 |
| [MobileNetV2](mobilenet/#mobilenetv2-function) | 14 MB | 0.713 | 0.901 | 3,538,984 | 88 |
| [DenseNet121](densenet/#densenet121-function) | 33 MB | 0.750 | 0.923 | 8,062,504 | 121 |
| [DenseNet169](densenet/#densenet169-function) | 57 MB | 0.762 | 0.932 | 14,307,880 | 169 |
| [DenseNet201](densenet/#densenet201-function) | 80 MB | 0.773 | 0.936 | 20,242,984 | 201 |
| [NASNetMobile](nasnet/#nasnetmobile-function) | 23 MB | 0.744 | 0.919 | 5,326,716 | - |
| [NASNetLarge](nasnet/#nasnetlarge-function) | 343 MB | 0.825 | 0.960 | 88,949,818 | - |
| [EfficientNetB0](efficientnet/#efficientnetb0-function) | 29 MB | - | - | 5,330,571 | - |
| [EfficientNetB1](efficientnet/#efficientnetb1-function) | 31 MB | - | - | 7,856,239 | - |
| [EfficientNetB2](efficientnet/#efficientnetb2-function) | 36 MB | - | - | 9,177,569 | - |
| [EfficientNetB3](efficientnet/#efficientnetb3-function) | 48 MB | - | - | 12,320,535 | - |
| [EfficientNetB4](efficientnet/#efficientnetb4-function) | 75 MB | - | - | 19,466,823 | - |
| [EfficientNetB5](efficientnet/#efficientnetb5-function) | 118 MB | - | - | 30,562,527 | - |
| [EfficientNetB6](efficientnet/#efficientnetb6-function) | 166 MB | - | - | 43,265,143 | - |
| [EfficientNetB7](efficientnet/#efficientnetb7-function) | 256 MB | - | - | 66,658,687 | - |

<br>
The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.


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



