# Keras Applications

Keras Applications are deep learning models that are made available alongside pre-trained weights.
These models can be used for prediction, feature extraction, and fine-tuning.

Weights are downloaded automatically when instantiating a model. They are stored at `~/.keras/models/`.

Upon instantiation, the models will be built according to the image data format set in your Keras configuration file at `~/.keras/keras.json`.
For instance, if you have set `image_data_format=channels_last`,
then any model loaded from this repository will get built according to the TensorFlow data format convention, "Height-Width-Depth".


## Available models

| Model | Size (MB)| Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth | Time (ms) per inference step (CPU) | Time (ms) per inference step (GPU) |
| ----- | ----: | --------------: | --------------: |  ----------: | -----: | ----------------: | ----------------: |
| [Xception](xception) | 88 | 0.790 | 0.945 | 22,910,480 | 126 | 109.42 | 8.06 | 
| [VGG16](vgg/#vgg16-function) | 528 | 0.713 | 0.901 | 138,357,544 | 23 | 69.50 | 4.16 | 
| [VGG19](vgg/#vgg19-function) | 549 | 0.713 | 0.900 | 143,667,240 | 26 | 84.75 | 4.38 | 
| [ResNet50](resnet/#resnet50-function) | 98 | 0.749 | 0.921 | 25,636,712 | - | 58.20 | 4.55 | 
| [ResNet101](resnet/#resnet101-function) | 171 | 0.764 | 0.928 | 44,707,176 | - | 89.59 | 5.19 | 
| [ResNet152](resnet/#resnet152-function) | 232 | 0.766 | 0.931 | 60,419,944 | - | 127.43 | 6.54 | 
| [ResNet50V2](resnet/#resnet50v2-function) | 98 | 0.760 | 0.930 | 25,613,800 | - | 45.63 | 4.42 | 
| [ResNet101V2](resnet/#resnet101v2-function) | 171 | 0.772 | 0.938 | 44,675,560 | - | 72.73 | 5.43 | 
| [ResNet152V2](resnet/#resnet152v2-function) | 232 | 0.780 | 0.942 | 60,380,648 | - | 107.50 | 6.64 | 
| [InceptionV3](inceptionv3) | 92 | 0.779 | 0.937 | 23,851,784 | 159 | 42.25 | 6.86 | 
| [InceptionResNetV2](inceptionresnetv2) | 215 | 0.803 | 0.953 | 55,873,736 | 572 | 130.19 | 10.02 | 
| [MobileNet](mobilenet) | 16 | 0.704 | 0.895 | 4,253,864 | 88 | 22.60 | 3.44 | 
| [MobileNetV2](mobilenet/#mobilenetv2-function) | 14 | 0.713 | 0.901 | 3,538,984 | 88 | 25.90 | 3.83 | 
| [DenseNet121](densenet/#densenet121-function) | 33 | 0.750 | 0.923 | 8,062,504 | 121 | 77.14 | 5.38 | 
| [DenseNet169](densenet/#densenet169-function) | 57 | 0.762 | 0.932 | 14,307,880 | 169 | 96.40 | 6.28 | 
| [DenseNet201](densenet/#densenet201-function) | 80 | 0.773 | 0.936 | 20,242,984 | 201 | 127.24 | 6.67 | 
| [NASNetMobile](nasnet/#nasnetmobile-function) | 23 | 0.744 | 0.919 | 5,326,716 | - | 27.04 | 6.70 | 
| [NASNetLarge](nasnet/#nasnetlarge-function) | 343 | 0.825 | 0.960 | 88,949,818 | - | 344.51 | 19.96 | 
| [EfficientNetB0](efficientnet/#efficientnetb0-function) | 29 | - | - | 5,330,571 | - | 46.00 | 4.91 | 
| [EfficientNetB1](efficientnet/#efficientnetb1-function) | 31 | - | - | 7,856,239 | - | 60.20 | 5.55 | 
| [EfficientNetB2](efficientnet/#efficientnetb2-function) | 36 | - | - | 9,177,569 | - | 80.79 | 6.50 | 
| [EfficientNetB3](efficientnet/#efficientnetb3-function) | 48 | - | - | 12,320,535 | - | 139.97 | 8.77 | 
| [EfficientNetB4](efficientnet/#efficientnetb4-function) | 75 | - | - | 19,466,823 | - | 308.33 | 15.12 | 
| [EfficientNetB5](efficientnet/#efficientnetb5-function) | 118 | - | - | 30,562,527 | - | 579.18 | 25.29 | 
| [EfficientNetB6](efficientnet/#efficientnetb6-function) | 166 | - | - | 43,265,143 | - | 958.12 | 40.45 | 
| [EfficientNetB7](efficientnet/#efficientnetb7-function) | 256 | - | - | 66,658,687 | - | 1578.90 | 61.62 | 

<br>
The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.

Time per inference step is the average of 30 batchs and 10 repetitions.
   - CPU: AMD EPYC Processor (with IBPB) (92 core)
   - Ram: 1.7T
   - GPU: Tesla A100
   - Batch size: 32

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



