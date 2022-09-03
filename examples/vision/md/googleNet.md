# MultiClass Classification Using GoogleNet

**Author:** [Tarun R Jain](https://twitter.com/TRJ_0751)<br>
**Date created:** 2022/09/03 <br>
**Last modified:** 2022/09/03 <br>
**Description:** Going Deeper with Convolutions (GoogleNet aka Inception-1)

## Introduction

The [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) evaluates algorithms for object detection and image classification at large scale. Reseachers at Google introduced a Deep Convolution Neural Network Architecture codenamed "Inception" and named the paper ["Going Deeper with Convolutions"](https://arxiv.org/abs/1409.4842). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.

Reference: [Paperswithcode](https://paperswithcode.com/paper/going-deeper-with-convolutions)

<div align="center">
<img src="https://miro.medium.com/max/1400/0*Tt7kChwQq2XkcN2q.png">
<p style='text-align:center;'>GoogLeNet incarnation of the Inception architecture</p>
</div>


```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras import models, layers
import datetime
```

```python
IMAGE_SHAPE = (224,224) #(224,224,3) is the input image shape taken in GoogleNet paper
numEPOCHS = 5 #number of epochs, you can try training for 20 epochs
BATCH_SIZE = 64 #batch size to train the data
CLASS_NUM = 6 #data contains total 6 labels
```

## Data Collection

The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.
Download: [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

```python
TRAIN_DIR = "train" #while you download the dataset, the dir will be named seg_train. You can rename it to train
TEST_DIR = "test"
```

```python
classLabels = os.listdir(TRAIN_DIR)
```

```python
print(classLabels) #will display 6 different classes
```
    ['street', 'mountain', 'sea', 'forest', 'glacier', 'buildings']

```python
def totalImagesCount(DIR):
    totalImages = 0
    for img in classLabels:
        totalImages+= len(os.listdir(os.path.join(DIR,img)))
    print(f"{DIR} consists of {totalImages} images")
```


```python
totalImagesCount(TRAIN_DIR)
```

    train consists of 14034 images

```python
totalImagesCount(TEST_DIR)
```

    test consists of 3000 images

The data Consists of `14304` training images and `3000` testing images. A perfect dataset to perform MultiClass Classification

```python
def checkImgLen(DATA_DIR):
    totalImgPerClass = [len(os.listdir(os.path.join(DATA_DIR,overallImg))) for overallImg in classLabels]
    plt.bar(classLabels,totalImgPerClass)
```


```python
checkImgLen(TRAIN_DIR)
```
    
![png](/img/examples/vision/googleNet/googleNet_1.png)
    

```python
checkImgLen(TEST_DIR)
```
    
![png](/img/examples/vision/googleNet/googleNet_2.png)
    

## Prepare the Data


```python
def prepareData(DATA_DIR,augment=False,shuffle=False):
    """
    Augmentation techniques such as rescaling, rotation and zoom range is performed only on training data.
    On Validation data we only need to rescale the image data. 
    """
    if augment:
        aug = ImageDataGenerator(rescale=1./255,
                                  rotation_range=45,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
    else:
        aug = ImageDataGenerator(rescale=1./255)
        
    #prepare the training data and validation data out of ImageDataGenerator and local directory
    return aug.flow_from_directory(DATA_DIR,
                                   batch_size=BATCH_SIZE,
                                   target_size=IMAGE_SHAPE,
                                   shuffle=shuffle,
                                   class_mode='categorical')
```


```python
training_data = prepareData(TRAIN_DIR,shuffle=True,augment=True)
```

    Found 14034 images belonging to 6 classes.


```python
validation_data = prepareData(TEST_DIR)
```

    Found 3000 images belonging to 6 classes.


```python
#ImageDataGenerator converts the class indices into one hot encoding to represent each image
```

```python
print(training_data.class_indices)
```

    {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

## Visualize the prepared Data

#### Data Augmentation= True on Training Data

```python
fig = plt.figure(figsize = (10,10))
for i in range(9):
    plt.subplot(3,3, i+1)
    for x,y in training_data:
        img = x[0]
        plt.imshow(img)
        plt.axis('off')
        break
```

![png](/img/examples/vision/googleNet/googleNet_3.png)


#### Visualization on Validation data along with Labels

The validation data is of type tuple, where first index is image and second label associated to the image.
- To plot the image we shall consider the first index(`image[0]`), since each image is made of `64 batch_size`. 
- To decode the label we shall use `np.argmax`, since label is in form of `One Hot Encoding`.


```python
plt.figure(figsize=(15,10))
n = 12
for i in range(n):
    plt.subplot(3,4,i+1)
    validImages,validLabels = validation_data[i+(i*3)]
    plt.imshow(validImages[0])
    decodeLabel = np.argmax(validLabels)
    plt.title(decodeLabel)
    plt.axis("off")
```
 
![png](/img/examples/vision/googleNet/googleNet_4.png)

## Build GoogleNet Architecture(Model)

The main idea of the Inception architecture is based on finding out how an optimal local sparse
structure in a convolutional vision network can be approximated and covered by readily available
dense components. 


```python
def inception(_input, filters):
    """
    3a, 3b, 4a, 4b, 4c, 4d, 5a, 5b are inception layers
    Note:
    1. All inception layer should have strides of 1
    2. Same Padding
    3. Depth of 2 i.e., Stack two Conv2D layers
    """
    layer1 = layers.Conv2D(filters[0],(1,1), strides=1, padding='same', activation='relu')(_input)
    
    #In order to avoid patchalignment issues, current incarnations of the Inception architecture are restricted to filter sizes:
    #1×1, 3×3 and 5×5
    
    layer2 = layers.Conv2D(filters[1][0], (1,1), strides=1, padding='same', activation='relu')(_input)
    layer2 = layers.Conv2D(filters[1][1], (3,3), strides=1, padding='same', activation='relu')(layer2)
    
    layer3 = layers.Conv2D(filters[2][0], (1,1), strides=1, padding='same', activation='relu')(_input)
    layer3 = layers.Conv2D(filters[2][1], (5,5), strides=1, padding='same', activation='relu')(layer3)

    layer4 = layers.MaxPooling2D((3,3), strides=1, padding='same')(_input)
    layer4 = layers.Conv2D(filters[3],(1,1), strides=1, padding='same', activation='relu')(layer4)

    return layers.Concatenate(axis=-1)([layer1,layer2,layer3,layer4])
```

<div align="center">
<img src="https://www.researchgate.net/profile/Ferhat-Ozgur-Catak/publication/342003740/figure/fig2/AS:900073961582598@1591605763849/Inception-module-naive-version-37.jpg">
 <p style='text-align:center;'>Inception module, naive version</p>
</div>


```python
def GoogleNet():
    input_layer = layers.Input(shape=(224,224,3))
    
    layer = layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(input_layer)
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    layer = layers.BatchNormalization()(layer)
    
    layer = layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = layers.Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

    layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
    layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    
    layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
    layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
    layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
    layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
    layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
    layer = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
    layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
    #after 5th inception layers; apply Average Pooling with (7,7)/1
    layer = layers.AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
    #averagepooling is the only layer with valid padding

    layer = layers.Flatten()(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.Dense(units=256, activation='linear')(layer)
    output = layers.Dense(units=CLASS_NUM, activation='softmax')(layer)
    
    model = models.Model(inputs=input_layer, outputs=output)
    
    return model
```

All the convolutions, including those inside the Inception modules, use rectified linear activation.
The size of the receptive field in our network is 224×224 taking RGB color channels with mean subtraction. “#3×3 reduce” and “#5×5 reduce” stands for the number of 1×1 filters in the reduction
layer used before the 3×3 and 5×5 convolutions


```python
model = GoogleNet()
```


```python
#cross check the GoogleNet Architecture with model.summary()
model.summary()
```

    Model: "model_1"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                    )]                                                                
                                                                                                      
     conv2d_57 (Conv2D)             (None, 112, 112, 64  9472        ['input_2[0][0]']                
                                    )                                                                 
                                                                                                      
     max_pooling2d_13 (MaxPooling2D  (None, 56, 56, 64)  0           ['conv2d_57[0][0]']              
     )                                                                                                
                                                                                                      
     batch_normalization_2 (BatchNo  (None, 56, 56, 64)  256         ['max_pooling2d_13[0][0]']       
     rmalization)                                                                                     
                                                                                                      
     conv2d_58 (Conv2D)             (None, 56, 56, 64)   4160        ['batch_normalization_2[0][0]']  
                                                                                                      
     conv2d_59 (Conv2D)             (None, 56, 56, 192)  110784      ['conv2d_58[0][0]']              
                                                                                                      
     batch_normalization_3 (BatchNo  (None, 56, 56, 192)  768        ['conv2d_59[0][0]']              
     rmalization)                                                                                     
                                                                                                      
     max_pooling2d_14 (MaxPooling2D  (None, 28, 28, 192)  0          ['batch_normalization_3[0][0]']  
     )                                                                                                
                                                                                                      
     conv2d_61 (Conv2D)             (None, 28, 28, 96)   18528       ['max_pooling2d_14[0][0]']       
                                                                                                      
     conv2d_63 (Conv2D)             (None, 28, 28, 16)   3088        ['max_pooling2d_14[0][0]']       
                                                                                                      
     max_pooling2d_15 (MaxPooling2D  (None, 28, 28, 192)  0          ['max_pooling2d_14[0][0]']       
     )                                                                                                
                                                                                                      
     conv2d_60 (Conv2D)             (None, 28, 28, 64)   12352       ['max_pooling2d_14[0][0]']       
                                                                                                      
     conv2d_62 (Conv2D)             (None, 28, 28, 128)  110720      ['conv2d_61[0][0]']              
                                                                                                      
     conv2d_64 (Conv2D)             (None, 28, 28, 32)   12832       ['conv2d_63[0][0]']              
                                                                                                      
     conv2d_65 (Conv2D)             (None, 28, 28, 32)   6176        ['max_pooling2d_15[0][0]']       
                                                                                                      
     concatenate_9 (Concatenate)    (None, 28, 28, 256)  0           ['conv2d_60[0][0]',              
                                                                      'conv2d_62[0][0]',              
                                                                      'conv2d_64[0][0]',              
                                                                      'conv2d_65[0][0]']              
                                                                                                      
     conv2d_67 (Conv2D)             (None, 28, 28, 128)  32896       ['concatenate_9[0][0]']          
                                                                                                      
     conv2d_69 (Conv2D)             (None, 28, 28, 32)   8224        ['concatenate_9[0][0]']          
                                                                                                      
     max_pooling2d_16 (MaxPooling2D  (None, 28, 28, 256)  0          ['concatenate_9[0][0]']          
     )                                                                                                
                                                                                                      
     conv2d_66 (Conv2D)             (None, 28, 28, 128)  32896       ['concatenate_9[0][0]']          
                                                                                                      
     conv2d_68 (Conv2D)             (None, 28, 28, 192)  221376      ['conv2d_67[0][0]']              
                                                                                                      
     conv2d_70 (Conv2D)             (None, 28, 28, 96)   76896       ['conv2d_69[0][0]']              
                                                                                                      
     conv2d_71 (Conv2D)             (None, 28, 28, 64)   16448       ['max_pooling2d_16[0][0]']       
                                                                                                      
     concatenate_10 (Concatenate)   (None, 28, 28, 480)  0           ['conv2d_66[0][0]',              
                                                                      'conv2d_68[0][0]',              
                                                                      'conv2d_70[0][0]',              
                                                                      'conv2d_71[0][0]']              
                                                                                                      
     max_pooling2d_17 (MaxPooling2D  (None, 14, 14, 480)  0          ['concatenate_10[0][0]']         
     )                                                                                                
                                                                                                      
     conv2d_73 (Conv2D)             (None, 14, 14, 96)   46176       ['max_pooling2d_17[0][0]']       
                                                                                                      
     conv2d_75 (Conv2D)             (None, 14, 14, 16)   7696        ['max_pooling2d_17[0][0]']       
                                                                                                      
     max_pooling2d_18 (MaxPooling2D  (None, 14, 14, 480)  0          ['max_pooling2d_17[0][0]']       
     )                                                                                                
                                                                                                      
     conv2d_72 (Conv2D)             (None, 14, 14, 192)  92352       ['max_pooling2d_17[0][0]']       
                                                                                                      
     conv2d_74 (Conv2D)             (None, 14, 14, 208)  179920      ['conv2d_73[0][0]']              
                                                                                                      
     conv2d_76 (Conv2D)             (None, 14, 14, 48)   19248       ['conv2d_75[0][0]']              
                                                                                                      
     conv2d_77 (Conv2D)             (None, 14, 14, 64)   30784       ['max_pooling2d_18[0][0]']       
                                                                                                      
     concatenate_11 (Concatenate)   (None, 14, 14, 512)  0           ['conv2d_72[0][0]',              
                                                                      'conv2d_74[0][0]',              
                                                                      'conv2d_76[0][0]',              
                                                                      'conv2d_77[0][0]']              
                                                                                                      
     conv2d_79 (Conv2D)             (None, 14, 14, 112)  57456       ['concatenate_11[0][0]']         
                                                                                                      
     conv2d_81 (Conv2D)             (None, 14, 14, 24)   12312       ['concatenate_11[0][0]']         
                                                                                                      
     max_pooling2d_19 (MaxPooling2D  (None, 14, 14, 512)  0          ['concatenate_11[0][0]']         
     )                                                                                                
                                                                                                      
     conv2d_78 (Conv2D)             (None, 14, 14, 160)  82080       ['concatenate_11[0][0]']         
                                                                                                      
     conv2d_80 (Conv2D)             (None, 14, 14, 224)  226016      ['conv2d_79[0][0]']              
                                                                                                      
     conv2d_82 (Conv2D)             (None, 14, 14, 64)   38464       ['conv2d_81[0][0]']              
                                                                                                      
     conv2d_83 (Conv2D)             (None, 14, 14, 64)   32832       ['max_pooling2d_19[0][0]']       
                                                                                                      
     concatenate_12 (Concatenate)   (None, 14, 14, 512)  0           ['conv2d_78[0][0]',              
                                                                      'conv2d_80[0][0]',              
                                                                      'conv2d_82[0][0]',              
                                                                      'conv2d_83[0][0]']              
                                                                                                      
     conv2d_85 (Conv2D)             (None, 14, 14, 128)  65664       ['concatenate_12[0][0]']         
                                                                                                      
     conv2d_87 (Conv2D)             (None, 14, 14, 24)   12312       ['concatenate_12[0][0]']         
                                                                                                      
     max_pooling2d_20 (MaxPooling2D  (None, 14, 14, 512)  0          ['concatenate_12[0][0]']         
     )                                                                                                
                                                                                                      
     conv2d_84 (Conv2D)             (None, 14, 14, 128)  65664       ['concatenate_12[0][0]']         
                                                                                                      
     conv2d_86 (Conv2D)             (None, 14, 14, 256)  295168      ['conv2d_85[0][0]']              
                                                                                                      
     conv2d_88 (Conv2D)             (None, 14, 14, 64)   38464       ['conv2d_87[0][0]']              
                                                                                                      
     conv2d_89 (Conv2D)             (None, 14, 14, 64)   32832       ['max_pooling2d_20[0][0]']       
                                                                                                      
     concatenate_13 (Concatenate)   (None, 14, 14, 512)  0           ['conv2d_84[0][0]',              
                                                                      'conv2d_86[0][0]',              
                                                                      'conv2d_88[0][0]',              
                                                                      'conv2d_89[0][0]']              
                                                                                                      
     conv2d_91 (Conv2D)             (None, 14, 14, 144)  73872       ['concatenate_13[0][0]']         
                                                                                                      
     conv2d_93 (Conv2D)             (None, 14, 14, 32)   16416       ['concatenate_13[0][0]']         
                                                                                                      
     max_pooling2d_21 (MaxPooling2D  (None, 14, 14, 512)  0          ['concatenate_13[0][0]']         
     )                                                                                                
                                                                                                      
     conv2d_90 (Conv2D)             (None, 14, 14, 112)  57456       ['concatenate_13[0][0]']         
                                                                                                      
     conv2d_92 (Conv2D)             (None, 14, 14, 288)  373536      ['conv2d_91[0][0]']              
                                                                                                      
     conv2d_94 (Conv2D)             (None, 14, 14, 64)   51264       ['conv2d_93[0][0]']              
                                                                                                      
     conv2d_95 (Conv2D)             (None, 14, 14, 64)   32832       ['max_pooling2d_21[0][0]']       
                                                                                                      
     concatenate_14 (Concatenate)   (None, 14, 14, 528)  0           ['conv2d_90[0][0]',              
                                                                      'conv2d_92[0][0]',              
                                                                      'conv2d_94[0][0]',              
                                                                      'conv2d_95[0][0]']              
                                                                                                      
     conv2d_97 (Conv2D)             (None, 14, 14, 160)  84640       ['concatenate_14[0][0]']         
                                                                                                      
     conv2d_99 (Conv2D)             (None, 14, 14, 32)   16928       ['concatenate_14[0][0]']         
                                                                                                      
     max_pooling2d_22 (MaxPooling2D  (None, 14, 14, 528)  0          ['concatenate_14[0][0]']         
     )                                                                                                
                                                                                                      
     conv2d_96 (Conv2D)             (None, 14, 14, 256)  135424      ['concatenate_14[0][0]']         
                                                                                                      
     conv2d_98 (Conv2D)             (None, 14, 14, 320)  461120      ['conv2d_97[0][0]']              
                                                                                                      
     conv2d_100 (Conv2D)            (None, 14, 14, 128)  102528      ['conv2d_99[0][0]']              
                                                                                                      
     conv2d_101 (Conv2D)            (None, 14, 14, 128)  67712       ['max_pooling2d_22[0][0]']       
                                                                                                      
     concatenate_15 (Concatenate)   (None, 14, 14, 832)  0           ['conv2d_96[0][0]',              
                                                                      'conv2d_98[0][0]',              
                                                                      'conv2d_100[0][0]',             
                                                                      'conv2d_101[0][0]']             
                                                                                                      
     max_pooling2d_23 (MaxPooling2D  (None, 7, 7, 832)   0           ['concatenate_15[0][0]']         
     )                                                                                                
                                                                                                      
     conv2d_103 (Conv2D)            (None, 7, 7, 160)    133280      ['max_pooling2d_23[0][0]']       
                                                                                                      
     conv2d_105 (Conv2D)            (None, 7, 7, 32)     26656       ['max_pooling2d_23[0][0]']       
                                                                                                      
     max_pooling2d_24 (MaxPooling2D  (None, 7, 7, 832)   0           ['max_pooling2d_23[0][0]']       
     )                                                                                                
                                                                                                      
     conv2d_102 (Conv2D)            (None, 7, 7, 256)    213248      ['max_pooling2d_23[0][0]']       
                                                                                                      
     conv2d_104 (Conv2D)            (None, 7, 7, 320)    461120      ['conv2d_103[0][0]']             
                                                                                                      
     conv2d_106 (Conv2D)            (None, 7, 7, 128)    102528      ['conv2d_105[0][0]']             
                                                                                                      
     conv2d_107 (Conv2D)            (None, 7, 7, 128)    106624      ['max_pooling2d_24[0][0]']       
                                                                                                      
     concatenate_16 (Concatenate)   (None, 7, 7, 832)    0           ['conv2d_102[0][0]',             
                                                                      'conv2d_104[0][0]',             
                                                                      'conv2d_106[0][0]',             
                                                                      'conv2d_107[0][0]']             
                                                                                                      
     conv2d_109 (Conv2D)            (None, 7, 7, 192)    159936      ['concatenate_16[0][0]']         
                                                                                                      
     conv2d_111 (Conv2D)            (None, 7, 7, 48)     39984       ['concatenate_16[0][0]']         
                                                                                                      
     max_pooling2d_25 (MaxPooling2D  (None, 7, 7, 832)   0           ['concatenate_16[0][0]']         
     )                                                                                                
                                                                                                      
     conv2d_108 (Conv2D)            (None, 7, 7, 384)    319872      ['concatenate_16[0][0]']         
                                                                                                      
     conv2d_110 (Conv2D)            (None, 7, 7, 384)    663936      ['conv2d_109[0][0]']             
                                                                                                      
     conv2d_112 (Conv2D)            (None, 7, 7, 128)    153728      ['conv2d_111[0][0]']             
                                                                                                      
     conv2d_113 (Conv2D)            (None, 7, 7, 128)    106624      ['max_pooling2d_25[0][0]']       
                                                                                                      
     concatenate_17 (Concatenate)   (None, 7, 7, 1024)   0           ['conv2d_108[0][0]',             
                                                                      'conv2d_110[0][0]',             
                                                                      'conv2d_112[0][0]',             
                                                                      'conv2d_113[0][0]']             
                                                                                                      
     average_pooling2d_1 (AveragePo  (None, 1, 1, 1024)  0           ['concatenate_17[0][0]']         
     oling2D)                                                                                         
                                                                                                      
     flatten_1 (Flatten)            (None, 1024)         0           ['average_pooling2d_1[0][0]']    
                                                                                                      
     dropout_1 (Dropout)            (None, 1024)         0           ['flatten_1[0][0]']              
                                                                                                      
     dense_2 (Dense)                (None, 256)          262400      ['dropout_1[0][0]']              
                                                                                                      
     dense_3 (Dense)                (None, 6)            1542        ['dense_2[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 6,238,518
    Trainable params: 6,238,006
    Non-trainable params: 512
    __________________________________________________________________________________________________


## Train the Model

```python
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```


```python
chk_path = 'model.h5'
log_dir = "checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint = keras.callbacks.ModelCheckpoint(filepath=chk_path,
                             save_best_only=True,
                             verbose=1,
                             mode='min',
                             moniter='val_loss')

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=3, 
                          verbose=1, 
                          restore_best_weights=True)
                        
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=6, 
                              verbose=1, 
                              min_delta=0.0001)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = keras.callbacks.CSVLogger('training.log')

callbacks = [checkpoint, reduce_lr, csv_logger]
```


```python
history = model.fit(training_data,validation_data=validation_data,epochs=numEPOCHS,steps_per_epoch=100,callbacks=callbacks)
```

    Epoch 1/5
    100/100 [==============================] - ETA: 0s - loss: 0.8688 - acc: 0.6691
    Epoch 1: val_loss improved from 1.68706 to 1.20432, saving model to model.h5
    100/100 [==============================] - 554s 6s/step - loss: 0.8688 - acc: 0.6691 - val_loss: 1.2043 - val_acc: 0.6250 - lr: 1.0000e-04
    Epoch 2/5
    100/100 [==============================] - ETA: 0s - loss: 0.8322 - acc: 0.6903
    Epoch 2: val_loss improved from 1.20432 to 0.89507, saving model to model.h5
    100/100 [==============================] - 548s 5s/step - loss: 0.8322 - acc: 0.6903 - val_loss: 0.8951 - val_acc: 0.6780 - lr: 1.0000e-04
    Epoch 3/5
    100/100 [==============================] - ETA: 0s - loss: 0.8000 - acc: 0.7101
    Epoch 3: val_loss improved from 0.89507 to 0.80097, saving model to model.h5
    100/100 [==============================] - 550s 6s/step - loss: 0.8000 - acc: 0.7101 - val_loss: 0.8010 - val_acc: 0.6920 - lr: 1.0000e-04
    Epoch 4/5
    100/100 [==============================] - ETA: 0s - loss: 0.7617 - acc: 0.7249
    Epoch 4: val_loss improved from 0.80097 to 0.65973, saving model to model.h5
    100/100 [==============================] - 542s 5s/step - loss: 0.7617 - acc: 0.7249 - val_loss: 0.6597 - val_acc: 0.7597 - lr: 1.0000e-04
    Epoch 5/5
    100/100 [==============================] - ETA: 0s - loss: 0.7272 - acc: 0.7361
    Epoch 5: val_loss did not improve from 0.65973
    100/100 [==============================] - 555s 6s/step - loss: 0.7272 - acc: 0.7361 - val_loss: 0.7837 - val_acc: 0.6990 - lr: 1.0000e-04



```python
model.evaluate(validation_data)
```

    47/47 [==============================] - 68s 1s/step - loss: 0.7837 - acc: 0.6990





    [0.7836715579032898, 0.6990000009536743]



## Evaluate the Model


```python
predict = model.predict(validation_data)
```

    47/47 [==============================] - 71s 2s/step



```python
def plotAccNLoss(history):
    fig , ax = plt.subplots(1,2)
    fig.set_size_inches(12,4)

    ax[0].plot(history.history['acc'])
    ax[0].plot(history.history['val_acc'])
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Training Loss vs Validation Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
```


```python
plotAccNLoss(history)
```


![png](/img/examples/vision/googleNet/googleNet_5.png)    


Our training data has smoothly worked properly, training on 15 epochs can easily touch 90% Accuracy, GoogleNet is a powerful Convolution Networks that works well with Multi Class Classification and Object Detection.

## Visualize Predicted Results


```python
labelDict = {}
for k,v in validation_data.class_indices.items():
    labelDict[v] = k
```


```python
plt.figure(figsize=(18,18))
for i in range(12):
    plt.subplot(3,4,i+1)
    toPredict = validation_data[i+(i*2)]
    imageToPredict,predictedLabel = toPredict
    
    fixImage = np.expand_dims(imageToPredict[0],axis=0)
    fixLabel = np.argmax(predictedLabel)
    target = np.argmax(model.predict(fixImage))
    plt.imshow(imageToPredict[0])
    plt.axis("off")
    
    target = labelDict[target]
    fixLabel = labelDict[fixLabel]
    plt.title(f"Pred:{target} \n Actual:{fixLabel}")
```

    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 37ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 50ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 45ms/step
    1/1 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 42ms/step



![png](/img/examples/vision/googleNet/googleNet_6.png)


## Final Thoughts

<div>
<img src="https://i.kym-cdn.com/photos/images/newsfeed/000/531/557/a88.jpg">
</div>

The main advantage of this method is a significant quality gain at a modest increase of computational requirements compared to shallower and less wide networks. GoogleNet Conv results seem to yield a solid evidence that approximating the expected optimal sparse structure by readily available dense building blocks is a viable method for improving neural networks for computer vision.

Read complete Paper: [GoogleNet Implementation](https://arxiv.org/pdf/1409.4842v1.pdf)
