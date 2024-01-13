# 3D image classification from CT scans

**Author:** [Hasib Zunair](https://twitter.com/hasibzunair)<br>
**Date created:** 2020/09/23<br>
**Last modified:** 2024/01/11<br>
**Description:** Train a 3D convolutional neural network to predict presence of pneumonia.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/3D_image_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/3D_image_classification.py)



---
## Introduction

This example will show the steps needed to build a 3D convolutional neural network (CNN)
to predict the presence of viral pneumonia in computer tomography (CT) scans. 2D CNNs are
commonly used to process RGB images (3 channels). A 3D CNN is simply the 3D
equivalent: it takes as input a 3D volume or a sequence of 2D frames (e.g. slices in a CT scan),
3D CNNs are a powerful model for learning representations for volumetric data.

---
## References

- [A survey on Deep Learning Advances on Different 3D DataRepresentations](https://arxiv.org/abs/1808.01462)
- [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)
- [FusionNet: 3D Object Classification Using MultipleData Representations](https://arxiv.org/abs/1607.05695)
- [Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction](https://arxiv.org/abs/2007.13224)

---
## Setup


```python
import os
import zipfile
import numpy as np
import tensorflow as tf  # for data preprocessing

import keras
from keras import layers
```

---
## Downloading the MosMedData: Chest CT Scans with COVID-19 Related Findings

In this example, we use a subset of the
[MosMedData: Chest CT Scans with COVID-19 Related Findings](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1).
This dataset consists of lung CT scans with COVID-19 related findings, as well as without such findings.

We will be using the associated radiological findings of the CT scans as labels to build
a classifier to predict presence of viral pneumonia.
Hence, the task is a binary classification problem.


```python
# Download url of normal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename = os.path.join(os.getcwd(), "CT-0.zip")
keras.utils.get_file(filename, url)

# Download url of abnormal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename = os.path.join(os.getcwd(), "CT-23.zip")
keras.utils.get_file(filename, url)

# Make a directory to store the data.
os.makedirs("MosMedData")

# Unzip data in the newly created directory.
with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")
```

<div class="k-default-codeblock">
```
Downloading data from https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip

```
</div>
    
<div class="k-default-codeblock">
```
  8192/1065471431 [..............................] - ETA: 0s


688128/1065471431 [..............................] - ETA: 1:19


```
</div>
  11436032/1065471431 [..............................] - ETA: 9s  

<div class="k-default-codeblock">
```

```
</div>
  25034752/1065471431 [..............................] - ETA: 6s

<div class="k-default-codeblock">
```

```
</div>
  39018496/1065471431 [>.............................] - ETA: 5s

<div class="k-default-codeblock">
```

```
</div>
  53084160/1065471431 [>.............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
  67502080/1065471431 [>.............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
  81313792/1065471431 [=>............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
  95436800/1065471431 [=>............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
 109477888/1065471431 [==>...........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 123379712/1065471431 [==>...........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 137478144/1065471431 [==>...........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 151199744/1065471431 [===>..........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 165158912/1065471431 [===>..........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 179183616/1065471431 [====>.........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 193331200/1065471431 [====>.........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 207282176/1065471431 [====>.........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 221298688/1065471431 [=====>........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 234782720/1065471431 [=====>........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 248766464/1065471431 [======>.......................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 262520832/1065471431 [======>.......................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 276602880/1065471431 [======>.......................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 290791424/1065471431 [=======>......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 304685056/1065471431 [=======>......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 318791680/1065471431 [=======>......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 332824576/1065471431 [========>.....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 346308608/1065471431 [========>.....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 360824832/1065471431 [=========>....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 374407168/1065471431 [=========>....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 388694016/1065471431 [=========>....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 402489344/1065471431 [==========>...................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 416653312/1065471431 [==========>...................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 430489600/1065471431 [===========>..................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 444391424/1065471431 [===========>..................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 458489856/1065471431 [===========>..................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 472530944/1065471431 [============>.................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 486891520/1065471431 [============>.................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 500875264/1065471431 [=============>................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 514850816/1065471431 [=============>................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 529022976/1065471431 [=============>................] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 543113216/1065471431 [==============>...............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 557137920/1065471431 [==============>...............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 571334656/1065471431 [===============>..............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 584941568/1065471431 [===============>..............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 599048192/1065471431 [===============>..............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 613318656/1065471431 [================>.............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 627245056/1065471431 [================>.............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 641286144/1065471431 [=================>............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 655466496/1065471431 [=================>............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 669286400/1065471431 [=================>............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 683327488/1065471431 [==================>...........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 697319424/1065471431 [==================>...........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 711278592/1065471431 [===================>..........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 725073920/1065471431 [===================>..........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 739254272/1065471431 [===================>..........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 753377280/1065471431 [====================>.........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 767221760/1065471431 [====================>.........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 781451264/1065471431 [=====================>........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 795394048/1065471431 [=====================>........] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 809525248/1065471431 [=====================>........] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 823296000/1065471431 [======================>.......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 837705728/1065471431 [======================>.......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 851492864/1065471431 [======================>.......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 865689600/1065471431 [=======================>......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 879853568/1065471431 [=======================>......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 893665280/1065471431 [========================>.....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 908017664/1065471431 [========================>.....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 922288128/1065471431 [========================>.....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 936108032/1065471431 [=========================>....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 950190080/1065471431 [=========================>....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 964296704/1065471431 [==========================>...] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 978272256/1065471431 [==========================>...] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 992419840/1065471431 [==========================>...] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1006329856/1065471431 [===========================>..] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1020444672/1065471431 [===========================>..] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1034338304/1065471431 [============================>.] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1048510464/1065471431 [============================>.] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1062674432/1065471431 [============================>.] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1065471431/1065471431 [==============================] - 4s 0us/step


<div class="k-default-codeblock">
```
Downloading data from https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip

```
</div>
    
<div class="k-default-codeblock">
```
  8192/1045162547 [..............................] - ETA: 0s


```
</div>
   1032192/1045162547 [..............................] - ETA: 53s

<div class="k-default-codeblock">
```

```
</div>
  13041664/1045162547 [..............................] - ETA: 8s 

<div class="k-default-codeblock">
```

```
</div>
  26836992/1045162547 [..............................] - ETA: 5s

<div class="k-default-codeblock">
```

```
</div>
  41205760/1045162547 [>.............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
  55418880/1045162547 [>.............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
  69525504/1045162547 [>.............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
  83951616/1045162547 [=>............................] - ETA: 4s

<div class="k-default-codeblock">
```

```
</div>
  98058240/1045162547 [=>............................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 112181248/1045162547 [==>...........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 126640128/1045162547 [==>...........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 140484608/1045162547 [===>..........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 154542080/1045162547 [===>..........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 168722432/1045162547 [===>..........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 182976512/1045162547 [====>.........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 197001216/1045162547 [====>.........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 211140608/1045162547 [=====>........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 225280000/1045162547 [=====>........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 239091712/1045162547 [=====>........................] - ETA: 3s

<div class="k-default-codeblock">
```

```
</div>
 253616128/1045162547 [======>.......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 267698176/1045162547 [======>.......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 281632768/1045162547 [=======>......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 296083456/1045162547 [=======>......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 310181888/1045162547 [=======>......................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 324288512/1045162547 [========>.....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 338362368/1045162547 [========>.....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 352305152/1045162547 [=========>....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 366755840/1045162547 [=========>....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 380698624/1045162547 [=========>....................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 395034624/1045162547 [==========>...................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 409206784/1045162547 [==========>...................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 423329792/1045162547 [===========>..................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 437501952/1045162547 [===========>..................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 451854336/1045162547 [===========>..................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 465960960/1045162547 [============>.................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 480280576/1045162547 [============>.................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 494116864/1045162547 [=============>................] - ETA: 2s

<div class="k-default-codeblock">
```

```
</div>
 508182528/1045162547 [=============>................] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 522485760/1045162547 [=============>................] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 536961024/1045162547 [==============>...............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 550993920/1045162547 [==============>...............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 565010432/1045162547 [===============>..............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 579125248/1045162547 [===============>..............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 593182720/1045162547 [================>.............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 607207424/1045162547 [================>.............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 621510656/1045162547 [================>.............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 635863040/1045162547 [=================>............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 649879552/1045162547 [=================>............] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 664199168/1045162547 [==================>...........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 678363136/1045162547 [==================>...........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 692469760/1045162547 [==================>...........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 706560000/1045162547 [===================>..........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 720855040/1045162547 [===================>..........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 735051776/1045162547 [====================>.........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 749322240/1045162547 [====================>.........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 763543552/1045162547 [====================>.........] - ETA: 1s

<div class="k-default-codeblock">
```

```
</div>
 777723904/1045162547 [=====================>........] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 791773184/1045162547 [=====================>........] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 806125568/1045162547 [======================>.......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 820379648/1045162547 [======================>.......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 834314240/1045162547 [======================>.......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 848412672/1045162547 [=======================>......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 862887936/1045162547 [=======================>......] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 876806144/1045162547 [========================>.....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 891256832/1045162547 [========================>.....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 904970240/1045162547 [========================>.....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 919240704/1045162547 [=========================>....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 933265408/1045162547 [=========================>....] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 947625984/1045162547 [==========================>...] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 961527808/1045162547 [==========================>...] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 975880192/1045162547 [===========================>..] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
 990117888/1045162547 [===========================>..] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1004355584/1045162547 [===========================>..] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1018445824/1045162547 [============================>.] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1032798208/1045162547 [============================>.] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1045162547/1045162547 [==============================] - 4s 0us/step


---
## Loading data and preprocessing

The files are provided in Nifti format with the extension .nii. To read the
scans, we use the `nibabel` package.
You can install the package via `pip install nibabel`. CT scans store raw voxel
intensity in Hounsfield units (HU). They range from -1024 to above 2000 in this dataset.
Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold
between -1000 and 400 is commonly used to normalize CT scans.

To process the data, we do the following:

* We first rotate the volumes by 90 degrees, so the orientation is fixed
* We scale the HU values to be between 0 and 1.
* We resize width, height and depth.

Here we define several helper functions to process the data. These functions
will be used when building training and validation datasets.


```python

import nibabel as nib

from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

```

Let's read the paths of the CT scans from the class directories.


```python
# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-23", x)
    for x in os.listdir("MosMedData/CT-23")
]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

```

<div class="k-default-codeblock">
```
CT scans with normal lung tissue: 100
CT scans with abnormal lung tissue: 100

```
</div>
---
## Build train and validation datasets
Read the scans from the class directories and assign labels. Downsample the scans to have
shape of 128x128x64. Rescale the raw HU values to the range 0 to 1.
Lastly, split the dataset into train and validation subsets.


```python
# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)
```

<div class="k-default-codeblock">
```
Number of samples in train and validation are 140 and 60.

```
</div>
---
## Data augmentation

The CT scans also augmented by rotating at random angles during training. Since
the data is stored in rank-3 tensors of shape `(samples, height, width, depth)`,
we add a dimension of size 1 at axis 4 to be able to perform 3D convolutions on
the data. The new shape is thus `(samples, height, width, depth, 1)`. There are
different kinds of preprocessing and augmentation techniques out there,
this example shows a few simple ones to get started.


```python
import random

from scipy import ndimage


def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

```

While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates volume at different angles. Note that both
training and validation data are already rescaled to have values between 0 and 1.


```python
# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
```

Visualize an augmented CT scan.


```python
import matplotlib.pyplot as plt

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")

```

<div class="k-default-codeblock">
```
Dimension of the CT scan is: (128, 128, 64, 1)

<matplotlib.image.AxesImage at 0x7fe26b253c90>

```
</div>
    
![png](/img/examples/vision/3D_image_classification/3D_image_classification_17_2.png)
    


Since a CT scan has many slices, let's visualize a montage of the slices.


```python

def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])
```


    
![png](/img/examples/vision/3D_image_classification/3D_image_classification_19_0.png)
    


---
## Define a 3D convolutional neural network

To make the model easier to understand, we structure it into blocks.
The architecture of the 3D CNN used in this example
is based on [this paper](https://arxiv.org/abs/2007.13224).


```python

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()
```

<div class="k-default-codeblock">
```
Model: "3dcnn"

_________________________________________________________________

 Layer (type)                Output Shape              Param #   

=================================================================

 input_1 (InputLayer)        [(None, 128, 128, 64, 1   0         

                             )]                                  

                                                                 

 conv3d (Conv3D)             (None, 126, 126, 62, 64   1792      

                             )                                   

                                                                 

 max_pooling3d (MaxPooling3  (None, 63, 63, 31, 64)    0         

 D)                                                              

                                                                 

 batch_normalization (Batch  (None, 63, 63, 31, 64)    256       

 Normalization)                                                  

                                                                 

 conv3d_1 (Conv3D)           (None, 61, 61, 29, 64)    110656    

                                                                 

 max_pooling3d_1 (MaxPoolin  (None, 30, 30, 14, 64)    0         

 g3D)                                                            

                                                                 

 batch_normalization_1 (Bat  (None, 30, 30, 14, 64)    256       

 chNormalization)                                                

                                                                 

 conv3d_2 (Conv3D)           (None, 28, 28, 12, 128)   221312    

                                                                 

 max_pooling3d_2 (MaxPoolin  (None, 14, 14, 6, 128)    0         

 g3D)                                                            

                                                                 

 batch_normalization_2 (Bat  (None, 14, 14, 6, 128)    512       

 chNormalization)                                                

                                                                 

 conv3d_3 (Conv3D)           (None, 12, 12, 4, 256)    884992    

                                                                 

 max_pooling3d_3 (MaxPoolin  (None, 6, 6, 2, 256)      0         

 g3D)                                                            

                                                                 

 batch_normalization_3 (Bat  (None, 6, 6, 2, 256)      1024      

 chNormalization)                                                

                                                                 

 global_average_pooling3d (  (None, 256)               0         

 GlobalAveragePooling3D)                                         

                                                                 

 dense (Dense)               (None, 512)               131584    

                                                                 

 dropout (Dropout)           (None, 512)               0         

                                                                 

 dense_1 (Dense)             (None, 1)                 513       

                                                                 

=================================================================

Total params: 1352897 (5.16 MB)

Trainable params: 1351873 (5.16 MB)

Non-trainable params: 1024 (4.00 KB)

_________________________________________________________________

```
</div>
---
## Train model


```python
# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.keras", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
```

<div class="k-default-codeblock">
```
Epoch 1/100

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1705140149.343713   14302 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

70/70 - 42s - loss: 0.6334 - acc: 0.6429 - val_loss: 1.1152 - val_acc: 0.5000 - 42s/epoch - 597ms/step

Epoch 2/100

70/70 - 22s - loss: 0.6596 - acc: 0.5714 - val_loss: 1.4869 - val_acc: 0.5000 - 22s/epoch - 309ms/step

Epoch 3/100

70/70 - 21s - loss: 0.6798 - acc: 0.5929 - val_loss: 1.5380 - val_acc: 0.5000 - 21s/epoch - 304ms/step

Epoch 4/100

70/70 - 21s - loss: 0.5876 - acc: 0.6857 - val_loss: 3.2347 - val_acc: 0.5000 - 21s/epoch - 305ms/step

Epoch 5/100

70/70 - 21s - loss: 0.6484 - acc: 0.6143 - val_loss: 2.0283 - val_acc: 0.5000 - 21s/epoch - 306ms/step

Epoch 6/100

70/70 - 22s - loss: 0.6328 - acc: 0.6286 - val_loss: 3.1393 - val_acc: 0.5000 - 22s/epoch - 311ms/step

Epoch 7/100

70/70 - 22s - loss: 0.6397 - acc: 0.6571 - val_loss: 0.8956 - val_acc: 0.5000 - 22s/epoch - 313ms/step

Epoch 8/100

70/70 - 22s - loss: 0.6183 - acc: 0.6286 - val_loss: 1.2956 - val_acc: 0.5000 - 22s/epoch - 308ms/step

Epoch 9/100

70/70 - 22s - loss: 0.6087 - acc: 0.6643 - val_loss: 1.4353 - val_acc: 0.5000 - 22s/epoch - 307ms/step

Epoch 10/100

70/70 - 21s - loss: 0.5874 - acc: 0.6857 - val_loss: 0.7793 - val_acc: 0.6000 - 21s/epoch - 306ms/step

Epoch 11/100

70/70 - 22s - loss: 0.5808 - acc: 0.6571 - val_loss: 0.6733 - val_acc: 0.6333 - 22s/epoch - 308ms/step

Epoch 12/100

70/70 - 22s - loss: 0.5657 - acc: 0.7286 - val_loss: 0.5757 - val_acc: 0.7000 - 22s/epoch - 309ms/step

Epoch 13/100

70/70 - 21s - loss: 0.6134 - acc: 0.6571 - val_loss: 0.5621 - val_acc: 0.6333 - 21s/epoch - 306ms/step

Epoch 14/100

70/70 - 21s - loss: 0.5701 - acc: 0.6786 - val_loss: 0.6454 - val_acc: 0.7333 - 21s/epoch - 302ms/step

Epoch 15/100

70/70 - 21s - loss: 0.5410 - acc: 0.7429 - val_loss: 0.5566 - val_acc: 0.7167 - 21s/epoch - 306ms/step

Epoch 16/100

70/70 - 22s - loss: 0.5492 - acc: 0.6929 - val_loss: 0.6223 - val_acc: 0.6167 - 22s/epoch - 308ms/step

Epoch 17/100

70/70 - 21s - loss: 0.5476 - acc: 0.7286 - val_loss: 0.8013 - val_acc: 0.5667 - 21s/epoch - 304ms/step

Epoch 18/100

70/70 - 21s - loss: 0.5298 - acc: 0.7214 - val_loss: 1.1550 - val_acc: 0.5500 - 21s/epoch - 306ms/step

Epoch 19/100

70/70 - 22s - loss: 0.5063 - acc: 0.7857 - val_loss: 1.3667 - val_acc: 0.5000 - 22s/epoch - 309ms/step

Epoch 20/100

70/70 - 21s - loss: 0.5425 - acc: 0.7143 - val_loss: 1.2658 - val_acc: 0.5667 - 21s/epoch - 302ms/step

Epoch 21/100

70/70 - 21s - loss: 0.5109 - acc: 0.7429 - val_loss: 2.2053 - val_acc: 0.5167 - 21s/epoch - 307ms/step

Epoch 22/100

70/70 - 21s - loss: 0.5045 - acc: 0.7429 - val_loss: 0.7835 - val_acc: 0.6667 - 21s/epoch - 306ms/step

Epoch 23/100

70/70 - 22s - loss: 0.4948 - acc: 0.7786 - val_loss: 0.5462 - val_acc: 0.7500 - 22s/epoch - 309ms/step

Epoch 24/100

70/70 - 21s - loss: 0.4952 - acc: 0.7786 - val_loss: 0.5627 - val_acc: 0.7333 - 21s/epoch - 304ms/step

Epoch 25/100

70/70 - 21s - loss: 0.5076 - acc: 0.7214 - val_loss: 1.0972 - val_acc: 0.5333 - 21s/epoch - 303ms/step

Epoch 26/100

70/70 - 21s - loss: 0.4851 - acc: 0.7786 - val_loss: 0.7425 - val_acc: 0.5833 - 21s/epoch - 300ms/step

Epoch 27/100

70/70 - 22s - loss: 0.4107 - acc: 0.8429 - val_loss: 0.6109 - val_acc: 0.7000 - 22s/epoch - 307ms/step

Epoch 28/100

70/70 - 21s - loss: 0.4354 - acc: 0.8214 - val_loss: 0.6934 - val_acc: 0.6667 - 21s/epoch - 305ms/step

Epoch 29/100

70/70 - 21s - loss: 0.4223 - acc: 0.7857 - val_loss: 0.6203 - val_acc: 0.7500 - 21s/epoch - 306ms/step

Epoch 30/100

70/70 - 21s - loss: 0.4083 - acc: 0.8214 - val_loss: 0.6310 - val_acc: 0.7000 - 21s/epoch - 306ms/step

Epoch 31/100

70/70 - 21s - loss: 0.4440 - acc: 0.7929 - val_loss: 0.7576 - val_acc: 0.6833 - 21s/epoch - 307ms/step

Epoch 32/100

70/70 - 21s - loss: 0.4743 - acc: 0.7643 - val_loss: 0.9090 - val_acc: 0.6667 - 21s/epoch - 306ms/step

Epoch 33/100

70/70 - 21s - loss: 0.4072 - acc: 0.8429 - val_loss: 0.5330 - val_acc: 0.7667 - 21s/epoch - 303ms/step

Epoch 34/100

70/70 - 21s - loss: 0.3723 - acc: 0.8500 - val_loss: 1.2858 - val_acc: 0.5833 - 21s/epoch - 305ms/step

Epoch 35/100

70/70 - 21s - loss: 0.3814 - acc: 0.8214 - val_loss: 1.7184 - val_acc: 0.5667 - 21s/epoch - 304ms/step

Epoch 36/100

70/70 - 21s - loss: 0.4642 - acc: 0.7571 - val_loss: 1.1018 - val_acc: 0.6333 - 21s/epoch - 304ms/step

Epoch 37/100

70/70 - 21s - loss: 0.3560 - acc: 0.8571 - val_loss: 0.7784 - val_acc: 0.6167 - 21s/epoch - 301ms/step

Epoch 38/100

70/70 - 21s - loss: 0.4250 - acc: 0.7643 - val_loss: 0.7526 - val_acc: 0.7167 - 21s/epoch - 302ms/step

Epoch 39/100

70/70 - 21s - loss: 0.3707 - acc: 0.8429 - val_loss: 1.1117 - val_acc: 0.6667 - 21s/epoch - 303ms/step

Epoch 40/100

70/70 - 21s - loss: 0.4270 - acc: 0.8000 - val_loss: 0.5893 - val_acc: 0.7333 - 21s/epoch - 303ms/step

Epoch 41/100

70/70 - 21s - loss: 0.3551 - acc: 0.8429 - val_loss: 0.6869 - val_acc: 0.7167 - 21s/epoch - 303ms/step

Epoch 42/100

70/70 - 21s - loss: 0.3708 - acc: 0.8071 - val_loss: 0.6271 - val_acc: 0.7500 - 21s/epoch - 304ms/step

Epoch 43/100

70/70 - 21s - loss: 0.3398 - acc: 0.8071 - val_loss: 2.0827 - val_acc: 0.5167 - 21s/epoch - 304ms/step

Epoch 44/100

70/70 - 21s - loss: 0.3634 - acc: 0.8500 - val_loss: 1.5756 - val_acc: 0.5500 - 21s/epoch - 299ms/step

Epoch 45/100

70/70 - 21s - loss: 0.2861 - acc: 0.8786 - val_loss: 0.8812 - val_acc: 0.6500 - 21s/epoch - 303ms/step

Epoch 46/100

70/70 - 21s - loss: 0.3805 - acc: 0.8071 - val_loss: 0.6249 - val_acc: 0.7667 - 21s/epoch - 304ms/step

Epoch 47/100

70/70 - 21s - loss: 0.3003 - acc: 0.9000 - val_loss: 0.6349 - val_acc: 0.8000 - 21s/epoch - 302ms/step

Epoch 48/100

70/70 - 21s - loss: 0.3319 - acc: 0.8286 - val_loss: 0.6876 - val_acc: 0.6667 - 21s/epoch - 300ms/step

Epoch 49/100

70/70 - 21s - loss: 0.3166 - acc: 0.8429 - val_loss: 0.7354 - val_acc: 0.7333 - 21s/epoch - 302ms/step

Epoch 50/100

70/70 - 21s - loss: 0.3164 - acc: 0.8500 - val_loss: 0.8156 - val_acc: 0.7000 - 21s/epoch - 302ms/step

Epoch 51/100

70/70 - 21s - loss: 0.2540 - acc: 0.9071 - val_loss: 0.9337 - val_acc: 0.6667 - 21s/epoch - 301ms/step

Epoch 52/100

70/70 - 21s - loss: 0.3080 - acc: 0.8571 - val_loss: 0.6895 - val_acc: 0.7333 - 21s/epoch - 302ms/step

Epoch 53/100

70/70 - 21s - loss: 0.3115 - acc: 0.8286 - val_loss: 0.6846 - val_acc: 0.7167 - 21s/epoch - 304ms/step

Epoch 54/100

70/70 - 21s - loss: 0.2242 - acc: 0.9143 - val_loss: 3.2433 - val_acc: 0.5333 - 21s/epoch - 303ms/step

Epoch 55/100

70/70 - 21s - loss: 0.2799 - acc: 0.8571 - val_loss: 0.8969 - val_acc: 0.6833 - 21s/epoch - 302ms/step

Epoch 56/100

70/70 - 21s - loss: 0.2810 - acc: 0.8571 - val_loss: 1.0162 - val_acc: 0.6833 - 21s/epoch - 300ms/step

Epoch 57/100

70/70 - 21s - loss: 0.2742 - acc: 0.8786 - val_loss: 2.0031 - val_acc: 0.5333 - 21s/epoch - 305ms/step

Epoch 58/100

70/70 - 21s - loss: 0.2867 - acc: 0.8929 - val_loss: 0.8081 - val_acc: 0.6833 - 21s/epoch - 302ms/step

Epoch 59/100

70/70 - 21s - loss: 0.1943 - acc: 0.9214 - val_loss: 1.3980 - val_acc: 0.5667 - 21s/epoch - 304ms/step

Epoch 60/100

70/70 - 21s - loss: 0.2515 - acc: 0.8714 - val_loss: 0.9441 - val_acc: 0.6167 - 21s/epoch - 304ms/step

Epoch 61/100

70/70 - 21s - loss: 0.2389 - acc: 0.9000 - val_loss: 0.8526 - val_acc: 0.7167 - 21s/epoch - 302ms/step

Epoch 62/100

70/70 - 21s - loss: 0.2582 - acc: 0.8929 - val_loss: 0.6839 - val_acc: 0.7500 - 21s/epoch - 303ms/step

<keras.src.callbacks.History at 0x7fe2689d8b50>

```
</div>
It is important to note that the number of samples is very small (only 200) and we don't
specify a random seed. As such, you can expect significant variance in the results. The full dataset
which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
performance is observed in both cases.

---
## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.


```python
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
```


    
![png](/img/examples/vision/3D_image_classification/3D_image_classification_26_0.png)
    


---
## Make predictions on a single CT scan


```python
# Load best weights.
model.load_weights("3d_image_classification.keras")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )
```

    
1/1 [==============================] - ETA: 0s

<div class="k-default-codeblock">
```

```
</div>
1/1 [==============================] - 1s 629ms/step


<div class="k-default-codeblock">
```
This model is 44.90 percent confident that CT scan is normal
This model is 55.10 percent confident that CT scan is abnormal

```
</div>