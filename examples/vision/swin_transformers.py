"""
Title: CIFAR-10 Image Classificatiion with Swin Transformers
Author: [Rishit Dagli](https://twitter.com/rishit_dagli)
Date created: 2021/09/08
Last modified: 2021/09/08
Description: (one-line text description)
"""
"""
This example implements [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
paper by Liu et al. for image classification, and demonstrates it on the 
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

Swin Transformer (**S**hifted **Wi**ndow) capably serves as a general-purpose backbone 
for computer vision. Swin Transformer is a hierarchical Transformer whose 
representation is computed with shifted windows. The shifted windowing scheme 
brings greater efficiency by limiting self-attention computation to 
non-overlapping local windows while also allowing for cross-window connection. 
This architecture also has the flexibility to model at various scales and has 
linear computational complexity with respect to image size.

This example requires TensorFlow 2.5 or higher, as well as 
[Matplotlib](https://matplotlib.org/), which can be installed using the 
following command:
"""

"""shell
pip install -U matplotlib
"""