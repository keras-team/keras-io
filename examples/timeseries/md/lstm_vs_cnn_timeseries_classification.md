# Timeseries Classification with LSTM and CNN Models

**Author**: [Georgios Sklavounakos] (https://github.com/gsklavounakos) 
**Date created**: 2025/06/05  
**Last modified**: 2025/06/05  
**Description**: Compare LSTM and 1D CNN models for timeseries classification using the FordA dataset.  
**Accelerator**: GPU

---

## Introduction

This example shows how to classify univariate timeseries data using two types of deep learning models:

- A Long Short-Term Memory (LSTM) model, which processes sequences using recurrent layers.
- A 1D Convolutional Neural Network (CNN) model, which extracts local patterns via convolution.

We use the FordA dataset from the [UCR/UEA archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), which consists of engine noise measurements recorded over time. The classification task is to determine whether a fault is present in the signal.

---

## What the Example Demonstrates

- How to preprocess univariate timeseries data for model input
- How to build and train two deep learning models with different architectures
- How to compare performance in terms of accuracy and training dynamics
- How to visualize model learning curves

---

## Dataset Overview

- **Name**: FordA  
- **Source**: [UCR/UEA Timeseries Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)  
- **Samples**: 3601 train, 1320 test  
- **Length**: 500 time steps  
- **Channels**: 1 (univariate)  
- **Labels**: -1 (no fault), 1 (fault) → mapped to 0 and 1  

Each timeseries is z-normalized and represents a sensor reading from an automotive engine.

---

## Results

The models are trained with the same optimizer, loss function, and validation strategy. After training, the test accuracy and loss are reported for both models. Training and validation curves are also plotted for visual comparison.

---

## References

- [UCR/UEA Timeseries Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- Wang et al., “Time Series Classification from Scratch with Deep Neural Networks”  
  [[arXiv:1611.06455](https://arxiv.org/abs/1611.06455)]
