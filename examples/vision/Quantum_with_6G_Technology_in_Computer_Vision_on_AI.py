
"""
##### Copyright 2022 The Emirhan BULUT.
# Quantum with 6G Technology in Computer Vision on AI
**Author:** [Emirhan BULUT](https://www.linkedin.com/in/artificialintelligencebulut/)<br>
**Date created:** 2022/10/31<br>
**Last modified:** 2022/10/31<br>
**Description:** Processed with 2nd class land use image datasets accompanied by quantum neural network in a manner compatible with 6G with quantum computer and compared with CNN (at close parameters).
"""


"""
<table class="tfo-notebook-buttons" align="left">

  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1yS5W-EsBDc6RYGYvypveGCv0QaTBCZxo?usp=sharing"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/emirhanai"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>
"""


"""
It is the algorithmic form of Quantum Neural Network designed according to 6G technology. I have been researching about Quantum Computing + 6G technology for about 2 years. In this software (in a notebook), I processed it with 2nd class land use image datasets accompanied by quantum neural network in a manner compatible with 6G with quantum computer and compared it with CNN (at close parameters). The main purpose of this software is to prove that artificial intelligence has now risen to an advanced (Quantum6) state.
"""


"""
## Download and Unzip Data
"""


!git clone https://github.com/emirhanai/Quantum-with-6G-Technology-in-Computer-Vision-on-AI.git


!unzip "/content/Quantum-with-6G-Technology-in-Computer-Vision-on-AI/datasets_for_quantum6.zip"


"""
## Setup
"""


!pip install tensorflow==2.7.0


"""
Install TensorFlow Quantum Library:
"""


!pip install tensorflow-quantum==0.7.2


"""
Now import TensorFlow, Keras and the module dependencies:
"""


import tensorflow as tf
import tensorflow_quantum as tfq
import keras
from sklearn.preprocessing import LabelEncoder

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections
import pandas as pd

from sklearn.model_selection import train_test_split
# visualization tools
%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


"""
### A. Data Preparation


"""


"""
We will pull the ImageDataGenerator function from the Keras library to convert the images extracted from the zip format into mathematical array to make them ready for processing.
"""


from keras.preprocessing.image import ImageDataGenerator
# We prepare of data
train_datagen = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False, rescale=1.0/255.0, preprocessing_function=None, data_format=None, dtype=None)

train_generator = train_datagen.flow_from_directory("/content/quantum",target_size=(4,4), batch_size=128, class_mode='categorical', interpolation="lanczos", color_mode="grayscale")


#from keras_image_generator type to numpy array
x=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
y=np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])


#Split of data to x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True)


"""
Show the first example:
"""


x_train.shape


print(y_train[0])

plt.imshow(x_train[0, :, :, 0])
plt.colorbar()


"""
### A.B. Resize the images
"""


"""
An image size of 256x256 is much too large for my quantum computer. Resize the image down to 2x2:
"""


first,two = 3,6


#x_train_resize = np.array(tf.image.resize(x_train, (first,two)))
#x_test_resize = np.array(tf.image.resize(x_test, (first,two)))
#print(y_train[0])

#plt.imshow(x_train_resize[0, :, :, 0])
#plt.colorbar()


"""
### A.C. Encode the data as quantum circuits (with qubits)

To process images using a quantum computer.
"""


THRESHOLD = 0.7

x_train_bin = np.array(x_train > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test > THRESHOLD, dtype=np.float32)


x_train_bin.shape


"""
The qubits at pixel indices with values that exceed a threshold, are rotated through an $X$ gate. And we use (3,6) qubits.
"""


def convert_to_circuit(data):
    """Encode truncated classical data into quantum datapoint."""
    values = np.ndarray.flatten(data)
    qubits = cirq.GridQubit.rect(first,two)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit
x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x) for x in x_test_bin]


"""
Here is the circuit created for the first example (circuit diagrams do not show qubits with zero gates):
"""


SVGCircuit(x_train_circ[0])


"""
Compare this circuit to the indices where the image value exceeds the threshold:
"""


bin_img = x_train_bin[-1]
indices = np.array(np.where(bin_img)).T
indices


"""
Convert these `Cirq` circuits to tensors for `tfq`:
"""


x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)


"""
## B. Quantum6 prepared by Python
"""


"""
### B.A. Build the model circuit
"""


class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


"""
Build an example circuit layer to see how it looks:
"""


demo_builder = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(first,two),
                                   readout=cirq.GridQubit(-1,-1))

circuit = cirq.Circuit()
demo_builder.add_layer(circuit, gate = cirq.XX, prefix='Emirhan_Quantum6G_AI')
SVGCircuit(circuit)


"""
Now build a quantum model, matching the data-circuit size, and include the preparation and readout operations.
"""


def create_quantum_model():
    """Create a Quantum6 AI Brain circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(first,two)   # a 3x6 grid.
    readout = cirq.GridQubit(-1, -1)         # a quantum qubits at [-1,-1]
    circuit = cirq.Circuit()
    
    # Prepare the readout quantum qubits.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "emir1")
    builder.add_layer(circuit, cirq.ZZ, "bulut1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


model_circuit, model_readout = create_quantum_model()


"""
### B.C. Build a Sequential Model for Quantum
"""


# Build the Tensorflow/Keras Sequential model.
model = keras.Sequential([
    # The input is the data-circuit (data format), encoded as a tf.string (type)
    keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(model_circuit, model_readout),
])


"""
Model Compile
"""


model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])


print(model.summary())


"""
### Quantum6 model with training in Keras
"""


EPOCHS = 35
BATCH_SIZE = 128

NUM_EXAMPLES = len(x_train_tfcirc)


"""
Model fitting
"""


quantum6_history = model.fit(x_train_tfcirc, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=0,validation_data=(x_test_tfcirc, y_test))

quantum_6_results = model.evaluate(x_test_tfcirc, y_test)


def cnn_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(2,2,input_shape=(4,4,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2))
    return model


model = cnn_model()
model.compile(loss="mse",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()


cnn_model = model.fit(x_train_bin,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=0,validation_data=(x_test_bin, y_test))

cnn_model_results = model.evaluate(x_test_bin, y_test)


"""
## C. Results on Matplotlib
"""


sns.barplot(["Quantum6 Accuracy","Convolutional Neural Network"],
            [quantum_6_results[1],cnn_model_results[1]])