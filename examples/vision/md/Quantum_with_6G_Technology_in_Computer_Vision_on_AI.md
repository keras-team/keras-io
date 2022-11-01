##### Copyright 2022 The Emirhan BULUT.
# Quantum with 6G Technology in Computer Vision on AI
**Author:** [Emirhan BULUT](https://www.linkedin.com/in/artificialintelligencebulut/)<br>
**Date created:** 2022/10/31<br>
**Last modified:** 2022/10/31<br>
**Description:** Processed with 2nd class land use image datasets accompanied by quantum neural network in a manner compatible with 6G with quantum computer and compared with CNN (at close parameters).

<table class="tfo-notebook-buttons" align="left">

  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1yS5W-EsBDc6RYGYvypveGCv0QaTBCZxo?usp=sharing"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/emirhanai"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

It is the algorithmic form of Quantum Neural Network designed according to 6G technology. I have been researching about Quantum Computing + 6G technology for about 2 years. In this software (in a notebook), I processed it with 2nd class land use image datasets accompanied by quantum neural network in a manner compatible with 6G with quantum computer and compared it with CNN (at close parameters). The main purpose of this software is to prove that artificial intelligence has now risen to an advanced (Quantum6) state.

## Download and Unzip Data


```
!git clone https://github.com/emirhanai/Quantum-with-6G-Technology-in-Computer-Vision-on-AI.git
```

    Cloning into 'Quantum-with-6G-Technology-in-Computer-Vision-on-AI'...
    remote: Enumerating objects: 8, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 8 (delta 1), reused 0 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (8/8), done.



```
!unzip "/content/Quantum-with-6G-Technology-in-Computer-Vision-on-AI/datasets_for_quantum6.zip"
```

    Archive:  /content/Quantum-with-6G-Technology-in-Computer-Vision-on-AI/datasets_for_quantum6.zip
       creating: quantum/
       creating: quantum/first class/
      inflating: quantum/first class/first class - Copy (2).png  
      inflating: quantum/first class/first class - Copy (3).png  
      inflating: quantum/first class/first class - Copy (4).png  
      inflating: quantum/first class/first class - Copy (5).png  
      inflating: quantum/first class/first class - Copy (6).png  
      inflating: quantum/first class/first class - Copy (7).png  
      inflating: quantum/first class/first class - Copy (8).png  
      inflating: quantum/first class/first class - Copy (9).png  
      inflating: quantum/first class/first class - Copy.png  
      inflating: quantum/first class/first class.png  
       creating: quantum/second class/
      inflating: quantum/second class/second class - Copy (2).png  
      inflating: quantum/second class/second class - Copy (3).png  
      inflating: quantum/second class/second class - Copy (4).png  
      inflating: quantum/second class/second class - Copy (5).png  
      inflating: quantum/second class/second class - Copy (6).png  
      inflating: quantum/second class/second class - Copy (7).png  
      inflating: quantum/second class/second class - Copy (8).png  
      inflating: quantum/second class/second class - Copy (9).png  
      inflating: quantum/second class/second class - Copy.png  
      inflating: quantum/second class/second class.png  


## Setup


```
!pip install tensorflow==2.7.0
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting tensorflow==2.7.0
      Downloading https://us-python.pkg.dev/colab-wheels/public/tensorflow/tensorflow-2.7.0%2Bzzzcolab20220506150900-cp37-cp37m-linux_x86_64.whl (665.5 MB)
    [K     |████████████████████████████████| 665.5 MB 23 kB/s 
    [?25hRequirement already satisfied: protobuf>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (3.17.3)
    Requirement already satisfied: flatbuffers<3.0,>=1.12 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.12)
    Requirement already satisfied: libclang>=9.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (14.0.6)
    Requirement already satisfied: gast<0.5.0,>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.4.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (4.1.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (3.3.0)
    Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.15.0)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.27.0)
    Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.14.1)
    Requirement already satisfied: tensorboard~=2.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (2.9.1)
    Requirement already satisfied: absl-py>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.3.0)
    Requirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.21.6)
    Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.6.3)
    Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (3.1.0)
    Requirement already satisfied: wheel<1.0,>=0.32.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.37.1)
    Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.2.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (2.0.1)
    Collecting keras<2.8,>=2.7.0rc0
      Downloading keras-2.7.0-py2.py3-none-any.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 32.2 MB/s 
    [?25hRequirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.50.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.1.2)
    Collecting tensorflow-estimator<2.8,~=2.7.0rc0
      Downloading tensorflow_estimator-2.7.0-py2.py3-none-any.whl (463 kB)
    [K     |████████████████████████████████| 463 kB 74.3 MB/s 
    [?25hRequirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow==2.7.0) (1.5.2)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (0.4.6)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (1.35.0)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (2.23.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (1.8.1)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (3.4.1)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (0.6.1)
    Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (1.0.1)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (57.4.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (4.9)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (4.2.4)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow==2.7.0) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard~=2.6->tensorflow==2.7.0) (4.13.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard~=2.6->tensorflow==2.7.0) (3.10.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (0.4.8)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (2022.9.24)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (1.24.3)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow==2.7.0) (3.2.2)
    Installing collected packages: tensorflow-estimator, keras, tensorflow
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.9.0
        Uninstalling tensorflow-estimator-2.9.0:
          Successfully uninstalled tensorflow-estimator-2.9.0
      Attempting uninstall: keras
        Found existing installation: keras 2.9.0
        Uninstalling keras-2.9.0:
          Successfully uninstalled keras-2.9.0
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 2.9.2
        Uninstalling tensorflow-2.9.2:
          Successfully uninstalled tensorflow-2.9.2
    Successfully installed keras-2.7.0 tensorflow-2.7.0+zzzcolab20220506150900 tensorflow-estimator-2.7.0


Install TensorFlow Quantum Library:


```
!pip install tensorflow-quantum==0.7.2
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting tensorflow-quantum==0.7.2
      Downloading tensorflow_quantum-0.7.2-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (10.5 MB)
    [K     |████████████████████████████████| 10.5 MB 23.5 MB/s 
    [?25hRequirement already satisfied: protobuf==3.17.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow-quantum==0.7.2) (3.17.3)
    Collecting cirq-google>=0.13.1
      Downloading cirq_google-1.0.0-py3-none-any.whl (576 kB)
    [K     |████████████████████████████████| 576 kB 67.3 MB/s 
    [?25hCollecting cirq-core==0.13.1
      Downloading cirq_core-0.13.1-py3-none-any.whl (1.6 MB)
    [K     |████████████████████████████████| 1.6 MB 53.7 MB/s 
    [?25hCollecting google-api-core==1.21.0
      Downloading google_api_core-1.21.0-py2.py3-none-any.whl (90 kB)
    [K     |████████████████████████████████| 90 kB 10.3 MB/s 
    [?25hCollecting googleapis-common-protos==1.52.0
      Downloading googleapis_common_protos-1.52.0-py2.py3-none-any.whl (100 kB)
    [K     |████████████████████████████████| 100 kB 6.9 MB/s 
    [?25hCollecting sympy==1.8
      Downloading sympy-1.8-py3-none-any.whl (6.1 MB)
    [K     |████████████████████████████████| 6.1 MB 22.9 MB/s 
    [?25hCollecting google-auth==1.18.0
      Downloading google_auth-1.18.0-py2.py3-none-any.whl (90 kB)
    [K     |████████████████████████████████| 90 kB 8.9 MB/s 
    [?25hRequirement already satisfied: networkx~=2.4 in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (2.6.3)
    Collecting duet~=0.2.0
      Downloading duet-0.2.7-py3-none-any.whl (28 kB)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (4.1.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (1.3.5)
    Requirement already satisfied: matplotlib~=3.0 in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (3.2.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (1.7.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (4.64.1)
    Requirement already satisfied: sortedcontainers~=2.0 in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (2.4.0)
    Requirement already satisfied: numpy~=1.16 in /usr/local/lib/python3.7/dist-packages (from cirq-core==0.13.1->tensorflow-quantum==0.7.2) (1.21.6)
    Requirement already satisfied: setuptools>=34.0.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core==1.21.0->tensorflow-quantum==0.7.2) (57.4.0)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core==1.21.0->tensorflow-quantum==0.7.2) (1.15.0)
    Requirement already satisfied: pytz in /usr/local/lib/python3.7/dist-packages (from google-api-core==1.21.0->tensorflow-quantum==0.7.2) (2022.5)
    Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core==1.21.0->tensorflow-quantum==0.7.2) (2.23.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth==1.18.0->tensorflow-quantum==0.7.2) (0.2.8)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth==1.18.0->tensorflow-quantum==0.7.2) (4.2.4)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth==1.18.0->tensorflow-quantum==0.7.2) (4.9)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.7/dist-packages (from sympy==1.8->tensorflow-quantum==0.7.2) (1.2.1)
    Collecting cirq-google>=0.13.1
      Downloading cirq_google-0.15.0-py3-none-any.whl (641 kB)
    [K     |████████████████████████████████| 641 kB 56.1 MB/s 
    [?25hRequirement already satisfied: google-api-core[grpc]<2.0.0dev,>=1.14.0 in /usr/local/lib/python3.7/dist-packages (from cirq-google>=0.13.1->tensorflow-quantum==0.7.2) (1.31.6)
      Downloading cirq_google-0.14.1-py3-none-any.whl (541 kB)
    [K     |████████████████████████████████| 541 kB 55.6 MB/s 
    [?25h  Downloading cirq_google-0.14.0-py3-none-any.whl (541 kB)
    [K     |████████████████████████████████| 541 kB 72.2 MB/s 
    [?25h  Downloading cirq_google-0.13.1-py3-none-any.whl (437 kB)
    [K     |████████████████████████████████| 437 kB 62.1 MB/s 
    [?25hCollecting typing-extensions
      Downloading typing_extensions-3.10.0.0-py3-none-any.whl (26 kB)
    Collecting google-api-core[grpc]<2.0.0dev,>=1.14.0
      Downloading google_api_core-1.33.2-py3-none-any.whl (115 kB)
    [K     |████████████████████████████████| 115 kB 62.3 MB/s 
    [?25h  Downloading google_api_core-1.33.1-py3-none-any.whl (115 kB)
    [K     |████████████████████████████████| 115 kB 54.3 MB/s 
    [?25h  Downloading google_api_core-1.33.0-py3-none-any.whl (115 kB)
    [K     |████████████████████████████████| 115 kB 60.4 MB/s 
    [?25h  Downloading google_api_core-1.32.0-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.5 MB/s 
    [?25h  Downloading google_api_core-1.31.5-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.7 MB/s 
    [?25h  Downloading google_api_core-1.31.4-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.8 MB/s 
    [?25h  Downloading google_api_core-1.31.3-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 889 kB/s 
    [?25h  Downloading google_api_core-1.31.2-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.7 MB/s 
    [?25h  Downloading google_api_core-1.31.1-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.4 MB/s 
    [?25h  Downloading google_api_core-1.31.0-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 769 kB/s 
    [?25h  Downloading google_api_core-1.30.0-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 782 kB/s 
    [?25h  Downloading google_api_core-1.29.0-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.3 MB/s 
    [?25h  Downloading google_api_core-1.28.0-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 1.2 MB/s 
    [?25h  Downloading google_api_core-1.27.0-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 522 kB/s 
    [?25h  Downloading google_api_core-1.26.3-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.2 MB/s 
    [?25h  Downloading google_api_core-1.26.2-py2.py3-none-any.whl (93 kB)
    [K     |████████████████████████████████| 93 kB 1.4 MB/s 
    [?25h  Downloading google_api_core-1.26.1-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 491 kB/s 
    [?25h  Downloading google_api_core-1.26.0-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 748 kB/s 
    [?25h  Downloading google_api_core-1.25.1-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 279 kB/s 
    [?25h  Downloading google_api_core-1.25.0-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 189 kB/s 
    [?25h  Downloading google_api_core-1.24.1-py2.py3-none-any.whl (92 kB)
    [K     |████████████████████████████████| 92 kB 4.3 MB/s 
    [?25h  Downloading google_api_core-1.24.0-py2.py3-none-any.whl (91 kB)
    [K     |████████████████████████████████| 91 kB 12.1 MB/s 
    [?25h  Downloading google_api_core-1.23.0-py2.py3-none-any.whl (91 kB)
    [K     |████████████████████████████████| 91 kB 9.9 MB/s 
    [?25h  Downloading google_api_core-1.22.4-py2.py3-none-any.whl (91 kB)
    [K     |████████████████████████████████| 91 kB 3.4 MB/s 
    [?25h  Downloading google_api_core-1.22.3-py2.py3-none-any.whl (91 kB)
    [K     |████████████████████████████████| 91 kB 10.4 MB/s 
    [?25h  Downloading google_api_core-1.22.2-py2.py3-none-any.whl (91 kB)
    [K     |████████████████████████████████| 91 kB 10.8 MB/s 
    [?25h  Downloading google_api_core-1.22.1-py2.py3-none-any.whl (91 kB)
    [K     |████████████████████████████████| 91 kB 9.7 MB/s 
    [?25h  Downloading google_api_core-1.22.0-py2.py3-none-any.whl (91 kB)
    [K     |████████████████████████████████| 91 kB 8.8 MB/s 
    [?25hRequirement already satisfied: grpcio<2.0dev,>=1.29.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core==1.21.0->tensorflow-quantum==0.7.2) (1.50.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib~=3.0->cirq-core==0.13.1->tensorflow-quantum==0.7.2) (1.4.4)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib~=3.0->cirq-core==0.13.1->tensorflow-quantum==0.7.2) (2.8.2)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib~=3.0->cirq-core==0.13.1->tensorflow-quantum==0.7.2) (3.0.9)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib~=3.0->cirq-core==0.13.1->tensorflow-quantum==0.7.2) (0.11.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth==1.18.0->tensorflow-quantum==0.7.2) (0.4.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core==1.21.0->tensorflow-quantum==0.7.2) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core==1.21.0->tensorflow-quantum==0.7.2) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core==1.21.0->tensorflow-quantum==0.7.2) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core==1.21.0->tensorflow-quantum==0.7.2) (3.0.4)
    Installing collected packages: typing-extensions, googleapis-common-protos, google-auth, sympy, google-api-core, duet, cirq-core, cirq-google, tensorflow-quantum
      Attempting uninstall: typing-extensions
        Found existing installation: typing-extensions 4.1.1
        Uninstalling typing-extensions-4.1.1:
          Successfully uninstalled typing-extensions-4.1.1
      Attempting uninstall: googleapis-common-protos
        Found existing installation: googleapis-common-protos 1.56.4
        Uninstalling googleapis-common-protos-1.56.4:
          Successfully uninstalled googleapis-common-protos-1.56.4
      Attempting uninstall: google-auth
        Found existing installation: google-auth 1.35.0
        Uninstalling google-auth-1.35.0:
          Successfully uninstalled google-auth-1.35.0
      Attempting uninstall: sympy
        Found existing installation: sympy 1.7.1
        Uninstalling sympy-1.7.1:
          Successfully uninstalled sympy-1.7.1
      Attempting uninstall: google-api-core
        Found existing installation: google-api-core 1.31.6
        Uninstalling google-api-core-1.31.6:
          Successfully uninstalled google-api-core-1.31.6
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    pydata-google-auth 1.4.0 requires google-auth<3.0dev,>=1.25.0; python_version >= "3.6", but you have google-auth 1.18.0 which is incompatible.
    pydantic 1.10.2 requires typing-extensions>=4.1.0, but you have typing-extensions 3.10.0.0 which is incompatible.
    google-cloud-bigquery-storage 1.1.2 requires google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0dev,>=1.31.5, but you have google-api-core 1.21.0 which is incompatible.[0m
    Successfully installed cirq-core-0.13.1 cirq-google-0.13.1 duet-0.2.7 google-api-core-1.21.0 google-auth-1.18.0 googleapis-common-protos-1.52.0 sympy-1.8 tensorflow-quantum-0.7.2 typing-extensions-3.10.0.0




Now import TensorFlow, Keras and the module dependencies:


```
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
```

### A. Data Preparation



We will pull the ImageDataGenerator function from the Keras library to convert the images extracted from the zip format into mathematical array to make them ready for processing.


```
from keras.preprocessing.image import ImageDataGenerator
# We prepare of data
train_datagen = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False, rescale=1.0/255.0, preprocessing_function=None, data_format=None, dtype=None)

train_generator = train_datagen.flow_from_directory("/content/quantum",target_size=(4,4), batch_size=128, class_mode='categorical', interpolation="lanczos", color_mode="grayscale")
```

    Found 20 images belonging to 2 classes.



```
#from keras_image_generator type to numpy array
x=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
y=np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])
```


```
#Split of data to x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
```

Show the first example:


```
x_train.shape
```




    (16, 4, 4, 1)




```
print(y_train[0])

plt.imshow(x_train[0, :, :, 0])
plt.colorbar()
```

    [0. 1.]





    <matplotlib.colorbar.Colorbar at 0x7f0cacfdb190>




![png](/content/Quantum_with_6G_Technology_in_Computer_Vision_on_AI_19_2.png)


### A.B. Resize the images

An image size of 256x256 is much too large for my quantum computer. Resize the image down to 2x2:


```
first,two = 3,6
```


```
#x_train_resize = np.array(tf.image.resize(x_train, (first,two)))
#x_test_resize = np.array(tf.image.resize(x_test, (first,two)))
#print(y_train[0])

#plt.imshow(x_train_resize[0, :, :, 0])
#plt.colorbar()
```

### A.C. Encode the data as quantum circuits (with qubits)

To process images using a quantum computer.


```
THRESHOLD = 0.7

x_train_bin = np.array(x_train > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test > THRESHOLD, dtype=np.float32)
```


```
x_train_bin.shape
```




    (16, 4, 4, 1)



The qubits at pixel indices with values that exceed a threshold, are rotated through an $X$ gate. And we use (3,6) qubits.


```
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
```

Here is the circuit created for the first example (circuit diagrams do not show qubits with zero gates):


```
SVGCircuit(x_train_circ[0])
```




![svg](/content/Quantum_with_6G_Technology_in_Computer_Vision_on_AI_30_0.svg)



Compare this circuit to the indices where the image value exceeds the threshold:


```
bin_img = x_train_bin[-1]
indices = np.array(np.where(bin_img)).T
indices
```




    array([[0, 0, 0],
           [0, 1, 0],
           [0, 2, 0],
           [0, 3, 0],
           [3, 0, 0],
           [3, 1, 0],
           [3, 2, 0],
           [3, 3, 0]])



Convert these `Cirq` circuits to tensors for `tfq`:


```
x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
```

## B. Quantum6 prepared by Python

### B.A. Build the model circuit


```
class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)
```

Build an example circuit layer to see how it looks:


```
demo_builder = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(first,two),
                                   readout=cirq.GridQubit(-1,-1))

circuit = cirq.Circuit()
demo_builder.add_layer(circuit, gate = cirq.XX, prefix='Emirhan_Quantum6G_AI')
SVGCircuit(circuit)
```




![svg](/content/Quantum_with_6G_Technology_in_Computer_Vision_on_AI_39_0.svg)



Now build a quantum model, matching the data-circuit size, and include the preparation and readout operations.


```
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
```


```
model_circuit, model_readout = create_quantum_model()
```

### B.C. Build a Sequential Model for Quantum


```
# Build the Tensorflow/Keras Sequential model.
model = keras.Sequential([
    # The input is the data-circuit (data format), encoded as a tf.string (type)
    keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(model_circuit, model_readout),
])
```

Model Compile


```
model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"])
```


```
print(model.summary())
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     pqc_1 (PQC)                 (None, 1)                 36        
                                                                     
    =================================================================
    Total params: 36
    Trainable params: 36
    Non-trainable params: 0
    _________________________________________________________________
    None


### Quantum6 model with training in Keras


```
EPOCHS = 35
BATCH_SIZE = 128

NUM_EXAMPLES = len(x_train_tfcirc)
```

Model fitting


```
quantum6_history = model.fit(x_train_tfcirc, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=0,validation_data=(x_test_tfcirc, y_test))

quantum_6_results = model.evaluate(x_test_tfcirc, y_test)
```

    1/1 [==============================] - 0s 185ms/step - loss: 0.4939 - accuracy: 0.5000



```
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
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 3, 3, 2)           10        
                                                                     
     flatten (Flatten)           (None, 18)                0         
                                                                     
     dense (Dense)               (None, 2)                 38        
                                                                     
    =================================================================
    Total params: 48
    Trainable params: 48
    Non-trainable params: 0
    _________________________________________________________________



```
cnn_model = model.fit(x_train_bin,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=0,validation_data=(x_test_bin, y_test))

cnn_model_results = model.evaluate(x_test_bin, y_test)
```

    1/1 [==============================] - 0s 19ms/step - loss: 0.9469 - accuracy: 0.2500


## C. Results on Matplotlib


```
sns.barplot(["Quantum6 Accuracy","Convolutional Neural Network"],
            [quantum_6_results[1],cnn_model_results[1]])
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning





    <matplotlib.axes._subplots.AxesSubplot at 0x7f0caa894f10>




![png](/content/Quantum_with_6G_Technology_in_Computer_Vision_on_AI_55_2.png)

