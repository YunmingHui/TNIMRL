# TNIMRL

This is the code of paper "[Non-Progressive Influence Maximization in Temporal Social Networks](https://www.sciencedirect.com/science/article/pii/S095741742603294X?via%3Dihub)".

## Introduction

The influence maximization (IM) problem involves identifying a set of key individuals in a social network who can maximize the spread of influence through their network connections. In this paper, we focus on the temporal non-progressive IM problem, which considers the temporal nature of real-world social networks and the special case where the influence diffusion is non-progressive, i.e., nodes can be activated multiple times. While influence diffusion in many real-world social network scenarios is non-progressive, such as in marketing campaigns promoting updated products and public health campaigns promoting healthier lifestyle choices, this problem has received limited attention in prior research. We first extend an existing diffusion model to capture the non-progressive influence diffusion in temporal social networks. We then propose a method, named TNIMRL, which employs deep reinforcement learning and temporal graph embedding to solve the temporal non-progressive IM problem. In particular, we propose a novel algorithm that effectively leverages temporal graph embedding to capture the changes in temporal social networks and seamlessly integrates with deep reinforcement learning. The extensive experiments on different types of real-world social networks demonstrate that our method outperforms state-of-the-art baselines.

## Run the code

The code in written with Python. To use the correct version of Python, please look at the [requirement of PyG](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html) which is the package used in the code.

### Step 1 Install necessary packages

#### Public available packages

- torch_geometric
- pytorch 2.X
- numpy

#### Influence estimation package

For higher efficiency, influence estimation is performed by a package written in C++. To install this package:

1. pip install pybind11
2. cd DiffusionModel_pybind11/C++2Python
3. python setup.py build_ext --inplace
4. python setup.py install

> Note: Before building this C++ extension, make sure a C++ compiler (e.g. gcc/g++) is installed on your system.

To test if the package is installed successfully, back to the main folder and run the file

1. python test_influence_estimation.py
2. If the estimated influence of the test seed set is printed, the package is installed successfully

### Step 2 Data preprocessing

For new datasets, please use a .txt file to represent the temporal social network and put the text file in the `Data` folder. The .txt file should in the following format:

- The node is indexed continuously, starting from 0.
- Each line should be `source_node destination_node timestamp`

### Step 3 Run the code

To find the seed set, you need to run the file "TNIMRL.py". Following is an example with following configurations

- Dataset: Bitcoinalpha
- Seed set size: 10
- Length of a successful activation $\tau$: 30 days
- Threshold for action pruning $max_{ts}$: 30
- Reward scale: 0.00001 

``````bash
python train_DoubleDQN.py --dataset Bitcoinalpha --size_seed_set 10 --activation_length 30 --minimal_activated_nodes 30 --reward_scale 0.00001
``````

## Cite us

Please cite the paper if you use this code in your work:

``````latex
@article{HUI2027134390,
title = {Non-progressive influence maximization in temporal social networks},
journal = {Expert Systems with Applications},
volume = {334},
pages = {134390},
year = {2027},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.134390},
url = {https://www.sciencedirect.com/science/article/pii/S095741742603294X},
author = {Yunming Hui and Shihan Wang and Melisachew Wudage Chekol and Stevan Rudinac and Inez Maria Zwetsloot}
}
``````

