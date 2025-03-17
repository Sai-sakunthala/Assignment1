# Assignment1
This repository contains the code and resources for Assignment 1. The primary focus is on the assignment1.py script, which implements our sweeps and takes the input as specified in requirements. Other supporting files are also included which are basically colab notebooks of all different sweeps and experiments i lave logged and tried. lets look at each of them in detail.
## Assignment1.py
This script contains a Python implementation of a neural network from scratch using NumPy. The network is trained on the Fashion MNIST and MNIST datasets, and it supports various activation functions, loss functions, and optimization algorithms. Below is a detailed explanation of the code and its functionality.
## Table of Contents
1. **Repository Structure**
2. **Key Features**
3. **Code Explanation**
   - Activation Functions
   - Loss Functions
   - Weight Initialization
   - Forward Pass
   - Backpropagation
   - Optimization Algorithms
   - Training and Validation
4. **Usage**
5. **Dependencies**
6. **Contributing**
7. **License**
## Key Features
1. **Datasets**:
   - Supports both Fashion MNIST and MNIST datasets.
   - Data is normalized and split into training and validation sets.

2. **Activation Functions**:
   - Sigmoid
   - Tanh
   - ReLU

3. **Loss Functions**:
   - Categorical Cross-Entropy (in this i have named as BCE)
   - Mean Squared Error (MSE)

4. **Optimization Algorithms**:
   - Stochastic Gradient Descent (SGD)
   - Momentum
   - Nesterov Accelerated Gradient (NAG)
   - RMSProp
   - AdaGrad
   - AdaDelta
   - Adam
   - Nadam

5. **Weight Initialization**:
   - Random Initialization
   - Xavier Initialization

6. **Weights & Biases Integration**:
   - Logs training and validation metrics (loss, accuracy) for visualization.
