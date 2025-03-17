# Assignment1
This repository contains the code and resources for Assignment 1. The primary focus is on the assignment1.py script, which implements our sweeps and takes the input as specified in requirements. Other supporting files are also included which are basically colab notebooks of all different sweeps and experiments i lave logged and tried. lets look at each of them in detail.
## Assignment1.py sript
This script contains a Python implementation of a neural network from scratch using NumPy. The network is trained on the Fashion MNIST and MNIST datasets, and it supports various activation functions, loss functions, and optimization algorithms. Below is a detailed explanation of the code and its functionality.
## Table of Contents
1. **Key Features**
2. **Code Explanation**
   - Activation Functions
   - Loss Functions
   - Weight Initialization
   - Forward Pass
   - Backpropagation
   - Optimization Algorithms
   - Training and Validation
3. **Usage**
4. **Dependencies**
5. **Contributing**
6. **License**
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

## Code Explanation

### 1. **Activation Functions**
- **`sigmoid(a_x)`**: Computes the sigmoid activation function.
- **`der_sigmoid(a_x)`**: Computes the derivative of the sigmoid function.
- **`Relu(a_x)`**: Computes the ReLU activation function.
- **`der_Relu(a_x)`**: Computes the derivative of the ReLU function.
- **`tanh(a_x)`**: Computes the hyperbolic tangent (tanh) activation function.
- **`der_tanh(a_x)`**: Computes the derivative of the tanh function.
- **`softmax(a_x)`**: Computes the softmax function for multi-class classification.

### 2. **Loss Functions**
- **`bce_loss_function(h_x, y)`**: Computes the Binary Cross-Entropy (BCE) loss.
- **`mse_loss_function(h_x, y)`**: Computes the Mean Squared Error (MSE) loss.

### 3. **Weight Initialization**
- **`initialize_weights_xavier(num_neurons)`**: Initializes weights using Xavier initialization.
- **`initialize_weights_random(num_neurons)`**: Initializes weights randomly.

### 4. **Forward Pass**
- **`forward_pass(x, y, weights, biases, activation_func, n_hidden, loss_function)`**:
  - Computes the forward pass through the network.
  - Returns activations, pre-activations, and the loss.

### 5. **Backpropagation**
- **`back_propagation(activations, pre_activations, weights, biases, x, y, y_hat, n_hidden, activation_deriv, loss_function)`**:
  - Computes gradients for weights and biases using backpropagation.
  - Returns gradients for weights (`del_L_w`) and biases (`del_L_b`).

### 6. **Optimization Algorithms**
- **`gradient_descent(dw, db, weights, biases, learning_rate, weight_decay)`**: Implements standard gradient descent.
- **`momentum_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, momentum)`**: Implements gradient descent with momentum.
- **`nestrov_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, momentum)`**: Implements Nesterov Accelerated Gradient (NAG).
- **`rmsprop_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, beta, epsilon)`**: Implements RMSProp.
- **`adagrad_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, epsilon)`**: Implements AdaGrad.
- **`adadelta_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, prev_v_w, prev_v_b, weight_decay, beta, epsilon)`**: Implements AdaDelta.
- **`adam_gradient(dw, db, weights, biases, learning_rate, prev_m_w, prev_m_b, prev_v_w, prev_v_b, weight_decay, iteration, beta1, beta2, epsilon)`**: Implements Adam.
- **`nadam_gradient(dw, db, weights, biases, learning_rate, prev_m_w, prev_m_b, prev_v_w, prev_v_b, weight_decay, iteration, beta1, beta2, epsilon)`**: Implements Nadam.

### 7. **Training and Validation**
- **`Neuralnet(x_train, y_train, x_val, y_val, n_hidden, n_neurons_hidden, epochs, batch_size, activation, optimization, learning_rate, weight_decay, loss_function, weight_initialization, momentum, beta, beta1, beta2, epsilon, classes)`**:
  - Trains the neural network using the specified parameters.
  - Logs training and validation metrics using Weights & Biases.
- **`validation(x_val, y_val, weights, biases, activation_func, n_hidden, loss_function)`**:
  - Computes validation loss and accuracy.

## Dependencies
**Python 3.x**
**NumPy**
**Keras (for dataset loading)**
**Weights & Biases (for logging)**
