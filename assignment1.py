from keras.datasets import fashion_mnist, mnist
import wandb
import numpy as np
import math
import random
import argparse


def input_layer(x):
    x = np.array(x)
    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], -1)
    return x

def sigmoid(a_x):
    a_x = np.clip(a_x, -700, 700)
    h_x = 1 / (1 + np.exp(-a_x))
    return h_x

def der_sigmoid(a_x):
    sig_x = sigmoid(a_x)
    del_sig = sig_x * (1 - sig_x)
    return del_sig

def Relu(a_x):
    h_x = np.clip(np.maximum(0, a_x), 0, 1e4)
    return h_x

def der_Relu(a_x):
    del_Relu = (a_x > 0).astype(float)
    return del_Relu

def tanh(a_x):
    a_x = np.clip(a_x, -700, 700)
    h_x = (np.exp(a_x) - np.exp(-a_x))/(np.exp(a_x) + np.exp(-a_x))
    return h_x

def der_tanh(a_x):
    del_tanh = 1 - ((np.exp(a_x) - np.exp(-a_x))/(np.exp(a_x) + np.exp(-a_x)))**2
    return del_tanh

def softmax(a_x):
    a_x = a_x - np.max(a_x)
    h_x = np.exp(a_x)
    h_x = h_x/np.sum(h_x)
    return h_x

def initialize_weights_xavier(num_neurons):
    np.random.seed(450)
    weights = []
    biases = []
    for i in range(len(num_neurons)-1):
        W = np.random.randn(num_neurons[i+1], num_neurons[i])*np.sqrt(1 / num_neurons[i])
        b = np.zeros((1, num_neurons[i+1]))
        weights.append(W)
        biases.append(b)
    return weights, biases

def initialize_weights_random(num_neurons):
    np.random.seed(450)
    weights = []
    biases = []
    for i in range(len(num_neurons)-1):
        W = np.random.randn(num_neurons[i+1], num_neurons[i])
        b = np.zeros((1, num_neurons[i+1]))
        weights.append(W)
        biases.append(b)
    return weights, biases

def pre_activation(h_x, W, b):
    a_x = np.dot(W, h_x.T) + b.flatten()
    return a_x

def bce_loss_function(h_x, y):
    h_x = np.clip(h_x, 1e-8, 1.0)
    loss = -np.log(h_x[np.argmax(y)])
    return loss

def mse_loss_function(h_x, y):
    loss = np.sum((h_x - y)**2)
    return loss

def forward_pass(x, y, weights, biases, activation_func, n_hidden, loss_function):
    activations = []
    pre_activations = []
    for i in range(n_hidden+1):
        a_x = pre_activation(x if i == 0 else activations[-1], weights[i], biases[i])
        h_x = softmax(a_x) if i == n_hidden else activation_func(a_x)
        activations.append(h_x)
        pre_activations.append(a_x)
    loss = loss_function(h_x, y)
    return activations, pre_activations, loss

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def back_propagation(activations, pre_activations, weights, biases, x, y, y_hat, n_hidden, activation_deriv, loss_function):
    del_L_a = {}
    del_L_w = {}
    del_L_b = {}
    del_L_h = {}
    for i in range(n_hidden, -1, -1):
        if i == n_hidden:
            if loss_function == bce_loss_function:
                del_L_a[i] = y_hat - y
            elif loss_function == mse_loss_function:
                del_L_a[i] = 2 * (y_hat - y) * y_hat * (1 - y_hat)
        if i == 0:
            del_L_w[i] = np.dot(del_L_a[i][:, np.newaxis], x[np.newaxis, :])
            del_L_b[i] = del_L_a[i]
            break
        else:
            del_L_w[i] = np.dot(del_L_a[i][:, np.newaxis], activations[i-1][np.newaxis, :])
        del_L_b[i] = del_L_a[i]
        del_L_h[i-1] = np.matmul(weights[i].T, del_L_a[i])
        del_L_a[i-1] = del_L_h[i-1]*activation_deriv(pre_activations[i-1])
    return del_L_w, del_L_b

def gradient_descent(dw, db, weights, biases, learning_rate, weight_decay):
    for i in range(len(weights)):
        weights[i] -= learning_rate*(dw[i] + weight_decay * weights[i])
        biases[i] -= learning_rate*db[i]
    return weights, biases

def momentum_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, momentum):
    u_w = {}
    u_b = {}
    beta = momentum
    for i in range(len(weights)):
        dw[i] += weight_decay * weights[i]
        if prev_u_w == {} and prev_u_b == {}:
            u_w[i] = learning_rate*dw[i]
            u_b[i] = learning_rate*db[i]
        else:
            u_w[i] = beta*prev_u_w[i] + learning_rate*dw[i]
            u_b[i] = beta*prev_u_b[i] + learning_rate*db[i]
        weights[i] -= u_w[i]
        biases[i] -= u_b[i]
    return weights, biases, u_w, u_b

def nestrov_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, momentum):
    u_w = {}
    u_b = {}
    beta = momentum
    for i in range(len(weights)):
        dw[i] += weight_decay * weights[i]
        if prev_u_w == {} and prev_u_b == {}:
            u_w[i] = learning_rate*dw[i]
            u_b[i] = learning_rate*db[i]
        else:
            u_w[i] = beta*prev_u_w[i] + learning_rate*dw[i]
            u_b[i] = beta*prev_u_b[i] + learning_rate*db[i]
        weights[i] -= u_w[i]
        biases[i] -= u_b[i]
    return weights, biases, u_w, u_b

def rmsprop_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, beta, epsilon):
    u_w = {}
    u_b = {}
    beta = beta
    epsilon = epsilon
    for i in range(len(weights)):
        dw[i] += weight_decay * weights[i]
        if prev_u_w == {} and prev_u_b == {}:
            u_w[i] = (1 - beta) * (dw[i] ** 2)
            u_b[i] = (1 - beta) * (db[i] ** 2)
        else:
            u_w[i] = beta * prev_u_w[i] + (1 - beta) * (dw[i] ** 2)
            u_b[i] = beta * prev_u_b[i] + (1 - beta) * (db[i] ** 2)

        weights[i] -= learning_rate * dw[i] / (np.sqrt(u_w[i] + epsilon))
        biases[i] -= learning_rate * db[i] / (np.sqrt(u_b[i] + epsilon))

    return weights, biases, u_w, u_b

def adagrad_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, epsilon):
    u_w = {}
    u_b = {}
    epsilon = epsilon
    for i in range(len(weights)):
        dw[i] += weight_decay * weights[i]
        if prev_u_w == {} and prev_u_b == {}:
            u_w[i] = (dw[i] ** 2)
            u_b[i] = (db[i] ** 2)
        else:
            u_w[i] = prev_u_w[i] + (dw[i] ** 2)
            u_b[i] = prev_u_b[i] + (db[i] ** 2)

        weights[i] -= learning_rate * dw[i] / (np.sqrt(u_w[i] + epsilon))
        biases[i] -= learning_rate * db[i] / (np.sqrt(u_b[i] + epsilon))

    return weights, biases, u_w, u_b

def adadelta_gradient(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, prev_v_w, prev_v_b, weight_decay, beta, epsilon):
    u_w = {}
    u_b = {}
    v_w = {}
    v_b = {}
    beta = beta
    epsilon = epsilon

    for i in range(len(weights)):
        dw[i] += weight_decay * weights[i]
        if prev_v_b == {} and prev_v_w == {}:
            v_w[i] = (1 - beta) * (dw[i] ** 2)
            v_b[i] = (1 - beta) * (db[i] ** 2)
            update_w = (np.sqrt(epsilon) / np.sqrt(v_w[i] + epsilon)) * dw[i]
            update_b = (np.sqrt(epsilon) / np.sqrt(v_b[i] + epsilon)) * db[i]
            weights[i] -= update_w
            biases[i] -= update_b
            u_w[i] = (1 - beta) * (update_w ** 2)
            u_b[i] = (1 - beta) * (update_b ** 2)
        else:
            v_w[i] = beta * prev_v_w[i] + (1 - beta) * (dw[i] ** 2)
            v_b[i] = beta * prev_v_b[i] + (1 - beta) * (db[i] ** 2)
            update_w = (np.sqrt(prev_u_w[i] + epsilon) / np.sqrt(v_w[i] + epsilon)) * dw[i]
            update_b = (np.sqrt(prev_u_b[i] + epsilon) / np.sqrt(v_b[i] + epsilon)) * db[i]
            weights[i] -= update_w
            biases[i] -= update_b
            u_w[i] = beta * prev_u_w[i] + (1 - beta) * (update_w ** 2)
            u_b[i] = beta * prev_u_b[i] + (1 - beta) * (update_b ** 2)

    return weights, biases, u_w, u_b, v_w, v_b

def adam_gradient(dw, db, weights, biases, learning_rate, prev_m_w, prev_m_b, prev_v_w, prev_v_b, weight_decay, iteration, beta1, beta2, epsilon):
    m_w = {}
    m_b = {}
    v_w = {}
    v_b = {}

    beta1 = beta1
    beta2 = beta2
    epsilon = epsilon

    for i in range(len(weights)):
        dw[i] += weight_decay * weights[i]
        if prev_m_w == {} and prev_m_b == {}:
            m_w[i] = (1 - beta1) * dw[i]
            m_b[i] = (1 - beta1) * db[i]
            v_w[i] = (1 - beta2) * (dw[i] ** 2)
            v_b[i] = (1 - beta2) * (db[i] ** 2)
        else:
            m_w[i] = beta1 * prev_m_w[i] + (1 - beta1) * dw[i]
            m_b[i] = beta1 * prev_m_b[i] + (1 - beta1) * db[i]
            v_w[i] = beta2 * prev_v_w[i] + (1 - beta2) * (dw[i] ** 2)
            v_b[i] = beta2 * prev_v_b[i] + (1 - beta2) * (db[i] ** 2)
        m_w_hat = m_w[i] / (1 - beta1**iteration)
        m_b_hat = m_b[i] / (1 - beta1**iteration)
        v_w_hat = v_w[i] / (1 - beta2**iteration)
        v_b_hat = v_b[i] / (1 - beta2**iteration)
        weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return weights, biases, m_w, m_b, v_w, v_b

def nadam_gradient(dw, db, weights, biases, learning_rate, prev_m_w, prev_m_b, prev_v_w, prev_v_b, weight_decay, iteration, beta1, beta2, epsilon):
    m_w = {}
    m_b = {}
    v_w = {}
    v_b = {}

    beta1 = beta1
    beta2 = beta2
    epsilon = epsilon

    for i in range(len(weights)):
        dw[i] += weight_decay * weights[i]
        if prev_m_w == {} and prev_m_b == {}:
            m_w[i] = (1 - beta1) * dw[i]
            m_b[i] = (1 - beta1) * db[i]
            v_w[i] = (1 - beta2) * (dw[i] ** 2)
            v_b[i] = (1 - beta2) * (db[i] ** 2)
        else:
            m_w[i] = beta1 * prev_m_w[i] + (1 - beta1) * dw[i]
            m_b[i] = beta1 * prev_m_b[i] + (1 - beta1) * db[i]
            v_w[i] = beta2 * prev_v_w[i] + (1 - beta2) * (dw[i] ** 2)
            v_b[i] = beta2 * prev_v_b[i] + (1 - beta2) * (db[i] ** 2)
        m_w_hat = m_w[i] / (1 - beta1**iteration)
        m_b_hat = m_b[i] / (1 - beta1**iteration)
        v_w_hat = v_w[i] / (1 - beta2**iteration)
        v_b_hat = v_b[i] / (1 - beta2**iteration)
        lookahead_m_w = beta1 * m_w_hat + (1 - beta1) * dw[i] / (1 - beta1 ** iteration)
        lookahead_m_b = beta1 * m_b_hat + (1 - beta1) * db[i] / (1 - beta1 ** iteration)
        weights[i] -= learning_rate * lookahead_m_w / (np.sqrt(v_w_hat) + epsilon)
        biases[i] -= learning_rate * lookahead_m_b / (np.sqrt(v_b_hat) + epsilon)

    return weights, biases, m_w, m_b, v_w, v_b

def validation(x_val, y_val, weights, biases, activation_func, n_hidden, loss_function):
    val_loss_final = 0
    y_pred_val = []
    y_val_j = []
    for j in range(0, len(x_val)):
        x_val_each = x_val[j]
        y_val_each = y_val[j]
        activ, _,val_loss = forward_pass(x_val_each, y_val_each, weights, biases, activation_func, n_hidden, loss_function)
        a_1 = activ[-1]
        y_pred_val.append(np.argmax(a_1))
        y_val_j.append(np.argmax(y_val[j]))
        val_loss_final = val_loss_final + val_loss
    accuracy = np.mean(np.array(y_pred_val) == np.array(y_val_j))
    return val_loss_final/len(x_val), accuracy

def Neuralnet(x_train, y_train, x_val, y_val, n_hidden, n_neurons_hidden, epochs, batch_size, activation, optimization, learning_rate, weight_decay, loss_function, weight_initialization, momentum, beta, beta1, beta2, epsilon, classes):
    x_train = input_layer(x_train)
    y_train = one_hot_encode(y_train, classes)
    x_val = input_layer(x_val)
    y_val = one_hot_encode(y_val, classes)
    features = x_train.shape[1]
    num_neurons = [features] + [n_neurons_hidden]*(n_hidden) + [classes]
    initialize_weights = {"random": initialize_weights_random, "Xavier": initialize_weights_xavier}[weight_initialization]
    activation_func = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": Relu}[activation]
    activation_deriv = {"sigmoid": der_sigmoid, "tanh": der_tanh, "ReLU": der_Relu}[activation]
    optimization_func = {"momentum": momentum_gradient, "sgd": gradient_descent, "nag": nestrov_gradient, "rmsprop": rmsprop_gradient, "adagrad": adagrad_gradient, "adadelta": adadelta_gradient, "adam": adam_gradient, "nadam": nadam_gradient}[optimization]
    loss_function = {"bce": bce_loss_function, "mse": mse_loss_function}[loss_function]
    weights, biases = initialize_weights(num_neurons)

    if optimization_func == gradient_descent:
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                dw = {}
                db = {}
                for x,y in zip(x_batch,y_batch):
                    activations, pre_activations,_ = forward_pass(x, y, weights, biases, activation_func, n_hidden, loss_function)
                    del_L_w, del_L_b = back_propagation(activations, pre_activations, weights, biases, x, y, activations[-1], n_hidden, activation_deriv, loss_function)
                    for key,value in del_L_w.items():
                        if key not in dw:
                            dw[key] = value
                        else:
                            dw[key] = dw[key] + value
                    for key,value in del_L_b.items():
                        if key not in db:
                            db[key] = value
                        else:
                            db[key] = db[key] + value
                for key in dw:
                    dw[key] /= batch_size
                    db[key] /= batch_size
                weights, biases = optimization_func(dw, db, weights, biases, learning_rate, weight_decay)
            val_loss, val_accuracy = validation(x_val, y_val, weights, biases, activation_func, n_hidden, loss_function)
            train_loss, train_accuracy = validation(x_train, y_train, weights, biases, activation_func, n_hidden, loss_function)
            wandb.log({
                           "epoch": epoch,
                           "train_loss": train_loss,
                           "train_accuracy": train_accuracy,
                           "val_loss": val_loss,
                           "val_accuracy": val_accuracy
                       })

    elif optimization_func == nestrov_gradient:
        prev_u_w = {}
        prev_u_b = {}
        beta = 0.9
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                dw = {}
                db = {}
                for x,y in zip(x_batch,y_batch):
                    if i == 0 and epoch == 0:
                        activations, pre_activations,_ = forward_pass(x, y, weights, biases, activation_func, n_hidden, loss_function)
                        del_L_w, del_L_b = back_propagation(activations, pre_activations, weights, biases, x, y, activations[-1], n_hidden, activation_deriv, loss_function)
                    else:
                        look_ahead_weights = {key: weights[key] - beta*prev_u_w[key] for key in range(len(weights))}
                        look_ahead_biases = {key: biases[key] - beta*prev_u_b[key] for key in range(len(weights))}
                        activations, pre_activations,_ = forward_pass(x, y, look_ahead_weights, look_ahead_biases, activation_func, n_hidden, loss_function)
                        del_L_w, del_L_b = back_propagation(activations, pre_activations, look_ahead_weights, look_ahead_biases, x, y, activations[-1], n_hidden, activation_deriv, loss_function)
                    for key,value in del_L_w.items():
                        if key not in dw:
                            dw[key] = value
                        else:
                            dw[key] = dw[key] + value
                    for key,value in del_L_b.items():
                        if key not in db:
                            db[key] = value
                        else:
                            db[key] = db[key] + value
                for key in dw:
                    dw[key] /= batch_size
                    db[key] /= batch_size
                weights, biases, prev_u_w, prev_u_b = optimization_func(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, momentum)
            val_loss, val_accuracy = validation(x_val, y_val, weights, biases, activation_func, n_hidden, loss_function)
            train_loss, train_accuracy = validation(x_train, y_train, weights, biases, activation_func, n_hidden, loss_function)
            wandb.log({
                           "epoch": epoch,
                           "train_loss": train_loss,
                           "train_accuracy": train_accuracy,
                           "val_loss": val_loss,
                           "val_accuracy": val_accuracy
                       })

    elif optimization_func == momentum_gradient or optimization_func == rmsprop_gradient or optimization_func == adadelta_gradient or optimization_func == adam_gradient or optimization_func == nadam_gradient:
        if optimization_func == adadelta_gradient or optimization_func == adam_gradient or optimization_func == nadam_gradient:
            iteration = 0
            prev_u_w = {}
            prev_u_b = {}
            prev_v_w = {}
            prev_v_b = {}
        else:
            prev_u_w = {}
            prev_u_b = {}
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                dw = {}
                db = {}
                for x,y in zip(x_batch,y_batch):
                    activations, pre_activations,_ = forward_pass(x, y, weights, biases, activation_func, n_hidden, loss_function)
                    del_L_w, del_L_b = back_propagation(activations, pre_activations, weights, biases, x, y, activations[-1], n_hidden, activation_deriv, loss_function)
                    for key,value in del_L_w.items():
                        if key not in dw:
                            dw[key] = value
                        else:
                            dw[key] = dw[key] + value
                    for key,value in del_L_b.items():
                        if key not in db:
                            db[key] = value
                        else:
                            db[key] = db[key] + value
                for key in dw:
                    dw[key] /= batch_size
                    db[key] /= batch_size
                if optimization_func == adadelta_gradient:
                    weights, biases, prev_u_w, prev_u_b, prev_v_w, prev_v_b = optimization_func(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, prev_v_w, prev_v_b, weight_decay, beta, epsilon)
                elif optimization_func == adam_gradient or optimization_func == nadam_gradient:
                    iteration +=1
                    weights, biases, prev_u_w, prev_u_b, prev_v_w, prev_v_b = optimization_func(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, prev_v_w, prev_v_b, weight_decay, iteration, beta1, beta2, epsilon)
                elif optimization_func == rmsprop_gradient:
                    weights, biases, prev_u_w, prev_u_b = optimization_func(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, beta, epsilon)
                else:
                    weights, biases, prev_u_w, prev_u_b = optimization_func(dw, db, weights, biases, learning_rate, prev_u_w, prev_u_b, weight_decay, momentum)
            val_loss, val_accuracy = validation(x_val, y_val, weights, biases, activation_func, n_hidden, loss_function)
            train_loss, train_accuracy = validation(x_train, y_train, weights, biases, activation_func, n_hidden, loss_function)
            wandb.log({
                           "epoch": epoch,
                           "train_loss": train_loss,
                           "train_accuracy": train_accuracy,
                           "val_loss": val_loss,
                           "val_accuracy": val_accuracy
                       })
def main():
    parser = argparse.ArgumentParser(description="Train a neural network with specified parameters")

    # Weights & Biases arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="Fashion-mnist-test", help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="sai-sakunthala-indian-institute-of-technology-madras", help="WandB entity name")

    # Dataset & Training parameters
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("-l", "--loss", type=str, choices=["mse", "bce"], default="bce", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adadelta", "adam", "nadam"], default="adam", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum (used by momentum & nag)")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta (used by RMSProp)")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 (used by Adam & Nadam)")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 (used by Adam & Nadam)")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon (used by optimizers)")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay (used by optimizers)")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", help="Weight initialization method")

    # Network structure parameters
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons per layer")
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "ReLU"], default="tanh", help="Activation function")

    args = parser.parse_args()
    def train():
        try:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
            optimizer = args.optimizer
            epochs = args.epochs
            n_hidden = args.num_layers
            n_neurons_hidden = args.hidden_size
            batch_size = args.batch_size
            learning_rate = args.learning_rate
            weight_decay = args.weight_decay
            loss_function = args.loss
            activation_str = args.activation
            weight_initialization = args.weight_init
            dataset_func = mnist if args.dataset == "mnist" else fashion_mnist
            (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
            X_train = X_train/255
            X_test = X_test/255
            classes = len(np.unique(Y_train))
            split_index = int(0.9 * X_train.shape[0])
            x_train_final, x_val_final = X_train[:split_index], X_train[split_index:]
            y_train_final, y_val_final = Y_train[:split_index], Y_train[split_index:]
            momentum = args.momentum
            beta = args.beta
            beta1 = args.beta1
            beta2 = args.beta2
            epsilon = args.epsilon
            Neuralnet(x_train_final, y_train_final, x_val_final, y_val_final, n_hidden, n_neurons_hidden, epochs, batch_size, activation_str, optimizer, learning_rate, weight_decay, loss_function, weight_initialization, momentum, beta, beta1, beta2, epsilon, classes)
        finally:
            wandb.finish()
    train()
if __name__ == "__main__":
    main()