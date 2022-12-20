# -*- coding: utf-8 -*-
"""
FYS-STK4155

Project 3

Some common functions used in other scripts
"""

import autograd.numpy as np

def initialise_params(n_neurons_list, input_shape, output_shape):
    
    L = len(n_neurons_list) + 1 # no. of layers excl. input layer
    params = [] # list to contain weights and biases
    
    # initialising parameters for first hidden layer
    W_l = np.random.randn(input_shape, n_neurons_list[0])
    b_l = np.zeros(n_neurons_list[0])
    P_l = np.c_[W_l.T, b_l]
    params.append(P_l)
    
    for l in range(1,L-1):
        # initialising parameters for remaining hidden layers
        W_l = np.random.randn(n_neurons_list[l-1], n_neurons_list[l])
        b_l = np.zeros(n_neurons_list[l])
        P_l = np.c_[W_l.T, b_l]
        params.append(P_l)
    
    # initialising parameters for output layer
    W_l = np.random.randn(n_neurons_list[-1], output_shape)
    b_l = np.zeros(output_shape)
    P_l = np.c_[W_l.T, b_l]
    params.append(P_l)
    
    return params


def sigmoid(z):
    return 1./(1. + np.exp(-z))

def RELU(z):
    return np.maximum(0., z)

def leaky_RELU(z):
    alpha = 0.01
    return np.maximum(z, alpha*z)

def tanh(z):
    return np.tanh(z)