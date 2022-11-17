# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

Neural Network class

"""

import numpy as np

from gradientdescent import GradientDescent

class NeuralNetwork:
    def __init__(self, 
                 input_size, 
                 output_size, 
                 nodes_per_layer,
                 lamb = 0.,
                 classification=False, 
                 activation_functions="RELU"):
        '''
        Neural Network class
        
        inputs:
        input_size (int) : no. of components of input vector x
        output_size (int) : no. of components of output vector y
        nodes_per_layer (list) : list containing no. of nodes in each of the hidden layers
        lamb (float) : L2 regularisation hyperparameter
        classification (bool) : if True, NN will do classification and not regression
        activation_functions (str or list) : desired activation functions for each layer
        '''
        n_hidden = len(nodes_per_layer) # no. of hidden layers
        self.n_hidden = n_hidden
        self.L = n_hidden + 1 # no. of layers (excluding input layer)
        
        # initialising weights for each layer
        self.W_list = np.empty(n_hidden + 1, dtype=object)
        if n_hidden>=1:
            W_l = np.random.randn(input_size, nodes_per_layer[0])
            self.W_list[0] = W_l

            if n_hidden >= 2:
                for l in range(1, n_hidden):
                    W_l = np.random.randn(nodes_per_layer[l-1], nodes_per_layer[l])
                    self.W_list[l] = W_l

            W_l = np.random.randn(nodes_per_layer[-1], output_size)
            self.W_list[-1] = W_l
        else:
            W_l = np.random.randn(input_size, output_size)
            self.W_list[0] = W_l
        
        # initialising biases
        self.b_list = np.array([np.zeros(nodes_per_layer[l]) for l in range(n_hidden)] + [np.zeros(output_size)])
        
        # setting activation functions for each layer
        if type(activation_functions) is str:
            # e.g. activation_function = "RELU" for all layers
            activation_function = getattr(self, activation_functions)
            self.activation_function_list = [activation_function]*self.L
            activation_function_derivative = getattr(self, activation_functions + "_derivative")
            self.activation_function_derivative_list = [activation_function_derivative]*self.L
        else:
            # if list of different act. fn.s for each layer
            self.activation_function_list = []
            self.activation_function_derivative_list = []
            for function_name in activation_functions:
                # make list of activation functions
                activation_function = getattr(self, function_name)
                self.activation_function_list.append(activation_function)
                # make list of act. fn. derivatives
                activation_function_derivative = getattr(self, function_name + "_derivative")
                self.activation_function_derivative_list.append(activation_function_derivative)
        
        self.lamb = lamb
        self.classification = classification
        if classification:
            assert activation_functions[-1] == "sigmoid", "Activation function of final layer must be sigmoid"
        
    def feed_forward(self, x):
        '''
        Feed forward algorithm
        input:
            x (ndarray) - input values
        output:
            output (ndarray) - output values
            node_values (ndarray) - values for each node in network
            z_values (ndarray) - values that go into activation function
        '''
        
        # create array corresponding to input, hidden and output layers
        # first index -> layer
        # second index -> specimen (1,...,n)
        # third index -> node value
        node_values = np.empty(self.L + 1, dtype=object)
        # setting input to input layer
        node_values[0] = x
        # initialise array of input z to activation functions in each layer
        z_values = np.empty_like(node_values[1:])
        for l in range(self.L):
            # l = 1, 2, ..., L
            W = self.W_list[l] 
            b = self.b_list[l]
            z_l = node_values[l]@W + b
            z_values[l] = z_l
            activation_function = self.activation_function_list[l]
            node_values[l+1] = activation_function(z_l)
        output = node_values[-1]
        self.node_values = node_values # -> get nn.node_values to calculate dW = eta*delta*a_ in GD
        
        return output, node_values, z_values # returns array of n outputs, list of layers (n versions of each) w/ corr. z_vals
    
    def backpropagation(self, x, target):
        '''
        Backpropagation algorithm
        inputs:
            x (ndarray) : input data
        outputs:
            delta_list (ndarray) : error/gradient for each layer
        '''
        
        # backpropagation gives gradient delta_l
        if x.ndim == 2:
            n = x.shape[0]
        else:
            n = len(x)
        
        if target.ndim == 1:
            target = target.reshape(n,1)
        
        output, node_values, z_values = self.feed_forward(x)
        
        # create array of deltas (each layer has n "versions" of delta_l)
        delta_list = np.empty_like(node_values[1:])
        
        # calculating gradient starting at final layer
        if self.classification:
            delta_l = output - target
        else:
            derivative = self.activation_function_derivative_list[-1]
            dC_daL = 2./n * (output - target) # have n outputs (possibly each vector-valued making this possibly a 2d array)
            z_l = z_values[-1]
            delta_l = derivative(z_l) * dC_daL
        delta_list[-1] = delta_l
        
        # performing backpropagation:
        for l in range(self.L-1, 0, -1):
            derivative = self.activation_function_derivative_list[l-1]
            z_l = z_values[l-1]
            W = self.W_list[l]
            delta_l = derivative(z_l) * (delta_list[l] @ W.T)
            delta_list[l-1] = delta_l
        
        return delta_list
    
    
    def train(self, 
              inputs, 
              outputs, 
              learning_schedule="rmsprop", 
              eta = 0.001,
              momentum = 0.8, 
              stochastic=True, 
              batch_size=5,
              n_epochs=100,
              n_iter=10000):
        """
        Training the NN on data set (inputs, outputs)
        """
        gd = GradientDescent(eta, learning_schedule, momentum=momentum,stochastic=stochastic,n_epochs=n_epochs,batch_size=batch_size)
        # create a single array of parameter values
        beta = np.array([self.W_list, self.b_list], dtype=object)
        # running gradient descent
        beta = gd.gradient_descent(inputs,outputs,beta,self,n_iter=n_iter)
    
    
    def predict(self,x):
        return self.feed_forward(x)[0]
    
    """
    defining activation functions and their derivatives
    """
    
    def sigmoid(self,z):
        return 1./(1. + np.exp(-z))
    
    def sigmoid_derivative(self,z):
        sigma = self.sigmoid(z)
        return sigma*(1.-sigma)

    def RELU(self,z):
        return np.maximum(0., z)
    
    def RELU_derivative(self,z):
        # = 0 if z>=0 or 1 if z>0
        return z > 0.

    def leaky_RELU(self,z):
        alpha = 0.01
        return np.maximum(z, alpha*z)
    
    def leaky_RELU_derivative(self,z):
        alpha = 0.01
        temp = np.ones_like(z)
        temp[np.argwhere(z<0.)] = alpha
        return temp
    
    def tanh(self,z):
        return np.tanh(z)
    
    def tanh_derivative(self,z):
        return 1. - np.tanh(z)**2
    
    """
    defining metrics
    """

    def MSE(self, target, pred):
        # calculate Mean Squared Error
        return np.mean( np.sum((pred-target)**2, axis=1) )
    
    def accuracy(self, target, pred):
        # calculates accuracy (percent correctly predicted)
        pred = pred >=0.5 # round values to 0 or 1
        return np.mean(pred.ravel()==target.ravel())
    
    def R2(self, target, pred):
        return 1. - self.MSE(target, pred)/np.var(target)
    
    