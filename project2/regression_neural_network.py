# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

Calculates regression MSE for various setups of a neural network

uses n_loops loops when calculating the MSE in order to reduce uncertainty

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gradientdescent import GradientDescent
from neuralnetwork import NeuralNetwork

from utils import test_function

# generate input data
n = 1000
x = 0.8*np.random.randn(n).reshape(n,1)

# generate output data
coeffs = [2,0,2,2] # polynomial coeffs.
y = test_function(x, coeffs)

x_train, x_test, y_train, y_test = train_test_split(x,y)

architectures = [[3,2],
                 [5,3],
                 [5],
                 [3]]

learning_rate_schedules = ['adagrad', 'rmsprop', 'adam'] # sklearn nn uses adam only

activation_functions = ['sigmoid', 'RELU', 'tanh']

lambdas = np.logspace(-10,2,7)
lambdas[0] = 0. # lamb = 0, 1e-8, 1e-6, ..., 1e2

n_loops = 3 # no. of loops to improve statistics

regression_nn = lambda nodes_per_layer, lamb, activation_function: NeuralNetwork(x.shape[1], 
                                                                                 y.shape[1], 
                                                                                 nodes_per_layer, 
                                                                                 lamb=lamb, 
                                                                                 activation_functions=activation_function)

def run_regression(nodes_per_layer, lamb, activation_function, schedule_name, try_count):
    nn = regression_nn(nodes_per_layer, lamb, activation_function)
    if schedule_name=="adagrad":
        eta0 = 0.1
    else:
        eta0 = 0.001
    nn.train(x_train, y_train, learning_schedule=schedule_name, momentum=0.8, eta=eta0, stochastic=True)
    pred = nn.predict(x_test)
    if pred.all() == 0. and try_count <=5:
        # rerun if parameters got all messed up and predicted only zeros
        # try only 5 times to avoid potentially infinite loop
        try_count += 1
        print("Bad convergence, rerunning", try_count)
        return run_regression(nodes_per_layer, lamb, activation_function, schedule_name, try_count)
    else:
        return nn, pred


k=2
# looping
for nodes_per_layer in architectures[1:]:
    # create dataframes for the metrics
    df_MSE = pd.DataFrame()
    df_R2 = pd.DataFrame()
    df_MSE['$\lambda$'] = lambdas
    df_R2['$\lambda$'] = lambdas
    print("k",k)
    for schedule_name in learning_rate_schedules:
        
        for activation_function in activation_functions:
            
            df_MSE[f'{schedule_name},{activation_function}'] = np.zeros_like(lambdas)
            df_R2[f'{schedule_name},{activation_function}'] = np.zeros_like(lambdas)
            
            for idx, lamb in enumerate(lambdas):
                print("idx",idx)
                
                for i in range(n_loops):
                    try_count = 0
                    nn, pred = run_regression(nodes_per_layer, lamb, activation_function, schedule_name, try_count)
                    MSE_test = nn.MSE(y_test, pred)
                    R2_test = nn.R2(y_test, pred)
                    df_MSE[f'{schedule_name},{activation_function}'][idx] += MSE_test
                    df_R2[f'{schedule_name},{activation_function}'][idx] += R2_test
                df_MSE[f'{schedule_name},{activation_function}'][idx] /= n_loops
                df_R2[f'{schedule_name},{activation_function}'][idx] /= n_loops
    df_MSE.to_csv(f'regression_MSE_architecture_{k}.csv')
    df_R2.to_csv(f'regression_R2_architecture_{k}.csv')
    k+=1