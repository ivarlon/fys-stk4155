# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

Calculates classification accuracy for various setups of a neural network

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

from gradientdescent import GradientDescent
from neuralnetwork import NeuralNetwork

from utils import test_function, accuracy

# generate input data
X, y = load_breast_cancer(return_X_y=True)

# scale data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y)

architectures = [[15,8],
                 [10,5],
                 [5],
                 [8]]

learning_rate_schedules = ['adagrad', 'rmsprop', 'adam'] # sklearn nn uses adam only

activation_functions = ['sigmoid', 'RELU', 'tanh']

lambdas = np.logspace(-6,4,6)
lambdas[0] = 0. # lamb = 0, 1e-4, 1e-2, ..., 1e4

n_loops = 1 # no. of loops to improve statistics

classification_nn = lambda nodes_per_layer, lamb, activation_function: NeuralNetwork(X.shape[1], 
                                                                                 1, 
                                                                                 nodes_per_layer, 
                                                                                 lamb=lamb, 
                                                                                 activation_functions=[activation_function]*len(nodes_per_layer) + ["sigmoid"],
                                                                                 classification=True)


def run_classification(nodes_per_layer, lamb, activation_function, schedule_name, try_count):
    nn = classification_nn(nodes_per_layer, lamb, activation_function)
    if schedule_name=="adagrad":
        eta0 = 0.1
    else:
        eta0 = 0.001
    nn.train(X_train, y_train, learning_schedule=schedule_name, momentum=0.8, eta=eta0, stochastic=True)
    pred = nn.predict(X_test)
    if pred.all() == 0. and try_count<=3:
        # rerun if parameters got all messed up and predicted only zeros
        # try only 3 times, otherwise give up xD
        try_count += 1
        print("Bad convergence, rerunning", try_count)
        return run_classification(nodes_per_layer, lamb, activation_function, schedule_name, try_count)
    else:
        return pred


k=1
# looping
for nodes_per_layer in architectures:
    df = pd.DataFrame()
    df['$\lambda$'] = lambdas
    print("k",k)
    for schedule_name in learning_rate_schedules:
        
        for activation_function in activation_functions:
            
            df[f'{schedule_name},{activation_function}'] = np.zeros_like(lambdas)
            
            for idx, lamb in enumerate(lambdas):
                print("idx",idx)
                
                for i in range(n_loops):
                    try_count = 0
                    print(nodes_per_layer, lamb, activation_function, schedule_name)
                    pred = run_classification(nodes_per_layer, lamb, activation_function, schedule_name, try_count)
                    acc = accuracy(y_test, pred)
                    df[f'{schedule_name},{activation_function}'][idx] += acc
                
                df[f'{schedule_name},{activation_function}'][idx] /= n_loops
    df.to_csv(f'classification_accuracy_architecture_{k}.csv')
    k+=1