# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

Plotting classification with NNs

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

learning_rate_schedules = ['adagrad', 'rmsprop', 'adam']
activation_functions = ['sigmoid', 'RELU', 'tanh']

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(7,10.5))
for i in [1,2,3,4]:
    df = pd.read_csv(f'classification_accuracy_architecture_{i}.csv')
    lambdas = df['$\lambda$'].values
    for j, schedule_name in enumerate(learning_rate_schedules):
        ax = axs[i-1,j]
        for k,activation_function in enumerate(activation_functions):
            acc = df[f'{schedule_name},{activation_function}'].values
            ax.plot(lambdas,acc,label=f'{k}')
        ax.legend()
        axs[0,j].set_title(schedule_name)
        axs[-1,j].set_xlabel('$\lambda$')
        axs[i-1,j].set_xscale('log')
        axs[i-1,j].set_ylim([0.45,1.05])
    axs[i-1,0].set_ylabel('Accuracy')
        
fig.tight_layout()

fig.savefig('Classification NNs.pdf')