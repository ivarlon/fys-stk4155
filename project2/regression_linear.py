# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

OLS + ridge regression using matrix inversion

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from gradientdescent import GradientDescent
from utils import test_function, make_design_matrix, ridge_regression

metric = 'R2' # or 'MSE'
if metric=='R2':
    metric_function = r2_score
elif metric=='MSE':
    metric_function = mean_squared_error

n = 1000
x = 0.8*np.random.randn(n).reshape(n,1)

# generate output data
coeffs = [2,0,2,2] # polynomial coeffs.
y = test_function(x, coeffs)

p = 5 # model complexity / no. of features
X = make_design_matrix(x,p)
X_train, X_test, y_train, y_test = train_test_split(X,y)

lambdas = np.logspace(-10,2,7)
lambdas[0] = 0. # lamb = 0, 1e-8, 1e-6, ..., 1e2 
# (lamb = 0 gives OLS)

score_matrix = np.zeros(shape=(len(lambdas),8))
columns = ['Analytic', 'GD', 'GD + mom.', 'SGD (plain)', 'Adagrad', 'Adagrad + mom.','RMSprop','Adam']
df = pd.DataFrame(score_matrix, index=lambdas, columns=columns)

fig_, ax_ = plt.subplots()
ax_.scatter(X_test[:,1][::n//200], y_test[::n//200], label="Test data")
x_sorted = np.sort(X_test[:,1])

# analytic (matrix inversion)
print("Solving analytically...")
for i,lamb in enumerate(lambdas):
    beta = ridge_regression(X_train,y_train,lamb)
    pred = X_test@beta
    test_score = metric_function(y_test, pred)
    df['Analytic'].iloc()[i] = test_score
    if i==0:
        X_test_sorted = make_design_matrix(x_sorted,p)
        y_pred = X_test_sorted@beta
        ax_.plot(x_sorted, y_pred, label="OLS analytic")


eta0 = 0.001
# running regular GD
print("Using plain GD...")
for i,lamb in enumerate(lambdas):
    gd = GradientDescent(eta0,schedule_name="plain",stochastic=False)
    beta = np.zeros(p).reshape(p,1)
    beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=lamb)
    pred = X_test@beta
    test_score = metric_function(y_test, pred)
    df['GD'].iloc()[i] = test_score
    if i==0:
        y_pred = X_test_sorted@beta
        ax_.plot(x_sorted, y_pred, label="OLS GD")

# running GD w/momentum
print("Using GD w/ momentum...")
for i,lamb in enumerate(lambdas):
    gd = GradientDescent(eta0,schedule_name="plain",momentum=0.8,stochastic=False)
    beta = np.zeros(p).reshape(p,1)
    beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=lamb)
    pred = X_test@beta
    test_score = metric_function(y_test, pred)
    df['GD + mom.'].iloc()[i] = test_score

# running SGD
print("Using SGD...")
for i,lamb in enumerate(lambdas):
    gd = GradientDescent(eta0,schedule_name="plain", stochastic=True)
    beta = np.zeros(p).reshape(p,1)
    beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=lamb)
    pred = X_test@beta
    test_score = metric_function(y_test, pred)
    df['SGD (plain)'].iloc()[i] = test_score

# running SGD + adagrad
print("Using SGD + adagrad...")
for i,lamb in enumerate(lambdas):
    # notice learning rate eta0 has to be made bigger
    gd = GradientDescent(50*eta0,schedule_name="adagrad", stochastic=True)
    beta = np.zeros(p).reshape(p,1)
    beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=lamb)
    print(beta)
    pred = X_test@beta
    test_score = metric_function(y_test, pred)
    df['Adagrad'].iloc()[i] = test_score
    if i==0:
        y_pred = X_test_sorted@beta
        ax_.plot(x_sorted, y_pred, label="OLS SGD+AG")

# running SGD + adagrad + mom.
print("Using SGD + adagrad + momentum...")
for i,lamb in enumerate(lambdas):
    gd = GradientDescent(50*eta0,schedule_name="adagrad", momentum=0.8, stochastic=True)
    beta = np.zeros(p).reshape(p,1)
    beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=lamb)
    pred = X_test@beta
    test_score = metric_function(y_test, pred)
    df['Adagrad + mom.'].iloc()[i] = test_score

# running SGD + rmsprop
print("Using SGD + RMSprop...")
for i,lamb in enumerate(lambdas):
    gd = GradientDescent(eta0,schedule_name="rmsprop", momentum=0.8, stochastic=True)
    beta = np.zeros(p).reshape(p,1)
    beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=lamb)
    pred = X_test@beta
    print(beta)
    test_score = metric_function(y_test, pred)
    df['RMSprop'].iloc()[i] = test_score
    if lamb==lambdas[-2]:
        y_pred = X_test_sorted@beta
        ax_.plot(x_sorted, y_pred, label=f"RMSprop $\lambda$={lamb}")

# running SGD + adam
print("Using SGD + Adam...")
for i,lamb in enumerate(lambdas):
    gd = GradientDescent(eta0,schedule_name="adam", momentum=0.8, stochastic=True)
    beta = np.zeros(p).reshape(p,1)
    beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=lamb)
    pred = X_test@beta
    test_score = metric_function(y_test, pred)
    df['Adam'].iloc()[i] = test_score
    
df.to_csv("regression_linear.csv")

ax_.set_xlabel("x"); ax_.set_ylabel("y")
ax_.legend()
fig_.tight_layout()
fig_.savefig("regression_plots.pdf")
plt.show()

hm = sns.heatmap(df.round(2), cbar_kws={'label': metric})
hm.set(ylabel='$\lambda$')
fig = hm.get_figure()
fig.tight_layout()
fig.savefig(f'linear_regression_{metric}.pdf')
plt.show()

'''
Now seeing how performance of SGD depends on batch size and no. of epochs
'''

batch_sizes = [5,10,20]
n_epoch_list = [10,30,50,70,100]
n_runs = 3
MSEs = np.zeros(shape=(len(batch_sizes),len(n_epoch_list)))
for i,n_epochs in enumerate(n_epoch_list):
    for j,batch_size in enumerate(batch_sizes):
        gd = GradientDescent(eta0,schedule_name="rmsprop",stochastic=True, batch_size=batch_size, n_epochs=n_epochs)
        for _ in range(n_runs):
            beta = np.zeros(p).reshape(p,1)
            beta = gd.gradient_descent(X_train, y_train, beta, problem="ridge_regression", lamb=0.)
            pred = X_test@beta
            MSEs[j,i] += mean_squared_error(y_test, pred)
        # take average
        MSEs[j,i] /= n_runs

fig2, ax = plt.subplots()
for i, batch_size in enumerate(batch_sizes):
    ax.plot(n_epoch_list, MSEs[i], label=f'{batch_size}')
ax.set_xlabel("No. of epochs"); ax.set_ylabel("MSE")
ax.legend()
fig2.tight_layout()
fig2.savefig('SGD_n_epochs.pdf')

plt.show()