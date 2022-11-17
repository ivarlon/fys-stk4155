# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

Running logistic regression w/ SGD
plotting accuracy as function of regularisation parameter lambda

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from utils import accuracy
from gradientdescent import GradientDescent

# generate input data
X, y = load_breast_cancer(return_X_y=True)

# scale data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y)

lambdas = np.logspace(-6,4,6)
#lambdas[0] = 0.
n_features = X.shape[1]
eta = 0.001
learning_schedule = "rmsprop"
momentum = 0.8
stochastic = True
n_epochs = 50
problem = "logistic_regression"
grad_desc = GradientDescent(eta, learning_schedule, momentum=momentum,stochastic=stochastic,n_epochs=n_epochs)

accuracy_list_own = np.zeros_like(lambdas)
accuracy_list_sklearn = np.zeros_like(lambdas)

n_loops = 10

important_features = np.zeros(n_features)

for i,lamb in enumerate(lambdas):
    # own lr
    for _ in range(n_loops):
        beta = np.zeros(n_features).reshape(n_features,1)
        beta = grad_desc.gradient_descent(X_train,y_train,beta,problem=problem, lamb=lamb)
        pred = 1./(1. + np.exp(-X_test@beta))
        acc = accuracy(y_test, pred)
        accuracy_list_own[i] += acc
    accuracy_list_own[i] /= n_loops
    
    # get which features are associated with cancer risk
    # features w/ coefficient <30% of max coeff may be considered less important
    print("Important features are:")
    beta_max = np.max(np.abs(beta))
    for j in range(n_features):
        if np.abs(beta[j]) >= 0.3*beta_max:
            correlation = np.sign(beta[j])
            """
            if correlation>0:
                print(f"Feature {j+1}, associated with tumour being benign.")
            else:
                print(f"Feature {j+1}, associated with tumour being malignant.")
            """
            important_features[j] += correlation
    
    # scikit-learn lr
    sklearn_fit = LogisticRegression(fit_intercept=False, C=1./lamb, max_iter=1000).fit(X_train,y_train)
    accuracy_list_sklearn[i] = sklearn_fit.score(X_test,y_test)

"""
Now print feature names along with how frequently they were deemed important
"""
feature_names = load_breast_cancer()['feature_names']
for i,feature in enumerate(feature_names):
    print((feature + " "*40)[:30], important_features[i])


"""
Make accuracy plot
"""
fig, ax = plt.subplots()
ax.plot(lambdas, accuracy_list_own, label="Original code")
ax.plot(lambdas, accuracy_list_sklearn, label="Scikit-learn")
ax.legend()
ax.set_xlabel("$\lambda$"); ax.set_ylabel("Accuracy")
ax.set_title("Logistic regression")
ax.set_xscale('log')
#fig.savefig("Logistic_regression.pdf")
plt.show()