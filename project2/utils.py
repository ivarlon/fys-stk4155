# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

Some useful functions

"""

import numpy as np

def make_design_matrix(x,n_features):
    '''
    Creates the design matrix.
    
    Parameters
    ----------
    x : ndarray
        array of input values.
    n_features : int
        no. of features.

    Returns
    -------
    X : 2d array
        Design matrix with a row corresponding to a datapoint, a column to a feature.

    '''
    X = np.stack([x**i for i in range(n_features)], axis = 1)
    if X.ndim == 3:
        X = X.reshape(X.shape[0],X.shape[1])
    return X

def ridge_regression(X,y,lamb):
    '''
    Performs ridge regression

    Parameters
    ----------
    X : 2d array
        design matrix.
    y : 1d array
        target data.
    lamb : float
        Regularisation parameter.

    Returns
    -------
    beta : 1d array
        Optimal parameters.

    '''
    n, p = X.shape
    beta = np.linalg.inv(X.T @ X + n*lamb*np.eye(p)) @ (X.T @ y)
    return beta


def OLS_regression(X,y):
    '''
    Ordinary least squares
    '''
    return ridge_regression(X,y,0.)


def test_function(x,coeffs,noiselevel=1.0):
    '''
    Generates a test function.

    Parameters
    ----------
    x : ndarray
        input values
    coeffs : list
        list of coefficients
    noiselevel : float, optional
        determines the size of the stochastic noise. The default is 1.0.

    Returns
    -------
    y : ndarray
        output data (polynomial function of x + noise).
    '''
    y = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        y = y + coeff*x**i
    y = y + noiselevel*np.random.randn(y.shape[0]).reshape(y.shape)
    return y

def accuracy(target, pred):
        # calculates accuracy (percent correctly predicted)
        pred = pred >=0.5 # round values to 0 or 1
        return np.mean(pred.ravel()==target.ravel())