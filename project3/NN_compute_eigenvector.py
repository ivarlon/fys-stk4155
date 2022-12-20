# -*- coding: utf-8 -*-
"""
FYS-STK4155

Project 3

Finding eigenvectors of symmetric matrix by solving an ODE with a neural network
"""

import autograd.numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
from tqdm import trange
from useful_functions import *

def neural_network(params, t, activation_fns):
    n_points = np.size(t,0)
    L = len(params)
    
    # ============
    # feed forward
    # ============
    
    a_l = t
    
    for l in range(L):
        # add vector of ones in order to include bias
        a_l = np.c_[a_l, np.ones(n_points).reshape(n_points,1)]
        P_l = params[l] # = [W_l^T  b_l] (an N_(l) x (N_(l-1)+1) matrix )
        a_l = a_l@P_l.T
        a_l = activation_fns[l](a_l)
    
    return a_l

#x0 = np.array([-0.42470757,  0.04838288, -0.28953088,  0.77230282, -0.36467259, -0.06337738]).reshape(n,1)
def x_trial(t, params, activation_fns, x0):
    return np.outer((1-t),x0.T) + t*neural_network(params, t, activation_fns)

def cost(t, params, activation_fns, A, x0):
    n = A.shape[0]
    Nt = t.shape[0]
    dx_t = elementwise_grad(x_trial, 0)(t, params, activation_fns, x0)
    x_t = x_trial(t, params, activation_fns, x0)
    x_squared = np.einsum('ij,ij->i', x_t, x_t).reshape(Nt,1)
    xAx = np.einsum('ij,ij->i', x_t, x_t@A).reshape(Nt,1)
    #print(x_squared.shape, x_t.shape, A.shape, xAx.shape)
    f = (x_squared*x_t) @ A + (1 - xAx) * x_t
    ODE = dx_t + x_t - f
    return np.mean( np.einsum('ij,ij->i', ODE, ODE) )


def solve_eigval_problem(A, x0, t_train, params, activation_fns, n_epochs, batch_size, eta):
    
    L = len(params) # no. of layers excl. input layer
    Nt = t_train.shape[0]
    n_batches = Nt//batch_size
    
    print(f"Initial cost = {cost(t_train, params, activation_fns, A, x0)}")
    
    # ================
    # gradient descent
    # ================
    
    grad_P = grad(cost, 1)
    for epoch in trange(n_epochs):
        #print(f"{epoch/n_epochs*100}%")

        for i in range(n_batches):
            t_sel = np.random.randint(Nt, size=batch_size)
            t = t_train[t_sel]
            # evaluate gradient(s)
            gradient = grad_P(t, params, activation_fns, A, x0)

            for l in range(L):
                # update parameters for each layer
                params[l] = params[l] - eta*gradient[l]
            
            if epoch%(n_epochs//min(10,n_epochs)) == 0:
                c = cost(t_train, params, activation_fns, A, x0)
                if np.isnan(c):
                    return
                print("Cost =", c)
    
    return params


if __name__ == "__main__":
    
    Nt = 200
    t = np.linspace(0,1,Nt).reshape(Nt,1)
    n_neurons_list = [4,8]
    activation_fns = [sigmoid, sigmoid, lambda z: z]
    n_epochs = 2000
    batch_size = 20
    eta = 0.001
    
    n = 6
    np.random.seed(0)
    Q = np.random.randn(n,n)
    A = 0.5*(Q.T + Q)
    
    #x0 = np.random.randn(n).reshape(n,1)
    x0 = np.array([1.23029068, 1.20237985, -0.38732682, -0.30230275, -1.04855297, -1.42001794]).reshape(n,1)
    
    input_shape = 1 # time t
    output_shape = n # vector x
    params = initialise_params(n_neurons_list, input_shape, output_shape)
    
    params = solve_eigval_problem(A, x0, t, params, activation_fns, n_epochs, batch_size, eta)
    
    if type(params) is type(None):
        print("Parameters diverged. Retune hyperparameters or choose a different x0")
    
    else:
        eigvals, eigvecs = np.linalg.eigh(A)
        
        x_tilde = x_trial(np.array([[1.]]), params, activation_fns, x0).reshape(n,1)
        x_tilde = x_tilde/np.sqrt(x_tilde.T@x_tilde) # normalise
        
        lambda_tilde = x_tilde.T@(A@x_tilde)
        
        # check to see if this is an eigenvector
        found_eigenvector = np.mean((A@x_tilde - lambda_tilde*x_tilde)**2) < 1e-2
        
        if found_eigenvector:
            print("Found eigenvector")
            print(x_tilde.T)
            print("Eigenvalue", lambda_tilde)
            print()
            
            idx = np.argmin(np.abs(lambda_tilde-eigvals)) # find idx of eigenvalue
            
            # check if estimated eigenvector has flipped sign compared to numpy eigvec
            if np.dot(x_tilde.ravel(), eigvecs[:,idx]) < 0:
                eigvecs[:,idx] *= -1
            
            print("Actual eigenvector")
            print(eigvecs[:,idx].T)
            print("Actual eigenvalue", eigvals[idx])
            print()
            
            difference = x_tilde.ravel() - eigvecs[:,idx] # deviation of estimated from actual eigenvector
            print("Error in eigenvector:", np.sqrt(np.dot(difference,difference)))
            print("Relative error in eigenval:", np.abs( (lambda_tilde - eigvals[idx])/eigvals[idx] ))
            
            # plot evolution of x_tr
            fig, ax = plt.subplots(figsize=(4.5,4.5))
            x_array = x_trial(t, params, activation_fns, x0)
            x_array2 = np.einsum('ij,ij->i', x_array, x_array) # squared norms
            x_array = x_array/np.sqrt(x_array2).reshape(Nt,1) # normalise
            
            for i in range(n):
                ax.plot(t, x_array[:,i])
            
            ax.set_yticks(eigvecs[:,idx])
            ax.set_xlabel("$t$")
            ax.set_ylabel("$\\tilde{x}_i$")
            ax.grid()
            fig.tight_layout()
            fig.savefig("x_evolution.pdf")
            plt.show()