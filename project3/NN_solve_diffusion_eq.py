# -*- coding: utf-8 -*-
"""
FYS-STK4155

Project 3

Solving diffusion eq. using a neural network
"""

import autograd.numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import trange
from useful_functions import *
pi = 3.1415926535897932

def neural_network(params, x, t, activation_fns):
    '''
    Neural network function.
    
    Parameters
    ----------
    params : list of ndarrays
        This list contains the matrices containing weights and biases for each layer. Length = L
    x : ndarray
        Column vector of spatial points. length n_points
    t : ndarray
        Column vector of time points. length n_points
    activation_fns : list
        List containing the activation functions for each layer. Length = L

    Returns
    -------
    a_l : ndarray
        Array of outputs. Shape (n_points, 1)

    '''
    
    
    n_points = x.shape[0] # no. of points
    L = len(params) # no. of layers in NN
    
    # ============
    # feed forward
    # ============
    
    a_l = np.c_[x,t] # input layer
    
    for l in range(L):
        # add vector of ones in order to include bias
        a_l = np.c_[a_l, np.ones(n_points).reshape(n_points,1)]
        P_l = params[l] # = [W_l^T  b_l] (an N_(l) x (N_(l-1)+1) matrix )
        a_l = a_l@P_l.T
        a_l = activation_fns[l](a_l)
    
    return a_l

def u0(x):
    '''
    Initial condition function.
    '''
    
    return np.sin(pi*x)

def u_trial(x, t, params, activation_fns):
    '''
    Trial solution. 
    Sum of two terms: the first satisfies the initial conditions, the second depends on the NN.
    '''
    h2 = x*(1-x) * t * neural_network(params,x,t,activation_fns)
    return (1-t)*u0(x) + h2

def u_analytic(x,t):
    '''
    Analytical solution of diffusion equation.
    '''
    return np.exp(-pi**2 * t) * np.sin(pi*x)

def cost(x, t, params, activation_fns):
    '''
    Cost function.
    Equal to the squared diffusion equation.
    Returns the mean of the squared equation evaluated at all points (x_i, t_j)
    '''
    Nx = x.shape[0]
    Nt = t.shape[0]
    du_x = elementwise_grad(u_trial, 0)
    du_xx = elementwise_grad(du_x, 0)
    du_t = elementwise_grad(u_trial, 1)
    X, T = np.meshgrid(x,t)
    X = X.reshape(Nx*Nt,1)
    T = T.reshape(Nx*Nt,1)
    PDE = du_t(X, T, params, activation_fns) - du_xx(X, T, params, activation_fns)
    return np.mean( PDE**2 )



def solve_PDE(x_train, t_train, params, activation_fns, n_epochs, batch_size, eta, momentum=0.):
    '''
    Function that solves the PDE specified by the cost function.

    Parameters
    ----------
    x_train : ndarray
        Array of spatial points. Shape (N_x, 1)
    t_train : ndarray
        Array of timepoints. Shape (N_t, 1)
    params : list of ndarrays
        This list contains the matrices containing weights and biases for each layer. Length = L
    activation_fns : list
        List containing the activation functions for each layer. Length = L
    n_epochs : int
        No. of epochs for which the NN is trained
    batch_size : int
        Size of batch used to compute the gradient of the cost
    eta : float
        Learning rate
    momentum : float, optional
        Momentum used in the gradient descent. Non-zero values gave worse results. The default is 0..

    Returns
    -------
    params : list
        Optimised parameters for the NN.

    '''    
    L = len(params) # no. of layers excl. input layer
    Nx = x_train.shape[0] # no. of points
    Nt = t_train.shape[0] # no. of time points
    N_tot = Nx*Nt # total no. of points
    n_batches = N_tot//batch_size
    
    print(f"Initial cost = {cost(x_train, t_train, params, activation_fns)}")
    
    # ================
    # gradient descent
    # ================
    
    grad_P = grad(cost, 2)
    delta_params = [np.zeros_like(params_) for params_ in params] # initialise delta_P to include momentum
    
    for epoch in trange(n_epochs):
        for i in range(n_batches):
            x_sel = np.random.randint(Nx, size=batch_size)
            t_sel = np.random.randint(Nt, size=batch_size)
            x = x_train[x_sel]; t = t_train[t_sel] # randomly selected points
            
            # evaluate gradient(s)
            gradient = grad_P(x, t, params, activation_fns)

            # update parameters for each layer
            for l in range(L):
                delta_params[l] = eta*gradient[l] - momentum * delta_params[l]
                params[l] = params[l] - delta_params[l]
            
        if epoch%(n_epochs//min(10,n_epochs)) == 0:
            c = cost(x_train, t_train, params, activation_fns)
            if np.isnan(c): # check to see if GD has diverged
                return
            print("Cost =", c)
    
    return params


if __name__=="__main__":
    
    Nx = 10 # no. of spatial points
    Nt = 10 # no. of time points
    x = np.linspace(0.,1.,Nx)
    t = np.linspace(0.,1.,Nt)
    X, T = np.meshgrid(x,t) # grid of points
    
    n_neurons_list = [100, 25] # NN architecture
    activation_fns = [sigmoid, sigmoid, lambda z: z] # activation functions for each layer
    
    n_epochs = 25 # no. of epochs for SGD
    batch_size = 4 # size of mini batches
    eta = 0.01 # learning rate
    
    # initialise parameters of NN
    input_shape = 2 # tuples (x,t)
    output_shape = 1 # scalar u(x,t)
    params = initialise_params(n_neurons_list, input_shape, output_shape)
    
    # train NN
    params = solve_PDE(x, t, params, activation_fns, n_epochs, batch_size, eta)
    
    if type(params) is type(None):
        print("Parameters diverged. Retune hyperparameters")
    
    else:
        # calculate solution
        U = u_trial(X.reshape(Nx*Nt,1),T.reshape(Nx*Nt,1),params,activation_fns)
        U = U.reshape(Nt,Nx)
        
        U_anal = u_analytic(X,T) # analytic solution
        
        
        # =========
        # plotting
        # =========
        # NN solution
        fig1 = plt.figure(figsize=(6,6))
        ax1 = fig1.gca(projection='3d')
        ax1.plot_surface(T,X,U,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax1.set_xlabel('Time $t$')
        ax1.set_ylabel('Position $x$')
        ax1.set_title("NN solution")
        fig1.savefig("NN_diffusion.pdf")
        
        # analytic solution
        fig2 = plt.figure(figsize=(6,6))
        ax2 = fig2.gca(projection='3d')
        ax2.plot_surface(T,X,U_anal,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax2.set_xlabel('Time $t$')
        ax2.set_ylabel('Position $x$')
        ax2.set_title("Analytic solution")
        fig2.savefig("anal_diffusion.pdf")
        
        # difference
        fig3 = plt.figure(figsize=(6,6))
        ax3 = fig3.gca(projection='3d')
        ax3.plot_surface(T,X,U-U_anal,linewidth=0,antialiased=False,cmap=cm.viridis)
        ax3.set_xlabel('Time $t$')
        ax3.set_ylabel('Position $x$')
        ax3.set_title("Difference")
        fig3.savefig("difference_diffusion.pdf")
        
        # get mean relative error (not incl. boundaries or initial conditions)
        mean_rel_err = np.mean( np.abs( (U[1:,1:-1] - U_anal[1:,1:-1])/U_anal[0,1:-1] ) )
        print("Mean rel. error is", mean_rel_err)
        
        print("Max rel. error", np.max( np.abs( (U[1:,1:-1] - U_anal[1:,1:-1])/U_anal[0,1:-1] ) ))
        print("Min rel. error", np.min( np.abs( (U[1:,1:-1] - U_anal[1:,1:-1])/U_anal[0,1:-1] ) ))
        
        plt.show()