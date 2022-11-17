# -*- coding: utf-8 -*-
"""
FYS-STK4155
Project 2

Gradient descent class

"""

import numpy as np

class GradientDescent:
    def __init__(self, eta0, 
                 schedule_name="plain", 
                 rho = 0.2, 
                 rho1 = 0.9, 
                 rho2 = 0.999,                 
                 momentum=None, 
                 stochastic=True, 
                 batch_size=10,
                 n_epochs=50):
        '''
        Gradient Descent class. Method gradient_descent performs the actual GD.

        Parameters
        ----------
        eta0 : float
            learning rate.
        schedule_name : str, optional
            The desired learning rate schedule. Supported values: 'plain', 'linear', 'decay', 'exp_decay', 'adagrad', 'rmsprop', 'adam'
            The default is "plain".
        rho : float, optional
            Decay parameter in rmsprop. The default is 0.2.
        rho1 : float, optional
            Decay parameter for first moment of gradient in adam. The default is 0.9.
        rho2 : float, optional
            Decay parameter for second moment of gradient in adam. The default is 0.999.
        momentum : float, optional
            Momentum used in GD with momentum. The default is None.
        stochastic : bool, optional
            If True, runs stochastic GD, else regular GD. The default is True.
        batch_size : int, optional
            Size of minibatches used in SGD. The default is 10.
        n_epochs : int, optional
            No. of epochs used in SGD. The default is 50.

        '''
        
        self.eta0 = eta0
        self.delta = 1e-7
        
        self.schedule_name = schedule_name
        if schedule_name == "rmsprop":
            self.rho = rho
        elif schedule_name == "adam":
            self.rho1 = rho1
            self.rho2 = rho2
        
        self.momentum = momentum
        
        self.stochastic = stochastic
        if stochastic:
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            # set random seed
            #np.random.seed(0)
    
    
    def gradient_descent(self, X, y, beta, problem, n_iter=1000, lamb=None):
        '''
        Performs the gd algorithm to find optimal model parameters
        
        inputs:
            X (ndarray) : array of input data points
            y (ndarray) : array of output data points
            beta (ndarray) : array of parameter values
            problem (str or NeuralNetwork instance) : "OLS_regression", "ridge_regression", "logistic_regression" or neural network
            n_iter (int) : no. of iterations in case of regular GD
            lamb (float) : regularisation hyperparameter
        
        outputs:
            beta (ndarray) : array of parameter values
        '''
        n = X.shape[0]
        
        delta_beta = np.zeros_like(beta)

        # define gradient
        
        if problem == "OLS_regression":
            gradient = lambda beta, X, y: 2./X.shape[0] * X.T @ ( X@beta - y)

        elif problem == "ridge_regression":
            gradient = lambda beta, X, y: 2./X.shape[0] * X.T @ ( X@beta - y) + 2.*lamb*beta

        elif problem == "logistic_regression":
            def gradient(beta, X, y):
                # calculating probability vector using sigmoid function
                predictions = 1./(1. + np.exp(-X@beta))
                g = X.T @ (predictions - y.reshape(predictions.shape)) + 2*lamb*beta
                return g
        else:
            try:
                nn = problem
                lamb = nn.lamb
                def gradient(beta, X, y):
                    # first set new parameters in the network
                    nn.W_list = [W_l for W_l in beta[0]]
                    nn.b_list = [b_l for b_l in beta[1]]
                    
                    delta_list = nn.backpropagation(X,y)
                    """
                    delta[i] = layer i: n samples
                    delta[i][j] = layer i: jth sample
                    """
                    node_values = nn.node_values
                    
                    # compute outer product (get array of size L x n x n(l-1) x n(l) )
                    # then compute mean to get gradient of weight matrix
                    dW = [np.mean(node_values[l][:,:,np.newaxis]*delta_list[l][:,np.newaxis,:],axis=0) + 2.*lamb*nn.W_list[l]
                          for l in range(nn.L)]
                    dW = np.array([dW_ for dW_ in dW], dtype=object)
                    db = [np.mean(delta_list[l], axis=0) + 2.*lamb*nn.b_list[l] for l in range(nn.L)]
                    db = np.array(db, dtype=object)
                    return np.array([dW, db], dtype=object)
            except:
                print("Invalid value for parameter 'problem'. Allowed values: 'OLS_regression', 'ridge_regression', 'logistic_regression', instance of NeuralNetwork.")
                return
        
        
        """
        set learning rate schedule
        """
        
        if self.schedule_name=="plain":
            schedule = lambda g, i: self.eta0*g
            
        elif self.schedule_name=="linear":
            eta_tau = self.eta0/100
            tau = self.n_epochs*n
            def schedule(g,i):
                alpha = i/tau
                eta = (1. - alpha)*self.eta0 + alpha*eta_tau
                return eta*g
            
        elif self.schedule_name=="decay":
            eta_tau = self.eta0/100
            def schedule(g,i):
                return self.eta0 * g /(1. + i*eta_tau)
        
        elif self.schedule_name=="exp_decay":
            eta_tau = self.eta0/100
            def schedule(g,i):
                return self.eta0 * np.exp(-i*eta_tau) * g
        
        elif self.schedule_name=="adagrad":
            # initialise (diagonal of) r matrix
            self.r_diag = np.zeros_like(beta)
            def schedule(g,i):
                # update r matrix
                self.r_diag += g**2
                # evaluate learning rate
                eta = self.eta0/(self.r_diag**0.5 + self.delta)
                return eta*g
        
        elif self.schedule_name=="rmsprop":
            # initialise (diagonal of) r matrix
            self.r_diag = np.zeros_like(beta)
            def schedule(g,i):
                # update r matrix
                self.r_diag = self.rho*self.r_diag + (1-self.rho) * g**2
                # evaluate learning rate
                eta = self.eta0/(self.r_diag**0.5 + self.delta)
                return eta*g
        
        elif self.schedule_name=="adam":
            # initialise (diagonal of) r matrix (in denominator of learning rate)
            self.r_diag = np.zeros_like(beta)
            # initialise (diagonal of) s matrix (in numerator of learning rate)
            self.s_diag = np.zeros_like(beta)
            def schedule(g,i):
                i += 1
                # update s matrix
                self.s_diag = self.rho1*self.s_diag + (1-self.rho1) * g
                # update r matrix
                self.r_diag = self.rho2*self.r_diag + (1-self.rho2) * g**2
                # evaluate learning rate
                s_hat = self.s_diag/(1 - self.rho1**i)
                r_hat = self.r_diag/(1 - self.rho2**i)
                eta = self.eta0 * s_hat / (r_hat**0.5 + self.delta)
                return eta
        
        else:
            print("You have to input a supported learning schedule")
            return
        
        def check_convergence(delta_beta, i):
            # checks if gradient descent has stopped
            eps = 1e-5
            if np.linalg.norm(delta_beta, ord=np.inf) < eps:
                print(f"Converged at iter. no. {i}")
                return True
            else:
                return False
        
        """
        Stochastic GD
        """
        
        if self.stochastic:
            
            m = int(n/self.batch_size) # no. of mini-batches
            
            if self.momentum:
                # SGD w/ momentum
                for epoch in range(self.n_epochs):
                    for i in range(m):
                        selection = np.random.randint(n,size=self.batch_size)
                        X_i, y_i = X[selection], y[selection]
                        g = gradient(beta, X_i, y_i)
                        iteration = epoch*m + i
                        eta_g = schedule(g,iteration)
                        delta_beta = eta_g + self.momentum * delta_beta
                        beta -= delta_beta
            
            else:
                # SGD w/o mom.
                for epoch in range(self.n_epochs):
                    for i in range(m):
                        selection = np.random.randint(n,size=self.batch_size)
                        X_i, y_i = X[selection], y[selection]
                        g = gradient(beta, X_i, y_i)
                        iteration = epoch*m + i
                        eta_g = schedule(g,iteration)
                        delta_beta = eta_g
                        beta -= delta_beta

        
        else:
            """
            Regular GD
            """
            
            # define gradient on basis of all datapoints
            old_gradient = gradient
            gradient = lambda beta: old_gradient(beta, X, y)
            
            # regular GD w/ mom.
            if self.momentum:
                for i in range(n_iter):
                    g = gradient(beta)
                    eta_g = schedule(g,i)
                    delta_beta = eta_g + self.momentum * delta_beta
                    beta -= delta_beta
                    if check_convergence(delta_beta,i):
                        break
            
            # regular GD w/o mom.
            else:
                for i in range(n_iter):
                    g = gradient(beta)
                    eta_g = schedule(g,i)
                    delta_beta = eta_g
                    beta -= delta_beta
                    if check_convergence(delta_beta,i):
                        break
        
        return beta
