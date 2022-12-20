# FYS-STK4150
# Project 3
## Solving PDEs with NNs
This repo contains the following files:
#### forwardeuler.py
This script solves the diffusion equation using the forward Euler scheme. It plots the solution for two step sizes dx = 0.1, 0.01, each at two different time points. It also prints the mean relative error at these points.

#### NN_solve_diffusion.py
This script solves the diffusion equation using a neural network method. It uses a trial function that depends on the output of a NN. It plots the numerical solution, the analytical solution, and their difference, and prints the relative error in the numerical solution.

#### NN_compute_eigenvector.py
This solves a system of ODEs in order to estimate an eigenvector (and corresponding eigenvalue) of a symmetric matrix. It plots the time evolution of the trial solution and compares with the actual eigenvector.

#### useful_functions.py
This script contains some functions that both NN_solve_diffusion.py and NN_compute_eigenvector.py use; namely a method to initialise the neural network parameters, in addition to some activation functions such as the sigmoid.
