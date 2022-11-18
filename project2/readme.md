
# FYS-STK4155
## Project 2
## Regression and classification using Neural Networks and other models
This repository contains the following files:

### utils.py
Defines some useful functions: make_design_matrix, test_function (returns a polynomial function), ridge_regression and accuracy.

### neuralnetwork.py
This defines a NeuralNetwork class. More information can be found in its docstring.

### gradientdescent.py
This defines a GradientDescent class, which can call a gradient_descent method to optimise an inputted problem.

### regression_linear.py
This performs linear regression on polynomial data using the analytical solution as well as gradient descent. It also plots the MSE value for a model optimised by SGD as function of no. of epochs and batch size.

### regression_neural_network.py
This performs regression on polynomial data for a range of NN setups, including different numbers of nodes per layer, activation functions, regularisation strengths and learning rate schedules. It computes an average R2 and MSE score for each setup. It saves the data to .csv files.

### plot_regression_NN.py
This plots the data in the .csv files created by regression_neural_network.py

### logistic_regression.py
This performs logistic regression on the Wisconsin breast cancer dataset, using SGD. It also uses the scikit-learn logistic regression routine. It plots the prediction accuracy as function of regularisation strength.

### classification_neural_network.py
This performs regression on the breast cancer data for a range of NN setups, including different numbers of nodes per layer, activation functions, regularisation strengths and learning rate schedules. It computes the accuracy score for each setup. It saves the data to .csv files.

### plot_classification_NN.py
This plots the data in the .csv files created by classification_neural_network.py
