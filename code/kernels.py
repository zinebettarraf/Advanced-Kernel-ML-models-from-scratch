#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

#Some useful functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gaussian(x, y, gamma):
    return np.exp(- gamma*np.sum(np.square(x - y)))

def laplacian(x, y, gamma):
    return np.exp(- np.sum(np.abs(x - y))/gamma)

def polynomial(X, Y, degree):
    return (np.dot(X, Y.T))**degree

def khi_2(X, Y, param = None):
    return 2*np.sum(X*Y/(X+Y))

def generalized_hisogram_inter(x, y, alpha):
    x_alpha = np.power(np.abs(x), alpha)
    y_alpha = np.power(np.abs(y), alpha)
    return np.sum(np.minimum(x_alpha, y_alpha))

def log_kernel(x, y, d):
    return -np.log(np.sum(np.abs(x - y))**d + 1)


def kernel_matrix(X, f, param):
    """
    Generating the kernel matrix

    Parameters
    ----------
    X : Numpy array
        The dataset.
    f : function
        The choosen kernel.
    param : INT
        The parameter of the kernel.

    Returns
    -------
    matrix : Numpy array
        The kernel matrix.

    """
    n = X.shape[0]
    matrix = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i + 1):
            matrix[i][j] = f(X[i], X[j], param)
            matrix[j][i] = matrix[i][j]
    return matrix

