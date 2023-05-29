#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from cvxopt import matrix, solvers
import kernels

solvers.options['abstol'] = 1e-6
solvers.options['reltol'] = 1e-6
solvers.options['feastol'] = 1e-6

class SVMOneVsOne:

    def __init__(self, kernel, param, C):
        if kernel   == "polynomial":
            self.kernel = kernels.polynomial
        elif kernel == "gaussian":
            self.kernel = kernels.gaussian
        elif kernel == "laplacian":
            self.kernel = kernels.laplacian
        elif kernel == "khi":
            self.kernel = kernels.khi_2
        elif kernel == "GHI":
            self.kernel = kernels.generalized_hisogram_inter
        elif kernel == "log":
            self.kernel = kernels.log_kernel
        self.alphas = None
        self.param = param
        self.C = C
    
    def loss(self, K, Y, alpha):
        return np.dot(alpha.T, np.dot(K, alpha)) * .5 - np.dot(Y.T, alpha)
    
    def optimize(self, K, Y):
        n = len(Y)
        c = np.diag(Y)*1.
        Y = Y.reshape(-1,1) * 1.
        #Converting into cvxopt format - as previously
        P = matrix(np.dot(np.dot(c, K),c))
        q = matrix(-np.ones(n)*1.)
        G = matrix(np.vstack((np.eye(n)*-1., np.eye(n)*1.)))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        A = matrix(Y.reshape(1, -1))
        b = matrix(np.zeros(1))

        #Run solver
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol['x'])
    
    def fit(self, X, Y):
        print("==== Calculating the kernel matrix ====")
        self.X = X
        K = kernels.kernel_matrix(X, self.kernel, self.param)
        n = len(X)
        self.Y = Y
        self.W = np.zeros((10, n))
        self.losses = []
        self.ww = [[] for _ in range(10)]
        for i in range(10):
            print(f"==== Classifier {i} ====")
            for j in range(i+1, 10):
                print(f"==== Classifier {i, j} ====")
                indices = np.arange(n)[np.logical_or(Y == i, Y == j)]
                Y_ij = Y[indices]
                Y_ij = 2*(Y_ij == i).astype(int) - 1
                K_ij = K[np.ix_(indices, indices)]
                w = self.optimize(K_ij, Y_ij)
                self.ww[i].append((w.reshape(-1)* Y_ij))
                
            
    def predict(self, Z):
        n, p = self.X.shape[0], Z.shape[0]
        K = np.zeros((p, n))
        prediction = np.zeros((p, 10))
        
        for i in tqdm(range(p)):
            for j in range(n):
                K[i][j] = self.kernel(Z[i], self.X[j], self.param)
        
        for i in range(10):
            for j in range(i+1, 10):
                indices = np.arange(n)[np.logical_or(self.Y == i, self.Y == j)]
                K_ij = K[:, indices]
                print(K_ij.shape)
                Y_predict = np.dot(K_ij, self.ww[i][j-i-1].T)
                Y_predict = Y_predict >= 0
                prediction[:,i] = prediction[:,i] + Y_predict
                prediction[:,j] = prediction[:,j] + 1 - Y_predict
        Y_final = np.argmax(prediction, axis = 1)
        return Y_final
    
    

class SVMOneVsAll:

    def __init__(self, kernel, param, C):
        if kernel   == "polynomial":
            self.kernel = kernels.polynomial
        elif kernel == "gaussian":
            self.kernel = kernels.gaussian
        elif kernel == "laplacian":
            self.kernel = kernels.laplacian
        elif kernel == "khi":
            self.kernel = kernels.khi_2
        elif kernel == "GHI":
            self.kernel = kernels.generalized_hisogram_inter
        elif kernel == "log":
            self.kernel = kernels.log_kernel
        self.alphas = None
        self.param = param
        self.C = C
    
    def loss(self, K, Y, alpha):
        return np.dot(alpha.T, np.dot(K, alpha)) * .5 - np.dot(Y.T, alpha)
    
    def optimize(self, K, Y):
        n = len(Y)
        c = np.diag(Y)*1.
        Y = Y.reshape(-1,1) * 1.
        #Converting into cvxopt format - as previously
        P = matrix(np.dot(np.dot(c, K),c))
        q = matrix(-np.ones(n)*1.)
        G = matrix(np.vstack((np.eye(n)*-1., np.eye(n)*1.)))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        A = matrix(Y.reshape(1, -1))
        b = matrix(np.zeros(1))

        #Run solver
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol['x'])
    
    def fit(self, X, Y):
        print("==== Calculating the kernel matrix ====")
        self.X = X
        K = kernels.kernel_matrix(X, self.kernel, self.param)
        n = len(X)       
        self.W = np.zeros((10, n))
        self.losses = []
        for i in range(10):
            print(f"==== Classifier {i} ====")
            Y_i = 2*(Y == i).astype(int) - 1
            w = self.optimize(K, Y_i)
            self.W[i] = w.reshape(-1)
            self.W[i] = (self.W[i] * Y_i)
            
    def predict(self, Z):
        n, p = self.X.shape[0], Z.shape[0]
        K = np.zeros((p, n))
        for i in tqdm(range(p)):
            for j in range(n):
                K[i][j] = self.kernel(Z[i], self.X[j], self.param)
        Y_predict = np.dot(K, self.W.T)
        Y_predict = np.argmax(Y_predict, axis = 1)
        return Y_predict