#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import kernels


class KRROneVsOne:
    
    def __init__(self, kernel, param):
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
        self.alphas = []
        self.param = param
        
    def fit(self, X, Y, lambdaa):
        n = X.shape[0]
        print("==== Calculating the kernel matrix ====")
        self.X = X
        self.Y = Y
        K = kernels.kernel_matrix(X, self.kernel, self.param)
        self.ww = [[] for _ in range(10)]
        for i in range(10):
            print(f"==== Classifier {i} ====")
            for j in range(i + 1, 10):
                print(f"==== Classifier {i, j} ====")
                indices = np.arange(n)[np.logical_or(Y == i, Y == j)]
                Y_ij = Y[indices]
                Y_ij = 2*(Y_ij == i).astype(int) - 1
                K_ij = K[np.ix_(indices, indices)]
                n_ij = len(indices)
                term1 = np.linalg.inv(K_ij + lambdaa*n_ij*np.identity(n_ij))
                w = (np.dot(term1, Y_ij))
                self.ww[i].append((w.reshape(-1)))
                print("Status: Calculated Correctly")
            
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
                Y_predict = np.dot(K_ij, self.ww[i][j-i-1].T)
                Y_predict = Y_predict >= 0
                prediction[:,i] = prediction[:,i] + Y_predict
                prediction[:,j] = prediction[:,j] + 1 - Y_predict
        Y_final = np.argmax(prediction, axis = 1)
        return Y_final
    
    

class KRROneVsAll:
    
    def __init__(self, kernel, param):
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
        self.alphas = []
        self.param = param
        
    def fit(self, X, Y, lambdaa):
        n = X.shape[0]
        print("==== Calculating the kernel matrix ====")
        self.X = X
        K = kernels.kernel_matrix(X, self.kernel, self.param)
        commun_term = np.linalg.inv(K + lambdaa*n*np.identity(n))
        for i in range(10):
            print(f"==== Classifier {i} ====")
            Y_i = 2*(Y == i).astype(int) - 1
            self.alphas.append(np.dot(commun_term, Y_i))
            print("Status: Calculated Correctly")
    
    def predict(self, Z):
        n, p = self.X.shape[0], Z.shape[0]
        K = np.zeros((p, n))
        for i in tqdm(range(p)):
            for j in range(n):
                K[i][j] = self.kernel(Z[i], self.X[j], self.param)
        Y_predict = np.dot(K, np.array(self.alphas).T)
        Y_predict = np.argmax(Y_predict, axis = 1)
        return Y_predict
    
