#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import kernels
import pandas as pd

class KLROneVsAll:
    
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
        self.alphas = None
        self.param = param
        self.X = None
        
    def logistic_loss(self, y, f_x):
        return np.log(1 + np.exp(-y*f_x))
    
    def loss(self, Y, K, alpha, lambd):
        k_alpha = np.dot(K, alpha)
        return np.mean(self.logistic_loss(Y, k_alpha)) + (lambd/2)*np.dot(alpha.T, k_alpha)

    def gradient_descent(self, K, Y, eta, lambd, max_iter):
        n = Y.shape[0]
        alpha = np.zeros(n)
        step = 0
        loss_ = []
        steps = []
        for _ in tqdm(range(max_iter)):
            k_alpha = np.dot(K, alpha)
            alpha = alpha + eta * np.dot(K, Y*(kernels.sigmoid(-Y*k_alpha))) / n - eta * lambd * k_alpha
            loss_.append(self.loss(Y, K, alpha, lambd))
            steps.append(step)
            step += 1
        loss_steps = pd.DataFrame({
        'step': steps, 
        'loss': loss_
        })
        return alpha, loss_steps    

    def fit(self, X, Y, eta = 0.01, lambd = 0.1, max_iter = 5000):
        print("==== Calculating the kernel matrix ====")
        self.X = X
        K = kernels.kernel_matrix(X, self.kernel, self.param)
        n = len(X)
        self.alphas = np.zeros((10, n))
        self.losses = []
        for i in range(10):
            print(f"==== Classifier {i} ====")
            Y_i = 2*(Y == i).astype(int) - 1
            alpha, loss_steps = self.gradient_descent(K, Y_i, eta, lambd, max_iter)
            self.alphas[i] = alpha
            self.losses.append(loss_steps)
    

    def predict(self, Z):
        n, p = self.X.shape[0], Z.shape[0]
        K = np.zeros((p, n))
        for i in tqdm(range(p)):
            for j in range(n):
                K[i][j] = self.kernel(Z[i], self.X[j], self.param)
        Y_predict = np.dot(K, self.alphas.T)
        Y_predict = np.argmax(Y_predict, axis = 1)
        return Y_predict
    
    def loss_plot(self, n = 0):
        return self.losses[n].plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )


class KLROneVsOne:
    
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
        self.alphas = None
        self.param = param
        self.X = None
        
    def logistic_loss(self, y, f_x):
        return np.log(1 + np.exp(-y*f_x))
    
    def loss(self, Y, K, alpha, lambd):
        k_alpha = np.dot(K, alpha)
        return np.mean(self.logistic_loss(Y, k_alpha)) + (lambd/2)*np.dot(alpha.T, k_alpha)

    def gradient_descent(self, K, Y, eta, lambd, max_iter):
        n = Y.shape[0]
        alpha = np.zeros(n)
        step = 0
        loss_ = []
        steps = []
        for _ in tqdm(range(max_iter)):
            k_alpha = np.dot(K, alpha)
            alpha = alpha + eta * np.dot(K, Y*(kernels.sigmoid(-Y*k_alpha))) / n - eta * lambd * k_alpha
            loss_.append(self.loss(Y, K, alpha, lambd))
            steps.append(step)
            step += 1
        loss_steps = pd.DataFrame({
        'step': steps, 
        'loss': loss_
        })
        return alpha, loss_steps    

    def fit(self, X, Y, eta = 0.01, lambd = 0.1, max_iter = 5000):
        print("==== Calculating the kernel matrix ====")
        self.X = X
        K = kernels.kernel_matrix(X, self.kernel, self.param)
        n = len(X)
        self.alphas = np.zeros((10, n))
        self.losses = []
        
        self.ww = [[] for _ in range(10)]
        for i in range(10):
            print(f"==== Classifier {i} ====")
            for j in range(i + 1, 10):
                print(f"==== Classifier {i, j} ====")
                indices = np.arange(n)[np.logical_or(Y == i, Y == j)]
                Y_ij = Y[indices]
                Y_ij = 2*(Y_ij == i).astype(int) - 1
                K_ij = K[np.ix_(indices, indices)]
                alpha, loss_steps = self.gradient_descent(K_ij, Y_ij, eta, lambd, max_iter)
                self.ww[i].append((alpha.reshape(-1)))
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
    
    def loss_plot(self, n = 0):
        return self.losses[n].plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )
