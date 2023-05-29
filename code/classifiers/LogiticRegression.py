#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import kernels
import pandas as pd


def logistic_loss(y, f_x):
    return np.log(1 + np.exp(-y*f_x))

class LR():
    """
    class of logistic regression
    """
    def f(self, x, w):
        return w[0] + np.dot(x, w[1:])

    def loss(self, y, w, x, lambd):
        return np.mean(logistic_loss(y, self.f(x, w))) + lambd/2*(np.sum(np.square(w[1:])))

    def gradient_descent(self, X, Y, eta, lambd, max_iter = 5000):
        n = len(X)
        p = len(X[0])
        #X = np.concatenate((np.ones((n, 1)), X), axis = 1)
        w = np.random.rand(p + 1).T
        step = 0
        steps = []
        loss_ = []
        w[0] = 0
        for _ in tqdm(range(max_iter)):
            w_0 = w[0] - eta * np.sum(Y*(kernels.sigmoid(Y*self.f(X,w)) -1))/n - eta*lambd*w[0]
            w[1:] = w[1:] - eta * np.dot(X.T, Y*(kernels.sigmoid(Y*self.f(X,w)) -1))/n - eta*lambd*w[1:]
            w[0] = w_0
            step += 1
            loss_.append(self.loss(Y, w, X, lambd))
            steps.append(step)
        loss_steps = pd.DataFrame({
        'step': steps, 
        'loss': loss_
        })
        return w, loss_steps
    
    def fit(self, X, Y, eta = 0.001, lambd = 0.1, max_iter = 7000):
        p = len(X[0])
        self.W = np.zeros((10, p+1))
        self.losses = []
        for i in range(10):
            print(f"==== Classifier {i} ====")
            Y_i = 2*(Y == i).astype(int) - 1
            w, loss = self.gradient_descent(X, Y_i, eta, lambd, max_iter)
            self.W[i] = w
            self.losses.append(loss)

    def predict(self, X):
        Y_predict = kernels.sigmoid(self.f(X, self.W.T))
        Y_predict = np.argmax(Y_predict, axis = 1)
        return Y_predict
    
        
    def loss_plot(self, n = 0):
        return self.losses[n].plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )