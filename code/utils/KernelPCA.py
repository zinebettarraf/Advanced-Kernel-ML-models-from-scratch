#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import kernels



class KernelPCA:
    """
    Kernel PCA
    """
    
    def __init__(self, kernel, param, n_components):
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
        self.X = None
        self.n_components = n_components
        self.vectors = None
        self.K = None
        
    def fit(self, X, Z):
        print("==== Calculating the kernel matrix ====")
        X = np.concatenate([X, Z])
        n, p = X.shape
        self.X = X
        self.K = kernels.kernel_matrix(self.X, self.kernel, self.param)
        #Computing the principal components
        val, vec = np.linalg.eigh(self.K)
        val, vec = val[- self.n_components:], vec[:, - self.n_components:]
        for i in range(self.n_components):
            vec[i] /= np.sqrt(val[i])
        self.vectors = vec
    
    def fit_transform(self, X, Z):
        self.fit(X, Z)
        return self.vectors
    
    def transform(self, Z):
        n, p = self.X.shape[0], Z.shape[0]
        K = np.zeros((p, n))
        for i in tqdm(range(p)):
            for j in range(n):
                K[i][j] = self.kernel(Z[i], self.X[j], self.param)
        Z_pca = np.dot(K, self.vectors)
        return Z_pca