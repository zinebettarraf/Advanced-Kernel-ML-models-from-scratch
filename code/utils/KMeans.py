#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

class KMeans:
    
    def __init__(self, k):
        self.k = k
        self.centers = None
        self.repartition = None
        self.X = None
    
    def update_centers(self):
        for i in range(self.k):
            indices_i = self.repartition == i
            self.centers[i] = np.sum(self.X[indices_i], axis = 0) / np.sum(indices_i)
    
    def update_repartition(self):
        n = len(self.X)
        for i in range(n):
            dist_min, cluster = 1e6, None
            for j in range(self.k):
                distance = np.mean(np.square(self.X[i] - self.centers[j]))
                if distance < dist_min:
                    dist_min, cluster = dist_min, j
            self.repartition[i] = cluster
    
    def fit(self, X, max_iter):
        n, m = X.shape
        self.X = X
        index_start = np.random.choice(range(n), self.k)
        self.centers = self.X[index_start]
        self.repartition = np.zeros(n)
        for i in tqdm(range(max_iter)):
            self.update_repartition()
            self.update_centers()
    
    def predict(self, Z):
        p = Z.shape[0]
        result = np.zeros((p, self.k))
        for i in range(p):
            for j in range(self.k):
                distance = np.mean(np.square(self.X[i] - self.centers[j]))
                result[i][j] = distance
        return result