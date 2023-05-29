#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt


def visualize(X):
    """
    Function for the vizualisation

    Parameters
    ----------
    X : Numpy array
        Images.

    Returns
    -------
    None.

    """
    index = np.random.choice(X.index)
    image = X.loc[index].values
    pix = 32 * 32
    c = -min(image)
    d = .6
    R = (np.tanh(np.arctanh(image[:pix].reshape(32, 32, 1)) + c) + d)
    G = (np.tanh(np.arctanh(image[pix:2 * pix].reshape(32, 32, 1)) + c) + d)
    B = (np.tanh(np.arctanh(image[2 * pix:].reshape(32, 32, 1)) + c) + d)
    image = np.concatenate((R, G, B), axis = -1)
    plt.imshow(image)



def transform_hog(X):
    """
    HOG features extraction

    Parameters
    ----------
    X : TYPE
        The original images array.

    Returns
    -------
    X_final : np.array
        The extracted HOG features.

    """
    
    X_final = np.zeros((X.shape[0], 324))
    for i in range(len(X)):
        image = X[i]
        pix = 32 * 32
        R = image[:pix].reshape(32, 32, 1)
        G = image[pix:2 * pix].reshape(32, 32, 1)
        B = image[2 * pix:].reshape(32, 32, 1)
        image = np.concatenate((R, G, B), axis = -1)
        fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, 
                 multichannel=True, feature_vector = True)
        X_final[i] = fd
    return X_final

def normalize(X_train, X_test):
    """
    Normalizing the hole dataset

    """
    
    X_norm = X_train.copy()
    X_test = X_test.copy()
    X = np.concatenate((X_norm, X_test))
    for i in range(3072):
        X_norm[:, i] = (X_train[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        X_test[:, i] = (X_test[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    return X_norm, X_test


def split_data(X, Y_train, X_test, rate = 0.8):
    """
    - This function transform first the dataframes into arrays
    - Then it splits the training dataset into two parts : 
    a training data and a validation data
    - We take 80% as a training dataset and 20% as a validation one
    """
    np.random.seed(3)
    #We first transform the data from a dataframe to array type
    Y = Y_train.to_numpy()[:, -1]
    #We initialize the indexes with the same length of the training set
    n = X.shape[0] 
    indexes = list(range(n))
    #We shuffle the indexes in order to get the training and validation dataset shuffled
    np.random.shuffle(indexes)
    #We split the data into two parts : 80% for training and 20% for validation
    ind_train, ind_val = indexes[: int(n*rate)], indexes[int(n*rate) :]
    X_train, Y_train = X[ind_train], Y[ind_train]
    X_val, Y_val = X[ind_val], Y[ind_val]
    return X_train, Y_train, X_val, Y_val, X_test
