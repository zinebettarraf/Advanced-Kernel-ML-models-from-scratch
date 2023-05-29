
This repository contains the handing for the M2SIAM class: Kernel methods for machine learning


The challenge was a classification task on CIFAR 10 dataset : https://www.cs.toronto.edu/~kriz/cifar.html.

This challenge had the specificity that it requires to compute from scratch all the algorithms and the methods that we are using (no sklearn, libsvm, ...).

All our implementation has been done in python.

Hence in order to run our codes one will only needs few libraries: numpy, pandas, skimage, cvxopt, scipy, tqdm, os, sklearn.metrics, matplotlib.

The folder code contains the work we have done in different folders, and some files such as:

    data_processing -- different useful methods for preprocessing tha data
    kernels -- contains many kernel functions we have implemented
    start -- the main script executing the algorithm with the highest accuracy (you just should put the data in data folder in the following way : Xtr.csv : the images for the training - Xte.csv : le label for each image : Ytr.csv : the test dataset.


The folder Classifiers contains all the classifiers that we have implemented:

    KernelRidgeRegression -- Kernel ridge regression (takes as input the kernel of your choice and its parameters)
    LogesticRegression -- basic logestic regression (used in the baseline)
    KernelLogisticRegression -- Kernel logistic regression (takes as input the kernel of your choice and its parameters)
    SVM -- Support Vector Machines (takes as input the kernel of your choice and its parameters as well as the C parameters of SVM)

The folder utils contains the functions basic tools that we have used during our implementation:

    KMeans : Kernel KMeans for clustring the data
    KernelPCA : A basic implementation of Kernel PCA

In order to run the code, you first have to set properly the data in data folder as follows : 
    Xtr.csv : the images for the training - 
    Xte.csv : le label for each image : 
    Ytr.csv : the test dataset.. 
    
Then you run the script start
