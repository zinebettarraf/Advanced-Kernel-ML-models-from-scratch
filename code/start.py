#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import pandas as pd
from data_preprocessing import normalize, transform_hog, split_data
from sklearn.metrics import accuracy_score
from classifiers.KernelRidgeRegession import KRROneVsAll
import os

def main():
    
    print("           Data Challenge 2022: Kernel Methods For Machine Learning")
    print("Team members: Anas MEJGARI (M2 SIAM) - Youssef DAOUD (M2 SIAM)")
    print("Professor   : Julien Mairal ")
    
    #First step: Importing the data
    print("[INFO][START] Importing the data csv files ... ")
    X_train = pd.read_csv("./data/Xtr.csv", header = None)
    X_test  = pd.read_csv("./data/Xte.csv", header = None)
    Y_train = pd.read_csv("./data/Ytr.csv")
    print("[INFO][END] Data imported succefully")
    
    #Second step: Preprocessing the data 
    print("[INFO][START] Preprocessing the Data ...")
    #After visualizing the data, it looks that we should remove the last column
    X_train = X_train.drop([3072], axis = 1)
    X_test  = X_test.drop([3072], axis = 1)
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    X_train, X_test = normalize(X_train, X_test)
    X_train, X_test = transform_hog(X_train), transform_hog(X_test)
    X_train, Y_train, X_val, Y_val, X_test = split_data(X_train, Y_train, X_test)
    print("[INFO][END] Data preprocessed succefully")
    
    
    #Third step: evaluating the model
    print("[INFO][START] Running the classifier ... ")
    krr = KRROneVsAll("polynomial", param = 12)
    krr.fit(X_train, Y_train, 0.01)
    print("[INFO] Evaluating on the validation set")
    Y_pred_val = krr.predict(X_val)
    accuracy = accuracy_score(Y_val, Y_pred_val)
    print("[INFO][END] end of execution of the classifier ... ")
    print(f"\n Validation set accuracy: {accuracy}")
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    # Fourth step: Generating the prediction file
    print("[INFO][START] Generating Yte.csv ... ")
    Y_test = krr.predict(X_test)    
    dataframe = pd.DataFrame(Y_test)
    dataframe.columns = ["Prediction"]
    dataframe.index += 1
    dataframe.to_csv('Yte.csv',index_label='Id')
    print("[INFO][END] Yte.csv generated correctly.")
    cwd = os.getcwd()+"/Yte.csv"
    print(f"[INFO] The full path of the prediction file is {cwd}.")
    print("[END]")

if __name__ == "__main__":
    main()