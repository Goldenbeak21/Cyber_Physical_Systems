# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
rmse1 = np.zeros((5,))
mape1 = np.zeros((5,))
mae1 = np.zeros((5,))


for p in range(5):

    
    dataset = pd.read_csv('trainset.csv')
    data = np.array(dataset)
    # Always have X as a matrix and y as a vector
    Xtr = data[:,2:905]
    ytr = data[:,0]
    j=0
    X = np.zeros((7000,903))
    y = np.zeros((7000,))
    for i in range(7000):
        if(ytr[i,]>p*10 and ytr[i,]<=(p+1)*10):
            X[j,:] = Xtr[i,:]
            y[j,] = ytr[i,]
            j = j+1
    
    X = X[0:j,:]
    y = y[0:j,]
    y = y.reshape((j,1))
    
    testset = pd.read_csv('testset.csv')
    test = np.array(testset)
    Xts = test[:,2:905]
    yts = test[:,0]
    
    j=0
    X1 = np.zeros((3000,903))
    y1 = np.zeros((3000,))
    for i in range(3000):
        if(yts[i,]>p*10 and yts[i,]<=(p+1)*10):
            X1[j,:] = Xts[i,:]
            y1[j,] = yts[i,]
            j = j+1
     
    X1 = X1[0:j,:]
    y1 = y1[0:j,]       
    

    
    # NEED ONLY ARRAYS FOR FEATURE SCALING
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    sc_X = sc_X.fit(X)
    sc_y = sc_y.fit(y)
    X = sc_X.transform(X)
    y = sc_y.transform(y)
    
    X1 = sc_X.transform(X1)
    
    
    # fitting the model ("regressor" is created here)
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X,y)
    
    y_pred = regressor.predict(X1)
    y_hat = sc_y.inverse_transform(y_pred)
    
    
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
    
    
    def NRMSD(y_true, y_pred):
        rmsd = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        y_min = min(y_true)
        y_max = max(y_true)
    
        return rmsd / (y_max - y_min)
    
    
    def MAPE(y_true, y_pred):
        y_true_select = (y_true != 0)
    
        y_true = y_true[y_true_select]
        y_pred = y_pred[y_true_select]
    
        errors = y_true - y_pred
        return sum(abs(errors / y_true)) * 100.0 / len(y_true)
    
    y_true = y1
    predict = y_hat
    
    nrmsd = NRMSD(y_true, predict)
    mape = MAPE(y_true, predict)
    mae = mean_absolute_error(y_true, predict)
    rmse = np.sqrt(mean_squared_error(y_true, predict))
    print("NRMSD", nrmsd)
    print("MAPE", mape)
    print("neg_mean_absolute_error", mae)
    print("Root mean squared error", rmse)
    rmse1[p] = rmse
    mae1[p]= mae
    mape1[p]= mape
    
















