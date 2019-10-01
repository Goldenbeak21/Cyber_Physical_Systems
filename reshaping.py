# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:11:14 2019

@author: Saiprasad
"""

import numpy as np
import pandas as pd

train_df = pd.read_csv('trainset.csv')
X_train = train_df.drop(['period', 'powerSetPoint', 'sigma', 'delay'], axis=1)
y_train = train_df[['delay']]

X = X_train
X = np.array(X)

X_shape = list(X.shape)

X_shape[-1] = int(X_shape[-1] / 3)

step = int(X_shape[-1] / 20)

lengh = step * 20


X = X.reshape((X_shape[0], 3, -1))

X = np.transpose(X, axes=(0,2,1))


j=0
required = np.zeros((3000, 301, 3))

y_train= np.array(y_train)
for i in range(3000):
    if(y_train[i] == 0):
        required[j, :, :] = X[i,:,:]
        j = j + 1
        print(i,j)
        
required = required[0:j ,:,:]
required = required.reshape((1, j ,903))
required = required.reshape((1, 301*j ,3))

      
     
        
        
        
        
        
        
        
        