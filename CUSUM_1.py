# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 08:55:06 2019

@author: Saiprasad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

gamma  = stats.gamma

data = pd.read_csv('modelica_trainset.csv')
columnsName_train=['P','PE','T']
data.columns=columnsName_train
pressure=data[['P']]
power=data[['PE']]
temp=data[['T']]

temp = np.array(temp)
tem = temp[300:601,]

t = np.arange(0,150,1)

measure = temp[0:150]

mu = np.mean(measure)
sd = np.std(measure)
var = sd*sd
beta = var/mu
alpha = mu/beta
gamma.ppf(0.9,alpha, 1/beta)

val = np.arange(0,1,0.01)
t1 = np.arange(0,1,0.01)
for i in range(t.shape[0]):
    val[i]= gamma.ppf(t1[i],alpha, 1/beta)
    
# Found the gamma function by computing its parameters but the 0.9 percentile value is very high compared
# to that of the values of the data itself. So should I continue with that value itself or is there something
# wrong??
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    