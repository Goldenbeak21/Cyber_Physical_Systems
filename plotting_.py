# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:40:38 2019

@author: Saiprasad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras


output = pd.read_csv('output.csv')
dataset = pd.read_csv('final_values.csv')
data = np.array(dataset)
data = data[:,3]
data = data.reshape((900,1))
pres = data[0:300,:]
pow = data[300:600,:]
temp = data[600:900,:]

pre_t = np.zeros((20,1))
temp_t = np.zeros((20,1))
pow_t = np.zeros((20,1))

for i in range(20):
    pre_t[i,0] = np.sum(pres[i*15:(i+1)*15,0])
    pow_t[i,0] = np.sum(pow[i*15:(i+1)*15,0])
    temp_t[i,0] = np.sum(temp[i*15:(i+1)*15,0])
    
bar_pre = pre_t.reshape((1,20))
sum_bar = np.zeros((20,))
for i in range(20):
    sum_bar[i] = pow_t[i] + pre_t[i] + temp_t[i]

import seaborn as sns
sns.barplot(x=xc, y=sum_bar, color='blue')
plt.title("Parameters")
plt.xlabel("Time Steps")
plt.ylabel("Attention Weights")
plt.show()

X = np.arange(1,21,1)
X1 = np.arange(1.25,21.25,1)
plt.bar(X + 0.00, pre_t, color = 'b', width = 0.25, align='center', label="Pressure")
plt.bar(X + 0.25, pow_t, color = 'g', width = 0.25, align='center', label="Power")
plt.bar(X + 0.50, temp_t, color = 'r', width = 0.25, align='center',label="Temperature")
plt.xticks(X1,X)
plt.legend()
plt.ylabel("Attention Weights")
plt.xlabel("Time Steps")
plt.show()


heat = sum_bar.reshape(20,1)
output = np.array(output)
output = output.reshape(20,1)
heat = output

ml = np.arange(0.5,20.5,1)
v = np.arange(1,21,1)
ax = sns.heatmap(heat, cmap="BuPu", annot=True, annot_kws={"weight": "bold", "size":12})
plt.title("Attention Heat Map", fontweight = 'bold', fontsize = 18)
plt.xlabel("Attention(%)", fontweight = 'bold', fontsize = 16)
plt.xticks([])
plt.yticks(ml,v, fontweight='bold', fontsize=12)
plt.ylabel("Time Steps", fontweight = 'bold', fontsize = 16)
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams["font.weight"] = "bold"
plt.show()







pre_t = pre_t.reshape((20,1))
pow_t = pow_t.reshape((20,1))
temp_t = temp_t.reshape((20,1))
xc = np.arange(1,21,1)
plt.bar(xc, pre_t)











