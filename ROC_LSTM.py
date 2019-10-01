# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:53:30 2019

@author: Saiprasad
"""
import numpy as np
import matplotlib as plt
import pandas as pd
train = pd.read_csv('T_GRU_Two_cycle.csv')
train = np.array(train)
y_pred = train[:,0]
y_true = train[:,1]


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
testy = y_true
probs = y_pred
auc = roc_auc_score(testy, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(testy, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()