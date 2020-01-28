#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

###############################################################################
""" Libraries"""
from keras import regularizers
from keras import optimizers
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from IPython import get_ipython
import tensorflow as tf
import statsmodels.api as sm



#############################################################
"""User data input"""
#############################################################
'''BASIC PARAMETER CHOICES'''
CNN = False #false if estimator is RF or DNN, True for CNN 
UT=1.95# upper and lower thresholds to elimn=inate pure classes from fuzzy error estimates
LT=-0.05

if CNN:
    ValidData = np.load('F:\\MixClass\\SesiaArbo_T.npy')
    ValidLabel = np.load('F:\\MixClass\\SesiaArbo_L.npy')
else:
    DF=pd.read_feather('F:\MixClass\\ESEXpaper_FuzzyMaster_ArboVal.dat')
    FeatureSet = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B11', 'B12']
    Valid_DF = DF[DF.Type=='Valid']
    Valid_Features = Valid_DF[FeatureSet]
    Valid_Labels = Valid_DF[['Mship1', 'Mship2','Mship3']]




'''Begin data validation and error estimates'''
get_ipython().run_line_magic('matplotlib', 'qt') 

#validate the estimator
if CNN:
    X=ValidData
    Y=ValidLabel
else:
    X=np.asarray(Valid_Features)
    Y=np.asarray(Valid_Labels)
 
PredictedPixels = Estimator.predict(X)


Error1 = np.asarray(PredictedPixels[:,0]) - Y[:,0]
Error2 = PredictedPixels[:,1] - Y[:,1]
Error3 = PredictedPixels[:,2] - Y[:,2]


ErrFrame = pd.DataFrame({'C1 Err':Error1, 'C2 Err':Error2, 'C3 Err':Error3, 'C1 Obs':Y[:,0], 'C2 Obs':Y[:,1],'C3 Obs':Y[:,2], 'C1 Pred':PredictedPixels[:,0], 'C2 Pred':PredictedPixels[:,1],'C3 Pred':PredictedPixels[:,2]})
ErrFrame1 = ErrFrame[ErrFrame['C1 Obs']<UT]
ErrFrame1 = ErrFrame1[ErrFrame1['C1 Obs']>LT]
jplot = sns.jointplot("C1 Obs", 'C1 Pred', data=ErrFrame1, kind="kde", color='b', n_levels=500)
jplot.ax_marg_x.set_xlim(-0.2, 1.2)
jplot.ax_marg_y.set_ylim(-0.2, 1.2)


ErrFrame2 = ErrFrame[ErrFrame['C2 Obs']<UT]
ErrFrame2 = ErrFrame2[ErrFrame2['C2 Obs']>LT]
jplot = sns.jointplot("C2 Obs", 'C2 Pred', data=ErrFrame2, kind="kde", color='g', n_levels=500)
jplot.ax_marg_x.set_xlim(-0.2, 1.2)
jplot.ax_marg_y.set_ylim(-0.2, 1.2)


ErrFrame3 = ErrFrame[ErrFrame['C3 Obs']<UT]
ErrFrame3 = ErrFrame3[ErrFrame3['C3 Obs']>LT]
jplot = sns.jointplot("C3 Obs", 'C3 Pred', data=ErrFrame3, kind="kde", color='r', n_levels=500)
jplot.ax_marg_x.set_xlim(-0.2, 1.2)
jplot.ax_marg_y.set_ylim(-0.2, 1.2)


Error1 = ErrFrame1['C1 Err']
Error2 = ErrFrame2['C2 Err']
Error3 = ErrFrame3['C3 Err']
RMS1 = np.sqrt(np.mean(Error1*Error1))
RMS2 = np.sqrt(np.mean(Error2*Error2))
RMS3 = np.sqrt(np.mean(Error3*Error3))
Errors = np.concatenate((Error1, Error2, Error3))
RMSall = np.sqrt(np.mean(Errors*Errors))
print('mean error =', str(np.mean(Errors)))
print('\n')
print('RMS error =', str(RMSall))
print('\n')

#
#
print('Water Fit Stats')
X=sm.add_constant(ErrFrame1['C1 Obs'])
linemodel = sm.OLS(ErrFrame1['C1 Pred'], X)
reg = linemodel.fit()
print(reg.summary())

print('Veg Fit Stats')
X=sm.add_constant(ErrFrame2['C2 Obs'])
linemodel = sm.OLS(ErrFrame2['C2 Pred'], X)
reg = linemodel.fit()
print(reg.summary())

print('Sed Fit Stats')
X=sm.add_constant(ErrFrame3['C3 Obs'])
linemodel = sm.OLS(ErrFrame3['C3 Pred'], X)
reg = linemodel.fit()
print(reg.summary())