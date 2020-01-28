#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

###############################################################################
""" Libraries"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.multioutput import MultiOutputRegressor




#############################################################
"""User data input. Fill in the info below before running"""
#############################################################
DF=pd.read_feather('F:\MixClass\\ESEXpaper_FuzzyMaster_ArboVal.dat') #pre-compiled master dataframe in feather format. 
#DFnofuzz=pd.read_feather('F:\MixClass\\ESEXpaper_Master_SR.dat')


'''BASIC PARAMETER CHOICES'''

Trees = 150 #for random forest only
TTS = 0.0001 #Train test split fraction in test
NClasses = 3  #The number of end members in the fuzzy class
Chatty = 0 # set the verbosity of the model training.  Use 1 at first, 0 when confident that model is well tuned

'''Build data''' 

Train_DF = DF[DF.Type=='Train']
FeatureSet = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B11', 'B12']
Train_Features = Train_DF[FeatureSet] #['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B11', 'B12']
Train_Labels = Train_DF[['Mship1', 'Mship2','Mship3']]


'''Instantiate the RF model with multioutput regressor wrapper'''
Estimator = MultiOutputRegressor(RFR(n_estimators=Trees,n_jobs=-1,random_state=0,verbose=Chatty))

   

###############################################################################
"""Data"""
   

X = np.asarray(Train_Features) 
Y = np.asarray(Train_Labels)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TTS, random_state=101)





print('Fitting RF fuzzy regressor on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train) 
  
print('Fuzzy RF model trained, use other scripts to test validation data')  
  
