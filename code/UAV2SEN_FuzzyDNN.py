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
from keras.layers import Dense, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split




#############################################################
"""User data input. Fill in the info below before running"""
#############################################################
DF=pd.read_feather('F:\MixClass\\ESEXpaper_FuzzyMaster_ArboVal.dat') #pre-compiled master dataframe in feather format. 


'''BASIC PARAMETER CHOICES'''

TrainingEpochs = 150 #For NN only
TTS = 0.0001 #Train test split fraction in test. Keep at 10E-4 if not using
NClasses = 3  #The number of end members in the fuzzy class


'''Build data''' 

Train_DF = DF[DF.Type=='Train']
FeatureSet = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B11', 'B12']
Train_Features = Train_DF[FeatureSet] #['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B11', 'B12']
Train_Labels = Train_DF[['Mship1', 'Mship2','Mship3']]


'''MODEL PARAMETERS''' #These would usually not be edited
 
Ndims = len(Train_Features.columns) # Feature Dimensions. 4 if using entropy in phase 2, 3 if just RGB
LearningRate = 0.0001
Chatty = 1 # set the verbosity of the model training.  Use 1 at first, 0 when confident that model is well tuned







##############################################################################
"""Instantiate Neural Network""" 
   
	# create model
def DNN_model_L2D():
	# create model
    model = Sequential()
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), input_dim=Ndims, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(160, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))
    
    #Tune an optimiser
    Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Optim, metrics = ['accuracy'])
    #model.summary()
    return model



    

Estimator = KerasRegressor(build_fn=DNN_model_L2D, epochs=TrainingEpochs, batch_size=5000, verbose=Chatty)






###############################################################################
"""Data"""


X = np.asarray(Train_Features) 
Y = np.asarray(Train_Labels)
 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TTS, random_state=101)





print('Fitting NN fuzzy regressor on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=5000, epochs=TrainingEpochs, verbose=Chatty)#, class_weight=weights)

print('Fuzzy DNN model trained, use other scripts to test validation data')  

 