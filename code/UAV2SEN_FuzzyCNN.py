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
from sklearn.model_selection import train_test_split




#############################################################
"""User data input. Fill in the info below before running"""
#############################################################

# Edit this section to have as many inputs as the number of times you ran UAV2SEN_MakeFuzzyTensor
Train1 = np.load('F:\\MixClass\\SesiaCarr_T.npy')
Label1 = np.load('F:\\MixClass\\SesiaCarr_L.npy')

Train2 = np.load('F:\\MixClass\\BuonAmico_T.npy')
Label2 = np.load('F:\\MixClass\\BuonAmico_L.npy')

Train3 = np.load('F:\\MixClass\\BuonAmicoValle_T.npy')
Label3 = np.load('F:\\MixClass\\BuonAmicoVale_L.npy')

Train4 = np.load('F:\\MixClass\\Po_T.npy')
Label4 = np.load('F:\\MixClass\\Po_L.npy')


TrainData = np.concatenate([Train1,Train2, Train3, Train4], axis=0)
TrainLabel = np.concatenate([Label1,Label2, Label3, Label4], axis=0)



'''BASIC PARAMETER CHOICES'''

TrainingEpochs = 150 #For NN training
Nfilters=36
TTS = 0.0001 #Train test split fraction in test.  Keep to 10E-4 if not using
NClasses = 3  #The number of end members in the fuzzy class
inShape = (5,5,11) #adjust this to the template size in X, Y and number of bands
'''MODEL PARAMETERS''' #These would usually not be edited
 
Ndims = TrainData.shape[3] # Feature Dimensions. 4 if using entropy in phase 2, 3 if just RGB
LearningRate = 0.0001
Chatty = 1 # set the verbosity of the model training.  Use 1 at first, 0 when confident that model is well tuned
NAF = 'relu' #Neural network Activation Function

####################################################################################################################

##############################################################################
"""Instantiate the CNN regressor""" 
   

Estimator = Sequential()
Estimator.add(Conv2D(Nfilters,5, data_format='channels_last', input_shape=inShape, activation=NAF))
Estimator.add(Flatten())
Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))

#Tune an optimiser
Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)

# Compile model
Estimator.compile(loss='mean_squared_error', optimizer=Optim, metrics = ['accuracy'])
#Estimator.summary()




###############################################################################
"""Data"""


X = TrainData
Y = TrainLabel

#weights = class_weight.compute_class_weight('balanced', np.unique(Y),  Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TTS, random_state=101)


print('Fitting CNN fuzzy regressor on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)


print('Fuzzy CNN model trained, use other scripts to test validation data')