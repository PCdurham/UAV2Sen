#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

###############################################################################
""" Libraries"""
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Conv2D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump
import seaborn as sns
import statsmodels.api as sm




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################
MainData = 'E:\\UAV2SEN\\MLdata\\CNNDebug'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'E:\\UAV2SEN\\SiteList.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DatFolder = 'E:\\UAV2SEN\\FinalTif\\'  #location of above
TrainingEpochs = 100 #Typically this can be reduced
size=3 #size of the tiles used to compile the data

FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']
LabelSet = ['WaterMem', 'VegMem','SedMem' ]
LearningRate = 0.0001
Chatty = 1 # set the verbosity of the model training. 
UT=0.95# upper and lower thresholds to elimn=inate pure classes from fuzzy error estimates
LT=0.05


'''Load the tensors, remove data augmentation meant for CNNs, squeeze to get single pixel values and filter out the required training and validation data.'''
TensorFileName = MainData+'_fuzzy_'+str(size)+'_T.npy'
LabelFileName = MainData+'_fuzzy_'+str(size)+'_L.dat'

SiteDF = pd.read_csv(SiteList)
Tensor = np.load(TensorFileName)
MasterLabelDF=pd.read_feather(LabelFileName)

#Remove the 4X data augmentation only relevant to the CNNs and take only points 0,4,8,etc...
PointNums = np.asarray(range(0,len(MasterLabelDF.index)))
Spots = PointNums%2
Valid = Spots==0

#Subsample the labels and fix the index
MasterLabelDF = MasterLabelDF.loc[Valid]
MasterLabelDF.index = range(0,len(MasterLabelDF.index))

#Subsample the tensor
Tensor = np.compress(Valid, Tensor, axis=0)

#get the central pixels in the tensor to transform this into pixel-based data for the non-convolutional NN
Middle = Tensor.shape[1]//2
PixelData = np.squeeze(Tensor[:,Middle, Middle,:])
PixelDF = pd.DataFrame(data=PixelData, columns=['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'])
MasterLabelDF = pd.concat([MasterLabelDF, PixelDF], axis=1)

#Start the filter process to isolate training and validation data
TrainingSites = SiteDF[SiteDF.Training == 1]
ValidationSites = SiteDF[SiteDF.Validation == 1]


#isolate the site
TrainDF = MasterLabelDF[MasterLabelDF['Site'].isin(TrainingSites.Abbrev.to_list())]
ValidationDF = MasterLabelDF[MasterLabelDF['Site'].isin(ValidationSites.Abbrev.to_list())]

#isolate the year
TrainDF = TrainDF[TrainDF['Year'].isin(TrainingSites.Year.to_list())]
ValidationDF = ValidationDF[ValidationDF['Year'].isin(ValidationSites.Year.to_list())]

#isolate the month
TrainDF = TrainDF[TrainDF['Month'].isin(TrainingSites.Month.to_list())]
ValidationDF = ValidationDF[ValidationDF['Month'].isin(ValidationSites.Month.to_list())]

#Set the labels
TrainFeatures = TrainDF[FeatureSet]
ValidationFeatures = ValidationDF[FeatureSet]
TrainLabels = TrainDF[LabelSet]
ValidationLabels = ValidationDF[LabelSet]
    
#check for empty dataframes and raise an error if found

if (len(TrainDF.B1)==0):
    raise Exception('There is an empty dataframe for training')
    
if (len(ValidationDF.B1)==0):
    raise Exception('There is an empty dataframe for validation')
    


        
    
'''Normalise and scale the band values'''
Scaler = StandardScaler()
Scaler.fit(TrainFeatures)
TrainFeatures = Scaler.transform(TrainFeatures)
ValidationFeatures = Scaler.transform(ValidationFeatures)

 
Ndims = TrainFeatures.shape[1] # Feature Dimensions. 
NClasses = 3  #The number of classes in the data. This MUST be the same as the classes used to retrain the model





##############################################################################
"""Instantiate the Conv Neural Network  regressor""" 
#EstimatorRF = RFC(n_estimators = 150, n_jobs = 8, verbose = Chatty) #adjust this to your processors
   


Estimator = Sequential()
Estimator.add(Conv2D(Nfilters,KernelSize, data_format='channels_last', input_shape=inShape, activation=NAF))
Estimator.add(Conv2D(Nfilters,KernelSize, activation=NAF))
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
Estimator.summary()
  
###############################################################################
"""Data Fitting"""
#TrainLabels1Hot = to_categorical(TrainLabels)
#ValidationLabels1Hot = to_categorical(ValidationLabels)
X_train, X_test, y_train, y_test = train_test_split(TrainFeatures, TrainLabels, test_size=0.2, random_state=42)
print('Fitting MLP Classifier on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)#, class_weight=weights)
#EstimatorRF.fit(X_train, y_train)
    



'''Save model and scaler'''

'''Validate the model'''
#Test data
PredictedPixels = Estimator.predict(X_test)
Y=y_test

Error1 = PredictedPixels[:,0] - Y.WaterMem
Error2 = PredictedPixels[:,1] - Y.VegMem
Error3 = PredictedPixels[:,2] - Y.SedMem


ErrFrame = pd.DataFrame({'C1 Err':Error1, 'C2 Err':Error2, 'C3 Err':Error3, 'C1 Obs':Y.WaterMem, 'C2 Obs':Y.VegMem,'C3 Obs':Y.SedMem, 'C1 Pred':PredictedPixels[:,0], 'C2 Pred':PredictedPixels[:,1],'C3 Pred':PredictedPixels[:,2]})
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