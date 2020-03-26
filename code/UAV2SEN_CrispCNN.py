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
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################
MainData = 'E:\\UAV2SEN\\MLdata\\LargeDebug'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'E:\\UAV2SEN\\SiteListLong.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DatFolder = 'E:\\UAV2SEN\\FinalTif\\'  #location of above
TrainingEpochs = 20 #Typically this can be reduced

UAVtrain = True #if true use the UAV class data to train the model, if false use desk-based
UAVvalid = True #if true use the UAV class data to validate.  If false, use desk-based polygons

MajType= 'Pure' #Majority type. only used if UAVtrain or valid is true. The options are RelMaj (relative majority), Maj (majority) and Pure (95% unanimous).

FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']

LearningRate = 0.0001
Chatty = 1 # set the verbosity of the model training. 



'''Load the tensors and filter out the required training and validation data.'''
TensorFileName = MainData+'_crisp_T.npy'
LabelFileName = MainData+'_crisp_L.dat'

SiteDF = pd.read_csv(SiteList)
Tensor = np.load(TensorFileName)
MasterLabelDF=pd.read_feather(LabelFileName)

#Subsample the tensor
Tensor = np.compress(Valid, Tensor, axis=0)


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

#select desk-based or UAV-based for training and validation, if using UAV data, select the majority type
if UAVtrain & UAVvalid:
    TrainDF=TrainDF[TrainDF.PolyClass==-1]
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
    if 'RelMaj' in  MajType:
        TrainDF = TrainDF[TrainDF.RelMajClass>0]
        ValidationDF = ValidationDF[ValidationDF.RelMajClass>0]
        TrainFeatures = TrainDF[FeatureSet]
        ValidationFeatures = ValidationDF[FeatureSet]
        TrainLabels = TrainDF.RelMajClass
        ValidationLabels = ValidationDF.RelMajClass
    elif 'Maj' in MajType:
        TrainDF = TrainDF[TrainDF.MajClass>0]
        ValidationDF = ValidationDF[ValidationDF.MajClass>0]
        TrainFeatures = TrainDF[FeatureSet]
        ValidationFeatures = ValidationDF[FeatureSet]
        TrainLabels = TrainDF.MajClass
        ValidationLabels = ValidationDF.MajClass
    
    elif 'Pure' in MajType:
        TrainDF = TrainDF[TrainDF.PureClass>0]
        ValidationDF = ValidationDF[ValidationDF.PureClass>0]
        TrainFeatures = TrainDF[FeatureSet]
        ValidationFeatures = ValidationDF[FeatureSet]
        TrainLabels = TrainDF.PureClass
        ValidationLabels = ValidationDF.PureClass
        
elif UAVtrain and ~(UAVvalid):
    TrainDF=TrainDF[TrainDF.PolyClass==-1]
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    ValidationFeatures = ValidationDF[FeatureSet]
    ValidationLabels = ValidationDF.PolyClass
    if 'RelMaj' in  MajType:
        TrainDF = TrainDF[TrainDF.RelMajClass>0]
        TrainFeatures = TrainDF[FeatureSet]
        TrainLabels = TrainDF.RelMajClass

    elif 'Maj' in MajType:
        TrainDF = TrainDF[TrainDF.MajClass>0]
        TrainFeatures = TrainDF[FeatureSet]
        TrainLabels = TrainDF.MajClass

    
    elif 'Pure' in MajType:
        TrainDF = TrainDF[TrainDF.PureClass>0]
        TrainFeatures = TrainDF[FeatureSet]
        TrainLabels = TrainDF.PureClass
        
elif ~(UAVtrain) & UAVvalid:
    TrainDF=TrainDF[TrainDF.PolyClass>0]
    TrainFeatures = TrainDF[FeatureSet]
    TrainLabels = TrainDF.PolyClass
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
    if 'RelMaj' in  MajType:
        ValidationDF = ValidationDF[ValidationDF.RelMajClass>0]
        ValidationFeatures = ValidationDF[FeatureSet]
        ValidationLabels = ValidationDF.RelMajClass
    elif 'Maj' in MajType:
        ValidationDF = ValidationDF[ValidationDF.MajClass>0]
        ValidationFeatures = ValidationDF[FeatureSet]
        ValidationLabels = ValidationDF.MajClass
    
    elif 'Pure' in MajType:
       ValidationDF = ValidationDF[ValidationDF.PureClass>0]
       ValidationFeatures = ValidationDF[FeatureSet]
       ValidationLabels = ValidationDF.PureClass
    
else:
    TrainDF=TrainDF[TrainDF.PolyClass>0]
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    TrainFeatures = TrainDF[FeatureSet]
    TrainLabels = TrainDF.PolyClass
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    ValidationFeatures = ValidationDF[FeatureSet]
    ValidationLabels = ValidationDF.PolyClass
    
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
NClasses = len(np.unique(TrainLabels))  #The number of classes in the data. This MUST be the same as the classes used to retrain the model





##############################################################################
"""Instantiate the Neural Network pixel-based classifier""" 
#EstimatorRF = RFC(n_estimators = 150, n_jobs = 8, verbose = Chatty) #adjust this to your processors
   


# define the very deep model with L2 regularization and dropout

 	# create model
def nodrop_model_L2D():
 	# create model
    model = Sequential()
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), input_dim=Ndims, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    #model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))
    
    
    #Tune an optimiser
    Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    return model
    
EstimatorNN = KerasClassifier(build_fn=nodrop_model_L2D, epochs=TrainingEpochs, batch_size=1000, verbose=Chatty)

  
###############################################################################
"""Data Fitting"""
#TrainLabels1Hot = to_categorical(TrainLabels)
#ValidationLabels1Hot = to_categorical(ValidationLabels)
X_train, X_test, y_train, y_test = train_test_split(TrainFeatures, TrainLabels, test_size=0.2, random_state=42)
print('Fitting MLP Classifier on ' + str(len(X_train)) + ' pixels')
EstimatorNN.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)#, class_weight=weights)
#EstimatorRF.fit(X_train, y_train)
    
#Fit the predictor to test pixels
PredictedPixels = EstimatorNN.predict(X_test)

# #Produce TTS classification reports 
#y_test=np.argmax(y_test, axis=1)
# PredictedPixels 
report = metrics.classification_report(y_test, PredictedPixels, digits = 3)
print('20% Test classification results for ')
print(report)
      

# #Fit the predictor to the external validation site
PredictedPixels = EstimatorNN.predict(ValidationFeatures)
report = metrics.classification_report(ValidationLabels, PredictedPixels, digits = 3)
print('Out-of-Sample validation results for ')
print(report)


'''Save model and scaler'''