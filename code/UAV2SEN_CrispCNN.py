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
from tensorflow.keras.utils import normalize
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Conv2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################
MainData = 'E:\\UAV2SEN\\MLdata\\DNNDebug'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'E:\\UAV2SEN\\SiteList.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DatFolder = 'E:\\UAV2SEN\\FinalTif\\'  #location of above
TrainingEpochs = 100 #Typically this can be reduced
Nfilters = 32
UAVtrain = False #if true use the UAV class data to train the model, if false use desk-based
UAVvalid = True #if true use the UAV class data to validate.  If false, use desk-based polygons
size=5#size of the tensor tiles
KernelSize=5 # size of the convolution kernels
MajType= 'Maj' #Majority type. only used if UAVtrain or valid is true. The options are RelMaj (relative majority), Maj (majority) and Pure (95% unanimous).

FeatureSet =  ['B1','B2','B9','B10','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']

LearningRate = 0.001
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function

DoHistory = False #Plot the history of the training losses

'''Load the tensors and filter out the required training and validation data.'''
TensorFileName = MainData+'_crisp_'+str(size)+'_T.npy'
LabelFileName = MainData+'_crisp_'+str(size)+'_L.csv'

SiteDF = pd.read_csv(SiteList)
MasterTensor = np.load(TensorFileName)
MasterLabelDF=pd.read_csv(LabelFileName)

#Select the features in the tensor

Valid=np.zeros(12)
for n in range(1,13):
    if ('B'+str(n)) in FeatureSet:
        Valid[n-1]=1
        
MasterTensor = np.compress(Valid, MasterTensor, axis=3)
#Remove the 4X data augmentation only relevant to the CNNs and take only points 0,4,8,etc...
PointNums = np.asarray(range(0,len(MasterLabelDF.RelMajClass)))
Spots = PointNums%4
Valid = Spots==0

#Subsample the labels and fix the index
MasterLabelDF = MasterLabelDF.loc[Valid]
MasterLabelDF.index = range(0,len(MasterLabelDF.RelMajClass))
MasterTensor = np.compress(Valid, MasterTensor, axis=0)



#Start the filter process to isolate training and validation data
TrainingSites = SiteDF[SiteDF.Training == 1]
ValidationSites = SiteDF[SiteDF.Validation == 1]

#isolate the site. first labels then tensors
TrainDF = MasterLabelDF[MasterLabelDF['Site'].isin(TrainingSites.Abbrev.to_list())]
ValidationDF = MasterLabelDF[MasterLabelDF['Site'].isin(ValidationSites.Abbrev.to_list())]

Valid = MasterLabelDF['Site'].isin(TrainingSites.Abbrev.to_list())
TrainingTensor = np.compress(Valid, MasterTensor, axis=0)
Valid = MasterLabelDF['Site'].isin(ValidationSites.Abbrev.to_list())
ValidationTensor = np.compress(Valid, MasterTensor, axis=0)

#isolate the year
TrainDF = TrainDF[TrainDF['Year'].isin(TrainingSites.Year.to_list())]
ValidationDF = ValidationDF[ValidationDF['Year'].isin(ValidationSites.Year.to_list())]

Valid= TrainDF['Year'].isin(TrainingSites.Year.to_list())
TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
Valid = ValidationDF['Year'].isin(ValidationSites.Year.to_list())
ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)


#isolate the month
TrainDF = TrainDF[TrainDF['Month'].isin(TrainingSites.Month.to_list())]
ValidationDF = ValidationDF[ValidationDF['Month'].isin(ValidationSites.Month.to_list())]

Valid= TrainDF['Month'].isin(TrainingSites.Month.to_list())
TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
Valid = ValidationDF['Month'].isin(ValidationSites.Month.to_list())
ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)


#select desk-based or UAV-based for training and validation, if using UAV data, select the majority type
if UAVtrain & UAVvalid:
    TrainDF=TrainDF[TrainDF.PolyClass==-1]
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
    Valid= TrainDF.PolyClass==-1
    TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    Valid = ValidationDF.PolyClass==-1
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    if 'RelMaj' in  MajType:
        TrainDF = TrainDF[TrainDF.RelMajClass>0]
        ValidationDF = ValidationDF[ValidationDF.RelMajClass>0]
        TrainLabels = TrainDF.RelMajClass
        ValidationLabels = ValidationDF.RelMajClass
        Valid= TrainDF.RelMajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        Valid = ValidationDF.RelMajClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    elif 'Maj' in MajType:
        TrainDF = TrainDF[TrainDF.MajClass>0]
        ValidationDF = ValidationDF[ValidationDF.MajClass>0]
        TrainLabels = TrainDF.MajClass
        ValidationLabels = ValidationDF.MajClass
        Valid= TrainDF.MajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        Valid = ValidationDF.MajClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    
    elif 'Pure' in MajType:
        TrainDF = TrainDF[TrainDF.PureClass>0]
        ValidationDF = ValidationDF[ValidationDF.PureClass>0]
        TrainLabels = TrainDF.PureClass
        ValidationLabels = ValidationDF.PureClass
        Valid= TrainDF.PureClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        Valid = ValidationDF.PureClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
        
elif UAVtrain and ~(UAVvalid):
    TrainDF=TrainDF[TrainDF.PolyClass==-1]
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    ValidationLabels = ValidationDF.PolyClass
    Valid= TrainDF.PolyClass==-1
    TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    Valid = ValidationDF.PolyClass>0
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    if 'RelMaj' in  MajType:
        TrainDF = TrainDF[TrainDF.RelMajClass>0]
        TrainLabels = TrainDF.RelMajClass
        Valid= TrainDF.RelMajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)

    elif 'Maj' in MajType:
        TrainDF = TrainDF[TrainDF.MajClass>0]
        TrainLabels = TrainDF.MajClass
        Valid= TrainDF.MajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    
    elif 'Pure' in MajType:
        TrainDF = TrainDF[TrainDF.PureClass>0]
        TrainLabels = TrainDF.PureClass
        Valid= TrainDF.PureClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        
elif ~(UAVtrain) & UAVvalid:
    TrainDF=TrainDF[TrainDF.PolyClass>0]
    TrainLabels = TrainDF.PolyClass
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
    Valid= TrainDF.PolyClass>0
    TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    Valid = ValidationDF.PolyClass==-1
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    if 'RelMaj' in  MajType:
        ValidationDF = ValidationDF[ValidationDF.RelMajClass>0]
        ValidationLabels = ValidationDF.RelMajClass
        Valid = ValidationDF.RelMajClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    elif 'Maj' in MajType:
        ValidationDF = ValidationDF[ValidationDF.MajClass>0]
        ValidationLabels = ValidationDF.MajClass
        Valid = ValidationDF.MajClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    
    elif 'Pure' in MajType:
        ValidationDF = ValidationDF[ValidationDF.PureClass>0]
        ValidationLabels = ValidationDF.PureClass
        Valid = ValidationDF.ValidationDF.PureClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    
else:
    TrainDF=TrainDF[TrainDF.PolyClass>0]
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    TrainLabels = TrainDF.PolyClass
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    ValidationLabels = ValidationDF.PolyClass
    Valid= TrainDF.PolyClass>0
    TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    Valid = ValidationDF.PolyClass>0
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    
#check for empty dataframes and raise an error if found

if (len(TrainDF.index)==0):
    raise Exception('There is an empty dataframe for training')
    
if (len(ValidationDF.index)==0):
    raise Exception('There is an empty dataframe for validation')
    
#Check that tensor lengths match label lengths

if (len(TrainLabels.index)) != TrainingTensor.shape[0]:
    raise Exception('Sample number mismatch for TRAINING tensor and labels')
    
if (len(ValidationLabels.index)) != ValidationTensor.shape[0]:
    raise Exception('Sample number mismatch for VALIDATION tensor and labels')
    


        
    
'''Range the training tensor from 0-1'''
#NormFactor = np.max(np.unique(TrainingTensor.reshape(1,-1)))
#TrainingTensor = TrainingTensor/NormFactor
#ValidationTensor = ValidationTensor/NormFactor
#TrainingTensor = normalize(TrainingTensor)
#ValidationTensor = normalize(ValidationTensor)
 
Ndims = TrainingTensor.shape[3] # Feature Dimensions. 
NClasses = len(np.unique(TrainLabels))  #The number of classes in the data. This MUST be the same as the classes used to retrain the model
inShape = TrainingTensor.shape[1:]




##############################################################################
"""Instantiate the Neural Network pixel-based classifier""" 
#EstimatorRF = RFC(n_estimators = 150, n_jobs = 8, verbose = Chatty) #adjust this to your processors
   


# define the very deep model with L2 regularization and dropout

 	# create model
Estimator = Sequential()
Estimator.add(Conv2D(Nfilters,5, data_format='channels_last', input_shape=(5,5,Ndims), activation=NAF))
Estimator.add(Flatten())
Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax'))

#Tune an optimiser
Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)

# Compile model
Estimator.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
Estimator.summary()
  
###############################################################################
"""Data Fitting"""
TrainLabels1Hot = to_categorical(TrainLabels)
ValidationLabels1Hot = to_categorical(ValidationLabels)
X_train, X_test, y_train, y_test = train_test_split(TrainingTensor, TrainLabels1Hot, test_size=0.2, random_state=42)
print('Fitting CNN Classifier on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)#, class_weight=weights)
#EstimatorRF.fit(X_train, y_train)
    
#Fit the predictor to test pixels
PredictedPixels = Estimator.predict(X_test)

# #Produce TTS classification reports 
#y_test=np.argmax(y_test, axis=1)
# PredictedPixels 
report = metrics.classification_report(y_test, PredictedPixels, digits = 3)
print('20% Test classification results for ')
print(report)
      

# #Fit the predictor to the external validation site
PredictedPixels = Estimator.predict(ValidationTensor)
report = metrics.classification_report(ValidationLabels, PredictedPixels, digits = 3)
print('Out-of-Sample validation results for ')
print(report)


'''Save model and scaler'''