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
import seaborn as sns
import statsmodels.api as sm




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################
#############################################################
MainData = 'E:\\UAV2SEN\\MLdata\\FullData_4xnoise'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'E:\\UAV2SEN\\SiteList.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
ModelName = 'E:\\UAV2SEN\\MLdata\\CNNdebugged.h5'  #Name and location of the final model to be saved
DatFolder = 'E:\\UAV2SEN\\FinalTif\\'  #location of processed tif files
TrainingEpochs = 15 #Typically this can be reduced
Nfilters = 64
size=5#size of the tensor tiles
KernelSize=3 # size of the convolution kernels. 

UAVvalid = False #if true use the UAV class data to validate.  If false, use desk-based polygons and ignore majority type
MajType = 'Pure' #majority type to use in crisp validation
FeatureSet =  ['B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']
LabelSet = ['WaterMem', 'VegMem','SedMem' ]
LearningRate = 0.001
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function

DoHistory = False #Plot the history of the training losses
ShowValidation=True

#################################################################################
'''Function definitions'''
def slide_raster_to_tiles(im, size):
    h=im.shape[0]
    w=im.shape[1]
    di=im.shape[2]
    TileTensor = np.zeros(((h-size)*(w-size), size,size,di))

    
    B=0
    for y in range(0, h-size):
        for x in range(0, w-size):

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:]
            B+=1

    return TileTensor


####################################################################################
    

'''Load the tensors and filter out the required training and validation data.'''
TensorFileName = MainData+'_fuzzy_'+str(size)+'_T.npy'
LabelFileName = MainData+'_fuzzy_'+str(size)+'_L.csv'
ValidationTensorFileName = MainData+'_crisp_'+str(size)+'_T.npy'
ValidationLabelFileName= MainData+'_crisp_'+str(size)+'_L.csv'

SiteDF = pd.read_csv(SiteList)
MasterTensor = np.load(TensorFileName)
MasterValidationTensor = np.load(ValidationTensorFileName)
MasterLabelDF=pd.read_csv(LabelFileName)
MasterValidationDF = pd.read_csv(ValidationLabelFileName)
#Select the features in the tensor

Valid=np.zeros(12)
for n in range(1,13):
    if ('B'+str(n)) in FeatureSet:
        Valid[n-1]=1
        
MasterTensor = np.compress(Valid, MasterTensor, axis=3)
MasterValidationTensor = np.compress(Valid, MasterValidationTensor, axis=3)




#Start the filter process to isolate training and validation data
TrainingSites = SiteDF[SiteDF.Training == 1]
ValidationSites = SiteDF[SiteDF.Validation == 1]
TrainingSites.index = range(0,len(TrainingSites.Year))
ValidationSites.index = range(0,len(ValidationSites.Year))

#initialise the training and validation DFs to the master
TrainingDF = MasterLabelDF
TrainingTensor = MasterTensor
ValidationDF = MasterValidationDF
ValidationTensor = MasterValidationTensor

#isolate the sites, months and year and isolate the associated tensor values
MasterValid = (np.zeros(len(MasterLabelDF.index)))==1
for s in range(len(TrainingSites.Site)):
    Valid = (TrainingDF.Site == TrainingSites.Abbrev[s])&(TrainingDF.Year==TrainingSites.Year[s])&(TrainingDF.Month==TrainingSites.Month[s])
    MasterValid = MasterValid | Valid
    
TrainingDF = TrainingDF.loc[MasterValid]
TrainingTensor=np.compress(MasterValid,TrainingTensor, axis=0)

MasterValid = (np.zeros(len(MasterValidationDF.index)))
for s in range(len(ValidationSites.Site)):
    Valid = (MasterValidationDF.Site == ValidationSites.Abbrev[s])&(MasterValidationDF.Year==ValidationSites.Year[s])&(MasterValidationDF.Month==ValidationSites.Month[s])
    MasterValid = MasterValid | Valid
    
ValidationDF = ValidationDF.loc[MasterValid]
ValidationTensor = np.compress(MasterValid,ValidationTensor, axis=0)


#Set the labels
TrainingLabels = TrainingDF[LabelSet]
MajType=MajType+'Class'




#select desk-based or UAV-based for training and validation, if using UAV data, select the majority type
if UAVvalid:

    ValidationLabels = ValidationDF[MajType]
    ValidationTensor = np.compress(ValidationLabels>0,ValidationTensor, axis=0)
    ValidationLabels=ValidationLabels.loc[ValidationLabels>0]
  

#
else:

    ValidationLabels = ValidationDF.PolyClass
    ValidationTensor = np.compress(ValidationLabels>0,ValidationTensor, axis=0)
    ValidationLabels=ValidationLabels.loc[ValidationLabels>0]    
#check for empty dataframes and raise an error if found

if (len(TrainingDF.index)==0):
    raise Exception('There is an empty dataframe for TRAINING')
    
if (len(ValidationDF.index)==0):
    raise Exception('There is an empty dataframe for VALIDATION')
    
#Check that tensor lengths match label lengths

if (len(TrainingLabels.index)) != TrainingTensor.shape[0]:
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
NClasses = 3  #The number of classes in the data. This MUST be the same as the classes used to retrain the model
inShape = TrainingTensor.shape[1:]




##############################################################################
"""Instantiate the Neural Network pixel-based classifier""" 


# define the model with L2 regularization and dropout

 	# create model
if size==3: 
    Estimator = Sequential()
    Estimator.add(Conv2D(Nfilters,KernelSize, data_format='channels_last', input_shape=inShape, activation=NAF))
    Estimator.add(Flatten())
    Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))    


elif size==5:
    Estimator = Sequential()
    Estimator.add(Conv2D(Nfilters,KernelSize, data_format='channels_last', input_shape=inShape, activation=NAF))
    Estimator.add(Conv2D(Nfilters//2,KernelSize, activation=NAF))
    Estimator.add(Flatten())
    Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))
    
elif size==7:
    Estimator = Sequential()
    Estimator.add(Conv2D(Nfilters,KernelSize, data_format='channels_last', input_shape=inShape, activation=NAF))
    Estimator.add(Conv2D(Nfilters//2,KernelSize, activation=NAF))
    Estimator.add(Conv2D(Nfilters//4,KernelSize, activation=NAF))
    Estimator.add(Flatten())
    Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))
else:
    raise Exception('Invalid tile size, only 3,5 and 7 available')


#Tune an optimiser
Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)

# Compile model
Estimator.compile(loss='mean_squared_error', optimizer=Optim, metrics = ['accuracy'])
Estimator.summary()
  
###############################################################################
"""Data Fitting"""

X_train, X_test, y_train, y_test = train_test_split(TrainingTensor, TrainingLabels, test_size=0.0001, random_state=42)
print('Fitting CNN Classifier on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)#, class_weight=weights)

'''Validate the model by crisping up the test and validation data and predicting classes instead of memberships'''

PredictedPixels = Estimator.predict(ValidationTensor)
Y=ValidationLabels

##See if the dominant class is predicted correctly with F1
ClassPredicted = 1+np.argmax(PredictedPixels, axis=1)
ClassTrue = Y

report = metrics.classification_report(ClassTrue, ClassPredicted, digits = 3)
print('CRISP Validation results for relative majority ')
print(report)
print('\n \n')

ClassPredicted = 1+np.argmax(PredictedPixels, axis=1)
ClassPredicted_maj = ClassPredicted[np.max(PredictedPixels, axis=1)>0.50]
ClassTrue = Y[np.max(PredictedPixels, axis=1)>0.50]
report = metrics.classification_report(ClassTrue, ClassPredicted_maj, digits = 3)
print('CRISP Validation results for majority ')
print(report)
print('\n \n')

ClassPredicted = 1+np.argmax(PredictedPixels, axis=1)
ClassPredicted_pure = ClassPredicted[np.max(PredictedPixels, axis=1)>0.950]
ClassTrue = Y[np.max(PredictedPixels, axis=1)>0.950]
report = metrics.classification_report(ClassTrue, ClassPredicted_pure, digits = 3)
print('CRISP Validation results for pure class  ')
print(report)
print('\n \n')



