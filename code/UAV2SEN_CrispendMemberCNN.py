#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'


'''

This script will attempt to train a fuzzy classifier with endmembers infered from manual digitisation.
It is assumed that the image interpretation process will lead the user to digitise pure classes.
The pure class rasters are then transformed to categorical (1-hot encoding) which in effect means
that the digitised class pixels will be assigned a pure membership.

This script will use tis information in a CNN.  The script will only train and save the model.
Use UAV2SEN_GetErrorsCNN to estiate associated errors.

'''
###############################################################################
""" Libraries"""

from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Conv2D
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.transform import downscale_local_mean, resize
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os




##########################################################################################
"""User data input. Use the site template and list training and validation choices"""
#########################################################################################
'''Folder Settgings'''
MainData = 'EMPTY'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'EMPTY'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DataFolder = 'EMPTY' #location of processed tif files
ModelName = 'EMPTY'  #Name and location of the final model to be saved in DataFolder. Add .h5 extension

'''Model Features and Labels'''
UAVtrain = False #if true use the UAV class data to train the model, if false use desk-based data for training
MajType= 'Pure' #Majority type. only used if UAVtrain or valid is true. The options are RelMaj (relative majority class), Maj (majority) and Pure (95% unanimous).
FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']

'''CNN parameters'''
TrainingEpochs = 200#Use model tuning to adjust this and prevent overfitting
size=5#size of the tensor tiles
Nfilters=32 # size of the convolution kernels
LearningRate = 0.0005
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function
ModelTuning = False #Plot the history of the training losses.  Increase the TrainingEpochs if doing this.

'''Validation Settings'''
UAVvalid = True #if true use the UAV class data to validate.  If false, use desk-based polygons
ShowValidation = False #if true will show predicted class rasters for validation images from the site list






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
######################################################################################
    
    '''Check that the specified folders and files exist before processing starts'''
if not(os.path.isfile(MainData+'_fuzzy_'+str(size)+'_T.npy')):
    raise Exception('Main data file does not exist')
elif not(os.path.isfile(SiteList)):
    raise Exception('Site list csv file does not exist')
elif not(os.path.isdir(DataFolder)):
    raise Exception('Data folder with pre-processed data not defined')



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


#Start the filter process to isolate training and validation data
TrainingSites = SiteDF[SiteDF.Training == 1]
ValidationSites = SiteDF[SiteDF.Validation == 1]
TrainingSites.index = range(0,len(TrainingSites.Year))
ValidationSites.index = range(0,len(ValidationSites.Year))
#initialise the training and validation DFs to the master
TrainingDF = MasterLabelDF
TrainingTensor = MasterTensor
ValidationDF = MasterLabelDF
ValidationTensor = MasterTensor

#isolate the sites, months and year and isolate the associated tensor values
MasterValid = (np.zeros(len(MasterLabelDF.index)))==1
for s in range(len(TrainingSites.Site)):
    Valid = (TrainingDF.Site == TrainingSites.Abbrev[s])&(TrainingDF.Year==TrainingSites.Year[s])&(TrainingDF.Month==TrainingSites.Month[s])
    MasterValid = MasterValid | Valid
    
TrainingDF = TrainingDF.loc[MasterValid]
TrainingTensor=np.compress(MasterValid,TrainingTensor, axis=0)#will delete where valid is false

MasterValid = (np.zeros(len(MasterLabelDF.index)))==1
for s in range(len(ValidationSites.Site)):
    Valid = (ValidationDF.Site == ValidationSites.Abbrev[s])&(ValidationDF.Year==ValidationSites.Year[s])&(ValidationDF.Month==ValidationSites.Month[s])
    MasterValid = MasterValid | Valid
    
ValidationDF = ValidationDF.loc[MasterValid]
ValidationTensor = np.compress(MasterValid,ValidationTensor, axis=0)#will delete where valid is false 

    
MajType=MajType+'Class'


#select desk-based or UAV-based for training and validation, if using UAV data, select the majority type
if UAVtrain & UAVvalid:
    TrainLabels= TrainingDF[MajType]
    ValidationLabels = ValidationDF[MajType]
    TrainingTensor = np.compress(TrainLabels>0, TrainingTensor, axis=0)
    ValidationTensor = np.compress(ValidationLabels>0,ValidationTensor, axis=0)
    TrainLabels=TrainLabels.loc[TrainLabels>0]
    ValidationLabels=ValidationLabels.loc[ValidationLabels>0]
    
   
elif UAVtrain and ~(UAVvalid):
    TrainLabels= TrainingDF[MajType]
    ValidationLabels = ValidationDF.PolyClass
    TrainingTensor = np.compress(TrainLabels>0, TrainingTensor, axis=0)
    ValidationTensor = np.compress(ValidationLabels>0,ValidationTensor, axis=0)
    TrainLabels=TrainLabels.loc[TrainLabels>0]
    ValidationLabels=ValidationLabels.loc[ValidationLabels>0]
    
elif ~(UAVtrain) & UAVvalid:
    TrainLabels= TrainingDF.PolyClass
    ValidationLabels = ValidationDF[MajType]
    TrainingTensor = np.compress(TrainLabels>0, TrainingTensor, axis=0)
    ValidationTensor = np.compress(ValidationLabels>0,ValidationTensor, axis=0)
    TrainLabels=TrainLabels.loc[TrainLabels>0]
    ValidationLabels=ValidationLabels.loc[ValidationLabels>0]
    
#
else:
    TrainLabels= TrainingDF.PolyClass
    ValidationLabels = ValidationDF.PolyClass
    TrainingTensor = np.compress(TrainLabels>0, TrainingTensor, axis=0)
    ValidationTensor = np.compress(ValidationLabels>0,ValidationTensor, axis=0)
    TrainLabels=TrainLabels.loc[TrainLabels>0]
    ValidationLabels=ValidationLabels.loc[ValidationLabels>0]
    
#Select the central pixel in each tensor tile and make a table for non-convolutional NN classification
TrainingFeatures = np.squeeze(TrainingTensor[:,size//2,size//2,:])
ValidationFeatures = np.squeeze(ValidationTensor[:,size//2, size//2,:]) 
    
#check for empty dataframes and raise an error if found

if (len(TrainingDF.index)==0):
    raise Exception('There is an empty dataframe for training')
    
if (len(ValidationDF.index)==0):
    raise Exception('There is an empty dataframe for validation')
    
#Check that tensor lengths match label lengths

if (len(TrainLabels.index)) != TrainingTensor.shape[0]:
    raise Exception('Sample number mismatch for TRAINING tensor and labels')
    
if (len(ValidationLabels.index)) != ValidationTensor.shape[0]:
    raise Exception('Sample number mismatch for VALIDATION tensor and labels')
        
    

 





##############################################################################
##############################################################################
"""Instantiate the Neural Network pixel-based classifier""" 
#basic params
Ndims = TrainingTensor.shape[3] # Feature Dimensions. 
NClasses = 3  #The number of classes in the data.
inShape = TrainingTensor.shape[1:]



 	# create model

Estimator = Sequential()
Estimator.add(Conv2D(Nfilters,size, data_format='channels_last', input_shape=inShape, activation=NAF))
#Estimator.add(Conv2D(Nfilters,3, activation=NAF))#tentative deep architectures.
#Estimator.add(Conv2D(Nfilters,3,  activation=NAF))
#Estimator.add(Conv2D(Nfilters,3, activation=NAF))
Estimator.add(Flatten())
Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(NClasses+1, kernel_initializer='normal', activation='linear'))    




#Tune an optimiser
Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)

# Compile model
Estimator.compile(loss='mean_squared_error', optimizer=Optim, metrics = ['accuracy'])
Estimator.summary()
###############################################################################
"""Data Splitting"""
TrainLabels1Hot = to_categorical(TrainLabels)
ValidationLabels1Hot = to_categorical(ValidationLabels)
X_train, X_test, y_train, y_test = train_test_split(TrainingTensor, TrainLabels1Hot, test_size=0.2, random_state=42)



"""Data Fitting"""
print('Fitting CNN Classifier on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=5000, epochs=TrainingEpochs, verbose=Chatty)



'''Save model'''
ModelName=os.path.join(DataFolder,ModelName+'.h5')
Estimator.save(ModelName,save_format='h5')
