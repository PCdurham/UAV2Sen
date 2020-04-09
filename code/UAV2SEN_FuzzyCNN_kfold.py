#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

'''

This script performs fuzzy classification of river corridor features with a CNN.  the script allows 
the user to tune, train, validate and save CNN models.  The required inputs must be produced with the
UAV2SEN_MakeFuzzyTensor.py script.

'''

###############################################################################
""" Libraries"""
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Conv2D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
#import statsmodels.api as sm
from skimage import io
from skimage.transform import downscale_local_mean, resize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################

'''Folder Settgings'''
MainData = 'F:\\UAV2SEN\\MLdata\\Fulldata_4xnoise'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'F:\\UAV2SEN\\Results\\Experiments\\SiteList_exp1.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DataFolder = 'F:\\UAV2SEN\\FinalTif\\'  #location of processed tif files
ModelName = 'Best_deep932a'  #Name and location of the final model to be saved in DataFolder. Add .h5 extension

'''Model Features and Labels'''
FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12']
LabelSet = ['WaterMem', 'VegMem','SedMem' ]

'''CNN parameters'''
TrainingEpochs = 200 #Use model tuning to adjust this and prevent overfitting
Nfilters= 128
size=9#size of the tensor tiles
LearningRate = 0.0005
BatchSize=5000
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function
ModelTuning = False #Plot the history of the training losses.  Increase the TrainingEpochs if doing this.


'''Validation Settings'''

ShowValidation = False#if true fuzzy classified images of the validation sites will be displayed.  Warning: xpensive to compute.
PublishHist = True#best quality historgams
Ytop=6.5
SaveName='F:\\UAV2SEN\\Results\\Experiments\\Hist_deep932a.png'
OutDPI=600
Fname='Arial'
Fsize=14
Fweight='bold'
Lweight=1


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

def GetDominantClassErrors(Obs, Pred):
    Dominant=np.zeros((len(Obs),2))
    for s in range(len(Obs)):
        order=np.argsort(Obs[s,:])#dominant class in ground truth
        Dominant[s,0]=Pred[s,order[-1]]-Obs[s,order[-1]]
        Dominant[s,1]=Pred[s,order[-2]]-Obs[s,order[-2]]
    return Dominant
        


####################################################################################
'''Check that the specified folders and files exist before processing starts'''
if not(os.path.isfile(MainData+'_fuzzy_'+str(size)+'_T.npy')):
    raise Exception('Main data file does not exist')
elif not(os.path.isfile(SiteList)):
    raise Exception('Site list csv file does not exist')
elif not(os.path.isdir(DataFolder)):
    raise Exception('Data folder with pre-processed data not defined')
    

'''Load the tensors and filter out the required training and validation data.'''
TensorFileName = MainData+'_fuzzy_'+str(size)+'_T.npy'
LabelFileName = MainData+'_fuzzy_'+str(size)+'_L.csv'

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


#Set the labels
TrainingLabels = TrainingDF[LabelSet]
ValidationLabels = ValidationDF[LabelSet]

#remove the 4x data augmentation in the validation data
Data=np.asarray(range(0,len(ValidationLabels)))
Valid=(Data%4)==0
ValidationTensor = np.compress(Valid,ValidationTensor, axis=0)
ValidationLabels=ValidationLabels[Valid]
    
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
    

 





##############################################################################
"""Instantiate the Neural Network pixel-based classifier""" 
#basic params
Ndims = TrainingTensor.shape[3] # Feature Dimensions. 
NClasses = len(LabelSet)  #The number of classes in the data.
inShape = TrainingTensor.shape[1:]



 	# create model
def score_model():
    Estimator = Sequential()
    Estimator.add(Conv2D(Nfilters,3, data_format='channels_last', input_shape=inShape, activation=NAF))
    Estimator.add(Conv2D(Nfilters,3, activation=NAF))#tentative deep architecture. gives poor results!
    Estimator.add(Conv2D(Nfilters,3,  activation=NAF))
    Estimator.add(Conv2D(Nfilters,3, activation=NAF))
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
    return Estimator
  
###############################################################################
"""Data Splitting"""

X_train, X_test, y_train, y_test = train_test_split(TrainingTensor, TrainingLabels, test_size=0.0001, random_state=int(42*np.random.random(1)))




'''Model Fitting with a 5 fold kfold'''
print('Fitting CNN Classifier on ' + str(len(X_train)) + ' pixels')
#Estimator.fit(X_train, y_train, batch_size=BatchSize, epochs=TrainingEpochs, verbose=Chatty)
Estimator = KerasRegressor(build_fn=score_model, epochs=TrainingEpochs, batch_size=5000, verbose=Chatty)
kfold = KFold(n_splits=5)
results = cross_val_score(Estimator, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
print('FINISHED')
print(' ')
print("Validation: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    



