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
from tensorflow.keras.models import Sequential, load_model
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
MainData = 'E:\\UAV2SEN\\MLdata\\CNNDebug'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'E:\\UAV2SEN\\SiteList.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
ModelName = 'E:\\UAV2SEN\\MLdata\\CNNdebugged2.h5'  #Name and location of the trained CNN
TrainingEpochs = 10 #Typically this can be reduced
Nfilters = 32
size=5#size of the tensor tiles
KernelSize=5 # size of the convolution kernels
FeatureSet =  ['B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']
LabelSet = ['WaterMem', 'VegMem','SedMem' ]
UAVvalid = True #if true use the UAV class data to validate.  If false, use desk-based polygons
MajType= 'Pure' #Majority type. only used if UAVtrain or valid is true. The options are RelMaj (relative majority), Maj (majority) and Pure (95% unanimous).

LearningRate = 0.001
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function

DoHistory = False #Plot the history of the training losses

'''Load the trained fuzzy CNN'''
Estimator = load_model(ModelName)

'''Load the crisp tensors and filter out the required validation data.'''
TensorFileName = MainData+'_crisp_'+str(size)+'_T.npy'
LabelFileName = MainData+'_crisp_'+str(size)+'_L.dat'

SiteDF = pd.read_csv(SiteList)
MasterTensor = np.load(TensorFileName)
MasterLabelDF=pd.read_csv(LabelFileName)

#Remove the 4X data augmentation only relevant to the CNNs and take only points 0,4,8,etc...
PointNums = np.asarray(range(0,len(MasterLabelDF.RelMajClass)))
Spots = PointNums%4
Valid = Spots==0

#Subsample the labels and fix the index
MasterLabelDF = MasterLabelDF.loc[Valid]
MasterLabelDF.index = range(0,len(MasterLabelDF.RelMajClass))

#Subsample the tensor
MasterTensor = np.compress(Valid, MasterTensor, axis=0)


#Select the features in the tensor

Valid=np.zeros(12)
for n in range(1,13):
    if ('B'+str(n)) in FeatureSet:
        Valid[n-1]=1
        
MasterTensor = np.compress(Valid, MasterTensor, axis=3)

#Start the filter process to isolate training and validation data
ValidationSites = SiteDF[SiteDF.Validation == 1]

#isolate the site. first labels then tensors
ValidationDF = MasterLabelDF[MasterLabelDF['Site'].isin(ValidationSites.Abbrev.to_list())]
Valid = MasterLabelDF['Site'].isin(ValidationSites.Abbrev.to_list())
ValidationTensor = np.compress(Valid, MasterTensor, axis=0)

#isolate the year
ValidationDF = ValidationDF[ValidationDF['Year'].isin(ValidationSites.Year.to_list())]
Valid = ValidationDF['Year'].isin(ValidationSites.Year.to_list())
ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)

#isolate the month
ValidationDF = ValidationDF[ValidationDF['Month'].isin(ValidationSites.Month.to_list())]
Valid = ValidationDF['Month'].isin(ValidationSites.Month.to_list())
ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)

#Set the labels
#select desk-based or UAV-based for validation, if using UAV data, select the majority type
if UAVvalid:
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
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
        Valid = ValidationDF.PureClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
   
else:
    
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    ValidationLabels = ValidationDF.PolyClass
    Valid = ValidationDF.PolyClass>0
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    

#check for empty dataframes and raise an error if found

    
if (len(ValidationDF.index)==0):
    raise Exception('There is an empty dataframe for validation')
    
#Check that tensor lengths match label lengths
    
if (len(ValidationLabels.index)) != ValidationTensor.shape[0]:
    raise Exception('Sample number mismatch for VALIDATION tensor and labels')
    

'''Transform fuzzy predictions of the CNN to crisp predictions'''

'''Validate the model'''
# #Fit the predictor to the external validation site
PredictedPixels = Estimator.predict(ValidationTensor)
report = metrics.classification_report(ValidationLabels, PredictedPixels, digits = 3)
print('Out-of-Sample validation results for ')
print(report)