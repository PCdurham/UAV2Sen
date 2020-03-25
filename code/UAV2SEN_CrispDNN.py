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
from sklearn import metrics
from sklearn.model_selection import train_test_split




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################
MainData = 'F:\\MLdata\\NNdebug'  #no extensions, will be fleshed out below
SiteList = 'F:\\SiteList.csv'#this has the lists of sites with name, month and year
DatFolder = 'F:\\FinalTif\\' #location of above
TrainingEpochs = 150 #Typically this can be reduced
UAV = True #if true the crisp class will run with the UAV based classifications crisped up to majorities.  If false, use desk-based polygons
MajType= 'RelMaj' #only used if UAV is true. The options are RelMaj, Maj and Pure.
FeatureSet = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'] # pick predictor bands




'''Load the tensors, squeeze to get single pixel values and filter out the required training and validation data.'''
TensorFileName = MainData+'__crisp_T.npy'
LabelFileName = MainData+'__crisp_L.dat'

SiteDF = pd.read_csv(SiteList)
Tensor = np.load(TensorFileName)
MasterLabelDF=pd.read_feather(LabelFileName)

#Remove the 4X data augmentation and take only points 1,4,8,etc...
PointNums = np.asarray(range(0,len(MasterLabelDF.RelMajClass)))
Spots = PointNums%4
Valid = Spots==0

#Subsample the labels and fix the index
MasterLabelDF = MasterLabelDF.loc[Valid]
MasterLabelDF.index = range(0,len(MasterLabelDF.RelMajClass))

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

#select desk-based or UAV-based
if UAV:
    TrainDF=TrainDF[TrainDF.PolyClass==-1]
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
else:
    TrainDF=TrainDF[TrainDF.PolyClass>0]
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    
#check for empty dataframes and raise an error if found

if (len(TrainDF.B1)==0):
    raise Exception('There is an empty dataframe for training')
    
if (len(ValidationDF.B1)==0):
    raise Exception('There is an empty dataframe for validation')
    

#use the FeatureSet definition and class type to get desired features and labels

if not(UAV):
    TrainDF = TrainDF[TrainDF.PolyClass>0]
    ValidationDF = ValidationDF[ValidationDF.PolyClass>0]
    TrainFeatures = TrainDF[FeatureSet]
    ValidationFeatures = ValidationDF[FeatureSet]
    TrainLabels = TrainDF.PolyClass
    ValidationLabels = ValidationDF.PolyClass
    
elif 'RelMaj' in  MajType:
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
    
else:
    raise Exception('Wrong type of label selection')
        
    
    



'''MODEL PARAMETERS''' #These would usually not be edited
 
Ndims = len(TrainFeatures.columns) # Feature Dimensions. 
NClasses = len(np.unique(TrainLabels))  #The number of classes in the data. This MUST be the same as the classes used to retrain the model
LearningRate = 0.0001
Chatty = 1 # set the verbosity of the model training. 




##############################################################################
"""Instantiate the Neural Network pixel-based classifier""" 
   


# define the very deep model with L2 regularization and dropout

 	# create model
def nodrop_model_L2D():
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
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    return model
    
EstimatorNN = KerasClassifier(build_fn=nodrop_model_L2D, epochs=TrainingEpochs, 
                                  batch_size=1000, verbose=Chatty)

  
###############################################################################
"""Data Fitting"""
 
X_train, X_test, y_train, y_test = train_test_split(TrainFeatures, TrainLabels, test_size=0.2, random_state=42)
print('Fitting MLP Classifier on ' + str(len(X_train)) + ' pixels')
EstimatorNN.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)#, class_weight=weights)

    
#Fit the predictor to test pixels
PredictedPixels = EstimatorNN.predict(X_test)

# #Produce TTS classification reports 
report = metrics.classification_report(y_test, PredictedPixels, digits = 3)
print('20% Test classification results for ')
print(report)
      

# #Fit the predictor to the external validation site
PredictedPixels = EstimatorNN.predict(ValidationFeatures)
report = metrics.classification_report(ValidationLabels, PredictedPixels, digits = 3)
print('Out-of-Sample validation results for ')
print(report)