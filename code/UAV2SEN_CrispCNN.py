#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

###############################################################################
""" Libraries"""
import tensorflow as tf
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
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################
MainData = 'E:\\UAV2SEN\\MLdata\\FullData_4xnoise'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'E:\\UAV2SEN\\SiteList.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DatFolder = 'E:\\UAV2SEN\\FinalTif\\'  #location of above
TrainingEpochs = 5 #Typically this can be reduced
Nfilters = 128
UAVtrain = True #if true use the UAV class data to train the model, if false use desk-based
UAVvalid = True #if true use the UAV class data to validate.  If false, use desk-based polygons
size=5#size of the tensor tiles
KernelSize=3 # size of the convolution kernels
MajType= 'Pure' #Majority type. only used if UAVtrain or valid is true. The options are RelMaj (relative majority), Maj (majority) and Pure (95% unanimous).
ShowValidation = True #if true will show predicted class rasters for validation images from the site list
FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']

LearningRate = 0.001
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function

DoHistory = False #Plot the history of the training losses


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

MasterValid = (np.zeros(len(MasterLabelDF.index)))
for s in range(len(ValidationSites.Site)):
    Valid = (ValidationDF.Site == ValidationSites.Abbrev[s])&(ValidationDF.Year==ValidationSites.Year[s])&(ValidationDF.Month==ValidationSites.Month[s])
    MasterValid = MasterValid | Valid
    
ValidationDF = ValidationDF.loc[MasterValid]
ValidationTensor = np.compress(MasterValid,ValidationTensor, axis=0)#will delete where valid is false 

    



#select desk-based or UAV-based for training and validation, if using UAV data, select the majority type
if UAVtrain & UAVvalid:
    TrainingDF=TrainingDF[TrainingDF.PolyClass==-1]
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
    Valid= TrainingDF.PolyClass==-1
    TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    Valid = ValidationDF.PolyClass==-1
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    if 'RelMaj' in  MajType:
        TrainingDF = TrainingDF[TrainingDF.RelMajClass>0]
        ValidationDF = ValidationDF[ValidationDF.RelMajClass>0]
        TrainLabels = TrainingDF.RelMajClass
        ValidationLabels = ValidationDF.RelMajClass
        Valid= TrainingDF.RelMajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        Valid = ValidationDF.RelMajClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    elif 'Maj' in MajType:
        TrainingDF = TrainingDF[TrainingDF.MajClass>0]
        ValidationDF = ValidationDF[ValidationDF.MajClass>0]
        TrainLabels = TrainingDF.MajClass
        ValidationLabels = ValidationDF.MajClass
        Valid= TrainingDF.MajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        Valid = ValidationDF.MajClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    
    elif 'Pure' in MajType:
        TrainingDF = TrainingDF[TrainingDF.PureClass>0]
        ValidationDF = ValidationDF[ValidationDF.PureClass>0]
        TrainLabels = TrainingDF.PureClass
        ValidationLabels = ValidationDF.PureClass
        Valid= TrainingDF.PureClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        Valid = ValidationDF.PureClass>0
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
        
elif UAVtrain and ~(UAVvalid):
    TrainingDF=TrainingDF[TrainingDF.PolyClass==-1]
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    ValidationLabels = ValidationDF.PolyClass
    Valid= TrainingDF.PolyClass==-1
    TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    Valid = ValidationDF.PolyClass>0
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    if 'RelMaj' in  MajType:
        TrainingDF = TrainingDF[TrainingDF.RelMajClass>0]
        TrainLabels = TrainingDF.RelMajClass
        Valid= TrainingDF.RelMajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)

    elif 'Maj' in MajType:
        TrainingDF = TrainingDF[TrainingDF.MajClass>0]
        TrainLabels = TrainingDF.MajClass
        Valid= TrainingDF.MajClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    
    elif 'Pure' in MajType:
        TrainingDF = TrainingDF[TrainingDF.PureClass>0]
        TrainLabels = TrainingDF.PureClass
        Valid= TrainingDF.PureClass>0
        TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
        
elif ~(UAVtrain) & UAVvalid:
    TrainingDF=TrainingDF[TrainingDF.PolyClass>0]
    TrainLabels = TrainingDF.PolyClass
    ValidationDF=ValidationDF[ValidationDF.PolyClass==-1]
    Valid= TrainingDF.PolyClass>0
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
    TrainingDF=TrainingDF[TrainingDF.PolyClass>0]
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    TrainLabels = TrainingDF.PolyClass
    ValidationDF=ValidationDF[ValidationDF.PolyClass>0]
    ValidationLabels = ValidationDF.PolyClass
    Valid= TrainingDF.PolyClass>0
    TrainingTensor = np.compress(Valid, TrainingTensor, axis=0)
    Valid = ValidationDF.PolyClass>0
    ValidationTensor = np.compress(Valid, ValidationTensor, axis=0)
    
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
if size==3: 
    Estimator = Sequential()
    Estimator.add(Conv2D(Nfilters,KernelSize, data_format='channels_last', input_shape=inShape, activation=NAF))
    Estimator.add(Flatten())
    Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(NClasses+1, kernel_initializer='normal', activation='linear'))    


elif size==5:
    Estimator = Sequential()
    Estimator.add(Conv2D(Nfilters,KernelSize, data_format='channels_last', input_shape=inShape, activation=NAF))
    Estimator.add(Conv2D(Nfilters//2,KernelSize, activation=NAF))
    Estimator.add(Flatten())
    Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(NClasses+1, kernel_initializer='normal', activation='linear'))
    
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
    Estimator.add(Dense(NClasses+1, kernel_initializer='normal', activation='linear'))
else:
    raise Exception('Invalid tile size, only 3,5 and 7 available')
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
report = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(PredictedPixels, axis=1), digits = 3)
print('20% Test classification results for ')
print(report)
      

# #Fit the predictor to the external validation site
PredictedPixels = Estimator.predict(ValidationTensor)
report = metrics.classification_report(ValidationLabels, np.argmax(PredictedPixels, axis=1), digits = 3)
print('Out-of-Sample validation results for ')
print(report)


'''Show the classified validation images'''
if ShowValidation:
    for s in range(len(ValidationSites.index)):
        ValidRasterName = DatFolder+ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s])+'_S2.tif'
        ValidRaster = io.imread(ValidRasterName)
        ValidTensor = slide_raster_to_tiles(ValidRaster, size)
        print('bla')
        Valid=np.zeros(12)
        for n in range(1,13):
            if ('B'+str(n)) in FeatureSet:
                Valid[n-1]=1
        
        ValidTensor = np.compress(Valid, ValidTensor, axis=3)
        print(ValidTensor.shape)
        PredictedPixels = Estimator.predict(ValidTensor)
        PredictedPixels =np.argmax(PredictedPixels, axis=1)
        PredictedClassImage = np.uint8(PredictedPixels.reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedClassImage[0,0]=1
        PredictedClassImage[0,1]=2
        PredictedClassImage[0,2]=3
        PredictedClassImage[0,3]=0
        cmapCHM = colors.ListedColormap(['black','red','green','blue'])
        plt.figure()
        plt.imshow(PredictedClassImage, cmap=cmapCHM)
        class0_box = mpatches.Patch(color='black', label='Unclassified')
        class1_box = mpatches.Patch(color='red', label='Sediment')
        class2_box = mpatches.Patch(color='green', label='Veg.')
        class3_box = mpatches.Patch(color='blue', label='Water')
        ax=plt.gca()
        ax.legend(handles=[class0_box, class1_box,class2_box,class3_box])
        plt.title(ValidRasterName)


