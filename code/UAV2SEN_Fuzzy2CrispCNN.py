#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

'''

This script performs crisp classification of river corridor features with a fuzzy CNN.  the script allows 
the user to use a fuzzy classifier's predictions to predict crisp classes according to 3 rules:
    
    - a relative majority where 1 class has a higher membershio than any other
    - a majority where 1 class has more than 50% membership
    - a (quasi) pure class where 1 class has more than 95% membership)
   
The required inputs must be produced with the UAV2SEN_MakeFuzzyTensor.py script.

'''

###############################################################################
""" Libraries"""
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Conv2D
from sklearn import metrics
from sklearn.model_selection import train_test_split
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################

'''Folder Settgings'''
MainData = 'EMPTY'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'EMPTY'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DataFolder = 'EMPTY'  #location of processed tif files


'''Model Features and Labels'''
LabelSet = ['WaterMem', 'VegMem','SedMem' ]
FeatureSet = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']

'''CNN parameters'''
TrainingEpochs = 50 #Use model tuning to adjust this and prevent overfitting
Nfilters = 32
size=5#size of the tensor tiles
KernelSize=3 # size of the convolution kernels
LearningRate = 0.001
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function

'''Validation Settings'''
UAVvalid = False #if true use the UAV class data to validate.  If false, use desk-based polygons and ignore majority type
MajType = 'RelMaj' #majority type to use in the display only, the F1 scores will be produced for all majority types
PureThresh = 0.95 #If MajType is Pure, you can adjust the threshold for a pure class here.
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
ClassPredicted_pure = ClassPredicted[np.max(PredictedPixels, axis=1)>PureThresh]
ClassTrue = Y[np.max(PredictedPixels, axis=1)>PureThresh]
report = metrics.classification_report(ClassTrue, ClassPredicted_pure, digits = 3)
print('CRISP Validation results for pure class  ')
print(report)
print('\n \n')



'''Show the classified validation images'''
if ShowValidation:
    for s in range(len(ValidationSites.index)):
        UAVRaster = io.imread(DataFolder+ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s])+'_UAVCLS.tif')
        UAVRaster[UAVRaster>3] =0
        UAVRaster[UAVRaster<1] =0
        UAVRasterRGB = np.zeros((UAVRaster.shape[0], UAVRaster.shape[1], 3))
        UAVRasterRGB[:,:,0]=255*(UAVRaster==3)
        UAVRasterRGB[:,:,1]=255*(UAVRaster==2)
        UAVRasterRGB[:,:,2]=255*(UAVRaster==1)
        
        ValidRasterName = DataFolder+ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s])+'_S2.tif'
        ValidRaster = io.imread(ValidRasterName)
        ValidRasterIR = np.zeros((ValidRaster.shape[0], ValidRaster.shape[1],3))
        ValidRasterIR[:,:,0] = ValidRaster[:,:,10]
        ValidRasterIR[:,:,1] = ValidRaster[:,:,4]
        ValidRasterIR[:,:,2] = ValidRaster[:,:,3]
        ValidRasterIR = ValidRasterIR/np.max(np.unique(ValidRasterIR))
        ValidTensor = slide_raster_to_tiles(ValidRaster, size)
        Valid=np.zeros(12)
        for n in range(1,13):
            if ('B'+str(n)) in FeatureSet:
                Valid[n-1]=1
        
        ValidTensorFinal = np.compress(Valid, ValidTensor, axis=3)
        #print(ValidTensor.shape)
        PredictedPixels = Estimator.predict(ValidTensorFinal)
        
        if MajType=='RelMajClass':
            ClassPredicted = 1+np.argmax(PredictedPixels, axis=1)
        elif MajType=='MajClass':
            ClassPredicted = 1+np.argmax(PredictedPixels, axis=1)
            ClassPredicted[np.max(PredictedPixels, axis=1)<0.50] = 0
        elif MajType=='PureClass':
            ClassPredicted = 1+np.argmax(PredictedPixels, axis=1)
            ClassPredicted[np.max(PredictedPixels, axis=1)<PureThresh] =0

        PredictedPixelsRaster = np.int16(ClassPredicted.reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedClassImage=np.int16(np.zeros((PredictedPixelsRaster.shape[0], PredictedPixelsRaster.shape[1],3)))
        PredictedClassImage[:,:,0]=100*(PredictedPixelsRaster==3)
        PredictedClassImage[:,:,1]=100*(PredictedPixelsRaster==2)
        PredictedClassImage[:,:,2]=100*(PredictedPixelsRaster==1)



        cmapCHM = colors.ListedColormap(['black','red','lime','blue'])
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(np.int16(255*(ValidRasterIR)))
        plt.title(ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s]) + ' Bands (11,3,2)')
        plt.subplot(2,2,2)
        plt.imshow(PredictedClassImage)
        if MajType=='RelMajClass':
            plt.title(' Relative Majority Class Pixels')
            
        elif MajType=='MajClass':
            plt.title(' Majority Class Pixels')
            
        elif MajType=='PureClass':
            plt.title(' Pure Class Pixels')
           
        
        class1_box = mpatches.Patch(color='red', label='Sediment')
        class2_box = mpatches.Patch(color='lime', label='Veg.')
        class3_box = mpatches.Patch(color='blue', label='Water')
        ax=plt.gca()
        ax.legend(handles=[class1_box,class2_box,class3_box], bbox_to_anchor=(1, -0.2),prop={'size': 20})
        
        #ax.legend(handles=[class1_box,class2_box,class3_box])
        plt.subplot(2,2,3)
        plt.imshow(np.int16(UAVRasterRGB))
        plt.xlabel('UAV Ground-Truth Data')
        



