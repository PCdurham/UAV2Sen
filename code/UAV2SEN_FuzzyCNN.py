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
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os




#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################

'''Folder Settgings'''
MainData = 'E:\\UAV2SEN\\MLdata\\FullData_4xnoise'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'E:\\UAV2SEN\\SiteList.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DataFolder = 'E:\\UAV2SEN\\FinalTif\\'  #location of processed tif files
ModelName = 'test.h5'  #Name and location of the final model to be saved in DataFolder. Add .h5 extension

'''Model Features and Labels'''
FeatureSet =  ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']
LabelSet = ['WaterMem', 'VegMem','SedMem' ]

'''CNN parameters'''
TrainingEpochs = 50 #Typically this can be reduced
Nfilters = 64 
size=5#size of the tensor tiles
KernelSize=3 # size of the convolution kernels. Caution becasue mis-setting this could cause bugs in the network definition.  Best keep at 3.
LearningRate = 0.0001
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function
ModelTuning = False #Plot the history of the training losses.  Increase the TrainingEpochs if doing this.


'''Validation Settings'''
UT=0.95# upper and lower thresholds to elimninate pure classes from fuzzy error Pred vs Obs displays
LT=0.05
ShowValidation = True#if true fuzzy classified images of the validation sites will be displayed




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


#Set the labels
TrainingLabels = TrainingDF[LabelSet]
ValidationLabels = ValidationDF[LabelSet]
    
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
    Estimator.add(Conv2D(Nfilters,KernelSize, activation=NAF))
    Estimator.add(Flatten())
    Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    Estimator.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))
    
elif size==7:
    Estimator = Sequential()
    Estimator.add(Conv2D(Nfilters,KernelSize, data_format='channels_last', input_shape=inShape, activation=NAF))
    Estimator.add(Conv2D(Nfilters,KernelSize, activation=NAF))
    Estimator.add(Conv2D(Nfilters,KernelSize, activation=NAF))
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
"""Data Splitting"""

X_train, X_test, y_train, y_test = train_test_split(TrainingTensor, TrainingLabels, test_size=0.2, random_state=42)
if ModelTuning:
    #Split the data for tuning. Use a double pass of train_test_split to shave off some data
   
    history = Estimator.fit(X_train, y_train, epochs = TrainingEpochs, batch_size = 1000, validation_data = (X_test, y_test))
    #Plot the test results
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    
    epochs = range(1, len(loss_values) + 1)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(epochs, loss_values, 'ks', label = 'Training loss')
    plt.plot(epochs,val_loss_values, 'k:', label = 'Validation loss')
    #plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.subplot(1,2,2)
    plt.plot(epochs, acc_values, 'ko', label = 'Training accuracy')
    plt.plot(epochs, val_acc_values, 'k', label = 'Validation accuracy')
    #plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.rcParams.update({'font.size': 22})
    plt.rcParams.update({'font.weight': 'bold'}) 
    plt.show()

    
    raise Exception("Tuning Finished, adjust parameters and re-train the model") # stop the code if still in tuning phase.


'''Model Fitting'''
print('Fitting CNN Classifier on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)#, class_weight=weights)
#EstimatorRF.fit(X_train, y_train)
    


'''Save model'''
ModelName=os.path.join(DataFolder,ModelName)
Estimator.save(ModelName,save_format='h5')


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
jplot = sns.jointplot("C1 Obs", 'C1 Pred', data=ErrFrame1, kind="hex", color='b')
jplot.ax_marg_x.set_xlim(-0.2, 1.2)
jplot.ax_marg_y.set_ylim(-0.2, 1.2)


ErrFrame2 = ErrFrame[ErrFrame['C2 Obs']<UT]
ErrFrame2 = ErrFrame2[ErrFrame2['C2 Obs']>LT]
jplot = sns.jointplot("C2 Obs", 'C2 Pred', data=ErrFrame2, kind="hex", color='g')
jplot.ax_marg_x.set_xlim(-0.2, 1.2)
jplot.ax_marg_y.set_ylim(-0.2, 1.2)


ErrFrame3 = ErrFrame[ErrFrame['C3 Obs']<UT]
ErrFrame3 = ErrFrame3[ErrFrame3['C3 Obs']>LT]
jplot = sns.jointplot("C3 Obs", 'C3 Pred', data=ErrFrame3, kind="hex", color='r')
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
print('20% test mean error =', str(np.mean(Errors)))
print('20% test RMS error =', str(RMSall))
print('\n')
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
print(reg.summary())#


#Validation data
PredictedPixels = Estimator.predict(ValidationTensor)
Y=ValidationLabels

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
print('Validation mean error =', str(np.mean(Errors)))
print('Validation RMS error =', str(RMSall))
print('\n')
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
print(reg.summary())#


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
        
        ValidTensor = np.compress(Valid, ValidTensor, axis=3)
        #print(ValidTensor.shape)
        PredictedPixels = Estimator.predict(ValidTensor)
        

        PredictedWaterImage = np.int16(255*PredictedPixels[:,0].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedVegImage = np.int16(255*PredictedPixels[:,1].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedSedImage = np.int16(255*PredictedPixels[:,2].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedClassImage=np.int16(np.zeros((PredictedWaterImage.shape[0], PredictedWaterImage.shape[1],3)))
        PredictedClassImage[:,:,0]=PredictedSedImage
        PredictedClassImage[:,:,1]=PredictedVegImage
        PredictedClassImage[:,:,2]=PredictedWaterImage

        cmapCHM = colors.ListedColormap(['black','red','lime','blue'])
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(np.int16(255*(ValidRasterIR)))
        plt.title(ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s]) + ' Bands (11,3,2)')
        plt.subplot(2,2,2)
        plt.imshow(PredictedClassImage)
        plt.title(' Fuzzy Class')
        class1_box = mpatches.Patch(color='red', label='Sediment')
        class2_box = mpatches.Patch(color='lime', label='Veg.')
        class3_box = mpatches.Patch(color='blue', label='Water')
        ax=plt.gca()
        ax.legend(handles=[class1_box,class2_box,class3_box], bbox_to_anchor=(1, -0.2),prop={'size': 30})
        
        #ax.legend(handles=[class1_box,class2_box,class3_box])
        plt.subplot(2,2,3)
        plt.imshow(np.int16(UAVRasterRGB))
        plt.xlabel('UAV Ground-Truth Data')
        
        
        
    


