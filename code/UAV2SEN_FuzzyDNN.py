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
#import statsmodels.api as sm
from skimage import io
from skimage.transform import downscale_local_mean, resize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os
from IPython import get_ipython



#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################

'''Folder Settgings'''
MainData = 'F:\\UAV2SEN\\MLdata\\Fulldata_4xnoise'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'F:\\UAV2SEN\\Results\\Experiments\\SiteList_exp1.csv'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DataFolder = 'F:\\UAV2SEN\\FinalTif\\'  #location of processed tif files
ModelName = 'Fuzzy_DNN_exp1'  #Name and location of the final model to be saved in DataFolder. Add .h5 extension

'''Model Features and Labels'''
FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']
LabelSet = ['WaterMem', 'VegMem','SedMem' ]

'''DNN parameters'''
TrainingEpochs = 200 #Use model tuning to adjust this and prevent overfitting
size=3#size of the tensor tiles. very little effect for the DNN, size 3 tiles have a bit more data due to edge effects during tiling
LearningRate = 0.0005
BatchSize=5000
Chatty = 1 # set the verbosity of the model training. 
NAF = 'tanh' #NN activation function
ModelTuning = False #Plot the history of the training losses.  Increase the TrainingEpochs if doing this.


'''Validation Settings'''

ShowValidation = False#if true fuzzy classified images of the validation sites will be displayed.  Warning: xpensive to compute.
PublishHist = True#best quality historgams
Ytop=6.5
SaveName='F:\\UAV2SEN\\Results\\Experiments\\HistDNN_exp1.png'
OutDPI=600
Fname='Arial'
Fsize=10
Fweight='bold'
Lweight=1.5


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

#Select the central pixel in each tensor tile and make a table for non-convolutional NN classification
TrainingFeatures = np.squeeze(TrainingTensor[:,size//2,size//2,:])
ValidationFeatures = np.squeeze(ValidationTensor[:,size//2, size//2,:])





    
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
Ndims = TrainingFeatures.shape[1] # Feature Dimensions. 
NClasses = TrainingLabels.shape[1]  #The number of classes in the data. This MUST be the same as the classes used to retrain the model



 	# create model

Estimator = Sequential()
Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001),input_dim=Ndims, kernel_initializer='normal', activation=NAF))
#Estimator.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
Estimator.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
Estimator.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))    




#Tune an optimiser
Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)

# Compile model
Estimator.compile(loss='mean_squared_error', optimizer=Optim, metrics = ['accuracy'])
Estimator.summary()
  
###############################################################################
"""Data Splitting"""

X_train, X_test, y_train, y_test = train_test_split(TrainingFeatures, TrainingLabels, test_size=0.2, random_state=42)


if ModelTuning:
    #Split the data for tuning. Use a double pass of train_test_split to shave off some data
    #get_ipython().run_line_magic('matplotlib', 'qt')
    history = Estimator.fit(X_train, y_train, epochs = TrainingEpochs, batch_size = BatchSize, validation_data = (X_test, y_test))
    #Plot the test results
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    
    epochs = range(1, len(loss_values) + 1)
    mpl.rc('xtick', labelsize=20) 
    mpl.rc('ytick', labelsize=20) 
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
Estimator.fit(X_train, y_train, batch_size=BatchSize, epochs=TrainingEpochs, verbose=Chatty)

    


'''Save model'''
ModelName=os.path.join(DataFolder,ModelName+'.h5')
Estimator.save(ModelName,save_format='h5')


'''Validate the model'''
#Test data
PredictedPixels = Estimator.predict(X_test)


DominantErrors=GetDominantClassErrors(np.asarray(y_test), PredictedPixels)
D={'Dominant_Error':DominantErrors[:,0],'Sub-Dominant_Error':DominantErrors[:,1] }
DominantErrorsDF = pd.DataFrame(D)

RMSdom = np.sqrt(np.mean(DominantErrors[:,0]*DominantErrors[:,0]))
RMSsubdom = np.sqrt(np.mean(DominantErrors[:,1]*DominantErrors[:,1]))
QPAdom = np.sum(np.abs(DominantErrors[:,0])<0.25)/len(DominantErrors[:,0])
QPAsubdom = np.sum(np.abs(DominantErrors[:,1])<0.25)/len(DominantErrors[:,1])

print('20% test mean error for DOMINANT class= ', str(np.mean(DominantErrors[:,0])))
print('20% test RMS error for DOMINANT class= ', str(RMSdom))
print('20% test QPA for the DOMINANT class= '+ str(QPAdom))
print('20% test mean error for SUB-DOMINANT class= ', str(np.mean(DominantErrors[:,1])))
print('20% test RMS error for SUB-DOMINANT class= ', str(RMSsubdom))
print('20% test QPA for the SUB-DOMINANT class= '+ str(QPAsubdom))

print('\n')
#get_ipython().run_line_magic('matplotlib', 'qt')
if PublishHist:
    mpl.rcParams['font.family'] = Fname
    plt.rcParams['font.size'] = Fsize
    plt.rcParams['axes.linewidth'] = Lweight
    plt.rcParams['font.weight'] = Fweight
    datbins=np.linspace(-1,1,40)
    #get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure()
    plt.subplot(2,2,1)
    sns.distplot(DominantErrorsDF.Dominant_Error,axlabel=' ', bins=datbins, color='k', kde=False, norm_hist=True)
    #plt.ylim(0, Ytop)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.xticks(fontsize=Fsize)
    plt.yticks(fontsize=Fsize)
    plt.ylabel('Test Density', fontweight=Fweight)
    plt.subplot(2,2,2)
    sns.distplot(DominantErrorsDF['Sub-Dominant_Error'], axlabel=' ',bins=datbins, color='k', kde=False, norm_hist=True)
    #plt.ylim(0, Ytop)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.xticks(fontsize=Fsize)
    plt.yticks(fontsize=Fsize)


else:
    
    datbins=np.linspace(-1,1,40)
    #get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure()
    plt.subplot(1,2,1)
    sns.distplot(DominantErrorsDF.Dominant_Error, axlabel='', bins=datbins, color='k', kde=False)
    plt.title('(Median, RMS, QPA) = '+' ('+str(int(100*np.median(DominantErrors[:,0])))+ ', '+str(int(100*RMSdom))+', '+str(int(100*QPAdom))+ ')')
    plt.ylabel('20% Test Frequency')
    plt.subplot(1,2,2)
    sns.distplot(DominantErrorsDF['Sub-Dominant_Error'], axlabel='', bins=datbins, color='k', kde=False)
    plt.title('(Median, RMS, QPA) = '+' ('+str(int(100*np.median(DominantErrors[:,1])))+ ', '+str(int(100*RMSsubdom))+', '+str(int(100*QPAsubdom))+ ')')



#Validation data
PredictedPixels = Estimator.predict(ValidationFeatures)


DominantErrors=GetDominantClassErrors(np.asarray(ValidationLabels), PredictedPixels)
D={'Dominant_Error':DominantErrors[:,0],'Sub-Dominant_Error':DominantErrors[:,1] }
DominantErrorsDF = pd.DataFrame(D)

RMSdom = np.sqrt(np.mean(DominantErrors[:,0]*DominantErrors[:,0]))
RMSsubdom = np.sqrt(np.mean(DominantErrors[:,1]*DominantErrors[:,1]))
QPAdom = np.sum(np.abs(DominantErrors[:,0])<0.25)/len(DominantErrors[:,0])
QPAsubdom = np.sum(np.abs(DominantErrors[:,1])<0.25)/len(DominantErrors[:,1])


print('Validation mean error for DOMINANT class= ', str(np.mean(DominantErrors[:,0])))
print('Validation RMS error for DOMINANT class= ', str(RMSdom))
print('Validation QPA for the DOMINANT class= '+ str(QPAdom))
print('Validation mean error for SUB-DOMINANT class= ', str(np.mean(DominantErrors[:,1])))
print('Validation RMS error for SUB-DOMINANT class= ', str(RMSsubdom))
print('Validation QPA for the SUB-DOMINANT class= '+ str(QPAsubdom))
print('\n')


if PublishHist:
    mpl.rcParams['font.family'] = Fname
    plt.rcParams['font.size'] = Fsize
    plt.rcParams['axes.linewidth'] = Lweight
    plt.rcParams['font.weight'] = Fweight
    datbins=np.linspace(-1,1,40)
    plt.subplot(2,2,3)
    sns.distplot(DominantErrorsDF.Dominant_Error, bins=datbins, color='k',kde=False, norm_hist=True)
    #plt.ylim(0, Ytop)
    plt.ylabel('DNN Density', fontweight=Fweight)
    plt.xlabel('Dominant Class Error', fontweight=Fweight)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.xticks(fontsize=Fsize)
    plt.yticks(fontsize=Fsize)
    
    plt.subplot(2,2,4)
    sns.distplot(DominantErrorsDF['Sub-Dominant_Error'], bins=datbins, color='k', kde=False, norm_hist=True)
    plt.xlabel('Sub-Dominant Class Error', fontweight=Fweight)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.xticks(fontsize=Fsize)
    plt.yticks(fontsize=Fsize)
    #plt.ylim(0, Ytop)
    plt.savefig(SaveName, dpi=OutDPI, transparent=False, bbox_inches='tight')



else:
    #get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure()
    plt.subplot(1,2,1)
    sns.distplot(DominantErrorsDF.Dominant_Error, axlabel='Dominant Class Errors', bins=datbins, color='b')
    plt.title('(Median, RMS, QPA) = '+' ('+str(int(100*np.median(DominantErrors[:,0])))+ ', '+str(int(100*RMSdom))+', '+str(int(100*QPAdom))+ ')')
    plt.ylabel('Validation Frequency')
    plt.subplot(1,2,2)
    sns.distplot(DominantErrorsDF['Sub-Dominant_Error'], axlabel='Sub-Dominant Class Errors', bins=datbins, color='b')
    plt.title('(Median, RMS, QPA) = '+' ('+str(int(100*np.median(DominantErrors[:,1])))+ ', '+str(int(100*RMSsubdom))+', '+str(int(100*QPAsubdom))+ ')')


'''Show the classified validation images'''
if ShowValidation:
    for s in range(len(ValidationSites.index)):
        UAVRaster = io.imread(DataFolder+'Cropped_'+ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s])+'_UAVCLS.tif')
        UAVRaster[UAVRaster>3] =0
        UAVRaster[UAVRaster<1] =0
        
        UAVRasterRGB = np.zeros((UAVRaster.shape[0], UAVRaster.shape[1], 4))
        UAVRasterRGB[:,:,0]=255*(UAVRaster==3)
        UAVRasterRGB[:,:,1]=255*(UAVRaster==2)
        UAVRasterRGB[:,:,2]=255*(UAVRaster==1)
        UAVRasterRGB[:,:,3] = 255*np.float32(UAVRaster != 0.0)
        UAVRasterRGB= downscale_local_mean(UAVRasterRGB, (10,10,1))
        ValidRasterName = DataFolder+'Cropped_'+ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s])+'_S2.tif'
        ValidRaster = io.imread(ValidRasterName)
        ValidRasterIR = np.uint8(np.zeros((ValidRaster.shape[0], ValidRaster.shape[1],4)))
        stretch=2
        ValidRasterIR[:,:,0] = np.int16(stretch*255*ValidRaster[:,:,10])
        ValidRasterIR[:,:,1] = np.int16(stretch*255*ValidRaster[:,:,5])
        ValidRasterIR[:,:,2] = np.int16(stretch*255*ValidRaster[:,:,4])
        ValidRasterIR[:,:,3]= 255*np.int16((ValidRasterIR[:,:,0] != 0.0)&(ValidRasterIR[:,:,1] != 0.0))
#        ValidRasterIR[:,:,0] = adjust_sigmoid(ValidRasterIR[:,:,0], cutoff=0.5, gain=10, inv=False)
#        ValidRasterIR[:,:,1] = adjust_sigmoid(ValidRasterIR[:,:,1], cutoff=0.5, gain=10, inv=False)
#        ValidRasterIR[:,:,2] = adjust_sigmoid(ValidRasterIR[:,:,2], cutoff=0.5, gain=10, inv=False)
        #ValidRasterIR = ValidRasterIR/np.max(np.unique(ValidRasterIR))
        ValidationTensor = slide_raster_to_tiles(ValidRaster, size)
        Valid=np.zeros(12)
        for n in range(1,13):
            if ('B'+str(n)) in FeatureSet:
                Valid[n-1]=1
        
        ValidationTensor = np.compress(Valid, ValidationTensor, axis=3)
        #Select the central pixel in each tensor tile and make a table for non-convolutional NN classification
        ValidationFeatures = np.squeeze(ValidationTensor[:,size//2, size//2,:])
        PredictedPixels = Estimator.predict(ValidationFeatures)
        

        PredictedWaterImage = np.int16(255*PredictedPixels[:,0].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedVegImage = np.int16(255*PredictedPixels[:,1].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedSedImage = np.int16(255*PredictedPixels[:,2].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedClassImage=np.int16(np.zeros((PredictedWaterImage.shape[0], PredictedWaterImage.shape[1],4)))
        PredictedClassImage[:,:,0]=PredictedSedImage
        PredictedClassImage[:,:,1]=PredictedVegImage
        PredictedClassImage[:,:,2]=PredictedWaterImage
        mask = 255*np.int16((ValidRasterIR[:,:,0] != 0.0)&(ValidRasterIR[:,:,1] != 0.0))
        mask = np.int16(resize(mask, (PredictedClassImage.shape[0], PredictedClassImage.shape[1]), preserve_range=True))
        PredictedClassImage[:,:,3]=mask
        #PredictedClassImage[:,:,3]==255*np.float32((ValidRasterIR[1+size//2:-(size//2),1+size//2:-(size//2),0] != 0.0)&(ValidRasterIR[1+size//2:-(size//2),1+size//2:-(size//2),1] != 0.0))
#        DominantErrors=GetDominantClassErrors(np.asarray(ValidationLabels), PredictedPixels)
#        DominantErrorImage = np.int16(255*DominantErrors[:,0].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))

        cmapCHM = colors.ListedColormap(['black','red','lime','blue'])
        #get_ipython().run_line_magic('matplotlib', 'qt')
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(ValidRasterIR)
        plt.title(ValidationSites.Abbrev[s]+'_'+str(ValidationSites.Month[s])+'_'+str(ValidationSites.Year[s]) + ' Bands (11,4,3)')
        plt.subplot(1,3,2)
        plt.imshow(PredictedClassImage)
        plt.title(' Fuzzy Class')
        class1_box = mpatches.Patch(color='red', label='Sediment')
        class2_box = mpatches.Patch(color='lime', label='Veg.')
        class3_box = mpatches.Patch(color='blue', label='Water')
        ax=plt.gca()
        #ax.legend(handles=[class1_box,class2_box,class3_box], bbox_to_anchor=(1, -0.2),prop={'size': 24})
        
        #ax.legend(handles=[class1_box,class2_box,class3_box])
        plt.subplot(1,3,3)
        plt.imshow(np.int16(UAVRasterRGB))
        plt.title('UAV Ground-Truth Data')
        class1_box = mpatches.Patch(color='red', label='Sediment')
        class2_box = mpatches.Patch(color='lime', label='Veg.')
        class3_box = mpatches.Patch(color='blue', label='Water')
        ax=plt.gca()
        ax.legend(handles=[class1_box,class2_box,class3_box])
#        
#        plt.subplot(2,2,2)
#        plt.imshow(DominantErrorImage)
        
        
        
    


