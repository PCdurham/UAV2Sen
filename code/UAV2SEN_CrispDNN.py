#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'


'''

This script performs crisp classification of river corridor features with a DNN.  the script allows 
the user to tune, train, validate and save DNN models.  The required inputs must be produced with the
UAV2SEN_MakeCrispTensor.py script.

'''
###############################################################################
""" Libraries"""

from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
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
UAVtrain = True #if true use the UAV class data to train the model, if false use desk-based data for training
MajType= 'Pure' #Majority type. only used if UAVtrain or valid is true. The options are RelMaj (relative majority class), Maj (majority) and Pure (95% unanimous).
FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']

'''CNN parameters'''
TrainingEpochs = 200#Use model tuning to adjust this and prevent overfitting
Nfilters = 4
size=3#size of the tensor tiles
KernelSize=3 # size of the convolution kernels
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
    


        
    
'''Use a Standard scaler for NN classification'''
SCAL = StandardScaler()
SCAL.fit(TrainingFeatures)
TrainF_scaled = SCAL.transform(TrainingFeatures)
ValidF_scaled = SCAL.transform(ValidationFeatures)
 





##############################################################################
"""Instantiate the Neural Network pixel-based classifier""" 
Ndims = TrainF_scaled.shape[1] # Feature Dimensions. 
NClasses = len(np.unique(TrainLabels))  #The number of classes in the data. This MUST be the same as the classes used to retrain the model


 	# create model
 
Estimator = Sequential()
Estimator.add(Dense(64, kernel_regularizer= regularizers.l2(0.001),input_dim=Ndims, kernel_initializer='normal', activation=NAF))
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
"""Data Splitting"""
TrainLabels1Hot = to_categorical(TrainLabels)
ValidationLabels1Hot = to_categorical(ValidationLabels)
X_train, X_test, y_train, y_test = train_test_split(TrainF_scaled, TrainLabels1Hot, test_size=0.2, random_state=42)

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

"""Data Fitting"""
print('Fitting CNN Classifier on ' + str(len(X_train)) + ' pixels')
Estimator.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)



'''Save model'''
ModelName=os.path.join(DataFolder,ModelName)
Estimator.save(ModelName,save_format='h5')
    
#Fit the predictor to test pixels
PredictedPixels = Estimator.predict(X_test)

# #Produce TTS classification reports 
#y_test=np.argmax(y_test, axis=1)
# PredictedPixels 
report = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(PredictedPixels, axis=1), digits = 3)
print('20% Test classification results for ')
print(report)
      

# #Fit the predictor to the external validation site
PredictedPixels = Estimator.predict(ValidF_scaled)
report = metrics.classification_report(ValidationLabels, np.argmax(PredictedPixels, axis=1), digits = 3)
print('Out-of-Sample validation results for ')
print(report)


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
        ValidationFeatures = np.squeeze(ValidationTensor[:,size//2, size//2,:])
        ValidF_scaled = SCAL.transform(ValidationFeatures)
        #print(ValidTensor.shape)
        PredictedPixels = Estimator.predict(ValidF_scaled)
        

        PredictedWaterImage = np.int16(255*PredictedPixels[:,1].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedVegImage = np.int16(255*PredictedPixels[:,2].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
        PredictedSedImage = np.int16(255*PredictedPixels[:,3].reshape(ValidRaster.shape[0]-size, ValidRaster.shape[1]-size))
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
        
        