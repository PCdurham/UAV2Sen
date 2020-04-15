#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

'''

This script uses a pre-trained fuzzy CNN model, makes fuzzy membership predictions and computes the errors.
the stript is controlled with the same CSV file and erros will only be calculated for validation sites.

See equivalent script for CNN models if needed.

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
from tensorflow.keras.models import load_model


#############################################################
"""User data input. Use the site template and list training and validation choices"""
#############################################################

'''Folder Settgings'''
MainData = 'EMPTY'  #main data output from UAV2SEN_MakeCrispTensor.py. no extensions, will be fleshed out below
SiteList = 'EMPTY'#this has the lists of sites with name, month, year and 1s and 0s to identify training and validation sites
DataFolder = 'EMPTY'  #location of processed tif files
ModelName = 'EMPTY'  #Name and location of the final model to be saved in DataFolder. Add .h5 extension

'''Model Features and Labels'''
FeatureSet = ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12']
LabelSet = ['WaterMem', 'VegMem','SedMem' ]

'''Model parameters'''

size=5#size of the tensor tiles
BatchSize=5000
Chatty = 1 # set the verbosity of the model training. 
NAF = 'relu' #NN activation function



'''Validation Settings'''

ShowValidation = False#if true fuzzy classified images of the validation sites will be displayed.  Warning: xpensive to compute.
PublishHist = True#best quality historgams
Ytop=6.5
SaveName='EMPTY'
OutDPI=1200
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
"""CNN  model load"""
Estimator = load_model(os.path.join(DataFolder,ModelName+'.h5'))
  
###############################################################################
###############################################################################



'''Validate the model'''

#Validation data
PredictedPixels = Estimator.predict(ValidationTensor)
if PredictedPixels.shape[1]==4:
    PredictedPixels=PredictedPixels[:,1:]

DominantErrors=GetDominantClassErrors(np.asarray(ValidationLabels), PredictedPixels)
D={'Dominant_Error':DominantErrors[:,0],'Sub-Dominant_Error':DominantErrors[:,1] }
DominantErrorsDF = pd.DataFrame(D)

AbsMeandom = np.mean(np.abs(DominantErrors[:,0]))
AbsMeansubdom = np.mean(np.abs(DominantErrors[:,1]))
QPAdom = np.sum(np.abs(DominantErrors[:,0])<0.25)/len(DominantErrors[:,0])
QPAsubdom = np.sum(np.abs(DominantErrors[:,1])<0.25)/len(DominantErrors[:,1])


DominantErrors[:,0]
#print(np.median(DominantErrors[:,0]), np.var(DominantErrors[:,0]))

print('Validation abs. mean error for DOMINANT class= ', str(AbsMeandom))
print('Validation median error for DOMINANT class= ', str(np.median(DominantErrors[:,0])))
print('Validation variance of error for DOMINANT class= ', str(np.var(DominantErrors[:,0])))

#print('Validation QPA for the DOMINANT class= '+ str(QPAdom))
print('Validation abs. mean error for SUB-DOMINANT class= ', str(AbsMeansubdom))
print('Validation median error for SUB-DOMINANT class= ', str(np.median(DominantErrors[:,1])))
print('Validation variance of error for SUB-DOMINANT class= ', str(np.var(DominantErrors[:,1])))
print('\n')


if PublishHist:
    mpl.rcParams['font.family'] = Fname
    plt.rcParams['font.size'] = Fsize
    plt.rcParams['axes.linewidth'] = Lweight
    plt.rcParams['font.weight'] = Fweight
    datbins=np.linspace(-1,1,40)
    plt.subplot(2,2,1)
    sns.distplot(DominantErrorsDF.Dominant_Error, bins=datbins, color='k',kde=False, norm_hist=True)
    #plt.ylim(0, Ytop)
    plt.ylabel('CNN Error Density', fontweight=Fweight)
    plt.xlabel('Dominant Class Error', fontweight=Fweight)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.xticks(fontsize=Fsize)
    plt.yticks(fontsize=Fsize)
    
    plt.subplot(2,2,2)
    sns.distplot(DominantErrorsDF['Sub-Dominant_Error'], bins=datbins, color='k', kde=False, norm_hist=True)
    plt.xlabel('Sub-Dominant Class Error', fontweight=Fweight)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.xticks(fontsize=Fsize)
    plt.yticks(fontsize=Fsize)
    #plt.ylim(0, Ytop)
    plt.savefig(SaveName, dpi=OutDPI, transparent=False, bbox_inches='tight')




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
        #print(ValidTensor.shape)
        PredictedPixels = Estimator.predict(ValidationTensor)
        

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
        
        
        
    


