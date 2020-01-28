#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'
'''
This script is intended to run in the QGIS python console.  Check dependencies. 
It will train a CNN on the fly and then perform a fuzzy classification for the image saved
on disk as per line 33.  The output will be 3 separate rasters for each membership.
They can be merged if a unique map is needed.

'''

###############################################################################
""" Libraries"""
from keras import regularizers
from keras import optimizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D
from keras.wrappers.scikit_learn import KerasRegressor
import gdal, osr
from skimage import io
import time





'''Image to classify'''
ImName = 'F:\\MixClass\\ArboReach.tif'

'''Names for outputs of the membership rasters'''
WaterFuzzClass='F:\\MixClass\\ArboReach_watermem.tif'
VegFuzzClass='F:\\MixClass\\ArboReach_vegmem.tif'
SedFuzzClass='F:\\MixClass\\ArboReach_sedmem.tif'

'''BASIC PARAMETER CHOICES'''

TrainingEpochs = 150 #For NN only
Nfilters=36
Nfilters2=128#int(Nfilters/2)
NClasses = 3  #The number of end members in the fuzzy class


'''MODEL PARAMETERS''' #These would usually not be edited
 
Ndims = 11 # Feature Dimensions. 4 if using entropy in phase 2, 3 if just RGB
LearningRate = 0.0001
Chatty = 1 # set the verbosity of the model training.  Use 1 at first, 0 when confident that model is well tuned
#NAF = tf.keras.layers.LeakyReLU(alpha=0.25) #lambda expression for leaky relu
NAF = 'relu' #activation function


#############################################################
"""CNN  model data load"""
#############################################################
Train2 = np.load('F:\\MixClass\\SesiaCarr_T.npy')
Label2 = np.load('F:\\MixClass\\SesiaCarr_L.npy')

Train3 = np.load('F:\\MixClass\\BuonAmico_T.npy')
Label3 = np.load('F:\\MixClass\\BuonAmico_L.npy')

Train4 = np.load('F:\\MixClass\\BuonAmicoValle_T.npy')
Label4 = np.load('F:\\MixClass\\BuonAmicoVale_L.npy')

Train5 = np.load('F:\\MixClass\\Po_T.npy')
Label5 = np.load('F:\\MixClass\\Po_L.npy')


TrainData = np.concatenate([Train2,Train5, Train3, Train4], axis=0)
TrainLabel = np.concatenate([Label2,Label5, Label3, Label4], axis=0)

##############################################################
def slide_rasters_to_tiles(im, size):
    
    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape


    TileTensor = np.zeros(((h-size)*(w-size), size,size,d))

    B=0
    for y in range(0, h-size):
        for x in range(0, w-size):
            

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:].reshape(size,size,d)
            B+=1

    return TileTensor

##############################################################################
"""Instantiate and train fuzzy regressor""" 
   
	# create model
def CNN_model():
	# create model
    model = Sequential()
    model.add(Conv2D(Nfilters,5, data_format='channels_last', input_shape=(5,5,11), activation=NAF))
    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation=NAF))
    model.add(Dense(NClasses, kernel_initializer='normal', activation='linear'))
    
    #Tune an optimiser
    Optim = optimizers.Adam(lr=LearningRate, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Optim, metrics = ['accuracy'])
    #model.summary()
    return model



    
Estimator = KerasRegressor(build_fn=CNN_model, epochs=TrainingEpochs, batch_size=1000, verbose=Chatty)

X = TrainData
Y = TrainLabel

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.001, random_state=101)


print('Fitting CNN fuzzy regressor on ' + str(len(X_train)) + ' tiles')
Estimator.fit(X_train, y_train, batch_size=1000, epochs=TrainingEpochs, verbose=Chatty)

'''Process the new image'''
print('Loading raster')

Imultispec = io.imread(ImName)

#tile the image
Tiles = slide_rasters_to_tiles(Imultispec, 5)

#Predict the fuzzy memberships for the tiles
Predicted = Estimator.predict(Tiles)

#Loop through and reconstruct rasters
water = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
veg = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
sed = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
S=0
for n in range(2, Imultispec.shape[0]-3):
    for m in range(2, Imultispec.shape[1]-3):
        water[n,m]=Predicted[S,0]
        veg[n,m]=Predicted[S,1]
        sed[n,m]=Predicted[S,2]
        S+=1
        
water = np.int16(water*255)
veg = np.int16(veg*255)
sed = np.int16(sed*255)

#
#'''Georeferenced Exports'''

ImageFile = gdal.Open(ImName)
driver = gdal.GetDriverByName("GTiff")
outRaster = driver.Create(WaterFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
outRasterSRS = osr.SpatialReference()
project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
project_crs = int(project_crs_name[5:])
outRasterSRS.ImportFromEPSG(project_crs)
outRaster.SetProjection(outRasterSRS.ExportToWkt())
outRaster.GetRasterBand(1).WriteArray(water)
outRaster.FlushCache()  # saves image to disk
outRaster = None        # close output file
ImageFile = None        # close input file
##Open the new Class Raster data in QGIS
print('Displaying Classification')
time.sleep(1)
fileInfo = QFileInfo(WaterFuzzClass)
baseName = fileInfo.baseName()
rlayer = QgsRasterLayer(WaterFuzzClass, baseName)
if not rlayer.isValid():
    print ('Layer failed to load!')
else:
        QgsProject.instance().addMapLayer(rlayer)
# 
# 

ImageFile = gdal.Open(ImName)
driver = gdal.GetDriverByName("GTiff")
outRaster = driver.Create(VegFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
outRasterSRS = osr.SpatialReference()
project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
project_crs = int(project_crs_name[5:])
outRasterSRS.ImportFromEPSG(project_crs)
outRaster.SetProjection(outRasterSRS.ExportToWkt())
outRaster.GetRasterBand(1).WriteArray(veg)
outRaster.FlushCache()  # saves image to disk
outRaster = None        # close output file
ImageFile = None        # close input file
##Open the new Class Raster data in QGIS
print('Displaying Classification')
time.sleep(1)
fileInfo = QFileInfo(VegFuzzClass)
baseName = fileInfo.baseName()
rlayer = QgsRasterLayer(VegFuzzClass, baseName)
if not rlayer.isValid():
    print ('Layer failed to load!')
else:
        QgsProject.instance().addMapLayer(rlayer)
        

ImageFile = gdal.Open(ImName)
driver = gdal.GetDriverByName("GTiff")
outRaster = driver.Create(SedFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
outRasterSRS = osr.SpatialReference()
project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
project_crs = int(project_crs_name[5:])
outRasterSRS.ImportFromEPSG(project_crs)
outRaster.SetProjection(outRasterSRS.ExportToWkt())
outRaster.GetRasterBand(1).WriteArray(sed)
outRaster.FlushCache()  # saves image to disk
outRaster = None        # close output file
ImageFile = None        # close input file
##Open the new Class Raster data in QGIS
print('Displaying Classification')
time.sleep(1)
fileInfo = QFileInfo(SedFuzzClass)
baseName = fileInfo.baseName()
rlayer = QgsRasterLayer(SedFuzzClass, baseName)
if not rlayer.isValid():
    print ('Layer failed to load!')
else:
        QgsProject.instance().addMapLayer(rlayer)

 