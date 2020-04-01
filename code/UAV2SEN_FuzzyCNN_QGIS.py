#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'
'''
This script is intended to run in the QGIS python console.  Check dependencies. 
It will a pre-trained  CNN model and perform a fuzzy classification for the image saved
on disk as per line 33.  The output will be 3 separate rasters for each membership.
They can be merged if a unique map is needed using the merge tool.

The routine has a memory management feature.  If the size of the image is more than 250K pixels, 
the fuzzy classification will be produced row-by-row to avoid the creatoin of
large 4D tensors.

'''

###############################################################################
""" Libraries"""
import tensorflow as tf
import numpy as np
import gdal, osr
from skimage import io
import time


###############################################################################
'''User Inputs'''
#Pre-Trained CNN with full path
ModelName='E:\\UAV2SEN\\FinalTif\\QGIStest.h5'
size=5 #tile sized used to train the model
FeatureSet =  ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']
#Image to classify
ImName = 'E:\\UAV2SEN\\FinalTif\\Po_9_2017_S2.tif'

'''Names for outputs of the membership rasters'''
WaterFuzzClass='F:\\MixClass\\ArboReach_watermem.tif'
VegFuzzClass='F:\\MixClass\\ArboReach_vegmem.tif'
SedFuzzClass='F:\\MixClass\\ArboReach_sedmem.tif'




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


def Select_Features(Tensor, FeatureSet):
    Valid=np.zeros(12)
    for n in range(1,13):
        if ('B'+str(n)) in FeatureSet:
            Valid[n-1]=1
        
    Tensor = np.compress(Valid, Tensor, axis=3)
    return Tensor

##############################################################################

"""CNN  model load"""
Estimator = tf.keras.models.load_model(ModelName)



'''Process the new image'''
print('Loading raster')

Imultispec = io.imread(ImName)



#Predict the fuzzy memberships for the tiles
if Imultispec.shape[0]*Imultispec.shape[1]<2:
    print('Classifying whole image')
    #tile the image
    Tiles = slide_rasters_to_tiles(Imultispec, size)
    Tiles = Select_Features(Tiles, FeatureSet)
    Predicted = Estimator.predict(Tiles)
    #Loop through and reconstruct rasters
    water = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    veg = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    sed = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    S=0
    for n in range(2, Imultispec.shape[0]-size):
        for m in range(2, Imultispec.shape[1]-size):
            water[n,m]=Predicted[S,0]
            veg[n,m]=Predicted[S,1]
            sed[n,m]=Predicted[S,2]
            S+=1
else:
    print('Large Image detected. Classifying row-by-row')
    water = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    veg = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    sed = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))

    for r in range(0,Imultispec.shape[1]-size):
        print('row '+str(r)+' of '+str(Imultispec.shape[1]))
        Irow = Imultispec[r:r+size+1,:,:]
        #tile the image
        Tiles = slide_rasters_to_tiles(Irow, size)
        Tiles = Select_Features(Tiles, FeatureSet)
        Predicted = Estimator.predict(Tiles)
        S=0
        for m in range(size//2, Imultispec.shape[1]-size):
            water[r+size//2,m]=Predicted[S,0]
            veg[r+size//2,m]=Predicted[S,1]
            sed[r+size//2,m]=Predicted[S,2]
            S+=1
        

        
        
        
        

        
water = np.int16(water*255)
veg = np.int16(veg*255)
sed = np.int16(sed*255)

#
#'''Georeferenced Exports'''
#
#ImageFile = gdal.Open(ImName)
#driver = gdal.GetDriverByName("GTiff")
#outRaster = driver.Create(WaterFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
#outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
#outRasterSRS = osr.SpatialReference()
#project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
#project_crs = int(project_crs_name[5:])
#outRasterSRS.ImportFromEPSG(project_crs)
#outRaster.SetProjection(outRasterSRS.ExportToWkt())
#outRaster.GetRasterBand(1).WriteArray(water)
#outRaster.FlushCache()  # saves image to disk
#outRaster = None        # close output file
#ImageFile = None        # close input file
###Open the new Class Raster data in QGIS
#print('Displaying Classification')
#time.sleep(1)
#fileInfo = QFileInfo(WaterFuzzClass)
#baseName = fileInfo.baseName()
#rlayer = QgsRasterLayer(WaterFuzzClass, baseName)
#if not rlayer.isValid():
#    print ('Layer failed to load!')
#else:
#        QgsProject.instance().addMapLayer(rlayer)
## 
## 
#
#ImageFile = gdal.Open(ImName)
#driver = gdal.GetDriverByName("GTiff")
#outRaster = driver.Create(VegFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
#outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
#outRasterSRS = osr.SpatialReference()
#project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
#project_crs = int(project_crs_name[5:])
#outRasterSRS.ImportFromEPSG(project_crs)
#outRaster.SetProjection(outRasterSRS.ExportToWkt())
#outRaster.GetRasterBand(1).WriteArray(veg)
#outRaster.FlushCache()  # saves image to disk
#outRaster = None        # close output file
#ImageFile = None        # close input file
###Open the new Class Raster data in QGIS
#print('Displaying Classification')
#time.sleep(1)
#fileInfo = QFileInfo(VegFuzzClass)
#baseName = fileInfo.baseName()
#rlayer = QgsRasterLayer(VegFuzzClass, baseName)
#if not rlayer.isValid():
#    print ('Layer failed to load!')
#else:
#        QgsProject.instance().addMapLayer(rlayer)
#        
#
#ImageFile = gdal.Open(ImName)
#driver = gdal.GetDriverByName("GTiff")
#outRaster = driver.Create(SedFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
#outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
#outRasterSRS = osr.SpatialReference()
#project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
#project_crs = int(project_crs_name[5:])
#outRasterSRS.ImportFromEPSG(project_crs)
#outRaster.SetProjection(outRasterSRS.ExportToWkt())
#outRaster.GetRasterBand(1).WriteArray(sed)
#outRaster.FlushCache()  # saves image to disk
#outRaster = None        # close output file
#ImageFile = None        # close input file
###Open the new Class Raster data in QGIS
#print('Displaying Classification')
#time.sleep(1)
#fileInfo = QFileInfo(SedFuzzClass)
#baseName = fileInfo.baseName()
#rlayer = QgsRasterLayer(SedFuzzClass, baseName)
#if not rlayer.isValid():
#    print ('Layer failed to load!')
#else:
#        QgsProject.instance().addMapLayer(rlayer)

 