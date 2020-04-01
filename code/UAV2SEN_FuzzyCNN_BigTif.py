#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

'''
This script is intended to process large geocoded images, eg entire Sentinel 2 tiles. It will usea pre-trained  CNN model 
and perform a fuzzy classification for the image saved on disk as per line 33.  The image needs to be
a super-resolved image output by SNAP.  The output will be 3 separate rasters for each membership 
which will be geocoded to the same CRS as the input image.  Each membership raster has
the fuzzy membership encoded in %. They can be open in QGIS (and they have a correct CRS) 
where the river corridor can be isolated.

The routine has a memory management feature.  If the size of the image is more than a user-set thershold in pixels, 
the fuzzy classification will be produced row-by-row to avoid the creation of
large 4D tensors that provoke out-of-memory errors. However, this is less efficient and slow.


'''

###############################################################################
""" Libraries"""
import numpy as np
import gdal, osr
from skimage import io
from tensorflow.keras.models import load_model



###############################################################################
'''User Inputs'''
#Pre-Trained CNN with full path
ModelName='E:\\UAV2SEN\\FinalTif\\QGIStest.h5'
size=5 #tile sized used to train the model
FeatureSet =  ['B2','B3','B4','B5','B6','B7','B8','B9','B11','B12'] # pick predictor bands from: ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12']

#Image to classify
ImageName = 'Paglia2018_SR' #do not add the tif extension
ImageLocation = 'E:\\UAV2SEN\\SuperRes\\'#add slashes at the end
ImageUTM = 32 #Give UTM zone. Assumes WGS84 datum
North = True #set to false if wanting UTM for southern hemisphere
SizeThresh = 500000 #This sets the threshold to trigger slower but OoM-error free row-by-row processing.  Increase or Decrease depending on your computer.






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

def Select_Features_BigTif(Image, FeatureSet):
    Valid=np.zeros(12)
    for n in range(1,13):
        if ('B'+str(n)) in FeatureSet:
            Valid[n-1]=1
        
    Image = np.compress(Valid, Image, axis=2)
    return Image

##############################################################################
'''Names for outputs of the membership rasters'''
WaterFuzzClass=ImageLocation+ImageName+'_watermem.tif'
VegFuzzClass=ImageLocation+ImageName+'_vegmem.tif'
SedFuzzClass=ImageLocation+ImageName+'_sedmem.tif'


"""CNN  model load"""
Estimator = load_model(ModelName)



'''Process the new image'''
print('Loading raster')
ImName = ImageLocation+ImageName+'.tif'
Imultispec = io.imread(ImName)



#Predict the fuzzy memberships for the tiles
if Imultispec.shape[0]*Imultispec.shape[1]<SizeThresh:
    print('Classifying whole image')
    #tile the image
    Tiles = slide_rasters_to_tiles(Imultispec, size)
    Tiles = Select_Features(Tiles, FeatureSet)
    Predicted = Estimator.predict(Tiles)
    # reconstruct rasters
    water = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    veg = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    sed = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    S=0
    for n in range(Imultispec.shape[0]-size):
        for m in range(Imultispec.shape[1]-size):
            water[n+size//2,m+size//2]=Predicted[S,0]
            veg[n+size//2,m+size//2]=Predicted[S,1]
            sed[n+size//2,m+size//2]=Predicted[S,2]
            S+=1
else:
    print('Large Image detected. Classifying row-by-row')
    water = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    veg = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    sed = np.zeros((Imultispec.shape[0], Imultispec.shape[1]))
    Imultispec = Select_Features_BigTif(Imultispec, FeatureSet)
    for r in range(0,Imultispec.shape[0]-size):
        print('row '+str(r)+' of '+str(Imultispec.shape[0]))
        Irow = Imultispec[r:r+size+1,:,:]
        #tile the image
        Tiles = slide_rasters_to_tiles(Irow, size)
        Predicted = Estimator.predict(Tiles)
        S=0
        for m in range(size//2, Imultispec.shape[1]-size):
            water[r+size//2,m+size//2]=Predicted[S,0]
            veg[r+size//2,m+size//2]=Predicted[S,1]
            sed[r+size//2,m+size//2]=Predicted[S,2]
            S+=1
        

        

       
        
        


        
water = np.int16(water*100)#output rasters to be scaled directly in %
veg = np.int16(veg*100)
sed = np.int16(sed*100)

water[water>100]=100 #saturate extremes
veg[veg<0]=0
veg[veg>100]=100
sed[sed<0]=0
sed[sed>100]=100

#
#'''Georeferenced Exports'''
if North:
    HemUTM=1
else:
    HemUTM=0

ImageFile = gdal.Open(ImName)
driver = gdal.GetDriverByName("GTiff")
outRaster = driver.Create(WaterFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
outRasterSRS = osr.SpatialReference()
#project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
#project_crs = int(project_crs_name[5:])
#outRasterSRS.ImportFromEPSG(ImageEPSG)
outRasterSRS.SetUTM(ImageUTM, HemUTM)
outRasterSRS.SetWellKnownGeogCS("WGS84")
outRaster.SetProjection(outRasterSRS.ExportToWkt())
outRaster.GetRasterBand(1).WriteArray(water)
outRaster.FlushCache()  # saves image to disk
outRaster = None        # close output file
ImageFile = None        # close input file


ImageFile = gdal.Open(ImName)
driver = gdal.GetDriverByName("GTiff")
outRaster = driver.Create(VegFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
outRasterSRS = osr.SpatialReference()
#project_crs_name = iface.mapCanvas().mapSettings().destinationCrs().authid()
#project_crs = int(project_crs_name[5:])
outRasterSRS.SetUTM(ImageUTM, HemUTM)
outRasterSRS.SetWellKnownGeogCS("WGS84")
outRaster.SetProjection(outRasterSRS.ExportToWkt())
outRaster.GetRasterBand(1).WriteArray(veg)
outRaster.FlushCache()  # saves image to disk
outRaster = None        # close output file
ImageFile = None        # close input file

        

ImageFile = gdal.Open(ImName)
driver = gdal.GetDriverByName("GTiff")
outRaster = driver.Create(SedFuzzClass, water.shape[1], water.shape[0], gdal.GDT_Byte)
outRaster.SetGeoTransform(ImageFile.GetGeoTransform())
outRasterSRS = osr.SpatialReference()
outRasterSRS.SetUTM(ImageUTM, HemUTM)
outRasterSRS.SetWellKnownGeogCS("WGS84")
outRaster.SetProjection(outRasterSRS.ExportToWkt())
outRaster.GetRasterBand(1).WriteArray(sed)
outRaster.FlushCache()  # saves image to disk
outRaster = None        # close output file
ImageFile = None        # close input file

 