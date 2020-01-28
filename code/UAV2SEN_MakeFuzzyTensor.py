#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

''' This routine compiles sentinel data (preferably cropped) and 
UAV high resolution class rasters and creates tensors suitable for CNN work.
saves a tensor and label vector in npy format
'''

###############################################################################
""" Libraries"""

import pandas as pd
import numpy as np
from skimage import io
import rasterio
import os



#############################################################
"""Inputs"""
#############################################################

#
Image1='F:\MixClass\\S2_BAVAL_SR.tif'
Class1='F:\MixClass\\BA_Valle_S2_CLS.tif'
ClassUAV1='F:\MixClass\\BA_valle_Drone_CLS.tif'
#


#tile size and associated index of middle pixel
size=7
middle=3

#Output location
Outfile = 'F:\MixClass\\gitest' #no extensions needed, added later

'''Functions'''
def map2pix(rasterfile,xmap,ymap):
    with rasterio.open(rasterfile) as map_layer:
        coords2pix = map_layer.index(xmap,ymap)
    return coords2pix

def pix2map(rasterfile,xpix,ypix):
    with rasterio.open(rasterfile) as map_layer:
        pix2coords = map_layer.xy(xpix,ypix)
    return pix2coords

def GetPercentMixClass(CLS, UL, LR):# mixed classes, if more than 2, ignore that patch.
    Spot = CLS[UL[0]:LR[0], UL[1]:LR[1]]
    Spot[Spot>3]=0
    c,counts = np.unique(Spot, return_counts=True)
    #print(str(c))
    ClassOut=np.zeros((1,1,3))
    
    if (c.size>0):
        if np.min(c)>0:
        

            if 1 in c:
                C1 = np.where(c==1)
                ClassOut[0,0,0] = counts[C1]/np.sum(counts)
    
            if 2 in c:
                C2 = np.where(c==2)
                ClassOut[0,0,1] = counts[C2]/np.sum(counts)
    
            if 3 in c:
                C3 = np.where(c==3)
                ClassOut[0,0,2] = counts[C3]/np.sum(counts)
    else:
        ClassOut[0,0,0] = -1
        
                    
    return ClassOut

def MakeFuzzyClass(S2ClassName, UAVClassName):
    CLS_S2 = io.imread(S2ClassName)
    w = CLS_S2.shape[0]
    h = CLS_S2.shape[1] 
    CLS_UAV = io.imread(UAVClassName)
    MixClass = np.zeros((w,h,3))
    for W in range(w):
        for H in range(h):
            S2coords = pix2map(S2ClassName, W,H)
            UL = map2pix(UAVClassName, S2coords[0]-5, S2coords[1]+5)
            LR = map2pix(UAVClassName, S2coords[0]+5, S2coords[1]-5)
            MixClass[W,H,:] = GetPercentMixClass(CLS_UAV, UL, LR)
    return MixClass


def slide_rasters_to_tiles(im, CLS, size):
    
    di=im.shape[2]
    dc =CLS.shape[2]
    h=im.shape[0]
    w=im.shape[1]



    TileTensor = np.zeros(((h-size)*(w-size), size,size,di))
    LabelTensor = np.zeros(((h-size)*(w-size), size,size,dc))
    B=0
    for y in range(0, h-size):
        for x in range(0, w-size):
            LabelTensor[B] = CLS[y:y+size,x:x+size,:].reshape(size,size,dc)

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:].reshape(size,size,di)
            B+=1

    return TileTensor, LabelTensor





# Getting the data
I1=io.imread(Image1)
#Slice the bands to get desired set, use indexing to change band set
Isubset =I1#[:,:,8:11]  

#get both UAV class and S2 class and produce the fuzzy classification
Cfuzz1 = MakeFuzzyClass(Class1, ClassUAV1)



Ti, Tl = slide_rasters_to_tiles(Isubset, Cfuzz1, size)
labels = np.zeros((Tl.shape[0],3))
for t in range(0, Tl.shape[0]):
    labels[t,0]=Tl[t,middle,middle,0].reshape(1,-1)
    labels[t,1]=Tl[t,middle,middle,1].reshape(1,-1)
    labels[t,2]=Tl[t,middle,middle,2].reshape(1,-1)
    
dataspots = (labels[:,0]>0)|(labels[:,1]>0)|(labels[:,2]>0)
numel = np.sum(dataspots)
AugLabel = np.zeros((2*numel,3))
AugTensor = np.zeros((2*numel, size,size,Ti.shape[3]))


#clean zeros and a bit of data augmentation with 90 degree rotations of the tensor
E=0
for n in range(0, len(dataspots)):
    if dataspots[n]:
        AugTensor[E,:,:,:]=Ti[n,:,:,:]
        AugLabel[E,:] = labels[n,:]
        E+=1
        AugTensor[E,:,:,:]=np.rot90(Ti[n,:,:,:])
        AugLabel[E,:] = labels[n,:]
        E+=1

        
        
    


OutTrain = Outfile +'_T.npy'
OutLabel =  Outfile+'_L.npy'   
np.save(OutTrain, AugTensor)
np.save(OutLabel, AugLabel)


      

