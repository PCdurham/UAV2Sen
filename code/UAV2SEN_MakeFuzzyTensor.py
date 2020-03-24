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




#############################################################
"""Inputs"""
#############################################################

SiteList = 'E:\\UAV2SEN\\DebugList.csv'#this has the lists of sites with name, month and year
DatFolder = 'E:\\UAV2SEN\\Debug\\' #location of above
#
SiteDF = pd.read_csv(SiteList)

Set = 1 # 1 is all bands, 2 is 3,4,8 and 3 is 9,10,11 

#


#tile size and associated index of middle pixel
size=5
middle=2

#Output location
Outfile = 'E:\\UAV2SEN\\MLdata\\test3' #no extensions needed, added later

'''Functions'''
def map2pix(rasterfile,xmap,ymap):
    with rasterio.open(rasterfile) as map_layer:
        coords2pix = map_layer.index(xmap,ymap)
    return coords2pix

def pix2map(rasterfile,xpix,ypix):
    with rasterio.open(rasterfile) as map_layer:
        pix2coords = map_layer.xy(xpix,ypix)
    return pix2coords

def GetPercentMixClass(CLS, UL, LR):
    ClassOut=np.zeros((1,1,3))

    Spot = CLS[UL[0]:LR[0], UL[1]:LR[1]]#
    c,counts = np.unique(Spot, return_counts=True)
    
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
        ClassOut[0,0,0] = -1 #this flags a spot with no data
        
                    
    return ClassOut

def MakeFuzzyClass(S2Name, UAVClassName):
    S2 = io.imread(S2Name)
    w = S2.shape[0]
    h = S2.shape[1] 
    CLS_UAV = io.imread(UAVClassName)
    MixClass = np.zeros((w,h,3))
    for W in range(w):
        
        for H in range(h):
            S2coords = pix2map(S2Name, W,H)
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

'''Main processing'''

#initialise the main outputs
MasterLabelDict = {'VegMem':0,'WaterMem':0,'SedMem':0,'Month':0,'Year':0,'Site':'none'}
MasterLabelDF = pd.DataFrame(data=MasterLabelDict, index=[0])
if Set ==1:
    MasterTensor = np.zeros((1,size,size,12))
else:
    MasterTensor = np.zeros((1,size,size,3))
    
#run through the sites in the DF and extract the data
for s in range(len(SiteDF.Site)):
    # Getting the data
    S2Image = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_S2.tif'
    I1=io.imread(S2Image)
    #Slice the bands to get desired set, use indexing to change band set
    if Set==1:
        Isubset =I1
    elif Set==2:
        Isubset = I1[:,:,3:6]
        Isubset[:,:,2] =I1[:,:,8]
    else:
        Isubset = I1[:,:,9:12]
        
            
    ClassUAV = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_UAVCLS.tif'
    #get both UAV class and S2 class and produce the fuzzy classification
    Cfuzz1 = MakeFuzzyClass(S2Image, ClassUAV)
    
    
    
    Ti, Tl = slide_rasters_to_tiles(Isubset, Cfuzz1, size)
    labels = np.zeros((Tl.shape[0],6))
    LabelDF = pd.DataFrame(data=labels, columns=['VegMem','WaterMem','SedMem','Month','Year','Site'])
    #add the labels and membership to a DF for export
    for t in range(0, Tl.shape[0]):
        LabelDF.VegMem[t]=Tl[t,middle,middle,0].reshape(1,-1)
        LabelDF.WaterMem[t]=Tl[t,middle,middle,1].reshape(1,-1)
        LabelDF.SedMem[t]=Tl[t,middle,middle,2].reshape(1,-1)
        LabelDF.Month[t]=SiteDF.Month[s]
        LabelDF.Year[t]=SiteDF.Year[s]
        LabelDF.Site[t]=SiteDF.Abbrev[s]
        
    dataspots = LabelDF.VegMem != -1 #finds whre valid data was extracted
    numel = np.sum(dataspots)
    AugLabel = np.zeros((4*numel,6))
    AugLabelDF = pd.DataFrame(data=AugLabel, columns=['VegMem','WaterMem','SedMem','Month','Year','Site'])
    AugTensor = np.zeros((4*numel, size,size,Ti.shape[3]))
    
    
    #assemble valid data and a bit of data augmentation with 90 degree rotation and flips of the tensor
    E=0
    for n in range(0, len(dataspots)):
        if dataspots[n]:
            AugTensor[E,:,:,:]=Ti[n,:,:,:]
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            AugTensor[E,:,:,:]=np.rot90(Ti[n,:,:,:])
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            AugTensor[E,:,:,:]=np.fliplr(Ti[n,:,:,:])
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            AugTensor[E,:,:,:]=np.flipud(Ti[n,:,:,:])
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
    MasterLabelDF = pd.concat([MasterLabelDF, AugLabelDF])
    MasterTensor = np.concatenate((MasterTensor, AugTensor), axis=0)

        
        
#Clean up the final DFs for export    
MasterLabelDF = MasterLabelDF[MasterLabelDF.Site != 'none']
MasterLabelDF.index = range(0,len(MasterLabelDF.VegMem))
MasterTensor = MasterTensor[1:,:,:,:]


#export to feather for the DF and numpy for the tensor
OutTrain = Outfile +'__fuzzy_T.npy'
OutLabel =  Outfile+'__fuzzy_L.dat'   
np.save(OutTrain, MasterTensor)
MasterLabelDF.to_feather(OutLabel)


      

