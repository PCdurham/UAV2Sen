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

SiteList = 'E:\\UAV2SEN\\SiteListLong.csv'#this has the lists of sites with name, month and year
DatFolder = 'E:\\UAV2SEN\\FinalTif\\' #location of above

#tile size 
size=3


#Output location
Outfile = 'E:\\UAV2SEN\\MLdata\\LargeDebug' #no extensions needed, added later

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
    #print(str(len(c)))
    if (len(c)>0): 
        if (np.min(c)>0):#no UAV class pixels as no data. 10x10m area of S2 pixel is 100% classified
        

            if 1 in c:#in the drone classes, water=1, veg=2 and sed=3
                C1 = np.where(c==1)
                ClassOut[0,0,1] = counts[C1]/np.sum(counts)
        
            if 2 in c:
                C2 = np.where(c==2)
                ClassOut[0,0,0] = counts[C2]/np.sum(counts)
        
            if 3 in c:
                C3 = np.where(c==3)
                ClassOut[0,0,2] = counts[C3]/np.sum(counts)
                
        else:
            ClassOut[0,0,0] = -1
    else:
        ClassOut[0,0,0] = -1
    
        
                    
    return ClassOut

def MakeFuzzyClass(w,h,S2Name, UAVClassName, UAVClass):

    MixClass = np.zeros((w,h,3))
    for W in range(w):
        
        for H in range(h):
            S2coords = pix2map(S2Name, W,H)
            UL = map2pix(UAVClassName, S2coords[0]-5, S2coords[1]+5)
            LR = map2pix(UAVClassName, S2coords[0]+5, S2coords[1]-5)
            MixClass[W,H,:] = GetPercentMixClass( UAVClass, UL, LR)
                   
    return MixClass


def slide_rasters_to_tiles(im, CLS, size):
    
    di=im.shape[2]
    dc =CLS.shape[2]
    h=im.shape[0]
    w=im.shape[1]
    mid=size//2


    TileTensor = np.zeros(((h-size+mid)*(w-size+mid), size,size,di))
    LabelTensor = np.zeros(((h-size+mid)*(w-size+mid), size,size,dc))
    B=0
    for y in range(0, h-size+mid):
        for x in range(0, w-size+mid):
            LabelTensor[B] = CLS[y:y+size,x:x+size,:].reshape(size,size,dc)

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:].reshape(size,size,di)
            B+=1

    return TileTensor, LabelTensor

############################################################################################
'''Main processing'''
#load the site list
SiteDF = pd.read_csv(SiteList)

#Tile size
if size%2 != 0:
    middle=size//2
else:
    raise Exception('Tile size of '+str(size)+ ' is even and not valid. Please choose an odd tile size')

#initialise the main outputs
MasterLabelDict = {'VegMem':0,'WaterMem':0,'SedMem':0,'Month':0,'Year':0,'Site':'none'}
MasterLabelDF = pd.DataFrame(data=MasterLabelDict, index=[0])

MasterTensor = np.zeros((1,size,size,12))

    
#run through the sites in the DF and extract the data
for s in range(len(SiteDF.Site)):
    print('Processing '+SiteDF.Site[s]+' '+str(SiteDF.Month[s])+' '+str(SiteDF.Year[s]))
    # Getting the data
    S2Image = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_S2.tif'
    Isubset=io.imread(S2Image)
        
    
    #get both UAV class and S2 class and produce the fuzzy classification on the S2 image dimensions
    w = Isubset.shape[0]
    h = Isubset.shape[1]         
    ClassUAVName = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_UAVCLS.tif'
    ClassUAV = io.imread(ClassUAVName)
    ClassUAV[ClassUAV>3] = 0 #filter other classes and cases where 255 is the no data value
    ClassUAV[ClassUAV<1] = 0 #catch no data <1 but not 0 
    Cfuzz1 = MakeFuzzyClass(w,h,S2Image, ClassUAVName, ClassUAV)
    
    
    
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
    
    
    #assemble valid data and add a bit of data augmentation with three 90 degree rotations
    E=0
    for n in range(0, len(dataspots)):
        if dataspots[n]:
            AugTensor[E,:,:,:]=Ti[n,:,:,:]
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            Irotated = np.rot90(Ti[n,:,:,:])
            AugTensor[E,:,:,:]=Irotated 
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            Irotated = np.rot90(Irotated)
            AugTensor[E,:,:,:]=Irotated
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            Irotated = np.rot90(Irotated)
            AugTensor[E,:,:,:]=Irotated
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
    MasterLabelDF = pd.concat([MasterLabelDF, AugLabelDF])
    MasterTensor = np.concatenate((MasterTensor, AugTensor), axis=0)

        
        
#Clean up the final DFs for export    
MasterLabelDF = MasterLabelDF[MasterLabelDF.Site != 'none']
MasterLabelDF.index = range(0,len(MasterLabelDF.VegMem))
MasterTensor = MasterTensor[1:,:,:,:]


#export to feather for the DF and numpy for the tensor
OutTrain = Outfile +'_fuzzy_T.npy'
OutLabel =  Outfile+'_fuzzy_L.dat'   
np.save(OutTrain, MasterTensor)
MasterLabelDF.to_feather(OutLabel)


      

