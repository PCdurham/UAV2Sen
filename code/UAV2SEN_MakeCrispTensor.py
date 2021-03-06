#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'

''' This routine compiles sentinel data (preferably cropped) and 
UAV high resolution class rasters and creates tensors suitable for DNN work.
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

SiteList = 'EMPTY'#this has the lists of sites with name, month and year
DatFolder = 'EMPTY' #location of above

#tile size 
size=7


#Output location
Outfile = 'EMPTY' #no extensions needed, added later


###############################################################################
'''Functions'''
def map2pix(rasterfile,xmap,ymap):
    with rasterio.open(rasterfile) as map_layer:
        coords2pix = map_layer.index(xmap,ymap)
    return coords2pix

def pix2map(rasterfile,xpix,ypix):
    with rasterio.open(rasterfile) as map_layer:
        pix2coords = map_layer.xy(xpix,ypix)
    return pix2coords

def GetCrispClass(CLS, UL, LR):
    ClassOut=np.zeros((1,1,3))

    Spot = CLS[UL[0]:LR[0], UL[1]:LR[1]]#
    c,counts = np.unique(Spot, return_counts=True)
    
    if len(c)>0:
        if (np.min(c)>0):#no UAV class pixels as no data. 10x10m area of S2 pixel is 100% classified
    
            if np.max(counts)>=(0.95*np.sum(counts)):#pure class
                ClassOut[0,0,2]=c[np.argmax(counts)]
            if np.max(counts)>=(0.5*np.sum(counts)):#majority class
                ClassOut[0,0,1]=c[np.argmax(counts)]
            if np.max(counts)>(np.sum(counts)/3):#relative majority class, assumes a 3 class problem
                ClassOut[0,0,0]=c[np.argmax(counts)]
        else:
            ClassOut[0,0,0] = -1 #this flags a spot with no data

    else:
        ClassOut[0,0,0] = -1 #this flags a spot with no data
        
                    
    return ClassOut

def MakeCrispClass(S2Name, UAVClassName, CLS_UAV):
    S2 = io.imread(S2Name)
    w = S2.shape[0]
    h = S2.shape[1] 

    CrispClass = np.zeros((w,h,3))
    for W in range(w):
        
        for H in range(h):
            S2coords = pix2map(S2Name, W,H)
            UL = map2pix(UAVClassName, S2coords[0]-5, S2coords[1]+5)
            LR = map2pix(UAVClassName, S2coords[0]+5, S2coords[1]-5)
            CrispClass[W,H,:] = GetCrispClass(CLS_UAV, UL, LR)
                   
    return CrispClass


def slide_rasters_to_tiles(im, CLS, size):
    h=im.shape[0]
    w=im.shape[1]
    di=im.shape[2]

    try:
        dc =CLS.shape[2]
        LabelTensor = np.zeros(((h-size)*(w-size), size,size,dc))
    except:#case with desk-based polygons having 2D labels
         dc=1
         LabelTensor = np.zeros(((h-size)*(w-size), size,size,dc))

    TileTensor = np.zeros(((h-size)*(w-size), size,size,di))

    
    B=0
    for y in range(0, h-size):
        for x in range(0, w-size):
            #print(str(x)+' '+str(y))
            if dc>1:
                LabelTensor[B,:,:,:] = CLS[y:y+size,x:x+size,:]#.reshape(size,size,dc)
            else:
                LabelTensor[B,:,:,0] = CLS[y:y+size,x:x+size]#.reshape(size,size,1)

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:]#.reshape(size,size,di)
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

#initialise the main outputs with Relative majority, Majority and Pure class and Poygon class cases
MasterLabelDict = {'RelMajClass':0,'MajClass':0,'PureClass':0,'PolyClass':0,'Month':0,'Year':0,'Site':'none'}
MasterLabelDF = pd.DataFrame(data=MasterLabelDict, index=[0])
MasterTensor = np.zeros((1,size,size,12))

    
''''Pass 1: UAV classes'''


    
##run through the sites in the DF and extract the data
for s in range(len(SiteDF.Site)):
    print('Processing UAV classes '+SiteDF.Site[s]+' '+str(SiteDF.Month[s])+' '+str(SiteDF.Year[s]))
    # Getting the data
    S2Image = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_S2.tif'
    Isubset=io.imread(S2Image)
           
            
    #get both UAV class and S2 class and produce the fuzzy classification on the S2 image dimensions
      
    ClassUAVName = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_UAVCLS.tif'
    ClassUAV = io.imread(ClassUAVName)
    ClassUAV[ClassUAV<1] = 0 #catch no data <1 but not 0 cases
    ClassUAV[ClassUAV>3] = 0 #filter other classes and cases where 255 is the no data value
    Ccrisp1 = MakeCrispClass(S2Image, ClassUAVName, ClassUAV)
    
    
    
    Ti, Tl = slide_rasters_to_tiles(Isubset, Ccrisp1, size)
    labels = np.zeros((Tl.shape[0],7))
    LabelDF = pd.DataFrame(data=labels, columns=['RelMajClass','MajClass','PureClass','PolyClass','Month','Year','Site'])
    #add the labels and membership to a DF for export
    for t in range(0, Tl.shape[0]):
        LabelDF.RelMajClass[t]=Tl[t,middle,middle,0].reshape(1,-1)
        LabelDF.MajClass[t]=Tl[t,middle,middle,1].reshape(1,-1)
        LabelDF.PureClass[t]=Tl[t,middle,middle,2].reshape(1,-1)
        LabelDF.PolyClass[t]=-1 #this flags the fact that this part does not compute polygon classes
        LabelDF.Month[t]=SiteDF.Month[s]
        LabelDF.Year[t]=SiteDF.Year[s]
        LabelDF.Site[t]=SiteDF.Abbrev[s]
        
    dataspots = LabelDF.RelMajClass != -1 #finds whre valid data was extracted
    numel = np.sum(dataspots)
    AugLabel = np.zeros((4*numel,7))
    AugLabelDF = pd.DataFrame(data=AugLabel, columns=['RelMajClass','MajClass','PureClass','PolyClass','Month','Year','Site'])
    AugTensor = np.zeros((4*numel, size,size,Ti.shape[3]))
    
    
    #assemble valid data and a bit of data augmentation with 90 degree rotation and flips of the tensor
    E=0
    for n in range(0, len(dataspots)):
        if dataspots[n]:
            AugTensor[E,:,:,:]=Ti[n,:,:,:]
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            noise=0.001*np.random.random((1,size,size,12))
            Irotated = np.rot90(Ti[n,:,:,:])
            AugTensor[E,:,:,:]=Irotated #+ noise
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            noise=0.001*np.random.random((1,size,size,12))
            Irotated = np.rot90(Irotated)
            AugTensor[E,:,:,:]=Irotated + noise
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            noise=0.001*np.random.random((1,size,size,12))
            Irotated = np.rot90(Irotated)
            AugTensor[E,:,:,:]=Irotated + noise
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
    MasterLabelDF = pd.concat([MasterLabelDF, AugLabelDF])
    MasterTensor = np.concatenate((MasterTensor, AugTensor), axis=0)

''''Pass 2: Desk-Based Polygon classes'''


    
#run through the sites in the DF and extract the data
for s in range(len(SiteDF.Site)):
    print('Processing desk-based polygon classes '+SiteDF.Site[s]+' '+str(SiteDF.Month[s])+' '+str(SiteDF.Year[s]))
    # Getting the data
    S2Image = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_S2.tif'
    Isubset=io.imread(S2Image)

        
            
    ClassPolyFile = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_dbPoly.tif'
    #vectorise the Polygon classes
    ClassPoly = io.imread(ClassPolyFile)
    ClassPoly[ClassPoly>3] = 0 #filter other classes and cases where 255 is the no data value
    ClassPoly[ClassPoly<1] = 0
    
    
    
    Ti, Tl = slide_rasters_to_tiles(Isubset, ClassPoly, size)
#    if len(Tl.shape)==3:
#        Tl=Tl.reshape(Tl.shape[0], Tl.shape[1], Tl.shape[2], 1)
#        

    labels = np.zeros((Ti.shape[0],7))
    LabelDF = pd.DataFrame(data=labels, columns=['RelMajClass','MajClass','PureClass','PolyClass','Month','Year','Site'])
    #add the labels and membership to a DF for export
    for t in range(0, Ti.shape[0]):
        LabelDF.RelMajClass[t]=-1 #this flags the fact that this part does not compute classes from the UAV data
        LabelDF.MajClass[t]=-1
        LabelDF.PureClass[t]=-1
        LabelDF.PolyClass[t]=Tl[t,middle,middle,0].reshape(1,-1) 
        LabelDF.Month[t]=SiteDF.Month[s]
        LabelDF.Year[t]=SiteDF.Year[s]
        LabelDF.Site[t]=SiteDF.Abbrev[s]
        
    dataspots = LabelDF.PolyClass != 0 #finds where valid data was extracted
    numel = np.sum(dataspots)
    AugLabel = np.zeros((4*numel,7))
    AugLabelDF = pd.DataFrame(data=AugLabel, columns=['RelMajClass','MajClass','PureClass','PolyClass','Month','Year','Site'])
    AugTensor = np.zeros((4*numel, size,size,Ti.shape[3]))
    
    
    #assemble valid data and a bit of data augmentation with three 90 degree rotations
    E=0
    for n in range(0, len(dataspots)):
        if dataspots[n]:
            AugTensor[E,:,:,:]=Ti[n,:,:,:]
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            #noise=0.001*np.random.random((1,size,size,12))
            Irotated = np.rot90(Ti[n,:,:,:])
            AugTensor[E,:,:,:]=Irotated# + noise
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            noise=0.001*np.random.random((1,size,size,12))
            Irotated = np.rot90(Irotated)
            AugTensor[E,:,:,:]=Irotated + noise
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
            noise=0.001*np.random.random((1,size,size,12))
            Irotated = np.rot90(Irotated)
            AugTensor[E,:,:,:]=Irotated + noise
            AugLabelDF.iloc[E] = LabelDF.iloc[n]
            E+=1
    MasterLabelDF = pd.concat([MasterLabelDF, AugLabelDF])
    MasterTensor = np.concatenate((MasterTensor, AugTensor), axis=0)       
        
#Clean up the final DFs for export    
MasterLabelDF = MasterLabelDF[MasterLabelDF.Site != 'none']
MasterTensor = MasterTensor[1:,:,:,:]
MasterLabelDF.index = range(0,len(MasterLabelDF.RelMajClass))



#export to feather for the DF and numpy for the tensor
OutTrain = Outfile +'_crisp_'+str(size)+'_T.npy'
OutLabel =  Outfile+'_crisp_'+str(size)+'_L.csv'   
np.save(OutTrain, MasterTensor)
MasterLabelDF.to_csv(OutLabel)


      

