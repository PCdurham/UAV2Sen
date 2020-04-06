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
import seaborn as sns
from skimage.transform import downscale_local_mean, resize
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import os



#############################################################
"""Inputs"""
#############################################################

SiteName = 'E:\\UAV2SEN\\LinearUnmix\SiteList_all.csv'#this has the lists of sites with name, month and year
DatFolder = 'E:\\UAV2SEN\\FinalTif\\'
LuMFolder = 'E:\\UAV2SEN\\LinearUnmix\\' #location of above

'''Validation Settings'''


PublishHist = True#best quality historgams
Ytop=8
SaveName='E:\\UAV2SEN\\Results\\Experiments\\AllSites_LuM_Hist.png'
OutDPI=900
Fname='Arial'
Fsize=10
Fweight='bold'
Lweight=1.5

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


def GetDominantClassErrors(Obs, Pred):
    Dominant=np.zeros((Obs.shape[0],2))
  
    for s in range(Obs.shape[0]):
        order=np.argsort(Obs[s,:])#dominant class in ground truth
        Dominant[s,0]=Pred[s,order[-1]]-Obs[s,order[-1]]
        Dominant[s,1]=Pred[s,order[-2]]-Obs[s,order[-2]]
    return Dominant

############################################################################################
'''Main processing'''
#load the site list
SiteDF = pd.read_csv(SiteName)


#initialise the main outputs
MasterDominantErrorsDict ={'Dominant_Error':-10,'Sub-Dominant_Error':-10}
MasterDominantErrors=pd.DataFrame(MasterDominantErrorsDict, index=[0])


    
#run through the sites in the DF and extract the data
for s in range(len(SiteDF.Site)):
    print('Processing '+SiteDF.Site[s]+' '+str(SiteDF.Month[s])+' '+str(SiteDF.Year[s]))
    # Getting the data
    S2Image = DatFolder+'Cropped_'+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_S2.tif'
    Isubset=io.imread(S2Image)
        
    
    #get both UAV class and S2 class and produce the fuzzy classification on the S2 image dimensions
    w = Isubset.shape[0]
    h = Isubset.shape[1]         
    ClassUAVName = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_UAVCLS.tif'
    ClassUAV = io.imread(ClassUAVName)
    ClassUAV[ClassUAV>3] = 0 #filter other classes and cases where 255 is the no data value
    ClassUAV[ClassUAV<1] = 0 #catch no data <1 but not 0 
    S2Mix = np.zeros((w,h,3))
    for i in range(w):
        for j in range(h):
            S2coords = pix2map(S2Image, i,j)
            UL = map2pix(ClassUAVName, S2coords[0]-5, S2coords[1]+5)
            LR = map2pix(ClassUAVName, S2coords[0]+5, S2coords[1]-5)
            S2Mix[i,j,:]= GetPercentMixClass(ClassUAV, UL, LR)
            
    #get unmixed data
    UnMixImageName = LuMFolder+'Auto_LuM_'+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_S2.tif'
    UnMixImage = io.imread(UnMixImageName)
    UnMixMemberships=np.zeros((w,h,3))
    UnMixMemberships[:,:,2]=UnMixImage[:,:,SiteDF.LuM_Sed[s]-1]
    UnMixMemberships[:,:,1]=UnMixImage[:,:,SiteDF.LuM_Veg[s]-1]
    UnMixMemberships[:,:,0]=UnMixImage[:,:,SiteDF.LuM_Water[s]-1]
    
    #tabulate both UAV predictions and LuM
    UAVtable=np.zeros((w*h,3))
    LuMtable=np.zeros((w*h,3))
    UAVtable[:,0]=S2Mix[:,:,0].ravel()#.reshape(-1,1)
    UAVtable[:,1]=S2Mix[:,:,1].ravel()#.reshape(-1,1)
    UAVtable[:,2]=S2Mix[:,:,2].ravel()#.reshape(-1,1)
    LuMtable[:,0]=UnMixMemberships[:,:,0].ravel()#.reshape(-1,1)
    LuMtable[:,1]=UnMixMemberships[:,:,1].ravel()#.reshape(-1,1)
    LuMtable[:,2]=UnMixMemberships[:,:,2].ravel()#.reshape(-1,1)
    UAVclsDF=pd.DataFrame(data=UAVtable, columns=['Water','Veg','Sed'])
    LuMDF=pd.DataFrame(data=LuMtable, columns=['Water','Veg','Sed'])
    LuMDF=LuMDF[UAVclsDF.Water!=-1]
    UAVclsDF=UAVclsDF[UAVclsDF.Water!=-1]
#    LuMDF=LuMDF.query('Water!=0 and Veg!=0 and Sed!=0')
#    LuMDF=LuMDF[(LuMDF.Water!=0 & LuMDF.Veg!=0 & LuMDF.Sed!=0)]
    DominantErrors=GetDominantClassErrors(np.asarray(UAVclsDF), np.asarray(LuMDF))
    D={'Dominant_Error':DominantErrors[:,0],'Sub-Dominant_Error':DominantErrors[:,1] }
    DominantErrorsDF = pd.DataFrame(D)
    MasterDominantErrors=pd.concat([MasterDominantErrors,DominantErrorsDF])

        
        
#Clean up the final DF    
MasterDominantErrors = MasterDominantErrors[MasterDominantErrors != -10]
MasterDominantErrors = MasterDominantErrors.dropna()
DomErrors = MasterDominantErrors.Dominant_Error
SubDom_Errors =MasterDominantErrors['Sub-Dominant_Error']
RMSdom = np.sqrt(np.mean(DomErrors*DomErrors))
RMSsubdom = np.sqrt(np.mean(SubDom_Errors*SubDom_Errors))
QPAdom = np.sum(np.abs(DomErrors)<0.25)/len(DomErrors)
QPAsubdom = np.sum(np.abs(SubDom_Errors)<0.25)/len(SubDom_Errors)


print('Validation mean error for DOMINANT class= ', str(np.mean(DomErrors)))
print('Validation RMS error for DOMINANT class= ', str(RMSdom))
print('Validation QPA for the DOMINANT class= '+ str(QPAdom))
print('Validation mean error for SUB-DOMINANT class= ', str(np.mean(SubDom_Errors)))
print('Validation RMS error for SUB-DOMINANT class= ', str(RMSsubdom))
print('Validation QPA for the SUB-DOMINANT class= '+ str(QPAsubdom))
print('\n')

if PublishHist:
    mpl.rcParams['font.family'] = Fname
    plt.rcParams['font.size'] = Fsize
    plt.rcParams['axes.linewidth'] = Lweight
    plt.rcParams['font.weight'] = Fweight
    datbins=np.linspace(-1,1,40)
    plt.subplot(1,2,1)
    sns.distplot(DominantErrorsDF.Dominant_Error, bins=datbins, color='k', kde=False, norm_hist=True)
    plt.ylim(0,Ytop)
    plt.ylabel('Frequency [%]', fontweight=Fweight)
    plt.xlabel('Dominant Class Error', fontweight=Fweight)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.subplot(1,2,2)
    sns.distplot(DominantErrorsDF['Sub-Dominant_Error'], bins=datbins, color='k', kde=False, norm_hist=True)
    plt.xlabel('Sub-Dominant Class Error', fontweight=Fweight)
    plt.xticks((-1.0, -0.5, 0.0,0.5, 1.0))
    plt.ylim(0,Ytop)
    plt.savefig(SaveName, dpi=OutDPI, transparent=False, bbox_inches='tight')




else:
    plt.figure()
    plt.subplot(1,2,1)
    sns.distplot(DominantErrorsDF.Dominant_Error, axlabel='Dominant Class Errors', bins=datbins, color='b', kde=False)
    plt.title('(Median, RMS, QPA) = '+' ('+str(int(100*np.median(DominantErrors[:,0])))+ ', '+str(int(100*RMSdom))+', '+str(int(100*QPAdom))+ ')')
    plt.ylabel('Validation Frequency')
    plt.subplot(1,2,2)
    sns.distplot(DominantErrorsDF['Sub-Dominant_Error'], axlabel='Sub-Dominant Class Errors', bins=datbins, color='b', kde=False)
    plt.title('(Median, RMS, QPA) = '+' ('+str(int(100*np.median(DominantErrors[:,1])))+ ', '+str(int(100*RMSsubdom))+', '+str(int(100*QPAsubdom))+ ')')


      

