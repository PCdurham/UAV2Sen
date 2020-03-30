# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:34:01 2020

@author: dgg0pc

Checks that each site in the site list has the S2, UAVCLS and dbPoly tif files needed
"""


import pandas as pd
import skimage.io as io


SiteList = 'E:\\UAV2SEN\\SiteList.csv'
DatFolder = 'E:\\UAV2SEN\\FinalTif\\'
SiteDF = pd.read_csv(SiteList)
for s in range(len(SiteDF.Site)):
    print('Checking '+SiteDF.Site[s]+' '+str(SiteDF.Month[s])+' '+str(SiteDF.Year[s]))
    S2Image = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_S2.tif'
    Isubset=io.imread(S2Image)
    ClassUAVName = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_UAVCLS.tif'
    ClassUAV = io.imread(ClassUAVName)
    ClassPolyFile = DatFolder+SiteDF.Abbrev[s]+'_'+str(SiteDF.Month[s])+'_'+str(SiteDF.Year[s])+'_dbPoly.tif'
    ClassPoly = io.imread(ClassPolyFile)