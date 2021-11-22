# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:59:27 2019

@author: PC
"""


import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor,imageoperations
import os
import pandas as pd
from pandas import DataFrame as DF
import warnings
import time
from time import sleep
from tqdm import tqdm
from skimage import measure


    
def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

def Extract_Features(image,mask,params_path):
    paramsFile = os.path.abspath(params_path)
    extractor = featureextractor.RadiomicsFeaturesExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    start = time.clock()
    warnings.simplefilter('ignore')

    img_path = '../../TestData/DICOM'
    mask_root = '../../TestData/ROI'
    list_path = 'list_all.csv'

    test_file = os.listdir(img_path)
    
    f = open(list_path)
    Tumor_list = pd.read_csv(f)
    List_Num = Tumor_list['Num'].tolist()
    OS = Tumor_list['OS'].tolist()
    DFS = Tumor_list['DFS'].tolist()
    Osevents = Tumor_list['Osevents'].tolist()
    DFSevents = Tumor_list['DFSevents'].tolist()    
    
    Feature = []
    Error_list = []
    for i in tqdm(range(len(test_file))):
        sleep(0.01)
        dcm_File = test_file[i]
        if os.path.exists(img_path+'/'+str(dcm_File)):
            roi_path = img_path+'/'+str(dcm_File)
            mask_path = mask_root+'/'+str(dcm_File)+'.nii'
            if ~os.path.exists(mask_path):
                mask_path = mask_root+'/'+str(dcm_File)+'.nii.gz'
        try:
            ROI = readDCM_Img(roi_path)
            
            Mask = sitk.ReadImage(mask_path)
            
            LN_Mask_array = sitk.GetArrayFromImage(Mask)
            label_LN_Mask,num = measure.label(LN_Mask_array,return_num=True)
            
            if num>1:
                props = measure.regionprops(label_LN_Mask)
                LN_Area = []
                LN_label = []
                for region in props:
                    LN_Area.append(region.area)
                    LN_label.append(region.label)
                LN_ind = LN_Area.index(max(LN_Area))
                LN_mask_array = (label_LN_Mask==LN_label[LN_ind]).astype('int')
                LN_mask = sitk.GetImageFromArray(LN_mask_array)
                LN_mask.CopyInformation(Mask)
            else:
                LN_mask = Mask
            
        
        
            features, feature_info = Extract_Features(ROI, LN_mask, 'params.yaml')
    
    
            features['Name'] = test_file[i]
            ind = List_Num.index(int(test_file[i]))
            features['OS'] = OS[ind]
            features['DFS'] = DFS[ind]
            features['Osevents'] = Osevents[ind]
            features['DFSevents'] = DFSevents[ind]
            features['Num'] = num
            Feature.append(features)
        except:
            print(List_Num[i])
            Error_list.append(test_file[i])
    df = DF(Feature).fillna('0')
    df.to_csv('../../Results/Tumor_Feature_TestingData.csv',index=False,sep=',')

    end = time.clock()
    print(end-start)  
