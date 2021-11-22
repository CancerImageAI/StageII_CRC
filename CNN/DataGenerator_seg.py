# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:02:14 2019

@author: PC
"""

import pydicom as dicom
import numpy as np
import os
import pandas as pd
from pandas import DataFrame as DF
from skimage import draw, measure
import scipy
import SimpleITK as sitk
from tqdm import tqdm

# Load the scans in given folder path
def readDCM_Img(FilePath):
    img = {}
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image) # z,y,x
    Spacing = image.GetSpacing()
#    Origin = image.GetOrigin()
    img_array = img_array.transpose(2,1,0)#x,y,z
    img['array'] = img_array
    img['Spacing'] = np.array(Spacing).astype(float)
#    img['Origin'] = Origin
    return img

def resample(img, new_spacing=[1,1,5]):
    # Determine current pixel spacing
    image = img['array']
    spacing = img['Spacing']
    img_size = np.array(image.shape)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, img_size, real_resize_factor

def normalize_hu(image):
	#将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -125.0
    MAX_BOUND = 225.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    image = (image*255).astype('uint8')
    return image

def crop_roi(resampled_img, img_size, seed_pos, crop_size, resize_factor):
    initial_seed = [seed_pos[0], seed_pos[1], seed_pos[2]]
    trans_seed = initial_seed*resize_factor
    start = []
    end= []
    for i in range(3):
        s = np.floor(trans_seed[i]-(crop_size[i]/2))
        e = np.ceil(trans_seed[i]+(crop_size[i]/2))
        if s<0:
            s = 0
        if e>resampled_img.shape[i]:
            e = resampled_img.shape[i]
        if e-s != crop_size[i]:
            pad = e-s-crop_size[i]
            if s==0:
                e = e-pad
            else:
                s = s+pad
        start.append(int(s))
        end.append(int(e))       
#    print(start,end,pad)
    roi = resampled_img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]      
        
    return roi

def save_img(image, outputImageFileName):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)

if __name__ == '__main__':
    
    data_path = '../ColonDicom'
    mask_path = '../ColonMask'
    save_path = '../Tumor_crop'
    data_list = os.listdir(data_path)
    Seg_List = []
    Tumor_num = 0
    for patient_path in tqdm(data_list):
        img_path = os.path.join(data_path,patient_path)
        img = readDCM_Img(img_path)
        image, img_size, resize_factor = resample(img)
        image = normalize_hu(image)        

        Tumor_Mask = os.path.join(mask_path,patient_path+'/'+patient_path+'.nii')
        if os.path.exists(Tumor_Mask)==False:
            Tumor_Mask = os.path.join(mask_path,patient_path+'/'+patient_path+'.nii.gz')
            if os.path.exists(Tumor_Mask)==False:
                print(patient_path)
                mask_array = np.zeros(img['array'].shape)
#                Tumor_Mask = os.path.join(mask_path,patient_path+'/'+patient_path+'.nill.nii')
            else:
                mask = sitk.ReadImage(Tumor_Mask)
                mask_array = sitk.GetArrayFromImage(mask).transpose(2,1,0)#x,y,z
        else:
            mask = sitk.ReadImage(Tumor_Mask)
            mask_array = sitk.GetArrayFromImage(mask).transpose(2,1,0)#x,y,z
        try:
            LN_Mask = os.path.join(mask_path,patient_path+'/'+patient_path+' L.nii')
            if os.path.exists(LN_Mask)==False:
                LN_Mask = os.path.join(mask_path,patient_path+'/'+patient_path+' L.nii.gz')
                if os.path.exists(LN_Mask)==False:
                    print(patient_path)
                    LN_mask_array = np.zeros(img['array'].shape)
    #                LN_Mask = os.path.join(mask_path,patient_path+'/'+patient_path+' L.nill.nii')
                else:
                    LN_mask = sitk.ReadImage(LN_Mask)
                    LN_mask_array = sitk.GetArrayFromImage(LN_mask).transpose(2,1,0)#x,y,z                       
            else:
                LN_mask = sitk.ReadImage(LN_Mask)
                LN_mask_array = sitk.GetArrayFromImage(LN_mask).transpose(2,1,0)#x,y,z   
            
            mask_array = (mask_array+LN_mask_array*2).astype('uint8')
            if np.max(mask_array)>2:
                mask_array[mask_array>2]=2
            props = measure.regionprops(mask_array)
            for i in range(len(props)):
                x,y,z = props[i].centroid
                X = np.round(x+0.5)
                Y = np.round(y+0.5)
                Z = np.round(z+0.5)
            
                seed_pos = [X, Y, Z]
                ROI = crop_roi(image, img_size, seed_pos, [224,224,24] , resize_factor)
                new_spacing = [1,1,5]
                ROI_sitk = sitk.GetImageFromArray(ROI)
                ROI_sitk.SetSpacing(new_spacing)
                ROI = ROI[np.newaxis,...]
                ROI = (ROI.astype(np.float32)-128)/128.0 
                
                mask_new = {}
                mask_new['array'] = mask_array
                mask_new['Spacing'] = img['Spacing']
                label, label_size, resize_factor = resample(mask_new)
                
                ROI_label = crop_roi(label, label_size, seed_pos, [224,224,24], resize_factor)
                
                if not os.path.exists(save_path): 
                    os.mkdir(save_path)
                Tumor_num = Tumor_num+1
                np.save(os.path.join(save_path,str(Tumor_num)+'_roi.npy'), ROI)
                np.save(os.path.join(save_path,str(Tumor_num)+'_label.npy'), ROI_label)
                
                img_info = {}
                img_info['ID'] = patient_path
                img_info['Save_Num'] = Tumor_num
                Seg_List.append(img_info)
        except:
            print(patient_path)     
    df = DF(Seg_List).fillna('0')
    df.to_csv('./Seg_List.csv',index=False,sep=',')

                    
        


        
    
