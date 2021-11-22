# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:16:10 2019

@author: PC
"""

import numpy as np
import torch
from torch import nn
import os
from torch.autograd import Variable
import SimpleITK as sitk
import numpy as np
import scipy
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
from Net_Segment import *



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
    MIN_BOUND = -140.0
    MAX_BOUND = 260.0
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

if __name__ == "__main__":
    Pretrained_path = './model'
    model = Unet3D().cuda()
    classify_path = os.path.join(Pretrained_path, '100.ckpt')
    modelCheckpoint = torch.load(classify_path)
    pretrained_dict = modelCheckpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#filter out unnecessary keys
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    mask_path = '../ROI/9375142/9375142.nii'
    img_path = '../DICOM/9375142'
    LN_mask_path = '../ROI/9375142/9375142 L.nii.gz'
    img = readDCM_Img(img_path)
    image, img_size, resize_factor = resample(img)
    image = normalize_hu(image) 
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask).transpose(2,1,0)#x,y,z
    
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
#    data = np.load('../Tumor_crop/28_roi.npy')
    data = ROI
    data = data[np.newaxis,...]
    data1 = torch.from_numpy(data.astype(np.float32))
    
    with torch.no_grad():
        input_data = Variable(data1.cuda())
        predict = model(input_data)
        result = predict.data.cpu().numpy()
    
    roi = data[0,0,...]
    mask = np.array(result[0,0,...]>0.5).astype(int)
#    gt = np.load('../Tumor_crop/28_label.npy')
    gt = ROI_label
#    plt.figure(dpi=300)

    for i in range(8,mask.shape[2]-10):
        plt.subplot(1,6,i-7)
        contours = measure.find_contours(mask[:,:,i].transpose(1,0), 0.5)
        contours_gt = measure.find_contours(gt[:,:,i].transpose(1,0), 0.5)
        plt.imshow(roi[:,:,i].transpose(1,0),cmap='gray')
        plt.axis('off')
        for n, contour in enumerate(contours_gt):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=0.8, color='r')
            plt.axis('off')
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=0.8, color='b')
            plt.axis('off')

    plt.figure()
    for i in range(8,result.shape[4]-10):
        plt.subplot(1,6,i-7)
        plt.imshow(result[0,0,:,:,i].transpose(1,0),  cmap='jet')            
        plt.axis('off')
#    plt.savefig(os.path.join(r'.\ROI_jpg', 'GGO_'+str(ROI_num)+'.jpg')),vmin = -0.8,vmax = 1.5
#    plt.close()
            