# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:09:46 2019

@author: PC
"""

import os
import time
import numpy as np
from Net_Classification_test import *
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.ndimage.interpolation import rotate
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc,confusion_matrix,cohen_kappa_score
import pandas as pd
from tqdm import tqdm

def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

if __name__ == "__main__":
    Pretrained_path = './model/clf_model'
    model = Classify_Model().cuda()
    classify_path = os.path.join(Pretrained_path, 'OS.ckpt')
    modelCheckpoint = torch.load(classify_path)
    pretrained_dict = modelCheckpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#filter out unnecessary keys
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    img_path = '../Test_Crop/OS_Crop'
    list_path = './test_OS_List.csv'
    f = open(list_path)
    Tumor_list = pd.read_csv(f)
    List_Num = np.array(Tumor_list['Save_Num'].tolist())
    Class = np.array(Tumor_list['Class'].tolist()) 
    prob = []
    prob_label = []
    real_class = []
    test_result=[]
    for i in tqdm(range(len(List_Num))): 
        roi_path = os.path.join(img_path, str(List_Num[i])+'_roi.npy')
        data = np.load(roi_path)
        data = data[np.newaxis,...]
        data = torch.from_numpy(data.astype(np.float32))
        with torch.no_grad():
            input_data = Variable(data.cuda())
            predict = model(input_data)            
            result = predict.data.cpu().numpy()
            prob.append(result[0][1])
            real_class.append(Class[i])
            prob_label.append(np.argmax(result[0]))
            test = {}
            test['Num'] = List_Num[i]
            test['Class'] = Class[i]
            test['Prob'] = result[0][1]
            test_result.append(test)
    df = pd.DataFrame(test_result).fillna('null')
    df.to_csv('./test_result_OS.csv',index=False,sep=',')
    print('Our Model ACC:',accuracy_score(real_class,prob_label)*100)      
    fpr_OS,tpr_OS,threshold_OS = roc_curve(np.array(real_class),np.array(prob)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(prob), np.array(real_class))
    print('Tumor OS Feature AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')
    print('Tumor OS Feature ACC:%.2f'%accuracy_score(real_class,prob_label)) 