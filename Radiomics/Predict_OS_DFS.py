# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:25:14 2019

@author: PC
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel,mutual_info_classif
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import scipy.stats as stats
from pandas import DataFrame as DF
from imblearn.over_sampling import SMOTE


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

if __name__ == '__main__': 
    train_path = '../../Results/Tumor_Feature_TrainingData.csv'

    f_train = open(train_path)
    Tumor_list = pd.read_csv(f_train)
    List_Num = np.array(Tumor_list['Name'].tolist())
    train_OS_tumor = Tumor_list['Osevents'].tolist()
    train_DFS_tumor = Tumor_list['DFSevents'].tolist()

    train_Feature_initial = Tumor_list.values[:,6:]
    Feature_Name = np.array(list(Tumor_list.head(0))[6:])
 
    test_path = '../../Results/Tumor_Feature_TestingData.csv'
    f_test = open(test_path)
    Tumor_test = pd.read_csv(f_test)
    test_OS_tumor = np.array(Tumor_test['Osevents'].tolist())
    test_DFS_tumor = np.array(Tumor_test['DFSevents'].tolist())
    OS = np.array(Tumor_test['OS'].tolist())
    DFS = np.array(Tumor_test['DFS'].tolist())
    test_Feature_initial = Tumor_test.values[:,6:]

    # Feature normalization
    min_max_scaler = MinMaxScaler()
    Tumor_Feature = np.vstack((train_Feature_initial,test_Feature_initial))
    Tumor_Feature = min_max_scaler.fit_transform(Tumor_Feature)
    train_Feature = Tumor_Feature[:len(List_Num),:]
    test_Feature = Tumor_Feature[len(List_Num):,:]


    estimator = linear_model.Ridge(random_state=0)
    selector_OS = RFE(estimator, 15, step=1)
#        selector_OS = SelectKBest(f_classif, k=a)
    selector_OS.fit(train_Feature, train_OS_tumor)
    train_OS = selector_OS.transform(train_Feature)
    test_OS = selector_OS.transform(test_Feature)
    
    idx_OS = np.arange(0, train_Feature.shape[1])  #create an index array
    selected_index_OS = idx_OS[selector_OS.get_support() == True]  #get index positions of kept features
    
    selected_feature_OS = train_Feature[:,selected_index_OS]
    selected_f_name_OS = Feature_Name[selected_index_OS]
    print('OS Tumor QI Feature:',selected_f_name_OS)
    
    train_OS, OS_train = SMOTE(sampling_strategy='auto',random_state=0).fit_sample(train_OS, train_OS_tumor)
    clf_OS = svm.SVC(kernel="rbf", gamma='auto', probability=True, random_state=48)
    clf_OS.fit(train_OS, OS_train)
    pred_prob_OS = clf_OS.predict_proba(test_OS)[:,1]
    pred_label_OS = clf_OS.predict(test_OS)
    pred_label_OS = np.array(pred_label_OS).astype(int)
    fpr_OS,tpr_OS,threshold_OS = roc_curve(test_OS_tumor, np.array(pred_prob_OS)) ###计算真正率和假正率
    auc_OS = auc(fpr_OS,tpr_OS)
    auc_l_OS, auc_h_OS, auc_std_OS = confindence_interval_compute(np.array(pred_prob_OS), test_OS_tumor)
    print('Tumor OS Feature AUC:%.2f'%auc_OS,'+/-%.2f'%auc_std_OS,'  95% CI:[','%.2f,'%auc_l_OS,'%.2f'%auc_h_OS,']')
    print('Tumor OS Feature ACC:%.2f'%accuracy_score(test_OS_tumor,pred_label_OS)) 
    Tumor_OS_result = []
    Tumor_OS_result.append(test_OS_tumor)
    Tumor_OS_result.append(OS)
    Tumor_OS_result.append(pred_prob_OS)
    Tumor_OS_result.append(pred_label_OS)
    Tumor_OS_result = np.array(Tumor_OS_result)
    Tumor_OS_result = Tumor_OS_result.T
#    save_OS = pd.DataFrame(Tumor_OS_result, columns = ['OS_Event', 'OS_Time', 'Prob', 'Pred_Label']) 
#    save_OS.to_csv('../../Results/Tumor_OS_Result.csv',index=False,header=True)

    estimator = linear_model.Ridge(random_state=0)
    selector_DFS = RFE(estimator, 14, step=1)
    selector_DFS.fit(train_Feature, train_DFS_tumor)
    train_DFS = selector_DFS.transform(train_Feature)
    test_DFS = selector_DFS.transform(test_Feature)
    
    idx_DFS = np.arange(0, train_Feature.shape[1])  #create an index array
    selected_index_DFS = idx_DFS[selector_DFS.get_support() == True]  #get index positions of kept features
    
    selected_feature_DFS = train_Feature[:,selected_index_DFS]
    selected_f_name_DFS = Feature_Name[selected_index_DFS]
    print('DFS Tumor QI Feature:',selected_f_name_DFS)
    
    train_DFS, DFS_train = SMOTE(sampling_strategy='auto',random_state=0).fit_sample(train_DFS, train_DFS_tumor)
    clf_DFS = svm.SVC(kernel="rbf", gamma='auto', probability=True, random_state=48)
    clf_DFS.fit(train_DFS, DFS_train)
    test_prob_DFS = clf_DFS.predict_proba(test_DFS)[:,1]
    pred_label_DFS = clf_DFS.predict(test_DFS)
    pred_label_DFS = np.array(pred_label_DFS).astype(int)
    fpr_DFS,tpr_DFS,threshold_DFS = roc_curve(test_DFS_tumor, np.array(test_prob_DFS)) ###计算真正率和假正率
    auc_DFS = auc(fpr_DFS,tpr_DFS)
    auc_l_DFS, auc_h_DFS, auc_std_DFS = confindence_interval_compute(np.array(test_prob_DFS), test_DFS_tumor)
    print('Tumor DFS Feature AUC:%.2f'%auc_DFS,'+/-%.2f'%auc_std_DFS,'  95% CI:[','%.2f,'%auc_l_DFS,'%.2f'%auc_h_DFS,']')
    print('Tumor DFS Feature ACC:%.2f'%accuracy_score(test_DFS_tumor,pred_label_DFS)) 
    Tumor_DFS_result = []
    Tumor_DFS_result.append(test_DFS_tumor)
    Tumor_DFS_result.append(DFS)
    Tumor_DFS_result.append(test_prob_DFS)
    Tumor_DFS_result.append(pred_label_DFS)
    Tumor_DFS_result = np.array(Tumor_DFS_result)
    Tumor_DFS_result = Tumor_DFS_result.T
#    save_DFS = pd.DataFrame(Tumor_DFS_result, columns = ['OS_Event', 'OS_Time', 'Prob', 'Pred_Label']) 
#    save_DFS.to_csv('../../Results/Tumor_DFS_Result.csv',index=False,header=True)
    
    
    LN_train_path = '../../Results/LN_Feature_TrainingData.csv'
    f_LN = open(LN_train_path)
    LN_list = pd.read_csv(f_LN)
    LN_List_Num = np.array(LN_list['Name'].tolist())
    train_LN_OS = LN_list['Osevents'].tolist()
    train_LN_DFS = LN_list['DFSevents'].tolist()
    Num = np.array(LN_list['Num'].tolist())
    train_LN_Feature = np.array(LN_list.values[:,6:])
    LN_Feature_Name = np.array(['Num']+list(LN_list.head(0))[6:])
    train_LN_Feature_initial = np.hstack((Num[:,np.newaxis],train_LN_Feature))
    
    LN_test_path = '../../Results/Tumor_Feature_TestingData.csv'
    f_test_LN = open(LN_test_path)
    test_LN_list = pd.read_csv(f_test_LN)
    test_LN_OS = np.array(test_LN_list['Osevents'].tolist())
    test_LN_DFS = np.array(test_LN_list['DFSevents'].tolist())
    test_Num = np.array(test_LN_list['Num'].tolist())
    test_LN_Feature = np.array(test_LN_list.values[:,6:])
    test_LN_Feature_initial = np.hstack((test_Num[:,np.newaxis],test_LN_Feature))
    LN_OS = np.array(test_LN_list['OS'].tolist())
    LN_DFS = np.array(test_LN_list['DFS'].tolist())
    
    # Feature normalization
    min_max_scaler = MinMaxScaler()
    LN_Feature = np.vstack((train_LN_Feature_initial, test_LN_Feature_initial))
    LN_Feature = min_max_scaler.fit_transform(np.array(LN_Feature))
    train_LN_Feature = LN_Feature[:len(LN_List_Num),:]
    test_LN_Feature = LN_Feature[len(LN_List_Num):,:]
    
    estimator = linear_model.Lasso(random_state=0)
    LN_selector_OS = RFE(estimator, 14, step=1)
    LN_selector_OS.fit(train_LN_Feature, train_LN_OS)
    train_LN_Feature_OS = LN_selector_OS.transform(train_LN_Feature)
    test_LN_Feature_OS = LN_selector_OS.transform(test_LN_Feature)
    idx_OS_LN = np.arange(0, train_LN_Feature.shape[1])  #create an index array
    selected_index_OS = idx_OS_LN[LN_selector_OS.get_support() == True]  #get index positions of kept features
    
    selected_feature_OS = train_LN_Feature[:,selected_index_OS]
    selected_f_name_OS = LN_Feature_Name[selected_index_OS]
    print('OS LN QI Feature:',selected_f_name_OS)
    
    train_LN_Feature_OS, LN_OS_train = SMOTE(sampling_strategy='auto',random_state=0).fit_sample(train_LN_Feature_OS, train_LN_OS)
    LN_clf_OS = svm.SVC(kernel="rbf", gamma='auto', probability=True, random_state=48)
    LN_clf_OS.fit(train_LN_Feature_OS, LN_OS_train)
    LN_test_prob_OS = LN_clf_OS.predict_proba(test_LN_Feature_OS)[:,1]
    LN_pred_label_OS = LN_clf_OS.predict(test_LN_Feature_OS)
    LN_pred_label_OS = np.array(LN_pred_label_OS).astype(int)
    LN_fpr_OS,LN_tpr_OS,LN_threshold_OS = roc_curve(test_LN_OS, np.array(LN_test_prob_OS)) ###计算真正率和假正率
    LN_auc_OS = auc(LN_fpr_OS,LN_tpr_OS)
    LN_auc_l_OS, LN_auc_h_OS, LN_auc_std_OS = confindence_interval_compute(np.array(LN_test_prob_OS), test_LN_OS)
    print('LN OS Feature AUC:%.2f'%LN_auc_OS,'+/-%.2f'%LN_auc_std_OS,'  95% CI:[','%.2f,'%LN_auc_l_OS,'%.2f'%LN_auc_h_OS,']')
    print('LN OS Feature ACC:%.2f'%accuracy_score(test_LN_OS,LN_pred_label_OS)) 
    LN_OS_result = []
    LN_OS_result.append(test_LN_OS)
    LN_OS_result.append(LN_OS)
    LN_OS_result.append(LN_test_prob_OS)
    LN_OS_result.append(LN_pred_label_OS)
    LN_OS_result = np.array(LN_OS_result)
    LN_OS_result = LN_OS_result.T
#    save_LN_OS = pd.DataFrame(LN_OS_result, columns = ['OS_Event', 'OS_Time', 'Prob', 'Pred_Label']) 
#    save_LN_OS.to_csv('../../Results/LN_OS_Result.csv',index=False,header=True)
    estimator = linear_model.Lasso(alpha=0.01,random_state=0)
    LN_selector_DFS = RFE(estimator, 4, step=1)
    LN_selector_DFS.fit(train_LN_Feature, train_LN_DFS)
    train_LN_Feature_DFS = LN_selector_DFS.transform(train_LN_Feature)
    test_LN_Feature_DFS = LN_selector_DFS.transform(test_LN_Feature)
    
    idx_DFS = np.arange(0, train_LN_Feature.shape[1])  #create an index array
    selected_index_DFS = idx_DFS[LN_selector_DFS.get_support() == True]  #get index positions of kept features
    
    selected_feature_DFS = train_LN_Feature[:,selected_index_DFS]
    selected_f_name_DFS = LN_Feature_Name[selected_index_DFS]
    print('DFS LN QI Feature:',selected_f_name_DFS)
    
    train_LN_Feature_DFS, LN_DFS_train = SMOTE(sampling_strategy='auto',random_state=0).fit_sample(train_LN_Feature_DFS, train_LN_DFS)
    LN_clf_DFS = svm.SVC(kernel="rbf", gamma='auto', probability=True, random_state=48)
    LN_clf_DFS.fit(train_LN_Feature_DFS, LN_DFS_train)
    LN_test_prob_DFS = LN_clf_DFS.predict_proba(test_LN_Feature_DFS)[:,1]
    LN_pred_label_DFS = LN_clf_DFS.predict(test_LN_Feature_DFS)
    LN_pred_label_DFS = np.array(LN_pred_label_DFS).astype(int)
    LN_fpr_DFS,LN_tpr_DFS,LN_threshold_DFS = roc_curve(test_LN_DFS, np.array(LN_test_prob_DFS)) ###计算真正率和假正率
    LN_auc_DFS = auc(LN_fpr_DFS,LN_tpr_DFS)
    LN_auc_l_DFS, LN_auc_h_DFS, LN_auc_std_DFS = confindence_interval_compute(np.array(LN_test_prob_DFS), test_LN_DFS)
    print('LN DFS Feature AUC:%.2f'%LN_auc_DFS,'+/-%.2f'%LN_auc_std_DFS,'  95% CI:[','%.2f,'%LN_auc_l_DFS,'%.2f'%LN_auc_h_DFS,']')
    print('LN DFS Feature ACC:%.2f'%accuracy_score(test_LN_DFS,LN_pred_label_DFS))
    LN_DFS_result = []
    LN_DFS_result.append(test_LN_DFS)
    LN_DFS_result.append(LN_DFS)
    LN_DFS_result.append(LN_test_prob_DFS)
    LN_DFS_result.append(LN_pred_label_DFS)
    LN_DFS_result = np.array(LN_DFS_result)
    LN_DFS_result = LN_DFS_result.T
#    save_LN_DFS = pd.DataFrame(LN_DFS_result, columns = ['OS_Event', 'OS_Time', 'Prob', 'Pred_Label']) 
#    save_LN_DFS.to_csv('../../Results/LN_DFS_Result.csv',index=False,header=True)

    scales = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for scale in scales:
        prob_fusion = scale*np.array(pred_prob_OS)+(1-scale)*np.array(LN_test_prob_OS)
        auc_value = roc_auc_score(np.array(test_LN_OS),prob_fusion)
        auc_fl, auc_fh, auc_fstd = confindence_interval_compute(np.array(prob_fusion), test_LN_OS)
        print('Fusion Scale',scale,'AUC:%.2f'%auc_value,'+/-%.2f'%auc_fstd,
              '  95% CI:[','%.2f,'%auc_fl,'%.2f'%auc_fh,']')      
    Fusion = np.zeros([len(LN_test_prob_OS),2])
    Fusion[:,0] = np.array(pred_prob_OS)
    Fusion[:,1] = np.array(LN_test_prob_OS)
    Fusion_min = Fusion.min(1)
    Fusion_max = Fusion.max(1)


    auc_min = roc_auc_score(np.array(test_LN_OS),Fusion_min)
    auc_fl_min, auc_fh_min, auc_fstd_min = confindence_interval_compute(np.array(Fusion_min), test_LN_DFS)
    print('Min Fusion AUC:%.2f'%auc_min,'+/-%.2f'%auc_fstd_min,'  95% CI:[','%.2f,'%auc_fl_min,'%.2f'%auc_fh_min,']')

    auc_max = roc_auc_score(np.array(test_LN_OS),Fusion_max)
    auc_fl_max, auc_fh_max,auc_fstd_max = confindence_interval_compute(np.array(Fusion_max), test_LN_DFS)
    print('Max Fusion AUC:%.2f'%auc_max,'+/-%.2f'%auc_fstd_max,'  95% CI:[','%.2f,'%auc_fl_max,'%.2f'%auc_fh_max,']')
    
    OS_result = []
    OS_result.append(test_LN_OS)
    OS_result.append(LN_OS)
    OS_result.append(Fusion_min)
    OS_result.append(Fusion_min>0.5)
    OS_result = np.array(OS_result)
    OS_result = OS_result.T
#    Fusion_OS = pd.DataFrame(OS_result, columns = ['OS_Event', 'OS_Time', 'Prob', 'Pred_Label']) 
#    Fusion_OS.to_csv('../../Results/Fusion_OS_Result.csv',index=False,header=True)
    print(accuracy_score(test_LN_OS,Fusion_min>0.5))
    
    for scale in scales:
        prob_fusion = scale*np.array(test_prob_DFS)+(1-scale)*np.array(LN_test_prob_DFS)
#        fpr,tpr,threshold = roc_curve(np.array(real_class),prob_fusion)
#        auc_value = auc(fpr,tpr)
        auc_value = roc_auc_score(np.array(test_LN_DFS),prob_fusion)
        auc_fl, auc_fh, auc_fstd = confindence_interval_compute(np.array(prob_fusion), test_LN_DFS)
        print('Fusion Scale',scale,'AUC:%.2f'%auc_value,'+/-%.2f'%auc_fstd,
              '  95% CI:[','%.2f,'%auc_fl,'%.2f'%auc_fh,']')      
    Fusion = np.zeros([len(LN_test_prob_DFS),2])
    Fusion[:,0] = np.array(test_prob_DFS)
    Fusion[:,1] = np.array(LN_test_prob_DFS)
    Fusion_min = Fusion.min(1)
    Fusion_max = Fusion.max(1)

    auc_min = roc_auc_score(np.array(test_LN_DFS),Fusion_min)
    auc_fl_min, auc_fh_min, auc_fstd_min = confindence_interval_compute(np.array(Fusion_min), test_LN_DFS)
    print('Min Fusion AUC:%.2f'%auc_min,'+/-%.2f'%auc_fstd_min,'  95% CI:[','%.2f,'%auc_fl_min,'%.2f'%auc_fh_min,']')

    auc_max = roc_auc_score(np.array(test_LN_DFS),Fusion_max)
    auc_fl_max, auc_fh_max,auc_fstd_max = confindence_interval_compute(np.array(Fusion_max), test_LN_DFS)
    print('Max Fusion AUC:%.2f'%auc_max,'+/-%.2f'%auc_fstd_max,'  95% CI:[','%.2f,'%auc_fl_max,'%.2f'%auc_fh_max,']')
    
    Fusion_DFS_Optimal = 0.1*np.array(test_prob_DFS)+(1-0.1)*np.array(LN_test_prob_DFS)
    
    DFS_result = []
    DFS_result.append(test_LN_DFS)
    DFS_result.append(LN_DFS)
    DFS_result.append(Fusion_DFS_Optimal)
    DFS_result.append(Fusion_DFS_Optimal>0.5)
    DFS_result = np.array(DFS_result)
    DFS_result = DFS_result.T
#    Fusion_DFS = pd.DataFrame(DFS_result, columns = ['OS_Event', 'OS_Time', 'Prob', 'Pred_Label']) 
#    Fusion_DFS.to_csv('../../Results/Fusion_DFS_Result.csv',index=False,header=True)
    print(accuracy_score(test_LN_DFS,Fusion_DFS_Optimal>0.5))

