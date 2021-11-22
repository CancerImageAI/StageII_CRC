# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:57:11 2019

@author: LJY-GJ
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:17:56 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.metrics import accuracy_score,roc_curve,recall_score,roc_auc_score,auc,confusion_matrix,cohen_kappa_score, f1_score, precision_score,matthews_corrcoef 
from tqdm import tqdm
from sklearn.svm import SVC,LinearSVC
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectFromModel, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import seaborn as sns

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
#    test_Num = np.array(Tumor_test['Name'].tolist())
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

#    lsvc_os = LinearSVC(C=0.05, penalty="l1", dual=False)
#    selector_OS = SelectFromModel(lsvc_os)
#    for a in range(2,18):
#        print(a)

    
    train_OS_tumor = np.array(train_OS_tumor)
    OS_index_0 = [i for i in range(len(train_OS_tumor)) if train_OS_tumor[i]==0]
    OS_index_1 = [i for i in range(len(train_OS_tumor)) if train_OS_tumor[i]==1]
    
    Tumor_x_OS = np.vstack((train_Feature[OS_index_0,:],train_Feature[OS_index_1,:]))
    OS = np.hstack((train_OS_tumor[OS_index_0],train_OS_tumor[OS_index_1]))

    estimator = linear_model.Ridge(random_state=0)
    selector_OS = RFE(estimator, 15, step=1)
    selector_OS.fit(Tumor_x_OS, OS)
    train_OS = selector_OS.transform(Tumor_x_OS)
    
    idx_OS = np.arange(0, Tumor_x_OS.shape[1])  #create an index array
    selected_index_OS = idx_OS[selector_OS.get_support() == True]  #get index positions of kept features
    
    selected_feature_OS = Tumor_x_OS[:,selected_index_OS]
    selected_f_name_OS = Feature_Name[selected_index_OS]
    #    Class_Type = [i if i==0  else '' for i in train_y]
#    Class_Type = [i if i=='IA' else 'Non_IA' for i in Class_Type]
    selected_Features_OS = pd.DataFrame(selected_feature_OS,index=OS,columns=selected_f_name_OS.tolist())


    selected_Features_OS['Class_Type'] = OS
    plt.figure()
    font = {'family' : 'Times New Roman',
			'weight' : 'normal',
			'size'   : 12,}
    plt.rc('font', **font)

#    cmap = sns.diverging_palette(260,5,sep=10,as_cmap=True)
    sns.heatmap(selected_feature_OS.transpose(), yticklabels=selected_f_name_OS, xticklabels=OS, center=1.8)
    plt.yticks(rotation=30)
    
    plt.figure()
    font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10,}
#    sns.boxplot(x="Class_Type", y=selected_f_name[0],data=selected_Features,palette="Set3",orient="v")
    
    for i in range(len(selected_index_OS)):
        ax = plt.subplot(1,15,i+1)  
        
        sns.boxplot(x="Class_Type", y=selected_f_name_OS[i],data=selected_Features_OS,
                    palette='bright',orient="v")
        plt.ylabel('')

        plt.ylabel(selected_f_name_OS[i],font)
        plt.tick_params(labelsize=10)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        
#    estimator = linear_model.Lasso(alpha=0.01,random_state=0)
#    selector_DFS = RFE(estimator, 2, step=1)
#    lsvc = LinearSVC(C=0.1, penalty="l1", dual=False)
#    selector_DFS = SelectFromModel(lsvc)
#    selector_DFS.fit(Tumor_x_DFS, DFS)
#    idx_DFS = np.arange(0, Tumor_x_DFS.shape[1])  #create an index array
#    selected_index_DFS = idx_DFS[selector_DFS.get_support() == True]  #get index positions of kept features
#    
#    selected_feature_DFS = Tumor_x_DFS[:,selected_index_DFS]
#    selected_f_name_DFS = Feature_Name[selected_index_DFS]
##

#    DFS_index_0 = [i for i in range(len(DFS)) if DFS[i]==0]
#    DFS_index_1 = [i for i in range(len(DFS)) if DFS[i]==1]
#    
#    Tumor_x_DFS = np.vstack((Tumor_x[DFS_index_0,:],Tumor_x[DFS_index_1,:]))
#    DFS = np.hstack((DFS[DFS_index_0],DFS[DFS_index_1]))
#
#    # Feature normalization
#    min_max_scaler = MinMaxScaler()
#    Tumor_x_OS = min_max_scaler.fit_transform(np.array(Tumor_x_OS))
#    Tumor_x_DFS = min_max_scaler.fit_transform(np.array(Tumor_x_DFS))
#    
#    estimator = linear_model.Lasso(alpha=0.01,random_state=0)
#    selector_OS = RFE(estimator, 2, step=1)
#
##    lsvc_os = LinearSVC(C=0.1, penalty="l1", dual=False)
##    selector_OS = SelectFromModel(lsvc_os)
##        SelectKBest(f_classif, k=3)
#    selector_OS.fit(Tumor_x_OS, OS)
#    
#
##        if i%3 != 0 :
##            plt.gca().get_yaxis().set_ticks([])
#
#    selected_Features_DFS = pd.DataFrame(selected_feature_DFS,index=DFS,columns=selected_f_name_DFS.tolist())
#
#
#    selected_Features_DFS['Class_Type'] = DFS
#    plt.figure()
#    font = {'family' : 'Times New Roman',
#			'weight' : 'normal',
#			'size'   : 12,}
#    plt.rc('font', **font)
#
##    cmap = sns.diverging_palette(260,5,sep=10,as_cmap=True)
#    sns.heatmap(selected_feature_DFS.transpose(), yticklabels=selected_f_name_DFS, xticklabels=DFS, center=1.8)
#    plt.yticks(rotation=30)
#    
#    plt.figure()
#    font = {'family' : 'Times New Roman',
#        'weight' : 'normal',
#        'size'   : 12,}
##    sns.boxplot(x="Class_Type", y=selected_f_name[0],data=selected_Features,palette="Set3",orient="v")
#    
#    for i in range(len(selected_index_DFS)):
#        ax = plt.subplot(1,2,i+1)  
#        
#        sns.boxplot(x="Class_Type", y=selected_f_name_DFS[i],data=selected_Features_DFS,
#                    palette='bright',orient="v")
#        plt.ylabel('')
#
#        plt.xlabel(selected_f_name_DFS[i],font)
#        plt.tick_params(labelsize=12)
#        labels = ax.get_xticklabels() + ax.get_yticklabels()
#        [label.set_fontname('Times New Roman') for label in labels]
#        if i%3 != 0 :
#            plt.gca().get_yaxis().set_ticks([])

    plt.show()

    
#    LN_csv = r'.\LN_Radiomics_Feature.csv'
#    f_testing = open(testing_csv)
#    test_list = pd.read_csv(f_testing)
#   
#    test_x = np.array(test_list.values[:,3:])
#    test_y = np.array(test_list['Class'].tolist())
#    
##    list_no = np.array(test_list['Name'])
##    test_x = test_x[[i for i,x in enumerate(list_no) if x in List_Num]]
##    test_y = test_y[[i for i,x in enumerate(list_no) if x in List_Num]]
#    
#    # Feature normalization
#    min_max_scaler = MinMaxScaler()
#    train_x = min_max_scaler.fit_transform(np.array(train_x,dtype=np.float64))
#    test_x = min_max_scaler.transform(test_x)
##    lsvc_T1 = LinearSVC(C=0.26, penalty="l1", dual=False).fit(x_T1, y_T1)
##    model_T1 = SelectFromModel(lsvc_T1, prefit=True)
##    estimator = linear_model.Lasso(alpha=0.005)
##    model_T1 = RFE(estimator, 12, step=1).fit(train_x, train_y)
#    selector = SelectKBest(f_classif, 20)
#    train_x = selector.fit_transform(train_x, train_y)
#    test_x = selector.transform(test_x)
#    selector.fit(train_x, train_y)
#    idx = np.arange(0, train_x.shape[1])  #create an index array
#    selected_index = idx[selector.get_support() == True]  #get index positions of kept features
#    
#    selected_feature = train_x[:,selected_index]
#    selected_f_name = feature_name[selected_index]
##    plt.figure()
#
##    sns.heatmap(selected_feature.transpose(), yticklabels=selected_f_name, xticklabels=y, center = 1.8)#
##    plt.yticks(rotation=15) 
#    # plt.rcParams.update({'font.family': 'Times New Roman'})
#    
##    index = (selected_index+2).tolist()
##    index.append(0)
##    selected_Features = T2_List.iloc[:,index]
#
