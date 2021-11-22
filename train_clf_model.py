# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:53:55 2019

@author: PC
"""

import os
import time
import numpy as np
from Net_Classification import *
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy.ndimage.interpolation import rotate
import glob
import pandas as pd
import SimpleITK as sitk

def augment(roi, ifflip = True, ifrotate=True, ifswap = True):
    if ifrotate:
        angle1 = np.random.rand()*180
        roi = rotate(roi,angle1,axes=(1,2),reshape=False)
    if ifswap:
        axisorder = np.array([1, 0, 2])
        roi = np.transpose(roi,np.concatenate([[0],axisorder+1]))
    if ifflip:
        flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
        roi = np.ascontiguousarray(roi[:,::flipid[0],::flipid[1],::flipid[2]])
    return roi

class GGODataGenerator(Dataset):
    def __init__(self, img_path, csv_path, phase='train'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        f = open(csv_path)
        GGO_list = pd.read_csv(f) 
        List_Num = np.array(GGO_list['Save_Num'].tolist())
        Label = GGO_list['Class'].tolist()
        self.List_Num = List_Num    
        self.Label = Label
        self.img_path = img_path
        self.phase = phase
                
    def __getitem__(self,idx):  
        if self.phase =='train':
            if idx>=len(self.List_Num):
                idx = idx%len(self.List_Num)
                ifflip = True
                ifrotate= False
                ifswap = False
            elif idx>=(len(self.List_Num)*2):
                idx = idx%(len(self.List_Num)*2)
                ifflip = False
                ifrotate= True
                ifswap = False
            elif idx>=(len(self.List_Num)*3):
                idx = idx%(len(self.List_Num)*3)
                ifflip = False
                ifrotate= False
                ifswap = True
            else:
                ifflip = False
                ifrotate= False
                ifswap = False

        if self.phase == 'train':
            File = self.List_Num[idx]
            roi_path = self.img_path+'/'+str(File)+'_roi.npy'
            ROI = np.load(roi_path)
            ROI = augment(ROI, ifflip = ifflip, ifrotate=ifrotate, ifswap = ifswap)
        else:
            File = self.List_Num[idx]
            roi_path = self.img_path+'/'+str(File)+'_roi.npy'
            ROI = np.load(roi_path)
        Label = self.Label[idx]
        return ROI, np.array(Label)
        
    def __len__(self):
        if self.phase == 'train':
            return len(self.List_Num)*4
        else:
            return len(self.List_Num)

def get_lr(epoch, lr):
    if epoch <= epochs * 0.5:
        lr = lr
    elif epoch <= epochs * 0.8:
        lr = 0.1 * lr
    else:
        lr = 0.01 * lr
    return lr

def get_optimizer(parameters, st, lr, momentum=0.9):
    if st == 'sgd':
        return torch.optim.SGD(parameters, lr = lr, momentum=momentum)
    elif st == 'adam':
        return torch.optim.Adam(parameters, lr = lr, weight_decay=1e-4)


def train(dataloader, net, loss_fun,epoch, optimizer, get_lr, save_freq, save_dir):
    starttime = time.time()
    net.train()
    lr = get_lr(epoch, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    metrics = []
    acc = []
    ok = 0
#    lambda2 = 0.01
    lambda1 = 0.001
    for i, (data,Class) in enumerate(data_loader):
        optimizer.zero_grad()   # clear gradients for next train
        data = Variable(data.cuda(async = True))
        target = Variable(Class.cuda(async = True))
        l1_regularization = torch.tensor(0.).cuda()
#        l2_regularization = torch.tensor(0.).cuda()
        target = target.long()
        output = net(data)
        
        loss_output = loss_fun(output, target)
        for param in [clf_model.fc.weight, clf_model.fc.bias
                      ]:
#             clf_model.fc3.weight, clf_model.fc3.bias,
#            l2_regularization += torch.norm(param,2)
            l1_regularization += torch.sum(torch.abs(param))
        loss = loss_output +  lambda1*l1_regularization #+ lambda2*l2_regularization  
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        metrics.append(loss_output)
        _, predicted = torch.max(output.data,1)
        ok = ok+(predicted==target).sum()            
        traind_total = (i + 1) * len(target)
        acc_output = 100. * ok / traind_total
#        print(len(target), traind_total, acc_output)
        acc.append(acc_output)
    if epoch % save_freq == 0:            
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()         
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict},
            os.path.join(save_dir, 'OS_%03d.ckpt' % epoch))
     
    endtime = time.time()
    metrics = np.asarray(metrics, np.float32)
    acc = np.asarray(acc, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('loss %2.4f' % (np.mean(metrics)))
    print('Accuracy %2.4f' % (np.mean(acc)))
    print('time:%3.2f'%(endtime-starttime))
    
if __name__ == '__main__':
    Pretrained_path = './model'
    clf_model = Classify_Model().cuda()
    classify_path = os.path.join(Pretrained_path, '200.ckpt')
    modelCheckpoint = torch.load(classify_path)
    pretrained_dict = modelCheckpoint['state_dict']
    model_dict = clf_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#filter out unnecessary keys
    model_dict.update(pretrained_dict)
    clf_model.load_state_dict(model_dict)
    torch.cuda.set_device(0)
    for k,v in clf_model.named_parameters():
        if k!='fc.weight' and k!='fc.bias':   
            v.requires_grad = False    
#              k!='fc3.weight' and k!='fc3.bias' and \
#clf_model.fc3.weight, clf_model.fc3.bias,
    optimizer = get_optimizer(parameters=[clf_model.fc.weight, clf_model.fc.bias
                            ], st='adam', lr=0.0001)
#    class_weight = torch.FloatTensor([1/38, 1/135]).cuda()
    loss = torch.nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True
    img_path = '../Train_Crop/OS_Crop'
    csv_path = './train_OS_List.csv'
    
    dataset = GGODataGenerator(img_path,csv_path, phase='train')
    data_loader = DataLoader(dataset, batch_size = 4,shuffle = True, num_workers = 0, pin_memory=True)
    save_freq = 5
    epochs = 25
    lr = 0.0001
#    l1_crit = nn.L1Loss(size_average=False)
    
    save_dir = './model/clf_model'
    for epoch in range(0, epochs + 1):
        train(data_loader, clf_model, loss, epoch, optimizer, get_lr, save_freq, save_dir)
