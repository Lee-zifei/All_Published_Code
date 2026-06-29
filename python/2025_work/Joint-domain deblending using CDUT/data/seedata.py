# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：stetrain.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/06/07/
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import sys, os, platform
import os
import torch
import torch.distributed as dist
from myprog import myimtocol
import numpy as np
from myprog import *
import matplotlib.pyplot as plt
from seis import seis
def bin2npy():
    k1=1
    n1 = 2560
    n2 = 96
    clip = 0.01
    for i in range(1,280):
        print('Bin2npy:'+str(i))
        ii = '{0:04}'.format(i)
        filename1 = './input2_select_patch_training/'+str(ii)+'.dat';
        filename2 = './target2_select_patch_training/'+str(ii)+'.dat';
        with open(filename1, "rb") as fid:
            fid.seek(n1*n2*(k1-1)*4,0)
            signal1 = np.fromfile(fid,dtype=np.float32,count=n1*n2).reshape((n2,n1)).T
            # plt.subplot(221)
            # plt.imshow(signal1,cmap=seis(2),vmax=clip, vmin=-clip)
        with open(filename2, "rb") as fid:
            fid.seek(n1*n2*(k1-1)*4,0)
            signal2 = np.fromfile(fid,dtype=np.float32,count=n1*n2).reshape((n2,n1)).T
            # plt.subplot(222)
            # plt.imshow(signal2,cmap=seis(2),vmax=clip, vmin=-clip)

        np.save('./input2_select_patch_training/'+str(ii)+'.npy', signal1)
        np.save('./target2_select_patch_training/'+str(ii)+'.npy', signal2)


def build_dataset():
    n1 = 2560  # the training volumn size   nx
    n2 = 96  # the training volumn sie    nt
    n3=1
    patch_rows = 96  # patch size
    patch_cols = 96
    tslide = 96  # time stride
    xslide = 32  # space stride
    trainsize = 0.8
    seed = 20240605

    for i in range(1,280):
        print('Setting Patch:'+str(i))
        ii = '{0:04}'.format(i)
        filename3 = './input2_select_patch_training/'+str(ii)+'.npy'
        inputs_patch_y = np.load(filename3)
        inputs_patch = inputs_patch_y/np.max(inputs_patch_y)
        filename4 = './target2_select_patch_training/'+str(ii)+'.npy'
        targets_patch = np.load(filename4)
        targets_patch = targets_patch/np.max(inputs_patch_y)
        
        # #############
        inputs_patch_col = myimtocol(inputs_patch, patch_rows, patch_rows, n1, n2,tslide, xslide, 1)
        targets_patch_col = myimtocol(targets_patch, patch_rows, patch_rows, n1, n2,tslide, xslide, 1)
        if i == 1:
            inputs = inputs_patch_col
            targets = targets_patch_col
        else:
            inputs = np.concatenate((inputs, inputs_patch_col), axis=0)
            targets = np.concatenate((targets, targets_patch_col), axis=0)

    print (inputs.shape)
    print (targets.shape)
    np.save('./targets2_combine_select_patch_training.npy', targets)
    np.save('./inputs2_combine_seletc_patch_training.npy', inputs)
    np.save('./targets2_noise_combine_select_patch_training.npy', inputs-targets)
def fig():
    clip = 0.1
    ii = 80
    # aelmp
    for j in range(1,5,1):
        if j == 1:
            data_type = 'a'
        elif j == 2:
            data_type = 'n'
        for i in range(1,2,1):
            targets = np.load('./2single_csg/Train/target/'+data_type+str(i)+'.npy')
            inputs = np.load('./2single_csg/Train/sample/'+data_type+str(i)+'.npy')
            print (targets.shape)
            mm = seis(2)
            print (np.max(inputs))
            fig = plt.figure(figsize=(16, 16),dpi=100)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            ax3 = fig.add_subplot(121)
            ax3.imshow(inputs, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            ax4 = fig.add_subplot(122)
            ax4.imshow(targets, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            plt.show()
if __name__ == '__main__':
#      bin2npy()
#      build_dataset()
    fig()


