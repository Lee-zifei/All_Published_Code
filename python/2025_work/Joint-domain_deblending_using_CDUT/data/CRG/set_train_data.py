# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：fig.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：2024010196@stu.cdut.edu.cn
#    Work：2024/11/01/
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import sys, os, platform
if 'macos' in platform.platform().lower(): 
    myprog_path='/Users/lzf/Documents/cdut_zsh_group/python/subfuctions' 
elif 'linux' in platform.platform().lower(): 
    myprog_path='/media/lzf/Work/code/python' 
    myprog_path_survey='/home/lzf/code/python' 
else: 
    myprog_path='L:\data\code\python' 
sys.path.append(myprog_path)
sys.path.append(myprog_path_survey)
from subfunctions import *  
import numpy as np
import matplotlib.pyplot as plt

import sys, os, platform
import os
import torch
import torch.distributed as dist
# from myprog import myimtocol
import numpy as np
# from myprog import *
import matplotlib.pyplot as plt
# from seis import seis
import random

def generate_random_01_matrix(n1, n2,sparsity):
    """
    生成一个大小为 n1 x n2 的随机 0-1 矩阵
    """
    sampling_matrix = np.random.choice([0, 1], size=(n1, n2), p=[1-sparsity, sparsity])
    
    return sampling_matrix

def build_dataset():
    clip = 0.1
    ii = 80
    # aelmp
    files=os.listdir('./Train/target/')
    for f in files:
        filename=f.split('.')[0]
        targets = np.load('./Train/target/'+filename+'.npy')
        print('Is setting data:'+filename)
        
        b,n1,n2 = targets.shape
        sp = np.random.choice([0.2, 0.5])
        windows = generate_random_01_matrix(n1,n2,sp)
        inputs = np.zeros_like(targets)
        # print (targets.shape)
        mm = seis(2)
        target1 = targets[0,:,:]
        target2 = targets[1,:,:]

        input1 = target1+target2*windows
        input2 = target2+target1*windows

        inputs[0,:,:] = input1
        inputs[1,:,:] = input2

        np.save('./Train/sample/'+filename+'.npy',inputs)
            
            # fig = plt.figure(figsize=(16, 16),dpi=100)
            # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            # ax3 = fig.add_subplot(141)
            # ax3.imshow(target1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            # ax4 = fig.add_subplot(142)
            # ax4.imshow(target2, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            # ax1 = fig.add_subplot(143)
            # ax1.imshow(input1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            # ax2 = fig.add_subplot(144)
            # ax2.imshow(input2, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
            # plt.show()

def fig():
    clip = 0.1
    ii = 80
    # aelmp
    for j in range(1,6,1):
        if j == 1:
            data_type = 'a'
            ed_nm = 400
        elif j == 2:
            data_type = 'e'
            ed_nm = 140
        elif j == 3:
            data_type = 'l'
            ed_nm = 63
        elif j == 4:
            data_type = 'm'
            ed_nm = 57
        elif j == 5:
            data_type = 'p'
            ed_nm = 553
        for i in range(1,2,1):
            print(i)
            targets = np.load('./Train/target/'+str(data_type)+str(i)+'.npy')
            inputs = np.load('./Train/sample/'+str(data_type)+str(i)+'.npy')
            # print (targets.shape)
            mm = seis(2)
            # print (np.max(targets))
            fig = plt.figure(figsize=(16, 16),dpi=100)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

            ax3 = fig.add_subplot(221)
            ax3.imshow(inputs[0,:,:], cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            ax4 = fig.add_subplot(222)
            ax4.imshow(inputs[1,:,:], cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            ax3 = fig.add_subplot(223)
            ax3.imshow(targets[0,:,:], cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        
            ax4 = fig.add_subplot(224)
            ax4.imshow(targets[1,:,:], cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
            
            plt.show()
if __name__ == '__main__':
#      bin2npy()
    # build_dataset()
    fig()


