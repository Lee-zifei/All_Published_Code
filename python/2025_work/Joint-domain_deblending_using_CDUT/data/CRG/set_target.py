# -*- coding: utf-8 -*-
# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：main_crg.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/08/21/
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import argparse
import os,sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_config
from logger import create_logger
from models import build_model
from seislet import DataParallelWithCallback

myprog_path='../../' 
sys.path.append(myprog_path)
from subfunctions import *

def build_dataset():
    clip = 0.1
    ii = 80
    # aelmp
    files=os.listdir('./Train/target/')
    # set a
    datapath = '/media/lzf/Backup/data/All_you_need_data/BGP3_groud_rool'
    for i in range ()
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


