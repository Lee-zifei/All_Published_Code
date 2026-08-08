# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：dither.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/08/18/
#    Function：
#
#    estimate_interference   : estimate the interference according to the shot time
#    IN        input         : A continuous aliasing signal of a seismic source
#              maintime      : Excitation time series of the source
#              assistime     : Continuous aliasing signal of the target source
#              dt            : Sampling time
#    OUT       interference  : Pseudo-deblending records with maintime
# 
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import sys, os, platform
if 'macos' in platform.platform().lower(): 
    myprog_path='/Users/lzf/Documents/cdut_zsh_group/python' 
elif 'linux' in platform.platform().lower(): 
    myprog_path='/media/lzf/Work/code/python' 
    myprog_path_survey='/home/lzf/code/python' 
else: 
    myprog_path='L:\data\code\python' 
sys.path.append(myprog_path)
sys.path.append(myprog_path_survey)
from subfunctions import *

import numpy as np
import math
# import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import pandas as pd
def estimate_interference(input, maintime, assistime, dt):

    
    n1, n2 = input.shape
    interference = np.zeros((n1, n2), dtype=np.float32)
    
    for i in range(n2):
        temp = (assistime[i] - maintime) / dt
        
        # Finding indices where the condition is satisfied
        bef = np.where((temp >= 0) & (temp < n1))[0]
        aft = np.where((temp <= 0) & (temp > -n1))[0]
        
        if len(bef) > 0:
            for j in range(len(bef)):
                index = int(np.floor((maintime[bef[j]] + n1 * dt - assistime[i]) / dt))
                interference[:index+1 , i] += input[n1 - index-1 :n1, bef[j]]
        
        if len(aft) > 0:
            for j in range(len(aft)):
                index = int(np.floor((assistime[i] + n1 * dt - maintime[aft[j]]) / dt))
                interference[n1 - index :n1, i] += input[:index, aft[j]]
    
    return interference

if __name__ == '__main__':
    datapath = '/home/lzf/code/python/2024_work/CSG_souece_different_deblending/test/bpdata/temp'
    file1_single = datapath+'/../blend_data/d1.dat'
    file2_single = datapath+'/../blend_data/d2.dat'

    file1_ble = datapath+'/../blend_data/d1b.dat'
    file2_ble = datapath+'/../blend_data/d2b.dat'

    file1_t1 = datapath+'/../blend_data/t1.dat'
    file2_t2 = datapath+'/../blend_data/t2.dat'
    
    n1 = 1333
    n2 = 128
    n3 = 600
    dt = 6e-3


    t1d = bin2npy_3d(file1_t1,1,1,n3)
    t1 = t1d.reshape(-1)[:n3]
    
    t2d = bin2npy_3d(file2_t2,1,1,n3)
    t2 = t2d.reshape(-1)[:n3]
    # 取前 n3 个数据

    trace = 100

    d1_blend = read_d3(file1_ble,[n1,n2,n3],trace)
    d2_blend = read_d3(file2_ble,[n1,n2,n3],trace)
    
    d1 = read_d3(file1_single,[n1,n2,n3],trace)
    d2 = read_d3(file2_single,[n1,n2,n3],trace)
    
    print(t1.shape)
    d1_blend_rev = d1+estimate_interference(d2,t2,t1,dt)
    d2_blend_rev = d2+estimate_interference(d1,t1,t2,dt)

    clip = 0.01
    mm = seis(2)
    fig = plt.figure(figsize=(16, 16),dpi=100)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    asp = 0.6

    ax3 = fig.add_subplot(161)
    ax3.imshow(d1_blend,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax3 = fig.add_subplot(162)
    ax3.imshow(d2_blend,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax5 = fig.add_subplot(163)
    ax5.imshow(d1_blend-d1_blend_rev,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax6 = fig.add_subplot(164)
    ax6.imshow(d2_blend-d2_blend_rev,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax6 = fig.add_subplot(165)
    ax6.imshow(d1_blend-estimate_interference(d2,t2,t1,dt),cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    
    ax6 = fig.add_subplot(166)
    ax6.imshow(d2_blend-estimate_interference(d1,t1,t2,dt),cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    plt.show()