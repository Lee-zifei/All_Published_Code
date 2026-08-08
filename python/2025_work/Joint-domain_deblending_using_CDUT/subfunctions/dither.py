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
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import sys, os, platform
if 'macos' in platform.platform().lower(): 
    myprog_path='/Users/lzf/Documents/cdut_zsh_group/python' 
elif 'linux' in platform.platform().lower(): 
    myprog_path='/media/lzf/Work/code/python' 
else: 
    myprog_path='L:\data\code\python' 
sys.path.append(myprog_path)
from subfunctions import *  
import numpy as np
import math
# 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def dither(input1, time, pytorch = False):
    n1, n2 = input1.shape
    if pytorch:
        import torch
        out1 = torch.zeros((n1,n2), dtype=torch.float32)
    else:
        out1 = np.zeros((n1, n2), dtype='float32')
    n22 = len(time)
    if n2 != n22:
        print ('Error in size of delay time')

    for ix in range(n2):
        temp = int(time[ix])
        if temp > 0:
            if temp>=0 and temp<n1:
                out1[temp:,ix] = input1[:n1-temp,ix]
        else:
            begin = -temp
            if begin>=0 and begin<n1:
                out1[:n1-begin,ix] = input1[begin:,ix]
    return out1  
  
  
 
