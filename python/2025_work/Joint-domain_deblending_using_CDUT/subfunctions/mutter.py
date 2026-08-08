# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：mutter.py
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
# import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def mutter(inputs, x0, t0, k):
    nt,nx=inputs.shape[0],inputs.shape[1]
    output=inputs
    for i in range(1,nt):###axis y
        for j in range(1,nx):###axis x
            if j < x0 :
                if i < math.floor(-k*j+t0+k*x0):
                    output[i,j]=0
            if j >=x0:
                if i < math.floor(k*j+t0-k*x0):
                    output[i,j]=0
    return output
