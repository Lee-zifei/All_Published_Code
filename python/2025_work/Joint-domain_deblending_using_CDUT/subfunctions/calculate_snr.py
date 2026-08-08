# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：calculate_snr.py
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

  
 
def calculate_snr(signal,noisy):
    error = signal - noisy
    temp1 = np.sum(np.sum(signal*signal))
    temp2 = np.sum(np.sum(error*error))
    snr = 10*np.log10(temp1/temp2)
    return snr