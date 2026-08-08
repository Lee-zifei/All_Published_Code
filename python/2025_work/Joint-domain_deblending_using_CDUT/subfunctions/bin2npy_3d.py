# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：bin2npy.py
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
import scipy.io as io

def bin2npy_3d(filename,n1,n2,n3):
    single_3d = np.zeros((n1,n2,n3))
    for i in range(n3):
        with open(filename, "rb") as fid:
            fid.seek(n1*n2*(i)*4,0)
            signal = np.fromfile(fid,dtype=np.float32,count=n1*n2).reshape((n2,n1)).T
        single_3d[:,:,i] = signal
    return single_3d

if __name__ == '__main__':
    filename = ''
    n1 = 1
    n2 = 1
    n3 = 1
    npy_data = bin2npy_3d(filename,n1,n2,1)
    # io.savemat("dataname.mat", {'data': npy_data})