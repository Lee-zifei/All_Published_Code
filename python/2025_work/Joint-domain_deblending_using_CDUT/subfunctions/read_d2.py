# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：read_d2.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/08/21/
#    Function：指针读取第二个维度数据
#    input:
#          string filename       :    the data path need to be plotted
#          int    a              :    the data size of 3 dimension
#          int    trace          :    the sample trace of 2nd dimension
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
    myprog_path_survey='home/lzf/code/python' 
else: 
    myprog_path='L:\data\code\python' 
sys.path.append(myprog_path)
sys.path.append(myprog_path_survey)
from subfunctions import *  
import numpy as np

def read_d2(filename,a,trace):
    n1 = a[0]
    n2 = a[1]
    n3 = a[2]
    with open(filename, "rb") as fid:
        fid.seek(n1*n2*(trace)*4,0)
        signal = np.fromfile(fid,dtype=np.float32,count=n1*n2).reshape((n2,n1)).T
    return signal
