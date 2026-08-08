# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li
#    
#    Filename：read_d3.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：2024010196@cdut.edu.cn
#    Data：2024/11/19/
    # Function: 指针读取第三个维度数据
    # Args:
    #     filename (str): 数据文件路径
    #     a (tuple): 数据尺寸 (n1, n2, n3)
    #     trace (int): 第三维度中需要采样的某个 trace
    # Returns:
    #     numpy.ndarray: 读取的二维数据 
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

def read_d3(filename,a,trace):
    n1 = a[0]
    n2 = a[1]
    n3 = a[2]
    data1 = np.zeros((n1, n3), dtype=np.float32)

    with open(filename, "rb") as fid:
        for j in range(n3):
            offset = n1 * n2 * j * 4 + (trace) * n1 * 4
            fid.seek(offset, 0) 
            data1[:, j] = np.fromfile(fid, dtype=np.float32, count=n1)
    return data1


  
  
  
 
