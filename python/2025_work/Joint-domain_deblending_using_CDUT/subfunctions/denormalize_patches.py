# ==================================================================================
#    Copyright (C) 2025 Chengdu University of Technology.
#    Copyright (C) 2025 Zifei LI.
#    
#    Filename：denormalize_patches.py
#    Author：Zifei LI
#    Institute：Chengdu University of Technology
#    Email：2024010196@stu.cdut.edu.cn
#    Work：2025/05/11/
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
def denormalize_patches(norm_patches, stats):
    """
    使用保存的 stats 对归一化数据进行反归一化。
    
    参数:
        norm_patches: numpy array, shape (n, h, w)
        stats: dict，包含 keys: 'mean', 'max_abs'

    返回:
        orig_patches: numpy array, shape (n, h, w)
    """
    means = stats['mean']
    max_abs_vals = stats['max_abs']
    n = norm_patches.shape[0]
    orig_patches = np.empty_like(norm_patches, dtype=np.float32)

    for i in range(n):
        orig_patches[i] = norm_patches[i] * max_abs_vals[i] + means[i]

    return orig_patches
