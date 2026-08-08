# ==================================================================================
#    Copyright (C) 2025 Chengdu University of Technology.
#    Copyright (C) 2025 Zifei LI.
#    
#    Filename：normalize_patches.py
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



def normalize_patches(patches):
    """
    对每个 patch 独立归一化，使其均值为0，最大绝对值为1。
    同时保存每个 patch 的 mean 和 max_abs 用于反归一化。
    
    参数:
        patches: numpy array, shape (n, h, w)

    返回:
        norm_patches: numpy array, shape (n, h, w)
        stats: dict，包含 keys: 'mean', 'max_abs'
    """
    n = patches.shape[0]
    norm_patches = np.empty_like(patches, dtype=np.float32)
    means = np.empty(n, dtype=np.float32)
    max_abs_vals = np.empty(n, dtype=np.float32)

    for i in range(n):
        patch = patches[i]
        mean = np.mean(patch)
        centered = patch - mean
        max_abs = np.max(np.abs(centered)) + 1e-16  # 防止除0
        norm_patch = centered / max_abs

        norm_patches[i] = norm_patch
        means[i] = mean
        max_abs_vals[i] = max_abs

    stats = {'mean': means, 'max_abs': max_abs_vals}
    return norm_patches, stats
