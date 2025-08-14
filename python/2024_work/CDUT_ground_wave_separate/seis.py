# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：seis.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/05/20/
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def seis(input):
    N=40
    L=40
    if input == 1:
        u1 = np.concatenate((0.5 * np.ones(N), np.linspace(0.5, 1, 128-N), np.linspace(1, 0, 128-N), np.zeros(N)))
        u2 = np.concatenate((0.25 * np.ones(N), np.linspace(0.25, 1, 128-N), np.linspace(1, 0, 128-N), np.zeros(N)))
        u3 = np.concatenate((np.zeros(N), np.linspace(0, 1, 128-N), np.linspace(1, 0, 128-N), np.zeros(N)))  
    elif input == 2:
        u1 = np.concatenate((np.ones(N), np.linspace(1, 1, 128-N), np.linspace(1, 0, 128-N), np.zeros(N)))
        u2 = np.concatenate((np.zeros(N), np.linspace(0, 1, 128-N), np.linspace(1, 0, 128-N), np.zeros(N)))
        u3 = np.concatenate((np.zeros(N), np.linspace(0, 1, 128-N), np.linspace(1, 0, 128-N), np.zeros(N)))
    elif input == 3:
        u1 = np.concatenate((np.zeros(N), np.linspace(0., 1, 128 - N - L//2), np.ones(L), np.linspace(1, 0.5, 128 - L//2)))
        u2 = np.concatenate((np.zeros(N), np.linspace(0., 1, 128 - N - L//2), np.ones(L), np.linspace(1, 0., 128 - N - L//2), np.zeros(N)))
        u3 = np.concatenate((np.linspace(0.5, 1, 128 - L//2), np.ones(L), np.linspace(1, 0., 128 - N - L//2), np.zeros(N)))
    elif input == 4:
        N = 40   # 黑色前段长度
        L = 40   # 白色中心段长度
        n = 128  # 总长度
        u1 = np.concatenate((
            np.zeros(N),                                 # 黑色段
            np.linspace(0., 1., (n - 2*N - L)//2),      # 渐变到白色
            np.ones(L),                                  # 白色中心
            np.ones((n - 2*N - L)//2),                   # 黄到红段 (R=1)
            np.ones(N)                                   # 红色段
        ))

        # G通道
        u2 = np.concatenate((
            np.zeros(N),                                 # 黑色段
            np.linspace(0., 1., (n - 2*N - L)//2),      # 渐变到白色
            np.ones(L),                                  # 白色中心
            np.linspace(1., 0., (n - 2*N - L)//2),      # 黄到红
            np.zeros(N)                                  # 红色段
        ))

        # B通道
        u3 = np.concatenate((
            np.linspace(0., 1., (n - L)//2),            # 黑到白
            np.ones(L),                                  # 白色中心
            np.zeros((n - L)//2)                         # 黄到红 (B=0)
        ))

    M = np.column_stack((u1, u2, u3))
    # 创建自定义的colormap
    custom_colormap = mcolors.ListedColormap(M)
    return custom_colormap
  
  
  
  
 
