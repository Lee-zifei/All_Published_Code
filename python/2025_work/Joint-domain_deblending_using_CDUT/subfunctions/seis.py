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
        S = 128 - N - L // 2  # 中段长度

        # R: 黑→白→(白到黄保持1)→(黄到红保持1)→红
        u1 = np.concatenate((
            np.zeros(N),
            np.linspace(0., 1, S),
            np.ones(L),
            np.ones(S),
            np.ones(N)
        ))

        # G: 黑→白→(白到黄保持1)→黄到橙(1→0.5)→橙到红(0.5→0)
        u2 = np.concatenate((
            np.zeros(N),
            np.linspace(0., 1, S),
            np.ones(L),
            np.linspace(1, 0.5, S),
            np.linspace(0.5, 0., N)
        ))

        # B: 黑→白→白到黄(1→0)→(黄到红保持0)→(结尾保持0)
        u3 = np.concatenate((
            np.zeros(N),
            np.linspace(0., 1, S),
            np.linspace(1, 0., L),
            np.zeros(S),
            np.zeros(N)
        ))

    M = np.column_stack((u1, u2, u3))
    # 创建自定义的colormap
    custom_colormap = mcolors.ListedColormap(M)
    return custom_colormap
  
  
  
  
 
