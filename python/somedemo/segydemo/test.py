# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：test.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Data：2024/01/16/01:15
#    Function：
#    
#    This program is free software: you can redistribute it and/or modify it 
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation, either version 3 of the License, or an later version.
#=================================================================================
import sys, os, platform
import numpy as np
import matplotlib.pyplot as plt
k1 = 101
k2 = 101
filename = "./data/data.dat"  # 请替换为你的文件路径
n1 = 1001
n2 = 401
n3 = 101
with open(filename, "rb") as fid:
    fid.seek(n1*n2*(k1-1)*4,0)
    signal_d1 = np.fromfile(fid,dtype=np.float32,count=n1*n2).reshape((n2,n1)).T
    plt.subplot(221)
    plt.imshow(signal_d1,cmap='gray')

    signal_d2 = np.zeros((n1,n3))
    j=0
    for i in range(n3):
        fid.seek(n1*n2*j*4+(k2-1)*n1*4,0)
        signal_d2[:,i] = np.fromfile(fid,dtype=np.float32,count=n1)
        j = j+1
    plt.subplot(222)
    plt.imshow(signal_d2,cmap='gray')
  
np.save('signal_d1.npy', signal_d1)
np.save('signal_d2.npy', signal_d2)

data1 = np.load('signal_d1.npy') 
data2 = np.load('signal_d2.npy') 
plt.subplot(223)
plt.imshow(data1,cmap='gray')
plt.subplot(224)
plt.imshow(data2,cmap='gray')

plt.show()
 

