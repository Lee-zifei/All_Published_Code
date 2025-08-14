# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：pptfig.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/10/02/
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
from seis import seis
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

def myConv2d(images, in_channels, out_channels, kernel_size, stride, padding, weights=None, bias=None):
    if weights is None:
        weights = torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1])
    if bias is None:
        bias = torch.zeros(out_channels)
    n, c, w, h = images.shape
    # 给原始图片加上padding
    # new_image = torch.zeros(n, c, w + 2 * padding, h + 2 * padding)
    images = images.clone()
    images = torch.cat((torch.zeros(n, c, padding, h), images), 2)
    images = torch.cat((images, torch.zeros(n, c, padding, h)), 2)
    images = torch.cat((torch.zeros(n, c, w + 2 * padding, padding), images), 3)
    images = torch.cat((images, torch.zeros(n, c, w + 2 * padding, padding)), 3)
    n, c, w, h = images.shape
    output = []
    # 循环batch_size
    for i, im in enumerate(images):
        imout = []
        # 循环feature map count, 也就是输出通道数
        for j in range(out_channels):
            feature_map = []
            row = 0
            # 下面两层循环为使用kernel滑动窗口遍历输入图片
            while row + kernel_size[0] <= h:
                row_feat_map = []
                col = 0
                while col + kernel_size[1] <= w:
                    # 卷积计算每个点的值，此处为了方便理解定义了channels,其实可以直接定义point=0，然后进行累加，最后再加上偏置
                    channels = [0 for x in range(c)]
                    for ch in range(c):
                        for y in range(kernel_size[0]):
                            for x in range(kernel_size[1]):
                                channels[ch] += im[ch][row + y][col + x] * weights[j][ch][y][x]
                    point = sum(channels) + bias[j]
                    row_feat_map.append(point)
                    col += stride[1]
                feature_map.append(row_feat_map)
                row += stride[0]
            imout.append(feature_map)
        output.append(imout)
    return torch.Tensor(output)
 
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

if __name__ == "__main__":
    # 测试参数
    image_w, image_h = 7,7
    in_channels = 1
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1,1)
    padding = 1
    
    n1 = 3000
    n2 = 256
    n3 = 1
    # 输入图片与网络权重
    filename = './test/BGP/patch3.dat'
    data = bin2npy_3d(filename,n1,n2*n3,1)
    # image = data.float() 
    image = data

    image = torch.from_numpy(image).float() 
    image = image.reshape(1, 1, n1, n2)
    print (image.shape)
    weights = torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1])
    bias = torch.zeros(out_channels)
 
    # pytorch运算结果
    net = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    net.weight = nn.Parameter(weights)
    net.bias = nn.Parameter(bias)
    net.eval()
    output = net(image)
    print(output.shape)
 
    output1 = dwt_init(image)
    # # 自己实现的结果
    # output = myConv2d(image, in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, weights=weights, bias=bias)
    # print(output)
    
    
    image = image.detach().numpy()
    output = output.detach().numpy()
    
    clip = 0.0001
    clip1 = 1
    # mm = seis(4)
    mm = seis(2)
    
    asp = n2/n1
    fig = plt.figure(figsize=(16, 16),dpi=100)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    
    ax5 = fig.add_subplot(151)
    ax5.imshow(image[0,0,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax4 = fig.add_subplot(152)
    ax4.imshow(output[0,0,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    
    ax4 = fig.add_subplot(153)
    ax4.imshow(output[0,1,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    
    ax4 = fig.add_subplot(154)
    ax4.imshow(output[0,2,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    
    ax4 = fig.add_subplot(155)
    ax4.imshow(output[0,3,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    
    plt.show()