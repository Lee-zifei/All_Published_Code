# -*- coding: utf-8 -*-
# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：turnone.py
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
# if 'macos' in platform.platform().lower(): 
#     myprog_path='/Users/lzf/Documents/cdut_zsh_group/python' 
# elif 'linux' in platform.platform().lower(): 
#     myprog_path='/media/lzf/Work/code/python' 
# else: 
#     myprog_path='L:\data\code\python' 
# sys.path.append(myprog_path)
# from subfunctions import *
from myprog import *
import torch
from models import build_model

# import os
import argparse
import numpy as np
from config import get_config
from collections import OrderedDict
import matplotlib.pyplot as plt
from seis import seis
from scipy import io
# from metrics import snr
# from seislet import patch_replication_callback, DataParallelWithCallback

def parse_option():
	parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
	parser.add_argument('--cfg', type=str, default='./configs/WUDT_STAnet.yaml', metavar="FILE", help='path to config file' )
	parser.add_argument('--batch-size', type=int, default=1, help="batch size for single GPU")
	parser.add_argument('--data-path', type=str, help='path to dataset')
	parser.add_argument('--resume', help='resume from checkpoint')
	parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
	parser.add_argument('--use-checkpoint', action='store_true',
			    help="whether to use gradient checkpointing to save memory")
	parser.add_argument('--output', default='output', type=str, metavar='PATH',
			    help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
	parser.add_argument('--tag', help='tag of experiment')
	parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
	parser.add_argument('--throughput', action='store_true', help='Test throughput only')
	parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
	parser.add_argument(
		"--opts",
		help="Modify config options by adding 'KEY VALUE' pairs. ",
		default=None,
		nargs='+',
	)
	args, unparsed = parser.parse_known_args()
	config = get_config(args)
	return args, config

def bin2npy(filename,n1,n2,n3):
    with open(filename, "rb") as fid:
        fid.seek(n1*n2*(n3-1)*4,0)
        signal = np.fromfile(fid,dtype=np.float32,count=n1*n2).reshape((n2,n1)).T
    return signal


def test_data(inputs):
	datasize=inputs.shape[0]
	for i in range(datasize):
		yield inputs[i,:,:,:]

def mutter_parabola_window(data, x1,x2,x0, y0):
    #  初始化矩阵
    n1,n2 = data.shape
    data1 = data
    y0 = n1-y0
    #  抛物线方程
    for i in range(n1):
        for j in range(n2): 
            if (i < -y0/ ((x0-x1)*(x0-x2)) *(j-x1)*(j-x2)+n1):
                data1[i, j] = 0
    # H = H[::-1,:]
    return data1

def find_x0(input,ns=20):
    nt,nx=input.shape
    patch = np.zeros((nt,nx//ns))
    n2 = nx//ns
    for i in range(ns):
        patch = patch+np.abs(input[:,i*n2:(i+1)*n2])

    psum = np.sum(patch,axis=0)  # 沿着列方向取最大值
    m = np.max(psum)              # 取最大值
    n = math.floor(np.argmax(psum))           # 取最大值的索引
    print(n)
    return n

def input_data(hyper,patch_size=1,xslide=1,x1=0,x2=0,x0=0,y0=0):
    patch_size = 96  # patch size
    xslide = 32      # space stride
    # 输入原始大小的数据
    Normalization = np.max(hyper)
    hyper = hyper/Normalization
    # 设置输入网络的部分
    n1,n2 = hyper.shape
    window = np.ones_like(hyper)
    hyper_mutter = mutter_parabola_window(hyper,x1,x2,x0,y0)
    window = mutter_parabola_window(hyper,x1,x2,x0,y0)
    
    hyper1 = np.zeros((n1,int(np.abs(x2-x1))))
    hyper1[:,:] = hyper_mutter[:,x1:x2]
    colhyper = myimtocol(hyper1, patch_size, patch_size, n1, n2, xslide, xslide, 1)
    return hyper1,colhyper,patch_size,xslide,Normalization,window

def output_data(hyper,pre_data,x1=0,x2=0,x0=0,y0=0):
    n1,n2 = hyper.shape
    outdata = np.zeros((n1,n2))
    # print(outdata.shape)
    n11,n22 = pre_data.shape
    outdata[:,x1:x2] = pre_data[:,:]
    outdata = mutter_parabola_window(outdata,x1,x2,x0,y0)
    return outdata

def main(config,times=0,hyper_y=None,patch_size=1,xslide=1,x1=0,x2=0,x0=0,y0=0):
 
    # filename = './data/patch1.npy'
    # hyper_y = np.load(filename)
    hyper, colhyper, patch_size, xslide, Normalization,window = input_data(hyper_y,patch_size,xslide,x1,x2,x0,y0)
    ## data input 3000*256->3000*96
    n1, n2 = hyper.shape
    colhyper_test = colhyper

    model = build_model(config)
    torch.backends.cudnn.benchmark = False
    # Load checkpoint
    model = torch.nn.DataParallel(model,device_ids=[0])
    model.cuda()

    model_parameters1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model1 parameters: {model_parameters1}")
    checkpoint = torch.load("./output/"+str(config.MODEL.NAME)+"/default/ckpt_epoch_599.pth", map_location='cpu', weights_only=False)
    checkpoint_dict = checkpoint['model']
    new_state_dict = OrderedDict()

    for k, v in checkpoint_dict.items():
        name = 'module.'+k  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)###need modify
    model.eval()

    colhyper = np.reshape(colhyper, colhyper.shape + (1,))
    colhyper = colhyper.transpose(0, 3, 1, 2)
    colhyper = torch.from_numpy(colhyper)

    with torch.no_grad():
        data_loader1=test_data(colhyper)

        for idx, datas in enumerate(data_loader1):
            datas=np.reshape(datas,(1,)+datas.shape)
            datas = datas.cuda()
            # datas = datas.cuda()

            output1 = model(datas)		####datasize,C H W
            out1=output1.cpu().numpy()
            if idx==0:
                results=out1
            else:
                results=np.concatenate((results,out1),axis=0)
    outputs1 = np.squeeze(results)
    modelname=config.MODEL.TYPE
    gwden = myimtocol(outputs1, patch_size,patch_size, n1, n2, xslide,xslide, 0)
    gwden = gwden*Normalization
	## data output 3000*96->3000*256
    out_data = output_data(hyper_y,gwden,x1,x2,x0,y0)
    np.save(modelname+'_result%d.npy'%times,out_data)
    return hyper_y,out_data,colhyper_test,outputs1,window

def main1(config,times=0,hyper_y=None,patch_size=1,xslide=1,x1=0,x2=0,x0=0,y0=0):
 
    n1,n2 = hyper_y.shape
    Normalization =  np.max(hyper_y)
    colhyper = myimtocol(hyper_y, patch_size, patch_size, n1, n2, xslide, xslide, 1)/Normalization
    ## data input 3000*256->3000*96
    n1, n2 = hyper.shape
    colhyper_test = colhyper

    model = build_model(config)
    torch.backends.cudnn.benchmark = False
    # Load checkpoint
    model = torch.nn.DataParallel(model,device_ids=[0])
    model.cuda()
    checkpoint = torch.load("./output/"+str(config.MODEL.NAME)+"/default/ckpt_epoch_599.pth", map_location='cpu', weights_only=False)
    checkpoint_dict = checkpoint['model']
    new_state_dict = OrderedDict()

    for k, v in checkpoint_dict.items():
        name = 'module.'+k  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)###need modify
    model.eval()

    colhyper = np.reshape(colhyper, colhyper.shape + (1,))
    colhyper = colhyper.transpose(0, 3, 1, 2)
    colhyper = torch.from_numpy(colhyper)

    with torch.no_grad():
        data_loader1=test_data(colhyper)

        for idx, datas in enumerate(data_loader1):
            datas=np.reshape(datas,(1,)+datas.shape)
            datas = datas.cuda()
            # datas = datas.cuda()

            output1 = model(datas)		####datasize,C H W
            out1=output1.cpu().numpy()
            if idx==0:
                results=out1
            else:
                results=np.concatenate((results,out1),axis=0)
    outputs1 = np.squeeze(results)
    modelname=config.MODEL.TYPE
    gwden = myimtocol(outputs1, patch_size,patch_size, n1, n2, xslide,xslide, 0)
    gwden = gwden*Normalization
	## data output 3000*96->3000*256
    # out_data = output_data(hyper_y,gwden,x1,x2,x0,y0)
    # np.save(modelname+'_result%d.npy'%times,out_data)
    return hyper_y,gwden,colhyper_test,outputs1

# def npy2bin(filename,data):
#     with open(filename,'wb') as f:
#         data.T.tofile(f)

  
if __name__ == '__main__':
    _, config = parse_option()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    n1 = 3000
    n2 = 256
    times = 10
    mm = seis(2)
    n3=1
    filename = './test/BGP/patch2.dat'
    hyper_y = bin2npy(filename,n1,n2*n3,1)

    for denoise in range(2):
        for i in range(8,9,1):
            hyper_in = hyper_y
            n1,n2 = hyper_in[:,:].shape
            
            x0 = 128
            print (x0)
            y0=500
            patch_size = 96  # patch size
            window_size = 128
            x1=x0-window_size//2
            x2=x0+window_size//2
            xslide = 16      # space stride
            clip  = np.max(hyper_in[:,:])/1000
            if denoise == 0:
                hyper,hyper_out,colhyper_test,outputs1,window = main(config,times,hyper_in,patch_size,xslide,x1,x2,x0,y0)
                res1 = hyper_in-hyper_out
            elif denoise == 1:
                hyper,hyper_out,colhyper_test,outputs1 = main1(config,times,hyper_in,patch_size,xslide,x1,x2,x0,y0)
                res2 = hyper_in-hyper_out
            
        print('source number'+str(i+1)+'......')
        # print(platform.platform().lower())
        print (hyper.shape)
        # np.save('hyper_out.npy', hyper_in-hyper_out)
    add = (res1+res2+res2*window*2)/2
        # ax1 = fig.add_subplot(3,20,(i+1))
        # # ax1.set_title('bpg ground roll patch')
        # ax1.imshow(hyper_in[:,:,i], cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)
        # # ax2 = fig.add_subplot(122)
        # # ax2.set_title('noise patch')
        # # ax2.imshow(gwden, cmap=mm, vmax=clip, vmin=-clip,aspect=0.1)

        # ax2 = fig.add_subplot(3,20,(i+1)+20)
        # # ax2.set_title('denoise patch')
        # ax2.imshow(hyper_in[:,:,i]-hyper_out[:,:,i], cmap=mm, vmax=clip, vmin=-clip,aspect=0.4)

        # ax3 = fig.add_subplot(3,20,(i+1)+40)
        # # ax2.set_title('denoise patch')
        # ax3.imshow(hyper_out[:,:,i], cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)



    fig = plt.figure(figsize=(16, 16),dpi=100)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    ax5 = fig.add_subplot(141)
    ax5.imshow(hyper_in,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)

    ax4 = fig.add_subplot(142)
    ax4.imshow(res1,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)
    
    ax6 = fig.add_subplot(143)
    ax6.imshow(res2,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)
    
    ax6 = fig.add_subplot(144)
    ax6.imshow(add,cmap=mm,vmax = clip,vmin = -clip,aspect=0.1)
    
    plt.show()
    filename1 = './figppt/hyper_in_p2.mat'
    filename2 = './figppt/wudt_STA_add_p2.mat'
    filename3 = './figppt/wudt_STA_res1_p1.mat'
    filename4 = './figppt/windows.mat'
    
    io.savemat(filename1, {'hyper_in_p2': hyper_in})
    io.savemat(filename2, {'wudt_STA_res1_p2': res1})
    io.savemat(filename3, {'wudt_STA_add_p1': add})
    io.savemat(filename4, {'window': window})
    
    # npy2bin(filenam1,hyper_in)
    # npy2bin(filenam2,add)
    # npy2bin(filenam3,res1)
