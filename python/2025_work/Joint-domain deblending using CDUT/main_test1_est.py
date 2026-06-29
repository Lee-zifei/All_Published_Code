import numpy as np
import os
# -*- coding: utf-8 -*-
# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：main_crg.py
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
from subfunctions import dither,snr,myimtocol,seis,read_d3,bin2npy_3d,mutter,read_d2,calculate_snr as snr,estimate_interference

import torch
from logger import create_logger
from models import build_model
# import os
# from metrics import AverageMeter,snr
import argparse
import numpy as np
from config import get_config
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
from models import build_model
import os
import argparse
import numpy as np
from config import get_config
from collections import OrderedDict
from seislet import patch_replication_callback, DataParallelWithCallback
from models.NAFNet_arch import  NAFNet
import time

def parse_option():
	parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
	parser.add_argument('--cfg', type=str, default='./configs/CDUTnet_2single_crg_1.yaml', metavar="FILE", help='path to config file' )
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

def input_data(filed,seeds):
	hyper = filed
	B,n1,n2=hyper.shape
	hyper1 = np.zeros((n1,n2))
	hyper2 = np.zeros((n1,n2))
	hyper1 = hyper[0,:,:]
	hyper2 = hyper[1,:,:]
	
	# hyper = turnone(hyper)
	# np.random.seed(seeds)
	# hyper2 = np.flip(hyper, axis=1)
	# delay=np.random.randint(-100,100,n2,dtype='int16')
	return hyper1,hyper2

def build_test_loader(config,inputs):

    inputs = np.reshape(inputs, (1,)+inputs.shape)
    inputs = inputs.transpose(1,0,2,3)
    inputs = torch.from_numpy(inputs)

    dataset_test= torch.utils.data.TensorDataset(inputs,inputs)
    data_loader_test= torch.utils.data.DataLoader(
        dataset_test,
        batch_size=256,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return data_loader_test

def iter_in_crg(config,filed,net_model,SROW,SCOL):

    seed = 10
    blend_csg1, blend_csg2 = input_data(filed,seed)
    ROW,COL = blend_csg1.shape
    ## data input 3000*256->3000*96
    n1, n2 = blend_csg1.shape

    model = build_model(config)
    torch.backends.cudnn.benchmark = False

    checkpoint_dict = torch.load('./output/'+net_model+'/default/ckpt_epoch_199.pth', map_location='cpu')['model']
    model.load_state_dict(checkpoint_dict, strict=True)
    model.cuda()
    model = DataParallelWithCallback(model)
    model.eval()

    blend_csg1_col = myimtocol(blend_csg1, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)
    blend_csg2_col = myimtocol(blend_csg2, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)

    with torch.no_grad():
        data_loader1 = build_test_loader(config,blend_csg1_col)
        for idx, (datas,_) in enumerate(data_loader1):
            datas = datas.cuda()
            output1 = model(datas)####datasize,C H W
            output1 = output1.cpu().numpy()
            if idx==0:
                results1=output1
            else:
                results1=np.concatenate((results1,output1),axis=0)
        data_loader2=build_test_loader(config,blend_csg2_col)
        for idx, (datas,_)  in enumerate(data_loader2):
            datas=datas.cuda()

            output2 = model(datas)####datasize,C H W
            output2=output2.cpu().numpy()
            if idx==0:
                results2=output2
            else:
                results2=np.concatenate((results2,output2),axis=0)

    outputs1 = np.squeeze(results1)
    outputs2 = np.squeeze(results2)
    # print(outputs1.shape)

    d1de1 = myimtocol(outputs1, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)
    d2de1 = myimtocol(outputs2, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)

    return d1de1,d2de1

def iter_in_csg(config,filed,net_model,SROW,SCOL):

    seed = 10
    blend_csg1, blend_csg2_y = input_data(filed,seed)
    blend_csg2 = blend_csg2_y[:,::-1]
    ROW,COL = blend_csg1.shape
    ## data input 3000*256->3000*96
    n1, n2 = blend_csg1.shape

    model = build_model(config)
    torch.backends.cudnn.benchmark = False

    checkpoint_dict = torch.load('./output/'+net_model+'/default/ckpt_epoch_199.pth', map_location='cpu')['model']

    model.load_state_dict(checkpoint_dict, strict=True)
    model.cuda()
    model = DataParallelWithCallback(model)
    model.eval()

    blend_csg1_col = myimtocol(blend_csg1,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)
    blend_csg2_col = myimtocol(blend_csg2,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)
    # blend_csg2_col = myimtocol(blend_csg2, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)

    with torch.no_grad():
        data_loader1 = build_test_loader(config,blend_csg1_col)
        for idx, (datas,_) in enumerate(data_loader1):
            datas = datas.cuda()
            output1 = model(datas)####datasize,C H W
            output1 = output1.cpu().numpy()
            if idx==0:
                results1=output1
            else:
                results1=np.concatenate((results1,output1),axis=0)
        data_loader2=build_test_loader(config,blend_csg2_col)
        for idx, (datas,_)  in enumerate(data_loader2):
            datas=datas.cuda()

            output2 = model(datas)####datasize,C H W
            output2=output2.cpu().numpy()
            if idx==0:
                results2=output2
            else:
                results2=np.concatenate((results2,output2),axis=0)

    outputs1 = np.squeeze(results1)
    outputs2 = np.squeeze(results2)
    # print(outputs1.shape)

    d1de1 = myimtocol(outputs1, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)
    d2de1_f = myimtocol(outputs2, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)
    d2de1 = d2de1_f[:,::-1]
    return d1de1,d2de1

if __name__ == '__main__':
    _, config = parse_option()
    datapath = '/home/lzf/code/python/2024_work/CSG_souece_different_deblending/test/viking/temp'
    logger = create_logger(output_dir=datapath,name=f"{config.MODEL.TYPE}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    clip = 5
    times = 20
    mm = seis(2)

    n1 = 1500
    n2 = 120
    n3 = 1001
    dt=6e-3;
    
    file1_single = datapath+'/../blend_data/d1.dat'
    file2_single = datapath+'/../blend_data/d2.dat'

    file1_ble = datapath+'/../blend_data/d1b.dat'
    file2_ble = datapath+'/../blend_data/d2b.dat'
    
    file1_t1 = datapath+'/../blend_data/t1.dat'
    file2_t2 = datapath+'/../blend_data/t2.dat'

    t1d = bin2npy_3d(file1_t1,1,1,n3)
    t1 = t1d.reshape(-1)[:n3]

    t2d = bin2npy_3d(file2_t2,1,1,n3)
    t2 = t2d.reshape(-1)[:n3]

    deblend1 = np.zeros((n1,n2,n3))
    deblend2 = np.zeros((n1,n2,n3))

    d_temp = np.zeros((2,n1,n3))
    dom = ['CSG','CRG','CRG','CRG','CRG'
           ]
    net_mod = [     'CDUTnet_2single_csg_0.4',
                    'CDUTnet_2single_crg',
                    'CDUTnet_2single_crg_1_iter',
                    'CDUTnet_2single_crg_1_iter_5'
                    ]
    SROW = 32
    SCOL = 32


    for epoch in range(4):
        print(epoch)
        domain = dom[epoch]
        net_model = net_mod[epoch]

        if domain == 'CRG':
            niter = 5
            for i in range(n2):
                trace = i
                k1 = 3.8
                tr_1 = 0+k1*trace
                tr_2 = 500-k1*trace

                
                if epoch==0:
                    d1_new = np.zeros((n1,n3))
                    d2_new = np.zeros((n1,n3))
                else:
                    d1_new = np.load('./d1_new_3d_est_viking.npy')[:,i,:]
                    d2_new = np.load('./d2_new_3d_est_viking.npy')[:,i,:]
                    d1_new = mutter(d1_new,0,tr_1,0)
                    d2_new = mutter(d2_new,0,tr_2,0)

                for iter in range(niter):
                    
                    # delay = delay_2d[trace,:]
                    d1_blend = read_d3(file1_ble,[n1,n2,n3],trace)
                    d2_blend = read_d3(file2_ble,[n1,n2,n3],trace)
 
                    # print(delay_old[0])
                    d1 = read_d3(file1_single,[n1,n2,n3],trace)
                    d2 = read_d3(file2_single,[n1,n2,n3],trace)

                    
                    temp_input1 = np.zeros((n1,n3))
                    temp_input2 = np.zeros((n1,n3))

                    temp_input1 = (d1_blend-estimate_interference(d2_new,t2,t1,dt))
                    temp_input2 = (d2_blend-estimate_interference(d1_new,t1,t2,dt))
                            
                    temp_input1 = mutter(temp_input1,0,tr_1,0)
                    temp_input2 = mutter(temp_input2,0,tr_2,0)
                    
                    max1 = np.max(np.abs(d1_blend))
                    
                    temp_input=np.zeros((2,n1,n3))
                    
                    temp_input[0,:,:] = temp_input1
                    temp_input[1,:,:] = temp_input2

                    d1de1,d2de2 = iter_in_crg(config,temp_input/max1,net_model,SROW,SCOL)

                    snr1 =  snr(d1de1*max1, d1)
                    snr2 =  snr(d2de2*max1, d2)

                    n = 0.5
                    A = 1.5

                    d1_new = d1de1*max1
                    d2_new = d2de2*max1

                    d1_neww = temp_input1
                    d2_neww = d2de2*max1

                    d1_neww = mutter(d1_neww,0,tr_1,0)
                    d2_neww = mutter(d2_neww,0,tr_2,0)
                    logger.info(f"正在进行第{str(iter+1)}次迭代，当前处理到{domain}第{str(i+1)}/{str(n2)}道,SNR1={'%.2f'%snr1},SNR2={'%.2f'%snr2}")
                deblend1[:,i,:] = d1_neww
                deblend2[:,i,:] = d2_neww
            np.save('./d1_new_3d_est_viking.npy',deblend1)
            np.save('./d2_new_3d_est_viking.npy',deblend2)
        if domain == 'CSG':
            for i in range(n3):
                trace = i
                niter = 0

                for iter in range(niter+1):
                    
                    # delay = delay_2d[trace,:]
                    if epoch==0:
                        d1_blend = read_d2(file1_ble,[n1,n2,n3],trace)
                        d2_blend = read_d2(file2_ble,[n1,n2,n3],trace)
                    else:
                        d1_blend = np.load('./d1_new_3d_est_viking.npy')[:,:,i]
                        d2_blend = np.load('./d2_new_3d_est_viking.npy')[:,:,i]

                    # print(delay_old[0])
                    d1 = read_d2(file1_single,[n1,n2,n3],trace)
                    d2 = read_d2(file2_single,[n1,n2,n3],trace)

                    temp_input1 = np.zeros((n1,n2))
                    temp_input2 = np.zeros((n1,n2))

                    temp_input1 = d1_blend
                    temp_input2 = d2_blend
                    if epoch == 0:
                        max1 = np.max(np.abs(d1_blend))
                    else:
                        max1 =1
                    
                    temp_input=np.zeros((2,n1,n2))
                    temp_input11 = np.zeros((2,n1,n2))
                    
                    temp_input[0,:,:] = temp_input1
                    temp_input[1,:,:] = temp_input2

                    d1de11,d2de21 = iter_in_csg(config,temp_input/max1,net_model,SROW,SCOL)
                    
                    temp_input11[0,:,:] = d1de11
                    temp_input11[1,:,:] = d2de21
                    
                    d1de1,d2de2 = iter_in_csg(config,temp_input11,net_model+'_temp',SROW,SCOL)
                    
                    snr1 =  snr(d1de1*max1, d1)
                    snr2 =  snr(d2de2*max1, d2)

                    d1_neww = d1de1*max1
                    d2_neww = d2de2*max1

                    logger.info(f"正在进行第{str(iter+1)}次迭代，当前处理到{domain}第{str(i+1)}/{str(n3)}道,SNR1={'%.2f'%snr1},SNR2={'%.2f'%snr2}")
                    
                    # fig = plt.figure(figsize=(16, 16),dpi=100)
                    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
                    # asp = 0.6

                    # ax3 = fig.add_subplot(121)
                    # ax3.imshow(d1_neww,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)


                    # ax3 = fig.add_subplot(122)
                    # ax3.imshow(d2_neww,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)


                    # plt.show()
                deblend1[:,:,i] = d1_neww
                deblend2[:,:,i] = d2_neww
            np.save('./d1_new_3d_est_viking.npy',deblend1)
            np.save('./d2_new_3d_est_viking.npy',deblend2)
    fig = plt.figure(figsize=(16, 16),dpi=100)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    asp = 0.6
    # print(d1_blend.max)
    ax1 = fig.add_subplot(241)
    ax1.imshow(d1,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    # ax2 = fig.add_subplot(162)
    # ax2.imshow(hyper2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax3 = fig.add_subplot(242)
    ax3.imshow(d1_blend,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax5 = fig.add_subplot(243)
    ax5.imshow(d1_neww,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax6 = fig.add_subplot(244)
    ax6.imshow(d1-d1_neww,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax1 = fig.add_subplot(245)
    ax1.imshow(d2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    # ax2 = fig.add_subplot(162)
    # ax2.imshow(hyper2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax3 = fig.add_subplot(246)
    ax3.imshow(d2_blend,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax5 = fig.add_subplot(247)
    ax5.imshow(d2_neww,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    ax6 = fig.add_subplot(248)
    ax6.imshow(d2-d2_neww,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    # ax6 = fig.add_subplot(166)
    # ax6.imshow(d1-temp_input[0,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    # ax5 = fig.add_subplot(256)
    # ax5.imshow(d2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    # ax6 = fig.add_subplot(257)
    # ax6.imshow(d2_blend,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    # ax6 = fig.add_subplot(258)
    # ax6.imshow(d2_new,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    # ax6 = fig.add_subplot(259)
    # ax6.imshow(temp_input[1,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    # ax6 = fig.add_subplot(2,5,10)
    # ax6.imshow(d2-temp_input[1,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    plt.show()

    # fig1 = plt.figure(figsize=(16, 16),dpi=100)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    # asp = 0.5

    # ax6 = fig1.add_subplot(1,2,1)
    # ax6.imshow((d1_new+temp_input[0,:,:])/2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

    # ax6 = fig1.add_subplot(1,2,2)
    # ax6.imshow((d2_new+temp_input[1,:,:])/2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)


    # plt.show()


    # fig = plt.figure(figsize=(16, 16),dpi=100)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    # asp = 0.5

    # ax6 = fig.add_subplot(111)
    # ax6.imshow((d1_new+temp_input[0,:,:])/2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
    # plt.show()
