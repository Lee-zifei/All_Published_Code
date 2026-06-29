# -*- coding: utf-8 -*-
# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：env.py
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

import torch
from models import build_model
# import os
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
from data import build_test_loader
from seislet import patch_replication_callback, DataParallelWithCallback
from data.build_iter_dataset import patching_test
from models.NAFNet_arch import  NAFNet
# from seis import seis
import random
import time
# from metrics import snr
# from seislet import patch_replication_callback, DataParallelWithCallback

def parse_option():
	parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
	parser.add_argument('--cfg', type=str, default='./configs/WUDTnet.yaml', metavar="FILE", help='path to config file' )
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
	n1,n2=hyper.shape
	# hyper1 = np.zeros((n1,n2))
	# hyper2 = np.zeros((n1,n2))
	# hyper1 = hyper[0,:,:]
	# hyper2 = hyper[1,:,:]
	
	# hyper = turnone(hyper)
	# np.random.seed(seeds)
	# hyper2 = np.flip(hyper, axis=1)
	# delay=np.random.randint(-100,100,n2,dtype='int16')
	return hyper

def main(config,filed,SROW,SCOL):

 
	seed = 10
	blend_csg1  = input_data(filed,seed)
	ROW,COL = blend_csg1.shape
	## data input 3000*256->3000*96
	n1, n2 = blend_csg1.shape

	model = build_model(config)
	torch.backends.cudnn.benchmark = False

	checkpoint_dict = torch.load('./output/'+config.MODEL.NAME+'/default/ckpt_epoch_199.pth', map_location='cpu')['model']

	model.load_state_dict(checkpoint_dict, strict=True)
	model.cuda()
	model = DataParallelWithCallback(model)
	model.eval()

	blend_csg1_col = myimtocol(blend_csg1,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,1)
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
		# data_loader2=build_test_loader(config,blend_csg2_col)
		# for idx, (datas,_)  in enumerate(data_loader2):
		# 	datas=datas.cuda()

		# 	output2 = model(datas)####datasize,C H W
		# 	output2=output2.cpu().numpy()
		# 	if idx==0:
		# 		results2=output2
		# 	else:
		# 		results2=np.concatenate((results2,output2),axis=0)

	outputs1 = np.squeeze(results1)
	# outputs2 = np.squeeze(results2)
	# print(outputs1.shape)
 
	d1de1 = myimtocol(outputs1, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)
	# d2de1 = myimtocol(outputs2, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)

	return d1de1

def fig_train_data():
    clip = 0.01
    ii = 80
    # aelmp
    for iii in range(1):
        for j in range(1,5,1):
            if j == 1:
                data_type = 'a'
                ed_nm = 400
                aps = 0.2
            elif j == 2:
                data_type = 'n'
                ed_nm = 140
                aps = 0.25
            elif j == 3:
                data_type = 'm'
                ed_nm = 63
                aps = 0.5
            elif j == 4:
                data_type = 'p'
                ed_nm = 553
                aps = 0.5
            for i in range(1,2,1):
                print(i)
                if iii == 0:
                    targets = np.load('./data/2single_csg/Train_temp_02_1/target/'+str(data_type)+str(i)+'.npy')
                    inputs = np.load('./data/2single_csg/Train_temp_02_1/sample/'+str(data_type)+str(i)+'.npy')
                elif iii == 1:
                    targets = np.load('./data/2single_csg/Train_temp_02_1/target/'+str(data_type)+str(i)+'.npy')
                    inputs = np.load('./data/2single_csg/Train_temp_02_1/sample_08/'+str(data_type)+str(i)+'.npy')
                mm = seis(2)
                print (inputs.shape)
                # print (np.max(targets))
                fig = plt.figure(figsize=(16, 16),dpi=100)
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

                ax3 = fig.add_subplot(121)
                ax3.imshow(inputs, cmap=mm, vmax=clip, vmin=-clip,aspect=aps)
            
                ax3 = fig.add_subplot(122)
                ax3.imshow(targets, cmap=mm, vmax=clip, vmin=-clip,aspect=aps)
                
                plt.show()

def set_train_data():
	_, config = parse_option()
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config.GPU)	
	os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'	
	clip = 4
	times = 10
	mm = seis(2)

	# n1 = 3000
	# n2 =  564
	# n3 = 344
	clip = 0.0001
	ii = 80
	# aelmp
	files=os.listdir('./data/2single_csg/Train/target_y/')
	for f in files:
		filename=f.split('.')[0]
		targets = np.load('./data/2single_csg/Train/target_y/'+filename+'.npy')
		print('Is setting data:'+filename)
		d1 = targets
		d2 = np.zeros_like(targets)
		
		n1,n2 = d1.shape
		
		dtimme1 = random.randint(-500, 500)
		delay1 = np.ones((n2))*dtimme1

		dtimme2 = random.randint(100, 500)
		delay2 = np.ones((n2))*dtimme2

		dtimme3 = random.randint(500, 1000)
		delay3 = np.ones((n2))*dtimme3

		# dtimme3 = random.randint(300, 400)
		# delay3 = np.ones((n2))*dtimme3
		input1 = np.zeros_like(d1)
		d2 = d1[:, ::-1]*0.2
		
		max1 = np.max(np.abs(d1))
		input1 = d1+dither(d2,delay1)+dither(d2,delay2)+dither(d2,delay3)
		# +dither(d2,delay2)+dither(d2,delay3)
		input1 = input1/max1
		targets = targets/max1
		SROW = 16
		SCOL = 16
		d1de1 = main(config,input1,SROW,SCOL)
 
		np.save('./data/2single_csg/Train_temp_02_1/sample/'+filename+'.npy',d1de1)
		np.save('./data/2single_csg/Train_temp_02_1/target/'+filename+'.npy',targets)
  
		# fig = plt.figure(figsize=(16, 16),dpi=100)
		# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

		# ax3 = fig.add_subplot(131)
		# ax3.imshow(d1de1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)

		# ax3 = fig.add_subplot(132)
		# ax3.imshow(d1de1, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)

		# ax3 = fig.add_subplot(133)
		# ax3.imshow(targets, cmap=mm, vmax=clip, vmin=-clip,aspect=0.5)

		# plt.show()

if __name__ == '__main__':
	# set_train_data()
	fig_train_data()