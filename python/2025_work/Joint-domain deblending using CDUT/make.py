import numpy as np
import os
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

import time
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


def main(config,filed,SROW,SCOL):

 
	seed = 10
	blend_csg1, blend_csg2 = input_data(filed,seed)
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

            
if __name__ == '__main__':
	_, config = parse_option()
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'
	clip = 0.01
	times = 10
	mm = seis(2)
 
	datatype = config.DATA.TEST_PATH
	files=os.listdir(datatype+'/sample')
	os.makedirs(datatype+'/swap',exist_ok=True)

	for f in files:

		data_type=f.split('.')[0][0]
		dataname = datatype+'/sample/'+f
		dataname_hyper = datatype+'/target/'+f
  
		outdata = np.zeros_like(np.load(dataname_hyper))
  
		hyper1 = np.load(dataname_hyper)[0,:,:]
		hyper2 = np.load(dataname_hyper)[1,:,:]
	
		ble = np.load(dataname).astype(np.float32)
		outname = datatype+'/swap/'+f
		print("Is setting data:"+outname)
	
		bl1 = np.load(dataname)[0,:,:]
		bl2 = np.load(dataname)[1,:,:]

		if data_type == 'p':
			ROW, COL, SROW, SCOL = 1024, 354, 46, 58
		elif data_type == 'm':
			ROW, COL, SROW, SCOL = 725, 207, 32, 32
		elif data_type == 'l':
			ROW, COL, SROW, SCOL = 900, 300, 40, 48
		elif data_type == 'a':
			ROW, COL, SROW, SCOL = 1200, 120, 56, 48
		elif data_type == 'e':
			ROW, COL, SROW, SCOL = 1200, 151, 27, 48

		d1de1,d2de2 = main(config,ble,SROW,SCOL)
		outdata[0,:,:] = d1de1
		outdata[1,:,:] = d2de2
	
		np.save(outname,outdata)
		# fig = plt.figure(figsize=(16, 16),dpi=100)
		# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
		# asp = 0.5

		# ax1 = fig.add_subplot(141)
		# ax1.imshow(np.load(dataname_hyper)[0,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# # ax2 = fig.add_subplot(162)
		# # ax2.imshow(hyper2,cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# ax3 = fig.add_subplot(142)
		# ax3.imshow(np.load(dataname_hyper)[1,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# ax5 = fig.add_subplot(143)
		# ax5.imshow(outdata[0,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)

		# ax6 = fig.add_subplot(144)
		# ax6.imshow(outdata[1,:,:],cmap=mm,vmax = clip,vmin = -clip,aspect=asp)
	
		# plt.show()