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
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import get_config
from logger import create_logger
from models import build_model
from seislet import DataParallelWithCallback
from subfunctions import dither,snr,myimtocol,seis,read_d3,bin2npy_3d,mutter,read_d2,calculate_snr as snr,estimate_interference

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

def generate_random_sequence(n3, t1, t2):
    # 生成 n3 个随机整数，范围在 [t1, t2]
    random_sequence = np.random.randint(t1, t2 + 1, size=n3)
    return random_sequence

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

def input_data(filed,seeds):
	hyper = filed
	# hyper1 = np.zeros((n1,n2))
	# hyper2 = np.zeros((n1,n2))
	# hyper1 = hyper[0,:,:]
	# hyper2 = hyper[1,:,:]
	
	# hyper = turnone(hyper)
	# np.random.seed(seeds)
	# hyper2 = np.flip(hyper, axis=1)
	# delay=np.random.randint(-100,100,n2,dtype='int16')
	return hyper

def main(config,style,filed,SROW,SCOL):

 
	seed = 10
	blend_csg1, blend_csg2 = input_data(filed,seed)
	ROW,COL = blend_csg1.shape
	## data input 3000*256->3000*96
	n1, n2 = blend_csg1.shape

	model = build_model(config)
	torch.backends.cudnn.benchmark = False

	if style=='CSG':
		checkpoint_dict = torch.load('./'+style+'/output/'+config.MODEL.TYPE+'_'+style+'_randonsample/default/ckpt_epoch_199.pth', map_location='cpu')['model']
	elif style == 'CRG':
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


def set_train_data():
    _, config = parse_option()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config.GPU)	
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'	
    clip = 4
    times = 10
    mm = seis(2)

    # n1 = 3000
    # n2 =  564
    # n3 = 344
    clip = 0.05
    ii = 80
    # aelmp
    aps = 0.8
    files=os.listdir('./data/CRG/Train/target/')
    for f in files:
        filename=f.split('.')[0]
        targets = np.load('./data/CRG/Train/target/'+filename+'.npy')
        print('Is setting data:'+filename)
        
        b,n1,n2 = targets.shape
        sp = np.random.choice([0.2, 0.5])
        # windows = generate_random_01_matrix(n1,n2,sp)
        inputs = np.zeros_like(targets)
        # print (targets.shape)
        mm = seis(2)
        target1 = targets[0,:,:]
        target2 = targets[1,:,:]
    
        d1 = targets[0,:,:]
        d2 = targets[1,:,:]
        # print(d1.shape)
        n1,n2 = d1.shape
        t1=-50
        t2= 50
        delay1 = generate_random_sequence(n2, t1, t2)
        n1,n2 = d1.shape

        # dtimme3 = random.randint(300, 400)
        # delay3 = np.ones((n2))*dtimme3
        d1b = np.zeros_like(d1)
        d2b = np.zeros_like(d2)
    
        d1b = d1+dither(d2,delay1)
        d2b = d2+dither(d1,-delay1)

        input1=np.zeros((2,n1,n2))
        

        SROW = 16
        SCOL = 16
        d1_new = np.zeros_like(d1)
        d2_new = np.zeros_like(d1)
        for i in range(1):
            d1t = d1b-dither(d2_new,delay1)
            d2t = d2b-dither(d1_new,-delay1)
            input1[0,:,:] = d1t
            input1[1,:,:] = d2t
            maxin = np.max(np.abs(input1))
            d1de1,d2de2 = main(config,'CRG',input1/maxin,config.DATA.SROW,config.DATA.SCOL)

            output=np.zeros((2,n1,n2))
            d1_new = d1de1*maxin
            d2_new = d2de2*maxin
            
            output[0,:,:] =  d1b-dither(d2_new,delay1)
            output[1,:,:] =  d2b-dither(d1_new,-delay1)
            
        np.save('./data/CRG/Train_iter/sample/'+filename+'.npy',inputs)


if __name__ == '__main__':
	set_train_data()
	# fig_train_data()