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
import random
from config import get_config
from logger import create_logger
from models import build_model
from seislet import DataParallelWithCallback
from subfunctions import dither,snr,myimtocol,seis,read_d3,bin2npy_3d,mutter,read_d2,calculate_snr as snr,estimate_interference

def parse_option():
	parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
	parser.add_argument('--cfg', type=str, default='./configs/CDUTnet_2single_csg_0.4_temp.yaml', metavar="FILE", help='path to config file' )
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

def iter_in_csg(config,filed,net_model,SROW,SCOL):

    seed = 10
    blend_csg1 = filed
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

    outputs1 = np.squeeze(results1)
    # print(outputs1.shape)

    d1de1 = myimtocol(outputs1, config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,ROW,COL,SROW,SCOL,0)
    return d1de1

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
	files=os.listdir('./data/CSG/Train_04/target/')
	source_model = config.MODEL.NAME.replace('_temp', '')
	save_sample_dir = './data/CSG/Train_temp_04_1/sample'
	save_target_dir = './data/CSG/Train_temp_04_1/target'
	os.makedirs(save_sample_dir, exist_ok=True)
	os.makedirs(save_target_dir, exist_ok=True)
	for f in files:
		filename=f.split('.')[0]
		targets = np.load('./data/CSG/Train_04/target/'+filename+'.npy')
		print('Is setting data:'+filename)
		
		n1,n2 = targets.shape
		
		sp1 = np.random.choice([0.3, 0.5])
		sp2 = np.random.choice([0.3, 0.5])
		sp3 = np.random.choice([0.3, 0.5])


		
		inputs = np.zeros_like(targets)
		# print (targets.shape)
		mm = seis(2)
		target1 = targets
		target2 = targets[:,::-1]

		dtimme1 = random.randint(-200, 200)
		delay1 = np.ones((n2))*dtimme1

		dtimme2 = random.randint(300, 400)
		delay2 = np.ones((n2))*dtimme2

		dtimme3 = random.randint(500, 600)
		delay3 = np.ones((n2))*dtimme3
		
		input1 = target1+0.4*dither(target2,delay1)+0.4*dither(target2,delay2)+0.4*dither(target2,delay3)
		

		SROW = 16
		SCOL = 16
		for i in range(1):
			output = iter_in_csg(config, input1, source_model, config.DATA.SROW, config.DATA.SCOL)

			fig = plt.figure(figsize=(16, 16),dpi=100)
			plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

			ax3 = fig.add_subplot(121)
			ax3.imshow(input1, cmap=mm, vmax=clip, vmin=-clip,aspect=aps)

			# ax4 = fig.add_subplot(142)
			# ax4.imshow(inputs[1,:,:], cmap=mm, vmax=clip, vmin=-clip,aspect=aps)

			ax3 = fig.add_subplot(122)
			ax3.imshow(output, cmap=mm, vmax=clip, vmin=-clip,aspect=aps)  

			plt.show()
		np.save(os.path.join(save_sample_dir, filename + '.npy'), output.astype(np.float32))
		np.save(os.path.join(save_target_dir, filename + '.npy'), targets.astype(np.float32))

if __name__ == '__main__':
	set_train_data()
	# fig_train_data()
