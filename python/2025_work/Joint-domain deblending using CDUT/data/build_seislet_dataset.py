
import torch
import numpy as np
import os
from myprog import myimtocol
#####need load
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

def build_load_folder(config):
    config.defrost()
    dataset_train, dataset_val = build_fromfolder(config)
    config.freeze()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val


class NewDataFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.image_path=root_dir + '/sample'
        self.label_path=root_dir + '/target'
        image_temp=os.listdir(self.image_path)
        self.image_list=[]
        self.label_list=[]
        for image in image_temp:
            if '.npy' in image:
                    self.image_list.append(image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        X_train = np.load(self.image_path+'/'+self.image_list[index]).astype(np.float32) ### C H W
        Y_train = np.load(self.label_path+'/'+self.image_list[index]).astype(np.float32) ### H W
        return X_train,Y_train

class DatasetFolder(torch.utils.data.Dataset):

    def __init__(self, root_dir,test=False):

        self.root_dir = root_dir
        self.image_path=root_dir + '/input'
        self.label_path=root_dir + '/output'
        image_temp=os.listdir(self.image_path)
        self.image_list=[]
        self.label_list=[]
        for image in image_temp:
            if '.npy' in image:
                if test == True:
                    if int(image.split('.')[0]) >= 1000:
                        self.image_list.append(image)
                else:
                    if int(image.split('.')[0]) < 1000:
                        self.image_list.append(image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        X_train = np.load(self.image_path+'/'+self.image_list[index]).astype(np.float32) ### C H W
        Y_train = np.load(self.label_path+'/'+self.image_list[index]).astype(np.float32) ### H W
        # X_train = np.reshape(X_train, (1,) + X_train.shape ) ### C H W
        # Y_train = np.reshape(Y_train, (1,) + Y_train.shape ) ### C H W
        # X_traint = torch.from_numpy(X_train)
        # Y_traint = torch.from_numpy(Y_train)
        return X_train,Y_train

def build_fromfolder(config):
    # dataset_train=DatasetFolder(config.DATA.DATA_PATH)
    # dataset_val=DatasetFolder(config.DATA.DATA_PATH,test=True)

    dataset_train = NewDataFolder(config.DATA.DATA_PATH)
    dataset_val = NewDataFolder(config.DATA.TEST_PATH)
    return dataset_train,dataset_val

def patch_single(config,inputs,targets=None):
    inputs=inputs.numpy()
    outputs=myimtocol(inputs[0,:,:],config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,1)
    B,H,W=outputs.shape
    outputs=outputs.reshape(B,1,H,W)
    if targets is not None:
        outs = myimtocol(targets[0,:,:],config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,1)
        B, H, W = outs.shape
        outs = outs.reshape(B, 1, H, W)
        Y_traint = torch.from_numpy(outs)
        X_traint = torch.from_numpy(outputs)
        return X_traint, Y_traint
    else:
        X_traint = torch.from_numpy(outputs)
        return X_traint

def patching(config,inputs,targets=None,vmax=None):
    inputs=inputs.numpy()
    B,C,H,W=inputs.shape
    seislet=inputs[0,1,:,:]
    # value= np.max(seislet)
    # seislet= np.tanh(seislet/value)
    if vmax is None:
        vmax=np.max(np.abs(seislet))/2
    seislet=seislet/vmax
    inputs[0,1,:,:]=seislet
    ##patching###
    for i in range(C):
        if i==0:
            outputs=myimtocol(inputs[0,i,:,:],config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,1)
            B,H,W=outputs.shape
            outputs=outputs.reshape(B,1,H,W)
        else:
            temp=myimtocol(inputs[0,i,:,:],config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,1)
            B,H,W=temp.shape
            temp=temp.reshape(B,1,H,W)
            outputs=np.concatenate((outputs,temp),axis=1)
    ###process outputs
    outputs[:,2,0,1]=vmax
    if targets is not None:
        targets=targets/vmax
        # targets = np.tanh(targets / value)
        outs = myimtocol(targets[0,:,:],config.DATA.IMG_SIZE,config.DATA.IMG_SIZE,config.DATA.ROW,config.DATA.COL,config.DATA.SROW,config.DATA.SCOL,1)
        B, H, W = outs.shape
        outs = outs.reshape(B, 1, H, W)
        Y_traint = torch.from_numpy(outs)
        X_traint = torch.from_numpy(outputs)
        return X_traint, Y_traint
    else:
        X_traint = torch.from_numpy(outputs)
        return X_traint, vmax





