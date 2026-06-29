import torch
import numpy as np
import os
from myprog import myimtocol
import copy

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
        self.image_path = root_dir + '/sample'
        self.label_path = root_dir + '/target'
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
        X_train = X_train.reshape((1,) + X_train.shape)
        Y_train = Y_train.reshape((1,) + Y_train.shape)
        return X_train,Y_train

def build_fromfolder(config):
    dataset_train = NewDataFolder(config.DATA.DATA_PATH)
    dataset_val = NewDataFolder(config.DATA.TEST_PATH)
    return dataset_train,dataset_val





