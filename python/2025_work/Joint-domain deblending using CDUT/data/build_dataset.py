import torch
import numpy as np
import os

class DatasetFolder(torch.utils.data.Dataset):

    def __init__(self, root_dir,test=False):

        self.root_dir = root_dir
        self.image_path=root_dir+'/image'
        self.label_path=root_dir + '/label'
        image_temp=os.listdir(self.image_path)
        self.image_list=[]
        self.label_list=[]
        for image in image_temp:
            if '.npy' in image:
                if test == True:
                    if int(image.split('.')[0]) >= 100000:
                        self.image_list.append(image)
                else:
                    if int(image.split('.')[0]) < 100000:
                        self.image_list.append(image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        X_train = np.load(self.image_path+'/'+self.image_list[index]).astype(np.float32) ### C H W
        Y_train = np.load(self.label_path+'/'+self.image_list[index]).astype(np.float32) ### H W
        # X_train = np.reshape(X_train, (1,) + X_train.shape ) ### C H W
        Y_train = np.reshape(Y_train, (1,) + Y_train.shape ) ### C H W
        X_traint = torch.from_numpy(X_train)
        Y_traint = torch.from_numpy(Y_train)
        return X_traint,Y_traint

def build_fromfolder(config):

    dataset_train=DatasetFolder(config.DATA.DATA_PATH)

    dataset_val=DatasetFolder(config.DATA.DATA_PATH,test=True)

    return dataset_train,dataset_val


