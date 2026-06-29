import os.path
import torch
from sklearn.model_selection import train_test_split
from myprog import *
import torch.optim as optim


def build_loader(config):
    config.defrost()
    dataset_train, dataset_val = build_dataset(config=config)
    config.freeze()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
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


#### need to modify
def build_dataset(config):
    # n1 = 300  # the training volumn size   nz
    # n2 = 1024  # the training volumn sie    nt
    # n3 = 512  # the training volumn size   nx
    # patch_rows = 128  # patch size
    # patch_cols = 128
    # tslide = 128  # time stride
    # xslide = 128  # space stride
    # trainsize = 0.8
    # seed = 20240607
    n1 = 300  # the training volumn size   nz
    n2 = 1024  # the training volumn sie    nt
    n3 = 512  # the training volumn size   nx
    patch_rows = 128  # patch size
    patch_cols = 128
    tslide = 128  # time stride
    xslide = 128  # space stride
    trainsize = 0.8
    seed = 20240607
    
    ###need to modify later
    if config.DATA.FINE_TUNE:
        filename3 = 'data/Data/fine-tune/tars.npy';
        targets = np.load(filename3)

        filename4 = 'data/Data/fine-tune/inpts.npy';
        inputs = np.load(filename4)
    else:
        if os.path.exists('data/gwave/inputs2_combine_seletc_patch_training.npy'):
            targets = np.load('data/gwave/targets2_noise_combine_select_patch_training.npy')
            inputs = np.load('data/gwave/inputs2_combine_seletc_patch_training.npy')
            # targets = inputs-targets1
        else:
            filename3 = 'data/gwave/data1ss.npy';
            oup1 = np.load(filename3)[:,:512*n1]

            filename4 = 'data/gwave/data2ss.npy';
            oup2 = np.load(filename4)[:,:512*n1]

            delay1 = np.random.randint(-300, 300, n1 * n3, dtype='int16')

            inp1 = oup1 + dither(oup2, delay1)
            inp2 = oup2 + dither(oup1, -delay1)
            # #############
            oupt1 = myimtocol2(oup1, patch_rows, patch_rows, n1, n2, n3, tslide, xslide, 1)
            oupt2 = myimtocol2(oup2, patch_rows, patch_rows, n1, n2, n3, tslide, xslide, 1)
            targets = np.concatenate((oupt1, oupt2), axis=0)

            inpt1 = myimtocol2(inp1, patch_rows, patch_rows, n1, n2, n3, tslide, xslide, 1)
            inpt2 = myimtocol2(inp2, patch_rows, patch_rows, n1, n2, n3, tslide, xslide, 1)
            inputs = np.concatenate((inpt1, inpt2), axis=0)

            np.save('data/Data/targets.npy', targets)
            np.save('data/Data/inputs.npy', inputs)

    ################################################
    X_train, X_dev, Y_train, Y_dev = train_test_split(inputs, targets, train_size=trainsize, random_state=seed)
    X_train = np.reshape(X_train, X_train.shape + (1,))
    X_train = X_train.transpose(0, 3, 1, 2)
    Y_train = np.reshape(Y_train, Y_train.shape + (1,))
    Y_train = Y_train.transpose(0, 3, 1, 2)
    X_dev = np.reshape(X_dev, X_dev.shape + (1,))
    X_dev = X_dev.transpose(0, 3, 1, 2)
    Y_dev = np.reshape(Y_dev, Y_dev.shape + (1,))
    Y_dev = Y_dev.transpose(0, 3, 1, 2)

    ###(1,128,128,1)   B H W C    to  (1,1,128,128) B C H W

    X_traint = torch.from_numpy(X_train)
    Y_traint = torch.from_numpy(Y_train)
    dataset_train = torch.utils.data.TensorDataset(X_traint, Y_traint)
    X_devt = torch.from_numpy(X_dev)
    Y_devt = torch.from_numpy(Y_dev)
    dataset_val = torch.utils.data.TensorDataset(X_devt, Y_devt)

    return dataset_train, dataset_val
