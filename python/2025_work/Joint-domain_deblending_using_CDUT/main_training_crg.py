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
import argparse
import copy
import datetime
import functools
import logging
import math
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from torch import nn
from torch import optim as optim

try:
    from termcolor import colored
except ImportError:
    colored = None

try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
    from timm.scheduler.step_lr import StepLRScheduler
    from timm.scheduler.scheduler import Scheduler
except ImportError:
    CosineLRScheduler = None
    StepLRScheduler = None
    Scheduler = object

try:
    from apex import amp
except ImportError:
    amp = None


class ConfigNode(dict):
    def __init__(self, init_dict=None):
        super().__init__()
        init_dict = init_dict or {}
        for key, value in init_dict.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value):
        if isinstance(value, ConfigNode):
            return value
        if isinstance(value, dict):
            return ConfigNode(value)
        if isinstance(value, list):
            return [ConfigNode._wrap(item) for item in value]
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def clone(self):
        return copy.deepcopy(self)

    def freeze(self):
        return None

    def defrost(self):
        return None

    def merge_from_file(self, cfg_file):
        with open(cfg_file, "r") as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
        self.merge_from_dict(yaml_cfg)

    def merge_from_dict(self, other):
        for key, value in other.items():
            if (
                key in self
                and isinstance(self[key], ConfigNode)
                and isinstance(value, dict)
            ):
                self[key].merge_from_dict(value)
            else:
                self[key] = self._wrap(value)

    def merge_from_list(self, opts):
        if len(opts) % 2 != 0:
            raise ValueError("opts should be KEY VALUE pairs")
        for key, value in zip(opts[0::2], opts[1::2]):
            self._set_by_path(key.split("."), self._decode_value(value))

    def _set_by_path(self, keys, value):
        node = self
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], ConfigNode):
                node[key] = ConfigNode()
            node = node[key]
        node[keys[-1]] = self._wrap(value)

    @staticmethod
    def _decode_value(value):
        if not isinstance(value, str):
            return value
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            return value

    def to_dict(self):
        out = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [
                    item.to_dict() if isinstance(item, ConfigNode) else item
                    for item in value
                ]
            else:
                out[key] = value
        return out

    def dump(self):
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _default_config():
    cfg = ConfigNode()
    cfg.BASE = [""]

    cfg.DATA = ConfigNode()
    cfg.DATA.BATCH_SIZE = 64
    cfg.DATA.IMG_SIZE = 64
    cfg.DATA.PIN_MEMORY = True
    cfg.DATA.NUM_WORKERS = 1
    cfg.DATA.DATA_PATH = "data/NewData3/Train"
    cfg.DATA.TEST_PATH = "data/NewData3/Test"
    cfg.DATA.TEST_TYPE = "field"
    cfg.DATA.ROW = 1024
    cfg.DATA.SROW = 32
    cfg.DATA.COL = 512
    cfg.DATA.SCOL = 35

    cfg.GPU = [4, 5, 0]
    cfg.TESTF = False
    cfg.VAL = False
    cfg.DENOISE = False

    cfg.MODEL = ConfigNode()
    cfg.MODEL.TYPE = "dt"
    cfg.MODEL.NAME = "DTv2"
    cfg.MODEL.RESUME = ""
    cfg.MODEL.NUM_CLASSES = 1
    cfg.MODEL.DROP_RATE = 0.1
    cfg.MODEL.DROP_PATH_RATE = 0.1

    cfg.MODEL.DT = ConfigNode()
    cfg.MODEL.DT.IN_CHANS = 1
    cfg.MODEL.DT.EMBED_DIM = 64
    cfg.MODEL.DT.DEPTHS = [2, 2, 2, 2]
    cfg.MODEL.DT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.DT.WINDOW_SIZE = [8, 16, 32, 64]
    cfg.MODEL.DT.MLP_RATIO = 4.0
    cfg.MODEL.DT.QKV_BIAS = True
    cfg.MODEL.DT.QK_SCALE = 0
    cfg.MODEL.DT.PATCH_NORM = True

    cfg.MODEL.DT2 = ConfigNode()
    cfg.MODEL.DT2.IN_CHANS = 1
    cfg.MODEL.DT2.EMBED_DIM = 64
    cfg.MODEL.DT2.DEPTHS = [4, 7, 19, 8]
    cfg.MODEL.DT2.NUM_HEADS = [2, 3, 7, 10]
    cfg.MODEL.DT2.NITER = [1, 1, 1, 1]
    cfg.MODEL.DT2.STOKEN_SIZE = [8, 4, 1, 1]
    cfg.MODEL.DT2.MLP_RATIO = 4.0
    cfg.MODEL.DT2.QKV_BIAS = True
    cfg.MODEL.DT2.QK_SCALE = 0
    cfg.MODEL.DT2.PATCH_NORM = True

    cfg.TRAIN = ConfigNode()
    cfg.TRAIN.START_EPOCH = 0
    cfg.TRAIN.EPOCHS = 100
    cfg.TRAIN.WARMUP_EPOCHS = 4
    cfg.TRAIN.WEIGHT_DECAY = 1e-4
    cfg.TRAIN.BASE_LR = 2e-4
    cfg.TRAIN.WARMUP_LR = 1e-4
    cfg.TRAIN.MIN_LR = 1e-6
    cfg.TRAIN.CLIP_GRAD = 5
    cfg.TRAIN.AUTO_RESUME = True
    cfg.TRAIN.ACCUMULATION_STEPS = 0
    cfg.TRAIN.USE_CHECKPOINT = False
    cfg.TRAIN.GID = (2, 3, 4, 5)

    cfg.TRAIN.LR_SCHEDULER = ConfigNode()
    cfg.TRAIN.LR_SCHEDULER.NAME = "cosine"
    cfg.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
    cfg.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

    cfg.TRAIN.OPTIMIZER = ConfigNode()
    cfg.TRAIN.OPTIMIZER.NAME = "adamw"
    cfg.TRAIN.OPTIMIZER.EPS = 1e-8
    cfg.TRAIN.OPTIMIZER.BETAS = (0.9, 0.99)
    cfg.TRAIN.OPTIMIZER.MOMENTUM = 0.9

    cfg.TRAIN.LOSS = ConfigNode()
    cfg.TRAIN.LOSS.NAME = "MSE"

    cfg.TEST = ConfigNode()
    cfg.TEST.TYPE = "test3"
    cfg.TEST.MODE = False

    cfg.AMP_OPT_LEVEL = "O0"
    cfg.OUTPUT = ""
    cfg.TAG = "default"
    cfg.SAVE_FREQ = 1
    cfg.PRINT_FREQ = 1
    cfg.SEED = 3207
    cfg.EVAL_MODE = False
    cfg.THROUGHPUT_MODE = False
    cfg.LOCAL_RANK = 0
    return cfg


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader) or {}

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    config.LOCAL_RANK = args.local_rank if args.local_rank else 0
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()


def get_config(args):
    config = _default_config().clone()
    update_config(config, args)
    return config


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=""):
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.DEBUG)
    logger_obj.propagate = False
    logger_obj.handlers.clear()

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    if colored:
        color_fmt = (
            colored("[%(asctime)s %(name)s]", "green")
            + colored("(%(filename)s %(lineno)d)", "yellow")
            + ": %(levelname)s %(message)s"
        )
    else:
        color_fmt = fmt

    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger_obj.addHandler(console_handler)

    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger_obj.addHandler(file_handler)
    return logger_obj


class L1(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, x1, y):
        diff = torch.add(x1, -y)
        error = torch.sqrt(diff * diff + self.eps)
        return torch.mean(error)


class MIX2(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, x1, x2, y):
        diff = torch.add(x1, -y)
        error = torch.sqrt(diff * diff + self.eps)
        return 0.5 * torch.mean(error) + 0.5 * F.mse_loss(x2, y)


class MIX3(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, x1, x2, y):
        diff = torch.add(x1, -y)
        error = torch.sqrt(diff * diff + self.eps)
        return torch.mean(error) + F.mse_loss(x2, y)


class Mixloss(nn.Module):
    def __init__(self, weight1=0.5, weight2=0.5, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, pred, target):
        pred01 = pred[:, :, 0::2, :] / 2
        pred02 = pred[:, :, 1::2, :] / 2
        pred1 = pred01[:, :, :, 0::2]
        pred2 = pred02[:, :, :, 0::2]
        pred3 = pred01[:, :, :, 1::2]
        pred4 = pred02[:, :, :, 1::2]
        pred_hl = -pred1 - pred2 + pred3 + pred4
        pred_lh = -pred1 + pred2 - pred3 + pred4

        target01 = target[:, :, 0::2, :] / 2
        target02 = target[:, :, 1::2, :] / 2
        target1 = target01[:, :, :, 0::2]
        target2 = target02[:, :, :, 0::2]
        target3 = target01[:, :, :, 1::2]
        target4 = target02[:, :, :, 1::2]
        target_hl = -target1 - target2 + target3 + target4
        target_lh = -target1 + target2 - pred3 + target4
        return (
            F.mse_loss(pred, target)
            + self.weight1 * F.l1_loss(pred_hl, target_hl)
            + self.weight2 * F.l1_loss(pred_lh, target_lh)
        )


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def snr(output, target):
    eps = 1e-5
    results = 0
    batch_size = target.size(0)
    for i in range(batch_size):
        loss1 = torch.sum((target[i, 0, :, :] - output[i, 0, :, :]) ** 2) + eps
        loss2 = torch.sum(target[i, 0, :, :] ** 2) + eps
        results += 10 * torch.log(loss2 / loss1)
    return results / batch_size


def myimtocol(input1, rn1, rn2, n1, n2, tslide, xslide, f):
    if f == 1:
        n1, n2 = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        datasize = num1 * num2
        output1 = np.zeros((datasize, rn1, rn2), dtype="float32")
        for i in range(num2):
            for j in range(num1):
                if i < num2 - 1:
                    if j < num1 - 1:
                        output1[i * num1 + j, :, :] = input1[
                            j * tslide : j * tslide + rn1,
                            i * xslide : i * xslide + rn2,
                        ]
                    else:
                        output1[i * num1 + j, :, :] = input1[
                            n1 - rn1 : n1,
                            i * xslide : i * xslide + rn2,
                        ]
                else:
                    if j < num1 - 1:
                        output1[i * num1 + j, :, :] = input1[
                            j * tslide : j * tslide + rn1,
                            n2 - rn2 : n2,
                        ]
                    else:
                        output1[i * num1 + j, :, :] = input1[
                            n1 - rn1 : n1,
                            n2 - rn2 : n2,
                        ]
    else:
        datasize, rn1, rn2 = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        output1 = np.zeros((n1, n2), dtype="float32")
        weight = np.zeros((n1, n2), dtype="float32")
        one = np.ones((rn1, rn2), dtype="float32")
        for i in range(num2):
            for j in range(num1):
                patch = np.squeeze(input1[i * num1 + j, :, :])
                if i < num2 - 1:
                    if j < num1 - 1:
                        output1[j * tslide : j * tslide + rn1, i * xslide : i * xslide + rn2] += patch
                        weight[j * tslide : j * tslide + rn1, i * xslide : i * xslide + rn2] += one
                    else:
                        output1[n1 - rn1 : n1, i * xslide : i * xslide + rn2] += patch
                        weight[n1 - rn1 : n1, i * xslide : i * xslide + rn2] += one
                else:
                    if j < num1 - 1:
                        output1[j * tslide : j * tslide + rn1, n2 - rn2 : n2] += patch
                        weight[j * tslide : j * tslide + rn1, n2 - rn2 : n2] += one
                    else:
                        output1[n1 - rn1 : n1, n2 - rn2 : n2] += patch
                        weight[n1 - rn1 : n1, n2 - rn2 : n2] += one
        output1 = output1 / weight
    return output1


def dither(input1, delay_time):
    n1, n2 = input1.shape
    out1 = np.zeros((n1, n2), dtype="float32")
    n22 = len(delay_time)
    if n2 != n22:
        print("Error in size of delay time")
    for ix in range(n2):
        for it in range(n1):
            itt = it + int(delay_time[ix])
            if 0 <= itt < n1:
                out1[itt, ix] = input1[it, ix]
    return out1


class NewDataFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, model_type=""):
        self.root_dir = root_dir
        if model_type == "NAFnet":
            self.image_path = root_dir + "/swap"
        else:
            self.image_path = root_dir + "/sample"
        self.label_path = root_dir + "/target"
        image_temp = os.listdir(self.image_path)
        self.image_list = []
        self.label_list = []
        for image in image_temp:
            if ".npy" in image:
                self.image_list.append(image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        x_train = np.load(self.image_path + "/" + self.image_list[index]).astype(np.float32)
        y_train = np.load(self.label_path + "/" + self.image_list[index]).astype(np.float32)
        return x_train, y_train, self.image_list[index]


def build_fromfolder(config):
    dataset_train = NewDataFolder(config.DATA.DATA_PATH, config.MODEL.TYPE)
    dataset_val = NewDataFolder(config.DATA.TEST_PATH, config.MODEL.TYPE)
    return dataset_train, dataset_val


def build_load_folder(config):
    config.defrost()
    dataset_train, dataset_val = build_fromfolder(config)
    config.freeze()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_test_loader(config, inputs):
    inputs = np.reshape(inputs, (1,) + inputs.shape)
    inputs = inputs.transpose(1, 0, 2, 3)
    inputs = torch.from_numpy(inputs)
    dataset_test = torch.utils.data.TensorDataset(inputs, inputs)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    return data_loader_test


def patch_single(config, inputs, targets=None):
    inputs = inputs.numpy()
    outputs = myimtocol(
        inputs[0, :, :],
        config.DATA.IMG_SIZE,
        config.DATA.IMG_SIZE,
        config.DATA.ROW,
        config.DATA.COL,
        config.DATA.SROW,
        config.DATA.SCOL,
        1,
    )
    b, h, w = outputs.shape
    outputs = outputs.reshape(b, 1, h, w)
    if targets is not None:
        outs = myimtocol(
            targets[0, :, :],
            config.DATA.IMG_SIZE,
            config.DATA.IMG_SIZE,
            config.DATA.ROW,
            config.DATA.COL,
            config.DATA.SROW,
            config.DATA.SCOL,
            1,
        )
        b, h, w = outs.shape
        outs = outs.reshape(b, 1, h, w)
        return torch.from_numpy(outputs), torch.from_numpy(outs)
    return torch.from_numpy(outputs)


def patching_test(config, inputs):
    index = torch.zeros((2,), dtype=torch.float32)
    index[0] = 10
    index[1] = 10
    index = torch.cat([index, index], axis=0)
    index = torch.cat([index, index], axis=0)
    c, h, w = inputs.shape
    outputs = None
    for i in range(c):
        temp = myimtocol(inputs[i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, h, w, 32, 35, 1)
        b, patch_h, patch_w = temp.shape
        temp = temp.reshape(b, 1, patch_h, patch_w)
        outputs = temp if outputs is None else np.concatenate((outputs, temp), axis=0)
    return torch.from_numpy(outputs), index


def patching(config, inputs, name, targets=None, test=False, denoise=False):
    inputs = inputs.numpy()
    flag1 = flag2 = False
    index = torch.zeros((2,), dtype=torch.float32)
    names = name.split(".")[0]
    index[1] = int(names[1:])

    if names[0] == "p":
        index[0] = 5 if test else 0
        h, w, sh, sw = 1024, 354, 46, 58
        newinputs = np.zeros((1, 2, h, w), dtype=np.float32)
        newinputs[0, 0, :, :], newinputs[0, 1, :, :] = inputs[0, 0, :, :w], inputs[0, 1, :, :w]
        inputs = newinputs
        newtargets = np.zeros((1, 2, h, w), dtype=np.float32)
        newtargets[0, 0, :, :], newtargets[0, 1, :, :] = targets[0, 0, :, :w], targets[0, 1, :, :w]
        targets = newtargets
    elif names[0] == "m":
        index[0] = 6 if test else 1
        h, w, sh, sw = 725, 207, 32, 32
    elif names[0] == "l":
        index[0] = 7 if test else 2
        h, w, sh, sw = 900, 300, 40, 48
    elif names[0] == "a":
        index[0] = 8 if test else 3
        h, w, sh, sw = 1200, 120, 56, 48
        flag1 = True
    elif names[0] == "e":
        index[0] = 9 if test else 4
        h, w, sh, sw = 1200, 151, 27, 48
    else:
        raise ValueError(f"Unsupported sample name prefix: {name}")

    index = torch.cat([index, index], axis=0)
    index = torch.cat([index, index], axis=0)
    _, channels, h, w = inputs.shape

    outputs = None
    for i in range(channels):
        temp = myimtocol(inputs[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, h, w, sh, sw, 1)
        b, patch_h, patch_w = temp.shape
        temp = temp.reshape(b, 1, patch_h, patch_w)
        outputs = temp if outputs is None else np.concatenate((outputs, temp), axis=0)

    if targets is None:
        return torch.from_numpy(outputs), index

    outs = None
    for i in range(channels):
        temp = myimtocol(targets[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, h, w, sh, sw, 1)
        b, patch_h, patch_w = temp.shape
        temp = temp.reshape(b, 1, patch_h, patch_w)
        outs = temp if outs is None else np.concatenate((outs, temp), axis=0)

    if denoise:
        if flag1 and not test:
            abspath = os.getcwd()
            delay = np.load(abspath + "/" + config.DATA.DATA_PATH + "/delay/" + name)
            obsers = np.load(abspath + "/" + config.DATA.DATA_PATH + "/sample/" + name)
            obser1, obser2 = obsers[0, :, :], obsers[1, :, :]
            ntemp1 = obser1 - dither(inputs[0, 1, :, :], delay) * np.random.uniform(0.75, 0.91)
            ntemp2 = obser2 - dither(inputs[0, 0, :, :], -delay) * np.random.uniform(0.75, 0.91)
            ntemp1 = myimtocol(ntemp1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, h, w, sh, sw, 1)
            ntemp2 = myimtocol(ntemp2, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, h, w, sh, sw, 1)
            out2 = np.concatenate((ntemp1, ntemp2), axis=0)
            b, patch_h, patch_w = out2.shape
            outputs = out2.reshape(b, 1, patch_h, patch_w)
            outputs = np.concatenate((outputs, outputs), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
        if flag2 and not test:
            outputs = np.concatenate((outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = outs[:264, :, :, :]
    else:
        if flag1 and not test:
            outputs = np.concatenate((outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = np.concatenate((outs, outs), axis=0)
            outs = outs[:264, :, :, :]
        if flag2 and not test:
            outputs2 = copy.deepcopy(outputs)
            for i in range(outputs2.shape[0]):
                outputs2[i, :, :, :] *= np.random.uniform(0.95, 1.05)
            outputs = np.concatenate((outputs, outputs2), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = outs[:264, :, :, :]

    b, _, _, _ = outs.shape
    if config.DATA.BATCH_SIZE < b:
        random_slice = np.random.choice(b, config.DATA.BATCH_SIZE, replace=False)
        y_train = torch.from_numpy(outs[random_slice, :, :, :])
        x_train = torch.from_numpy(outputs[random_slice, :, :, :])
    else:
        y_train = torch.from_numpy(outs[:256, :, :, :])
        x_train = torch.from_numpy(outputs[:256, :, :, :])
    return x_train, y_train, index


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "DnCNN":
        from models.Dn_CNN import DnCNN

        return DnCNN()
    if model_type == "Unet":
        from models.unet import Unet

        return Unet()
    if model_type == "NakaUnet":
        from models.nakaunet import NakaUnet

        return NakaUnet()
    if model_type == "Hydranetv2":
        from models.hydranetv2 import Hydranetv2

        return Hydranetv2(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.DT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DT.EMBED_DIM,
            depths=config.MODEL.DT.DEPTHS,
            num_heads=config.MODEL.DT.NUM_HEADS,
            window_size=config.MODEL.DT.WINDOW_SIZE,
            mlp_ratio=2,
            qkv_bias=config.MODEL.DT.QKV_BIAS,
            qk_scale=config.MODEL.DT.QK_SCALE,
            drop_rate=0.0,
            drop_path_rate=0.1,
            patch_norm=config.MODEL.DT.PATCH_NORM,
            config=config,
        )
    if model_type == "WUDTnet":
        from models.wudtnet import WUDTnet

        return WUDTnet(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.DT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DT.EMBED_DIM,
            depths=config.MODEL.DT.DEPTHS,
            num_heads=config.MODEL.DT.NUM_HEADS,
            window_size=config.MODEL.DT.WINDOW_SIZE,
            mlp_ratio=2,
            qkv_bias=config.MODEL.DT.QKV_BIAS,
            qk_scale=config.MODEL.DT.QK_SCALE,
            drop_rate=0.0,
            drop_path_rate=0.1,
            patch_norm=config.MODEL.DT.PATCH_NORM,
            config=config,
        )
    if model_type == "DT":
        from models.dt import DeblendingTransformer

        return DeblendingTransformer(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.DT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DT.EMBED_DIM,
            depths=config.MODEL.DT.DEPTHS,
            num_heads=config.MODEL.DT.NUM_HEADS,
            window_size=config.MODEL.DT.WINDOW_SIZE,
            mlp_ratio=config.MODEL.DT.MLP_RATIO,
            qkv_bias=config.MODEL.DT.QKV_BIAS,
            qk_scale=config.MODEL.DT.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.DT.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
    if model_type == "swir":
        from models.network_swinir import SwinIR

        return SwinIR(
            upscale=1,
            in_chans=1,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler=None,
            resi_connection="3conv",
            talking_heads=False,
            use_attn_fn="softmax",
            head_scale=False,
            on_attn=False,
            use_mask=True,
            mask_ratio1=75,
            mask_ratio2=75,
            mask_is_diff=False,
            type="stand",
        )
    if model_type == "restormer":
        from models.restormer import Restormer

        return Restormer(
            inp_channels=1,
            out_channels=1,
            dim=32,
            num_blocks=[2, 3, 3, 4],
            num_refinement_blocks=2,
            ffn_expansion_factor=2,
        )
    if model_type == "WUDT_STAnet":
        from models.wudt_STAnet import WUDT_STAnet

        return WUDT_STAnet(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.DT2.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DT2.EMBED_DIM,
            depths=config.MODEL.DT2.DEPTHS,
            num_heads=config.MODEL.DT2.NUM_HEADS,
            n_iter=config.MODEL.DT2.NITER,
            stoken_size=config.MODEL.DT2.STOKEN_SIZE,
            projection=1024,
            mlp_ratio=config.MODEL.DT2.MLP_RATIO,
            qkv_bias=config.MODEL.DT2.QKV_BIAS,
            qk_scale=config.MODEL.DT2.QK_SCALE,
            drop_rate=0,
            drop_path_rate=0.6,
            layerscale=[False, False, True],
            init_values=1e-6,
            config=config,
        )
    raise NotImplementedError(f"Unkown model: {model_type}")


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(config, model):
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else {}
    skip_keywords = (
        model.no_weight_decay_keywords()
        if hasattr(model, "no_weight_decay_keywords")
        else {}
    )
    parameters = set_weight_decay(model, skip, skip_keywords)
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    if opt_lower == "sgd":
        return optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    if opt_lower == "adamw":
        return optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=tuple(config.TRAIN.OPTIMIZER.BETAS),
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
    raise NotImplementedError(f"Unknown optimizer: {config.TRAIN.OPTIMIZER.NAME}")


class _FallbackScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_values = [group["lr"] for group in optimizer.param_groups]
        self.last_t = 0

    def update_groups(self, values):
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value

    def step_update(self, num_updates):
        self.last_t = num_updates
        values = self.get_update_values(num_updates)
        if values is not None:
            self.update_groups(values)

    def state_dict(self):
        return {"base_values": self.base_values, "last_t": self.last_t}

    def load_state_dict(self, state_dict):
        self.last_t = state_dict.get("last_t", state_dict.get("t", self.last_t))
        self.base_values = state_dict.get("base_values", self.base_values)


class FallbackCosineLRScheduler(_FallbackScheduler):
    def __init__(self, optimizer, t_initial, lr_min, warmup_lr_init=0.0, warmup_t=0):
        super().__init__(optimizer)
        self.t_initial = max(1, t_initial)
        self.lr_min = lr_min
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        if self.warmup_t:
            self.update_groups([warmup_lr_init for _ in self.base_values])

    def get_update_values(self, t):
        if self.warmup_t and t < self.warmup_t:
            return [
                self.warmup_lr_init + (base - self.warmup_lr_init) * t / self.warmup_t
                for base in self.base_values
            ]
        t = min(max(t - self.warmup_t, 0), max(1, self.t_initial - self.warmup_t))
        total_t = max(1, self.t_initial - self.warmup_t)
        return [
            self.lr_min + 0.5 * (base - self.lr_min) * (1 + math.cos(math.pi * t / total_t))
            for base in self.base_values
        ]


class LinearLRScheduler(Scheduler if Scheduler is not object else _FallbackScheduler):
    def __init__(
        self,
        optimizer,
        t_initial,
        lr_min_rate,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ):
        if Scheduler is object:
            _FallbackScheduler.__init__(self, optimizer)
        else:
            super().__init__(
                optimizer,
                param_group_field="lr",
                noise_range_t=noise_range_t,
                noise_pct=noise_pct,
                noise_std=noise_std,
                noise_seed=noise_seed,
                initialize=initialize,
            )
        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            self.update_groups([self.warmup_lr_init for _ in self.base_values])
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * s for s in self.warmup_steps]
        t = t - self.warmup_t
        total_t = self.t_initial - self.warmup_t
        return [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]

    def get_epoch_values(self, epoch):
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates):
        return self._get_lr(num_updates) if not self.t_in_epochs else None


class DecayLRScheduler(Scheduler if Scheduler is not object else _FallbackScheduler):
    def __init__(
        self,
        optimizer,
        decay_t,
        min_lr,
        basic_lr=5e-4,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ):
        if Scheduler is object:
            _FallbackScheduler.__init__(self, optimizer)
        else:
            super().__init__(
                optimizer,
                param_group_field="lr",
                noise_range_t=noise_range_t,
                noise_pct=noise_pct,
                noise_std=noise_std,
                noise_seed=noise_seed,
                initialize=initialize,
            )
        self.min_lr = min_lr
        self.decay_t = decay_t
        self.basic_lr = basic_lr
        self.t_in_epochs = t_in_epochs

    def _get_lr(self, t):
        if t < self.decay_t:
            return [v - (self.basic_lr - self.min_lr) * t / self.decay_t for v in self.base_values]
        return [self.min_lr for _ in self.base_values]

    def get_epoch_values(self, epoch):
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates):
        return self._get_lr(num_updates) if not self.t_in_epochs else None


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    scheduler_name = config.TRAIN.LR_SCHEDULER.NAME
    if scheduler_name == "cosine":
        if CosineLRScheduler is not None:
            return CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=config.TRAIN.MIN_LR,
                warmup_lr_init=config.TRAIN.WARMUP_LR,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )
        return FallbackCosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
        )
    if scheduler_name == "linear":
        return LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    if scheduler_name == "step":
        if StepLRScheduler is not None:
            return StepLRScheduler(
                optimizer,
                decay_t=decay_steps,
                decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                warmup_lr_init=config.TRAIN.WARMUP_LR,
                warmup_t=warmup_steps,
                t_in_epochs=False,
            )
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, decay_steps),
            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
        )
    if scheduler_name == "decay":
        return DecayLRScheduler(
            optimizer,
            decay_t=decay_steps,
            min_lr=config.TRAIN.MIN_LR,
            basic_lr=config.TRAIN.BASE_LR,
            t_in_epochs=False,
        )
    if scheduler_name == "None":
        return None
    raise NotImplementedError(f"Unknown lr scheduler: {scheduler_name}")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger_obj):
    logger_obj.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")

    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    logger_obj.info(msg)
    max_accuracy = 0.0
    min_loss = 1.0
    if (
        not config.EVAL_MODE
        and isinstance(checkpoint, dict)
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        if (
            amp is not None
            and "amp" in checkpoint
            and config.AMP_OPT_LEVEL != "O0"
            and checkpoint["config"].AMP_OPT_LEVEL != "O0"
        ):
            amp.load_state_dict(checkpoint["amp"])
        logger_obj.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        max_accuracy = checkpoint.get("max_accuracy", max_accuracy)
        min_loss = checkpoint.get("min_loss", min_loss)

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, min_loss


def save_checkpoint(config, epoch, model, max_accuracy, min_loss, optimizer, lr_scheduler, logger_obj):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "max_accuracy": max_accuracy,
        "min_loss": min_loss,
        "epoch": epoch,
        "config": config,
    }
    if amp is not None and config.AMP_OPT_LEVEL != "O0":
        save_state["amp"] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger_obj.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger_obj.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        return latest_checkpoint
    return None


def parse_option():
    parser = argparse.ArgumentParser(
        "CRG training script with local helper functions bundled",
        add_help=False,
    )
    parser.add_argument(
        "--cfg",
        type=str,
        metavar="FILE",
        default="configs/CDUTnet_2single_crg_1_iter_5.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs="+",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")
    parser.add_argument("--local_rank", type=int, required=False, help="local rank for DistributedDataParallel")
    args, _ = parser.parse_known_args()
    config = get_config(args)
    return args, config


def build_criterion(config):
    if config.TRAIN.LOSS.NAME == "MSE":
        return torch.nn.MSELoss()
    if config.DENOISE or config.MODEL.TYPE == "NAFnet":
        return L1()
    return torch.nn.MSELoss()


def maybe_model_forward(model, samples, index, denoise):
    if denoise:
        return model(samples, index)
    return model(samples)


def main(config, pre_train=False):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_load_folder(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    if pre_train:
        model_dict = model.state_dict()
        pretrained_dict2 = torch.load("WUDTnet.pth", map_location="cpu")["model"]
        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            new_state_dict[k] = v
        for k, v in pretrained_dict2.items():
            if "denoise" not in k:
                new_state_dict[k] = v
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=True)
        for name, param in model.named_parameters():
            if "denoise" not in name:
                param.requires_grad = False

    optimizer = build_optimizer(config, model)
    if len(config.GPU) > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    model_without_ddp = model.module if len(config.GPU) > 1 else model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_accuracy = 0.0
    min_loss = 1.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    criterion = build_criterion(config)
    if config.MODEL.RESUME:
        max_accuracy, min_loss = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, loss = validate(config, data_loader_val, model, criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        criterion = build_criterion(config)
        if config.MODEL.TYPE == "NAFnet":
            criterion = L1()

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        if (
            (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1))
            and "preprocess" not in config.MODEL.NAME
        ):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, min_loss, optimizer, lr_scheduler, logger)

        acc1, loss = validate(config, data_loader_val, model, criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss:.8f}")
        min_loss = min(min_loss, loss)
        if min_loss == loss and "preprocess" not in config.MODEL.NAME:
            torch.save(model.state_dict(), f"{config.MODEL.NAME}.pth")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max snr: {max_accuracy:.3f}")
        logger.info(f"Min loss: {min_loss:.8f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    snr_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (images, labels, names) in enumerate(data_loader):
        samples, targets, index = patching(config, images, names[0], targets=labels, denoise=config.DENOISE)
        if config.DENOISE:
            index = index.cuda()
        samples = samples.cuda()
        targets = targets.cuda()

        output = maybe_model_forward(model, samples, index, config.DENOISE)
        loss = criterion(output, targets)
        snr1 = snr(output, targets)

        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "step_update"):
                lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                lr_scheduler.step()

        snr_meter.update(snr1.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(float(grad_norm))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t"
                f"snr {snr_meter.val:.4f} ({snr_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})"
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, labels, names) in enumerate(data_loader):
        samples, targets, index = patching(
            config,
            images,
            names[0],
            targets=labels,
            test=True,
            denoise=config.DENOISE,
        )
        if config.DENOISE:
            index = index.cuda(non_blocking=True)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = maybe_model_forward(model, samples, index, config.DENOISE)
        loss = criterion(output, targets)
        acc1 = snr(output, targets)

        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss:  {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t"
                f"snr:  {acc1_meter.val:.4f} ({acc1_meter.avg:.4f})"
            )
    logger.info(f" * snr: {acc1_meter.avg:.3f} ")
    return acc1_meter.avg, loss_meter.avg


logger = logging.getLogger("main_training_crg")


if __name__ == "__main__":
    _, config = parse_option()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in config.GPU)

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())

    pretrain = False
    main(config, pretrain)
