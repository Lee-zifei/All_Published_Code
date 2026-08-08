# -*- coding: utf-8 -*-
# ==================================================================================
#    Copyright (C) 2024 Chengdu University of Technology.
#    Copyright (C) 2024 Zifei Li.
#    
#    Filename：main_csg.py
#    Author：Zifei Li
#    Institute：Chengdu University of Technology
#    Email：202005050218@stu.cdut.edu.cn
#    Work：2024/08/19/
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
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

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

from torch import nn
from torch import optim as optim
from torch.nn import functional as F
from yacs.config import CfgNode as CN

from models.wudt_STAnet import WUDT_STAnet

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 64
_C.DATA.IMG_SIZE = 64
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 1
_C.GPU = []
_C.DATA.DATA_PATH = "data/NewData3/Train"
_C.DATA.TEST_PATH = "data/NewData3/Test"
_C.DATA.TEST_TYPE = "field"
_C.DATA.ROW = 1024
_C.DATA.SROW = 32
_C.DATA.COL = 512
_C.DATA.SCOL = 35

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = "dt"
_C.MODEL.NAME = "DTv2"
_C.MODEL.RESUME = ""
_C.MODEL.NUM_CLASSES = 1
_C.MODEL.DROP_RATE = 0.1
_C.MODEL.DROP_PATH_RATE = 0.1

_C.MODEL.DT = CN()
_C.MODEL.DT.IN_CHANS = 1
_C.MODEL.DT.EMBED_DIM = 64
_C.MODEL.DT.DEPTHS = [2, 2, 2, 2]
_C.MODEL.DT.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.DT.WINDOW_SIZE = [8, 16, 32, 64]
_C.MODEL.DT.MLP_RATIO = 4.0
_C.MODEL.DT.QKV_BIAS = True
_C.MODEL.DT.QK_SCALE = 0
_C.MODEL.DT.PATCH_NORM = True
_C.TESTF = False
_C.VAL = False
_C.DENOISE = False

_C.MODEL.DT2 = CN()
_C.MODEL.DT2.IN_CHANS = 1
_C.MODEL.DT2.EMBED_DIM = 64
_C.MODEL.DT2.DEPTHS = [4, 7, 19, 8]
_C.MODEL.DT2.NUM_HEADS = [2, 3, 7, 10]
_C.MODEL.DT2.NITER = [1, 1, 1, 1]
_C.MODEL.DT2.STOKEN_SIZE = [8, 4, 1, 1]
_C.MODEL.DT2.MLP_RATIO = 4.0
_C.MODEL.DT2.QKV_BIAS = True
_C.MODEL.DT2.QK_SCALE = 0
_C.MODEL.DT2.PATCH_NORM = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 4
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 2e-4
_C.TRAIN.WARMUP_LR = 1e-4
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.CLIP_GRAD = 5
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.GID = (2, 3, 4, 5)

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.99)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = "MSE"
_C.TRAIN.LOSS.AGC_WINDOW = 31
_C.TRAIN.LOSS.AGC_AXIS = -2
_C.TRAIN.LOSS.AGC_EPS = 1e-6
_C.TRAIN.LOSS.AGC_MAX_GAIN = 0.0

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.TYPE = "test3"
_C.TEST.MODE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP_OPT_LEVEL = "O0"
_C.OUTPUT = ""
_C.TAG = "default"
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 1
_C.SEED = 3207
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

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
    config = _C.clone()
    update_config(config, args)
    return config


class L1(nn.Module):
    """L1 Charbonnier loss."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, x1, y):
        diff = torch.add(x1, -y)
        error = torch.sqrt(diff * diff + self.eps)
        return torch.mean(error)


class L1_agc(nn.Module):
    """L1 Charbonnier loss after applying target-driven AGC."""

    def __init__(self, agc_window=31, agc_axis=-2, eps=1e-6, gain_eps=1e-6, max_gain=0.0):
        super().__init__()
        if agc_window < 1:
            raise ValueError("agc_window must be >= 1")
        if agc_window % 2 == 0:
            agc_window += 1

        self.eps = eps
        self.gain_eps = gain_eps
        self.agc_window = agc_window
        self.agc_axis = agc_axis
        self.max_gain = max_gain
        self.last_gain = None

        agc_kernel = torch.ones(1, 1, self.agc_window, dtype=torch.float32) / float(self.agc_window)
        self.register_buffer("agc_kernel", agc_kernel)

    def compute_gain(self, target):
        axis = self.agc_axis if self.agc_axis >= 0 else target.dim() + self.agc_axis
        if axis < 0 or axis >= target.dim():
            raise ValueError(f"agc_axis={self.agc_axis} is out of range for target dim={target.dim()}")

        moved_target = torch.movedim(target, axis, -1)
        moved_shape = moved_target.shape
        moved_target = moved_target.reshape(-1, 1, moved_shape[-1])

        agc_kernel = self.agc_kernel.to(device=target.device, dtype=target.dtype)
        local_power = F.conv1d(moved_target * moved_target, agc_kernel, padding=self.agc_window // 2)
        gain = torch.rsqrt(local_power + self.gain_eps)

        if self.max_gain > 0:
            gain = torch.clamp(gain, max=self.max_gain)

        gain = gain.reshape(moved_shape)
        gain = torch.movedim(gain, -1, axis)
        return gain.detach()

    def apply_agc(self, data, gain):
        return data * gain

    def forward(self, X1, Y):
        gain = self.compute_gain(Y)
        self.last_gain = gain

        X1_agc = self.apply_agc(X1, gain)
        Y_agc = self.apply_agc(Y, gain)

        diff = torch.add(X1_agc, -Y_agc)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


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
                            j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2
                        ]
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, i * xslide:i * xslide + rn2]
                else:
                    if j < num1 - 1:
                        output1[i * num1 + j, :, :] = input1[j * tslide:j * tslide + rn1, n2 - rn2:n2]
                    else:
                        output1[i * num1 + j, :, :] = input1[n1 - rn1:n1, n2 - rn2:n2]
    else:
        [datasize, rn1, rn2] = input1.shape
        num1 = int(np.floor((n1 - rn1) / tslide) + 1 + (np.mod(n1 - rn1, tslide) != 0))
        num2 = int(np.floor((n2 - rn2) / xslide) + 1 + (np.mod(n2 - rn2, xslide) != 0))
        output1 = np.zeros((n1, n2), dtype="float32")
        weight = np.zeros((n1, n2), dtype="float32")
        one = np.ones((rn1, rn2), dtype="float32")

        for i in range(num2):
            for j in range(num1):
                if i < num2 - 1:
                    if j < num1 - 1:
                        output1[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] += np.squeeze(
                            input1[i * num1 + j, :, :]
                        )
                        weight[j * tslide:j * tslide + rn1, i * xslide:i * xslide + rn2] += one
                    else:
                        output1[n1 - rn1:n1, i * xslide:i * xslide + rn2] += np.squeeze(
                            input1[i * num1 + j, :, :]
                        )
                        weight[n1 - rn1:n1, i * xslide:i * xslide + rn2] += one
                else:
                    if j < num1 - 1:
                        output1[j * tslide:j * tslide + rn1, n2 - rn2:n2] += np.squeeze(input1[i * num1 + j, :, :])
                        weight[j * tslide:j * tslide + rn1, n2 - rn2:n2] += one
                    else:
                        output1[n1 - rn1:n1, n2 - rn2:n2] += np.squeeze(input1[i * num1 + j, :, :])
                        weight[n1 - rn1:n1, n2 - rn2:n2] += one

        output1 = output1 / weight
    return output1


def dither(input1, delay_time):
    [n1, n2] = input1.shape
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


def atomic_torch_save(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_save_path = f"{save_path}.tmp"
    torch.save(state, tmp_save_path)
    os.replace(tmp_save_path, save_path)


def load_loss_history(output_dir, logger=None):
    history_dir = os.path.join(output_dir, "loss_history")
    history_path = os.path.join(history_dir, "latest.pth")
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_snr": [],
    }
    if not os.path.exists(history_path):
        return history

    history_state = torch.load(history_path, map_location='cpu')
    for key in history:
        values = history_state.get(key, [])
        if isinstance(values, np.ndarray):
            values = values.tolist()
        history[key] = list(values)

    if logger is not None and history["epochs"]:
        logger.info(f"Loaded loss history from {history_path} ({len(history['epochs'])} epochs)")
    return history


def trim_loss_history(history, next_epoch, logger=None):
    trimmed_history = {key: [] for key in history}
    kept = 0
    for idx, epoch in enumerate(history.get("epochs", [])):
        if epoch < next_epoch:
            for key in trimmed_history:
                trimmed_history[key].append(history[key][idx])
            kept += 1

    if logger is not None and kept != len(history.get("epochs", [])):
        logger.info(f"Trimmed loss history to {kept} epochs before resume epoch {next_epoch}")
    return trimmed_history


def save_loss_history(output_dir, epoch, train_loss, val_loss=None, val_snr=None, history=None, logger=None):
    if history is None:
        history = load_loss_history(output_dir)

    history = {
        "epochs": list(history.get("epochs", [])),
        "train_loss": list(history.get("train_loss", [])),
        "val_loss": list(history.get("val_loss", [])),
        "val_snr": list(history.get("val_snr", [])),
    }

    epoch_state = {
        "epoch": int(epoch),
        "train_loss": None if train_loss is None else float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "val_snr": None if val_snr is None else float(val_snr),
        "saved_at": time.time(),
    }

    if epoch in history["epochs"]:
        idx = history["epochs"].index(epoch)
        history["train_loss"][idx] = epoch_state["train_loss"]
        history["val_loss"][idx] = epoch_state["val_loss"]
        history["val_snr"][idx] = epoch_state["val_snr"]
    else:
        history["epochs"].append(epoch_state["epoch"])
        history["train_loss"].append(epoch_state["train_loss"])
        history["val_loss"].append(epoch_state["val_loss"])
        history["val_snr"].append(epoch_state["val_snr"])

    history_dir = os.path.join(output_dir, "loss_history")
    epoch_path = os.path.join(history_dir, f"epoch_{epoch:04d}.pth")
    latest_path = os.path.join(history_dir, "latest.pth")

    atomic_torch_save(epoch_state, epoch_path)
    atomic_torch_save(
        {
            **history,
            "last_epoch": int(epoch),
            "saved_at": epoch_state["saved_at"],
        },
        latest_path,
    )

    if logger is not None:
        logger.info(f"Saved loss history to {epoch_path} (train={epoch_state['train_loss']}, val={epoch_state['val_loss']})")
    return history


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    min_loss = 1.0
    if not config.EVAL_MODE and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        if "amp" in checkpoint and amp is not None and config.AMP_OPT_LEVEL != "O0" and checkpoint["config"].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint["amp"])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]
        if "min_loss" in checkpoint:
            min_loss = checkpoint["min_loss"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, min_loss


def save_checkpoint(config, epoch, model, max_accuracy, min_loss, optimizer, lr_scheduler, logger):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "max_accuracy": max_accuracy,
        "min_loss": min_loss,
        "epoch": epoch,
        "config": config,
    }
    if config.AMP_OPT_LEVEL != "O0" and amp is not None:
        save_state["amp"] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    logger.info(f"{save_path} saving......")
    atomic_torch_save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


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
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def build_optimizer(config, model):
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()
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


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


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
        return FallbackStepLRScheduler(
            optimizer,
            decay_t=max(1, decay_steps),
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
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


class FallbackCosineLRScheduler:
    def __init__(self, optimizer, t_initial, lr_min, warmup_lr_init=0.0, warmup_t=0):
        self.optimizer = optimizer
        self.base_values = [group["lr"] for group in optimizer.param_groups]
        self.t_initial = max(1, t_initial)
        self.lr_min = lr_min
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        self.last_t = 0
        if self.warmup_t:
            self.update_groups([warmup_lr_init for _ in self.base_values])

    def update_groups(self, values):
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value

    def get_update_values(self, t):
        if self.warmup_t and t < self.warmup_t:
            return [
                self.warmup_lr_init + (base - self.warmup_lr_init) * t / self.warmup_t
                for base in self.base_values
            ]
        t = min(max(t - self.warmup_t, 0), max(1, self.t_initial - self.warmup_t))
        total_t = max(1, self.t_initial - self.warmup_t)
        return [
            self.lr_min + 0.5 * (base - self.lr_min) * (1 + np.cos(np.pi * t / total_t))
            for base in self.base_values
        ]

    def step_update(self, num_updates):
        self.last_t = num_updates
        self.update_groups(self.get_update_values(num_updates))

    def state_dict(self):
        return {"base_values": self.base_values, "last_t": self.last_t}

    def load_state_dict(self, state_dict):
        self.base_values = state_dict.get("base_values", self.base_values)
        self.last_t = state_dict.get("last_t", state_dict.get("t", self.last_t))


class FallbackStepLRScheduler:
    def __init__(self, optimizer, decay_t, decay_rate, warmup_lr_init=0.0, warmup_t=0):
        self.optimizer = optimizer
        self.base_values = [group["lr"] for group in optimizer.param_groups]
        self.decay_t = max(1, decay_t)
        self.decay_rate = decay_rate
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        self.last_t = 0
        if self.warmup_t:
            self.update_groups([warmup_lr_init for _ in self.base_values])

    def update_groups(self, values):
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value

    def get_update_values(self, t):
        if self.warmup_t and t < self.warmup_t:
            return [
                self.warmup_lr_init + (base - self.warmup_lr_init) * t / self.warmup_t
                for base in self.base_values
            ]
        step_index = max(0, (t - self.warmup_t) // self.decay_t)
        return [base * (self.decay_rate ** step_index) for base in self.base_values]

    def step_update(self, num_updates):
        self.last_t = num_updates
        self.update_groups(self.get_update_values(num_updates))

    def state_dict(self):
        return {"base_values": self.base_values, "last_t": self.last_t}

    def load_state_dict(self, state_dict):
        self.base_values = state_dict.get("base_values", self.base_values)
        self.last_t = state_dict.get("last_t", state_dict.get("t", self.last_t))


class LinearLRScheduler(Scheduler if Scheduler is not object else object):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        if Scheduler is object:
            self.optimizer = optimizer
            self.base_values = [group["lr"] for group in optimizer.param_groups]
            self.last_t = 0
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
            if Scheduler is object:
                self.update_groups([self.warmup_lr_init for _ in self.base_values])
            else:
                super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def update_groups(self, values):
        if Scheduler is object:
            for group, value in zip(self.optimizer.param_groups, values):
                group["lr"] = value
        else:
            super().update_groups(values)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def step_update(self, num_updates):
        if Scheduler is object:
            values = self.get_update_values(num_updates)
            if values is not None:
                self.update_groups(values)
        else:
            super().step_update(num_updates)

    def state_dict(self):
        if Scheduler is object:
            return {
                "base_values": self.base_values,
                "last_t": getattr(self, "last_t", 0),
            }
        return super().state_dict()

    def load_state_dict(self, state_dict):
        if Scheduler is object:
            self.base_values = state_dict.get("base_values", self.base_values)
            self.last_t = state_dict.get("last_t", state_dict.get("t", getattr(self, "last_t", 0)))
        else:
            super().load_state_dict(state_dict)


class DecayLRScheduler(Scheduler if Scheduler is not object else object):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_t: float,
        min_lr: float,
        basic_lr: float = 5e-4,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        if Scheduler is object:
            self.optimizer = optimizer
            self.base_values = [group["lr"] for group in optimizer.param_groups]
            self.last_t = 0
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

    def update_groups(self, values):
        if Scheduler is object:
            for group, value in zip(self.optimizer.param_groups, values):
                group["lr"] = value
        else:
            super().update_groups(values)

    def _get_lr(self, t):
        if t < self.decay_t:
            lrs = [v - (self.basic_lr - self.min_lr) * t / self.decay_t for v in self.base_values]
        else:
            lrs = [self.min_lr for _ in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def step_update(self, num_updates):
        if Scheduler is object:
            values = self.get_update_values(num_updates)
            if values is not None:
                self.update_groups(values)
        else:
            super().step_update(num_updates)

    def state_dict(self):
        if Scheduler is object:
            return {
                "base_values": self.base_values,
                "last_t": getattr(self, "last_t", 0),
            }
        return super().state_dict()

    def load_state_dict(self, state_dict):
        if Scheduler is object:
            self.base_values = state_dict.get("base_values", self.base_values)
            self.last_t = state_dict.get("last_t", state_dict.get("t", getattr(self, "last_t", 0)))
        else:
            super().load_state_dict(state_dict)


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
        X_train = np.load(self.image_path + "/" + self.image_list[index]).astype(np.float32)
        Y_train = np.load(self.label_path + "/" + self.image_list[index]).astype(np.float32)
        return X_train, Y_train, self.image_list[index]


def build_fromfolder(config):
    dataset_train = NewDataFolder(config.DATA.DATA_PATH, config.MODEL.TYPE)
    dataset_val = NewDataFolder(config.DATA.TEST_PATH, config.MODEL.TYPE)
    return dataset_train, dataset_val


def patching(config, inputs, name, targets=None, test=False, denoise=False):
    inputs = inputs.numpy()
    flag1 = flag2 = False
    index = torch.zeros((2,), dtype=torch.float32)
    names = name.split(".")[0]

    index[1] = int(names[1:])
    if names[0] == "a":
        index[0] = 5 if test else 0
        H, W, sh, sw = 1200, 120, 48, 48
        flag1 = True
    elif names[0] == "n":
        index[0] = 6 if test else 1
        H, W, sh, sw = 1500, 172, 48, 48
    elif names[0] == "m":
        index[0] = 7 if test else 1
        H, W, sh, sw = 2048, 256, 64, 64
    elif names[0] == "p":
        index[0] = 8 if test else 1
        H, W, sh, sw = 1600, 256, 48, 48

    index = torch.cat([index, index], axis=0)
    index = torch.cat([index, index], axis=0)

    newinputs = np.zeros((1, 1, H, W), dtype=np.float32)
    newinputs[0, 0, :, :] = inputs[0, :, :W]
    inputs = newinputs
    newtargets = np.zeros((1, 1, H, W), dtype=np.float32)
    newtargets[0, 0, :, :] = targets[0, :, :W]
    targets = newtargets

    B, C, H, W = inputs.shape
    for i in range(C):
        if i == 0:
            outputs = myimtocol(inputs[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            B, H, W = outputs.shape
            outputs = outputs.reshape(B, 1, H, W)
        else:
            temp = myimtocol(inputs[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            B, H, W = temp.shape
            temp = temp.reshape(B, 1, H, W)
            outputs = np.concatenate((outputs, temp), axis=0)

    if targets is not None:
        for i in range(C):
            if i == 0:
                outs = myimtocol(targets[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
                B, H, W = outs.shape
                outs = outs.reshape(B, 1, H, W)
            else:
                temp = myimtocol(targets[0, i, :, :], config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
                B, H, W = temp.shape
                temp = temp.reshape(B, 1, H, W)
                outs = np.concatenate((outs, temp), axis=0)
    else:
        x_train = torch.from_numpy(outputs)
        return x_train, index
    if denoise:
        if flag1 and not test:
            abspath = os.getcwd()
            delay = np.load(abspath + "/" + config.DATA.DATA_PATH + "/delay/" + name)
            obsers = np.load(abspath + "/" + config.DATA.DATA_PATH + "/sample/" + name)
            obser1, obser2 = obsers[0, :, :], obsers[1, :, :]
            ntemp1 = obser1 - dither(inputs[0, 1, :, :], delay) * np.random.uniform(0.75, 0.91)
            ntemp2 = obser2 - dither(inputs[0, 0, :, :], -delay) * np.random.uniform(0.75, 0.91)
            ntemp1 = myimtocol(ntemp1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            ntemp2 = myimtocol(ntemp2, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, H, W, sh, sw, 1)
            out2 = np.concatenate((ntemp1, ntemp2), axis=0)
            B, H, W = out2.shape
            outputs = out2.reshape(B, 1, H, W)
            outputs = np.concatenate((outputs, outputs), axis=0)
            outs = np.concatenate((outs, outs), axis=0)
        if flag2 and not test:
            outputs = np.concatenate(
                (outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)),
                axis=0,
            )
            outs = np.concatenate((outs, outs), axis=0)
            outputs = outputs[:264, :, :, :]
            outs = outs[:264, :, :, :]
    else:
        if flag1 and not test:
            outputs = np.concatenate(
                (outputs, outputs + np.random.normal(0, 0.05, outputs.shape).astype(np.float32)),
                axis=0,
            )
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
    B, C, H, W = outs.shape
    if config.DATA.BATCH_SIZE < B:
        random_slice = np.random.choice(B, config.DATA.BATCH_SIZE, replace=False)
        y_train = torch.from_numpy(outs[random_slice, :, :, :])
        x_train = torch.from_numpy(outputs[random_slice, :, :, :])
    else:
        y_train = torch.from_numpy(outs[:256, :, :, :])
        x_train = torch.from_numpy(outputs[:256, :, :, :])
    return x_train, y_train, index


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "WUDT_STAnet":
        model = WUDT_STAnet(
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
    else:
        raise NotImplementedError(f"Unknown model: {model_type}. This open-source package keeps only WUDT_STAnet.")

    return model


def parse_option():
    parser = argparse.ArgumentParser("Swin Transformer training and evaluation script", add_help=False)
    parser.add_argument("--cfg", type=str, metavar="FILE", default="configs/DT2_6.yaml", help="path to config file")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64, help="batch size for single GPU")
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
    loss_name = config.TRAIN.LOSS.NAME.upper()

    if loss_name == "MSE":
        if config.MODEL.TYPE == "NAFnet":
            return L1()
        return torch.nn.MSELoss()
    if loss_name == "L1":
        return L1()
    if loss_name in ("L1_AGC", "AGC_L1"):
        return L1_agc(
            agc_window=config.TRAIN.LOSS.AGC_WINDOW,
            agc_axis=config.TRAIN.LOSS.AGC_AXIS,
            eps=1e-6,
            gain_eps=config.TRAIN.LOSS.AGC_EPS,
            max_gain=config.TRAIN.LOSS.AGC_MAX_GAIN,
        )

    raise ValueError(f"Unsupported loss: {config.TRAIN.LOSS.NAME}")


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
    if config.MODEL.RESUME:
        max_accuracy, min_loss = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        criterion = build_criterion(config)
        acc1, loss = validate(config, data_loader_val, model, criterion)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    loss_history = load_loss_history(config.OUTPUT, logger)
    loss_history = trim_loss_history(loss_history, config.TRAIN.START_EPOCH, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        criterion = build_criterion(config)
        train_loss = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)
        loss_history = save_loss_history(
            config.OUTPUT,
            epoch,
            train_loss=train_loss,
            history=loss_history,
            logger=logger,
        )
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)) and "preprocess" not in config.MODEL.NAME:
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, min_loss, optimizer, lr_scheduler, logger)

        acc1, loss = validate(config, data_loader_val, model, criterion)
        loss_history = save_loss_history(
            config.OUTPUT,
            epoch,
            train_loss=train_loss,
            val_loss=loss,
            val_snr=acc1,
            history=loss_history,
            logger=logger,
        )
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

        output = model(samples)
        if config.DENOISE:
            output = model(samples, index)
        loss = criterion(output, targets)
        snr1 = snr(output, targets)

        model.train()
        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        snr_meter.update(snr1.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
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
    logger.info(f"EPOCH {epoch} average training loss: {loss_meter.avg:.8f}")
    return loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, labels, names) in enumerate(data_loader):
        samples, targets, index = patching(config, images, names[0], targets=labels, test=True, denoise=config.DENOISE)
        if config.DENOISE:
            index = index.cuda(non_blocking=True)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        output = model(samples)
        if config.DENOISE:
            output = model(samples, index)
        loss = criterion(output, targets)
        acc1 = snr(output, targets)

        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))

        # measure elapsed time
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
