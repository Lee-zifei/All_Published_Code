# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import torch
import torch.distributed as dist
from myprog import myimtocol
import numpy as np
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def atomic_torch_save(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_save_path = f'{save_path}.tmp'
    torch.save(state, tmp_save_path)
    os.replace(tmp_save_path, save_path)


def load_loss_history(output_dir, logger=None):
    history_dir = os.path.join(output_dir, 'loss_history')
    history_path = os.path.join(history_dir, 'latest.pth')
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_snr': [],
    }
    if not os.path.exists(history_path):
        return history

    history_state = torch.load(history_path, map_location='cpu')
    for key in history:
        values = history_state.get(key, [])
        if isinstance(values, np.ndarray):
            values = values.tolist()
        history[key] = list(values)

    if logger is not None and history['epochs']:
        logger.info(
            f"Loaded loss history from {history_path} "
            f"({len(history['epochs'])} epochs)"
        )
    return history


def trim_loss_history(history, next_epoch, logger=None):
    trimmed_history = {key: [] for key in history}
    kept = 0
    for idx, epoch in enumerate(history.get('epochs', [])):
        if epoch < next_epoch:
            for key in trimmed_history:
                trimmed_history[key].append(history[key][idx])
            kept += 1

    if logger is not None and kept != len(history.get('epochs', [])):
        logger.info(
            f"Trimmed loss history to {kept} epochs before resume epoch {next_epoch}"
        )
    return trimmed_history


def save_loss_history(output_dir, epoch, train_loss, val_loss=None, val_snr=None, history=None, logger=None):
    if history is None:
        history = load_loss_history(output_dir)

    history = {
        'epochs': list(history.get('epochs', [])),
        'train_loss': list(history.get('train_loss', [])),
        'val_loss': list(history.get('val_loss', [])),
        'val_snr': list(history.get('val_snr', [])),
    }

    epoch_state = {
        'epoch': int(epoch),
        'train_loss': None if train_loss is None else float(train_loss),
        'val_loss': None if val_loss is None else float(val_loss),
        'val_snr': None if val_snr is None else float(val_snr),
        'saved_at': time.time(),
    }

    if epoch in history['epochs']:
        idx = history['epochs'].index(epoch)
        history['train_loss'][idx] = epoch_state['train_loss']
        history['val_loss'][idx] = epoch_state['val_loss']
        history['val_snr'][idx] = epoch_state['val_snr']
    else:
        history['epochs'].append(epoch_state['epoch'])
        history['train_loss'].append(epoch_state['train_loss'])
        history['val_loss'].append(epoch_state['val_loss'])
        history['val_snr'].append(epoch_state['val_snr'])

    history_dir = os.path.join(output_dir, 'loss_history')
    epoch_path = os.path.join(history_dir, f'epoch_{epoch:04d}.pth')
    latest_path = os.path.join(history_dir, 'latest.pth')

    atomic_torch_save(epoch_state, epoch_path)
    atomic_torch_save(
        {
            **history,
            'last_epoch': int(epoch),
            'saved_at': epoch_state['saved_at'],
        },
        latest_path,
    )

    if logger is not None:
        logger.info(
            f"Saved loss history to {epoch_path} "
            f"(train={epoch_state['train_loss']}, val={epoch_state['val_loss']})"
        )
    return history


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'min_loss' in checkpoint:
            min_loss = checkpoint['min_loss']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, min_loss


def save_checkpoint(config, epoch, model, max_accuracy, min_loss, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'min_loss': min_loss,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
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
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file



