import torch
from torch import nn as nn
from torch.nn import functional as F


class L1(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1, self).__init__()
        self.eps = 1e-6

    def forward(self, X1, Y):
        diff = torch.add(X1, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss 


class L1_agc(nn.Module):
    """L1 Charbonnier loss after applying target-driven AGC."""

    def __init__(self, agc_window=31, agc_axis=-2, eps=1e-6, gain_eps=1e-6, max_gain=0.0):
        super(L1_agc, self).__init__()
        if agc_window < 1:
            raise ValueError('agc_window must be >= 1')
        if agc_window % 2 == 0:
            agc_window += 1

        self.eps = eps
        self.gain_eps = gain_eps
        self.agc_window = agc_window
        self.agc_axis = agc_axis
        self.max_gain = max_gain
        self.last_gain = None

        agc_kernel = torch.ones(1, 1, self.agc_window, dtype=torch.float32) / float(self.agc_window)
        self.register_buffer('agc_kernel', agc_kernel)

    def compute_gain(self, target):
        axis = self.agc_axis if self.agc_axis >= 0 else target.dim() + self.agc_axis
        if axis < 0 or axis >= target.dim():
            raise ValueError(f'agc_axis={self.agc_axis} is out of range for target dim={target.dim()}')

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

class MIX2(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(MIX2, self).__init__()
        self.eps = 1e-6

    def forward(self, X1, X2, Y):
        diff = torch.add(X1, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return 0.5*loss + 0.5*F.mse_loss(X2,Y)

class MIX3(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(MIX3, self).__init__()
        self.eps = 1e-6

    def forward(self, X1,X2, Y):
        diff = torch.add(X1, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss + F.mse_loss(X2,Y)

class Mixloss(nn.Module):
    def __init__(self, weight1=0.5,weight2=0.5, reduction='mean'):
        super(Mixloss, self).__init__()
        self.reduction = reduction
        self.weight1=weight1
        self.weight2=weight2


    def forward(self,pred,target):
        pred01 = pred[:, :, 0::2, :] / 2
        pred02 = pred[:, :, 1::2, :] / 2
        pred1 = pred01[:, :, :, 0::2]
        pred2 = pred02[:, :, :, 0::2]
        pred3 = pred01[:, :, :, 1::2]
        pred4 = pred02[:, :, :, 1::2]
        pred_HL = -pred1 - pred2 + pred3 + pred4 ##vertical
        pred_LH = -pred1 + pred2 - pred3 + pred4  ## hozational

        target01 = target[:, :, 0::2, :] / 2
        target02 = target[:, :, 1::2, :] / 2
        target1 = target01[:, :, :, 0::2]
        target2 = target02[:, :, :, 0::2]
        target3 = target01[:, :, :, 1::2]
        target4 = target02[:, :, :, 1::2]
        target_HL = -target1 - target2 + target3 + target4 ##vertical
        target_LH = -target1 + target2 - pred3 + target4  ## hozational

        return F.mse_loss(pred,target)+self.weight1*F.l1_loss(pred_HL, target_HL)+self.weight2*F.l1_loss(pred_LH, target_LH)
