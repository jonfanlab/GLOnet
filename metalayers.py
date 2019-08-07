import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import scipy.stats as st
from typing import Tuple
import math
import logging

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    

def index_along(tensor, key, axis):
    indexer = [slice(None)] * len(tensor.shape)
    indexer[axis] = key
    return tensor[tuple(indexer)]


def pad_periodic(inputs, padding: int, axis: int, center: bool = True):
    
    if padding == 0:
        return inputs
    if center:
        if padding % 2 != 0:
            raise ValueError('cannot do centered padding if padding is not even')
        inputs_list = [index_along(inputs, slice(-padding//2, None), axis),
                       inputs,
                       index_along(inputs, slice(None, padding//2), axis)]
    else:
        inputs_list = [inputs, index_along(inputs, slice(None, padding), axis)]
    return torch.cat(inputs_list, dim=axis)


def pad1d_meta(inputs, padding: int):
    return pad_periodic(inputs, padding, axis=-1, center=True)


def gkern1D(kernlen=7, nsig=4):
    """Returns a 1D Gaussian kernel array."""

    x_cord = torch.arange(0., kernlen)

    mean = (kernlen - 1)/2.
    variance = nsig**2.

    # variables (in this case called x and y)
    gaussian_kernel = 1./(2.*math.pi*variance)**0.5 * torch.exp(-(x_cord - mean)**2. / (2.*variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel.type(Tensor).requires_grad_(False)



def conv1d(inputs, kernel, padding='same'):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 1d kernel
    """
    B, C, _ = inputs.size()
    kH = kernel.size()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, C, 1)

    if padding == 'valid':
        return F.conv1d(inputs, kernel)
    elif padding == 'same':
        pad = (kH-1)//2
        return F.conv1d(inputs, kernel, padding = pad)


def conv1d_meta(inputs, kernel):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 1d kernel
    """
    kH = kernel.size(0)
    padded_inputs = pad1d_meta(inputs, kH-1)
    
    return conv1d(padded_inputs, kernel, padding='valid')


class AvgPool1d_meta(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__()
        self.padding = kernel_size - 1
        self.avgpool1d = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, inputs):
        padded_inputs = pad1d_meta(inputs, self.padding)
        return self.avgpool1d(padded_inputs) 


class ConvTranspose1d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=True, dilation=1):
        super().__init__()
        self.padding = kernel_size - 1
        self.trim = self.padding * stride // 2
        pad = (kernel_size - stride) // 2 
        self.output_padding = (kernel_size - stride) % 2 
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=pad,
                                          output_padding=0, groups=groups, bias=bias, dilation=dilation)
    
    def forward(self, inputs):
        padded_inputs = pad1d_meta(inputs, self.padding)
        padded_outputs = self.conv1d_transpose(padded_inputs)
        if self.output_padding:
            padded_outputs = padded_outputs[:, :, 1:]
        
        if self.trim:
            return padded_outputs[:, :, self.trim:-self.trim]
        else:
            return padded_outputs
 
    
class Conv1d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1)*dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = 0, 
                                dilation = dilation, groups = groups, bias = bias)
    
    def forward(self, inputs):
        padded_inputs = pad1d_meta(inputs, self.padding)
        outputs = self.conv1d(padded_inputs)
        return outputs








        