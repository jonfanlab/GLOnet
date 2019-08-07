import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from metalayers import * 
import os
import utils


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = int(params.noise_dims)
        self.label_dim = int(params.label_dims)

        self.min_feat = 8
       # self.min_feat_kernel = torch.ones(self.min_feat).type(Tensor)/self.min_feat

        self.gkernel = gkern1D(params.gkernlen, params.gkernsig)

        self.FC = nn.Sequential(
            nn.Linear(self.noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(256, 32*16, bias=False),
            nn.BatchNorm1d(32*16),
            nn.LeakyReLU(0.2),
        )

        self.CONV = nn.Sequential(
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(4, 1, 5),
            )

        self.MIN_FEAT = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.min_feat, stride=self.min_feat),
            nn.Upsample(scale_factor=self.min_feat)
            )
        


    def forward(self, noise):
        net = self.FC(noise)
        net = net.view(-1, 16, 32)
        net = self.CONV(net)    
        net = conv1d_meta(net, self.gkernel)
        net = torch.tanh(net * 10) * 1.02

        return net
    
if __name__ == '__main__':
    last_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    last_path = os.path.join(last_path, 'results')
    
	# Load parameters from json file
    json_path = os.path.join(last_path, 'Params.json')
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path) 
    '''
	# Add attributes to params
	params.output_dir = output_dir
    params.lambda_gp  = 10.0
    params.n_critic = 1
    params.cuda = torch.cuda.is_available()
    params.restore_from = restore_from
    params.batch_size = int(params.batch_size)
    params.numIter = int(params.numIter)
    params.noise_dims = int(params.noise_dims)
    params.label_dims = int(params.label_dims)
    params.gkernlen = int(params.gkernlen)
    params.n_solver = int(params.n_solver)
    params.n_solver_th = int(params.n_solver_th)
    params.step_size = int(params.step_size)	
    params.w = int(args.wavelength)
    params.a = int(args.angle)
    '''
    net = Generator(params) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = net.to(device)
    summary(model, tuple([256]))
    #print(net)