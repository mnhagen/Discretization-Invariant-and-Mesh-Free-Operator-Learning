import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F

class CNO_LReLu(nn.Module):
    """This class is responsible for applying non-linearities without aliasing. This is
        done by first upsampling, then applying the non-linearity, then downsampling using 
        a low-pass filter."""
    def __init__(self, in_size, out_size):
        super(CNO_LReLu, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(2), size = (1,2 * self.in_size), mode = "bicubic", antialias = True)
        x = self.act(x)
        x = F.interpolate(x, size = (1, self.out_size), mode = "bicubic", antialias = True)
        return x[:,:,0]
    
class CNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size, use_bn = True):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size

        self.convolution = torch.nn.Conv1d(in_channels = self.in_channels, out_channels = self.out_channels,
                                           kernel_size = 3, padding = 1)
        
        if use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()
        
    def forward(self,x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)
    
class LiftProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, latent_dim = 64):
        super(LiftProjectBlock, self).__init__()

        self.inter_CNOBlock = CNOBlock(in_channels = in_channels, out_channels = out_channels,
                                       in_size = size, out_size = size, use_bn = False)
        
        self.convolution = torch.nn.Conv1d(in_channels = latent_dim, out_channels = out_channels,
                                           kernel_size = 3, padding = 1)
        
    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, size, use_bn = True):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size = size

        self.convolution1 = torch.nn.Conv1d(in_channels = self.channels, out_channels = self.channels,
                                            kernel_size = 3, padding = 1)
        
        self.convolution2 = torch.nn.Conv1d(in_channels = self.channels, out_channels = self.channels,
                                            kernel_size = 3, padding = 1)
        
        if use_bn:
            self.batch_norm1 = nn.BatchNorm1d(self.channels)
            self.batch_norm2 = nn.BatchNorm1d(self.channels)
        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.act = CNO_LReLu(in_size = self.size, out_size = self.size)

    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out
    
class ResNet(nn.Module):
    def __init__(self, channels, size, num_blocks, use_bn = True):
        super(ResNet, self).__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(ResidualBlock(channels = channels, size = self.size, use_bn = use_bn))

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x
    
class CNO1D(nn.Module):
    def __init__(self, in_dim, out_dim, size, N_layers, N_res = 4, 
                 N_res_neck = 4, channel_multiplier = 16, use_bn = True):

        super(CNO1D, self).__init__()

        self.N_layers = int(N_layers)
        self.lift_dim = channel_multiplier//2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_multiplier = channel_multiplier

        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2**i*self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i]
            
        