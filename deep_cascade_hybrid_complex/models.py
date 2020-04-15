import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 

    def forward(self,predicted_img,us_kspace,kspace_flag=False):

        predicted_img = predicted_img.permute(0,2,3,1)

        if not kspace_flag:
            kspace_predicted_img = torch.fft(predicted_img,2,True)
        else:
            kspace_predicted_img = predicted_img

        us_kspace = us_kspace.permute(0,2,3,1)
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        #print (us_kspace.dtype,predicted_img.dtype,kspace_predicted_img.dtype,self.us_mask.dtype)
        
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1 + updated_kspace2
        
        updated_img    = torch.ifft(updated_kspace,2,True) 

        updated_img   = updated_img.permute(0,3,1,2)
        updated_kspace = updated_kspace.permute(0,3,1,2)

        #print (updated_img.shape,updated_kspace.shape)
        
        return updated_img,updated_kspace


class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=2, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

        print(args)
        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).float().to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs1 = []
        kspace_conv_blocks = []
        dcs2 = []

        conv_layer = conv_block


        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs1.append(DataConsistencyLayer(us_mask))
            kspace_conv_blocks.append(conv_block(n_channels, nd, **kwargs))
            dcs2.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs1 = nn.ModuleList(dcs1)
        self.kspace_conv_blocks = nn.ModuleList(kspace_conv_blocks)
        self.dcs2 = nn.ModuleList(dcs2)

        self.convn_1 = conv_layer(n_channels,nd)#n-1 th x_cnn layer 
        self.dcsn_1 = DataConsistencyLayer(us_mask)#n-1 th dc layer
        self.convn = conv_layer(n_channels,nd) # nth x_cnn layer
        self.dcsn = DataConsistencyLayer(us_mask)#nth dc layer

    def forward(self,x,k):

        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x,ksp = self.dcs1[i](x,k,False)
            ksp_cnn = self.kspace_conv_blocks[i](ksp)
            ksp = ksp + ksp_cnn
            x, ksp = self.dcs2[i](ksp,k,True)
        x_cnn = self.convn_1(x)
        x = x + x_cnn
        x,ksp = self.dcsn_1(x,k,False)
        x_cnn = self.convn(x)
        x = x + x_cnn
        x,ksp = self.dcsn(x,k,False)
        
        return x
'''
x = torch.rand(1,2,160,160)
layer = conv_block(n_ch=2, nd=5)
print(layer)
y = layer(x)
print(y.shape)
'''
