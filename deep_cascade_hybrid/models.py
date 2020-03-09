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

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        if kspace_flag == False:
            kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        else:
            kspace_predicted_img = predicted_img
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        
        updated_kspace1  = self.us_mask * us_kspace
        #print(self.us_mask.dtype, kspace_predicted_img.dtype) 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img.double()
        #print(updated_kspace1.shape, updated_kspace2.shape)
        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        #update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        update_img_abs = updated_img[:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float(), updated_kspace.float()


class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=3, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

        print(args)
        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs1 = []
        kspace_conv_blocks = []
        dcs2 = []

        conv_layer = conv_block


        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs1.append(DataConsistencyLayer(us_mask))
            kspace_conv_blocks.append(conv_block(2, nd, **kwargs))
            dcs2.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs1 = dcs1
        self.kspace_conv_blocks = nn.ModuleList(kspace_conv_blocks)
        self.dcs2 = dcs2
        self.convn_1 = conv_layer(n_channels,nd)#n-1 th x_cnn layer 
        self.dcsn_1 = DataConsistencyLayer(us_mask)#n-1 th dc layer
        self.convn = conv_layer(n_channels,nd) # nth x_cnn layer
        self.dcsn = DataConsistencyLayer(us_mask)#nth dc layer

    def forward(self,x,k):

        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x,ksp = self.dcs1[i](x,k,False)
            ksp = ksp.permute(0,3,1,2)
            ksp_cnn = self.kspace_conv_blocks[i](ksp)
            ksp = ksp + ksp_cnn
            ksp = ksp.unsqueeze(4).permute(0,4,2,3,1)
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
