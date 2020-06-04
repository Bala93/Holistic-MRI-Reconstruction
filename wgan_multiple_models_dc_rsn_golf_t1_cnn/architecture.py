import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import itertools
from collections import OrderedDict
import math
from models import DnCnFeatureLoop

## Dataconsistency implementation

class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 
        self.us_mask = self.us_mask.float() 

    def forward(self,predicted_img,us_kspace):

        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False)
        us_kspace = us_kspace.permute(0,2,3,1)
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img
        updated_kspace   = updated_kspace1 + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        update_img_abs = updated_img[:,:,:,0]
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float(), updated_kspace

# cnn blocks used instead of unet

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



class DnCnRefine(nn.Module):

    def __init__(self,args,nc=3,mode='train'):

        super(DnCnRefine, self).__init__()

        self.dataset_type = args.dataset_type

        us_mask_path = os.path.join(args.usmask_path,'mask_{}x.npy'.format(args.acceleration))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)
        
        self.dncn = DnCnFeatureLoop(args, n_channels=1)

        if mode == 'train': 
            ckpt = torch.load(args.dcrsnpath,map_location='cpu')['model']
            self.dncn.load_state_dict(ckpt)
        
            for param in self.dncn.parameters():
                param.requires_grad = False
        
        self.refine = conv_block(n_ch=1,nd=5,n_out=1)
        self.dc = DataConsistencyLayer(us_mask)

    def forward(self,x,k,feat,us_mask,seg,t1):

        x = self.dncn(x,k,feat,us_mask,seg,t1)
        x = self.refine(x)
        if self.dataset_type == 'cardiac':
            x = x[:,:,5:x.shape[2]-5,5:x.shape[3]-5]
        x,_ = self.dc(x,k)

        return x

