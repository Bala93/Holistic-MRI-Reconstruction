import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
from model_rsn import *


class RecursiveDilatedBlock(nn.Module):
    def __init__(self, n_ch, nd=5, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=1):
        super().__init__()

        self.nd = nd

        # dilated convolution
        pad_conv = 1

        def conv_i(n_i, n_out, dilation):
            return nn.Conv2d(n_i, n_out, ks, stride=1, padding=dilation, dilation=dilation, bias=True)

        conv_1 = nn.Conv2d(n_ch, nf, ks, stride=1, padding=pad_conv, dilation=1, bias=True)
        conv_n = nn.Conv2d(nf, n_out, ks, stride=1, padding=pad_conv, dilation=1, bias=True)
        #print(conv_1) 
        #print(conv_n) 
        # relu
        #nll = relu if nl == 'relu' else lrelu
        self.layers = nn.ModuleList([])
        self.layers += [nn.Sequential(conv_1, nn.LeakyReLU(0.01, True))]
        self.layers += [nn.Sequential(conv_i(nf,nf,1), nn.LeakyReLU(0.01, True))]
        self.layers += [nn.Sequential(conv_i(nf,nf,2), nn.LeakyReLU(0.01, True))]
        self.layers += [nn.Sequential(conv_i(nf,nf,3), nn.LeakyReLU(0.01, True))]
        self.layers += [nn.Sequential(conv_n, nn.LeakyReLU(0.01, True))]

    def forward(self,x):

        xtemp = self.layers[0](x) # first conv that takes image input, gives out 32 feature maps
        x1 = xtemp
        for i in range(1,self.nd):
            xtemp = self.layers[1](xtemp)#dilation 1 conv layer + relu
            xtemp = self.layers[2](xtemp) #dilation 2 conv layer + relu
            xtemp = self.layers[3](xtemp) #dilation3 conv layer + relu
            xtemp = x1 + xtemp
            #print("xtemp: ", torch.sum(xtemp))
            #print("x1: ", torch.sum(x1))
        out = self.layers[4](xtemp) 
        return out+x


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 

    def forward(self,predicted_img,us_kspace):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        #update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        update_img_abs = updated_img[:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()


class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=4, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        #conv_layer = conv_block
        self.recon_synergy_model = ReconSynergyNetAblative(args)
        self.dc_block = DataConsistencyLayer(us_mask)


        for i in range(nc):
            conv_blocks.append(RecursiveDilatedBlock(n_channels, **kwargs))
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k):


        k_inp = k.permute(0,3,1,2).float()
        k = k.unsqueeze(1)
        x = self.recon_synergy_model(x,k_inp)
        x = self.dc_block(x,k)


        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            x = self.dcs[i](x,k)

        return x

