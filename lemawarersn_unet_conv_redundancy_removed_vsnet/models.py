import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import itertools
from collections import OrderedDict
import math
from dautomap import * 
from unetmodels import *

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        intout= self.layers[:4](input)
        output = self.layers[4:](intout)
        return intout, output

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'



# the segunet model which gives LEM features
class UnetModelTakeLatentDecoder(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        lf=[]
        output = input
        H,W = input.shape[2],input.shape[3]
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            intout,output = layer(output)
            #intoutup = F.upsample(intout,size=(H,W),mode='bilinear')
            outputup = F.upsample(output,size=(H,W),mode='bilinear')
            #print(output.shape)
            #feat=torch.cat([intoutup,outputup],dim=1)
            #lf.append(outputup)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        intout,output = self.conv(output)
        #latentfeat = torch.cat([intout,output],dim=1) 
        #print("latent shape: ",latentfeat.shape)
        #intoutup = F.upsample(intout,size=(H,W),mode='bilinear')
        outputup = F.upsample(output,size=(H,W),mode='bilinear')
        #feat = torch.cat([intoutup, outputup],dim=1)
        lf.append(outputup)
        #print(intoutup.shape,outputup.shape)
        #print(output.shape)
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            intout,output = layer(output)
            #intoutup = F.upsample(intout,size=(H,W),mode='bilinear')
            outputup = F.upsample(output,size=(H,W),mode='bilinear')
            #print(output.shape)
            #feat=torch.cat([intoutup,outputup],dim=1)
            lf.append(outputup)
        finalfeat=torch.cat(lf,dim=1)    
        #print(thinfeat.shape)
     
        return finalfeat,self.conv2(output)

class ConvFeatureBlock(nn.Module):
    def __init__(self,n_ch,nf=32,n_out=2):
        super(ConvFeatureBlock,self).__init__()

        #self.conv1x1=nn.Conv2d(1984,nf,kernel_size=1)
        #self.conv1x1=nn.Conv2d(992,nf,kernel_size=1)
        self.conv1=nn.Sequential(nn.Conv2d(n_ch,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(nf+32,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(nf+32,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv2d(nf+32,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv5=nn.Conv2d(nf+32,n_out,kernel_size=3,padding=1)

      
    def forward(self, x, feat):

        #feat = self.conv1x1(feat)

        x = self.conv1(x)
        x = torch.cat([x,feat],dim=1)
       
        x = self.conv2(x)
        x = torch.cat([x,feat],dim=1)
        
        x = self.conv3(x)
        x = torch.cat([x,feat],dim=1)

        x = self.conv4(x)
        x = torch.cat([x,feat],dim=1)

        x = self.conv5(x)

        return x



class ReconSynergyNetAblativeFeature(nn.Module):

    def __init__(self):

        super(ReconSynergyNetAblativeFeature, self).__init__()

        patch_size_row = 640
        patch_size_col = 368

        model_params = {
          'input_shape': (2, patch_size_row, patch_size_col),
          'output_shape': (2, patch_size_row, patch_size_col),
          'tfx_params': {
            'nrow': patch_size_row,
            'ncol': patch_size_col,
            'nch_in': 2,
            'kernel_size': 1,
            'nl': 'relu',
            'init_fourier': False,
            'init': 'xavier_uniform_',
            'bias': True,
            'share_tfxs': False,
            'learnable': True,
          },
          'depth': 2,
          'nl':'relu'
        }

        dautomap_model = dAUTOMAPFeat(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'])
        unet_model = UnetModelAssistLatentDecoder(in_chans=2, out_chans=2,chans=32,num_pool_layers=4, drop_prob = 0 )
        srcnnlike_model =  ConvFeatureBlock(n_ch=6,n_out=2)

        self.KI_layer = dautomap_model        
        self.II_layer = unet_model
        self.Re_layer = srcnnlike_model        
        

    def forward(self, x, xk,feat):

        dautomap_pred = self.KI_layer(xk,feat)

        #y = x[:,:,:,:-4] # crop based on axial  484 to 480
        unet_pred     = self.II_layer(x,feat)
        #unet_pred = F.pad(unet_pred,[2,2]) # pad based on axial 

        #print (dautomap_pred.shape,unet_pred.shape,x.shape)
        pred_cat = torch.cat([unet_pred,dautomap_pred,x],dim=1)
        #print (pred_cat.shape)
        recons = self.Re_layer(pred_cat,feat)
        
        return recons

#model = ReconSynergyNetAblative()
#

