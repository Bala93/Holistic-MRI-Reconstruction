import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import common

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

class ConvBlock2X_to_X(nn.Module):
    """
    A Convolutional Block that consists of one convolution layer each followed by
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

        self.layer = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        output = self.layer(input)
        return output



class UnetModelAssistEverywhere(nn.Module):
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
        #self.conv1x1=nn.Conv2d(1984,32,kernel_size=1)
        #self.conv1x1=nn.Conv2d(992,32,kernel_size=1)
        na=96
        new_downsample_block = nn.ModuleDict({'conv1':ConvBlock(in_chans,chans,drop_prob),'conv2':ConvBlock2X_to_X(chans+na,chans,drop_prob)})
        self.down_sample_layers = nn.ModuleList([new_downsample_block])
        ch = chans
        
        for i in range(num_pool_layers - 1):
            new_downsample_block = nn.ModuleDict({'conv1':ConvBlock(ch,ch*2,drop_prob),'conv2':ConvBlock2X_to_X((ch*2)+na,ch*2,drop_prob)})
            self.down_sample_layers += nn.ModuleList([new_downsample_block])
            ch *= 2

        new_downsample_block = nn.ModuleDict({'conv1':ConvBlock(ch,ch,drop_prob),'conv2':ConvBlock2X_to_X(ch+na,ch,drop_prob)})
        self.conv = nn.ModuleList([new_downsample_block])

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            new_upsample_block = nn.ModuleDict({'conv1':ConvBlock(ch*2,ch//2,drop_prob),'conv2':ConvBlock2X_to_X((ch//2)+na,ch//2,drop_prob)})
            self.up_sample_layers += nn.ModuleList([new_upsample_block])
            ch //= 2

        new_upsample_block = nn.ModuleDict({'conv1':ConvBlock(ch*2,ch,drop_prob),'conv2':ConvBlock2X_to_X(ch+na,ch,drop_prob)})
        self.up_sample_layers += nn.ModuleList([new_upsample_block])
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
        print("#############################")
        print(self.down_sample_layers)

    def forward(self, input, feat):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        H,W = input.shape[2],input.shape[3]
        #feat = self.conv1x1(feat)
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            #print(layer)
            _,output = layer['conv1'](output)
            intH,intW = output.shape[2],output.shape[3]
            resampledfeat = F.upsample(feat,size=(intH,intW), mode='bilinear') 
            #print(intout.shape,output.shape)
            intmfeat=torch.cat([resampledfeat,output],dim=1)
            output = layer['conv2'](intmfeat)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
            #print("output: ",output.shape)

        _,output = self.conv[0]['conv1'](output)
        intH,intW = output.shape[2],output.shape[3]
        resampledfeat = F.upsample(feat,size=(intH,intW), mode='bilinear') 
        latentfeat = torch.cat([resampledfeat, output],dim=1)
        #print("latent shape: ",latentfeat.shape)
        output = self.conv[0]['conv2'](latentfeat)
        #print("output lat:  ",output.shape)
        #print(output.shape)
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            _,output = layer['conv1'](output)
            #print("output up: ",output.shape)
            intH,intW = output.shape[2],output.shape[3]
            resampledfeat = F.upsample(feat,size=(intH,intW), mode='bilinear') 
            #print("output resampled up: ",resampledfeat.shape)
            intmfeat=torch.cat([resampledfeat,output],dim=1)
            output = layer['conv2'](intmfeat)
     
        return self.conv2(output)


class UnetModelAssistLatentDecoder(nn.Module):
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
        #self.conv1x1=nn.Conv2d(1984,32,kernel_size=1)
        #self.conv1x1=nn.Conv2d(992,32,kernel_size=1)

        new_downsample_block = nn.ModuleDict({'conv1':ConvBlock(in_chans,chans,drop_prob)})
        self.down_sample_layers = nn.ModuleList([new_downsample_block])
        ch = chans

        for i in range(num_pool_layers - 1):
            new_downsample_block = nn.ModuleDict({'conv1':ConvBlock(ch,ch*2,drop_prob)})
            self.down_sample_layers += nn.ModuleList([new_downsample_block])
            ch *= 2

        new_downsample_block = nn.ModuleDict({'conv1':ConvBlock(ch,ch,drop_prob),'conv2':ConvBlock2X_to_X(ch+32,ch,drop_prob)})
        self.conv = nn.ModuleList([new_downsample_block])

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            new_upsample_block = nn.ModuleDict({'conv1':ConvBlock(ch*2,ch//2,drop_prob),'conv2':ConvBlock2X_to_X((ch//2)+32,ch//2,drop_prob)})
            self.up_sample_layers += nn.ModuleList([new_upsample_block])
            ch //= 2

        new_upsample_block = nn.ModuleDict({'conv1':ConvBlock(ch*2,ch,drop_prob),'conv2':ConvBlock2X_to_X(ch+32,ch,drop_prob)})
        self.up_sample_layers += nn.ModuleList([new_upsample_block])
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
        print("#############################")
        print(self.down_sample_layers)

    def forward(self, input, feat):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        H,W = input.shape[2],input.shape[3]
        #feat = self.conv1x1(feat)
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            #print(layer)
            _,output = layer['conv1'](output)
            #intH,intW = output.shape[2],output.shape[3]
            #resampledfeat = F.upsample(feat,size=(intH,intW), mode='bilinear') 
            #print(intout.shape,output.shape)
            #intmfeat=torch.cat([resampledfeat,output],dim=1)
            #output = layer['conv2'](intmfeat)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
            #print("output: ",output.shape)

        _,output = self.conv[0]['conv1'](output)
        intH,intW = output.shape[2],output.shape[3]
        resampledfeat = F.upsample(feat,size=(intH,intW), mode='bilinear') 
        latentfeat = torch.cat([resampledfeat, output],dim=1)
        #print("latent shape: ",latentfeat.shape)
        output = self.conv[0]['conv2'](latentfeat)
        #print("output lat:  ",output.shape)
        #print(output.shape)
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            _,output = layer['conv1'](output)
            #print("output up: ",output.shape)
            intH,intW = output.shape[2],output.shape[3]
            resampledfeat = F.upsample(feat,size=(intH,intW), mode='bilinear') 
            #print("output resampled up: ",resampledfeat.shape)
            intmfeat=torch.cat([resampledfeat,output],dim=1)
            output = layer['conv2'](intmfeat)
     
        return self.conv2(output)



#########################################################################

class MWCNN(nn.Module):

    def __init__(self, args, conv=common.default_conv):
        super(MWCNN, self).__init__()
        #n_resblocks = args.n_resblocks
        n_feats = 64#args.n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = 1#args.n_colors

        act = nn.ReLU(True)

        self.DWT = common.DWT()
        self.IWT = common.IWT(args)

        n = 1
        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))


        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)]
        i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))

        i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))

        i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        #print ("forward device:",x2.device,dir(self.DWT))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.tail(self.i_l0(x_))# + x #here commented +x since it is taken care in the calling forward method

        return x



'''

model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob = 0
     )

x = torch.rand([1,1,240,240])
y = model(x)
print (y.shape)


'''
 
