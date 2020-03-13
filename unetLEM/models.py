import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable, grad
import numpy as np
import os 
from unetmodel import UnetModel

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)

def relu():
    return nn.ReLU(inplace=True)

class LEM(nn.Module):

    def __init__(self):#,device):
        super(LEM,self).__init__()

        HSOBEL_WEIGHTS = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]) / 4.0
        HSOBEL_WEIGHTS = HSOBEL_WEIGHTS.astype(np.float64)
        VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

        HSOBEL_WEIGHTS = torch.from_numpy(HSOBEL_WEIGHTS)
        VSOBEL_WEIGHTS = torch.from_numpy(VSOBEL_WEIGHTS)

        HSOBEL_WEIGHTS = HSOBEL_WEIGHTS #.to(device)
        VSOBEL_WEIGHTS = VSOBEL_WEIGHTS #.to(device)

        self.HSOBEL_WEIGHTS = HSOBEL_WEIGHTS.unsqueeze(0).unsqueeze(0)
        self.VSOBEL_WEIGHTS = VSOBEL_WEIGHTS.unsqueeze(0).unsqueeze(0)

    def forward(self,img):
#         x = torch.log(x+1e-2)
        x = (torch.log(img+1)/(torch.log(1+torch.max(img))))
        #print (self.HSOBEL_WEIGHTS.shape,self.VSOBEL_WEIGHTS.shape)
        edge_torch_H = F.conv2d(x,self.HSOBEL_WEIGHTS,padding=1)
        edge_torch_V = F.conv2d(x,self.VSOBEL_WEIGHTS,padding=1)
        edge_abs = torch.sqrt(edge_torch_H **2 + edge_torch_V **2 ) #/ torch.sqrt(torch.Tensor([2]))
        return edge_abs

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
        return output

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class UnetModel(nn.Module):
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
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)



def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: if none, it is assumed to be the same as n_ch
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

class ConvFeatureBlock(nn.Module):
    def __init__(self,n_ch,nf=32):
        super(ConvFeatureBlock,self).__init__()

        self.conv1=nn.Sequential(nn.Conv2d(n_ch,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(nf,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(nf,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv2d(nf,nf,kernel_size=3,padding=1),nn.ReLU())
        self.conv5=nn.Conv2d(nf,1,kernel_size=3,padding=1)

        self.conv1x1_1=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1,padding=1),nn.ReLU())
        self.conv1x1_2=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1,padding=1),nn.ReLU())
        self.conv1x1_3=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1,padding=1),nn.ReLU())
        self.conv1x1_4=nn.Sequential(nn.Conv2d(nf+nf,nf,kernel_size=1,padding=1),nn.ReLU())
      
    def forward(x, feat):

        x = self.conv1(x)
        x = torch.cat([x,feat],dim=1)
        x = self.conv1x1_1(x)
       
        x = self.conv2(x)
        x = torch.cat([x,feat],dim=1)
        x = self.conv1x1_2(x)
        
        x = self.conv3(x)
        x = torch.cat([x,feat],dim=1)
        x = self.conv1x1_3(x)

        x = self.conv4(x)
        x = torch.cat([x,feat],dim=1)
        x = self.conv1x1_4(x)

        x = self.conv5(x)

        return x


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

    def __init__(self,args,n_channels=2, nc=5, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block


        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k):

        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            x = self.dcs[i](x,k)

        return x

class DnCnFeature(nn.Module):

    def __init__(self,args,n_channels=2, nc=5, nd=5,**kwargs):

        super(DnCnFeature, self).__init__()

        self.nc = nc
        self.nd = nd

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []


        for i in range(nc):
            conv_feature_block=ConvFeatureBlock(1)
            conv_blocks.append(conv_feature_block)
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k,feat):

        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x,feat)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            x = self.dcs[i](x,k)

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
#print (y.shape)
'''
 
