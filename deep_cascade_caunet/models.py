import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable, grad
import numpy as np
import os 
import math
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.inp = torch.nn.Conv2d(1,3,kernel_size=1)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.inp(X)
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return h_relu2_2


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False):
        super(double_conv, self).__init__()
        if dropout:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )


    def forward(self, x):
        x = self.conv(x)
        return x



class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, padding)


    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, padding, dropout)
        )


    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dropout=False, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, padding, dropout)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (math.ceil(diffY / 2), int(diffY / 2),
                        math.ceil(diffX / 2), int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)


    def forward(self, x):
        x = self.conv(x)
        return x


class CSE_Block(nn.Module):

    def __init__(self, in_channel, r, w, h):
        super(CSE_Block, self).__init__()

        self.layer = nn.Sequential(
            nn.AvgPool2d((w, h)),
            nn.Conv2d(in_channel, int(in_channel/r), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channel/r), in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.layer(x)
        return s*x

#    def forward(self, x):
#        #print("x entering CSE block: ", x.shape)
#        x1 = self.avgpoollayer(x)
        #print("x1 after avg pool block: ", x1.shape)
#        x2 = self.conv1(x1)
        #print("x2 after conv block: ", x2.shape)
#        x3 = self.relu1(x2)
#        x4 = self.conv2(x3)
        #print("x4 after conv block: ", x4.shape)
#        s = self.sigmoid1(x4)
#        return s*x

'''
x = torch.rand(1,128,37,37)
#layer = UNetCSE(1,1)
layer = CSE_Block(128, 8, 40, 36)
print (layer)
y = layer(x)
print (y.shape)

'''


# UNet with channel-wise attention, input arguments of CSE_block should change according to image size
class UNetCSE(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetCSE, self).__init__()
        self.inc = inconv(n_channels, 32, 1)
        self.down1 = down(32, 64, 1)
        self.down2 = down(64, 128, 1)
        self.se3 = CSE_Block(128, 8, 40, 36)
        self.up2 = up(128, 64, 1)
        self.se4 = CSE_Block(64, 8, 80, 72)
        self.up1 = up(64, 32, 1)
        self.se5 = CSE_Block(32, 8, 160, 144)
        self.outc = outconv(32, n_classes)


    def forward(self, x):
        #print("x entering: ", x.shape)
        x1 = self.inc(x)
        #print("x1 after inc: ", x1.shape)
        x2 = self.down1(x1)
        #print("x2 after down1: ", x2.shape)
        x3 = self.down2(x2)
        #print("x3 after inc: ", x3.shape)
        x3 = self.se3(x3)
        #print("x3 after se3: ", x3.shape)
        x = self.up2(x3, x2)
        #print("x after up2: ", x.shape)
        x = self.se4(x)
        #print("x after se4: ", x.shape)
        x = self.up1(x, x1)
        #print("x after up1: ", x.shape)
        x = self.se5(x)
        #print("x after se5: ", x.shape)
        x = self.outc(x)
        #print("x after outc: ", x.shape)
        return x

#x = torch.rand(1,1,160,160)
#layer = UNetCSE(1,1)
#layer = CSE_Block(128, 8, 40, 36)
#print (layer)
#y = layer(x)
#print (y.shape)



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



#class ConvBlock(nn.Module):
#    """
#    A Convolutional Block that consists of two convolution layers each followed by
#    instance normalization, relu activation and dropout.
#    """
#
#    def __init__(self, in_chans, out_chans, drop_prob):
#        """
#        Args:
#            in_chans (int): Number of channels in the input.
#            out_chans (int): Number of channels in the output.
#            drop_prob (float): Dropout probability.
#        """
#        super().__init__()
#
#        self.in_chans = in_chans
#        self.out_chans = out_chans
#        self.drop_prob = drop_prob
#
#        self.layers = nn.Sequential(
#            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
#            nn.InstanceNorm2d(out_chans),
#            nn.ReLU(),
#            nn.Dropout2d(drop_prob),
#            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
#            nn.InstanceNorm2d(out_chans),
#            nn.ReLU(),
#            nn.Dropout2d(drop_prob)
#        )
#
#    def forward(self, input):
#        """
#        Args:
#            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
#        Returns:
#            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
#        """
#        return self.layers(input)
#
#    def __repr__(self):
#        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
#            f'drop_prob={self.drop_prob})'
#
#
#class UnetModel(nn.Module):
#    """
#    PyTorch implementation of a U-Net model.
#    This is based on:
#        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
#        for biomedical image segmentation. In International Conference on Medical image
#        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
#    """
#
#    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
#        """
#        Args:
#            in_chans (int): Number of channels in the input to the U-Net model.
#            out_chans (int): Number of channels in the output to the U-Net model.
#            chans (int): Number of output channels of the first convolution layer.
#            num_pool_layers (int): Number of down-sampling and up-sampling layers.
#            drop_prob (float): Dropout probability.
#        """
#        super().__init__()
#
#        self.in_chans = in_chans
#        self.out_chans = out_chans
#        self.chans = chans
#        self.num_pool_layers = num_pool_layers
#        self.drop_prob = drop_prob
#
#        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
#        ch = chans
#        for i in range(num_pool_layers - 1):
#            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
#            ch *= 2
#        self.conv = nn.Sequential(ConvBlock(ch, ch, drop_prob),CALayer(ch)) # Added CALayer for the output of bottleneck layer also.
#        #self.ca1  = CALayer(ch)
#
#        self.up_sample_layers = nn.ModuleList()
#        for i in range(num_pool_layers - 1):
#            self.up_sample_layers += [nn.Sequential(ConvBlock(ch * 2, ch // 2, drop_prob),CALayer(ch//2))]
#            ch //= 2
#
#        self.up_sample_layers += [nn.Sequential(ConvBlock(ch * 2, ch, drop_prob),CALayer(ch))]
#
#        self.conv2 = nn.Sequential(
#            nn.Conv2d(ch, ch // 2, kernel_size=1),
#            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
#            nn.Conv2d(out_chans, out_chans, kernel_size=1),
#        )
#
#    def forward(self, input):
#        """
#        Args:
#            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
#        Returns:
#            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
#        """
#        stack = []
#        output = input
#        # Apply down-sampling layers
#        for layer in self.down_sample_layers:
#            output = layer(output)
#            stack.append(output)
#            output = F.max_pool2d(output, kernel_size=2)
#
#        output = self.conv(output)
#
#        # Apply up-sampling layers
#        for layer in self.up_sample_layers:
#            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
#            output = torch.cat([output, stack.pop()], dim=1)
#            output = layer(output)
#        return self.conv2(output)
################################################ Alternative implementation of CSEUnet using squeeze and excitation ################################
class cSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        
        super(cSELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        
        y = self.avg_pool(x)
        y = self.conv_du(y)
        
        return x * y

class sSELayer(nn.Module):
    def __init__(self, channel):
        super(sSELayer, self).__init__()
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 1, 1, padding=0, bias=True),
                nn.Sigmoid())


    def forward(self, x):
        
        y = self.conv_du(x)
        
        return x * y

class scSELayer(nn.Module):
    def __init__(self, channel,reduction=16):
        super(scSELayer, self).__init__()
        
        self.cSElayer = cSELayer(channel,reduction)
        self.sSElayer = sSELayer(channel)

    def forward(self, x):
        
        y1 = self.cSElayer(x)
        y2 = self.sSElayer(x)
        
        y  = torch.max(y1,y2)
        
        return y

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob,attention,attention_type,reduction): # cSE,scSE
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
        self.attention = attention
        self.reduction = reduction

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
        
        if self.attention:
            if attention_type == 'cSE':
                self.attention_layer = cSELayer(channel=self.out_chans,reduction=reduction)
            if attention_type == 'scSE':
                self.attention_layer = scSELayer(channel=self.out_chans,reduction=reduction)
                  
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]

        """
        out = self.layers(input)
        
        if self.attention:
            out = self.attention_layer(out)
        
        return out

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class CSEUnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,attention_type='cSE',reduction=16):
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
        self.reduction = reduction

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
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

################################################ end of Alternative implementation of CSEUnet using squeeze and excitation ################################


class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 

    def forward(self,predicted_img,us_kspace):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        # print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.mask.shape)
        
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        
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

        #conv_layer = conv_block


        for i in range(nc):
            #unetModelCSE = UNetCSE(1,1)
            #conv_blocks.append(UnetModel(in_chans=1,out_chans=1,chans=args.num_chans,num_pool_layers=args.num_pools,drop_prob=args.drop_prob))
            conv_blocks.append(CSEUnetModel(in_chans=1,out_chans=1,chans=32,num_pool_layers=4,drop_prob = 0,attention_type='cSE',reduction=16))            
            #conv_blocks.append(unetModelCSE)
            #conv_blocks.append(UnetModel(in_chans=1,out_chans=1,chans=args.num_chans,num_pool_layers=args.num_pools,drop_prob=args.drop_prob))
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k):

        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
#            x = self.dcs[i](x,k)
            xcrop = x
            if x.shape[2]==160 and x.shape[3]==160:
                xcrop = x[:,:,5:x.shape[2]-5,5:x.shape[3]-5]
            x = self.dcs[i](xcrop,k)
            if x.shape[2]==150 and x.shape[3]==150:
                x = F.pad(x,(5,5,5,5),"constant",0)

        return x



